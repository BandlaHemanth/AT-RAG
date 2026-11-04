"""Train the min_topic_size=50 reference on every frozen 150k occurrence.

Kaggle/Linux entry point. Completed embedding batches can be verified and copied
from a stopped run into a NEW output directory. No existing run is modified.
See docs/bertopic_150k.md. Full training is intentionally not part of the tests.
"""

import argparse
from collections import Counter
from contextlib import closing
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback

# Establish thread controls before NumPy/Numba/Torch imports. An incompatible
# caller setting is refused at startup rather than silently ignored.
THREAD_SETTINGS = {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
for _name, _value in THREAD_SETTINGS.items():
    os.environ.setdefault(_name, _value)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

# Sibling helpers remain unchanged; their checksums are part of cache identity.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_150k_corpus as preparation
import embed_full_2wiki_corpus as embedding
import train_full_2wiki_topics as full


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/tae_v2_150k_bertopic"
MODEL_ID = embedding.MODEL_ID
MODEL_REVISION = embedding.MODEL_REVISION
DIMENSIONS = 384
MAX_TOKENS = 256
DTYPE = np.dtype("<f4")
SEED = 42
GIB = 1024 ** 3
EXPECTED = {"sha256": "d9edc9d9bc3ff4bbb98de8690c5b79c71aa6e2878f035df55cd01891ea322c33",
            "occurrences": 150_000, "unique_texts": 66_208,
            "duplicate_excess_occurrences": 83_792}
MODEL_FILES = ("config.json", "config_sentence_transformers.json", "modules.json",
               "sentence_bert_config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "1_Pooling/config.json")
REQUIRED_MODEL_FILES = {"config.json", "modules.json", "sentence_bert_config.json", "model.safetensors",
                        "tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json"}
json_bytes = preparation.json_bytes
sha256 = preparation.sha256
require = preparation.require


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=preparation.reject_constant,
                       object_pairs_hook=preparation.unique_object)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    preparation.write_file(temporary, json_bytes(value))  # exclusive create + fsync
    # Atomic, non-overwriting commit. An interruption during serialization leaves
    # only .part; a reader can never mistake a half-written JSON for a checkpoint.
    os.link(temporary, path)
    sync_directory(path.parent)
    temporary.unlink()
    sync_directory(path.parent)


def file_evidence(path):
    return {"sha256": sha256(path), "bytes": Path(path).stat().st_size}


def copy_verified(source, destination, evidence):
    require(source.is_file() and not source.is_symlink(), f"Unsafe/missing cache file: {source}")
    require(file_evidence(source) == evidence, f"Cache integrity failure: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    require(file_evidence(destination) == evidence, f"Cache changed during copy: {source}")


def corpus_rows(path, digest):
    """Only the Step-1 allowlist; preserve text and every occurrence in file order."""
    with Path(path).open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            require(bool(raw.strip()), f"Unexpected blank corpus row: {line}")
            row = json.loads(raw.decode("utf-8"), parse_constant=preparation.reject_constant,
                             object_pairs_hook=preparation.unique_object)
            require(isinstance(row, dict) and row.keys() == preparation.CORPUS_FIELDS,
                    f"Corpus row {line}: missing or unapproved fields")
            text = row["paragraph_text"]
            require(isinstance(text, str) and bool(text.strip()), f"Corpus row {line}: invalid text")
            require(row["text_sha256"] == preparation.text_hash(text), f"Corpus row {line}: text hash mismatch")
            require(row["source_split"] == "train", f"Corpus row {line}: nontraining source")
            require(isinstance(row["question_id"], str) and bool(row["question_id"].strip()), "Invalid question_id")
            require(isinstance(row["title"], str), "Invalid title")
            row["question_id"].encode("utf-8")
            row["title"].encode("utf-8")
            for key in ("source_line", "context_position", "idx"):
                require(type(row[key]) is int and row[key] >= (1 if key == "source_line" else 0),
                        f"Invalid {key} in corpus row {line}")
            prefix = "2wikimultihopqa:train:"
            oid = row["occurrence_id"]
            require(isinstance(oid, str) and oid.startswith(prefix), "Invalid occurrence_id")
            parts = oid[len(prefix):].split(":")
            require(len(parts) == 3 and len(parts[0]) == 64 and
                    all(c in "0123456789abcdef" for c in parts[0]) and
                    parts[1:] == [str(row["source_line"]), str(row["context_position"])],
                    "Occurrence ID does not match source line/position")
            yield row


def validate_corpus(path, output, expected):
    digest, count, utf8_bytes, python_bytes = hashlib.sha256(), 0, 0, 0
    index = output / ".corpus_validation.sqlite3"
    try:
        with closing(sqlite3.connect(index)) as connection:
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            connection.execute("CREATE TABLE texts (hash TEXT PRIMARY KEY, text BLOB) WITHOUT ROWID")
            for row in corpus_rows(path, digest):
                try:
                    connection.execute("INSERT INTO ids VALUES (?)", (row["occurrence_id"],))
                except sqlite3.IntegrityError as error:
                    raise ValueError("Duplicate occurrence_id in training corpus") from error
                encoded = row["paragraph_text"].encode("utf-8")
                prior = connection.execute("SELECT text FROM texts WHERE hash=?", (row["text_sha256"],)).fetchone()
                if prior is None:
                    connection.execute("INSERT INTO texts VALUES (?,?)", (row["text_sha256"], encoded))
                else:
                    require(prior[0] == encoded, "SHA-256 text collision")
                count += 1
                utf8_bytes += len(encoded)
                python_bytes += sys.getsizeof(row["paragraph_text"]) + 8
                if count % 1000 == 0:
                    connection.commit()
            unique = connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        actual = {"sha256": digest.hexdigest(), "occurrences": count, "unique_texts": unique,
                  "duplicate_excess_occurrences": count - unique}
        require(actual == expected, f"Frozen corpus identity mismatch: expected={expected}, actual={actual}")
        return {**actual, "valid_nonempty_texts": count, "unique_occurrence_ids": count,
                "text_utf8_bytes": utf8_bytes, "estimated_python_text_bytes": python_bytes,
                "source_split": "train"}
    finally:
        for suffix in ("", "-journal", "-wal", "-shm"):
            index.with_name(index.name + suffix).unlink(missing_ok=True)


def reference_parameters():
    parameters = full.topic_configuration(min_cluster_size=50, min_samples=50,
                                           n_neighbors=15, umap_epochs=200)
    # None reproduces HDBSCAN's original default: min_samples=min_cluster_size.
    parameters["hdbscan"]["min_samples"] = None
    parameters["vectorizer"]["min_df"] = 1
    return parameters


def runtime_environment(device):
    environment = embedding.runtime_environment(device)
    environment["libraries"].update(full.environment()["libraries"])
    environment["libraries"]["llvmlite"] = metadata.version("llvmlite")
    environment["installed_packages"] = dict(sorted(
        (distribution.metadata["Name"], distribution.version) for distribution in metadata.distributions()
        if distribution.metadata["Name"]))
    if device == "cpu" and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                environment["device_identity"]["cpu_model"] = line.partition(":")[2].strip()
                break
    return environment


def preflight(evidence, output, min_ram_gib, min_disk_gib):
    for value in (min_ram_gib, min_disk_gib):
        require(math.isfinite(value) and value >= 0, "Resource floors must be finite and nonnegative")
    payload = evidence["occurrences"] * DIMENSIONS * DTYPE.itemsize
    # Planning allowances, not bounds on third-party working memory.
    ram = max(int(min_ram_gib * GIB), 3 * evidence["estimated_python_text_bytes"] +
              4 * payload + 3 * GIB + evidence["occurrences"] * (15 * 32 + 5 * 128))
    disk = max(int(min_disk_gib * GIB), 6 * payload + evidence["text_utf8_bytes"] + 2 * GIB)
    snapshot = full.memory_snapshot()
    free = shutil.disk_usage(output).free
    report = {"memory": snapshot, "disk_free_bytes": free, "embedding_payload_bytes": payload,
              "required_available_ram_bytes": ram, "required_free_disk_bytes": disk,
              "status": "passed" if snapshot["available_memory_bytes"] >= ram and free >= disk else "failed",
              "policy": "One admission check per invocation; later snapshots are observations, not repeated gates"}
    write_json(output / "preflight.json", report)
    require(report["status"] == "passed", f"Resource preflight failed: need {ram/GIB:.2f} GiB available RAM "
            f"and {disk/GIB:.2f} GiB free disk. Use a larger session; training rows will not be reduced.")
    return report


def stage_encoder(output, reuse=None, encoder_dir=None):
    """Pinned download or hash-verified copy; never write to the source encoder."""
    destination = output / "encoder"
    destination.mkdir()
    if encoder_dir is not None:
        source = Path(encoder_dir)
        original = read_json(source / "embedding_verification_manifest.json")
        require(original.get("model_id") == MODEL_ID and original.get("revision") == MODEL_REVISION,
                "Local encoder manifest has wrong model/revision")
        files = original.get("sha256", {})
        require(REQUIRED_MODEL_FILES <= set(files) <= set(MODEL_FILES), "Incomplete/unsafe encoder manifest")
        for name, expected_hash in files.items():
            path = source / name
            copy_verified(path, destination / name, {"sha256": expected_hash, "bytes": path.stat().st_size})
    elif reuse is not None:
        original = read_json(reuse / "encoder_manifest.json")
        require(original.get("model_id") == MODEL_ID and original.get("revision") == MODEL_REVISION,
                "Reusable encoder has wrong model/revision")
        files = original.get("files", {})
        require(REQUIRED_MODEL_FILES <= set(files) <= set(MODEL_FILES), "Incomplete/unsafe encoder manifest")
        for name, evidence in files.items():
            copy_verified(reuse / "encoder" / name, destination / name, evidence)
    else:
        from huggingface_hub import snapshot_download
        # local_dir keeps downloaded model/cache metadata inside this NEW run.
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination),
                          allow_patterns=list(MODEL_FILES))
    files = {name: file_evidence(destination / name) for name in MODEL_FILES if (destination / name).is_file()}
    require(REQUIRED_MODEL_FILES <= set(files), "Pinned encoder snapshot is incomplete")
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    write_json(output / "encoder_manifest.json", manifest)
    return manifest


class MiniLMEncoder:
    def __init__(self, directory, device):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = SentenceTransformer(str(directory), device=device, local_files_only=True,
                                         trust_remote_code=False)
        self.model.max_seq_length = MAX_TOKENS
        self.model.eval()
        require(self.model.get_sentence_embedding_dimension() == DIMENSIONS, "Wrong encoder dimension")
        self.tokenizer = self.model.tokenizer
        self.device = device
        if device.startswith("cuda:"):
            torch.cuda.reset_peak_memory_stats(device)

    def encode_windows(self, features):
        import torch
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        padded = {key: value.to(self.device) for key, value in padded.items()}
        with torch.inference_mode():
            vectors = self.model(padded)["sentence_embedding"]
        return vectors.detach().cpu().numpy().astype(DTYPE, copy=False)

    def memory_statistics(self):
        if self.device == "cpu":
            return {"device": "cpu", "cuda_used": False}
        import torch
        torch.cuda.synchronize(self.device)
        return {"device": self.device, "cuda_used": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}


def encode_occurrences(texts, encoder, window_batch_size):
    """Cover every token; one normalized vector per original occurrence.

    Single windows match normalized MiniLM encoding. For long paragraphs, encode
    nonoverlapping content-token windows, normalize each window, take a weighted
    mean (content-token counts), then normalize the paragraph vector. Full original
    strings, including their tails, still enter BERTopic's text representation.
    """
    tokenizer = encoder.tokenizer
    capacity = MAX_TOKENS - tokenizer.num_special_tokens_to_add(pair=False)
    require(capacity > 0 and window_batch_size > 0, "Invalid encoder window capacity/batch size")
    sums = np.zeros((len(texts), DIMENSIONS), dtype=np.float64)
    pending, owners, weights = [], [], []
    statistics = Counter(content_tokens=0, windows=0, multiwindow_occurrences=0)

    def flush():
        if not pending:
            return
        vectors = np.asarray(encoder.encode_windows(pending))
        require(vectors.shape == (len(pending), DIMENSIONS) and vectors.dtype == DTYPE and
                np.isfinite(vectors).all(), "Invalid encoder window vectors")
        norms = np.linalg.norm(vectors, axis=1)
        require(bool(np.all(norms > 0)), "Zero encoder window vector")
        for vector, norm, owner, weight in zip(vectors, norms, owners, weights):
            sums[owner] += (vector / norm) * weight
        pending.clear()
        owners.clear()
        weights.clear()

    for owner, text in enumerate(texts):
        # No encode(texts) call: its implicit max_seq_length truncation would lose tails.
        tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        require(bool(tokens), "Nonempty paragraph produced zero tokenizer content tokens")
        windows = (len(tokens) + capacity - 1) // capacity
        statistics["content_tokens"] += len(tokens)
        statistics["windows"] += windows
        statistics["multiwindow_occurrences"] += int(windows > 1)
        for start in range(0, len(tokens), capacity):
            ids = tokens[start:start + capacity]
            features = tokenizer.prepare_for_model(ids, add_special_tokens=True, truncation=False,
                                                   return_attention_mask=True)
            require(len(features["input_ids"]) <= MAX_TOKENS, "Tokenizer window exceeds model limit")
            pending.append(features)
            owners.append(owner)
            weights.append(len(ids))
            if len(pending) == window_batch_size:
                flush()
    flush()
    norms = np.linalg.norm(sums, axis=1)
    require(bool(np.all(norms > 0)), "Zero aggregated paragraph vector")
    vectors = (sums / norms[:, None]).astype(DTYPE)
    embedding.validate_vectors(vectors, len(texts))
    return vectors, dict(statistics)


def embedding_configuration(inputs, encoder_manifest, environment, device, batch_size, window_batch_size):
    files = [Path(__file__), Path(preparation.__file__), Path(embedding.__file__), Path(full.__file__), full.RESOURCE_HELPER]
    return {
        "schema_version": 1, "corpus": {key: inputs[key] for key in EXPECTED},
        "encoder": encoder_manifest, "dimensions": DIMENSIONS, "dtype": "<f4", "normalize_embeddings": True,
        "max_window_tokens": MAX_TOKENS, "long_text_policy": "all content tokens; nonoverlapping windows; "
            "normalized window vectors weighted by content-token count; final L2 normalization",
        "seed": SEED, "device": device, "batch_occurrences": batch_size, "window_batch_size": window_batch_size,
        "implementation_sha256": {path.name: sha256(path) for path in files},
        "runtime": {key: environment[key] for key in
                    ("python", "libraries", "device_identity", "cublas_workspace_config")},
    }


def embedding_batches(corpus, output, config, *, reuse=None, encoder_factory=MiniLMEncoder, event=None):
    signature = hashlib.sha256(json_bytes(config)).hexdigest()
    if reuse is not None:
        require(read_json(reuse / "embedding_config.json") == config,
                "Reusable embedding configuration/code/runtime differs; use a matching environment or a fresh run")
    directory = output / "embedding_batches"
    directory.mkdir()
    count, number, reused, encoder = 0, 0, 0, None
    totals, pending, digest = Counter(content_tokens=0, windows=0, multiwindow_occurrences=0), [], hashlib.sha256()
    can_reuse = reuse is not None

    def finish(batch):
        nonlocal count, number, reused, encoder, can_reuse
        start, stop = count, count + len(batch)
        stem = f"batch_{number:06d}"
        matrix, marker = directory / (stem + ".npy"), directory / (stem + ".json")
        ids_sha = hashlib.sha256(json_bytes([row["occurrence_id"] for row in batch])).hexdigest()
        old_marker = reuse / "embedding_batches" / marker.name if can_reuse else None
        if old_marker is not None and old_marker.is_file():
            saved = read_json(old_marker)
            require(saved.get("signature_sha256") == signature and saved.get("start") == start and
                    saved.get("stop") == stop and saved.get("occurrence_ids_sha256") == ids_sha,
                    "Reusable batch row/configuration identity mismatch")
            source = reuse / "embedding_batches" / matrix.name
            copy_verified(source, matrix, saved["array"])
            vectors = np.load(matrix, mmap_mode="r", allow_pickle=False)
            embedding.validate_vectors(vectors, len(batch))
            del vectors
            statistics = saved["encoding_statistics"]
            reused += len(batch)
        else:
            # A missing commit marker means the batch was never completed. The
            # source's orphan bytes are ignored; it remains completely unchanged.
            can_reuse = False
            if encoder is None:
                encoder = encoder_factory(output / "encoder", config["device"])
            vectors, statistics = encode_occurrences([row["paragraph_text"] for row in batch], encoder,
                                                    config["window_batch_size"])
            with matrix.open("xb") as stream:
                np.save(stream, vectors, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            del vectors
            saved = {"signature_sha256": signature, "start": start, "stop": stop,
                     "occurrence_ids_sha256": ids_sha, "array": file_evidence(matrix),
                     "encoding_statistics": statistics}
        require(set(statistics) == set(totals) and all(type(v) is int and v >= 0 for v in statistics.values()),
                "Invalid cached encoding statistics")
        require(statistics["windows"] >= len(batch) and statistics["content_tokens"] >= len(batch) and
                statistics["multiwindow_occurrences"] <= len(batch), "Invalid cached token coverage counts")
        sync_directory(directory)
        write_json(marker, saved)  # Commit only after complete durable .npy and integrity checks.
        totals.update(statistics)
        count, number = stop, number + 1
        if event:
            event("embedding_batch_complete", rows=count, reused_rows=reused)

    for row in corpus_rows(corpus, digest):
        pending.append(row)
        if len(pending) == config["batch_occurrences"]:
            finish(pending)
            pending.clear()
    if pending:
        finish(pending)
    require(count == config["corpus"]["occurrences"] and digest.hexdigest() == config["corpus"]["sha256"],
            "Corpus changed during embedding")
    device_memory = encoder.memory_statistics() if encoder is not None else {
        "device": config["device"], "encoder_loaded_this_attempt": False}
    del encoder
    gc.collect()
    return {"occurrences": count, "completed_batches": number, "reused_occurrences": reused,
            "newly_encoded_occurrences": count - reused, "encoding_statistics": dict(totals),
            "device_memory": device_memory}


def assemble_embeddings(output, count, batches):
    path = output / "embeddings.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=DTYPE, shape=(count, DIMENSIONS))
    offset = 0
    for number in range(batches):
        stem = f"batch_{number:06d}"
        saved = read_json(output / "embedding_batches" / (stem + ".json"))
        shard = output / "embedding_batches" / (stem + ".npy")
        require(file_evidence(shard) == saved["array"], "Embedding batch changed before assembly")
        vectors = np.load(shard, mmap_mode="r", allow_pickle=False)
        require(saved["start"] == offset and saved["stop"] <= count, "Embedding batch order/size mismatch")
        embedding.validate_vectors(vectors, saved["stop"] - offset)
        matrix[offset:saved["stop"]] = vectors
        offset = saved["stop"]
        del vectors
    require(offset == count, "Assembled embedding row count mismatch")
    matrix.flush()
    del matrix
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(output)
    return file_evidence(path)


def export_topics(output, corpus, model, expected):
    topics, strengths, counts = full.model_assignments(model, expected["occurrences"])
    require(strengths is not None, "Missing scalar HDBSCAN membership strengths")
    digest, written = hashlib.sha256(), 0
    with (output / "occurrence_topics.jsonl").open("xb") as stream:
        for index, row in enumerate(corpus_rows(corpus, digest)):
            require(index < len(topics), "More corpus rows than topic assignments")
            topic = int(topics[index])
            stream.write(json_bytes({"row": index, "occurrence_id": row["occurrence_id"],
                                     "text_sha256": row["text_sha256"], "question_id": row["question_id"],
                                     "source_split": "train", "source_line": row["source_line"],
                                     "context_position": row["context_position"], "idx": row["idx"],
                                     "title": row["title"], "topic": topic, "bertopic": f"Topic {topic}",
                                     "membership_strength": float(strengths[index])}))
            written += 1
        stream.flush()
        os.fsync(stream.fileno())
    require(written == expected["occurrences"] and digest.hexdigest() == expected["sha256"],
            "Corpus changed while exporting assignments")
    representations = model.get_topics()
    require(set(representations) == set(counts), "Topic representations and assignments disagree")
    metadata = [{"topic": int(topic), "occurrences": size,
                 "representation": [{"term": term, "score": float(score)} for term, score in representations[topic]]}
                for topic, size in sorted(counts.items())]
    summary = {"occurrences": written, "non_outlier_topics": len(counts) - int(-1 in counts),
               "outlier_occurrences": counts.get(-1, 0), "outlier_fraction": counts.get(-1, 0) / written,
               "topic_sizes": {str(topic): size for topic, size in sorted(counts.items())}}
    write_json(output / "topic_metadata.json", {"topics": metadata, "summary": summary})
    return summary


def run_training(corpus, output_dir, *, reuse=None, encoder_dir=None, device="cpu", batch_size=128,
                 window_batch_size=64, min_ram_gib=12, min_disk_gib=6, expected=None,
                 encoder_stager=stage_encoder, encoder_factory=MiniLMEncoder,
                 trainer=full.fit_model, loader=full.load_model, environment_getter=runtime_environment):
    """Expected identity/service injection is for tiny synthetic tests, never CLI."""
    corpus, output = Path(corpus).absolute(), Path(output_dir).absolute()
    expected = EXPECTED if expected is None else expected
    require(corpus.is_file(), f"Missing frozen training corpus: {corpus}")
    require(os.environ.get("PYTHONHASHSEED") == "42", "Launch Python with PYTHONHASHSEED=42")
    require(all(os.environ.get(k) == v for k, v in THREAD_SETTINGS.items()), "Start with one Numba/BLAS/OpenMP thread")
    require(type(batch_size) is int and batch_size > 0 and type(window_batch_size) is int and window_batch_size > 0,
            "Batch sizes must be positive integers")
    device = embedding.canonical_device(device)
    reuse = None if reuse is None else Path(reuse).absolute()
    if reuse is not None:
        require(not output.resolve().is_relative_to(reuse.resolve()) and
                not reuse.resolve().is_relative_to(output.resolve()),
                "New output and read-only reuse source must be separate directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # includes empty dirs, symlinks and concurrent creators
    # Keep runtime download/compiler caches within this new run too. Existing
    # HF/Torch credentials and network/proxy settings are left intact.
    runtime_cache = output / "runtime_cache"
    for key, subdir in (("HF_HUB_CACHE", "hub"), ("TRANSFORMERS_CACHE", "transformers"),
                        ("NUMBA_CACHE_DIR", "numba")):
        os.environ[key] = str(runtime_cache / subdir)
    stage, started, timings = "corpus_validation", time.perf_counter(), {}
    resources = {}
    log = (output / "events.jsonl").open("xb")

    def event(name, **values):
        log.write(json_bytes({"event": name, "elapsed_seconds": time.perf_counter() - started, **values}))
        log.flush()
        os.fsync(log.fileno())
        print(f"{name}: {values}", flush=True)

    def snapshot(name):
        resources[name] = full.memory_snapshot()
        event(name, memory=resources[name])

    try:
        event("started", corpus=str(corpus), output=str(output), expected_occurrences=expected["occurrences"])
        inputs = validate_corpus(corpus, output, expected)
        write_json(output / "input_validation.json", inputs)
        stage = "preflight"
        resources["preflight"] = preflight(inputs, output, min_ram_gib, min_disk_gib)
        stage = "encoder_setup"
        environment = environment_getter(device)
        encoder_manifest = encoder_stager(output, reuse=reuse, encoder_dir=encoder_dir)
        config = embedding_configuration(inputs, encoder_manifest, environment, device, batch_size, window_batch_size)
        write_json(output / "embedding_config.json", config)
        parameters = reference_parameters()
        training = {"reference": "original-style BERTopic min_topic_size=50; explicit seeded estimators",
                    "parameters": parameters, "environment": environment, "seed": SEED,
                    "training_occurrences": expected["occurrences"], "corpus_sha256": inputs["sha256"],
                    "embedding_config_sha256": hashlib.sha256(json_bytes(config)).hexdigest(),
                    "process_settings": {key: os.environ.get(key) for key in
                                         ("PYTHONHASHSEED", "TOKENIZERS_PARALLELISM", "CUBLAS_WORKSPACE_CONFIG", *THREAD_SETTINGS)},
                    "resource_floors_gib": {"available_ram": min_ram_gib, "free_disk": min_disk_gib},
                    "git_commit": preparation.git_revision(), "training_text_input": "paragraph_text only",
                    "row_policy": "every original occurrence, including duplicates; no reduction or outlier reassignment",
                    "serialization": "complete BERTopic pickle with fitted UMAP, HDBSCAN prediction data, vectorizer and c-TF-IDF; "
                                     "pinned encoder files saved separately in encoder/"}
        write_json(output / "training_config.json", training)
        snapshot("before_embedding")
        stage, phase_start = "embedding", time.perf_counter()
        embedding_stats = embedding_batches(corpus, output, config, reuse=reuse,
                                           encoder_factory=encoder_factory, event=event)
        timings["embedding_seconds"] = time.perf_counter() - phase_start
        write_json(output / "embedding_statistics.json", embedding_stats)
        snapshot("after_embedding")
        stage, phase_start = "embedding_assembly", time.perf_counter()
        matrix_evidence = assemble_embeddings(output, expected["occurrences"], embedding_stats["completed_batches"])
        timings["embedding_assembly_seconds"] = time.perf_counter() - phase_start
        write_json(output / "embedding_matrix.json", matrix_evidence)
        stage = "loading_all_training_texts"
        digest = hashlib.sha256()
        texts = [row["paragraph_text"] for row in corpus_rows(corpus, digest)]
        require(len(texts) == expected["occurrences"] and digest.hexdigest() == expected["sha256"],
                "Corpus changed before fitting")
        matrix = np.load(output / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        require(matrix.shape == (len(texts), DIMENSIONS) and matrix.dtype == DTYPE, "Invalid fitting matrix")
        snapshot("before_fit")
        stage, phase_start = "fitting_all_occurrences", time.perf_counter()
        event(stage, occurrences=len(texts), unique_texts=inputs["unique_texts"])
        model = trainer(parameters, texts, matrix)
        timings["fit_seconds"] = time.perf_counter() - phase_start
        full.model_assignments(model, len(texts))
        del texts, matrix
        gc.collect()
        snapshot("after_fit")
        stage, phase_start = "model_and_assignment_export", time.perf_counter()
        model_path = output / "bertopic_model.pkl"
        require(not os.path.lexists(model_path), "Model destination already exists")
        model.save(str(model_path), serialization="pickle", save_embedding_model=False)
        with model_path.open("rb") as stream:
            os.fsync(stream.fileno())
        summary = export_topics(output, corpus, model, expected)
        write_json(output / "resolved_parameters.json", full.resolved_parameters(model))
        model_evidence = file_evidence(model_path)
        del model
        gc.collect()
        stage = "verifying_saved_model"
        require(file_evidence(model_path) == model_evidence, "Saved model hash mismatch")
        restored = loader(model_path)
        _, strengths, restored_counts = full.model_assignments(restored, expected["occurrences"])
        require(strengths is not None and {str(k): v for k, v in restored_counts.items()} == summary["topic_sizes"],
                "Reloaded model lost assignments/confidence state")
        del restored
        gc.collect()
        timings["export_and_reload_seconds"] = time.perf_counter() - phase_start
        stage = "final_integrity_verification"
        require(sha256(corpus) == expected["sha256"], "Input corpus changed during training")
        require(file_evidence(output / "embeddings.npy") == matrix_evidence, "Embeddings changed during training")
        snapshot("after_saved_model_verification")
        timings["total_seconds"] = time.perf_counter() - started
        write_json(output / "runtime_statistics.json", {"timings": timings, "memory": resources,
                   "embedding": embedding_stats, "topic_summary": summary,
                   "memory_measurement": "Host/cgroup availability and kernel process RSS/high-water mark at named stages; "
                                         "GPU allocator peak is recorded separately if CUDA was used"})
        event("complete", occurrences=expected["occurrences"], topic_summary=summary)
        log.close()
        artifacts = {str(path.relative_to(output)): file_evidence(path) for path in sorted(output.rglob("*"))
                     if path.is_file() and ".cache" not in path.relative_to(output).parts and
                     "runtime_cache" not in path.relative_to(output).parts}
        manifest = {"schema_version": 1, "status": "complete", "occurrences": expected["occurrences"],
                    "corpus_sha256": expected["sha256"], "encoder_id": MODEL_ID, "encoder_revision": MODEL_REVISION,
                    "artifacts": artifacts, "topic_summary": summary,
                    "elapsed_seconds_before_manifest_write": time.perf_counter() - started,
                    "inference": "Supply embeddings from the saved pinned encoder/windowing policy to BERTopic.transform. "
                                 "Scalar approximate-predict strength is available; HDBSCAN prediction data enables "
                                 "query-batch multi-topic membership later. Strength is not calibrated evidence adequacy.",
                    "reproducibility": "Fixed input order, seeds, parameters and threads; record/reuse exact Python, packages "
                                       "and device. Bitwise clustering equivalence across hardware is not guaranteed."}
        write_json(output / "manifest.json", manifest)  # sole success marker, written LAST
        return manifest
    except BaseException as error:
        try:
            if not log.closed:
                event("failed", stage=stage, error_type=type(error).__name__, error=str(error))
                log.close()
            write_json(output / "failure.json", {"status": "failed", "stage": stage,
                       "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                       "elapsed_seconds": time.perf_counter() - started,
                       "recovery": "Preserve this run. Use a NEW output directory with --reuse-embeddings-from pointing here. "
                                   "Completed, compatible batches are hash-verified; no mid-fit UMAP/HDBSCAN resume."})
        except (OSError, ValueError):
            pass  # Disk exhaustion must not hide the original failure.
        raise
    finally:
        if not log.closed:
            log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts/tae_v2_150k_corpus/run_001/corpus.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "run_001")
    parser.add_argument("--reuse-embeddings-from", type=Path,
                        help="Read-only previous run/checkpoint; always writes to a fresh --output-dir")
    parser.add_argument("--encoder-dir", type=Path, help="Optional verified local MiniLM directory with its verification manifest")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N; no automatic device fallback")
    parser.add_argument("--batch-size", type=int, default=128, help="Occurrences per durable embedding checkpoint")
    parser.add_argument("--window-batch-size", type=int, default=64, help="MiniLM token windows per forward pass")
    parser.add_argument("--min-available-ram-gib", type=float, default=12)
    parser.add_argument("--min-free-disk-gib", type=float, default=6)
    args = parser.parse_args(argv)
    try:
        require(args.output_dir.absolute().parent.resolve() == OUTPUT_ROOT.resolve(),
                f"Output must be a NEW run directory directly under {OUTPUT_ROOT}")
        manifest = run_training(args.corpus, args.output_dir, reuse=args.reuse_embeddings_from,
                                encoder_dir=args.encoder_dir, device=args.device, batch_size=args.batch_size,
                                window_batch_size=args.window_batch_size,
                                min_ram_gib=args.min_available_ram_gib, min_disk_gib=args.min_free_disk_gib)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Complete: {manifest['occurrences']} occurrence assignments in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
