"""Restartable full-training 2Wiki preparation, original-config topics, and Chroma.

Heavy operations are explicit subcommands for Kaggle. Imports and prepare/status
never instantiate a model or Chroma client. Original AT-RAG files are not edited.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"questions": 167454, "occurrences": 1674540, "unique_texts": 396978, "empty_texts": 0}
PROTOCOL = "full-train-occurrences-original-bertopic-v1"
RECIPE = {"constructor": {"min_topic_size": 50}, "reduction_target": 20,
          "fit_uses_all_occurrences": True, "fit_uses_precomputed_embeddings": False,
          "seed_or_component_overrides": None}
BATCH = 1000
# Audited PR #5 commit b042922: preparation schema/parsing/count rules unchanged.
# Retain its provenance and immutable DB hash rather than redo 1.67M occurrences.
PREPARATION_IMPLEMENTATIONS = {"3dcb7d5352646e0af5040ae21dd1e9bff0cc3948be95fbad9cea1843b1c46aa8"}


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".workflow.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is using this workflow directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def database(path, write=False):
    connection = (sqlite3.connect(path) if write else
                  sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True))
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute("PRAGMA mmap_size=0")
    if write:
        connection.execute("PRAGMA synchronous=FULL")
    try:
        yield connection
    finally:
        connection.close()


def state(db):
    row = db.execute("SELECT value FROM state WHERE name='job'").fetchone()
    return json.loads(row[0]) if row else None


def save_state(db, value):
    db.execute("INSERT OR REPLACE INTO state VALUES ('job',?)", (encoded(value).decode(),))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def adapter():
    return module("_full_minilm", ROOT / "compat/minilm/utils.py")


def environments():
    return module("_full_profiles", ROOT / "scripts/full_2wiki_environment.py")


def runtime(freeze, profile="retrieval"):
    compatibility = module("_full_environment", ROOT / "scripts/ingest_original_2wiki_minilm.py")
    for relative, expected in compatibility.SOURCE_SHA256.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Original source changed: {relative}")
    result = environments().inspect_environment(freeze, profile)
    result["implementation_sha256"] = digest(__file__)
    result["adapter_sha256"] = digest(ROOT / "compat/minilm/utils.py")
    result["thread_environment"] = {key: os.environ.get(key) for key in (
        "PYTHONHASHSEED", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    return result


def compatible_preparation(left, right):
    accepted = PREPARATION_IMPLEMENTATIONS | {digest(__file__)}
    return (left.get("implementation_sha256") in accepted and right.get("implementation_sha256") in accepted and
            {k: v for k, v in left.items() if k != "implementation_sha256"} ==
            {k: v for k, v in right.items() if k != "implementation_sha256"})


def occurrence_id(source_sha, number):
    return hashlib.sha256(f"2wiki:{source_sha}:{number}".encode()).hexdigest()


def prepare(source, heldout, root, expected=COUNTS, checkpoint=BATCH, progress=None):
    """One question in memory; SQLite transaction commits rows AND source cursor."""
    source, heldout, root = Path(source), Path(heldout), Path(root)
    if source.resolve() == heldout.resolve():
        raise ValueError("Training and held-out files must differ")
    config = {"protocol": PROTOCOL, "source_sha256": digest(source),
              "heldout_sha256": digest(heldout), "expected_counts": expected,
              "implementation_sha256": digest(__file__)}
    manifest_path = root / "preparation.json"
    if manifest_path.exists():
        result = prepared(root)
        if not compatible_preparation(result["configuration"], config):
            raise ValueError("Training/held-out source or preparation configuration changed")
        return result
    with database(root / "corpus.sqlite", write=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and "state" not in tables:
            raise ValueError("Unrecognized corpus database; choose a fresh workdir")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS state(name TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS texts(number INTEGER PRIMARY KEY,text_id TEXT UNIQUE,text TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS occurrences(number INTEGER PRIMARY KEY,text_id TEXT NOT NULL,metadata TEXT NOT NULL);
        """)
        job = state(db)
        if job and not compatible_preparation(job["configuration"], config):
            raise ValueError("Refusing resume against different source bytes/configuration")
        if job:
            config = job["configuration"]  # Preserve the approved producer's provenance.
        if not job:
            job = {"configuration": config, "offset": 0, "questions": 0, "occurrences": 0,
                   "unique_texts": 0, "empty_texts": 0, "text_bytes": 0, "complete": False}
            with db:
                save_state(db, job)
        committed = job["occurrences"]
        with source.open("rb") as stream:
            stream.seek(job["offset"])
            while not job["complete"]:
                raw = stream.readline()
                if not raw:
                    break
                question = json.loads(raw.decode("utf-8"))
                qid = question["question_id"]
                if not isinstance(qid, str) or not qid:
                    raise ValueError("Training question_id must be a nonempty string")
                db.execute("INSERT INTO questions VALUES (?)", (qid,))
                for context in question["contexts"]:
                    text = context["paragraph_text"]
                    if not isinstance(text, str):
                        raise ValueError("paragraph_text must be a string; nothing is silently skipped")
                    text_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    existing = db.execute("SELECT text FROM texts WHERE text_id=?", (text_id,)).fetchone()
                    if existing is None:
                        db.execute("INSERT INTO texts VALUES (?,?,?)", (job["unique_texts"], text_id, text))
                        job["unique_texts"] += 1
                    elif existing[0] != text:
                        raise ValueError("Exact-text SHA-256 collision")
                    metadata = {"idx": context["idx"], "title": context.get("title", "Unknown"),
                                "is_supporting": context.get("is_supporting", "N/A")}
                    if any(not isinstance(v, (str, int, float, bool)) for v in metadata.values()):
                        raise ValueError("Metadata must satisfy the original Chroma scalar contract")
                    db.execute("INSERT INTO occurrences VALUES (?,?,?)", (
                        job["occurrences"], text_id, encoded(metadata).decode()))
                    job["occurrences"] += 1
                    job["empty_texts"] += not bool(text.strip())
                    job["text_bytes"] += len(text.encode("utf-8"))
                job["questions"] += 1
                job["offset"] = stream.tell()
                if job["occurrences"] - committed >= checkpoint:
                    save_state(db, job)
                    db.commit()
                    committed = job["occurrences"]
                    if progress:
                        progress(dict(job))
        if any(job[key] != value for key, value in expected.items()):
            raise ValueError(f"Full-training counts differ: { {k: job[k] for k in expected} }")
        heldout_count = 0
        with heldout.open(encoding="utf-8") as stream:
            for line in stream:
                qid = json.loads(line)["question_id"]  # No answers/contexts/support labels are used.
                if db.execute("SELECT 1 FROM questions WHERE id=?", (qid,)).fetchone():
                    raise ValueError(f"Training/evaluation question-ID overlap: {qid}")
                heldout_count += 1
        if not heldout_count:
            raise ValueError("Held-out question file is empty")
        if digest(source) != config["source_sha256"] or digest(heldout) != config["heldout_sha256"]:
            raise ValueError("Inputs changed during preparation")
        job.update(complete=True, heldout_questions=heldout_count)
        save_state(db, job)
        db.commit()
    result = {"status": "complete", **job, "corpus_sha256": digest(root / "corpus.sqlite"),
              "duplicate_occurrences": job["occurrences"] - job["unique_texts"]}
    atomic(manifest_path, result)
    return result


def prepared(root):
    result = read(root / "preparation.json")
    if (result.get("status") != "complete" or result["configuration"]["protocol"] != PROTOCOL or
            result["configuration"]["implementation_sha256"] not in PREPARATION_IMPLEMENTATIONS | {digest(__file__)}):
        raise ValueError("Preparation is incomplete or belongs to another protocol")
    if digest(root / "corpus.sqlite") != result["corpus_sha256"]:
        raise ValueError("Prepared corpus changed")
    return result


def rows(db, start, size=None):
    size = BATCH if size is None else size
    return db.execute("""SELECT o.number,o.text_id,t.text,o.metadata FROM occurrences o
                         JOIN texts t ON t.text_id=o.text_id
                         WHERE o.number>=? ORDER BY o.number LIMIT ?""", (start, size)).fetchall()


def all_documents(root):
    # Required global BERTopic operations, not used by cache or Chroma ingestion.
    # Intern repeated strings so the occurrence list reuses text objects.
    with database(root / "corpus.sqlite") as db:
        texts = dict(db.execute("SELECT text_id,text FROM texts"))
        return [texts[text_id] for (text_id,) in db.execute("SELECT text_id FROM occurrences ORDER BY number")]


def memory_guard(minimum_gib):
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    limit, usage = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if limit.exists() and limit.read_text().strip() != "max":
        available = min(available or math.inf, max(0, int(limit.read_text()) - int(usage.read_text())))
    if available is None or available < minimum_gib * 1024 ** 3:
        raise ValueError(f"Stage requires at least {minimum_gib} GiB available. No model-setting or index-setting fallback is performed")


def saved_model(model, path):
    temporary = path.with_name(path.name + ".tmp")
    model.save(str(temporary))  # Original BERTopic default serialization, including encoder.
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def model_class():
    from bertopic import BERTopic
    return BERTopic


def resolved_parameters(model):
    def clean(value):
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else repr(value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return repr(value)
    return {name: clean(getattr(model, name).get_params(deep=False))
            for name in ("umap_model", "hdbscan_model", "vectorizer_model") if hasattr(model, name)}


def topic_manifest(root, info, environment, allow_unverified=False):
    manifest = read(root / "topics.json")
    producer = manifest.get("configuration", {}).get("environment")
    expected = {"protocol": PROTOCOL, "corpus_sha256": info["corpus_sha256"],
                "recipe": RECIPE, "environment": producer}
    if manifest.get("status") != "complete" or manifest.get("configuration") != expected:
        raise ValueError("Topic model provenance is incompatible; the old custom 418-topic model is not accepted")
    if digest(root / "bertopic_model") != manifest["model_sha256"]:
        raise ValueError("Saved full-training topic model changed")
    if producer != environment:
        try:
            matching = environments().model_contract(producer) == environments().model_contract(environment)
        except (KeyError, TypeError):
            matching = False
        if not matching:
            raise ValueError("Topic model provenance: Python, recorded freeze, or retained model dependencies differ")
        if not allow_unverified:
            certificate = root / "model_compatibility.json"
            reference = root / "model_probe_reference.json"
            expected_certificate = {"status": "verified", "model_sha256": manifest["model_sha256"],
                                    "source_environment_sha256": identity(producer),
                                    "target_environment_sha256": identity(environment),
                                    "reference_sha256": digest(reference) if reference.exists() else None}
            if not certificate.exists() or read(certificate) != expected_certificate:
                raise ValueError("Run verify-model in this environment before loading/ingesting the training model")
    return manifest


def probe_outputs(model, cases):
    import numpy as np
    result = []
    for texts in cases:
        topics, probabilities = model.transform(texts)
        labels = np.asarray(topics)
        if labels.shape != (len(texts),) or labels.dtype.kind not in "iu":
            raise ValueError("Invalid topic labels during model compatibility verification")
        probabilities = None if probabilities is None else np.asarray(probabilities, dtype="float64")
        if probabilities is not None and not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite model probe probabilities")
        result.append({"topics": labels.tolist(), "probabilities": None if probabilities is None else probabilities.tolist()})
    return result


def compare_probes(expected, actual):
    import numpy as np
    if len(expected) != len(actual):
        raise ValueError("Model compatibility probe count differs")
    for reference, candidate in zip(expected, actual):
        if reference["topics"] != candidate["topics"]:
            raise ValueError("Model compatibility probe topic labels differ")
        left, right = reference["probabilities"], candidate["probabilities"]
        if (left is None) != (right is None):
            raise ValueError("Model compatibility probe probability availability differs")
        if left is not None:
            left, right = np.asarray(left), np.asarray(right)
            if left.shape != right.shape or not np.allclose(left, right, atol=1e-6, rtol=1e-6):
                raise ValueError("Model compatibility probe probabilities differ (atol=rtol=1e-6)")


def verify_model(root, environment, loader=None):
    """Explicit Kaggle check, separate from the original global topic transform.

    Source environment writes reference outputs from a freshly loaded pickle.
    A destination must reproduce them before its model transfer is certified.
    Only training paragraph text and a fixed neutral query are used, never gold.
    """
    info = prepared(root)
    manifest = topic_manifest(root, info, environment, allow_unverified=True)
    producer = manifest["configuration"]["environment"]
    reference_path = root / "model_probe_reference.json"
    if not reference_path.exists() and producer != environment:
        raise ValueError("First run verify-model in the original training environment to create the reference")
    if reference_path.exists():
        reference = read(reference_path)
        if (reference["model_sha256"] != manifest["model_sha256"] or
                reference["corpus_sha256"] != info["corpus_sha256"] or
                reference["source_environment_sha256"] != identity(producer)):
            raise ValueError("Model probe reference provenance changed")
        cases = reference["cases"]
    else:
        with database(root / "corpus.sqlite") as db:
            texts = [r[0] for r in db.execute("SELECT text FROM texts ORDER BY number LIMIT 3")]
        cases = [[text] for text in texts] + [["Which person or place is described in the passage?"], texts]
        reference = {"model_sha256": manifest["model_sha256"], "corpus_sha256": info["corpus_sha256"],
                     "source_environment_sha256": identity(producer), "cases": cases}
    loaded = (loader or model_class().load)(str(root / "bertopic_model"))
    outputs = probe_outputs(loaded, cases)
    if digest(root / "bertopic_model") != manifest["model_sha256"]:
        raise ValueError("Model file changed during verification")
    if reference_path.exists():
        compare_probes(reference["outputs"], outputs)
    else:
        reference["outputs"] = outputs
        atomic(reference_path, reference)
    certificate = {"status": "verified", "model_sha256": manifest["model_sha256"],
                   "source_environment_sha256": identity(producer),
                   "target_environment_sha256": identity(environment), "reference_sha256": digest(reference_path)}
    atomic(root / "model_compatibility.json", certificate)
    return certificate


def train_topics(root, environment, minimum_gib=24, factory=None, loader=None):
    if environment.get("profile", "training") != "training":
        raise ValueError("train-topics requires the training-only environment")
    info = prepared(root)
    configuration = {"protocol": PROTOCOL, "corpus_sha256": info["corpus_sha256"],
                     "recipe": RECIPE, "environment": environment}
    if (root / "topics.json").exists():
        return topic_manifest(root, info, environment)
    memory_guard(minimum_gib)
    documents = all_documents(root)
    checkpoint = root / "fit.json"
    if checkpoint.exists():
        fit = read(checkpoint)
        if fit["configuration"] != configuration or digest(root / "fit_model") != fit["sha256"]:
            raise ValueError("Fit checkpoint changed or belongs to different data/settings/environment")
        model = (loader or model_class().load)(str(root / "fit_model"))
    else:
        model = (factory or model_class())(min_topic_size=50)
        model.fit_transform(documents)  # All occurrences; no cached retrieval vectors supplied.
        saved_model(model, root / "fit_model")
        atomic(checkpoint, {"configuration": configuration, "sha256": digest(root / "fit_model")})
    model = model.reduce_topics(documents, nr_topics=20)
    saved_model(model, root / "bertopic_model")
    manifest = {"status": "complete", "configuration": configuration,
                "model_sha256": digest(root / "bertopic_model"),
                "resolved_components": resolved_parameters(model),
                "embedding_backend": repr(getattr(model, "embedding_model", None)),
                "regular_topics": len(set(getattr(model, "topics_", [])) - {-1})}
    atomic(root / "topics.json", manifest)
    return manifest


def assignment_manifest(root, info, model):
    manifest = read(root / "assignments.json")
    if (manifest.get("status") != "complete" or manifest["corpus_sha256"] != info["corpus_sha256"] or
            manifest["model_sha256"] != model["model_sha256"] or
            digest(root / "topics.npy") != manifest["assignments_sha256"]):
        raise ValueError("Topic assignments are incomplete or incompatible")
    return manifest


def assign_topics(root, environment, minimum_gib=24, loader=None):
    if environment.get("profile", "training") != "training":
        raise ValueError("assign-topics requires the training-only environment")
    import numpy as np
    info = prepared(root)
    model = topic_manifest(root, info, environment)
    if (root / "assignments.json").exists():
        return assignment_manifest(root, info, model)
    memory_guard(minimum_gib)
    loaded = (loader or model_class().load)(str(root / "bertopic_model"))
    documents = all_documents(root)
    topics, _ = loaded.transform(documents)  # Original single whole-corpus transform.
    topics = np.asarray(topics)
    if topics.shape != (info["occurrences"],) or topics.dtype.kind not in "iu" or np.any(topics < -1):
        raise ValueError("Invalid occurrence-level topic assignments")
    temporary = root / "topics.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, topics.astype("<i4"), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / "topics.npy")
    manifest = {"status": "complete", "corpus_sha256": info["corpus_sha256"],
                "model_sha256": model["model_sha256"], "assignments_sha256": digest(root / "topics.npy"),
                "occurrences": len(topics), "regular_topics": int(len(set(topics.tolist()) - {-1})),
                "outlier_occurrences": int(np.sum(topics == -1))}
    atomic(root / "assignments.json", manifest)
    return manifest


def validate_vectors(vectors, count):
    import numpy as np
    vectors = np.asarray(vectors, dtype="<f4")
    if (vectors.shape != (count, 384) or not np.isfinite(vectors).all() or
            not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0)):
        raise ValueError("Embeddings must be finite normalized float32 rows of dimension 384")
    return vectors


def legacy_embeddings(folder, info, environment, settings):
    """Conservative reuse: verify files, frozen encoder AND producer runtime."""
    import numpy as np
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    config = manifest.get("configuration", {})
    expected = {"model_id": settings["model_id"], "model_revision": settings["revision"],
                "dimensions": 384, "dtype": "float32", "normalize_embeddings": True,
                "max_seq_length": 256, "device": "cpu", "batch_size": 32,
                "preprocessing": "paragraph_text unchanged; no title/prompt prefix"}
    if (manifest.get("status") != "complete" or manifest.get("processed_count") != info["unique_texts"] or
            any(config.get(k) != v for k, v in expected.items()) or
            manifest.get("config_sha256") != identity(config)):
        raise ValueError("Cached embedding configuration differs from the frozen adapter; recompute the unique-text cache")
    producer = config.get("environment", {}).get("libraries", {})
    for name in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub"):
        if not producer.get(name) or producer[name] != environment["packages"].get(name):
            raise ValueError(f"Cached embedding producer runtime differs for {name}; no silent reuse")
    for name in ("embeddings.npy", "document_ids.jsonl"):
        entry = manifest.get("artifacts", {}).get(name, {})
        if entry.get("sha256") != digest(folder / name) or entry.get("bytes") != (folder / name).stat().st_size:
            raise ValueError(f"Cached embedding artifact changed: {name}")
    matrix = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (info["unique_texts"], 384) or matrix.dtype != np.dtype("<f4"):
        raise ValueError("Cached embedding shape/dtype differs")
    raw_hash = hashlib.sha256()
    for start in range(0, len(matrix), BATCH):
        raw_hash.update(validate_vectors(matrix[start:start+BATCH], min(BATCH, len(matrix)-start)).tobytes())
    if raw_hash.hexdigest() != manifest.get("embedding_data_sha256"):
        raise ValueError("Cached raw-vector checksum differs")
    return matrix, {"manifest_sha256": digest(folder / "manifest.json"), "configuration": config}


def build_cache(root, environment, reuse=None, embedder=None, progress=None):
    import numpy as np
    info = prepared(root)
    helper = adapter()
    settings = helper.embedding_settings()
    source = {"mode": "fresh"}
    matrix = None
    if reuse:
        matrix, source = legacy_embeddings(reuse, info, environment, settings)
    configuration = {"protocol": PROTOCOL, "corpus_sha256": info["corpus_sha256"],
                     "embedding": settings, "environment": environment, "source": source}
    if (root / "cache.json").exists():
        result = cache_manifest(root, info, environment)
        if result["configuration"] != configuration:
            raise ValueError("Requested embedding-cache source differs from the completed cache")
        return result
    with database(root / "corpus.sqlite") as corpus, database(root / "vectors.sqlite", write=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and "state" not in tables:
            raise ValueError("Unrecognized vector cache; choose a fresh workdir")
        db.executescript("CREATE TABLE IF NOT EXISTS state(name TEXT PRIMARY KEY,value TEXT NOT NULL);"
                         "CREATE TABLE IF NOT EXISTS vectors(number INTEGER PRIMARY KEY,text_id TEXT UNIQUE,value BLOB NOT NULL);")
        job = state(db)
        if job and job["configuration"] != configuration:
            raise ValueError("Embedding-cache configuration changed; refusing incompatible restart")
        if not job:
            job = {"configuration": configuration, "next": 0}
            with db:
                save_state(db, job)
        provider = embedder
        if reuse:
            def cached_batches():
                batch = []
                with (Path(reuse) / "document_ids.jsonl").open() as stream:
                    for number, line in enumerate(stream):
                        if number >= info["unique_texts"]:
                            raise ValueError("Cached ID mapping has too many rows")
                        mapping = json.loads(line)
                        if mapping.get("row") != number:
                            raise ValueError("Cached ID row order is invalid")
                        if number < job["next"]:
                            continue
                        text_id = mapping["document_id"]
                        if not corpus.execute("SELECT 1 FROM texts WHERE text_id=?", (text_id,)).fetchone():
                            raise ValueError("Cached paragraph is absent from this training corpus")
                        batch.append((number, text_id, None))
                        if len(batch) == BATCH:
                            yield batch
                            batch = []
                if batch:
                    yield batch
            batches = cached_batches()
        else:
            def fresh_batches():
                cursor = corpus.execute("SELECT number,text_id,text FROM texts WHERE number>=? ORDER BY number", (job["next"],))
                while batch := cursor.fetchmany(BATCH):
                    yield batch
            batches = fresh_batches()
        for batch in batches:
            if matrix is not None:
                vectors = validate_vectors(matrix[[row[0] for row in batch]], len(batch))
            else:
                if provider is None:
                    provider = helper.get_embedding_model()
                vectors = validate_vectors(provider.embed_documents([row[2] for row in batch]), len(batch))
            with db:
                for (number, text_id, _), vector in zip(batch, vectors):
                    db.execute("INSERT INTO vectors VALUES (?,?,?)", (number, text_id, vector.tobytes()))
                job["next"] = batch[-1][0] + 1
                save_state(db, job)
            if progress:
                progress(dict(job))
        count = db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        if count != info["unique_texts"]:
            raise ValueError("Embedding cache does not cover every unique training text exactly once")
    result = {"status": "complete", "configuration": configuration, "vectors": count,
              "database_sha256": digest(root / "vectors.sqlite")}
    atomic(root / "cache.json", result)
    return result


def cache_manifest(root, info, environment):
    result = read(root / "cache.json")
    config = result.get("configuration", {})
    if (result.get("status") != "complete" or config.get("corpus_sha256") != info["corpus_sha256"] or
            config.get("embedding") != adapter().embedding_settings() or config.get("environment") != environment or
            result.get("vectors") != info["unique_texts"] or
            digest(root / "vectors.sqlite") != result.get("database_sha256")):
        raise ValueError("Embedding cache changed or belongs to a different configuration")
    return result


def chroma_collection(path, create=True):
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    method = client.get_or_create_collection if create else client.get_collection
    return method(name="langchain", embedding_function=None)


def records(batch, source_sha, topics):
    return [{"id": occurrence_id(source_sha, number), "document": text,
             "metadata": {**json.loads(metadata), "bertopic": f"Topic {int(topics[number])}"}}
            for number, _, text, metadata in batch]


def verify_batch(collection, expected):
    actual = collection.get(ids=[r["id"] for r in expected], include=["documents", "metadatas"])
    observed = {key: (text, meta) for key, text, meta in zip(actual["ids"], actual["documents"], actual["metadatas"])}
    if len(observed) != len(expected) or any(observed.get(r["id"]) != (r["document"], r["metadata"]) for r in expected):
        raise ValueError("Stored Chroma IDs/documents/metadata disagree with the occurrence catalog")


def verify_store(root, collection, info, topics):
    if collection.count() != info["occurrences"]:
        raise ValueError("Chroma count differs from the full occurrence count")
    result = hashlib.sha256()
    with database(root / "corpus.sqlite") as db:
        for start in range(0, info["occurrences"], BATCH):
            expected = records(rows(db, start), info["configuration"]["source_sha256"], topics)
            verify_batch(collection, expected)
            for record in expected:
                result.update(encoded(record))
    return result.hexdigest()


def index_configuration(info, model, assignments, cache, environment):
    return {"protocol": PROTOCOL, "corpus_sha256": info["corpus_sha256"],
            "model_sha256": model["model_sha256"], "assignments_sha256": assignments["assignments_sha256"],
            "cache_sha256": cache["database_sha256"], "environment": environment,
            "collection": "langchain", "chroma_settings": "unmodified defaults", "batch_size": BATCH}


def remaining_disk_estimate(info, next_row):
    remaining = max(0, info["occurrences"] - next_row)
    return remaining * (384 * 4 * 3 + 512) + math.ceil(2 * info["text_bytes"] * remaining / info["occurrences"])


def ingest(root, environment, factory=None, progress=None, minimum_free_bytes=None, minimum_gib=8):
    import numpy as np
    info = prepared(root)
    model = topic_manifest(root, info, environment)
    assignments = assignment_manifest(root, info, model)
    cache = cache_manifest(root, info, environment)
    topics = np.load(root / "topics.npy", mmap_mode="r", allow_pickle=False)
    if topics.shape != (info["occurrences"],) or topics.dtype.kind != "i":
        raise ValueError("Topic array shape/type differs")
    configuration = index_configuration(info, model, assignments, cache, environment)
    path = root / "ingestion.json"
    job = read(path) if path.exists() else None
    if job and job["configuration"] != configuration:
        raise ValueError("Index provenance changed; refusing incompatible restart")
    if not job and (root / "chroma").exists():
        raise ValueError("Existing Chroma store has no matching workflow checkpoint")
    required = (remaining_disk_estimate(info, job["next"] if job else 0)
                if minimum_free_bytes is None else minimum_free_bytes)
    if shutil.disk_usage(root).free < required and not (job and job.get("complete")):
        raise ValueError(f"Insufficient free disk for conservative Chroma estimate ({required} bytes)")
    if job is None:
        job = {"configuration": configuration, "next": 0, "complete": False}
        atomic(path, job)
    memory_guard(minimum_gib)
    collection = (factory or chroma_collection)(root / "chroma")
    count = collection.count()
    if not job["next"] <= count <= min(info["occurrences"], job["next"] + BATCH):
        raise ValueError("Store/checkpoint counts disagree; not a valid one-batch interrupted upsert")
    with database(root / "corpus.sqlite") as corpus, database(root / "vectors.sqlite") as vectors:
        if job["next"]:
            start = max(0, job["next"] - BATCH)
            verify_batch(collection, records(rows(corpus, start, job["next"]-start), info["configuration"]["source_sha256"], topics))
        while job["next"] < info["occurrences"]:
            batch = rows(corpus, job["next"])
            expected = records(batch, info["configuration"]["source_sha256"], topics)
            matrix = []
            for _, text_id, _, _ in batch:
                row = vectors.execute("SELECT value FROM vectors WHERE text_id=?", (text_id,)).fetchone()
                if row is None:
                    raise ValueError("Missing cached text vector")
                matrix.append(np.frombuffer(row[0], dtype="<f4"))
            matrix = validate_vectors(matrix, len(batch))
            collection.upsert(ids=[r["id"] for r in expected], documents=[r["document"] for r in expected],
                              metadatas=[r["metadata"] for r in expected], embeddings=matrix.tolist())
            # A crash here replays the same occurrence IDs, never a second copy.
            if progress:
                progress({"stage": "after_upsert", "next": job["next"]})
            verify_batch(collection, expected)
            job["next"] += len(batch)
            atomic(path, job)
    job["record_sha256"] = verify_store(root, collection, info, topics)
    job["complete"] = True
    atomic(path, job)
    return job


def verify_completed(root, environment, factory=None):
    import numpy as np
    job = read(root / "ingestion.json")
    if not job.get("complete") or not (root / "chroma").is_dir():
        raise ValueError("No completed Chroma store; verification never creates a new one")
    info = prepared(root)
    model = topic_manifest(root, info, environment)
    assignments = assignment_manifest(root, info, model)
    cache = cache_manifest(root, info, environment)
    if job["configuration"] != index_configuration(info, model, assignments, cache, environment):
        raise ValueError("Completed index provenance differs")
    collection = factory(root / "chroma") if factory else chroma_collection(root / "chroma", create=False)
    observed = verify_store(root, collection, info, np.load(root / "topics.npy", mmap_mode="r", allow_pickle=False))
    if observed != job["record_sha256"]:
        raise ValueError("Completed record checksum differs")
    return {"status": "verified", "occurrences": collection.count(), "record_sha256": observed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "preflight", "verify-model", "cache", "train-topics", "assign-topics", "ingest", "verify", "status"])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Complete processed train.jsonl (prepare only)")
    parser.add_argument("--heldout", type=Path, help="Held-out dev/test JSONL; only question IDs are inspected")
    parser.add_argument("--training-requirements", type=Path, help="Original immutable 62-package training freeze (Python 3.12.3)")
    parser.add_argument("--reuse-embeddings", type=Path, help="Optional complete legacy embedding directory (cache only)")
    parser.add_argument("--environment-profile", choices=["training", "retrieval"],
                        help="preflight/verify-model: default training; other stages select and enforce their profile")
    args = parser.parse_args(argv)
    root = args.workdir.resolve()
    try:
        with lock(root):
            if args.stage == "prepare":
                if not args.input or args.input.name != "train.jsonl" or not args.heldout:
                    raise ValueError("prepare requires the full train.jsonl and --heldout JSONL")
                result = prepare(args.input, args.heldout, root,
                                 progress=lambda state: print(f"Prepared {state['occurrences']} occurrences", flush=True))
            elif args.stage == "status":
                result = {name: read(root / name) for name in (
                    "preparation.json", "cache.json", "fit.json", "topics.json", "assignments.json", "model_compatibility.json", "ingestion.json") if (root / name).exists()}
            else:
                if not args.training_requirements:
                    raise ValueError("This stage requires --training-requirements")
                required_profile = ("training" if args.stage in {"train-topics", "assign-topics"} else "retrieval")
                if args.stage in {"preflight", "verify-model"}:
                    required_profile = args.environment_profile or "training"
                elif args.environment_profile and args.environment_profile != required_profile:
                    raise ValueError(f"{args.stage} requires the {required_profile} profile")
                environment = runtime(args.training_requirements, required_profile)
                if args.stage == "preflight":
                    result = environment
                elif args.stage == "verify-model":
                    result = verify_model(root, environment)
                elif args.stage == "cache":
                    result = build_cache(root, environment, reuse=args.reuse_embeddings,
                                         progress=lambda state: print(f"Cached {state['next']} unique texts", flush=True))
                elif args.stage == "train-topics":
                    result = train_topics(root, environment)
                elif args.stage == "assign-topics":
                    result = assign_topics(root, environment)
                elif args.stage == "ingest":
                    result = ingest(root, environment,
                                    progress=lambda state: print(f"Upserting from occurrence {state['next']}", flush=True))
                else:
                    result = verify_completed(root, environment)
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError, sqlite3.Error) as error:
        print(f"Full-training workflow stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
