"""CPU BERTopic training from the verified full training corpus and embeddings.

No encoder is constructed. Final outputs are published atomically; interrupted
runs remain in a separate .inprogress directory. See docs/full_2wiki_topics.md.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import runpy
import shutil
import sys
import time

import numpy as np


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CORPUS_SHA256 = "f1824d1dc9275731243c6f5a89cc4d702c77b3607413343c9c75bb242fff3160"
DOCUMENT_COUNT = 396_978
SEED = 42
GIB = 1024 ** 3
RESOURCE_HELPER = Path(__file__).with_name("run_bertopic_pilot.py")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_file(path):
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def documents(path, digest):
    """Project only document ID and unchanged text; annotations are ignored."""
    with Path(path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
                text, identifier = row["paragraph_text"], row["document_id"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("paragraph_text must be a nonempty string")
                if identifier != hashlib.sha256(text.encode("utf-8")).hexdigest():
                    raise ValueError("document ID differs from the exact UTF-8 text hash")
            except (ValueError, UnicodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid corpus line {line_number}: {error}") from error
            yield identifier, text


def validate_inputs(corpus, embeddings_path, mapping, manifest_path, expected_sha, expected_count):
    """Verify bytes and every aligned ID before any fitting or text accumulation."""
    manifest = read_object(manifest_path)
    config = manifest.get("configuration", {})
    required = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "corpus_sha256": expected_sha, "document_count": expected_count,
                "dimensions": 384, "dtype": "float32", "byte_order": "little",
                "normalize_embeddings": True, "max_seq_length": 256}
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete" or
            manifest.get("processed_count") != expected_count or not isinstance(config, dict) or
            any(config.get(key) != value for key, value in required.items())):
        raise ValueError("Embedding manifest is incomplete or incompatible with the fixed corpus/encoder")
    if manifest.get("config_sha256") != hashlib.sha256(json_bytes(config)).hexdigest():
        raise ValueError("Embedding manifest configuration checksum mismatch")
    if not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("Embedding manifest lacks artifact checksums")
    evidence = {}
    for name, path in (("embeddings.npy", embeddings_path), ("document_ids.jsonl", mapping)):
        recorded = manifest.get("artifacts", {}).get(name, {})
        actual = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if recorded != actual:
            raise ValueError(f"Embedding artifact checksum/size mismatch: {name}")
        evidence[name] = actual
    matrix = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if (not isinstance(matrix, np.memmap) or matrix.shape != (expected_count, 384) or
            matrix.dtype != np.dtype("<f4") or not matrix.flags.c_contiguous or
            embeddings_path.stat().st_size != matrix.offset + matrix.nbytes):
        raise ValueError("Embeddings must be a complete C-order (document_count, 384) float32 .npy matrix")
    vector_digest = hashlib.sha256()
    for start in range(0, expected_count, 1024):
        block = matrix[start:start + 1024]
        if (not np.isfinite(block).all() or
                not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Embeddings contain nonfinite or non-normalized rows")
        vector_digest.update(block.tobytes(order="C"))
    if vector_digest.hexdigest() != manifest.get("embedding_data_sha256"):
        raise ValueError("Embedding raw-vector checksum mismatch")
    digest, count, text_bytes, python_bytes = hashlib.sha256(), 0, 0, 0
    with mapping.open("rb") as ids:
        for index, (identifier, text) in enumerate(documents(corpus, digest)):
            item = json.loads(ids.readline())
            if item != {"row": index, "document_id": identifier}:
                raise ValueError(f"Document-ID mapping is not aligned at row {index}")
            count += 1
            text_bytes += len(text.encode("utf-8"))
            python_bytes += sys.getsizeof(text) + 8
        if ids.read(1):
            raise ValueError("Document-ID mapping contains extra rows/bytes")
    if count != expected_count or digest.hexdigest() != expected_sha:
        raise ValueError(f"Corpus checksum/count mismatch: count={count}, sha256={digest.hexdigest()}")
    evidence.update({"corpus": {"sha256": digest.hexdigest(), "bytes": corpus.stat().st_size},
                     "embedding_manifest": {"sha256": file_sha256(manifest_path),
                                            "bytes": manifest_path.stat().st_size},
                     "document_count": count, "embedding_bytes": matrix.nbytes,
                     "text_utf8_bytes": text_bytes, "estimated_python_text_bytes": python_bytes})
    return matrix, evidence


def topic_configuration(*, min_cluster_size=100, min_samples=10, n_neighbors=15, umap_epochs=200):
    if min_cluster_size < 2 or min_samples < 1 or n_neighbors < 2 or umap_epochs < 10:
        raise ValueError("Invalid clustering/UMAP parameters")
    return {
        "umap": {"n_components": 5, "n_neighbors": n_neighbors, "metric": "cosine", "min_dist": 0.0,
                 "n_epochs": umap_epochs, "init": "spectral", "random_state": SEED,
                 "transform_seed": SEED, "low_memory": True, "n_jobs": 1,
                 "force_approximation_algorithm": True},
        "hdbscan": {"min_cluster_size": min_cluster_size, "min_samples": min_samples,
                    "metric": "euclidean", "algorithm": "boruvka_kdtree", "leaf_size": 40,
                    "alpha": 1.0, "approx_min_span_tree": True, "gen_min_span_tree": False,
                    "core_dist_n_jobs": 1, "prediction_data": True,
                    "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.0,
                    "allow_single_cluster": False},
        "vectorizer": {"stop_words": "english", "min_df": 2, "max_features": 20_000,
                       "ngram_range": [1, 1], "dtype": "int32"},
        "ctfidf": {"bm25_weighting": False, "reduce_frequent_words": False},
        "bertopic": {"embedding_model": None, "language": "english", "min_topic_size": min_cluster_size,
                     "representation_model": None, "nr_topics": None, "calculate_probabilities": False,
                     "low_memory": True, "top_n_words": 10, "verbose": False},
    }


def environment():
    versions = {}
    for package in ("numpy", "bertopic", "umap-learn", "hdbscan", "scikit-learn", "scipy", "pandas",
                    "numba", "pynndescent", "joblib", "threadpoolctl", "sentence-transformers", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "libraries": versions,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "numba_threads": 1, "blas_threads": 1}


def memory_snapshot():
    # Reuse the pilot's host + all-visible-ancestor cgroup v1/v2 checks unchanged.
    return runpy.run_path(str(RESOURCE_HELPER))["memory_snapshot"]()


def resource_check(evidence, parameters, output_parent, min_ram_gib, min_disk_gib):
    text = evidence["estimated_python_text_bytes"]
    matrix = evidence["embedding_bytes"]
    # A planning allowance, not a prediction/bound on third-party peak memory.
    estimate = (3 * text + 3 * matrix + 2 * GIB + evidence["document_count"] *
                (parameters["umap"]["n_neighbors"] * 32 + parameters["umap"]["n_components"] * 128))
    required_ram = max(int(min_ram_gib * GIB), estimate)
    required_disk = max(int(min_disk_gib * GIB), 4 * matrix + text + GIB)
    existing = output_parent
    while not existing.exists():
        existing = existing.parent
    snapshot = memory_snapshot()
    free = shutil.disk_usage(existing).free
    report = {"memory": snapshot, "disk_free_bytes": free, "required_ram_bytes": required_ram,
              "required_disk_bytes": required_disk, "planning_ram_bytes": estimate}
    if snapshot["available_memory_bytes"] < required_ram:
        raise ValueError(f"Insufficient RAM: {snapshot['available_memory_bytes'] / GIB:.2f} GiB available; "
                         f"need at least {required_ram / GIB:.2f} GiB. Use a higher-RAM CPU session.")
    if free < required_disk:
        raise ValueError(f"Insufficient disk: {free / GIB:.2f} GiB free; need {required_disk / GIB:.2f} GiB.")
    return report


def fit_model(parameters, texts, embeddings):
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import numba
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from threadpoolctl import threadpool_limits
    from umap import UMAP

    random.seed(SEED)
    np.random.seed(SEED)
    numba.set_num_threads(1)
    vectorizer = {**parameters["vectorizer"], "dtype": np.int32,
                  "ngram_range": tuple(parameters["vectorizer"]["ngram_range"])}
    model = BERTopic(**parameters["bertopic"], umap_model=UMAP(**parameters["umap"]),
                     hdbscan_model=HDBSCAN(**parameters["hdbscan"]),
                     vectorizer_model=CountVectorizer(**vectorizer),
                     ctfidf_model=ClassTfidfTransformer(**parameters["ctfidf"]))
    with threadpool_limits(limits=1):
        # Pass the original read-only mmap; do not astype(), normalize, or re-encode.
        model.fit_transform(texts, embeddings=embeddings)
    return model


def load_model(path):
    from bertopic import BERTopic
    return BERTopic.load(str(path))


def model_assignments(model, count):
    topics = np.asarray(model.topics_)
    if topics.shape != (count,) or topics.dtype.kind not in "iu" or np.any(topics < -1):
        raise ValueError("Invalid fitted topic assignments")
    strengths = model.probabilities_
    if strengths is not None:
        strengths = np.asarray(strengths)
        if (strengths.shape != (count,) or not np.isfinite(strengths).all() or
                np.any(strengths < 0) or np.any(strengths > 1 + 1e-6)):
            raise ValueError("Expected one finite HDBSCAN membership strength per document, not a dense matrix")
    if (not model.hdbscan_model.prediction_data or model.hdbscan_model.prediction_data_ is None or
            not callable(model.umap_model.transform)):
        raise ValueError("Model does not preserve UMAP transform and HDBSCAN prediction data")
    counts = Counter(int(topic) for topic in topics)
    if {int(k): int(v) for k, v in model.topic_sizes_.items()} != dict(counts):
        raise ValueError("Model topic-size metadata disagrees with assignments")
    return topics, strengths, counts


def resolved_parameters(model):
    """Record constructor defaults too, without serializing estimator objects."""
    def value(item):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite_float": str(item)}
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, (list, tuple)):
            return [value(part) for part in item]
        if isinstance(item, dict):
            return {str(key): value(part) for key, part in item.items()}
        if isinstance(item, type):
            return {"type": f"{item.__module__}.{item.__qualname__}"}
        return {"class": f"{type(item).__module__}.{type(item).__qualname__}"}

    result = {}
    for name, component in (("bertopic", model), ("umap", model.umap_model), ("hdbscan", model.hdbscan_model),
                             ("vectorizer", getattr(model, "vectorizer_model", None)),
                             ("ctfidf", getattr(model, "ctfidf_model", None))):
        result[name] = {"class": f"{type(component).__module__}.{type(component).__qualname__}",
                        "parameters": value(component.get_params(deep=False))
                        if hasattr(component, "get_params") else None}
    return result


@contextmanager
def job_lock(workspace):
    with (workspace / ".lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another training process is using this workspace") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_topic_artifacts(artifacts, corpus, model, count, expected_sha):
    topics, strengths, counts = model_assignments(model, count)
    digest, written = hashlib.sha256(), 0
    with (artifacts / "document_topics.jsonl").open("wb") as stream:
        for row, (identifier, _) in enumerate(documents(corpus, digest)):
            if row >= count:
                raise ValueError("Corpus grew during training")
            topic = int(topics[row])
            stream.write(json_bytes({"row": row, "document_id": identifier, "topic": topic,
                                     "bertopic": f"Topic {topic}",
                                     "membership_strength": None if strengths is None else float(strengths[row])}))
            written += 1
        stream.flush()
        os.fsync(stream.fileno())
    if written != count or digest.hexdigest() != expected_sha:
        raise ValueError("Corpus changed during training; assignments are incomplete")
    representations = model.get_topics()
    if set(int(topic) for topic in representations) != set(counts):
        raise ValueError("Topic representations disagree with fitted topic IDs")
    with (artifacts / "topic_metadata.jsonl").open("wb") as stream:
        for topic, size in sorted(counts.items()):
            terms = [{"term": term, "score": float(score)} for term, score in representations[topic]]
            stream.write(json_bytes({"topic": topic, "bertopic": f"Topic {topic}", "size": size,
                                     "proportion": size / count, "representation": terms}))
        stream.flush()
        os.fsync(stream.fileno())
    histogram = Counter(size for topic, size in counts.items() if topic != -1)
    summary = {"document_count": count, "topics_discovered_excluding_outliers": sum(t != -1 for t in counts),
               "outlier_count": counts.get(-1, 0), "outlier_rate": counts.get(-1, 0) / count,
               "topic_size_distribution": [{"topic": t, "size": n} for t, n in sorted(counts.items())],
               "topic_size_histogram_excluding_outliers": [{"size": n, "topics": k} for n, k in sorted(histogram.items())],
               "all_documents_outliers": counts.get(-1, 0) == count,
               "assignment_semantics": "fit labels; -1 preserved; no outlier reassignment or topic reduction",
               "strength_semantics": "scalar HDBSCAN membership strength, not a calibrated adequacy probability"}
    atomic_json(artifacts / "topic_summary.json", summary)
    return summary


def run_training(corpus, embeddings, mapping, embedding_manifest, output_dir, *, resume=False,
                 expected_sha=CORPUS_SHA256, expected_count=DOCUMENT_COUNT,
                 min_available_ram_gib=24, min_free_disk_gib=8, parameters=None,
                 trainer=None, loader=None):
    """Trainer/loader injection is for synthetic tests, not a command-line option."""
    corpus, embeddings, mapping, embedding_manifest = map(Path, (corpus, embeddings, mapping, embedding_manifest))
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Completed output/destination already exists: {output}")
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise ValueError("Start Python with PYTHONHASHSEED=42 (setting it after startup is insufficient)")
    if expected_count < 7 or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("Expected corpus count/hash is invalid")
    if any(not math.isfinite(x) or x < 0 for x in (min_available_ram_gib, min_free_disk_gib)):
        raise ValueError("Resource floors must be finite and nonnegative")
    parameters = parameters or topic_configuration()
    if parameters["umap"]["n_neighbors"] >= expected_count:
        raise ValueError("UMAP n_neighbors must be smaller than the corpus count")
    matrix, inputs = validate_inputs(corpus, embeddings, mapping, embedding_manifest, expected_sha, expected_count)
    resources = {"preflight": resource_check(inputs, parameters, output.parent,
                                              min_available_ram_gib, min_free_disk_gib)}
    config = {"schema_version": 1, "seed": SEED, "parameters": parameters, "training_only": True,
              "encoder": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "dimensions": 384,
                          "dtype": "float32", "normalized": True, "max_seq_length": 256},
              "inputs": inputs, "environment": environment(),
              "min_available_ram_gib": min_available_ram_gib, "min_free_disk_gib": min_free_disk_gib,
              "implementation_sha256": file_sha256(__file__),
              "resource_helper_sha256": file_sha256(RESOURCE_HELPER),
              "serialization": "pickle; encoder excluded; fitted UMAP and HDBSCAN prediction data included"}
    config_sha = hashlib.sha256(json_bytes(config)).hexdigest()
    workspace = output.with_name(output.name + ".inprogress")
    output.parent.mkdir(parents=True, exist_ok=True)
    if workspace.is_symlink():
        raise ValueError("Refusing a symlink training workspace")
    workspace.mkdir(exist_ok=resume)
    with job_lock(workspace):
        if os.path.lexists(output):
            raise FileExistsError("Final output appeared during startup")
        config_path = workspace / "job.json"
        if config_path.exists():
            if read_object(config_path) != config:
                raise ValueError("Input/configuration/code/environment changed; refusing incompatible resume")
        else:
            if any(p.name not in {".lock", "job.json.tmp"} for p in workspace.iterdir()):
                raise ValueError("Unrecognized training workspace")
            atomic_json(config_path, config)
        artifacts = workspace / "artifacts"
        artifacts.mkdir(exist_ok=True)
        stage = "validated"
        started = time.perf_counter()
        try:
            atomic_json(workspace / "progress.json", {"status": "running", "stage": stage})
            atomic_json(artifacts / "training_config.json", config)
            atomic_json(artifacts / "input_validation.json", inputs)
            checkpoint_path = workspace / "fit_checkpoint.json"
            model_path = artifacts / "bertopic_model.pkl"
            if checkpoint_path.exists():
                stage = "loading_fitted_checkpoint"
                saved = read_object(checkpoint_path)
                if (saved.get("status") != "fitted" or saved.get("config_sha256") != config_sha or
                        saved.get("model_sha256") != file_sha256(model_path) or
                        saved.get("model_bytes") != model_path.stat().st_size or
                        not isinstance(saved.get("fit_resources"), dict)):
                    raise ValueError("Fitted model checkpoint is corrupt or incompatible")
                resources["fitted_attempt"] = {"resources": saved["fit_resources"],
                                               "elapsed_seconds_through_checkpoint": saved["fit_elapsed_seconds"]}
                # Only an explicitly resumed, hash-checked model from this job is loaded.
                model = (loader or load_model)(model_path)
                model_assignments(model, expected_count)
            else:
                stage = "loading_training_texts"
                digest = hashlib.sha256()
                texts = [text for _, text in documents(corpus, digest)]
                if len(texts) != expected_count or digest.hexdigest() != expected_sha:
                    raise ValueError("Corpus changed before fitting")
                resources["before_fit"] = memory_snapshot()
                remaining = max(256 * 1024 ** 2, resources["preflight"]["planning_ram_bytes"] -
                                inputs["estimated_python_text_bytes"])
                if resources["before_fit"]["available_memory_bytes"] < remaining:
                    raise ValueError("Insufficient RAM remaining after loading training texts")
                stage = "fitting"
                atomic_json(workspace / "progress.json", {"status": "running", "stage": stage,
                                                         "document_count": expected_count})
                print(f"Fitting {expected_count} training-only documents with precomputed embeddings", flush=True)
                model = (trainer or fit_model)(parameters, texts, matrix)
                model_assignments(model, expected_count)
                del texts
                gc.collect()
                resources["after_fit"] = memory_snapshot()
                stage = "saving_fitted_checkpoint"
                if shutil.disk_usage(output.parent).free < resources["preflight"]["required_disk_bytes"]:
                    raise ValueError("Insufficient disk remaining before model serialization")
                temporary = artifacts / "bertopic_model.pkl.tmp"
                model.save(str(temporary), serialization="pickle", save_embedding_model=False)
                sync_file(temporary)
                os.replace(temporary, model_path)
                sync_directory(artifacts)
                saved = {"status": "fitted", "config_sha256": config_sha,
                         "model_sha256": file_sha256(model_path), "model_bytes": model_path.stat().st_size,
                         "fit_resources": resources, "fit_elapsed_seconds": time.perf_counter() - started}
                atomic_json(checkpoint_path, saved)
            stage = "writing_assignments"
            atomic_json(artifacts / "resolved_parameters.json", resolved_parameters(model))
            summary = write_topic_artifacts(artifacts, corpus, model, expected_count, expected_sha)
            del model
            gc.collect()
            stage = "revalidating_inputs"
            verified_matrix, final_inputs = validate_inputs(corpus, embeddings, mapping, embedding_manifest,
                                                            expected_sha, expected_count)
            del verified_matrix
            if final_inputs != inputs:
                raise ValueError("Inputs changed during training")
            resources["after_artifacts"] = memory_snapshot()
            resources["elapsed_seconds_this_attempt"] = time.perf_counter() - started
            atomic_json(artifacts / "resource_report.json", resources)
            artifacts_evidence = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                                  for p in sorted(artifacts.iterdir())
                                  if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")}
            # The fitted file must still be the exact checkpoint whose model was used.
            if artifacts_evidence["bertopic_model.pkl"]["sha256"] != saved["model_sha256"]:
                raise ValueError("Fitted model changed during artifact generation")
            manifest = {"schema_version": 1, "status": "complete", "processed_count": expected_count,
                        "config_sha256": config_sha, "inputs": inputs, "environment": config["environment"],
                        "topic_summary": summary, "artifacts": artifacts_evidence,
                        "prediction_support": "saved UMAP.transform and HDBSCAN prediction_data; pass query embeddings",
                        "reproducibility": "fixed parameters/seeds/threads; identical results across environments not guaranteed"}
            atomic_json(artifacts / "manifest.json", manifest)
            sync_directory(artifacts)
            stage = "publishing"
            if os.path.lexists(output):
                raise FileExistsError("Final output appeared during training")
            os.rename(artifacts, output)
            sync_directory(output.parent)
        except BaseException as error:
            # SIGKILL cannot run this handler; an incomplete directory still has no final path.
            try:
                atomic_json(workspace / "progress.json", {"status": "failed", "stage": stage,
                                                         "error_type": type(error).__name__, "error": str(error)})
            except OSError:
                pass  # A disk-full reporting failure must not mask the original error.
            raise
        shutil.rmtree(workspace)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/at-rag-full-2wiki-topics"))
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--umap-epochs", type=int, default=200)
    parser.add_argument("--min-available-ram-gib", type=float, default=24)
    parser.add_argument("--min-free-disk-gib", type=float, default=8)
    parser.add_argument("--expected-count", type=int, default=DOCUMENT_COUNT)
    parser.add_argument("--expected-corpus-sha256", default=CORPUS_SHA256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        parameters = topic_configuration(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                                         n_neighbors=args.n_neighbors, umap_epochs=args.umap_epochs)
        run_training(args.corpus, args.embeddings, args.mapping, args.embedding_manifest, args.output_dir,
                     resume=args.resume, expected_count=args.expected_count,
                     expected_sha=args.expected_corpus_sha256, parameters=parameters,
                     min_available_ram_gib=args.min_available_ram_gib, min_free_disk_gib=args.min_free_disk_gib)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Topic training failed: {error}. Only a final directory with a complete manifest is usable.",
              file=sys.stderr)
        return 1
    print(f"Completed topic model: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
