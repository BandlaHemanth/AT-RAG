"""Fixed, offline CPU BERTopic pilot; never fits on development or test data.

Run: python -B scripts/run_bertopic_pilot.py
Exit codes: 0 success, 1 failure, 2 insufficient available physical RAM.
Existing research outputs are never overwritten. A directory containing only
preflight.jsonl can be reused after a memory-blocked attempt.
"""

import argparse
from collections import Counter
import ctypes
import gc
import hashlib
from importlib import metadata
from itertools import islice
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import sys
import tempfile
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 42
TRAIN_COUNT = 500
DEV_COUNT = 20
MIN_AVAILABLE_BYTES = 2 * 1024**3


if os.name == "nt":
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in
            ["total_physical", "available_physical", "total_pagefile", "available_pagefile",
             "total_virtual", "available_virtual", "available_extended_virtual"]
        ]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ["peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
             "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private_usage"]
        ]


def _windows_memory_snapshot():
    """Windows OS measurements, not sampled Python-allocation estimates."""
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    result = {"total_physical_bytes": status.total_physical,
              "available_physical_bytes": status.available_physical,
              "available_physical_gib": status.available_physical / 1024**3}
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = ProcessMemory()
    process.cb = ctypes.sizeof(process)
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb):
        result.update({"process_working_set_mib": process.working_set / 1024**2,
                       "process_peak_working_set_mib": process.peak_working_set / 1024**2,
                       "process_private_commit_mib": process.private_usage / 1024**2,
                       "process_peak_commit_mib": process.peak_pagefile / 1024**2})
    else:
        result["process_memory_error"] = str(ctypes.WinError(ctypes.get_last_error()))
    return result


def _read_kib_fields(path):
    """Read Linux /proc fields whose explicitly stated unit is kB (1024 bytes)."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        parts = value.split()
        if separator and len(parts) == 2 and parts[1] == "kB":
            result[key] = int(parts[0]) * 1024
    return result


def _mount_path(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _cgroup_memory_constraints(proc_root):
    """Resolve v1/v2 process membership and inspect all visible ancestor limits.

    memory.current/usage_in_bytes includes other processes and charged cache.
    limit minus usage is therefore conservative container headroom, not RSS or
    an estimate that assumes all page cache can be reclaimed.
    """
    memberships = []
    for line in (proc_root / "self/cgroup").read_text(encoding="utf-8").splitlines():
        _, controllers, group = line.split(":", 2)
        if not controllers:
            memberships.append(("cgroup2", PurePosixPath(group)))
        elif "memory" in controllers.split(","):
            memberships.append(("cgroup", PurePosixPath(group)))
    constraints = []
    visited = set()
    for line in (proc_root / "self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        mount_fields, filesystem_fields = left.split(), right.split()
        if len(mount_fields) < 5 or len(filesystem_fields) < 3:
            continue
        filesystem = filesystem_fields[0]
        if filesystem not in {"cgroup", "cgroup2"}:
            continue
        if filesystem == "cgroup" and "memory" not in filesystem_fields[2].split(","):
            continue
        mount_root = PurePosixPath(_mount_path(mount_fields[3]))
        mount_point = Path(_mount_path(mount_fields[4])).resolve()
        for kind, group in memberships:
            if kind != filesystem:
                continue
            # A cgroup namespace may expose membership relative to its root,
            # while mountinfo retains the original host-side mount root.
            relative = group.relative_to(mount_root) if group.is_relative_to(mount_root) else group.relative_to("/")
            directory = mount_point.joinpath(*relative.parts).resolve()
            if not directory.is_relative_to(mount_point):
                raise RuntimeError(f"Memory cgroup resolves outside its visible mount: {directory}")
            if not directory.is_dir():
                raise RuntimeError(f"Cannot locate process memory cgroup: {directory}")
            while True:
                identity = (kind, str(directory))
                if identity not in visited:
                    visited.add(identity)
                    limit_name, usage_name = (("memory.max", "memory.current") if kind == "cgroup2"
                                              else ("memory.limit_in_bytes", "memory.usage_in_bytes"))
                    limit_path = directory / limit_name
                    if limit_path.is_file():
                        text = limit_path.read_text(encoding="utf-8").strip()
                        limit = None if text == "max" else int(text)
                        # The v1 unlimited sentinel is close to the signed-long maximum.
                        if limit is not None and (kind != "cgroup" or limit < 2**60):
                            usage = int((directory / usage_name).read_text(encoding="utf-8").strip())
                            constraints.append({"version": 2 if kind == "cgroup2" else 1,
                                "directory": str(directory), "limit_bytes": limit,
                                "current_usage_bytes": usage, "headroom_bytes": max(0, limit - usage)})
                if directory == mount_point:
                    break
                directory = directory.parent
    return constraints


def _linux_memory_snapshot(proc_root=Path("/proc")):
    system = _read_kib_fields(proc_root / "meminfo")
    if "MemTotal" not in system or "MemAvailable" not in system:
        raise RuntimeError("Linux memory gate requires /proc/meminfo MemTotal and MemAvailable.")
    constraints = _cgroup_memory_constraints(proc_root)
    system_available = system["MemAvailable"]
    available = min([system_available] + [item["headroom_bytes"] for item in constraints])
    result = {"backend": "linux_proc_and_cgroup", "total_physical_bytes": system["MemTotal"],
              "available_physical_bytes": system_available,
              "available_physical_gib": system_available / 1024**3,
              "available_memory_bytes": available, "available_memory_gib": available / 1024**3,
              "available_memory_scope": "minimum of system availability and visible cgroup headroom",
              "cgroup_constraints": constraints,
              "container_memory_limit_bytes": min((item["limit_bytes"] for item in constraints), default=None),
              "container_memory_headroom_bytes": min((item["headroom_bytes"] for item in constraints), default=None),
              "process_peak_measurement": "kernel-reported VmHWM; not a sampled peak"}
    process = _read_kib_fields(proc_root / "self/status")
    if "VmRSS" in process:
        result["process_rss_mib"] = process["VmRSS"] / 1024**2
    if "VmHWM" in process:
        result["process_peak_rss_mib"] = process["VmHWM"] / 1024**2
    return result


def memory_snapshot():
    """Keep host availability, container headroom, process RSS and OS peaks separate."""
    if sys.platform.startswith("linux"):
        return _linux_memory_snapshot()
    if os.name == "nt":
        result = _windows_memory_snapshot()
        result.update({"backend": "windows_os", "available_memory_scope": "system physical memory",
                       "available_memory_bytes": result["available_physical_bytes"],
                       "available_memory_gib": result["available_physical_gib"],
                       "process_peak_measurement": "OS-reported peak working set and commit; not sampled"})
        return result
    raise RuntimeError(f"Memory preflight supports Windows and Linux, not {sys.platform}.")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def build_corpora():
    """Selection never consults support labels, answers, or reasoning steps."""
    training = []
    seen = set()
    source_questions_read = 0
    contexts_read = 0
    for line_number, question in jsonl_rows(ROOT / "processed_data/2wikimultihopqa/train.jsonl"):
        source_questions_read += 1
        for position, context in enumerate(question["contexts"]):
            contexts_read += 1
            text = context["paragraph_text"]
            if not isinstance(text, str) or not text.strip() or text in seen:
                continue
            seen.add(text)  # Exact text identity; no normalization or truncation here.
            training.append({"document_id": document_id(text), "split": "train",
                             "source_line": line_number, "question_id": question.get("question_id"),
                             "idx": context.get("idx"), "context_position": position,
                             "title": context.get("title"), "paragraph_text": text})
            if len(training) == TRAIN_COUNT:
                break
        if len(training) == TRAIN_COUNT:
            break
    if len(training) != TRAIN_COUNT:
        raise RuntimeError(f"Only {len(training)} unique nonempty training paragraphs found.")
    dev_rows = list(islice(jsonl_rows(ROOT / "processed_data/2wikimultihopqa/dev_subsampled.jsonl"), DEV_COUNT))
    if len(dev_rows) != DEV_COUNT:
        raise RuntimeError("Fewer than 20 development questions available.")
    dev = [row for _, row in dev_rows]  # Preserve the full original records for later scoring.
    dev_ids = [row["question_id"] for row in dev]
    if len(set(dev_ids)) != DEV_COUNT:
        raise RuntimeError("Duplicate development question identifiers.")
    if set(dev_ids) & {row["question_id"] for row in training}:
        raise RuntimeError("Selected training and development question identifiers overlap.")
    candidates = [{"document_id": row["document_id"], "paragraph_text": row["paragraph_text"],
                   "origin_splits": ["train"]} for row in training]
    positions = {row["paragraph_text"]: i for i, row in enumerate(candidates)}
    contexts = []
    for question in dev:
        if not isinstance(question["question_text"], str) or not question["question_text"].strip():
            raise RuntimeError("Empty development query.")
        for position, context in enumerate(question["contexts"]):
            text = context["paragraph_text"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Empty development context; cannot silently omit a candidate.")
            if text not in positions:
                positions[text] = len(candidates)
                candidates.append({"document_id": document_id(text), "paragraph_text": text,
                                   "origin_splits": ["development_context"]})
            candidate = candidates[positions[text]]
            if "development_context" not in candidate["origin_splits"]:
                candidate["origin_splits"].append("development_context")
            contexts.append({"occurrence_id": f"dev:{question['question_id']}:{position}",
                             "question_id": question["question_id"], "idx": context.get("idx"),
                             "context_position": position, "title": context.get("title"),
                             "paragraph_text": text, "document_id": candidate["document_id"],
                             "candidate_position": positions[text]})
    return training, dev, contexts, candidates, {
        "training_source_questions_read": source_questions_read,
        "training_source_contexts_read": contexts_read,
        "dev_source_lines": [line for line, _ in dev_rows],
    }


def verify_encoder(directory):
    manifest = json.loads((directory / "embedding_verification_manifest.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise RuntimeError("Encoder identifier/revision does not match the fixed official model.")
    required = {"config.json", "modules.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "1_Pooling/config.json", "sentence_bert_config.json"}
    if not required.issubset(manifest["sha256"]):
        raise RuntimeError("Incomplete verified encoder manifest.")
    for name, expected in manifest["sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Missing or changed official encoder file: {name}")
    return manifest


def execute(output, encoder_dir, report):
    timings = report["timings_seconds"]
    stage = "dataset_selection"
    try:
        started = time.perf_counter()
        training, dev, contexts, candidates, selection = build_corpora()
        timings[stage] = time.perf_counter() - started
        report.update(selection)
        report.update({"training_paragraphs": len(training), "development_questions": len(dev),
                       "development_context_occurrences": len(contexts),
                       "unique_candidate_paragraphs": len(candidates)})
        write_jsonl(output / "training_manifest.jsonl", training)
        write_jsonl(output / "development_questions.jsonl", dev)
        write_jsonl(output / "development_context_manifest.jsonl", contexts)
        write_jsonl(output / "candidate_corpus.jsonl", candidates)
        stage = "encoder_validation"
        write_json(output / "embedding_model.json", verify_encoder(encoder_dir))
        report["package_versions"] = {}
        for package in ["bertopic", "sentence-transformers", "torch", "umap-learn", "hdbscan",
                        "scikit-learn", "numpy", "scipy", "pandas", "numba", "huggingface-hub"]:
            report["package_versions"][package] = metadata.version(package)
        write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "packages": report["package_versions"]})

        stage = "embedding_dependencies"
        started = time.perf_counter()
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        timings[stage] = time.perf_counter() - started
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        np.random.seed(SEED)
        random.seed(SEED)
        stage = "encoder_loading"
        started = time.perf_counter()
        encoder = SentenceTransformer(str(encoder_dir), device="cpu", local_files_only=True,
                                      trust_remote_code=False)
        encoder.max_seq_length = 256
        encoder.eval()
        timings[stage] = time.perf_counter() - started
        stage = "embedding_generation"
        started = time.perf_counter()
        with torch.inference_mode():
            candidate_embeddings = encoder.encode([row["paragraph_text"] for row in candidates],
                batch_size=4, device="cpu", normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
            query_embeddings = encoder.encode([row["question_text"] for row in dev],
                batch_size=4, device="cpu", normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
        timings[stage] = time.perf_counter() - started
        for embeddings, count in [(candidate_embeddings, len(candidates)), (query_embeddings, len(dev))]:
            if embeddings.shape != (count, 384) or not np.isfinite(embeddings).all():
                raise RuntimeError("Invalid embedding shape or nonfinite values.")
            if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5, rtol=0):
                raise RuntimeError("Embeddings are not normalized.")
        np.save(output / "candidate_embeddings.npy", candidate_embeddings, allow_pickle=False)
        np.save(output / "query_embeddings.npy", query_embeddings, allow_pickle=False)
        report["memory"]["after_encoding"] = memory_snapshot()
        del encoder
        gc.collect()

        stage = "clustering_dependencies"
        started = time.perf_counter()
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        timings[stage] = time.perf_counter() - started
        model = BERTopic(embedding_model=None,
            umap_model=UMAP(n_components=5, n_neighbors=15, metric="cosine", min_dist=0.0,
                            random_state=SEED, transform_seed=SEED, low_memory=True, n_jobs=1),
            hdbscan_model=HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True,
                                  metric="euclidean", core_dist_n_jobs=1),
            vectorizer_model=CountVectorizer(stop_words="english", min_df=2, max_features=5000,
                                              ngram_range=(1, 1)),
            nr_topics=None, calculate_probabilities=False, low_memory=True, verbose=False)
        stage = "fitting"
        print("FITTING: 500 training-only paragraphs; development embeddings are excluded.", flush=True)
        report["memory"]["before_fitting"] = memory_snapshot()
        started = time.perf_counter()
        # Candidates start with exactly the selected training texts. The slice is a view.
        fit_topics, fit_probabilities = model.fit_transform(
            [row["paragraph_text"] for row in training], embeddings=candidate_embeddings[:TRAIN_COUNT])
        timings[stage] = time.perf_counter() - started
        del fit_probabilities
        counts = Counter(int(topic) for topic in fit_topics)
        report.update({"topics_discovered_excluding_outliers": len([topic for topic in counts if topic != -1]),
                       "training_outlier_count": counts.get(-1, 0),
                       "training_outlier_proportion": counts.get(-1, 0) / TRAIN_COUNT,
                       "training_topic_counts": dict(sorted(counts.items()))})
        if counts.get(-1, 0) == TRAIN_COUNT:
            report["experiment_warnings"].append("All training paragraphs are outliers; no parameters were retuned.")
        model.get_topic_info().to_csv(output / "topic_info.csv", index=False)
        write_jsonl(output / "training_assignments.jsonl", (
            {"document_id": row["document_id"], "question_id": row["question_id"],
             "idx": row["idx"], "fitted_topic": int(topic)} for row, topic in zip(training, fit_topics)))
        report["memory"]["after_fitting"] = memory_snapshot()

        stage = "frozen_topic_assignment"
        print("ASSIGNING: frozen model; no fit, update, or topic reduction on development data.", flush=True)
        started = time.perf_counter()
        candidate_topics, candidate_probabilities = model.transform(
            [row["paragraph_text"] for row in candidates], embeddings=candidate_embeddings)
        query_topics, query_probabilities = model.transform(
            [row["question_text"] for row in dev], embeddings=query_embeddings)
        timings[stage] = time.perf_counter() - started
        del candidate_probabilities, query_probabilities
        write_jsonl(output / "candidate_topic_assignments.jsonl", (
            {"document_id": row["document_id"], "topic": int(topic), "bertopic": f"Topic {int(topic)}"}
            for row, topic in zip(candidates, candidate_topics)))
        write_jsonl(output / "query_topic_assignments.jsonl", (
            {"question_id": row["question_id"], "question_text": row["question_text"],
             "topic": int(topic), "bertopic": f"Topic {int(topic)}"} for row, topic in zip(dev, query_topics)))
        write_jsonl(output / "development_context_topic_assignments.jsonl", (
            {**row, "topic": int(candidate_topics[row["candidate_position"]]),
             "bertopic": f"Topic {int(candidate_topics[row['candidate_position']])}"} for row in contexts))
        report["assignment_counts"] = {"queries": len(query_topics), "candidate_paragraphs": len(candidate_topics),
                                       "development_context_occurrences": len(contexts)}
        stage = "model_serialization"
        started = time.perf_counter()
        # Pickle preserves UMAP and HDBSCAN/prediction_data. Compact tensor serialization
        # would not preserve this exact clustering inference pipeline.
        model.save(str(output / "bertopic_model.pkl"), serialization="pickle", save_embedding_model=False)
        timings[stage] = time.perf_counter() - started
        write_json(output / "artifact_hashes.json", {
            path.name: sha256_file(path) for path in output.iterdir()
            if path.is_file() and path.name not in {"run_report.json", "preflight.jsonl", "artifact_hashes.json"}})
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"stage": stage, "type": type(exc).__name__,
                           "message": str(exc), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/bertopic_pilot_500")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(path.name != "preflight.jsonl" for path in output.iterdir()):
        print(f"STOP: output directory already contains research artifacts; nothing overwritten: {output}")
        return 1
    before = memory_snapshot()  # Gate precedes dataset reads and heavyweight imports.
    output.mkdir(parents=True, exist_ok=True)
    preflight = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "minimum_available_bytes": MIN_AVAILABLE_BYTES, "memory": before,
                 "status": "ready" if before["available_memory_bytes"] >= MIN_AVAILABLE_BYTES else "blocked_memory"}
    with (output / "preflight.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(preflight) + "\n")
    print(f"Available memory ({before['available_memory_scope']}): "
          f"{before['available_memory_gib']:.3f} GiB; required: 2.000 GiB.", flush=True)
    if preflight["status"] != "ready":
        print("STOP: free memory before retrying. No dataset selection, encoding, fitting, or assignment was run.")
        return 2

    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        os.environ[name] = "2"
    os.environ["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "at_rag_readiness_numba")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONHASHSEED"] = str(SEED)
    # Python hash randomization is fixed at interpreter startup, not by runtime env changes.
    # Re-exec once, before dependencies or dataset reads, if the caller omitted it.
    if getattr(sys.flags, "hash_randomization", 0) and os.environ.get("AT_RAG_PILOT_SEEDED") != "1":
        os.environ["AT_RAG_PILOT_SEEDED"] = "1"
        os.execv(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]])
    config = {"seed": SEED, "training_paragraphs": TRAIN_COUNT, "development_questions": DEV_COUNT,
        "selection": "first unique nonempty exact training texts in deterministic file/context order",
        "training_path": str(ROOT / "processed_data/2wikimultihopqa/train.jsonl"),
        "development_path": str(ROOT / "processed_data/2wikimultihopqa/dev_subsampled.jsonl"),
        "held_out_test_used": False, "development_used_for_fitting_or_tuning": False,
        "embedding": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu",
                      "batch_size": 4, "max_seq_length": 256, "normalized": True, "dtype": "float32"},
        "umap": {"n_components": 5, "n_neighbors": 15, "metric": "cosine", "random_state": SEED,
                 "transform_seed": SEED, "low_memory": True, "n_jobs": 1, "min_dist": 0.0},
        "hdbscan": {"min_cluster_size": 10, "min_samples": 5, "prediction_data": True,
                    "metric": "euclidean", "core_dist_n_jobs": 1},
        "vectorizer": {"stop_words": "english", "min_df": 2, "max_features": 5000, "ngram_range": [1, 1]},
        "bertopic": {"embedding_model": None, "nr_topics": None, "calculate_probabilities": False,
                     "low_memory": True, "verbose": False},
        "torch_threads": 2, "torch_interop_threads": 1, "serialization": "pickle; encoder excluded",
        "unspecified_library_parameters": "installed-version defaults; see environment.json",
        "assignment_policy": "frozen model.transform for every candidate and query; retain topic -1 unchanged",
        "embedding_rows": "candidate_corpus.jsonl and development_questions.jsonl order respectively",
        "script_sha256": sha256_file(Path(__file__))}
    write_json(output / "config.json", config)
    report = {"status": "running", "started_utc": preflight["utc"], "timings_seconds": {},
              "memory": {"preflight": before}, "experiment_warnings": [],
              "memory_measurement_scope": "OS-reported process peaks, not sampled or whole-system peaks; "
                                          "Linux cgroup usage includes all group processes and charged cache"}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        execute(output, ROOT / "artifacts/embedding_models/all-MiniLM-L6-v2", report)
        report["library_warnings"] = [{"category": item.category.__name__, "message": str(item.message)} for item in captured]
    report["timings_seconds"]["total_execution"] = time.perf_counter() - started
    report["memory"]["final"] = memory_snapshot()
    write_json(output / "run_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
