"""Preflight or run the unedited 2Wiki Ingestor with local MiniLM embeddings.

Preflight is the default and reads only files/package metadata. Execution is
explicitly opt-in and belongs in Kaggle, not the Codex Cloud preparation task.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "compat/minilm/utils.py"
PROFILE_PATH = ROOT / "compat/minilm/requirements-py312.txt"
DATASET_SHA256 = "4948757b90627f40b77cd6bf81a62c1a7ad525d6a6a0c91b17a2e0e2fa88f5bc"
TOPIC_MODEL_SHA256 = "4a626723a82563556a7567706b742b7c8ed4444efefe1a411c23d7554078749c"
SOURCE_SHA256 = {
    "vectorDB/dataset_ingestion.py": "027f059b61031ad54ff7ce849a63ede1a06401283e4c5fb54e48f87c8240b212",
    "vectorDB/train_topic_model.py": "19cc7321a32293171683bddb48705811b3f187de7fd1c5f5b6296c88cbcc20b6",
    "models/AT_RAG.py": "dbefa9fc12333b1a7c47167087a5ff4d08e5a83c7aaf9cc5c2d7f2842ca9108c",
}


def load_module(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# This module is deliberately lightweight; its encoder is loaded only on encode.
ADAPTER = load_module("_at_rag_minilm", ADAPTER_PATH)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_inputs(dataset, topic_model, destination):
    dataset = Path(dataset).resolve(strict=True)
    topic_model = Path(topic_model).resolve(strict=True)
    if Path(destination).is_symlink():
        raise ValueError("Destination must not be a symlink")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination must not exist; the original ingestor appends records")
    if not dataset.is_file() or file_sha256(dataset) != DATASET_SHA256:
        raise ValueError("Dataset SHA-256 differs from the verified original evaluation subset")
    if not topic_model.is_file() or file_sha256(topic_model) != TOPIC_MODEL_SHA256:
        raise ValueError("Saved BERTopic SHA-256 differs; the custom full-corpus model is not accepted")
    for relative, expected in SOURCE_SHA256.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"Original source changed: {relative}")
    questions = contexts = 0
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            questions += 1
            contexts += len(sample["contexts"])
    if (questions, contexts) != (500, 5000):
        raise ValueError(f"Expected 500 questions/5000 occurrences, got {questions}/{contexts}")
    return {
        "dataset": str(dataset), "dataset_sha256": DATASET_SHA256,
        "questions": questions, "context_occurrences": contexts,
        "topic_model": str(topic_model), "topic_model_sha256": TOPIC_MODEL_SHA256,
        "persist_directory": str(destination), "original_source_sha256": SOURCE_SHA256,
    }


def inspect_environment():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the dedicated Python 3.12 baseline environment")
    versions = {}
    for line in PROFILE_PATH.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==")
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise ValueError(f"Missing dependency: {name}=={expected}") from error
        versions[name] = actual
        if actual != expected:
            raise ValueError(f"Dependency mismatch: {name}=={actual}; profile requires {expected}")
    # Record, but do not guess or replace, the saved model's scientific stack.
    for name in ("torch", "pandas", "scipy", "scikit-learn", "hdbscan", "umap-learn", "numba", "llvmlite"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise ValueError(f"Missing saved-model dependency: {name}") from error
    return {"python": platform.python_version(), "packages": versions}


@contextmanager
def original_ingestor():
    """Supply the missing module and import-only API placeholder, then restore."""
    previous_path = sys.path[:]
    sentinel = object()
    names = ("utils", "train_topic_model")
    previous_modules = {name: sys.modules.get(name, sentinel) for name in names}
    if previous_modules["utils"] is not sentinel:
        raise RuntimeError("Run in a fresh process; a top-level utils module is already loaded")
    if previous_modules["train_topic_model"] is not sentinel:
        raise RuntimeError("Run in a fresh process; train_topic_model is already loaded")
    had_key = "OPENAI_API_KEY" in os.environ
    try:
        sys.path[:0] = [str(ROOT / "vectorDB"), str(ROOT)]
        sys.modules["utils"] = ADAPTER
        # Original ingestion reads this at import. Local MiniLM never uses it.
        if not had_key:
            os.environ["OPENAI_API_KEY"] = "unused-local-minilm-import-only"
        module = load_module("_at_rag_original_ingestion", ROOT / "vectorDB/dataset_ingestion.py")
        if not had_key:
            del os.environ["OPENAI_API_KEY"]
        yield module.Ingestor
    finally:
        if not had_key:
            os.environ.pop("OPENAI_API_KEY", None)
        sys.path[:] = previous_path
        for name, previous in previous_modules.items():
            if previous is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def create_original_store(dataset, topic_model, destination):
    """Call the original create_vectordb verbatim; never fit a topic model."""
    if Path(destination).is_symlink():
        raise ValueError("Destination must not be a symlink")
    dataset, topic_model, destination = (Path(p).resolve() for p in (dataset, topic_model, destination))
    destination.mkdir(parents=True, exist_ok=False)
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="at-rag-original-ingestion-") as working:
            # Original BERTopicTrainer() loads the relative name bertopic_model.
            # A symlink supports arbitrary read-only Kaggle backup filenames.
            (Path(working) / "bertopic_model").symlink_to(topic_model)
            os.chdir(working)
            with original_ingestor() as Ingestor:
                ingestor = Ingestor(
                    dataset_path=str(dataset), persist_directory=str(destination),
                    openai_api_key="unused-by-local-minilm",
                )
                return ingestor.create_vectordb()
    finally:
        os.chdir(previous_cwd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "processed_data/2wikimultihopqa/test_subsampled.jsonl")
    parser.add_argument("--topic-model", type=Path, required=True, help="The saved model file, not its containing directory")
    parser.add_argument("--persist-directory", type=Path, default=ROOT / "vectorDB/2wikimultihopqa_minilm")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Read-only validation (the default)")
    modes.add_argument("--execute", action="store_true", help="Explicitly create the index in Kaggle")
    args = parser.parse_args(argv)
    try:
        inputs = inspect_inputs(args.dataset, args.topic_model, args.persist_directory)
        environment = inspect_environment()
        report = {
            "status": "preflight_passed", **inputs, "environment": environment,
            "retrieval_embedding_substitution": ADAPTER.embedding_settings(),
            "compatibility_file_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in (ADAPTER_PATH, PROFILE_PATH, Path(__file__).resolve())
            },
            "ingestion_batch_size": 1000, "deduplication": False,
            "limitation": "File/package checks only; saved-model unpickling and real MiniLM/Chroma execution remain untested until Kaggle execution",
        }
        if args.execute:
            store = create_original_store(args.dataset, args.topic_model, args.persist_directory)
            count = store._collection.count()
            if count != 5000:
                raise ValueError(f"Chroma contains {count} records; expected 5000")
            report.update(status="ingestion_completed", stored_records=count)
            report["limitation"] = "Scientific dependency versions must still be matched to the saved training-environment backup"
            manifest = args.persist_directory / "minilm_ingestion_manifest.json"
            manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Ingestion preparation failed: {error}", file=sys.stderr)
        if args.execute:
            print("Any partial destination is retained. Use a fresh destination for a retry; do not append.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
