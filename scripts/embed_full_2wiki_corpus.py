"""Reproducible, resumable full-corpus MiniLM encoding (Linux/Kaggle).

Only a final directory containing manifest.json with status=complete is a
successful artifact. See docs/full_2wiki_embeddings.md for commands and recovery.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CORPUS_SHA256 = "f1824d1dc9275731243c6f5a89cc4d702c77b3607413343c9c75bb242fff3160"
DOCUMENT_COUNT = 396_978
DIMENSIONS = 384
DTYPE = np.dtype("<f4")
MAX_SEQUENCE_LENGTH = 256
SEED = 42


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_file(path):
    with Path(path).open("rb") as stream:
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
    """Yield one paragraph at a time; hash the original JSONL bytes, not JSON."""
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
                text, identifier = row["paragraph_text"], row["document_id"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("expected a nonempty paragraph_text string")
                if identifier != hashlib.sha256(text.encode("utf-8")).hexdigest():
                    raise ValueError("document_id does not match exact UTF-8 paragraph_text")
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                raise ValueError(f"Invalid corpus document at source line {line_number}: {error}") from error
            yield identifier, text


def verify_corpus(source, summary_path, expected_sha256, expected_count):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (not isinstance(summary, dict) or summary.get("documents_sha256") != expected_sha256 or
            summary.get("unique_paragraphs") != expected_count):
        raise ValueError("Companion summary disagrees with expected corpus checksum/count")
    digest = hashlib.sha256()
    count = sum(1 for _ in documents(source, digest))
    if count != expected_count or digest.hexdigest() != expected_sha256:
        raise ValueError(f"Corpus verification failed: count={count}, sha256={digest.hexdigest()}")
    return file_sha256(summary_path)


def canonical_device(device):
    if device == "cuda":
        return "cuda:0"
    if device != "cpu" and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("Device must be cpu, cuda, or cuda:N; automatic fallback is disabled")
    return device


def runtime_environment(device):
    # Set before importing torch/initializing CUDA; retain and record an explicit
    # caller setting rather than quietly changing their execution environment.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    versions = {}
    for package in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers",
                    "huggingface-hub", "safetensors", "scipy", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    identity = {"machine": platform.machine(), "processor": platform.processor()}
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no CPU fallback was performed")
        index = int(device.split(":")[1])
        properties = torch.cuda.get_device_properties(index)
        identity.update({"gpu_name": properties.name, "gpu_capability": [properties.major, properties.minor],
                         "gpu_memory_bytes": properties.total_memory,
                         "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()})
    return {"python": platform.python_version(), "platform": platform.platform(),
            "libraries": versions, "device_identity": identity,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def load_encoder(device):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    encoder = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device,
                                  trust_remote_code=False)
    encoder.max_seq_length = MAX_SEQUENCE_LENGTH
    encoder.eval()
    if encoder.get_sentence_embedding_dimension() != DIMENSIONS:
        raise ValueError("Pinned encoder returned an unexpected embedding dimension")
    return encoder


@contextmanager
def job_lock(workspace):
    # flock is released by the OS after crashes, unlike a stale PID lock file.
    with (workspace / ".lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is using this embedding workspace") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_vectors(vectors, count):
    if vectors.shape != (count, DIMENSIONS) or vectors.dtype != DTYPE:
        raise ValueError("Encoder/output must contain 384-dimensional float32 rows")
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite embedding values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Embeddings are not L2-normalized")


def checkpoint(workspace, matrix, mapping, count, embedding_digest, mapping_digest, config_sha):
    # Data must be durable BEFORE the checkpoint may claim these rows committed.
    matrix.flush()
    sync_file(workspace / "artifacts/embeddings.npy")
    mapping.flush()
    os.fsync(mapping.fileno())
    sync_directory(workspace / "artifacts")
    state = {"schema_version": 1, "config_sha256": config_sha, "processed_count": count,
             "mapping_bytes": mapping.tell(), "embedding_prefix_sha256": embedding_digest.hexdigest(),
             "mapping_prefix_sha256": mapping_digest.hexdigest()}
    atomic_json(workspace / "checkpoint.json", state)
    return state


def restore(workspace, expected_count, config_sha):
    state = json.loads((workspace / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("Invalid progress checkpoint: expected an object")
    count, offset = state.get("processed_count"), state.get("mapping_bytes")
    if (state.get("schema_version") != 1 or state.get("config_sha256") != config_sha or
            type(count) is not int or not 0 <= count <= expected_count or
            type(offset) is not int or offset < 0 or
            any(not isinstance(state.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", state[key])
                for key in ("embedding_prefix_sha256", "mapping_prefix_sha256"))):
        raise ValueError("Invalid or incompatible progress checkpoint")
    matrix_path = workspace / "artifacts/embeddings.npy"
    matrix = np.load(matrix_path, mmap_mode="r+", allow_pickle=False)
    if (matrix.shape != (expected_count, DIMENSIONS) or matrix.dtype != DTYPE or
            not matrix.flags.c_contiguous or matrix_path.stat().st_size != matrix.offset + matrix.nbytes):
        raise ValueError("Invalid embedding matrix header/size")
    embedding_digest, mapping_digest = hashlib.sha256(), hashlib.sha256()
    for start in range(0, count, 1024):
        block = matrix[start:min(start + 1024, count)]
        validate_vectors(block, len(block))
        embedding_digest.update(block.tobytes(order="C"))
    mapping_path = workspace / "artifacts/document_ids.jsonl"
    with mapping_path.open("rb") as stream:
        for row in range(count):
            raw = stream.readline()
            item = json.loads(raw)
            if (not isinstance(item, dict) or item.get("row") != row or
                    not isinstance(item.get("document_id"), str) or
                    not re.fullmatch(r"[0-9a-f]{64}", item["document_id"])):
                raise ValueError("Invalid committed document-ID mapping")
            mapping_digest.update(raw)
        if stream.tell() != offset:
            raise ValueError("Checkpoint mapping offset disagrees with committed rows")
    if (embedding_digest.hexdigest() != state["embedding_prefix_sha256"] or
            mapping_digest.hexdigest() != state["mapping_prefix_sha256"]):
        raise ValueError("Committed artifact prefix checksum mismatch; refusing resume")
    return matrix, state, embedding_digest, mapping_digest


def run_job(input_path, output_dir, *, summary_path=None, batch_size=64, device="cpu",
            resume=False, checkpoint_batches=16, expected_sha256=CORPUS_SHA256,
            expected_count=DOCUMENT_COUNT, encoder_factory=None):
    """Encode a verified corpus; encoder_factory is a test seam, not a CLI option."""
    source, output = Path(input_path).absolute(), Path(output_dir).absolute()
    device = canonical_device(device)
    if batch_size < 1 or checkpoint_batches < 1 or expected_count < 1:
        raise ValueError("Batch size, checkpoint interval and expected count must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 must be 64 lowercase hexadecimal characters")
    if os.path.lexists(output):
        raise FileExistsError(f"Final output already exists; it will not be overwritten: {output}")
    summary_path = Path(summary_path) if summary_path else source.with_name("summary.json")
    summary_sha = verify_corpus(source, summary_path, expected_sha256, expected_count)
    config = {"schema_version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "corpus_sha256": expected_sha256, "document_count": expected_count,
              "corpus_summary_sha256": summary_sha, "dimensions": DIMENSIONS, "dtype": "float32",
              "byte_order": "little",
              "normalize_embeddings": True, "max_seq_length": MAX_SEQUENCE_LENGTH,
              "preprocessing": "paragraph_text unchanged; no title/prompt prefix",
              "batch_size": batch_size, "checkpoint_batches": checkpoint_batches, "device": device,
              "seed": SEED, "deterministic_algorithms": True, "allow_tf32": False,
              "torch_threads": 2, "torch_interop_threads": 1,
              "environment": runtime_environment(device), "implementation_sha256": file_sha256(__file__)}
    config_sha = hashlib.sha256(json_bytes(config)).hexdigest()
    workspace = output.with_name(output.name + ".inprogress")
    output.parent.mkdir(parents=True, exist_ok=True)
    if workspace.is_symlink():
        raise ValueError("Refusing a symlink embedding workspace")
    workspace.mkdir(exist_ok=resume)
    with job_lock(workspace):
        if os.path.lexists(output):
            raise FileExistsError("Final output appeared while waiting for the workspace")
        config_path = workspace / "job.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Input/model/configuration/environment differs from the saved job; choose a new output")
        else:
            if any(path.name not in {".lock", "job.json.tmp"} for path in workspace.iterdir()):
                raise ValueError("Unrecognized workspace without job configuration")
            atomic_json(config_path, config)
        artifacts = workspace / "artifacts"
        artifacts.mkdir(exist_ok=True)
        if not (workspace / "checkpoint.json").exists():
            # Initialization crashed before the first checkpoint: these zero-row
            # scratch files belong to this verified configuration and can be reset.
            if any(path.name not in {"embeddings.npy", "document_ids.jsonl"} for path in artifacts.iterdir()):
                raise ValueError("Unrecognized artifacts without an initial checkpoint")
            matrix = np.lib.format.open_memmap(artifacts / "embeddings.npy", mode="w+", dtype=DTYPE,
                                               shape=(expected_count, DIMENSIONS))
            with (artifacts / "document_ids.jsonl").open("w+b") as mapping:
                checkpoint(workspace, matrix, mapping, 0, hashlib.sha256(), hashlib.sha256(), config_sha)
            del matrix
        matrix, state, embedding_digest, mapping_digest = restore(workspace, expected_count, config_sha)
        processed = state["processed_count"]
        encoder = None
        pending_texts, pending_ids = [], []
        source_digest = hashlib.sha256()
        seen = 0
        batches_since_checkpoint = 0
        try:
            with (artifacts / "document_ids.jsonl").open("r+b") as mapping:
                # Validate the saved prefix before truncating any uncommitted tail.
                for identifier, text in documents(source, source_digest):
                    seen += 1
                    if seen > expected_count:
                        raise ValueError("Input grew beyond the verified document count")
                    if seen <= state["processed_count"]:
                        if json.loads(mapping.readline()) != {"row": seen - 1, "document_id": identifier}:
                            raise ValueError("Committed document-ID order differs from input")
                        continue
                    if not pending_texts:
                        mapping.truncate(mapping.tell())
                    pending_texts.append(text)
                    pending_ids.append(identifier)
                    if len(pending_texts) < batch_size and seen < expected_count:
                        continue
                    if encoder is None:
                        encoder = (encoder_factory or load_encoder)(device)
                    vectors = np.asarray(encoder.encode(
                        pending_texts, batch_size=batch_size, device=device, normalize_embeddings=True,
                        precision="float32", convert_to_numpy=True, show_progress_bar=False))
                    validate_vectors(vectors, len(pending_texts))
                    matrix[processed:processed + len(vectors)] = vectors
                    embedding_digest.update(np.asarray(vectors, dtype=DTYPE).tobytes(order="C"))
                    for index, doc_id in enumerate(pending_ids, processed):
                        raw = json_bytes({"row": index, "document_id": doc_id})
                        mapping.write(raw)
                        mapping_digest.update(raw)
                    processed += len(vectors)
                    pending_texts.clear()
                    pending_ids.clear()
                    batches_since_checkpoint += 1
                    if batches_since_checkpoint == checkpoint_batches:
                        checkpoint(workspace, matrix, mapping, processed, embedding_digest, mapping_digest, config_sha)
                        batches_since_checkpoint = 0
                        print(f"Checkpoint: {processed}/{expected_count} documents", flush=True)
                if seen != expected_count or source_digest.hexdigest() != expected_sha256:
                    raise ValueError("Corpus checksum/count changed during encoding; output is incomplete")
                if file_sha256(summary_path) != summary_sha:
                    raise ValueError("Companion summary changed during encoding")
                mapping.truncate(mapping.tell())
                checkpoint(workspace, matrix, mapping, processed, embedding_digest, mapping_digest, config_sha)
        finally:
            del matrix
        if processed != expected_count:
            raise ValueError("Embedding count does not match the verified corpus")
        # Re-read the durable files rather than trusting in-memory hashes alone.
        matrix, state, _, _ = restore(workspace, expected_count, config_sha)
        del matrix
        manifest = {"schema_version": 1, "status": "complete", "processed_count": processed,
                    "configuration": config, "config_sha256": config_sha,
                    "embedding_data_sha256": state["embedding_prefix_sha256"],
                    "artifacts": {name: {"sha256": file_sha256(artifacts / name),
                                         "bytes": (artifacts / name).stat().st_size}
                                  for name in ("embeddings.npy", "document_ids.jsonl")}}
        atomic_json(artifacts / "manifest.json", manifest)
        sync_directory(artifacts)
        if os.path.lexists(output):
            raise FileExistsError("Final output appeared during processing")
        os.rename(artifacts, output)  # Existing nonempty completed directories cannot be replaced.
        sync_directory(output.parent)
        shutil.rmtree(workspace)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Defaults to summary.json beside the input")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/full_2wiki_embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N (no automatic fallback)")
    parser.add_argument("--checkpoint-batches", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-sha256", default=CORPUS_SHA256)
    parser.add_argument("--expected-count", type=int, default=DOCUMENT_COUNT)
    args = parser.parse_args(argv)
    try:
        run_job(args.input, args.output_dir, summary_path=args.summary, batch_size=args.batch_size,
                device=args.device, resume=args.resume, checkpoint_batches=args.checkpoint_batches,
                expected_sha256=args.expected_sha256, expected_count=args.expected_count)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Embedding job failed: {error}. Only a final directory with a complete manifest is usable.",
              file=sys.stderr)
        return 1
    print(f"Completed embeddings: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
