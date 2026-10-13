"""Verify the official MiniLM encoder on CPU, including a fresh offline reload.

Run with the Python environment containing sentence-transformers:
    python -B scripts/verify_local_embeddings.py

Only files from sentence-transformers/all-MiniLM-L6-v2 may be downloaded.
The first successful run populates artifacts/embedding_models/all-MiniLM-L6-v2.
Subsequent runs verify its manifest and load offline, without contacting the Hub.
No repository training or retrieval modules are imported.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "artifacts" / "embedding_models" / "all-MiniLM-L6-v2"
MANIFEST_NAME = "embedding_verification_manifest.json"
MAX_SEQUENCE_LENGTH = 256
SENTENCES = [
    "The cat sits on the mat.",
    "Multi-hop questions require evidence from multiple documents.",
]
# Exclude alternate weight formats, ONNX/OpenVINO exports, and unrelated assets.
MODEL_FILES = [
    "config.json", "config_sentence_transformers.json", "modules.json",
    "sentence_bert_config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    "1_Pooling/config.json",
]
RESULT_PREFIX = "VERIFICATION_RESULT="


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(model_dir):
    manifest = json.loads((model_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("Local manifest identifies a different model or revision.")
    files = manifest.get("sha256", {})
    if not files or any(name not in MODEL_FILES for name in files):
        raise RuntimeError("Invalid local model manifest file list.")
    for name, expected_hash in files.items():
        path = model_dir / name
        if not path.is_file() or path.is_symlink() or file_hash(path) != expected_hash:
            raise RuntimeError(f"Local model file is missing, changed, or a symlink: {name}")
    return manifest


def worker(args):
    """Run in a fresh process; set offline flags before importing ML libraries."""
    offline = args.worker == "offline"
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    result = {
        "status": "failed", "mode": args.worker, "model_id": MODEL_ID,
        "requested_revision": args.revision, "resolved_revision": args.revision,
        "model_directory": str(args.model_dir),
        "python_version": sys.version, "python_executable": sys.executable,
        "device": "cpu", "batch_size": 2, "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalize_embeddings": True, "sentences": SENTENCES,
    }
    stage = "dependency_import"
    try:
        print(f"STAGE {args.worker}: importing dependencies", flush=True)
        import numpy as np
        import torch
        import sentence_transformers
        from huggingface_hub import __version__ as hub_version, snapshot_download
        from sentence_transformers import SentenceTransformer

        result.update({"pytorch_version": torch.__version__,
                       "sentence_transformers_version": sentence_transformers.__version__,
                       "cuda_available": torch.cuda.is_available(),
                       "huggingface_hub_version": hub_version})
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        np.random.seed(42)
        torch.use_deterministic_algorithms(True)

        stage = "local_manifest_validation" if offline else "official_snapshot_download"
        print(f"STAGE {args.worker}: {'verifying local files' if offline else 'downloading official files to local directory'}", flush=True)
        started = time.perf_counter()
        snapshot = args.model_dir
        if offline:
            manifest = verify_manifest(snapshot)
            result["download_method"] = "none; verified local directory; offline flags enabled"
            result["original_download_method"] = manifest.get("download_method")
        else:
            download_kwargs = {
                "repo_id": MODEL_ID, "revision": args.revision,
                "local_dir": str(snapshot), "allow_patterns": MODEL_FILES,
                "token": False, "max_workers": 2,
            }
            if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
                download_kwargs["local_dir_use_symlinks"] = False
            result["download_method"] = {"function": "huggingface_hub.snapshot_download",
                                         "arguments": download_kwargs}
            downloaded = Path(snapshot_download(**download_kwargs)).resolve()
            if downloaded != snapshot:
                raise RuntimeError(f"Unexpected snapshot download directory: {downloaded}")
        result["snapshot_resolution_seconds"] = time.perf_counter() - started
        result["snapshot_path"] = str(snapshot)

        stage = "snapshot_validation"
        required = ["config.json", "modules.json", "model.safetensors",
                    "tokenizer_config.json", "sentence_bert_config.json",
                    "1_Pooling/config.json"]
        missing = [name for name in required if not (snapshot / name).is_file()]
        if not any((snapshot / name).is_file() for name in ["tokenizer.json", "vocab.txt"]):
            missing.append("tokenizer.json or vocab.txt")
        if missing:
            raise RuntimeError(f"Official snapshot is incomplete: {missing}")
        present_files = [name for name in MODEL_FILES if (snapshot / name).is_file()]
        if any((snapshot / name).is_symlink() for name in present_files):
            raise RuntimeError("Expected regular model files in the dedicated local directory.")
        result["regular_model_files"] = present_files
        result["complete_model_files"] = True
        modules = json.loads((snapshot / "modules.json").read_text(encoding="utf-8"))
        expected = ["sentence_transformers.models.Transformer",
                    "sentence_transformers.models.Pooling",
                    "sentence_transformers.models.Normalize"]
        if [module.get("type") for module in modules] != expected:
            raise RuntimeError("Unexpected official module layout; stop for review.")

        stage = "model_loading"
        print(f"STAGE {args.worker}: loading SentenceTransformer on CPU", flush=True)
        started = time.perf_counter()
        model = SentenceTransformer(str(snapshot), device="cpu",
                                    local_files_only=True, trust_remote_code=False)
        model.max_seq_length = MAX_SEQUENCE_LENGTH
        model.eval()
        result["model_loading_seconds"] = time.perf_counter() - started
        result["actual_device"] = str(model.device)
        if model.device.type != "cpu":
            raise RuntimeError(f"Expected CPU, got {model.device}")

        stage = "embedding_inference"
        print(f"STAGE {args.worker}: encoding two sentences twice", flush=True)
        durations = []
        outputs = []
        with torch.inference_mode():
            for _ in range(2):
                started = time.perf_counter()
                outputs.append(model.encode(
                    SENTENCES, batch_size=2, device="cpu", normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ))
                durations.append(time.perf_counter() - started)
        embeddings, repeated = outputs
        norms = np.linalg.norm(embeddings, axis=1)
        checks = {
            "model_loaded": True,
            "shape_2_by_384": embeddings.shape == (2, 384),
            "all_values_finite": bool(np.isfinite(embeddings).all()),
            "unit_norms": bool(np.allclose(norms, 1.0, atol=1e-5, rtol=0)),
            "repeat_consistency": bool(np.allclose(embeddings, repeated, atol=1e-6, rtol=1e-5)),
        }
        result.update({"checks": checks, "embedding_shape": list(embeddings.shape),
                       "embedding_norms": norms.tolist(),
                       "embedding_generation_seconds": durations[0],
                       "repeated_embedding_generation_seconds": durations[1],
                       "repeat_max_absolute_difference": float(np.max(np.abs(embeddings - repeated)))})
        if not all(checks.values()):
            raise RuntimeError(f"Embedding validation failed: {checks}")
        if not offline:
            stage = "local_manifest_write"
            manifest = {"model_id": MODEL_ID, "revision": args.revision,
                        "huggingface_hub_version": hub_version,
                        "download_method": result["download_method"],
                        "sha256": {name: file_hash(snapshot / name) for name in present_files}}
            (snapshot / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result["_embeddings"] = embeddings.tolist()
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = {"stage": stage, "type": type(exc).__name__,
                           "message": str(exc), "traceback": traceback.format_exc()}
    print(RESULT_PREFIX + json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


def run_worker(mode, revision, model_dir):
    command = [sys.executable, "-B", str(Path(__file__).resolve()),
               "--worker", mode, "--revision", revision, "--model-dir", str(model_dir)]
    diagnostics = []
    result = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX):])
            elif line.startswith("STAGE "):
                print(line, flush=True)
            elif line:
                diagnostics.append(line)
        return_code = process.wait()
    if result is None:
        raise RuntimeError(f"{mode} worker exited {return_code} without a result:\n" + "\n".join(diagnostics[-30:]))
    result["diagnostics"] = diagnostics[-30:]
    if return_code != 0 and result["status"] == "passed":
        raise RuntimeError(f"{mode} worker exited unexpectedly: {return_code}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Dedicated local directory for the official model files")
    parser.add_argument("--revision", default=MODEL_REVISION, choices=[MODEL_REVISION],
                        help="Pinned official model revision")
    parser.add_argument("--offline", action="store_true", help="Require existing verified local files; never download")
    parser.add_argument("--report", type=Path, default=REPOSITORY_ROOT / "results" / "local_embeddings_verification.json")
    parser.add_argument("--worker", choices=["online", "offline"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    if args.worker:
        return worker(args)
    report = {"status": "failed", "model_id": MODEL_ID, "revision": MODEL_REVISION,
              "model_directory": str(args.model_dir),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        initial_mode = "offline" if args.offline or (args.model_dir / MANIFEST_NAME).is_file() else "online"
        report["initial_mode"] = initial_mode
        online = run_worker(initial_mode, args.revision, args.model_dir)
        reference = online.pop("_embeddings", None)
        report["initial_load"] = online
        if online["status"] != "passed":
            raise RuntimeError("Initial verification failed; no reconstruction or alternate model attempted.")
        revision = online["resolved_revision"]
        offline = run_worker("offline", revision, args.model_dir)
        reloaded = offline.pop("_embeddings", None)
        report["offline_reload"] = offline
        if offline["status"] != "passed":
            raise RuntimeError("Fresh-process cache-only reload failed.")
        differences = [abs(a - b) for row_a, row_b in zip(reference, reloaded)
                       for a, b in zip(row_a, row_b)]
        consistent = all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
                         for row_a, row_b in zip(reference, reloaded)
                         for a, b in zip(row_a, row_b))
        report["offline_reload_max_absolute_difference"] = max(differences)
        report["offline_reload_consistent"] = consistent
        if not consistent or offline["resolved_revision"] != revision:
            raise RuntimeError("Cached reload revision or embeddings differ.")
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"REPORT {args.report}")
    print(f"STATUS {report['status'].upper()}")
    for key in ["initial_load", "offline_reload"]:
        item = report.get(key, {})
        if item.get("status") == "passed":
            print(f"{key}: revision={item['resolved_revision']} shape={item['embedding_shape']} "
                  f"norms={item['embedding_norms']} load={item['model_loading_seconds']:.3f}s "
                  f"encode={item['embedding_generation_seconds']:.3f}s "
                  f"repeat_max_diff={item['repeat_max_absolute_difference']:.3g}")
        elif item.get("error"):
            print(json.dumps(item["error"], indent=2))
    if report.get("error"):
        print("ERROR " + report["error"]["message"])
    if report["status"] == "passed":
        print(f"offline_reload_max_diff={report['offline_reload_max_absolute_difference']:.3g}")
        print("READY: official sentence encoder verified for CPU-based BERTopic experiments.")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
