"""Read-only question encoding and frozen BERTopic inference (Step 4D).

Use the recorded Python 3.12/package environment and trusted training directory.
No training, document encoding, retrieval, adequacy thresholds or output writes.
"""

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import copy
import hashlib
from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import threading

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_bertopic_150k as training
import shared_150k_retrieval as shared


DEFAULT_TRAINING = training.OUTPUT_ROOT / "run_001"
LIBRARIES = ("numpy", "bertopic", "umap-learn", "hdbscan", "scikit-learn", "scipy", "pandas", "numba",
             "llvmlite", "pynndescent", "joblib", "threadpoolctl", "sentence-transformers", "transformers",
             "tokenizers", "torch", "huggingface-hub", "safetensors")
POLICY = ("all content tokens; nonoverlapping windows; normalized window vectors weighted by "
          "content-token count; final L2 normalization")
UNCALIBRATED = "Uncalibrated HDBSCAN membership signals; not probabilities of evidence adequacy."
_RUNTIME_LOCK = threading.RLock()
require = shared.require


def check_environment(saved):
    """Reject incompatible pickle/encoder libraries before deserialization."""
    python = platform.python_version()
    require(isinstance(saved, dict), "Missing saved training environment")
    require(python.split(".")[:2] == ["3", "12"] and
            str(saved.get("python", "")).split(".")[:2] == ["3", "12"], "Use the saved Python 3.12 environment")
    versions = saved.get("libraries", {})
    require(isinstance(versions, dict), "Missing recorded training libraries")
    current = {}
    for library in LIBRARIES:
        try:
            current[library] = metadata.version(library)
        except metadata.PackageNotFoundError as error:
            raise shared.ArtifactValidationError(f"Missing recorded dependency: {library}") from error
        require(isinstance(versions.get(library), str) and current[library] == versions[library],
                f"Incompatible {library}: saved={versions.get(library)}, current={current[library]}")
    return {"python": python, "saved_python": saved["python"], "libraries": current}


@contextmanager
def verified_artifacts(directory, expected_count, expected_corpus_sha256, trusted_manifest_sha256=None):
    """Verify every manifest entry, keeping the verified pickle descriptor open."""
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    before_manifest = shared._stamp(manifest_path)
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if trusted_manifest_sha256 is not None:
        require(shared._valid_hash(trusted_manifest_sha256) and manifest_sha == trusted_manifest_sha256,
                "Training manifest differs from the separately trusted manifest hash")
    manifest = shared._json(raw, "Training manifest")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
            "Unsupported training manifest schema")
    require(manifest.get("status") == "complete", "Training manifest must be complete")
    require(type(manifest.get("occurrences")) is int and manifest["occurrences"] == expected_count and
            manifest.get("corpus_sha256") == expected_corpus_sha256, "Unexpected frozen corpus identity/count")
    require(manifest.get("encoder_id") == training.MODEL_ID and
            manifest.get("encoder_revision") == training.MODEL_REVISION, "Wrong saved MiniLM model/revision")
    records = manifest.get("artifacts")
    required = {"bertopic_model.pkl", "encoder_manifest.json", "embedding_config.json", "training_config.json"}
    require(isinstance(records, dict) and required <= records.keys(), "Incomplete training artifact inventory")
    require("manifest.json" not in records, "Training manifest cannot contain a circular self-hash")
    stamps, model_stream = {manifest_path: before_manifest}, None
    with ExitStack() as resources:
        for name, evidence in records.items():
            require(isinstance(name, str) and bool(name) and "\\" not in name, "Unsafe artifact path")
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
                    f"Unsafe artifact path: {name}")
            path = directory / name
            require(path.resolve().is_relative_to(directory) and path.is_file() and not path.is_symlink(),
                    f"Missing/unsafe artifact: {name}")
            require(isinstance(evidence, dict) and shared._valid_hash(evidence.get("sha256")) and
                    type(evidence.get("bytes")) is int and evidence["bytes"] >= 0, f"Invalid artifact evidence: {name}")
            stamps[path] = shared._stamp(path)
            require(path.stat().st_size == evidence["bytes"], f"Artifact size mismatch: {name}")
            digest = hashlib.sha256()
            if name == "bertopic_model.pkl":
                stream = resources.enter_context(path.open("rb"))
                model_stream = stream
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            else:
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            require(digest.hexdigest() == evidence["sha256"], f"Artifact SHA-256 mismatch: {name}")
        _check_stamps(stamps)
        model_stream.seek(0)
        yield manifest, manifest_sha, model_stream, stamps


def _check_stamps(stamps):
    require(all(shared._stamp(path) == stamp for path, stamp in stamps.items()),
            "Training artifacts changed during verification/inference")


def validate_encoding_policy(directory, manifest, expected_count, expected_corpus_sha256):
    encoder = training.read_json(directory / "encoder_manifest.json")
    config = training.read_json(directory / "embedding_config.json")
    fit_config = training.read_json(directory / "training_config.json")
    require(encoder.get("model_id") == training.MODEL_ID and encoder.get("revision") == training.MODEL_REVISION,
            "Wrong encoder snapshot identity")
    files = encoder.get("files", {})
    require(isinstance(files, dict) and training.REQUIRED_MODEL_FILES <= files.keys() <= set(training.MODEL_FILES),
            "Incomplete/unsafe encoder snapshot inventory")
    for name, evidence in files.items():
        require(manifest["artifacts"].get("encoder/" + name) == evidence, "Encoder hashes differ from training manifest")
    require(type(config.get("schema_version")) is int and config["schema_version"] == 1 and
            config.get("encoder") == encoder and config.get("dimensions") == training.DIMENSIONS and
            config.get("dtype") == "<f4" and config.get("normalize_embeddings") is True and
            config.get("max_window_tokens") == training.MAX_TOKENS and config.get("long_text_policy") == POLICY and
            config.get("seed") == training.SEED, "Saved encoding policy differs from the training helper")
    corpus = config.get("corpus", {})
    require(corpus.get("occurrences") == expected_count and corpus.get("sha256") == expected_corpus_sha256,
            "Embedding configuration corpus identity differs")
    batch_size = shared._positive_integer(config.get("window_batch_size"), "saved window_batch_size")
    helpers = [Path(training.__file__), Path(training.preparation.__file__), Path(training.embedding.__file__),
               Path(training.full.__file__), training.full.RESOURCE_HELPER]
    hashes = config.get("implementation_sha256", {})
    require(all(hashes.get(path.name) == training.sha256(path) for path in helpers),
            "Training helper code differs; use the recorded training source before inference")
    require(fit_config.get("embedding_config_sha256") == hashlib.sha256(training.json_bytes(config)).hexdigest() and
            fit_config.get("training_occurrences") == expected_count and
            fit_config.get("corpus_sha256") == expected_corpus_sha256, "Training/embedding configurations disagree")
    return config, fit_config, batch_size


@contextmanager
def runtime_cache(directory):
    """Third-party runtime caches stay temporary; frozen inputs are never writable caches."""
    settings = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_CACHE": str(directory / "hub"),
                "TRANSFORMERS_CACHE": str(directory / "transformers"), "NUMBA_CACHE_DIR": str(directory / "numba")}
    with _RUNTIME_LOCK:
        previous = {key: os.environ.get(key) for key in settings}
        numba = sys.modules.get("numba")
        previous_numba_cache = None if numba is None else numba.config.CACHE_DIR
        try:
            os.environ.update(settings)
            if numba is not None:
                numba.config.CACHE_DIR = settings["NUMBA_CACHE_DIR"]
            yield
        finally:
            loaded_numba = sys.modules.get("numba")
            if loaded_numba is not None:
                loaded_numba.config.CACHE_DIR = previous_numba_cache if numba is not None else (previous["NUMBA_CACHE_DIR"] or "")
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def load_trusted_model(stream):
    import joblib
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    model = joblib.load(stream)  # exact descriptor hashed above, positioned at byte zero
    require(isinstance(model, BERTopic) and isinstance(model.hdbscan_model, HDBSCAN) and
            isinstance(model.umap_model, UMAP), "Expected the complete fitted BERTopic/UMAP/HDBSCAN pickle")
    return model


class ReadOnlyMiniLM(training.MiniLMEncoder):
    """Reuse training's exact forward pass with offline, verified local weights."""

    def __init__(self, directory, device):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(training.SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = SentenceTransformer(str(directory), device=device, local_files_only=True, trust_remote_code=False,
                                         cache_folder=os.environ["HF_HUB_CACHE"])
        self.model.max_seq_length = training.MAX_TOKENS
        self.model.eval()
        require(self.model.get_sentence_embedding_dimension() == training.DIMENSIONS, "Wrong encoder dimension")
        self.tokenizer, self.device = self.model.tokenizer, device


def approximate_predict(clusterer, reduced):
    import hdbscan
    return hdbscan.approximate_predict(clusterer, reduced)


def membership_vector(clusterer, reduced):
    import hdbscan
    return hdbscan.membership_vector(clusterer, reduced)


def validate_mapping(model, summary, count):
    """Validate raw HDBSCAN labels -> BERTopic IDs against every fitted row."""
    clusterer = model.hdbscan_model
    raw, topics = np.asarray(clusterer.labels_), np.asarray(model.topics_)
    require(raw.shape == topics.shape == (count,) and raw.dtype.kind in "iu" and topics.dtype.kind in "iu" and
            bool(np.all(raw >= -1)) and bool(np.all(topics >= -1)), "Invalid fitted raw/topic assignment arrays")
    original = model.topic_mapper_.get_mappings(original_topics=True)
    require(isinstance(original, dict), "Missing BERTopic original-topic mapper")
    mapping = {}
    for source, target in original.items():
        require(not isinstance(source, (bool, np.bool_)) and isinstance(source, (int, np.integer)) and source >= -1 and
                not isinstance(target, (bool, np.bool_)) and isinstance(target, (int, np.integer)) and target >= -1,
                "Invalid BERTopic topic mapping")
        mapping[int(source)] = int(target)
    require(mapping.get(-1, -1) == -1, "Raw outliers must remain topic -1")
    raw_ids, inverse = np.unique(raw, return_inverse=True)
    require(set(mapping) - {-1} == set(int(value) for value in raw_ids if value != -1), "Raw cluster mapping is incomplete")
    require(all(mapping[int(value)] >= 0 for value in raw_ids if value != -1), "Non-outlier cluster mapped to -1")
    mapped = np.array([mapping.get(int(value), -1) for value in raw_ids], dtype=np.int64)
    require(np.array_equal(mapped[inverse], topics), "BERTopic mapper disagrees with fitted row assignments")
    counts = Counter(int(value) for value in topics)
    require({int(key): int(value) for key, value in model.topic_sizes_.items()} == dict(counts), "Invalid fitted topic sizes")
    require(isinstance(summary, dict) and summary.get("occurrences") == count and
            summary.get("topic_sizes") == {str(topic): size for topic, size in counts.items()} and
            summary.get("non_outlier_topics") == len(counts) - int(-1 in counts) and
            summary.get("outlier_occurrences") == counts.get(-1, 0), "Model disagrees with manifest topic summary")
    require(callable(model.umap_model.transform), "Saved UMAP transform is unavailable")
    require(getattr(clusterer, "prediction_data_", None) is not None, "Saved HDBSCAN prediction data is unavailable")
    return mapping, frozenset(counts), frozenset(int(value) for value in raw_ids if value != -1)


def membership_columns(clusterer, raw_ids):
    """HDBSCAN membership_vector columns follow sorted selected tree-cluster IDs."""
    if not raw_ids:
        return None, "no_non_outlier_clusters"
    tree = getattr(clusterer, "condensed_tree_", None)
    data = clusterer.prediction_data_
    if not callable(getattr(tree, "_select_clusters", None)) or not hasattr(data, "cluster_map") or not hasattr(data, "reverse_cluster_map"):
        return None, "membership_column_metadata_unavailable"
    selected = sorted(tree._select_clusters())
    columns = []
    for tree_id in selected:
        require(isinstance(tree_id, (int, np.integer)) and not isinstance(tree_id, (bool, np.bool_)), "Invalid selected tree cluster")
        raw = data.cluster_map.get(tree_id)
        require(isinstance(raw, (int, np.integer)) and not isinstance(raw, (bool, np.bool_)) and
                int(raw) in raw_ids and data.reverse_cluster_map.get(raw) == tree_id,
                "Ambiguous/inconsistent HDBSCAN membership column mapping")
        columns.append(int(raw))
    require(len(columns) == len(set(columns)) and set(columns) == set(raw_ids), "Membership columns do not cover raw clusters")
    exemplars = getattr(data, "exemplars", None)
    if exemplars is None:
        return None, "membership_exemplars_unavailable"
    require(len(exemplars) == len(columns), "Membership exemplars and selected clusters disagree")
    return tuple(columns), None


class QueryInference150k:
    """Reusable single-question inference. Source files and saved parameters stay unchanged.

    Private service/identity arguments support synthetic tests only; no CLI bypass.
    The caller supplies its trusted complete training directory. An optional
    separately backed-up manifest hash provides an additional external trust pin.
    """

    def __init__(self, training_dir=DEFAULT_TRAINING, *, device="cpu", trusted_manifest_sha256=None,
                 _expected_count=shared.OCCURRENCES, _expected_corpus_sha256=shared.CORPUS_SHA256,
                 _model_loader=load_trusted_model, _encoder_factory=ReadOnlyMiniLM, _runtime_checker=check_environment,
                 _predictor=approximate_predict, _membership_function=membership_vector):
        self._closed, self._temporary = False, None
        self._model, self._encoder = None, None
        self._lock = threading.RLock()
        self._directory = Path(training_dir).resolve()
        self._predictor, self._membership_function = _predictor, _membership_function
        device = training.embedding.canonical_device(device)
        try:
            with verified_artifacts(self._directory, _expected_count, _expected_corpus_sha256, trusted_manifest_sha256) as verified:
                manifest, manifest_sha, stream, stamps = verified
                config, fit_config, self._window_batch_size = validate_encoding_policy(
                    self._directory, manifest, _expected_count, _expected_corpus_sha256)
                environment = _runtime_checker(fit_config.get("environment", {}))
                _check_stamps(stamps)
                self._temporary = tempfile.TemporaryDirectory(prefix="at-rag-query-inference-")
                with runtime_cache(Path(self._temporary.name)):
                    self._model = _model_loader(stream)
                    self._mapping, self._topics, raw_ids = validate_mapping(self._model, manifest.get("topic_summary"), _expected_count)
                    self._columns, self._membership_reason = membership_columns(self._model.hdbscan_model, raw_ids)
                    self._encoder = _encoder_factory(self._directory / "encoder", device)
                _check_stamps(stamps)
                self._stamps = {path: stamp for path, stamp in stamps.items()
                                if path.name in {"manifest.json", "bertopic_model.pkl", "training_config.json",
                                                 "embedding_config.json", "encoder_manifest.json"} or
                                "encoder" in path.relative_to(self._directory).parts}
                self._report = {"status": "validated", "training_manifest_sha256": manifest_sha,
                    "verified_artifact_count": len(manifest["artifacts"]), "model_sha256": manifest["artifacts"]["bertopic_model.pkl"]["sha256"],
                    "corpus_sha256": _expected_corpus_sha256, "training_occurrences": _expected_count,
                    "encoder_id": training.MODEL_ID, "encoder_revision": training.MODEL_REVISION,
                    "encoding_policy": config["long_text_policy"], "window_batch_size": self._window_batch_size,
                    "environment": environment, "device": device,
                    "raw_cluster_to_topic": {str(k): v for k, v in sorted(self._mapping.items())},
                    "membership_column_raw_clusters": None if self._columns is None else list(self._columns),
                    "membership_unavailable_reason": self._membership_reason, "signal_interpretation": UNCALIBRATED}
        except BaseException:
            self.close()
            raise

    @property
    def validation_report(self):
        return copy.deepcopy(self._report)

    def infer(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a nonempty string")
        question.encode("utf-8")
        with self._lock:
            if self._closed:
                raise RuntimeError("Query inference component is closed")
            _check_stamps(self._stamps)
            with runtime_cache(Path(self._temporary.name)):
                vectors, encoding_stats = training.encode_occurrences([question], self._encoder, self._window_batch_size)
                shared.Shared150kRetrieval._query(vectors[0])  # exact retrieval input contract
                reduced = np.asarray(self._model.umap_model.transform(vectors))
                require(reduced.ndim == 2 and reduced.shape[0] == 1 and reduced.shape[1] > 0 and
                        bool(np.isfinite(reduced).all()), "Invalid saved UMAP transform output")
                raw, strengths = self._predictor(self._model.hdbscan_model, reduced)
                raw = np.asarray(raw)
                require(raw.shape == (1,) and raw.dtype.kind in "iu" and int(raw[0]) >= -1, "Invalid predicted raw cluster")
                label = int(raw[0])
                require(label == -1 or label in self._mapping, "Predicted raw cluster has no validated BERTopic mapping")
                topic = -1 if label == -1 else self._mapping[label]
                strength = None
                if strengths is not None:
                    strengths = np.asarray(strengths)
                    require(strengths.shape == (1,) and strengths.dtype.kind in "fiu" and
                            bool(np.isfinite(strengths).all()) and 0 <= float(strengths[0]) <= 1 + 1e-6,
                            "Expected a finite scalar HDBSCAN strength, not a topic distribution")
                    strength = float(strengths[0])
                signals, mass, reason = None, None, self._membership_reason
                if self._columns is not None:
                    try:
                        values = self._membership_function(self._model.hdbscan_model, reduced)
                    except NotImplementedError:
                        values, reason = None, "membership_api_not_supported"
                    if values is None:
                        reason = reason or "membership_api_returned_no_distribution"
                    else:
                        values = np.asarray(values)
                        require(values.shape == (1, len(self._columns)) and values.dtype.kind in "fiu" and
                                bool(np.isfinite(values).all()) and bool(np.all(values >= 0)) and
                                bool(np.all(values <= 1 + 1e-6)) and float(values.sum()) <= 1 + 1e-6,
                                "Invalid per-cluster membership matrix; no substitute distribution was produced")
                        by_topic = {}
                        for column, raw_cluster in enumerate(self._columns):
                            target = self._mapping[raw_cluster]
                            by_topic[target] = by_topic.get(target, 0.0) + float(values[0, column])
                        signals = [{"topic_id": key, "membership": value}
                                   for key, value in sorted(by_topic.items(), key=lambda pair: (-pair[1], pair[0]))]
                        mass = float(values.sum())
            vector = vectors[0].copy()
            vector.flags.writeable = False
            return {"embedding": vector, "topic_id": topic, "raw_cluster_id": label,
                    "membership_strength": strength, "topic_memberships": signals,
                    "membership_status": "available" if signals is not None else "unavailable",
                    "membership_unavailable_reason": reason, "membership_sum": mass,
                    "encoding_statistics": encoding_stats, "signal_interpretation": UNCALIBRATED}

    def close(self):
        with self._lock:
            self._model, self._encoder = None, None
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Query inference component is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trusted-manifest-sha256", help="Optional separately backed-up manifest SHA-256")
    args = parser.parse_args(argv)
    try:
        with QueryInference150k(args.training_dir, device=args.device,
                               trusted_manifest_sha256=args.trusted_manifest_sha256) as inference:
            result = inference.infer(args.question)
            result["embedding"] = result["embedding"].tolist()
            print(training.json_bytes({"validation": inference.validation_report, "result": result}).decode("utf-8"), end="")
    except (Exception, KeyboardInterrupt) as error:
        parser.exit(1, f"Query inference failed: {type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
