"""Synthetic validation plus an optional real, offline BERTopic fit/save/predict."""

from collections import Counter, namedtuple
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_full_2wiki_topics.py"
SPEC = importlib.util.spec_from_file_location("train_full_2wiki_topics", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
DiskUsage = namedtuple("DiskUsage", "total used free")


def make_inputs(root, texts, vectors):
    corpus, matrix, mapping, manifest = [root / name for name in
                                        ("documents.jsonl", "embeddings.npy", "document_ids.jsonl", "manifest.json")]
    identifiers = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    corpus.write_bytes(b"".join(builder.json_bytes({"document_id": identifier, "paragraph_text": text,
                                                   "title": "not an encoder input", "is_supporting": False,
                                                   "answers_objects": ["ignored"], "reasoning_steps": ["ignored"]})
                                for identifier, text in zip(identifiers, texts)))
    np.save(matrix, vectors, allow_pickle=False)
    mapping.write_bytes(b"".join(builder.json_bytes({"row": row, "document_id": identifier})
                                 for row, identifier in enumerate(identifiers)))
    sha = builder.file_sha256(corpus)
    config = {"model_id": builder.MODEL_ID, "model_revision": builder.MODEL_REVISION, "dimensions": 384,
              "dtype": "float32", "byte_order": "little", "max_seq_length": 256,
              "document_count": len(texts), "normalize_embeddings": True, "corpus_sha256": sha}
    data = {"schema_version": 1, "status": "complete", "processed_count": len(texts),
            "configuration": config, "config_sha256": hashlib.sha256(builder.json_bytes(config)).hexdigest(),
            "embedding_data_sha256": hashlib.sha256(vectors.tobytes(order="C")).hexdigest(),
            "artifacts": {path.name: {"sha256": builder.file_sha256(path), "bytes": path.stat().st_size}
                          for path in (matrix, mapping)}}
    manifest.write_bytes(builder.json_bytes(data))
    return corpus, matrix, mapping, manifest, sha, identifiers


class FakeModel:
    def __init__(self, topics):
        self.topics_ = list(topics)
        self.probabilities_ = np.array([0.0 if topic == -1 else 0.8 for topic in topics])
        self.topic_sizes_ = dict(Counter(topics))
        self.hdbscan_model = types.SimpleNamespace(prediction_data=True, prediction_data_=object())
        self.umap_model = types.SimpleNamespace(transform=lambda x: x[:, :5])

    def get_topics(self):
        return {topic: [("word", 0.5)] for topic in self.topic_sizes_}

    def save(self, path, serialization, save_embedding_model):
        assert serialization == "pickle" and save_embedding_model is False
        Path(path).write_bytes(builder.json_bytes({"topics": self.topics_}))

    @staticmethod
    def load(path):
        return FakeModel(builder.read_object(path)["topics"])


class TopicTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.texts = [f" Training Café 東京 paragraph {i} " for i in range(12)]
        vectors = np.random.default_rng(42).normal(size=(12, 384)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.corpus, self.matrix, self.mapping, self.manifest, self.sha, self.ids = make_inputs(self.root, self.texts, vectors)
        self.output = self.root / "topics"
        self.work = self.root / "topics.inprogress"
        self.parameters = builder.topic_configuration(min_cluster_size=3, min_samples=2, n_neighbors=3, umap_epochs=30)
        self.model = FakeModel([0, 0, -1, 1] * 3)
        self.trainer = mock.Mock(return_value=self.model)
        self.loader = mock.Mock(side_effect=FakeModel.load)
        for patch in (
            mock.patch.dict(os.environ, {"PYTHONHASHSEED": "42"}),
            mock.patch.object(builder, "environment", return_value={"libraries": {"numpy": np.__version__}}),
            mock.patch.object(builder, "memory_snapshot", return_value={"available_memory_bytes": 64 * builder.GIB}),
            mock.patch.object(builder.shutil, "disk_usage", return_value=DiskUsage(100 * builder.GIB, 0, 100 * builder.GIB)),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def run_training(self, **options):
        defaults = {"expected_sha": self.sha, "expected_count": 12, "parameters": self.parameters,
                    "min_available_ram_gib": 0, "min_free_disk_gib": 0,
                    "trainer": self.trainer, "loader": self.loader}
        defaults.update(options)
        output = defaults.pop("output_dir", self.output)
        return builder.run_training(self.corpus, self.matrix, self.mapping, self.manifest, output, **defaults)

    def update_manifest(self, mutate):
        data = builder.read_object(self.manifest)
        mutate(data)
        self.manifest.write_bytes(builder.json_bytes(data))

    def refresh_artifact_checksums(self):
        def refresh(data):
            for path in (self.matrix, self.mapping):
                data["artifacts"][path.name] = {"sha256": builder.file_sha256(path), "bytes": path.stat().st_size}
        self.update_manifest(refresh)

    def interrupt_publication(self):
        with mock.patch.object(builder.os, "rename", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.run_training()
        self.assertFalse(self.output.exists())
        self.assertEqual(builder.read_object(self.work / "progress.json")["status"], "failed")

    def test_training_uses_unchanged_texts_and_readonly_mmap(self):
        self.run_training()
        parameters, texts, matrix = self.trainer.call_args.args
        self.assertEqual(parameters, self.parameters)
        self.assertEqual(texts, self.texts)
        self.assertIsInstance(matrix, np.memmap)
        self.assertFalse(matrix.flags.writeable)
        self.assertEqual(matrix.shape, (12, 384))
        self.assertEqual(matrix.dtype, np.float32)

    def test_assignments_metadata_outliers_and_checksums(self):
        manifest = self.run_training()
        rows = [json.loads(raw) for raw in (self.output / "document_topics.jsonl").read_bytes().splitlines()]
        self.assertEqual([row["document_id"] for row in rows], self.ids)
        self.assertEqual([row["row"] for row in rows], list(range(12)))
        self.assertEqual([row["topic"] for row in rows], self.model.topics_)
        self.assertEqual(rows[2]["topic"], -1)
        self.assertEqual(rows[2]["bertopic"], "Topic -1")
        self.assertEqual(rows[2]["membership_strength"], 0)
        summary = builder.read_object(self.output / "topic_summary.json")
        self.assertEqual(summary["outlier_count"], 3)
        self.assertEqual(summary["outlier_rate"], 0.25)
        self.assertEqual(summary["topics_discovered_excluding_outliers"], 2)
        self.assertEqual(sum(row["size"] for row in summary["topic_size_distribution"]), 12)
        metadata_rows = [json.loads(raw) for raw in (self.output / "topic_metadata.jsonl").read_bytes().splitlines()]
        self.assertEqual({row["topic"]: row["size"] for row in metadata_rows}, {-1: 3, 0: 6, 1: 3})
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["processed_count"], 12)
        for name, evidence in manifest["artifacts"].items():
            self.assertEqual(evidence, {"sha256": builder.file_sha256(self.output / name),
                                        "bytes": (self.output / name).stat().st_size})
        self.assertFalse(self.work.exists())
        self.assertEqual(builder.read_object(self.output / "training_config.json")["parameters"], self.parameters)
        self.assertEqual(set(builder.read_object(self.output / "resolved_parameters.json")),
                         {"bertopic", "umap", "hdbscan", "vectorizer", "ctfidf"})

    def test_incomplete_or_wrong_model_manifest_is_rejected(self):
        original = self.manifest.read_bytes()
        for change in (lambda data: data.update(status="running"),
                       lambda data: data["configuration"].update(model_revision="wrong"),
                       lambda data: data["configuration"].update(normalize_embeddings=False),
                       lambda data: data.update(processed_count=11)):
            with self.subTest(change=change):
                self.manifest.write_bytes(original)
                self.update_manifest(change)
                with self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                    self.run_training()
        self.trainer.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_corpus_checksum_and_count_are_verified(self):
        with self.assertRaises(ValueError):
            self.run_training(expected_sha="0" * 64)
        with self.assertRaises(ValueError):
            self.run_training(expected_count=13)
        with self.corpus.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Corpus checksum/count mismatch"):
            self.run_training()
        self.trainer.assert_not_called()

    def test_alignment_is_checked_even_with_valid_mapping_checksum(self):
        records = self.mapping.read_bytes().splitlines(keepends=True)
        records[0], records[1] = records[1], records[0]
        self.mapping.write_bytes(b"".join(records))
        self.refresh_artifact_checksums()
        with self.assertRaisesRegex(ValueError, "not aligned at row 0"):
            self.run_training()
        self.trainer.assert_not_called()

    def test_extra_mapping_rows_are_rejected(self):
        with self.mapping.open("ab") as stream:
            stream.write(builder.json_bytes({"row": 12, "document_id": self.ids[0]}))
        self.refresh_artifact_checksums()
        with self.assertRaisesRegex(ValueError, "extra rows/bytes"):
            self.run_training()

    def test_embedding_checksum_prevents_silent_mutation(self):
        matrix = np.load(self.matrix, mmap_mode="r+")
        matrix[0] = np.roll(matrix[0], 1)
        matrix.flush()
        del matrix
        with self.assertRaisesRegex(ValueError, "artifact checksum/size mismatch"):
            self.run_training()
        self.trainer.assert_not_called()

    def test_wrong_shape_dtype_and_nonfinite_vectors_are_rejected(self):
        original = np.load(self.matrix)
        for vectors in (original[:, :383], original.astype(np.float64), np.full_like(original, np.nan), original * 2):
            with self.subTest(shape=vectors.shape, dtype=vectors.dtype):
                np.save(self.matrix, vectors)
                self.refresh_artifact_checksums()
                with self.assertRaises(ValueError):
                    self.run_training()
        self.trainer.assert_not_called()

    def test_resource_gates_prevent_fitting_and_output_creation(self):
        with mock.patch.object(builder, "memory_snapshot", return_value={"available_memory_bytes": 1}):
            with self.assertRaisesRegex(ValueError, "Insufficient RAM"):
                self.run_training()
        with mock.patch.object(builder.shutil, "disk_usage", return_value=DiskUsage(10, 9, 1)):
            with self.assertRaisesRegex(ValueError, "Insufficient disk"):
                self.run_training()
        self.trainer.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_completed_or_empty_destination_is_not_overwritten(self):
        self.run_training()
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.run_training(resume=True)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_training(output_dir=empty)

    def test_failed_fit_is_incomplete_and_explicit_resume_refits(self):
        self.trainer.side_effect = RuntimeError("fit interrupted")
        with self.assertRaisesRegex(RuntimeError, "fit interrupted"):
            self.run_training()
        self.assertFalse(self.output.exists())
        self.assertFalse((self.work / "fit_checkpoint.json").exists())
        self.assertEqual(builder.read_object(self.work / "progress.json")["stage"], "fitting")
        with self.assertRaises(FileExistsError):
            self.run_training()
        self.trainer.side_effect = None
        self.run_training(resume=True)
        self.assertEqual(self.trainer.call_count, 2)

    def test_fitted_checkpoint_can_resume_without_retraining(self):
        self.interrupt_publication()
        self.run_training(resume=True)
        self.assertEqual(self.trainer.call_count, 1)
        self.loader.assert_called_once()
        report = builder.read_object(self.output / "resource_report.json")
        self.assertIn("after_fit", report["fitted_attempt"]["resources"])
        self.assertGreater(report["fitted_attempt"]["elapsed_seconds_through_checkpoint"], 0)

    def test_changed_parameters_are_not_mixed_on_resume(self):
        self.interrupt_publication()
        changed = builder.topic_configuration(min_cluster_size=4, min_samples=2, n_neighbors=3, umap_epochs=30)
        with self.assertRaisesRegex(ValueError, "incompatible resume"):
            self.run_training(resume=True, parameters=changed)
        self.loader.assert_not_called()

    def test_corrupt_fitted_checkpoint_is_never_loaded(self):
        self.interrupt_publication()
        with (self.work / "artifacts/bertopic_model.pkl").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checkpoint is corrupt"):
            self.run_training(resume=True)
        self.loader.assert_not_called()

    def test_all_outliers_are_preserved_and_reported(self):
        self.trainer.return_value = FakeModel([-1] * 12)
        result = self.run_training()
        self.assertTrue(result["topic_summary"]["all_documents_outliers"])
        self.assertEqual(result["topic_summary"]["topics_discovered_excluding_outliers"], 0)
        self.assertEqual(result["topic_summary"]["outlier_rate"], 1)

    def test_missing_prediction_data_and_dense_probabilities_fail(self):
        self.model.hdbscan_model.prediction_data = False
        with self.assertRaisesRegex(ValueError, "prediction data"):
            self.run_training()
        self.assertFalse(self.output.exists())
        self.model.hdbscan_model.prediction_data = True
        self.model.probabilities_ = np.ones((12, 3))
        with self.assertRaisesRegex(ValueError, "not a dense matrix"):
            self.run_training(resume=True)

    def test_model_metadata_inconsistency_is_rejected(self):
        self.model.topic_sizes_ = {0: 12}
        with self.assertRaisesRegex(ValueError, "topic-size metadata"):
            self.run_training()

    def test_concurrent_completed_output_is_preserved(self):
        rename = builder.os.rename

        def concurrent(source, destination):
            Path(destination).mkdir()
            (Path(destination) / "manifest.json").write_bytes(b"other completed run")
            rename(source, destination)

        with mock.patch.object(builder.os, "rename", side_effect=concurrent):
            with self.assertRaises(OSError):
                self.run_training()
        self.assertEqual((self.output / "manifest.json").read_bytes(), b"other completed run")

    def test_concurrent_writer_is_locked_out(self):
        self.work.mkdir()
        with builder.job_lock(self.work):
            with self.assertRaisesRegex(ValueError, "Another training process"):
                self.run_training(resume=True)
        self.trainer.assert_not_called()

    def test_hashseed_and_cli_validation(self):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "random"}):
            with self.assertRaisesRegex(ValueError, "PYTHONHASHSEED=42"):
                self.run_training()
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_configuration_avoids_dense_distance_and_membership_matrices(self):
        config = builder.topic_configuration()
        self.assertTrue(config["umap"]["force_approximation_algorithm"])
        self.assertTrue(config["umap"]["low_memory"])
        self.assertEqual(config["umap"]["n_components"], 5)
        self.assertEqual(config["hdbscan"]["algorithm"], "boruvka_kdtree")
        self.assertTrue(config["hdbscan"]["prediction_data"])
        self.assertFalse(config["bertopic"]["calculate_probabilities"])
        self.assertIsNone(config["bertopic"]["nr_topics"])
        self.assertIsNone(config["bertopic"]["embedding_model"])


HAS_TOPIC_DEPS = all(importlib.util.find_spec(name) is not None for name in ("bertopic", "umap", "hdbscan"))


@unittest.skipUnless(HAS_TOPIC_DEPS, "BERTopic/UMAP/HDBSCAN dependencies are not installed")
class BERTopicIntegration(unittest.TestCase):
    def test_real_offline_fit_save_reload_and_query_assignment(self):
        from bertopic import BERTopic

        rng = np.random.default_rng(42)
        texts, vectors = [], []
        vocab = ["space rocket planet orbit astronaut telescope galaxy star",
                 "ocean coral whale fish reef sea marine dolphin",
                 "food kitchen bread soup recipe cooking dinner chef"]
        for group, words in enumerate(vocab):
            for index in range(40):
                texts.append(f"{words} unique training example {group} {index}")
                vector = rng.normal(0, 0.01, 384).astype(np.float32)
                vector[group] += 1
                vectors.append(vector / np.linalg.norm(vector))
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"PYTHONHASHSEED": "42"}):
            root = Path(temporary)
            corpus, matrix, mapping, manifest, sha, _ = make_inputs(root, texts, np.stack(vectors))
            output = root / "topics"
            parameters = builder.topic_configuration(min_cluster_size=8, min_samples=3, n_neighbors=10, umap_epochs=30)
            # A real integration failure must not silently download or encode text.
            with mock.patch.object(BERTopic, "_extract_embeddings", side_effect=AssertionError("Re-encoding forbidden")):
                result = builder.run_training(corpus, matrix, mapping, manifest, output, expected_sha=sha,
                                              expected_count=len(texts), parameters=parameters,
                                              min_available_ram_gib=0, min_free_disk_gib=0)
                loaded = BERTopic.load(str(output / "bertopic_model.pkl"))
                query_vector = rng.normal(0, 0.01, 384).astype(np.float32)
                query_vector[0] += 1
                query_vector /= np.linalg.norm(query_vector)
                query_topics, strengths = loaded.transform(["unseen space rocket orbit query"],
                                                          embeddings=query_vector[None, :])
            self.assertEqual(len(query_topics), 1)
            self.assertTrue(np.isfinite(strengths).all())
            self.assertIsNotNone(loaded.hdbscan_model.prediction_data_)
            self.assertTrue(loaded.umap_model.force_approximation_algorithm)
            self.assertGreaterEqual(result["topic_summary"]["topics_discovered_excluding_outliers"], 2)
            rows = [json.loads(raw) for raw in (output / "document_topics.jsonl").read_bytes().splitlines()]
            self.assertEqual([row["topic"] for row in rows], loaded.topics_)
            self.assertEqual(sum(loaded.topic_sizes_.values()), len(texts))
            resolved = builder.read_object(output / "resolved_parameters.json")
            self.assertEqual(resolved["hdbscan"]["parameters"]["algorithm"], "boruvka_kdtree")
            self.assertTrue(resolved["umap"]["parameters"]["force_approximation_algorithm"])


if __name__ == "__main__":
    unittest.main()
