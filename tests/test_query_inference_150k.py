"""Synthetic artifact/encoder tests plus an opt-in tiny real fitted-model test."""

from collections import Counter
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/query_inference_150k.py"
SPEC = importlib.util.spec_from_file_location("query_inference_150k", SCRIPT)
query = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(query)


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": [ord(char) for char in text]}

    def prepare_for_model(self, ids, add_special_tokens, truncation, return_attention_mask):
        assert truncation is False
        return {"input_ids": [101] + list(ids) + [102], "attention_mask": [1] * (len(ids) + 2)}


class Encoder:
    def __init__(self, *args):
        self.tokenizer = Tokenizer()
        self.tokens, self.batch_sizes = [], []

    def encode_windows(self, features):
        self.batch_sizes.append(len(features))
        vectors = np.zeros((len(features), 384), dtype=np.float32)
        for row, feature in enumerate(features):
            ids = feature["input_ids"][1:-1]
            self.tokens.extend(ids)
            for token in ids:
                vectors[row, token % 384] += 1
        return vectors


class Mapper:
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def get_mappings(self, original_topics=True):
        assert original_topics is True
        return dict(self.mapping)


class Model:
    def __init__(self, mapping=None):
        mapping = mapping or {-1: -1, 0: 8, 1: 3, 2: 15}
        self.topic_mapper_ = Mapper(mapping)
        raw = np.array([0, 0, 1, 1, 2, 2, -1, -1])
        self.topics_ = [mapping[int(value)] for value in raw]
        self.topic_sizes_ = dict(Counter(self.topics_))
        self.hdbscan_model = types.SimpleNamespace(labels_=raw,
            prediction_data_=types.SimpleNamespace(cluster_map={42: 2, 90: 0, 160: 1},
                reverse_cluster_map={2: 42, 0: 90, 1: 160}, exemplars=[object(), object(), object()]),
            condensed_tree_=types.SimpleNamespace(_select_clusters=lambda: [160, 42, 90]))
        self.umap_model = types.SimpleNamespace(transform=lambda x: x[:, :5])
        self.calculate_probabilities = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(query.training.json_bytes(value))


def make_fixture(root, model, *, vectors=None, environment=None, save_model=None):
    root.mkdir(exist_ok=True)
    count = len(model.topics_)
    corpus_rows, assignments = [], []
    for i, topic in enumerate(model.topics_):
        text = f"Synthetic original paragraph {i}"
        row = {"occurrence_id": f"2wikimultihopqa:train:{'a'*64}:{i+1}:0",
               "paragraph_text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "question_id": f"q-{i}", "source_split": "train", "source_line": i+1,
               "context_position": 0, "idx": 0, "title": "Fixture title"}
        corpus_rows.append(row)
        assignments.append({**{k: v for k, v in row.items() if k != "paragraph_text"}, "row": i,
                            "topic": int(topic), "bertopic": f"Topic {topic}", "membership_strength": .8})
    (root / "corpus.jsonl").write_bytes(b"".join(query.training.json_bytes(row) for row in corpus_rows))
    (root / "occurrence_topics.jsonl").write_bytes(b"".join(query.training.json_bytes(row) for row in assignments))
    if vectors is None:
        vectors = np.zeros((count, 384), dtype=np.float32)
        vectors[:, 0] = 1
    np.save(root / "embeddings.npy", vectors, allow_pickle=False)
    identity = {"sha256": query.training.sha256(root / "corpus.jsonl"), "occurrences": count,
                "unique_texts": count, "duplicate_excess_occurrences": 0}
    files = {}
    for name in query.training.REQUIRED_MODEL_FILES:
        path = root / "encoder" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("Synthetic model-file fixture: " + name).encode())
        files[name] = query.training.file_evidence(path)
    encoder = {"model_id": query.training.MODEL_ID, "revision": query.training.MODEL_REVISION, "files": files}
    write(root / "encoder_manifest.json", encoder)
    environment = environment or {"python": "3.12.3", "libraries": {name: "synthetic" for name in query.LIBRARIES},
                "device_identity": {"machine": "fixture"}, "cublas_workspace_config": ":4096:8"}
    config = query.training.embedding_configuration(identity, encoder, environment, "cpu", 128, 2)
    write(root / "embedding_config.json", config)
    write(root / "training_config.json", {"training_occurrences": count, "corpus_sha256": identity["sha256"],
          "embedding_config_sha256": hashlib.sha256(query.training.json_bytes(config)).hexdigest(), "environment": environment})
    if save_model is None:
        (root / "bertopic_model.pkl").write_bytes(b"synthetic-loader fixture, not a real pickle")
    else:
        save_model(root / "bertopic_model.pkl")
    (root / "extra_checkpoint.txt").write_bytes(b"Verify this unconsumed manifest entry too")
    counts = Counter(int(topic) for topic in model.topics_)
    manifest = {"schema_version": 1, "status": "complete", "occurrences": count, "corpus_sha256": identity["sha256"],
                "encoder_id": query.training.MODEL_ID, "encoder_revision": query.training.MODEL_REVISION,
                "topic_summary": {"occurrences": count, "topic_sizes": {str(k): v for k, v in counts.items()},
                                  "non_outlier_topics": len(counts)-int(-1 in counts),
                                  "outlier_occurrences": counts.get(-1, 0), "outlier_fraction": counts.get(-1, 0)/count},
                "artifacts": {str(p.relative_to(root)): query.training.file_evidence(p)
                              for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}}
    write(root / "manifest.json", manifest)
    return identity


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Model()
        self.identity = make_fixture(self.root, self.model)
        self.encoder = Encoder()
        self.loader = mock.Mock(side_effect=self.load)
        self.factory = mock.Mock(return_value=self.encoder)
        self.runtime_checker = mock.Mock(return_value={"python": "3.12.3", "libraries": {}})
        self.predictor = mock.Mock(return_value=(np.array([0]), np.array([.71])))
        self.membership = mock.Mock(return_value=np.array([[.1, .2, .5]]))

    def load(self, stream):
        self.assertEqual(stream.tell(), 0)
        self.assertEqual(stream.read(), (self.root / "bertopic_model.pkl").read_bytes())
        return self.model

    def component(self, **options):
        arguments = {"_expected_count": self.identity["occurrences"], "_expected_corpus_sha256": self.identity["sha256"],
                     "_model_loader": self.loader, "_encoder_factory": self.factory, "_runtime_checker": self.runtime_checker,
                     "_predictor": self.predictor, "_membership_function": self.membership}
        arguments.update(options)
        component = query.QueryInference150k(self.root, **arguments)
        self.addCleanup(component.close)
        return component

    def refresh(self):
        manifest = query.training.read_json(self.root / "manifest.json")
        for name in manifest["artifacts"]:
            manifest["artifacts"][name] = query.training.file_evidence(self.root / name)
        write(self.root / "manifest.json", manifest)

    def change_config(self, mutate):
        config = query.training.read_json(self.root / "embedding_config.json")
        mutate(config)
        write(self.root / "embedding_config.json", config)
        fit = query.training.read_json(self.root / "training_config.json")
        fit["embedding_config_sha256"] = hashlib.sha256(query.training.json_bytes(config)).hexdigest()
        write(self.root / "training_config.json", fit)
        self.refresh()

    def test_complete_token_windows_embedding_and_retrieval_compatibility(self):
        component = self.component()
        question = "a" * 800 + " ORIGINAL TAIL Ω "
        result = component.infer(question)
        self.assertEqual(self.encoder.tokens, [ord(char) for char in question])
        self.assertLessEqual(max(self.encoder.batch_sizes), 2)
        self.assertEqual(result["encoding_statistics"]["content_tokens"], len(question))
        self.assertEqual(result["encoding_statistics"]["multiwindow_occurrences"], 1)
        self.assertEqual(result["embedding"].shape, (384,))
        self.assertEqual(result["embedding"].dtype, np.float32)
        self.assertFalse(result["embedding"].flags.writeable)
        query.shared.Shared150kRetrieval._query(result["embedding"])
        with query.shared.Shared150kRetrieval(self.root / "corpus.jsonl", self.root / "embeddings.npy",
                self.root / "occurrence_topics.jsonl", self.root / "manifest.json", expected_count=8,
                expected_corpus_sha256=self.identity["sha256"]) as engine:
            hits = engine.retrieve(result["embedding"], mode="single", topic_id=result["topic_id"], k=5)
            self.assertEqual(len(hits), 2)
            self.assertEqual({hit["topic"] for hit in hits}, {8})

    def test_tree_columns_and_raw_clusters_are_mapped_to_bertopic_ids(self):
        component = self.component()
        result = component.infer("Question?")
        self.assertEqual(component.validation_report["membership_column_raw_clusters"], [2, 0, 1])
        self.assertEqual(result["raw_cluster_id"], 0)
        self.assertEqual(result["topic_id"], 8)
        self.assertEqual(result["membership_strength"], .71)
        self.assertEqual(result["topic_memberships"], [
            {"topic_id": 3, "membership": .5}, {"topic_id": 8, "membership": .2}, {"topic_id": 15, "membership": .1}])
        self.assertAlmostEqual(result["membership_sum"], .8)
        self.assertNotEqual(result["topic_id"], result["topic_memberships"][0]["topic_id"])
        self.assertIn("Uncalibrated", result["signal_interpretation"])
        self.assertNotIn(-1, [item["topic_id"] for item in result["topic_memberships"]])

    def test_many_raw_clusters_mapping_to_one_topic_are_summed(self):
        self.model = Model({-1: -1, 0: 8, 1: 8, 2: 15})
        self.identity = make_fixture(self.root, self.model)
        result = self.component().infer("Question?")
        self.assertEqual([item["topic_id"] for item in result["topic_memberships"]], [8, 15])
        self.assertAlmostEqual(result["topic_memberships"][0]["membership"], .7)
        self.assertAlmostEqual(result["membership_sum"], .8)

    def test_predicted_outlier_and_zero_strength_are_preserved(self):
        self.predictor.return_value = np.array([-1]), np.array([0.0])
        result = self.component().infer("Novel question")
        self.assertEqual(result["topic_id"], -1)
        self.assertEqual(result["membership_strength"], 0.0)
        self.assertEqual(result["membership_status"], "available")

    def test_no_non_outlier_clusters_reports_no_distribution(self):
        self.model.hdbscan_model.labels_ = np.full(8, -1)
        self.model.topics_ = [-1] * 8
        self.model.topic_sizes_ = {-1: 8}
        self.model.topic_mapper_ = Mapper({-1: -1})
        self.identity = make_fixture(self.root, self.model)
        self.predictor.return_value = np.array([-1]), np.array([0.0])
        result = self.component().infer("Question")
        self.assertEqual(result["topic_id"], -1)
        self.assertEqual(result["membership_strength"], 0.0)
        self.assertIsNone(result["topic_memberships"])
        self.assertEqual(result["membership_unavailable_reason"], "no_non_outlier_clusters")
        self.membership.assert_not_called()

    def test_new_query_can_be_outlier_when_training_had_no_outliers(self):
        self.model.hdbscan_model.labels_ = self.model.hdbscan_model.labels_[:6]
        self.model.topics_ = self.model.topics_[:6]
        self.model.topic_sizes_ = dict(Counter(self.model.topics_))
        self.model.topic_mapper_.mapping.pop(-1)
        self.identity = make_fixture(self.root, self.model)
        self.predictor.return_value = np.array([-1]), np.array([0.0])
        self.assertEqual(self.component().infer("Novel question")["topic_id"], -1)

    def test_tied_memberships_sort_by_topic_id_without_renormalization(self):
        self.membership.return_value = np.array([[.2, .2, .2]])
        result = self.component().infer("Question")
        self.assertEqual([value["topic_id"] for value in result["topic_memberships"]], [3, 8, 15])
        self.assertAlmostEqual(result["membership_sum"], .6)

    def test_unavailable_distribution_is_explicit_and_never_uses_scalar_as_distribution(self):
        self.model.hdbscan_model.prediction_data_.exemplars = None
        result = self.component().infer("Question")
        self.assertEqual(result["membership_strength"], .71)
        self.assertIsNone(result["topic_memberships"])
        self.assertIsNone(result["membership_sum"])
        self.assertEqual(result["membership_status"], "unavailable")
        self.assertEqual(result["membership_unavailable_reason"], "membership_exemplars_unavailable")
        self.membership.assert_not_called()

    def test_unsupported_membership_api_and_missing_scalar_are_independent(self):
        self.membership.side_effect = NotImplementedError("unsupported")
        self.predictor.return_value = np.array([0]), None
        result = self.component().infer("Question")
        self.assertIsNone(result["membership_strength"])
        self.assertIsNone(result["topic_memberships"])
        self.assertEqual(result["membership_unavailable_reason"], "membership_api_not_supported")

    def test_invalid_membership_values_fail_instead_of_substitution(self):
        component = self.component()
        for values in (np.array([.1, .2, .3]), np.array([[.1, .2]]), np.array([[np.nan, .1, .2]]),
                       np.array([[-.1, .2, .3]]), np.array([[.5, .5, .5]]), np.array([[0, 0, 1.1]])):
            with self.subTest(values=values):
                self.membership.return_value = values
                with self.assertRaisesRegex(ValueError, "Invalid per-cluster membership matrix"):
                    component.infer("Question")

    def test_invalid_predictions_and_strength_shapes_fail(self):
        component = self.component()
        for raw, strength in ((np.array([99]), np.array([.5])), (np.array([0.5]), np.array([.5])),
                              (np.array([0]), np.array([[.5, .5]])), (np.array([0]), np.array([np.nan]))):
            self.predictor.return_value = raw, strength
            with self.assertRaises(ValueError):
                component.infer("Question")

    def test_all_manifest_entries_verified_before_any_deserialization(self):
        (self.root / "extra_checkpoint.txt").write_bytes(b"CORRUPT")
        with self.assertRaisesRegex(ValueError, "extra_checkpoint"):
            self.component()
        self.loader.assert_not_called()
        self.factory.assert_not_called()
        self.runtime_checker.assert_not_called()

    def test_model_hash_checked_before_deserialization(self):
        path = self.root / "bertopic_model.pkl"
        original = path.read_bytes()
        path.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: bertopic_model.pkl"):
            self.component()
        self.loader.assert_not_called()

    def test_incomplete_manifest_or_external_trust_pin_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "separately trusted manifest"):
            self.component(trusted_manifest_sha256="0" * 64)
        saved = query.training.read_json(self.root / "manifest.json")
        saved["status"] = "running"
        write(self.root / "manifest.json", saved)
        with self.assertRaisesRegex(ValueError, "must be complete"):
            self.component()
        self.loader.assert_not_called()

    def test_unsafe_or_missing_manifest_artifacts_rejected_before_loading(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        manifest = query.training.read_json(path)
        manifest["artifacts"]["../outside.pkl"] = {"bytes": 0, "sha256": "0" * 64}
        write(path, manifest)
        with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
            self.component()
        path.write_bytes(original)
        (self.root / "extra_checkpoint.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/unsafe artifact"):
            self.component()
        self.loader.assert_not_called()

    def test_mismatched_runtime_stops_before_deserialization(self):
        self.runtime_checker.side_effect = ValueError("incompatible saved environment")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.component()
        self.loader.assert_not_called()

    def test_wrong_encoding_policy_helper_hashes_and_revision_rejected(self):
        original = (self.root / "embedding_config.json").read_bytes()
        for mutate in (lambda c: c.update(max_window_tokens=128), lambda c: c.update(normalize_embeddings=False),
                       lambda c: c.update(long_text_policy="truncate"),
                       lambda c: c["implementation_sha256"].update({"train_bertopic_150k.py": "0" * 64}),
                       lambda c: c["encoder"].update(revision="wrong")):
            (self.root / "embedding_config.json").write_bytes(original)
            self.change_config(mutate)
            with self.assertRaises(ValueError):
                self.component()
        self.loader.assert_not_called()

    def test_invalid_topic_mapping_column_mapping_and_prediction_state_fail(self):
        baseline = copy.deepcopy(self.model)
        for mutate in (lambda m: m.topic_mapper_.mapping.pop(2), lambda m: m.topic_mapper_.mapping.update({0: 99}),
                       lambda m: m.topic_mapper_.mapping.update({-1: 8}),
                       lambda m: m.hdbscan_model.prediction_data_.reverse_cluster_map.update({2: 90}),
                       lambda m: setattr(m.hdbscan_model, "prediction_data_", None)):
            self.model = copy.deepcopy(baseline)
            mutate(self.model)
            with self.assertRaises(ValueError):
                self.component()

    def test_source_artifacts_unchanged_reusable_and_parameters_not_toggled(self):
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        with self.component() as component:
            first = component.infer("Same question")
            second = component.infer("Same question")
            np.testing.assert_array_equal(first["embedding"], second["embedding"])
            self.assertFalse(self.model.calculate_probabilities)
            self.assertEqual(self.loader.call_count, 1)
            self.assertEqual(self.factory.call_count, 1)
            report = component.validation_report
            report["raw_cluster_to_topic"]["0"] = 999
            self.assertNotEqual(report, component.validation_report)
        with self.assertRaises(RuntimeError):
            component.infer("Question")
        self.assertEqual(before, {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_invalid_questions_and_changed_sources_fail(self):
        component = self.component()
        for question in (None, "", " \t\n", 4, ["Question"]):
            with self.assertRaises(ValueError):
                component.infer(question)
        with (self.root / "manifest.json").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "changed during"):
            component.infer("Question")

    def test_environment_checker_rejects_wrong_python_and_dependency_versions(self):
        saved = {"python": "3.12.3", "libraries": {name: "1" for name in query.LIBRARIES}}
        with mock.patch.object(query.platform, "python_version", return_value="3.11.9"):
            with self.assertRaisesRegex(ValueError, "Python 3.12"):
                query.check_environment(saved)
        with mock.patch.object(query.platform, "python_version", return_value="3.12.10"), \
             mock.patch.object(query.metadata, "version", return_value="2"):
            with self.assertRaisesRegex(ValueError, "Incompatible numpy"):
                query.check_environment(saved)
        with mock.patch.object(query.platform, "python_version", return_value="3.12.10"), \
             mock.patch.object(query.metadata, "version", return_value="1"):
            self.assertEqual(query.check_environment(saved)["saved_python"], "3.12.3")

    def test_cli_serializes_embedding_and_exposes_no_identity_bypass(self):
        with mock.patch.object(query, "QueryInference150k") as constructor:
            fake = constructor.return_value.__enter__.return_value
            fake.infer.return_value = {"embedding": np.zeros(384, dtype=np.float32), "topic_id": -1}
            fake.validation_report = {"status": "validated"}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(query.main(["--question", "Question?"]), 0)
            self.assertEqual(len(json.loads(output.getvalue())["result"]["embedding"]), 384)
            self.assertEqual(constructor.call_args.kwargs, {"device": "cpu", "trusted_manifest_sha256": None})


@unittest.skipUnless(os.environ.get("ATRAG_RUN_TINY_QUERY_TEST") == "1", "Opt-in tiny real model test; no downloads")
class TinyIntegration(unittest.TestCase):
    def test_real_pickle_mapping_and_memberships_match_bertopic_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # New synthetic 120-row model only. No saved thesis model is loaded.
            with query.runtime_cache(root / "test_runtime"):
                rng = np.random.default_rng(42)
                vectors = rng.normal(0, .01, (120, 384)).astype(np.float32)
                vectors[:60, 0] += 1
                vectors[60:, 1] += 1
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                texts = ["apple orchard fruit"] * 60 + ["ocean marine water"] * 60
                model = query.training.full.fit_model(query.training.reference_parameters(), texts, vectors)
                self.assertGreater(len(set(model.topics_) - {-1}), 0)
                environment = query.training.runtime_environment("cpu")
                artifact_dir = root / "artifacts"
                identity = make_fixture(artifact_dir, model, vectors=vectors, environment=environment,
                    save_model=lambda path: model.save(str(path), serialization="pickle", save_embedding_model=False))
                encoder = Encoder()
                encoder.encode_windows = lambda features: np.tile(vectors[0], (len(features), 1))
                with query.QueryInference150k(artifact_dir, _expected_count=120,
                        _expected_corpus_sha256=identity["sha256"], _encoder_factory=lambda *args: encoder) as component:
                    result = component.infer("A synthetic question with no downloaded encoder")
                    predicted, strengths = model.transform(["query"], embeddings=result["embedding"][None, :])
                    self.assertEqual(result["topic_id"], int(predicted[0]))
                    self.assertAlmostEqual(result["membership_strength"], float(strengths[0]), places=12)
                    model.calculate_probabilities = True
                    _, memberships = model.transform(["query"], embeddings=result["embedding"][None, :])
                    by_topic = {item["topic_id"]: item["membership"] for item in result["topic_memberships"]}
                    for topic, value in by_topic.items():
                        self.assertAlmostEqual(value, float(memberships[0, topic]), places=12)
                    self.assertAlmostEqual(result["membership_sum"], float(memberships.sum()), places=12)
                    self.assertTrue(component._model.hdbscan_model.prediction_data_ is not None)


if __name__ == "__main__":
    unittest.main()
