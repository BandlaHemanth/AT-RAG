"""Synthetic frozen-artifact fixtures; no model, encoder, network or 150k run."""

from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/shared_150k_retrieval.py"
SPEC = importlib.util.spec_from_file_location("shared_150k_retrieval", SCRIPT)
retrieval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retrieval)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus, self.matrix_path, self.assignments, self.manifest = [self.root / name for name in
            ("corpus.jsonl", "embeddings.npy", "occurrence_topics.jsonl", "manifest.json")]
        self.labels = [7, 20, 7, 20, 7, -1, 42, 20, 42, -1, 7, 20, 42, -1, 20, 7, -1, 42]
        self.vectors = np.zeros((len(self.labels), 384), dtype=np.float32)
        self.vectors[:, :2] = np.array([
            [1, 0], [1, 0], [1, 0], [.8, .6], [-1, 0], [1, 0], [.6, .8], [0, 1], [0, -1],
            [1, 1], [.5, np.sqrt(.75)], [-.5, np.sqrt(.75)], [1, 0], [-1, 0], [1, 0], [0, 1], [1, 0], [0, 0],
        ], dtype=np.float32)
        self.vectors[17, 2] = 1
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.query = np.zeros(384, dtype=np.float32)
        self.query[0] = 1
        self.rows, self.topics = [], []
        for index, topic in enumerate(self.labels):
            text = " Repeated original Ω text\n " if index in (0, 1, 16) else f"Original paragraph {index}"
            row = {"occurrence_id": f"2wikimultihopqa:train:{'a'*64}:{index//2+1}:{index%2}",
                   "paragraph_text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "question_id": f"q-{index//2}", "source_split": "train", "source_line": index//2+1,
                   "context_position": index % 2, "idx": index % 2, "title": f"Title {index} Ω"}
            assignment = {key: value for key, value in row.items() if key != "paragraph_text"}
            assignment.update(row=index, topic=topic, bertopic=f"Topic {topic}",
                              membership_strength=0.0 if topic == -1 else 0.8)
            self.rows.append(row)
            self.topics.append(assignment)
        self.write_rows()
        np.save(self.matrix_path, self.vectors, allow_pickle=False)
        counts = Counter(self.labels)
        self.saved = {
            "schema_version": 1, "status": "complete", "occurrences": len(self.labels),
            "corpus_sha256": sha(self.corpus), "encoder_id": retrieval.ENCODER_ID,
            "encoder_revision": retrieval.ENCODER_REVISION,
            "topic_summary": {"occurrences": len(self.labels), "non_outlier_topics": 3,
                              "outlier_occurrences": counts[-1], "outlier_fraction": counts[-1] / len(self.labels),
                              "topic_sizes": {str(key): value for key, value in counts.items()}},
            "artifacts": {},
        }
        self.expected_sha = sha(self.corpus)
        self.refresh_manifest()

    def write_rows(self):
        self.corpus.write_bytes(b"".join(json_bytes(row) for row in self.rows))
        self.assignments.write_bytes(b"".join(json_bytes(row) for row in self.topics))

    def refresh_manifest(self):
        for path in (self.matrix_path, self.assignments):
            self.saved["artifacts"][path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        self.manifest.write_bytes(json_bytes(self.saved))

    def engine(self, **options):
        settings = {"block_size": 3, "expected_count": len(self.labels), "expected_corpus_sha256": self.expected_sha}
        settings.update(options)
        engine = retrieval.Shared150kRetrieval(self.corpus, self.matrix_path, self.assignments, self.manifest, **settings)
        self.addCleanup(engine.close)
        return engine

    @staticmethod
    def indices(results):
        return [result["row"] for result in results]

    def oracle(self, labels=None, k=5):
        # Independent high-accuracy row-wise reference. A BLAS matrix multiply can
        # round identical vectors differently at different row positions.
        vectors = self.vectors.astype(np.float64)
        query = self.query.astype(np.float64)
        query_norm = math.sqrt(math.fsum(float(value) ** 2 for value in query))
        scores = [math.fsum(float(value) * float(q) for value, q in zip(vector, query)) /
                  (math.sqrt(math.fsum(float(value) ** 2 for value in vector)) * query_norm)
                  for vector in vectors]
        eligible = [i for i, topic in enumerate(self.labels) if labels is None or topic in labels]
        return sorted(eligible, key=lambda i: (-float(scores[i]), i))[:k]

    def test_single_topic_retrieval(self):
        engine = self.engine()
        results = engine.retrieve(self.query, mode="single", topic_id=7, k=5)
        self.assertEqual(self.indices(results), [0, 2, 10, 15, 4])
        self.assertEqual({r["topic"] for r in results}, {7})

    def test_multi_topic_or_semantics_and_global_ranking(self):
        engine = self.engine()
        results = engine.retrieve(self.query, mode="multi", topic_ids={7, 42}, k=5)
        self.assertEqual(self.indices(results), self.oracle({7, 42}))
        self.assertEqual(self.indices(results)[:3], [0, 2, 12])
        self.assertTrue(all(r["topic"] in {7, 42} for r in results))
        self.assertEqual(results, engine.retrieve(self.query, mode="multi", topic_ids=[42, 7, 42], k=5))

    def test_unfiltered_k5_k10_and_configurable_positive_k(self):
        engine = self.engine()
        for k in (1, 5, 10, 18, 1000):
            with self.subTest(k=k):
                results = engine.retrieve(self.query, mode="unfiltered", k=k)
                self.assertEqual(self.indices(results), self.oracle(k=k))
                self.assertEqual(len(results), min(k, len(self.labels)))
        self.assertEqual(self.indices(engine.retrieve(self.query, mode="unfiltered", k=5)), [0, 1, 2, 5, 12])

    def test_outliers_only_enter_filters_when_selected_explicitly(self):
        engine = self.engine()
        strict = engine.retrieve(self.query, mode="single", topic_id=7, k=100)
        self.assertNotIn(-1, {r["topic"] for r in strict})
        multi = engine.retrieve(self.query, mode="multi", topic_ids={7, 20, 42}, k=100)
        self.assertNotIn(-1, {r["topic"] for r in multi})
        outliers = engine.retrieve(self.query, mode="single", topic_id=-1, k=10)
        self.assertEqual(self.indices(outliers), [5, 16, 9, 13])
        with_outliers = engine.retrieve(self.query, mode="multi", topic_ids={7, -1}, k=10)
        self.assertEqual(self.indices(with_outliers), self.oracle({7, -1}, k=10))
        self.assertIn(-1, {r["topic"] for r in engine.retrieve(self.query, mode="unfiltered", k=10)})

    def test_duplicate_occurrences_preserved_and_top_k_not_refilled(self):
        results = self.engine().retrieve(self.query, mode="unfiltered", k=2)
        self.assertEqual(self.indices(results), [0, 1])
        self.assertEqual(results[0]["paragraph_text"], " Repeated original Ω text\n ")
        self.assertEqual(results[0]["paragraph_text"], results[1]["paragraph_text"])
        self.assertEqual(results[0]["text_sha256"], results[1]["text_sha256"])
        self.assertNotEqual(results[0]["occurrence_id"], results[1]["occurrence_id"])
        self.assertEqual({r["topic"] for r in results}, {7, 20})

    def test_exact_ties_cross_block_boundaries_and_modes(self):
        expected = self.engine(block_size=1).retrieve(self.query, mode="unfiltered", k=18)
        tied = [r["row"] for r in expected if r["similarity"] == 1.0]
        self.assertEqual(tied, [0, 1, 2, 5, 12, 14, 16])
        for size in (2, 3, 4, 7, 4096):
            with self.subTest(block_size=size):
                engine = self.engine(block_size=size)
                self.assertEqual(engine.retrieve(self.query, mode="unfiltered", k=18), expected)
                self.assertEqual(engine.retrieve(self.query, mode="multi", topic_ids=engine.available_topics, k=18), expected)
                by_row = {r["row"]: r["similarity"] for r in expected}
                for result in engine.retrieve(self.query, mode="single", topic_id=20, k=18):
                    self.assertEqual(result["similarity"], by_row[result["row"]])

    def test_cosine_scores_and_original_metadata(self):
        engine = self.engine()
        results = engine.retrieve(self.query.astype(np.float64), mode="unfiltered", k=18)
        for result in results:
            row, vector = self.rows[result["row"]], self.vectors[result["row"]].astype(np.float64)
            self.assertAlmostEqual(result["similarity"], float(vector[0] / np.linalg.norm(vector)), places=14)
            self.assertEqual(result["paragraph_text"], row["paragraph_text"])
            self.assertEqual(result["occurrence_id"], row["occurrence_id"])
            self.assertEqual(result["text_sha256"], row["text_sha256"])
            self.assertEqual(set(result), {"row", "occurrence_id", "topic", "text_sha256", "similarity", "paragraph_text"})

    def test_seeded_dense_vectors_match_independent_cosine_reference(self):
        rng = np.random.default_rng(42)
        self.vectors = rng.normal(size=(18, 384)).astype(np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors[1] = self.vectors[16] = self.vectors[0]  # preserve natural repeated text/vector pairs
        self.query = rng.normal(size=384).astype(np.float32)
        self.query /= np.linalg.norm(self.query)
        np.save(self.matrix_path, self.vectors, allow_pickle=False)
        self.refresh_manifest()
        reference = None
        for block_size in (1, 5, 4096):
            engine = self.engine(block_size=block_size)
            for selected, settings in ((None, {"mode": "unfiltered"}), ({7}, {"mode": "single", "topic_id": 7}),
                                       ({7, 20}, {"mode": "multi", "topic_ids": [20, 7]}),
                                       ({-1}, {"mode": "single", "topic_id": -1})):
                for k in (1, 5, 10, 30):
                    results = engine.retrieve(self.query, k=k, **settings)
                    self.assertEqual(self.indices(results), self.oracle(selected, k=k))
            all_results = engine.retrieve(self.query, mode="unfiltered", k=18)
            if reference is None:
                reference = all_results
            else:
                self.assertEqual(all_results, reference)

    def test_empty_and_nonexistent_topic_filters_return_empty(self):
        engine = self.engine()
        for topics in ([], (), set(), frozenset(), np.array([], dtype=np.int64), [999999999999999999999999]):
            with self.subTest(topics=topics):
                self.assertEqual(engine.retrieve(self.query, mode="multi", topic_ids=topics), [])
        self.assertEqual(engine.retrieve(self.query, mode="single", topic_id=999999999999999999999999), [])

    def test_invalid_query_dimensions_types_values_and_normalization(self):
        engine = self.engine()
        bad = [np.zeros(383), np.zeros((1, 384)), np.zeros((384, 1)), np.zeros(384),
               np.ones(384), np.full(384, np.nan), np.full(384, np.inf),
               self.query.astype(np.complex64), self.query.astype(np.int32), ["1"] * 384, None]
        for query in bad:
            with self.subTest(query_type=type(query), shape=np.shape(query)):
                with self.assertRaises(ValueError):
                    engine.retrieve(query, mode="unfiltered")
                with self.assertRaises(ValueError):
                    engine.retrieve(query, mode="multi", topic_ids=[])
        self.assertEqual(engine.retrieve(self.query.tolist(), mode="unfiltered"),
                         engine.retrieve(self.query, mode="unfiltered"))

    def test_invalid_modes_topic_ids_filters_and_k(self):
        engine = self.engine()
        bad = [dict(mode="unknown"), dict(mode=[]), dict(mode="single"),
               dict(mode="unfiltered", topic_id=7), dict(mode="unfiltered", topic_ids=[]),
               dict(mode="single", topic_id=7, topic_ids=[7]), dict(mode="multi"),
               dict(mode="multi", topic_ids=7), dict(mode="multi", topic_ids="7"),
               dict(mode="multi", topic_ids={7: True}), dict(mode="multi", topic_ids=[[7]]),
               dict(mode="multi", topic_ids=np.array([[7]])), dict(mode="multi", topic_ids=[7], topic_id=7)]
        for topic in (-2, 1.0, "7", True, np.bool_(False), None, np.nan):
            bad += [dict(mode="single", topic_id=topic), dict(mode="multi", topic_ids=[topic])]
        for settings in bad:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    engine.retrieve(self.query, **settings)
        for k in (0, -1, 1.5, True, np.bool_(True), "5"):
            with self.assertRaises(ValueError):
                engine.retrieve(self.query, mode="unfiltered", k=k)
        self.assertEqual(len(engine.retrieve(self.query, mode="single", topic_id=np.int64(7), k=np.int64(5))), 5)

    def test_sources_and_query_are_read_only_and_integrity_is_not_repeated(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        for path in self.root.iterdir():
            path.chmod(0o444)
        with mock.patch.object(retrieval, "_sha256", wraps=retrieval._sha256) as hashed:
            engine = self.engine()
            self.assertIsInstance(engine._matrix, np.memmap)
            self.assertFalse(engine._matrix.flags.writeable)
            with self.assertRaises(ValueError):
                engine._matrix[0, 0] = 2
            calls = hashed.call_count
            query_before = self.query.copy()
            self.query.flags.writeable = False
            for mode, settings in (("single", {"topic_id": 7}), ("multi", {"topic_ids": {7, 20}}), ("unfiltered", {})):
                engine.retrieve(self.query, mode=mode, k=10, **settings)
            self.assertEqual(hashed.call_count, calls)
            np.testing.assert_array_equal(self.query, query_before)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_production_defaults_require_150000_rows_and_frozen_identity(self):
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "exactly 150000"):
            retrieval.Shared150kRetrieval(self.corpus, self.matrix_path, self.assignments, self.manifest)
        self.saved["corpus_sha256"] = "0" * 64
        self.refresh_manifest()
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "frozen protocol"):
            self.engine()
        self.saved["corpus_sha256"] = self.expected_sha
        self.refresh_manifest()
        # A byte-only JSON whitespace change preserves row metadata/text hashes,
        # so this specifically exercises the frozen whole-file hash check.
        self.corpus.write_bytes(self.corpus.read_bytes().replace(b'": ', b'":  ', 1))
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "Corpus SHA-256"):
            self.engine()

    def test_manifest_complete_schema_encoder_and_artifact_records_required(self):
        original = json.loads(json.dumps(self.saved))
        mutations = [lambda d: d.update(status="running"), lambda d: d.update(schema_version=2),
                     lambda d: d.update(occurrences=True), lambda d: d.update(encoder_revision="wrong"),
                     lambda d: d["artifacts"].pop("embeddings.npy"),
                     lambda d: d["artifacts"]["embeddings.npy"].update(sha256="invalid")]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                data = json.loads(json.dumps(original))
                mutate(data)
                self.manifest.write_bytes(json_bytes(data))
                with self.assertRaises(retrieval.ArtifactValidationError):
                    self.engine()

    def test_corrupted_embedding_and_assignment_hashes_rejected(self):
        original = self.matrix_path.read_bytes()
        with self.matrix_path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "Embedding SHA-256 mismatch"):
            self.engine()
        self.matrix_path.write_bytes(original)
        data = self.assignments.read_bytes().replace(b'"membership_strength": 0.8', b'"membership_strength": 0.7', 1)
        self.assignments.write_bytes(data)
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "Assignment SHA-256 mismatch"):
            self.engine()

    def test_invalid_embedding_shapes_dtype_values_and_norms_rejected(self):
        for name, vectors in (("dimension", self.vectors[:, :383]), ("rows", self.vectors[:-1]),
                              ("dtype", self.vectors.astype(np.float64)), ("zero", np.zeros_like(self.vectors)),
                              ("nan", np.full_like(self.vectors, np.nan)), ("scale", self.vectors * 2)):
            with self.subTest(case=name):
                np.save(self.matrix_path, vectors, allow_pickle=False)
                self.refresh_manifest()
                with self.assertRaises(retrieval.ArtifactValidationError):
                    self.engine()

    def test_misaligned_assignment_rows_occurrences_and_text_hashes_rejected(self):
        original = json.loads(json.dumps(self.topics))
        for name in ("row-order", "occurrence", "text", "provenance", "topic-label", "topic-type", "provenance-type"):
            with self.subTest(case=name):
                self.topics = json.loads(json.dumps(original))
                if name == "row-order":
                    self.topics[0], self.topics[1] = self.topics[1], self.topics[0]
                elif name == "occurrence":
                    self.topics[0]["occurrence_id"] = self.topics[1]["occurrence_id"]
                elif name == "text":
                    self.topics[0]["text_sha256"] = "0" * 64
                elif name == "provenance":
                    self.topics[0]["question_id"] = "wrong question"
                elif name == "topic-label":
                    self.topics[0]["bertopic"] = "Topic 999"
                elif name == "provenance-type":
                    self.topics[0]["idx"] = False  # JSON bool must not match local idx=0.
                else:
                    self.topics[0]["topic"] = True
                self.write_rows()
                self.refresh_manifest()
                with self.assertRaises(retrieval.ArtifactValidationError):
                    self.engine()

    def test_wrong_corpus_text_hash_split_or_duplicate_ids_rejected(self):
        original_rows, original_topics = json.loads(json.dumps(self.rows)), json.loads(json.dumps(self.topics))
        for name in ("text", "split", "duplicate"):
            with self.subTest(case=name):
                self.rows, self.topics = json.loads(json.dumps(original_rows)), json.loads(json.dumps(original_topics))
                if name == "text":
                    self.rows[0]["paragraph_text"] = "Modified text"
                elif name == "split":
                    self.rows[0]["source_split"] = "dev"
                else:
                    self.rows[-1] = dict(self.rows[0])
                    self.topics[-1] = dict(self.topics[0], row=len(self.labels)-1)
                self.write_rows()
                self.expected_sha = self.saved["corpus_sha256"] = sha(self.corpus)
                self.refresh_manifest()
                with self.assertRaises(retrieval.ArtifactValidationError):
                    self.engine()

    def test_missing_extra_and_summary_mismatched_rows_rejected(self):
        self.assignments.write_bytes(self.assignments.read_bytes().splitlines(keepends=True)[0])
        self.refresh_manifest()
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "too small"):
            self.engine()
        self.write_rows()
        with self.assignments.open("ab") as stream:
            stream.write(b"\n")
        self.refresh_manifest()
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "extra rows"):
            self.engine()
        self.write_rows()
        self.saved["topic_summary"]["topic_sizes"]["7"] += 1
        self.refresh_manifest()
        with self.assertRaisesRegex(retrieval.ArtifactValidationError, "topic sizes"):
            self.engine()

    def test_source_mutation_after_initialization_is_detected_without_rehashing(self):
        engine = self.engine()
        with self.corpus.open("ab") as stream:
            stream.write(b"\n")
        with mock.patch.object(retrieval, "_sha256", side_effect=AssertionError("Must not rehash per query")):
            with self.assertRaisesRegex(retrieval.ArtifactValidationError, "changed after validation"):
                engine.retrieve(self.query, mode="unfiltered")

    def test_context_manager_closes_engine_and_report_is_a_copy(self):
        with self.engine() as engine:
            report = engine.validation_report
            self.assertEqual(report["unique_occurrence_ids"], 18)
            self.assertEqual(engine.available_topics, (-1, 7, 20, 42))
            report["topic_sizes"]["7"] = 999
            self.assertNotEqual(report, engine.validation_report)
            with self.assertRaises(AttributeError):
                engine.occurrence_count = 1
        engine.close()
        with self.assertRaises(RuntimeError):
            engine.retrieve(self.query, mode="unfiltered")

    def test_cli_uses_existing_query_vector_and_no_test_identity_overrides(self):
        query = self.root / "query.npy"
        np.save(query, self.query, allow_pickle=False)
        with mock.patch.object(retrieval, "Shared150kRetrieval") as constructor:
            fake = constructor.return_value.__enter__.return_value
            fake.retrieve.return_value = [{"row": 0}]
            fake.validation_report = {"status": "validated"}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(retrieval.main(["--query-embedding", str(query), "--mode", "unfiltered", "--k", "10"]), 0)
            self.assertEqual(json.loads(output.getvalue())["results"], [{"row": 0}])
            self.assertEqual(constructor.call_args.kwargs, {"block_size": 4096})
            self.assertEqual(fake.retrieve.call_args.kwargs, {"mode": "unfiltered", "topic_id": None, "topic_ids": None, "k": 10})


if __name__ == "__main__":
    unittest.main()
