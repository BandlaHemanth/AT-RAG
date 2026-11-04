"""Offline synthetic tests; opt-in tiny fit uses no encoder/network/full corpus."""

from collections import Counter, namedtuple
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_bertopic_150k.py"
SPEC = importlib.util.spec_from_file_location("train_bertopic_150k", SCRIPT)
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)
Disk = namedtuple("Disk", "total used free")


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": [ord(char) for char in text]}

    def prepare_for_model(self, ids, add_special_tokens, truncation, return_attention_mask):
        assert add_special_tokens is True and truncation is False
        return {"input_ids": [101] + list(ids) + [102], "attention_mask": [1] * (len(ids) + 2)}


class Encoder:
    def __init__(self, *args):
        self.tokenizer = Tokenizer()
        self.seen_tokens, self.window_batch_sizes = [], []

    def encode_windows(self, features):
        self.window_batch_sizes.append(len(features))
        vectors = np.zeros((len(features), job.DIMENSIONS), dtype=job.DTYPE)
        for i, feature in enumerate(features):
            content = feature["input_ids"][1:-1]
            self.seen_tokens.extend(content)
            for token in content:
                vectors[i, token % job.DIMENSIONS] += 1
        return vectors

    def memory_statistics(self):
        return {"device": "cpu", "cuda_used": False}


class Model:
    def __init__(self, count):
        self.topics_ = [(-1 if i % 5 == 0 else i % 2) for i in range(count)]
        self.probabilities_ = np.array([0.0 if t == -1 else 0.8 for t in self.topics_])
        self.topic_sizes_ = dict(Counter(self.topics_))
        self.hdbscan_model = types.SimpleNamespace(prediction_data=True, prediction_data_=object())
        self.umap_model = types.SimpleNamespace(transform=lambda x: x[:, :5])

    def get_topics(self):
        return {topic: [("term", 0.5)] for topic in self.topic_sizes_}

    def save(self, path, serialization, save_embedding_model):
        assert serialization == "pickle" and save_embedding_model is False
        Path(path).write_bytes(job.json_bytes({"count": len(self.topics_)}))

    @staticmethod
    def load(path):
        return Model(job.read_json(path)["count"])


def synthetic_corpus(path, texts):
    rows = []
    for i, text in enumerate(texts):
        rows.append({"occurrence_id": f"2wikimultihopqa:train:{'b'*64}:{i+1}:0",
                     "paragraph_text": text, "text_sha256": job.preparation.text_hash(text),
                     "question_id": f"q-{i}", "source_split": "train", "source_line": i + 1,
                     "context_position": 0, "idx": 0, "title": "Original Ω title"})
    path.write_bytes(b"".join(job.json_bytes(row) for row in rows))
    return {"sha256": job.sha256(path), "occurrences": len(texts), "unique_texts": len(set(texts)),
            "duplicate_excess_occurrences": len(texts) - len(set(texts))}


def stage_fake_encoder(output, reuse=None, encoder_dir=None):
    if reuse is not None:
        manifest = job.read_json(reuse / "encoder_manifest.json")
    else:
        manifest = {"model_id": job.MODEL_ID, "revision": job.MODEL_REVISION,
                    "files": {"fake": {"sha256": "pinned-synthetic-weights"}}}
    (output / "encoder").mkdir()
    job.write_json(output / "encoder_manifest.json", manifest)
    return manifest


def environment(device):
    return {"python": "synthetic-python", "libraries": {}, "platform": "test",
            "device_identity": {"machine": "synthetic"}, "cublas_workspace_config": ":4096:8"}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus.jsonl"
        self.texts = ["alpha training text", " repeated Ω text ", "beta training text"] * 4
        self.expected = synthetic_corpus(self.corpus, self.texts)
        self.factory = mock.Mock(side_effect=Encoder)
        self.trainer = mock.Mock(side_effect=lambda p, texts, matrix: Model(len(texts)))
        self.loader = mock.Mock(side_effect=Model.load)
        self.snapshot = mock.patch.object(job.full, "memory_snapshot", return_value={
            "available_memory_bytes": 64 * job.GIB, "process_rss_mib": 10, "process_peak_rss_mib": 20}).start()
        self.addCleanup(mock.patch.stopall)
        self.disk = mock.patch.object(job.shutil, "disk_usage", return_value=Disk(100 * job.GIB, 0, 100 * job.GIB)).start()
        mock.patch.dict(os.environ, {**job.THREAD_SETTINGS, "PYTHONHASHSEED": "42"}).start()
        mock.patch("sys.stdout").start()

    def run_job(self, name="run", **options):
        settings = {"expected": self.expected, "encoder_stager": stage_fake_encoder,
                    "encoder_factory": self.factory, "trainer": self.trainer, "loader": self.loader,
                    "environment_getter": environment, "batch_size": 4, "window_batch_size": 3,
                    "min_ram_gib": 0, "min_disk_gib": 0}
        settings.update(options)
        return job.run_training(self.corpus, self.root / name, **settings)

    def test_fit_receives_every_original_occurrence_and_readonly_mmap(self):
        self.run_job()
        parameters, texts, vectors = self.trainer.call_args.args
        self.assertEqual(texts, self.texts)
        self.assertEqual(len(texts), 12)
        self.assertEqual(texts.count(" repeated Ω text "), 4)
        self.assertIsInstance(vectors, np.memmap)
        self.assertFalse(vectors.flags.writeable)
        self.assertEqual(vectors.shape, (12, 384))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(parameters["bertopic"]["min_topic_size"], 50)
        self.assertEqual(parameters["hdbscan"]["min_cluster_size"], 50)
        self.assertIsNone(parameters["hdbscan"]["min_samples"])
        self.assertTrue(parameters["hdbscan"]["prediction_data"])
        self.assertFalse(parameters["bertopic"]["calculate_probabilities"])
        np.testing.assert_allclose(vectors[0], vectors[3])

    def test_assignments_model_reload_statistics_and_all_artifact_hashes(self):
        manifest = self.run_job()
        output = self.root / "run"
        rows = [json.loads(line) for line in (output / "occurrence_topics.jsonl").read_bytes().splitlines()]
        self.assertEqual([r["row"] for r in rows], list(range(12)))
        self.assertEqual(len({r["occurrence_id"] for r in rows}), 12)
        self.assertEqual(len({r["text_sha256"] for r in rows}), 3)
        self.assertEqual({r["idx"] for r in rows}, {0})
        self.assertEqual(rows[0]["topic"], -1)
        self.assertEqual(rows[0]["membership_strength"], 0)
        self.assertEqual(manifest["status"], "complete")
        self.loader.assert_called_once()
        self.assertEqual(job.sha256(self.corpus), self.expected["sha256"])
        for name, evidence in manifest["artifacts"].items():
            self.assertEqual(evidence, job.file_evidence(output / name))
        stats = job.read_json(output / "runtime_statistics.json")
        self.assertEqual(stats["embedding"]["occurrences"], 12)
        self.assertEqual(stats["embedding"]["encoding_statistics"]["content_tokens"], sum(map(len, self.texts)))
        self.assertIn("before_fit", stats["memory"])
        self.assertFalse(stats["embedding"]["device_memory"]["cuda_used"])
        self.assertEqual(self.disk.call_count, 1)  # one admission gate, no polling/repeated disk gates

    def test_long_text_covers_every_token_in_bounded_window_batches(self):
        encoder = Encoder()
        text = "a" * 900 + "UNIQUE TAIL Ω"
        vectors, stats = job.encode_occurrences([text, "short"], encoder, 2)
        self.assertEqual(encoder.seen_tokens, [ord(c) for c in text + "short"])
        self.assertLessEqual(max(encoder.window_batch_sizes), 2)
        self.assertEqual(stats["multiwindow_occurrences"], 1)
        self.assertEqual(stats["content_tokens"], len(text) + 5)
        self.assertEqual(vectors.shape, (2, 384))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        changed, _ = job.encode_occurrences(["a" * len(text)], Encoder(), 2)
        self.assertFalse(np.allclose(vectors[0], changed[0]))

    def test_short_paragraph_matches_normalized_single_window(self):
        encoder = Encoder()
        vector, stats = job.encode_occurrences(["alpha"], encoder, 2)
        raw = Encoder().encode_windows([encoder.tokenizer.prepare_for_model(
            [ord(c) for c in "alpha"], True, False, True)])[0]
        np.testing.assert_allclose(vector[0], raw / np.linalg.norm(raw), atol=1e-7)
        self.assertEqual(stats["windows"], 1)

    def test_complete_embeddings_reused_without_loading_encoder(self):
        self.run_job("first")
        source = self.root / "first"
        before = {str(p.relative_to(source)): job.sha256(p) for p in source.rglob("*") if p.is_file()}
        no_encoder = mock.Mock(side_effect=AssertionError("Should not encode completed work"))
        self.run_job("second", reuse=source, encoder_factory=no_encoder)
        no_encoder.assert_not_called()
        second = job.read_json(self.root / "second/embedding_statistics.json")
        self.assertEqual(second["reused_occurrences"], 12)
        self.assertEqual(second["newly_encoded_occurrences"], 0)
        self.assertEqual(job.sha256(source / "embeddings.npy"), job.sha256(self.root / "second/embeddings.npy"))
        self.assertEqual(before, {str(p.relative_to(source)): job.sha256(p) for p in source.rglob("*") if p.is_file()})

    def test_interrupted_embedding_reuses_only_completed_prefix(self):
        encoder = Encoder()
        original = encoder.encode_windows
        calls = 0
        def interrupted(features):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("synthetic interruption")
            return original(features)
        encoder.encode_windows = interrupted
        # Four windows per batch; the first batch is committed in one forward pass.
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.run_job("interrupted", encoder_factory=lambda *a: encoder, window_batch_size=8)
        failed = self.root / "interrupted"
        self.assertTrue((failed / "embedding_batches/batch_000000.json").is_file())
        self.assertFalse((failed / "manifest.json").exists())
        self.run_job("recovered", reuse=failed, window_batch_size=8)
        stats = job.read_json(self.root / "recovered/embedding_statistics.json")
        self.assertEqual(stats["reused_occurrences"], 4)
        self.assertEqual(stats["newly_encoded_occurrences"], 8)
        self.assertEqual(self.trainer.call_args.args[1], self.texts)

    def test_fit_failure_retains_embeddings_and_explicit_failure_stage(self):
        with self.assertRaisesRegex(MemoryError, "synthetic fit OOM"):
            self.run_job("failed", trainer=mock.Mock(side_effect=MemoryError("synthetic fit OOM")))
        failure = job.read_json(self.root / "failed/failure.json")
        self.assertEqual(failure["stage"], "fitting_all_occurrences")
        self.assertFalse((self.root / "failed/manifest.json").exists())
        self.assertTrue((self.root / "failed/embedding_batches/batch_000002.json").is_file())
        self.run_job("retry", reuse=self.root / "failed")
        self.assertEqual(job.read_json(self.root / "retry/embedding_statistics.json")["reused_occurrences"], 12)

    def test_corrupt_committed_batch_refused_before_fitting(self):
        self.run_job("first")
        self.trainer.reset_mock()
        array = self.root / "first/embedding_batches/batch_000000.npy"
        with array.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"X")
        with self.assertRaisesRegex(ValueError, "Cache integrity failure"):
            self.run_job("corrupt", reuse=self.root / "first")
        self.trainer.assert_not_called()
        self.assertFalse((self.root / "corrupt/manifest.json").exists())

    def test_wrong_batch_occurrence_mapping_refused(self):
        self.run_job("first")
        marker = self.root / "first/embedding_batches/batch_000000.json"
        data = job.read_json(marker)
        data["occurrence_ids_sha256"] = "wrong"
        marker.write_bytes(job.json_bytes(data))
        with self.assertRaisesRegex(ValueError, "row/configuration identity mismatch"):
            self.run_job("wrong-mapping", reuse=self.root / "first")

    def test_changed_cache_configuration_refused(self):
        self.run_job("first")
        with self.assertRaisesRegex(ValueError, "configuration/code/runtime differs"):
            self.run_job("wrong-settings", reuse=self.root / "first", batch_size=3)

    def test_verified_encoder_files_copied_without_modifying_source(self):
        source = self.root / "verified-encoder"
        source.mkdir()
        files = {}
        for name in job.REQUIRED_MODEL_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic encoder file " + name).encode())
            files[name] = job.sha256(path)
        (source / "embedding_verification_manifest.json").write_bytes(job.json_bytes({
            "model_id": job.MODEL_ID, "revision": job.MODEL_REVISION, "sha256": files}))
        before = {str(p.relative_to(source)): job.sha256(p) for p in source.rglob("*") if p.is_file()}
        output = self.root / "copied-encoder"
        output.mkdir()
        manifest = job.stage_encoder(output, encoder_dir=source)
        self.assertEqual(manifest["revision"], job.MODEL_REVISION)
        for name in files:
            self.assertEqual((source / name).read_bytes(), (output / "encoder" / name).read_bytes())
        self.assertEqual(before, {str(p.relative_to(source)): job.sha256(p) for p in source.rglob("*") if p.is_file()})
        again = self.root / "copied-from-run"
        again.mkdir()
        self.assertEqual(job.stage_encoder(again, reuse=output), manifest)

    def test_corrupt_or_wrong_revision_local_encoder_refused(self):
        source = self.root / "bad-encoder"
        source.mkdir()
        path = source / "embedding_verification_manifest.json"
        path.write_bytes(job.json_bytes({"model_id": job.MODEL_ID, "revision": "wrong", "sha256": {}}))
        output = self.root / "wrong-encoder"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "wrong model/revision"):
            job.stage_encoder(output, encoder_dir=source)
        files = {}
        for name in job.REQUIRED_MODEL_FILES:
            file = source / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"synthetic")
            files[name] = job.sha256(file)
        path.write_bytes(job.json_bytes({"model_id": job.MODEL_ID, "revision": job.MODEL_REVISION, "sha256": files}))
        (source / "model.safetensors").write_bytes(b"corrupted weights")
        output = self.root / "corrupt-encoder"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "Cache integrity failure"):
            job.stage_encoder(output, encoder_dir=source)

    def test_invalid_window_vectors_are_not_checkpointed(self):
        for name in ("nonfinite", "wrong-dimension"):
            encoder = Encoder()
            vectors = np.zeros((3, 384), dtype=job.DTYPE)
            if name == "nonfinite":
                vectors[0, 0] = np.nan
            else:
                vectors = vectors[:, :12]
            encoder.encode_windows = mock.Mock(return_value=vectors)
            with self.assertRaisesRegex(ValueError, "Invalid encoder window vectors"):
                self.run_job(name, encoder_factory=lambda *args: encoder)
            self.assertFalse((self.root / name / "embedding_batches/batch_000000.json").exists())

    def test_orphan_without_commit_marker_is_recomputed_in_new_directory(self):
        self.run_job("first")
        marker = self.root / "first/embedding_batches/batch_000001.json"
        marker.unlink()
        marker.with_name(marker.name + ".part").write_bytes(b'{"interrupted":')
        self.run_job("new", reuse=self.root / "first")
        self.assertEqual(job.read_json(self.root / "new/embedding_statistics.json")["reused_occurrences"], 4)
        self.assertFalse(marker.exists())

    def test_checkpoint_commit_is_atomic_and_non_overwriting(self):
        marker = self.root / "checkpoint.json"
        with mock.patch.object(job.os, "link", side_effect=OSError("synthetic crash before commit")):
            with self.assertRaisesRegex(OSError, "before commit"):
                job.write_json(marker, {"complete": True})
        self.assertFalse(marker.exists())
        self.assertTrue(marker.with_name(marker.name + ".part").exists())
        existing = self.root / "existing.json"
        existing.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            job.write_json(existing, {"replacement": True})
        self.assertEqual(existing.read_bytes(), b"preserve")

    def test_resource_preflight_failure_does_not_load_model_or_reduce_rows(self):
        self.snapshot.return_value = {"available_memory_bytes": 1}
        stager = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Resource preflight failed"):
            self.run_job(encoder_stager=stager)
        stager.assert_not_called()
        self.trainer.assert_not_called()
        self.assertEqual(job.read_json(self.root / "run/preflight.json")["status"], "failed")
        self.assertFalse((self.root / "run/manifest.json").exists())

    def test_wrong_corpus_identity_and_duplicate_occurrence_ids_refused(self):
        expected = dict(self.expected, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Frozen corpus identity mismatch"):
            self.run_job("wrong-sha", expected=expected)
        rows = [json.loads(line) for line in self.corpus.read_bytes().splitlines()]
        rows[-1] = dict(rows[0])
        self.corpus.write_bytes(b"".join(job.json_bytes(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "Duplicate occurrence_id"):
            self.run_job("duplicate")

    def test_forbidden_fields_invalid_texts_and_nontraining_rows_refused(self):
        original = self.corpus.read_bytes()
        for name, mutation in (("gold", {"is_supporting": True}), ("empty", {"paragraph_text": ""}),
                               ("split", {"source_split": "dev"}), ("local", {"idx": True})):
            with self.subTest(case=name):
                rows = [json.loads(line) for line in original.splitlines()]
                rows[-1].update(mutation)
                self.corpus.write_bytes(b"".join(job.json_bytes(row) for row in rows))
                with self.assertRaises(ValueError):
                    self.run_job(name)
        self.trainer.assert_not_called()

    def test_dense_probabilities_lost_prediction_data_or_reduced_rows_fail(self):
        for name in ("dense", "prediction", "rows"):
            with self.subTest(case=name):
                model = Model(12)
                if name == "dense":
                    model.probabilities_ = np.ones((12, 2))
                elif name == "prediction":
                    model.hdbscan_model.prediction_data_ = None
                else:
                    model.topics_ = model.topics_[:-1]
                with self.assertRaises(ValueError):
                    self.run_job(name, trainer=mock.Mock(return_value=model))
                self.assertFalse((self.root / name / "manifest.json").exists())

    def test_existing_outputs_files_and_dangling_symlinks_never_modified(self):
        for name, kind in (("directory", "dir"), ("file", "file"), ("link", "link")):
            path = self.root / name
            if kind == "dir":
                path.mkdir()
                (path / "sentinel").write_bytes(b"preserve")
            elif kind == "file":
                path.write_bytes(b"preserve")
            else:
                path.symlink_to(self.root / "not-present")
            with self.assertRaises(FileExistsError):
                self.run_job(name)
            if kind == "dir":
                self.assertEqual((path / "sentinel").read_bytes(), b"preserve")
            elif kind == "file":
                self.assertEqual(path.read_bytes(), b"preserve")
            else:
                self.assertTrue(path.is_symlink())

    def test_reuse_source_cannot_contain_the_new_output_directory(self):
        source = self.root / "source"
        source.mkdir()
        with self.assertRaisesRegex(ValueError, "separate directories"):
            job.run_training(self.corpus, source / "nested", reuse=source)
        self.assertEqual(list(source.iterdir()), [])

    def test_corpus_mutation_during_fit_aborts_publication(self):
        def mutation(parameters, texts, vectors):
            with self.corpus.open("ab") as stream:
                stream.write(b"\n")
            return Model(len(texts))
        with self.assertRaisesRegex(ValueError, "blank corpus row"):
            self.run_job(trainer=mutation)
        self.assertFalse((self.root / "run/manifest.json").exists())

    def test_fixed_production_identity_parameters_and_cli_output_boundary(self):
        self.assertEqual(job.EXPECTED["occurrences"], 150_000)
        self.assertEqual(job.EXPECTED["unique_texts"], 66_208)
        self.assertEqual(job.EXPECTED["duplicate_excess_occurrences"], 83_792)
        self.assertEqual(job.MODEL_REVISION, "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        with mock.patch.object(job, "run_training") as runner, mock.patch("sys.stderr"):
            self.assertEqual(job.main(["--output-dir", str(self.root / "outside")]), 1)
            runner.assert_not_called()


@unittest.skipUnless(os.environ.get("ATRAG_RUN_TINY_BERTOPIC_TEST") == "1",
                     "Opt-in tiny real estimator/save/load test; no model downloads")
class TinyBERTopicIntegration(unittest.TestCase):
    def test_reference_fit_save_reload_and_confidence_on_synthetic_vectors(self):
        # 120 synthetic points, under 0.2 MiB of vectors. Production parameters
        # (including min_topic_size=50) are retained; no encoder/data downloads.
        rng = np.random.default_rng(42)
        vectors = rng.normal(0, .01, (120, 384)).astype(np.float32)
        vectors[:60, 0] += 1
        vectors[60:, 1] += 1
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        texts = ["apple orchard fruit evidence"] * 60 + ["ocean marine water evidence"] * 60
        with tempfile.TemporaryDirectory() as temporary:
            # Heavy libraries are first imported below; keep any runtime cache
            # in this synthetic fixture rather than the user's home directory.
            environment_patch = mock.patch.dict(os.environ, {
                "TRANSFORMERS_CACHE": str(Path(temporary) / "transformers-cache"),
                "HF_HUB_CACHE": str(Path(temporary) / "hub-cache"),
                "NUMBA_CACHE_DIR": str(Path(temporary) / "numba-cache")})
            environment_patch.start()
            self.addCleanup(environment_patch.stop)
            matrix_path = Path(temporary) / "embeddings.npy"
            np.save(matrix_path, vectors, allow_pickle=False)
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            model = job.full.fit_model(job.reference_parameters(), texts, matrix)
            topics, strength, counts = job.full.model_assignments(model, 120)
            self.assertEqual(topics.shape, (120,))
            self.assertEqual(strength.shape, (120,))
            self.assertEqual(sum(counts.values()), 120)
            path = Path(temporary) / "model.pkl"
            model.save(str(path), serialization="pickle", save_embedding_model=False)
            del model
            restored = job.full.load_model(path)
            job.full.model_assignments(restored, 120)
            _, confidence = restored.transform([texts[0]], embeddings=vectors[:1])
            self.assertEqual(np.asarray(confidence).shape, (1,))
            restored.calculate_probabilities = True
            _, memberships = restored.transform([texts[0]], embeddings=vectors[:1])
            self.assertEqual(np.asarray(memberships).shape[0], 1)
            self.assertTrue(np.isfinite(memberships).all())


if __name__ == "__main__":
    unittest.main()
