"""Small stand-ins only: no model weights, BERTopic execution, or Chroma DB."""

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full_workflow", ROOT / "scripts/full_2wiki_workflow.py")
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


class Interrupted(RuntimeError):
    pass


def interrupt(_):
    raise Interrupted("simulated process interruption")


class Encoder:
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        result = np.zeros((len(texts), 384), dtype="<f4")
        for row, text in enumerate(texts):
            result[row, hashlib.sha256(text.encode()).digest()[0]] = 1
        return result.tolist()


class TopicModel:
    fits = []
    transforms = []
    reductions = []
    fail_reduce = False

    def __init__(self, min_topic_size=50):
        assert min_topic_size == 50
        self.topics_ = []

    def fit_transform(self, documents):
        self.fits.append(list(documents))
        self.topics_ = [0] * len(documents)
        return self.topics_, None

    def reduce_topics(self, documents, nr_topics):
        assert nr_topics == 20
        self.reductions.append(list(documents))
        if self.fail_reduce:
            raise Interrupted("after global fit checkpoint")
        return self

    def transform(self, documents):
        self.transforms.append(list(documents))
        # Preserve per-occurrence labels even when text is repeated.
        return [i % 3 - 1 for i in range(len(documents))], None

    def save(self, path):
        Path(path).write_text(json.dumps(self.topics_))

    @classmethod
    def load(cls, path):
        result = cls()
        result.topics_ = json.loads(Path(path).read_text())
        return result


class Store:
    """Emulates Chroma's public get/upsert/count contract without a database."""

    def __init__(self):
        self.data = {}
        self.calls = []

    def factory(self, path):
        path.mkdir(exist_ok=True)
        return self

    def upsert(self, **batch):
        self.calls.append(batch)
        for key, text, meta, vector in zip(batch["ids"], batch["documents"],
                                           batch["metadatas"], batch["embeddings"]):
            self.data[key] = (text, meta.copy(), vector)

    def count(self):
        return len(self.data)

    def get(self, ids, include):
        assert include == ["documents", "metadatas"]
        # Deliberately return a different order than the request.
        keys = [key for key in reversed(ids) if key in self.data]
        return {"ids": keys, "documents": [self.data[k][0] for k in keys],
                "metadatas": [self.data[k][1] for k in keys]}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "work"
        self.root.mkdir()
        self.train = self.directory / "train.jsonl"
        self.heldout = self.directory / "dev.jsonl"
        self.questions = [
            {"question_id": "train-0", "contexts": [
                {"paragraph_text": " shared\n東京 ", "idx": 0},
                {"paragraph_text": "beta", "idx": 1, "title": "B", "is_supporting": True},
                {"paragraph_text": " shared\n東京 ", "idx": 2, "title": "Other", "is_supporting": False}]},
            {"question_id": "train-1", "contexts": [
                {"paragraph_text": "gamma", "idx": 0}, {"paragraph_text": "beta", "idx": 1}]},
            {"question_id": "train-2", "contexts": [
                {"paragraph_text": "delta", "idx": 0}, {"paragraph_text": "beta", "idx": 1}]},
        ]
        self.write_training()
        self.heldout.write_text(json.dumps({"question_id": "dev-0", "contexts": "unused", "answers": "unused"}) + "\n")
        self.counts = {"questions": 3, "occurrences": 7, "unique_texts": 4, "empty_texts": 0}
        self.environment = {"packages": {name: "test-version" for name in (
            "numpy", "torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub")}}
        self.guard = mock.patch.object(w, "memory_guard")
        self.guard.start()
        self.addCleanup(self.guard.stop)
        for calls in (TopicModel.fits, TopicModel.transforms, TopicModel.reductions):
            calls.clear()
        TopicModel.fail_reduce = False

    def write_training(self):
        self.train.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in self.questions), encoding="utf-8")

    def prepare(self, **kwargs):
        return w.prepare(self.train, self.heldout, self.root, expected=self.counts, checkpoint=2, **kwargs)

    def topics(self):
        self.prepare()
        w.train_topics(self.root, self.environment, factory=TopicModel, loader=TopicModel.load)
        w.assign_topics(self.root, self.environment, loader=TopicModel.load)

    def ready(self):
        self.topics()
        w.build_cache(self.root, self.environment, embedder=Encoder())

    def legacy(self):
        folder = self.directory / "legacy"
        folder.mkdir()
        with w.database(self.root / "corpus.sqlite") as db:
            rows = db.execute("SELECT text_id,text FROM texts ORDER BY number DESC").fetchall()
        vectors = np.asarray(Encoder().embed_documents([r[1] for r in rows]), dtype="<f4")
        np.save(folder / "embeddings.npy", vectors, allow_pickle=False)
        (folder / "document_ids.jsonl").write_text("".join(
            json.dumps({"row": i, "document_id": row[0]}) + "\n" for i, row in enumerate(rows)))
        settings = w.adapter().embedding_settings()
        config = {"model_id": settings["model_id"], "model_revision": settings["revision"],
                  "dimensions": 384, "dtype": "float32", "normalize_embeddings": True,
                  "max_seq_length": 256, "device": "cpu", "batch_size": 32,
                  "preprocessing": "paragraph_text unchanged; no title/prompt prefix",
                  "environment": {"libraries": self.environment["packages"]}}
        manifest = {"status": "complete", "processed_count": 4, "configuration": config,
                    "config_sha256": w.identity(config), "embedding_data_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
                    "artifacts": {name: {"sha256": w.digest(folder / name), "bytes": (folder / name).stat().st_size}
                                  for name in ("embeddings.npy", "document_ids.jsonl")}}
        w.atomic(folder / "manifest.json", manifest)
        return folder

    def test_prepare_preserves_occurrences_order_exact_text_metadata_and_hashes(self):
        result = self.prepare()
        self.assertEqual({k: result[k] for k in self.counts}, self.counts)
        self.assertEqual(result["duplicate_occurrences"], 3)
        self.assertEqual(result["configuration"]["source_sha256"], w.digest(self.train))
        self.assertEqual(result["heldout_questions"], 1)
        self.assertEqual(w.all_documents(self.root), [c["paragraph_text"] for q in self.questions for c in q["contexts"]])
        with w.database(self.root / "corpus.sqlite") as db:
            batch = w.rows(db, 0)
        self.assertEqual(json.loads(batch[0][3]), {"idx": 0, "title": "Unknown", "is_supporting": "N/A"})
        self.assertEqual(json.loads(batch[2][3])["title"], "Other")
        self.assertEqual(self.prepare(), result)

    def test_prepare_resumes_at_committed_question_boundary_without_duplicates(self):
        with self.assertRaises(Interrupted):
            self.prepare(progress=interrupt)
        with w.database(self.root / "corpus.sqlite") as db:
            self.assertEqual(w.state(db)["occurrences"], 3)
        self.assertEqual(self.prepare()["occurrences"], 7)
        with w.database(self.root / "corpus.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM questions").fetchone()[0], 3)

    def test_prepare_refuses_changed_source_after_interruption(self):
        with self.assertRaises(Interrupted):
            self.prepare(progress=interrupt)
        self.train.write_text(self.train.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "different source"):
            self.prepare()

    def test_heldout_overlap_blocks_completion_and_answers_never_needed(self):
        self.heldout.write_text('{"question_id":"train-1"}\n')
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.prepare()
        self.assertFalse((self.root / "preparation.json").exists())

    def test_exact_counts_fail_closed(self):
        self.counts["occurrences"] = 8
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.prepare()

    def test_duplicate_training_question_ids_fail_closed(self):
        import sqlite3
        self.questions[1]["question_id"] = "train-0"
        self.write_training()
        with self.assertRaises(sqlite3.IntegrityError):
            self.prepare()
        self.assertFalse((self.root / "preparation.json").exists())

    def test_prepared_catalog_tampering_is_detected(self):
        self.prepare()
        with (self.root / "corpus.sqlite").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "corpus changed"):
            w.prepared(self.root)

    def test_original_recipe_global_fit_reduction_and_single_transform(self):
        self.topics()
        self.assertEqual([len(d) for d in TopicModel.fits], [7])
        self.assertEqual([len(d) for d in TopicModel.reductions], [7])
        self.assertEqual([len(d) for d in TopicModel.transforms], [7])
        self.assertEqual(TopicModel.fits[0], TopicModel.transforms[0])
        manifest = w.read(self.root / "topics.json")
        self.assertEqual(manifest["configuration"]["recipe"], w.RECIPE)
        self.assertFalse(w.RECIPE["fit_uses_precomputed_embeddings"])
        w.train_topics(self.root, self.environment, factory=TopicModel)
        w.assign_topics(self.root, self.environment, loader=TopicModel.load)
        self.assertEqual(len(TopicModel.fits), 1)
        self.assertEqual(len(TopicModel.transforms), 1)

    def test_resume_reduction_uses_saved_global_fit_without_refitting(self):
        self.prepare()
        TopicModel.fail_reduce = True
        with self.assertRaises(Interrupted):
            w.train_topics(self.root, self.environment, factory=TopicModel)
        self.assertTrue((self.root / "fit.json").exists())
        self.assertFalse((self.root / "topics.json").exists())
        TopicModel.fail_reduce = False
        w.train_topics(self.root, self.environment, factory=TopicModel, loader=TopicModel.load)
        self.assertEqual(len(TopicModel.fits), 1)

    def test_custom_topic_model_recipe_is_refused(self):
        self.topics()
        manifest = w.read(self.root / "topics.json")
        manifest["configuration"]["recipe"] = {"topics": 418, "custom": True}
        w.atomic(self.root / "topics.json", manifest)
        with self.assertRaisesRegex(ValueError, "provenance"):
            w.assign_topics(self.root, self.environment, loader=TopicModel.load)

    def split_profiles(self):
        producer = {**self.environment, "profile": "training", "python": "3.12.3",
                    "recorded_training_packages": {**self.environment["packages"], "packaging": "26.3"},
                    "training_requirements_sha256": "fixture-original-freeze",
                    "packages": {**self.environment["packages"], "packaging": "26.3"}}
        consumer = {**producer, "profile": "retrieval", "packages": {
            **producer["packages"], "packaging": "24.2", "langchain": "0.3.2"}}
        self.environment = producer
        return producer, consumer

    def test_cross_environment_model_requires_reference_and_successful_transform_check(self):
        producer, consumer = self.split_profiles()
        self.topics()
        info = w.prepared(self.root)
        with self.assertRaisesRegex(ValueError, "verify-model"):
            w.topic_manifest(self.root, info, consumer)
        with self.assertRaisesRegex(ValueError, "original training environment"):
            w.verify_model(self.root, consumer, loader=TopicModel.load)
        w.verify_model(self.root, producer, loader=TopicModel.load)
        with self.assertRaisesRegex(ValueError, "verify-model"):
            w.topic_manifest(self.root, info, consumer)
        w.verify_model(self.root, consumer, loader=TopicModel.load)
        self.assertEqual(w.topic_manifest(self.root, info, consumer)["configuration"]["environment"], producer)
        w.build_cache(self.root, consumer, embedder=Encoder())
        store = Store()
        self.assertTrue(w.ingest(self.root, consumer, factory=store.factory, minimum_free_bytes=0)["complete"])
        self.assertEqual(store.count(), 7)

    def test_model_transfer_refuses_scientific_version_or_certificate_changes(self):
        producer, consumer = self.split_profiles()
        self.topics()
        w.verify_model(self.root, producer, loader=TopicModel.load)
        incompatible = {**consumer, "packages": {**consumer["packages"], "torch": "other"}}
        loader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "dependencies differ"):
            w.verify_model(self.root, incompatible, loader=loader)
        loader.assert_not_called()
        w.verify_model(self.root, consumer, loader=TopicModel.load)
        changed = {**consumer, "extra_environment_change": True}
        with self.assertRaisesRegex(ValueError, "verify-model"):
            w.topic_manifest(self.root, w.prepared(self.root), changed)

    def test_model_transfer_rejects_changed_topic_labels_or_probabilities(self):
        producer, consumer = self.split_profiles()
        self.topics()
        w.verify_model(self.root, producer, loader=TopicModel.load)

        class WrongLabels(TopicModel):
            def transform(self, documents):
                return [418] * len(documents), None

        with self.assertRaisesRegex(ValueError, "topic labels differ"):
            w.verify_model(self.root, consumer, loader=lambda _: WrongLabels())
        with self.assertRaisesRegex(ValueError, "verify-model"):
            w.topic_manifest(self.root, w.prepared(self.root), consumer)
        w.compare_probes([{"topics": [1], "probabilities": [.5]}], [{"topics": [1], "probabilities": [.5000001]}])
        with self.assertRaisesRegex(ValueError, "probabilities differ"):
            w.compare_probes([{"topics": [1], "probabilities": [.5]}], [{"topics": [1], "probabilities": [.6]}])

    def test_probe_never_uses_heldout_answers_or_changes_global_assignments(self):
        producer, _ = self.split_profiles()
        self.topics()
        before = w.digest(self.root / "topics.npy")
        w.verify_model(self.root, producer, loader=TopicModel.load)
        reference = w.read(self.root / "model_probe_reference.json")
        self.assertEqual([len(case) for case in reference["cases"]], [1, 1, 1, 1, 3])
        self.assertNotIn("unused", json.dumps(reference))
        self.assertEqual(w.digest(self.root / "topics.npy"), before)
        self.assertEqual([len(d) for d in TopicModel.transforms[:1]], [7])

    def test_approved_previous_preparation_is_reused_but_unknown_code_is_refused(self):
        result = self.prepare()
        previous = next(iter(w.PREPARATION_IMPLEMENTATIONS))
        with w.database(self.root / "corpus.sqlite", write=True) as db:
            job = w.state(db)
            job["configuration"]["implementation_sha256"] = previous
            w.save_state(db, job)
            db.commit()
        result["configuration"]["implementation_sha256"] = previous
        result["corpus_sha256"] = w.digest(self.root / "corpus.sqlite")
        w.atomic(self.root / "preparation.json", result)
        self.assertEqual(self.prepare(), result)
        self.assertEqual(w.digest(self.root / "corpus.sqlite"), result["corpus_sha256"])
        result["configuration"]["implementation_sha256"] = "unapproved"
        w.atomic(self.root / "preparation.json", result)
        with self.assertRaisesRegex(ValueError, "another protocol"):
            w.prepared(self.root)

    def test_cli_selects_training_profile_for_fit_and_assign_without_retrieval_checker(self):
        for stage in ("train-topics", "assign-topics"):
            target = "train_topics" if stage == "train-topics" else "assign_topics"
            with mock.patch.object(w, "runtime", return_value=self.environment) as runtime:
                with mock.patch.object(w, target, return_value={}), mock.patch("builtins.print"):
                    code = w.main([stage, "--workdir", str(self.root), "--training-requirements", "original-62.txt"])
            self.assertEqual(code, 0)
            self.assertEqual(runtime.call_args.args[1], "training")

    def test_training_runtime_uses_new_profile_checker_and_never_combined_checker(self):
        old = w.module("_source_guards", ROOT / "scripts/ingest_original_2wiki_minilm.py")
        old.inspect_environment = mock.Mock(side_effect=AssertionError("Combined checker must not run"))
        checker = mock.Mock()
        checker.inspect_environment.return_value = {"profile": "training", "packages": {"packaging": "26.3"}}
        with mock.patch.object(w, "module", return_value=old), mock.patch.object(w, "environments", return_value=checker):
            result = w.runtime("original-62.txt", "training")
        checker.inspect_environment.assert_called_once_with("original-62.txt", "training")
        old.inspect_environment.assert_not_called()
        self.assertEqual(result["profile"], "training")

    def test_fit_and_assign_explicitly_refuse_retrieval_profile(self):
        for method in (w.train_topics, w.assign_topics):
            with self.assertRaisesRegex(ValueError, "training-only"):
                method(self.root, {"profile": "retrieval"})

    def test_memory_guard_runs_before_model_creation(self):
        self.prepare()
        factory = mock.Mock()
        with mock.patch.object(w, "memory_guard", side_effect=ValueError("memory")):
            with self.assertRaisesRegex(ValueError, "memory"):
                w.train_topics(self.root, self.environment, factory=factory)
        factory.assert_not_called()

    def test_cache_embeds_unique_texts_only_and_resumes_transactions(self):
        self.prepare()
        encoder = Encoder()
        with mock.patch.object(w, "BATCH", 2):
            with self.assertRaises(Interrupted):
                w.build_cache(self.root, self.environment, embedder=encoder, progress=interrupt)
            self.assertEqual(len(encoder.texts), 2)
            result = w.build_cache(self.root, self.environment, embedder=encoder)
        self.assertEqual(len(encoder.texts), 4)
        self.assertEqual(result["vectors"], 4)
        self.assertEqual(len(set(encoder.texts)), 4)
        w.build_cache(self.root, self.environment, embedder=encoder)
        self.assertEqual(len(encoder.texts), 4)

    def test_bad_embedding_dimension_or_norm_never_commits_vectors(self):
        self.prepare()
        encoder = mock.Mock()
        encoder.embed_documents.return_value = [[0.] * 384] * 4
        with self.assertRaisesRegex(ValueError, "normalized"):
            w.build_cache(self.root, self.environment, embedder=encoder)
        with w.database(self.root / "vectors.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)

    def test_compatible_legacy_embeddings_are_remapped_by_exact_text_hash(self):
        self.prepare()
        folder = self.legacy()
        encoder = mock.Mock()
        result = w.build_cache(self.root, self.environment, reuse=folder, embedder=encoder)
        encoder.embed_documents.assert_not_called()
        self.assertEqual(result["vectors"], 4)
        with w.database(self.root / "vectors.sqlite") as db:
            for text in ["beta", "delta"]:
                value = db.execute("SELECT value FROM vectors WHERE text_id=?", (hashlib.sha256(text.encode()).hexdigest(),)).fetchone()[0]
                self.assertEqual(np.frombuffer(value, dtype="<f4").tolist(), Encoder().embed_documents([text])[0])

    def test_legacy_device_batch_revision_runtime_or_hash_mismatch_refused(self):
        info = self.prepare()
        folder = self.legacy()
        original = w.read(folder / "manifest.json")
        for key, value in (("batch_size", 64), ("device", "cuda"), ("model_revision", "other")):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(original))
                changed["configuration"][key] = value
                changed["config_sha256"] = w.identity(changed["configuration"])
                w.atomic(folder / "manifest.json", changed)
                with self.assertRaisesRegex(ValueError, "configuration"):
                    w.legacy_embeddings(folder, info, self.environment, w.adapter().embedding_settings())
        w.atomic(folder / "manifest.json", original)
        other_environment = {"packages": {**self.environment["packages"], "torch": "other"}}
        with self.assertRaisesRegex(ValueError, "runtime"):
            w.legacy_embeddings(folder, info, other_environment, w.adapter().embedding_settings())
        with (folder / "embeddings.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            w.legacy_embeddings(folder, info, self.environment, w.adapter().embedding_settings())

    def test_inconsistent_legacy_id_mapping_never_completes_cache(self):
        self.prepare()
        folder = self.legacy()
        mapping = folder / "document_ids.jsonl"
        lines = mapping.read_text().splitlines()
        lines[1] = json.dumps({"row": 1, "document_id": "not-in-training"})
        mapping.write_text("\n".join(lines) + "\n")
        manifest = w.read(folder / "manifest.json")
        manifest["artifacts"][mapping.name] = {"sha256": w.digest(mapping), "bytes": mapping.stat().st_size}
        w.atomic(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "absent"):
            w.build_cache(self.root, self.environment, reuse=folder)
        self.assertFalse((self.root / "cache.json").exists())

    def test_upsert_crash_replays_same_ids_and_preserves_all_duplicate_occurrences(self):
        self.ready()
        store = Store()
        with mock.patch.object(w, "BATCH", 2):
            with self.assertRaises(Interrupted):
                w.ingest(self.root, self.environment, factory=store.factory, progress=interrupt, minimum_free_bytes=0)
            self.assertEqual(w.read(self.root / "ingestion.json")["next"], 0)
            self.assertEqual(store.count(), 2)
            first_ids = store.calls[0]["ids"]
            result = w.ingest(self.root, self.environment, factory=store.factory, minimum_free_bytes=0)
            self.assertEqual(store.calls[1]["ids"], first_ids)
            self.assertEqual([len(call["ids"]) for call in store.calls], [2, 2, 2, 2, 1])
            self.assertEqual(result["next"], 7)
            self.assertEqual(store.count(), 7)

    def test_restart_preserves_original_metadata_labels_vectors_and_batch_1000(self):
        self.ready()
        store = Store()
        with self.assertRaises(Interrupted):
            w.ingest(self.root, self.environment, factory=store.factory, progress=interrupt, minimum_free_bytes=0)
        first_ids = store.calls[0]["ids"]
        result = w.ingest(self.root, self.environment, factory=store.factory, minimum_free_bytes=0)
        self.assertEqual(store.calls[1]["ids"], first_ids)
        self.assertEqual(store.count(), 7)
        self.assertEqual(result["next"], 7)
        self.assertTrue(result["complete"])
        self.assertEqual(result["configuration"]["batch_size"], 1000)
        self.assertEqual(len(set(first_ids)), 7)
        beta = [row for row in store.data.values() if row[0] == "beta"]
        self.assertEqual(len(beta), 3)
        self.assertTrue(all(row[2] == beta[0][2] for row in beta))
        source_sha = w.digest(self.train)
        for number, context in enumerate(c for q in self.questions for c in q["contexts"]):
            text, meta, _ = store.data[w.occurrence_id(source_sha, number)]
            self.assertEqual(text, context["paragraph_text"])
            self.assertEqual(meta, {"idx": context["idx"], "title": context.get("title", "Unknown"),
                                    "is_supporting": context.get("is_supporting", "N/A"),
                                    "bertopic": f"Topic {number % 3 - 1}"})
        verified = w.verify_completed(self.root, self.environment, factory=store.factory)
        self.assertEqual(verified["occurrences"], 7)
        calls = len(store.calls)
        w.ingest(self.root, self.environment, factory=store.factory, minimum_free_bytes=0)
        self.assertEqual(len(store.calls), calls)

    def test_production_batch_boundary_and_restart_after_committed_batch(self):
        self.questions[-1]["contexts"].extend(
            {"paragraph_text": "beta", "idx": number} for number in range(996))
        self.counts["occurrences"] = 1003
        self.write_training()
        self.ready()
        store = Store()

        def stop_on_second_batch(status):
            if status["next"] == 1000:
                interrupt(status)

        with self.assertRaises(Interrupted):
            w.ingest(self.root, self.environment, factory=store.factory, progress=stop_on_second_batch, minimum_free_bytes=0)
        self.assertEqual(w.read(self.root / "ingestion.json")["next"], 1000)
        self.assertEqual(store.count(), 1003)
        result = w.ingest(self.root, self.environment, factory=store.factory, minimum_free_bytes=0)
        self.assertEqual([len(call["ids"]) for call in store.calls], [1000, 3, 3])
        self.assertEqual(result["next"], 1003)
        self.assertEqual(store.count(), 1003)

    def test_native_client_uses_original_default_collection_without_embedding_fallback(self):
        import types
        client = mock.Mock()
        package = types.SimpleNamespace(PersistentClient=mock.Mock(return_value=client))
        with mock.patch.dict(sys.modules, {"chromadb": package}):
            w.chroma_collection(self.root / "chroma")
            package.PersistentClient.assert_called_once_with(path=str(self.root / "chroma"))
            client.get_or_create_collection.assert_called_once_with(name="langchain", embedding_function=None)
            w.chroma_collection(self.root / "chroma", create=False)
            client.get_collection.assert_called_once_with(name="langchain", embedding_function=None)

    def test_verification_detects_changed_metadata_and_missing_records(self):
        self.ready()
        store = Store()
        w.ingest(self.root, self.environment, factory=store.factory, minimum_free_bytes=0)
        key = next(iter(store.data))
        store.data[key][1]["bertopic"] = "Topic 418"
        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            w.verify_completed(self.root, self.environment, factory=store.factory)
        del store.data[key]
        with self.assertRaisesRegex(ValueError, "count differs"):
            w.verify_completed(self.root, self.environment, factory=store.factory)

    def test_existing_untracked_store_or_incompatible_runtime_is_refused(self):
        self.ready()
        (self.root / "chroma").mkdir()
        with self.assertRaisesRegex(ValueError, "no matching"):
            w.ingest(self.root, self.environment, factory=mock.Mock(), minimum_free_bytes=0)
        with self.assertRaisesRegex(ValueError, "provenance"):
            w.topic_manifest(self.root, w.prepared(self.root), {"packages": {}})

    def test_verify_never_creates_missing_store(self):
        w.atomic(self.root / "ingestion.json", {"complete": True})
        factory = mock.Mock()
        with self.assertRaisesRegex(ValueError, "never creates"):
            w.verify_completed(self.root, self.environment, factory=factory)
        factory.assert_not_called()

    def test_lock_prevents_two_writers_and_disk_estimate_decreases_on_resume(self):
        with w.lock(self.root):
            with self.assertRaisesRegex(ValueError, "Another process"):
                with w.lock(self.root):
                    self.fail("second writer acquired lock")
        info = self.prepare()
        self.assertGreater(w.remaining_disk_estimate(info, 0), w.remaining_disk_estimate(info, 5))
        self.assertEqual(w.remaining_disk_estimate(info, 7), 0)

    def test_import_and_status_do_not_load_models_or_chroma(self):
        code = ("import runpy,sys; runpy.run_path(sys.argv[1],run_name='import_only'); "
                "assert not {'numpy','bertopic','torch','sentence_transformers','chromadb'} & set(sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "scripts/full_2wiki_workflow.py")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_sampled_input_before_loading_model_libraries(self):
        with redirect_stderr(io.StringIO()) as errors:
            code = w.main(["prepare", "--workdir", str(self.root), "--input", "test_subsampled.jsonl", "--heldout", str(self.heldout)])
        self.assertEqual(code, 1)
        self.assertIn("full train.jsonl", errors.getvalue())

    def test_original_sources_still_match_merged_compatibility_guards(self):
        original = w.module("_original_guard_test", ROOT / "scripts/ingest_original_2wiki_minilm.py")
        for relative, expected in original.SOURCE_SHA256.items():
            self.assertEqual(w.digest(ROOT / relative), expected, relative)


if __name__ == "__main__":
    unittest.main()
