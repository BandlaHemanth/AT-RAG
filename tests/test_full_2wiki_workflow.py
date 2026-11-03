"""Small stand-ins only: no model weights, BERTopic execution, or Chroma DB."""

from contextlib import redirect_stderr, redirect_stdout
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

    def test_unique_mode_fits_and_reduces_every_exact_paragraph_without_sampling(self):
        info = self.prepare()
        before = w.digest(self.root / "corpus.sqlite")
        events = []
        factory = mock.Mock(side_effect=TopicModel)
        result = w.train_topics(self.root, self.environment, factory=factory,
                                training_mode="unique-paragraphs", progress=events.append)
        factory.assert_called_once_with(min_topic_size=50)
        expected = [" shared\n東京 ", "beta", "gamma", "delta"]
        self.assertEqual(TopicModel.fits, [expected])
        self.assertEqual(TopicModel.reductions, [expected])
        self.assertEqual(result["configuration"]["training_documents"], info["unique_texts"])
        self.assertEqual(result["configuration"]["recipe"], {
            **w.RECIPE, "fit_uses_all_occurrences": False, "fit_uses_all_unique_paragraphs": True})
        self.assertEqual(result["configuration_sha256"], w.identity(result["configuration"]))
        self.assertEqual([e["stage"] for e in events], [
            "resource-preflight", "load-training-documents", "fit-start", "fit-checkpoint-saved",
            "reduction-start", "topics-checkpoint-saved"])
        self.assertEqual(w.digest(self.root / "corpus.sqlite"), before)

    def test_unique_training_assigns_every_occurrence_in_original_order_and_preserves_metadata(self):
        self.prepare()
        w.train_topics(self.root, self.environment, factory=TopicModel, training_mode="unique-paragraphs")
        events = []
        assignments = w.assign_topics(self.root, self.environment, loader=TopicModel.load,
                                      training_mode="unique-paragraphs", progress=events.append)
        # Position-sensitive fake: deduplicating/batching this transform would change the result.
        expected = [c["paragraph_text"] for q in self.questions for c in q["contexts"]]
        self.assertEqual(TopicModel.transforms, [expected])
        self.assertEqual(assignments["occurrences"], 7)
        self.assertEqual(assignments["method"], w.ASSIGNMENT_METHOD)
        self.assertEqual(events[-1]["stage"], "assignments-checkpoint-saved")
        w.build_cache(self.root, self.environment, embedder=Encoder())
        store = Store()
        w.ingest(self.root, self.environment, factory=store.factory, minimum_free_bytes=0,
                 training_mode="unique-paragraphs")
        self.assertEqual(store.count(), 7)
        self.assertEqual(sum(row[0] == "beta" for row in store.data.values()), 3)
        with w.database(self.root / "corpus.sqlite") as db:
            labels = np.load(self.root / "topics.npy", allow_pickle=False)
            for record in w.records(w.rows(db, 0), w.digest(self.train), labels):
                self.assertEqual(store.data[record["id"]][:2], (record["document"], record["metadata"]))
        self.assertEqual(w.verify_completed(self.root, self.environment, factory=store.factory)["occurrences"], 7)

    def test_separate_mode_fit_checkpoints_resume_without_cross_reuse(self):
        self.prepare()
        TopicModel.fail_reduce = True
        for mode in w.TRAINING_MODES:
            with self.assertRaises(Interrupted):
                w.train_topics(self.root, self.environment, factory=TopicModel, training_mode=mode)
        occurrence_fit = w.read(self.root / "fit.json")
        unique_fit = w.read(self.root / "fit.unique-paragraphs.json")
        self.assertNotEqual(occurrence_fit["configuration_sha256"], unique_fit["configuration_sha256"])
        self.assertEqual([len(d) for d in TopicModel.fits], [7, 4])
        TopicModel.fail_reduce = False
        w.train_topics(self.root, self.environment, factory=TopicModel, loader=TopicModel.load,
                       training_mode="unique-paragraphs")
        self.assertEqual([len(d) for d in TopicModel.fits], [7, 4])
        with self.assertRaisesRegex(ValueError, "training mode"):
            w.train_topics(self.root, self.environment, training_mode="occurrences")

    def test_incompatible_or_renamed_fit_is_rejected_before_model_loading(self):
        self.prepare()
        TopicModel.fail_reduce = True
        with self.assertRaises(Interrupted):
            w.train_topics(self.root, self.environment, factory=TopicModel)
        (self.root / "fit.unique-paragraphs.json").write_bytes((self.root / "fit.json").read_bytes())
        (self.root / "fit_model.unique-paragraphs").write_bytes((self.root / "fit_model").read_bytes())
        loader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "training mode"):
            w.train_topics(self.root, self.environment, loader=loader, training_mode="unique-paragraphs")
        loader.assert_not_called()
        # Legacy occurrence checkpoint without a mode/fingerprint must also stop.
        checkpoint = w.read(self.root / "fit.json")
        del checkpoint["configuration"]["training_mode"]
        w.atomic(self.root / "fit.json", checkpoint)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            w.train_topics(self.root, self.environment, loader=loader)

    def test_assignment_mode_and_fingerprint_mismatch_are_refused(self):
        self.prepare()
        w.train_topics(self.root, self.environment, factory=TopicModel, training_mode="unique-paragraphs")
        loader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "training mode"):
            w.assign_topics(self.root, self.environment, loader=loader, training_mode="occurrences")
        loader.assert_not_called()
        w.assign_topics(self.root, self.environment, loader=TopicModel.load)
        model = w.read(self.root / "topics.json")
        manifest = w.read(self.root / "assignments.json")
        for key, bad in (("topic_configuration_sha256", "other-mode"), ("occurrences", 4), ("method", "unique-transform")):
            w.atomic(self.root / "assignments.json", {**manifest, key: bad})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                w.assignment_manifest(self.root, w.prepared(self.root), model)

    def test_changed_topic_configuration_fingerprint_is_refused(self):
        self.topics()
        model = w.read(self.root / "topics.json")
        model["configuration_sha256"] = "invalid"
        w.atomic(self.root / "topics.json", model)
        with self.assertRaisesRegex(ValueError, "provenance"):
            w.topic_manifest(self.root, w.prepared(self.root), self.environment)

    def test_disk_preflight_stops_before_documents_or_model_and_does_not_fallback(self):
        self.prepare()
        with mock.patch.object(w.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            for mode in w.TRAINING_MODES:
                with mock.patch.object(w, "all_documents") as documents, mock.patch.object(w, "model_class") as factory:
                    with self.assertRaisesRegex(ValueError, "No sampling or parameter fallback"):
                        w.train_topics(self.root, self.environment, training_mode=mode)
                    documents.assert_not_called()
                    factory.assert_not_called()
        self.assertFalse((self.root / "topics.json").exists())

    def test_preflight_distinguishes_modes_and_includes_full_assignment_coverage(self):
        info = self.prepare()
        unique = w.topic_preflight(self.root, info, "unique-paragraphs")
        occurrences = w.topic_preflight(self.root, info, "occurrences")
        assignment = w.topic_preflight(self.root, info, "unique-paragraphs", stage="assign-topics")
        self.assertEqual((unique["documents"], occurrences["documents"], assignment["documents"]), (4, 7, 7))
        self.assertLess(unique["estimated_free_disk_bytes"], occurrences["estimated_free_disk_bytes"])
        self.assertFalse(unique["capacity_guaranteed"])

    def test_corpus_coverage_refuses_missing_or_orphaned_occurrences_and_numbering_gaps(self):
        info = self.prepare()
        with w.database(self.root / "corpus.sqlite", write=True) as db:
            db.execute("UPDATE occurrences SET text_id='absent' WHERE number=6")
            db.commit()
        with self.assertRaisesRegex(ValueError, "no paragraph mapping"):
            w.corpus_coverage(self.root, info)
        with w.database(self.root / "corpus.sqlite", write=True) as db:
            db.execute("UPDATE occurrences SET number=7 WHERE number=6")
            db.commit()
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            w.corpus_coverage(self.root, info)
        with w.database(self.root / "corpus.sqlite", write=True) as db:
            db.execute("DELETE FROM occurrences WHERE number=7")
            db.commit()
        with self.assertRaisesRegex(ValueError, "count differs"):
            w.corpus_coverage(self.root, info)

    def test_invalid_or_short_transform_never_publishes_assignments(self):
        self.prepare()
        w.train_topics(self.root, self.environment, factory=TopicModel, training_mode="unique-paragraphs")
        for labels in ([0] * 4, [0.] * 7, [-2] * 7):
            fake = mock.Mock()
            fake.transform.return_value = labels, None
            with self.assertRaisesRegex(ValueError, "occurrence-level"):
                w.assign_topics(self.root, self.environment, loader=lambda _: fake)
            self.assertFalse((self.root / "assignments.json").exists())

    def test_full_size_occurrence_assignment_mapping_covers_last_record_and_batch_remainder(self):
        # Full occurrence count with synthetic arrays/rows; no dataset preparation,
        # BERTopic training, model weights, or Chroma creation.
        count = w.COUNTS["occurrences"]
        labels = np.arange(count, dtype="<i4") % 21 - 1
        source_sha = "synthetic-full-size"
        observed = 0
        for start in range(0, count, w.BATCH):
            batch = [(n, f"text-{n % 4}", f"paragraph-{n % 4}", json.dumps({"idx": n}))
                     for n in range(start, min(start + w.BATCH, count))]
            mapped = w.records(batch, source_sha, labels)
            self.assertEqual(len(mapped), len(batch))
            for number, record in zip(range(start, start + len(batch)), mapped):
                self.assertEqual(record["metadata"], {"idx": number, "bertopic": f"Topic {int(labels[number])}"})
            observed += len(mapped)
        self.assertEqual(observed, 1674540)
        self.assertEqual(mapped[-1]["id"], w.occurrence_id(source_sha, 1674539))
        self.assertEqual(len(mapped), 540)

    def test_unique_cli_passes_explicit_mode_and_preserves_training_profile(self):
        with mock.patch.object(w, "runtime", return_value=self.environment) as runtime:
            with mock.patch.object(w, "train_topics", return_value={}) as train, mock.patch("builtins.print"):
                code = w.main(["train-topics", "--training-mode", "unique-paragraphs", "--workdir", str(self.root),
                               "--training-requirements", "original-62.txt"])
        self.assertEqual(code, 0)
        self.assertEqual(runtime.call_args.args[1], "training")
        self.assertEqual(train.call_args.kwargs["training_mode"], "unique-paragraphs")

    def test_both_approved_backup_producers_train_unique_without_repreparation_or_db_changes(self):
        actual_digest = w.digest
        for producer in sorted(w.PREPARATION_IMPLEMENTATIONS):
            with self.subTest(producer=producer):
                self.root = self.directory / f"backup-{producer[:8]}"
                self.root.mkdir()

                def producer_digest(path):
                    return producer if Path(path).resolve() == Path(w.__file__).resolve() else actual_digest(path)

                with mock.patch.object(w, "digest", side_effect=producer_digest):
                    backup = self.prepare()
                before = actual_digest(self.root / "corpus.sqlite")
                with mock.patch.object(w, "prepare", side_effect=AssertionError("Do not repeat preparation")):
                    self.assertEqual(w.prepared(self.root), backup)
                    w.train_topics(self.root, self.environment, factory=TopicModel, training_mode="unique-paragraphs")
                self.assertEqual(actual_digest(self.root / "corpus.sqlite"), before)
                self.assertEqual(w.read(self.root / "preparation.json"), backup)

    def test_failed_unique_fit_does_not_create_success_checkpoint(self):
        self.prepare()
        broken = mock.Mock()
        broken.fit_transform.side_effect = Interrupted("during global fit")
        with self.assertRaises(Interrupted):
            w.train_topics(self.root, self.environment, factory=lambda **_: broken, training_mode="unique-paragraphs")
        broken.save.assert_not_called()
        self.assertFalse((self.root / "fit.unique-paragraphs.json").exists())
        self.assertFalse((self.root / "topics.json").exists())
        w.train_topics(self.root, self.environment, factory=TopicModel, training_mode="unique-paragraphs")
        self.assertEqual([len(d) for d in TopicModel.fits], [4])

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


class MemoryGuardTests(unittest.TestCase):
    """Synthetic /proc and cgroup-v2 counters; no real resource assumptions."""

    GIB = 1024 ** 3

    def snapshot(self, host_gib=30, limit_gib=30, unused_gib=16, inactive_gib=12):
        return {"/proc/meminfo": f"MemAvailable: {int(host_gib * self.GIB) // 1024} kB\n",
                "/sys/fs/cgroup/memory.max": str(int(limit_gib * self.GIB)),
                "/sys/fs/cgroup/memory.current": str(int((limit_gib - unused_gib) * self.GIB)),
                "/sys/fs/cgroup/memory.stat": f"inactive_file {int(inactive_gib * self.GIB)}\n"}

    def measure(self, files, minimum_gib=24):
        def read_text(path, *args, **kwargs):
            if str(path) not in files:
                raise FileNotFoundError(str(path))
            return files[str(path)]

        output = io.StringIO()
        with mock.patch.object(Path, "read_text", autospec=True, side_effect=read_text), redirect_stdout(output):
            try:
                return w.memory_guard(minimum_gib)
            finally:
                self.details = json.loads(output.getvalue())

    def test_reported_kaggle_resources_pass_with_two_gib_reserve_and_complete_diagnostics(self):
        files = self.snapshot(host_gib=29.69, unused_gib=15.98, inactive_gib=12.19)
        available = self.measure(files)
        self.assertAlmostEqual(available / self.GIB, 26.17, places=6)
        details = self.details
        self.assertEqual(details["minimum_available_memory_gib"], 24)
        self.assertAlmostEqual(details["raw_unused_bytes"] / self.GIB, 15.98, places=6)
        self.assertAlmostEqual(details["inactive_file_bytes"] / self.GIB, 12.19, places=6)
        self.assertAlmostEqual(details["estimated_reclaimable_bytes"] / self.GIB, 10.19, places=6)
        self.assertEqual(details["safety_margin_bytes"], 2 * self.GIB)
        self.assertEqual(details["effective_available_bytes"], available)
        self.assertEqual(details["container_limit_bytes"], 30 * self.GIB)
        self.assertEqual(details["reclaim_estimate_status"], "inactive-file-less-reserve")

    def test_other_file_active_anon_and_slab_memory_get_no_reclaim_credit(self):
        files = self.snapshot(unused_gib=16, inactive_gib=0)
        files["/sys/fs/cgroup/memory.stat"] += "file 14000000000\nactive_file 14000000000\nanon 1000000000\nslab_reclaimable 1000000000\n"
        with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
            self.measure(files)
        self.assertEqual(self.details["estimated_reclaimable_bytes"], 0)
        self.assertEqual(self.details["effective_available_bytes"], 16 * self.GIB)

    def test_cache_below_reserve_cannot_reduce_unused_or_give_negative_credit(self):
        for inactive in (0, 1, 2):
            with self.subTest(inactive=inactive):
                self.assertEqual(self.measure(self.snapshot(unused_gib=25, inactive_gib=inactive)), 25 * self.GIB)
                self.assertEqual(self.details["estimated_reclaimable_bytes"], 0)

    def test_exact_24_gib_boundary_passes_but_one_byte_less_fails(self):
        files = self.snapshot(unused_gib=14, inactive_gib=12)
        self.assertEqual(self.measure(files), 24 * self.GIB)
        files["/sys/fs/cgroup/memory.current"] = str(16 * self.GIB + 1)
        with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
            self.measure(files)
        self.assertEqual(self.details["effective_available_bytes"], 24 * self.GIB - 1)

    def test_estimate_is_capped_by_host_available_and_container_limit(self):
        self.assertEqual(self.measure(self.snapshot(host_gib=25, unused_gib=16, inactive_gib=14)), 25 * self.GIB)
        with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
            self.measure(self.snapshot(host_gib=100, limit_gib=23, unused_gib=23, inactive_gib=0))
        self.assertEqual(self.details["effective_available_bytes"], 23 * self.GIB)

    def test_zero_host_available_is_never_replaced_with_container_headroom(self):
        with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
            self.measure(self.snapshot(host_gib=0))
        self.assertEqual(self.details["effective_available_bytes"], 0)

    def test_missing_or_malformed_host_available_fails_closed(self):
        for content in (None, "MemFree: 999999999 kB\n", "MemAvailable: -1 kB\n",
                        "MemAvailable: nan kB\n", "MemAvailable: 999999999 MB\n",
                        "MemAvailable: 999999999\n", "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"):
            with self.subTest(content=content):
                files = self.snapshot()
                if content is None:
                    del files["/proc/meminfo"]
                else:
                    files["/proc/meminfo"] = content
                with self.assertRaisesRegex(ValueError, "Cannot establish safe available memory"):
                    self.measure(files)
                self.assertIsNone(self.details["effective_available_bytes"])

    def test_missing_or_malformed_limit_or_current_never_uses_host_as_fallback(self):
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"):
            for content in (None, "", "-1", "NaN", "inf", "3.0", "1_000", "１２３", "MAX"):
                with self.subTest(path=path, content=content):
                    files = self.snapshot(unused_gib=26, inactive_gib=0)
                    if content is None:
                        del files[path]
                    else:
                        files[path] = content
                    with self.assertRaisesRegex(ValueError, "Cannot establish safe available memory"):
                        self.measure(files)
                    self.assertIsNone(self.details["effective_available_bytes"])

    def test_missing_malformed_or_inconsistent_cache_gets_zero_credit(self):
        invalid = (None, "", "file 99999999999\n", "inactive_file\n", "inactive_file -1\n",
                   "inactive_file nan\n", "inactive_file 1 2\n", "inactive_file 1\ninactive_file 2\n",
                   f"inactive_file {15 * self.GIB}\n")  # current is only 14 GiB
        for content in invalid:
            with self.subTest(content=content):
                files = self.snapshot()
                if content is None:
                    del files["/sys/fs/cgroup/memory.stat"]
                else:
                    files["/sys/fs/cgroup/memory.stat"] = content
                with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
                    self.measure(files)
                self.assertEqual(self.details["estimated_reclaimable_bytes"], 0)
                self.assertEqual(self.details["effective_available_bytes"], 16 * self.GIB)
                self.assertEqual(self.details["reclaim_estimate_status"], "unavailable-no-credit")

    def test_missing_cache_can_pass_only_when_verified_unused_memory_alone_is_sufficient(self):
        files = self.snapshot(unused_gib=25, inactive_gib=0)
        del files["/sys/fs/cgroup/memory.stat"]
        self.assertEqual(self.measure(files), 25 * self.GIB)
        self.assertIsNone(self.details["inactive_file_bytes"])
        self.assertEqual(self.details["estimated_reclaimable_bytes"], 0)

    def test_over_limit_usage_is_subtracted_before_reclaim_credit(self):
        files = self.snapshot(unused_gib=-2, inactive_gib=6)
        with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
            self.measure(files)
        self.assertEqual(self.details["raw_unused_bytes"], 0)
        self.assertEqual(self.details["estimated_reclaimable_bytes"], 4 * self.GIB)
        self.assertEqual(self.details["effective_available_bytes"], 2 * self.GIB)

    def test_explicit_unlimited_container_uses_host_once_without_adding_cache(self):
        files = self.snapshot(host_gib=25)
        files["/sys/fs/cgroup/memory.max"] = "max\n"
        del files["/sys/fs/cgroup/memory.current"]
        del files["/sys/fs/cgroup/memory.stat"]
        self.assertEqual(self.measure(files), 25 * self.GIB)
        self.assertEqual(self.details["reclaim_estimate_status"], "unlimited-host-only")
        self.assertIsNone(self.details["container_limit_bytes"])
        self.assertIsNone(self.details["raw_unused_bytes"])
        self.assertEqual(self.details["estimated_reclaimable_bytes"], 0)
        files["/proc/meminfo"] = f"MemAvailable: {23 * self.GIB // 1024} kB\n"
        with self.assertRaisesRegex(ValueError, "at least 24 GiB"):
            self.measure(files)


if __name__ == "__main__":
    unittest.main()
