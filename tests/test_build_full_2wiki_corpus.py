"""Synthetic correctness and publication-safety checks; no dataset/model needed."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_full_2wiki_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_full_2wiki_corpus", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "train.jsonl"
        self.output = self.root / "corpus"
        self.rows = [
            {"question_id": "q1", "contexts": [
                {"idx": 8, "title": "First title", "paragraph_text": "Café 東京"},
                {"idx": 2, "title": "Repeat", "paragraph_text": "Café 東京"},
                {"paragraph_text": ""}, {"paragraph_text": " \t\n"},
                {"paragraph_text": None}, {},
                {"idx": 9, "paragraph_text": " Café 東京 "},
            ]},
            {"question_id": "q2", "contexts": [
                {"idx": 0, "title": "Later title", "paragraph_text": "Café 東京"},
                {"idx": 1, "title": "Third", "paragraph_text": "alpha"},
                {"paragraph_text": "Cafe\u0301 東京"},
            ]},
        ]
        self.write_rows(self.rows)

    def write_rows(self, rows):
        # A blank physical line verifies source-line provenance independently
        # of question count. The last record has no trailing newline.
        self.source.write_bytes((json.dumps(rows[0], ensure_ascii=False) + "\n\n" +
                                 "\n".join(json.dumps(row, ensure_ascii=False)
                                           for row in rows[1:])).encode("utf-8"))

    def documents(self, output=None):
        return [json.loads(line) for line in (output or self.output).joinpath(
            "documents.jsonl").read_text(encoding="utf-8").splitlines()]

    def assert_no_publication(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["train.jsonl"])

    def test_exact_deduplication_order_metadata_and_sha256(self):
        summary = builder.build_corpus(self.source, self.output)
        documents = self.documents()
        self.assertEqual([doc["paragraph_text"] for doc in documents],
                         ["Café 東京", " Café 東京 ", "alpha", "Cafe\u0301 東京"])
        self.assertEqual(documents[0], {
            "document_id": hashlib.sha256("Café 東京".encode("utf-8")).hexdigest(),
            "paragraph_text": "Café 東京", "question_id": "q1", "source_line": 1,
            "context_position": 0, "idx": 8, "title": "First title",
        })
        self.assertEqual(documents[2]["source_line"], 3)
        self.assertEqual(documents[2]["context_position"], 1)
        self.assertEqual(documents[2]["question_id"], "q2")
        self.assertIsNone(documents[1]["title"])
        for document in documents:
            self.assertEqual(document["document_id"], hashlib.sha256(
                document["paragraph_text"].encode("utf-8")).hexdigest())
        self.assertEqual(builder.document_id("abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual({key: summary[key] for key in builder.FULL_DATASET_COUNTS}, {
            "questions": 2, "context_occurrences": 10, "unique_paragraphs": 4,
            "duplicate_occurrences": 2, "empty_contexts": 4,
        })
        self.assertEqual(summary["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(summary["source_bytes"], self.source.stat().st_size)
        self.assertEqual(summary["documents_sha256"], hashlib.sha256(
            (self.output / "documents.jsonl").read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.output / "summary.json").read_bytes()), summary)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         ["documents.jsonl", "summary.json"])

    def test_byte_determinism(self):
        builder.build_corpus(self.source, self.output)
        second = self.root / "second"
        builder.build_corpus(self.source, second)
        for name in ("documents.jsonl", "summary.json"):
            self.assertEqual((self.output / name).read_bytes(), (second / name).read_bytes())

    def test_gold_annotations_never_affect_documents(self):
        builder.build_corpus(self.source, self.output)
        annotated = copy.deepcopy(self.rows)
        for row in annotated:
            row.update({"answers_objects": [{"spans": ["secret answer"]}],
                        "reasoning_steps": ["secret reasoning"], "supporting_facts": ["secret label"]})
            for context in row["contexts"]:
                context["is_supporting"] = False
        self.write_rows(annotated)
        second = self.root / "annotated"
        builder.build_corpus(self.source, second)
        self.assertEqual((self.output / "documents.jsonl").read_bytes(),
                         (second / "documents.jsonl").read_bytes())

    def test_matching_counts_are_checked_after_processing_to_eof(self):
        expected = {"questions": 2, "context_occurrences": 10, "unique_paragraphs": 4,
                    "duplicate_occurrences": 2, "empty_contexts": 4}
        summary = builder.build_corpus(self.source, self.output, expected_counts=expected)
        self.assertEqual(summary["count_validation"], {"status": "passed", "expected": expected})

    def test_count_mismatch_publishes_nothing(self):
        with self.assertRaisesRegex(ValueError, '"actual": 2, "expected": 1'):
            builder.build_corpus(self.source, self.output, expected_counts={"questions": 1})
        self.assert_no_publication()

    def test_completed_and_empty_destinations_are_never_overwritten(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                target = self.root / str(completed)
                target.mkdir()
                if completed:
                    (target / "summary.json").write_bytes(b"preserve me")
                before = {path.name: path.read_bytes() for path in target.iterdir()}
                with self.assertRaises(FileExistsError):
                    builder.build_corpus(self.source, target)
                self.assertEqual(before, {path.name: path.read_bytes() for path in target.iterdir()})

    def test_symlink_destination_is_refused(self):
        self.output.symlink_to(self.root / "missing", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            builder.build_corpus(self.source, self.output)
        self.assertTrue(self.output.is_symlink())

    def test_late_invalid_json_or_context_aborts_without_partial_artifacts(self):
        for invalid in (b"{bad json", b'\xff', b'{"contexts": {}}',
                        b'{"contexts": [false]}', b'{"contexts": [{"paragraph_text": 123}]}'):
            with self.subTest(invalid=invalid):
                self.write_rows(self.rows)
                with self.source.open("ab") as stream:
                    stream.write(b"\n" + invalid)
                with self.assertRaises(ValueError):
                    builder.build_corpus(self.source, self.output)
                self.assert_no_publication()

    def test_simulated_hash_collision_is_not_silently_deduplicated(self):
        with mock.patch.object(builder, "document_id", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 collision"):
                builder.build_corpus(self.source, self.output)
        self.assert_no_publication()

    def test_interrupted_publication_cleans_up_staging(self):
        with mock.patch.object(builder.os, "rename", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                builder.build_corpus(self.source, self.output)
        self.assert_no_publication()

    def test_atomic_publish_cannot_replace_a_concurrent_completed_output(self):
        original_rename = builder.os.rename

        def concurrent_publication(source, destination):
            self.assertEqual(sorted(path.name for path in Path(source).iterdir()),
                             ["documents.jsonl", "summary.json"])
            Path(destination).mkdir()
            (Path(destination) / "summary.json").write_bytes(b"other completed run")
            original_rename(source, destination)

        with mock.patch.object(builder.os, "rename", side_effect=concurrent_publication):
            with self.assertRaises(OSError):
                builder.build_corpus(self.source, self.output)
        self.assertEqual((self.output / "summary.json").read_bytes(), b"other completed run")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["corpus", "train.jsonl"])

    def test_source_is_iterated_without_whole_file_reads(self):
        original_open = Path.open

        class IterationOnly:
            def __enter__(self):
                self.stream = original_open(self_path, "rb")
                return self

            def __exit__(self, *args):
                self.stream.close()

            def __iter__(self):
                return iter(self.stream)

            def read(self, *args):
                raise AssertionError("Whole-input read forbidden")

            readlines = read

        self_path = self.source

        def guarded_open(path, *args, **kwargs):
            if path == self.source:
                return IterationOnly()
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            summary = builder.build_corpus(self.source, self.output)
        self.assertEqual(summary["questions"], 2)

    def test_empty_input(self):
        self.source.write_bytes(b"")
        summary = builder.build_corpus(self.source, self.output)
        self.assertEqual((self.output / "documents.jsonl").read_bytes(), b"")
        self.assertTrue(all(summary[key] == 0 for key in builder.FULL_DATASET_COUNTS))

    def test_cli_enforces_reference_counts_unless_explicitly_disabled(self):
        command = [sys.executable, "-B", str(SCRIPT), "--input", str(self.source),
                   "--output-dir", str(self.output)]
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1, failed.stderr)
        self.assertIn("Dataset count validation failed", failed.stderr)
        self.assert_no_publication()
        success = subprocess.run(command + ["--skip-count-validation"], capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads((self.output / "summary.json").read_bytes())[
            "count_validation"]["status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
