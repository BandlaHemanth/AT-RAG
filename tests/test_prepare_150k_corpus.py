"""Small synthetic JSONL fixtures; no dataset downloads or ML dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_150k_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_150k", SCRIPT)
corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corpus)


def context(text, idx=0, title="Title"):
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": True}


def question(qid, texts):
    return {"question_id": qid, "question_text": "Query " + qid,
            "contexts": [context(text, idx=i) for i, text in enumerate(texts)],
            "answers_objects": [{"spans": ["GOLD ANSWER"]}], "reasoning_steps": ["GOLD REASONING"]}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.training = [question(f"train-{i}", [f"text-{i}", " repeated text \nΩ "]) for i in range(6)]
        self.development = [question(f"dev-{i}", [f"dev-text-{i}"]) for i in range(4)]
        self.write("train.jsonl", self.training)
        self.write("dev.jsonl", self.development)
        self.write("dev_subsampled.jsonl", self.development[:2])
        self.write("test_subsampled.jsonl", self.development[2:])

    def write(self, name, rows):
        (self.data / name).write_bytes(b"".join(corpus.json_bytes(row) for row in rows))

    def run_prepare(self, name="run", count=7, seed=42):
        return corpus.prepare(self.data, self.root / name, sample_size=count, seed=seed)

    def saved(self, name="run"):
        return [json.loads(line) for line in (self.root / name / "corpus.jsonl").read_text().splitlines()]

    def test_deterministic_selection_and_hashes(self):
        first = self.run_prepare("a")
        second = self.run_prepare("b")
        self.assertEqual((self.root / "a/corpus.jsonl").read_bytes(), (self.root / "b/corpus.jsonl").read_bytes())
        self.assertEqual(first, second)
        self.assertEqual(len(self.saved("a")), 7)
        self.assertEqual(first["sampling"]["seed"], 42)
        self.assertEqual(first["sampling"]["eligible_occurrences"], 12)
        for name, stats in first["inputs"].items():
            self.assertEqual(stats["sha256"], corpus.sha256(self.data / name))
        for line in (self.root / "a/SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ")
            self.assertEqual(expected, corpus.sha256(self.root / "a" / name))

    def test_selection_matches_frozen_fixture_and_reaches_tail(self):
        # Pin the seed-42 fixture result so algorithm/RNG changes require review.
        expected = [2, 5, 6, 7, 8, 9, 11]
        self.run_prepare()
        actual = [(row["source_line"] - 1) * 2 + row["context_position"] for row in self.saved()]
        self.assertEqual(actual, sorted(expected))
        self.assertGreater(max(actual), 6)
        self.run_prepare("other_seed", seed=43)
        self.assertNotEqual([r["occurrence_id"] for r in self.saved()],
                            [r["occurrence_id"] for r in self.saved("other_seed")])

    def test_duplicates_preserved_exactly_with_distinct_occurrence_ids(self):
        report = self.run_prepare(count=12)
        saved = self.saved()
        repeated = [r for r in saved if r["context_position"] == 1]
        self.assertEqual([r["paragraph_text"] for r in repeated], [" repeated text \nΩ "] * 6)
        self.assertEqual(len({r["text_sha256"] for r in repeated}), 1)
        self.assertEqual(len({r["occurrence_id"] for r in saved}), 12)
        self.assertEqual({r["idx"] for r in saved}, {0, 1})
        self.assertEqual(report["validation"]["unique_texts"], 7)
        self.assertEqual(report["validation"]["duplicate_text_excess_occurrences"], 5)
        self.assertEqual(report["validation"]["duplicated_text_groups"], 1)
        self.assertEqual(report["validation"]["occurrences_in_duplicated_text_groups"], 6)
        self.assertEqual(report["validation"]["maximum_text_multiplicity"], 6)
        self.assertEqual(report["validation"]["source_integrity_verified_occurrences"], 12)
        for r in saved:
            self.assertEqual(set(r), corpus.CORPUS_FIELDS)
            original = self.training[r["source_line"] - 1]
            c = original["contexts"][r["context_position"]]
            self.assertEqual(r["question_id"], original["question_id"])
            for key in ("idx", "title", "paragraph_text"):
                self.assertEqual(r[key], c[key])
            self.assertIn(report["inputs"]["train.jsonl"]["sha256"], r["occurrence_id"])

    def test_context_position_distinguishes_repeated_local_idx_in_one_question(self):
        self.training[0]["contexts"][1]["idx"] = 0
        self.write("train.jsonl", self.training)
        self.run_prepare(count=12)
        first = self.saved()[:2]
        self.assertEqual([r["idx"] for r in first], [0, 0])
        self.assertEqual([r["context_position"] for r in first], [0, 1])
        self.assertNotEqual(first[0]["occurrence_id"], first[1]["occurrence_id"])

    def test_hash_collision_is_not_misreported_as_duplicate_text(self):
        with mock.patch.object(corpus, "text_hash", return_value="collision"):
            with self.assertRaisesRegex(ValueError, "SHA-256 text collision"):
                self.run_prepare(count=12)
        self.assertFalse((self.root / "run").exists())

    def test_empty_texts_counted_and_excluded_without_modifying_nonempty_text(self):
        self.training[0]["contexts"].extend([context(None, 3), context("", 4), context(" \t\n", 5)])
        self.write("train.jsonl", self.training)
        report = self.run_prepare(count=12)
        self.assertEqual(report["inputs"]["train.jsonl"]["empty_contexts"], 3)
        self.assertEqual(report["inputs"]["train.jsonl"]["context_occurrences"], 15)
        self.assertTrue(all(r["paragraph_text"].strip() for r in self.saved()))

    def test_gold_changes_do_not_affect_sampling_or_training_projection(self):
        first = self.run_prepare("a")
        for q in self.training:
            q["answers_objects"] = "changed answer"
            q["reasoning_steps"] = {"different": "reasoning"}
            for c in q["contexts"]:
                c["is_supporting"] = not c["is_supporting"]
        self.write("train.jsonl", self.training)
        second = self.run_prepare("b")
        self.assertEqual(first["sampling"]["eligible_ordinal_selection_sha256"],
                         second["sampling"]["eligible_ordinal_selection_sha256"])
        # Input-byte-qualified occurrence IDs change, while selected source records do not.
        projected = lambda name: [{k: v for k, v in r.items() if k != "occurrence_id"} for r in self.saved(name)]
        self.assertEqual(projected("a"), projected("b"))
        serialized = (self.root / "b/corpus.jsonl").read_text()
        for forbidden in ("is_supporting", "answers_objects", "reasoning_steps", "question_text"):
            self.assertNotIn(forbidden, serialized)

    def test_malformed_records_abort_without_publishing(self):
        mutations = [
            lambda q: q.pop("question_id"),
            lambda q: q.update(question_id=""),
            lambda q: q.update(contexts={}),
            lambda q: q["contexts"].append("invalid context"),
            lambda q: q["contexts"][0].pop("paragraph_text"),
            lambda q: q["contexts"][0].update(paragraph_text=123),
            lambda q: q["contexts"][0].update(idx=True),
            lambda q: q["contexts"][0].update(idx=-1),
            lambda q: q["contexts"][0].pop("title"),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                changed = copy.deepcopy(self.training)
                mutate(changed[-1])  # Even malformed records beyond the initial reservoir must fail.
                self.write("train.jsonl", changed)
                with self.assertRaises(ValueError):
                    self.run_prepare(f"bad-{i}", count=1)
                self.assertFalse((self.root / f"bad-{i}").exists())

    def test_invalid_json_utf8_duplicate_keys_and_nonfinite_constants_rejected(self):
        for i, raw in enumerate((b"{invalid}\n", b"\xff\n", b'{"x":NaN}\n',
                                 b'{"question_id":"a","question_id":"b","contexts":[]}\n')):
            with self.subTest(case=i):
                (self.data / "train.jsonl").write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.run_prepare(f"bad-{i}")
                self.assertFalse((self.root / f"bad-{i}").exists())

    def test_duplicate_training_question_ids_rejected(self):
        self.write("train.jsonl", self.training + [self.training[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate training question_id"):
            self.run_prepare()

    def test_insufficient_eligible_population_rejected(self):
        with self.assertRaisesRegex(ValueError, "found 12"):
            self.run_prepare(count=13)
        self.assertFalse((self.root / "run").exists())

    def test_blank_source_lines_preserve_physical_line_provenance(self):
        path = self.data / "train.jsonl"
        path.write_bytes(b"\n" + path.read_bytes() + b" \n")
        report = self.run_prepare(count=12)
        self.assertEqual(report["inputs"]["train.jsonl"]["blank_lines"], 2)
        self.assertEqual(self.saved()[0]["source_line"], 2)

    def test_evaluation_split_overlap_rejected(self):
        self.write("test_subsampled.jsonl", [self.development[0]])
        with self.assertRaisesRegex(ValueError, "Evaluation split question-ID overlap"):
            self.run_prepare()
        self.assertFalse((self.root / "run").exists())

    def test_train_evaluation_overlap_checks_full_training_population(self):
        self.write("test_subsampled.jsonl", [self.training[-1]])
        with self.assertRaisesRegex(ValueError, "Training/evaluation question-ID overlap"):
            self.run_prepare(count=1)

    def test_train_original_dev_overlap_rejected_even_outside_subsamples(self):
        self.write("dev.jsonl", self.development + [self.training[-1]])
        with self.assertRaisesRegex(ValueError, "Original development/training question-ID overlap"):
            self.run_prepare()

    def test_unverified_eval_provenance_and_query_changes_rejected(self):
        self.write("test_subsampled.jsonl", [question("unknown", ["unknown text"])])
        with self.assertRaisesRegex(ValueError, "missing from original dev"):
            self.run_prepare("missing")
        changed = copy.deepcopy(self.development[2:])
        changed[0]["question_text"] = "Different query"
        self.write("test_subsampled.jsonl", changed)
        with self.assertRaisesRegex(ValueError, "question_text differs"):
            self.run_prepare("changed")

    def test_duplicate_eval_and_original_dev_ids_rejected(self):
        self.write("dev_subsampled.jsonl", [self.development[0]] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
            self.run_prepare("duplicate-eval")
        self.write("dev_subsampled.jsonl", self.development[:2])
        self.write("dev.jsonl", self.development + [self.development[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate original development"):
            self.run_prepare("duplicate-source")

    def test_evaluation_context_positions_keep_original_local_idx(self):
        changed = copy.deepcopy(self.development[2:])
        changed[0]["contexts"][0]["idx"] += 10
        self.write("test_subsampled.jsonl", changed)
        with self.assertRaisesRegex(ValueError, "context idx differs"):
            self.run_prepare()

    def test_protocol_freezes_actual_ids_and_reports_context_remapping(self):
        changed = copy.deepcopy(self.development[2:])
        changed[0]["contexts"][0]["paragraph_text"] = "remapped text"
        self.write("test_subsampled.jsonl", changed)
        report = self.run_prepare()
        protocol = json.loads((self.root / "run/frozen_evaluation_protocol.json").read_text())
        test = protocol["evaluation_audit"]["splits"]["test_subsampled.jsonl"]
        self.assertEqual(test["ordered_question_ids"], ["dev-2", "dev-3"])
        self.assertFalse(test["official_test_set"])
        self.assertEqual(test["original_source_split"], "development")
        self.assertEqual(test["context_comparison"]["changed_context_positions"], 1)
        self.assertIn("Keep uncovered", protocol["retrieval_reporting"]["unconditional_main"])
        self.assertEqual(protocol["corpus_sha256"], report["validation"]["corpus_sha256"])
        self.assertFalse(any(r["paragraph_text"] == "remapped text" for r in self.saved()))

    def test_existing_output_directory_file_or_symlink_never_overwritten(self):
        for name, kind in (("empty", "dir"), ("full", "full"), ("file", "file"), ("link", "link")):
            with self.subTest(kind=kind):
                path = self.root / name
                if kind in ("dir", "full"):
                    path.mkdir()
                    if kind == "full":
                        (path / "sentinel").write_bytes(b"preserve")
                elif kind == "file":
                    path.write_bytes(b"preserve")
                else:
                    path.symlink_to(self.root / "missing-target")
                with self.assertRaises(FileExistsError):
                    self.run_prepare(name)
                if kind == "full":
                    self.assertEqual((path / "sentinel").read_bytes(), b"preserve")
                elif kind == "file":
                    self.assertEqual(path.read_bytes(), b"preserve")
                elif kind == "link":
                    self.assertTrue(path.is_symlink())
                else:
                    self.assertEqual(list(path.iterdir()), [])

    def test_concurrent_empty_destination_is_not_replaced(self):
        original = corpus.audit_evaluation
        def race(*args):
            result = original(*args)
            (self.root / "run").mkdir()
            return result
        with mock.patch.object(corpus, "audit_evaluation", side_effect=race):
            with self.assertRaises(FileExistsError):
                self.run_prepare()
        self.assertEqual(list((self.root / "run").iterdir()), [])

    def test_source_change_between_passes_aborts(self):
        original = corpus.write_corpus
        def changed(*args):
            original(*args)
            with (self.data / "train.jsonl").open("ab") as stream:
                stream.write(b"\n")
        with mock.patch.object(corpus, "write_corpus", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed between passes"):
                self.run_prepare()
        self.assertFalse((self.root / "run").exists())

    def test_corrupted_corpus_source_metadata_and_extra_rows_rejected(self):
        original = corpus.write_corpus
        for name, extra in (("metadata", False), ("extra", True)):
            def corrupt(*args):
                original(*args)
                path = args[1]
                saved = [json.loads(line) for line in path.read_text().splitlines()]
                if extra:
                    saved.append(saved[-1])
                else:
                    saved[0]["idx"] += 100
                path.write_bytes(b"".join(corpus.json_bytes(row) for row in saved))
            with mock.patch.object(corpus, "write_corpus", side_effect=corrupt):
                with self.assertRaises(ValueError):
                    self.run_prepare(name)
            self.assertFalse((self.root / name).exists())

    def test_missing_original_development_fails_without_creating_output(self):
        (self.data / "dev.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_prepare()
        self.assertFalse((self.root / "run").exists())

    def test_cli_fixes_production_size_and_seed_and_restricts_output(self):
        self.assertEqual(corpus.SAMPLE_SIZE, 150_000)
        self.assertEqual(corpus.SEED, 42)
        with mock.patch.object(corpus, "prepare") as prepare, mock.patch("sys.stderr"):
            self.assertEqual(corpus.main(["--output-dir", str(self.root / "outside")]), 1)
            prepare.assert_not_called()
        with mock.patch.object(corpus, "prepare", return_value={"status": "passed", "validation": {}}) as prepare:
            with mock.patch("sys.stdout"):
                self.assertEqual(corpus.main([]), 0)
            self.assertEqual(prepare.call_args.args[1], corpus.OUTPUT_ROOT / "run_001")
            self.assertEqual(prepare.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main()
