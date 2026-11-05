import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_2wiki_corpus_design.py"
spec = importlib.util.spec_from_file_location("audit_design", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def question(qid, texts):
    return {"question_id": qid, "question_text": "Question " + qid,
            "answers_objects": [{"spans": ["unused answer"]}],
            "contexts": [{"idx": i, "title": "Title " + str(i), "paragraph_text": t,
                          "is_supporting": i == 0} for i, t in enumerate(texts)]}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.records = {
            "train": [question("t1", ["A", "A", "B", " ", None]),
                      question("t2", ["C", " A", "é", "e\u0301"])],
            "original_dev": [question("d1", ["A", "D"]), question("d2", ["E", "D"])],
            "development": [question("d1", ["A", "D"])],
            "heldout": [question("d2", ["F", "D"])]}
        self.write()

    def write(self):
        for split, name in audit.NAMES.items():
            (self.data / name).write_bytes(b"".join(audit.encoded(q) + b"\n" for q in self.records[split]))

    def run_audit(self):
        return audit.audit(self.data, scratch_dir=self.root)

    def test_counts_exact_bytes_and_distributions(self):
        r = self.run_audit()
        t = r["splits"]["train"]
        self.assertEqual(t["context_occurrences"], 9)
        self.assertEqual(t["nonempty_occurrences"], 7)
        self.assertEqual(t["empty_occurrences"], 2)
        self.assertEqual(t["unique_nonempty_texts"], 6)
        self.assertEqual(t["duplicate_excess_occurrences"], 1)
        self.assertEqual(t["maximum_text_multiplicity"], 2)
        self.assertEqual(t["duplicated_text_groups"], 1)
        self.assertEqual(t["occurrences_in_duplicated_text_groups"], 2)
        self.assertEqual(t["text_multiplicity_distribution"], {1: 5, 2: 1})
        self.assertEqual(t["contexts_per_question_distribution"], {4: 1, 5: 1})
        self.assertEqual(t["nonempty_contexts_per_question_distribution"], {3: 1, 4: 1})
        self.assertEqual(r["splits"]["original_dev"]["unique_nonempty_texts"], 3)
        self.assertEqual(r["splits"]["heldout"]["contexts_per_question_distribution"], {2: 1})

    def test_overlap_and_fitting_exclusion(self):
        r = self.run_audit()
        overlap = r["pairwise_overlap"]["train__original_dev"]
        self.assertEqual(overlap, {"shared_unique_texts": 1, "left_occurrences_of_shared_texts": 2,
                                  "right_occurrences_of_shared_texts": 1, "shared_distinct_question_ids": 0})
        self.assertEqual(r["pairwise_overlap"]["original_dev__heldout"]["shared_distinct_question_ids"], 1)
        self.assertEqual(r["fresh_topic_fitting"]["train_unique_excluding_all_original_dev_and_evaluation_texts"], 5)

    def test_budget_boundaries_and_background_feasibility(self):
        with patch.object(audit, "BUDGETS", (2, 3, 4, 8, 9)):
            r = self.run_audit()["budget_feasibility"]
        e = r["all_evaluation_contexts"]
        self.assertEqual(e["mandatory_unique_texts"], 3)
        self.assertEqual(e["training_only_background_available"], 5)
        self.assertFalse(e["budgets"]["2"]["mandatory_fits"])
        self.assertTrue(e["budgets"]["3"]["mandatory_fits"])
        self.assertFalse(e["budgets"]["3"]["fits_with_at_least_one_training_background_text"])
        self.assertTrue(e["budgets"]["8"]["can_fill_exact_budget"])
        self.assertFalse(e["budgets"]["9"]["can_fill_exact_budget"])
        self.assertEqual(r["full_original_dev_plus_evaluation_contexts"]["mandatory_unique_texts"], 4)

    def test_remapping_and_original_gold_diagnostic(self):
        r = self.run_audit()
        d = r["context_remapping"]["development"]
        h = r["context_remapping"]["heldout"]
        self.assertEqual(d["questions_with_context_changes"], 0)
        self.assertEqual(h["text_changes"], 1)
        self.assertEqual(h["original_gold_nonempty_occurrences"], 1)
        self.assertEqual(h["original_gold_texts_present_in_own_contexts"], 0)
        self.assertEqual(r["splits"]["heldout"]["own_annotation_diagnostics"]["nonempty_support_occurrences"], 1)
        serialized = json.dumps(r)
        for private in ('"d1"', '"d2"', '"F"', 'Question d2', 'unused answer'):
            self.assertNotIn(private, serialized)

    def test_labels_and_answers_cannot_change_structural_design(self):
        before = self.run_audit()
        for records in self.records.values():
            for q in records:
                q["answers_objects"] = "different"
                q["reasoning_steps"] = ["arbitrary"]
                for ctx in q["contexts"]:
                    ctx["is_supporting"] = not ctx["is_supporting"]
        self.write()
        after = self.run_audit()
        for key in ("budget_feasibility", "fresh_topic_fitting", "pairwise_overlap"):
            self.assertEqual(before[key], after[key])
        for split in audit.NAMES:
            for key in ("unique_nonempty_texts", "text_multiplicity_distribution",
                        "contexts_per_question_distribution", "distinct_question_ids"):
                self.assertEqual(before["splits"][split][key], after["splits"][split][key])

    def test_duplicate_ids_are_counted_and_ambiguous_comparison_skipped(self):
        self.records["heldout"].append(copy.deepcopy(self.records["heldout"][0]))
        self.records["train"][0]["question_id"] = "d1"
        self.write()
        r = self.run_audit()
        self.assertEqual(r["splits"]["heldout"]["distinct_question_ids"], 1)
        self.assertEqual(r["splits"]["heldout"]["duplicate_question_id_excess_records"], 1)
        self.assertEqual(r["context_remapping"]["heldout"]["ambiguous_duplicate_ids"], 1)
        self.assertEqual(r["pairwise_overlap"]["train__development"]["shared_distinct_question_ids"], 1)
        self.assertIn("train/development: question-ID overlap", r["design_validation"]["structural_blockers"])

    def test_remap_count_idx_title_and_missing_original(self):
        q = self.records["development"][0]
        q["contexts"][0]["title"] = "new"
        q["contexts"][0]["idx"] = 9
        q["contexts"].append(copy.deepcopy(q["contexts"][0]))
        q["question_text"] = "changed"
        self.records["heldout"][0]["question_id"] = "unmatched"
        self.write()
        r = self.run_audit()["context_remapping"]
        for field in ("title_changes", "idx_changes", "added_context_positions",
                      "context_count_changes", "question_text_changes"):
            self.assertEqual(r["development"][field], 1)
        self.assertEqual(r["heldout"]["missing_original_ids"], 1)

    def test_invalid_flags_and_empty_support_are_aggregate_diagnostics(self):
        q = self.records["heldout"][0]
        q["contexts"][0]["paragraph_text"] = " "
        q["contexts"][1]["is_supporting"] = 1
        self.write()
        d = self.run_audit()["splits"]["heldout"]["own_annotation_diagnostics"]
        self.assertEqual(d["empty_support_occurrences"], 1)
        self.assertEqual(d["invalid_or_missing_flags"], 1)

    def test_hashes_and_read_only_sources(self):
        before = {p.name: p.read_bytes() for p in self.data.iterdir()}
        r = self.run_audit()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.data.iterdir()})
        for split, name in audit.NAMES.items():
            self.assertEqual(r["splits"][split]["sha256"], hashlib.sha256(before[name]).hexdigest())
        self.assertFalse(list(self.root.glob("2wiki-audit-*")))

    def test_complete_cli_custom_filename_and_json_roundtrip(self):
        (self.data / "dev_subsampled.jsonl").rename(self.data / "dev_100_subsampled.jsonl")
        output = self.root / "complete.json"
        self.assertEqual(audit.main(["--data-dir", str(self.data), "--output", str(output),
                                     "--development-name", "dev_100_subsampled.jsonl"]), 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["design_validation"]["structural_blockers"], [])
        self.assertEqual(report["splits"]["heldout"]["contexts_per_question_distribution"], {"2": 1})
        self.assertTrue(report["design_validation"]["context_source_choice_requires_freezing"])

    def test_blank_lines_are_hashed_and_do_not_add_questions(self):
        path = self.data / "train.jsonl"
        path.write_bytes(b"\n" + path.read_bytes() + b" \n")
        t = self.run_audit()["splits"]["train"]
        self.assertEqual(t["blank_lines"], 2)
        self.assertEqual(t["questions"], 2)
        self.assertEqual(t["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_missing_files_generate_incomplete_json_and_exit_two(self):
        (self.data / audit.NAMES["heldout"]).unlink()
        output = self.root / "missing.json"
        self.assertEqual(audit.main(["--data-dir", str(self.data), "--output", str(output)]), 2)
        r = json.loads(output.read_text())
        self.assertEqual(r["missing_splits"], ["heldout"])
        self.assertIsNone(r["splits"]["heldout"])
        self.assertIsNone(r["budget_feasibility"])

    def test_empty_splits_have_defined_zero_counts(self):
        self.records = {s: [] for s in audit.NAMES}
        self.write()
        r = self.run_audit()
        self.assertEqual(r["splits"]["train"]["maximum_text_multiplicity"], 0)
        self.assertEqual(r["budget_feasibility"]["all_evaluation_contexts"]["mandatory_unique_texts"], 0)

    def test_malformed_records_fail_without_private_values(self):
        for content in (b'{"question_id":"secret","question_id":"other"}\n',
                        b'{"x":NaN}\n', b'\xff\n', b'[]\n'):
            (self.data / "test_subsampled.jsonl").write_bytes(content)
            with self.assertRaises(ValueError) as caught:
                self.run_audit()
            self.assertNotIn("secret", str(caught.exception))

    def test_invalid_context_types_and_surrogates_fail(self):
        for value in (7, {"bad": "value"}, "\ud800"):
            self.records["heldout"][0]["contexts"][0]["paragraph_text"] = value
            # ensure_ascii allows malformed Unicode to reach the parser validation.
            (self.data / "test_subsampled.jsonl").write_text(json.dumps(self.records["heldout"][0]) + "\n")
            with self.assertRaises(ValueError):
                self.run_audit()

    def test_cli_refuses_overwrite_symlink_and_dataset_output(self):
        output = self.root / "existing.json"
        output.write_text("preserved")
        self.assertEqual(audit.main(["--data-dir", str(self.data), "--output", str(output)]), 1)
        self.assertEqual(output.read_text(), "preserved")
        link = self.root / "link.json"
        link.symlink_to(self.root / "absent")
        self.assertEqual(audit.main(["--data-dir", str(self.data), "--output", str(link)]), 1)
        self.assertEqual(audit.main(["--data-dir", str(self.data), "--output", str(self.data / "new.json")]), 1)

    def test_scratch_and_filename_guards(self):
        with self.assertRaises(ValueError):
            audit.audit(self.data, scratch_dir=self.data)
        names = dict(audit.NAMES, heldout="../outside.jsonl")
        with self.assertRaises(ValueError):
            audit.audit(self.data, names=names)

    def test_input_mutation_detected(self):
        real = audit.sha256
        def changed(path):
            return "0" * 64 if Path(path).name == "train.jsonl" else real(path)
        with patch.object(audit, "sha256", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                self.run_audit()


if __name__ == "__main__":
    unittest.main()
