import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


splits = load("build_tae_r_splits")
corpora = load("build_tae_r_corpora")


def ctx(idx, title, text, support=False):
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": support}


def question(qid, contexts):
    return {"question_id": qid, "question_text": "Question " + qid,
            "answers_objects": [{"spans": ["a"]}], "contexts": contexts}


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        train = [question(f"t{i:02d}", [
            ctx(0, f"Own {i}", f"train-only-{i}", True), ctx(1, "Common", "common"),
            ctx(2, "Pair", f"pair-{i // 2}"), ctx(3, " dev  ENTITY ", f"variant-{i}"),
            ctx(4, "Shared Title", f"shared-title-{i}"), ctx(5, "Blank", " ")]) for i in range(20)]
        dev = [question(f"d{i}", [ctx(0, f"D{i}", f"dev-only-{i}", True), ctx(1, "Common", "common"),
                                  ctx(2, "Dev Entity", "dev-ent")]) for i in range(10)]
        write(self.data / "train.jsonl", train)
        write(self.data / "dev.jsonl", dev)
        write(self.data / "dev_subsampled.jsonl", [question("d0", [ctx(0, "D0", "remapped-a")])])
        write(self.data / "test_subsampled.jsonl", [question("d1", [ctx(0, "D1", "remapped-b")]),
                                                    question("d2", [ctx(0, "D2", "dev-only-2")])])
        self.splits = splits.build(self.data, self.root / "splits", seed=7, router_train_size=4,
                                   validation_size=2, e2e_sizes=[2, 3])
        self.n_fit = self.splits["split_files"]["topic_fit"]["count"]

    def run_build(self, name="out", mode="build", title="evaluation", **kw):
        args = dict(mode=mode, title_exclusion=title,
                    frozen_manifest_sha256=self.splits["manifest_sha256"],
                    expected_text_population=self.splits["counts"]["unique_texts"]["topic_fit_population"])
        args.update(kw)
        return corpora.build(self.data, self.root / "splits", self.root / name, **args)

    def lines(self, name, file):
        return [json.loads(x) for x in (self.root / name / file).read_text(encoding="utf-8").splitlines()]

    def test_measure_reports_corrected_title_diagnostic_and_writes_nothing(self):
        result = self.run_build(mode="measure")
        counts = result["counts"]
        diag = counts["title_overlap_diagnostic"]
        self.assertEqual(diag["first_title_exact_vs_evaluation_titles_original_diagnostic"], 0)
        self.assertEqual(diag["any_title_exact_vs_evaluation_titles"], 0)
        self.assertEqual(diag["any_title_normalized_vs_evaluation_titles"], self.n_fit)
        self.assertEqual(counts["texts_excluded_by_title"], self.n_fit)
        self.assertEqual(counts["topic_fit_population_after_title_exclusion"],
                         counts["text_level_population"] - self.n_fit)
        self.assertFalse((self.root / "out").exists())

    def test_heldout_title_mode_excludes_more(self):
        counts = self.run_build(mode="measure", title="evaluation_and_heldout")["counts"]
        ids = lambda f: (self.root / "splits" / f).read_text().split()
        fit, held = ids("topic_fit_ids.txt"), set(ids("router_train_ids.txt")) | set(ids("validation_ids.txt"))
        group = lambda q: int(q[1:]) // 2
        pair_only = {group(q) for q in fit} - {group(q) for q in held}
        # variant + shared-title texts, plus pair texts (title "Pair" occurs in held-out contexts)
        self.assertEqual(counts["texts_excluded_by_title"], 2 * self.n_fit + len(pair_only))
        self.assertEqual(counts["title_overlap_diagnostic"]["any_title_normalized_vs_router_train_validation_titles"],
                         2 * self.n_fit + len(pair_only))

    def test_build_writes_disjoint_corpora_with_stable_ids(self):
        manifest = self.run_build()
        self.assertTrue(manifest["inputs_match_split_manifest"])
        population = self.lines("out", "topic_fit_population.jsonl")
        final = self.lines("out", "retrieval_final.jsonl")
        validation = self.lines("out", "retrieval_validation.jsonl")
        pop_ids, final_ids, val_ids = ({r["doc_id"] for r in x} for x in (population, final, validation))
        self.assertFalse(pop_ids & final_ids or pop_ids & val_ids)
        self.assertEqual(len(final), 12)
        for record in population + final + validation:
            self.assertEqual(record["doc_id"], hashlib.sha256(record["text"].encode("utf-8")).hexdigest())
        self.assertFalse(any(r["text"].startswith("variant-") for r in population))
        self.assertTrue(any(r["text"].startswith("shared-title-") for r in population))
        excluded = (self.root / "out" / "excluded_by_title_doc_ids.txt").read_text().split()
        self.assertEqual(len(excluded), self.n_fit)
        self.assertFalse(set(excluded) & pop_ids)
        for info in manifest["files"].values():
            self.assertEqual(sha(self.root / "out" / info["file"]), info["sha256"])

    def test_question_files_carry_gold_inside_their_corpus(self):
        self.run_build()
        final_ids = {r["doc_id"] for r in self.lines("out", "retrieval_final.jsonl")}
        held_ids = {r["doc_id"] for r in self.lines("out", "retrieval_validation.jsonl")}
        questions = self.lines("out", "questions_final_eval.jsonl")
        self.assertEqual([q["question_id"] for q in questions], [f"d{i}" for i in range(3, 10)])
        for q in questions:
            self.assertEqual(len(q["gold_doc_ids"]), 1)
            self.assertTrue(set(q["context_doc_ids"]) <= final_ids)
        for file in ("questions_router_train.jsonl", "questions_validation.jsonl"):
            for q in self.lines("out", file):
                self.assertEqual(q["blank_contexts"], 1)
                self.assertTrue(set(q["gold_doc_ids"]) <= held_ids)
                self.assertIn("answers_objects", q)
                self.assertNotIn("contexts", q)

    def test_deterministic(self):
        first, second = self.run_build("a"), self.run_build("b")
        self.assertEqual({k: v["sha256"] for k, v in first["files"].items()},
                         {k: v["sha256"] for k, v in second["files"].items()})

    def test_refusals(self):
        self.run_build()
        with self.assertRaises(ValueError):
            self.run_build()
        with self.assertRaises(ValueError):
            self.run_build("x", frozen_manifest_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.run_build("y", expected_text_population=1)
        with self.assertRaises(ValueError):
            self.run_build("z", title="bogus")
        with self.assertRaises(ValueError):
            corpora.build(self.data, self.root / "splits", None, mode="build", title_exclusion="evaluation",
                          frozen_manifest_sha256=self.splits["manifest_sha256"], expected_text_population=1)
        split_file = self.root / "splits" / "validation_ids.txt"
        split_file.chmod(0o644)
        split_file.write_text(split_file.read_text() + "extra\n")
        with self.assertRaises(ValueError):
            self.run_build("w")

    def test_changed_input_is_refused(self):
        lines = (self.data / "train.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        (self.data / "train.jsonl").write_text("".join(lines[:-1]), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_build("v")


if __name__ == "__main__":
    unittest.main()
