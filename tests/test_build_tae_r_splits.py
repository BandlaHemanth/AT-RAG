import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_tae_r_splits.py"
spec = importlib.util.spec_from_file_location("tae_r_splits", SCRIPT)
splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splits)


def question(qid, texts, question_text=None):
    return {"question_id": qid, "question_text": question_text or "Question " + qid,
            "contexts": [{"idx": i, "title": "T-" + t, "paragraph_text": t,
                          "is_supporting": i == 0} for i, t in enumerate(texts)]}


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class SplitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        # 20 train questions; each has one private text, and the shared texts
        # "common" (also in dev) and "pair-k" (shared by neighbouring questions).
        train = [question(f"t{i:02d}", [f"train-only-{i}", "common", f"pair-{i // 2}", " "])
                 for i in range(20)]
        dev = [question(f"d{i}", [f"dev-only-{i}", "common"]) for i in range(10)]
        write(self.data / "train.jsonl", train)
        write(self.data / "dev.jsonl", dev)
        # Subsampled files hold dev IDs with altered ("remapped") context text.
        write(self.data / "dev_subsampled.jsonl", [question("d0", ["remapped-a", "train-only-3"])])
        write(self.data / "test_subsampled.jsonl", [question("d1", ["remapped-b"]),
                                                    question("d2", ["dev-only-2"])])

    def build(self, name="out", **kw):
        args = dict(seed=7, router_train_size=4, validation_size=2, e2e_sizes=[2, 3])
        args.update(kw)
        return splits.build(self.data, self.root / name, **args)

    def ids(self, name, file):
        return (self.root / name / file).read_text(encoding="utf-8").split()

    def test_splits_are_disjoint_and_complete(self):
        m = self.build()
        self.assertFalse(any(m["question_id_overlaps"].values()))
        self.assertEqual(self.ids("out", "exposed_ids.txt"), ["d0", "d1", "d2"])
        self.assertEqual(self.ids("out", "final_eval_ids.txt"), [f"d{i}" for i in range(3, 10)])
        train_parts = [self.ids("out", f) for f in
                       ("router_train_ids.txt", "validation_ids.txt", "topic_fit_ids.txt")]
        self.assertEqual([len(p) for p in train_parts], [4, 2, 14])
        self.assertEqual(sorted(sum(train_parts, [])), [f"t{i:02d}" for i in range(20)])
        self.assertEqual(m["exposed_ids_not_in_original_dev"], 0)
        self.assertFalse(m["labels_used_for_selection"])

    def test_e2e_subsets_are_nested_inside_final_eval(self):
        self.build()
        small, large = self.ids("out", "e2e_2_ids.txt"), self.ids("out", "e2e_3_ids.txt")
        self.assertTrue(set(small) < set(large) <= set(self.ids("out", "final_eval_ids.txt")))

    def test_topic_fit_population_excludes_every_held_out_text(self):
        m = self.build()
        fit_ids = set(self.ids("out", "topic_fit_ids.txt"))
        held = set(self.ids("out", "router_train_ids.txt")) | set(self.ids("out", "validation_ids.txt"))
        records = [json.loads(line) for line in
                   (self.data / "train.jsonl").read_text(encoding="utf-8").splitlines()]

        def texts(ids):
            return {c["paragraph_text"] for r in records if r["question_id"] in ids
                    for c in r["contexts"] if c["paragraph_text"].strip()}
        evaluation = {"common", "remapped-a", "remapped-b", "train-only-3"} | {
            f"dev-only-{i}" for i in range(10)}
        expected = texts(fit_ids) - texts(held) - evaluation
        counts = m["counts"]["unique_texts"]
        self.assertEqual(counts["topic_fit_population"], len(expected))
        self.assertNotIn("common", expected)
        self.assertEqual(counts["final_retrieval_pool_original_dev"], 11)
        self.assertEqual(counts["validation_retrieval_pool_router_train_plus_validation"],
                         len(texts(held)))

    def test_deterministic_and_independent_of_file_order(self):
        first = self.build("a")
        lines = (self.data / "train.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        (self.data / "train.jsonl").write_text("".join(reversed(lines)), encoding="utf-8")
        second = self.build("b")
        for key in ("router_train", "validation", "topic_fit", "final_eval", "e2e_2"):
            self.assertEqual(first["split_files"][key]["sha256"], second["split_files"][key]["sha256"])
        other = self.build("c", seed=8)
        self.assertNotEqual(first["split_files"]["router_train"]["sha256"],
                            other["split_files"]["router_train"]["sha256"])

    def test_refuses_existing_output_missing_input_and_duplicates(self):
        self.build()
        with self.assertRaises(ValueError):
            self.build()
        with self.assertRaises(ValueError):
            self.build("inside", router_train_size=30)
        write(self.data / "dev.jsonl", [question("d0", ["x"]), question("d0", ["y"])])
        with self.assertRaises(ValueError):
            self.build("dup")
        (self.data / "dev_subsampled.jsonl").unlink()
        self.assertEqual(splits.main(["--data-dir", str(self.data),
                                      "--output-dir", str(self.root / "missing")]), 1)

    def test_reports_shared_question_text_and_audit_hash_mismatch(self):
        records = [json.loads(line) for line in
                   (self.data / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
        records[5]["question_text"] = "Question t00"
        write(self.data / "dev.jsonl", records)
        m = self.build(router_train_size=18, validation_size=1)
        shared = m["counts"]["question_text_shared_with_original_dev"]
        self.assertEqual(sum(shared.values()) + (1 if "t00" in self.ids("out", "topic_fit_ids.txt") else 0), 1)
        self.assertFalse(m["inputs_match_audit_run_001"])


if __name__ == "__main__":
    unittest.main()
