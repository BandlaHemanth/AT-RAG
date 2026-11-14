import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_final_results", ROOT / "scripts" / "analyze_final_results.py")
ana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana)


def outcome(all5, searched):
    return {"hits@5": 2 * all5, "hits@10": 2, "all@5": all5, "all@10": 1, "searched": searched}


def row(qid, oracle, strict, s1, s3, tae_route, all5):
    routes = {"unfiltered": None, "strict": strict, "oracle": oracle, "S1": s1, "S3": s3, "P1_S1": s1, "P1_S3": s3,
              "tae_v1": strict, "tae_v2": s3, "fixed_three": s3}
    chosen = {m: m for m in ("unfiltered", "strict", "oracle", "tae_v1", "tae_v2", "fixed_three")}
    chosen.update(tae_r=tae_route, tae_r_p1="unfiltered")
    outcomes = {k: outcome(all5.get(k, 0), 10) for k in routes}
    return {"question_id": qid, "routes": routes, "chosen": chosen, "outcomes": outcomes, "features": {}}


ROWS = [
    row("a", [1], [1], [1], [1, 2, 3], "S1", {"strict": 1, "S1": 1, "unfiltered": 1, "oracle": 1}),
    row("b", [1, 2], [-1], [1], [1, 2, 3], "S3", {"S3": 1, "unfiltered": 1, "oracle": 1}),
    row("c", [4], [2], [1], [1, 2, 3], "unfiltered", {"oracle": 1}),
    row("d", [5], [5], [1], [1, 2, 3], "S1", {"strict": 1, "unfiltered": 1, "oracle": 1}),
]


class RetrievalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.stats = ana._load("tae_r_router", "tae_r_router.py")

    def test_availability_routes_gap_and_anatomy(self):
        types = {"a": "comparison", "b": "comparison", "c": "inference", "d": "inference"}
        out = ana.retrieval_analysis(ROWS, self.stats, types=types)
        self.assertEqual(out["availability"]["unfiltered"], 100.0)
        self.assertEqual(out["availability"]["strict"], 50.0)  # a and d
        self.assertEqual(out["availability"]["tae_r"], 75.0)   # a, b, c(unfiltered); d's S1 misses topic 5
        self.assertEqual(out["allgold@5"]["tae_r"], 50.0)
        self.assertEqual(out["tae_r_routes"]["S1"]["route_adequate"], 50.0)
        self.assertEqual(out["tae_r_routes"]["unfiltered"]["S3_would_have_been_adequate"], 0.0)
        self.assertEqual(out["oracle_gap"]["questions"], 1)
        self.assertEqual(out["oracle_gap"]["tae_r_route"], {"unfiltered": 1})
        a = out["anatomy"]
        self.assertEqual((a["tae_r_vs_strict"]["helped"], a["tae_r_vs_strict"]["hurt"]), (1, 1))
        self.assertEqual((a["tae_r_vs_unfiltered"]["helped"], a["tae_r_vs_unfiltered"]["hurt"]), (0, 1))
        self.assertEqual(a["tae_r_vs_unfiltered"]["hurt_route_inadequate"], 1)
        self.assertEqual(out["by_type"]["comparison"]["allgold@5"]["tae_r"], 100.0)
        self.assertEqual(out["gold_topics_per_question"], {"1": 3, "2": 1})

    def test_e2e_conditioning_and_errors(self):
        def rec(system, qid, em, first, err=None, it=5):
            return {"system": system, "question_id": qid, "exact_match": em, "f1": em, "first_allgold@5": first,
                    "first_hits": first * 2, "n_gold": 2, "error": err, "iterations": it, "retrievals": [{"route": "S1"}]}
        records = [rec("original", "a", 0, 0), rec("tae_r", "a", 1, 1), rec("nofilter", "a", 1, 1),
                   rec("original", "b", 1, 1, it=1), rec("tae_r", "b", 1, 1), rec("nofilter", "b", 0, 1, "OutputParserError: x")]
        out = ana.e2e_analysis(records, self.stats, types={"a": "comparison", "b": "inference"})
        self.assertEqual(out["questions"], 2)
        self.assertEqual(out["systems"]["original"]["em_given_first_allgold=1"], 100.0)
        self.assertEqual(out["systems"]["nofilter"]["errors"]["kinds"], {"OutputParserError": 1})
        self.assertEqual(out["tae_r_vs_original_em"]["helped_by_first_retrieval"], {"tae_r_first_retrieval_better": 1})
        self.assertEqual(out["by_type"]["comparison"]["tae_r"], 100.0)

    def test_full_run_writes_tables_and_figures(self):
        methods = {m: {"questions": 4, "allgold@5": 0.5, "recall@5": 0.6, "allgold@10": 0.7, "mean_searched_share": 0.5,
                       "mean_similarity_computations": 100.0} for m in ana.ORDER}
        comps = {"tae_r_vs_strict": {"difference": 0.0, "ci95": [-0.5, 0.5], "helps": 1, "hurts": 1, "exact_p": 1.0}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "final_retrieval_H2.json").write_text(json.dumps({"methods": methods, "comparisons": comps}))
            (tmp / "final_retrieval_H2.questions.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
            raw = tmp / "dev.json"
            raw.write_text(json.dumps([{"_id": q, "type": "comparison"} for q in "abcd"]))
            ana.run(tmp, ["H2"], None, None, raw, tmp, log=lambda *_: None)
            tex = (tmp / "tables.tex").read_text()
            self.assertIn(r"\textbf{TAE-R}", tex)
            self.assertIn("Comparison", tex)
            self.assertTrue((tmp / "cost_accuracy_H2.pdf").is_file() and (tmp / "by_type_H2.png").is_file())


if __name__ == "__main__":
    unittest.main()
