import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load("evaluate_topics", ROOT / "scripts" / "evaluate_tae_r_topic_candidates.py")
fixture = load("fit_fixture", ROOT / "tests" / "test_fit_tae_r_topic_models.py")


def metrics(allgold, recall, searched, three=0.9):
    return {"strict": {"allgold@5": allgold, "recall@5": recall, "mean_searched_share": searched},
            "oracle_three_topic_coverage": three}


class UnitTests(unittest.TestCase):
    def test_top_k_masks_and_breaks_ties_by_index(self):
        scores = np.array([0.5, 0.9, 0.9, 0.1, 0.7])
        allowed = np.array([True, True, True, False, True])
        self.assertEqual(ev.top_k(scores, allowed, 3).tolist(), [1, 2, 4])
        self.assertEqual(ev.top_k(scores, np.zeros(5, bool), 3).tolist(), [])
        self.assertEqual(ev.top_k(scores, np.array([0, 0, 0, 1, 0], bool), 3).tolist(), [3])

    def test_selection_rule(self):
        m = {"A": metrics(0.30, 0.50, 0.60), "B": metrics(0.25, 0.40, 0.20), "C": metrics(0.245, 0.45, 0.10)}
        chosen, trace = ev.select(m)
        self.assertEqual(trace[0]["excluded_for_cost"], ["A"])
        self.assertEqual(chosen, "C")  # B and C tie on AllGold@5 (<1 pt); C wins on Recall@5
        m = {"B": metrics(0.25, 0.40, 0.20, 0.8), "C": metrics(0.25, 0.405, 0.10, 0.8)}
        self.assertEqual(ev.select(m)[0], "C")  # all ties -> lower searched share
        self.assertEqual(ev.select({"A": metrics(0.3, 0.5, 0.9)})[0], None)

    def test_evaluate_filters_matches_brute_force(self):
        rng = np.random.RandomState(1)
        docs = rng.normal(size=(40, 8)).astype(np.float32)
        queries = rng.normal(size=(6, 8)).astype(np.float32)
        golds = [np.array([i, i + 10]) for i in range(6)]
        labels = np.arange(40) % 3
        out, _, info = ev.evaluate_filters(queries, docs, golds, {"all": lambda q: np.ones(40, bool),
                                                                  "lab": lambda q: labels == q % 3}, chunk=4)
        hits = 0
        for q in range(6):
            top = np.argsort(-(queries[q] @ docs.T), kind="stable")[:5]
            hits += np.isin(golds[q], top).sum()
        self.assertAlmostEqual(out["all"]["recall@5"], hits / 12)
        self.assertEqual(out["all"]["mean_searched_share"], 1.0)
        self.assertAlmostEqual(out["lab"]["mean_searched_share"], np.mean([(labels == q % 3).mean() for q in range(6)]))
        self.assertEqual(info["gold_paragraphs"], 12)

    def test_oracle_coverage(self):
        c = ev.oracle_coverage([[1, 1], [1, 2], [-1, 3, 4, 5]])
        self.assertAlmostEqual(c["oracle_one_topic_coverage"], 1 / 3)
        self.assertAlmostEqual(c["oracle_three_topic_coverage"], 2 / 3)
        self.assertAlmostEqual(c["gold_outlier_share"], 1 / 8)


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FitTests.setUpClass()
        cls.f = fixture.FitTests

    @classmethod
    def tearDownClass(cls):
        fixture.FitTests.tearDownClass()

    def test_end_to_end_on_synthetic_blobs(self):
        report = ev.run(self.f.corpora, self.f.emb, self.f.out, list(fixture.SMALL["candidates"]),
                        frozen=self.f.frozen, log=lambda m: None)
        k1 = report["candidates"]["K1"]
        self.assertAlmostEqual(k1["cross_topic_rate"], 0.5)  # odd questions have gold in a second blob
        self.assertEqual(k1["oracle"]["eligibility"], 1.0)
        self.assertLessEqual(k1["strict"]["eligibility"], k1["oracle"]["eligibility"])
        self.assertLessEqual(k1["strict"]["mean_searched_share"], 0.5)
        self.assertEqual(report["baseline"]["unfiltered"]["mean_searched_share"], 1.0)
        self.assertEqual(report["baseline"]["info"]["questions"], 12)
        self.assertIn(report["selected"], fixture.SMALL["candidates"])
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
