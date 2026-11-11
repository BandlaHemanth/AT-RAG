import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


router = load("tae_r_router", ROOT / "scripts" / "tae_r_router.py")
fixture = load("fit_fixture_router", ROOT / "tests" / "test_fit_tae_r_topic_models.py")


class UnitTests(unittest.TestCase):
    def test_topic_mass_and_entropy(self):
        ranked = router.topic_mass(np.array([3, 1, 3, -1]), np.array([0.9, 0.8, 0.5, -0.2]))
        self.assertEqual([t for t, _ in ranked], [3, 1, -1])
        self.assertAlmostEqual(ranked[0][1], 1.4)
        self.assertAlmostEqual(router.softmax_entropy(np.array([1.0, 1.0, 1.0])), 1.0)
        self.assertLess(router.softmax_entropy(np.array([1.0, 0.0, 0.0])), 0.01)

    def test_v2_inference_renormalises_rounding(self):
        inf = router.v2_inference(2, 0.7, np.array([0.2, 0.3, 0.5000004], dtype=np.float32))
        self.assertLessEqual(inf["membership_sum"], 1.0 + 1e-12)
        decision = router.policy.decide(inf, policy="tae_v2")
        self.assertNotEqual(decision["reason"], "invalid_membership_mass")

    def test_gate2_rules(self):
        self.assertEqual(router.gate2({"ci95": [0.001, 0.03]}), "better")
        self.assertEqual(router.gate2({"ci95": [-0.009, 0.008]}), "equivalent")
        self.assertEqual(router.gate2({"ci95": [-0.02, 0.005]}), "worse_or_inconclusive")
        m = router.mcnemar([1, 1, 0, 0, 1], [0, 1, 1, 0, 0])
        self.assertEqual((m["helps"], m["hurts"], m["ties"]), (2, 1, 2))

    def test_retriever_outcome_brute_force(self):
        rng = np.random.RandomState(3)
        docs = rng.normal(size=(30, 4)).astype(np.float32)
        labels = np.arange(30) % 4
        r = router.Retriever(docs, labels)
        q = rng.normal(size=4).astype(np.float32)
        _, ranking = r.rank(q)
        gold = np.array([1, 5])
        out = r.outcome(ranking, gold, [1])
        top = [i for i in np.argsort(-(docs @ q), kind="stable") if labels[i] == 1][:5]
        self.assertEqual(out["hits@5"], int(np.isin(gold, top).sum()))
        self.assertEqual(out["searched"], int((labels == 1).sum()))
        self.assertEqual(r.outcome(ranking, gold, None)["searched"], 30)


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FitTests.setUpClass()
        cls.f = fixture.FitTests

    @classmethod
    def tearDownClass(cls):
        fixture.FitTests.tearDownClass()

    def run_candidate(self, name):
        return router.run(self.f.corpora, self.f.emb, self.f.out, name, frozen=self.f.frozen, log=lambda m: None)

    def test_hdbscan_and_kmeans_candidates(self):
        for name in ("H1", "K1"):
            report, rows, choices = self.run_candidate(name)
            val = [r for r in rows if r["split"] == "validation"]
            self.assertEqual(len(val), 12)
            for r in val:
                for route, out in r["outcomes"].items():
                    for k in (5, 10):
                        self.assertGreaterEqual(r["outcomes"]["oracle"][f"all@{k}"], out[f"all@{k}"], (name, route))
            for method, summary in report["methods"].items():
                self.assertEqual(sum(summary["route_counts"].values()), 12)
                self.assertLessEqual(summary["mean_searched_share"], 1.0)
            self.assertEqual(report["methods"]["unfiltered"]["mean_searched_share"], 1.0)
            self.assertIn("tae_r", report["gate2"])
            low, high = report["gate2"]["tae_r"]["ci95"]
            self.assertLessEqual(low, high)
            if name == "H1":
                self.assertIn("tae_v2", report["methods"])
            else:
                self.assertNotIn("tae_v2", report["methods"])
            json.dumps(report)

    def test_deterministic(self):
        a, b = self.run_candidate("K1")[0], self.run_candidate("K1")[0]
        self.assertEqual(a["per_question_sha256"], b["per_question_sha256"])
        self.assertEqual(a["thresholds"], b["thresholds"])


if __name__ == "__main__":
    unittest.main()
