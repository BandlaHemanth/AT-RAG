import importlib.util
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plus = load("tae_r_plus_explore", ROOT / "scripts" / "tae_r_plus_explore.py")
fixture = load("fixture_plus", ROOT / "tests" / "test_fit_tae_r_topic_models.py")


class UnitTests(unittest.TestCase):
    def test_helpers(self):
        labels = np.array([0, 0, 0, 1, 2])
        self.assertEqual(plus.topk_diverse(np.arange(5), labels, 1, k=3).tolist(), [0, 3, 4])
        scores = np.array([0.9, 0.8, 0.1, 0.85, 0.2])
        self.assertEqual(plus.topk_boost(scores, labels, [2], 1.0, k=2).tolist(), [4, 0])
        self.assertEqual(plus.coverage_topics([(5, 6.0), (3, 3.0), (1, 1.0)], 0.6), [5])
        self.assertEqual(plus.coverage_topics([(5, 6.0), (3, 3.0), (1, 1.0)], 0.85), [5, 3])
        self.assertIn(plus.half("abc"), ("tune", "confirm"))


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FitTests.setUpClass()
        cls.f = fixture.FitTests

    @classmethod
    def tearDownClass(cls):
        fixture.FitTests.tearDownClass()

    def test_runs_and_selects_on_tune_only(self):
        for name in ("H1", "K1"):
            report = plus.router.run(self.f.corpora, self.f.emb, self.f.out, name, frozen=self.f.frozen, log=lambda m: None)[0]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "m.joblib"
                joblib.dump(report["router_model"], path)
                out = plus.run(self.f.corpora, self.f.emb, self.f.out, name, path, frozen=self.f.frozen, log=lambda m: None)
            self.assertNotIn(out["chosen"], ("unfiltered", "tae_r"))
            self.assertEqual(set(out["confirm"]), {out["chosen"], "unfiltered", "tae_r"})
            self.assertEqual(out["tune"]["unfiltered"]["mean_searched_share"], 1.0)
            self.assertIn("gbm", out["tune"])
            self.assertEqual(out["tune"]["unfiltered"]["questions"] + out["confirm"]["unfiltered"]["questions"], 12)


if __name__ == "__main__":
    unittest.main()
