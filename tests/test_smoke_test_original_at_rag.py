import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke", ROOT / "scripts" / "smoke_test_original_at_rag.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class HelperTests(unittest.TestCase):
    def test_exact_match_and_f1_normalise(self):
        self.assertEqual(smoke.exact_match("The Beatles.", ["beatles"]), 1.0)
        self.assertEqual(smoke.exact_match("Paris", ["London"]), 0.0)
        self.assertAlmostEqual(smoke.f1_score("John Smith Jr", ["John Smith"]), 0.8)
        self.assertEqual(smoke.f1_score("", ["x"]), 0.0)

    def test_stale_flags(self):
        self.assertEqual(smoke.stale_flags(["a", "a", "b", "c"], [None, None, "Err", None]),
                         [False, True, True, False])

    def test_gold_answers(self):
        self.assertEqual(smoke.gold_answers({"answers_objects": [{"spans": ["x", "y"]}]}), ["x", "y"])
        self.assertEqual(smoke.gold_answers({}), [""])


if __name__ == "__main__":
    unittest.main()
