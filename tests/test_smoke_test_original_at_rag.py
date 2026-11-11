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



class ClarifiedTemplateTests(unittest.TestCase):
    def test_sentence_inserted_once_before_json_instruction(self):
        released = "Here is the answer: {generation} \n\n            Please respond in valid JSON format using x \n {format_instructions}"
        out = smoke.clarified_template(released, smoke.CLARIFICATIONS["hallucination_grader"])
        self.assertIn("is not. Please respond in valid JSON format using x", out)
        self.assertEqual(out.replace(smoke.CLARIFICATIONS["hallucination_grader"] + " ", ""), released)
        with self.assertRaises(ValueError):
            smoke.clarified_template("no anchor here", "s")


class RecorderTests(unittest.TestCase):
    def test_recorder_passes_through_and_logs(self):
        class Fake:
            def invoke(self, x):
                return {"score": "Yes"}
            extra = 7
        log = []
        holder = type("P", (), {})()
        for name in ("hallucination_grader", "answer_grader", "question_rewriter", "rag_chain"):
            setattr(holder, name, Fake())
        smoke.install_recorders(holder, log)
        self.assertEqual(holder.answer_grader.invoke({"q": 1}), {"score": "Yes"})
        self.assertEqual(holder.answer_grader.extra, 7)
        self.assertEqual(log, [{"component": "answer_grader", "output": {"score": "Yes"}}])


if __name__ == "__main__":
    unittest.main()
