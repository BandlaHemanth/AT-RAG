import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ctl = load("controlled_at_rag", ROOT / "scripts" / "controlled_at_rag.py")
smoke = load("smoke_for_ctl", ROOT / "scripts" / "smoke_test_original_at_rag.py")


def langchain_available():
    extra = os.environ.get("LANGCHAIN_03_PATH")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        from langchain.output_parsers import ResponseSchema, StructuredOutputParser  # noqa: F401
        from langchain_core.prompts import PromptTemplate  # noqa: F401
        return True
    except Exception:
        return False


class PromptTests(unittest.TestCase):
    def test_extracts_all_released_templates(self):
        prompts = ctl.released_prompts()
        self.assertEqual(set(prompts), set(ctl.RELEASED_METHODS))
        self.assertIn("{generation}", prompts["hallucination_grader"]["template"])
        self.assertEqual([s[0] for s in prompts["rag_chain"]["schemas"]], ["answer"])
        self.assertEqual([s[0] for s in prompts["cot"]["schemas"]], ["thoughts"])
        self.assertEqual(prompts["question_rewriter"]["schemas"], [])

    def test_clarifications_identical_to_sanity_check_and_inserted_once(self):
        self.assertEqual(ctl.CLARIFICATIONS, smoke.CLARIFICATIONS)
        released, controlled = ctl.released_prompts(), ctl.controlled_prompts()
        for name, sentence in ctl.CLARIFICATIONS.items():
            self.assertEqual(controlled[name]["template"].replace(sentence + " ", "", 1), released[name]["template"])
        for name in ("rag_chain", "cot", "question_rewriter"):
            self.assertEqual(controlled[name], released[name])

    @unittest.skipUnless(langchain_available(), "LangChain 0.3 not importable (set LANGCHAIN_03_PATH)")
    def test_rendering_and_parsing_match_langchain(self):
        from langchain.output_parsers import ResponseSchema, StructuredOutputParser
        from langchain_core.prompts import PromptTemplate
        values = {"hallucination_grader": dict(documents="Doc A\n\nDoc B", generation="Paris"),
                  "answer_grader": dict(question="Capital?", generation="Paris"),
                  "question_rewriter": dict(question="Capital?", context="Doc A"),
                  "rag_chain": dict(question="T--Capital?", context="Doc A"),
                  "cot": dict(question="Capital?", context="Doc A")}
        for name, prompt in ctl.released_prompts().items():
            partial = {}
            if prompt["schemas"]:
                parser = StructuredOutputParser.from_response_schemas(
                    [ResponseSchema(name=n, type=t, description=d) for n, t, d in prompt["schemas"]])
                partial = {"format_instructions": parser.get_format_instructions()}
                self.assertEqual(ctl.format_instructions(prompt["schemas"]), partial["format_instructions"])
            lc = PromptTemplate(template=prompt["template"], input_variables=list(values[name]), partial_variables=partial)
            self.assertEqual(ctl.render(prompt, **values[name]), lc.format(**values[name]), name)
        parser = StructuredOutputParser.from_response_schemas([ResponseSchema(name="score", description="d")])
        for text in ('```json\n{"score": "yes"}\n```', 'Sure! ```json\n{\n "score": "no"\n}\n``` done',
                     '{"score": "yes"}', '{"score": "yes"', 'no json here', '{"other": 1}'):
            try:
                expected = parser.parse(text)
            except Exception as error:
                expected = type(error).__name__
            try:
                got = ctl.parse_structured(text, ["score"])
            except Exception as error:
                got = type(error).__name__
            if isinstance(expected, str):
                self.assertIsInstance(got, str, text)
            else:
                self.assertEqual(got, expected, text)


class FakeEngine:
    texts = ["doc zero", "doc one", "doc two"]

    def __init__(self):
        self.queries = []

    def retrieve(self, system, text):
        self.queries.append(text)
        return [0, 1], {"system": system, "similarity_computations": 3, "doc_ids": ["d0", "d1"]}


def scripted_llm(hallucination, answer=("yes",), bad=None):
    state = {"h": list(hallucination), "a": list(answer), "cot": 0, "prompts": []}

    def llm(prompt):
        state["prompts"].append(prompt)
        if bad and bad in prompt:
            return "not json at all", False
        if "grounded in a set of facts" in prompt:
            return '```json\n{"score": "%s"}\n```' % (state["h"].pop(0) if len(state["h"]) > 1 else state["h"][0]), False
        if "useful to resolve a question" in prompt:
            return '```json\n{"score": "%s"}\n```' % (state["a"].pop(0) if len(state["a"]) > 1 else state["a"][0]), False
        if "question re-writer" in prompt:
            return "rewritten question", False
        if "chain of thought" in prompt:
            state["cot"] += 1
            return '```json\n{"thoughts": "T%d"}\n```' % state["cot"], False
        return '```json\n{"answer": "Paris"}\n```', False
    return llm, state


class GraphTests(unittest.TestCase):
    prompts = ctl.controlled_prompts()

    def test_grader_always_no_runs_to_cap(self):
        llm, _ = scripted_llm(["no"])
        out = ctl.run_question("nofilter", "Q?", FakeEngine(), llm, self.prompts)
        self.assertEqual((out["iterations"], out["llm_calls"], len(out["retrievals"])), (5, 11, 1))
        self.assertEqual(out["answer"], "Paris")
        self.assertIsNone(out["error"])

    def test_rewrite_path_prepends_thoughts_and_reretrieves(self):
        llm, state = scripted_llm(["yes"], answer=["no", "yes"])
        engine = FakeEngine()
        out = ctl.run_question("original", "Q?", engine, llm, self.prompts)
        self.assertEqual((out["iterations"], out["llm_calls"], len(out["retrievals"])), (3, 9, 2))
        self.assertEqual(engine.queries, ["Q?", "rewritten question"])
        rag_prompts = [p for p in state["prompts"] if "question answering angent" in p]
        self.assertIn("T2T1--rewritten question", rag_prompts[-1])
        self.assertIn("T1--Q?", rag_prompts[0])

    def test_parse_failure_is_recorded_not_inherited(self):
        llm, _ = scripted_llm(["no"], bad="chain of thought")
        out = ctl.run_question("tae_r", "Q?", FakeEngine(), llm, self.prompts)
        self.assertIsNone(out["answer"])
        self.assertIn("OutputParserError", out["error"])


class ClientTests(unittest.TestCase):
    def test_cache_hits_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            sent = []
            transport = lambda payload: (sent.append(payload), "reply")[1]
            client = ctl.LLMClient("http://x/v1", "m", Path(tmp) / "cache.jsonl", digest="d", transport=transport)
            self.assertEqual(client("hello"), ("reply", False))
            self.assertEqual(client("hello"), ("reply", True))
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0]["temperature"], 0)
            self.assertEqual(sent[0]["seed"], 42)
            self.assertEqual(sent[0]["messages"], [{"role": "user", "content": "hello"}])
            again = ctl.LLMClient("http://x/v1", "m", Path(tmp) / "cache.jsonl", digest="d", transport=transport)
            self.assertEqual(again("hello"), ("reply", True))


class EngineTests(unittest.TestCase):
    def test_routing_and_costs(self):
        router = load("router_for_ctl", ROOT / "scripts" / "tae_r_router.py")
        rng = np.random.RandomState(0)
        docs = rng.normal(size=(40, 8)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        labels = np.array([-1, 0, 1, 2] * 10)
        centroid_ids = np.array([-1, 0, 1, 2])
        centroids = np.stack([docs[labels == t].mean(0) for t in centroid_ids])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        q = docs[5]
        model = {"features": list(router.FULL_FEATURES), "single": "S1", "multi": "S3",
                 "model_single": 1.0, "model_multi": 0.0, "t1": 0.5, "t3": 0.5}
        engine = ctl.Engine([f"t{i}" for i in range(40)], [f"d{i}" for i in range(40)], docs, labels,
                            centroid_ids, centroids, model, {"Q": (q, 1, 0.9)}, None, None, router)
        top, info = engine.retrieve("original", "Q")
        self.assertTrue(all(labels[i] == 1 for i in top))
        self.assertEqual(info["similarity_computations"], 10)
        top, info = engine.retrieve("nofilter", "Q")
        self.assertEqual(top[0], 5)
        self.assertEqual(info["similarity_computations"], 40)
        top, info = engine.retrieve("tae_r", "Q")
        self.assertEqual(info["route"], "S1")
        self.assertEqual(len(set(labels[top].tolist())), 1)
        self.assertEqual(info["similarity_computations"], 44)


if __name__ == "__main__":
    unittest.main()
