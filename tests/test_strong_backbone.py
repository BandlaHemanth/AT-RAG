import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strong_backbone", ROOT / "scripts" / "strong_backbone.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

DIM = 256


def bow(text, salt):
    v = np.zeros(DIM, dtype=np.float32)
    for w in text.lower().replace(".", " ").replace(":", " ").split():
        v[int(hashlib.sha256((salt + w).encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEncoder:
    def __init__(self, name):
        self.salt = name

    def encode(self, texts, **kw):
        return np.stack([bow(t, self.salt) for t in texts])


class FakeReranker:
    """Scores by shared rare words: the 'smart' model the dense step lacks."""
    def __init__(self, name):
        self.name = name

    def predict(self, pairs, **kw):
        return [len({w for w in q.lower().split() if w.startswith("ent")} & set(p.lower().split())) +
                (0.01 if "bge" in self.name else 0.0) for q, p in pairs]


def world(tmp):
    rng = np.random.RandomState(0)
    filler = [f"w{i}" for i in range(40)]
    docs, questions, gold = [], [], {}
    for qi in range(30):
        g1 = {"text": f"ent{qi}a " + " ".join(rng.choice(filler, 6)), "titles": [f"Page {qi}a"]}
        g2 = {"text": f"ent{qi}b " + " ".join(rng.choice(filler, 6)), "titles": [f"Page {qi}b"]}
        docs += [g1, g2]
        questions.append({"question_id": f"{(qi % 3) + 2}hop__{qi}", "question_text": f"about ent{qi}a and ent{qi}b " +
                          " ".join(rng.choice(filler, 4)), "gold_texts": [g1["text"], g2["text"]]})
    for i in range(140):
        docs.append({"text": " ".join(rng.choice(filler, 8)), "titles": [f"Filler {i}"]})
    for d in docs:
        d["doc_id"] = hashlib.sha256(d["text"].encode()).hexdigest()
    by_text = {d["text"]: d["doc_id"] for d in docs}
    for q in questions:
        q["gold_doc_ids"] = [by_text[t] for t in q.pop("gold_texts")]
    for name in ("retrieval_validation", "retrieval_final"):
        (tmp / f"{name}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    for name in ("questions_validation", "questions_final_eval"):
        (tmp / f"{name}.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    labels = np.array([i % 5 for i in range(len(docs))])
    np.save(tmp / "labels.npy", labels)
    index = {d["doc_id"]: i for i, d in enumerate(docs)}
    rows = []
    for q in questions:
        gt = sorted({int(labels[index[g]]) for g in q["gold_doc_ids"]})
        first = int(labels[index[q["gold_doc_ids"][0]]])
        rows.append({"question_id": q["question_id"],
                     "routes": {"unfiltered": None, "strict": [first], "oracle": gt, "S3": [0, 1, 2], "P1_S3": None},
                     "chosen": {"unfiltered": "unfiltered", "strict": "strict", "oracle": "oracle", "tae_r": "S3",
                                "tae_r_p1": "P1_S3"}})
    (tmp / "frozen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return docs, questions, labels


class BackboneTests(unittest.TestCase):
    def models(self, cache=None):
        return sb.Models("cpu", encoder_factory=FakeEncoder, reranker_factory=FakeReranker, cache_dir=cache)

    def test_dense_top_ties_and_allowed(self):
        s = np.array([0.5, 0.9, 0.9, 0.1, 0.9], dtype=np.float32)
        self.assertEqual(sb.dense_top(s, None, 2).tolist(), [1, 2])
        self.assertEqual(sb.dense_top(s, np.array([0, 3, 4]), 2).tolist(), [4, 0])
        self.assertEqual(len(sb.dense_top(s, np.array([], dtype=np.int64), 3)), 0)

    def test_passage_uses_titles(self):
        self.assertEqual(sb.passage({"text": "body", "titles": ["A", "B"]}), "A / B. body")
        self.assertEqual(sb.passage({"text": "body", "titles": []}), "body")

    def test_select_prefers_reranker_and_final_reports_every_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            docs, questions, labels = world(tmp)
            selection, summary = sb.run_select(tmp, tmp / "sel", self.models(tmp / "cache"), log=lambda *_: None)
            self.assertEqual(len(summary), len(sb.ENCODERS) * (1 + 2 * len(sb.DEPTHS)))
            self.assertNotEqual(selection["reranker"], "none")
            best = summary[selection["key"]]["allgold@5"]
            self.assertTrue(all(best >= v["allgold@5"] for v in summary.values()))
            with self.assertRaises(ValueError):
                sb.run_final(tmp, tmp / "labels.npy", tmp / "frozen.jsonl", tmp / "sel/backbone_selection.json",
                             tmp / "fin", self.models(), dataset="toy")
            report = sb.run_final(tmp, tmp / "labels.npy", tmp / "frozen.jsonl", tmp / "sel/backbone_selection.json",
                                  tmp / "fin", self.models(tmp / "cache"), dataset="toy", allow_final=True, log=lambda *_: None)
            m = report["methods"]
            # the reranker only reorders the dense top-R, so even the oracle route can miss a gold paragraph
            self.assertGreaterEqual(m["oracle|rerank"]["allgold@5"], m["strict|rerank"]["allgold@5"])
            self.assertGreaterEqual(m["oracle|rerank"]["allgold@5"], m["oracle|dense"]["allgold@5"])
            self.assertGreaterEqual(m["unfiltered|rerank"]["allgold@5"], m["unfiltered|dense"]["allgold@5"])
            self.assertLessEqual(m["strict|rerank"]["allgold@5"], m["unfiltered|rerank"]["allgold@5"])
            self.assertIn("tae_r_vs_strict|rerank", report["comparisons"])
            self.assertEqual(set(report["by_type"]), {"2-hop", "3-hop", "4-hop"})

    def test_rerank_cache_scores_each_pair_once(self):
        calls = []

        class Counting(FakeReranker):
            def predict(self, pairs, **kw):
                calls.append(len(pairs))
                return super().predict(pairs)
        models = sb.Models("cpu", encoder_factory=FakeEncoder, reranker_factory=Counting)
        cache = sb.RerankCache(models, "rr", ["ent1 x", "y", "ent1 z"])
        cache.ensure(0, "q ent1", [0, 1])
        cache.ensure(0, "q ent1", [1, 2])
        self.assertEqual(calls, [2, 1])
        self.assertEqual(cache.order(0, [1, 2, 0]), [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
