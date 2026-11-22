import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hop = load("tae_r_hop", ROOT / "scripts" / "tae_r_hop.py")
sys.path.insert(0, str(ROOT / "tests"))
reptest = load("test_run_replication_for_hop", ROOT / "tests" / "test_run_replication.py")


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def two_hop_world(seed=0, n=200, dim=128):
    """Question ~ paragraph A; gold B is near A but far from the question (an unnamed second hop)."""
    rng = np.random.RandomState(seed)
    docs = np.stack([unit(rng.normal(size=dim)) for _ in range(n)])
    q = unit(rng.normal(size=dim))
    r = rng.normal(size=dim)
    r = unit(r - (r @ q) * q)                                        # a direction the question does not mention
    docs[0] = unit(q + 1.0 * r)                                      # A: first hop (question + bridge entity)
    docs[1] = unit(r + 0.1 * unit(rng.normal(size=dim)))             # B: second hop, about the bridge entity only
    for i in range(2, 5):                                            # distractors similar to the question
        docs[i] = unit(q + 1.2 * unit(rng.normal(size=dim)))
    labels = np.array([i % 4 for i in range(n)])
    centroid_ids = np.arange(4)
    centroids = np.stack([unit(docs[labels == t].mean(0)) for t in centroid_ids])
    ctx = {"centroid_ids": centroid_ids, "centroids": centroids, "hdbscan": False}
    model = {"features": list(hop.router.FULL_FEATURES), "single": "S1", "multi": "S3",
             "model_single": 0.0, "model_multi": 0.0, "t1": 0.5, "t3": 0.5}   # never filters: isolates the hop logic
    return q, docs, labels, ctx, model


class HopTests(unittest.TestCase):
    def test_second_hop_reaches_unnamed_paragraph(self):
        q, docs, labels, ctx, model = two_hop_world()
        single = hop.stable_order(docs @ q)[:5]
        self.assertIn(0, single)
        self.assertNotIn(1, single)
        top, cost, _ = hop.hop_topk(q, 0, 1.0, docs, labels, ctx, model, {"lam": 1.0, "beams": 1, "keep": 3}, "none")
        self.assertEqual(len(top), 5)
        self.assertEqual(len(set(top.tolist())), 5)
        self.assertEqual(top[0], single[0])
        self.assertIn(1, top)
        self.assertEqual(cost, 2 * len(labels) + len(ctx["centroid_ids"]))

    def test_topic_arm_costs_centroids_and_keeps_hop1_prefix(self):
        q, docs, labels, ctx, model = two_hop_world(1)
        cfg = {"lam": 0.5, "beams": 2, "keep": 2}
        top_t, cost_t, h1 = hop.hop_topk(q, 0, 1.0, docs, labels, ctx, model, cfg, "topics")
        top_n, cost_n, _ = hop.hop_topk(q, 0, 1.0, docs, labels, ctx, model, cfg, "none")
        self.assertEqual(top_t[:2].tolist(), h1[:2].tolist())
        self.assertEqual(cost_t - cost_n, 2 * len(ctx["centroid_ids"]))

    def test_choose_and_parse(self):
        s = {"topics:lam=0.5,B=1,a=2": {"allgold@5": 0.2, "recall@5": 0.5, "mean_similarity_computations": 10},
             "topics:lam=1.0,B=2,a=3": {"allgold@5": 0.2, "recall@5": 0.5, "mean_similarity_computations": 9},
             "none:lam=0.5,B=1,a=2": {"allgold@5": 0.1, "recall@5": 0.5, "mean_similarity_computations": 1}}
        self.assertEqual(hop.choose(s, "topics"), "topics:lam=1.0,B=2,a=3")
        self.assertEqual(hop.parse_key("none:lam=0.5,B=2,a=3"), {"lam": 0.5, "beams": 2, "keep": 3})
        self.assertEqual(hop.hop_counts("3hop1__1_2"), "3-hop")

    def test_select_then_final_on_small_replication(self):
        rep = reptest.rep
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            data.mkdir()
            reptest.write(data / "train.jsonl", [reptest.question(f"tr{i}", i, "train") for i in range(160)])
            dev = [reptest.question(f"dev{i}", i, "dev") for i in range(70)]
            reptest.write(data / "dev.jsonl", dev)
            reptest.write(data / "dev_subsampled.jsonl", dev[:3])
            reptest.write(data / "test_subsampled.jsonl", dev[3:6])
            old = rep.E2E_SIZES
            rep.E2E_SIZES = (10, 20)
            rep.embed_module().runtime_environment = lambda device: {"test": True}
            try:
                rep.run("toy", data, tmp / "work", rep.STAGES, device="cpu", allow_final=True, log=lambda *_: None,
                        config=reptest.SMALL, encode=reptest.fake_encode, encoder_factory=lambda d: reptest.FakeEncoder())
            finally:
                rep.E2E_SIZES = old
            args = type("A", (), {"replication_work_dir": tmp / "work"})()
            paths, frozen = hop.resolve_paths(args)
            report = hop.run_select(paths, tmp / "hop", frozen, log=lambda *_: None)
            self.assertEqual(set(report["selection"]["chosen"]), {"topics", "none"})
            self.assertEqual(len(report["summary"]), 3 + 2 * len(hop.GRID))
            with self.assertRaises(ValueError):
                hop.run_final(paths, tmp / "hop" / "selection.json", tmp / "hopfinal", frozen, log=lambda *_: None)
            final = hop.run_final(paths, tmp / "hop" / "selection.json", tmp / "hopfinal", frozen, log=lambda *_: None,
                                  allow_final=True, dataset="toy")
            self.assertEqual(final["methods"]["unfiltered"]["questions"], 64)
            self.assertIn("tae_r_h_topics_vs_tae_r_h_none", final["comparisons"])
            # hop-1 baselines inside the hop script agree with the frozen replication final run
            frozen_final = json.loads((tmp / "work/final/final_retrieval_H2.json").read_text())
            for m in ("strict", "unfiltered", "tae_r"):
                self.assertAlmostEqual(final["methods"][m]["allgold@5"], frozen_final["methods"][m]["allgold@5"], msg=m)


if __name__ == "__main__":
    unittest.main()
