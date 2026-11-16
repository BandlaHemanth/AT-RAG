import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_replication", ROOT / "scripts" / "run_replication.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

CENTERS = 6
BASIS = np.random.RandomState(0).normal(size=(CENTERS, 384))


def vec(text):
    """Texts look like 'blob<c> ...'; vectors cluster by c, with text-seeded noise."""
    c = int(text.split()[0][4:])
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = BASIS[c] + np.random.RandomState(seed).normal(scale=0.15, size=384)
    return (v / np.linalg.norm(v)).astype(np.float32)


def fake_encode(texts):
    return np.stack([vec(t) for t in texts])


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return fake_encode(texts)


def question(qid, i, prefix):
    c = i % CENTERS
    contexts = []
    for j in range(8):
        cc = c if j < 4 else (c + j) % CENTERS
        contexts.append({"title": f"{prefix} title {i} {j}", "paragraph_text": f"blob{cc} {prefix} para {i} {j}",
                         "is_supporting": j in (0, 1) or (i % 2 == 1 and j == 5)})
    return {"question_id": qid, "question_text": f"blob{c} {prefix} question {i}",
            "answers_objects": [{"spans": ["x"]}], "contexts": contexts}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


SMALL = copy.deepcopy(rep.fit.CONFIG)
SMALL["umap"].update(n_neighbors=10)
SMALL["candidates"] = {"H2": {"family": "hdbscan", "min_cluster_size": 20, "min_samples": 2}}


class ReplicationTests(unittest.TestCase):
    def test_fold_sizes_keep_pool_parity_and_ratio(self):
        self.assertEqual(rep.fold_sizes(12576), (8000, 4576))
        self.assertEqual(rep.fold_sizes(7405), (4711, 2694))
        self.assertEqual(rep.fold_sizes(2417), (1538, 879))

    def test_end_to_end_small_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            data.mkdir()
            write(data / "train.jsonl", [question(f"tr{i}", i, "train") for i in range(160)])
            dev = [question(f"dev{i}", i, "dev") for i in range(70)]
            write(data / "dev.jsonl", dev)
            write(data / "dev_subsampled.jsonl", dev[:3])
            write(data / "test_subsampled.jsonl", dev[3:6])
            old = rep.E2E_SIZES
            rep.E2E_SIZES = (10, 20)
            rep.embed_module().runtime_environment = lambda device: {"test": True}
            try:
                logs = []
                results = rep.run("toy", data, tmp / "work", rep.STAGES, device="cpu", allow_final=True,
                                  log=logs.append, config=SMALL, encode=fake_encode,
                                  encoder_factory=lambda device: FakeEncoder())
            finally:
                rep.E2E_SIZES = old
            self.assertEqual(results["splits"]["router_train"] + results["splits"]["validation"], 70)
            self.assertEqual(results["splits"]["split_files"]["final_eval"]["count"], 64)
            self.assertIn("tae_r", results["validate"]["allgold@5"])
            self.assertIn("tae_r_vs_strict", results["final"]["comparisons"])
            final = json.loads((tmp / "work/final/final_retrieval_H2.json").read_text())
            self.assertEqual(final["methods"]["unfiltered"]["questions"], 64)
            # rerun: every stage is skipped, nothing is overwritten
            again = rep.run("toy", data, tmp / "work", rep.STAGES, device="cpu", allow_final=True,
                            log=logs.append, config=SMALL, encode=fake_encode,
                            encoder_factory=lambda device: FakeEncoder())
            self.assertEqual(again["final"]["report_sha256"], results["final"]["report_sha256"])
            self.assertTrue(any("already complete" in m for m in logs))

    def test_final_needs_allow_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rep.stage_final(rep.Work(tmp), print, "cpu", allow_final=False)

    def test_cap_keeps_seeded_subset(self):
        corpora = rep.corpora_mod
        self.assertIn("fit_cap", corpora.build.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
