import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fit_topics", ROOT / "scripts" / "fit_tae_r_topic_models.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)

CENTERS = 6
RNG = np.random.RandomState(0)
BASIS = RNG.normal(size=(CENTERS, 384))


def vector(center, seed):
    noise = np.random.RandomState(seed).normal(scale=0.15, size=384)
    v = BASIS[center] + noise
    return v / np.linalg.norm(v)


def fake_encode(texts):
    return np.stack([vector(int(t.split("-")[1]), int(t.split("-")[2]) + 10_000) for t in texts])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SMALL = copy.deepcopy(fit.CONFIG)
SMALL["umap"].update(n_neighbors=10)
SMALL["candidates"] = {
    "H1": {"family": "hdbscan", "min_cluster_size": 20, "min_samples": 10},
    "H2": {"family": "hdbscan", "min_cluster_size": 20, "min_samples": 2},
    "K1": {"family": "kmeans", "k": 6},
    "K2": {"family": "kmeans", "k": 3},
}


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.corpora, cls.emb = root / "corpora", root / "emb"
        cls.corpora.mkdir()
        cls.emb.mkdir()
        docs = {}
        for name, per_center, offset in (("topic_fit_population", 60, 0), ("retrieval_validation", 15, 5000)):
            records = []
            for c in range(CENTERS):
                for i in range(per_center):
                    text = f"{name}-{c}-{i}"
                    doc = hashlib.sha256(text.encode()).hexdigest()
                    docs[doc] = vector(c, offset + c * 1000 + i)
                    records.append({"doc_id": doc, "titles": ["t"], "text": text})
            records.sort(key=lambda r: r["doc_id"])
            cls.write(name + ".jsonl", records)
        for split, n in (("router_train", 24), ("validation", 12)):
            cls.write(f"questions_{split}.jsonl",
                      [{"question_id": f"{split}{i}", "question_text": f"q-{i % CENTERS}-{i}"} for i in range(n)])
        (cls.corpora / "retrieval_final.jsonl").write_text("not json: must never be read\n")
        (cls.corpora / "questions_final_eval.jsonl").write_text("not json: must never be read\n")
        files = {p.stem: {"file": p.name, "sha256": sha(p)} for p in cls.corpora.glob("*.jsonl")}
        (cls.corpora / "manifest.json").write_text(json.dumps({"files": files}))
        ids = sorted(docs)
        np.save(cls.emb / "embeddings.npy", np.stack([docs[d] for d in ids]).astype(np.float32))
        (cls.emb / "document_ids.jsonl").write_text(
            "".join(json.dumps({"document_id": d, "row": i}) + "\n" for i, d in enumerate(ids)))
        cls.frozen = {"corpus_manifest": sha(cls.corpora / "manifest.json"),
                      "embeddings.npy": sha(cls.emb / "embeddings.npy"),
                      "document_ids.jsonl": sha(cls.emb / "document_ids.jsonl")}
        cls.out = root / "out"
        cls.messages = cls.run_all(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def write(cls, name, records):
        (cls.corpora / name).write_text("".join(json.dumps(r) + "\n" for r in records))

    @classmethod
    def run_all(cls, out, stages=fit.STAGES, names=None):
        messages = []
        fit.run(cls.corpora, cls.emb, out, list(stages), names or list(SMALL["candidates"]),
                frozen=cls.frozen, config=SMALL, encode=fake_encode, log=messages.append)
        return messages

    def stage(self, name, out=None):
        return fit.Stage(out or self.out, name)

    def test_all_stages_complete_with_expected_shapes(self):
        for name in ["umap", "questions"] + [f"{c}/{s}" for c in SMALL["candidates"] for s in ("fit", "assign_validation")]:
            self.assertTrue(self.stage(name).complete(), name)
        self.assertEqual(self.stage("umap").load("fit_5d.npy").shape, (360, 5))
        self.assertEqual(self.stage("umap").load("validation_corpus_5d.npy").shape, (90, 5))
        q = self.stage("questions")
        self.assertEqual(q.load("question_embeddings.npy").shape, (36, 384))
        self.assertEqual(json.loads((q.dir / "question_ids.json").read_text())[0], ["router_train", "router_train0"])
        h = self.stage("H1/assign_validation")
        topics = len(self.stage("H1/fit").load("centroid_topic_ids.npy"))
        self.assertGreaterEqual(topics, 2)
        self.assertEqual(h.load("question_membership.npy").shape, (36, topics))
        self.assertEqual(self.stage("K2/assign_validation").load("question_centroid_scores.npy").shape, (36, 3))
        for name in SMALL["candidates"]:
            c = self.stage(name + "/fit").load("centroids_384.npy")
            np.testing.assert_allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-5)

    def test_kmeans_recovers_blobs_and_labels_agree(self):
        labels = self.stage("K1/fit").load("fit_labels.npy")
        truth = np.repeat(np.arange(CENTERS), 60)
        order = np.argsort([hashlib.sha256(f"topic_fit_population-{c}-{i}".encode()).hexdigest()
                            for c in range(CENTERS) for i in range(60)])
        purity = sum(np.bincount(labels[truth[order] == c]).max() for c in range(CENTERS)) / 360
        self.assertEqual(purity, 1.0)
        q_labels = self.stage("K1/assign_validation").load("question_labels.npy")
        scores = self.stage("K1/assign_validation").load("question_centroid_scores.npy")
        np.testing.assert_array_equal(q_labels, scores.argmax(axis=1))

    def test_resume_skips_and_detects_tampering(self):
        before = json.loads((self.out / "H2/fit/stage_manifest.json").read_text())["files"]
        messages = self.run_all(self.out)
        self.assertTrue(all("skipped" in m for m in messages if m.startswith(("umap", "questions", "cluster", "assign"))))
        labels = self.out / "H2/fit/fit_labels.npy"
        labels.write_bytes(labels.read_bytes() + b"x")
        self.assertFalse(self.stage("H2/fit").complete())
        messages = self.run_all(self.out, ["cluster"], ["H2"])
        self.assertFalse(any("skipped" in m for m in messages if m.startswith("cluster")))
        after = json.loads((self.out / "H2/fit/stage_manifest.json").read_text())["files"]
        self.assertEqual(before["fit_labels.npy"], after["fit_labels.npy"])

    def test_deterministic_across_runs(self):
        other = Path(self.tmp.name) / "again"
        self.run_all(other)
        for name, file in (("umap", "fit_5d.npy"), ("H1/fit", "fit_labels.npy"), ("K2/fit", "centroids_384.npy"),
                           ("H2/assign_validation", "question_labels.npy")):
            self.assertEqual(sha(self.out / name / file), sha(other / name / file), name)

    def test_refusals(self):
        bad = dict(self.frozen, corpus_manifest="0" * 64)
        with self.assertRaises(ValueError):
            fit.run(self.corpora, self.emb, Path(self.tmp.name) / "bad", ["umap"], ["H1"], frozen=bad,
                    config=SMALL, encode=fake_encode, log=lambda m: None)
        with self.assertRaises(ValueError):
            fit.run(self.corpora, self.emb, Path(self.tmp.name) / "early", ["assign"], ["H1"], frozen=self.frozen,
                    config=SMALL, encode=fake_encode, log=lambda m: None)
        with self.assertRaises(ValueError):
            fit.run(self.corpora, self.emb, Path(self.tmp.name) / "x", ["cluster"], ["Z9"], frozen=self.frozen,
                    config=SMALL, encode=fake_encode, log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
