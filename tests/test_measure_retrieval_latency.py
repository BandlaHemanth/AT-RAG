import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lat = load("measure_retrieval_latency", ROOT / "scripts" / "measure_retrieval_latency.py")
router = load("router_for_lat", ROOT / "scripts" / "tae_r_router.py")
policy = load("policy_for_lat", ROOT / "scripts" / "tae_v2_policy.py")


def fixture(seed=0, n=300, dim=16):
    rng = np.random.RandomState(seed)
    docs = rng.normal(size=(n, dim)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    labels = rng.randint(-1, 6, size=n)
    centroid_ids = np.arange(-1, 6)
    centroids = np.stack([docs[labels == t].mean(0) for t in centroid_ids]).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    vectors = {f"q{i}": docs[i * 7] + 0.1 * rng.normal(size=dim).astype(np.float32) for i in range(12)}
    for v in vectors.values():
        v /= np.linalg.norm(v)
    topic_of = {k: (int(labels[i * 7]) if i % 3 else -1, 0.7) for i, k in enumerate(vectors)}
    by_vec = {v.tobytes(): topic_of[k] for k, v in vectors.items()}
    components = {"embed": lambda text: vectors[text], "topic": lambda q: by_vec[q.tobytes()],
                  "membership": lambda q: np.full(6, 0.15, dtype=np.float32),
                  "centroid_ids": centroid_ids, "centroids": centroids}
    model = {"features": list(router.FULL_FEATURES), "single": "S1", "multi": "S3",
             "model_single": 0.0, "model_multi": 1.0, "t1": 0.5, "t3": 0.5}
    models = {"tae_r": model, "tae_r_p1": dict(model, single="P1_S1", multi="P1_S3")}
    return lat.TopicIndex(docs, labels), components, models, list(vectors)


class LatencyTests(unittest.TestCase):
    def test_topk_matches_stable_argsort(self):
        scores = np.array([0.5, 0.9, 0.5, 0.1, 0.9, 0.5], dtype=np.float32)
        self.assertEqual(lat.topk(scores, 4).tolist(), np.argsort(-scores, kind="stable")[:4].tolist())

    def test_every_path_equals_reference_ranking(self):
        index, components, models, texts = fixture()
        for text in texts:
            results, q, label = lat.run_methods(text, index, components, router, policy, models)
            scores = index.docs @ q
            for method, allowed in (("unfiltered", None), ("strict", [label]),
                                    ("tae_v1", None if label == -1 else [label])):
                self.assertEqual(results[method][0].tolist(), lat.reference_topk(scores, index.labels, allowed).tolist())
            features, p1, p2 = router.compute_features(q, label, 0.7, scores, np.argsort(-scores, kind="stable"),
                                                       index.labels, components["centroid_ids"], components["centroids"], True)
            for variant in ("tae_r", "tae_r_p1"):
                _, allowed = router.route_tae_r(features, p1, p2, models[variant])
                self.assertEqual(results[variant][0].tolist(), lat.reference_topk(scores, index.labels, allowed).tolist())
                self.assertEqual(results[variant][1]["topic"], results["strict"][1]["topic"])
            self.assertEqual(results["unfiltered"][1]["topic"], 0.0)

    def test_measure_skips_warmup_and_reports_components(self):
        index, components, models, texts = fixture()
        summary, mismatches, n = lat.measure(texts, index, components, router, policy, models, warmup=2, log=lambda *_: None)
        self.assertEqual(n, len(texts) - 2)
        self.assertEqual(set(summary), set(lat.METHODS))
        self.assertTrue(all(v == 0 for v in mismatches.values()))
        self.assertEqual(summary["unfiltered"]["components_mean_ms"]["membership"], 0.0)
        self.assertGreater(summary["tae_r"]["components_mean_ms"]["route"], 0.0)


if __name__ == "__main__":
    unittest.main()
