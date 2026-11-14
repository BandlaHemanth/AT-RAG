"""Controlled retrieval latency per method (validation fold only; descriptive, no accuracy claim).

For each of the first N validation questions (after W warm-up questions that are not recorded), every
method's retrieval path is executed and timed component by component, in one process and one thread:

  embed        MiniLM encoding of the question text (shared by every method)
  topic        UMAP transform + HDBSCAN approximate_predict (methods that use the question's topic)
  membership   HDBSCAN membership vector (TAE V2 only)
  route        routing decision: TAE-R = centroid probe + top-20 probe features + two classifiers;
               TAE V2 = policy decision; others 0
  scan         similarity computations (cosine scores) over the searched paragraphs
  rank         top-k selection from those scores (k = 5)

Methods: unfiltered, strict (original AT-RAG single-topic filter), tae_v1, tae_v2, tae_r, tae_r_p1.
Filtered methods search a per-topic index (paragraphs grouped by topic, as a metadata filter would),
so their scan cost is proportional to the topic sizes. TAE-R ranks its filtered set from the probe's
full scan (no second scan), as in the accuracy evaluation. The retrieved top-k of each timed path is
checked against the reference ranking of the accuracy evaluation, so the timing measures the same method.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = 5
METHODS = ("unfiltered", "strict", "tae_v1", "tae_v2", "tae_r", "tae_r_p1")
COMPONENTS = ("embed", "topic", "membership", "route", "scan", "rank")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def topk(scores, k):
    """Indices of the k largest scores, ordered by (score desc, index asc), like a stable argsort."""
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


class TopicIndex:
    """Paragraph vectors grouped contiguously by topic, so a topic filter scans only its own rows."""

    def __init__(self, docs, labels):
        self.docs, self.labels = docs, np.asarray(labels)
        order = np.argsort(self.labels, kind="stable")
        self.rows = order
        self.grouped = np.ascontiguousarray(docs[order])
        topics, starts, counts = np.unique(self.labels[order], return_index=True, return_counts=True)
        self.block = {int(t): (int(s), int(s + c)) for t, s, c in zip(topics, starts, counts)}

    def filtered(self, q, topics, timer):
        """Scan only the paragraphs of `topics`; returns global indices of the top-k."""
        with timer("scan"):
            spans = [self.block[t] for t in sorted(set(topics)) if t in self.block]
            rows = np.concatenate([self.rows[a:b] for a, b in spans]) if spans else np.empty(0, dtype=np.int64)
            scores = (np.concatenate([self.grouped[a:b] @ q for a, b in spans]) if spans
                      else np.empty(0, dtype=np.float32))
        with timer("rank"):
            best = topk(scores, K)
            # Order ties by global index, as the reference ranking does.
            result = rows[best]
            result = result[np.lexsort((result, -scores[best]))]
        return result

    def full_scan(self, q, timer):
        with timer("scan"):
            scores = self.docs @ q
        return scores


class Timer:
    def __init__(self):
        self.parts = dict.fromkeys(COMPONENTS, 0.0)

    def __call__(self, part):
        timer = self

        class _Span:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.parts[part] += time.perf_counter() - self.t
        return _Span()


def run_methods(text, index, components, router, policy, models):
    """Execute and time every method for one question. Returns {method: (top-k, {component: s})}."""
    out = {}
    base = Timer()
    with base("embed"):
        q = components["embed"](text)
    with base("topic"):
        label, strength = components["topic"](q)
    label, strength = int(label), float(strength)
    shared_embed, shared_topic = base.parts["embed"], base.parts["topic"]

    def fresh(topic=True):
        t = Timer()
        t.parts["embed"] = shared_embed
        t.parts["topic"] = shared_topic if topic else 0.0
        return t

    t = fresh(topic=False)
    scores = index.full_scan(q, t)
    with t("rank"):
        top = topk(scores, K)
    out["unfiltered"] = (top, t.parts)

    t = fresh()
    out["strict"] = (index.filtered(q, [label], t), t.parts)

    t = fresh()
    if label == -1:
        scores_v1 = index.full_scan(q, t)
        with t("rank"):
            top = topk(scores_v1, K)
    else:
        top = index.filtered(q, [label], t)
    out["tae_v1"] = (top, t.parts)

    t = fresh()
    with t("membership"):
        membership = components["membership"](q)
    with t("route"):
        decision = policy.decide(router.v2_inference(label, strength, membership), policy="tae_v2")
        topics = None if decision["route"] == "unfiltered" else decision["selected_topic_ids"]
    if topics is None:
        s = index.full_scan(q, t)
        with t("rank"):
            top = topk(s, K)
    else:
        top = index.filtered(q, topics, t)
    out["tae_v2"] = (top, t.parts)

    for variant in ("tae_r", "tae_r_p1"):
        t = fresh()
        s = index.full_scan(q, t)  # the probe scan; reused for the final ranking
        with t("route"):
            probe = topk(s, router.PROBE_M)
            features, p1, p2 = router.compute_features(q, label, strength, s, probe, index.labels,
                                                       components["centroid_ids"], components["centroids"], True)
            route, allowed = router.route_tae_r(features, p1, p2, models[variant])
        with t("rank"):
            if allowed is None:
                top = probe[:K]
            else:
                keep = np.flatnonzero(np.isin(index.labels, allowed))
                top = keep[topk(s[keep], K)]
        out[variant] = (top, t.parts)
    return out, q, label


def reference_topk(scores, labels, allowed):
    """The accuracy evaluation's ranking: stable argsort, then the topic filter."""
    order = np.argsort(-scores, kind="stable")
    if allowed is not None:
        order = order[np.isin(labels[order], allowed)]
    return order[:K]


def summarise(per_question):
    """per_question: list of {method: {component: seconds}} -> milliseconds summary per method."""
    report = {}
    for method in METHODS:
        totals = [sum(q[method].values()) * 1000 for q in per_question]
        entry = {"mean_ms": statistics.fmean(totals), "median_ms": statistics.median(totals),
                 "p95_ms": float(np.percentile(totals, 95)), "components_mean_ms": {}}
        for part in COMPONENTS:
            entry["components_mean_ms"][part] = statistics.fmean(q[method][part] * 1000 for q in per_question)
        entry["retrieval_only_mean_ms"] = entry["mean_ms"] - entry["components_mean_ms"]["embed"]
        report[method] = entry
    return report


def measure(texts, index, components, router, policy, models, warmup, log=print):
    per_question, mismatches = [], {m: 0 for m in METHODS}
    for i, text in enumerate(texts):
        results, q, label = run_methods(text, index, components, router, policy, models)
        if i < warmup:
            continue
        per_question.append({m: dict(parts) for m, (_, parts) in results.items()})
        # Same method as the accuracy evaluation: compare with the reference ranking (untimed).
        scores = index.docs @ q
        refs = {"unfiltered": None, "strict": [label], "tae_v1": None if label == -1 else [label]}
        for m, allowed in refs.items():
            mismatches[m] += int(not np.array_equal(results[m][0], reference_topk(scores, index.labels, allowed)))
        if (i + 1) % 100 == 0:
            log(f"{i + 1}/{len(texts)} questions timed")
    return summarise(per_question), mismatches, len(per_question)


def load_components(corpora_dir, embeddings_dir, topics_dir, router_model_path, device, router, fit):
    import joblib
    import hdbscan
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, fit.FROZEN)
    val_ids = inputs["val_ids"]
    labels = fit.Stage(topics_dir, "H2/assign_validation").load("validation_corpus_labels.npy")
    docs = np.ascontiguousarray(fit.rows_for(inputs, val_ids), dtype=np.float32)
    f_stage = fit.Stage(topics_dir, "H2/fit")
    centroid_ids, centroids = fit.topic_centroids(fit.rows_for(inputs, inputs["fit_ids"]), f_stage.load("fit_labels.npy") + 1)
    hdb = f_stage.load("hdbscan_model.joblib")
    umap_model = fit.Stage(topics_dir, "umap").load("umap_model.joblib")
    from sentence_transformers import SentenceTransformer
    enc = fit.CONFIG["encoder"]
    encoder = SentenceTransformer(enc["model_id"], revision=enc["revision"], device=device)
    encoder.max_seq_length = enc["max_seq_length"]
    saved = joblib.load(router_model_path)

    def embed(text):
        return encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)[0].astype(np.float32)

    def topic(q):
        labels_, strengths = hdbscan.approximate_predict(hdb, umap_model.transform(q[None, :]).astype(np.float32))
        return labels_[0], strengths[0]

    def membership(q):
        return hdbscan.membership_vector(hdb, umap_model.transform(q[None, :]).astype(np.float32))[0]

    components = {"embed": embed, "topic": topic, "membership": membership,
                  "centroid_ids": centroid_ids - 1, "centroids": centroids}
    texts = [t for _, t in inputs["questions"]["validation"]]
    return TopicIndex(docs, labels), components, saved["variants"], texts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--topics-dir", type=Path, required=True)
    parser.add_argument("--router-model", type=Path, required=True, help="tae_r_router_H2_v1.joblib")
    parser.add_argument("--questions", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="encoder device (default cpu, single thread)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    router = _load("tae_r_router", "tae_r_router.py")
    policy = _load("tae_v2_policy", "tae_v2_policy.py")
    with threadpool_limits(limits=1):
        index, components, models, texts = load_components(args.corpora_dir, args.embeddings_dir, args.topics_dir,
                                                           args.router_model, args.device, router, router.fit)
        texts = texts[:args.warmup + args.questions]
        summary, mismatches, n = measure(texts, index, components, router, policy, models, args.warmup)
    report = {"split": "validation", "candidate": "H2", "questions_timed": n, "warmup": args.warmup, "k": K,
              "corpus_paragraphs": int(len(index.labels)), "threads": 1, "encoder_device": args.device,
              "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
              "router_model_sha256": hashlib.sha256(args.router_model.read_bytes()).hexdigest(),
              "topk_mismatches_vs_reference": mismatches, "methods": summary}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps({m: {"mean_ms": round(s["mean_ms"], 2), "retrieval_only_mean_ms": round(s["retrieval_only_mean_ms"], 2),
                          **{c: round(v, 2) for c, v in s["components_mean_ms"].items()}}
                      for m, s in summary.items()}, indent=1))
    print("top-k mismatches vs reference:", mismatches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
