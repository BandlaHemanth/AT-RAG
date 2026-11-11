"""TAE-R: retrieval-grounded topic adequacy routing (protocol v1.3, sections 5-7, 11-12).

Retrieval-only evaluation on the validation fold. For one topic-model candidate
(H2 primary, K2 secondary) it:

1. computes, for every router_train and validation question, label-free features:
   Probe 1 - cosine to every topic centroid (incl. a -1 centroid for HDBSCAN models);
   Probe 2 - an exact unfiltered top-m probe and its similarity-weighted topic mass;
2. defines candidate routes: S1 (one topic), S3 (up to three topics), unfiltered;
3. trains two adequacy classifiers on router_train only (label: all gold paragraphs
   lie inside S1 / S3) and picks thresholds on router_train out-of-fold predictions;
4. routes the validation questions and compares TAE-R, its Probe-1-only ablation,
   strict (original mechanism), TAE V1, TAE V2, fixed three, unfiltered and oracle;
5. reports Gate 2 (TAE-R minus unfiltered on AllGold@5: paired bootstrap 95% CI,
   McNemar) and the full cost of every method.

Never reads final_eval questions or the final retrieval corpus.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = _load("fit_tae_r_topic_models", "fit_tae_r_topic_models.py")
policy = _load("tae_v2_policy", "tae_v2_policy.py")

PROBE_M = 20
KS = (5, 10)
ENTROPY_TEMPERATURE = 0.05
THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)] + [1.01]  # 1.01 = never
NEAR_BEST_POINTS = 0.0025  # prefer cheaper configurations within 0.25 AllGold@5 points of the best
FOLDS = 5
SEED = 42
BOOTSTRAP = 10_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def softmax_entropy(scores, temperature=ENTROPY_TEMPERATURE):
    z = (scores - scores.max()) / temperature
    p = np.exp(z)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p))) if len(p) > 1 else 0.0


def topic_mass(labels, sims):
    """Similarity-weighted topic mass of a probe list, ranked (mass desc, topic asc)."""
    mass = {}
    for label, sim in zip(labels.tolist(), sims.tolist()):
        mass[label] = mass.get(label, 0.0) + max(sim, 0.0)
    return sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    """Exact cosine ranking over one corpus; outcomes for any topic restriction."""

    def __init__(self, doc_vectors, doc_labels):
        self.docs = doc_vectors
        self.labels = doc_labels
        topics, counts = np.unique(doc_labels, return_counts=True)
        self.size = dict(zip(topics.tolist(), counts.tolist()))
        self.n = len(doc_labels)

    def rank(self, query):
        scores = self.docs @ query
        order = np.argsort(-scores, kind="stable")
        return scores, (order, self.labels[order])

    def outcome(self, ranking, gold, topics):
        """topics None = unfiltered. Returns hits@5, hits@10, all@5, all@10, searched documents."""
        order, ranked_labels = ranking
        if topics is None:
            ranked, searched = order[:max(KS)], self.n
        else:
            topics = sorted(set(topics))
            keep = np.isin(ranked_labels, topics)
            ranked = order[keep][:max(KS)]
            searched = sum(self.size.get(t, 0) for t in topics)
        out = {}
        for k in KS:
            hits = int(np.isin(gold, ranked[:k]).sum())
            out[f"hits@{k}"], out[f"all@{k}"] = hits, int(hits == len(gold))
        out["searched"] = searched
        return out


def load_candidate(inputs, topics_dir, corpora_dir, name):
    q_stage = fit.Stage(topics_dir, "questions")
    a_stage = fit.Stage(topics_dir, name + "/assign_validation")
    f_stage = fit.Stage(topics_dir, name + "/fit")
    for stage in (q_stage, a_stage, f_stage):
        require(stage.complete(), f"incomplete stage: {stage.dir}")
    ids = json.loads((q_stage.dir / "question_ids.json").read_text(encoding="utf-8"))
    q_vectors = q_stage.load("question_embeddings.npy")
    doc_labels = a_stage.load("validation_corpus_labels.npy")
    q_labels, q_strengths = a_stage.load("question_labels.npy"), a_stage.load("question_strengths.npy")
    membership = (a_stage.load("question_membership.npy")
                  if (a_stage.dir / "question_membership.npy").is_file() else None)
    # Centroids for every fit label, including -1 for HDBSCAN models (from the fitting population).
    fit_labels = f_stage.load("fit_labels.npy")
    fit_points = fit.rows_for(inputs, inputs["fit_ids"])
    centroid_ids, centroids = fit.topic_centroids(fit_points, fit_labels + 1)
    centroid_ids = centroid_ids - 1
    gold = {}
    for split in ("router_train", "validation"):
        with open(corpora_dir / f"questions_{split}.jsonl", encoding="utf-8") as stream:
            for record in map(json.loads, stream):
                gold[(split, record["question_id"])] = record["gold_doc_ids"]
    return {"ids": ids, "q_vectors": q_vectors, "doc_labels": doc_labels, "q_labels": q_labels,
            "q_strengths": q_strengths, "membership": membership, "centroid_ids": centroid_ids,
            "centroids": centroids, "gold": gold, "hdbscan": membership is not None}


def v2_inference(label, strength, membership_row):
    weights = np.asarray(membership_row, dtype=np.float64)
    if weights.sum() > 1.0:  # float32 rounding can push the sum just above 1; renormalise
        weights = weights / weights.sum()
    entries = [{"topic_id": int(t), "membership": float(w)} for t, w in enumerate(weights) if w > 0]
    total = math.fsum(e["membership"] for e in entries)
    return {"topic_id": int(label), "membership_strength": float(min(max(strength, 0.0), 1.0)),
            "membership_status": "available", "membership_unavailable_reason": None,
            "topic_memberships": entries or [{"topic_id": 0, "membership": 0.0}], "membership_sum": total}


def question_rows(cand, inputs):
    """Features and route outcomes for every router_train and validation question."""
    doc_index = {doc: i for i, doc in enumerate(inputs["val_ids"])}
    retriever = Retriever(fit.rows_for(inputs, inputs["val_ids"]), cand["doc_labels"])
    rows = []
    for qi, (split, qid) in enumerate(cand["ids"]):
        q = cand["q_vectors"][qi]
        gold = np.array([doc_index[d] for d in cand["gold"][(split, qid)]], dtype=np.int64)
        scores, ranking = retriever.rank(q)
        order = ranking[0]
        label, strength = int(cand["q_labels"][qi]), float(cand["q_strengths"][qi])
        # Probe 1: centroid scores (no corpus scan).
        c_scores = cand["centroids"] @ q
        c_rank = np.argsort(-c_scores, kind="stable")
        p1_topics = cand["centroid_ids"][c_rank].tolist()
        top1, top2 = float(c_scores[c_rank[0]]), float(c_scores[c_rank[1]]) if len(c_rank) > 1 else 0.0
        # Probe 2: exact unfiltered top-m probe and its topic mass.
        probe = order[:PROBE_M]
        ranked_mass = topic_mass(cand["doc_labels"][probe], scores[probe])
        total_mass = sum(m for _, m in ranked_mass) or 1.0
        p2_topics = [t for t, _ in ranked_mass]
        probe_labels = cand["doc_labels"][probe]
        features = {
            "p1_top": top1, "p1_margin": top1 - top2, "p1_entropy": softmax_entropy(c_scores),
            "hdb_strength": strength if cand["hdbscan"] else 0.0,
            "hdb_outlier": float(label == -1) if cand["hdbscan"] else 0.0,
            "p1_agrees_label": float(p1_topics[0] == label),
            "p2_top_share": ranked_mass[0][1] / total_mass,
            "p2_top3_share": sum(m for _, m in ranked_mass[:3]) / total_mass,
            "p2_distinct": len(ranked_mass) / PROBE_M,
            "p2_outlier_share": float((probe_labels == -1).mean()),
            "p2_sim1": float(scores[probe[0]]), "p2_sim_gap": float(scores[probe[0]] - scores[probe[4]]),
            "p2_sim_mean": float(scores[probe].mean()),
            "p2_agrees_p1": float(p2_topics[0] == p1_topics[0]),
            "p2_agrees_label": float(p2_topics[0] == label),
        }
        gold_topics = sorted(set(cand["doc_labels"][gold].tolist()))
        routes = {
            "unfiltered": None, "strict": [label], "oracle": gold_topics,
            "S1": p2_topics[:1], "S3": p2_topics[:3], "P1_S1": p1_topics[:1], "P1_S3": p1_topics[:3],
        }
        if cand["hdbscan"]:
            routes["tae_v1"] = None if label == -1 else [label]
            for name in ("tae_v2", "fixed_three"):
                decision = policy.decide(v2_inference(label, strength, cand["membership"][qi]), policy=name)
                routes[name] = None if decision["route"] == "unfiltered" else decision["selected_topic_ids"]
        else:
            routes["tae_v1"] = [label]
        rows.append({
            "split": split, "question_id": qid, "features": features,
            "adequate_S1": int(set(gold_topics) <= set(p2_topics[:1])),
            "adequate_S3": int(set(gold_topics) <= set(p2_topics[:3])),
            "adequate_P1_S1": int(set(gold_topics) <= set(p1_topics[:1])),
            "adequate_P1_S3": int(set(gold_topics) <= set(p1_topics[:3])),
            "routes": {k: (None if v is None else [int(t) for t in v]) for k, v in routes.items()},
            "outcomes": {k: retriever.outcome(ranking, gold, v) for k, v in routes.items()},
            "n_gold": int(len(gold)),
        })
    return rows, retriever.n, len(cand["centroid_ids"])


P1_FEATURES = ("p1_top", "p1_margin", "p1_entropy", "hdb_strength", "hdb_outlier", "p1_agrees_label")
FULL_FEATURES = P1_FEATURES + ("p2_top_share", "p2_top3_share", "p2_distinct", "p2_outlier_share",
                               "p2_sim1", "p2_sim_gap", "p2_sim_mean", "p2_agrees_p1", "p2_agrees_label")


def make_classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def matrix(rows, names):
    return np.array([[r["features"][n] for n in names] for r in rows], dtype=np.float64)


def predict_proba(x_train, y_train, x_test):
    if len(set(y_train.tolist())) < 2:  # degenerate fold: constant prediction
        return np.full(len(x_test), float(y_train.mean()))
    model = make_classifier().fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def out_of_fold(x, y):
    from sklearn.model_selection import StratifiedKFold, KFold
    probs = np.zeros(len(y))
    splitter = (StratifiedKFold(FOLDS, shuffle=True, random_state=SEED) if min(np.bincount(y, minlength=2)) >= FOLDS
                else KFold(FOLDS, shuffle=True, random_state=SEED))
    for train, test in splitter.split(x, y):
        probs[test] = predict_proba(x[train], y[train], x[test])
    return probs


def summarise(rows, choice, n_docs, extra_cost):
    n = len(rows)
    gold_total = sum(r["n_gold"] for r in rows)
    out = {"questions": n}
    for k in KS:
        out[f"recall@{k}"] = sum(r["outcomes"][c][f"hits@{k}"] for r, c in zip(rows, choice)) / gold_total
        out[f"allgold@{k}"] = sum(r["outcomes"][c][f"all@{k}"] for r, c in zip(rows, choice)) / n
    searched = sum(r["outcomes"][c]["searched"] for r, c in zip(rows, choice)) / n
    out["mean_searched_share"] = searched / n_docs
    out["mean_similarity_computations"] = searched + extra_cost
    out["route_counts"] = {c: choice.count(c) for c in sorted(set(choice))}
    return out


def choose_thresholds(rows, p1, p3, single, multi, n_docs):
    best = []
    for t1 in THRESHOLD_GRID:
        for t3 in THRESHOLD_GRID:
            routes = ["S1" if a >= t1 else ("S3" if b >= t3 else "unfiltered") for a, b in zip(p1, p3)]
            choice = [single if c == "S1" else multi if c == "S3" else "unfiltered" for c in routes]
            s = summarise(rows, choice, n_docs, 0)
            best.append((s["allgold@5"], -s["mean_searched_share"], t1, t3))
    top = max(b[0] for b in best)
    near = [b for b in best if top - b[0] <= NEAR_BEST_POINTS]
    near.sort(key=lambda b: (b[1], b[0], b[2], b[3]), reverse=True)  # cheapest, then best, then strictest
    allgold, neg_cost, t1, t3 = near[0]
    return {"t1": t1, "t3": t3, "oof_allgold@5": allgold, "oof_searched_share": -neg_cost, "oof_best_allgold@5": top}


def paired_bootstrap(a, b, seed=SEED, resamples=BOOTSTRAP):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.RandomState(seed)
    stats = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(resamples)])
    return {"difference": float(diff.mean()), "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]}


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, int), np.asarray(b, int)
    helps, hurts = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(helps, helps + hurts, 0.5).pvalue if helps + hurts else 1.0
    return {"helps": helps, "hurts": hurts, "ties": int(len(a) - helps - hurts), "exact_p": float(p)}


def gate2(diff):
    low, high = diff["ci95"]
    if low > 0:
        return "better"
    if low >= -0.01 and high <= 0.01:
        return "equivalent"
    return "worse_or_inconclusive"


def run(corpora_dir, embeddings_dir, topics_dir, name, *, frozen=fit.FROZEN, log=print):
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, frozen)
    cand = load_candidate(inputs, topics_dir, corpora_dir, name)
    rows, n_docs, n_centroids = question_rows(cand, inputs)
    train = [r for r in rows if r["split"] == "router_train"]
    val = [r for r in rows if r["split"] == "validation"]
    log(f"{name}: {len(train)} router_train, {len(val)} validation questions")
    variants = {
        "tae_r": (FULL_FEATURES, "S1", "S3", n_docs + n_centroids),  # probe 2 scans the corpus once
        "tae_r_p1": (P1_FEATURES, "P1_S1", "P1_S3", n_centroids),
    }
    report = {"protocol": fit.PROTOCOL, "candidate": name, "validation_documents": n_docs,
              "topic_centroids": n_centroids, "probe_m": PROBE_M, "methods": {}, "thresholds": {}, "gate2": {}}
    choices = {}
    for variant, (features, single, multi, extra) in variants.items():
        x_tr, x_va = matrix(train, features), matrix(val, features)
        y1 = np.array([r[f"adequate_{single}"] for r in train])
        y3 = np.array([r[f"adequate_{multi}"] for r in train])
        oof1, oof3 = out_of_fold(x_tr, y1), out_of_fold(x_tr, y3)
        th = choose_thresholds(train, oof1, oof3, single, multi, n_docs)
        p1, p3 = predict_proba(x_tr, y1, x_va), predict_proba(x_tr, y3, x_va)
        routes = ["S1" if a >= th["t1"] else ("S3" if b >= th["t3"] else "unfiltered") for a, b in zip(p1, p3)]
        choices[variant] = [single if c == "S1" else multi if c == "S3" else "unfiltered" for c in routes]
        th["router_train_base_rates"] = {single: float(y1.mean()), multi: float(y3.mean())}
        report["thresholds"][variant] = th
        report["methods"][variant] = summarise(val, choices[variant], n_docs, extra)
        log(f"{variant}: thresholds {th['t1']}/{th['t3']}, validation AllGold@5 {report['methods'][variant]['allgold@5']:.4f}")
    baselines = ["unfiltered", "strict", "tae_v1", "oracle"] + (["tae_v2", "fixed_three"] if cand["hdbscan"] else [])
    for method in baselines:
        choices[method] = [method] * len(val)
        report["methods"][method] = summarise(val, choices[method], n_docs, 0)
    base = [r["outcomes"]["unfiltered"]["all@5"] for r in val]
    for method in ("tae_r", "tae_r_p1", "strict", "tae_v2", "oracle"):
        if method not in choices:
            continue
        mine = [r["outcomes"][c]["all@5"] for r, c in zip(val, choices[method])]
        diff = paired_bootstrap(mine, base)
        report["gate2"][method] = dict(diff, **mcnemar(mine, base), decision=gate2(diff))
    report["per_question_sha256"] = hashlib.sha256(json.dumps(
        [[r["question_id"], {m: r["outcomes"][c] for m, c in ((m, choices[m][i]) for m in choices)}]
         for i, r in enumerate(val)], sort_keys=True).encode()).hexdigest()
    return report, rows, choices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--topics-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="H2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, rows, choices = run(args.corpora_dir, args.embeddings_dir, args.topics_dir, args.candidate)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
        detail = args.output.with_suffix(".questions.jsonl")
        with detail.open("x", encoding="utf-8") as stream:
            val = [r for r in rows if r["split"] == "validation"]
            for i, r in enumerate(val):
                stream.write(json.dumps({"question_id": r["question_id"], "routes": r["routes"],
                                         "chosen": {m: choices[m][i] for m in choices},
                                         "outcomes": r["outcomes"], "features": r["features"]}, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError) as error:
        print(f"TAE-R failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"methods": {m: {k: v for k, v in s.items() if k != "route_counts"}
                                  for m, s in report["methods"].items()},
                      "thresholds": report["thresholds"], "gate2": report["gate2"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
