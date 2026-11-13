"""TAE-R+ exploration on the validation fold only (post-hoc, declared after the final run of v1).

The validation fold is split by a fixed hash into TUNE and CONFIRM halves. Every variant and setting is
scored on TUNE; the single best by AllGold@5 (ties: Recall@5, then lower cost) is chosen there. Only the
chosen variant is then scored on CONFIRM, against unfiltered search and the frozen TAE-R (paired
bootstrap 95% CI, McNemar). No final_eval question or final-corpus paragraph is read.

Variant families (retrieval only; exact MiniLM search over the validation corpus):
  boost(l)      soft topic boosting: score + l * P(S3 adequate) for paragraphs in the probe's top-3 topics
  diverse(c)    topic-diverse top-k: at most c paragraphs per topic, taken in unfiltered order
  boost+diverse combination of the two
  coverage(t)   hard filter to the fewest probe topics whose similarity mass reaches share t
  gbm           the frozen TAE-R design with a gradient-boosting predictor (thresholds out-of-fold on
                router_train, as in v1)
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


router = _load("tae_r_router", "tae_r_router.py")
fit = router.fit
KS = (5, 10)
BOOSTS = (0.01, 0.02, 0.03, 0.05, 0.08)
CAPS = (1, 2, 3)
COVERAGE = (0.5, 0.6, 0.7, 0.8, 0.9)
NEAR = 1e-9


def half(question_id):
    return "tune" if int(hashlib.sha256(f"tae-r-plus:{question_id}".encode()).hexdigest(), 16) % 2 == 0 else "confirm"


def topk_boost(scores, labels, topics, bonus, k=max(KS)):
    boosted = scores + bonus * np.isin(labels, topics)
    part = np.argpartition(-boosted, k)[:k]
    return part[np.lexsort((part, -boosted[part]))]


def topk_diverse(order, labels, cap, k=max(KS)):
    picked, used = [], {}
    for i in order:
        t = int(labels[i])
        if used.get(t, 0) < cap:
            picked.append(i)
            used[t] = used.get(t, 0) + 1
            if len(picked) == k:
                break
    return np.array(picked)


def coverage_topics(ranked_mass, share):
    total = sum(m for _, m in ranked_mass) or 1.0
    chosen, acc = [], 0.0
    for topic, mass in ranked_mass:
        chosen.append(topic)
        acc += mass
        if acc / total >= share:
            break
    return chosen


def outcome(top, gold):
    out = {}
    for k in KS:
        hits = int(np.isin(gold, top[:k]).sum())
        out[f"hits@{k}"], out[f"all@{k}"] = hits, int(hits == len(gold))
    return out


def evaluate(cand, inputs, model, gbm_model=None):
    """Per validation question: outcomes of every variant. Returns {variant: [outcome...]}, rows."""
    doc_index = {doc: i for i, doc in enumerate(inputs["val_ids"])}
    docs = fit.rows_for(inputs, inputs["val_ids"])
    labels = cand["doc_labels"]
    n_docs = len(labels)
    sizes = dict(zip(*[a.tolist() for a in np.unique(labels, return_counts=True)]))
    results, meta = {}, []

    def add(name, top, searched):
        results.setdefault(name, []).append(dict(outcome(top, gold), searched=searched))

    for qi, (split, qid) in enumerate(cand["ids"]):
        if split != "validation":
            continue
        q = cand["q_vectors"][qi]
        gold = np.array([doc_index[d] for d in cand["gold"][(split, qid)]], dtype=np.int64)
        scores = docs @ q
        order = np.argsort(-scores, kind="stable")
        label, strength = int(cand["q_labels"][qi]), float(cand["q_strengths"][qi])
        features, p1t, p2t = router.compute_features(q, label, strength, scores, order, labels,
                                                     cand["centroid_ids"], cand["centroids"], cand["hdbscan"])
        x = np.array([[features[n] for n in model["features"]]])
        p3 = float(model["model_multi"].predict_proba(x)[0, 1]) if hasattr(model["model_multi"], "predict_proba") else float(model["model_multi"])
        route, allowed = router.route_tae_r(features, p1t, p2t, model)
        ranked = order[np.isin(labels[order], allowed)] if allowed is not None else order
        add("unfiltered", order[:10], n_docs)
        add("tae_r", ranked[:10], n_docs + len(cand["centroid_ids"]))
        if gbm_model is not None:
            r2, a2 = router.route_tae_r(features, p1t, p2t, gbm_model)
            add("gbm", (order[np.isin(labels[order], a2)] if a2 is not None else order)[:10], n_docs + len(cand["centroid_ids"]))
        s3 = p2t[:3]
        for b in BOOSTS:
            add(f"boost({b})", topk_boost(scores, labels, s3, b * p3), n_docs + len(cand["centroid_ids"]))
        for c in CAPS:
            add(f"diverse({c})", topk_diverse(order, labels, c), n_docs)
            for b in BOOSTS:
                boosted = scores + b * p3 * np.isin(labels, s3)
                add(f"boost({b})+diverse({c})", topk_diverse(np.argsort(-boosted, kind="stable"), labels, c),
                    n_docs + len(cand["centroid_ids"]))
        mass = router.topic_mass(labels[order[:router.PROBE_M]], scores[order[:router.PROBE_M]])
        for t in COVERAGE:
            topics = coverage_topics(mass, t)
            add(f"coverage({t})", order[np.isin(labels[order], topics)][:10], n_docs)
        meta.append({"question_id": qid, "half": half(qid), "n_gold": len(gold)})
    return results, meta


def summarise(outs, meta, which, n_docs):
    idx = [i for i, m in enumerate(meta) if m["half"] == which]
    gold_total = sum(meta[i]["n_gold"] for i in idx)
    return {"questions": len(idx),
            "allgold@5": sum(outs[i]["all@5"] for i in idx) / len(idx),
            "allgold@10": sum(outs[i]["all@10"] for i in idx) / len(idx),
            "recall@5": sum(outs[i]["hits@5"] for i in idx) / gold_total,
            "mean_searched_share": sum(outs[i]["searched"] for i in idx) / len(idx) / n_docs}


def gbm_predictor(cand, inputs, base_model):
    """Frozen TAE-R design with a gradient-boosting predictor; thresholds out-of-fold on router_train."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows, n_docs, _ = router.question_rows(cand, inputs)
    train = [r for r in rows if r["split"] == "router_train"]
    original = router.make_classifier
    router.make_classifier = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                     max_leaf_nodes=15, random_state=router.SEED)
    try:
        x = router.matrix(train, router.FULL_FEATURES)
        y1 = np.array([r["adequate_S1"] for r in train])
        y3 = np.array([r["adequate_S3"] for r in train])
        th = router.choose_thresholds(train, router.out_of_fold(x, y1), router.out_of_fold(x, y3), "S1", "S3", n_docs)
        model = dict(base_model, model_single=router.fit_model(x, y1), model_multi=router.fit_model(x, y3),
                     t1=th["t1"], t3=th["t3"])
    finally:
        router.make_classifier = original
    return model, th


def run(corpora_dir, embeddings_dir, topics_dir, name, model_path, *, frozen=fit.FROZEN, log=print, with_gbm=True):
    import joblib
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, frozen)
    cand = router.load_candidate(inputs, topics_dir, corpora_dir, name)
    model = joblib.load(model_path)["variants"]["tae_r"]
    gbm, gbm_th = gbm_predictor(cand, inputs, model) if with_gbm else (None, None)
    results, meta = evaluate(cand, inputs, model, gbm)
    n_docs = len(cand["doc_labels"])
    tune = {v: summarise(o, meta, "tune", n_docs) for v, o in results.items()}
    candidates = [v for v in tune if v not in ("unfiltered", "tae_r")]
    chosen = max(candidates, key=lambda v: (tune[v]["allgold@5"], tune[v]["recall@5"], -tune[v]["mean_searched_share"], v))
    confirm = {v: summarise(results[v], meta, "confirm", n_docs) for v in (chosen, "unfiltered", "tae_r")}
    idx = [i for i, m in enumerate(meta) if m["half"] == "confirm"]
    stats = {}
    for ref in ("unfiltered", "tae_r"):
        mine = [results[chosen][i]["all@5"] for i in idx]
        base = [results[ref][i]["all@5"] for i in idx]
        diff = router.paired_bootstrap(mine, base)
        stats[f"{chosen}_vs_{ref}"] = dict(diff, **router.mcnemar(mine, base), decision=router.gate2(diff))
    log(f"{name}: chosen on tune = {chosen}")
    return {"candidate": name, "split": "validation (tune/confirm halves)", "gbm_thresholds": gbm_th,
            "tune": tune, "chosen": chosen, "confirm": confirm, "confirm_comparisons": stats}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--topics-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="H2")
    parser.add_argument("--router-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.corpora_dir, args.embeddings_dir, args.topics_dir, args.candidate, args.router_model)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    top = sorted(report["tune"].items(), key=lambda kv: -kv[1]["allgold@5"])[:8]
    print(json.dumps({"tune_top8": dict(top), "chosen": report["chosen"], "confirm": report["confirm"],
                      "confirm_comparisons": report["confirm_comparisons"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
