"""Select the TAE-R topic model on the validation fold (protocol v1.2, section 11).

For every fitted candidate, on the 4,576 validation questions and the validation
retrieval corpus (exact cosine search, MiniLM embeddings):

  strict      original AT-RAG mechanism: search only paragraphs whose label equals
              the question's label (including -1 for HDBSCAN candidates)
  unfiltered  search everything (identical for all candidates; computed once)
  oracle      search the union of the topics that hold the question's gold
              paragraphs (uses gold labels; an upper bound only)

Measures Recall@5/10, AllGold@5/10, pre-ranking eligibility, searched share of
the corpus, similarity computations, oracle one-topic and three-topic coverage
and the cross-topic rate, then applies the frozen selection rule. Reads no
final_eval question and no final retrieval corpus.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fit_tae_r_topic_models", HERE / "fit_tae_r_topic_models.py")
fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit)

KS = (5, 10)
MAX_SEARCHED_SHARE = 0.50
TIE_POINTS = 0.01


def require(condition, message):
    if not condition:
        raise ValueError(message)


def top_k(scores, allowed, k):
    """Indices of the k best allowed documents, by score then index (deterministic)."""
    masked = np.where(allowed, scores, -np.inf)
    count = int(allowed.sum())
    if count == 0:
        return np.array([], dtype=np.int64)
    take = min(k, count)
    part = np.argpartition(-masked, take - 1)[:take]
    return part[np.lexsort((part, -masked[part]))]


def evaluate_filters(query_vectors, doc_vectors, golds, filters, chunk=256):
    """filters: name -> function(question_index) -> boolean mask over documents."""
    n_docs = doc_vectors.shape[0]
    stats = {name: {"hits": {k: 0 for k in KS}, "all": {k: 0 for k in KS}, "eligible": 0, "searched": 0}
             for name in filters}
    per_question = {name: [] for name in filters}
    gold_total = sum(len(g) for g in golds)
    for start in range(0, len(golds), chunk):
        scores = query_vectors[start:start + chunk] @ doc_vectors.T
        for offset in range(scores.shape[0]):
            qi = start + offset
            gold = golds[qi]
            for name, make in filters.items():
                allowed = make(qi)
                ranked = top_k(scores[offset], allowed, max(KS))
                s = stats[name]
                s["eligible"] += int(allowed[gold].sum())
                s["searched"] += int(allowed.sum())
                row = {}
                for k in KS:
                    hit = int(np.isin(gold, ranked[:k]).sum())
                    s["hits"][k] += hit
                    s["all"][k] += int(hit == len(gold))
                    row[k] = hit == len(gold)
                per_question[name].append(row[5])
    out = {}
    n = len(golds)
    for name, s in stats.items():
        out[name] = {"recall@5": s["hits"][5] / gold_total, "recall@10": s["hits"][10] / gold_total,
                     "allgold@5": s["all"][5] / n, "allgold@10": s["all"][10] / n,
                     "eligibility": s["eligible"] / gold_total, "mean_searched_share": s["searched"] / (n * n_docs),
                     "mean_similarity_computations": s["searched"] / n,
                     "hits@5": s["hits"][5], "allgold_questions@5": s["all"][5]}
    return out, per_question, {"questions": n, "gold_paragraphs": gold_total, "documents": n_docs}


def oracle_coverage(gold_labels):
    one = sum(1 for labels in gold_labels if len(set(labels)) == 1)
    three = sum(1 for labels in gold_labels if len(set(labels)) <= 3)
    n = len(gold_labels)
    return {"oracle_one_topic_coverage": one / n, "oracle_three_topic_coverage": three / n,
            "cross_topic_rate": 1 - one / n,
            "gold_outlier_share": sum(labels.count(-1) for labels in gold_labels) / sum(map(len, gold_labels))}


def select(metrics):
    """Frozen rule: admissible if strict searched share <= 50%; highest strict AllGold@5;
    within 1 point -> strict Recall@5; within 1 point -> oracle three-topic coverage; then lower cost."""
    trace = []
    admissible = {c: m for c, m in metrics.items() if m["strict"]["mean_searched_share"] <= MAX_SEARCHED_SHARE}
    trace.append({"admissible": sorted(admissible),
                  "excluded_for_cost": sorted(set(metrics) - set(admissible))})
    if not admissible:
        return None, trace
    pool = sorted(admissible)
    for key, label in ((lambda c: admissible[c]["strict"]["allgold@5"], "strict AllGold@5"),
                       (lambda c: admissible[c]["strict"]["recall@5"], "strict Recall@5"),
                       (lambda c: admissible[c]["oracle_three_topic_coverage"], "oracle three-topic coverage")):
        best = max(key(c) for c in pool)
        pool = [c for c in pool if best - key(c) < TIE_POINTS]
        trace.append({"criterion": label, "best": best, "remaining": pool})
        if len(pool) == 1:
            return pool[0], trace
    pool.sort(key=lambda c: (admissible[c]["strict"]["mean_searched_share"], c))
    trace.append({"criterion": "lower strict searched share", "remaining": pool})
    return pool[0], trace


def run(corpora_dir, embeddings_dir, topics_dir, names, *, frozen=fit.FROZEN, log=print):
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, frozen)
    q_stage = fit.Stage(topics_dir, "questions")
    require(q_stage.complete(), "questions stage incomplete")
    ids = json.loads((q_stage.dir / "question_ids.json").read_text(encoding="utf-8"))
    rows = [i for i, (split, _) in enumerate(ids) if split == "validation"]
    require(len(rows) == len(inputs["questions"]["validation"]), "validation question count differs")
    q_vectors = q_stage.load("question_embeddings.npy")[rows]
    gold_by_id = {}
    with open(corpora_dir / "questions_validation.jsonl", encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            gold_by_id[record["question_id"]] = record["gold_doc_ids"]
    doc_index = {doc: i for i, doc in enumerate(inputs["val_ids"])}
    golds = [np.array([doc_index[d] for d in gold_by_id[ids[r][1]]], dtype=np.int64) for r in rows]
    doc_vectors = fit.rows_for(inputs, inputs["val_ids"])
    n_docs = len(inputs["val_ids"])
    everything = np.ones(n_docs, dtype=bool)
    results, base, per_question = {}, None, {}
    for name in names:
        stage = fit.Stage(topics_dir, name + "/assign_validation")
        require(stage.complete(), f"{name}: assign stage incomplete")
        doc_labels = stage.load("validation_corpus_labels.npy")
        q_labels = stage.load("question_labels.npy")[rows]
        filters = {"strict": lambda qi, d=doc_labels, q=q_labels: d == q[qi],
                   "oracle": lambda qi, d=doc_labels: np.isin(d, np.unique(d[golds[qi]]))}
        if base is None:
            filters["unfiltered"] = lambda qi: everything
        metrics, flags, base_info = evaluate_filters(q_vectors, doc_vectors, golds, filters)
        if base is None:
            base = {"unfiltered": metrics.pop("unfiltered"), "info": base_info}
            per_question["unfiltered"] = flags.pop("unfiltered")
        fit_info = json.loads((topics_dir / name / "fit" / "stage_manifest.json").read_text())["info"]
        results[name] = dict(metrics, **oracle_coverage([doc_labels[g].tolist() for g in golds]),
                             topics=fit_info["topics"], fit_outlier_rate=fit_info["outlier_rate"],
                             question_outlier_rate=float((q_labels < 0).mean()))
        per_question[name] = flags["strict"]
        log(f"{name}: strict AllGold@5 {metrics['strict']['allgold@5']:.4f}, Recall@5 {metrics['strict']['recall@5']:.4f}, "
            f"searched {metrics['strict']['mean_searched_share']:.3f}, cross-topic {results[name]['cross_topic_rate']:.3f}")
    chosen, trace = select(results)
    return {"protocol": fit.PROTOCOL, "split": "validation", "candidates": results, "baseline": base,
            "selected": chosen, "selection_trace": trace,
            "per_question_allgold5_sha256": hashlib.sha256(json.dumps(per_question, sort_keys=True).encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--topics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=list(fit.CONFIG["candidates"]))
    args = parser.parse_args(argv)
    try:
        report = run(args.corpora_dir, args.embeddings_dir, args.topics_dir, args.candidates)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"selected": report["selected"], "trace": report["selection_trace"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
