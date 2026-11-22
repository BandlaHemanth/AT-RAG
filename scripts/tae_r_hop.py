"""TAE-R-H: hop-aware retrieval with topic routing on the second hop (protocol amendment v3.0).

Hop 1 is TAE-R v1 unchanged. The top `a` hop-1 paragraphs are kept; for each of the top `B` hop-1 paragraphs p a
second query e2 = normalise(e_q + lam * e_p) searches the corpus (excluding the kept paragraphs), routed by the same
TAE-R procedure and saved predictor ("topics" arm) or over the whole corpus ("none" arm, the ablation). The final
top 5 is the kept paragraphs followed by the best hop-2 candidates by their best hop-2 score.

  selection (2WikiMultiHopQA validation only):
    python scripts/tae_r_hop.py select --corpora-dir .. --embeddings-dir .. --topics-dir .. --router-model .. --output-dir ..
  final (each dataset, once; the frozen selection.json is applied unchanged):
    python scripts/tae_r_hop.py final --selection selection.json --allow-final \
        (--replication-work-dir <dir> | --corpora-dir .. --embeddings-dir .. --topics-dir .. --final-dir .. --router-model ..) \
        --output-dir ..
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = "tae-r-protocol-v3.0-hop"
K = 5
GRID = [{"lam": lam, "beams": b, "keep": a} for lam in (0.5, 1.0) for b in (1, 2) for a in (2, 3)]
ARMS = ("topics", "none")
CANDIDATE = "H2"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


router = _load("tae_r_router", "tae_r_router.py")
fit = router.fit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def config_key(arm, cfg):
    return f"{arm}:lam={cfg['lam']},B={cfg['beams']},a={cfg['keep']}"


def stable_order(scores):
    return np.argsort(-scores, kind="stable")


def route(qvec, scores, order, label, strength, labels, ctx, model):
    features, p1, p2 = router.compute_features(qvec, label, strength, scores, order, labels,
                                               ctx["centroid_ids"], ctx["centroids"], ctx["hdbscan"])
    return router.route_tae_r(features, p1, p2, model)


def hop_topk(q, label, strength, docs, labels, ctx, model, cfg, arm, hop1=None):
    """Top-K of TAE-R-H for one question. Returns (indices, similarity computations, hop-1 ranked list)."""
    n_docs, n_cent = len(labels), len(ctx["centroid_ids"])
    if hop1 is None:
        scores = docs @ q
        order = stable_order(scores)
        _, allowed = route(q, scores, order, label, strength, labels, ctx, model)
        hop1 = order if allowed is None else order[np.isin(labels[order], allowed)]
    cost = n_docs + n_cent
    kept = [int(i) for i in hop1[:cfg["keep"]]]
    kept_set = set(kept)
    best = {}
    for j in range(min(cfg["beams"], len(hop1))):
        q2 = q + cfg["lam"] * docs[hop1[j]]
        q2 = (q2 / np.linalg.norm(q2)).astype(np.float32)
        s2 = docs @ q2
        o2 = stable_order(s2)
        cost += n_docs
        if arm == "topics":
            _, allowed2 = route(q2, s2, o2, label, strength, labels, ctx, model)
            cost += n_cent
            if allowed2 is not None:
                o2 = o2[np.isin(labels[o2], allowed2)]
        taken = 0
        for d in o2:
            d = int(d)
            if d in kept_set:
                continue
            if s2[d] > best.get(d, -np.inf):
                best[d] = float(s2[d])
            taken += 1
            if taken >= K:
                break
    extra = sorted(best, key=lambda d: (-best[d], d))[:K - len(kept)]
    top = kept + extra
    for d in hop1:
        if len(top) >= K:
            break
        if int(d) not in top:
            top.append(int(d))
    return np.array(top[:K], dtype=np.int64), cost, hop1


def outcome(top, gold):
    hits = int(np.isin(gold, top[:K]).sum())
    return {"hits@5": hits, "all@5": int(hits == len(gold))}


def evaluate(questions, docs, labels, ctx, model, configs, log=print):
    """questions: list of dicts {question_id, q, label, strength, gold (indices), type}. configs: {key: (arm, cfg)}.
    Also scores strict, unfiltered and TAE-R v1 from the same scans. Returns {method: [outcome per question]}."""
    out = {m: [] for m in ["strict", "unfiltered", "tae_r", *configs]}
    costs = {m: [] for m in out}
    n_docs, n_cent = len(labels), len(ctx["centroid_ids"])
    started = time.time()
    for i, item in enumerate(questions):
        q, gold = item["q"], np.asarray(item["gold"], dtype=np.int64)
        scores = docs @ q
        order = stable_order(scores)
        _, allowed = route(q, scores, order, item["label"], item["strength"], labels, ctx, model)
        hop1 = order if allowed is None else order[np.isin(labels[order], allowed)]
        strict = order[labels[order] == item["label"]]
        for m, top, c in (("unfiltered", order, n_docs), ("strict", strict, int((labels == item["label"]).sum())),
                          ("tae_r", hop1, n_docs + n_cent)):
            out[m].append(outcome(top, gold))
            costs[m].append(c)
        for key, (arm, cfg) in configs.items():
            top, c, _ = hop_topk(q, item["label"], item["strength"], docs, labels, ctx, model, cfg, arm, hop1=hop1)
            out[key].append(outcome(top, gold))
            costs[key].append(c)
        if (i + 1) % 500 == 0:
            log(f"{i + 1}/{len(questions)} questions, {time.time() - started:.0f} s")
    return out, costs


def summarise(outs, costs, questions):
    n, total_gold = len(questions), sum(len(q["gold"]) for q in questions)
    return {"questions": n, "allgold@5": sum(o["all@5"] for o in outs) / n,
            "recall@5": sum(o["hits@5"] for o in outs) / total_gold,
            "mean_similarity_computations": float(np.mean(costs))}


def choose(summary, arm):
    keys = [k for k in summary if k.startswith(arm + ":")]
    def rank(k):
        s, cfg = summary[k], parse_key(k)
        return (-s["allgold@5"], -s["recall@5"], s["mean_similarity_computations"], cfg["lam"], cfg["beams"], cfg["keep"])
    return min(keys, key=rank)


def parse_key(key):
    arm, rest = key.split(":")
    parts = dict(p.split("=") for p in rest.split(","))
    return {"lam": float(parts["lam"]), "beams": int(parts["B"]), "keep": int(parts["a"])}


def compare(outs, a, b):
    mine = [o["all@5"] for o in outs[a]]
    base = [o["all@5"] for o in outs[b]]
    diff = router.paired_bootstrap(mine, base)
    return dict(diff, **router.mcnemar(mine, base), decision=router.gate2(diff))


# ------------------------------------------------------------------------------------------- loading

def load_validation(corpora_dir, embeddings_dir, topics_dir, frozen):
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, frozen)
    cand = router.load_candidate(inputs, topics_dir, corpora_dir, CANDIDATE)
    docs = np.ascontiguousarray(fit.rows_for(inputs, inputs["val_ids"]), dtype=np.float32)
    index = {d: i for i, d in enumerate(inputs["val_ids"])}
    types = {}
    with open(Path(corpora_dir) / "questions_validation.jsonl", encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            types[record["question_id"]] = record.get("type")
    questions = []
    for i, (split, qid) in enumerate(cand["ids"]):
        if split != "validation":
            continue
        questions.append({"question_id": qid, "q": cand["q_vectors"][i], "label": int(cand["q_labels"][i]),
                          "strength": float(cand["q_strengths"][i]), "type": types.get(qid),
                          "gold": [index[d] for d in cand["gold"][(split, qid)]]})
    return questions, docs, cand["doc_labels"], context(cand)


def load_final(corpora_dir, embeddings_dir, topics_dir, final_dir, frozen):
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, frozen)
    final_ids = fit.jsonl_ids(Path(corpora_dir) / "retrieval_final.jsonl", "doc_id")
    stage, f_stage = fit.Stage(final_dir, CANDIDATE + "/assign_final"), fit.Stage(topics_dir, CANDIDATE + "/fit")
    require(stage.complete() and f_stage.complete(), "final assignment or fit stage incomplete")
    q_ids = json.loads((stage.dir / "question_ids.json").read_text(encoding="utf-8"))
    centroid_ids, centroids = fit.topic_centroids(fit.rows_for(inputs, inputs["fit_ids"]), f_stage.load("fit_labels.npy") + 1)
    membership_file = stage.dir / "question_membership.npy"
    ctx = {"centroid_ids": centroid_ids - 1, "centroids": centroids, "hdbscan": membership_file.is_file()}
    docs = np.ascontiguousarray(fit.rows_for(inputs, final_ids), dtype=np.float32)
    index = {d: i for i, d in enumerate(final_ids)}
    meta = {}
    with open(Path(corpora_dir) / "questions_final_eval.jsonl", encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            meta[record["question_id"]] = record
    vecs, qlab, qstr = stage.load("question_embeddings.npy"), stage.load("question_labels.npy"), stage.load("question_strengths.npy")
    questions = [{"question_id": qid, "q": vecs[i], "label": int(qlab[i]), "strength": float(qstr[i]),
                  "type": meta[qid].get("type"), "gold": [index[d] for d in meta[qid]["gold_doc_ids"]]}
                 for i, qid in enumerate(q_ids)]
    return questions, docs, stage.load("final_corpus_labels.npy"), ctx


def context(cand):
    return {"centroid_ids": cand["centroid_ids"], "centroids": cand["centroids"], "hdbscan": cand["hdbscan"]}


def hop_counts(qid):
    import re
    m = re.match(r"^(\d)hop", str(qid))
    return f"{m.group(1)}-hop" if m else None


def by_type(outs, questions, methods):
    types = [q["type"] or hop_counts(q["question_id"]) for q in questions]
    if not any(types):
        return None
    result = {}
    for t in sorted({t for t in types if t}):
        idx = [i for i, x in enumerate(types) if x == t]
        result[t] = {"questions": len(idx), **{m: sum(outs[m][i]["all@5"] for i in idx) / len(idx) for m in methods}}
    return result


# ------------------------------------------------------------------------------------------- commands

def run_select(paths, out_dir, frozen=None, log=print):
    import joblib
    questions, docs, labels, ctx = load_validation(paths["corpora"], paths["embeddings"], paths["topics"], frozen or fit.FROZEN)
    model = joblib.load(paths["model"])["variants"]["tae_r"]
    configs = {config_key(arm, cfg): (arm, cfg) for arm in ARMS for cfg in GRID}
    log(f"selection: {len(questions)} validation questions, {len(configs)} configurations")
    outs, costs = evaluate(questions, docs, labels, ctx, model, configs, log)
    summary = {m: summarise(outs[m], costs[m], questions) for m in outs}
    chosen = {arm: choose(summary, arm) for arm in ARMS}
    selection = {"protocol": PROTOCOL, "selected_on": "2WikiMultiHopQA validation", "candidate": CANDIDATE,
                 "chosen": {arm: {"key": k, **parse_key(k)} for arm, k in chosen.items()},
                 "router_model_sha256": fit.sha256_file(paths["model"])}
    report = {"selection": selection, "summary": summary,
              "comparisons": {f"{k}_vs_{ref}": compare(outs, k, ref) for k in chosen.values() for ref in ("unfiltered", "tae_r")},
              "topics_vs_none": compare(outs, chosen["topics"], chosen["none"]),
              "by_type": by_type(outs, questions, ["strict", "unfiltered", "tae_r", *chosen.values()])}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_final(paths, selection_path, out_dir, frozen=None, log=print, allow_final=False, dataset=None):
    import joblib
    require(allow_final, "final evaluation needs --allow-final (run once)")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    require(selection["protocol"] == PROTOCOL, "selection from another protocol")
    questions, docs, labels, ctx = load_final(paths["corpora"], paths["embeddings"], paths["topics"], paths["final"],
                                              frozen or fit.FROZEN)
    model = joblib.load(paths["model"])["variants"]["tae_r"]
    configs = {f"tae_r_h_{arm}": (arm, {k: selection["chosen"][arm][k] for k in ("lam", "beams", "keep")}) for arm in ARMS}
    log(f"final: {len(questions)} final_eval questions, {len(docs)} paragraphs")
    outs, costs = evaluate(questions, docs, labels, ctx, model, configs, log)
    methods = ["strict", "unfiltered", "tae_r", *configs]
    report = {"protocol": PROTOCOL, "dataset": dataset, "selection": selection,
              "selection_sha256": fit.sha256_file(selection_path), "router_model_sha256": fit.sha256_file(paths["model"]),
              "methods": {m: summarise(outs[m], costs[m], questions) for m in methods},
              "comparisons": {f"{a}_vs_{b}": compare(outs, a, b) for a in configs
                              for b in ("unfiltered", "tae_r", "strict")},
              "by_type": by_type(outs, questions, methods)}
    report["comparisons"]["tae_r_h_topics_vs_tae_r_h_none"] = compare(outs, "tae_r_h_topics", "tae_r_h_none")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_hop_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "final_hop_questions.jsonl").open("w", encoding="utf-8") as stream:
        for i, q in enumerate(questions):
            stream.write(json.dumps({"question_id": q["question_id"], **{m: outs[m][i] for m in methods}}) + "\n")
    return report


def resolve_paths(args):
    if args.replication_work_dir:
        rep = _load("run_replication", "run_replication.py")
        work = rep.Work(args.replication_work_dir)
        return ({"corpora": work.dir("corpora") / "corpora", "embeddings": work.dir("embed") / "embeddings",
                 "topics": work.dir("topics") / "topics", "final": work.dir("final") / "assign",
                 "model": work.dir("validate") / f"tae_r_router_{CANDIDATE}.joblib"}, work.frozen())
    return ({"corpora": args.corpora_dir, "embeddings": args.embeddings_dir, "topics": args.topics_dir,
             "final": args.final_dir, "model": args.router_model}, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "final"))
    parser.add_argument("--corpora-dir", type=Path)
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--topics-dir", type=Path)
    parser.add_argument("--final-dir", type=Path)
    parser.add_argument("--router-model", type=Path)
    parser.add_argument("--replication-work-dir", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--allow-final", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths, frozen = resolve_paths(args)
    log = lambda m: print(m, flush=True)
    if args.command == "select":
        report = run_select(paths, args.output_dir, frozen, log)
        brief = {"chosen": report["selection"]["chosen"],
                 "allgold@5": {k: round(v["allgold@5"], 4) for k, v in report["summary"].items()},
                 "comparisons": report["comparisons"], "topics_vs_none": report["topics_vs_none"], "by_type": report["by_type"]}
    else:
        report = run_final(paths, args.selection, args.output_dir, frozen, log, args.allow_final, args.dataset)
        brief = {"methods": report["methods"], "comparisons": report["comparisons"], "by_type": report["by_type"]}
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
