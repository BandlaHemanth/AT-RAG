"""Stronger retrieval backbone for every method (protocol amendment v4.0).

Routing is frozen: each method's allowed topics per question are read from the frozen per-question outputs
(final_retrieval_H2.questions.jsonl). Only the ranking inside the allowed set changes: a dense encoder (exact search)
then, optionally, a cross-encoder reranker over the top R.

  select (2WikiMultiHopQA validation, unfiltered search as the method-neutral objective):
    python scripts/strong_backbone.py select --corpora-dir <2wiki corpora> --output-dir <dir>
  final (each dataset once, frozen selection):
    python scripts/strong_backbone.py final --selection backbone_selection.json --allow-final \
        --corpora-dir <corpora> --final-labels <final_corpus_labels.npy> --frozen-questions <final_retrieval_H2.questions.jsonl> \
        --dataset <name> --output-dir <dir>
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = "tae-r-protocol-v4.0-backbone"
K = 5
ENCODERS = {
    "BAAI/bge-base-en-v1.5": {"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
    "intfloat/e5-base-v2": {"query": "query: ", "passage": "passage: "},
}
RERANKERS = ("none", "cross-encoder/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base")
DEPTHS = (20, 50)
METHODS = ("strict", "unfiltered", "tae_r", "tae_r_p1", "oracle")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def passage(record):
    titles = record.get("titles") or ([record["title"]] if record.get("title") else [])
    titles = [t for t in titles if t]
    return (" / ".join(titles) + ". " + record["text"]) if titles else record["text"]


def load_corpus(path):
    records = read_jsonl(path)
    return [r["doc_id"] for r in records], [passage(r) for r in records]


def load_questions(path):
    return [{"question_id": r["question_id"], "text": r["question_text"], "gold_ids": r["gold_doc_ids"],
             "type": r.get("type")} for r in read_jsonl(path)]


# ------------------------------------------------------------------------------ models (injectable for tests)

class Models:
    def __init__(self, device="cuda", encoder_factory=None, reranker_factory=None, cache_dir=None):
        self.device, self.cache_dir = device, Path(cache_dir) if cache_dir else None
        self.encoder_factory = encoder_factory or self._st_encoder
        self.reranker_factory = reranker_factory or self._st_reranker
        self._enc, self._rr, self.revisions = {}, {}, {}

    def _st_encoder(self, name):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name, device=self.device)
        model.max_seq_length = 512
        return model

    def _st_reranker(self, name):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(name, device=self.device, max_length=512)

    def encoder(self, name):
        if name not in self._enc:
            self._enc[name] = self.encoder_factory(name)
            self.revisions[name] = revision_of(self._enc[name])
        return self._enc[name]

    def reranker(self, name):
        if name not in self._rr:
            self._rr[name] = self.reranker_factory(name)
            self.revisions[name] = revision_of(self._rr[name])
        return self._rr[name]

    def embed(self, name, texts, kind, tag=None):
        prefix = ENCODERS[name][kind]
        key = hashlib.sha256((name + kind + "\n".join(texts)).encode("utf-8")).hexdigest()[:20]
        if self.cache_dir is not None:
            f = self.cache_dir / f"{name.replace('/', '_')}_{kind}_{tag or 'x'}_{key}.npy"
            if f.is_file():
                return np.load(f)
        vecs = self.encoder(name).encode([prefix + t for t in texts], batch_size=128, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(f, vecs)
        return vecs

    def rerank(self, name, pairs):
        if not pairs:
            return np.empty(0, dtype=np.float32)
        return np.asarray(self.reranker(name).predict(pairs, batch_size=64, show_progress_bar=False), dtype=np.float32)


def revision_of(model):
    try:
        cfg = model[0].auto_model.config if hasattr(model, "__getitem__") else model.model.config
        return getattr(cfg, "_commit_hash", None) or getattr(cfg, "_name_or_path", None)
    except Exception:
        return None


# ------------------------------------------------------------------------------ ranking

def dense_top(scores, allowed_idx, depth):
    """Indices of the `depth` best-scoring allowed paragraphs, ordered by (score desc, index asc)."""
    cand = allowed_idx if allowed_idx is not None else np.arange(len(scores))
    if len(cand) == 0:
        return cand
    s = scores[cand]
    d = min(depth, len(cand))
    part = np.argpartition(-s, d - 1)[:d]
    kth = s[part].min()
    part = np.flatnonzero(s >= kth)                       # all ties at the boundary, then break ties by index
    chosen = cand[part]
    order = np.lexsort((chosen, -scores[chosen]))
    return chosen[order][:d]


class RerankCache:
    """Cross-encoder scores per (question, paragraph), computed once and shared across methods."""

    def __init__(self, models, name, passages):
        self.models, self.name, self.passages, self.scores = models, name, passages, {}

    def ensure(self, qi, question, docs):
        missing = [int(d) for d in docs if (qi, int(d)) not in self.scores]
        if missing:
            values = self.models.rerank(self.name, [(question, self.passages[d]) for d in missing])
            for d, v in zip(missing, values):
                self.scores[(qi, d)] = float(v)

    def order(self, qi, docs):
        return sorted((int(d) for d in docs), key=lambda d: (-self.scores[(qi, d)], d))


class Backbone:
    """The frozen backbone as an online ranker for the end-to-end pipeline (amendment v4.0).

    top(text, allowed_idx) ranks the allowed paragraphs (None = whole corpus) for any query text, including
    rewritten questions: dense top R by the selected encoder, then the selected cross-encoder, top k."""

    def __init__(self, models, selection, passages, tag="corpus", k=K):
        import threading
        require(selection["protocol"] == PROTOCOL, "selection from another protocol")
        self.models, self.passages, self.k = models, passages, k
        self.enc, self.rr, self.depth = selection["encoder"], selection["reranker"], selection["depth"] or 50
        self.key = selection["key"]
        self.vectors = models.embed(self.enc, passages, "passage", tag=tag)
        self.lock = threading.Lock()

    def top(self, text, allowed_idx):
        with self.lock:
            q = self.models.encoder(self.enc).encode([ENCODERS[self.enc]["query"] + text], normalize_embeddings=True,
                                                     convert_to_numpy=True, show_progress_bar=False)[0]
            cand = dense_top(self.vectors @ np.asarray(q, dtype=np.float32), allowed_idx, self.depth)
            if self.rr == "none" or len(cand) == 0:
                return np.asarray(cand[:self.k], dtype=np.int64)
            scores = self.models.rerank(self.rr, [(text, self.passages[int(d)]) for d in cand])
        order = sorted(range(len(cand)), key=lambda i: (-float(scores[i]), int(cand[i])))
        return np.asarray([int(cand[i]) for i in order[:self.k]], dtype=np.int64)


def outcome(top, gold):
    hits = int(np.isin(gold, np.asarray(top[:K], dtype=np.int64)).sum())
    return {"hits@5": hits, "all@5": int(hits == len(gold))}


def summarise(outs, questions):
    n, total = len(outs), sum(len(q["gold"]) for q in questions)
    return {"questions": n, "allgold@5": sum(o["all@5"] for o in outs) / n,
            "recall@5": sum(o["hits@5"] for o in outs) / total}


# ------------------------------------------------------------------------------ select

def run_select(corpora_dir, out_dir, models, log=print, limit=None):
    corpora_dir = Path(corpora_dir)
    ids, passages = load_corpus(corpora_dir / "retrieval_validation.jsonl")
    index = {d: i for i, d in enumerate(ids)}
    questions = load_questions(corpora_dir / "questions_validation.jsonl")[:limit]
    for q in questions:
        q["gold"] = np.array([index[d] for d in q["gold_ids"]], dtype=np.int64)
    texts = [q["text"] for q in questions]
    results, timing = {}, {}
    for enc in ENCODERS:
        started = time.time()
        pv = models.embed(enc, passages, "passage", tag="val")
        qv = models.embed(enc, texts, "query", tag="valq")
        timing[f"embed:{enc}"] = time.time() - started
        tops = [dense_top(pv @ qv[i], None, max(DEPTHS)) for i in range(len(questions))]
        results[f"{enc}|none"] = [outcome(t, q["gold"]) for t, q in zip(tops, questions)]
        for rr in RERANKERS[1:]:
            started = time.time()
            cache = RerankCache(models, rr, passages)
            for qi, (q, t) in enumerate(zip(questions, tops)):
                cache.ensure(qi, q["text"], t)
            for depth in DEPTHS:
                results[f"{enc}|{rr}|R={depth}"] = [outcome(cache.order(qi, t[:depth]), q["gold"])
                                                    for qi, (q, t) in enumerate(zip(questions, tops))]
            timing[f"rerank:{enc}:{rr}"] = time.time() - started
            log(f"{enc} + {rr}: done")
    summary = {k: summarise(v, questions) for k, v in results.items()}
    def cost(k):  # tie-break: no reranker < smaller depth < smaller reranker
        return (0, 0) if k.endswith("|none") else (int(k.split("R=")[1]), 0 if "MiniLM" in k else 1)
    best = min(summary, key=lambda k: (-summary[k]["allgold@5"], -summary[k]["recall@5"], cost(k), k))
    enc, rr, *rest = best.split("|")
    selection = {"protocol": PROTOCOL, "selected_on": "2WikiMultiHopQA validation, unfiltered search",
                 "encoder": enc, "reranker": rr, "depth": int(rest[0].split("=")[1]) if rest else None,
                 "key": best, "revisions": dict(models.revisions)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backbone_selection.json").write_text(json.dumps(selection, indent=1, sort_keys=True) + "\n")
    (out_dir / "backbone_selection_report.json").write_text(json.dumps(
        {"selection": selection, "summary": summary, "timing_seconds": timing, "questions": len(questions)},
        indent=1, sort_keys=True) + "\n")
    return selection, summary


# ------------------------------------------------------------------------------ final

def allowed_indices(allowed_topics, labels):
    if allowed_topics is None:
        return None
    return np.flatnonzero(np.isin(labels, np.asarray(allowed_topics)))


def run_final(corpora_dir, final_labels, frozen_questions, selection_path, out_dir, models, *, dataset,
              allow_final=False, log=print, limit=None):
    require(allow_final, "final evaluation needs --allow-final (run once)")
    selection = json.loads(Path(selection_path).read_text())
    require(selection["protocol"] == PROTOCOL, "selection from another protocol")
    corpora_dir = Path(corpora_dir)
    ids, passages = load_corpus(corpora_dir / "retrieval_final.jsonl")
    labels = np.load(final_labels)
    require(len(labels) == len(ids), "final labels do not match the final corpus")
    index = {d: i for i, d in enumerate(ids)}
    questions = {q["question_id"]: q for q in load_questions(corpora_dir / "questions_final_eval.jsonl")}
    frozen = read_jsonl(frozen_questions)[:limit]
    items = []
    for row in frozen:
        q = questions[row["question_id"]]
        q["gold"] = np.array([index[d] for d in q["gold_ids"]], dtype=np.int64)
        q["allowed"] = {m: row["routes"][row["chosen"][m]] for m in METHODS if m in row["chosen"]}
        items.append(q)
    enc, rr, depth = selection["encoder"], selection["reranker"], selection["depth"] or 50
    started = time.time()
    pv = models.embed(enc, passages, "passage", tag=f"{dataset}_final")
    qv = models.embed(enc, [q["text"] for q in items], "query", tag=f"{dataset}_finalq")
    log(f"{dataset}: embedded {len(passages)} paragraphs and {len(items)} questions in {time.time() - started:.0f} s")
    cache = RerankCache(models, rr, passages) if rr != "none" else None
    outs = {f"{m}|{stage}": [] for m in METHODS for stage in ("dense", "rerank")}
    for qi, q in enumerate(items):
        scores = pv @ qv[qi]
        for m in METHODS:
            if m not in q["allowed"]:
                continue
            top = dense_top(scores, allowed_indices(q["allowed"][m], labels), depth)
            outs[f"{m}|dense"].append(outcome(top, q["gold"]))
            if cache is not None:
                cache.ensure(qi, q["text"], top)
                outs[f"{m}|rerank"].append(outcome(cache.order(qi, top), q["gold"]))
        if (qi + 1) % 1000 == 0:
            log(f"{dataset}: {qi + 1}/{len(items)} questions, {time.time() - started:.0f} s")
    outs = {k: v for k, v in outs.items() if v}
    router = _load("tae_r_router", "tae_r_router.py")
    stats = {}
    stage = "rerank" if cache is not None else "dense"
    for a, b in (("tae_r", "strict"), ("tae_r", "unfiltered"), ("tae_r_p1", "unfiltered"), ("unfiltered", "strict"),
                 ("oracle", "unfiltered")):
        for st in {stage, "dense"}:
            if f"{a}|{st}" in outs and f"{b}|{st}" in outs:
                mine, base = [o["all@5"] for o in outs[f"{a}|{st}"]], [o["all@5"] for o in outs[f"{b}|{st}"]]
                diff = router.paired_bootstrap(mine, base)
                stats[f"{a}_vs_{b}|{st}"] = dict(diff, **router.mcnemar(mine, base), decision=router.gate2(diff))
    types = [q["type"] or hop_type(q["question_id"]) for q in items]
    by_type = None
    if any(types):
        by_type = {}
        for t in sorted({t for t in types if t}):
            idx = [i for i, x in enumerate(types) if x == t]
            by_type[t] = {"questions": len(idx), **{k: sum(v[i]["all@5"] for i in idx) / len(idx) for k, v in outs.items()}}
    report = {"protocol": PROTOCOL, "dataset": dataset, "selection": selection, "selection_sha256": sha256_file(selection_path),
              "frozen_questions_sha256": sha256_file(frozen_questions), "revisions": dict(models.revisions),
              "methods": {k: summarise(v, items) for k, v in outs.items()}, "comparisons": stats, "by_type": by_type,
              "seconds": time.time() - started}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_backbone_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    with (out_dir / "final_backbone_questions.jsonl").open("w", encoding="utf-8") as stream:
        for i, q in enumerate(items):
            stream.write(json.dumps({"question_id": q["question_id"], **{k: v[i] for k, v in outs.items()}}) + "\n")
    return report


def hop_type(qid):
    import re
    m = re.match(r"^(\d)hop", str(qid))
    return f"{m.group(1)}-hop" if m else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "final"))
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--final-labels", type=Path)
    parser.add_argument("--frozen-questions", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--allow-final", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    models = Models(args.device, cache_dir=args.cache_dir or args.output_dir / "cache")
    log = lambda m: print(m, flush=True)
    if args.command == "select":
        selection, summary = run_select(args.corpora_dir, args.output_dir, models, log)
        print(json.dumps({"selection": selection, "allgold@5": {k: round(v["allgold@5"], 4) for k, v in summary.items()},
                          "recall@5": {k: round(v["recall@5"], 4) for k, v in summary.items()}}, indent=1))
    else:
        report = run_final(args.corpora_dir, args.final_labels, args.frozen_questions, args.selection, args.output_dir,
                           models, dataset=args.dataset, allow_final=args.allow_final, log=log)
        print(json.dumps({"methods": report["methods"], "comparisons": report["comparisons"],
                          "by_type": report["by_type"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
