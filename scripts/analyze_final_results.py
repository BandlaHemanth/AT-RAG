"""Descriptive analysis of the frozen final results (no new retrieval, no new LLM call).

Reads the per-question outputs already produced once on final_eval (section 18: final_retrieval_<cand>
.questions.jsonl and .json; section 20: the end-to-end records.jsonl) and writes:

  analysis.json   gold availability per method (all gold paragraphs inside the searched set),
                  gold-topic spread, TAE-R route use and route adequacy, oracle-gap anatomy,
                  help/hurt anatomy and examples, results by 2Wiki question type (from the raw
                  dev.json `type`, joined on question_id = `_id`), end-to-end EM conditioned on
                  first-retrieval AllGold@5, end-to-end failures, iterations
  tables.tex      LaTeX (booktabs) tables for the thesis
  figures         cost_accuracy_<cand>.pdf/.png, by_type_<cand>.pdf/.png

Nothing here selects or tunes anything; examples are picked by a fixed hash order.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"strict": "Strict (original AT-RAG)", "fixed_three": "Fixed three", "tae_v1": "TAE V1",
          "tae_v2": "TAE V2", "tae_r": "TAE-R", "tae_r_p1": "TAE-R-P1", "unfiltered": "Unfiltered",
          "oracle": "Oracle (upper bound)"}
ORDER = ("strict", "fixed_three", "tae_v1", "tae_v2", "tae_r", "tae_r_p1", "unfiltered", "oracle")
SYSTEM_LABELS = {"original": "Original AT-RAG (strict)", "nofilter": "AT-RAG without filter", "tae_r": "AT-RAG with TAE-R"}
TYPE_LABELS = {"comparison": "Comparison", "inference": "Inference", "compositional": "Compositional",
               "bridge_comparison": "Bridge comparison"}
N_EXAMPLES = 5


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fixed_order(ids, salt):
    return sorted(ids, key=lambda q: hashlib.sha256(f"{salt}:{q}".encode()).hexdigest())


def pct(x):
    return round(100.0 * x, 2)


# ------------------------------------------------------------------------------ retrieval analysis

def available(row, method):
    """All gold paragraphs lie inside the set the method searched (its AllGold upper bound)."""
    allowed = row["routes"][row["chosen"][method]]
    return allowed is None or set(row["routes"]["oracle"]) <= set(allowed)


def all5(row, method):
    return row["outcomes"][row["chosen"][method]]["all@5"]


def retrieval_analysis(rows, stats, questions=None, titles=None, types=None):
    n = len(rows)
    methods = [m for m in ORDER if m in rows[0]["chosen"]]
    out = {"questions": n}
    out["availability"] = {m: pct(sum(available(r, m) for r in rows) / n) for m in methods}
    out["allgold@5"] = {m: pct(sum(all5(r, m) for r in rows) / n) for m in methods}
    out["allgold@5_given_available"] = {}
    for m in methods:
        av = [r for r in rows if available(r, m)]
        out["allgold@5_given_available"][m] = pct(sum(all5(r, m) for r in av) / len(av)) if av else None
    spread = Counter(len(r["routes"]["oracle"]) for r in rows)
    out["gold_topics_per_question"] = {str(k): spread[k] for k in sorted(spread)}
    out["gold_in_outlier_topic"] = pct(sum(-1 in r["routes"]["oracle"] for r in rows) / n)
    out["strict_label_is_outlier"] = pct(sum(r["routes"]["strict"] == [-1] for r in rows) / n)

    # TAE-R routes: how often, how adequate, and accuracy per route against unfiltered on the same questions.
    routes = {}
    for route in sorted({r["chosen"]["tae_r"] for r in rows}):
        group = [r for r in rows if r["chosen"]["tae_r"] == route]
        entry = {"questions": len(group), "share": pct(len(group) / n),
                 "tae_r_allgold@5": pct(sum(all5(r, "tae_r") for r in group) / len(group)),
                 "unfiltered_allgold@5": pct(sum(all5(r, "unfiltered") for r in group) / len(group))}
        if route == "unfiltered":
            entry["S3_would_have_been_adequate"] = pct(sum(available(dict(r, chosen={"x": "S3"}), "x") for r in group) / len(group))
        else:
            entry["route_adequate"] = pct(sum(available(r, "tae_r") for r in group) / len(group))
        routes[route] = entry
    out["tae_r_routes"] = routes

    # Oracle gap: questions the oracle topic filter gets fully right but unfiltered search does not.
    gap = [r for r in rows if all5(r, "oracle") and not all5(r, "unfiltered")]
    out["oracle_gap"] = {"questions": len(gap), "share": pct(len(gap) / n),
                         "tae_r_route": dict(Counter(r["chosen"]["tae_r"] for r in gap)),
                         "tae_r_solves": sum(all5(r, "tae_r") for r in gap),
                         "gold_topics": dict(Counter(len(r["routes"]["oracle"]) for r in gap)),
                         "examples": [example(r, questions, titles) for r in pick(gap, "oracle-gap")]}

    anatomy = {}
    for ref in ("strict", "unfiltered"):
        helped = [r for r in rows if all5(r, "tae_r") and not all5(r, ref)]
        hurt = [r for r in rows if not all5(r, "tae_r") and all5(r, ref)]
        anatomy[f"tae_r_vs_{ref}"] = {
            "helped": len(helped), "hurt": len(hurt),
            "helped_by_tae_r_route": dict(Counter(r["chosen"]["tae_r"] for r in helped)),
            "hurt_by_tae_r_route": dict(Counter(r["chosen"]["tae_r"] for r in hurt)),
            "hurt_route_inadequate": sum(not available(r, "tae_r") for r in hurt),
            "helped_examples": [example(r, questions, titles) for r in pick(helped, f"help-{ref}")],
            "hurt_examples": [example(r, questions, titles) for r in pick(hurt, f"hurt-{ref}")]}
    if "strict" in methods:
        strict_failed_unavailable = [r for r in rows if not available(r, "strict")]
        anatomy["strict_gold_unavailable"] = {"questions": len(strict_failed_unavailable),
                                              "share": pct(len(strict_failed_unavailable) / n)}
    out["anatomy"] = anatomy

    if types:
        by_type = {}
        for t in sorted({types.get(r["question_id"]) for r in rows} - {None}):
            group = [r for r in rows if types.get(r["question_id"]) == t]
            entry = {"questions": len(group),
                     "allgold@5": {m: pct(sum(all5(r, m) for r in group) / len(group)) for m in methods},
                     "availability": {m: pct(sum(available(r, m) for r in group) / len(group)) for m in methods},
                     "gold_topics_mean": round(float(np.mean([len(r["routes"]["oracle"]) for r in group])), 3),
                     "tae_r_routes": dict(Counter(r["chosen"]["tae_r"] for r in group))}
            for ref in ("strict", "unfiltered"):
                mine = [all5(r, "tae_r") for r in group]
                base = [all5(r, ref) for r in group]
                diff = stats.paired_bootstrap(mine, base)
                entry[f"tae_r_vs_{ref}"] = {"difference_pt": pct(diff["difference"]),
                                            "ci95_pt": [pct(x) for x in diff["ci95"]], **stats.mcnemar(mine, base)}
            by_type[t] = entry
        out["by_type"] = by_type
        out["type_coverage"] = sum(r["question_id"] in types for r in rows)
    return out


def pick(rows, salt):
    by_id = {r["question_id"]: r for r in rows}
    return [by_id[q] for q in fixed_order(by_id, salt)[:N_EXAMPLES]]


def example(row, questions, titles):
    q = (questions or {}).get(row["question_id"], {})
    return {"question_id": row["question_id"], "question": q.get("question_text"),
            "answer": [s for o in q.get("answers_objects", []) for s in o.get("spans", [])],
            "gold_titles": [titles.get(d) for d in q.get("gold_doc_ids", [])] if titles else None,
            "gold_topics": row["routes"]["oracle"], "strict_topic": row["routes"]["strict"],
            "tae_r_route": row["chosen"]["tae_r"], "tae_r_topics": row["routes"][row["chosen"]["tae_r"]],
            "allgold@5": {m: all5(row, m) for m in ("strict", "tae_r", "unfiltered", "oracle") if m in row["chosen"]}}


# ------------------------------------------------------------------------------ end-to-end analysis

def e2e_analysis(records, stats, types=None):
    systems = [s for s in SYSTEM_LABELS if any(r["system"] == s for r in records)]
    by = {s: {r["question_id"]: r for r in records if r["system"] == s} for s in systems}
    ids = sorted(set.intersection(*(set(v) for v in by.values())))
    out = {"questions": len(ids), "systems": {}}
    for s in systems:
        rs = [by[s][q] for q in ids]
        entry = {"exact_match": pct(np.mean([r["exact_match"] for r in rs])), "f1": pct(np.mean([r["f1"] for r in rs]))}
        for flag in (1, 0):
            group = [r for r in rs if r["first_allgold@5"] == flag]
            entry[f"em_given_first_allgold={flag}"] = pct(np.mean([r["exact_match"] for r in group])) if group else None
            entry[f"questions_first_allgold={flag}"] = len(group)
        partial = [r for r in rs if 0 < r["first_hits"] < r["n_gold"]]
        entry["em_given_partial_gold"] = pct(np.mean([r["exact_match"] for r in partial])) if partial else None
        entry["questions_partial_gold"] = len(partial)
        errors = [r for r in rs if r["error"]]
        entry["errors"] = {"count": len(errors),
                           "kinds": dict(Counter(r["error"].split(":")[0] for r in errors)),
                           "question_ids": [r["question_id"] for r in errors]}
        entry["iterations"] = dict(sorted(Counter(r["iterations"] for r in rs).items()))
        entry["hit_iteration_cap"] = pct(np.mean([r["iterations"] >= 5 for r in rs]))
        entry["rewrites"] = pct(np.mean([len(r["retrievals"]) > 1 for r in rs]))
        if s == "tae_r":
            entry["first_route"] = dict(Counter((r["retrievals"][0].get("route") if r["retrievals"] else None) for r in rs))
        out["systems"][s] = entry
    shared_errors = set.intersection(*(set(out["systems"][s]["errors"]["question_ids"]) for s in systems))
    out["errors_on_same_question_all_systems"] = len(shared_errors)
    if "tae_r" in by and "original" in by:
        helped = [q for q in ids if by["tae_r"][q]["exact_match"] and not by["original"][q]["exact_match"]]
        hurt = [q for q in ids if not by["tae_r"][q]["exact_match"] and by["original"][q]["exact_match"]]
        def retrieval_explains(qs):
            return dict(Counter("tae_r_first_retrieval_better" if by["tae_r"][q]["first_allgold@5"] > by["original"][q]["first_allgold@5"]
                                else "same_first_retrieval" if by["tae_r"][q]["first_allgold@5"] == by["original"][q]["first_allgold@5"]
                                else "original_first_retrieval_better" for q in qs))
        out["tae_r_vs_original_em"] = {"helped": len(helped), "hurt": len(hurt),
                                       "helped_by_first_retrieval": retrieval_explains(helped),
                                       "hurt_by_first_retrieval": retrieval_explains(hurt)}
    if types:
        out["by_type"] = {}
        for t in sorted({types.get(q) for q in ids} - {None}):
            group = [q for q in ids if types.get(q) == t]
            out["by_type"][t] = {"questions": len(group),
                                 **{s: pct(np.mean([by[s][q]["exact_match"] for q in group])) for s in systems}}
    return out


# ------------------------------------------------------------------------------ LaTeX and figures

def latex_retrieval(report, cand):
    m = report["methods"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Final retrieval results ({cand}, {report['methods']['unfiltered']['questions']:,} final\_eval questions, "
             r"$k=5$). Searched share is the mean fraction of the corpus scored per question.}",
             rf"\label{{tab:final-retrieval-{cand.lower()}}}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Method & AllGold@5 & Recall@5 & AllGold@10 & Searched share & Similarity comp. \\", r"\midrule"]
    for key in ORDER:
        if key not in m:
            continue
        s = m[key]
        name = LABELS[key] if key != "tae_r" else r"\textbf{TAE-R}"
        lines.append(f"{name} & {pct(s['allgold@5']):.2f}\\% & {pct(s['recall@5']):.2f}\\% & {pct(s['allgold@10']):.2f}\\% & "
                     f"{s['mean_searched_share']:.3f} & {round(s['mean_similarity_computations']):,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_comparisons(report, cand):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Paired comparisons on AllGold@5 ({cand}): difference, bootstrap 95\% CI (10{{,}}000 resamples), "
             r"questions helped/hurt, exact McNemar $p$.}",
             rf"\label{{tab:final-comparisons-{cand.lower()}}}",
             r"\begin{tabular}{lrcrr}", r"\toprule",
             r"Comparison & $\Delta$ (pt) & 95\% CI & Helps/hurts & $p$ \\", r"\midrule"]
    for key in ("tae_r_vs_strict", "tae_r_vs_unfiltered", "tae_r_p1_vs_unfiltered", "tae_v2_vs_strict",
                "strict_vs_unfiltered", "oracle_vs_unfiltered"):
        c = report["comparisons"].get(key)
        if not c:
            continue
        a, b = key.split("_vs_")
        lo, hi = c["ci95"]
        lines.append(f"{LABELS.get(a, a)} vs {LABELS.get(b, b).split(' (')[0].lower()} & {pct(c['difference']):+.2f} & "
                     f"[{pct(lo):+.2f}, {pct(hi):+.2f}] & {c['helps']:,}/{c['hurts']:,} & {fmt_p(c['exact_p'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def fmt_p(p):
    if p == 0 or p < 1e-200:
        return r"$<10^{-200}$"
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return f"{p:.3f}"


def latex_by_type(analysis, cand):
    by = analysis.get("by_type")
    if not by:
        return ""
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{AllGold@5 by 2WikiMultiHopQA question type ({cand}, final\_eval). $\Delta$: TAE-R minus strict, "
             r"with bootstrap 95\% CI.}", rf"\label{{tab:by-type-{cand.lower()}}}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Type & $n$ & Strict & TAE-R & Unfiltered & Oracle & $\Delta$ vs strict \\", r"\midrule"]
    for t, e in by.items():
        a, d = e["allgold@5"], e["tae_r_vs_strict"]
        lines.append(f"{TYPE_LABELS.get(t, t)} & {e['questions']:,} & {a['strict']:.2f} & {a['tae_r']:.2f} & {a['unfiltered']:.2f} & "
                     f"{a['oracle']:.2f} & {d['difference_pt']:+.2f} [{d['ci95_pt'][0]:+.2f}, {d['ci95_pt'][1]:+.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_availability(analysis, cand):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Gold availability ({cand}): share of questions whose gold paragraphs all lie in the searched set "
             r"(the method's AllGold ceiling), and AllGold@5 overall and when gold is available.}",
             rf"\label{{tab:availability-{cand.lower()}}}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Method & Gold available & AllGold@5 & AllGold@5 $\mid$ available \\", r"\midrule"]
    for key in ORDER:
        if key not in analysis["availability"]:
            continue
        g = analysis["allgold@5_given_available"][key]
        lines.append(f"{LABELS[key]} & {analysis['availability'][key]:.2f}\\% & {analysis['allgold@5'][key]:.2f}\\% & "
                     f"{'--' if g is None else f'{g:.2f}' + chr(92) + '%'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_e2e(e2e):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{End-to-end QA on {e2e['questions']} final\_eval questions (same LLM, prompts and loop for all systems). "
             r"EM conditioned on whether the first retrieval contained all gold paragraphs.}",
             r"\label{tab:e2e-analysis}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"System & EM & F1 & EM $\mid$ gold retrieved & EM $\mid$ not retrieved & Parse failures \\", r"\midrule"]
    for s, e in e2e["systems"].items():
        g1, g0 = e["em_given_first_allgold=1"], e["em_given_first_allgold=0"]
        lines.append(f"{SYSTEM_LABELS[s]} & {e['exact_match']:.1f}\\% & {e['f1']:.1f}\\% & "
                     f"{'--' if g1 is None else f'{g1:.1f}'} ($n$={e['questions_first_allgold=1']}) & "
                     f"{'--' if g0 is None else f'{g0:.1f}'} ($n$={e['questions_first_allgold=0']}) & {e['errors']['count']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def figures(report, analysis, cand, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = report["methods"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for key in ORDER:
        if key not in m:
            continue
        x, y = m[key]["mean_searched_share"], 100 * m[key]["allgold@5"]
        style = dict(marker="*", s=180, color="#c0392b", zorder=3) if key == "tae_r" else \
            dict(marker="D", s=50, color="#7f8c8d", zorder=2) if key == "oracle" else dict(marker="o", s=50, color="#2c3e50", zorder=2)
        ax.scatter([x], [y], **style)
        ax.annotate(LABELS[key].replace(" (original AT-RAG)", "\n(original AT-RAG)").replace(" (upper bound)", ""),
                    (x, y), textcoords="offset points", xytext=(6, -3), fontsize=7.5)
    ax.set_xlabel("Mean share of the corpus searched per question")
    ax.set_ylabel("AllGold@5 (%)")
    ax.set_xlim(0, 1.18)
    ax.grid(alpha=0.25)
    ax.set_title(f"Cost and accuracy of topic routing ({cand}, final_eval)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cost_accuracy_{cand}.{ext}", dpi=200)
    plt.close(fig)
    by = analysis.get("by_type")
    if by:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        keys = ("strict", "tae_r", "unfiltered", "oracle")
        colors = ("#95a5a6", "#c0392b", "#2c3e50", "#bdc3c7")
        width = 0.2
        for i, (key, color) in enumerate(zip(keys, colors)):
            xs = [j + (i - 1.5) * width for j in range(len(by))]
            ax.bar(xs, [e["allgold@5"][key] for e in by.values()], width, label=LABELS[key].split(" (")[0], color=color)
        ax.set_xticks(range(len(by)))
        ax.set_xticklabels([f"{TYPE_LABELS.get(t, t)}\n(n={e['questions']:,})" for t, e in by.items()], fontsize=8)
        ax.set_ylabel("AllGold@5 (%)")
        ax.legend(fontsize=7.5, ncol=4, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"by_type_{cand}.{ext}", dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------------------ inputs and main

def load_types(raw_dev):
    data = json.loads(Path(raw_dev).read_text(encoding="utf-8"))
    return {r["_id"]: r["type"] for r in data}


def load_titles(corpus_path, wanted):
    titles = {}
    with open(corpus_path, encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["doc_id"] in wanted:
                t = record.get("titles") or record.get("title")
                titles[record["doc_id"]] = t[0] if isinstance(t, list) and len(t) == 1 else t
    return titles


def run(final_dir, candidates, e2e_records, corpora_dir, raw_dev, out_dir, log=print):
    stats = _load("tae_r_router", "tae_r_router.py")
    types = load_types(raw_dev) if raw_dev else None
    questions = None
    if corpora_dir:
        questions = {r["question_id"]: r for r in read_jsonl(Path(corpora_dir) / "questions_final_eval.jsonl")}
    titles = None
    if corpora_dir and questions:
        wanted = {d for q in questions.values() for d in q.get("gold_doc_ids", [])}
        titles = load_titles(Path(corpora_dir) / "retrieval_final.jsonl", wanted)
    result, tex = {"inputs": {}}, []
    for cand in candidates:
        report_path = Path(final_dir) / f"final_retrieval_{cand}.json"
        rows_path = Path(final_dir) / f"final_retrieval_{cand}.questions.jsonl"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = read_jsonl(rows_path)
        for p in (report_path, rows_path):
            result["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        analysis = retrieval_analysis(rows, stats, questions, titles, types)
        result[cand] = analysis
        tex += [f"% ---- {cand}", latex_retrieval(report, cand), latex_comparisons(report, cand),
                latex_availability(analysis, cand), latex_by_type(analysis, cand)]
        figures(report, analysis, cand, out_dir)
        log(f"{cand}: availability {analysis['availability']}")
    if e2e_records:
        records = read_jsonl(e2e_records)
        result["inputs"][Path(e2e_records).name] = hashlib.sha256(Path(e2e_records).read_bytes()).hexdigest()
        result["end_to_end"] = e2e_analysis(records, stats, types)
        tex.append(latex_e2e(result["end_to_end"]))
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "tables.tex").write_text("% Generated by scripts/analyze_final_results.py; needs \\usepackage{booktabs}\n"
                                        + "\n".join(tex), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-dir", type=Path, required=True, help="holds final_retrieval_<cand>.json/.questions.jsonl")
    parser.add_argument("--candidates", nargs="+", default=["H2", "K2"])
    parser.add_argument("--e2e-records", type=Path, help="end-to-end records.jsonl (section 20)")
    parser.add_argument("--corpora-dir", type=Path, help="for question texts and gold titles in examples")
    parser.add_argument("--raw-dev", type=Path, help="raw 2WikiMultiHopQA dev.json (question types)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.final_dir, args.candidates, args.e2e_records, args.corpora_dir, args.raw_dev, args.output_dir)
    brief = {c: {k: result[c][k] for k in ("availability", "tae_r_routes", "oracle_gap")} for c in args.candidates}
    for c in args.candidates:
        brief[c]["oracle_gap"] = {k: v for k, v in brief[c]["oracle_gap"].items() if k != "examples"}
        if "by_type" in result[c]:
            brief[c]["by_type_allgold@5"] = {t: dict(n=e["questions"], **{m: e["allgold@5"][m] for m in ("strict", "tae_r", "unfiltered", "oracle")})
                                             for t, e in result[c]["by_type"].items()}
    if "end_to_end" in result:
        brief["end_to_end"] = {s: {k: v for k, v in e.items() if k not in ("errors",)} | {"errors": e["errors"]["kinds"]}
                               for s, e in result["end_to_end"]["systems"].items()}
        brief["end_to_end"]["by_type"] = result["end_to_end"].get("by_type")
        brief["end_to_end"]["tae_r_vs_original_em"] = result["end_to_end"].get("tae_r_vs_original_em")
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
