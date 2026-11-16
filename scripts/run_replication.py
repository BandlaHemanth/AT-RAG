"""Replicate the frozen TAE-R v1 pipeline on another dataset (protocol amendment v2.0).

Everything is fixed in advance; nothing is selected or tuned on the new dataset's validation fold:
  * splits: the v1 builder; router_train + validation = |dev.jsonl| (same pool-scale parity as 2Wiki),
    split in the 2Wiki ratio 8,000 : 4,576; e2e_300 / e2e_500 nested in final_eval; seed 20261110
  * corpora: the v1 builder, title exclusion "evaluation_and_heldout"; the topic-fitting population is
    capped at 300,579 paragraphs (the 2Wiki size), first in seeded SHA-256 order, when larger
  * embeddings: the v1 MiniLM encoder and settings (all-MiniLM-L6-v2 @ 1110a243, normalised, batch 128)
  * topic model: H2 settings only (shared 5-D UMAP; HDBSCAN min cluster 100, min samples 10; seed 42)
  * TAE-R: the frozen design; predictors trained on router_train, thresholds chosen out-of-fold there
  * validation results are reported for comparability only; no decision depends on them
  * final: retrieval on final_eval with the saved predictor, run once (--allow-final)

Stages (resumable; a stage whose output directory holds its completion record is skipped):
  splits corpora embed topics validate final

  python -B scripts/run_replication.py --dataset hotpotqa \
      --data-dir <processed_data>/hotpotqa --work-dir /kaggle/working/rep_hotpotqa \
      --stages splits corpora embed topics validate final --allow-final
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "tae-r-protocol-v2.0-replication"
SEED = 20261110
REFERENCE_FOLDS = (8000, 4576)            # 2Wiki router_train, validation
FIT_CAP = 300_579                         # 2Wiki topic-fitting population
E2E_SIZES = (300, 500)
CANDIDATE = "H2"
STAGES = ("splits", "corpora", "embed", "topics", "validate", "final")
DONE = "replication_stage.json"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


splits_mod = _load("build_tae_r_splits", "build_tae_r_splits.py")
corpora_mod = _load("build_tae_r_corpora", "build_tae_r_corpora.py")
prepare_mod = _load("prepare_tae_r_embedding_input", "prepare_tae_r_embedding_input.py")
router = _load("tae_r_router", "tae_r_router.py")
fit = router.fit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return fit.sha256_file(path)


def fold_sizes(n_dev):
    router_train = round(n_dev * REFERENCE_FOLDS[0] / sum(REFERENCE_FOLDS))
    return router_train, n_dev - router_train


def count_lines(path):
    with open(path, encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


class Work:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, stage):
        return self.root / stage

    def done(self, stage):
        return (self.dir(stage) / DONE).is_file()

    def record(self, stage):
        return json.loads((self.dir(stage) / DONE).read_text(encoding="utf-8"))

    def finish(self, stage, record):
        record = dict(record, stage=stage, protocol=PROTOCOL, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (self.dir(stage) / DONE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return record

    def frozen(self):
        """The 'frozen' record the v1 code verifies: this replication's own corpus and embedding hashes."""
        corpora, embed = self.record("corpora"), self.record("embed")
        return {"corpus_manifest": corpora["manifest_sha256"],
                "embeddings.npy": embed["artifacts"]["embeddings.npy"]["sha256"],
                "document_ids.jsonl": embed["artifacts"]["document_ids.jsonl"]["sha256"]}


def stage_splits(work, data_dir, log):
    n_dev = count_lines(Path(data_dir) / "dev.jsonl")
    router_train, validation = fold_sizes(n_dev)
    out = work.dir("splits") / "splits"
    work.dir("splits").mkdir(exist_ok=True)
    manifest = splits_mod.build(Path(data_dir), out, SEED, router_train, validation, E2E_SIZES)
    log(f"splits: router_train {router_train}, validation {validation}, "
        f"final_eval {manifest['split_files']['final_eval']['count']}")
    return work.finish("splits", {"manifest_sha256": manifest["manifest_sha256"], "dev_questions": n_dev,
                                  "router_train": router_train, "validation": validation,
                                  "split_files": manifest["split_files"], "input_sha256": manifest["input_sha256"]})


def stage_corpora(work, data_dir, log):
    splits = work.record("splits")
    out = work.dir("corpora") / "corpora"
    work.dir("corpora").mkdir(exist_ok=True)
    manifest = corpora_mod.build(Path(data_dir), work.dir("splits") / "splits", out, mode="build",
                                 title_exclusion="evaluation_and_heldout",
                                 frozen_manifest_sha256=splits["manifest_sha256"], expected_text_population=None,
                                 fit_cap=FIT_CAP, cap_seed=SEED)
    counts = manifest["counts"]
    log(f"corpora: fit population {counts['topic_fit_population']} "
        f"(before cap {counts['topic_fit_population_after_title_exclusion']}), "
        f"validation corpus {counts['retrieval_validation_unique_texts']}, final corpus {counts['retrieval_final_unique_texts']}")
    return work.finish("corpora", {"manifest_sha256": manifest["manifest_sha256"], "counts": counts,
                                   "files": {k: {"count": v["count"], "sha256": v["sha256"]}
                                             for k, v in manifest["files"].items() if isinstance(v, dict) and "count" in v}})


_EMBED = {}


def embed_module():
    if "m" not in _EMBED:
        _EMBED["m"] = _load("embed_full_2wiki_corpus", "embed_full_2wiki_corpus.py")
    return _EMBED["m"]


def stage_embed(work, log, device, encoder_factory=None, batch_size=128):
    embed = embed_module()
    corpora = work.record("corpora")
    inp = work.dir("embed") / "input"
    work.dir("embed").mkdir(exist_ok=True)
    if not (inp / "summary.json").is_file():
        prepare_mod.prepare(work.dir("corpora") / "corpora", inp, frozen_manifest_sha256=corpora["manifest_sha256"])
    summary = json.loads((inp / "summary.json").read_text(encoding="utf-8"))
    out = work.dir("embed") / "embeddings"
    manifest = embed.run_job(inp / "documents.jsonl", out, summary_path=inp / "summary.json", batch_size=batch_size,
                             device=device, resume=(out.with_name(out.name + ".inprogress")).exists(),
                             expected_sha256=summary["documents_sha256"], expected_count=summary["unique_paragraphs"],
                             encoder_factory=encoder_factory)
    log(f"embed: {manifest['processed_count']} paragraphs")
    return work.finish("embed", {"processed_count": manifest["processed_count"], "artifacts": manifest["artifacts"],
                                 "embedding_data_sha256": manifest["embedding_data_sha256"],
                                 "documents_sha256": summary["documents_sha256"]})


def stage_topics(work, log, device, config=None, encode=None):
    out = work.dir("topics") / "topics"
    work.dir("topics").mkdir(exist_ok=True)
    fit.run(work.dir("corpora") / "corpora", work.dir("embed") / "embeddings", out, list(fit.STAGES), [CANDIDATE],
            frozen=work.frozen(), config=config or fit.CONFIG, encode=encode, device=device, log=log)
    hashes = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("stage_manifest.json"))}
    return work.finish("topics", {"candidate": CANDIDATE, "stage_manifests": hashes})


def stage_validate(work, log):
    d = work.dir("validate")
    d.mkdir(exist_ok=True)
    report, rows, choices = router.run(work.dir("corpora") / "corpora", work.dir("embed") / "embeddings",
                                       work.dir("topics") / "topics", CANDIDATE, frozen=work.frozen(), log=log)
    model = report.pop("router_model")
    import joblib
    import sklearn
    model_path = d / f"tae_r_router_{CANDIDATE}.joblib"
    joblib.dump(dict(model, sklearn=sklearn.__version__, protocol=PROTOCOL), model_path)
    (d / "validation_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    val = [r for r in rows if r["split"] == "validation"]
    with (d / "validation_report.questions.jsonl").open("w", encoding="utf-8") as stream:
        for i, r in enumerate(val):
            stream.write(json.dumps({"question_id": r["question_id"], "routes": r["routes"],
                                     "chosen": {m: choices[m][i] for m in choices}, "outcomes": r["outcomes"],
                                     "features": r["features"]}, sort_keys=True) + "\n")
    return work.finish("validate", {"router_model_sha256": sha256_file(model_path),
                                    "report_sha256": sha256_file(d / "validation_report.json"),
                                    "thresholds": report["thresholds"],
                                    "allgold@5": {m: s["allgold@5"] for m, s in report["methods"].items()},
                                    "gate2": report["gate2"]})


def stage_final(work, log, device, allow_final, config=None, encode=None):
    require(allow_final, "the final stage needs --allow-final (final evaluation, run once)")
    require(work.done("validate"), "the final stage needs the saved predictor from the validate stage")
    d = work.dir("final")
    d.mkdir(exist_ok=True)
    corpora, emb, topics = work.dir("corpora") / "corpora", work.dir("embed") / "embeddings", work.dir("topics") / "topics"
    assign = d / "assign"
    fit.run(corpora, emb, assign, ["assign_final"], [CANDIDATE], frozen=work.frozen(), config=config or fit.CONFIG,
            encode=encode, device=device, log=log, allow_final=True, source_dir=topics)
    model_path = work.dir("validate") / f"tae_r_router_{CANDIDATE}.joblib"
    require(sha256_file(model_path) == work.record("validate")["router_model_sha256"], "saved predictor changed")
    report, rows, choices = router.run_final(corpora, emb, topics, assign, CANDIDATE, model_path,
                                             frozen=work.frozen(), log=log)
    (d / f"final_retrieval_{CANDIDATE}.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                                                         encoding="utf-8")
    with (d / f"final_retrieval_{CANDIDATE}.questions.jsonl").open("w", encoding="utf-8") as stream:
        for i, r in enumerate(rows):
            stream.write(json.dumps({"question_id": r["question_id"], "routes": r["routes"],
                                     "chosen": {m: choices[m][i] for m in choices}, "outcomes": r["outcomes"],
                                     "features": r["features"]}, sort_keys=True) + "\n")
    return work.finish("final", {"report_sha256": sha256_file(d / f"final_retrieval_{CANDIDATE}.json"),
                                 "allgold@5": {m: s["allgold@5"] for m, s in report["methods"].items()},
                                 "comparisons": report["comparisons"]})


def run(dataset, data_dir, work_dir, stages, *, device="cuda", allow_final=False, log=print,
        config=None, encode=None, encoder_factory=None):
    require(set(stages) <= set(STAGES), f"stages must be among {STAGES}")
    work = Work(work_dir)
    (work.root / "replication.json").write_text(json.dumps({
        "dataset": dataset, "protocol": PROTOCOL, "seed": SEED, "reference_folds": REFERENCE_FOLDS,
        "fit_cap": FIT_CAP, "candidate": CANDIDATE, "e2e_sizes": E2E_SIZES,
        "script_sha256": sha256_file(Path(__file__))}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    results = {}
    for stage in STAGES:
        if stage not in stages:
            continue
        if work.done(stage):
            log(f"{stage}: already complete, skipped")
            results[stage] = work.record(stage)
            continue
        started = time.time()
        if stage == "splits":
            results[stage] = stage_splits(work, data_dir, log)
        elif stage == "corpora":
            results[stage] = stage_corpora(work, data_dir, log)
        elif stage == "embed":
            results[stage] = stage_embed(work, log, device, encoder_factory)
        elif stage == "topics":
            results[stage] = stage_topics(work, log, device, config, encode)
        elif stage == "validate":
            results[stage] = stage_validate(work, log)
        elif stage == "final":
            results[stage] = stage_final(work, log, device, allow_final, config, encode)
        log(f"{stage}: done in {time.time() - started:.0f} s")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="processed_data/<dataset> with train/dev files")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-final", action="store_true")
    args = parser.parse_args(argv)
    results = run(args.dataset, args.data_dir, args.work_dir, args.stages, device=args.device,
                  allow_final=args.allow_final, log=lambda m: print(m, flush=True))
    brief = {s: {k: v for k, v in r.items() if k in ("allgold@5", "thresholds", "gate2", "comparisons", "counts",
                                                       "router_train", "validation", "processed_count")}
             for s, r in results.items()}
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
