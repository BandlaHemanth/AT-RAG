"""Build the frozen TAE-R corpora from the frozen question splits.

Reads the processed 2WikiMultiHopQA JSONL files and the frozen split files
written by scripts/build_tae_r_splits.py. Fits no model and computes no
embeddings. Modes:

  measure  print the population counts and title-overlap diagnostics only;
           writes nothing.
  build    write the corpora, question files and a manifest to a fresh
           directory.

Corpora (docs/tae_r_protocol_v1.md, section 3):
  topic_fit_population   unique paragraph texts of topic_fit questions that occur
                         in no dev.jsonl / subsampled / router_train / validation
                         context, minus texts whose title also occurs in the
                         excluded title set (see --title-exclusion)
  retrieval_final        every unique paragraph of every dev.jsonl context
  retrieval_validation   every unique paragraph of router_train + validation contexts

Text identity is exact UTF-8 paragraph_text (null/empty/whitespace-only ignored);
doc_id is its SHA-256 hex digest, so equal text has the same id in every corpus.
Title comparison is Unicode NFC, case-folded, whitespace-collapsed. Question files
carry gold_doc_ids for evaluation only; support labels never select any corpus.

Standard library only. Example:
  python -B scripts/build_tae_r_corpora.py --mode measure \
      --data-dir /kaggle/working/data/processed_data/2wikimultihopqa \
      --splits-dir /kaggle/working/tae_r_splits_v1_run01
"""
import argparse
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

PROTOCOL = "tae-r-protocol-v1"
FROZEN_MANIFEST_SHA256 = "44e79c259e5f0ac55eee8f83fb65493b70432fb276a80809621919ec35ba38ce"
EXPECTED_TEXT_POPULATION = 312555
NAMES = {"train": "train.jsonl", "original_dev": "dev.jsonl",
         "development": "dev_subsampled.jsonl", "heldout": "test_subsampled.jsonl"}
SPLIT_KEYS = ("exposed", "final_eval", "router_train", "validation", "topic_fit")
TITLE_MODES = ("evaluation", "evaluation_and_heldout")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def norm_title(title):
    return " ".join(unicodedata.normalize("NFC", title).casefold().split())


class Pool:
    """Unique nonempty paragraph texts with every title they appear under."""

    def __init__(self):
        self.texts = {}
        self.titles = {}

    def add(self, ctx):
        text = ctx.get("paragraph_text")
        if not isinstance(text, str) or not text.strip():
            return None
        key = hashlib.sha256(text.encode("utf-8")).digest()
        if key not in self.texts:
            self.texts[key] = text
            self.titles[key] = []
        title = ctx.get("title")
        if isinstance(title, str) and title not in self.titles[key]:
            self.titles[key].append(title)
        return key

    def all_titles(self):
        return {t for titles in self.titles.values() for t in titles}


def rows(path, file_hashes):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw.decode("utf-8"))
            ok = (isinstance(row, dict) and isinstance(row.get("question_id"), str)
                  and row["question_id"].strip() and isinstance(row.get("contexts"), list))
            require(ok, f"{path.name}: line {line}: invalid question record")
            yield row
    file_hashes[path.name] = digest.hexdigest()


def question_record(row, pool):
    """Question fields plus context/gold doc ids; adds the contexts to the pool."""
    context_ids, gold, blank = [], [], 0
    for ctx in row["contexts"]:
        key = pool.add(ctx)
        if key is None:
            blank += 1
            continue
        doc = key.hex()
        context_ids.append(doc)
        if ctx.get("is_supporting") is True and doc not in gold:
            gold.append(doc)
    record = {k: v for k, v in row.items() if k != "contexts"}
    record.update(context_doc_ids=context_ids, gold_doc_ids=gold, blank_contexts=blank)
    return record


def load_splits(splits_dir, frozen_manifest_sha256):
    data = (splits_dir / "manifest.json").read_bytes()
    require(sha256_hex(data) == frozen_manifest_sha256,
            "Split manifest hash differs from the frozen protocol record")
    manifest = json.loads(data.decode("utf-8"))
    ids = {}
    for key in SPLIT_KEYS:
        info = manifest["split_files"][key]
        raw = (splits_dir / info["file"]).read_bytes()
        require(sha256_hex(raw) == info["sha256"], f"{info['file']}: hash differs from manifest")
        listed = raw.decode("utf-8").split()
        require(len(listed) == info["count"] == len(set(listed)), f"{info['file']}: bad count")
        ids[key] = set(listed)
    return manifest, ids


def write_jsonl(path, records):
    digest, count = hashlib.sha256(), 0
    with path.open("xb") as stream:
        for record in records:
            line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
            count += 1
    return {"file": path.name, "count": count, "sha256": digest.hexdigest()}


def write_lines(path, lines):
    data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    with path.open("xb") as stream:
        stream.write(data)
    return {"file": path.name, "count": len(lines), "sha256": sha256_hex(data)}


def collect(data_dir, splits, names):
    paths = {k: data_dir / v for k, v in names.items()}
    for key, path in paths.items():
        require(path.is_file(), f"Missing input file for {key}: {path.name}")
    hashes = {}
    final_pool, sub_pool, held_eval_pool = Pool(), Pool(), Pool()
    final_questions = []
    for row in rows(paths["original_dev"], hashes):
        record = question_record(row, final_pool)
        if row["question_id"] in splits["final_eval"]:
            final_questions.append(record)
    for row in rows(paths["development"], hashes):
        for ctx in row["contexts"]:
            sub_pool.add(ctx)
    for row in rows(paths["heldout"], hashes):
        for ctx in row["contexts"]:
            held_eval_pool.add(ctx)
    held_pool, fit_pool = Pool(), Pool()
    held_questions = {"router_train": [], "validation": []}
    seen = set()
    for row in rows(paths["train"], hashes):
        qid = row["question_id"]
        require(qid not in seen, "train: duplicate question_id")
        seen.add(qid)
        fold = ("router_train" if qid in splits["router_train"]
                else "validation" if qid in splits["validation"] else None)
        if fold:
            held_questions[fold].append(question_record(row, held_pool))
        else:
            require(qid in splits["topic_fit"], f"train question {qid} is in no frozen split")
            for ctx in row["contexts"]:
                fit_pool.add(ctx)
    require(len(final_questions) == len(splits["final_eval"]), "final_eval ids missing from dev.jsonl")
    require(len(held_questions["router_train"]) == len(splits["router_train"])
            and len(held_questions["validation"]) == len(splits["validation"])
            and len(seen) == sum(len(splits[k]) for k in ("router_train", "validation", "topic_fit")),
            "train ids differ from the frozen splits")
    return hashes, final_pool, sub_pool, held_eval_pool, held_pool, fit_pool, final_questions, held_questions


def build(data_dir, splits_dir, output_dir, *, mode, title_exclusion, frozen_manifest_sha256,
          expected_text_population, names=None):
    require(mode in ("measure", "build"), "mode must be measure or build")
    require(title_exclusion in TITLE_MODES, f"title_exclusion must be one of {TITLE_MODES}")
    names = dict(NAMES, **(names or {}))
    if mode == "build":
        require(output_dir is not None, "--output-dir is required for build")
        require(not os.path.lexists(output_dir), "Output directory exists; use a fresh one")
        require(not output_dir.resolve().is_relative_to(data_dir.resolve()),
                "Output directory must be outside the dataset directory")
    split_manifest, splits = load_splits(splits_dir, frozen_manifest_sha256)
    (hashes, final_pool, sub_pool, held_eval_pool, held_pool, fit_pool,
     final_questions, held_questions) = collect(data_dir, splits, names)
    input_hashes = {key: hashes[names[key]] for key in names}
    require(input_hashes == split_manifest["input_sha256"], "Input files differ from the split manifest")

    eval_keys = set(final_pool.texts) | set(sub_pool.texts) | set(held_eval_pool.texts)
    held_keys = set(held_pool.texts)
    fit_keys = set(fit_pool.texts) - eval_keys - held_keys
    require(len(fit_keys) == expected_text_population,
            f"Text-level population {len(fit_keys)} differs from the frozen {expected_text_population}")

    eval_titles = final_pool.all_titles() | sub_pool.all_titles() | held_eval_pool.all_titles()
    eval_titles_norm = {norm_title(t) for t in eval_titles}
    held_titles_norm = {norm_title(t) for t in held_pool.all_titles()}

    def hits(key, exact=None, norm=None):
        titles = fit_pool.titles[key]
        if exact is not None:
            return any(t in exact for t in titles)
        return any(norm_title(t) in norm for t in titles)

    ordered = sorted(fit_keys, key=lambda k: k.hex())
    first_title_exact = sum(1 for k in ordered if fit_pool.titles[k][:1] and fit_pool.titles[k][0] in eval_titles)
    any_exact = {k for k in ordered if hits(k, exact=eval_titles)}
    any_norm_eval = {k for k in ordered if hits(k, norm=eval_titles_norm)}
    any_norm_held = {k for k in ordered if hits(k, norm=held_titles_norm)}
    excluded = any_norm_eval if title_exclusion == "evaluation" else any_norm_eval | any_norm_held
    population = [k for k in ordered if k not in excluded]

    counts = {
        "text_level_population": len(fit_keys),
        "texts_with_more_than_one_title": sum(1 for k in ordered if len(fit_pool.titles[k]) > 1),
        "title_overlap_diagnostic": {
            "first_title_exact_vs_evaluation_titles_original_diagnostic": first_title_exact,
            "any_title_exact_vs_evaluation_titles": len(any_exact),
            "any_title_normalized_vs_evaluation_titles": len(any_norm_eval),
            "any_title_normalized_vs_router_train_validation_titles": len(any_norm_held),
            "any_title_normalized_vs_evaluation_or_heldout_titles": len(any_norm_eval | any_norm_held),
        },
        "title_exclusion_applied": title_exclusion,
        "texts_excluded_by_title": len(excluded),
        "topic_fit_population_after_title_exclusion": len(population),
        "unique_titles": {"evaluation": len(eval_titles), "router_train_validation": len(held_pool.all_titles())},
        "retrieval_final_unique_texts": len(final_pool.texts),
        "retrieval_validation_unique_texts": len(held_pool.texts),
        "texts_in_both_retrieval_pools": len(set(final_pool.texts) & held_keys),
        "blank_contexts_skipped": {
            "final_eval": sum(q["blank_contexts"] for q in final_questions),
            "router_train": sum(q["blank_contexts"] for q in held_questions["router_train"]),
            "validation": sum(q["blank_contexts"] for q in held_questions["validation"])},
        "questions_without_gold": {
            "final_eval": sum(1 for q in final_questions if not q["gold_doc_ids"]),
            "router_train": sum(1 for q in held_questions["router_train"] if not q["gold_doc_ids"]),
            "validation": sum(1 for q in held_questions["validation"] if not q["gold_doc_ids"])},
    }
    pop_set = set(population)
    require(not pop_set & set(final_pool.texts) and not pop_set & held_keys,
            "Topic-fitting population overlaps a retrieval pool")
    if mode == "measure":
        return {"mode": "measure", "counts": counts}

    output_dir.mkdir(parents=True)

    def corpus_records(pool, keys):
        for key in sorted(keys, key=lambda k: k.hex()):
            yield {"doc_id": key.hex(), "titles": pool.titles[key], "text": pool.texts[key]}

    def by_id(records):
        return sorted(records, key=lambda r: r["question_id"])

    files = {
        "topic_fit_population": write_jsonl(output_dir / "topic_fit_population.jsonl",
                                            corpus_records(fit_pool, population)),
        "retrieval_final": write_jsonl(output_dir / "retrieval_final.jsonl",
                                       corpus_records(final_pool, final_pool.texts)),
        "retrieval_validation": write_jsonl(output_dir / "retrieval_validation.jsonl",
                                            corpus_records(held_pool, held_pool.texts)),
        "questions_final_eval": write_jsonl(output_dir / "questions_final_eval.jsonl", by_id(final_questions)),
        "questions_router_train": write_jsonl(output_dir / "questions_router_train.jsonl",
                                              by_id(held_questions["router_train"])),
        "questions_validation": write_jsonl(output_dir / "questions_validation.jsonl",
                                            by_id(held_questions["validation"])),
        "excluded_by_title_doc_ids": write_lines(output_dir / "excluded_by_title_doc_ids.txt",
                                                 sorted(k.hex() for k in excluded)),
    }
    manifest = {
        "protocol": PROTOCOL, "script_sha256": sha256_hex(Path(__file__).read_bytes()),
        "frozen_split_manifest_sha256": frozen_manifest_sha256,
        "input_sha256": input_hashes, "inputs_match_split_manifest": True,
        "text_identity": "exact UTF-8 paragraph_text; doc_id = SHA-256 hex; blank texts excluded",
        "title_identity": "NFC, casefold, whitespace-collapsed; a text is excluded if ANY of its titles matches",
        "labels_used_for_selection": False,
        "gold_doc_ids_note": "present in question files for evaluation only",
        "files": files, "counts": counts,
    }
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with (output_dir / "manifest.json").open("xb") as stream:
        stream.write(data)
    manifest["manifest_sha256"] = sha256_hex(data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("measure", "build"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Fresh directory (build mode)")
    parser.add_argument("--title-exclusion", choices=TITLE_MODES, default="evaluation")
    args = parser.parse_args(argv)
    try:
        result = build(args.data_dir, args.splits_dir, args.output_dir, mode=args.mode,
                       title_exclusion=args.title_exclusion,
                       frozen_manifest_sha256=FROZEN_MANIFEST_SHA256,
                       expected_text_population=EXPECTED_TEXT_POPULATION)
    except (OSError, ValueError, KeyError) as error:
        print(f"Corpus build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
