"""Build the frozen, leakage-safe question splits for the TAE-R experiments.

Reads the processed 2WikiMultiHopQA JSONL files and writes only question-ID
lists and a manifest. It builds no corpus, fits no model, and reads support
labels only to count them for the manifest; labels never influence a split.

Splits (see docs/tae_r_protocol_v1.md):
  exposed        development questions already used in earlier experiments
                 (every ID in dev_subsampled.jsonl and test_subsampled.jsonl)
  final_eval     dev.jsonl questions minus exposed; touched once, at the end
  e2e_300/500    seeded nested subsets of final_eval for end-to-end LLM runs
  router_train   held-out training questions for the adequacy predictor
  validation     held-out training questions for tuning and decision gates
  topic_fit      all remaining training questions

The topic-fitting paragraph population is measured here as the unique
paragraph texts of topic_fit questions that occur in no development,
exposed, router_train or validation context (exact UTF-8 text identity,
empty/whitespace-only texts ignored, as in corpus-design-audit-v1).

Standard library only. Example:
  python -B scripts/build_tae_r_splits.py \
      --data-dir /kaggle/working/data/processed_data/2wikimultihopqa \
      --output-dir /kaggle/working/tae_r_splits_v1
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROTOCOL = "tae-r-protocol-v1"
NAMES = {"train": "train.jsonl", "original_dev": "dev.jsonl",
         "development": "dev_subsampled.jsonl", "heldout": "test_subsampled.jsonl"}
# SHA-256 of the files measured by corpus_design_audit_v1/audit_run_001.json.
AUDITED_SHA256 = {
    "train": "14c670fbbcbca728fca1c6f5a0787dbba3a7f8c20f9193f0ddb5f7abc9121a4a",
    "original_dev": "c6e1d9faee2202f2c72b68e05ae75ef220329e73819635a1172838fa81f79055",
    "development": "5db565705239f8389a0eb25a40ec320199f5059eece6221fbff0888b4873e089",
    "heldout": "4948757b90627f40b77cd6bf81a62c1a7ad525d6a6a0c91b17a2e0e2fa88f5bc",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_key(text):
    """Digest of a nonempty paragraph text, or None for null/blank text."""
    if not isinstance(text, str) or not text.strip():
        return None
    return hashlib.sha256(text.encode("utf-8")).digest()


def rows(path, file_hashes):
    """Yield parsed question records and record the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw.decode("utf-8"))
            ok = (isinstance(row, dict) and isinstance(row.get("question_id"), str)
                  and row["question_id"].strip() and isinstance(row.get("contexts"), list)
                  and isinstance(row.get("question_text"), str))
            require(ok, f"{path.name}: line {line}: invalid question record")
            yield row
    file_hashes[path.name] = digest.hexdigest()


def seeded_order(ids, seed):
    """Deterministic pseudo-random order that does not depend on file order."""
    return sorted(ids, key=lambda q: (hashlib.sha256(f"{seed}:{q}".encode("utf-8")).hexdigest(), q))


def scan_small(path, file_hashes):
    """Return (ids, text digests, title set, question texts, support count)."""
    ids, texts, titles, questions, support = [], set(), set(), set(), 0
    for row in rows(path, file_hashes):
        ids.append(row["question_id"])
        questions.add(row["question_text"].strip())
        for ctx in row["contexts"]:
            key = text_key(ctx.get("paragraph_text"))
            if key is not None:
                texts.add(key)
                if isinstance(ctx.get("title"), str):
                    titles.add(ctx["title"])
                support += ctx.get("is_supporting") is True
    require(len(ids) == len(set(ids)), f"{path.name}: duplicate question_id")
    return ids, texts, titles, questions, support


def write_ids(path, ids):
    data = ("\n".join(ids) + "\n").encode("utf-8") if ids else b""
    with path.open("xb") as stream:
        stream.write(data)
    return {"file": path.name, "count": len(ids), "sha256": hashlib.sha256(data).hexdigest()}


def build(data_dir, output_dir, seed, router_train_size, validation_size, e2e_sizes, names=None):
    names = dict(NAMES, **(names or {}))
    paths = {split: data_dir / name for split, name in names.items()}
    for split, path in paths.items():
        require(path.is_file(), f"Missing input file for {split}: {path.name}")
    require(not os.path.lexists(output_dir), "Output directory exists; use a fresh one")
    require(not output_dir.resolve().is_relative_to(data_dir.resolve()),
            "Output directory must be outside the dataset directory")
    file_hashes = {}

    dev_ids, dev_texts, dev_titles, dev_questions, dev_support = scan_small(
        paths["original_dev"], file_hashes)
    sub_ids, sub_texts, sub_titles, _, _ = scan_small(paths["development"], file_hashes)
    held_ids, held_texts, held_titles, _, _ = scan_small(paths["heldout"], file_hashes)

    exposed = set(sub_ids) | set(held_ids)
    dev_set = set(dev_ids)
    exposed_not_in_dev = sorted(exposed - dev_set)
    final_eval = sorted(dev_set - exposed)
    ordered_final = seeded_order(final_eval, seed)
    e2e = {}
    for size in sorted(e2e_sizes):
        require(0 < size <= len(final_eval), f"e2e size {size} exceeds final_eval")
        e2e[size] = sorted(ordered_final[:size])

    # Pass 1 over train: question IDs only, to choose the held-out folds.
    train_ids = [row["question_id"] for row in rows(paths["train"], file_hashes)]
    train_set = set(train_ids)
    require(len(train_ids) == len(train_set), "train: duplicate question_id")
    require(router_train_size + validation_size < len(train_ids), "Held-out folds exceed train")
    ordered_train = seeded_order(train_ids, seed)
    router_train = set(ordered_train[:router_train_size])
    validation = set(ordered_train[router_train_size:router_train_size + validation_size])
    topic_fit = train_set - router_train - validation

    # Pass 2 over train: paragraph texts per fold.
    fold_texts = {"router_train": set(), "validation": set(), "topic_fit": set()}
    fold_support = {"router_train": 0, "validation": 0}
    fold_questions = {"router_train": set(), "validation": set()}
    title_of = {}
    for row in rows(paths["train"], {}):
        qid = row["question_id"]
        fold = "router_train" if qid in router_train else "validation" if qid in validation else "topic_fit"
        if fold != "topic_fit":
            fold_questions[fold].add(row["question_text"].strip())
        for ctx in row["contexts"]:
            key = text_key(ctx.get("paragraph_text"))
            if key is None:
                continue
            fold_texts[fold].add(key)
            if fold == "topic_fit":
                title_of.setdefault(key, ctx.get("title"))
            else:
                fold_support[fold] += ctx.get("is_supporting") is True

    evaluation_texts = dev_texts | sub_texts | held_texts
    heldout_train_texts = fold_texts["router_train"] | fold_texts["validation"]
    fit_population = fold_texts["topic_fit"] - evaluation_texts - heldout_train_texts
    evaluation_titles = dev_titles | sub_titles | held_titles
    fit_titles_shared = sum(1 for key in fit_population if title_of.get(key) in evaluation_titles)

    id_sets = {"exposed": exposed, "final_eval": set(final_eval), "router_train": router_train,
               "validation": validation, "topic_fit": topic_fit}
    labels = sorted(id_sets)
    overlaps = {f"{a}__{b}": len(id_sets[a] & id_sets[b])
                for i, a in enumerate(labels) for b in labels[i + 1:]}
    require(not any(overlaps.values()), "Question-ID overlap between splits")
    require(all(set(v) <= id_sets["final_eval"] for v in e2e.values()), "e2e not within final_eval")

    output_dir.mkdir(parents=True)
    files = {
        "exposed": write_ids(output_dir / "exposed_ids.txt", sorted(exposed)),
        "final_eval": write_ids(output_dir / "final_eval_ids.txt", final_eval),
        "router_train": write_ids(output_dir / "router_train_ids.txt", sorted(router_train)),
        "validation": write_ids(output_dir / "validation_ids.txt", sorted(validation)),
        "topic_fit": write_ids(output_dir / "topic_fit_ids.txt", sorted(topic_fit)),
    }
    for size, ids in e2e.items():
        files[f"e2e_{size}"] = write_ids(output_dir / f"e2e_{size}_ids.txt", ids)

    input_hashes = {split: file_hashes[paths[split].name] for split in names}
    manifest = {
        "protocol": PROTOCOL, "seed": seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_filenames": names, "input_sha256": input_hashes,
        "inputs_match_audit_run_001": input_hashes == AUDITED_SHA256,
        "text_identity": "exact UTF-8 paragraph_text; null/empty/whitespace-only excluded",
        "labels_used_for_selection": False,
        "split_files": files,
        "question_id_overlaps": overlaps,
        "exposed_ids_not_in_original_dev": len(exposed_not_in_dev),
        "provenance": {
            "exposed": "all question IDs in dev_subsampled.jsonl and test_subsampled.jsonl",
            "final_eval": "dev.jsonl question IDs minus exposed",
            "e2e": "first N of final_eval in seeded SHA-256 order; nested",
            "router_train": "first router_train_size train IDs in seeded SHA-256 order",
            "validation": "next validation_size train IDs in the same order",
            "topic_fit": "all remaining train IDs",
        },
        "counts": {
            "train_questions": len(train_ids), "original_dev_questions": len(dev_ids),
            "final_eval_support_occurrences_including_exposed": dev_support,
            "router_train_support_occurrences": fold_support["router_train"],
            "validation_support_occurrences": fold_support["validation"],
            "unique_texts": {
                "final_retrieval_pool_original_dev": len(dev_texts),
                "all_evaluation_files": len(evaluation_texts),
                "router_train_contexts": len(fold_texts["router_train"]),
                "validation_contexts": len(fold_texts["validation"]),
                "validation_retrieval_pool_router_train_plus_validation": len(heldout_train_texts),
                "topic_fit_question_contexts": len(fold_texts["topic_fit"]),
                "topic_fit_population": len(fit_population),
            },
            "topic_fit_population_removed": {
                "shared_with_evaluation_files": len(fold_texts["topic_fit"] & evaluation_texts),
                "shared_with_router_train_or_validation": len(
                    (fold_texts["topic_fit"] - evaluation_texts) & heldout_train_texts),
            },
            "topic_fit_population_texts_whose_title_occurs_in_evaluation_files": fit_titles_shared,
            "heldout_train_pool_texts_also_in_evaluation_files": len(
                heldout_train_texts & evaluation_texts),
            "question_text_shared_with_original_dev": {
                fold: len(fold_questions[fold] & dev_questions) for fold in fold_questions},
        },
    }
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with (output_dir / "manifest.json").open("xb") as stream:
        stream.write(data)
    manifest["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--router-train-size", type=int, default=8000)
    parser.add_argument("--validation-size", type=int, default=4576)
    parser.add_argument("--e2e-sizes", type=int, nargs="+", default=[300, 500])
    args = parser.parse_args(argv)
    try:
        manifest = build(args.data_dir, args.output_dir, args.seed, args.router_train_size,
                         args.validation_size, args.e2e_sizes)
    except (OSError, ValueError) as error:
        print(f"Split build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
