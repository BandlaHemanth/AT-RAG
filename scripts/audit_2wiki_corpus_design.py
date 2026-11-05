"""Read-only structural 2Wiki audit. Standard library only; no retrieval or fitting.

See docs/corpus_design_audit_v1.md. Only a fresh JSON report and temporary
SQLite index are written. No question IDs, texts, or per-question outcomes escape.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile


NAMES = {"train": "train.jsonl", "original_dev": "dev.jsonl",
         "development": "dev_subsampled.jsonl", "heldout": "test_subsampled.jsonl"}
BUDGETS = (100_000, 150_000, 200_000)
ROOT = Path(__file__).resolve().parents[1]
INSPECTED = (
    "processing_scripts/process_2wikimultihopqa.py",
    "processing_scripts/subsample_dataset_and_remap_paras.py", "processing_scripts/lib.py",
    "scripts/build_full_2wiki_corpus.py", "scripts/prepare_150k_corpus.py",
    "docs/tae_v2_150k_protocol.md", "scripts/train_bertopic_150k.py",
    "scripts/shared_150k_retrieval.py", "scripts/query_inference_150k.py",
    "scripts/tae_v2_policy.py", "vectorDB/train_topic_model.py", "vectorDB/dataset_ingestion.py")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def rows(path, stats):
    digest = hashlib.sha256()
    stats.update(bytes=0, blank_lines=0, questions=0)
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            stats["bytes"] += len(raw)
            if not raw.strip():
                stats["blank_lines"] += 1
                continue
            try:
                row = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                                 parse_constant=reject_constant)
                require(isinstance(row, dict), "Expected object")
                require(isinstance(row.get("question_id"), str) and row["question_id"].strip(),
                        "Expected nonempty question_id")
                require(isinstance(row.get("contexts"), list), "Expected contexts list")
                require(isinstance(row.get("question_text"), str), "Expected question_text")
                # Validate UTF-8 even for escaped surrogate code points.
                row["question_id"].encode("utf-8")
                row["question_text"].encode("utf-8")
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"{path.name}: line {line}: invalid question record") from error
            stats["questions"] += 1
            yield line, row
    stats["sha256"] = digest.hexdigest()


def project_contexts(row):
    projected = []
    for ctx in row["contexts"]:
        require(isinstance(ctx, dict), "Expected context object")
        require(type(ctx.get("idx")) is int and ctx["idx"] >= 0, "Invalid context idx")
        require(isinstance(ctx.get("title"), str), "Invalid context title")
        require("paragraph_text" in ctx, "Missing paragraph_text")
        text = ctx["paragraph_text"]
        require(text is None or isinstance(text, str), "Invalid paragraph_text")
        projected.append({k: ctx[k] for k in ("idx", "title", "paragraph_text")})
    encoded(projected)
    return projected


def gold_diagnostic(contexts):
    """Local annotations only: availability here means a nonempty annotated text.

    This cannot independently establish that the source annotation was correct.
    No labels are consumed by any structural count or budget calculation.
    """
    result = Counter(annotated_support_occurrences=0, nonempty_support_occurrences=0,
                     invalid_or_missing_flags=0)
    texts = []
    for ctx in contexts:
        flag = ctx.get("is_supporting")
        if type(flag) is not bool:
            result["invalid_or_missing_flags"] += 1
        elif flag:
            result["annotated_support_occurrences"] += 1
            text = ctx["paragraph_text"]
            texts.append(text)
            if text is not None and text.strip():
                result["nonempty_support_occurrences"] += 1
    result["questions_without_annotated_support"] = int(not result["annotated_support_occurrences"])
    result["empty_support_occurrences"] = (result["annotated_support_occurrences"] -
                                           result["nonempty_support_occurrences"])
    return result, texts


def scan(db, split, path):
    stats, distribution, nonempty_distribution = {}, Counter(), Counter()
    counts = Counter(context_occurrences=0, nonempty_occurrences=0, empty_occurrences=0)
    gold = Counter()
    for line, row in rows(path, stats):
        try:
            projected = project_contexts(row)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"{path.name}: line {line}: invalid context record") from error
        qid = row["question_id"]
        # Keep structural metadata and scoring diagnostics in separate columns.
        diagnostic, gold_texts = ({}, []) if split == "train" else gold_diagnostic(row["contexts"])
        gold.update(diagnostic)
        signature = b"[]" if split == "train" else encoded(projected)
        old = db.execute("SELECT signature, query FROM questions WHERE split=? AND qid=?",
                         (split, qid)).fetchone()
        if old is None:
            db.execute("INSERT INTO questions VALUES (?,?,?,?,?,1)",
                       (split, qid, signature, b"" if split == "train" else
                        row["question_text"].encode("utf-8"), encoded(gold_texts)))
        else:
            db.execute("UPDATE questions SET n=n+1 WHERE split=? AND qid=?", (split, qid))
        distribution[len(projected)] += 1
        eligible = 0
        for ctx in projected:
            counts["context_occurrences"] += 1
            text = ctx["paragraph_text"]
            if text is None or not text.strip():
                counts["empty_occurrences"] += 1
                continue
            eligible += 1
            counts["nonempty_occurrences"] += 1
            # Exact UTF-8 bytes are the key: no normalization or hash collision risk.
            blob = text.encode("utf-8")
            db.execute("INSERT OR IGNORE INTO texts(text) VALUES (?)", (blob,))
            tid = db.execute("SELECT id FROM texts WHERE text=?", (blob,)).fetchone()[0]
            db.execute("INSERT INTO frequencies VALUES (?,?,1) ON CONFLICT(split,tid) "
                       "DO UPDATE SET n=n+1", (split, tid))
        nonempty_distribution[eligible] += 1
        if stats["questions"] % 1000 == 0:
            db.commit()
    db.commit()
    unique, maximum, groups, repeated = db.execute(
        "SELECT COUNT(*),COALESCE(MAX(n),0),COALESCE(SUM(n>1),0),"
        "COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0) FROM frequencies WHERE split=?",
        (split,)).fetchone()
    ids, duplicate_groups = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(n>1),0) FROM questions WHERE split=?", (split,)).fetchone()
    result = {**stats, **counts, "unique_nonempty_texts": unique,
              "duplicate_excess_occurrences": counts["nonempty_occurrences"] - unique,
              "duplicated_text_groups": groups, "occurrences_in_duplicated_text_groups": repeated,
              "maximum_text_multiplicity": maximum, "distinct_question_ids": ids,
              "duplicate_question_id_groups": duplicate_groups,
              "duplicate_question_id_excess_records": stats["questions"] - ids,
              "text_multiplicity_distribution": dict(db.execute(
                  "SELECT n,COUNT(*) FROM frequencies WHERE split=? GROUP BY n ORDER BY n", (split,))),
              "contexts_per_question_distribution": dict(sorted(distribution.items())),
              "nonempty_contexts_per_question_distribution": dict(sorted(nonempty_distribution.items()))}
    if split != "train":
        result["own_annotation_diagnostics"] = dict(gold)
    return result


def remapping(db, split):
    counts = Counter(matched_unique_ids=0, missing_original_ids=0, ambiguous_duplicate_ids=0,
                     question_text_changes=0, questions_with_context_changes=0,
                     context_count_changes=0, changed_context_positions=0,
                     text_changes=0, title_changes=0, idx_changes=0,
                     added_context_positions=0, removed_context_positions=0,
                     original_gold_occurrences=0, original_gold_nonempty_occurrences=0,
                     original_gold_texts_present_in_own_contexts=0)
    for qid, signature, query, n in db.execute(
            "SELECT qid,signature,query,n FROM questions WHERE split=?", (split,)):
        original = db.execute("SELECT signature,query,gold,n FROM questions "
                              "WHERE split='original_dev' AND qid=?", (qid,)).fetchone()
        if original is None:
            counts["missing_original_ids"] += 1
            continue
        if n != 1 or original[3] != 1:
            counts["ambiguous_duplicate_ids"] += 1
            continue
        counts["matched_unique_ids"] += 1
        counts["question_text_changes"] += int(query != original[1])
        a, b = json.loads(original[0]), json.loads(signature)
        counts["questions_with_context_changes"] += int(a != b)
        counts["context_count_changes"] += int(len(a) != len(b))
        counts["added_context_positions"] += max(0, len(b) - len(a))
        counts["removed_context_positions"] += max(0, len(a) - len(b))
        for left, right in zip(a, b):
            counts["changed_context_positions"] += int(left != right)
            for key, label in (("paragraph_text", "text_changes"), ("title", "title_changes"),
                               ("idx", "idx_changes")):
                counts[label] += int(left[key] != right[key])
        own = {c["paragraph_text"] for c in b if c["paragraph_text"] and c["paragraph_text"].strip()}
        gold_texts = json.loads(original[2])
        counts["original_gold_occurrences"] += len(gold_texts)
        nonempty = [t for t in gold_texts if t is not None and t.strip()]
        counts["original_gold_nonempty_occurrences"] += len(nonempty)
        counts["original_gold_texts_present_in_own_contexts"] += sum(t in own for t in nonempty)
    # Only aggregate counts leave this function, including for the held-out split.
    return dict(counts)


def union_size(db, splits):
    marks = ",".join("?" for _ in splits)
    return db.execute(f"SELECT COUNT(DISTINCT tid) FROM frequencies WHERE split IN ({marks})",
                      tuple(splits)).fetchone()[0]


def design_counts(db, mandatory_splits):
    mandatory = union_size(db, mandatory_splits)
    marks = ",".join("?" for _ in mandatory_splits)
    background = db.execute("SELECT COUNT(*) FROM frequencies t WHERE t.split='train' AND NOT EXISTS "
                            f"(SELECT 1 FROM frequencies e WHERE e.tid=t.tid AND e.split IN ({marks}))",
                            tuple(mandatory_splits)).fetchone()[0]
    return {"mandatory_unique_texts": mandatory, "training_only_background_available": background,
            "budgets": {str(b): {"mandatory_fits": mandatory <= b,
                                 "background_slots": max(0, b - mandatory),
                                 "fits_with_at_least_one_training_background_text": mandatory < b and background > 0,
                                 "can_fill_exact_budget": mandatory <= b <= mandatory + background}
                        for b in BUDGETS}}


def audit(data_dir, *, names=None, scratch_dir=None):
    data_dir = Path(data_dir).resolve()
    names = dict(NAMES if names is None else names)
    require(set(names) == set(NAMES), "Four split roles are required")
    require(all(Path(n).name == n and n not in ("", ".", "..") for n in names.values()),
            "Input names must be plain filenames under --data-dir")
    require(len(set(names.values())) == 4, "Split filenames must differ")
    if scratch_dir is not None:
        require(not Path(scratch_dir).resolve().is_relative_to(data_dir), "Scratch must be outside data directory")
    inputs = {s: data_dir / n for s, n in names.items()}
    missing = [s for s, p in inputs.items() if not p.is_file()]
    report = {"schema_version": 1, "audit_protocol": "corpus-design-audit-v1",
              "status": "incomplete_missing_inputs" if missing else "complete",
              "missing_splits": missing, "input_filenames": names, "splits": {},
              "text_identity": "exact UTF-8 paragraph_text; exclude null/empty/whitespace-only; no normalization",
              "budgets_are_unique_nonempty_texts": True,
              "heldout_reporting": "aggregate structural and annotation diagnostics only; no IDs or question outcomes",
              "gold_used_for_budget_or_selection": False, "corpus_constructed": False,
              "model_fitted": False, "pairwise_overlap": {}, "context_remapping": {},
              "budget_feasibility": None, "fresh_topic_fitting": None,
              "design_validation": None,
              "script_sha256": sha256(Path(__file__)), "python": sys.version,
              "inspected_implementation_sha256": {p: sha256(ROOT / p) if (ROOT / p).is_file() else None
                                                   for p in INSPECTED}}
    try:
        report["repository_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report["repository_revision"] = None
    with tempfile.TemporaryDirectory(prefix="2wiki-audit-", dir=scratch_dir) as tmp:
        with closing(sqlite3.connect(Path(tmp) / "index.sqlite3")) as db:
            db.executescript("""
                PRAGMA cache_size=-8192;
                PRAGMA mmap_size=0;
                CREATE TABLE texts(id INTEGER PRIMARY KEY, text BLOB NOT NULL UNIQUE);
                CREATE TABLE frequencies(split TEXT,tid INTEGER,n INTEGER,PRIMARY KEY(split,tid)) WITHOUT ROWID;
                CREATE TABLE questions(split TEXT,qid TEXT,signature BLOB,query BLOB,gold BLOB,n INTEGER,
                                       PRIMARY KEY(split,qid)) WITHOUT ROWID;
            """)
            for split, path in inputs.items():
                report["splits"][split] = None if split in missing else scan(db, split, path)
            available = [s for s in names if s not in missing]
            for a, b in itertools.combinations(available, 2):
                shared, na, nb = db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(a.n),0),COALESCE(SUM(b.n),0) "
                    "FROM frequencies a JOIN frequencies b ON a.tid=b.tid WHERE a.split=? AND b.split=?",
                    (a, b)).fetchone()
                ids = db.execute("SELECT COUNT(*) FROM questions a JOIN questions b ON a.qid=b.qid "
                                 "WHERE a.split=? AND b.split=?", (a, b)).fetchone()[0]
                report["pairwise_overlap"][a + "__" + b] = {
                    "shared_unique_texts": shared, "left_occurrences_of_shared_texts": na,
                    "right_occurrences_of_shared_texts": nb, "shared_distinct_question_ids": ids}
            if "original_dev" in available:
                for split in ("development", "heldout"):
                    if split in available:
                        report["context_remapping"][split] = remapping(db, split)
            if not missing:
                blockers = []
                for split in NAMES:
                    stats = report["splits"][split]
                    if not stats["questions"]:
                        blockers.append(split + ": empty split")
                    if stats["duplicate_question_id_groups"]:
                        blockers.append(split + ": duplicate question IDs")
                for a, b in (("train", "original_dev"), ("train", "development"),
                             ("train", "heldout"), ("development", "heldout")):
                    if report["pairwise_overlap"][a + "__" + b]["shared_distinct_question_ids"]:
                        blockers.append(a + "/" + b + ": question-ID overlap")
                for split, changes in report["context_remapping"].items():
                    if any(changes[k] for k in ("missing_original_ids", "ambiguous_duplicate_ids",
                                               "question_text_changes")):
                        blockers.append(split + ": source membership/query identity unresolved")
                report["design_validation"] = {
                    "structural_blockers": blockers,
                    "context_source_choice_requires_freezing": any(
                        x["questions_with_context_changes"] for x in report["context_remapping"].values()),
                    "counts_are_not_authorization_to_generate_or_train": True}
                report["budget_feasibility"] = {
                    "all_evaluation_contexts": design_counts(db, ("development", "heldout")),
                    # Include remapped texts as well as every original-dev context.
                    "full_original_dev_plus_evaluation_contexts": design_counts(
                        db, ("original_dev", "development", "heldout")),
                    "full_original_dev_unique_texts": union_size(db, ("original_dev",)),
                    "full_original_dev_fits": {str(b): union_size(db, ("original_dev",)) <= b for b in BUDGETS}}
                report["fresh_topic_fitting"] = {
                    "train_unique_excluding_evaluation_context_texts": design_counts(
                        db, ("development", "heldout"))["training_only_background_available"],
                    "train_unique_excluding_all_original_dev_and_evaluation_texts": design_counts(
                        db, ("original_dev", "development", "heldout"))["training_only_background_available"],
                    "recommended_exclusion": "all original development plus both processed evaluation subsets",
                    "fitting_input_allowlist": ["paragraph_text"]}
            # Detect changes after scanning; no success report for unstable inputs.
            for split in available:
                require(sha256(inputs[split]) == report["splits"][split]["sha256"],
                        f"Input changed during audit: {names[split]}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON file outside the dataset directory")
    parser.add_argument("--scratch-dir", type=Path, help="Existing writable directory outside dataset; temporary SQLite")
    for split, name in NAMES.items():
        parser.add_argument("--" + split.replace("_", "-") + "-name", default=name)
    args = parser.parse_args(argv)
    try:
        require(not args.output.resolve().is_relative_to(args.data_dir.resolve()),
                "Report must be outside dataset directory")
        require(not os.path.lexists(args.output), "Output exists; use a fresh JSON filename")
        report = audit(args.data_dir, names={s: getattr(args, s + "_name") for s in NAMES},
                       scratch_dir=args.scratch_dir)
        with args.output.open("xb") as stream:
            stream.write(encoded(report) + b"\n")
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return 2 if report["missing_splits"] else 0
    except (OSError, ValueError, sqlite3.Error) as error:
        # Validation errors never interpolate IDs, texts, or held-out row contents.
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
