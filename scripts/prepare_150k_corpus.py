"""Prepare a frozen training-occurrence corpus and audit 2Wiki evaluation splits.

Standard library only. No fitting, embeddings, retrieval, or network access.
See docs/tae_v2_150k_protocol.md. Production CLI fixes size=150000 and seed=42.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/tae_v2_150k_corpus"
SAMPLE_SIZE = 150_000
SEED = 42
INPUT_NAMES = ("train.jsonl", "dev.jsonl", "dev_subsampled.jsonl", "test_subsampled.jsonl")
CORPUS_FIELDS = {"occurrence_id", "question_id", "source_split", "source_line",
                 "context_position", "idx", "title", "paragraph_text", "text_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reject_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def rows(path, stats):
    """Hash every byte, including blank lines; reject malformed records explicitly."""
    digest = hashlib.sha256()
    stats.update(bytes=0, blank_lines=0, questions=0)
    with Path(path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            stats["bytes"] += len(raw)
            if not raw.strip():
                stats["blank_lines"] += 1
                continue
            try:
                row = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                                 object_pairs_hook=unique_object)
                require(isinstance(row, dict), "Expected a question object")
                qid = row.get("question_id")
                require(isinstance(qid, str) and bool(qid.strip()), "Invalid question_id")
                qid.encode("utf-8")
                require(isinstance(row.get("contexts"), list), "Expected a contexts list")
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            stats["questions"] += 1
            yield line_number, row
    stats["sha256"] = digest.hexdigest()


def contexts(row, location):
    """Validate provenance even for empty contexts; never inspect gold fields."""
    for position, context in enumerate(row["contexts"]):
        where = f"{location}, context {position}"
        require(isinstance(context, dict), f"{where}: Expected a context object")
        require(type(context.get("idx")) is int and context["idx"] >= 0,
                f"{where}: idx must be a nonnegative question-local integer")
        require(isinstance(context.get("title"), str), f"{where}: Missing/string title required")
        require("paragraph_text" in context, f"{where}: Missing paragraph_text")
        text = context["paragraph_text"]
        require(text is None or isinstance(text, str), f"{where}: Invalid paragraph_text type")
        context["title"].encode("utf-8")
        if text is not None:
            text.encode("utf-8")
        yield position, context, text is not None and bool(text.strip())


def scan_training(path, connection, sample_size, seed):
    """Algorithm R: every eligible occurrence has inclusion probability k/N.

    Retain only eligible ordinal integers in RAM. Read the whole input even when
    the reservoir is full. Repeated text is never deduplicated before sampling.
    """
    require(type(sample_size) is int and sample_size > 0, "sample_size must be positive")
    require(type(seed) is int, "seed must be an integer")
    connection.execute("CREATE TABLE train_ids (qid TEXT PRIMARY KEY) WITHOUT ROWID")
    stats, reservoir = {}, []
    counts = Counter(context_occurrences=0, eligible_occurrences=0, empty_contexts=0)
    rng = random.Random(seed)
    for line, row in rows(path, stats):
        try:
            connection.execute("INSERT INTO train_ids VALUES (?)", (row["question_id"],))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"{path}:{line}: Duplicate training question_id") from error
        for _, _, eligible in contexts(row, f"{path}:{line}"):
            counts["context_occurrences"] += 1
            if not eligible:
                counts["empty_contexts"] += 1
                continue
            ordinal = counts["eligible_occurrences"]
            counts["eligible_occurrences"] += 1
            if ordinal < sample_size:
                reservoir.append(ordinal)
            else:
                slot = rng.randrange(ordinal + 1)
                if slot < sample_size:
                    reservoir[slot] = ordinal
        if stats["questions"] % 1000 == 0:
            connection.commit()
    connection.commit()
    require(counts["eligible_occurrences"] >= sample_size,
            f"Need {sample_size} eligible training occurrences; found {counts['eligible_occurrences']}")
    require(sum(counts[k] for k in ("eligible_occurrences", "empty_contexts")) ==
            counts["context_occurrences"], "Context accounting mismatch")
    require(len(reservoir) == sample_size and len(set(reservoir)) == sample_size,
            "Selected ordinal count or uniqueness mismatch")
    return sorted(reservoir), {**stats, **counts}


def project(row, context, source_line, position, source_sha256):
    """An occurrence identity is distinct from an exact-text identity and idx."""
    return {
        "occurrence_id": f"2wikimultihopqa:train:{source_sha256}:{source_line}:{position}",
        "question_id": row["question_id"], "source_split": "train", "source_line": source_line,
        "context_position": position, "idx": context["idx"], "title": context["title"],
        "paragraph_text": context["paragraph_text"],
        "text_sha256": text_hash(context["paragraph_text"]),
    }


def selected_rows(path, selected, source_sha256, stats):
    """Reread the entire source, projecting only the selected eligible ordinals."""
    wanted = iter(selected)
    target = next(wanted, None)
    ordinal = 0
    for line, row in rows(path, stats):
        for position, context, eligible in contexts(row, f"{path}:{line}"):
            if not eligible:
                continue
            if ordinal == target:
                yield project(row, context, line, position, source_sha256)
                target = next(wanted, None)
            ordinal += 1
    require(target is None, "Selected occurrence missing during source reread")
    require(stats["sha256"] == source_sha256, "Training source changed between passes")


def write_file(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_corpus(source, destination, selected, source_sha256):
    stats = {}
    with destination.open("xb") as stream:
        for row in selected_rows(source, selected, source_sha256, stats):
            stream.write(json_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())


def validate_corpus(source, corpus, selected, source_sha256, connection):
    """Independent third source pass, compared row-for-row with saved output.

    The ordered source-line/position pair and source-qualified identity establish
    uniqueness. SQLite counts exact UTF-8 duplicates without keeping texts in RAM.
    """
    connection.execute("CREATE TABLE sample_texts (hash TEXT PRIMARY KEY, text BLOB NOT NULL, "
                       "n INTEGER NOT NULL) WITHOUT ROWID")
    connection.execute("CREATE TABLE sample_questions (qid TEXT PRIMARY KEY) WITHOUT ROWID")
    stats, count, previous, digest = {}, 0, (0, -1), hashlib.sha256()
    with corpus.open("rb") as stream:
        for expected in selected_rows(source, selected, source_sha256, stats):
            raw = stream.readline()
            require(bool(raw), "Corpus has fewer rows than requested")
            digest.update(raw)
            actual = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                                object_pairs_hook=unique_object)
            require(isinstance(actual, dict) and actual.keys() == CORPUS_FIELDS,
                    "Corpus projection contains missing or unapproved fields")
            require(actual == expected, "Corpus text, occurrence identity, or source provenance differs")
            order = (actual["source_line"], actual["context_position"])
            require(order > previous, "Duplicate or unordered source occurrence")
            previous = order
            encoded = actual["paragraph_text"].encode("utf-8")
            existing = connection.execute("SELECT text FROM sample_texts WHERE hash=?",
                                          (actual["text_sha256"],)).fetchone()
            if existing is None:
                connection.execute("INSERT INTO sample_texts VALUES (?, ?, 1)",
                                   (actual["text_sha256"], encoded))
            else:
                require(existing[0] == encoded, "SHA-256 text collision")
                connection.execute("UPDATE sample_texts SET n=n+1 WHERE hash=?", (actual["text_sha256"],))
            connection.execute("INSERT OR IGNORE INTO sample_questions VALUES (?)", (actual["question_id"],))
            count += 1
            if count % 1000 == 0:
                connection.commit()
        require(not stream.read(1), "Corpus has extra rows")
    connection.commit()
    require(count == len(selected), "Corpus occurrence count mismatch")
    unique, groups, repeated, maximum = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(n>1),0), COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0), "
        "COALESCE(MAX(n),0) FROM sample_texts").fetchone()
    return {
        "status": "passed", "occurrences": count, "valid_nonempty_texts": count,
        "unique_occurrence_ids": count, "source_integrity_verified_occurrences": count,
        "source_questions": connection.execute("SELECT COUNT(*) FROM sample_questions").fetchone()[0],
        "unique_texts": unique, "duplicate_text_excess_occurrences": count - unique,
        "duplicated_text_groups": groups, "occurrences_in_duplicated_text_groups": repeated,
        "maximum_text_multiplicity": maximum, "corpus_sha256": digest.hexdigest(),
    }


def question_projection(row, location):
    query = row.get("question_text")
    require(isinstance(query, str) and bool(query.strip()), f"{location}: Invalid question_text")
    query.encode("utf-8")
    projected = []
    for _, context, _ in contexts(row, location):
        projected.append({key: context[key] for key in ("idx", "title", "paragraph_text")})
    return query, projected


def audit_evaluation(data_dir, connection):
    """Verify actual IDs against full train and original processed development.

    Context remapping is permitted and counted; query identity must match dev.
    Gold labels, answers and reasoning are not used by this audit or selection.
    """
    evaluation, source_stats, seen_eval = {}, {}, set()
    for name in INPUT_NAMES[2:]:
        stats, saved, seen = {}, {}, set()
        path = data_dir / name
        for line, row in rows(path, stats):
            qid = row["question_id"]
            require(qid not in seen, f"{path}:{line}: Duplicate evaluation question_id")
            require(qid not in seen_eval, f"Evaluation split question-ID overlap: {qid}")
            require(connection.execute("SELECT 1 FROM train_ids WHERE qid=?", (qid,)).fetchone() is None,
                    f"Training/evaluation question-ID overlap: {qid}")
            query, projected = question_projection(row, f"{path}:{line}")
            saved[qid] = (query, projected)
            seen.add(qid)
        require(bool(saved), f"Empty evaluation file: {path}")
        seen_eval.update(seen)
        evaluation[name] = saved
        source_stats[name] = stats

    connection.execute("CREATE TABLE dev_ids (qid TEXT PRIMARY KEY) WITHOUT ROWID")
    original_stats, verified = {}, set()
    changes = {name: Counter(questions_with_context_changes=0, changed_context_positions=0)
               for name in evaluation}
    path = data_dir / "dev.jsonl"
    for line, row in rows(path, original_stats):
        qid = row["question_id"]
        try:
            connection.execute("INSERT INTO dev_ids VALUES (?)", (qid,))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"{path}:{line}: Duplicate original development question_id") from error
        require(connection.execute("SELECT 1 FROM train_ids WHERE qid=?", (qid,)).fetchone() is None,
                f"Original development/training question-ID overlap: {qid}")
        query, projected = question_projection(row, f"{path}:{line}")
        for name, saved in evaluation.items():
            if qid not in saved:
                continue
            eval_query, eval_contexts = saved[qid]
            require(query == eval_query, f"Evaluation question_text differs from original dev: {qid}")
            require(len(projected) == len(eval_contexts), f"Evaluation context count differs from original dev: {qid}")
            require(all(a["idx"] == b["idx"] for a, b in zip(projected, eval_contexts)),
                    f"Evaluation context idx differs from original dev at its source position: {qid}")
            changed = sum(a != b for a, b in zip(projected, eval_contexts))
            changes[name]["questions_with_context_changes"] += int(changed > 0)
            changes[name]["changed_context_positions"] += changed
            verified.add(qid)
        if original_stats["questions"] % 1000 == 0:
            connection.commit()
    require(verified == seen_eval, "Evaluation IDs missing from original dev.jsonl: " +
            ", ".join(sorted(seen_eval - verified)[:5]))
    connection.commit()
    source_stats["dev.jsonl"] = original_stats
    splits = {}
    for name, saved in evaluation.items():
        ordered_ids = list(saved)
        splits[name] = {
            "role": "calibration_and_development" if name.startswith("dev_") else "final_local_evaluation",
            "original_source_split": "development", "official_test_set": False,
            "question_count": len(saved), "ordered_question_ids": ordered_ids,
            "question_id_set_sha256": hashlib.sha256(json_bytes(sorted(ordered_ids))).hexdigest(),
            "question_id_set_hash_encoding": "sorted IDs as canonical UTF-8 JSON array plus LF",
            "source_membership_and_question_text": "verified_against_dev.jsonl",
            "context_comparison": dict(changes[name]),
        }
    return source_stats, {
        "status": "passed", "train_vs_original_dev_overlap": 0,
        "train_vs_evaluation_overlap": 0, "dev_subsampled_vs_test_subsampled_overlap": 0,
        "splits": splits,
        "provenance_basis": "Actual question-ID membership and exact question text in processed dev.jsonl; "
                            "repository subsampler reads dev.jsonl for both names with seed 13370. "
                            "Context remapping can change text/title; changes are reported, not silently normalized.",
        "historical_exposure": "dev_subsampled questions 1-20 were exploratory and 21-100 used for TAE V1; "
                               "test_subsampled is a local development-derived holdout, not a claim of historical nonuse.",
    }


def frozen_protocol(audit, input_hashes, corpus_sha256, sample_size, seed):
    return {
        "protocol_version": 1, "dataset": "2WikiMultiHopQA", "source_split_for_corpus_and_fitting": "train",
        "input_sha256": input_hashes, "corpus_sha256": corpus_sha256, "evaluation_audit": audit,
        "sampling": {"occurrences": sample_size, "seed": seed},
        "corpus_policy": f"Exactly {sample_size} training paragraph occurrences sampled without replacement, seed {seed}. "
                         "No development contexts appended; repeated training text retained. "
                         "Only paragraph_text may be a future fitting input; provenance is metadata.",
        "freeze_rule": "Use every question in each saved ordered ID list. Tune only on dev_subsampled. "
                       "Freeze all later method settings before scoring test_subsampled; record settings and hashes. "
                       "Never select questions, replenish the corpus, or tune using evidence coverage or final scores.",
        "gold_scope": "Gold supporting paragraph occurrences identified by boolean is_supporting in evaluation records; "
                      "answers/support/reasoning may enter scoring only, never model fitting or routing inputs.",
        "evidence_identity": "For each gold occurrence, exact UTF-8 paragraph_text equality with any corpus row establishes "
                             "availability. text_sha256 is an index, verified with exact bytes. No fuzzy/title/idx-only match. "
                             "Different corpus occurrences of identical text are equivalent for evidence matching. "
                             "Each annotated gold occurrence remains in the denominator; a retrieved matching text can satisfy "
                             "repeated annotations. These are paragraph metrics, not official sentence-support scores.",
        "overall_coverage": {
            "micro": "sum_q |A_q| / sum_q |G_q|, where G_q is all gold occurrences and A_q those present in the fixed corpus",
            "question_reporting": "Report N, gold and available occurrence counts, macro |A_q|/|G_q|, "
                                  "and numbers of questions with all, some, and zero gold present, separately for each split.",
        },
        "retrieval_reporting": {
            "k": [5, 10],
            "retrieved_set": "First k occurrence IDs from the ranked list; validate membership in the frozen corpus. "
                             "Count duplicate-text slots as consumed ranks; do not refill top-k after deduplication.",
            "unconditional_main": "Recall@k = sum_q |R_qk| / sum_q |G_q|; complete-evidence success@k = "
                                  "count_q(R_qk = G_q) / N. R_qk is gold occurrences matched in top-k. "
                                  "Keep uncovered and partially covered questions in main answer and retrieval results.",
            "conditional_secondary": "Conditional micro Recall@k = sum_q |R_qk| / sum_q |A_q|. "
                                     "Also report macro |R_qk|/|A_q| over |A_q|>0 and complete-evidence success@k "
                                     "over questions with |A_q|=|G_q|. Always show these subset denominators.",
            "zero_denominators": "Report null/undefined with count=0, never replace an undefined rate by zero or drop it silently. "
                                 "Missing/nonboolean supporting flags or questions with no gold are validation errors.",
            "distinction": "Corpus availability, pre-ranking topic eligibility, ranked retrieval, and answer quality are separate outcomes. "
                           "All methods must share the identical corpus and question lists.",
        },
        "step_1_scope": "Preparation and protocol only; evidence coverage and retrieval scores are not computed in this step.",
    }


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(data_dir, output_dir, *, sample_size=SAMPLE_SIZE, seed=SEED):
    """Synthetic tests may override size/seed; production CLI cannot override them.

    All work is staged beside the destination. An exclusive mkdir reserves a new
    output only after every check passes, even if another process creates an empty
    directory during processing. No existing destination is reused or replaced.
    """
    data_dir, output = Path(data_dir).absolute(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists; choose a fresh run directory: {output}")
    for name in INPUT_NAMES:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Required processed source missing: {data_dir / name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "complete"
        staged.mkdir()
        with closing(sqlite3.connect(workspace / "audit.sqlite3")) as connection:
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            selected, source = scan_training(data_dir / "train.jsonl", connection, sample_size, seed)
            corpus = staged / "corpus.jsonl"
            write_corpus(data_dir / "train.jsonl", corpus, selected, source["sha256"])
            validation = validate_corpus(data_dir / "train.jsonl", corpus, selected, source["sha256"], connection)
            eval_stats, audit = audit_evaluation(data_dir, connection)
        inputs = {"train.jsonl": source, **eval_stats}
        # A final hash check detects source mutation after each completed scan.
        for name, stats in inputs.items():
            require(sha256(data_dir / name) == stats["sha256"], f"Input changed during preparation: {name}")
        protocol = frozen_protocol(audit, {name: stats["sha256"] for name, stats in inputs.items()},
                                   validation["corpus_sha256"], sample_size, seed)
        write_file(staged / "frozen_evaluation_protocol.json", json_bytes(protocol))
        report = {
            "schema_version": 1, "status": "passed", "source_split": "train", "inputs": inputs,
            "input_directory": str(data_dir),
            "processed_source_provenance": "Repository download/processed_data.sh documents Google Drive archive "
                                           "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC; this run identifies supplied files "
                                           "by their byte hashes and verifies processed split membership.",
            "sampling": {"algorithm": "Algorithm R reservoir over all eligible occurrence ordinals; random.Random(seed).randrange(i+1)",
                         "seed": seed, "requested_occurrences": sample_size, "without_replacement": True,
                         "eligible_occurrences": source["eligible_occurrences"],
                         "inclusion_probability": sample_size / source["eligible_occurrences"],
                         "eligible_ordinal_selection_sha256": hashlib.sha256(json_bytes(selected)).hexdigest(),
                         "output_order": "original source line then zero-based context position; selection is uniform, not a prefix"},
            "eligibility": "paragraph_text must exist; null/empty/whitespace-only texts excluded and counted; "
                           "other malformed records abort. Nonempty text and title are preserved without normalization.",
            "validation": validation, "evaluation_audit": audit,
            "training_input_allowlist": ["paragraph_text"], "gold_used_for_selection_or_fitting": False,
            "occurrence_id": "2wikimultihopqa:train:<input SHA-256>:<1-based source line>:<0-based context position>; "
                             "idx is preserved as question-local metadata, never a global key",
            "duplicate_policy": "Exact UTF-8 equality; all natural occurrences retained in the sampling population. "
                                "excess=N-unique_texts; duplicated groups count texts with multiplicity>1.",
            "reproducibility": {"python": sys.version, "implementation": platform.python_implementation(),
                                "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
                                "git_commit": git_revision(), "script_sha256": sha256(Path(__file__)),
                                "dependencies": "Python standard library only",
                                "training_parsing_passes": 3, "final_input_byte_hash_passes": 1,
                                "repeat": "Same input bytes, script, size, seed and Python RNG implementation yield identical "
                                          "corpus bytes and ordered IDs. Use a fresh output directory; compare corpus and selection hashes. "
                                          "Report environment/Git fields may differ across hosts."},
            "output_sha256": {"corpus.jsonl": validation["corpus_sha256"],
                              "frozen_evaluation_protocol.json": sha256(staged / "frozen_evaluation_protocol.json")},
        }
        write_file(staged / "validation_report.json", json_bytes(report))
        names = ("corpus.jsonl", "frozen_evaluation_protocol.json", "validation_report.json")
        write_file(staged / "SHA256SUMS", "".join(f"{sha256(staged / name)}  {name}\n" for name in names).encode("ascii"))
        # mkdir is exclusive even against an empty directory, file or dangling symlink.
        output.mkdir(exist_ok=False)
        # The report is published last and is the success marker. A crash during
        # publication leaves an incomplete, non-reusable directory, never a replaced run.
        for name in ("corpus.jsonl", "frozen_evaluation_protocol.json", "SHA256SUMS", "validation_report.json"):
            os.link(staged / name, output / name)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "processed_data/2wikimultihopqa",
                        help="Directory with train.jsonl, dev.jsonl and both subsampled evaluation files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "run_001",
                        help="Fresh directory inside artifacts/tae_v2_150k_corpus/")
    args = parser.parse_args(argv)
    try:
        require(args.output_dir.resolve().is_relative_to(OUTPUT_ROOT.resolve()) and
                args.output_dir.resolve() != OUTPUT_ROOT.resolve(),
                f"Output must be a fresh run directory under {OUTPUT_ROOT}")
        report = prepare(args.data_dir, args.output_dir)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir),
                      "validation": report["validation"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
