"""Stream the complete 2Wiki training corpus without consulting gold annotations.

Standard library only; see docs/full_2wiki_corpus.md for Kaggle commands.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FULL_DATASET_COUNTS = {
    "questions": 167_454,
    "context_occurrences": 1_674_540,
    "unique_paragraphs": 396_978,
    "duplicate_occurrences": 1_277_562,
    "empty_contexts": 0,
}


def document_id(paragraph_text):
    """Use the same unprefixed SHA-256 IDs as the existing thesis pilot."""
    return hashlib.sha256(paragraph_text.encode("utf-8")).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sync_directory(path):
    """Persist directory entries on POSIX (including Kaggle Linux)."""
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def reject_nonfinite(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def stream_documents(source, destination, index_path):
    """Keep one question in Python memory; retain exact texts in a disk index.

    Comparing stored UTF-8 bytes also detects SHA-256 collisions rather than
    silently treating different paragraphs with the same hash as duplicates.
    """
    counts = dict.fromkeys(FULL_DATASET_COUNTS, 0)
    source_digest = hashlib.sha256()
    documents_digest = hashlib.sha256()
    source_bytes = 0
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA cache_size = -8192")  # 8 MiB page cache.
        connection.execute("PRAGMA mmap_size = 0")
        connection.execute(
            "CREATE TABLE seen (document_id TEXT PRIMARY KEY, paragraph BLOB NOT NULL) WITHOUT ROWID"
        )
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            for source_line, raw_line in enumerate(input_stream, 1):
                source_digest.update(raw_line)
                source_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    question = json.loads(raw_line.decode("utf-8"), parse_constant=reject_nonfinite)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"Source line {source_line}: invalid UTF-8 JSON: {error}") from error
                if not isinstance(question, dict) or not isinstance(question.get("contexts"), list):
                    raise ValueError(f"Source line {source_line}: expected an object with a contexts list")
                counts["questions"] += 1
                for position, context in enumerate(question["contexts"]):
                    location = f"Source line {source_line}, context {position}"
                    counts["context_occurrences"] += 1
                    if not isinstance(context, dict):
                        raise ValueError(f"{location}: expected a context object")
                    paragraph = context.get("paragraph_text")
                    if paragraph is None or (isinstance(paragraph, str) and not paragraph.strip()):
                        counts["empty_contexts"] += 1
                        continue
                    if not isinstance(paragraph, str):
                        raise ValueError(f"{location}: paragraph_text must be a string or null")
                    encoded = paragraph.encode("utf-8")
                    identifier = document_id(paragraph)
                    previous = connection.execute(
                        "SELECT paragraph FROM seen WHERE document_id = ?", (identifier,)
                    ).fetchone()
                    if previous is not None:
                        if previous[0] != encoded:
                            raise ValueError(f"{location}: SHA-256 collision for {identifier}")
                        counts["duplicate_occurrences"] += 1
                        continue
                    connection.execute("INSERT INTO seen VALUES (?, ?)", (identifier, encoded))
                    # Explicit projection: no support labels, answers, question text,
                    # reasoning steps, or other annotations enter the documents.
                    document = {
                        "document_id": identifier,
                        "paragraph_text": paragraph,
                        "question_id": question.get("question_id"),
                        "source_line": source_line,
                        "context_position": position,
                        "idx": context.get("idx"),
                        "title": context.get("title"),
                    }
                    serialized = json_bytes(document)
                    output_stream.write(serialized)
                    documents_digest.update(serialized)
                    counts["unique_paragraphs"] += 1
                # Bound transaction/journal growth without changing stream order.
                if counts["questions"] % 1000 == 0:
                    connection.commit()
            connection.commit()
            output_stream.flush()
            os.fsync(output_stream.fileno())
    finally:
        connection.close()
    if counts["context_occurrences"] != sum(counts[key] for key in (
        "unique_paragraphs", "duplicate_occurrences", "empty_contexts"
    )):
        raise ValueError("Context accounting invariant failed")
    return {
        **counts,
        "schema_version": 1,
        "source_sha256": source_digest.hexdigest(),
        "source_bytes": source_bytes,
        "documents_sha256": documents_digest.hexdigest(),
        "document_id_algorithm": "sha256(UTF-8 paragraph_text), lowercase hex",
        "deduplication": "exact UTF-8 text; first occurrence retained",
        "source_line_base": 1,
        "context_position_base": 0,
        "empty_context_policy": "missing, null, empty, or whitespace-only paragraph_text",
    }


def build_corpus(input_path, output_dir, *, expected_counts=None):
    """Publish documents.jsonl and summary.json as one complete directory.

    expected_counts are validation targets, never limits on input processing.
    A missing destination is required; no overwrite/resume option is provided.
    """
    source = Path(input_path).absolute()
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if not source.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Staging and publication share a filesystem, so the final rename is atomic.
    # The temporary SQLite index stays outside the published artifact directory.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "complete"
        staged.mkdir()
        summary = stream_documents(source, staged / "documents.jsonl", workspace / "seen.sqlite3")
        mismatches = {}
        if expected_counts is not None:
            mismatches = {key: {"expected": value, "actual": summary[key]}
                          for key, value in expected_counts.items() if summary[key] != value}
        if mismatches:
            raise ValueError("Dataset count validation failed: " + json.dumps(mismatches, sort_keys=True))
        summary["count_validation"] = {
            "status": "passed" if expected_counts is not None else "not_requested",
            "expected": expected_counts,
        }
        with (staged / "summary.json").open("xb") as stream:
            stream.write(json_bytes(summary))
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(staged)
        if os.path.lexists(output):
            raise FileExistsError(f"Output appeared during processing: {output}")
        # Never use os.replace: rename refuses an existing nonempty directory,
        # protecting completed artifacts even if concurrent builders finish here.
        os.rename(staged, output)
        sync_directory(output.parent)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "processed_data/2wikimultihopqa/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/full_2wiki_corpus")
    parser.add_argument("--skip-count-validation", action="store_true",
                        help="For synthetic/alternate inputs only; disables the full-dataset count check.")
    args = parser.parse_args(argv)
    try:
        summary = build_corpus(args.input, args.output_dir,
                               expected_counts=None if args.skip_count_validation else FULL_DATASET_COUNTS)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Corpus build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"Corpus written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
