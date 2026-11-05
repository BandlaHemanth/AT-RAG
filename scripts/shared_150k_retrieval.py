"""Read-only, exact shared retrieval over the frozen 150k training occurrences.

NumPy + standard library only. No model loading, encoding, database creation,
topic selection policy, LLM calls or evaluation. See docs/shared_150k_retrieval.md.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/tae_v2_150k_corpus/run_001/corpus.jsonl"
TRAINING = ROOT / "artifacts/tae_v2_150k_bertopic/run_001"
OCCURRENCES = 150_000
DIMENSIONS = 384
CORPUS_SHA256 = "d9edc9d9bc3ff4bbb98de8690c5b79c71aa6e2878f035df55cd01891ea322c33"
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NORM_TOLERANCE = 1e-5
CORPUS_FIELDS = {"occurrence_id", "paragraph_text", "text_sha256", "question_id", "source_split",
                 "source_line", "context_position", "idx", "title"}
ASSIGNMENT_FIELDS = (CORPUS_FIELDS - {"paragraph_text"}) | {"row", "topic", "bertopic", "membership_strength"}


class ArtifactValidationError(ValueError):
    """A frozen artifact is incomplete, corrupt, incompatible or misaligned."""


def require(condition, message):
    if not condition:
        raise ArtifactValidationError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ArtifactValidationError(f"Nonfinite JSON constant: {value}")


def _json(raw, label):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        require(isinstance(value, dict), "Expected a JSON object")
        return value
    except (ValueError, UnicodeError) as error:
        raise ArtifactValidationError(f"{label}: {error}") from error


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _positive_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


class Shared150kRetrieval:
    """Initialize once; reuse this object for single, multi and unfiltered search.

    expected_count/expected_corpus_sha256 exist for synthetic fixtures. Production
    callers must keep their frozen defaults; the CLI provides no such overrides.
    Keep source files immutable while open. Close the object after queries finish.
    """

    def __init__(self, corpus=CORPUS, embeddings=TRAINING / "embeddings.npy",
                 assignments=TRAINING / "occurrence_topics.jsonl", manifest=TRAINING / "manifest.json", *,
                 block_size=4096, expected_count=OCCURRENCES, expected_corpus_sha256=CORPUS_SHA256):
        self._closed = False
        self._matrix = None
        self._corpus_file = None
        self._read_lock = threading.Lock()
        self._block_size = _positive_integer(block_size, "block_size")
        self._count = _positive_integer(expected_count, "expected_count")
        require(_valid_hash(expected_corpus_sha256), "Invalid expected corpus hash")
        paths = tuple(Path(path).absolute() for path in (corpus, embeddings, assignments, manifest))
        self._paths = paths
        try:
            require(all(path.is_file() for path in paths), "Missing corpus/embedding/assignment/training-manifest file")
            self._stamps = tuple(_stamp(path) for path in paths)
            raw_manifest = paths[3].read_bytes()
            saved = _json(raw_manifest, "Training manifest")
            require(type(saved.get("schema_version")) is int and saved["schema_version"] == 1,
                    "Unsupported training manifest schema")
            require(saved.get("status") == "complete", "Training manifest status must be complete")
            require(type(saved.get("occurrences")) is int and saved["occurrences"] == self.occurrence_count,
                    f"Training count must be exactly {self.occurrence_count}")
            require(saved.get("corpus_sha256") == expected_corpus_sha256,
                    "Manifest corpus identity differs from the frozen protocol")
            require(saved.get("encoder_id") == ENCODER_ID and saved.get("encoder_revision") == ENCODER_REVISION,
                    "Training manifest identifies a different encoder/revision")
            records = saved.get("artifacts")
            require(isinstance(records, dict), "Missing artifact hashes in training manifest")
            evidence = {}
            for name, path in (("embeddings.npy", paths[1]), ("occurrence_topics.jsonl", paths[2])):
                record = records.get(name)
                require(isinstance(record, dict) and _valid_hash(record.get("sha256")) and
                        type(record.get("bytes")) is int and record["bytes"] >= 0,
                        f"Missing/invalid manifest hash evidence: {name}")
                require(path.stat().st_size == record["bytes"], f"Artifact byte-size mismatch: {name}")
                evidence[name] = record
            require(_sha256(paths[1]) == evidence["embeddings.npy"]["sha256"], "Embedding SHA-256 mismatch")
            self._matrix = np.load(paths[1], mmap_mode="r", allow_pickle=False)
            require(isinstance(self._matrix, np.memmap), "Embeddings must be a memory-mappable .npy array")
            require(self._matrix.shape == (self.occurrence_count, DIMENSIONS), "Embedding shape mismatch")
            require(self._matrix.dtype == np.dtype("float32"), "Embeddings must have float32 dtype")
            require(not self._matrix.flags.writeable, "Embedding map must be read-only")
            self._norms = np.empty(self.occurrence_count, dtype=np.float64)
            for start in range(0, self.occurrence_count, self._block_size):
                stop = min(start + self._block_size, self.occurrence_count)
                block = self._matrix[start:stop]
                require(bool(np.isfinite(block).all()), "Nonfinite document embeddings")
                norms = np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64, optimize=False))
                require(bool(np.all(np.abs(norms - 1.0) <= NORM_TOLERANCE)),
                        "Document embeddings are not unit-normalized")
                self._norms[start:stop] = norms
            self._offsets = np.empty(self.occurrence_count, dtype=np.int64)
            self._topics = np.empty(self.occurrence_count, dtype=np.int64)
            self._corpus_file = paths[0].open("rb")
            corpus_digest, assignment_digest = hashlib.sha256(), hashlib.sha256()
            seen, counts = set(), Counter()
            with paths[2].open("rb") as topic_stream:
                for index in range(self.occurrence_count):
                    self._offsets[index] = self._corpus_file.tell()
                    raw_corpus, raw_topic = self._corpus_file.readline(), topic_stream.readline()
                    require(bool(raw_corpus) and bool(raw_topic), "Corpus/assignment row count is too small")
                    corpus_digest.update(raw_corpus)
                    assignment_digest.update(raw_topic)
                    row = _json(raw_corpus, f"Corpus row {index}")
                    topic = _json(raw_topic, f"Assignment row {index}")
                    self._validate_row(row, topic, index)
                    require(row["occurrence_id"] not in seen, "Duplicate corpus occurrence_id")
                    seen.add(row["occurrence_id"])
                    self._topics[index] = topic["topic"]
                    counts[topic["topic"]] += 1
                require(not self._corpus_file.read(1) and not topic_stream.read(1),
                        "Corpus/assignment has extra rows or blank lines")
            require(corpus_digest.hexdigest() == expected_corpus_sha256, "Corpus SHA-256 differs from frozen protocol")
            require(assignment_digest.hexdigest() == evidence["occurrence_topics.jsonl"]["sha256"],
                    "Assignment SHA-256 mismatch")
            summary = saved.get("topic_summary")
            require(isinstance(summary, dict) and type(summary.get("occurrences")) is int and
                    summary["occurrences"] == self.occurrence_count, "Invalid manifest topic-summary count")
            sizes = summary.get("topic_sizes")
            require(isinstance(sizes, dict) and all(type(value) is int and value > 0 for value in sizes.values()) and
                    sizes == {str(topic): count for topic, count in counts.items()},
                    "Manifest topic sizes differ from assignments")
            non_outliers = len(counts) - int(-1 in counts)
            require(type(summary.get("non_outlier_topics")) is int and summary["non_outlier_topics"] == non_outliers and
                    type(summary.get("outlier_occurrences")) is int and summary["outlier_occurrences"] == counts.get(-1, 0),
                    "Manifest topic/outlier totals differ from assignments")
            fraction = summary.get("outlier_fraction")
            require(type(fraction) in (int, float) and math.isfinite(fraction) and
                    math.isclose(fraction, counts.get(-1, 0) / self.occurrence_count, abs_tol=1e-12, rel_tol=0),
                    "Invalid manifest outlier fraction")
            for array in (self._topics, self._norms, self._offsets):
                array.flags.writeable = False
            self._available = frozenset(counts)
            self._report = {
                "status": "validated", "occurrences": self.occurrence_count,
                "unique_occurrence_ids": len(seen), "dimensions": DIMENSIONS, "dtype": "float32",
                "corpus_sha256": corpus_digest.hexdigest(),
                "embeddings_sha256": evidence["embeddings.npy"]["sha256"],
                "assignments_sha256": assignment_digest.hexdigest(),
                "training_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                "encoder_id": ENCODER_ID, "encoder_revision": ENCODER_REVISION,
                "non_outlier_topics": non_outliers, "outlier_occurrences": counts.get(-1, 0),
                "topic_sizes": {str(topic): count for topic, count in sorted(counts.items())},
                "ranking": "float64 cosine reductions over saved float32 vectors; descending score, ascending corpus row",
                "normalization_tolerance": NORM_TOLERANCE, "integrity_validation": "once at initialization",
            }
            self._check_sources()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate_row(row, assignment, index):
        require(row.keys() == CORPUS_FIELDS and assignment.keys() == ASSIGNMENT_FIELDS,
                f"Unrecognized corpus/assignment fields at row {index}")
        require(isinstance(row["paragraph_text"], str) and bool(row["paragraph_text"].strip()), "Invalid paragraph text")
        require(row["text_sha256"] == hashlib.sha256(row["paragraph_text"].encode("utf-8")).hexdigest(),
                f"Paragraph text hash mismatch at row {index}")
        require(row["source_split"] == "train", "Nontraining corpus source")
        require(isinstance(row["question_id"], str) and bool(row["question_id"].strip()) and
                isinstance(row["title"], str), "Invalid question/title provenance")
        for key in ("source_line", "context_position", "idx"):
            require(type(row[key]) is int and row[key] >= (1 if key == "source_line" else 0), f"Invalid {key}")
        oid = row["occurrence_id"]
        prefix = "2wikimultihopqa:train:"
        require(isinstance(oid, str) and oid.startswith(prefix), "Invalid source-qualified occurrence ID")
        parts = oid[len(prefix):].split(":")
        require(len(parts) == 3 and _valid_hash(parts[0]) and
                parts[1:] == [str(row["source_line"]), str(row["context_position"])], "Occurrence provenance mismatch")
        require(type(assignment["row"]) is int and assignment["row"] == index, "Assignment corpus-row mismatch")
        require(all(type(assignment[key]) is type(row[key]) and assignment[key] == row[key]
                    for key in CORPUS_FIELDS - {"paragraph_text"}),
                f"Assignment occurrence/text/provenance mismatch at row {index}")
        require(type(assignment["topic"]) is int and -1 <= assignment["topic"] <= np.iinfo(np.int64).max,
                "Invalid assigned topic ID")
        require(assignment["bertopic"] == f"Topic {assignment['topic']}", "Assigned topic metadata differs")
        strength = assignment["membership_strength"]
        require(type(strength) in (int, float) and math.isfinite(strength) and 0 <= strength <= 1 + 1e-6,
                "Invalid saved membership strength")

    @property
    def occurrence_count(self):
        return self._count

    @property
    def validation_report(self):
        return copy.deepcopy(self._report)

    @property
    def available_topics(self):
        return tuple(sorted(self._available))

    def _check_sources(self):
        # Cheap metadata guards only: no per-query hashes or full artifact scans.
        require(tuple(_stamp(path) for path in self._paths) == self._stamps,
                "Source artifacts changed after validation; construct a new engine")

    @staticmethod
    def _topic_id(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < -1:
            raise ValueError("Topic IDs must be integers >= -1; -1 is selected explicitly")
        return int(value)

    def _filter(self, mode, topic_id, topic_ids):
        if not isinstance(mode, str) or mode not in ("single", "multi", "unfiltered"):
            raise ValueError("mode must be single, multi or unfiltered")
        if mode == "unfiltered":
            if topic_id is not None or topic_ids is not None:
                raise ValueError("Unfiltered mode does not accept topic filters")
            return None
        if mode == "single":
            if topic_ids is not None:
                raise ValueError("Single-topic mode accepts topic_id only")
            return frozenset((self._topic_id(topic_id),)) & self._available
        if topic_id is not None or not isinstance(topic_ids, (list, tuple, set, frozenset, np.ndarray)):
            raise ValueError("Multi-topic mode requires a finite collection in topic_ids and no topic_id")
        if isinstance(topic_ids, np.ndarray) and topic_ids.ndim != 1:
            raise ValueError("topic_ids must be one-dimensional")
        return frozenset(self._topic_id(value) for value in topic_ids) & self._available

    @staticmethod
    def _query(query_embedding):
        query = np.asarray(query_embedding)
        if query.shape != (DIMENSIONS,) or query.dtype.kind != "f":
            raise ValueError(f"Query embedding must be a floating vector of shape ({DIMENSIONS},)")
        query = np.asarray(query, dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or not math.isfinite(norm) or abs(norm - 1.0) > NORM_TOLERANCE:
            raise ValueError("Query embedding must be finite and unit-normalized")
        return query, norm

    def retrieve(self, query_embedding, *, mode, topic_id=None, topic_ids=None, k=5):
        """Return min(k, eligible_count) occurrence dictionaries without deduplication."""
        if self._closed:
            raise RuntimeError("Retrieval engine is closed")
        limit = min(_positive_integer(k, "k"), self.occurrence_count)
        selected = self._filter(mode, topic_id, topic_ids)
        query, query_norm = self._query(query_embedding)
        self._check_sources()
        if selected == frozenset():
            return []
        selected_array = None if selected is None else np.array(sorted(selected), dtype=np.int64)
        best_rows, best_scores = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        for start in range(0, self.occurrence_count, self._block_size):
            stop = min(start + self._block_size, self.occurrence_count)
            if selected_array is None:
                rows = np.arange(start, stop, dtype=np.int64)
                block = self._matrix[start:stop]
            else:
                local = np.flatnonzero(np.isin(self._topics[start:stop], selected_array))
                if not len(local):
                    continue
                rows = local + start
                # At most block_size rows are copied, never the full eligible matrix.
                block = self._matrix[start:stop]
                if len(local) != stop - start:
                    block = block[local]
            scores = np.einsum("ij,j->i", block, query, dtype=np.float64, optimize=False)
            scores /= self._norms[rows] * query_norm
            np.clip(scores, -1.0, 1.0, out=scores)  # bound floating roundoff
            order = np.lexsort((rows, -scores))[:limit]
            merged_rows = np.concatenate((best_rows, rows[order]))
            merged_scores = np.concatenate((best_scores, scores[order]))
            order = np.lexsort((merged_rows, -merged_scores))[:limit]
            best_rows, best_scores = merged_rows[order], merged_scores[order]
        results = []
        for index, score in zip(best_rows, best_scores):
            with self._read_lock:
                self._corpus_file.seek(int(self._offsets[index]))
                # Text is fetched only for final hits, from the validated read-only source.
                row = json.loads(self._corpus_file.readline())
            results.append({"row": int(index), "occurrence_id": row["occurrence_id"],
                            "topic": int(self._topics[index]), "text_sha256": row["text_sha256"],
                            "similarity": float(score), "paragraph_text": row["paragraph_text"]})
        return results

    def close(self):
        if self._corpus_file is not None:
            self._corpus_file.close()
            self._corpus_file = None
        if isinstance(self._matrix, np.memmap):
            self._matrix._mmap.close()
        elif self._matrix is not None and hasattr(self._matrix, "close"):
            self._matrix.close()
        self._matrix = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Retrieval engine is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--embeddings", type=Path, default=TRAINING / "embeddings.npy")
    parser.add_argument("--assignments", type=Path, default=TRAINING / "occurrence_topics.jsonl")
    parser.add_argument("--manifest", type=Path, default=TRAINING / "manifest.json")
    parser.add_argument("--query-embedding", type=Path, required=True, help="Existing normalized (384,) .npy query vector")
    parser.add_argument("--mode", required=True, choices=("single", "multi", "unfiltered"))
    parser.add_argument("--topic", type=int)
    parser.add_argument("--topics", type=int, nargs="*")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        with args.query_embedding.open("rb") as stream:
            query = np.load(stream, allow_pickle=False)
        with Shared150kRetrieval(args.corpus, args.embeddings, args.assignments, args.manifest,
                                block_size=args.block_size) as engine:
            results = engine.retrieve(query, mode=args.mode, topic_id=args.topic, topic_ids=args.topics, k=args.k)
            print(json.dumps({"validation": engine.validation_report, "mode": args.mode, "k": args.k,
                              "results": results}, ensure_ascii=False, allow_nan=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Retrieval failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
