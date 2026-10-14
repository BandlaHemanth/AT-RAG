"""Offline evidence eligibility under AT-RAG's exact topic equality filter.

Uses saved topics only: no ML imports, embedding generation, fitting, ranking,
fallbacks, or API calls. Gold labels select evaluation occurrences, never topics.
Run: python -B scripts/analyze_topic_filter_retention.py
Outputs are never overwritten. Validation failure produces diagnostics only.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "config.json", "development_questions.jsonl", "candidate_corpus.jsonl",
    "candidate_topic_assignments.jsonl", "query_topic_assignments.jsonl",
    "development_context_manifest.jsonl",
    "development_context_topic_assignments.jsonl",
]


class ValidationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def keyed(records, key, label):
    result = {}
    for row in records:
        value = row[key]
        require(value not in result, f"Duplicate {label}: {value}")
        result[value] = row
    return result


def valid_topic(row):
    require(type(row.get("topic")) is int and row["topic"] >= -1,
            "Invalid integer topic assignment")
    require(row.get("bertopic") == f"Topic {row['topic']}",
            "Integer topic and AT-RAG metadata string disagree")


def mapping(context, question_id, position, text_index, candidates, occurrences):
    """Resolve exact text using the saved question-local occurrence metadata.

    Repeated gold text remains multiple occurrences. Multiple candidate rows
    with identical text can be resolved only by a unique, metadata-consistent
    saved occurrence link. idx is checked locally, never used as a global key.
    """
    matches = text_index.get(context["paragraph_text"], [])
    if not matches:
        return "unmatched", None, "Exact paragraph text absent from candidate corpus"
    occurrence = occurrences.get(f"dev:{question_id}:{position}")
    if occurrence is None:
        return ("ambiguous" if len(matches) > 1 else "invalid_metadata"), None, "Saved occurrence missing"
    fields = {"question_id": question_id, "context_position": position,
              "idx": context.get("idx"), "title": context.get("title"),
              "paragraph_text": context["paragraph_text"]}
    if any(occurrence.get(key) != value for key, value in fields.items()):
        return "invalid_metadata", None, "Saved occurrence metadata differs from development context"
    candidate_position = occurrence.get("candidate_position")
    if type(candidate_position) is not int or candidate_position not in matches:
        return ("ambiguous" if len(matches) > 1 else "invalid_metadata"), None, "Saved candidate position does not resolve exact text"
    candidate = candidates[candidate_position]
    if candidate["document_id"] != occurrence.get("document_id"):
        return "invalid_metadata", None, "Occurrence and candidate document IDs differ"
    return "verified", candidate, "exact text + question/position + idx/title + saved candidate link"


def aggregate(questions):
    n = len(questions)
    gold = sum(row["gold_supporting_paragraphs"] for row in questions)
    covered = sum(row["gold_verified_present"] for row in questions)
    retained = sum(row["gold_retained"] for row in questions)
    excluded = sum(row["gold_excluded"] for row in questions)
    all_retained = sum(row["all_gold_evidence_retained"] for row in questions)
    require(retained + excluded == covered, "Aggregate denominator mismatch")
    return {
        "questions": n, "gold_supporting_occurrences": gold,
        "verified_corpus_occurrences": covered,
        "verified_corpus_coverage_percentage": 100 * covered / gold if gold else None,
        "gold_retained": retained, "gold_excluded": excluded,
        "micro_retention_percentage": 100 * retained / covered if covered else None,
        "macro_per_question_retention_percentage":
            100 * sum(row["evidence_retention_ratio"] for row in questions) / n if n else None,
        "questions_losing_at_least_one_gold": n - all_retained,
        "questions_losing_at_least_one_gold_percentage": 100 * (n - all_retained) / n if n else None,
        "questions_retaining_all_gold": all_retained,
        "questions_retaining_all_gold_percentage": 100 * all_retained / n if n else None,
        "unfiltered_pre_ranking_reference": {
            "eligible_gold_occurrences": covered,
            "evidence_availability_percentage": 100 * covered / gold if gold else None,
            "questions_with_all_gold_available": sum(row["gold_verified_present"] == row["gold_supporting_paragraphs"] for row in questions),
        },
    }


def analyze(pilot, development, validation, gold_rows):
    expected_hashes = json.loads((pilot / "artifact_hashes.json").read_text())
    for name in INPUTS:
        require(name in expected_hashes, f"No pilot hash recorded for {name}")
        actual = sha256(pilot / name)
        require(actual == expected_hashes[name], f"Pilot input hash mismatch: {name}")
        validation["input_sha256"][name] = actual
    config = json.loads((pilot / "config.json").read_text())
    require(config["development_questions"] == 20 and config["training_paragraphs"] == 500,
            "Expected the fixed 500-training/20-development pilot")
    require(config["held_out_test_used"] is False and config["development_used_for_fitting_or_tuning"] is False,
            "Pilot split policy differs from the expected experiment")
    questions = list(islice(jsonl(development), 20))
    saved_questions = list(jsonl(pilot / "development_questions.jsonl"))
    require(len(questions) == 20 and questions == saved_questions,
            "Saved development questions differ from the first 20 source questions")
    validation["development_source_sha256"] = sha256(development)
    qmap = keyed(questions, "question_id", "development question ID")
    query_assignments = keyed(list(jsonl(pilot / "query_topic_assignments.jsonl")), "question_id", "query topic ID")
    require(qmap.keys() == query_assignments.keys(), "Query assignment question IDs differ")
    for qid, assignment in query_assignments.items():
        valid_topic(assignment)
        require(assignment["question_text"] == qmap[qid]["question_text"], "Query assignment text differs")
    candidates = list(jsonl(pilot / "candidate_corpus.jsonl"))
    cmap = keyed(candidates, "document_id", "candidate document ID")
    candidate_assignments = keyed(list(jsonl(pilot / "candidate_topic_assignments.jsonl")), "document_id", "candidate topic ID")
    require(cmap.keys() == candidate_assignments.keys(), "Candidate assignment IDs differ")
    text_index = defaultdict(list)
    for position, candidate in enumerate(candidates):
        text = candidate["paragraph_text"]
        require(isinstance(text, str) and bool(text.strip()), "Empty candidate paragraph")
        require(hashlib.sha256(text.encode("utf-8")).hexdigest() == candidate["document_id"],
                "Candidate exact-text hash differs from document ID")
        text_index[text].append(position)
        valid_topic(candidate_assignments[candidate["document_id"]])
    # The pilot's exact-text deduplication contract forbids duplicate candidate
    # rows. Repeated source context occurrences are allowed and counted below.
    require(all(len(positions) == 1 for positions in text_index.values()),
            "Duplicate exact text in deduplicated pilot candidate corpus")
    occurrences = keyed(list(jsonl(pilot / "development_context_manifest.jsonl")), "occurrence_id", "context occurrence ID")
    context_assignments = keyed(list(jsonl(pilot / "development_context_topic_assignments.jsonl")), "occurrence_id", "context assignment occurrence ID")
    expected_occurrence_ids = {f"dev:{qid}:{position}" for qid, q in qmap.items() for position in range(len(q["contexts"]))}
    require(occurrences.keys() == context_assignments.keys() == expected_occurrence_ids,
            "Context occurrences/assignments do not exactly cover development contexts")
    mapping_errors = []
    for q in questions:
        qid = q["question_id"]
        require(isinstance(qid, str) and bool(qid.strip()), "Invalid question ID")
        require(isinstance(q["question_text"], str) and bool(q["question_text"].strip()), "Invalid question text")
        require(isinstance(q["contexts"], list) and bool(q["contexts"]), "Empty question contexts")
        query_topic = query_assignments[qid]["topic"]
        for position, context in enumerate(q["contexts"]):
            require(type(context.get("is_supporting")) is bool, "Supporting flag must be boolean")
            require(isinstance(context.get("paragraph_text"), str) and bool(context["paragraph_text"].strip()), "Invalid context paragraph text")
            occurrence_id = f"dev:{qid}:{position}"
            status, candidate, detail = mapping(context, qid, position, text_index, candidates, occurrences)
            if status != "verified":
                mapping_errors.append({"occurrence_id": occurrence_id, "gold": context["is_supporting"], "status": status, "detail": detail})
            else:
                saved = context_assignments[occurrence_id]
                require(all(saved.get(key) == value for key, value in occurrences[occurrence_id].items()), "Context assignment metadata differs from manifest")
                valid_topic(saved)
                require(saved["topic"] == candidate_assignments[candidate["document_id"]]["topic"], "Context/candidate topic mismatch")
            if context["is_supporting"]:
                candidate_topic = candidate_assignments[candidate["document_id"]]["topic"] if candidate else None
                gold_rows.append({
                    "question_id": qid, "query_topic": query_topic,
                    "occurrence_id": occurrence_id, "context_position": position,
                    "idx": context.get("idx"), "title": context.get("title"),
                    "paragraph_text": context["paragraph_text"],
                    "document_id": candidate["document_id"] if candidate else "",
                    "candidate_topic": candidate_topic, "mapping_status": status,
                    "mapping_detail": detail, "verified_present": status == "verified",
                    "retained_by_exact_filter": query_topic == candidate_topic if status == "verified" else None,
                    "excluded_by_exact_filter": query_topic != candidate_topic if status == "verified" else None,
                })
    validation["mapping_errors"] = mapping_errors
    validation["gold_mapping_status_counts"] = dict(Counter(row["mapping_status"] for row in gold_rows))
    validation["duplicate_handling"] = {
        "candidate_duplicate_text_rows": sum(len(v) - 1 for v in text_index.values()),
        "gold_repeated_text_occurrences": len(gold_rows) - len({row["paragraph_text"] for row in gold_rows}),
        "gold_repeated_text_within_question_occurrences": len(gold_rows) - len({(row["question_id"], row["paragraph_text"]) for row in gold_rows}),
        "policy": "Count every gold occurrence; match exact text and question-local metadata; never join on idx alone",
    }
    require(not mapping_errors, "Unmatched, ambiguous, or inconsistent context mappings; retention metrics withheld")
    require(bool(gold_rows), "No gold supporting occurrences")
    by_question = defaultdict(list)
    for row in gold_rows:
        by_question[row["question_id"]].append(row)
    per_question = []
    topic_counts = Counter(row["topic"] for row in candidate_assignments.values())
    for q in questions:
        qid = q["question_id"]
        gold = by_question[qid]
        require(bool(gold), f"No gold supporting contexts for question {qid}")
        covered = sum(row["verified_present"] for row in gold)
        retained = sum(row["retained_by_exact_filter"] for row in gold)
        excluded = sum(row["excluded_by_exact_filter"] for row in gold)
        require(retained + excluded == covered == len(gold), "Per-question denominator mismatch")
        topic = query_assignments[qid]["topic"]
        per_question.append({"question_id": qid, "query_topic": topic,
            "gold_supporting_paragraphs": len(gold), "gold_verified_present": covered,
            "gold_retained": retained, "gold_excluded": excluded,
            "evidence_retention_ratio": retained / covered,
            "all_gold_evidence_retained": retained == len(gold),
            "query_topic_is_minus_1": topic == -1,
            "unfiltered_eligible_gold": covered,
            "eligible_candidate_documents_after_filter": topic_counts[topic]})
    summary = {
        "status": "passed", "diagnostic": "pre-ranking gold evidence eligibility under exact AT-RAG topic equality",
        "filter_rule": "candidate_topic == query_topic; -1 is an ordinary equality match; no fallback",
        "retention_denominator": "verified corpus gold supporting-paragraph occurrences; metrics withheld if any mapping is unresolved",
        "aggregate": aggregate(per_question),
        "by_query_topic_group": {label: aggregate([row for row in per_question if row["query_topic_is_minus_1"] == is_outlier])
                                 for label, is_outlier in [("topic_minus_1", True), ("non_outlier", False)]},
        "candidate_corpus_documents": len(candidates),
        "unmatched_gold_occurrences": sum(row["mapping_status"] == "unmatched" for row in gold_rows),
        "ambiguous_gold_occurrences": sum(row["mapping_status"] == "ambiguous" for row in gold_rows),
        "limitations": ["20 development questions, fixed small pilot; descriptive results only",
                        "Candidate corpus includes all selected development contexts plus training paragraphs",
                        "Gold occurrences, including repeated text, are counted separately",
                        "Pre-ranking eligibility only; no similarity ranking, recall@k, answer accuracy, or full AT-RAG execution",
                        "No parameter tuning, fallback, confidence threshold, or Topic Adequacy Evaluation"],
    }
    validation.update({"status": "passed", "checks": [
        "saved input hashes", "first 20 source questions match saved records",
        "unique question/candidate/occurrence IDs", "complete topic assignment joins",
        "exact text hashes and question-local metadata", "context/candidate topic agreement",
        "boolean gold labels", "occurrence-preserving duplicate handling",
        "per-question and aggregate denominators"], "questions": len(questions),
        "context_occurrences": len(occurrences), "script_sha256": sha256(Path(__file__))})
    return per_question, summary


def markdown(summary):
    a = summary["aggregate"]
    lines = ["# Exact topic-filter evidence-retention diagnostic", "",
        "This measures pre-ranking evidence availability in the saved pilot candidate corpus. It does not measure retrieval recall@5 or answer accuracy.", "",
        f"All {a['gold_supporting_occurrences']} gold supporting-paragraph occurrences were verified in the corpus ({a['verified_corpus_coverage_percentage']:.2f}% coverage).",
        f"Exact topic equality retained {a['gold_retained']} and excluded {a['gold_excluded']} occurrences. Micro retention was {a['micro_retention_percentage']:.2f}%; macro per-question retention was {a['macro_per_question_retention_percentage']:.2f}%.",
        f"{a['questions_losing_at_least_one_gold']}/{a['questions']} questions lost at least one gold paragraph ({a['questions_losing_at_least_one_gold_percentage']:.2f}%). {a['questions_retaining_all_gold']}/{a['questions']} retained all gold paragraphs ({a['questions_retaining_all_gold_percentage']:.2f}%).", "",
        "| Query topic group | Questions | Gold occurrences | Retained | Excluded | Micro retention | Macro retention | All gold retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, v in summary["by_query_topic_group"].items():
        micro = f"{v['micro_retention_percentage']:.2f}%" if v['micro_retention_percentage'] is not None else "N/A"
        macro = f"{v['macro_per_question_retention_percentage']:.2f}%" if v['macro_per_question_retention_percentage'] is not None else "N/A"
        lines.append(f"| {group} | {v['questions']} | {v['gold_supporting_occurrences']} | {v['gold_retained']} | {v['gold_excluded']} | {micro} | {macro} | {v['questions_retaining_all_gold']} |")
    lines += ["", f"Without filtering, the same corpus makes {a['verified_corpus_occurrences']} gold occurrences eligible across all {a['questions']} questions. Filtering removes the excluded occurrences before ranking can consider them. Retained evidence is eligible, but is not guaranteed to appear in a retrieved top-k set.", "",
        "Topic -1 was treated as an ordinary equality match. No gold labels altered topics or the filter. Exact text, saved occurrence links, question IDs, positions, idx and title were checked together. No missing or ambiguous mappings were counted as filter exclusions.", "", "## Limitations", ""]
    lines += ["- " + limitation for limitation in summary["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "artifacts/bertopic_pilot_500")
    parser.add_argument("--development-file", type=Path, default=ROOT / "processed_data/2wikimultihopqa/dev_subsampled.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/topic_filter_diagnostic_20")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        print(f"STOP: refusing to overwrite existing diagnostic outputs: {output}")
        return 1
    output.mkdir(parents=True, exist_ok=True)
    validation = {"status": "failed", "input_sha256": {}, "mapping_errors": []}
    gold_rows = []
    try:
        per_question, summary = analyze(args.pilot_dir.resolve(), args.development_file.resolve(), validation, gold_rows)
    except (ValidationError, KeyError, TypeError, ValueError, OSError) as exc:
        validation["status"] = "failed"
        validation["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "validation_report.json", validation)
        write_csv(output / "per_gold_paragraph.csv", gold_rows)
        print("STOP: " + validation["error"])
        return 1
    write_csv(output / "per_question.csv", per_question)
    write_csv(output / "per_gold_paragraph.csv", gold_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "validation_report.json", validation)
    (output / "report.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"OUTPUT: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
