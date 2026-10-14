"""Preregistered TAE V1 evaluation on development questions 21--100.

Frozen inference only. Gold labels enter scoring after routing is saved and
hashed. Measures pre-ranking evidence eligibility, never retrieval recall@k.
Run with the existing environment: python -B scripts/evaluate_tae_v1.py
Existing output directories containing files are never overwritten.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import runpy
import statistics
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
GATE_RULE = "UNFILTERED iff predicted_topic == -1 or membership_strength is nonfinite or membership_strength <= 0; otherwise STRICT_TOPIC_FILTER"
METHODS = ("C1_strict", "C2_unfiltered", "C3_tae_v1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def id_set_hash(ids):
    return text_id("\n".join(sorted(ids)))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    require(bool(rows), "Refusing to write an empty result table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_map(rows, key, label):
    result = {}
    for row in rows:
        require(row[key] not in result, f"Duplicate {label}: {row[key]}")
        result[row[key]] = row
    return result


def project_question(question):
    """Explicit inference allowlist: support flags and answers are not inputs."""
    return {"question_id": question["question_id"], "question_text": question["question_text"],
            "contexts": [{"idx": c.get("idx"), "title": c.get("title"),
                          "paragraph_text": c["paragraph_text"]} for c in question["contexts"]]}


def route_query(topic, strength):
    """The fixed rule has no labels, answers, corpus outcomes, or thresholds."""
    triggers = []
    if topic == -1:
        triggers.append("predicted_topic_minus_1")
    if not math.isfinite(strength):
        triggers.append("membership_strength_nonfinite")
    elif strength <= 0:
        triggers.append("membership_strength_nonpositive")
    return ("UNFILTERED", triggers) if triggers else ("STRICT_TOPIC_FILTER", ["non_outlier_positive_finite_strength"])


def prepare_corpus(development, training, exploratory):
    source = read_jsonl(development)
    require(len(source) == 100, "Expected exactly 100 development questions")
    # Project all records before selecting or building candidates. No gold fields
    # are passed beyond this point until scoring rereads the source later.
    projected = [project_question(q) for q in source]
    del source
    all_questions = unique_map(projected, "question_id", "development question ID")
    require(len(all_questions) == 100, "Development ID count differs")
    first, evaluation = projected[:20], projected[20:100]
    first_pairs = [(q["question_id"], q["question_text"]) for q in first]
    require(first_pairs == [(q["question_id"], q["question_text"]) for q in exploratory],
            "First 20 questions differ from the exploratory pilot")
    evaluation_ids = {q["question_id"] for q in evaluation}
    require(not evaluation_ids & {q["question_id"] for q in first}, "Exploratory/evaluation ID overlap")
    require(not {q["question_text"] for q in evaluation} & {q["question_text"] for q in first},
            "Exploratory/evaluation query-text overlap")
    require(not evaluation_ids & {t["question_id"] for t in training}, "Training/evaluation question-ID overlap")
    require(len(training) == 500, "Expected 500 frozen training paragraphs")
    candidates, text_positions, occurrences = [], {}, []
    for t in training:
        text = t["paragraph_text"]
        require(t["split"] == "train" and text_id(text) == t["document_id"], "Invalid training manifest provenance")
        require(isinstance(text, str) and bool(text.strip()) and text not in text_positions,
                "Training manifest contains empty or duplicate exact text")
        text_positions[text] = len(candidates)
        candidates.append({"document_id": t["document_id"], "paragraph_text": text, "origin_splits": ["train"]})
    for question_number, q in enumerate(evaluation, 21):
        require(isinstance(q["question_id"], str) and bool(q["question_id"].strip()), "Invalid evaluation question ID")
        require(isinstance(q["question_text"], str) and bool(q["question_text"].strip()), "Invalid evaluation question text")
        require(bool(q["contexts"]), "Evaluation question has no contexts")
        for position, c in enumerate(q["contexts"]):
            text = c["paragraph_text"]
            require(isinstance(text, str) and bool(text.strip()), "Empty evaluation paragraph; cannot silently skip")
            if text not in text_positions:
                text_positions[text] = len(candidates)
                candidates.append({"document_id": text_id(text), "paragraph_text": text,
                                   "origin_splits": ["evaluation_development_context"]})
            candidate_position = text_positions[text]
            candidate = candidates[candidate_position]
            if "evaluation_development_context" not in candidate["origin_splits"]:
                candidate["origin_splits"].append("evaluation_development_context")
            occurrences.append({"occurrence_id": f"dev:{q['question_id']}:{position}",
                "question_id": q["question_id"], "source_question_number": question_number,
                "context_position": position, "idx": c["idx"], "title": c["title"],
                "paragraph_text": text, "document_id": candidate["document_id"],
                "candidate_position": candidate_position})
    unique_map(candidates, "document_id", "candidate document ID")
    unique_map(occurrences, "occurrence_id", "context occurrence ID")
    return evaluation, candidates, occurrences


def summarize(rows, corpus_size):
    gold = sum(row["gold_occurrences"] for row in rows)
    coverage = sum(row["verified_corpus_occurrences"] for row in rows)
    retained = sum(row["gold_retained"] for row in rows)
    excluded = sum(row["gold_excluded"] for row in rows)
    require(retained + excluded == coverage == gold, "Aggregate denominator mismatch")
    sizes = [row["candidate_documents"] for row in rows]
    n = len(rows)
    all_retained = sum(row["all_gold_retained"] for row in rows)
    return {"questions": n, "gold_occurrences": gold,
        "verified_corpus_occurrences": coverage, "corpus_coverage_percentage": 100 * coverage / gold,
        "gold_retained": retained, "gold_excluded": excluded,
        "micro_retention_percentage": 100 * retained / coverage,
        "macro_question_retention_percentage": 100 * statistics.mean(row["retention_ratio"] for row in rows),
        "questions_retaining_all_gold": all_retained,
        "questions_retaining_all_gold_percentage": 100 * all_retained / n,
        "questions_losing_at_least_one_gold": n - all_retained,
        "questions_losing_at_least_one_gold_percentage": 100 * (n - all_retained) / n,
        "candidate_documents": {"mean": statistics.mean(sizes), "median": statistics.median(sizes),
            "minimum": min(sizes), "maximum": max(sizes),
            "mean_survivor_fraction": statistics.mean(sizes) / corpus_size,
            "total_query_document_eligibilities": sum(sizes)}}


def score(development, inference_questions, candidates, occurrences, candidate_assignments, routing):
    """Gold labels are first used here, after all routes have been persisted."""
    original = read_jsonl(development)[20:100]
    require([project_question(q) for q in original] == inference_questions,
            "Source records changed after inference")
    source_questions = unique_map(original, "question_id", "scoring question ID")
    cmap = unique_map(candidates, "document_id", "candidate document ID")
    amap = unique_map(candidate_assignments, "document_id", "candidate topic ID")
    require(cmap.keys() == amap.keys(), "Candidate assignment IDs differ from corpus")
    rmap = unique_map(routing, "question_id", "routing question ID")
    require(rmap.keys() == source_questions.keys(), "Routing/source IDs differ")
    omap = unique_map(occurrences, "occurrence_id", "context occurrence ID")
    text_positions = defaultdict(list)
    topic_ids = defaultdict(set)
    for position, candidate in enumerate(candidates):
        text_positions[candidate["paragraph_text"]].append(position)
        topic_ids[amap[candidate["document_id"]]["topic"]].add(candidate["document_id"])
    full_ids = set(cmap)
    require(all(len(v) == 1 for v in text_positions.values()), "Ambiguous duplicate candidate text")
    per_question, per_gold, gold_manifest = [], [], []
    gold_document_ids = []
    for q in original:
        qid = q["question_id"]
        decision = rmap[qid]
        strict_ids = topic_ids[decision["predicted_topic"]]
        gated_ids = full_ids if decision["route"] == "UNFILTERED" else strict_ids
        strategies = {"C1_strict": strict_ids, "C2_unfiltered": full_ids, "C3_tae_v1": gated_ids}
        require(strict_ids <= full_ids and gated_ids <= full_ids, "A strategy changed the candidate corpus")
        retained = Counter()
        gold_count = 0
        for position, context in enumerate(q["contexts"]):
            require(type(context.get("is_supporting")) is bool, "Supporting flags must be boolean")
            oid = f"dev:{qid}:{position}"
            require(oid in omap, f"Missing context occurrence: {oid}")
            occurrence = omap[oid]
            require(occurrence["question_id"] == qid and occurrence["context_position"] == position,
                    "Context occurrence identifiers differ")
            require(all(occurrence[key] == context.get(key) for key in ["idx", "title", "paragraph_text"]),
                    "Context exact text or metadata differs")
            matches = text_positions.get(context["paragraph_text"], [])
            require(len(matches) == 1, f"Missing/ambiguous gold-to-corpus mapping: {oid}")
            candidate = candidates[matches[0]]
            require(occurrence["candidate_position"] == matches[0] and occurrence["document_id"] == candidate["document_id"],
                    "Saved context/candidate link differs")
            gold_manifest.append({**occurrence, "is_supporting": context["is_supporting"]})
            if not context["is_supporting"]:
                continue
            gold_count += 1
            document_id = candidate["document_id"]
            gold_document_ids.append(document_id)
            outcomes = {method: document_id in eligible for method, eligible in strategies.items()}
            for method, survives in outcomes.items():
                retained[method] += int(survives)
            per_gold.append({"question_id": qid, "occurrence_id": oid,
                "context_position": position, "idx": context.get("idx"), "title": context.get("title"),
                "paragraph_text": context["paragraph_text"], "document_id": document_id,
                "predicted_query_topic": decision["predicted_topic"], "candidate_topic": amap[document_id]["topic"],
                "mapping_status": "verified_exact_text_and_occurrence_metadata", "verified_present": True,
                **{method + "_retained": survives for method, survives in outcomes.items()}})
        require(gold_count > 0, f"Question has no gold supporting contexts: {qid}")
        for method in METHODS:
            count = retained[method]
            require(0 <= count <= gold_count and count + (gold_count - count) == gold_count, "Per-question denominator mismatch")
            per_question.append({"question_id": qid, "method": method,
                "predicted_topic": decision["predicted_topic"], "membership_strength": decision["membership_strength"],
                "tae_route": decision["route"], "tae_routing_reasons": decision["routing_reasons"],
                "gold_occurrences": gold_count, "verified_corpus_occurrences": gold_count,
                "gold_retained": count, "gold_excluded": gold_count - count,
                "retention_ratio": count / gold_count, "all_gold_retained": count == gold_count,
                "candidate_documents": len(strategies[method]),
                "survivor_fraction": len(strategies[method]) / len(candidates),
                "eligible_candidate_id_set_sha256": id_set_hash(strategies[method])})
    require(len(gold_manifest) == len(occurrences), "Not all context occurrences were scored")
    metrics = {method: summarize([r for r in per_question if r["method"] == method], len(candidates)) for method in METHODS}
    require(all(v["gold_occurrences"] == len(per_gold) for v in metrics.values()), "Method gold denominators differ")
    require(metrics["C2_unfiltered"]["gold_excluded"] == 0, "Unfiltered corpus does not contain all gold evidence")
    for qid in source_questions:
        pair = {r["method"]: r for r in per_question if r["question_id"] == qid}
        reference = "C2_unfiltered" if rmap[qid]["route"] == "UNFILTERED" else "C1_strict"
        require(pair["C3_tae_v1"]["eligible_candidate_id_set_sha256"] == pair[reference]["eligible_candidate_id_set_sha256"], "TAE candidate set differs from its logged route")
    duplicates = {"gold_occurrences": len(per_gold), "unique_gold_documents": len(set(gold_document_ids)),
        "repeated_gold_document_occurrences": len(gold_document_ids) - len(set(gold_document_ids)),
        "candidate_duplicate_exact_text": 0, "policy": "Count occurrences separately; deduplicate corpus by exact text; never join on idx alone"}
    return original, gold_manifest, per_question, per_gold, metrics, duplicates


def report_markdown(summary):
    lines = ["# Frozen TAE V1 evaluation: 80 development questions", "",
        "Preregistered rule: " + GATE_RULE + ".", "",
        f"The same {summary['candidate_documents']} exact-text-deduplicated candidates were used for all {summary['evaluation_queries']} queries and all three methods.", "",
        "| Method | Gold retained/total | Micro retention | Macro retention | All gold retained | Losing gold | Mean candidates | Mean survivor fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, v in summary["methods"].items():
        sizes = v["candidate_documents"]
        lines.append(f"| {method} | {v['gold_retained']}/{v['gold_occurrences']} | {v['micro_retention_percentage']:.2f}% | {v['macro_question_retention_percentage']:.2f}% | {v['questions_retaining_all_gold']}/{v['questions']} | {v['questions_losing_at_least_one_gold']}/{v['questions']} | {sizes['mean']:.2f} | {100*sizes['mean_survivor_fraction']:.2f}% |")
    f = summary["fallback"]
    lines += ["", f"TAE used unfiltered eligibility for {f['queries']}/{summary['evaluation_queries']} queries ({f['percentage']:.2f}%). Routing decisions were written and hashed before any gold-support scoring.", "",
        f"Compared with strict filtering, TAE changed micro eligibility by {summary['paired_comparison']['micro_retention_change_percentage_points']:.2f} percentage points and mean eligible candidates by {summary['paired_comparison']['mean_candidate_increase']:.2f}. These changes are descriptive, paired measurements on this corpus.", "", "## Limits", ""]
    lines += ["- " + s for s in summary["limitations"]]
    lines += ["", "## Runtime and memory", "", f"Total evaluator time: {summary['timings_seconds']['total_execution']:.2f} seconds. Peak process RSS: {summary['memory']['final'].get('process_peak_rss_mib', 'unavailable')} MiB. Cgroup usage includes other processes and charged cache; candidate counts are search-space measurements, not latency estimates.", ""]
    return "\n".join(lines)


def execute(args, output, validation, memory_snapshot):
    started = time.perf_counter()
    timings, memory, events = {}, {"preflight": memory_snapshot()}, []
    stage = "provenance"
    def mark(name):
        events.append({"event": name, "elapsed_seconds": time.perf_counter() - started,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_json(output / "execution_events.json", events)
    try:
        require(memory["preflight"]["available_memory_bytes"] >= 2 * 1024**3, "Insufficient available memory (<2 GiB)")
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        require(branch == "thesis/topic-filter-pilot", "Unexpected Git branch")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        pilot = args.pilot_dir.resolve()
        development = args.development_file.resolve()
        model_file = pilot / "bertopic_model.pkl"
        pilot_hashes = json.loads((pilot / "artifact_hashes.json").read_text())
        for name, expected in pilot_hashes.items():
            require(Path(name).name == name, "Unsafe pilot artifact hash path")
            require(sha256(pilot / name) == expected, f"Frozen pilot artifact hash mismatch: {name}")
        original_report = json.loads((pilot / "run_report.json").read_text())
        require(original_report["status"] == "passed", "Pilot run did not pass")
        config = json.loads((pilot / "config.json").read_text())
        require(config["training_paragraphs"] == 500 and config["development_questions"] == 20,
                "Unexpected pilot selection")
        require(config["seed"] == 42 and config["bertopic"]["nr_topics"] is None,
                "Pilot seed/topic reduction differs")
        require(config["held_out_test_used"] is False and config["development_used_for_fitting_or_tuning"] is False,
                "Pilot training split provenance differs")
        require(sha256(ROOT / "scripts/run_bertopic_pilot.py") == config["script_sha256"], "Pilot implementation changed")
        expected_packages = original_report["package_versions"]
        installed = {package: metadata.version(package) for package in expected_packages}
        require(installed == expected_packages, "Installed dependency versions differ from frozen pilot")
        helpers = runpy.run_path(str(ROOT / "scripts/run_bertopic_pilot.py"), run_name="tae_provenance_helpers")
        encoder_dir = args.encoder_dir.resolve()
        encoder_manifest = helpers["verify_encoder"](encoder_dir)
        require(encoder_manifest["model_id"] == config["embedding"]["model_id"] and encoder_manifest["revision"] == config["embedding"]["revision"], "Encoder and pilot provenance differ")
        training = read_jsonl(pilot / "training_manifest.jsonl")
        exploratory = read_jsonl(pilot / "development_questions.jsonl")
        original_corpus = read_jsonl(pilot / "candidate_corpus.jsonl")
        require([r["document_id"] for r in original_corpus[:500]] == [r["document_id"] for r in training], "Pilot training/candidate row order differs")
        stage = "label_independent_corpus_construction"
        queries, candidates, occurrences = prepare_corpus(development, training, exploratory)
        source_hash = sha256(development)
        provenance = {"gate_rule": GATE_RULE, "gate_thresholds_tuned": False,
            "branch": branch, "head": head, "script_sha256": sha256(Path(__file__)),
            "pilot_directory": str(pilot), "pilot_artifact_sha256": pilot_hashes,
            "frozen_model_sha256": sha256(model_file), "encoder_directory": str(encoder_dir),
            "encoder_manifest": encoder_manifest, "development_file": str(development),
            "development_file_sha256": source_hash,
            "training_manifest": str(pilot / "training_manifest.jsonl"),
            "evaluation_split": "development", "source_question_numbers": [21, 100],
            "selection_order": "First the 500 saved training paragraphs in manifest order; then every context of development records 21--100 in file/context order; exact text deduplication, no normalization",
            "candidate_selection_uses_gold_labels": False, "inference_fields": ["question_id", "question_text", "context idx/title/paragraph_text"],
            "evaluation_queries": len(queries), "candidate_documents": len(candidates),
            "context_occurrences": len(occurrences), "seed": SEED,
            "embedding_settings": config["embedding"], "frozen_umap_settings": config["umap"],
            "frozen_hdbscan_settings": config["hdbscan"], "frozen_bertopic_settings": config["bertopic"],
            "membership_semantics": "One scalar HDBSCAN approximate-prediction membership strength per query; not a calibrated evidence confidence or a complete topic distribution",
            "packages": installed, "python": sys.version, "executable": sys.executable, "platform": platform.platform()}
        write_json(output / "experiment_config.json", provenance)
        write_jsonl(output / "query_inputs.jsonl", [{"question_id": q["question_id"], "question_text": q["question_text"]} for q in queries])
        write_jsonl(output / "candidate_corpus.jsonl", candidates)
        write_jsonl(output / "context_occurrence_inputs.jsonl", occurrences)
        mark("label_independent_corpus_saved")
        timings["provenance_and_corpus"] = time.perf_counter() - started
        stage = "dependency_import_and_frozen_model_loading"
        before = time.perf_counter()
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        np.random.seed(SEED)
        random.seed(SEED)
        model = BERTopic.load(str(model_file))
        require(isinstance(model.hdbscan_model, HDBSCAN), "Frozen model is not the expected HDBSCAN model")
        require(model.calculate_probabilities is False and model.hdbscan_model.prediction_data is True,
                "Frozen probability configuration differs")
        require(model.embedding_model is None and model.nr_topics is None, "Unexpected frozen embedding backend or topic reduction")
        for key, expected in config["umap"].items():
            require(getattr(model.umap_model, key) == expected, f"Frozen UMAP parameter mismatch: {key}")
        for key, expected in config["hdbscan"].items():
            require(getattr(model.hdbscan_model, key) == expected, f"Frozen HDBSCAN parameter mismatch: {key}")
        for key, expected in config["vectorizer"].items():
            if key == "ngram_range": expected = tuple(expected)
            require(getattr(model.vectorizer_model, key) == expected, f"Frozen vectorizer parameter mismatch: {key}")
        fitted = read_jsonl(pilot / "training_assignments.jsonl")
        require(list(model.topics_) == [r["fitted_topic"] for r in fitted], "Frozen training topic assignments differ")
        frozen_centroids = np.array(model.topic_embeddings_, copy=True)
        encoder = SentenceTransformer(str(encoder_dir), device="cpu", local_files_only=True, trust_remote_code=False)
        encoder.max_seq_length = 256
        encoder.eval()
        require(encoder.device.type == "cpu", "Encoder did not load on CPU")
        timings["imports_and_model_loading"] = time.perf_counter() - before
        memory["after_model_loading"] = memory_snapshot()
        stage = "offline_encoding"
        before = time.perf_counter()
        # Reuse the exact 500 saved training embeddings. Encode only new context
        # paragraphs and the 80 query strings; no support flags reach the encoder.
        ce = np.load(pilot / "candidate_embeddings.npy", allow_pickle=False)[:500]
        with torch.inference_mode():
            if len(candidates) > 500:
                extra = encoder.encode([c["paragraph_text"] for c in candidates[500:]], batch_size=4,
                    device="cpu", normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                ce = np.concatenate([ce, extra], axis=0)
            qe = encoder.encode([q["question_text"] for q in queries], batch_size=4,
                device="cpu", normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        for array, count in [(ce, len(candidates)), (qe, 80)]:
            require(array.shape == (count, 384) and bool(np.isfinite(array).all()), "Invalid embedding shape or values")
            require(bool(np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5, rtol=0)), "Embeddings are not normalized")
        np.save(output / "candidate_embeddings.npy", ce, allow_pickle=False)
        np.save(output / "query_embeddings.npy", qe, allow_pickle=False)
        del encoder
        gc.collect()
        timings["encoding"] = time.perf_counter() - before
        memory["after_encoding"] = memory_snapshot()
        stage = "frozen_topic_inference"
        before = time.perf_counter()
        print(f"INFERENCE: {len(candidates)} candidates and 80 queries; frozen model only", flush=True)
        candidate_topics, candidate_strengths = model.transform([c["paragraph_text"] for c in candidates], embeddings=ce)
        query_topics, query_strengths = model.transform([q["question_text"] for q in queries], embeddings=qe)
        valid_topics = set(model.topics_)
        for topics, count in [(candidate_topics, len(candidates)), (query_topics, 80)]:
            require(len(topics) == count and all(isinstance(t, (int, np.integer)) and int(t) in valid_topics for t in topics), "Invalid predicted topic IDs")
        require(isinstance(query_strengths, np.ndarray) and query_strengths.shape == (80,),
                "Query membership strength must be scalar per query, shape (80,)")
        require(np.issubdtype(query_strengths.dtype, np.floating), "Query membership output is not numeric floating-point strength")
        finite = np.isfinite(query_strengths)
        require(bool(((query_strengths[finite] >= 0) & (query_strengths[finite] <= 1)).all()), "Finite HDBSCAN strengths outside [0,1]")
        require(np.shape(candidate_strengths) == (len(candidates),), "Candidate membership shape differs")
        require(bool(np.array_equal(frozen_centroids, model.topic_embeddings_)), "Frozen topic centroids changed during inference")
        require(list(model.topics_) == [r["fitted_topic"] for r in fitted], "Frozen fitted assignments changed")
        require(sha256(model_file) == provenance["frozen_model_sha256"], "Frozen model file changed")
        np.save(output / "query_membership_strengths.npy", query_strengths, allow_pickle=False)
        candidate_assignments = [{"document_id": c["document_id"], "topic": int(t), "bertopic": f"Topic {int(t)}"} for c, t in zip(candidates, candidate_topics)]
        write_jsonl(output / "candidate_topic_assignments.jsonl", candidate_assignments)
        routing = []
        for q, t, strength in zip(queries, query_topics, query_strengths):
            strength = float(strength)
            route, reasons = route_query(int(t), strength)
            routing.append({"question_id": q["question_id"], "predicted_topic": int(t),
                "membership_strength": strength if math.isfinite(strength) else None,
                "membership_strength_representation": repr(strength), "membership_strength_is_finite": math.isfinite(strength),
                "route": route, "routing_reasons": "|".join(reasons), "primary_reason": reasons[0]})
        # Persist every decision before reading gold fields for evaluation.
        write_csv(output / "routing_decisions.csv", routing)
        write_jsonl(output / "query_topic_assignments.jsonl", routing)
        routing_hash = sha256(output / "routing_decisions.csv")
        decisions_hash = sha256(output / "query_topic_assignments.jsonl")
        candidate_assignment_hash = sha256(output / "candidate_topic_assignments.jsonl")
        candidate_corpus_hash = sha256(output / "candidate_corpus.jsonl")
        write_json(output / "inference_checkpoint.json", {"routing_decisions_sha256": routing_hash,
            "query_assignments_sha256": decisions_hash, "candidate_assignments_sha256": candidate_assignment_hash,
            "candidate_corpus_sha256": candidate_corpus_hash, "gold_labels_used": False,
            "membership_shape": list(query_strengths.shape), "membership_dtype": str(query_strengths.dtype),
            "nonfinite_membership_strengths": int((~finite).sum())})
        mark("routing_saved_and_hashed_before_gold_scoring")
        timings["frozen_inference_and_routing"] = time.perf_counter() - before
        memory["after_inference"] = memory_snapshot()
        stage = "gold_label_scoring"
        mark("gold_label_scoring_started")
        print("SCORING: routes frozen on disk; gold labels used only for eligibility evaluation", flush=True)
        before = time.perf_counter()
        original, gold_manifest, question_results, gold_results, metrics, duplicates = score(
            development, queries, candidates, occurrences, candidate_assignments, routing)
        require(sha256(development) == source_hash, "Development source changed during evaluation")
        for name, expected in [("routing_decisions.csv", routing_hash), ("query_topic_assignments.jsonl", decisions_hash),
                               ("candidate_topic_assignments.jsonl", candidate_assignment_hash), ("candidate_corpus.jsonl", candidate_corpus_hash)]:
            require(sha256(output / name) == expected, f"Inference checkpoint changed after scoring: {name}")
        write_jsonl(output / "evaluation_questions.jsonl", original)
        write_jsonl(output / "context_occurrence_manifest.jsonl", gold_manifest)
        write_csv(output / "per_question_results.csv", question_results)
        write_csv(output / "per_gold_occurrence_results.csv", gold_results)
        timings["scoring_and_result_tables"] = time.perf_counter() - before
        fallback = [r for r in routing if r["route"] == "UNFILTERED"]
        fallback_groups = {}
        for label, is_outlier in [("topic_minus_1", True), ("non_outlier", False)]:
            group = [r for r in routing if (r["predicted_topic"] == -1) == is_outlier]
            count = sum(r["route"] == "UNFILTERED" for r in group)
            fallback_groups[label] = {"queries": len(group), "fallback_queries": count,
                                      "fallback_percentage": 100 * count / len(group) if group else None}
        c1, c3 = metrics["C1_strict"], metrics["C3_tae_v1"]
        by_q = {method: {r["question_id"]: r for r in question_results if r["method"] == method} for method in METHODS}
        paired = {"micro_retention_change_percentage_points": c3["micro_retention_percentage"] - c1["micro_retention_percentage"],
            "macro_retention_change_percentage_points": c3["macro_question_retention_percentage"] - c1["macro_question_retention_percentage"],
            "mean_candidate_increase": c3["candidate_documents"]["mean"] - c1["candidate_documents"]["mean"],
            "ratio_of_mean_candidate_counts_tae_to_strict": c3["candidate_documents"]["mean"] / c1["candidate_documents"]["mean"] if c1["candidate_documents"]["mean"] else None,
            "additional_gold_occurrences_eligible": c3["gold_retained"] - c1["gold_retained"],
            "questions_with_increased_retention": sum(by_q["C3_tae_v1"][qid]["gold_retained"] > r["gold_retained"] for qid, r in by_q["C1_strict"].items()),
            "questions_with_unchanged_retention": sum(by_q["C3_tae_v1"][qid]["gold_retained"] == r["gold_retained"] for qid, r in by_q["C1_strict"].items()),
            "questions_with_decreased_retention": sum(by_q["C3_tae_v1"][qid]["gold_retained"] < r["gold_retained"] for qid, r in by_q["C1_strict"].items())}
        memory["final"] = memory_snapshot()
        timings["total_execution"] = time.perf_counter() - started
        summary = {"status": "passed", "gate_rule": GATE_RULE, "evaluation_queries": 80,
            "candidate_documents": len(candidates), "context_occurrences": len(occurrences),
            "methods": metrics, "fallback": {"queries": len(fallback), "percentage": 100 * len(fallback) / 80,
                "by_query_topic_group": fallback_groups,
                "primary_reason_counts": dict(Counter(r["primary_reason"] for r in routing)),
                "fallback_trigger_counts": dict(Counter(reason for r in fallback for reason in r["routing_reasons"].split("|")))},
            "paired_comparison": paired, "timings_seconds": timings, "memory": memory,
            "limitations": ["Pre-ranking evidence eligibility only; no ranking, recall@5, answer accuracy, or benchmark superiority",
                "All supplied development contexts are included regardless of support flags, so unfiltered gold availability is favorable by construction",
                "One frozen 500-paragraph topic model and 80 questions from one development dataset; no generalization guarantee",
                "The gate handles outlier/zero/nonfinite membership cases only; positive membership does not imply evidence adequacy",
                "HDBSCAN membership strength is not calibrated evidence confidence; no thresholds were tuned",
                "Gold paragraph occurrences are counted separately even when exact text repeats"]}
        validation.update({"status": "passed", "checks": ["15 original pilot artifact hashes", "exact installed package versions",
            "local pinned encoder manifest", "training/evaluation IDs disjoint", "exploratory/evaluation IDs and query text disjoint",
            "candidate construction includes every context without gold selection", "frozen parameters and centroids unchanged",
            "query membership numeric scalar shape (80,)", "routing saved and hashed before gold scoring",
            "same immutable candidate IDs and topic assignments for C1/C2/C3", "all context exact-text/metadata mappings verified",
            "all gold occurrences present; no missing/ambiguous mappings", "per-question and aggregate denominators"],
            "mapping_unmatched_gold": 0, "mapping_ambiguous_gold": 0, "duplicate_handling": duplicates,
            "evaluation_queries": 80, "gold_occurrences": len(gold_results), "membership_shape": list(query_strengths.shape),
            "nonfinite_membership_strengths": int((~finite).sum()), "inference_checkpoint": json.loads((output / "inference_checkpoint.json").read_text())})
        mark("validation_and_scoring_completed")
        summary["library_warnings"] = []  # Filled by main's warning capture.
        return summary
    except Exception as exc:
        validation.update({"status": "failed", "error_stage": stage,
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "timings_seconds": timings, "execution_events": events, "memory": memory})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "artifacts/bertopic_pilot_500")
    parser.add_argument("--development-file", type=Path, default=ROOT / "processed_data/2wikimultihopqa/dev_subsampled.jsonl")
    parser.add_argument("--encoder-dir", type=Path, default=ROOT / "artifacts/embedding_models/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tae_v1_eval_80")
    args = parser.parse_args()
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        os.environ[name] = "2"
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": str(SEED),
        "NUMBA_CACHE_DIR": "/tmp/at_rag_readiness_numba", "HF_HOME": "/workspace/.cache/huggingface",
        "XDG_CACHE_HOME": "/workspace/.cache"})
    if getattr(sys.flags, "hash_randomization", 0) and os.environ.get("AT_RAG_TAE_V1_SEEDED") != "1":
        os.environ["AT_RAG_TAE_V1_SEEDED"] = "1"
        os.execv(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]])
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        print(f"STOP: refusing to overwrite existing evaluation outputs: {output}")
        return 1
    output.mkdir(parents=True, exist_ok=True)
    validation = {"status": "failed"}
    try:
        helper = runpy.run_path(str(ROOT / "scripts/run_bertopic_pilot.py"), run_name="memory_helper")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            summary = execute(args, output, validation, helper["memory_snapshot"])
        summary["library_warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
        write_json(output / "validation_report.json", validation)
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(report_markdown(summary), encoding="utf-8")
        hashes = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "output_artifact_hashes.json"}
        write_json(output / "output_artifact_hashes.json", hashes)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        print(f"OUTPUT: {output}")
        return 0
    except Exception as exc:
        validation.setdefault("error", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        validation["status"] = "failed"
        write_json(output / "validation_report.json", validation)
        print(f"STOP: {validation.get('error_stage', 'startup')}: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
