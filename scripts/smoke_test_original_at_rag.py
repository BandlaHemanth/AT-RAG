"""Smoke-test the released AT-RAG pipeline end to end, without editing it.

Faithful-reproduction setting (protocol section 5): the original 500-question
evaluation subset (test_subsampled.jsonl, whose 5,000 context occurrences form
the store), the supplied original BERTopic model, local MiniLM retrieval
embeddings (compat/minilm), and models/AT_RAG.py unchanged. The only substitution
is the LLM: the released ChatOpenAI(model="gpt-4o") is pointed at a local
OpenAI-compatible server (Ollama) where a free model is registered under the
name "gpt-4o". Run with the separate old-LangChain environment.

Per question it records the answer exactly as the original loop reads it
(pipeline.last_answer), iterations, LLM calls, seconds, errors, and whether the
answer is identical to the previous question's (possible stale state).
"""
import argparse
import contextlib
import importlib.util
import json
import os
import re
import string
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, truths):
    return float(any(normalize_answer(prediction) == normalize_answer(t) for t in truths))


def f1_score(prediction, truths):
    best = 0.0
    for truth in truths:
        p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
        common = sum((Counter(p) & Counter(t)).values())
        if common:
            precision, recall = common / len(p), common / len(t)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def stale_flags(answers, errors):
    """A question's answer is suspect if it errored or repeats the previous answer verbatim."""
    flags = []
    for i, answer in enumerate(answers):
        flags.append(bool(errors[i]) or (i > 0 and answer is not None and answer == answers[i - 1]))
    return flags


def gold_answers(sample):
    answers = []
    for obj in sample.get("answers_objects", []):
        answers.extend(obj.get("spans", []))
    return answers or [""]


def count_llm_calls():
    from langchain_community.chat_models.openai import ChatOpenAI
    counter = {"calls": 0}
    original = ChatOpenAI._generate

    def wrapped(self, *args, **kwargs):
        counter["calls"] += 1
        return original(self, *args, **kwargs)

    ChatOpenAI._generate = wrapped
    return counter


@contextlib.contextmanager
def original_environment(topic_model):
    """Working directory with the relative 'bertopic_model' name the code loads,
    the compat MiniLM 'utils' module, and the original source directories on sys.path."""
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    with tempfile.TemporaryDirectory(prefix="at-rag-smoke-") as working:
        (Path(working) / "bertopic_model").symlink_to(Path(topic_model).resolve())
        sys.modules["utils"] = load_module("utils", ROOT / "compat/minilm/utils.py")
        sys.path[:0] = [str(ROOT / "models"), str(ROOT / "vectorDB"), str(ROOT)]
        os.chdir(working)
        try:
            yield
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path


def run(dataset, topic_model, store, n_questions, server):
    ingest = load_module("ingest_original", ROOT / "scripts/ingest_original_2wiki_minilm.py")
    dataset, topic_model, store = Path(dataset).resolve(), Path(topic_model).resolve(), Path(store).resolve()
    if not store.exists():
        inputs = ingest.inspect_inputs(dataset, topic_model, store)  # hash checks of data, model, source
        started = time.time()
        ingest.create_original_store(dataset, topic_model, store)
        build = {"built": True, "seconds": round(time.time() - started, 1), "verified_inputs": inputs}
    else:
        if ingest.file_sha256(dataset) != ingest.DATASET_SHA256 or ingest.file_sha256(topic_model) != ingest.TOPIC_MODEL_SHA256:
            raise ValueError("dataset or topic model differs from the verified originals")
        build = {"built": False}
    os.environ.update(OPENAI_API_KEY="local-ollama", GEMINI_API="unused", OPENAI_API_BASE=server)
    samples = [json.loads(line) for line in dataset.open(encoding="utf-8")][:n_questions]
    counter = count_llm_calls()
    records = []
    with original_environment(topic_model):
        at_rag = load_module("AT_RAG", ROOT / "models/AT_RAG.py")
        pipeline = at_rag.TopicCoTSelfRAG(max_iter=5, nr_topics=10, max_doc_retrived=5,
                                          dataset_path=str(dataset), vectorDB_path=str(store),
                                          bert_topic_model_path="bertopic_model")
        for sample in samples:
            question = sample["question_text"]
            topic = int(pipeline.trainer.get_topics_with_probabilities([question])[0][0])
            calls_before, started, error = counter["calls"], time.time(), None
            try:
                pipeline.run_pipeline(question=question)
            except Exception as exc:  # recorded, as the original loop continues on errors
                error = f"{type(exc).__name__}: {exc}"[:500]
            answer = pipeline.last_answer if hasattr(pipeline, "last_answer") else "***"
            answer = answer.lstrip() if answer else ""
            truths = gold_answers(sample)
            records.append({"question_id": sample["question_id"], "question": question, "gold": truths,
                            "answer": answer, "question_topic": topic, "iterations": pipeline.iter,
                            "llm_calls": counter["calls"] - calls_before,
                            "seconds": round(time.time() - started, 1), "error": error,
                            "exact_match": exact_match(answer, truths), "f1": round(f1_score(answer, truths), 4)})
    flags = stale_flags([r["answer"] for r in records], [r["error"] for r in records])
    for record, flag in zip(records, flags):
        record["suspect_stale_or_error"] = flag
    n = len(records)
    summary = {"questions": n, "errors": sum(bool(r["error"]) for r in records),
               "suspect_stale_or_error": sum(flags),
               "mean_iterations": sum(r["iterations"] for r in records) / n,
               "mean_llm_calls": sum(r["llm_calls"] for r in records) / n,
               "mean_seconds": sum(r["seconds"] for r in records) / n,
               "exact_match": sum(r["exact_match"] for r in records) / n,
               "f1": sum(r["f1"] for r in records) / n,
               "question_topic_minus_one": sum(r["question_topic"] == -1 for r in records)}
    return {"setting": "faithful reproduction smoke test; LLM substituted via local OpenAI-compatible server",
            "server": server, "store": build, "summary": summary, "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="test_subsampled.jsonl (verified by hash)")
    parser.add_argument("--topic-model", type=Path, required=True, help="the supplied original BERTopic model file")
    parser.add_argument("--store", type=Path, required=True, help="Chroma directory; built once if absent")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--server", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.topic_model, args.store, args.questions, args.server)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
