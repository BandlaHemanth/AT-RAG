"""Controlled end-to-end comparison of three AT-RAG systems (protocol v1.3, sections 5 and 13).

  original  strict single-topic filter (the released mechanism; H2 label of the current question,
            including -1)
  nofilter  AT-RAG without topic filtering
  tae_r     AT-RAG with the frozen TAE-R router (saved predictor, protocol section 14)

Everything except retrieval routing is identical across systems and follows the released
`models/AT_RAG.py` graph exactly: retrieve -> generate_cot -> generate -> grade_generation
(hallucination grader, called before the iteration cap is checked; then answer grader) ->
END / generate again ("not supported") / transform_query -> retrieve; at most 5 iterations,
k = 5. Prompt templates are read verbatim from the released source file; the format
instructions are rebuilt by LangChain 0.3's StructuredOutputParser rule and outputs are parsed
with LangChain's JSON-markdown logic (vendored below). Controlled settings (section 13): the two
fixed grader clarifications, temperature 0, seed 42, per-question state, failures recorded (never
inherited), every LLM call cached on disk.

Retrieval for all systems: exact cosine search with the frozen MiniLM embeddings over the chosen
corpus; question embeddings and H2 labels are the frozen precomputed ones for the original
questions and computed with the same encoder / UMAP / HDBSCAN models for rewritten questions.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SYSTEMS = ("original", "nofilter", "tae_r")
MAX_ITER, K, TEMPERATURE, SEED = 5, 5, 0, 42
MODEL = "qwen2.5:7b-instruct-q8_0"
MODEL_DIGEST = "2d9500c948416525"  # prefix of the Ollama digest fixed in protocol section 16
CLARIFICATIONS = {  # protocol v1.3, section 13 (identical to scripts/smoke_test_original_at_rag.py)
    "hallucination_grader": "Give 'yes' if the answer is supported by the facts, and 'no' if it is not.",
    "answer_grader": "Give 'yes' if the answer resolves the question, and 'no' if it does not.",
}
ANCHOR = "Please respond in valid JSON format"
RELEASED_METHODS = {"hallucination_grader": "_create_hallucination_grader",
                    "answer_grader": "_create_answer_grader",
                    "question_rewriter": "_create_question_rewriter",
                    "rag_chain": "_create_rag_chain", "cot": "get_cot_chain"}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- prompts (verbatim from source)

def released_prompts(source_path=ROOT / "models" / "AT_RAG.py"):
    """Template strings and response schemas of the released chains, read with `ast` (exact text)."""
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    out = {}
    for key, method in RELEASED_METHODS.items():
        node = methods[method]
        template, schemas = None, []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "PromptTemplate":
                for kw in sub.keywords:
                    if kw.arg == "template" and isinstance(kw.value, ast.Constant):
                        template = kw.value.value
            if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "ResponseSchema":
                kws = {kw.arg: kw.value.value for kw in sub.keywords if isinstance(kw.value, ast.Constant)}
                schemas.append((kws["name"], kws.get("type", "string"), kws["description"]))
            if (key == "cot" and isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant)
                    and any(getattr(t, "id", None) == "prompt" for t in sub.targets)):
                template = sub.value.value
        if template is None:
            raise ValueError(f"template not found in {method}")
        out[key] = {"template": template, "schemas": schemas}
    return out


def format_instructions(schemas):
    """LangChain 0.3 StructuredOutputParser.get_format_instructions()."""
    lines = "\n".join(f'\t"{name}": {kind}  // {description}' for name, kind, description in schemas)
    return ('The output should be a markdown code snippet formatted in the following schema, including the '
            'leading and trailing "```json" and "```":\n\n```json\n{\n' + lines + "\n}\n```")


def controlled_prompts(source_path=ROOT / "models" / "AT_RAG.py"):
    prompts = released_prompts(source_path)
    for name, sentence in CLARIFICATIONS.items():
        template = prompts[name]["template"]
        if template.count(ANCHOR) != 1:
            raise ValueError("released template does not contain the JSON instruction exactly once")
        prompts[name]["template"] = template.replace(ANCHOR, sentence + " " + ANCHOR, 1)
    return prompts


def render(prompt, **values):
    if prompt["schemas"]:
        values["format_instructions"] = format_instructions(prompt["schemas"])
    return prompt["template"].format(**values)


# ------------------------------------- output parsing (vendored from langchain-core 0.3.63, MIT)

_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)
_json_strip_chars = " \n\r\t`"


def _replace_new_line(match):
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)
    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string):
    return re.sub(r'("action_input"\:\s*")(.*?)(")', _replace_new_line, multiline_string, flags=re.DOTALL)


def parse_partial_json(s, *, strict=False):
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass
    new_chars, stack, is_inside_string, escaped = [], [], False, False
    for char in s:
        new_char = char
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                new_char = "\\n"
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        elif char == '"':
            is_inside_string = True
            escaped = False
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                return None
        new_chars.append(new_char)
    if is_inside_string:
        if escaped:
            new_chars.pop()
        new_chars.append('"')
    stack.reverse()
    while new_chars:
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
            new_chars.pop()
    return json.loads(s, strict=strict)


def _parse_json(json_str):
    return parse_partial_json(_custom_parser(json_str.strip(_json_strip_chars)))


def parse_json_markdown(json_string):
    try:
        return _parse_json(json_string)
    except json.JSONDecodeError:
        match = _json_markdown_re.search(json_string)
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str)


class OutputParserError(ValueError):
    pass


def parse_structured(text, keys):
    """StructuredOutputParser.parse: JSON from a markdown snippet, with the expected keys."""
    try:
        obj = parse_json_markdown(text)
    except json.JSONDecodeError as error:
        raise OutputParserError(f"Got invalid JSON object. Error: {error}") from error
    for key in keys:
        if key not in obj:  # a None result raises TypeError here, as in LangChain
            raise OutputParserError(f"Got invalid return object. Expected key `{key}`")
    return obj


# ---------------------------------------------------------------------------------- LLM client

class LLMClient:
    """OpenAI-compatible chat completions with an on-disk cache (one user message, as ChatOpenAI)."""

    def __init__(self, server, model, cache_path, temperature=TEMPERATURE, seed=SEED, digest=None, transport=None):
        self.server, self.model, self.temperature, self.seed = server.rstrip("/"), model, temperature, seed
        self.digest = digest
        self.transport = transport or self._post
        self.cache_path = Path(cache_path)
        self.lock = threading.Lock()
        self.cache = {}
        if self.cache_path.is_file():
            with self.cache_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry["content"]

    def _post(self, payload):
        request = urllib.request.Request(self.server + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    return json.load(response)["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    def __call__(self, prompt):
        key = hashlib.sha256(json.dumps([self.model, self.digest, prompt, self.temperature, self.seed]).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key], True
        payload = {"model": self.model, "temperature": self.temperature, "seed": self.seed,
                   "messages": [{"role": "user", "content": prompt}]}
        content = self.transport(payload)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = content
                with self.cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"key": key, "content": content}) + "\n")
        return content, False


def ollama_digest(server, model):
    base = server.rstrip("/").rsplit("/v1", 1)[0]
    with urllib.request.urlopen(base + "/api/tags", timeout=30) as response:
        for entry in json.load(response)["models"]:
            if entry["name"] == model:
                return entry["digest"]
    raise ValueError(f"model {model} not served")


# ------------------------------------------------------------------------------ answer metrics

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


def gold_answers(record):
    answers = []
    for obj in record.get("answers_objects", []):
        answers.extend(obj.get("spans", []))
    return answers or [""]


# --------------------------------------------------------------------------------- the graph

def run_question(system, question, engine, llm, prompts):
    """One question through the released graph; returns the record (never raises)."""
    calls = cached = 0

    def ask(prompt):
        nonlocal calls, cached
        content, hit = llm(prompt)
        calls += 1
        cached += hit
        return content

    started, retrievals, answer, iterations, error = time.time(), [], None, 0, None
    state = {"question": question, "thoughts": ""}
    try:
        while True:
            doc_idx, info = engine.retrieve(system, state["question"])
            retrievals.append(info)
            documents = "\n\n".join(engine.texts[i] for i in doc_idx)
            cot = parse_structured(ask(render(prompts["cot"], question=state["question"], context=documents)), ["thoughts"])
            state["thoughts"] = cot["thoughts"] + state["thoughts"]
            while True:
                generation = parse_structured(ask(render(prompts["rag_chain"], context=documents,
                                                         question=f"{state['thoughts']}--{state['question']}")), ["answer"])
                answer = generation["answer"]
                iterations += 1
                score = parse_structured(ask(render(prompts["hallucination_grader"], documents=documents,
                                                    generation=answer)), ["score"])
                if iterations >= MAX_ITER:
                    decision = "useful"
                elif score["score"] == "yes":
                    useful = parse_structured(ask(render(prompts["answer_grader"], question=state["question"],
                                                         generation=answer)), ["score"])
                    decision = "useful" if useful["score"] == "yes" else "not useful"
                else:
                    decision = "not supported"
                if decision != "not supported":
                    break
            if decision == "useful":
                break
            state["question"] = ask(render(prompts["question_rewriter"], question=state["question"], context=documents))
            iterations += 1
    except Exception as exc:  # recorded as a failure; never inherits another question's answer
        error = f"{type(exc).__name__}: {exc}"[:400]
        answer = None
    return {"system": system, "answer": None if answer is None else str(answer), "error": error,
            "iterations": iterations, "llm_calls": calls, "cached_calls": cached,
            "retrievals": retrievals, "seconds": round(time.time() - started, 2)}


# ------------------------------------------------------------------------------ retrieval engine

class Engine:
    """Exact MiniLM search over one corpus with the three routing mechanisms."""

    def __init__(self, texts, doc_ids, doc_vectors, doc_labels, centroid_ids, centroids, router_model,
                 known_questions, embed_fn, topic_fn, router_module):
        self.texts, self.doc_ids, self.docs, self.labels = texts, doc_ids, doc_vectors, doc_labels
        self.centroid_ids, self.centroids, self.model = centroid_ids, centroids, router_model
        self.known = dict(known_questions)  # text -> (vector, label, strength)
        self.embed_fn, self.topic_fn, self.router = embed_fn, topic_fn, router_module
        self.lock = threading.Lock()
        topics, counts = np.unique(doc_labels, return_counts=True)
        self.size = dict(zip(topics.tolist(), counts.tolist()))

    def encode(self, text):
        with self.lock:
            if text not in self.known:
                vector = self.embed_fn(text)
                label, strength = self.topic_fn(vector)
                self.known[text] = (vector, int(label), float(strength))
            return self.known[text]

    def retrieve(self, system, text):
        q, label, strength = self.encode(text)
        scores = self.docs @ q
        order = np.argsort(-scores, kind="stable")
        info = {"system": system, "question_label": label}
        if system == "nofilter":
            allowed, cost = None, len(self.labels)
        elif system == "original":
            allowed = [label]
            cost = self.size.get(label, 0)
        elif system == "tae_r":
            features, p1, p2 = self.router.compute_features(q, label, strength, scores, order, self.labels,
                                                            self.centroid_ids, self.centroids, True)
            route, allowed = self.router.route_tae_r(features, p1, p2, self.model)
            info["route"] = route
            cost = len(self.labels) + len(self.centroid_ids)  # probe scan reused for the ranking
        else:
            raise ValueError(system)
        if allowed is None:
            top = order[:K]
        else:
            top = order[np.isin(self.labels[order], sorted(set(allowed)))][:K]
        info.update(topics=None if allowed is None else [int(t) for t in allowed],
                    similarity_computations=int(cost), doc_ids=[self.doc_ids[i] for i in top])
        return top, info


def build_engine(corpora_dir, embeddings_dir, topics_dir, split, router_path, *, allow_final=False,
                 device="cuda", frozen=None):
    fit = _load("fit_tae_r_topic_models", "fit_tae_r_topic_models.py")
    router = _load("tae_r_router", "tae_r_router.py")
    import joblib
    inputs = fit.load_inputs(corpora_dir, embeddings_dir, frozen or fit.FROZEN)
    if split == "validation":
        corpus_file, stage_name, q_file = "retrieval_validation.jsonl", "H2/assign_validation", "questions_validation.jsonl"
    elif split == "final_eval":
        if not allow_final:
            raise ValueError("final_eval requires --allow-final (protocol: run once, after freezing)")
        corpus_file, stage_name, q_file = "retrieval_final.jsonl", "H2/assign_final", "questions_final_eval.jsonl"
    else:
        raise ValueError(split)
    records = [json.loads(line) for line in (corpora_dir / corpus_file).open(encoding="utf-8")]
    doc_ids = [r["doc_id"] for r in records]
    texts = [r["text"] for r in records]
    stage = fit.Stage(topics_dir, stage_name)
    if not stage.complete():
        raise ValueError(f"incomplete stage {stage.dir}")
    labels_file = "validation_corpus_labels.npy" if split == "validation" else "final_corpus_labels.npy"
    doc_labels = stage.load(labels_file)
    if len(doc_labels) != len(doc_ids):
        raise ValueError("corpus labels do not match the corpus")
    fit_stage = fit.Stage(topics_dir, "H2/fit")
    centroid_ids, centroids = fit.topic_centroids(fit.rows_for(inputs, inputs["fit_ids"]), fit_stage.load("fit_labels.npy") + 1)
    centroid_ids = centroid_ids - 1
    # Frozen precomputed question vectors and labels for the original question texts.
    if split == "validation":
        q_stage = fit.Stage(topics_dir, "questions")
        ids = json.loads((q_stage.dir / "question_ids.json").read_text(encoding="utf-8"))
        vectors, q_labels, q_strengths = (q_stage.load("question_embeddings.npy"), stage.load("question_labels.npy"),
                                          stage.load("question_strengths.npy"))
        keep = {qid: i for i, (s, qid) in enumerate(ids) if s == "validation"}
    else:
        ids = json.loads((stage.dir / "question_ids.json").read_text(encoding="utf-8"))
        vectors, q_labels, q_strengths = (stage.load("question_embeddings.npy"), stage.load("question_labels.npy"),
                                          stage.load("question_strengths.npy"))
        keep = {qid: i for i, qid in enumerate(ids)}
    questions = {}
    for line in (corpora_dir / q_file).open(encoding="utf-8"):
        record = json.loads(line)
        questions[record["question_id"]] = record
    known = {questions[qid]["question_text"]: (vectors[i], int(q_labels[i]), float(q_strengths[i]))
             for qid, i in keep.items() if qid in questions}
    from sentence_transformers import SentenceTransformer
    enc = fit.CONFIG["encoder"]
    encoder = SentenceTransformer(enc["model_id"], revision=enc["revision"], device=device)
    encoder.max_seq_length = enc["max_seq_length"]
    umap_model = fit.Stage(topics_dir, "umap").load("umap_model.joblib")
    hdb = fit_stage.load("hdbscan_model.joblib")
    import hdbscan

    def embed_fn(text):
        return encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    def topic_fn(vector):
        point = umap_model.transform(vector[None, :]).astype(np.float32)
        labels, strengths = hdbscan.approximate_predict(hdb, point)
        return labels[0], strengths[0]

    router_model = joblib.load(router_path)["variants"]["tae_r"]
    engine = Engine(texts, doc_ids, fit.rows_for(inputs, doc_ids), doc_labels, centroid_ids, centroids,
                    router_model, known, embed_fn, topic_fn, router)
    return engine, questions


def summarise(rows):
    out = {}
    for system in SYSTEMS:
        mine = [r for r in rows if r["system"] == system]
        if not mine:
            continue
        n = len(mine)
        out[system] = {"questions": n, "exact_match": sum(r["exact_match"] for r in mine) / n,
                       "f1": sum(r["f1"] for r in mine) / n, "errors": sum(bool(r["error"]) for r in mine),
                       "first_retrieval_allgold@5": sum(r["first_allgold@5"] for r in mine) / n,
                       "mean_iterations": sum(r["iterations"] for r in mine) / n,
                       "mean_llm_calls": sum(r["llm_calls"] for r in mine) / n,
                       "mean_retrievals": sum(len(r["retrievals"]) for r in mine) / n,
                       "mean_similarity_computations": sum(sum(x["similarity_computations"] for x in r["retrievals"]) for r in mine) / n,
                       "mean_seconds": sum(r["seconds"] for r in mine) / n}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--topics-dir", type=Path, required=True, help="tae-r-topics (needs H2/assign_final for final_eval)")
    parser.add_argument("--router-model", type=Path, required=True, help="tae_r_router_H2_v1.joblib")
    parser.add_argument("--split", choices=("validation", "final_eval"), required=True)
    parser.add_argument("--question-ids", type=Path, help="file of question ids (e.g. e2e_300_ids.txt)")
    parser.add_argument("--limit", type=int, help="first N questions (validation dry run)")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument("--allow-final", action="store_true")
    parser.add_argument("--server", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = ollama_digest(args.server, MODEL)
    if not digest.startswith(MODEL_DIGEST):
        raise SystemExit(f"served {MODEL} digest {digest} differs from the protocol")
    engine, questions = build_engine(args.corpora_dir, args.embeddings_dir, args.topics_dir, args.split,
                                     args.router_model, allow_final=args.allow_final)
    if args.question_ids:
        order = args.question_ids.read_text(encoding="utf-8").split()
    else:
        order = list(questions)
    order = order[:args.limit] if args.limit else order
    prompts = controlled_prompts()
    llm = LLMClient(args.server, MODEL, args.output_dir / "llm_cache.jsonl", digest=digest)
    out_path = args.output_dir / "records.jsonl"
    done = set()
    if out_path.is_file():
        done = {(r["system"], r["question_id"]) for r in map(json.loads, out_path.open(encoding="utf-8"))}
    tasks = [(s, qid) for qid in order for s in args.systems if (s, qid) not in done]
    write_lock = threading.Lock()
    print(f"{len(tasks)} runs to do ({len(done)} already done)", flush=True)

    def work(task):
        system, qid = task
        record = questions[qid]
        result = run_question(system, record["question_text"], engine, llm, prompts)
        truths = gold_answers(record)
        gold = set(record["gold_doc_ids"])
        first = set(result["retrievals"][0]["doc_ids"]) if result["retrievals"] else set()
        result.update(question_id=qid, gold_answers=truths,
                      exact_match=exact_match(result["answer"], truths) if result["answer"] is not None else 0.0,
                      f1=round(f1_score(result["answer"], truths), 4) if result["answer"] is not None else 0.0,
                      first_allgold=int(gold <= first), first_hits=len(gold & first), n_gold=len(gold))
        result["first_allgold@5"] = result.pop("first_allgold")
        with write_lock:
            with out_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        return system, qid

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(work, tasks), 1):
            if i % 10 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} runs, {time.time() - started:.0f} s", flush=True)
    rows = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    summary = {"protocol": "tae-r-protocol-v1.3", "split": args.split, "model": MODEL, "digest": digest,
               "temperature": TEMPERATURE, "seed": SEED, "max_iter": MAX_ITER, "k": K,
               "questions": len(order), "systems": summarise(rows),
               "records_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["systems"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
