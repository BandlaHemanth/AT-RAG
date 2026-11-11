# Smoke test of the released AT-RAG code (faithful-reproduction setting), run 1

Date: 11 November 2026. Driver: `scripts/smoke_test_original_at_rag.py` (commit `4d5c0ae`).
Questions: the first 5 of `test_subsampled.jsonl` (exposed questions; not part of final_eval).

## Setting

- Code: `models/AT_RAG.py`, `vectorDB/dataset_ingestion.py`, `vectorDB/train_topic_model.py` unedited
  (source hashes checked by `scripts/ingest_original_2wiki_minilm.py`).
- Store: original `create_vectordb()` over the 5,000 context occurrences of `test_subsampled.jsonl`
  (SHA-256 `4948757b...`), original supplied BERTopic model (SHA-256 `4a626723...`; 12 regular topics + -1),
  local MiniLM retrieval embeddings via `compat/minilm/utils.py`. Build time 98.2 s.
- Runtime: Python 3.12.3 (the model's training Python), torch 2.14.1+cu130, numpy 1.26.4, numba 0.68.0,
  llvmlite 0.50.0, cloudpickle 3.1.2, scikit-learn 1.9.1, scipy 1.17.1, umap-learn 0.5.12, hdbscan 0.8.44,
  bertopic 0.16.3, transformers 4.45.1, sentence-transformers 5.7.0, tokenizers 0.20.3, huggingface-hub 0.36.2
  (scientific pins from the backed-up 62-package training freeze); langchain 0.3.2, langchain-community 0.3.1,
  chromadb 0.5.11, langgraph 0.2.34, openai 1.51.2, httpx 0.27.2, python-decouple 3.8, posthog 3.7.0.
  The freeze's `packaging==26.3` was not enforced (incompatible with langchain-core 0.3; resolved to 24.2).
- LLM substitution (the only change): released `ChatOpenAI(model="gpt-4o")` pointed at a local Ollama
  server (`OPENAI_API_BASE`) where `qwen2.5:7b-instruct-q8_0` (digest `2d9500c948416525`) is registered
  as `gpt-4o`. Temperature left at the released default (0.7).

## Reproducibility findings

1. The supplied BERTopic artifact does not load on Python 3.13 (numba-pickled UMAP functions); it needs
   the original Python 3.12 runtime. It also contains CUDA tensors, so it needs a CUDA build of torch.
2. In the released code `iter` is reset per question but `last_answer` is not; the evaluation loop reads
   `last_answer` even after an exception, so a failed question would inherit the previous answer.
   (No exception occurred in this run.)
3. `grade_documents` exists but is not wired into the graph.

## Results (5 questions)

| Measure | Value |
|---|---|
| Errors / suspect stale answers | 0 / 0 |
| Exact match / F1 | 0.20 / 0.20 |
| Iterations | 5 on every question (the cap) |
| LLM calls per question | 11-13 (mean 11.6) |
| Seconds per question | 11.2-27.6 (mean 17.7) |
| Question topic -1 | 1 of 5 (others: topic 0) |

Every question reached the iteration cap, including the one answered correctly; the log shows the same
answer regenerated repeatedly, i.e. the hallucination/answer graders did not return "yes".
Hypothesis, not yet verified: the released code compares `score == "yes"` exactly, and the substituted
model may answer "Yes". Next: a diagnostic run recording the graders' raw outputs.

## Diagnostic run (same 5 questions, `--trace-graders`, commit `4d45a9b`)

Grader outputs were recorded without changing behaviour. The case hypothesis is rejected: the graders
answer in lowercase. The hallucination grader returned "no" in 21 of 22 calls, including for the
correctly answered question (answer "The Devil's Miner"), so `grade_generation` returned
"not supported" and the graph regenerated with the same documents until the iteration cap.
The only "yes" led to the answer grader, which returned "no" (question rewritten, then capped).

Cause (from the released prompts): the hallucination and answer grader prompts ask for
"a score 'yes' or 'no'" but never state which value means grounded / useful. A strong model may infer
it; the 7B substitute mostly answers "no". With this model the released loop therefore behaves as
"regenerate up to the cap and keep the last answer", at about 12 LLM calls per question.
Mean over the diagnostic run: 5.2 iterations, 11.6 LLM calls, 13.6 s per question, EM 0.20.
