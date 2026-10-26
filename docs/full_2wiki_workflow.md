# Full-training 2Wiki workflow on Kaggle

This is a **full-training-split extension using the released AT-RAG model configuration**, not an exact reproduction of the released 500-question protocol. It leaves the original source and sampled runner unchanged. There is no TAE implementation here.

Expected processed `train.jsonl`: **167,454 questions, 1,674,540 context occurrences, 396,978 unique exact paragraph strings, 1,277,562 duplicate occurrences, zero empty paragraphs**. Preparation verifies these counts and records the input SHA-256; it does not assume that the earlier unique-corpus SHA is the training-file SHA. Supply the complete original processed training file, not `documents.jsonl` from the old deduplicating builder.

## Environment and first cell

Use the Python **3.12.3** environment and complete **62-package training freeze** from the baseline Kaggle backup. Retain the merged [MiniLM dependency profile](../compat/minilm/requirements-py312.txt) and [environment instructions](original_chroma_minilm.md). In particular, BERTopic is 0.16.3 and sentence-transformers is 5.7.0. The workflow's heavy stages check every backed-up pin, integration pin, `pip check`, and original-source hashes before loading anything. They never install or replace dependencies. Python 3.13 or the current Codex test environment is not a validated training environment.

After checking out the reviewed feature code into `/kaggle/working/AT-RAG`, execute this **one preparation cell**. Replace the two paths with mounted full training and held-out processed dev/test files. Preparation needs only the standard library; it neither trains nor indexes.

```python
import subprocess, sys
subprocess.run([
    sys.executable, "-B", "/kaggle/working/AT-RAG/scripts/full_2wiki_workflow.py",
    "prepare",
    "--input", "/kaggle/input/YOUR_DATASET/processed_data/2wikimultihopqa/train.jsonl",
    "--heldout", "/kaggle/input/YOUR_DATASET/processed_data/2wikimultihopqa/dev.jsonl",
    "--workdir", "/kaggle/working/at-rag-full-original",
], check=True)
```

The held-out file supplies **question IDs only**, to reject train/evaluation ID overlap. Its answers, contexts, and gold support labels never enter fitting, embeddings, or routing. Training `is_supporting` is stored only to preserve the original metadata schema. Full-file input hashes and the held-out question count are recorded. Shared paragraph text across splits is legitimate; evaluation question IDs must be disjoint. Do not add held-out contexts to this requested training-only corpus. Report evidence availability separately when evaluating retrieval.

## Remaining stages

In the verified Python 3.12.3 environment, set `PY` to that interpreter and `FREEZE` to the backed-up exact 62-package requirements file. Run from the reviewed repository. These are **separate commands**, in order, not an automatic training job:

```bash
export PY=/path/to/python3.12.3
export FREEZE=/kaggle/input/YOUR_BACKUP/training-requirements.txt
export WORK=/kaggle/working/at-rag-full-original
"$PY" -B scripts/full_2wiki_workflow.py train-topics --workdir "$WORK" --training-requirements "$FREEZE"
"$PY" -B scripts/full_2wiki_workflow.py assign-topics --workdir "$WORK" --training-requirements "$FREEZE"
"$PY" -B scripts/full_2wiki_workflow.py cache --workdir "$WORK" --training-requirements "$FREEZE"
"$PY" -B scripts/full_2wiki_workflow.py ingest --workdir "$WORK" --training-requirements "$FREEZE"
"$PY" -B scripts/full_2wiki_workflow.py verify --workdir "$WORK" --training-requirements "$FREEZE"
```

Re-run the **same command and workdir** after interruption. Do not delete checkpoints or change source bytes, environment, code, or settings to force a restart. `status --workdir "$WORK"` reports completed stages without model imports. Heavy commands run the environment preflight automatically. A mismatch fails before model execution.

### Topic-model methodology and memory limit

`train-topics` exactly follows the calls in [vectorDB/train_topic_model.py](../vectorDB/train_topic_model.py): `BERTopic(min_topic_size=50)`, `fit_transform(all_occurrences)`, then `reduce_topics(all_occurrences, nr_topics=20)`, then default pickle serialization. It does not set a random seed, replace UMAP/HDBSCAN/vectorizer defaults, or pass cached retrieval embeddings. BERTopic selects its default English embedding backend; the retrieval adapter's pinned encoder is a separate component. The actual backend description, component parameters, library versions, model hash, and observed topic count are recorded. Requesting 20 does not guarantee exactly 20 regular topics. Original stochastic defaults mean fresh fits are not bitwise reproducible; freeze one saved model for all comparisons.

The original **global fit cannot be made minibatch or incrementally resumed without changing the methodology**. A completed fit is checkpointed before topic reduction. A crash before that checkpoint requires refitting; a crash during reduction reuses the completed fit. `assign-topics` loads the saved final model and makes **one whole-corpus `transform` call**, matching original ingestion. Batched UMAP transforms can change topic labels, so labels are frozen once in an occurrence-indexed `topics.npy` rather than recomputed by ingestion batch.

These two global stages require at least **24 GiB available RAM** to start; that guard is not a proven peak-memory bound. Use the largest available Kaggle RAM session. The raw 1,674,540 × 384 float32 embedding array alone is about 2.40 GiB, and UMAP/HDBSCAN consume additional RAM. The occurrence list shares repeated text strings, but the scientific libraries still process every occurrence. If the global fit exhausts memory or the session duration, this exact configuration is blocked on that Kaggle resource tier; it is not silently replaced with the custom 418-topic model or a different algorithm.

Only this workflow's full-training model plus matching `topics.json`/corpus/environment hashes is automatically loadable. For an existing model without that provenance, methodology is unverified and loading is refused. Back up the whole workdir after the process exits. Fit/model checkpoints use original pickle serialization; load only trusted backups.

### Retrieval embeddings and reusable artifacts

The existing [compat/minilm/utils.py](../compat/minilm/utils.py) supplies **both document and query embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, float32, normalization enabled, maximum sequence length 256, encoder batch size 32, unchanged input strings. This remains the documented substitution for the undocumented original retrieval service. No adapter changes are made.

`cache` embeds only the 396,978 exact unique strings into a bounded SQLite vector cache. Ingestion expands those vectors back to **every occurrence**, including repeated text with different metadata/topic labels. This computational reuse is valid because the frozen retrieval encoder receives text alone. BERTopic still fits and transforms the full occurrence list.

To request reuse instead of encoding, add `--reuse-embeddings /kaggle/input/YOUR_EMBEDDINGS/COMPLETED_DIRECTORY` to `cache`. The completed directory produced by `scripts/embed_full_2wiki_corpus.py` must contain `manifest.json`, `embeddings.npy`, and `document_ids.jsonl`. The workflow checks revision, normalization, dimensions, dtype, preprocessing, max length, CPU/batch-32 configuration, six producer-library versions, file and raw-vector checksums, row order, exact-text SHA mapping, uniqueness, and complete training-text coverage. A different row order is supported through the text IDs.

The old encoder script defaults to **batch 64**, and some old artifacts may use CUDA or different libraries. Those artifacts are **not automatically compatible** with the current frozen adapter; inspect their manifest rather than assume reuse. A rejected cache must be recomputed with this workflow. No access to private Kaggle artifacts is assumed. The **418-topic BERTopic model and its labels are incompatible**: its unique-text fit, precomputed normalized vectors, custom components/seed, and absence of the original reduction differ from the required recipe. Raw full training data and a verified environment freeze remain useful; the old deduplicated paragraph file alone cannot reconstruct occurrence metadata.

### Chroma persistence and restart guarantees

Preparation streams one question at a time into a disk-backed SQLite catalog. Cache construction and ingestion use at most 1,000 text/vector rows per application batch. Chroma's native HNSW index itself grows with the corpus, so total process RAM is not constant; ingestion requires at least **8 GiB available RAM** at startup and checks a conservative remaining-disk estimate. Disk/RAM estimates are safeguards, not measured capacity guarantees. Keep enough writable Kaggle disk for the catalog, vector cache, fit/final models, and Chroma.

Operational differences from `vectorDB/dataset_ingestion.py` are deliberate: SQLite text/vector caching, durable phase/cursor manifests, per-occurrence deterministic IDs in place of random UUIDs, and native Chroma `upsert` with precomputed vectors. The collection is still **`langchain`**, with unmodified persistent Chroma defaults (including the default L2 metric and HNSW settings); original metadata is exactly `idx`, `title`, `is_supporting`, `bertopic="Topic <id>"`. No titles, answers, or gold labels enter encoder inputs. Context order, all duplicates, topic labels, and batches of **1,000** are retained. IDs are SHA-256 of training-file hash plus the zero-based occurrence number, never text IDs.

Each cache/preparation transaction commits its rows and cursor together. Each Chroma batch is read back before its cursor advances. A crash after upsert replays the same IDs. Concurrent writers, unknown stores, changed inputs/configuration, bad counts, and corrupt immutable artifacts are rejected. Final verification checks all 1,674,540 IDs/documents/metadata and records a canonical record checksum; supplied vectors are checked before upsert, but final verification does not re-read every stored vector. Checkpoints fingerprint input, model, label, cache, implementation, and environment artifacts. Back up the **entire quiescent workdir**, not a live Chroma folder, to survive Kaggle session replacement. Checksummed immutable artifacts are re-read on resume; budget I/O time for verification.

For original inference, put the absolute `compat/minilm` directory first on `PYTHONPATH` and pass the verified `WORK/chroma` and `WORK/bertopic_model` paths to the existing `TopicCoTSelfRAG` constructor. Use the same interpreter/freeze and adapter for evaluation. No inference files, prompts, strict topic filter, total retrieval **k=5**, grader, rewriter, reasoning, or LangGraph code are changed. Future baseline/TAE comparisons must share this frozen index/model, held-out IDs, question order, LLM settings, and evaluation code. This workflow does not execute QA or introduce TAE.

## Focused validation

```bash
python -B -m unittest discover -s tests -p 'test_full_2wiki_workflow.py' -v
python -B -m unittest discover -s tests -p 'test_original_chroma_minilm.py' -v
```

Tests use tiny SQLite fixtures and fake encoders/topic models/collections. They exercise interruption/replay, source/configuration/hash rejection, split separation, original topic calls, duplicate metadata and labels, cache remapping, count validation, and the merged adapter's matching document/query interface. They do not train BERTopic, fetch weights, or create a Chroma index. Full-scale capacity and actual Kaggle execution remain to be validated there.
