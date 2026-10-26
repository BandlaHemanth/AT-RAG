# Full-training 2Wiki workflow on Kaggle

This is a **full-training-split extension using the released AT-RAG model configuration**, not an exact reproduction of the released 500-question protocol. It leaves the original source and sampled runner unchanged. There is no TAE implementation here.

Expected processed `train.jsonl`: **167,454 questions, 1,674,540 context occurrences, 396,978 unique exact paragraph strings, 1,277,562 duplicate occurrences, zero empty paragraphs**. Preparation verifies these counts and records the input SHA-256; it does not assume that the earlier unique-corpus SHA is the training-file SHA. Supply the complete original processed training file, not `documents.jsonl` from the old deduplicating builder.

## Separate verified environments

The full workflow no longer calls the sampled ingestor's combined environment checker. **Do not install the combined MiniLM requirements into the BERTopic training environment.** The saved `packaging==26.3` conflicts with `langchain-core==0.3.63` (`packaging>=23.2,<25`), reached through `langchain==0.3.2`. Even core 0.3.83 requires packaging <26. BERTopic 0.16.3 itself has no LangChain/Chroma requirement. Sources: [core 0.3.63 metadata](https://pypi.org/pypi/langchain-core/0.3.63/json), [core 0.3.83 metadata](https://pypi.org/pypi/langchain-core/0.3.83/json), [BERTopic metadata](https://pypi.org/pypi/bertopic/0.16.3/json).

Both profiles require **Python 3.12.3** and the immutable **original 62-package training freeze**. The [training overlay](../compat/full_2wiki/training-py312.txt) preserves packaging 26.3 and all eleven recorded scientific versions: BERTopic 0.16.3, transformers 4.45.1, sentence-transformers 5.7.0, torch 2.14.1, numpy 1.26.4, scikit-learn 1.9.1, scipy 1.17.1, UMAP 0.5.12, HDBSCAN 0.8.44, numba 0.68.0, llvmlite 0.50.0. Recorded Torch local CUDA/build tags take precedence over the public pin. All other retained recorded packages are checked too; this is not an eleven-package-only check.

The [retrieval/inference overlay](../compat/full_2wiki/retrieval-py312.txt) retains those same scientific and model-serialization dependencies and **explicitly replaces packaging 26.3 with 24.2 only in that environment**. It retains LangChain 0.3.2/community 0.3.1 and Chroma 0.5.11; pins core 0.3.63, text-splitters 0.3.8, LangGraph 0.2.28/checkpoint 1.0.12 for the existing interfaces; and carries the merged integration pins. These additional deployment pins do not change the original graph code. Any other conflict with a retained freeze pin stops setup rather than upgrading it. If the input record contains LangChain/LangGraph/Chroma integration pins, their exclusion from the training profile is explicitly listed in the plan/report. No scientific pin is excluded. Training rejects installed retrieval packages, including those leaked from an inherited environment.

The installer creates a dedicated venv without system-site packages, prints exclusions/changes, writes the effective exact requirements, runs a **resolver dry-run before installation**, then verifies every retained pin, `pip check`, and imports. It never uses `--no-deps`, upgrades the notebook's system Python, downloads weights, or constructs a model/index. The input freeze is never edited. Failed installs can retry the same installer-owned venv; unknown or changed-profile destinations are refused. Keep the reported `verified-environment.json`, effective requirements, and pip resolution report with experiment artifacts. A complete installed package inventory is recorded, including additional resolved dependencies; freeze and restore that inventory when moving an environment.

### Next single Kaggle cell: install training only

Fetch/check out the current PR #5 feature code first. Replace `PY312` with the existing **3.12.3** interpreter and `FREEZE` with the original 62-package record. The notebook kernel may remain on another Python version; all model stages must use the venv interpreter. No training is started by this cell.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/AT-RAG
PY312=/path/to/existing/python3.12.3
FREEZE=/kaggle/input/YOUR_BACKUP/training-requirements.txt
"$PY312" -B scripts/full_2wiki_environment.py install \
  --profile training --training-requirements "$FREEZE" \
  --venv /kaggle/working/at-rag-bertopic-py312
```

Published Python/direct-dependency metadata was checked for the 12 training pins and 23 combined retrieval/scientific pins, with no direct pin conflicts. The private complete freeze and target binary wheels have not been installed/tested in Codex. Success is declared **only after Kaggle resolution, version, dependency, and import checks pass**. Preserve recorded Torch/CUDA wheels; if a local build is unavailable at the configured index, supply its original wheel/index, not a replacement build. Any additional freeze conflict remains an explicit blocker. Python 3.13 and the Cloud tooling environment are not validated model environments. Two CUDA venvs plus the index may exceed notebook disk limits: the profiles can run in separate Kaggle sessions with the quiescent workdir/reference restored. Keep compatible accelerator availability for CUDA pickles; no automatic CPU conversion is performed.

## Full-corpus preparation

Preparation needs only the standard library and neither trains nor indexes. Replace paths with the complete processed training file and held-out file; skip this command if the existing preparation is complete.

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

## Model stages, transfer check, then ingestion

Run these **separate commands**, in order, from the reviewed repository. `train-topics` and `assign-topics` automatically select the training-only checker; cache/ingest/verify select the retrieval checker. Selecting the wrong profile is refused. `preflight` checks package metadata, dependency health, and original-source hashes without loading a model. Runtime checks never install or change packages.

```bash
export TRAIN_PY=/kaggle/working/at-rag-bertopic-py312/bin/python
export FREEZE=/kaggle/input/YOUR_BACKUP/training-requirements.txt
export WORK=/kaggle/working/at-rag-full-original
"$TRAIN_PY" -B scripts/full_2wiki_workflow.py preflight --environment-profile training --workdir "$WORK" --training-requirements "$FREEZE"
"$TRAIN_PY" -B scripts/full_2wiki_workflow.py train-topics --workdir "$WORK" --training-requirements "$FREEZE"
"$TRAIN_PY" -B scripts/full_2wiki_workflow.py verify-model --environment-profile training --workdir "$WORK" --training-requirements "$FREEZE"
"$TRAIN_PY" -B scripts/full_2wiki_workflow.py assign-topics --workdir "$WORK" --training-requirements "$FREEZE"

# Later: install a DIFFERENT venv for Chroma and original AT-RAG inference.
"$TRAIN_PY" -B scripts/full_2wiki_environment.py install --profile retrieval --training-requirements "$FREEZE" --venv /kaggle/working/at-rag-retrieval-py312
export RETRIEVAL_PY=/kaggle/working/at-rag-retrieval-py312/bin/python
"$RETRIEVAL_PY" -B scripts/full_2wiki_workflow.py verify-model --environment-profile retrieval --workdir "$WORK" --training-requirements "$FREEZE"
"$RETRIEVAL_PY" -B scripts/full_2wiki_workflow.py cache --workdir "$WORK" --training-requirements "$FREEZE"
"$RETRIEVAL_PY" -B scripts/full_2wiki_workflow.py ingest --workdir "$WORK" --training-requirements "$FREEZE"
"$RETRIEVAL_PY" -B scripts/full_2wiki_workflow.py verify --workdir "$WORK" --training-requirements "$FREEZE"
```

`verify-model` freshly loads the trusted pickle in the producer environment and writes `model_probe_reference.json`. It transforms three training paragraph strings singly, a fixed neutral query singly, and the three paragraphs together. The retrieval environment must match the **exact Python and actual versions of every retained recorded model dependency except the declared packaging replacement**, load the same model bytes, reproduce all reference topic labels exactly, and reproduce available probabilities within `atol=rtol=1e-6`. A successful destination check writes an environment/model/reference-bound `model_compatibility.json`; ingestion refuses a different environment until that check succeeds. Missing producer references cannot be generated by the destination. No held-out questions, answers, or supporting labels enter these checks. Probe transforms run in separate commands and do not replace or precede the original global transform within its process.

Before original QA in the retrieval environment, rerun that environment's `verify-model` command and then launch original `TopicCoTSelfRAG` with the same verified paths/interpreter and the existing MiniLM `PYTHONPATH`. The original QA source is not instrumented: this explicit pre-launch verification is required by the procedure. A finite probe check establishes loading/basic transform agreement, not mathematical equivalence for every query or portability across arbitrary hardware. GPU/CPU changes, numerical libraries, and pickle compatibility can still fail the check; stop and investigate rather than relax tolerances. Keep one verified inference environment for all baseline/future-TAE comparisons.

Re-run the **same command and workdir** after interruption. Do not delete checkpoints to force a restart. `status --workdir "$WORK"` reports completed stages without model imports. This fix explicitly accepts the unchanged preparation implementation from PR #5 commit `b042922` (SHA `3dcb7d5352646e0af5040ae21dd1e9bff0cc3948be95fbad9cea1843b1c46aa8`), retaining its producer SHA and corpus DB checksum. Unknown preparation implementations, changed inputs, and changed counts remain rejected. Existing topic/fit/cache checkpoints under the old combined environment are not silently relabeled as the new profiles.

### Topic-model methodology and memory limit

`train-topics` exactly follows the calls in [vectorDB/train_topic_model.py](../vectorDB/train_topic_model.py): `BERTopic(min_topic_size=50)`, `fit_transform(all_occurrences)`, then `reduce_topics(all_occurrences, nr_topics=20)`, then default pickle serialization. It does not set a random seed, replace UMAP/HDBSCAN/vectorizer defaults, or pass cached retrieval embeddings. BERTopic selects its default English embedding backend; the retrieval adapter's pinned encoder is a separate component. The actual backend description, component parameters, library versions, model hash, and observed topic count are recorded. Requesting 20 does not guarantee exactly 20 regular topics. Original stochastic defaults mean fresh fits are not bitwise reproducible; freeze one saved model for all comparisons.

The original **global fit cannot be made minibatch or incrementally resumed without changing the methodology**. A completed fit is checkpointed before topic reduction. A crash before that checkpoint requires refitting; a crash during reduction reuses the completed fit. `assign-topics` loads the saved final model and makes **one whole-corpus `transform` call**, matching original ingestion. Batched UMAP transforms can change topic labels, so labels are frozen once in an occurrence-indexed `topics.npy` rather than recomputed by ingestion batch.

These two global stages require at least **24 GiB available RAM** to start; that guard is not a proven peak-memory bound. Use the largest available Kaggle RAM session. The raw 1,674,540 × 384 float32 embedding array alone is about 2.40 GiB, and UMAP/HDBSCAN consume additional RAM. The occurrence list shares repeated text strings, but the scientific libraries still process every occurrence. If the global fit exhausts memory or the session duration, this exact configuration is blocked on that Kaggle resource tier; it is not silently replaced with the custom 418-topic model or a different algorithm.

Only this workflow's full-training model plus matching `topics.json`/corpus hashes and verified environment provenance is loadable. Cross-environment access additionally requires the explicit verification above. An existing model without that provenance is refused. Back up the whole workdir after the process exits. Fit/model checkpoints use original pickle serialization; load only trusted backups.

### Retrieval embeddings and reusable artifacts

The existing [compat/minilm/utils.py](../compat/minilm/utils.py) supplies **both document and query embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, float32, normalization enabled, maximum sequence length 256, encoder batch size 32, unchanged input strings. This remains the documented substitution for the undocumented original retrieval service. No adapter changes are made.

`cache` embeds only the 396,978 exact unique strings into a bounded SQLite vector cache. Ingestion expands those vectors back to **every occurrence**, including repeated text with different metadata/topic labels. This computational reuse is valid because the frozen retrieval encoder receives text alone. BERTopic still fits and transforms the full occurrence list.

To request reuse instead of encoding, add `--reuse-embeddings /kaggle/input/YOUR_EMBEDDINGS/COMPLETED_DIRECTORY` to `cache`. The completed directory produced by `scripts/embed_full_2wiki_corpus.py` must contain `manifest.json`, `embeddings.npy`, and `document_ids.jsonl`. The workflow checks revision, normalization, dimensions, dtype, preprocessing, max length, CPU/batch-32 configuration, six producer-library versions, file and raw-vector checksums, row order, exact-text SHA mapping, uniqueness, and complete training-text coverage. A different row order is supported through the text IDs.

The old encoder script defaults to **batch 64**, and some old artifacts may use CUDA or different libraries. Those artifacts are **not automatically compatible** with the current frozen adapter; inspect their manifest rather than assume reuse. A rejected cache must be recomputed with this workflow. No access to private Kaggle artifacts is assumed. The **418-topic BERTopic model and its labels are incompatible**: its unique-text fit, precomputed normalized vectors, custom components/seed, and absence of the original reduction differ from the required recipe. Raw full training data and a verified environment freeze remain useful; the old deduplicated paragraph file alone cannot reconstruct occurrence metadata.

### Chroma persistence and restart guarantees

Preparation streams one question at a time into a disk-backed SQLite catalog. Cache construction and ingestion use at most 1,000 text/vector rows per application batch. Chroma's native HNSW index itself grows with the corpus, so total process RAM is not constant; ingestion requires at least **8 GiB available RAM** at startup and checks a conservative remaining-disk estimate. Disk/RAM estimates are safeguards, not measured capacity guarantees. Keep enough writable Kaggle disk for the catalog, vector cache, fit/final models, and Chroma.

Operational differences from `vectorDB/dataset_ingestion.py` are deliberate: SQLite text/vector caching, durable phase/cursor manifests, per-occurrence deterministic IDs in place of random UUIDs, and native Chroma `upsert` with precomputed vectors. The collection is still **`langchain`**, with unmodified persistent Chroma defaults (including the default L2 metric and HNSW settings); original metadata is exactly `idx`, `title`, `is_supporting`, `bertopic="Topic <id>"`. No titles, answers, or gold labels enter encoder inputs. Context order, all duplicates, topic labels, and batches of **1,000** are retained. IDs are SHA-256 of training-file hash plus the zero-based occurrence number, never text IDs.

Each cache/preparation transaction commits its rows and cursor together. Each Chroma batch is read back before its cursor advances. A crash after upsert replays the same IDs. Concurrent writers, unknown stores, changed inputs/configuration, bad counts, and corrupt immutable artifacts are rejected. Final verification checks all 1,674,540 IDs/documents/metadata and records a canonical record checksum; supplied vectors are checked before upsert, but final verification does not re-read every stored vector. Checkpoints fingerprint input, model, label, cache, implementation, and environment artifacts. Back up the **entire quiescent workdir**, not a live Chroma folder, to survive Kaggle session replacement. Checksummed immutable artifacts are re-read on resume; budget I/O time for verification.

For original inference, put the absolute `compat/minilm` directory first on `PYTHONPATH` and pass the verified `WORK/chroma` and `WORK/bertopic_model` paths to the existing `TopicCoTSelfRAG` constructor. Use the verified **retrieval interpreter**, freeze/report, and adapter for evaluation. No inference files, prompts, strict topic filter, total retrieval **k=5**, grader, rewriter, reasoning, or LangGraph code are changed. Future baseline/TAE comparisons must share this frozen index/model, held-out IDs, question order, LLM settings, and evaluation code. This workflow does not execute QA or introduce TAE.

## Focused validation

```bash
python -B -m unittest discover -s tests -p 'test_full_2wiki_workflow.py' -v
python -B -m unittest discover -s tests -p 'test_full_2wiki_environment.py' -v
python -B -m unittest discover -s tests -p 'test_original_chroma_minilm.py' -v
```

Tests use tiny SQLite fixtures, fake models/collections, and synthetic 62-package metadata. They cover profile isolation, preservation of scientific/local-build pins, explicit packaging replacement, failed resolver ordering, cross-environment load/transform gating, retained restart guards, duplicate handling, and original retrieval interfaces. They do not install the private environment, train BERTopic, fetch weights, or create a Chroma index. Full-scale capacity and actual Kaggle execution remain to be validated there.
