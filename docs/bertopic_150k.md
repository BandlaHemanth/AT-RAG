# BERTopic reference training on all 150,000 occurrences

Use `thesis/tae-v2-150k`. This implementation trains only the frozen training
corpus. It does not read development/test data or labels, evaluate retrieval,
build a vector database, or implement TAE V2. Existing training/embedding helpers,
models, corpus files and results remain unchanged.

## Frozen input and reference settings

Input: `artifacts/tae_v2_150k_corpus/run_001/corpus.jsonl`. Admission checks require
exactly **150,000 occurrences**, **66,208 unique texts**, **83,792 duplicate-excess
occurrences**, and SHA-256
`d9edc9d9bc3ff4bbb98de8690c5b79c71aa6e2878f035df55cd01891ea322c33`.
Every occurrence must have a unique, source-qualified occurrence ID and valid
training provenance. `idx` remains local metadata. Only unchanged `paragraph_text`
strings enter embedding and fitting. All 150,000 rows, including repeats, enter
BERTopic in corpus order. There is no deduplication, subsampling, text shortening,
topic reduction or outlier reassignment, and no automatic smaller-corpus fallback.

| Component | Explicit reference configuration |
| --- | --- |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; normalized 384D float32 |
| BERTopic | `min_topic_size=50`, `nr_topics=None`, `calculate_probabilities=False`, `low_memory=True`, `top_n_words=10`; precomputed vectors |
| UMAP | 5 components, 15 neighbors, cosine, min_dist=0, 200 epochs, spectral init, random_state=42, transform_seed=42, low_memory=True, n_jobs=1, force_approximation_algorithm=True |
| HDBSCAN | min_cluster_size=50, min_samples=None (thus 50, matching the original default), Euclidean, boruvka_kdtree, leaf_size=40, alpha=1, approx_min_span_tree=True, gen_min_span_tree=False, core_dist_n_jobs=1 |
| Cluster selection | EOM, cluster_selection_epsilon=0, allow_single_cluster=False, prediction_data=True |
| CountVectorizer | English stop words, unigrams, min_df=1, max_features=20000, int32 sparse counts |
| c-TF-IDF | bm25_weighting=False, reduce_frequent_words=False |
| Reproducibility | Seed 42; interpreter hash seed 42; one Numba/BLAS/OpenMP thread; deterministic Torch algorithms and TF32 disabled |

This retains the original `BERTopic(min_topic_size=50)` reference size and original
HDBSCAN min-samples convention, while making estimator seeds/algorithms explicit.
English stop words and the vocabulary cap follow the existing thesis memory
conventions. `min_df=1` avoids an implicit requirement for multiple topic groups.
These are frozen reference settings, not tuned parameters; fitted library defaults
are also exported. Successful all-outlier fits remain reported as such.

MiniLM's native window is 256 tokens including special tokens. Ordinary
`encode(texts)` would truncate long paragraphs. This script tokenizes without
truncation, encodes bounded batches of nonoverlapping content-token windows,
normalizes each window vector, averages with content-token-count weights, and
normalizes the resulting paragraph vector. A short paragraph uses one normalized
window. Every content token is covered, including long-paragraph tails; the full
original string also enters c-TF-IDF. This documented aggregation is necessary to
cover long text and differs from the earlier single-window embedding policy.
Future query/document embedding must use this same policy. Window/token counts
and numbers of multiwindow occurrences are saved.

## Memory, dependencies and commands

Linux/Kaggle, Python **3.12** is the tested reference. In a separate environment,
install `requirements/bertopic_150k.txt`; do not upgrade an older saved-model
environment. The file pins the installed reference NumPy, BERTopic, MiniLM/Torch,
UMAP, HDBSCAN and scientific dependencies. Save the complete resolved environment
for subsequent model loading. A different Python/Kaggle image must resolve
compatible wheels and pass the synthetic integration test first; cross-version
pickle compatibility and bitwise cross-hardware clustering are not promised.

```bash
python -m pip install -r /kaggle/working/AT-RAG/requirements/bertopic_150k.txt
PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATRAG_RUN_TINY_BERTOPIC_TEST=1 \
python -B -m unittest discover -s /kaggle/working/AT-RAG/tests -p 'test_train_bertopic_150k.py' -v
```

Full training command (shell or Kaggle `%%bash` cell):

```bash
PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B /kaggle/working/AT-RAG/scripts/train_bertopic_150k.py \
  --corpus /kaggle/working/AT-RAG/artifacts/tae_v2_150k_corpus/run_001/corpus.jsonl \
  --output-dir /kaggle/working/AT-RAG/artifacts/tae_v2_150k_bertopic/run_001 \
  --device cpu --batch-size 128 --window-batch-size 64 \
  --min-available-ram-gib 12 --min-free-disk-gib 6
```

If using a read-only Kaggle Dataset, replace only `--corpus` with its exact mounted
path. Output must stay in a **new run directory** directly under the repository's
`artifacts/tae_v2_150k_bertopic/`. The first run downloads the exact pinned encoder
snapshot into its own `encoder/`; Kaggle Internet must be enabled. For offline
use, add `--encoder-dir /path/to/verified/all-MiniLM-L6-v2`, containing the existing
`embedding_verification_manifest.json` and matching model files. Files are checked
and copied, and that source directory is never modified.

CPU is the reference. Explicit `--device cuda:0` accelerates only embedding if a
GPU is available; UMAP/HDBSCAN remain CPU. There is no automatic device fallback.
Record and retain the chosen device and environment, including CUDA/cuDNN where
applicable. A GPU session still needs enough **host RAM**.

The one-time preflight, before ML initialization/text-list allocation, considers
host MemAvailable, visible cgroup ancestor headroom, and free output disk. It
requires the larger of 12 GiB available RAM or a text/vector/graph allowance, and
the larger of 6 GiB disk or a model/cache allowance. Prefer a high-RAM CPU session;
these heuristics do not guarantee protection against OOM. Do not lower the floors
to force a run into inadequate memory.

The `(150000,384)` embedding payload is **230,400,000 bytes (~219.7 MiB)**.
Encoding/checkpoint verification and assembly operate in bounded batches;
BERTopic receives a read-only NumPy memmap. SQLite checks text duplicates and IDs
without retaining a full corpus index in RAM. A complete BERTopic fit still needs
all 150,000 strings, UMAP graph/index working memory, HDBSCAN prediction data and
topic-aggregated text for c-TF-IDF. Libraries may copy arrays internally; the
vocabulary cap does not bound the vectorizer's transient vocabulary. This is not
an out-of-core fit. Named-stage RSS/kernel high-water marks, host/cgroup memory,
CUDA allocator peaks if used, and stage runtimes are recorded. No full-scale
runtime/memory measurement is claimed by the implementation tests.

## Interruption recovery and outputs

Each embedding batch has a durable `.npy` plus a JSON commit marker recording
its hash/size, exact occurrence-ID/order hash, row range, token statistics and
embedding configuration signature. A marker is atomically published without
replacement only after array fsync; interrupted JSON writes remain `.part` files.
Resume accepts only a verified contiguous prefix. An orphan without a marker is
recomputed; a corrupt committed batch is an error. Code, corpus, encoder hashes,
token policy, batch settings, Python/library versions and device identity must
match. Keep an interrupted run unchanged; restart in a **fresh directory**:

```bash
PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B /kaggle/working/AT-RAG/scripts/train_bertopic_150k.py \
  --corpus /kaggle/working/AT-RAG/artifacts/tae_v2_150k_corpus/run_001/corpus.jsonl \
  --output-dir /kaggle/working/AT-RAG/artifacts/tae_v2_150k_bertopic/run_002 \
  --reuse-embeddings-from /kaggle/working/AT-RAG/artifacts/tae_v2_150k_bertopic/run_001 \
  --device cpu --batch-size 128 --window-batch-size 64 \
  --min-available-ram-gib 12 --min-free-disk-gib 6
```

Verified encoder files and completed batches are copied into the new run. A fully
encoded failed fit reuses all embeddings without loading the encoder. There is
no mid-fit UMAP/HDBSCAN checkpoint: a stopped fit restarts on **all 150,000 rows**.
Existing directories, including empty ones, files and symlinks are refused.

Expected outputs include `embeddings.npy`, `embedding_batches/`, encoder files
and manifest, `bertopic_model.pkl`, `occurrence_topics.jsonl` (150,000 rows),
`topic_metadata.json`, `training_config.json`, `resolved_parameters.json`,
`embedding_config.json`, `input_validation.json`, `preflight.json`,
`embedding_statistics.json`, `embedding_matrix.json`, `runtime_statistics.json`,
and `events.jsonl`. The final **manifest.json with status=complete** is the sole
success marker; it hashes every experiment/model file, excluding itself and
regenerable `.cache`/`runtime_cache` contents. Runtime caches are kept inside the
new run too. Require that marker and verify hashes
before downstream use. On handled failure, `failure.json` records the stage,
exception and recovery instructions. SIGKILL/OOM may prevent a failure handler;
absence of a complete manifest still identifies an incomplete run.

The pickle preserves the complete fitted BERTopic, UMAP transform, HDBSCAN
prediction data, topic mapping, sparse c-TF-IDF and vectorizer state. Reload is
verified before completion, without retaining two fitted model objects at once.
Load only a trusted, hash-verified pickle in its recorded environment. Encoder
weights are saved separately; pass compatible embeddings to model inference.
With `calculate_probabilities=False`, new-query transform returns scalar
HDBSCAN membership strengths. Future code can request memberships for small query
batches using the saved prediction state (and BERTopic's topic-ID mapping), without
ever building a 150,000-by-topic training probability matrix. These strengths are
not calibrated evidence adequacy. Adaptive retrieval remains future work.

All large outputs stay ignored by Git. Preserve the entire stopped run privately
before a Kaggle session ends, restore it read-only or into a separate writable
location, and point `--reuse-embeddings-from` at it. Never treat a copy taken while
the original process was writing as a consistent checkpoint.
