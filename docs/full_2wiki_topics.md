# Full-corpus BERTopic training for TAE V2 preparation

This trainer fits only the verified 396,978 unique 2WikiMultiHopQA **training**
paragraphs using their existing `(396978, 384)` normalized float32 MiniLM vectors.
It never instantiates an encoder, reads development questions, uses support/answer
annotations, implements TAE V2 selection, or constructs a retrieval index.

## Proposed first full-corpus configuration

| Component | Explicit settings |
| --- | --- |
| UMAP | 5 components, 15 neighbors, cosine, min_dist=0, 200 epochs, spectral initialization |
| UMAP reproducibility/memory | random_state=42, transform_seed=42, n_jobs=1, low_memory=True, force_approximation_algorithm=True |
| HDBSCAN | min_cluster_size=100, min_samples=10, Euclidean on the 5-dimensional UMAP output |
| HDBSCAN algorithm | boruvka_kdtree, leaf_size=40, alpha=1, approx_min_span_tree=True, gen_min_span_tree=False, core_dist_n_jobs=1 |
| HDBSCAN selection/prediction | EOM, cluster_selection_epsilon=0, allow_single_cluster=False, prediction_data=True |
| CountVectorizer | English stop words, unigrams, min_df=2, max_features=20000, int32 counts |
| c-TF-IDF | bm25_weighting=False, reduce_frequent_words=False |
| BERTopic | embedding_model=None, nr_topics=None, calculate_probabilities=False, low_memory=True, top_n_words=10 |
| Runtime | seed 42, PYTHONHASHSEED=42 at interpreter startup, one Numba/BLAS thread |

These are a proposed, frozen starting configuration, not parameters tuned on
development labels or an assertion of optimal topic quality. The CLI exposes
minimum cluster size, minimum samples, neighbors and epochs; every selection is
recorded. Keep defaults for the first full thesis run and use a new output path
for any later planned comparison. No topics are reduced or merged after fitting,
and no outliers are reassigned. Topic `-1` remains a valid outlier label.

### Comparison with existing conventions

`vectorDB/train_topic_model.py` creates `BERTopic(min_topic_size=50)`, embeds text
inside fitting, leaves UMAP seeds/clustering details implicit, and optionally
reduces to `nr_topics` afterward. Its example selects a test subsample. This new
entry point exclusively fits the frozen training corpus, requires the pinned
precomputed encoder identity, and does not perform optional topic reduction.

The 500-paragraph pilot in `scripts/run_bertopic_pilot.py` already uses 5D cosine
UMAP with 15 neighbors, min_dist=0, fixed fit/transform seeds, low-memory mode and
one job; HDBSCAN min_cluster_size=10/min_samples=5 with prediction data; normalized
384D embeddings; and a bounded unigram vocabulary. These conventions are retained.
For the full corpus, the minimum cluster size increases to 100 and min_samples to
10 to avoid treating tiny pilot-scale groups as the initial full-corpus setting.
The vocabulary cap increases from 5,000 to 20,000 for broader coverage. Epochs are
explicitly 200 (UMAP's usual large-dataset default), approximate neighbors are
forced, and KD-tree HDBSCAN is selected explicitly. Topic counts remain discovered
by clustering. English stop words and min_df=2 match the pilot; BERTopic applies
that vectorizer to **topic-aggregated texts**, so min_df here is not a raw-paragraph
frequency threshold. No existing implementation files are changed.

## Verify and run on Kaggle

Use the implementation from PR branch `codex/train-full-2wiki-topics`, or the thesis
branch after a separate merge, checked out at `/kaggle/working/AT-RAG`. Use a
high-RAM **CPU** session: this trainer uses CPU UMAP/HDBSCAN and needs no GPU,
Hugging Face download, or encoder weights. Required packages are NumPy, BERTopic,
umap-learn, hdbscan, scikit-learn, scipy, pandas, numba, pynndescent, joblib and
threadpoolctl (BERTopic's dependencies may additionally import transformer libraries).
Do not install the repository's entire legacy requirements for this phase.

The real synthetic integration test was exercised in the managed Python 3.12
environment with BERTopic 0.16.3; the final PR records the tested library versions.
Use Python-3.13-compatible package wheels on Kaggle, run the synthetic test there
first, and retain that session's exact versions. This is not a claim that all
legacy package pins support Python 3.13. In a fresh environment, a starting install
is `python -m pip install 'bertopic==0.16.3' umap-learn hdbscan threadpoolctl`;
dependencies still need to resolve for the installed Python. Do not upgrade a
saved-model environment before reloading it.

Exact full run (in a notebook, use a `%%bash` cell or prefix with `!env`):

```bash
PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B /kaggle/working/AT-RAG/scripts/train_full_2wiki_topics.py \
  --corpus /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-corpus-zip/documents.jsonl \
  --embeddings /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-embeddings/embeddings.npy \
  --mapping /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-embeddings/document_ids.jsonl \
  --embedding-manifest /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-embeddings/manifest.json \
  --output-dir /kaggle/working/at-rag-full-2wiki-topics \
  --min-cluster-size 100 --min-samples 10 --n-neighbors 15 --umap-epochs 200 \
  --min-available-ram-gib 24 --min-free-disk-gib 8
```

Setting PYTHONHASHSEED inside an already running Python process cannot fix its hash
seed. Launch the script as a separate process with the environment prefix above.

The default expected count is 396,978 and the corpus SHA-256 is fixed to
`f1824d1dc9275731243c6f5a89cc4d702c77b3607413343c9c75bb242fff3160`.
The embedding manifest must be complete and agree on the exact
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, dimensions, dtype, normalization,
256-token setting, document count and corpus hash. Configuration/artifact/vector
checksums, finite unit-length rows, file sizes and every document-ID/row pair are
verified before fitting and again before completion. Overrides of expected count
and corpus hash are for explicitly chosen alternate/synthetic inputs only.

## Memory and disk planning

`np.load(..., mmap_mode="r", allow_pickle=False)` supplies the original read-only
float32 matrix directly to BERTopic. Validation operates in 1,024-row blocks and
file hashes stream in 1 MiB blocks. The script does not make a full `astype`,
renormalization or encoder copy. The vector payload itself is about 582 MiB.

UMAP's approximate neighbor graph avoids a dense document-by-document distance
matrix; KD-tree HDBSCAN operates on only five dimensions. A full float64 pairwise
matrix for 396,978 documents would be about 1.15 TiB and is not this configuration.
Disabling `calculate_probabilities` avoids a dense document-by-topic membership
matrix. Scalar HDBSCAN membership strengths are retained instead.

This is **not** an out-of-core BERTopic fit: its API needs the complete text list,
UMAP/PyNNDescent build graph/index working data, HDBSCAN retains prediction data,
and BERTopic groups documents into topic texts for c-TF-IDF. Third-party code can
copy embeddings internally and aggregate/copy text. The vocabulary cap bounds
final vectorizer features, not its transient pre-pruning vocabulary. Sparse
c-TF-IDF avoids an intentional dense corpus-term matrix but cannot eliminate
vocabulary or topic-text memory. Model pickle size can include UMAP's input vectors
and neighbor structures even though no embedding encoder is saved.

The preflight uses the pilot's unchanged Linux host/cgroup v1/v2 memory helper,
taking the minimum of system MemAvailable and visible cgroup headroom. It requires
the larger of **24 GiB available RAM** and a planning allowance of three text-list
sizes, three embedding payloads, graph work and 2 GiB overhead. It requires the
larger of **8 GiB free disk** and four embedding payloads plus text size and 1 GiB.
Text size is estimated by a streaming pass before materializing the list. Remaining
memory is checked after loading texts and disk is checked again before saving.
Resource snapshots include the kernel-reported process RSS/high-water mark.

These conservative gates are planning heuristics, not guaranteed peak bounds.
Start with a 32 GiB-or-larger environment that actually has 24 GiB available;
prefer more RAM if possible. Low-RAM GPU sessions may fail preflight. Do not lower
the floors just to force a full run through a small session. An OS OOM kill is
still possible. Runtime may exceed a Kaggle session; single-thread controls favor
reproducibility over speed. No full-corpus runtime or RAM benchmark is claimed.

## Recovery and output contract

The sibling `<output>.inprogress/` contains an advisory lock, immutable `job.json`,
atomic `progress.json`, staged artifacts, and (only after a complete fit/save)
`fit_checkpoint.json`. No final output directory exists during fitting. Failures
record their stage when the process can run its handler; SIGKILL/OOM cannot do so.
An existing final destination, including an empty directory, is refused.

Rerun the exact command with `--resume` to recover. Inputs, code/helper checksums,
parameters, resource settings and recorded environment must match. If the complete
fitted-model checkpoint exists and its hash/size match, it is reloaded to finish
artifact export without fitting again. The checkpoint retains the original fit's
resource snapshots and elapsed time so resumed export does not lose that evidence.
If fitting or checkpoint serialization was
interrupted, `--resume` **restarts fitting from the beginning**. There is no mid-fit
UMAP/HDBSCAN resume. A corrupt checkpoint is refused, not silently retrained.
Use a new output directory for a new configuration or environment.

On success the staged directory is atomically renamed to the final path. It contains:

- `bertopic_model.pkl`: full fitted BERTopic/UMAP/HDBSCAN state, including prediction
  data and c-TF-IDF/vectorizer, without an embedding encoder.
- `document_topics.jsonl`: original corpus order; zero-based `row`, `document_id`,
  fitted `topic`, `bertopic` string (`Topic -1` is preserved), membership strength.
- `topic_metadata.jsonl`: topic IDs, sizes, corpus proportions and c-TF-IDF terms/scores.
- `topic_summary.json`: number of non-outlier topics, outlier count/rate, every topic
  size and size histogram. Successful all-outlier assignments are reported without
  retuning. A fit with too few topic groups can fail vectorization with min_df=2;
  this is an explicit failed run, not a trigger to silently change the parameters.
- `training_config.json`, `input_validation.json`, `resource_report.json`: exact
  configuration, seeds, library versions, input evidence, resource observations.
- `resolved_parameters.json`: the actual fitted estimators' constructor parameters,
  including library defaults and their class identities. Nonfinite float defaults
  (such as an infinite clustering threshold) are explicitly encoded as objects.
- `manifest.json`: complete status, document count, configuration hash, topic
  summary, input identity and SHA-256/size checks for every saved artifact.

Require the final path, complete manifest, count and checksum checks before use.
Do not consume `.inprogress/artifacts/`, even if a failed publication left a
manifest there. Pickle must only be loaded from trusted thesis artifacts after
checksum verification, in the same Python/library environment. Compact tensor
serialization is deliberately not used because it omits the fitted UMAP/HDBSCAN
pipeline needed here. This script and its resource helper must remain available
from the repository for a new run or checkpoint recovery.

Later work can join `document_topics.jsonl` to the corpus/embedding rows by exact
document ID (and row) to construct a topic-aware index. For new queries, supply
embeddings from the **same pinned encoder and normalization/tokenizer settings**
to `loaded_model.transform([query_text], embeddings=query_vectors)`. The saved
UMAP transform and HDBSCAN prediction data support this without refitting. Do not
call transform without supplied embeddings, and do not interpret membership
strength as calibrated adequacy or implement TAE V2 routing in this phase.

## Private Kaggle Dataset and reproducibility

After successful completion, export the entire final directory (or a ZIP), use
Kaggle's **New Dataset** flow, and select **Private** before creation. A suitable
name is `at-rag-full-2wiki-topics`. Verify privacy, retain the Dataset URL/version
with the manifest, and attach it read-only to later experiments. Keep the matching
Python/library environment record. Do not commit the generated model, assignments,
embeddings, input data, checkpoints or model weights to Git.

Kaggle working storage can disappear. To preserve a stopped run, archive the entire
`.inprogress` directory privately with an explicit incomplete/checkpoint name;
restore it to writable `/kaggle/working` before `--resume`. Never copy a live run as
a consistent backup. An archive without a completed fitted checkpoint will still
need to restart training. Seeds, fixed order, explicit algorithms and threads are
reproducibility controls; approximate neighbors, floating-point arithmetic,
platform/library changes and topic-label permutation prevent any promise of
identical clustering or model bytes across environments. Do not compare topic
numbers from separately fitted models as though they were stable semantic IDs.

## Tests

```bash
PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 python -B -m unittest discover -s tests -p 'test_train_full_2wiki_topics.py' -v
```

Synthetic tests validate integrity/alignment, mmap use, metadata/outliers, resource
gates, failure/checkpoint recovery, locks and publication. When BERTopic/UMAP/
HDBSCAN are installed, the integration test fits 120 synthetic training documents
with precomputed vectors, saves/reloads the real model and assigns a held-out query
using supplied embeddings. An encoder call is forbidden by the test. It verifies
prediction support, not full-corpus quality, resource usage or cross-platform equality.
