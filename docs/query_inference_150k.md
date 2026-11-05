# Step 4D: read-only query encoding and topic inference

`scripts/query_inference_150k.py` exposes `QueryInference150k`. It consumes the
existing trusted `artifacts/tae_v2_150k_bertopic/run_001/` directory. It neither
trains nor saves a model, regenerates document embeddings, selects a retrieval
route, computes evaluation metrics, nor changes the original AT-RAG pipeline or
`Shared150kRetrieval`. Real 150k inference verification belongs in Kaggle only.

## Initialization, trust and environment

Use the complete, externally backed-up training directory, including every file
listed in `manifest.json`. Initialization verifies status `complete`, schema 1,
150,000 occurrences, the frozen corpus SHA-256, pinned encoder ID/revision, and
**every listed artifact's SHA-256 and byte size** before deserialization. Large
arrays/checkpoints are hashed in streaming blocks without loading them into RAM.
This full inventory check is a one-time startup cost; missing batch/checkpoint
files cause an error rather than regeneration. An optional separately recorded
`trusted_manifest_sha256` pins the manifest itself. Hashes establish integrity
relative to that manifest; use only your own trusted training backup, since a
self-consistent untrusted manifest cannot establish pickle provenance.

The model is deserialized from the same read-only descriptor that was hashed.
The default loader requires the complete BERTopic/UMAP/HDBSCAN pickle. The saved
model's counts and raw-label-to-topic mapping are checked against every fitted
row and the manifest topic summary. Missing prediction data is an error; no
cosine-to-topic-embedding fallback is substituted.

Python **3.12** is required for both saved and current environments. Before
loading, the component compares exact saved versions of NumPy, BERTopic, UMAP,
HDBSCAN, scikit-learn, scipy, pandas, numba/llvmlite, pynndescent, joblib,
threadpoolctl, Sentence Transformers, Transformers/tokenizers, Torch,
huggingface-hub and safetensors. Python 3.12 patch versions may differ and are
reported. Restore the actual training freeze; the repository's
`requirements/bertopic_150k.txt` is the reference environment, not permission to
upgrade a saved-model environment. There is no CLI version/hash/count bypass.

Saved encoder files are loaded locally with `local_files_only=True` and
`trust_remote_code=False`. Offline flags prevent downloads. Regenerable library
runtime caches live in a temporary directory outside the artifact tree and are
removed on close; source files and backups are never written. Initialization
reuses the exact recorded window batch size and checks the fingerprints of the
unchanged training helpers. A changed helper, model file, encoder file or policy
fails validation. Keep inputs immutable throughout inference.

## Exact encoding and signal semantics

Questions pass unchanged to `train_bertopic_150k.encode_occurrences`. The pinned
encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. All content tokens are encoded in
nonoverlapping windows within the 256-token model limit. Normalized window vectors
are averaged using content-token-count weights and normalized again. No question
tail is silently discarded. The output is a read-only, normalized float32 NumPy
vector with shape `(384,)`, accepted directly by `Shared150kRetrieval`.

For each question, the saved UMAP transform runs once. HDBSCAN
`approximate_predict` supplies a raw cluster label and a separate scalar strength;
the saved BERTopic original-topic mapper supplies the final topic ID. A query
may predict `-1`, including when training had no outliers. A missing scalar is
returned as `None`; zero remains zero.

For supported prediction state, `hdbscan.membership_vector` runs only on this
query, never on the training corpus. In the recorded HDBSCAN implementation,
columns follow sorted selected condensed-tree cluster IDs. The component checks
those IDs through prediction data's `cluster_map` and `reverse_cluster_map`,
then through BERTopic's original-cluster-to-final-topic mapping. It validates
coverage and uniqueness and sums columns if several raw clusters map to one
topic. It never assumes a column index equals a BERTopic topic ID.

Ranked signals sort by descending membership, then ascending topic ID for ties.
Their total mass is preserved: no softmax, forced unit sum, invented outlier
component, residual `1-sum` probability, or scalar-strength substitute is used.
The predicted topic need not be the largest soft membership. Missing optional
membership metadata/API is reported explicitly as unavailable; inconsistent
mapping, wrong dimensions, nonfinite/negative values and invalid mass are errors.
**Both scalar strengths and per-topic memberships are uncalibrated signals, not
probabilities that retrieved evidence will be adequate.**

## API and output

From a fresh compatible Kaggle process:

```python
import sys
sys.path.insert(0, "/kaggle/working/AT-RAG/scripts")
from query_inference_150k import QueryInference150k

with QueryInference150k() as inference:
    result = inference.infer("Which river flows through Paris?")
    q = result["embedding"]             # pass directly to engine.retrieve(q, ...)
    predicted = result["topic_id"]      # may be -1
    strength = result["membership_strength"]
    ranked = result["topic_memberships"]
    print(inference.validation_report)
```

`topic_memberships` is a list of `{"topic_id": int, "membership": float}` or
`None`. Other fields include `raw_cluster_id`, `membership_status`,
`membership_unavailable_reason`, `membership_sum`, `encoding_statistics` and
`signal_interpretation`. The validation report records manifest/model hashes,
verified artifact count, environment, encoder identity and the validated mapping.
Reuse one component across questions; close it after active calls finish.
Callers choose any later explicit retrieval mode. This step contains no adequacy
thresholds, topic-set selection, fallback routing or LLM question rewriting.

## Tests and real Kaggle verification

Synthetic tests use fake encoder weights and controlled inference state. The
opt-in integration test fits a **new 120-row synthetic model**, saves/loads its
pickle, and checks predictions, scalar strengths and memberships against
BERTopic's native transform. It uses a synthetic encoder and downloads no model.

```bash
python -B -m unittest discover -s tests -p 'test_query_inference_150k.py' -v

PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATRAG_RUN_TINY_QUERY_TEST=1 \
python -B -m unittest discover -s tests -p 'test_query_inference_150k.py' -v
```

Run this **only in Kaggle** to verify one natural-language question against the
already-trained real model, using the saved Python/package environment:

```bash
PYTHONHASHSEED=42 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B /kaggle/working/AT-RAG/scripts/query_inference_150k.py \
  --training-dir /kaggle/working/AT-RAG/artifacts/tae_v2_150k_bertopic/run_001 \
  --question 'Which river flows through Paris?' --device cpu
```

Replace `--training-dir` with a complete read-only Dataset mount if appropriate.
Optionally add `--trusted-manifest-sha256` with the hash recorded separately with
your backup. The CLI prints one JSON object containing the validation report and
result, with the vector serialized as a 384-element list; it creates no saved
output artifact. Confirm successful manifest validation, a unit vector, a mapped
topic or `-1`, a finite scalar when available, and either mapped ranked signals
or an explicit unavailable reason. A long question should report multiple
windows and all content tokens. Do not interpret a single-question smoke check
as retrieval evaluation. No real model latency or full-scale inference claim is
made by the Codex synthetic tests, and TAE V2 policy remains a later step.
