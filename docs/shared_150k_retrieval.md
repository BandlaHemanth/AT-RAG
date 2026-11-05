# Step 4A: shared occurrence retrieval

`scripts/shared_150k_retrieval.py` supplies exact cosine ranking for original-style
single-topic, multi-topic OR, and unfiltered retrieval. It reads the existing
training artifacts and never fits a model, encodes documents/queries, loads a
pickle, creates a database, or edits any artifact. `models/AT_RAG.py`, its answer
graders, generation and query-rewriting graph remain unchanged.

## Inputs and initialization

Defaults are the frozen `artifacts/tae_v2_150k_corpus/run_001/corpus.jsonl` and
`artifacts/tae_v2_150k_bertopic/run_001/{embeddings.npy,occurrence_topics.jsonl,manifest.json}`.
Explicit paths support read-only Kaggle Dataset mounts. Initialization requires
manifest status `complete`, schema 1, exactly 150,000 rows, the pinned MiniLM ID
and revision, and corpus SHA-256
`d9edc9d9bc3ff4bbb98de8690c5b79c71aa6e2878f035df55cd01891ea322c33`.
It verifies consumed embedding/assignment hashes and byte counts against the
training manifest; checks embedding shape `(150000,384)`, float32, finite values
and unit norms; and joins assignments to corpus rows, unique occurrence IDs,
exact text hashes and source provenance. Topic totals must match the manifest.

These expensive checks run once per engine initialization. `validation_report`
records the verified hashes (including the manifest hash), row count, encoder
identity and topic/outlier totals. Kaggle can compare those totals with the
reported 684 non-outlier topics and 54,697 outliers; the engine requires the
manifest and actual assignments to agree. Those artifacts are absent in Codex.
Unconsumed model/checkpoint/cache files are
not opened or hashed by this retrieval-only component.

The matrix is mapped read-only. Initialization uses a temporary in-memory set to
check occurrence-ID uniqueness; persistent state is small row/topic/norm/byte-offset
arrays. Original texts are read only for final hits. Scoring uses bounded blocks
(default 4096); filtered copies never exceed a block. There is no full-matrix
cast or copy. Each query performs cheap file-metadata guards, not another integrity
scan. Sources must remain immutable while the engine is open. Close it only after
active queries finish.

## Python interface

NumPy plus the standard library are sufficient; use the existing thesis Python
3.12 / NumPy 1.26.4 environment. From the repository root in Kaggle:

```python
import sys
sys.path.insert(0, "/kaggle/working/AT-RAG/scripts")
from shared_150k_retrieval import Shared150kRetrieval

# q is an already-computed, finite, normalized floating vector of shape (384,).
with Shared150kRetrieval() as engine:
    strict = engine.retrieve(q, mode="single", topic_id=7, k=5)
    multi = engine.retrieve(q, mode="multi", topic_ids={7, 20}, k=10)
    full = engine.retrieve(q, mode="unfiltered", k=10)
    print(engine.validation_report)
```

For mounted artifacts pass `corpus=...`, `embeddings=...`, `assignments=...`, and
`manifest=...` to the constructor. Reuse the **same engine** across methods and
queries. Constructor count/hash overrides exist only to build tiny synthetic
fixtures; production callers must retain the frozen defaults. No CLI override
can change the required production count or identity.

Each result dictionary contains `row` (zero-based corpus row), `occurrence_id`,
`topic`, `text_sha256`, `similarity`, and original `paragraph_text`. The function
returns `min(k, eligible_count)` occurrences. Duplicates consume slots and are
never deduplicated or replaced. `k` must be a positive integer. Topic IDs are
integers >= -1; unknown nonnegative IDs have no matches. Multi-topic input accepts
a one-dimensional list, tuple, set, frozenset or NumPy array, with OR semantics.
An empty collection returns `[]`. Topic -1 enters filtered search only when
explicitly selected; unfiltered search includes every topic and outlier.
Incompatible filters/modes, strings/floats/bools as topic IDs, malformed query
shapes, nonfinite/nonunit queries and invalid k raise `ValueError`.

All eligible rows use the same float64 cosine reduction over unchanged saved
float32 vectors, divided by their measured norms. Queries must already be unit
normalized (absolute tolerance 1e-5); cosine norm correction does not rewrite
either input. Roundoff is bounded to [-1,1]. Results sort by descending score,
then ascending corpus row for **exact computed-score ties**, without epsilon
grouping. Per-block top-k lists are merged globally, including across selected
topics; no per-topic quota or concatenation is used.

## Tests and Kaggle command

Run synthetic checks without downloads, BERTopic or the real artifacts:

```bash
python -B -m unittest discover -s tests -p 'test_shared_150k_retrieval.py' -v
```

A single-query Kaggle command accepts an **existing** normalized `(384,)` query
vector saved as `.npy`; it writes JSON only to stdout:

```bash
python -B /kaggle/working/AT-RAG/scripts/shared_150k_retrieval.py \
  --query-embedding /kaggle/working/query_embedding.npy --mode single --topic 7 --k 5
```

Use `--mode multi --topics 7 20 --k 10` or `--mode unfiltered --k 10` for the other
modes. CLI flags also accept explicit corpus, embedding, assignment and manifest
paths. Each CLI process initializes once; the Python interface avoids repeating
initialization across many queries. This is an interface example, not a full-scale
evaluation command. Query encoding and topic selection remain caller responsibilities;
queries must use the saved MiniLM revision and training's complete-token-window
aggregation/normalization policy.

## Fair comparison and scope

Only eligibility changes across modes. They share corpus occurrence IDs, saved
embedding rows, saved topic assignments, cosine arithmetic, tie rules and k.
Later original-style AT-RAG can supply its single predicted topic; later TAE V2
can supply an explicit selected-topic set or unfiltered mode. This step chooses
no thresholds, topics or fallback route. It implements neither LLM generation nor
question rewriting and computes no evaluation metrics. Future comparisons must
retain the frozen question lists and report overall evidence availability and
retrieval conditional on availability separately; uncovered questions remain in
main results. `test_subsampled` is development-derived, not the official test set.
