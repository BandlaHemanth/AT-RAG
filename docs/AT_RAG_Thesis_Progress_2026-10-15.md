# AT-RAG M.Tech Thesis Progress — October 15, 2026

Professor progress meeting; numerical results verified against saved JSON/CSV reports on October 14, 2026.
Repository: `BandlaHemanth/AT-RAG` · Branch: `thesis/topic-filter-pilot` · Experiment base commit: `ef185ff004499dd5860f00aaee68b63944066ba5`.

**Main finding:** The fixed TAE V1 abstention gate increased gold-evidence eligibility from 63.30% to 84.04% on 80 separate development questions, while increasing the mean eligible candidate set by 2.05×. This is an offline eligibility result, not retrieval or answer accuracy.

## Slide 1 — Research problem and original AT-RAG behavior

- Multi-hop questions require multiple supporting paragraphs, which can span topics. A restrictive topic filter can remove evidence before similarity ranking considers it.
- Original implementation: `models/AT_RAG.py`, `retrieve()`, assigns the query topic and applies `{"bertopic": "Topic <assigned_topic>"}` as the vector-store metadata filter before top-k similarity retrieval.
- A document survives exactly when `candidate_topic == query_topic`; topic `-1` is an ordinary equality match. The original method receives BERTopic probabilities but does not use them to decide whether filtering is adequate.
- Thesis direction: inference-time abstention from topic filtering when the frozen model cannot assign a positive-strength non-outlier cluster.

## Slide 2 — Proposed TAE V1 and experimental methodology

```text
IF predicted_topic == -1 OR membership_strength is nonfinite OR membership_strength <= 0:
    UNFILTERED
ELSE:
    STRICT_TOPIC_FILTER
```

- This rule was fixed before scoring the remaining 80 questions. No tunable threshold, gold labels, answers, or evidence-retention outcomes determine routing.
- Membership strength is the scalar HDBSCAN approximate-prediction output, not calibrated evidence confidence. `calculate_probabilities=False`; the output is a one-dimensional strength vector, not a complete multi-topic distribution.
- Processed 2WikiMultiHopQA source: repository-documented Google Drive file `1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC`. Inputs: `processed_data/2wikimultihopqa/train.jsonl` and `dev_subsampled.jsonl`; original splits were retained.
- Training: first 500 unique nonempty exact-text paragraphs in training file/context order. Only these paragraphs fit BERTopic; development data is never fitted or used for tuning.
- Exploration: development questions 1–20. Fixed-rule evaluation: questions 21–100 (80 questions), with no question-ID or query-text overlap with exploration and no selected-training/evaluation question-ID overlap.
- Evaluation corpus: 500 frozen training paragraphs plus **all** 800 context occurrences from the 80 questions, selected without support labels and deduplicated by exact text: **1,077 unique documents**. Gold labels enter scoring only after routing decisions are saved and hashed.

**Frozen model and environment**

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; offline CPU inference, normalized 384-dimensional embeddings, maximum sequence length 256, batch size 4.
- Seed 42; UMAP: 5 components, 15 neighbors, cosine metric, minimum distance 0, one job. HDBSCAN: minimum cluster size 10, minimum samples 5, Euclidean metric, prediction data enabled, one core-distance job.
- CountVectorizer: English stop words, `min_df=2`, maximum 5,000 features, unigrams. No topic reduction or parameter changes.
- Linux; Python 3.12.14; BERTopic 0.16.3; sentence-transformers 3.1.1; PyTorch 2.5.1+cpu; UMAP 0.5.12; HDBSCAN 0.8.44. Approximately 32 GiB container RAM; no paid APIs.
- Training outcome: **17 non-outlier topics**, **90/500 outliers (18%)**. Fitting: 9.74 s; complete pilot: 38.93 s.

## Slide 3 — Exploratory evidence-retention diagnostic (20 questions)

- All **48/48 gold occurrences** were verified present in the 667-document pilot corpus; no unmatched or ambiguous mappings.
- Strict filtering retained **31/48 (64.58%)**, excluding 17. Macro question retention: 62.50%.
- **13/20 questions (65%)** lost at least one gold paragraph; 7/20 retained all.
- Topic `-1`: 13 queries, 7 lost evidence. Non-outlier topics: 7 queries, 6 lost evidence.
- These exploratory outcomes motivated studying abstention; they were not reused to tune a numerical threshold. The exploratory and evaluation corpora differ, so their percentages are not a before/after performance comparison.

## Slide 4 — Fixed-rule evaluation (80 separate questions)

All **188/188 gold paragraph occurrences** were verified in the same candidate corpus. Occurrence-level scoring preserves duplicate text and validates question-local position, text and metadata; `idx` is never a global document identifier. One repeated gold document contributes two occurrences.

| Strategy | Gold retained | Excluded | Micro retention | Macro retention | All gold retained | Losing ≥1 gold |
|---|---:|---:|---:|---:|---:|---:|
| C1: strict | 119/188 | 69 | 63.30% | 63.44% | 35/80 (43.75%) | 45/80 (56.25%) |
| C2: unfiltered | 188/188 | 0 | 100.00% | 100.00% | 80/80 (100.00%) | 0/80 (0.00%) |
| C3: TAE V1 | 158/188 | 30 | 84.04% | 83.75% | 60/80 (75.00%) | 20/80 (25.00%) |

- TAE fallback: **47/80 (58.75%)**, all topic `-1` queries with zero strength. All 33 positive-strength non-outlier queries stayed strict; no nonfinite strengths occurred.
- Versus C1: **39 additional gold occurrences eligible**, **+20.74 percentage points** micro retention. Eligibility increased for 25 questions and was unchanged for 55.
- Negative finding: TAE still excludes **30 gold occurrences** and **20/80 questions** still lose evidence. Positive membership strength does not guarantee topic adequacy.

## Slide 5 — Candidate-set trade-off and validity

| Strategy | Mean eligible candidates | Median | Minimum–maximum | Mean fraction of corpus |
|---|---:|---:|---:|---:|
| C1: strict | 324.06 | 498 | 14–498 | 30.09% |
| C2: unfiltered | 1077.00 | 1077 | 1077–1077 | 100.00% |
| C3: TAE V1 | 664.23 | 1077 | 14–1077 | 61.67% |

- TAE expands the average candidate set by **340.16 documents (2.05× C1)**, while leaving 38.33% fewer candidates than unfiltered eligibility on average.
- Evaluator runtime: **27.15 s**; encoding 9.42 s; frozen inference/routing 7.80 s. Peak process RSS: **668.72 MiB**. These are evaluator measurements, not retrieval latency.
- Validation passed: model/artifact hashes, local encoder provenance, installed package versions, frozen parameters/centroids, scalar strength shape `(80,)`, split separation, route-before-label event order, immutable candidate assignments, mappings and metric denominators.
- Only warning: TensorFlow unavailable for ParametricUMAP; the experiment uses ordinary UMAP.

## Slide 6 — Limitations and next thesis experiments

- **Pre-ranking eligibility only:** retained paragraphs are allowed into the search space, not guaranteed to be retrieved. No recall@5, answer accuracy, end-to-end AT-RAG performance, or benchmark-superiority claim is supported.
- All supplied development contexts are included irrespective of support labels. Consequently, unfiltered gold availability is favorable by construction; this is not realistic large-corpus retrieval coverage.
- One small frozen topic model and one 80-question development evaluation. The rule was informed by exploration on the same dataset; the 80 questions are separate evaluation records, not an untouched benchmark test.
- Gains from falling back to the full corpus are mechanical eligibility gains and must be weighed against candidate-set expansion. Membership strength is not calibrated evidence confidence.

**Next experiments — proposed, not implemented**

1. Add identical offline similarity ranking to C1/C2/C3 and measure gold Recall@5/10 and complete multi-hop evidence retrieval; keep routing fixed before ranking.
2. Evaluate a larger candidate corpus whose construction does not depend on evaluation-question contexts; report verified corpus coverage separately from filter exclusions.
3. Measure actual retrieval latency and memory alongside evidence retention; candidate counts alone are not a latency model.
4. Assess stability across additional topic-training sizes/seeds and development datasets with a preregistered protocol. Use a separate calibration split before investigating additional gates; reserve final evaluation until the design is frozen.

## Reproducibility and preservation

Existing pilot/download verification scripts are on the experimental branch. New source scripts: `scripts/analyze_topic_filter_retention.py` and `scripts/evaluate_tae_v1.py`. Both passed compile-only syntax checks; measurements were not rerun for this document.

Local evidence files (Git-ignored; not included in the source commit):

- `artifacts/bertopic_pilot_500/`: training manifest, frozen model, configuration, environment, assignments, run report and hashes.
- `artifacts/topic_filter_diagnostic_20/`: per-question/per-gold CSVs, summary, validation and report.
- `artifacts/tae_v1_eval_80/`: inference inputs, frozen assignments, routing CSV, inference checkpoint, per-question/per-gold CSVs, summary, validation, provenance and hashes.

Numeric counts, retention fractions, macro averages, candidate statistics and fallback decisions were independently reconciled from saved CSV and JSON files; pilot/evaluation artifact hashes were checked. Rounded percentages use the original occurrence denominators.

| Evidence JSON | SHA-256 |
|---|---|
| `artifacts/bertopic_pilot_500/run_report.json` | `c1af701ca3598765ee208b4e47ad223713cf82842e8f63e7430e9807994e0be3` |
| `artifacts/topic_filter_diagnostic_20/summary.json` | `dccf385ccebe9c6637f8d7e5b239bb507cd335627352655bcdd9805f9cf2f331` |
| `artifacts/tae_v1_eval_80/summary.json` | `020c7cf4ffb7767aef48ba2933c31fa63097b2e7191b9cb1da6c7ee7e91b20e2` |

**Preservation note:** GitHub preserves the committed scripts and this numerical progress record, not the ignored cloud datasets/models/results. Keep or export the latest experiment artifact directories separately before relying on fresh cloud-task restoration.
