# TAE-R experimental protocol v1 (DRAFT, not yet frozen)

Status: draft for review on 10 November 2026. It becomes frozen when the split
manifest hash and the commit that contains this file are recorded in section 9.
After freezing, any change is a new protocol version with a stated reason.

This protocol governs the controlled TAE-R experiments. It does not amend
`docs/tae_v2_150k_protocol.md`; the pilot, TAE V1 and 150k results remain
preliminary studies under their own rules and are never compared with results
produced here.

## 1. Data and provenance

Source: processed 2WikiMultiHopQA archive, Google Drive file
`1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC`, measured by
`docs/corpus_design_audit_v1/audit_run_001.json`
(SHA-256 `2c2c4a06d52814f9a83e02d6416462403adade4b22f4d810f91b6535f5d8131b`).

| File | Questions | SHA-256 |
|---|---:|---|
| `train.jsonl` | 167,454 | `14c670fb…c9121a4a` |
| `dev.jsonl` | 12,576 | `c6e1d9fa…81f79055` |
| `dev_subsampled.jsonl` | 100 | `5db56570…4873e089` |
| `test_subsampled.jsonl` | 500 | `4948757b…fa88f5bc` |

Facts from the audit that shape this protocol:

- No question ID is shared between `train.jsonl` and `dev.jsonl`.
- All 100 and all 500 subsampled question IDs are in `dev.jsonl`; the two
  subsampled files share no question ID. `test_subsampled.jsonl` is a
  development-derived holdout, not the official test split.
- The subsampled files carry altered context text (39 of 100 and 232 of 500
  questions differ from `dev.jsonl`). Final evaluation therefore reads
  questions and contexts from `dev.jsonl` only.
- 38,994 unique paragraph texts occur in both `train.jsonl` and `dev.jsonl`.

Text identity everywhere: exact UTF-8 `paragraph_text`; null, empty and
whitespace-only texts are excluded; no normalization.

## 2. Question splits

Built by `scripts/build_tae_r_splits.py` (seed 20261110, seeded SHA-256
ordering, independent of file order). Support labels never influence a split.

| Split | Definition | Size | Used for |
|---|---|---:|---|
| exposed | every ID in the two subsampled files | 600 | nothing in this protocol; reported separately only if labelled |
| final_eval | `dev.jsonl` IDs minus exposed | 11,976 | one final run after everything is frozen |
| e2e_300 / e2e_500 | first 300 / 500 of final_eval in seeded order (nested) | 300 / 500 | end-to-end LLM runs |
| router_train | first 8,000 train IDs in seeded order | 8,000 | training the adequacy predictor |
| validation | next 4,576 train IDs | 4,576 | all tuning, model selection, both gates |
| topic_fit | remaining train IDs | 154,878 | source of the topic-fitting paragraphs |

router_train plus validation equals 12,576 questions, the size of `dev.jsonl`,
so the validation retrieval pool is built at the same scale as the final one.

Rules:

1. No method, threshold, topic-model configuration or prompt is chosen using
   final_eval or exposed questions.
2. final_eval is run once. Its results change nothing afterwards.
3. The 600 exposed questions informed earlier design work; this is disclosed.

## 3. Corpora (kept separate)

**Topic-fitting population.** Unique paragraph texts of topic_fit questions
that occur in no `dev.jsonl`, subsampled, router_train or validation context.
Upper bound from the audit before removing the held-out training folds:
357,742. The exact count is the `topic_fit_population` value in the split
manifest and is recorded in section 9. Used only to fit the topic model.

**Final retrieval corpus.** Every unique paragraph of every `dev.jsonl`
context, supporting and distractor, selected without reading support flags
(56,680 unique texts by the audit). All of it is searchable. This is a
context-pooled benchmark, easier than open Wikipedia retrieval; that
limitation is stated in the thesis.

**Validation retrieval corpus.** Built the same way from the router_train plus
validation questions.

Documents in both retrieval corpora receive topics from the frozen topic model
by inference only. The topic model is never refitted on them. Consequently
every searchable document is out-of-sample for the topic model, under
validation and under final evaluation alike.

Known limitation: exclusion is by exact text. The split manifest reports how
many topic-fitting texts share a title with an evaluation paragraph
(`topic_fit_population_texts_whose_title_occurs_in_evaluation_files`); the
decision to exclude by title as well is taken from that number before freezing
and written in section 9.

## 4. Topic model

1. Recover, if possible, the earlier 396,978-paragraph model and inspect its
   saved configuration, topic sizes and outlier diagnostics. It was fitted on
   text that includes development paragraphs, so it is a diagnostic and a
   comparison, not the final model.
2. Compare a small set of configurations fitted on the topic-fitting
   population: a few HDBSCAN settings and, optionally, spherical k-means.
3. Select on the validation fold by retrieval criteria: the share of gold
   paragraphs that fall in the question's selected topic(s), strict-filter
   Recall@5 and AllGold@5, the oracle-topic ceiling, and seed stability.
   Outlier percentage is reported but does not decide the choice.

## 5. Methods compared

Controlled end-to-end comparison (same frozen corpus, embeddings, questions,
LLM, prompts, k = 5, at most 5 iterations):

1. Original AT-RAG (strict single-topic filter).
2. AT-RAG without topic filtering.
3. AT-RAG with TAE-R.

Retrieval-only baselines where feasible: TAE V1, TAE V2 (BERTopic partition
only), fixed three-topic retrieval, oracle-topic upper bound.

A separate small **faithful reproduction** runs the released code in its own
setup, changed only where it cannot run. In the controlled comparison the
following fixes are applied to all three systems and documented: per-question
state reset, errors raised instead of reusing the previous answer, pinned
model snapshot, temperature 0, cached LLM calls.

## 6. Measurements

Per method and per question: Recall@5, Recall@10, AllGold@5, AllGold@10;
route taken; retrieval operations and similarity computations summed over
every probe and widening step; candidates scored; retrieval latency; LLM
calls; rewriting iterations; EM and F1 for end-to-end runs.

Statistics: paired bootstrap 95% confidence intervals over questions (10,000
resamples, fixed seed) for differences; McNemar's test for AllGold. Results
are also broken down by question type.

## 7. Decision gates (validation fold only; thresholds fixed here in advance)

**Gate 1, cross-topic evidence.** With the selected topic model, the
cross-topic rate is the share of validation questions whose gold paragraphs
lie in two or more distinct topics.

- 20% or more: multi-topic routing is a main contribution.
- 5% to under 20%: it is a secondary contribution.
- under 5%: the router is reported as a side experiment; the thesis leads with
  the oracle analysis, the reproduction and the no-filter ablation.

**Gate 2, TAE-R against unfiltered retrieval** on AllGold@5.

- Better: the 95% interval of (TAE-R minus unfiltered) lies above 0.
- Equivalent: the interval lies within plus or minus 1 percentage point.
- Otherwise: worse or inconclusive, reported as such.

An efficiency advantage is claimed only if TAE-R is better or equivalent
**and** its full measured cost (section 6, all probes included) is lower.

## 8. Reporting rules

- Corpus availability, pre-ranking eligibility, ranked recall and answer
  accuracy are reported as separate quantities with denominators.
- No significance claim from differences whose interval includes 0.
- Negative results are reported.

## 9. Freeze record (to be completed)

- Split manifest SHA-256:
- Measured topic-fitting population:
- Title-level exclusion decision:
- Commit containing this protocol:
- Date frozen:
