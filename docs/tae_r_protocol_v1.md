# TAE-R experimental protocol v1 (FROZEN 10 November 2026; amendment v1.1 in section 10)

Status: frozen on 10 November 2026 with the record in section 9, before any
validation or final-evaluation result exists. Any later change is a new
protocol version with a stated reason.

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
The audit's 357,742 was an upper bound that removed only evaluation-file texts.
The split manifest measured the true value after also removing router_train and
validation texts: 312,555. Title-level exclusion (below) reduces it by 7,114 to
305,441. Used only to fit the topic model.

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

Exclusion is by exact text and also by title: the 7,114 topic-fitting texts
whose title occurs in any evaluation file are excluded as well
(`topic_fit_population_texts_whose_title_occurs_in_evaluation_files`), because
same-title paragraphs are likely near-copies of evaluation paragraphs. The
population builder applies this rule and reports the resulting count.

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

## 9. Freeze record

- Split manifest SHA-256: `44e79c259e5f0ac55eee8f83fb65493b70432fb276a80809621919ec35ba38ce`
- Split script SHA-256: `864f474f9c0ff1125f431826f8cba55270d50f05bd27b12e96113a59bdb3fce8` (seed 20261110, run on commit `5d7fd89`)
- Split archive `tae_r_splits_v1_run01.zip` SHA-256: `1df5550e379dd566688ebcee795dac48a1b107723a58952fd271c7c0df63d387`
- Split sizes: exposed 600; final_eval 11,976; e2e 300 and 500; router_train 8,000; validation 4,576; topic_fit 154,878
- Measured topic-fitting population (text exclusion): 312,555
- Title-level exclusion decision: exclude 7,114 texts; expected population 305,441
- Retrieval corpora: final 56,680 unique texts; validation 56,655
- Gates and thresholds: as in section 7, unchanged from the draft
- Commit containing this protocol: the commit that adds this record
- Date frozen: 2026-11-10

## 10. Amendment v1.1 (10 November 2026, before any experiment)

Reason: `scripts/build_tae_r_corpora.py --mode measure` (commit `3366b3a`)
showed that 6,564 topic-fitting texts share a title with router_train or
validation paragraphs, in addition to 7,123 that share a title with evaluation
files. Under v1 these would stay in the topic-fitting set while the validation
corpus is searched, so the validation gates would be measured under a weaker
leakage control than the final run.

Changes (nothing else in v1 changes; no result existed when this was decided):

1. Title matching is Unicode NFC, case-folded and whitespace-collapsed, and a
   text is excluded if any of its titles matches (v1 used exact titles; the
   exact count was 7,114, the normalized count is 7,123).
2. The excluded title set is the union of titles in evaluation files and in
   router_train and validation contexts: 11,976 texts.
3. Expected topic-fitting population: 312,555 - 11,976 = 300,579 paragraphs.

Measured values in this section come from the measure run; the build manifest
records the final figures and corpus hashes.
