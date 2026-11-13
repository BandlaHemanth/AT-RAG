# TAE-R experimental protocol v1 (FROZEN 10 November 2026; amendments v1.1 in section 10, v1.2 in section 11; selection and Gate 1 in section 12; v1.3 in section 13; TAE-R design frozen in section 14; validation results and Gate 2 in section 15; LLM choice rule in section 16; final evaluation plan frozen in section 17; final retrieval results in section 18)

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

### Corpus build record (run 01, 10 November 2026)

Built by `scripts/build_tae_r_corpora.py --mode build --title-exclusion
evaluation_and_heldout` at commit `2a280e5`, from the frozen splits.

- Corpus manifest SHA-256: `d6905fc1c4c64f71c394b15c1fbdcc3c8d6e7a15c7e3b55c710673181057d96e`
- topic_fit_population: 300,579 paragraphs, `cf62736df31d260b2c07ed8387b6d14eef38897e4cd02491a4e4ae01c73faa49`
- retrieval_final: 56,680 paragraphs, `9ec5e936b624505dccd612dabb150527851187ab83cd3781e9be725cc3f0d973`
- retrieval_validation: 56,655 paragraphs, `31a79687e023a2a332f5b14e32f4d96bf9d4ad04bfdf610250c0716509b9357f`
- questions_final_eval: 11,976, `6d05dc628e5a838bd20dfc8bbc23afd6bade1c5f9556a546a36cef14927c8035`
- questions_router_train: 8,000, `0c39dd5c728680915b537eeb653e0b828eea7d8f30f53cbb61b55fc63cd5616f`
- questions_validation: 4,576, `0c2a252ae7787b105e321ba2822822d2759eeab9490a8439c28685f276b476af`
- excluded_by_title_doc_ids: 11,976, `4a7319171d81fe6886dabc42807264ac4cece601f8243ccd1b7769ad1118fc49`
- Measured: 312,555 text-level population - 11,976 title-excluded = 300,579, as expected in v1.1.
- Blank contexts skipped: 0. Questions without gold paragraphs: 0.

### Embedding record (run 01, 10 November 2026)

Built by `scripts/prepare_tae_r_embedding_input.py` and
`scripts/embed_full_2wiki_corpus.py` at commit `88da6bc`; one embedding per unique
paragraph of the topic-fitting population, final and validation retrieval corpora.

- Embedding input `documents.jsonl`: 402,504 paragraphs, SHA-256 `92cbd1f8427ba0dd3bedaf58bbf3d56642f290ba7d2d9ef35728a73c79f3fbcc`
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; text unchanged, no prefix; max length 256; L2-normalized; float32; 384 dimensions
- Run: Tesla T4, CUDA, batch size 128, deterministic algorithms; torch 2.11.0+cu128, sentence-transformers 5.7.0, transformers 5.16.1
- `embeddings.npy` (402,504 x 384): 618,246,272 bytes; file SHA-256 `19c5b4d6cb3bf0773f848a0a490f8787d8af8025ef616f5c6a074df0c0abd809` (checked on the laptop copy); embedding data SHA-256 `31f4ca2c70384577d18d11da3b7f8f92943d5c468516888ad1f401c1a56eab6d`
- `document_ids.jsonl`: 38,529,274 bytes; file SHA-256 `af08575835f1db2dd3ac445e1ff47650faab416e796d50558fee093fec9d80ee`; row order equals the input file order (sorted by document_id)
- Verified: all artifact hashes match the manifest; ids match input order; all values finite; norms 0.9999999 to 1.0000001
- These embeddings are a new set. They are not mixed with the earlier 396,978-paragraph embeddings. (Correction, 11 November: those were made with the same library versions, but with batch size 64 and over a different corpus file; the earlier statement that they came from a different software environment was wrong.)

## 11. Amendment v1.2 (11 November 2026, before any topic model is fitted)

Reason: section 4 left the candidate topic models open. They are fixed here,
after inspecting the two earlier full models and before fitting any new one.

Evidence from the earlier models (both fitted on the same 396,978 unique
paragraphs, which include development text; diagnostic only):

| Model | Kaggle dataset | HDBSCAN min cluster / min samples | UMAP seed | Topics | Outliers |
|---|---|---|---|---:|---:|
| B (reported in the 6 Nov brief) | at-rag-bertopic-trained-backup | 50 / default (= 50) | none | 19 | 193,646 (48.78%) |
| A (later fit) | at-rag-full-2wiki-topics | 100 / 10 | 42 | 418 | 189,270 (47.68%) |

Lowering min_samples from 50 to 10 multiplied the topic count about 22-fold
but moved the outlier share by about 1 point. The models also differ in seed,
software and corpus file, so this is consistent with, not proof of, outliers
being driven mainly by the data.

### Candidates (fitted on the 300,579-paragraph topic-fitting population only)

Two different families, kept distinct:

- **HDBSCAN on one shared 5-D UMAP representation.** UMAP: cosine metric,
  15 neighbours, 5 components, min_dist 0.0, random_state 42; fitted once and
  shared, so H1 to H3 differ only in clustering. HDBSCAN: Euclidean metric on the
  5-D output, cluster selection "eom", prediction_data enabled.
  - H1: min_cluster_size 50, min_samples 50 (Model B settings)
  - H2: min_cluster_size 100, min_samples 10 (Model A settings)
  - H3: min_cluster_size 100, min_samples 2 (low density threshold)
- **Spherical k-means on the original L2-normalised 384-D MiniLM embeddings**
  (no UMAP; cosine assignment; centroids re-normalised each iteration;
  k-means++ initialisation; seed 42; no outlier topic).
  - K1: K = 100
  - K2: K = 300

Documents of the retrieval corpora and questions receive topics by inference
only: for H1 to H3, UMAP transform followed by HDBSCAN approximate prediction
(the operation BERTopic's `transform` performs); for K1 and K2, the most similar
centroid.

### Strict baseline preserves the original AT-RAG mechanism

As in `models/AT_RAG.py`: the question's topic is the single label returned by
`transform`, and retrieval is restricted to corpus paragraphs carrying the same
label, which are also labelled by `transform`. For HDBSCAN candidates this
includes -1: a question predicted as -1 searches only the paragraphs labelled -1.
No outlier reassignment and no topic reduction. For k-means the label is the
nearest centroid.

### Selection rule (validation fold and validation retrieval corpus only)

Measured for every candidate: strict-filter Recall@5 and AllGold@5; oracle topic
coverage (share of questions whose gold paragraphs all lie in one topic, and
within the best three topics); retrieval cost (mean share of the corpus that
the strict filter makes searchable, and similarity computations). Topic count
and outlier share are reported and never decide the choice.

1. Admissible: the strict filter makes at most 50% of the corpus searchable on
   average (otherwise it is not filtering in any meaningful sense).
2. Among admissible candidates, choose the highest strict AllGold@5.
3. If the best two differ by less than 1 percentage point, use strict Recall@5;
   if still within 1 point, oracle three-topic coverage; then lower cost.

All candidates' results are reported. final_eval questions and the final
retrieval corpus are not used in this selection; final_eval questions are not
embedded or assigned until the final evaluation.

## 12. Topic-model selection, Gate 1 and secondary analysis (11 November 2026)

Fitted by `scripts/fit_tae_r_topic_models.py` (commit `67d0202`; stages run at
`b04f100`) and evaluated by `scripts/evaluate_tae_r_topic_candidates.py`
(commit `b04f100`) on the validation fold (4,576 questions) and the validation
retrieval corpus (56,655 paragraphs) only. No final_eval question and no final
corpus paragraph was read.

### Fitting (topic-fitting population, 300,579 paragraphs)

Shared UMAP: 548.2 s, peak RSS 5,869 MiB.

| Candidate | Topics | Fit outliers | Validation-corpus outliers | Validation-question outliers |
|---|---:|---:|---:|---:|
| H1 | 394 | 45.03% | 47.09% | 58.26% |
| H2 | 315 | 51.38% | 56.07% | 67.61% |
| H3 | 371 | 49.87% | 57.13% | 66.90% |
| K1 | 100 | 0% | 0% | 0% |
| K2 | 300 | 0% | 0% | 0% |

### Validation retrieval (exact cosine search, MiniLM)

Unfiltered: AllGold@5 16.28%, Recall@5 48.54%, AllGold@10 20.41%, Recall@10 52.50%.

| Candidate | Strict AllGold@5 | Strict Recall@5 | Strict eligibility | Strict searched share | Oracle AllGold@5 | Oracle searched share | One-topic coverage | Three-topic coverage | Gold outlier share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 4.70% | 27.95% | 44.95% | 0.283 | 20.39% | 0.348 | 32.28% | 96.77% | 0.456 |
| H2 | 5.22% | 31.01% | 52.97% | 0.383 | 19.93% | 0.468 | 36.54% | 97.64% | 0.590 |
| H3 | 6.03% | 30.74% | 51.32% | 0.387 | 19.84% | 0.472 | 35.36% | 96.88% | 0.573 |
| K1 | 3.12% | 16.44% | 25.44% | 0.019 | 22.47% | 0.038 | 16.22% | 91.67% | 0 |
| K2 | 1.73% | 13.76% | 18.27% | 0.005 | 25.31% | 0.013 | 8.76% | 84.24% | 0 |

### Selection (rule of section 11, applied mechanically)

1. Admissible (strict searched share <= 0.50): all five.
2. Strict AllGold@5: best H3 6.03% (276/4,576); H2 5.22% (239/4,576) is within
   1 point (gap 0.81); H1 (gap 1.33), K1 and K2 drop out.
3. Strict Recall@5: H2 31.01%, H3 30.74%; gap 0.27, tie.
4. Oracle three-topic coverage: H2 97.64%, H3 96.88%; gap 0.76, tie.
5. Lower strict searched share: H2 0.383 < H3 0.387.

**Selected primary topic model: H2.** H3 was numerically higher on strict
AllGold@5 by 37 questions, inside the pre-set 1-point margin; H2 and H3 are
treated as practically equivalent.

### Gate 1 (section 7)

Cross-topic rate with H2 = 1 - one-topic coverage = **63.46%** of validation
questions (label -1 counted as a topic, as in the original mechanism). This is
at least 20%, so **multi-topic routing is a main contribution**. With k-means
(no outlier label) the rate is 83.78% (K1) and 91.24% (K2), so the outcome does
not depend on outlier handling.

### Secondary analysis declared before TAE-R is implemented: K2 topics

Motivation (validation-based, stated openly): the oracle on K2 topics reaches
25.31% AllGold@5 while searching 1.3% of the corpus, versus 19.93% at 46.8% for
H2; the selection rule optimises the strict baseline and does not measure this
headroom.

- Primary results remain H2 for every system and are reported first; K2 never
  replaces them.
- Secondary: TAE-R built with K2 topics, compared with strict-K2, unfiltered,
  TAE V1/TAE V2-style baselines where applicable to K2, and the K2 oracle.
- Identical conditions to the primary comparison: same splits (router_train for
  training, validation for tuning, final_eval once), same retrieval corpora,
  embeddings, exact search, k = 5 and 10, same TAE-R design and training
  procedure (labels recomputed for K2 topics), same probe budget and cost
  measurement, and the same statistics.
- Retrieval-only. End-to-end LLM runs use H2 only.

### Backup record (11 November 2026)

- Selection report `tae_r_topic_selection_validation_v1.json`: 6,270 bytes, SHA-256 `814e2f85c98cfabae123b67ccaa80e3e7e817d03e52af08398e988c278c677fd`
- All fitted artifacts (shared UMAP model 1,627.8 MiB; H1-H3 HDBSCAN models; K1-K2 centroids; validation assignments; question embeddings): private Kaggle dataset `bandlahemanth/tae-r-topics-v1`, uploaded directly from the notebook; all 12 stage manifests and the report re-verified by SHA-256 from the dataset copy.
- Earlier artifacts: `tae-r-frozen-artifacts-v1` (splits, corpora) and `tae-r-embeddings-v1` (embeddings), both verified.

## 13. Amendment v1.3 (12 November 2026, before any controlled end-to-end run)

Reason: the faithful smoke test (`docs/smoke_test_original_at_rag_v1.md`) showed that the released
hallucination and answer grader prompts ask for "a score 'yes' or 'no'" without saying which value means
grounded or useful. With the free substitute LLM the hallucination grader answered "no" in 21 of 22
calls, so every question looped to the iteration cap and LLM-cost comparisons would carry no information.

### Faithful reproduction (unchanged)

Released code and prompts, released default temperature (0.7), LLM substitution only. The grader
behaviour above is reported as a finding.

### Controlled comparison (all three systems identically)

The released prompts are kept verbatim except for one added sentence each, placed immediately before
"Please respond in valid JSON format":

- hallucination grader: "Give 'yes' if the answer is supported by the facts, and 'no' if it is not."
- answer grader: "Give 'yes' if the answer resolves the question, and 'no' if it does not."

The wording is fixed here and will not be revised. Other controlled settings, as in section 5:
per-question state reset; a failed question is recorded as a failure and never inherits another answer;
temperature 0 and seed 42; the LLM `qwen2.5:7b-instruct-q8_0` (Ollama digest `2d9500c948416525`);
cached LLM calls; retrieval grader not used (as in the released graph); k = 5; at most 5 iterations.

A sanity check is run only on exposed questions (never final_eval or validation): grader "yes" rates
with and without the added sentence are reported. It is not used to tune the wording.

### LLM substitution (all settings)

The paper used GPT-4o. All end-to-end results here use the free model above. Absolute EM/F1 are not
comparable with the paper; comparisons between systems are, because every system uses the same model,
prompts and settings.

## 14. TAE-R design, frozen before its first validation run (12 November 2026)

Implementation: `scripts/tae_r_router.py` at commit `5a6cea3`.

- Probe 1 (no corpus scan): cosine of the question to the centroid of every topic of the selected model
  (for HDBSCAN models including a centroid of the -1 paragraphs, all from the topic-fitting population).
  Features: top score, top-minus-second margin, softmax entropy (temperature 0.05), HDBSCAN membership
  strength and outlier flag of the question, agreement of the top centroid with the predicted label.
- Probe 2 (one exact unfiltered scan, top m = 20): similarity-weighted topic mass of the 20 paragraphs.
  Features: share of the top topic and of the top three, distinct topics / 20, share of -1, top similarity,
  similarity gap rank 1 to 5, mean similarity, agreement with Probe 1 and with the predicted label.
- Candidate routes: S1 = the top Probe-2 topic; S3 = the top three; unfiltered.
- Adequacy predictor: two standardised logistic regressions (C = 1), labels "all gold paragraphs lie in
  S1" and "... in S3", trained on router_train only. Thresholds t1, t3 from the grid 0.30-0.95 (step 0.05)
  plus "never", chosen on router_train 5-fold out-of-fold predictions (seed 42) to maximise AllGold@5;
  among configurations within 0.25 points of the best, the lowest searched share is taken.
  Rule: S1 if P(S1 adequate) >= t1; else S3 if P(S3 adequate) >= t3; else unfiltered.
- Ablation TAE-R-P1: Probe 1 features only, routes from the centroid ranking; no corpus scan.
- Cost: TAE-R = one full scan (probe) + centroid scores (final filtered ranking reuses the probe scores);
  TAE-R-P1 = centroid scores + searched paragraphs; unfiltered = one full scan.
- Baselines on the same retrieval engine: strict (original mechanism), TAE V1 (-1 falls back to
  unfiltered), TAE V2 and fixed three (released `scripts/tae_v2_policy.py`, provisional thresholds,
  H2 soft membership), unfiltered, oracle (gold topics; upper bound).
- Gate 2 on the validation fold: TAE-R minus unfiltered AllGold@5, paired bootstrap 95% CI (10,000
  resamples, seed 42), exact McNemar test; decision per section 7.

## 15. TAE-R validation results and Gate 2 (12 November 2026)

Run: `scripts/tae_r_router.py` at commit `b34cdee` (design of section 14), validation fold 4,576
questions, validation retrieval corpus 56,655 paragraphs, MiniLM exact search. Predictor trained and
thresholds chosen on router_train (8,000) only.

### H2 (primary)

| Method | AllGold@5 | Recall@5 | AllGold@10 | Searched share | Similarity computations* | Routes |
|---|---:|---:|---:|---:|---:|---|
| Unfiltered | 16.28% | 48.54% | 20.39% | 1.000 | 56,655 | all unfiltered |
| TAE-R | 16.17% | 48.41% | 20.17% | 0.922 | 56,971 | S1 53, S3 769, unfiltered 3,754 |
| TAE-R-P1 | 16.13% | 48.04% | 20.19% | 0.975 | 55,535 | S1 212, S3 31, unfiltered 4,333 |
| TAE V2 | 13.51% | 43.46% | 16.98% | 0.804 | 45,524 | |
| TAE V1 | 11.10% | 39.78% | 13.96% | 0.681 | 38,600 | |
| Fixed three | 6.38% | 32.64% | 7.93% | 0.386 | 21,861 | |
| Strict (original) | 5.22% | 31.01% | 6.53% | 0.383 | 21,686 | |
| Oracle (upper bound) | 19.93% | 51.67% | 24.83% | 0.468 | 26,517 | |

Thresholds: TAE-R t1 = 0.75, t3 = 0.80 (router_train base rates: S1 adequate 32.95%, S3 66.58%);
TAE-R-P1 t1 = 0.40, t3 = 0.35.

\* Correction: the first report counted TAE-R's searched paragraphs on top of its probe scan
(109,205). Per the frozen design the filtered ranking reuses the probe scores, so the cost is one scan
plus 316 centroid scores = 56,971. Fixed in the script and covered by a test; retrieval results are
unaffected.

Gate 2 (AllGold@5 minus unfiltered, paired bootstrap 95% CI, exact McNemar):

| Method | Difference | 95% CI | Helps / hurts | p | Decision |
|---|---:|---|---|---:|---|
| TAE-R | -0.11 pt | [-0.22, 0.00] | 1 / 6 | 0.125 | **equivalent** |
| TAE-R-P1 | -0.15 pt | [-0.28, -0.02] | 1 / 8 | 0.039 | equivalent (within the 1-point margin) |
| TAE V2 | -2.78 pt | [-3.26, -2.29] | 2 / 129 | <0.001 | worse |
| Strict | -11.06 pt | [-12.02, -10.10] | 32 / 538 | <0.001 | worse |
| Oracle | +3.65 pt | [+3.10, +4.20] | 167 / 0 | <0.001 | better |

**Gate 2 outcome: TAE-R is equivalent to unfiltered retrieval.** No efficiency claim (its cost is one
full scan plus centroid scores, about 0.6% above unfiltered; TAE-R-P1 saves about 2% at -0.15 points).

### K2 (secondary, same conditions)

Unfiltered 16.28%; TAE-R 16.06% (-0.22 pt, CI [-0.37, -0.09], 1 / 11, equivalent); TAE-R-P1 16.02%
(equivalent); strict 1.73% (-14.55 pt, worse); TAE V1 = strict for k-means; oracle 25.31% (+9.03 pt,
searched share 0.013). TAE-R thresholds 0.65 / 0.65; routes S1 19, S3 317, unfiltered 4,240.

### Interpretation recorded before the final run

The strict single-topic filter loses most complete-evidence questions; TAE-R recovers that loss and
matches unfiltered search, mainly by declining to filter (82% unfiltered on H2). The oracle shows a
filtering gain of +3.65 (H2) and +9.03 (K2) points that the probe features do not predict reliably;
this gap is reported as the main open problem.

## 16. Pre-declared LLM choice test (12 November 2026, before any controlled end-to-end run)

The fitted TAE-R predictors were saved (`--save-model`, commit `caf5d90`); rerunning reproduced the
section 15 per-question results and thresholds exactly for H2 and K2.

Because the 7B substitute grades poorly (section 13 sanity check), one larger free model is tested on
the same 20 exposed questions, original code and store, temperature 0, clarified graders:
`qwen2.5:14b-instruct-q4_K_M` (Ollama). Decision rule, fixed before the test and judged only on
these exposed questions (answer accuracy is not a criterion):

- switch the controlled comparison's LLM to the 14B model if its mean iterations are below 3.5 **and**
  its mean seconds per question are at most 30;
- otherwise keep `qwen2.5:7b-instruct-q8_0`.

Reference (7B, same setting): mean iterations 4.40, 14.1 s per question.

### Results backup (12 November 2026, private Kaggle dataset `bandlahemanth/tae-r-results-v1`)

| File | SHA-256 |
|---|---|
| tae_r_validation_H2_v1.json | `e65a97f7de9cff8b74abfccd7b3a5900a9858b4abf5455fcdb856dee0bec552c` |
| tae_r_validation_H2_v1.questions.jsonl | `039f7e7d3b0c49476496e8a7ef07270d076b4314666640a1ae91a88fac1d1a34` |
| tae_r_validation_K2_v1.json | `9e33d7c8cc19050c00f5d6785c3d0792e6755004f2807ebb98fb4ba12831ad00` |
| tae_r_validation_K2_v1.questions.jsonl | `89d94f9e603b89f62eae563e3913e7b221936a88c67069836a682966bd190f6a` |
| tae_r_router_H2_v1.joblib (saved predictor) | `724c287c519027b65fff0cc1a2161c388e9b38e3ee2dd171420dd5fa9f7cfcf3` |
| tae_r_router_K2_v1.joblib (saved predictor) | `37c30a3ec54b3919a53f8f9f861160adf095d5589cec10539b2eb183c90d0093` |
| tae_r_validation_H2_rerun.json | `4b72924fe4fab7ed79d8ce86deae04fd31a0fc4f04c1eae1c5b0b3266e71c4b7` |
| tae_r_validation_K2_rerun.json | `9bbcfe1cb69984b08d1a08b64ad3fd288fd98a2b7588d740c17462c6b0af008f` |
| smoke_original_at_rag_v1.json | `9633971db176d6e8d083b7c768eeb564c6e188dcc68cdb6b9613c305d97191b7` |
| smoke_original_at_rag_trace_v1.json | `8e0772df8af5396d251fdaacd0dbfaa3eb0491d94506a87c8f7366898aa573d3` |
| grader_sanity_released_t0_v1.json | `ff710323b4c25ba39fa40c6702de8ec4fb71d8f334fa05eb6076c026503c8783` |
| grader_sanity_clarified_t0_v1.json | `8775b7f4159070adb02783852f36e1bbdd44541ce6834960245af84143893ec0` |

### LLM choice test result (13 November 2026)

Same setting as the 7B sanity check (original code and store, 20 exposed questions, temperature 0,
clarified graders), `qwen2.5:14b-instruct-q4_K_M` (digest `7cdf5a0187d5c58c`):
EM 0.25, F1 0.29, mean iterations 4.90, mean LLM calls 11.6, **109.9 s per question**, 0 errors;
hallucination grader "yes" 20 / 84, answer grader "yes" 1 / 15.

Rule outcome: iterations 4.90 is not below 3.5 and 109.9 s exceeds 30 s, so **the controlled comparison
uses `qwen2.5:7b-instruct-q8_0`** (digest `2d9500c948416525`).

Note: the original-runtime setup was rebuilt on 13 November with `scripts/setup_kaggle_session.sh`
(commit `a5e0171`), which installs the full training freeze except the CUDA wheels, torch (installed as
2.14.1+cu130) and packaging; pynndescent 0.6.0 as frozen.

## 17. Controlled pipeline dry run and frozen final-evaluation plan (13 November 2026)

### Dry run (validation, pipeline check only)

`scripts/controlled_at_rag.py` (commit `957c7b6`; prompts read from the released source and verified equal
to LangChain 0.3 rendering and parsing), first 20 validation questions x 3 systems, Qwen-7B, temperature 0:
60 runs in 370 s with 4 workers; 1 parse error (original). Exact match original 0.10, no filter 0.35,
TAE-R 0.30 (20 questions; not a result). First-retrieval AllGold@5 was 0 for all systems; the validation
retrieval run gives the same 0 complete-evidence questions and the same 23 gold hits on these 20 questions,
so the pipeline's retrieval reproduces the frozen evaluation exactly.

### Final evaluation (run once, in this order; nothing is changed afterwards)

Code frozen at commit `d72ce5e`.

1. `fit_tae_r_topic_models.py --stages assign_final --allow-final --source-dir <tae-r-topics-v1>` for H2 and K2:
   label the 56,680 final-corpus paragraphs, embed and label the 11,976 final_eval questions.
2. `tae_r_router.py --final --allow-final` with the saved predictors (H2 primary, K2 secondary): every method
   on all 11,976 final_eval questions; AllGold@5/10, Recall@5/10, eligibility, cost; paired bootstrap 95% CI
   and McNemar against unfiltered and against strict; by question type where available.
3. `controlled_at_rag.py --split final_eval --allow-final --question-ids e2e_500_ids.txt`, systems original,
   nofilter, tae_r (H2). Primary end-to-end analysis: all 500 questions if they complete; if the runs cannot
   finish, the nested e2e_300 is reported instead (declared here, before any run). Measures: EM, F1,
   errors, first-retrieval AllGold@5, iterations, LLM calls, similarity computations, seconds; paired
   bootstrap CIs on EM and F1.
4. Back up every output with SHA-256 before analysis.

## 18. Final retrieval results (13 November 2026, run once at commit `857295b`)

The session was restarted by mistake before the first final run printed anything; no output was seen. The
run below is the first and only one whose results were viewed (deterministic code and inputs).

Final assignment: 11,976 final_eval questions; 56,680 final-corpus paragraphs. H2: corpus outliers 55.66%,
question outliers 66.51%. K2: no outliers.

### H2 (primary), 11,976 questions, saved predictor (no retraining)

| Method | AllGold@5 | Recall@5 | AllGold@10 | Searched share | Similarity computations |
|---|---:|---:|---:|---:|---:|
| Strict (original AT-RAG) | 7.08% | 30.83% | 9.74% | 0.373 | 21,134 |
| Fixed three | 8.07% | 32.61% | 11.12% | 0.377 | 21,355 |
| TAE V1 | 12.50% | 40.63% | 17.13% | 0.668 | 37,848 |
| TAE V2 | 14.21% | 44.01% | 19.36% | 0.788 | 44,645 |
| **TAE-R** | **16.52%** | **48.72%** | **22.44%** | 0.868 | 56,996 |
| TAE-R-P1 | 16.38% | 48.32% | 22.34% | 0.975 | 55,560 |
| Unfiltered | 16.65% | 48.88% | 22.68% | 1.000 | 56,680 |
| Oracle (upper bound) | 20.87% | 52.24% | 27.05% | 0.453 | 25,652 |

Comparisons on AllGold@5 (paired bootstrap 95% CI, exact McNemar):

| Comparison | Difference | 95% CI | Helps / hurts | p | Decision (section 7 rule) |
|---|---:|---|---|---:|---|
| TAE-R vs strict | +9.44 pt | [+8.87, +10.03] | 1,233 / 102 | <1e-200 | better |
| TAE-R vs unfiltered | -0.13 pt | [-0.23, -0.03] | 10 / 25 | 0.017 | equivalent (within +/-1 pt) |
| TAE-R-P1 vs unfiltered | -0.27 pt | [-0.37, -0.18] | 2 / 34 | <1e-7 | equivalent |
| TAE V2 vs strict | +7.13 pt | [+6.60, +7.66] | 956 / 102 | <1e-170 | better |
| Strict vs unfiltered | -9.57 pt | [-10.15, -8.98] | 111 / 1,257 | <1e-200 | worse |
| Oracle vs unfiltered | +4.22 pt | [+3.86, +4.58] | 505 / 0 | <1e-150 | better |

The TAE-R deficit against unfiltered search is statistically detectable at this sample size (p = 0.017) but
negligible in size (0.13 points) and inside the pre-declared equivalence margin.

### K2 (secondary)

Strict 3.38%; TAE-R 16.44% (vs strict +13.06 pt [+12.42, +13.70], 1,614 / 50; vs unfiltered -0.21 pt
[-0.32, -0.10], 9 / 34, equivalent); TAE-R-P1 16.51% (equivalent); unfiltered 16.65%; oracle 26.77% at
searched share 0.016 (+10.12 pt over unfiltered).

