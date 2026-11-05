# Revised corpus design — audit protocol v1 (Step 1 only)

This is a separate, versioned design audit for `thesis/tae-v2-150k`.
It does not amend `docs/tae_v2_150k_protocol.md` or its saved frozen protocol.
The old experiment remains reproducible under its original rules. No corpus,
question subset, topic model, embedding, retrieval policy, or final experimental
protocol is generated or changed here. The proposed context-inclusive corpus
would be a new experiment with a new frozen manifest and protocol in a later step.

## Findings from implementation inspection

- `processing_scripts/process_2wikimultihopqa.py` joins source sentences with
  spaces and strips the resulting paragraph. It sets support flags by supporting
  title membership. The audit preserves the **processed** paragraph bytes; it
  cannot verify the correctness of these flags against absent raw annotations.
- `subsample_dataset_and_remap_paras.py` reads original `dev.jsonl` for both
  evaluation files, seeds sampling with 13370, and attempts ID exclusion for the
  local held-out subset. `dev_diff_size` skips remapping. Other modes can replace
  titles/text through `lib.find_matching_paragraph_text`, an Elasticsearch lookup
  accepted when fuzzy partial ratio exceeds 95. This is why original-dev and
  processed-subset contexts must both be counted. The audit does not invoke it.
- `build_full_2wiki_corpus.py` deduplicates exact training texts;
  `prepare_150k_corpus.py` instead uniformly samples **occurrences**, preserves
  repetitions, checks split membership, and freezes ordered evaluation IDs.
  Its constants or historical reports are not substitutes for measuring files.
- `train_bertopic_150k.py` fits all 150k occurrences, including repeated texts,
  with pinned MiniLM embeddings, long-text token window pooling, seeded
  UMAP/HDBSCAN and min_topic_size=50. A fresh unique-text fitting population will
  change frequency weights and clustering. Reusing old assignments would not
  establish a comparable new model. Retraining may be appropriate later.
- `shared_150k_retrieval.py` validates a fixed corpus hash, aligned embeddings and
  assignments and ranks occurrences by cosine similarity with deterministic row
  ties. Repeats consume top-k slots. `query_inference_150k.py` checks the matching
  model/encoding policy; `tae_v2_policy.py` uses inference signals, not support
  flags. A new unique-text corpus needs its own compatible retrieval artifacts
  and ranking/duplicate rules, rather than relaxing those validators.
- Legacy `vectorDB/train_topic_model.py` and `dataset_ingestion.py` default to
  `test_subsampled`; the former fits its contexts and the latter stores support
  metadata. They must not be used as the revised training path. This audit
  imports no legacy data/model/retrieval implementation and makes no network call.

`test_subsampled.jsonl` is a local holdout derived from development, not the
official 2Wiki test set. The prior protocol documents historical development
exposure; a new experiment cannot erase it or establish an untouched test claim.
The audit records SHA-256 hashes of the inspected implementations and each
supplied dataset file, including blank lines.

## Running in Kaggle

Use Python 3.9+; only the standard library is needed. No GPU, model download,
retriever service, pip install, or dataset regeneration is required. Attach the
actual processed dataset as read-only Kaggle input, or use the existing processed
directory. Do not run the subsampler again: that would change the audited inputs.

```bash
cd /kaggle/working/AT-RAG
python -B -m unittest discover -s tests -p 'test_audit_2wiki_corpus_design.py' -v
mkdir -p /kaggle/working/corpus_design_audit_v1
python -B scripts/audit_2wiki_corpus_design.py \
  --data-dir /kaggle/input/YOUR-DATASET/processed_data/2wikimultihopqa \
  --scratch-dir /kaggle/working \
  --output /kaggle/working/corpus_design_audit_v1/audit_run_001.json
sha256sum /kaggle/working/corpus_design_audit_v1/audit_run_001.json
```

The four filenames default to `train.jsonl`, `dev.jsonl`,
`dev_subsampled.jsonl`, `test_subsampled.jsonl`. Explicit alternatives are
`--train-name`, `--original-dev-name`, `--development-name`, and `--heldout-name`;
each must be a distinct plain filename under `--data-dir`. For example use
`--development-name dev_100_subsampled.jsonl` only if that is the existing frozen
development file. Do not choose a different question sample based on coverage.

Use a fresh output filename on every run. Existing files and symlinks are refused;
output and scratch must be outside the dataset directory. Output parents must
already exist. Only the report is published, with exclusive creation. SQLite is
temporary and deleted even after validation failure. SQLite stores exact bytes
on disk and keeps its page cache at 8 MiB; Python holds one question plus small
aggregate distributions. Allow several GB of scratch disk for a full split;
runtime and disk use depend on paragraph sizes and storage. No input, backup,
model, corpus, result or old protocol is opened for writing.

Exit status: **0** = all four files parsed and audited; **2** = one or more files
missing, with an explicitly incomplete report; **1** = invalid input, changed
source bytes, output conflict or I/O failure, with no report published. Status 0
does not certify scientific readiness: inspect `design_validation` and the
annotation diagnostics. Empty files, duplicate IDs, overlapping question sets,
and context differences are findings to report, not silently repaired or removed.
Malformed records/UTF-8, duplicate JSON keys, nonfinite constants, missing text
fields, invalid text/idx/title types, and escaped unpaired surrogates fail.
Keep inputs immutable for the run; a second full byte-hash pass detects changes.

## JSON definitions and interpretation

`splits` reports physical question records, distinct IDs, repeated-ID groups and
excess records, context occurrences, nonempty occurrences, empty occurrences,
unique nonempty texts, excess duplicate occurrences (N-U), duplicated-text groups,
occurrences in those groups, maximum multiplicity and the full multiplicity
histogram. Context-count distributions include empty contexts; a second histogram
counts only nonempty texts. Null, empty and whitespace-only text are excluded
from corpus budgets. Nonempty strings are never stripped, case-folded or Unicode
normalized: `A`, ` A`, and alternative Unicode encodings are distinct texts.

`pairwise_overlap` covers every pair of files: shared unique texts, occurrences
of those shared texts on each side, and shared distinct question IDs. Subsets
overlap original-dev by design; overlap of train with dev/evaluation or development
with heldout is a structural blocker. Duplicate question records remain in
occurrence/distribution counts while distinct-ID counts remain distinct.

`context_remapping` compares unique matching IDs with original dev, preserving
positional order. It counts missing IDs, duplicate-ID ambiguities (excluded from
comparison), question-text changes, changed context counts, changed common
positions, changed text/title/idx, and added/removed positions. An order change
therefore appears as positional changes. This identifies observed differences;
it cannot prove which historical process caused them. It reports no IDs or
question-level findings, including for heldout.

Gold diagnostics have **no role in budget feasibility, corpus selection, question
selection, fitting, or retrieval policy optimization**:

- `own_annotation_diagnostics` counts boolean support annotations and how many
  refer to nonempty text in that record's own contexts, plus empty support texts,
  missing/nonboolean flags and questions with no annotated support. Since flags
  are attached to contexts, nonempty availability is partly tautological; this
  is not independent validation of support correctness or retrieval success.
- The remapping diagnostic also counts original-dev annotated support occurrences
  and how many nonempty original supporting texts appear anywhere in the matched
  subset question's own contexts, by exact text equality. This can expose loss
  due to remapping. Repeated gold occurrences each count; one present text can
  satisfy multiple repeats. Missing/ambiguous IDs are excluded and separately
  counted. Invalid source flags are reported in original-dev diagnostics; these
  totals are diagnostic counts, not complete gold denominators when flags are
  invalid. No gold rates or question outcomes are emitted. No candidate-corpus
  coverage or ranked-retrieval diagnostic is run.

`budget_feasibility` reports full original-dev fit at 100k, 150k and 200k and two
label-blind mandatory sets: (a) both processed evaluation context collections,
(b) all original-dev contexts plus both processed evaluation collections. The
latter includes remapped texts absent from original dev. For each, let M be the
mandatory unique-text count and T be unique train texts outside the mandatory
set. A budget B admits all mandatory texts if M <= B; it admits at least one
training background text if M < B and T > 0. An exactly B-sized corpus is
possible iff M <= B <= M+T. A budget is otherwise an upper bound, not a requirement
to invent padding. `background_slots` is max(0,B-M); `mandatory_fits` must be
checked first. These are arithmetic feasibility checks, not construction.

`fresh_topic_fitting` gives train-text counts after excluding (a) both processed
evaluation collections, and (b) their union with all original dev. Use the second,
more conservative population for a revised experiment that includes original-dev
contexts. Train/dev textual overlap is possible even when question IDs are
disjoint. Fitting must exclude those texts; metadata, titles, question strings,
support flags, answers and reasoning are not fitting inputs. Evaluation contexts
may later be **transformed** by the train-fitted model, without refitting any
encoder, UMAP, HDBSCAN, vocabulary or c-TF-IDF component on them.

Missing split counts, budgets and fitting counts are null/unknown, never zero
estimates. `docs/corpus_design_audit_v1.cloud_status.json` records this session's
actual absence of the required dataset files. It contains code hashes, not a
fabricated full-dataset result. Kaggle must produce the authoritative counts.
File hashes identify supplied processed files; without official checksums they
do not establish that a file named train/dev is the complete official split.
Preserve and document the original data source and extraction provenance.

## Criteria for choosing a budget in the next design step

1. Resolve empty splits, duplicate IDs, split overlap and source/query identity
   blockers. Freeze every existing evaluation question without conditioning on
   labels, context gold availability or retrieval success. Review context
   remapping structurally and freeze which context representations define the
   revised corpus. Preserve the original files regardless of the decision.
2. Prefer a fixed mandatory set containing **all original development contexts
   plus all processed evaluation contexts**, if feasible. This uses question
   contexts without evidence selection and reduces dependence on a particular
   evaluation sample. Declare the resulting context-inclusive retrieval setting;
   its results will not be directly comparable to the old train-only corpus.
3. Predeclare a minimum training-background size or fraction based on scientific
   scope and resource limits, before examining diagnostic gold counts. Choose
   the smallest of 100k/150k/200k meeting M plus that background allowance and
   independently established RAM/disk/latency constraints. If none fit, report
   infeasibility and revise the budget/design; never trim mandatory contexts or
   evaluation questions by support labels or retrieval success. No numeric budget
   is recommended until actual counts and the background requirement are known.
4. For a later builder, sample **unique** train texts outside the mandatory set
   with a predeclared deterministic label-blind seed/hash rule. Selection must
   not consult answers, supporting evidence, topic coverage or retrieval scores.
   Fix identical corpus bytes for every comparison method. A fitting population
   can be larger than retrieval background, but its size, exclusions, seed and
   hashes must be specified separately and use training text only.
5. Freeze a new versioned experimental protocol, data/corpus/fitting manifests,
   encoder policy, duplicate handling, model parameters and resource plan before
   generation/training in a separately authorized step. Use entirely new artifact
   paths; retain the old 150k model, corpus, backups, frozen protocol and results.
   Freeze retrieval policy using development only; do not use these held-out
   aggregate diagnostics to optimize routing or thresholds.

**Stop after this audit.** No corpus generation or BERTopic retraining is part
of this step.
