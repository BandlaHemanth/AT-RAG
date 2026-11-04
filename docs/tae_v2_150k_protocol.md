# TAE V2: 150,000-occurrence corpus, Step 1

Branch: `thesis/tae-v2-150k`, based on pilot commit
`e6761b56057706d2e4f0c844aa1b25d7d30a7a04`. This step prepares data and freezes
evaluation rules. It does not fit BERTopic, embed paragraphs, build a vector
database, implement TAE V2, or run retrieval/answer experiments.

## Inspection and provenance

Inspected `processing_scripts/process_2wikimultihopqa.py`,
`subsample_dataset_and_remap_paras.py`, `lib.py`, `scripts/build_full_2wiki_corpus.py`,
`run_bertopic_pilot.py`, `analyze_topic_filter_retention.py`, `evaluate_tae_v1.py`,
`vectorDB/{train_topic_model,dataset_ingestion}.py`, and `evaluation/evaluate.py`.

The processor joins source sentences into `paragraph_text` and stores question
IDs, local `idx`, title, support flags, answers, and reasoning. Preserve the
processed paragraph string exactly; this step does not reconstruct raw sentences.
The pilot selected the first 500 unique training texts. Its evaluation candidates
included development contexts, so its favorable evidence availability cannot be
assumed for a training-only sampled corpus. The full-corpus builder deduplicates
texts; this occurrence sampler deliberately retains natural repeats.

The repository subsampler uses seed **13370**, reads **dev.jsonl** for both
`dev_subsampled.jsonl` and `test_subsampled.jsonl`, and attempts to exclude dev IDs
when creating test. It can remap paragraph text/title through the retriever.
Its default sample count is 500, but actual input counts must be measured, not
assumed; earlier pilot work used a 100-question dev file. **test_subsampled comes
from the original development split and is not the official test set.**

The implementation session had no processed JSONL files. Actual file counts,
hashes, disjointness and source membership are therefore **pending the Kaggle
run**, not claimed as verified. The script requires all four files and fails
unless both evaluation files are question-ID disjoint, disjoint from the entire
training split, and subsets of processed original dev with identical question
text. It also checks full train versus original dev disjointness. Context
differences caused by remapping are counted in the audit. This verifies provenance
relative to the supplied processed sources, not an external official checksum.

## Selection and validation

Use every eligible nonempty paragraph occurrence in `train.jsonl` as the sampling
population. Algorithm R reservoir sampling with Python `random.Random(42)` gives
each occurrence probability `150000/N`, without replacement. Read the entire
file; retain only selected eligible ordinal integers in memory. Output order is
source order, independent of the uniform inclusion decision. Freeze Python/RNG
version and input hashes when reproducing selection.

Null, empty, and whitespace-only paragraph strings are counted and excluded.
Missing paragraph fields, wrong field types, malformed UTF-8/JSON, duplicate JSON
keys, duplicate question IDs, or insufficient eligible occurrences abort the run.
`idx` must be a nonnegative integer and title a string. No nonempty paragraph is
normalized, shortened, deduplicated, filtered using labels, or rewritten.

Each corpus row contains `paragraph_text`, `question_id`, `source_split=train`,
1-based `source_line`, 0-based `context_position`, original `idx`, original title,
`text_sha256`, and `occurrence_id`. Occurrence IDs combine dataset, split, source
SHA-256, physical line and position, distinguishing repeats and local idx values.
`text_sha256` is shared by repeated texts and is never a unique occurrence ID.
Only **paragraph_text** is an allowed future fitting input. Metadata, question
text, labels, answers, and reasoning are not fitting inputs.

Three full parsing passes select ordinals, write the corpus, and independently
compare saved rows against their exact source occurrences. A final byte hash
checks every input for mutation. SQLite bounds duplicate/provenance index memory;
temporary indexes are removed. The report validates count=150000, valid text,
unique occurrence IDs, source integrity, unique texts, excess repeats (`N-U`),
duplicated-text groups, occurrences in those groups and maximum multiplicity.

## Frozen evaluation rules

The generated protocol freezes all actual question IDs in their file order and
hashes all four inputs and the fixed corpus. Use `dev_subsampled` for calibration
and `test_subsampled` for final local evaluation after method settings are frozen.
Dev questions 1–20 and 21–100 already informed exploration and TAE V1 evaluation.
Do not claim a historically untouched or official test set. Do not append
evaluation contexts to the corpus or replenish it based on coverage.

For question `q`, let `Gq` be its gold supporting **paragraph occurrences**, `Aq`
the gold occurrences with exact UTF-8 text present anywhere in the fixed corpus,
and `Rqk` those matched in the first `k` retrieved corpus occurrences (`k=5,10`).
Gold flags are scoring-only. Verify text bytes when matching hashes. No fuzzy,
title-only, or idx-only match is accepted. Repeated gold annotations each count;
one matching retrieved text can satisfy repeated annotations. Duplicate corpus
texts consume ranked slots; do not refill top-k after deduplication.

1. **Overall gold evidence coverage:** report `sum|Aq| / sum|Gq|`, macro per-question
   coverage, and all/some/zero-covered question counts, separately for each split.
2. **Retrieval conditional on availability:** report `sum|Rqk| / sum|Aq|`, macro
   recall over questions with `|Aq|>0`, and complete-evidence success over questions
   with `Aq=Gq`. Print every numerator, denominator, and subset size.

Main results retain **every frozen evaluation question**, including uncovered
ones: unconditional Recall@k is `sum|Rqk| / sum|Gq|`, complete-evidence success is
`count(Rqk=Gq)/N`, and answer metrics retain all N questions. Conditional metrics
are secondary. Zero denominators produce null/undefined with count=0. Missing or
nonboolean supporting flags and questions with zero gold are scoring validation
errors. Report corpus coverage, filter eligibility, ranked retrieval and answer
quality separately. These paragraph metrics are not official sentence-support
scores. Step 1 defines these rules; it computes no coverage/retrieval scores.

## Kaggle execution and outputs

Check out this branch at `/kaggle/working/AT-RAG` and provide the four processed
files under `processed_data/2wikimultihopqa/`. No packages or GPU are needed.

```bash
python -B /kaggle/working/AT-RAG/scripts/prepare_150k_corpus.py --data-dir /kaggle/working/AT-RAG/processed_data/2wikimultihopqa --output-dir /kaggle/working/AT-RAG/artifacts/tae_v2_150k_corpus/run_001
```

Expected new outputs in that run directory:

- `corpus.jsonl`: exactly 150000 occurrence rows; no gold fields.
- `validation_report.json`: successful checks, source/duplicate counts, input and
  output hashes, seed, selection hash, script/Git/Python/RNG provenance.
- `frozen_evaluation_protocol.json`: actual ordered evaluation IDs, file hashes,
  split provenance/disjointness audit, and evidence reporting rules.
- `SHA256SUMS`: SHA-256 for the other three outputs; no circular self-hash.

Use `sha256sum -c SHA256SUMS` from the run directory to verify the saved files.
For reproduction choose a fresh directory, such as `run_002`, and compare corpus
and selected-ordinal hashes. Existing directories (even empty), files, symlinks,
and concurrent output reservations are refused. Checks finish in temporary
storage before an exclusive output mkdir; the report is published last as the
success marker. A crash during publication may leave a partial directory that
must be preserved and investigated; rerun only into a new directory.

All production outputs must remain under `artifacts/tae_v2_150k_corpus/`, already
ignored by the inherited `.gitignore`. Existing models, backups, pilot outputs
and earlier results are never opened for writing. Commit only the new script,
synthetic tests and this document. Run the lightweight checks with:

```bash
python -B -m unittest discover -s tests -p 'test_prepare_150k_corpus.py' -v
```
