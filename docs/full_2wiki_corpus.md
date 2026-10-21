# Full 2Wiki training corpus preparation

This phase only constructs a paragraph corpus from the complete processed training
split. It does not train models, embed documents, or run retrieval/evaluation.
The builder uses only `question_id` and context `paragraph_text`, `idx`, and `title`.
It does not select paragraphs using supporting-fact labels, answers, or reasoning
annotations. Titles are metadata, not prepended to paragraph text.

## Run on Kaggle (Python 3.13)

With the repository checked out at `/kaggle/working/AT-RAG` on
`thesis/topic-filter-pilot` and the complete processed training file at
`/kaggle/working/AT-RAG/processed_data/2wikimultihopqa/train.jsonl`, run:

```bash
python -B /kaggle/working/AT-RAG/scripts/build_full_2wiki_corpus.py --input /kaggle/working/AT-RAG/processed_data/2wikimultihopqa/train.jsonl --output-dir /kaggle/working/AT-RAG/artifacts/full_2wiki_corpus
```

In a Kaggle notebook code cell, prefix that command with `!`. If the input is
attached as a read-only Kaggle dataset, replace only `--input` with its actual
`/kaggle/input/.../processed_data/2wikimultihopqa/train.jsonl` path. Keep the output
under writable `/kaggle/working`. No pip installs, credentials, network access,
GPU, or external models are required; all imports are Python standard library.

From the repository root, the equivalent command is:

```bash
python -B scripts/build_full_2wiki_corpus.py
```

The CLI reads to EOF and then requires these previously measured counts:

| Metric | Expected |
| --- | ---: |
| Questions (nonblank JSONL records) | 167,454 |
| Context occurrences (all context entries) | 1,674,540 |
| Unique nonempty paragraphs | 396,978 |
| Duplicate nonempty occurrences | 1,277,562 |
| Empty contexts | 0 |

These values are validation targets, never processing limits. A mismatch exits
with status 1 and publishes no artifact directory. For deliberately synthetic or
alternate inputs, use `--skip-count-validation`; the summary explicitly records
that count validation was not requested. Do not use this flag for the thesis
full-dataset run. Failures and attempts to reuse an output directory exit with 1;
successful builds exit with 0.

## Artifact contract

- `documents.jsonl`: one unique paragraph per line, in first-occurrence order.
  Fields: `document_id` (64 lowercase SHA-256 hex characters of the exact UTF-8
  paragraph text), `paragraph_text`, `question_id`, `source_line` (physical line,
  one-based), `context_position` (zero-based), `idx`, and `title`.
- `summary.json`: all five measured counts, validation status/targets, source
  byte count and SHA-256, documents SHA-256, and schema/identity conventions.

Missing optional metadata is retained as JSON null. Paragraphs that are missing,
null, empty, or whitespace-only count as empty; other non-string paragraph values
are errors. Blank input lines are skipped but count toward physical source line
numbers. Malformed JSON or invalid context structures abort the build. Paragraphs
are never stripped, normalized, truncated, or sorted. Leading/trailing whitespace,
case differences, and Unicode normalization differences remain distinct.

Only one question is parsed at a time. A temporary disk-backed SQLite index stores
hashes and exact paragraph bytes, with an 8 MiB SQLite page cache. A hash collision
between distinct paragraphs aborts rather than merging them. Disk use scales with
unique text: allow space for both the output JSONL and SQLite index/journal.
Peak Python memory also depends on the largest single input record.

Outputs use UTF-8, sorted JSON keys, fixed compact separators, and LF endings;
there are no timestamps or machine-specific paths. Identical input bytes and
validation settings yield identical artifact bytes. The source checksum covers
the entire input file for provenance, including ignored annotation fields;
those fields do not influence document selection, IDs, text, or metadata.

Both files are flushed and fsynced in a temporary sibling directory before an
atomic directory rename publishes them together. Any existing destination is
refused, including empty directories and symlinks. Concurrent completed outputs
are protected by the nonempty-directory rename rule. Ordinary failures clean up
staging; forced termination may leave a hidden `.full_2wiki_corpus.staging-*`
directory that can be removed after confirming its process is no longer active.
Choose a new output directory for a new build; there is no overwrite flag.

Generated files under `artifacts/` and data under `processed_data/` are already
Git-ignored. Export the completed directory separately; do not commit it.

## Synthetic validation

From the repository root:

```bash
python -B -m unittest discover -s tests -p 'test_build_full_2wiki_corpus.py' -v
```
