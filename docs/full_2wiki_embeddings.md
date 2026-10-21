# Full 2Wiki paragraph embeddings (TAE V2 preparation)

This phase encodes the verified training corpus only. It does not train BERTopic,
fit a topic model, construct a retrieval index, or change the existing AT-RAG
pipeline, corpus builder, BERTopic pilot, or TAE V1 evaluator.

## Fixed experiment conventions

| Setting | Value |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Corpus documents | 396,978 |
| Corpus SHA-256 | `f1824d1dc9275731243c6f5a89cc4d702c77b3607413343c9c75bb242fff3160` |
| Dimensions / dtype | 384 / little-endian float32 |
| Normalization | `normalize_embeddings=True` |
| Maximum sequence length | 256 tokens, as in the pilot |
| Input | Exact `paragraph_text`; no stripping, title prefix, or annotations |
| Seed / PyTorch CPU threads / interop threads | 42 / 2 / 1 |

The original text remains intact on disk; the pinned encoder's tokenizer applies
the same 256-token limit as the pilot. Corpus titles, question metadata, supporting
labels, answers, and reasoning are never sent to the encoder. The script uses
`SentenceTransformer.encode()` with float32 precision and no mixed precision.
Deterministic PyTorch algorithms are enabled and TF32 is disabled. Unsupported
deterministic CUDA operations fail explicitly; there is no device/model fallback.

## Kaggle commands

Use a Linux Kaggle notebook with Python 3.13, the repository checked out at
`/kaggle/working/AT-RAG`, and this implementation available from its PR branch
(`codex/embed-full-2wiki-corpus`) or the thesis branch after a separate merge.
Enable Internet for the first download of the pinned Hugging Face snapshot;
subsequent runs can reuse the Hugging Face cache. Do not download a different
model revision if the pinned snapshot cannot be reached.

NumPy, PyTorch, and sentence-transformers are required for a real run. Use a
PyTorch installation matching the selected Kaggle accelerator. If
sentence-transformers is absent, the pilot's version can be installed with:

```bash
python -m pip install 'sentence-transformers==3.1.1'
```

Keep the same installed library versions when resuming. The script records key
versions automatically. Do not install the repository's full legacy requirements
just for this preparation phase. Real model loading on Kaggle must still be
validated; automated tests use a mock encoder and require only NumPy.

GPU run (enable a GPU accelerator first):

```bash
python -B /kaggle/working/AT-RAG/scripts/embed_full_2wiki_corpus.py \
  --input /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-corpus-zip/documents.jsonl \
  --output-dir /kaggle/working/at-rag-full-2wiki-embeddings \
  --device cuda:0 --batch-size 64 --checkpoint-batches 16
```

CPU run with the pilot's batch size (choose a distinct output directory):

```bash
python -B /kaggle/working/AT-RAG/scripts/embed_full_2wiki_corpus.py \
  --input /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-corpus-zip/documents.jsonl \
  --output-dir /kaggle/working/at-rag-full-2wiki-embeddings-cpu \
  --device cpu --batch-size 4 --checkpoint-batches 16
```

In notebook cells, use `!python ...` or a `%%bash` cell for these shell commands.
`summary.json` is required beside the input; `--summary` can select another path.
Expected corpus count and checksum default to the verified values above. The
`--expected-count` and `--expected-sha256` overrides are for deliberate synthetic
or alternate datasets; there is no checksum/count bypass option.

To resume an interrupted GPU job, rerun the exact GPU command with `--resume`:

```bash
python -B /kaggle/working/AT-RAG/scripts/embed_full_2wiki_corpus.py \
  --input /kaggle/input/datasets/bandlahemanth/at-rag-full-2wiki-corpus-zip/documents.jsonl \
  --output-dir /kaggle/working/at-rag-full-2wiki-embeddings \
  --device cuda:0 --batch-size 64 --checkpoint-batches 16 --resume
```

Resumption requires the same corpus bytes and companion summary, model revision,
batch size, checkpoint interval, device and recorded hardware/software environment,
and script checksum. Changing a batch size after an out-of-memory error requires
a fresh output path. Switching CPU/GPU or upgrading libraries also requires a new
job. The model is not loaded again if all rows were committed before an interrupted
final publication.

## Storage, checkpoints, and completion

Only one input record and one encoder batch are kept in Python memory. The `.npy`
matrix is written using NumPy `open_memmap`; operating-system page caching still
uses reclaimable memory. Matrix payload size is 609,758,208 bytes (about 582 MiB),
plus the `.npy` header, row mapping, model cache, and checkpoints. Model/tokenizer
memory and GPU activations depend on the batch size. The input is scanned once
before encoding and once during encoding; resumes repeat verification. Final
integrity verification also streams over the saved matrix and mapping.

During a run, work is stored in
`/kaggle/working/at-rag-full-2wiki-embeddings.inprogress/`:

- `job.json` records the immutable configuration.
- `checkpoint.json` records the committed row count, mapping byte offset, and
  SHA-256 digests of both committed prefixes. It is atomically replaced only after
  matrix and mapping writes have been flushed and fsynced.
- `artifacts/` holds the preallocated matrix and mapping. Their allocated shapes
  do not imply that all rows are valid.
- `.lock` provides an OS-released advisory lock against concurrent script writers.

After interruption, rows written beyond the last checkpoint are discarded or
overwritten on resume. At most `checkpoint-batches * batch-size` rows may need to
be encoded again. Saved prefix bytes and row order are verified before continuing;
corruption aborts rather than silently resetting progress. A crash before the
initial zero-row checkpoint can safely restart initialization with `--resume`.
Do not edit checkpoint files or delete the workspace while its process is running.

Only after a second corpus checksum/count check, ID alignment checks, normalized
finite-vector validation, and durable artifact re-reading does an atomic directory
rename publish the final output. The final directory contains exactly:

- `embeddings.npy`: `(396978, 384)` float32, row-major matrix.
- `document_ids.jsonl`: `{ "row": 0, "document_id": "..." }` per row, in original
  corpus order. Row indices are zero-based, with no JSONL header.
- `manifest.json`: `status: "complete"`, processed count, complete configuration
  (including library versions and model revision), and SHA-256/file-size evidence
  for both artifacts. The manifest also records the committed raw vector digest.

Consumers must require the final directory and a complete manifest, then check its
artifact checksums and counts. Never consume `.inprogress/artifacts/` directly,
even if a manifest remains there after an interrupted final rename. Rerunning an
already completed job fails without modifying outputs, including with `--resume`.
Linux atomic directory rename, fsync, and flock semantics are required; Windows
and remote/object-store output paths are not supported.

## Save as a private Kaggle Dataset

1. Confirm the command exited successfully and inspect `manifest.json`: complete
   status, 396,978 processed rows, correct model revision and corpus checksum.
2. Save/export the completed output directory from `/kaggle/working`. Upload its
   three files, or a ZIP containing them, using Kaggle's **New Dataset** flow.
3. Set visibility to **Private** before creating the Dataset. Verify the resulting
   Dataset is private, and keep its URL/version with the thesis experiment record.
   A suitable name is `at-rag-full-2wiki-minilm-embeddings`.
4. For future work, attach that private Dataset read-only and load the matrix with
   `np.load(path, mmap_mode="r", allow_pickle=False)`. Verify manifest checksums
   after download or extraction. Do not rebuild a retrieval index in this phase.

Kaggle working storage is not a durable checkpoint service. To preserve an unfinished
run across notebook/session loss, stop the process gracefully first, archive the
entire `.inprogress` directory, and save that archive privately with an explicit
**incomplete/checkpoint** name. In a new session, restore it to writable
`/kaggle/working/<output-name>.inprogress`, keeping its structure intact, then use
the same script, environment, device class, and command with `--resume`. Uncommitted
tails in a stopped-process snapshot are expected and will be recovered. A live
copy is not a consistent checkpoint backup. Never present this archive as the
completed embeddings Dataset. Persist the model cache separately if Internet
will be disabled in the new session.

Do not commit the corpus, generated vectors, row mapping, checkpoints, model cache,
or model weights to Git. The repository's `artifacts/` output path is already ignored.

## Tests and reproducibility limits

```bash
python -B -m unittest discover -s tests -p 'test_embed_full_2wiki_corpus.py' -v
```

The synthetic tests use an encoder mock and real memory-mapped files. They exercise
batching, original text/order, IDs, checksums, checkpoint recovery, incompatible
configuration rejection, corruption, malformed vectors, locks, and atomic
publication. They do not benchmark the full corpus or prove real GPU kernel
determinism. Floating-point results can differ between hardware, library versions,
or batch sizes; seeds and deterministic flags do not promise cross-platform
bitwise equality. Record and retain the manifest for each separately run job.
