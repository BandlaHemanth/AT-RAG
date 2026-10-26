# Original 2Wiki Chroma ingestion with the declared MiniLM substitution

This prepares the released AT-RAG ingestor for the original evaluation subset.
It is a controlled reproduction with a replacement retrieval encoder, not a claim
to reproduce the paper's undocumented embedding model or its exact scores.
TAE V2 is not implemented. No original source file is edited.

## The one deliberate methodological substitution

The selected branch imports missing `utils.get_embedding_model()`. The author's
`mixtra_dev` helper supplies `HuggingFaceHubEmbeddings(model=config("TEI_URL"), ...)`,
but its underlying model is undocumented. The external `compat/minilm/utils.py`
supplies that same two-method embedding interface with these fixed settings:

| Setting | Value |
| --- | --- |
| Retrieval encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Embedding size | 384 |
| Device | CPU |
| Precision | float32 |
| Normalization | L2 normalization enabled for documents and queries |
| Maximum sequence length | 256 tokenizer tokens; encoder truncation only |
| Encoder batch size | 32; does not change the original 1,000-record ingestion batches |
| Text input | Paragraph/query strings passed unchanged; no title concatenation or cleaning |

Use these exact settings and the same saved Chroma store for the later original
AT-RAG versus AT-RAG + TAE V2 comparison. Do not use the prior full-corpus vectors.
The adapter lazily loads pretrained weights; it never fits or fine-tunes a model.
Public Hugging Face access/cache is needed only when execution loads the encoder.
No TEI endpoint, HF token, or OpenAI API call is used by this local encoder.

## What remains original

- `Ingestor.load_data()` keeps all 5,000 context occurrences, including duplicate
  paragraphs, in file/context order. Metadata remains `idx`, `title`, and
  `is_supporting`, including the original missing-value defaults.
- `create_vectordb()` loads the existing BERTopic model and transforms paragraphs;
  it adds `bertopic: "Topic <id>"`. Probabilities are discarded as in the source.
  No topic model is fitted, reduced, or reconfigured. The supplied model has 12
  reported regular topics; do not force it to 20 or substitute the 418-topic model.
- The original Chroma constructor, UUID assignment by its LangChain adapter,
  five calls of 1,000 records to `add_texts`, and persistence remain intact.
- Collection/metric settings are not overridden. The pinned LangChain/Chroma
  profile defaults to collection `langchain` and Chroma L2 distance. Normalization
  is part of the declared encoder substitution, not a switch to cosine search.
- The original similarity retrieval, topic equality filter, total top-5 limit,
  generation, prompts, grading, rewriting, and graph remain untouched. The runner
  does not invoke QA. The same helper supports `embed_query` when the store is
  later queried. No additional document filtering or deduplication is introduced.

Original references at commit `b05341305d3ecce01e6dba554f60122da1648333`:
`vectorDB/dataset_ingestion.py:30-105`, `vectorDB/train_topic_model.py:39-55`,
`models/AT_RAG.py:172-192`; upstream helper:
https://github.com/MrRezaeiUofT/AT-RAG/blob/ace4cdce77db0cb58da946812e1f14ccf0510a54/utils.py#L24-L27

## Kaggle Python 3.12 environment

Use a dedicated **Python 3.12.3** environment, matching the saved model's training
record exactly, including the patch version. Preflight rejects other Python
versions rather than asserting untested pickle compatibility. Do not replace the notebook's system
Python in place, install into the old Python 3.13 preparation environment, or
upgrade original LangChain to work around its NumPy requirement.

The owner supplied these versions from the saved model's training environment:

| Scientific package | Required training version |
| --- | --- |
| bertopic | 0.16.3 |
| transformers | 4.45.1 |
| sentence-transformers | 5.7.0 |
| torch | 2.14.1 |
| numpy | 1.26.4 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| umap-learn | 0.5.12 |
| hdbscan | 0.8.44 |
| numba | 0.68.0 |
| llvmlite | 0.50.0 |

All eleven are now pinned in `compat/minilm/requirements-py312.txt`; the previous
sentence-transformers 3.1.1 pin was incorrect for this saved model. Do not use the
Codex Cloud tooling environment as the saved-model environment. The Cloud test
runtime has different Python, sentence-transformers, and torch versions and is
expected to fail the real preflight.

Restore the complete 62-package environment record from the Kaggle baseline
backup. Set `AT_RAG_TRAINING_REQUIREMENTS` to its flat `pip freeze` record with
exact `name==version` pins. If the backup is JSON, export its recorded package
names and versions to that format without guessing missing versions or freezing
the current machine instead. The runner requires all 62 entries, including the
eleven scientific packages. It rejects ranges, URLs, editable requirements,
markers, wildcards, and duplicate normalized names rather than ignoring them.
Other training packages such as pandas and huggingface-hub are preserved from
this record; their versions are not invented or separately pinned by the adapter.
The 62-package record could not be retrieved through the available Kaggle access
during this correction, so it is supplied as an external provenance input.

With `python` resolving to the restored Python 3.12.3 interpreter:

```bash
cd /kaggle/working/AT-RAG
python --version
python -m pip install --dry-run \
  -c "$AT_RAG_TRAINING_REQUIREMENTS" \
  -r compat/minilm/requirements-py312.txt
```

The freeze is a constraint file for this operation: pip must not change any
backed-up version. If resolution conflicts, stop and report the conflicting pins.
Do not bypass the conflict with `--no-deps`, change scientific versions, or run a
blind upgrade. After the constrained dry-run succeeds, install the additional
ingestion integrations into that same restored environment and check:

```bash
python -m pip install \
  -c "$AT_RAG_TRAINING_REQUIREMENTS" \
  -r compat/minilm/requirements-py312.txt
python -m pip check
```

Preserve the original PyTorch CPU/CUDA wheel/build and its associated packages;
the adapter's CPU inference setting is not permission to replace that wheel. A
pin such as `torch==2.14.1` follows PEP 440 and can match a local `+cu` build. If the
backup records a local build tag, preflight requires that exact tag as well.
Avoid installing the entire original `requirements.txt` merely for ingestion: it
contains unrelated dependencies and omits integrations. It remains unedited.

### Dependency conflicts and verification limits

No unavoidable conflict was found in the published direct requirements of the
supplied scientific pins and the ingestion profile. In particular:

- sentence-transformers 5.7.0 permits transformers >=4.41,<6, torch >=1.11,
  numpy >=1.20, and scipy >=1.0: the supplied pins satisfy these bounds.
- transformers 4.45.1 requires huggingface-hub >=0.23.2,<1 and tokenizers >=0.20,<0.21.
  Preserve matching versions from the complete freeze; do not independently upgrade them.
- scipy 1.17.1 requires numpy >=1.26.4,<2.7; numba 0.68.0 requires numpy >=1.22,<2.6
  and llvmlite >=0.50.0dev0,<0.51. The supplied pins satisfy these bounds.
- Original langchain 0.3.2 requires numpy >=1.26,<2 on Python >=3.12, which the
  saved numpy 1.26.4 satisfies. Do not install Transformers optional `[all]` or
  `[flax]` extras: their older SciPy constraint would conflict with scipy 1.17.1.

Sources: published PyPI metadata for
[sentence-transformers 5.7.0](https://pypi.org/pypi/sentence-transformers/5.7.0/json),
[transformers 4.45.1](https://pypi.org/pypi/transformers/4.45.1/json),
[scipy 1.17.1](https://pypi.org/pypi/scipy/1.17.1/json),
[numba 0.68.0](https://pypi.org/pypi/numba/0.68.0/json), and
[langchain 0.3.2](https://pypi.org/pypi/langchain/0.3.2/json).

The remaining 51 training versions have not been independently inspected here.
There may be a conflict between a frozen integration/transitive package and the
additional ingestion profile. Such a conflict must stop setup and be reported;
the runner never changes packages. Full resolution against the actual freeze,
binary compatibility, and model unpickling still need Kaggle verification.

The unused original OpenAI imports require its package. Its import-time key lookup
is satisfied only inside the runner with a temporary non-secret placeholder when
no key is set; existing environment values are preserved. No real key is required
for ingestion, no `.env` is written, and that placeholder must not be used for QA.

## Read-only preflight: the next Kaggle action

Copy these five new files from the shared workspace into the same relative paths
of the Kaggle checkout. Existing original files must remain unchanged:

- `compat/minilm/utils.py`
- `compat/minilm/requirements-py312.txt`
- `scripts/ingest_original_2wiki_minilm.py`
- `tests/test_original_chroma_minilm.py`
- `docs/original_chroma_minilm.md`

Attach the dataset/environment backup and BERTopic-model backup. Locate the actual
saved **model file** in `/kaggle/input`; no mounted filename is assumed here.
Set `AT_RAG_SAVED_TOPIC_MODEL` to that file, `AT_RAG_ORIGINAL_DATASET` to the
processed `test_subsampled.jsonl`, and `AT_RAG_TRAINING_REQUIREMENTS` to the original
62-package freeze. The runner makes a temporary symlink named
`bertopic_model` so no copy, rename, or source-path patch is needed.

```bash
python -B scripts/ingest_original_2wiki_minilm.py \
  --dataset "$AT_RAG_ORIGINAL_DATASET" \
  --topic-model "$AT_RAG_SAVED_TOPIC_MODEL" \
  --training-requirements "$AT_RAG_TRAINING_REQUIREMENTS" \
  --persist-directory /kaggle/working/at-rag-original-2wiki-chroma-minilm \
  --preflight
```

Preflight only reads files/installed package metadata and runs read-only `pip check`. It does not import
BERTopic/Chroma/sentence-transformers, unpickle, download, encode, index, or create
the destination. It verifies:

- dataset SHA-256 `4948757b90627f40b77cd6bf81a62c1a7ad525d6a6a0c91b17a2e0e2fa88f5bc`;
- exactly 500 questions and 5,000 context occurrences;
- saved-model SHA-256 `4a626723a82563556a7567706b742b7c8ed4444efefe1a411c23d7554078749c`;
- unchanged original ingestion/training/AT-RAG source hashes;
- exact Python 3.12.3, all eleven scientific pins, all other profile pins, and
  installed versions of every package in the 62-package freeze;
- compatibility between overlapping profile/freeze pins and successful `pip check`;
- an absent destination, preventing accidental appending or use of an older index.

Any missing package or version mismatch fails preflight before the destination is
created. The report records the training freeze's SHA-256 and all checked versions.
Preflight success is metadata agreement with the supplied record, not proof of
saved-model loading or real MiniLM execution. Do not overwrite or load the prior
custom full-corpus model.

## Later execution, only after preflight and environment reconciliation

```bash
python -B scripts/ingest_original_2wiki_minilm.py \
  --dataset "$AT_RAG_ORIGINAL_DATASET" \
  --topic-model "$AT_RAG_SAVED_TOPIC_MODEL" \
  --training-requirements "$AT_RAG_TRAINING_REQUIREMENTS" \
  --persist-directory /kaggle/working/at-rag-original-2wiki-chroma-minilm \
  --execute
```

The runner calls the original `Ingestor.create_vectordb()` unchanged, verifies
5,000 stored records, and writes `minilm_ingestion_manifest.json` beside the Chroma
files with input hashes, source hashes, encoder settings, and installed versions.
Do not run `python vectorDB/dataset_ingestion.py`: its main selects MuSiQue.
If ingestion fails, a partial destination may remain. Use a new destination;
the runner never resumes, deduplicates, deletes, or appends to a partial store.

For later QA, add the absolute `/kaggle/working/AT-RAG/compat/minilm` directory to
`PYTHONPATH` before importing original AT-RAG,
and pass this store and the same saved topic-model paths through its existing
constructor. Do not modify QA source, run the ingestion script's demo as a strict
retrieval test, or mix embedding configurations between indexing and querying.
QA still needs its genuine OpenAI credentials and original environment settings.

## Focused verification and limits

```bash
python -B -m unittest discover -s tests -p 'test_original_chroma_minilm.py' -v
```

These tests exercise the original ingestor with capture-only Chroma and topic
stand-ins. They verify 5,000 occurrences/duplicates, topic labels including `-1`,
metadata/defaults, five original batches, persistence invocation, encoder/query
settings, artifact rejection, and state cleanup. One test also exercises the real
LangChain wrapper against a mock collection, verifying duplicate UUIDs, query
embeddings, the similarity filter, and total top-5 forwarding. The wrapper's
constructor is bypassed. Tests never load a real model, train, construct a real
Chroma collection, or index data.

Environment tests use fictional 62-package fixtures and mocked installed metadata;
they do not pretend to reconstruct the unavailable backup. They reject each
scientific mismatch, Python patch differences, partial freezes, profile conflicts,
missing packages, differing recorded wheel tags, and `pip check` failures.

Real saved-model compatibility, MiniLM loading/encoding, and Chroma persistence
remain to be verified in Kaggle. No claim of full QA reproduction is made here.
