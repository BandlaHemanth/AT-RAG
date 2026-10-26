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

Use a dedicated Python 3.12 environment. Do not replace the notebook's system
Python in place, install into the old Python 3.13 preparation environment, or
upgrade original LangChain to work around its NumPy requirement.

First restore/inspect the environment record from the saved BERTopic backup.
BERTopic's default pickle format requires compatible Python and scientific
dependencies, especially sentence-transformers, PyTorch, NumPy, scikit-learn,
HDBSCAN, and UMAP/Numba. Their training versions are not available in this checkout.
The profile below is an ingestion compatibility profile, not an assertion that
all saved-model dependencies have been recovered. A matching BERTopic version
alone does not establish that unpickling/transform will succeed.

With `python` resolving to the dedicated Python 3.12 interpreter:

```bash
cd /kaggle/working/AT-RAG
python --version
python -m pip install --dry-run -r compat/minilm/requirements-py312.txt
```

If the proposal changes scientific dependencies from the training backup, stop
and reconcile the versions first. Do not silently replace a working saved-model
environment. In a compatible dedicated environment, install and check:

```bash
python -m pip install -r compat/minilm/requirements-py312.txt
python -m pip check
```

PyTorch and the scientific dependencies are resolved/preserved from that
environment; retain their exact versions in its backup. Avoid a blind installation
of the whole original `requirements.txt`: it includes unrelated BEIR/preprocessing
dependencies and omits several ingestion integrations. This task's extra manifest
does not replace or edit the original requirements.

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
Set `AT_RAG_SAVED_TOPIC_MODEL` to that file and `AT_RAG_ORIGINAL_DATASET` to the
processed `test_subsampled.jsonl` file. The runner makes a temporary symlink named
`bertopic_model` so no copy, rename, or source-path patch is needed.

```bash
python -B scripts/ingest_original_2wiki_minilm.py \
  --dataset "$AT_RAG_ORIGINAL_DATASET" \
  --topic-model "$AT_RAG_SAVED_TOPIC_MODEL" \
  --persist-directory /kaggle/working/at-rag-original-2wiki-chroma-minilm \
  --preflight
```

Preflight only reads files and installed package metadata. It does not import
BERTopic/Chroma/sentence-transformers, unpickle, download, encode, index, or create
the destination. It verifies:

- dataset SHA-256 `4948757b90627f40b77cd6bf81a62c1a7ad525d6a6a0c91b17a2e0e2fa88f5bc`;
- exactly 500 questions and 5,000 context occurrences;
- saved-model SHA-256 `4a626723a82563556a7567706b742b7c8ed4444efefe1a411c23d7554078749c`;
- unchanged original ingestion/training/AT-RAG source hashes and the package profile;
- an absent destination, preventing accidental appending or use of an older index.

Preflight success is not proof of saved-model loading or endpoint-free MiniLM
execution. Keep the printed environment record and compare it to the training
backup. Do not overwrite or load the prior custom full-corpus model.

## Later execution, only after preflight and environment reconciliation

```bash
python -B scripts/ingest_original_2wiki_minilm.py \
  --dataset "$AT_RAG_ORIGINAL_DATASET" \
  --topic-model "$AT_RAG_SAVED_TOPIC_MODEL" \
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

Real saved-model compatibility, MiniLM loading/encoding, and Chroma persistence
remain to be verified in Kaggle. No claim of full QA reproduction is made here.
