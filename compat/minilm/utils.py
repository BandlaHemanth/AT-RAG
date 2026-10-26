"""Local retrieval embeddings replacing the undocumented AT-RAG TEI service.

Only the missing utils.get_embedding_model interface is supplied here. Importing
this module does not import sentence-transformers, download, or execute a model.
"""

from functools import lru_cache
import math


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32


def embedding_settings():
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalize_embeddings": True,
        "precision": "float32",
        "device": "cpu",
        "encoder_batch_size": BATCH_SIZE,
        "text_preprocessing": "none; pass paragraph/query strings unchanged",
    }


class MiniLMEmbeddings:
    """The embed_documents/embed_query interface consumed by LangChain Chroma."""

    def __init__(self):
        self._encoder = None

    def _encode(self, texts):
        if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
            raise TypeError("Embeddings require a list of strings")
        if not texts:
            return []
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                MODEL_ID, revision=MODEL_REVISION, device="cpu", trust_remote_code=False
            )
            self._encoder.max_seq_length = MAX_SEQUENCE_LENGTH
        vectors = self._encoder.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            precision="float32",
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
        if len(vectors) != len(texts) or any(
            len(row) != DIMENSION or any(not math.isfinite(v) for v in row)
            for row in vectors
        ):
            raise ValueError("MiniLM must return one finite 384-dimensional vector per text")
        return vectors

    def embed_documents(self, texts):
        return self._encode(list(texts))

    def embed_query(self, text):
        return self._encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model():
    return MiniLMEmbeddings()
