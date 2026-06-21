"""Shared lazy-loading singleton for the sentence-transformers embedding model."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
_EMBEDDING_MODEL: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call.

    The model is initialised once per process and reused across all callers
    (chunker, embedder, and any future module that needs sentence embeddings).
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model %s", MODEL_NAME)
        _EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBEDDING_MODEL
