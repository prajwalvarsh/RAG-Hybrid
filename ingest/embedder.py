"""Embed a list of Chunk objects using BAAI/bge-base-en-v1.5 via sentence-transformers."""

import logging

import numpy as np

from ingest import model as _model_mod
from ingest.schemas import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed a batch of Chunk objects and return EmbeddedChunk instances.

    All chunk texts are encoded in a single model.encode() call for efficiency.
    The embedding model is loaded once as a module-level singleton via
    _get_embedding_model() and reused across calls.

    Vectors are L2-normalized at encode time (normalize_embeddings=True)
    because BGE models are trained for cosine similarity. Normalization
    ensures retrieval ranks by semantic direction regardless of vector
    magnitude, and reduces cosine similarity to a simple dot product.

    Args:
        chunks: List of Chunk objects produced by ingest/chunker.py.

    Returns:
        List of EmbeddedChunk objects in the same order as the input, each
        carrying the original Chunk, its float embedding vector, and the
        model name used to produce it.
    """
    if not chunks:
        return []

    model = _model_mod.get_embedding_model()
    texts = [chunk.text for chunk in chunks]

    logger.info("Embedding %d chunks with model %s", len(chunks), _model_mod.MODEL_NAME)
    raw: np.ndarray = model.encode(texts, normalize_embeddings=True)

    embedded: list[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, raw):
        embedded.append(
            EmbeddedChunk(
                chunk=chunk,
                embedding=vector.tolist(),
                embedding_model=_model_mod.MODEL_NAME,
            )
        )

    logger.debug("Produced %d EmbeddedChunk objects", len(embedded))
    return embedded
