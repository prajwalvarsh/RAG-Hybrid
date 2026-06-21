"""Tests for ingest/embedder.py."""

import numpy as np
import pytest

from ingest.embedder import embed_chunks
from ingest.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkStrategy,
    EmbeddedChunk,
)

_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_EMBED_DIM = 8  # small fake dimension used in all mocked embeddings


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Three minimal Chunk objects for use across embedder tests."""
    return [
        Chunk(
            chunk_id=f"doc-001_fixed_{i:04d}",
            doc_id="doc-001",
            text=f"Sample text for chunk {i}.",
            chunk_index=i,
            strategy=ChunkStrategy.FIXED,
            metadata=ChunkMetadata(),
        )
        for i in range(3)
    ]


def _mock_model(mocker, n_chunks: int):
    """Patch ingest.model.get_embedding_model to return a mock whose .encode() returns a numpy array."""
    fake_embeddings = np.ones((n_chunks, _EMBED_DIM), dtype=np.float32)
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = fake_embeddings
    mocker.patch("ingest.model.get_embedding_model", return_value=mock_model)
    return mock_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_chunks_returns_correct_type(
    mocker, sample_chunks: list[Chunk]
) -> None:
    """embed_chunks must return a list of EmbeddedChunk objects."""
    _mock_model(mocker, len(sample_chunks))
    result = embed_chunks(sample_chunks)

    assert isinstance(result, list)
    assert len(result) == len(sample_chunks)
    for item in result:
        assert isinstance(item, EmbeddedChunk)


def test_embed_chunks_batch(mocker, sample_chunks: list[Chunk]) -> None:
    """model.encode() must be called exactly once for the full batch, not per chunk."""
    mock_model = _mock_model(mocker, len(sample_chunks))
    embed_chunks(sample_chunks)

    mock_model.encode.assert_called_once()
    call_args, _ = mock_model.encode.call_args
    passed_texts = call_args[0]
    assert len(passed_texts) == len(sample_chunks)


def test_embedding_model_field(mocker, sample_chunks: list[Chunk]) -> None:
    """Every EmbeddedChunk must have embedding_model set to 'BAAI/bge-base-en-v1.5'."""
    _mock_model(mocker, len(sample_chunks))
    result = embed_chunks(sample_chunks)

    for ec in result:
        assert ec.embedding_model == _MODEL_NAME


def test_embed_chunks_empty_input(mocker) -> None:
    """embed_chunks returns an empty list when given an empty input."""
    mock_model = _mock_model(mocker, 0)
    result = embed_chunks([])

    assert result == []
    mock_model.encode.assert_not_called()


def test_embed_chunks_preserves_order(mocker, sample_chunks: list[Chunk]) -> None:
    """Output EmbeddedChunks must correspond to input Chunks in the same order."""
    _mock_model(mocker, len(sample_chunks))
    result = embed_chunks(sample_chunks)

    for original, embedded in zip(sample_chunks, result):
        assert embedded.chunk.chunk_id == original.chunk_id
