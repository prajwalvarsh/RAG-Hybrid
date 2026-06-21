"""Tests for ingest/indexer.py."""

import pytest

from ingest.indexer import get_collection, index_chunks
from ingest.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkStrategy,
    EmbeddedChunk,
)

_EMBED_DIM = 8


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_embedded_chunk(index: int) -> EmbeddedChunk:
    """Build a minimal EmbeddedChunk for testing."""
    chunk = Chunk(
        chunk_id=f"doc-001_fixed_{index:04d}",
        doc_id="doc-001",
        text=f"Test text for chunk {index}.",
        chunk_index=index,
        strategy=ChunkStrategy.FIXED,
        metadata=ChunkMetadata(),
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=[0.1] * _EMBED_DIM,
        embedding_model="BAAI/bge-base-en-v1.5",
    )


@pytest.fixture
def embedded_chunks() -> list[EmbeddedChunk]:
    """Three EmbeddedChunk objects for use across indexer tests."""
    return [_make_embedded_chunk(i) for i in range(3)]


def _mock_chroma_client(mocker):
    """Patch chromadb.PersistentClient and return (mock_client, mock_collection)."""
    mock_collection = mocker.MagicMock()
    mock_client = mocker.MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mocker.patch("ingest.indexer.chromadb.PersistentClient", return_value=mock_client)
    return mock_client, mock_collection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_index_chunks_upserts_correct_count(
    mocker, embedded_chunks: list[EmbeddedChunk]
) -> None:
    """index_chunks must upsert exactly as many ids as there are EmbeddedChunks."""
    _, mock_collection = _mock_chroma_client(mocker)

    index_chunks(embedded_chunks, collection_name="test_col")

    mock_collection.upsert.assert_called_once()
    kwargs = mock_collection.upsert.call_args.kwargs
    assert len(kwargs["ids"]) == len(embedded_chunks)
    assert len(kwargs["embeddings"]) == len(embedded_chunks)
    assert len(kwargs["documents"]) == len(embedded_chunks)
    assert len(kwargs["metadatas"]) == len(embedded_chunks)


def test_index_chunks_uses_chunk_id_as_id(
    mocker, embedded_chunks: list[EmbeddedChunk]
) -> None:
    """The upserted ids must match the chunk_id fields of the input chunks."""
    _, mock_collection = _mock_chroma_client(mocker)

    index_chunks(embedded_chunks, collection_name="test_col")

    kwargs = mock_collection.upsert.call_args.kwargs
    expected_ids = [ec.chunk.chunk_id for ec in embedded_chunks]
    assert kwargs["ids"] == expected_ids


def test_index_chunks_empty_input(mocker) -> None:
    """index_chunks must not call upsert when given an empty list."""
    mock_client, mock_collection = _mock_chroma_client(mocker)

    index_chunks([])

    mock_collection.upsert.assert_not_called()


def test_get_collection_returns_collection(mocker) -> None:
    """get_collection must return the ChromaDB collection object."""
    mock_client, mock_collection = _mock_chroma_client(mocker)

    result = get_collection("my_collection")

    assert result is mock_collection
    mock_client.get_or_create_collection.assert_called_once_with(
        name="my_collection", metadata={"hnsw:space": "cosine"}
    )


def test_get_collection_default_name(mocker) -> None:
    """get_collection with no args must request the 'rag_hybrid' collection."""
    mock_client, _ = _mock_chroma_client(mocker)

    get_collection()

    mock_client.get_or_create_collection.assert_called_once_with(
        name="rag_hybrid", metadata={"hnsw:space": "cosine"}
    )
