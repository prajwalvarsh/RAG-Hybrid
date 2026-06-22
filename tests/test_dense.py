"""Tests for retrieval/dense.py."""

import pytest

from retrieval.dense import retrieve_dense
from retrieval.schemas import RetrievalMethod, RetrievalRequest, RetrievalResult

_EMBED_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chroma_response(
    ids: list[str],
    documents: list[str],
    distances: list[float],
    metadatas: list[dict],
) -> dict:
    """Build the dict structure ChromaDB returns from collection.query()."""
    return {
        "ids": [ids],
        "documents": [documents],
        "distances": [distances],
        "metadatas": [metadatas],
    }


def _mock_model(mocker, query_vector=None):
    """Patch ingest.model.get_embedding_model with a mock whose .encode() returns a numpy array."""
    import numpy as np

    if query_vector is None:
        query_vector = [0.1] * _EMBED_DIM

    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = mocker.MagicMock(
        tolist=mocker.MagicMock(return_value=query_vector)
    )
    mocker.patch("ingest.model.get_embedding_model", return_value=mock_model)
    return mock_model


def _mock_collection(mocker, count: int, query_response: dict):
    """Patch ingest.indexer.get_collection with a mock collection."""
    mock_collection = mocker.MagicMock()
    mock_collection.count.return_value = count
    mock_collection.query.return_value = query_response
    mocker.patch("retrieval.dense.get_collection", return_value=mock_collection)
    return mock_collection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_request() -> RetrievalRequest:
    """A minimal RetrievalRequest used across tests."""
    return RetrievalRequest(query="What is RAG?", retrieval_top_k=3)


@pytest.fixture
def chroma_response_three() -> dict:
    """A fake ChromaDB response with three results in distance order."""
    return _make_chroma_response(
        ids=["chunk-001", "chunk-002", "chunk-003"],
        documents=["Text one.", "Text two.", "Text three."],
        distances=[0.1, 0.3, 0.5],
        metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_dense_returns_correct_type(
    mocker, base_request: RetrievalRequest, chroma_response_three: dict
) -> None:
    """retrieve_dense must return a list of RetrievalResult objects."""
    _mock_model(mocker)
    _mock_collection(mocker, count=3, query_response=chroma_response_three)

    results = retrieve_dense(base_request)

    assert isinstance(results, list)
    assert len(results) == 3
    for item in results:
        assert isinstance(item, RetrievalResult)


def test_retrieve_dense_ranking(
    mocker, base_request: RetrievalRequest, chroma_response_three: dict
) -> None:
    """Results must be ordered by score descending and rank must start at 1."""
    _mock_model(mocker)
    _mock_collection(mocker, count=3, query_response=chroma_response_three)

    results = retrieve_dense(base_request)

    # rank starts at 1
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3

    # scores are descending (lower distance → higher score)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    # first result has the highest score (distance 0.1 → score 0.9)
    assert pytest.approx(results[0].score, abs=1e-6) == 0.9
    assert pytest.approx(results[1].score, abs=1e-6) == 0.7
    assert pytest.approx(results[2].score, abs=1e-6) == 0.5


def test_retrieve_dense_empty_collection(
    mocker, base_request: RetrievalRequest
) -> None:
    """retrieve_dense must return an empty list when the collection is empty."""
    _mock_model(mocker)
    mock_collection = mocker.MagicMock()
    mock_collection.count.return_value = 0
    mocker.patch("retrieval.dense.get_collection", return_value=mock_collection)

    results = retrieve_dense(base_request)

    assert results == []
    mock_collection.query.assert_not_called()


def test_retrieve_dense_normalization(
    mocker, base_request: RetrievalRequest, chroma_response_three: dict
) -> None:
    """model.encode() must be called with normalize_embeddings=True."""
    mock_model = _mock_model(mocker)
    _mock_collection(mocker, count=3, query_response=chroma_response_three)

    retrieve_dense(base_request)

    mock_model.encode.assert_called_once()
    _, kwargs = mock_model.encode.call_args
    assert kwargs.get("normalize_embeddings") is True


def test_retrieve_dense_retrieval_method(
    mocker, base_request: RetrievalRequest, chroma_response_three: dict
) -> None:
    """Every RetrievalResult must have retrieval_method set to 'dense'."""
    _mock_model(mocker)
    _mock_collection(mocker, count=3, query_response=chroma_response_three)

    results = retrieve_dense(base_request)

    for r in results:
        assert r.retrieval_method == RetrievalMethod.DENSE
