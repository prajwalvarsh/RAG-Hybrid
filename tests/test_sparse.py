"""Tests for retrieval/sparse.py."""

import pytest

import retrieval.sparse as sparse_mod
from retrieval.schemas import RetrievalMethod, RetrievalRequest, RetrievalResult
from retrieval.sparse import retrieve_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collection(mocker, ids: list[str], documents: list[str]):
    """Return a mock ChromaDB collection whose .get() returns fake corpus data."""
    mock_collection = mocker.MagicMock()
    mock_collection.get.return_value = {
        "ids": ids,
        "documents": documents,
        "metadatas": [{} for _ in ids],
    }
    mocker.patch("retrieval.sparse.get_collection", return_value=mock_collection)
    return mock_collection


def _reset_index():
    """Reset the module-level BM25 singleton between tests."""
    sparse_mod._BM25_INDEX = None
    sparse_mod._BM25_BUILT = False
    sparse_mod._BM25_CHUNK_IDS = []
    sparse_mod._BM25_TEXTS = []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure each test starts with a clean BM25 singleton."""
    _reset_index()
    yield
    _reset_index()


@pytest.fixture
def base_request() -> RetrievalRequest:
    """Minimal RetrievalRequest used across tests."""
    return RetrievalRequest(query="retrieval augmented generation", top_k=3)


@pytest.fixture
def fake_corpus() -> tuple[list[str], list[str]]:
    """Three-document corpus with predictable BM25 overlap with the base query."""
    ids = ["chunk-001", "chunk-002", "chunk-003"]
    documents = [
        "retrieval augmented generation is a technique",  # hits all 3 query tokens
        "augmented generation helps LLMs",               # hits 2 query tokens
        "unrelated document about cooking recipes",      # hits 0 query tokens
    ]
    return ids, documents


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_sparse_returns_correct_type(
    mocker, base_request: RetrievalRequest, fake_corpus
) -> None:
    """retrieve_sparse must return a list of RetrievalResult objects."""
    ids, documents = fake_corpus
    _make_collection(mocker, ids, documents)

    results = retrieve_sparse(base_request)

    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, RetrievalResult)


def test_retrieve_sparse_ranking(
    mocker, base_request: RetrievalRequest, fake_corpus
) -> None:
    """Results must be ordered by BM25 score descending and rank must start at 1."""
    ids, documents = fake_corpus
    _make_collection(mocker, ids, documents)

    results = retrieve_sparse(base_request)

    assert len(results) >= 1
    assert results[0].rank == 1
    for i, result in enumerate(results):
        assert result.rank == i + 1

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_sparse_retrieval_method(
    mocker, base_request: RetrievalRequest, fake_corpus
) -> None:
    """Every RetrievalResult must have retrieval_method set to SPARSE."""
    ids, documents = fake_corpus
    _make_collection(mocker, ids, documents)

    results = retrieve_sparse(base_request)

    assert len(results) > 0
    for r in results:
        assert r.retrieval_method == RetrievalMethod.SPARSE


def test_retrieve_sparse_empty_collection(
    mocker, base_request: RetrievalRequest
) -> None:
    """retrieve_sparse must return an empty list when the collection is empty."""
    _make_collection(mocker, ids=[], documents=[])

    results = retrieve_sparse(base_request)

    assert results == []


def test_bm25_force_rebuild(mocker, fake_corpus) -> None:
    """get_bm25_index must rebuild the index when force_rebuild=True."""
    ids, documents = fake_corpus
    mock_collection = _make_collection(mocker, ids, documents)

    request = RetrievalRequest(query="retrieval", top_k=3)

    # First call builds the index.
    retrieve_sparse(request)
    assert mock_collection.get.call_count == 1

    # Second call with force_rebuild must re-fetch from ChromaDB.
    from retrieval.sparse import get_bm25_index

    get_bm25_index(request.collection_name, force_rebuild=True)
    assert mock_collection.get.call_count == 2


def test_zero_scores_filtered(mocker, base_request: RetrievalRequest) -> None:
    """Chunks with BM25 score 0.0 must not appear in the results."""
    ids = ["chunk-A", "chunk-B", "chunk-C"]
    documents = [
        "retrieval augmented generation overview",  # matches query
        "unrelated cooking article",                # no overlap → score 0
        "another unrelated sports news piece",      # no overlap → score 0
    ]
    _make_collection(mocker, ids, documents)

    results = retrieve_sparse(base_request)

    returned_ids = {r.chunk_id for r in results}
    assert "chunk-B" not in returned_ids
    assert "chunk-C" not in returned_ids
    for r in results:
        assert r.score > 0.0
