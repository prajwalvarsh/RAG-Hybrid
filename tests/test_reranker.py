"""Tests for retrieval/reranker.py — cross-encoder reranking."""

import numpy as np
import pytest

import retrieval.reranker as reranker_mod
from retrieval.reranker import rerank
from retrieval.schemas import RetrievalMethod, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str,
    rank: int,
    text: str = "chunk text",
    score: float = 0.5,
) -> RetrievalResult:
    """Construct a HYBRID RetrievalResult for use in reranker tests."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        rank=rank,
        metadata={},
        retrieval_method=RetrievalMethod.HYBRID,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_reranker_singleton():
    """Reset the module-level _RERANKER singleton between tests."""
    reranker_mod._RERANKER = None
    yield
    reranker_mod._RERANKER = None


@pytest.fixture
def mock_reranker(mocker):
    """Patch _get_reranker to return a mock CrossEncoder."""
    mock_model = mocker.MagicMock()
    mocker.patch("retrieval.reranker._get_reranker", return_value=mock_model)
    return mock_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rerank_returns_top_k(mock_reranker) -> None:
    """rerank must return at most rerank_top_k results."""
    results = [_make_result(f"c{i}", i + 1) for i in range(10)]
    mock_reranker.predict.return_value = np.array(
        [float(i) for i in range(10)], dtype=float
    )

    reranked = rerank("test query", results, rerank_top_k=5)

    assert len(reranked) == 5


def test_rerank_ordering(mock_reranker) -> None:
    """Higher reranker score must produce a lower rank number (rank 1 = best)."""
    results = [
        _make_result("c1", 1, text="chunk one"),
        _make_result("c2", 2, text="chunk two"),
        _make_result("c3", 3, text="chunk three"),
    ]
    # Assign scores so c3 is best, c1 is worst.
    mock_reranker.predict.return_value = np.array([0.1, 0.5, 0.9], dtype=float)

    reranked = rerank("test query", results, rerank_top_k=3)

    assert reranked[0].chunk_id == "c3"
    assert reranked[0].rank == 1
    assert reranked[1].chunk_id == "c2"
    assert reranked[1].rank == 2
    assert reranked[2].chunk_id == "c1"
    assert reranked[2].rank == 3


def test_rerank_empty_input(mock_reranker) -> None:
    """rerank must return an empty list when given an empty input list."""
    reranked = rerank("test query", [], rerank_top_k=5)

    assert reranked == []
    mock_reranker.predict.assert_not_called()


def test_rerank_updates_score(mock_reranker) -> None:
    """score field on returned results must reflect the reranker score, not the original RRF score."""
    original_score = 0.0123  # RRF score — should not appear in output
    reranker_score = 0.9876

    results = [_make_result("c1", 1, score=original_score)]
    mock_reranker.predict.return_value = np.array([reranker_score], dtype=float)

    reranked = rerank("test query", results, rerank_top_k=1)

    assert len(reranked) == 1
    assert reranked[0].score == pytest.approx(reranker_score)
    assert reranked[0].score != original_score
