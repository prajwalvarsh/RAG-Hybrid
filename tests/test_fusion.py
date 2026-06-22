"""Tests for retrieval/fusion.py — Reciprocal Rank Fusion."""

import pytest

from retrieval.fusion import fuse_results
from retrieval.schemas import RetrievalMethod, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str,
    rank: int,
    text: str = "some text",
    score: float = 1.0,
    metadata: dict | None = None,
    method: RetrievalMethod = RetrievalMethod.DENSE,
) -> RetrievalResult:
    """Construct a RetrievalResult with sensible defaults."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        rank=rank,
        metadata=metadata or {},
        retrieval_method=method,
    )


def _dense(chunk_id: str, rank: int, **kwargs) -> RetrievalResult:
    return _make_result(chunk_id, rank, method=RetrievalMethod.DENSE, **kwargs)


def _sparse(chunk_id: str, rank: int, **kwargs) -> RetrievalResult:
    return _make_result(chunk_id, rank, method=RetrievalMethod.SPARSE, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fuse_results_returns_hybrid_method() -> None:
    """Every result in the fused list must have retrieval_method=HYBRID."""
    dense = [_dense("c1", 1), _dense("c2", 2)]
    sparse = [_sparse("c3", 1), _sparse("c2", 2)]

    results = fuse_results(dense, sparse)

    assert len(results) > 0
    for r in results:
        assert r.retrieval_method == RetrievalMethod.HYBRID


def test_fuse_results_ranking() -> None:
    """RRF scores must produce correct ordering and ranks starting at 1."""
    # c1: rank-1 dense only  → 0.7*(1/61) ≈ 0.01148
    # c2: rank-1 sparse only → 0.3*(1/61) ≈ 0.00492
    dense = [_dense("c1", 1)]
    sparse = [_sparse("c2", 1)]

    results = fuse_results(dense, sparse)

    assert results[0].rank == 1
    assert results[1].rank == 2

    # c1 should rank higher because dense_weight (0.7) > sparse_weight (0.3)
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "c2"

    # Scores must be strictly descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_fuse_results_chunk_in_both_lists() -> None:
    """A chunk that appears in both lists must outrank a chunk in only one list."""
    # "shared" appears rank-1 in both → gets both dense and sparse contributions
    # "dense_only" appears rank-1 in dense only → gets only dense contribution
    dense = [_dense("shared", 1), _dense("dense_only", 2)]
    sparse = [_sparse("shared", 1)]

    results = fuse_results(dense, sparse)

    shared_result = next(r for r in results if r.chunk_id == "shared")
    dense_only_result = next(r for r in results if r.chunk_id == "dense_only")

    assert shared_result.score > dense_only_result.score
    assert shared_result.rank < dense_only_result.rank


def test_fuse_results_empty_inputs() -> None:
    """fuse_results with both lists empty must return an empty list."""
    results = fuse_results([], [])

    assert results == []


def test_fuse_results_top_k() -> None:
    """fuse_results must return no more than top_k results."""
    dense = [_dense(f"d{i}", i + 1) for i in range(15)]
    sparse = [_sparse(f"s{i}", i + 1) for i in range(15)]

    results = fuse_results(dense, sparse, top_k=10)

    assert len(results) <= 10


def test_fuse_results_dense_priority() -> None:
    """When a chunk appears in both lists, text must come from the dense result."""
    dense_text = "dense version of the text"
    sparse_text = "sparse version of the text"

    dense = [_dense("shared", 1, text=dense_text)]
    sparse = [_sparse("shared", 1, text=sparse_text)]

    results = fuse_results(dense, sparse)

    assert len(results) == 1
    assert results[0].text == dense_text
