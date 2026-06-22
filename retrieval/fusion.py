"""Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval results."""

import logging

from retrieval.schemas import RetrievalMethod, RetrievalResult

logger = logging.getLogger(__name__)


def fuse_results(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievalResult]:
    """Combine dense and sparse retrieval lists using Reciprocal Rank Fusion.

    RRF formula for each chunk:
        RRF_score = (dense_weight * 1/(rank_dense + k))
                  + (sparse_weight * 1/(rank_sparse + k))

    If a chunk appears in only one list, its contribution from the missing
    list is 0.0 — it is not penalised for absence, it simply misses the
    boost from that side.

    Why k=60?  The constant k prevents the top-ranked document from
    dominating by an extreme amount.  With k=60, rank-1 gives 1/61 ≈ 0.016
    and rank-100 gives 1/160 ≈ 0.006, a ~2.6× spread rather than ∞.
    The value 60 was validated empirically across TREC benchmarks and is the
    conventional default for RRF (Cormack et al., 2009).

    Why top_k=20?  top_k caps the fused candidate list before reranking.
    Reranking is expensive — passing 20 candidates instead of 100 reduces
    cross-encoder inference by 5x with negligible recall loss since RRF
    already surfaced the best candidates.

    Dense results take priority for text and metadata when a chunk appears
    in both lists, because the dense retriever has richer metadata (distances,
    full ChromaDB metadatas) compared to the sparse retriever.

    Args:
        dense_results:  Ranked list from dense (vector) retrieval.
        sparse_results: Ranked list from sparse (BM25) retrieval.
        dense_weight:   Multiplicative weight applied to the dense RRF term.
        sparse_weight:  Multiplicative weight applied to the sparse RRF term.
        k:              Smoothing constant that dampens rank-position extremes.
        top_k:          Maximum number of fused candidates to return, sized for
                        efficient downstream reranking.

    Returns:
        Up to top_k RetrievalResult objects ordered by RRF score descending,
        with retrieval_method=HYBRID and rank starting at 1.
    """
    # Build lookup: chunk_id → result, from each list.
    dense_map: dict[str, RetrievalResult] = {r.chunk_id: r for r in dense_results}
    sparse_map: dict[str, RetrievalResult] = {r.chunk_id: r for r in sparse_results}

    all_ids = set(dense_map) | set(sparse_map)

    if not all_ids:
        return []

    scored: list[tuple[str, float]] = []
    for chunk_id in all_ids:
        dense_term = 0.0
        if chunk_id in dense_map:
            dense_rank = dense_map[chunk_id].rank
            dense_term = dense_weight / (dense_rank + k)

        sparse_term = 0.0
        if chunk_id in sparse_map:
            sparse_rank = sparse_map[chunk_id].rank
            sparse_term = sparse_weight / (sparse_rank + k)

        rrf_score = dense_term + sparse_term
        scored.append((chunk_id, rrf_score))

    scored.sort(key=lambda t: t[1], reverse=True)
    scored = scored[:top_k]

    results: list[RetrievalResult] = []
    for new_rank, (chunk_id, rrf_score) in enumerate(scored, start=1):
        # Dense takes priority for text and metadata; fall back to sparse.
        source = dense_map.get(chunk_id) or sparse_map[chunk_id]
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=source.text,
                score=rrf_score,
                rank=new_rank,
                metadata=source.metadata,
                retrieval_method=RetrievalMethod.HYBRID,
            )
        )

    logger.debug(
        "RRF fused %d dense + %d sparse → %d unique chunks",
        len(dense_results),
        len(sparse_results),
        len(results),
    )

    return results
