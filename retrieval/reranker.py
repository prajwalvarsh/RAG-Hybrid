"""Cross-encoder reranking over fused hybrid retrieval results."""

import logging

from sentence_transformers import CrossEncoder

from retrieval.schemas import RetrievalMethod, RetrievalResult

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Return the cached CrossEncoder, loading it on first call.

    Lazy singleton: the model is downloaded and loaded into memory only when
    this function is first called, then cached in the module-level _RERANKER
    global.  Subsequent calls return the cached instance immediately, avoiding
    the multi-second load penalty on every rerank() invocation.

    Returns:
        A CrossEncoder instance ready for predict() calls.
    """
    global _RERANKER

    if _RERANKER is None:
        logger.info("Loading reranker model: %s", _RERANKER_MODEL)
        _RERANKER = CrossEncoder(_RERANKER_MODEL)

    return _RERANKER


def rerank(
    query: str,
    results: list[RetrievalResult],
    rerank_candidate_k: int = 20,
    rerank_top_k: int = 5,
) -> list[RetrievalResult]:
    """Score each (query, chunk) pair with a cross-encoder and return the top-k.

    A cross-encoder reads the query and chunk text jointly — unlike a
    bi-encoder that encodes them independently — which produces more accurate
    relevance scores at the cost of higher latency.  This is intentionally
    a second-pass step: run RRF first to cut candidates to ~20, then rerank
    those 20 rather than the full corpus.

    rerank_candidate_k caps how many candidates the cross-encoder scores.
    Scoring is O(n) inference passes — capping at 20 gives 5x speedup over
    scoring 100 candidates with negligible recall loss since RRF already
    surfaced the best candidates.

    batch_size=32 makes inference deterministic and memory-bounded.
    Implicit batch size varies by library version and loads all pairs into
    memory simultaneously on large inputs.

    Scoring: model.predict([(query, text), ...]) returns a float array in the
    same order as the input pairs.  We sort descending (higher = more relevant)
    and keep the rerank_top_k results.

    The score field of each returned result is replaced with the reranker score
    so downstream consumers see a single authoritative relevance number.
    retrieval_method stays HYBRID because the input was fused hybrid results.

    Args:
        query:               The user's query string.
        results:             Fused RetrievalResult list, typically from fuse_results().
        rerank_candidate_k:  Maximum number of candidates passed to the cross-encoder.
        rerank_top_k:        Maximum number of results to return after reranking.

    Returns:
        Up to rerank_top_k RetrievalResult objects ordered by reranker score
        descending, with rank starting at 1 and score updated to the
        reranker score.  Returns an empty list if results is empty.
    """
    if not results:
        return []

    model = _get_reranker()

    candidates = results[:rerank_candidate_k]
    pairs = [(query, r.text) for r in candidates]
    scores: list[float] = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    ).tolist()

    # Pair each candidate with its reranker score and sort descending.
    ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)

    reranked: list[RetrievalResult] = []
    for new_rank, (result, reranker_score) in enumerate(ranked[:rerank_top_k], start=1):
        reranked.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=reranker_score,
                rank=new_rank,
                metadata=result.metadata,
                retrieval_method=RetrievalMethod.HYBRID,
            )
        )

    logger.debug(
        "Reranked %d candidates → top %d results for query: %r",
        len(candidates),
        len(reranked),
        query,
    )

    return reranked
