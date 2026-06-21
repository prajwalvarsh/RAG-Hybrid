"""Sparse (BM25) retrieval over the ChromaDB corpus."""

import logging

from rank_bm25 import BM25Okapi

from ingest.indexer import get_collection
from retrieval.schemas import RetrievalMethod, RetrievalRequest, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_BM25_INDEX: BM25Okapi | None = None
_BM25_BUILT: bool = False  # True once _build_bm25_index has run (even for empty corpus)
_BM25_CHUNK_IDS: list[str] = []
_BM25_TEXTS: list[str] = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_bm25_index(collection_name: str) -> None:
    """Fetch all chunks from ChromaDB and build a BM25Okapi index in memory.

    Tokenization is intentionally simple: lowercase the text then split on
    whitespace. The same tokenization MUST be applied at query time in
    retrieve_sparse. If the two tokenizations diverge (e.g. one lowercases
    and the other does not), BM25 term frequencies will never match and every
    query will return zero scores.

    The built index and supporting lists are stored in the module-level
    globals so the index survives across calls without re-fetching from
    ChromaDB.

    Args:
        collection_name: Name of the ChromaDB collection to index.
    """
    global _BM25_INDEX, _BM25_BUILT, _BM25_CHUNK_IDS, _BM25_TEXTS

    collection = get_collection(collection_name)
    data = collection.get(include=["documents", "metadatas"])

    ids: list[str] = data.get("ids", [])
    documents: list[str] = data.get("documents", []) or []

    if not documents:
        logger.warning(
            "Collection '%s' returned no documents — BM25 index will be empty",
            collection_name,
        )
        _BM25_INDEX = None
        _BM25_BUILT = True
        _BM25_CHUNK_IDS = []
        _BM25_TEXTS = []
        return

    tokenized_corpus = [doc.lower().split() for doc in documents]

    _BM25_INDEX = BM25Okapi(tokenized_corpus)
    _BM25_BUILT = True
    _BM25_CHUNK_IDS = list(ids)
    _BM25_TEXTS = list(documents)

    logger.info(
        "BM25 index built for collection '%s' with %d documents",
        collection_name,
        len(documents),
    )


def get_bm25_index(
    collection_name: str,
    force_rebuild: bool = False,
) -> BM25Okapi:
    """Return the cached BM25Okapi index, building it on first call.

    Acts as a lazy singleton: the first call triggers _build_bm25_index,
    subsequent calls return the cached instance immediately. Pass
    force_rebuild=True when the underlying ChromaDB collection has changed
    (e.g. after re-ingestion) to discard the cache and rebuild from scratch.

    Args:
        collection_name: ChromaDB collection whose documents back this index.
        force_rebuild: If True, discard the existing index and rebuild.

    Returns:
        A BM25Okapi instance ready for get_scores() calls.
    """
    global _BM25_INDEX

    if not _BM25_BUILT or force_rebuild:
        _build_bm25_index(collection_name)

    if _BM25_INDEX is None:
        raise RuntimeError(
            "BM25 index is empty — ingest documents before querying. "
            f"Collection '{collection_name}' returned no documents."
        )

    return _BM25_INDEX


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------


def retrieve_sparse(request: RetrievalRequest) -> list[RetrievalResult]:
    """Score every indexed chunk against *request.query* using BM25 and return the top-k.

    Tokenization mirrors _build_bm25_index: lowercase then split on
    whitespace. The two sides MUST use identical tokenization — if indexing
    lowercases but querying does not (or vice-versa), BM25 term overlap is
    zero and all scores are 0.0, producing no useful results.

    Zero scores are filtered out before ranking because BM25 assigns 0.0 to
    any chunk that shares no tokens with the query. Including them would pad
    the result list with irrelevant chunks and pollute RRF fusion downstream.

    Args:
        request: RetrievalRequest with query, top_k, and collection_name.

    Returns:
        List of RetrievalResult ordered by BM25 score descending with rank
        starting at 1. Returns an empty list when the index is empty or when
        no chunk shares any token with the query.
    """
    try:
        index = get_bm25_index(request.collection_name)
    except RuntimeError:
        logger.warning(
            "BM25 index for collection '%s' is empty — returning no results for query: %r",
            request.collection_name,
            request.query,
        )
        return []

    tokenized_query = request.query.lower().split()
    scores: list[float] = index.get_scores(tokenized_query).tolist()

    # Pair each chunk with its score, filter zeros, sort descending.
    scored = [
        (chunk_id, text, score)
        for chunk_id, text, score in zip(_BM25_CHUNK_IDS, _BM25_TEXTS, scores)
        if score > 0.0
    ]
    scored.sort(key=lambda t: t[2], reverse=True)

    top = scored[: request.top_k]

    results: list[RetrievalResult] = []
    for rank_zero, (chunk_id, text, score) in enumerate(top):
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                rank=rank_zero + 1,
                metadata={},
                retrieval_method=RetrievalMethod.SPARSE,
            )
        )

    return results
