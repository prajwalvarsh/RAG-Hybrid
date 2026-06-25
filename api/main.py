"""FastAPI application for the RAG Hybrid Search system.

Exposes three endpoints:
  POST /query       — run the full RAG pipeline and return a grounded answer
  GET  /health      — service liveness + configuration summary
  GET  /collections — list all ChromaDB collections with document counts
"""

import logging
import time

import chromadb
from fastapi import FastAPI, HTTPException

from api.schemas import CitationResponse, QueryRequest, QueryResponse
from config import settings
from eval.runner import run_pipeline
from ingest.indexer import _CHROMA_PATH

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Hybrid Search",
    description="Hybrid dense+sparse retrieval with reranking and grounded generation.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collection_counts() -> dict[str, int]:
    """Return a mapping of collection_name → document count for every ChromaDB collection.

    Creates a fresh PersistentClient on each call so this helper is
    safe to call from health/collections endpoints without holding a
    long-lived client reference.
    """
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collections = client.list_collections()
    return {col.name: col.count() for col in collections}


def _execute_query(question: str, collection_name: str) -> tuple[dict, float]:
    """Call run_pipeline and measure total wall-clock latency.

    run_pipeline() from eval.runner bundles embed → retrieve → rerank →
    generate into one call.  Per-stage logs are emitted at INFO level by
    the individual stage modules (ingest.model, retrieval.*, generation.*).
    Total latency is returned as latency_ms for the API response.

    Args:
        question:        Natural-language query from the client.
        collection_name: ChromaDB collection to search.

    Returns:
        Tuple of (pipeline_result_dict, latency_ms).  result_dict is
        empty if the pipeline failed.
    """
    t0 = time.perf_counter()
    result = run_pipeline(question, collection_name=collection_name)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Pipeline completed in %.1f ms | collection=%s | question=%r",
        latency_ms,
        collection_name,
        question,
    )
    return result, latency_ms


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline for a natural-language question.

    Accepts a question and an optional collection name, passes them through
    the full hybrid retrieval + rerank + generation pipeline, and returns
    a grounded answer with citation metadata and total latency.

    Returns HTTP 503 if the pipeline fails to produce a result (e.g. both
    retrievers fail or the LLM call errors out).

    Args:
        body: QueryRequest containing the question and collection_name.

    Returns:
        QueryResponse with answer, citations, support score, and latency.

    Raises:
        HTTPException 503: If run_pipeline returns an empty dict.
    """
    logger.info(
        "POST /query | question=%r | collection=%s",
        body.question,
        body.collection_name,
    )

    result, latency_ms = _execute_query(body.question, body.collection_name)

    if not result:
        logger.error(
            "Pipeline returned empty result for question=%r", body.question
        )
        raise HTTPException(
            status_code=503,
            detail="Pipeline failed to produce a result. Check logs for details.",
        )

    citations = [CitationResponse(**c) for c in result.get("citations", [])]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        citations=citations,
        support_score=result["support_score"],
        retrieval_method=result["retrieval_method"],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health() -> dict:
    """Return service liveness status and configuration summary.

    Connects to ChromaDB to fetch per-collection document counts so the
    caller can confirm the index is populated.  All config values come from
    the central settings singleton so this endpoint always reflects the
    current runtime configuration.

    Returns:
        Dict with status, model, chroma_path, default_collection, and
        a collections dict mapping collection names to document counts.
    """
    try:
        counts = _collection_counts()
    except Exception as exc:
        logger.warning("ChromaDB unavailable during health check: %s", exc)
        counts = {}

    return {
        "status": "ok",
        "model": settings.llm_model,
        "chroma_path": settings.chroma_path,
        "default_collection": settings.default_collection,
        "collections": counts,
    }


@app.get("/collections")
async def collections() -> list[dict]:
    """List all ChromaDB collections with their document counts.

    Useful for confirming which chunk-strategy collections (rag_fixed,
    rag_structural, rag_semantic, rag_hybrid) have been populated.

    Returns:
        List of dicts, each with 'name' and 'count' keys.
    """
    try:
        counts = _collection_counts()
    except Exception as exc:
        logger.error("Failed to list ChromaDB collections: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB unavailable: {exc}",
        )

    return [{"name": name, "count": count} for name, count in counts.items()]
