"""FastAPI application for the RAG Hybrid Search system.

Exposes three endpoints:
  POST /query       — run the full RAG pipeline and return a grounded answer
  GET  /health      — service liveness + configuration summary
  GET  /collections — list all ChromaDB collections with document counts
"""

import logging
import tempfile
import time
from pathlib import Path

import chromadb
from fastapi import FastAPI, Form, HTTPException, UploadFile

from api.schemas import (
    CitationResponse,
    FileIngestResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from config import settings
from eval.runner import run_pipeline
from ingest.chunker import chunk_document
from ingest.embedder import embed_chunks
from ingest.indexer import _CHROMA_PATH, index_chunks
from ingest.loader import load_document
from ingest.schemas import ChunkStrategy

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


_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


def _ingest_one(
    content: bytes,
    filename: str,
    collection_name: str,
    strategy: ChunkStrategy,
) -> FileIngestResult:
    """Run the full ingest pipeline for a single file's content.

    Writes content to a NamedTemporaryFile, then executes
    load → chunk → embed → index sequentially, removing the temp file
    in a finally block regardless of success or failure.

    Args:
        content:         Raw bytes already read from the upload.
        filename:        Original filename — used only for logging and the
                         result record; the actual file extension determines
                         the loader branch.
        collection_name: ChromaDB collection to write chunks into.
        strategy:        Chunking strategy to apply.

    Returns:
        FileIngestResult with status "success" and populated chunk_count,
        or status "error" with a reason string if any pipeline stage raises.
    """
    ext = Path(filename).suffix.lower()
    t_start = time.perf_counter()
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, prefix="rag_ingest_"
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        logger.info("Saved %s to temp path %s", filename, tmp_path)

        t0 = time.perf_counter()
        document = load_document(str(tmp_path))
        logger.info("load_document %s: %.1f ms", filename, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        chunks = chunk_document(document, strategy)
        logger.info(
            "chunk_document %s (%s): %d chunks in %.1f ms",
            filename,
            strategy.value,
            len(chunks),
            (time.perf_counter() - t0) * 1000,
        )

        t0 = time.perf_counter()
        embedded = embed_chunks(chunks)
        logger.info("embed_chunks %s: %.1f ms", filename, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        index_chunks(embedded, collection_name=collection_name)
        logger.info("index_chunks %s: %.1f ms", filename, (time.perf_counter() - t0) * 1000)

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            "Ingest success | file=%s | collection=%s | chunks=%d | elapsed=%.1f ms",
            filename,
            collection_name,
            len(embedded),
            elapsed_ms,
        )
        return FileIngestResult(
            filename=filename,
            collection_name=collection_name,
            strategy=strategy,
            chunk_count=len(embedded),
            elapsed_ms=elapsed_ms,
            status="success",
        )

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.error("Ingest failed for %s: %s", filename, exc, exc_info=True)
        return FileIngestResult(
            filename=filename,
            collection_name=collection_name,
            strategy=strategy,
            chunk_count=0,
            elapsed_ms=elapsed_ms,
            status="error",
            error=str(exc),
        )

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
            logger.debug("Removed temp file %s", tmp_path)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile],
    collection_name: str = Form(...),
    strategy: ChunkStrategy = Form(...),
) -> IngestResponse:
    """Ingest one or more uploaded documents into a ChromaDB collection.

    Accepts a multipart batch of file uploads (pdf, txt, or md) with a
    shared collection name and chunking strategy as form fields.  Files are
    processed sequentially.  Per-file validation (extension and 10 MB size
    limit) produces an error entry in the results list rather than an HTTP
    error code so that partial-success batches can be reported cleanly.

    Args:
        files:           One or more uploaded files — each must be .pdf,
                         .txt, or .md and at most 10 MB.
        collection_name: ChromaDB collection to write all chunks into.
        strategy:        Chunking strategy applied to every file in the batch.

    Returns:
        IngestResponse with a FileIngestResult per uploaded file, each
        carrying its own status, chunk_count, and elapsed_ms.
    """
    logger.info(
        "POST /ingest | files=%d | collection=%s | strategy=%s",
        len(files),
        collection_name,
        strategy.value,
    )

    results: list[FileIngestResult] = []

    for upload in files:
        filename = upload.filename or "upload"
        ext = Path(filename).suffix.lower()

        logger.info("Processing file %d/%d: %s", len(results) + 1, len(files), filename)

        # --- extension validation ---
        if ext not in _ALLOWED_EXTENSIONS:
            logger.warning("Skipping %s — unsupported extension '%s'", filename, ext)
            results.append(
                FileIngestResult(
                    filename=filename,
                    collection_name=collection_name,
                    strategy=strategy,
                    chunk_count=0,
                    elapsed_ms=0.0,
                    status="error",
                    error=(
                        f"Unsupported file type '{ext}'. "
                        f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
                    ),
                )
            )
            continue

        content = await upload.read()

        # --- size validation ---
        if len(content) > _MAX_FILE_BYTES:
            size_mb = len(content) / (1024 * 1024)
            logger.warning("Skipping %s — %.1f MB exceeds 10 MB limit", filename, size_mb)
            results.append(
                FileIngestResult(
                    filename=filename,
                    collection_name=collection_name,
                    strategy=strategy,
                    chunk_count=0,
                    elapsed_ms=0.0,
                    status="error",
                    error=f"File size {size_mb:.1f} MB exceeds the 10 MB limit.",
                )
            )
            continue

        result = _ingest_one(content, filename, collection_name, strategy)
        results.append(result)

    logger.info(
        "Batch complete | total=%d | success=%d | error=%d",
        len(results),
        sum(1 for r in results if r.status == "success"),
        sum(1 for r in results if r.status == "error"),
    )
    return IngestResponse(files=results)


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
