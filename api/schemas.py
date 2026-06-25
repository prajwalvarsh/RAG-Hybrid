"""Pydantic request/response models for the RAG Hybrid API layer."""

from pydantic import BaseModel

from config import settings
from generation.schemas import CitationStatus
from ingest.schemas import ChunkStrategy


class QueryRequest(BaseModel):
    """Incoming request body for POST /query."""

    question: str
    collection_name: str = settings.default_collection


class CitationResponse(BaseModel):
    """API representation of a single verified citation.

    Mirrors CitationVerification from generation/schemas.py so the API
    surface stays decoupled from internal schema changes.
    """

    chunk_id: str
    citation_number: int
    claim: str
    supported: CitationStatus
    confidence: float


class QueryResponse(BaseModel):
    """Response body returned by POST /query."""

    question: str
    answer: str
    citations: list[CitationResponse]
    support_score: float
    retrieval_method: str
    latency_ms: float


class IngestRequest(BaseModel):
    """Body for a programmatic (non-multipart) ingest request."""

    collection_name: str
    strategy: ChunkStrategy


class FileIngestResult(BaseModel):
    """Result for a single file within a POST /ingest batch.

    status is "success" when the full pipeline completed, or "error" when
    any validation or pipeline stage failed for this file.  The error field
    carries a human-readable reason when status == "error" and is None
    otherwise.
    """

    filename: str
    collection_name: str
    strategy: ChunkStrategy
    chunk_count: int
    elapsed_ms: float
    status: str
    error: str | None = None


class IngestResponse(BaseModel):
    """Response body returned by POST /ingest.

    Always returns HTTP 200.  Per-file failures are reported as
    FileIngestResult entries with status == "error" rather than as HTTP
    error codes, which allows partial-success batches.
    """

    files: list[FileIngestResult]
