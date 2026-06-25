"""Pydantic request/response models for the RAG Hybrid API layer."""

from pydantic import BaseModel

from config import settings
from generation.schemas import CitationStatus


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
