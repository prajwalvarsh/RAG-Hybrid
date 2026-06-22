# generation/schemas.py

from enum import Enum

from pydantic import BaseModel

from retrieval.schemas import RetrievalMethod


class CitationStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


class CitationVerification(BaseModel):
    """Verification result for a single citation in the generated answer."""

    chunk_id: str
    citation_number: int
    claim: str
    supported: CitationStatus
    confidence: float  # 0.0 to 1.0


class GenerationResult(BaseModel):
    """Full result of a RAG generation call, including grounded answer and citation audit."""

    query: str
    answer: str
    citations: list[CitationVerification]
    chunks_used: list[str]   # chunk_ids in rank order
    support_score: float     # supported_citations / total_citations
    model: str               # e.g. "gpt-4o-mini"
    retrieval_method: RetrievalMethod
