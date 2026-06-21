# retrieval/schemas.py

from enum import Enum

from pydantic import BaseModel


class RetrievalMethod(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its score, rank, and retrieval provenance."""

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict
    retrieval_method: RetrievalMethod


class RetrievalRequest(BaseModel):
    """Parameters for a retrieval query."""

    query: str
    top_k: int = 10
    collection_name: str = "rag_hybrid"
