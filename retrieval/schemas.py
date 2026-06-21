# retrieval/schemas.py

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its score, rank, and retrieval provenance."""

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict
    retrieval_method: str  # "dense" | "sparse" | "hybrid"


class RetrievalRequest(BaseModel):
    """Parameters for a retrieval query."""

    query: str
    top_k: int = 10
    collection_name: str = "rag_hybrid"
