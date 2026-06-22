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
    """Parameters for a retrieval query.

    retrieval_top_k controls how many candidates each retriever returns.
    fusion_top_k controls how many fused results enter the reranker.
    rerank_top_k controls how many results the reranker returns to generation.
    """

    query: str
    retrieval_top_k: int = 10
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    collection_name: str = "rag_hybrid"
