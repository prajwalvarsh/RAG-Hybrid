"""Dense (vector) retrieval using ChromaDB and BAAI/bge-base-en-v1.5 embeddings."""

import logging

from ingest import model as _model_mod
from ingest.indexer import get_collection
from retrieval.schemas import RetrievalMethod, RetrievalRequest, RetrievalResult

logger = logging.getLogger(__name__)


def retrieve_dense(request: RetrievalRequest) -> list[RetrievalResult]:
    """Embed *request.query* and return the top-k closest chunks from ChromaDB.

    Query embedding uses normalize_embeddings=True, which MUST match the
    normalize_embeddings=True used during indexing in ingest/embedder.py.
    Both the indexer and the retriever use BGE models trained for cosine
    similarity. Normalizing to unit vectors ensures every cosine score sits
    on [-1, 1] and ranks purely by semantic direction, not vector magnitude.
    If one side is normalized and the other is not, dot-product distances
    become inconsistent and ranking degrades silently.

    ChromaDB returns results as a dict of parallel lists:
      - ids[0]:        list of chunk IDs (one query → one inner list)
      - documents[0]:  list of chunk texts
      - distances[0]:  list of cosine *distances* (0 = identical, 2 = opposite)
      - metadatas[0]:  list of metadata dicts

    We convert distance to similarity as: score = 1 - distance, then sort
    descending so the most relevant chunk is rank 1.

    Args:
        request: RetrievalRequest containing query, top_k, and collection_name.

    Returns:
        List of RetrievalResult ordered by score descending with rank starting
        at 1. Returns an empty list if the collection is empty.
    """
    collection = get_collection(request.collection_name)

    if collection.count() == 0:
        logger.warning(
            "Collection '%s' is empty — returning no results for query: %r",
            request.collection_name,
            request.query,
        )
        return []

    model = _model_mod.get_embedding_model()
    query_vector: list[float] = model.encode(
        request.query, normalize_embeddings=True
    ).tolist()

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=request.top_k,
        include=["documents", "distances", "metadatas"],
    )

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    distances = raw["distances"][0]
    metadatas = raw["metadatas"][0]

    results: list[RetrievalResult] = []
    for rank_zero, (chunk_id, text, distance, metadata) in enumerate(
        zip(ids, documents, distances, metadatas)
    ):
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=1.0 - distance,
                rank=rank_zero + 1,
                metadata=metadata,
                retrieval_method=RetrievalMethod.DENSE,
            )
        )

    return results
