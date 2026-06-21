"""Store and retrieve EmbeddedChunk objects in a ChromaDB persistent collection."""

import logging
from pathlib import Path

import chromadb

from ingest.schemas import EmbeddedChunk

logger = logging.getLogger(__name__)

# chroma_db/ lives at the project root (one level above this file's ingest/ package)
_CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"


def index_chunks(
    embedded_chunks: list[EmbeddedChunk],
    collection_name: str = "rag_hybrid",
) -> None:
    """Upsert a list of EmbeddedChunk objects into a ChromaDB collection.

    Opens (or creates) a persistent ChromaDB client at chroma_db/ in the
    project root, then upserts all chunks in a single call. Existing records
    with the same chunk_id are overwritten; new records are inserted.

    Collection is created with cosine distance metric (hnsw:space: cosine)
    to match the normalized embeddings produced by ingest/embedder.py.
    These two settings are coupled — changing one without the other
    produces inconsistent retrieval results.

    Args:
        embedded_chunks: List of EmbeddedChunk objects to store.
        collection_name: Name of the ChromaDB collection (default "rag_hybrid").
    """
    if not embedded_chunks:
        logger.warning("index_chunks called with an empty list — nothing to upsert")
        return

    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [ec.chunk.chunk_id for ec in embedded_chunks]
    embeddings = [ec.embedding for ec in embedded_chunks]
    documents = [ec.chunk.text for ec in embedded_chunks]
    metadatas = [ec.chunk.metadata.model_dump() for ec in embedded_chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        "Upserted %d chunks into ChromaDB collection '%s' at %s",
        len(embedded_chunks),
        collection_name,
        _CHROMA_PATH,
    )


def get_collection(collection_name: str = "rag_hybrid") -> chromadb.Collection:
    """Return a ChromaDB collection, creating it if it does not exist.

    Retrieval modules use this function to obtain a handle on the indexed
    chunk collection without needing to know the storage path.

    Args:
        collection_name: Name of the ChromaDB collection (default "rag_hybrid").

    Returns:
        A chromadb.Collection object ready for querying.
    """
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Opened ChromaDB collection '%s'", collection_name)
    return collection
