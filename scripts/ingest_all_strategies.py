"""Ingest data/raw into three separate ChromaDB collections, one per chunk strategy.

Run with:
    uv run python scripts/ingest_all_strategies.py
"""

import logging
from pathlib import Path

import chromadb

from ingest.loader import load_all
from ingest.chunker import chunk_document
from ingest.embedder import embed_chunks
from ingest.indexer import index_chunks, _CHROMA_PATH
from ingest.schemas import ChunkStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGY_COLLECTIONS: dict[ChunkStrategy, str] = {
    ChunkStrategy.FIXED: "rag_fixed",
    ChunkStrategy.STRUCTURAL: "rag_structural",
    ChunkStrategy.SEMANTIC: "rag_semantic",
}

RAW_DATA_DIR = "data/raw"


def collection_doc_count(collection_name: str) -> int:
    """Return the number of documents already stored in a ChromaDB collection.

    Creates the collection with cosine distance if it does not exist yet, so
    the count is always 0 for brand-new collections. This mirrors the settings
    used by index_chunks so both functions operate on the same collection.

    Args:
        collection_name: Name of the ChromaDB collection to inspect.

    Returns:
        Integer count of documents currently in the collection.
    """
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection.count()


def ingest_strategy(
    docs: list,
    strategy: ChunkStrategy,
    collection_name: str,
) -> None:
    """Chunk, embed, and index documents for a single strategy into its collection.

    Skips the collection entirely if it already contains any documents, logging
    a message so the caller can see which strategies were bypassed.

    Logs chunk counts before embedding (i.e. after chunking) and after indexing
    to give visibility into how many pieces each strategy produces.

    Args:
        docs: Pre-loaded list of Document objects from ingest/loader.py.
        strategy: ChunkStrategy enum value (FIXED, STRUCTURAL, or SEMANTIC).
        collection_name: Name of the ChromaDB collection to write into.
    """
    existing = collection_doc_count(collection_name)
    if existing > 0:
        logger.info(
            "Collection '%s' already has %d documents — skipping %s ingestion.",
            collection_name,
            existing,
            strategy.value,
        )
        return

    logger.info("=== Strategy: %s → collection '%s' ===", strategy.value, collection_name)

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, strategy)
        all_chunks.extend(chunks)

    logger.info(
        "[%s] Chunk count before indexing: %d",
        strategy.value,
        len(all_chunks),
    )

    embedded = embed_chunks(all_chunks)
    index_chunks(embedded, collection_name=collection_name)

    after_count = collection_doc_count(collection_name)
    logger.info(
        "[%s] Chunk count after indexing: %d",
        strategy.value,
        after_count,
    )


def main() -> None:
    """Load documents once, then ingest into each strategy collection in turn.

    Documents are loaded a single time from RAW_DATA_DIR to avoid redundant
    I/O. Each strategy receives the same Document list and produces its own
    independently chunked and indexed collection.
    """
    logger.info("Loading documents from '%s' …", RAW_DATA_DIR)
    docs = load_all(RAW_DATA_DIR)
    logger.info("Loaded %d document(s).", len(docs))

    if not docs:
        logger.error("No documents found in '%s'. Aborting.", RAW_DATA_DIR)
        return

    for strategy, collection_name in STRATEGY_COLLECTIONS.items():
        ingest_strategy(docs, strategy, collection_name)

    logger.info("All strategies processed.")


if __name__ == "__main__":
    main()
