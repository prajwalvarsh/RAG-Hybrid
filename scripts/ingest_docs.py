# scripts/ingest_docs.py
import logging
from pathlib import Path
from ingest.loader import load_all
from ingest.chunker import chunk_document
from ingest.embedder import embed_chunks
from ingest.indexer import index_chunks
from ingest.schemas import ChunkStrategy

logging.basicConfig(level=logging.INFO)

docs = load_all("data/raw")
print(f"Loaded {len(docs)} documents")

all_chunks = []
for doc in docs:
    chunks = chunk_document(doc, ChunkStrategy.FIXED)
    all_chunks.extend(chunks)

print(f"Created {len(all_chunks)} chunks")

embedded = embed_chunks(all_chunks)
print(f"Embedded {len(embedded)} chunks")

index_chunks(embedded)
print("Indexed successfully")