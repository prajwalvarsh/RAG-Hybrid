# RAG Hybrid Search — Project Brief

## What this project is
A production-grade RAG (Retrieval-Augmented Generation) system that answers
questions from a document corpus using hybrid search (dense vector + sparse
BM25), a reranker, and grounded generation with citation verification.

Built as a portfolio project targeting AI/ML Engineer roles for me (Prajwal Amoghavarsh).

## Stack
- Python 3.10+
- FastAPI — API layer
- ChromaDB — vector store (dense retrieval)
- rank_bm25 — sparse keyword retrieval
- BAAI/bge-base-en-v1.5 — embeddings (local, via sentence-transformers)
- Reciprocal Rank Fusion (RRF) — combining dense + sparse results
- RAGAS — evaluation metrics
- Docker — containerization

## Repo layout
ingest/       → document loading, chunking, embedding, indexing
retrieval/    → hybrid search logic (dense + sparse + rerank)
generation/   → prompt assembly, LLM call, citation verification
eval/         → golden test set and evaluation runner
api/          → FastAPI routes
data/raw/     → source documents (do not modify programmatically)
tests/        → unit tests per module
docs/         → SKILLS.md, agents.md

## Current phase
Phase 1 — Ingestion pipeline

## Key decisions already made

- Chunking strategy: compare fixed-size, structure-aware, and semantic in Phase 1.
  Late chunking (dynamic) documented as a future improvement after eval results.
- RRF weights: start at 0.7 dense / 0.3 sparse, tune after eval
- Embedding model: text-embedding-3-small (cost vs quality tradeoff)
- No LangChain — building retrieval logic from scratch to understand it
- Embeddings normalized at encode time (normalize_embeddings=True)
  BGE models are trained for cosine similarity — unit vectors ensure
  retrieval ranks by semantic direction not vector magnitude
- ChromaDB collection uses cosine distance (hnsw:space: cosine)
  Consistent with normalized embeddings — do not change to L2


## Coding conventions
- Pydantic BaseModel for all data contracts, never TypedDict or dataclass
- Enums for all fixed-value fields (FileType, ChunkStrategy, RetrievalMethod)
- Lazy singleton pattern for expensive resources (model, ChromaDB client)
- logging not print statements everywhere
- Docstrings on every function explaining what and why
- One module per responsibility — no cross-module side effects


## Rules for Claude
- Never modify anything inside eval/golden/ — that is the ground truth
- Never change the Pydantic schemas without asking me first
- One module at a time — do not refactor across folders in a single step
- Always add docstrings to every function you write
- When I say "explain this" — explain it in simple terms before showing code