# RAG Hybrid Search — Project Brief

## What this project is
A production-grade RAG (Retrieval-Augmented Generation) system that answers
questions from a document corpus using hybrid search (dense vector + sparse
BM25), a reranker, and grounded generation with citation verification.

Built as a portfolio project targeting AI/ML Engineer roles (Prajwal Amoghavarsh).

## Stack
- Python 3.11 (3.13 incompatible with ragas — pinned in .python-version)
- uv — package manager
- FastAPI — API layer
- ChromaDB — vector store (dense retrieval, cosine distance)
- rank_bm25 — sparse keyword retrieval
- BAAI/bge-base-en-v1.5 — local embeddings via sentence-transformers
- Reciprocal Rank Fusion (RRF) — combining dense + sparse results
- cross-encoder/ms-marco-MiniLM-L-6-v2 — reranker
- llama-3.1-8b-instant via Groq API — generation + citation verification
  (OpenAI-compatible interface, free tier)
- langchain-openai pointed at Groq base_url — used only for RAGAS eval
  (langchain-groq conflicts with groq>=1.5.0 — base_url workaround)
- RAGAS 0.1.21 — evaluation metrics
- Docker — containerization (pending)

## Repo layout
ingest/       → document loading, chunking, embedding, indexing
retrieval/    → hybrid search logic (dense + sparse + rerank)
generation/   → prompt assembly, LLM call, citation verification
eval/         → golden test set and evaluation runner
api/          → FastAPI routes (Agent 9 — pending)
data/raw/     → source documents (gitignored, do not modify programmatically)
scripts/      → one-off ingestion scripts
tests/        → unit tests per module (75 passing)
docs/         → SKILLS.md, agents.md, BUGS.md, IMPROVEMENTS.md

## Current phase
Phase 4 — API layer (FastAPI)

## Completed phases
Phase 1 — Ingestion pipeline ✓
  ingest/model.py      shared embedding singleton (BAAI/bge-base-en-v1.5)
  ingest/schemas.py    Pydantic contracts — Document, Chunk, EmbeddedChunk
  ingest/loader.py     txt, md, pdf loading via pypdf
  ingest/chunker.py    fixed, structural, semantic chunking strategies
  ingest/embedder.py   batch normalized embeddings (normalize_embeddings=True)
  ingest/indexer.py    ChromaDB persistent storage with cosine distance

Phase 2 — Retrieval pipeline ✓
  retrieval/schemas.py   RetrievalResult, RetrievalMethod enum
                         retrieval_top_k=50, fusion_top_k=20, rerank_top_k=5
  retrieval/dense.py     ChromaDB cosine similarity retrieval
  retrieval/sparse.py    BM25 with lazy singleton, zero-score filtering
  retrieval/fusion.py    RRF (0.7 dense / 0.3 sparse, k=60, top_k=20)
  retrieval/reranker.py  cross-encoder, rerank_candidate_k=20, rerank_top_k=5
                         explicit batch_size=32

Phase 3 — Generation + Evaluation ✓
  generation/schemas.py   CitationStatus enum, CitationVerification, GenerationResult
  generation/generator.py grounded prompt + Groq llama-3.1-8b-instant
  generation/citation.py  batched verification (max_tokens=500) with fallback
  eval/golden/golden.json 15 hand-written Q&A pairs (Reliance AGM 2026 doc)
  eval/runner.py          RAGAS 0.1.21 runner with RunConfig(max_workers=1)
                          AnswerRelevancy(strictness=1) for Groq compatibility

## Eval baseline (run 2 — sparse+dense hybrid, 6 bugs fixed)
faithfulness:      0.689
context_precision: 0.717
context_recall:    0.744
answer_relevancy:  pending (strictness=1 fix applied, run 3 needed)
unanswerable:      100% correct (4/4)

## Architectural decisions

- No LangChain in retrieval — built from scratch to understand deeply
  LangChain used only inside RAGAS eval (separate concern)
- BAAI/bge-base-en-v1.5 over OpenAI embeddings
  Free, local, top-3 MTEB retrieval benchmark, runs on 16GB RAM
- Embeddings normalized at encode time (normalize_embeddings=True)
  BGE trained for cosine similarity — unit vectors required
- ChromaDB collection uses cosine distance (hnsw:space: cosine)
  Coupled with normalized embeddings — do not change to L2
- RRF weights: 0.7 dense / 0.3 sparse, k=60 (TREC validated default)
- Groq llama-3.1-8b-instant over OpenAI
  OpenAI billing unavailable — Groq free tier, OpenAI-compatible API
- langchain-openai with custom base_url for RAGAS
  langchain-groq requires groq<1.0.0, conflicts with groq>=1.5.0
- Python 3.11 pinned — RAGAS incompatible with 3.13
- Singleton caches use dict[str, Entry] keyed by collection_name
  Supports multiple collections in eval phase

## Coding conventions
- Pydantic BaseModel for all data contracts, never TypedDict or dataclass
- Enums for all fixed-value fields (FileType, ChunkStrategy, RetrievalMethod)
- Lazy singleton pattern for expensive resources (model, ChromaDB client)
- logging not print statements everywhere
- Docstrings on every function explaining what and why
- One module per responsibility — no cross-module side effects
- normalize_embeddings=True on all model.encode() calls

## Rules for Claude
- Never modify anything inside eval/golden/ — that is the ground truth
- Never change the Pydantic schemas without asking first
- One module at a time — do not refactor across folders in a single step
- Always add docstrings to every function
- When I say "explain this" — explain in simple terms before showing code
- Use uv for all package management — never pip directly

## Known issues (see docs/BUGS.md for full details)
- BUG-002: answer_relevancy nan — fixed with strictness=1, needs run 3
- Re-ingestion requires deleting chroma_db/ first — UUID per run breaks upsert

## Environment
- Python 3.11.14 via uv
- GROQ_API_KEY in .env
- chroma_db/ at project root (gitignored)
- data/raw/ gitignored — add documents manually