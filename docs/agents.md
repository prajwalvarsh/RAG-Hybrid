# Agents — RAG Hybrid Search

Each agent below is a scoped task for Claude Code.
One agent = one focused session = one module touched.
Never run two agents in the same Claude Code session.

---

## Agent 1: Ingestion — Document Loader
**Scope:** ingest/loader.py
**Reads:** data/raw/
**Writes:** ingest/loader.py, tests/test_loader.py

**Task prompt:**
"Read docs/SKILLS.md Skill 1 (Chunking). Then read CLAUDE.md.
Build ingest/loader.py that loads .txt, .md, and .pdf files
from data/raw/ and returns a list of Document objects using
the Pydantic schema in ingest/schemas.py. Add docstrings.
Write tests in tests/test_loader.py."

---

## Agent 2: Ingestion — Chunker
**Scope:** ingest/chunker.py
**Reads:** ingest/loader.py, ingest/schemas.py
**Writes:** ingest/chunker.py, tests/test_chunker.py

**Task prompt:**
"Read docs/SKILLS.md Skill 1 (Chunking). Then read CLAUDE.md.
Build ingest/chunker.py implementing all three strategies:
fixed-size, structure-aware, and semantic. Each returns a
list of Chunk objects as defined in ingest/schemas.py.
Write tests in tests/test_chunker.py."

---

## Agent 3: Ingestion — Embedder + Indexer
**Scope:** ingest/embedder.py, ingest/indexer.py
**Reads:** ingest/chunker.py, ingest/schemas.py
**Writes:** ingest/embedder.py, ingest/indexer.py

**Task prompt:**
"Read CLAUDE.md. Build ingest/embedder.py that takes a list
of Chunk objects and calls OpenAI text-embedding-3-small to
return embeddings. Build ingest/indexer.py that stores chunks
and their embeddings in ChromaDB. Add docstrings to everything."

---

## Agent 4: Retrieval — Dense Search
**Scope:** retrieval/dense.py
**Reads:** ingest/indexer.py, ingest/schemas.py
**Writes:** retrieval/dense.py, retrieval/schemas.py

**Task prompt:**
"Read docs/SKILLS.md Skill 2 (Hybrid Search). Then read CLAUDE.md.
Build retrieval/dense.py that takes a query string, embeds it
using OpenAI text-embedding-3-small, and returns the top-k
chunks from ChromaDB by cosine similarity. Return a list of
RetrievalResult objects defined in retrieval/schemas.py."

---

## Agent 5: Retrieval — Sparse Search (BM25)
**Scope:** retrieval/sparse.py
**Reads:** retrieval/schemas.py
**Writes:** retrieval/sparse.py

**Task prompt:**
"Read docs/SKILLS.md Skill 2 (Hybrid Search). Then read CLAUDE.md.
Build retrieval/sparse.py that builds a BM25 index over all
indexed chunks using rank_bm25. Takes a query string, returns
top-k chunks as RetrievalResult objects. The BM25 index should
be buildable from the same chunks stored in ChromaDB."

---

## Agent 6: Retrieval — RRF Fusion + Reranker
**Scope:** retrieval/fusion.py, retrieval/reranker.py
**Reads:** retrieval/dense.py, retrieval/sparse.py, retrieval/schemas.py
**Writes:** retrieval/fusion.py, retrieval/reranker.py

**Task prompt:**
"Read docs/SKILLS.md Skill 2 and Skill 3. Then read CLAUDE.md.
Build retrieval/fusion.py that takes dense results and sparse
results and combines them using RRF (weights: 0.7 dense, 0.3
sparse, smoothing constant 60). Then build retrieval/reranker.py
that takes the top-20 fused results and reranks them using
cross-encoder/ms-marco-MiniLM-L-6-v2. Return top-5."

---

## Agent 7: Generation — Grounded Prompt + Citation Check
**Scope:** generation/generator.py, generation/citation.py
**Reads:** retrieval/reranker.py, retrieval/schemas.py
**Writes:** generation/generator.py, generation/citation.py
         generation/schemas.py

**Task prompt:**
"Read docs/SKILLS.md Skill 4 (Citation Verification). Read CLAUDE.md.
Build generation/generator.py that takes a query and top-5
reranked chunks, assembles a grounded prompt instructing the
LLM to only answer from context and cite with [1][2] etc,
calls gpt-4o-mini, and returns a GenerationResult object.
Build generation/citation.py that verifies each citation and
adds a support_score to the result."

---

## Agent 8: Evaluation — Runner
**Scope:** eval/runner.py
**Reads:** eval/golden/, generation/generator.py
**Writes:** eval/runner.py, eval/results/

**Task prompt:**
"Read docs/SKILLS.md Skill 5 (Evaluation). Read CLAUDE.md.
Build eval/runner.py that loads every question-answer pair
from eval/golden/golden.json, runs each question through the
full pipeline (ingest → retrieval → generation), scores with
RAGAS metrics (faithfulness, answer relevancy, context
precision, context recall), and writes results to
eval/results/ as JSON. Never modify eval/golden/."

---

## Agent 9: API Layer
**Scope:** api/routes.py, api/main.py
**Reads:** generation/generator.py, ingest/indexer.py
**Writes:** api/routes.py, api/main.py, api/schemas.py

**Task prompt:**
"Read CLAUDE.md. Build a FastAPI app in api/main.py with these
routes defined in api/routes.py:
POST /v1/ask — takes a query, returns a GenerationResult
POST /v1/ingest — triggers ingestion of data/raw/
GET  /v1/health — returns status
Add OpenAPI docstrings to every route."

---

## How to use these agents

1. Open Claude Code: `claude`
2. Copy the task prompt from the agent you're on
3. Paste it into Claude Code
4. Review every file Claude creates before saying "looks good"
5. Run the tests before moving to the next agent