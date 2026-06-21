"""Tests for ingest/chunker.py."""

import pytest

from ingest.chunker import chunk_document, chunk_fixed, chunk_semantic, chunk_structural
from ingest.schemas import (
    ChunkStrategy,
    Document,
    DocumentMetadata,
    FileType,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_doc() -> Document:
    """A minimal Document for testing fixed and structural chunking."""
    return Document(
        doc_id="test-doc-001",
        text="word " * 600,  # 600 tokens — enough to produce multiple fixed chunks
        metadata=DocumentMetadata(
            file_name="test.txt",
            file_path="/tmp/test.txt",
            file_type=FileType.TXT,
        ),
    )


@pytest.fixture
def markdown_doc() -> Document:
    """A Document with markdown heading structure for structural chunking tests."""
    text = (
        "# Introduction\n\n"
        "This is the introduction section.\n\n"
        "## Background\n\n"
        "This covers the background material.\n\n"
        "### Details\n\n"
        "Fine-grained details go here."
    )
    return Document(
        doc_id="test-doc-002",
        text=text,
        metadata=DocumentMetadata(
            file_name="readme.md",
            file_path="/tmp/readme.md",
            file_type=FileType.MD,
        ),
    )


@pytest.fixture
def sentence_doc() -> Document:
    """A short Document with distinct sentences for semantic chunking tests."""
    text = (
        "The sky is blue. "
        "Clouds float in the atmosphere. "
        "Quantum mechanics describes subatomic particles. "
        "Wave functions collapse upon measurement."
    )
    return Document(
        doc_id="test-doc-003",
        text=text,
        metadata=DocumentMetadata(
            file_name="sentences.txt",
            file_path="/tmp/sentences.txt",
            file_type=FileType.TXT,
        ),
    )


# ---------------------------------------------------------------------------
# Fixed chunking tests
# ---------------------------------------------------------------------------

def test_fixed_chunk_count(simple_doc: Document) -> None:
    """A 600-token document with chunk_size=512 and overlap=50 should yield 2 chunks."""
    chunks = chunk_fixed(simple_doc, chunk_size=512, overlap=50)
    assert len(chunks) == 2


def test_fixed_chunk_ids(simple_doc: Document) -> None:
    """chunk_id must follow the pattern {doc_id}_fixed_{index:04d}."""
    chunks = chunk_fixed(simple_doc, chunk_size=512, overlap=50)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{simple_doc.doc_id}_fixed_{i:04d}"


def test_fixed_overlap(simple_doc: Document) -> None:
    """Consecutive chunks must share approximately `overlap` tokens at their boundary."""
    overlap = 50
    chunks = chunk_fixed(simple_doc, chunk_size=512, overlap=overlap)
    assert len(chunks) >= 2

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    tail_tokens = enc.encode(chunks[0].text)[-overlap:]
    head_tokens = enc.encode(chunks[1].text)[:overlap]
    assert tail_tokens == head_tokens


def test_fixed_token_count_populated(simple_doc: Document) -> None:
    """Every fixed chunk must have token_count set and > 0."""
    chunks = chunk_fixed(simple_doc, chunk_size=512, overlap=50)
    for chunk in chunks:
        assert chunk.token_count is not None
        assert chunk.token_count > 0


def test_fixed_strategy_enum(simple_doc: Document) -> None:
    """strategy field must equal ChunkStrategy.FIXED."""
    chunks = chunk_fixed(simple_doc)
    for chunk in chunks:
        assert chunk.strategy == ChunkStrategy.FIXED


# ---------------------------------------------------------------------------
# Structural chunking tests
# ---------------------------------------------------------------------------

def test_structural_headings(markdown_doc: Document) -> None:
    """Each non-heading chunk must carry its nearest heading in metadata."""
    chunks = chunk_structural(markdown_doc)
    content_chunks = [c for c in chunks if not c.text.startswith("#")]

    assert len(content_chunks) > 0
    for chunk in content_chunks:
        assert chunk.metadata.heading is not None, (
            f"Chunk missing heading: {chunk.text!r}"
        )


def test_structural_heading_values(markdown_doc: Document) -> None:
    """Heading metadata should match the actual section headings in the document."""
    chunks = chunk_structural(markdown_doc)
    headings = {c.metadata.heading for c in chunks if c.metadata.heading}
    assert "Introduction" in headings
    assert "Background" in headings
    assert "Details" in headings


def test_structural_chunk_ids(markdown_doc: Document) -> None:
    """chunk_id must follow the pattern {doc_id}_structural_{index:04d}."""
    chunks = chunk_structural(markdown_doc)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{markdown_doc.doc_id}_structural_{i:04d}"


def test_structural_strategy_enum(markdown_doc: Document) -> None:
    """strategy field must equal ChunkStrategy.STRUCTURAL."""
    chunks = chunk_structural(markdown_doc)
    for chunk in chunks:
        assert chunk.strategy == ChunkStrategy.STRUCTURAL


# ---------------------------------------------------------------------------
# Semantic chunking tests (mocked sentence-transformers model)
# ---------------------------------------------------------------------------

import numpy as np


def _make_embedding(value: float, dim: int = 8) -> list[float]:
    """Create a unit-ish embedding pointing in one direction."""
    return [value] * dim


def _mock_model(mocker, embeddings: list[list[float]]):
    """Patch ingest.model.get_embedding_model to return a mock whose .encode() returns a numpy array."""
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = np.array(embeddings)
    mocker.patch("ingest.model.get_embedding_model", return_value=mock_model)
    return mock_model


def test_semantic_chunk_ids(sentence_doc: Document, mocker) -> None:
    """chunk_id must follow the pattern {doc_id}_semantic_{index:04d}."""
    _mock_model(mocker, [_make_embedding(1.0) for _ in range(4)])

    chunks = chunk_semantic(sentence_doc, threshold=0.3)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{sentence_doc.doc_id}_semantic_{i:04d}"


def test_semantic_splits_on_low_similarity(sentence_doc: Document, mocker) -> None:
    """Semantic chunker creates a new chunk when cosine similarity drops below threshold."""
    # Sentences 0,1 point along dimension 0; sentences 2,3 point along dimension 1.
    # Cosine similarity between the two groups is 0.0, which is below threshold=0.3.
    embeddings = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sentence 0
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sentence 1 — same direction
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sentence 2 — orthogonal → new chunk
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sentence 3 — same as 2
    ]
    _mock_model(mocker, embeddings)

    chunks = chunk_semantic(sentence_doc, threshold=0.3)
    assert len(chunks) >= 2


def test_semantic_strategy_enum(sentence_doc: Document, mocker) -> None:
    """strategy field must equal ChunkStrategy.SEMANTIC."""
    _mock_model(mocker, [_make_embedding(1.0) for _ in range(4)])

    chunks = chunk_semantic(sentence_doc, threshold=0.3)
    for chunk in chunks:
        assert chunk.strategy == ChunkStrategy.SEMANTIC


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

def test_chunk_document_router_fixed(simple_doc: Document) -> None:
    """chunk_document routes FIXED strategy correctly."""
    chunks = chunk_document(simple_doc, ChunkStrategy.FIXED)
    assert all(c.strategy == ChunkStrategy.FIXED for c in chunks)


def test_chunk_document_router_structural(markdown_doc: Document) -> None:
    """chunk_document routes STRUCTURAL strategy correctly."""
    chunks = chunk_document(markdown_doc, ChunkStrategy.STRUCTURAL)
    assert all(c.strategy == ChunkStrategy.STRUCTURAL for c in chunks)


def test_chunk_document_router_invalid() -> None:
    """chunk_document raises ValueError for an unrecognised strategy string."""
    doc = Document(
        doc_id="x",
        text="hello",
        metadata=DocumentMetadata(
            file_name="x.txt",
            file_path="/tmp/x.txt",
            file_type=FileType.TXT,
        ),
    )
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(doc, "bogus")  # type: ignore[arg-type]
