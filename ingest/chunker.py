"""Three chunking strategies for RAG ingestion: fixed-size, structural, and semantic."""

import logging
import re
from typing import Callable

import nltk
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from ingest.schemas import Chunk, ChunkMetadata, ChunkStrategy, Document

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_EMBEDDING_MODEL: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model BAAI/bge-base-en-v1.5")
        _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _EMBEDDING_MODEL


def _token_count(text: str) -> int:
    """Return the number of cl100k_base tokens in text."""
    return len(_ENCODING.encode(text))


def _decode_tokens(tokens: list[int]) -> str:
    """Decode a list of token IDs back to a string."""
    return _ENCODING.decode(tokens)


def _make_chunk(
    doc_id: str,
    text: str,
    chunk_index: int,
    strategy: ChunkStrategy,
    id_suffix: str,
    metadata: ChunkMetadata | None = None,
) -> Chunk:
    """Construct a Chunk with all required fields populated."""
    return Chunk(
        chunk_id=f"{doc_id}_{id_suffix}_{chunk_index:04d}",
        doc_id=doc_id,
        text=text,
        chunk_index=chunk_index,
        strategy=strategy,
        token_count=_token_count(text),
        metadata=metadata or ChunkMetadata(),
    )


def chunk_fixed(
    document: Document,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a document into fixed-size token chunks with overlap.

    Tokenises the full document text using cl100k_base, then slides a window
    of `chunk_size` tokens forward by `chunk_size - overlap` tokens each step.
    Overlap ensures that sentences split across two windows are not lost.

    Args:
        document: The source Document to chunk.
        chunk_size: Maximum number of tokens per chunk (default 512).
        overlap: Number of tokens shared between consecutive chunks (default 50).

    Returns:
        List of Chunk objects in order.
    """
    tokens = _ENCODING.encode(document.text)
    stride = chunk_size - overlap
    chunks: list[Chunk] = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        window = tokens[start:end]
        text = _decode_tokens(window)
        chunks.append(
            _make_chunk(document.doc_id, text, len(chunks), ChunkStrategy.FIXED, "fixed")
        )
        if end == len(tokens):
            break
        start += stride

    logger.debug("Fixed chunking: %d chunks from doc %s", len(chunks), document.doc_id)
    return chunks


def chunk_structural(document: Document) -> list[Chunk]:
    """Split a document at markdown headings and double newlines.

    Each heading (# / ## / ###) and blank-line-separated paragraph becomes
    its own chunk. The nearest heading above each section is stored in
    ChunkMetadata.heading so retrieval can surface section context.

    Args:
        document: The source Document to chunk.

    Returns:
        List of Chunk objects in document order.
    """
    # Split on heading lines OR double newlines, keeping the delimiter
    pattern = re.compile(r"((?:^|\n)#{1,3} .+)", re.MULTILINE)
    parts = pattern.split(document.text)

    chunks: list[Chunk] = []
    current_heading: str | None = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r"^#{1,3} ", part):
            current_heading = part.lstrip("#").strip()
            # A heading line alone becomes its own chunk
            chunks.append(
                _make_chunk(
                    document.doc_id,
                    part,
                    len(chunks),
                    ChunkStrategy.STRUCTURAL,
                    "structural",
                    ChunkMetadata(heading=current_heading),
                )
            )
        else:
            # Split non-heading content on double newlines (paragraph breaks)
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", part) if p.strip()]
            for paragraph in paragraphs:
                chunks.append(
                    _make_chunk(
                        document.doc_id,
                        paragraph,
                        len(chunks),
                        ChunkStrategy.STRUCTURAL,
                        "structural",
                        ChunkMetadata(heading=current_heading),
                    )
                )

    logger.debug("Structural chunking: %d chunks from doc %s", len(chunks), document.doc_id)
    return chunks


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_semantic(
    document: Document,
    threshold: float = 0.3,
) -> list[Chunk]:
    """Split a document into semantically coherent chunks using sentence embeddings.

    Tokenises the text into sentences with NLTK, embeds each sentence using
    the local BAAI/bge-base-en-v1.5 model via sentence-transformers, then
    starts a new chunk whenever the cosine similarity between consecutive
    sentence embeddings drops below `threshold`.

    The embedding model is loaded once as a module-level singleton via
    _get_embedding_model() and reused across calls.

    Args:
        document: The source Document to chunk.
        threshold: Cosine similarity threshold below which a new chunk starts
                   (default 0.3).

    Returns:
        List of Chunk objects ordered by position in the document.
    """
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer data")
        nltk.download("punkt_tab", quiet=True)

    sentences = nltk.sent_tokenize(document.text)
    if not sentences:
        return []

    model = _get_embedding_model()
    raw: np.ndarray = model.encode(sentences)
    embeddings: list[list[float]] = raw.tolist()

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        text = " ".join(group)
        chunks.append(
            _make_chunk(document.doc_id, text, idx, ChunkStrategy.SEMANTIC, "semantic")
        )

    logger.debug("Semantic chunking: %d chunks from doc %s", len(chunks), document.doc_id)
    return chunks


_STRATEGY_MAP: dict[ChunkStrategy, Callable[[Document], list[Chunk]]] = {
    ChunkStrategy.FIXED: chunk_fixed,
    ChunkStrategy.STRUCTURAL: chunk_structural,
    ChunkStrategy.SEMANTIC: chunk_semantic,
}


def chunk_document(document: Document, strategy: ChunkStrategy) -> list[Chunk]:
    """Route a document to the correct chunking strategy.

    Args:
        document: The source Document to chunk.
        strategy: One of ChunkStrategy.FIXED, STRUCTURAL, or SEMANTIC.

    Returns:
        List of Chunk objects produced by the chosen strategy.

    Raises:
        ValueError: If strategy is not a recognised ChunkStrategy value.
    """
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Valid options: {[s.value for s in ChunkStrategy]}"
        )
    logger.info("Chunking doc %s with strategy '%s'", document.doc_id, strategy.value)
    return _STRATEGY_MAP[strategy](document)
