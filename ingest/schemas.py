# ingest/schemas.py

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MD = "md"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


class DocumentMetadata(BaseModel):
    """Structured metadata attached to every loaded document."""
    file_name: str
    file_path: str
    file_type: FileType
    size_bytes: Optional[int] = None
    extra: dict = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Structured metadata attached to every chunk."""
    heading: Optional[str] = None        # nearest section heading above this chunk
    page_number: Optional[int] = None    # for PDFs
    char_start: Optional[int] = None     # character offset in original document
    char_end: Optional[int] = None


class Document(BaseModel):
    """
    Raw document loaded from data/raw/.
    Output of ingest/loader.py. Input to ingest/chunker.py.
    """
    doc_id: str
    text: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """
    Single indexed unit produced by ingest/chunker.py.
    Input to ingest/embedder.py.
    """
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    strategy: ChunkStrategy
    token_count: Optional[int] = None
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class EmbeddedChunk(BaseModel):
    """
    Chunk with its embedding vector.
    Output of ingest/embedder.py. Input to ingest/indexer.py.
    """
    chunk: Chunk
    embedding: list[float] = Field(..., min_length=1)
    embedding_model: str                 # e.g. "text-embedding-3-small"