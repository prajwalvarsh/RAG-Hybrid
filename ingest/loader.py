"""Document loading utilities for supported file types (.txt, .md, .pdf)."""

import logging
import os
import uuid
from pathlib import Path

from pypdf import PdfReader

from ingest.schemas import Document, DocumentMetadata, FileType

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt": FileType.TXT, ".md": FileType.MD, ".pdf": FileType.PDF}


def load_document(file_path: str) -> Document:
    """Load a single document from disk and return a Document object.

    Args:
        file_path: Absolute or relative path to a .txt, .md, or .pdf file.

    Returns:
        A Document with extracted text and populated metadata.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}' for file '{path.name}'. "
            f"Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    file_type = SUPPORTED_EXTENSIONS[ext]

    if file_type in (FileType.TXT, FileType.MD):
        text = path.read_text(encoding="utf-8")
    else:
        text = _extract_pdf_text(path)

    metadata = DocumentMetadata(
        file_name=path.name,
        file_path=str(path.resolve()),
        file_type=file_type,
        size_bytes=path.stat().st_size,
    )

    return Document(
        doc_id=str(uuid.uuid4()),
        text=text,
        metadata=metadata,
    )


def _extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF by concatenating all page texts.

    Args:
        path: Path to the PDF file.

    Returns:
        Full text content of the PDF as a single string.
    """
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_all(directory: str) -> list[Document]:
    """Recursively load all supported documents from a directory.

    Skips unsupported file types with a warning; does not raise on them.

    Args:
        directory: Path to the directory to walk (e.g. "data/raw/").

    Returns:
        List of Document objects for every supported file found.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: list[Document] = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", file_path)
            continue

        try:
            doc = load_document(str(file_path))
            documents.append(doc)
            logger.info("Loaded document: %s (%s)", file_path.name, doc.doc_id)
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)

    return documents
