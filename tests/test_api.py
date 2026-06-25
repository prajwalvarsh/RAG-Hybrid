"""Tests for api/main.py — FastAPI endpoints.

All LLM and ChromaDB calls are mocked so no real infrastructure is required.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_PIPELINE_RESULT = {
    "question": "What is the capital of France?",
    "answer": "Paris is the capital of France [1].",
    "citations": [
        {
            "chunk_id": "chunk-abc",
            "citation_number": 1,
            "claim": "Paris is the capital of France",
            "supported": "supported",
            "confidence": 0.95,
        }
    ],
    "support_score": 1.0,
    "retrieved_chunks": ["Paris is the capital of France."],
    "retrieval_method": "hybrid",
}

_FAKE_COLLECTION_COUNTS = {
    "rag_fixed": 43,
    "rag_hybrid": 0,
}


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous TestClient for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


def test_query_valid_returns_200(client: TestClient, mocker) -> None:
    """A valid question with a successful pipeline result returns HTTP 200.

    Mocks run_pipeline so no real embedding, retrieval, or LLM call is made.
    """
    mocker.patch("api.main.run_pipeline", return_value=_FAKE_PIPELINE_RESULT)

    response = client.post(
        "/query",
        json={"question": "What is the capital of France?"},
    )

    assert response.status_code == 200


def test_query_response_has_expected_fields(client: TestClient, mocker) -> None:
    """POST /query response body must contain all QueryResponse fields."""
    mocker.patch("api.main.run_pipeline", return_value=_FAKE_PIPELINE_RESULT)

    response = client.post(
        "/query",
        json={"question": "What is the capital of France?"},
    )

    body = response.json()
    assert "question" in body
    assert "answer" in body
    assert "citations" in body
    assert "support_score" in body
    assert "retrieval_method" in body
    assert "latency_ms" in body


def test_query_citations_are_well_formed(client: TestClient, mocker) -> None:
    """Each citation in the response must contain the CitationResponse fields."""
    mocker.patch("api.main.run_pipeline", return_value=_FAKE_PIPELINE_RESULT)

    response = client.post(
        "/query",
        json={"question": "What is the capital of France?"},
    )

    body = response.json()
    assert len(body["citations"]) == 1
    cit = body["citations"][0]
    assert cit["chunk_id"] == "chunk-abc"
    assert cit["citation_number"] == 1
    assert cit["supported"] == "supported"
    assert cit["confidence"] == pytest.approx(0.95)


def test_query_returns_503_on_empty_pipeline(client: TestClient, mocker) -> None:
    """POST /query must return HTTP 503 when run_pipeline returns an empty dict."""
    mocker.patch("api.main.run_pipeline", return_value={})

    response = client.post(
        "/query",
        json={"question": "This will fail."},
    )

    assert response.status_code == 503


def test_query_503_body_contains_detail(client: TestClient, mocker) -> None:
    """HTTP 503 response must include a human-readable detail message."""
    mocker.patch("api.main.run_pipeline", return_value={})

    response = client.post(
        "/query",
        json={"question": "This will fail."},
    )

    body = response.json()
    assert "detail" in body


def test_query_custom_collection_forwarded(client: TestClient, mocker) -> None:
    """POST /query must forward collection_name to run_pipeline."""
    mock_pipeline = mocker.patch(
        "api.main.run_pipeline", return_value=_FAKE_PIPELINE_RESULT
    )

    client.post(
        "/query",
        json={
            "question": "What is the capital of France?",
            "collection_name": "rag_fixed",
        },
    )

    mock_pipeline.assert_called_once_with(
        "What is the capital of France?",
        collection_name="rag_fixed",
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient, mocker) -> None:
    """GET /health must return HTTP 200."""
    mocker.patch("api.main._collection_counts", return_value=_FAKE_COLLECTION_COUNTS)

    response = client.get("/health")

    assert response.status_code == 200


def test_health_has_required_keys(client: TestClient, mocker) -> None:
    """GET /health response must contain status, model, chroma_path, default_collection, collections."""
    mocker.patch("api.main._collection_counts", return_value=_FAKE_COLLECTION_COUNTS)

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "chroma_path" in body
    assert "default_collection" in body
    assert "collections" in body
    assert isinstance(body["collections"], dict)


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------


def test_collections_returns_200(client: TestClient, mocker) -> None:
    """GET /collections must return HTTP 200."""
    mocker.patch("api.main._collection_counts", return_value=_FAKE_COLLECTION_COUNTS)

    response = client.get("/collections")

    assert response.status_code == 200


def test_collections_returns_list(client: TestClient, mocker) -> None:
    """GET /collections must return a JSON list."""
    mocker.patch("api.main._collection_counts", return_value=_FAKE_COLLECTION_COUNTS)

    response = client.get("/collections")

    body = response.json()
    assert isinstance(body, list)


def test_collections_entries_have_name_and_count(client: TestClient, mocker) -> None:
    """Each entry in /collections must have 'name' and 'count' keys."""
    mocker.patch("api.main._collection_counts", return_value=_FAKE_COLLECTION_COUNTS)

    response = client.get("/collections")

    body = response.json()
    assert len(body) == 2
    for entry in body:
        assert "name" in entry
        assert "count" in entry


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

# Minimal valid PDF bytes (uses a real but tiny single-page PDF)
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

_FAKE_CHUNK = type(
    "_FakeChunk",
    (),
    {"chunk_id": "c1", "text": "hello", "chunk_index": 0},
)()

_FAKE_CHUNKS = [_FAKE_CHUNK]

_FAKE_EMBEDDED = [
    type("_FakeEC", (), {"chunk": _FAKE_CHUNK, "embedding": [0.1] * 768})()
]


def _patch_ingest_pipeline(mocker):
    """Patch all four ingest pipeline stages so no real IO or GPU work runs."""
    fake_doc = type("_Doc", (), {"doc_id": "d1", "text": "hello world"})()
    mocker.patch("api.main.load_document", return_value=fake_doc)
    mocker.patch("api.main.chunk_document", return_value=_FAKE_CHUNKS)
    mocker.patch("api.main.embed_chunks", return_value=_FAKE_EMBEDDED)
    mocker.patch("api.main.index_chunks", return_value=None)


# ---------------------------------------------------------------------------
# Helpers — multipart field name is "files" (matches the List[UploadFile] param)
# ---------------------------------------------------------------------------


def test_ingest_valid_pdf_returns_200(client: TestClient, mocker) -> None:
    """A valid PDF upload with mocked pipeline stages returns HTTP 200."""
    _patch_ingest_pipeline(mocker)

    response = client.post(
        "/ingest",
        files=[("files", ("test.pdf", _MINIMAL_PDF, "application/pdf"))],
        data={"collection_name": "rag_test", "strategy": "fixed"},
    )

    assert response.status_code == 200


def test_ingest_valid_pdf_chunk_count_positive(client: TestClient, mocker) -> None:
    """POST /ingest response files[0].chunk_count must equal the mocked embed output length."""
    _patch_ingest_pipeline(mocker)

    response = client.post(
        "/ingest",
        files=[("files", ("test.pdf", _MINIMAL_PDF, "application/pdf"))],
        data={"collection_name": "rag_test", "strategy": "fixed"},
    )

    body = response.json()
    assert body["files"][0]["chunk_count"] > 0


def test_ingest_response_has_files_list(client: TestClient, mocker) -> None:
    """POST /ingest response must contain a 'files' list; each entry has required fields."""
    _patch_ingest_pipeline(mocker)

    response = client.post(
        "/ingest",
        files=[("files", ("doc.txt", b"hello world", "text/plain"))],
        data={"collection_name": "rag_test", "strategy": "structural"},
    )

    body = response.json()
    assert "files" in body
    assert isinstance(body["files"], list)
    assert len(body["files"]) == 1
    entry = body["files"][0]
    for field in ("filename", "collection_name", "strategy", "chunk_count", "elapsed_ms", "status"):
        assert field in entry, f"Missing field: {field}"


def test_ingest_unsupported_file_type_returns_error_status(client: TestClient) -> None:
    """Uploading a .docx file must return HTTP 200 with status 'error' in files list.

    Per-file validation failures are reported inside the response body rather
    than as HTTP error codes so that mixed batches can be handled gracefully.
    """
    response = client.post(
        "/ingest",
        files=[("files", ("report.docx", b"PK\x03\x04", "application/octet-stream"))],
        data={"collection_name": "rag_test", "strategy": "fixed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["files"][0]["status"] == "error"


def test_ingest_error_entry_mentions_file_type(client: TestClient) -> None:
    """Error entry for unsupported extension must describe the problem in the error field."""
    response = client.post(
        "/ingest",
        files=[("files", ("data.csv", b"col1,col2\n1,2", "text/csv"))],
        data={"collection_name": "rag_test", "strategy": "fixed"},
    )

    body = response.json()
    entry = body["files"][0]
    assert entry["status"] == "error"
    assert ".csv" in entry["error"]


def test_ingest_pipeline_called_with_correct_strategy(client: TestClient, mocker) -> None:
    """chunk_document must be called with the strategy value sent in the form."""
    _patch_ingest_pipeline(mocker)
    mock_chunk = mocker.patch("api.main.chunk_document", return_value=_FAKE_CHUNKS)

    client.post(
        "/ingest",
        files=[("files", ("notes.md", b"# Hello\nWorld", "text/markdown"))],
        data={"collection_name": "rag_notes", "strategy": "semantic"},
    )

    args, _ = mock_chunk.call_args
    assert args[1].value == "semantic"


def test_ingest_multiple_valid_files_returns_200_with_results_list(
    client: TestClient, mocker
) -> None:
    """Uploading two valid files returns HTTP 200 and a files list with two entries."""
    _patch_ingest_pipeline(mocker)

    response = client.post(
        "/ingest",
        files=[
            ("files", ("first.txt", b"First document text.", "text/plain")),
            ("files", ("second.md", b"# Second\nMarkdown doc.", "text/markdown")),
        ],
        data={"collection_name": "rag_multi", "strategy": "fixed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["files"]) == 2
    for entry in body["files"]:
        assert entry["status"] == "success"
        assert entry["chunk_count"] > 0


def test_ingest_oversized_file_returns_error_status(client: TestClient) -> None:
    """A file exceeding 10 MB must produce an error entry — not an HTTP error code.

    No pipeline mocks are needed because size validation fires before any IO.
    """
    oversized = b"x" * (10 * 1024 * 1024 + 1)  # 10 MB + 1 byte

    response = client.post(
        "/ingest",
        files=[("files", ("big.txt", oversized, "text/plain"))],
        data={"collection_name": "rag_test", "strategy": "fixed"},
    )

    assert response.status_code == 200
    body = response.json()
    entry = body["files"][0]
    assert entry["status"] == "error"
    assert "10 MB" in entry["error"]


def test_ingest_partial_success_mixed_batch(client: TestClient, mocker) -> None:
    """A batch with one valid file and one oversized file reports partial success.

    The valid file produces status 'success'; the oversized file produces
    status 'error' — both in the same 200 response.
    """
    _patch_ingest_pipeline(mocker)
    oversized = b"y" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/ingest",
        files=[
            ("files", ("good.txt", b"Normal content.", "text/plain")),
            ("files", ("huge.txt", oversized, "text/plain")),
        ],
        data={"collection_name": "rag_mixed", "strategy": "fixed"},
    )

    assert response.status_code == 200
    body = response.json()
    statuses = [r["status"] for r in body["files"]]
    assert "success" in statuses
    assert "error" in statuses
