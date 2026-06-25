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
