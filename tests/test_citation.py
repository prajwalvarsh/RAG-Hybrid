"""Tests for generation/citation.py — citation extraction and verification."""

import pytest

from generation.citation import (
    compute_support_score,
    extract_citations,
    verify_citations,
)
from generation.schemas import CitationStatus, CitationVerification
from retrieval.schemas import RetrievalMethod, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, rank: int = 1) -> RetrievalResult:
    """Construct a HYBRID RetrievalResult for citation tests."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        rank=rank,
        metadata={},
        retrieval_method=RetrievalMethod.HYBRID,
    )


def _make_verification(
    chunk_id: str,
    citation_number: int,
    supported: CitationStatus,
    confidence: float = 1.0,
) -> CitationVerification:
    """Construct a CitationVerification for use in score tests."""
    return CitationVerification(
        chunk_id=chunk_id,
        citation_number=citation_number,
        claim="some claim",
        supported=supported,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


def test_extract_citations_single() -> None:
    """A sentence with one citation marker must produce one pair."""
    answer = "Munich is the capital of Bavaria [1]."
    pairs = extract_citations(answer)

    assert len(pairs) == 1
    assert pairs[0][1] == 1
    assert "Munich is the capital of Bavaria [1]" in pairs[0][0]


def test_extract_citations_multiple() -> None:
    """A sentence with two citation markers must produce two pairs."""
    answer = "X is true [1] and Y is also true [2]."
    pairs = extract_citations(answer)

    assert len(pairs) == 2
    citation_numbers = [p[1] for p in pairs]
    assert 1 in citation_numbers
    assert 2 in citation_numbers


def test_extract_citations_multiple_sentences() -> None:
    """Separate sentences with different citations must each produce a pair."""
    answer = "Paris is in France [1]. Berlin is in Germany [2]."
    pairs = extract_citations(answer)

    assert len(pairs) == 2
    assert pairs[0][1] == 1
    assert pairs[1][1] == 2


def test_extract_citations_no_citations() -> None:
    """An answer with no citation markers must return an empty list."""
    answer = "This sentence has no citations at all."
    pairs = extract_citations(answer)

    assert pairs == []


def test_extract_citations_empty_string() -> None:
    """An empty answer string must return an empty list."""
    assert extract_citations("") == []


# ---------------------------------------------------------------------------
# compute_support_score
# ---------------------------------------------------------------------------


def test_compute_support_score_all_supported() -> None:
    """All SUPPORTED citations must yield a score of 1.0."""
    verifications = [
        _make_verification("c1", 1, CitationStatus.SUPPORTED),
        _make_verification("c2", 2, CitationStatus.SUPPORTED),
    ]
    assert compute_support_score(verifications) == pytest.approx(1.0)


def test_compute_support_score_partial() -> None:
    """One supported out of two must yield 0.5."""
    verifications = [
        _make_verification("c1", 1, CitationStatus.SUPPORTED),
        _make_verification("c2", 2, CitationStatus.UNSUPPORTED),
    ]
    assert compute_support_score(verifications) == pytest.approx(0.5)


def test_compute_support_score_none_supported() -> None:
    """All UNSUPPORTED citations must yield a score of 0.0."""
    verifications = [
        _make_verification("c1", 1, CitationStatus.UNSUPPORTED),
        _make_verification("c2", 2, CitationStatus.UNSUPPORTED),
    ]
    assert compute_support_score(verifications) == pytest.approx(0.0)


def test_compute_support_score_empty() -> None:
    """An empty verification list must return 1.0 (nothing to dispute)."""
    assert compute_support_score([]) == pytest.approx(1.0)


def test_compute_support_score_unverified_not_counted() -> None:
    """UNVERIFIED citations must not count as supported."""
    verifications = [
        _make_verification("c1", 1, CitationStatus.SUPPORTED),
        _make_verification("c2", 2, CitationStatus.UNVERIFIED),
    ]
    # 1 supported out of 2 total → 0.5
    assert compute_support_score(verifications) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# verify_citations (OpenAI mocked)
# ---------------------------------------------------------------------------


def test_verify_citations_supported(mocker) -> None:
    """A 'yes' reply from the LLM must produce a SUPPORTED CitationVerification."""
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = "yes"

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("generation.citation.OpenAI", return_value=mock_client)

    chunks = [_make_chunk("c1", "Munich is the capital of Bavaria.")]
    answer = "Munich is in Bavaria [1]."

    results = verify_citations(answer, chunks)

    assert len(results) == 1
    assert results[0].supported == CitationStatus.SUPPORTED
    assert results[0].confidence == pytest.approx(1.0)
    assert results[0].chunk_id == "c1"
    assert results[0].citation_number == 1


def test_verify_citations_unsupported(mocker) -> None:
    """A 'no' reply from the LLM must produce an UNSUPPORTED CitationVerification."""
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = "no"

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("generation.citation.OpenAI", return_value=mock_client)

    chunks = [_make_chunk("c1", "Unrelated content.")]
    answer = "Munich is in Bavaria [1]."

    results = verify_citations(answer, chunks)

    assert len(results) == 1
    assert results[0].supported == CitationStatus.UNSUPPORTED
    assert results[0].confidence == pytest.approx(1.0)


def test_verify_citations_out_of_range(mocker) -> None:
    """A citation number that exceeds the chunk list must be marked UNVERIFIED."""
    mocker.patch("generation.citation.OpenAI")

    chunks = [_make_chunk("c1", "Only one chunk.")]
    answer = "Some claim [5]."  # citation [5] but only 1 chunk

    results = verify_citations(answer, chunks)

    assert len(results) == 1
    assert results[0].supported == CitationStatus.UNVERIFIED
    assert results[0].chunk_id == ""


def test_verify_citations_empty_answer(mocker) -> None:
    """An answer with no citations must return an empty list."""
    mocker.patch("generation.citation.OpenAI")

    chunks = [_make_chunk("c1", "Some chunk.")]
    results = verify_citations("No citations here.", chunks)

    assert results == []


def test_verify_citations_api_error_returns_unverified(mocker) -> None:
    """An OpenAI API error during verification must produce an UNVERIFIED entry."""
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("api down")
    mocker.patch("generation.citation.OpenAI", return_value=mock_client)

    chunks = [_make_chunk("c1", "Some chunk.")]
    answer = "Some claim [1]."

    results = verify_citations(answer, chunks)

    assert len(results) == 1
    assert results[0].supported == CitationStatus.UNVERIFIED
    assert results[0].confidence == pytest.approx(0.0)
