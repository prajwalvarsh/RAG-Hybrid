"""Tests for generation/generator.py — prompt construction and generation."""

import pytest

from generation.generator import build_prompt, generate
from generation.schemas import GenerationResult
from retrieval.schemas import RetrievalMethod, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    text: str,
    rank: int = 1,
    score: float = 0.9,
) -> RetrievalResult:
    """Construct a HYBRID RetrievalResult for generator tests."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        rank=rank,
        metadata={},
        retrieval_method=RetrievalMethod.HYBRID,
    )


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------


def test_build_prompt_contains_chunks() -> None:
    """build_prompt must embed each chunk's text in the returned prompt."""
    chunks = [
        _make_chunk("c1", "Munich is the capital of Bavaria."),
        _make_chunk("c2", "The Oktoberfest is held in Munich every year."),
    ]
    prompt = build_prompt("Where is Oktoberfest held?", chunks)

    assert "Munich is the capital of Bavaria." in prompt
    assert "The Oktoberfest is held in Munich every year." in prompt


def test_build_prompt_numbers_chunks() -> None:
    """build_prompt must number context chunks starting at [1]."""
    chunks = [
        _make_chunk("c1", "First chunk text."),
        _make_chunk("c2", "Second chunk text."),
    ]
    prompt = build_prompt("test query", chunks)

    assert "[1]" in prompt
    assert "[2]" in prompt


def test_build_prompt_contains_instructions() -> None:
    """build_prompt must include grounding instructions in the prompt."""
    chunks = [_make_chunk("c1", "Some context.")]
    prompt = build_prompt("Any question?", chunks)

    assert "ONLY" in prompt or "only" in prompt
    assert "cite" in prompt.lower()
    assert "cannot answer from the provided context" in prompt.lower()


def test_build_prompt_contains_query() -> None:
    """build_prompt must embed the user's query in the prompt."""
    query = "What is the boiling point of water?"
    prompt = build_prompt(query, [_make_chunk("c1", "Some context.")])

    assert query in prompt


# ---------------------------------------------------------------------------
# generate tests
# ---------------------------------------------------------------------------


def test_generate_returns_correct_type(mocker) -> None:
    """generate must return a GenerationResult instance."""
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = "Munich is in Bavaria [1]."

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("generation.generator.OpenAI", return_value=mock_client)

    mocker.patch(
        "generation.generator.verify_citations",
        return_value=[],
    )

    chunks = [_make_chunk("c1", "Munich is the capital of Bavaria.")]
    result = generate("Where is Munich?", chunks)

    assert isinstance(result, GenerationResult)
    assert result.query == "Where is Munich?"
    assert result.model == "gpt-4o-mini"
    assert result.chunks_used == ["c1"]


def test_generate_populates_answer(mocker) -> None:
    """generate must store the LLM response text in result.answer."""
    expected_answer = "The answer is 42 [1]."

    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = expected_answer

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("generation.generator.OpenAI", return_value=mock_client)
    mocker.patch("generation.generator.verify_citations", return_value=[])

    chunks = [_make_chunk("c1", "Context text.")]
    result = generate("What is the answer?", chunks)

    assert result.answer == expected_answer


def test_generate_handles_api_error(mocker) -> None:
    """generate must log OpenAI errors and re-raise them."""
    from openai import APIError

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("network failure")
    mocker.patch("generation.generator.OpenAI", return_value=mock_client)

    mock_logger = mocker.patch("generation.generator.logger")

    chunks = [_make_chunk("c1", "Some context.")]

    with pytest.raises(RuntimeError, match="network failure"):
        generate("Any question?", chunks)

    mock_logger.error.assert_called_once()


def test_generate_uses_custom_model(mocker) -> None:
    """generate must pass the caller-specified model to the OpenAI API."""
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = "Answer [1]."

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("generation.generator.OpenAI", return_value=mock_client)
    mocker.patch("generation.generator.verify_citations", return_value=[])

    chunks = [_make_chunk("c1", "Context.")]
    generate("Question?", chunks, model="gpt-4o")

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"
