# generation/generator.py

import logging

from dotenv import load_dotenv
from groq import Groq

from generation.citation import compute_support_score, verify_citations
from generation.schemas import GenerationResult
from retrieval.schemas import RetrievalMethod, RetrievalResult

load_dotenv()

logger = logging.getLogger(__name__)

_GROUNDING_INSTRUCTIONS = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the numbered context chunks provided below. "
    "Cite every factual claim with the chunk number in square brackets, e.g. [1]. "
    "If the answer cannot be found in the provided context, respond with exactly: "
    '"I cannot answer from the provided context." '
    "Do not use any knowledge outside the provided chunks."
)


def build_prompt(query: str, chunks: list[RetrievalResult]) -> str:
    """Construct a grounded prompt that forces the LLM to cite its sources.

    Grounding strategy:
    - Each retrieved chunk is numbered sequentially as [1], [2], [3], ...
      so that the LLM can reference them unambiguously.
    - The system instruction prohibits the model from drawing on any
      knowledge outside the provided context, reducing hallucination.
    - The model is explicitly told to emit citation markers for every
      claim, enabling downstream citation verification.
    - An out-of-context fallback phrase is prescribed so that unanswerable
      queries produce a deterministic, parseable response.

    Args:
        query:  The user's question.
        chunks: Ranked retrieval results whose text forms the context.

    Returns:
        A single prompt string ready to be sent as a user message to the LLM.
    """
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(f"[{i}] {chunk.text}")
    context_block = "\n\n".join(context_lines)

    prompt = (
        f"{_GROUNDING_INSTRUCTIONS}\n\n"
        f"--- CONTEXT ---\n"
        f"{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite every claim with [n]):"
    )
    return prompt


def generate(
    query: str,
    chunks: list[RetrievalResult],
    model: str = "llama-3.1-8b-instant",
) -> GenerationResult:
    """Generate a grounded answer and verify every citation it contains.

    Steps:
    1. Build a grounded prompt via build_prompt().
    2. Call the Groq Chat Completions API.
    3. Extract the answer text from the response.
    4. Run citation verification against the retrieved chunks.
    5. Compute the overall support score.
    6. Return a fully populated GenerationResult.

    Groq API errors are logged at ERROR level and re-raised so callers
    can decide on retry / fallback behaviour.

    Args:
        query:  The user's question.
        chunks: Ranked retrieval results to ground the answer.
        model:  Groq model identifier (default: "llama-3.1-8b-instant").

    Returns:
        GenerationResult with the answer, citation audit, and metadata.
    """
    prompt = build_prompt(query, chunks)
    client = Groq()

    logger.info("Calling Groq model=%s for query: %r", model, query)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.error("Groq API error for query %r: %s", query, exc)
        raise

    answer = response.choices[0].message.content or ""
    logger.info("Received answer (%d chars)", len(answer))

    verifications = verify_citations(answer, chunks)
    support_score = compute_support_score(verifications)

    retrieval_method = (
        chunks[0].retrieval_method if chunks else RetrievalMethod.HYBRID
    )

    return GenerationResult(
        query=query,
        answer=answer,
        citations=verifications,
        chunks_used=[c.chunk_id for c in chunks],
        support_score=support_score,
        model=model,
        retrieval_method=retrieval_method,
    )
