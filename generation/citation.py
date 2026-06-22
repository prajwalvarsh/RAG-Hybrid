# generation/citation.py

import json
import logging
import re

from dotenv import load_dotenv
from openai import OpenAI

from generation.schemas import CitationStatus, CitationVerification
from retrieval.schemas import RetrievalResult

load_dotenv()

logger = logging.getLogger(__name__)

# Internal alias: (result_slot_index, claim, citation_number, chunk)
_Pending = tuple[int, str, int, RetrievalResult]


def extract_citations(answer: str) -> list[tuple[str, int]]:
    """Extract (claim, citation_number) pairs from a generated answer.

    Splits the answer into sentences, then finds every bracketed citation
    marker [n] within each sentence.  A sentence that contains multiple
    markers (e.g. "X is true [1] and Y is also true [2]") produces one pair
    per marker, both sharing the same sentence as the claim text.

    Returns:
        List of (claim_text, citation_number) tuples in document order.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    pairs: list[tuple[str, int]] = []
    for sentence in sentences:
        markers = re.findall(r"\[(\d+)\]", sentence)
        for num_str in markers:
            pairs.append((sentence, int(num_str)))
    return pairs


def verify_citations(
    answer: str,
    chunks: list[RetrievalResult],
    client: OpenAI | None = None,
) -> list[CitationVerification]:
    """Verify every citation in the answer against the retrieved chunks.

    All in-range citations are verified in a single batched OpenAI call that
    asks the model to return a JSON array of yes/no verdicts — one per claim —
    in a single round-trip rather than one API call per citation.  This keeps
    API cost proportional to answer length rather than citation count.

    If the batch call fails (network error, malformed JSON, wrong array length),
    the function transparently falls back to individual per-citation calls so
    that partial results are still returned rather than failing entirely.

    Out-of-range citations (citation number > number of chunks) are always
    marked UNVERIFIED without an API call.

    Confidence is set to the value returned by the model (0.0–1.0) for batch
    verdicts, 1.0 for unambiguous per-citation yes/no, and 0.5 for ambiguous
    per-citation replies.

    Args:
        answer: The generated answer text containing bracketed citations.
        chunks: Retrieved chunks in rank order (1-based citation indexing).
        client: Optional OpenAI client. A new client is created from the
                environment when not provided, allowing tests to inject a
                mock directly without module-level patching.

    Returns:
        List[CitationVerification] in the same order as extract_citations.
    """
    pairs = extract_citations(answer)
    if not pairs:
        return []

    if client is None:
        client = OpenAI()

    # Pre-allocate result slots; fill out-of-range entries immediately.
    results: list[CitationVerification | None] = [None] * len(pairs)
    pending: list[_Pending] = []

    for i, (claim, citation_number) in enumerate(pairs):
        idx = citation_number - 1
        if idx < 0 or idx >= len(chunks):
            logger.warning(
                "Citation [%d] is out of range (only %d chunks available).",
                citation_number,
                len(chunks),
            )
            results[i] = CitationVerification(
                chunk_id="",
                citation_number=citation_number,
                claim=claim,
                supported=CitationStatus.UNVERIFIED,
                confidence=0.0,
            )
        else:
            pending.append((i, claim, citation_number, chunks[idx]))

    if pending:
        _fill_results(pending, results, client)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fill_results(
    pending: list[_Pending],
    results: list[CitationVerification | None],
    client: OpenAI,
) -> None:
    """Try batch verification; fall back to per-citation on any failure."""
    prompt_lines = [
        "For each numbered claim below, answer whether the corresponding chunk "
        "supports it. Reply ONLY with a JSON array of objects with keys "
        "'supported' (yes/no) and 'confidence' (0.0-1.0), one entry per claim "
        "in order.\n",
    ]
    for j, (_, claim, _, chunk) in enumerate(pending, start=1):
        prompt_lines.append(f"Claim {j}: {claim}")
        prompt_lines.append(f"Chunk {j}: {chunk.text}\n")
    batch_prompt = "\n".join(prompt_lines)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": batch_prompt}],
            max_tokens=200,
        )
        raw = (response.choices[0].message.content or "").strip()
        verdicts: list[dict] = json.loads(raw)
        if len(verdicts) != len(pending):
            raise ValueError(
                f"Expected {len(pending)} verdicts, got {len(verdicts)}"
            )

        for j, (i, claim, citation_number, chunk) in enumerate(pending):
            verdict = verdicts[j]
            reply = str(verdict.get("supported", "")).strip().lower()
            confidence = float(verdict.get("confidence", 0.5))
            if reply == "yes":
                status = CitationStatus.SUPPORTED
            elif reply == "no":
                status = CitationStatus.UNSUPPORTED
            else:
                logger.warning(
                    "Ambiguous batch verdict for citation [%d]: %r",
                    citation_number,
                    reply,
                )
                status = CitationStatus.UNVERIFIED
                confidence = 0.5
            results[i] = CitationVerification(
                chunk_id=chunk.chunk_id,
                citation_number=citation_number,
                claim=claim,
                supported=status,
                confidence=confidence,
            )

    except Exception as exc:
        logger.warning(
            "Batch verification failed, falling back to per-citation calls: %s",
            exc,
        )
        _fill_per_citation(pending, results, client)


def _fill_per_citation(
    pending: list[_Pending],
    results: list[CitationVerification | None],
    client: OpenAI,
) -> None:
    """Verify citations one at a time (fallback path)."""
    for i, claim, citation_number, chunk in pending:
        prompt = (
            f"Does this chunk support this claim?\n\n"
            f"Chunk: {chunk.text}\n\n"
            f"Claim: {claim}\n\n"
            f"Answer only yes or no."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
            )
            reply = (response.choices[0].message.content or "").strip().lower()
        except Exception as exc:
            logger.error(
                "OpenAI error while verifying citation [%d]: %s",
                citation_number,
                exc,
            )
            results[i] = CitationVerification(
                chunk_id=chunk.chunk_id,
                citation_number=citation_number,
                claim=claim,
                supported=CitationStatus.UNVERIFIED,
                confidence=0.0,
            )
            continue

        if reply.startswith("yes"):
            status = CitationStatus.SUPPORTED
            confidence = 1.0
        elif reply.startswith("no"):
            status = CitationStatus.UNSUPPORTED
            confidence = 1.0
        else:
            logger.warning(
                "Ambiguous verification reply for citation [%d]: %r",
                citation_number,
                reply,
            )
            status = CitationStatus.UNVERIFIED
            confidence = 0.5

        results[i] = CitationVerification(
            chunk_id=chunk.chunk_id,
            citation_number=citation_number,
            claim=claim,
            supported=status,
            confidence=confidence,
        )


def compute_support_score(verifications: list[CitationVerification]) -> float:
    """Return the fraction of citations marked as SUPPORTED.

    Returns 1.0 when there are no citations — an answer with no claims
    to verify is considered fully grounded by convention.
    """
    if not verifications:
        return 1.0
    supported = sum(
        1 for v in verifications if v.supported == CitationStatus.SUPPORTED
    )
    return supported / len(verifications)
