# generation/citation.py

import logging
import re

from dotenv import load_dotenv
from openai import OpenAI

from generation.schemas import CitationStatus, CitationVerification
from retrieval.schemas import RetrievalResult

load_dotenv()

logger = logging.getLogger(__name__)


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
) -> list[CitationVerification]:
    """Verify every citation in the answer against the retrieved chunks.

    For each (claim, citation_number) pair extracted from the answer:
    - Looks up the chunk at that 1-based index.
    - Calls the OpenAI API to ask whether the chunk text supports the claim.
    - Marks the citation SUPPORTED, UNSUPPORTED, or UNVERIFIED (when the
      index is out of range).

    Confidence is set to 1.0 for a clear yes/no reply and 0.5 when the
    model's answer is ambiguous.

    Returns:
        List[CitationVerification] in the same order as extract_citations.
    """
    pairs = extract_citations(answer)
    if not pairs:
        return []

    client = OpenAI()
    verifications: list[CitationVerification] = []

    for claim, citation_number in pairs:
        idx = citation_number - 1  # convert 1-based to 0-based

        if idx < 0 or idx >= len(chunks):
            logger.warning(
                "Citation [%d] is out of range (only %d chunks available).",
                citation_number,
                len(chunks),
            )
            verifications.append(
                CitationVerification(
                    chunk_id="",
                    citation_number=citation_number,
                    claim=claim,
                    supported=CitationStatus.UNVERIFIED,
                    confidence=0.0,
                )
            )
            continue

        chunk = chunks[idx]
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
            )
            reply = (response.choices[0].message.content or "").strip().lower()
        except Exception as exc:
            logger.error(
                "OpenAI error while verifying citation [%d]: %s",
                citation_number,
                exc,
            )
            verifications.append(
                CitationVerification(
                    chunk_id=chunk.chunk_id,
                    citation_number=citation_number,
                    claim=claim,
                    supported=CitationStatus.UNVERIFIED,
                    confidence=0.0,
                )
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

        verifications.append(
            CitationVerification(
                chunk_id=chunk.chunk_id,
                citation_number=citation_number,
                claim=claim,
                supported=status,
                confidence=confidence,
            )
        )

    return verifications


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
