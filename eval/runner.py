"""End-to-end evaluation runner.

Executes the full RAG pipeline against the golden test set and scores
the results using four RAGAS metrics: faithfulness, answer_relevancy,
context_precision, and context_recall.
"""

import json
import logging
import os
import time
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

_REQUIRED_KEYS: frozenset[str] = frozenset({"id", "question", "expected_answer", "type"})
_PASS_THRESHOLD: float = 0.7


# ---------------------------------------------------------------------------
# Golden set loading
# ---------------------------------------------------------------------------


def load_golden(path: str) -> list[dict]:
    """Load and validate the golden test set from a JSON file.

    Checks that the file exists and that every entry contains the four
    required keys (id, question, expected_answer, type).

    Args:
        path: File path to golden.json.

    Returns:
        List of validated golden question dicts.

    Raises:
        FileNotFoundError: If no file exists at path.
        ValueError: If any entry is missing one or more required keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Golden test set not found at '{path}'. "
            "Ensure eval/golden/golden.json exists before running the runner."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for i, entry in enumerate(data):
        missing = _REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise ValueError(
                f"Golden entry at index {i} (id={entry.get('id', '?')!r}) "
                f"is missing required keys: {sorted(missing)}"
            )

    logger.info("Loaded %d golden questions from '%s'", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline(question: str, collection_name: str | None = None) -> dict:
    """Execute the full RAG pipeline for one question.

    Stages: embed → dense retrieval → sparse retrieval → RRF fusion →
    cross-encoder rerank → grounded generation.

    Failure handling:
      - Dense fails  → log warning, continue with sparse only.
      - Sparse fails → log warning, continue with dense only.
      - Both fail    → log error, return {}.
      - Generation fails → log error, return {}.

    Args:
        question:        Natural-language query.
        collection_name: ChromaDB collection to query. Defaults to
                         settings.default_collection.

    Returns:
        Dict with keys question, answer, citations, support_score,
        retrieved_chunks, retrieval_method.  Empty dict on fatal failure.
    """
    if collection_name is None:
        collection_name = settings.default_collection

    # Lazy imports keep module-level load time low and allow tests to mock them.
    from generation.generator import generate
    from ingest.model import get_embedding_model
    from retrieval.dense import retrieve_dense
    from retrieval.fusion import fuse_results
    from retrieval.reranker import rerank
    from retrieval.schemas import RetrievalRequest
    from retrieval.sparse import retrieve_sparse

    # Pre-warm embedding model once so retrieval latency is not skewed by
    # the first-call model-load penalty.
    get_embedding_model()

    request = RetrievalRequest(
        query=question,
        retrieval_top_k=settings.retrieval_top_k,
        fusion_top_k=settings.fusion_top_k,
        rerank_top_k=settings.rerank_top_k,
        collection_name=collection_name,
    )

    dense_results = []
    sparse_results = []

    try:
        dense_results = retrieve_dense(request)
    except Exception as exc:
        logger.warning("Dense retrieval failed for query %r: %s", question, exc)

    try:
        sparse_results = retrieve_sparse(request)
    except Exception as exc:
        logger.warning("Sparse retrieval failed for query %r: %s", question, exc)

    if not dense_results and not sparse_results:
        logger.error(
            "Both dense and sparse retrieval failed for query %r — cannot generate",
            question,
        )
        return {}

    fused = fuse_results(dense_results, sparse_results, top_k=request.fusion_top_k)
    reranked = rerank(
        question, fused,
        rerank_candidate_k=request.fusion_top_k,
        rerank_top_k=request.rerank_top_k,
    )

    try:
        gen_result = generate(question, reranked)
    except Exception as exc:
        logger.error("Generation failed for query %r: %s", question, exc)
        return {}

    retrieval_method = (
        reranked[0].retrieval_method.value if reranked else "hybrid"
    )

    return {
        "question": question,
        "answer": gen_result.answer,
        "citations": [c.model_dump() for c in gen_result.citations],
        "support_score": gen_result.support_score,
        "retrieved_chunks": [r.text for r in reranked],
        "retrieval_method": retrieval_method,
    }


# ---------------------------------------------------------------------------
# RAGAS helpers
# ---------------------------------------------------------------------------


def _build_ragas_llm():
    """Return a RAGAS-compatible LLM backed by Groq's OpenAI-compatible API.

    Wraps ChatOpenAI (pointed at Groq's endpoint) in a LangchainLLMWrapper so
    RAGAS can call it for faithfulness, answer_relevancy, and context_recall.
    Reads llm_api_key, llm_api_base, and llm_model from the central settings
    singleton so values stay in sync with the generation module.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.llm_api_key,
            openai_api_base=settings.llm_api_base,
        )
    )


def _build_ragas_embeddings():
    """Return a RAGAS-compatible embeddings backend using the project's local model.

    Wraps BAAI/bge-base-en-v1.5 via HuggingFaceEmbeddings so RAGAS can call it
    for answer_relevancy scoring (which requires embedding both question and
    answer to measure semantic closeness).
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # langchain_huggingface is preferred in newer langchain versions; fall back
    # to langchain_community for older installs.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    )


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------


def evaluate_with_ragas(
    results: list[dict],
    golden: list[dict],
) -> dict:
    """Score pipeline results using four RAGAS metrics.

    Builds a HuggingFace Dataset from (result, golden) pairs—preserving the
    original ordering so index i in results matches index i in golden—then
    evaluates faithfulness, answer_relevancy, context_precision, and
    context_recall using ragas 0.1.21's Dataset-based API.

    Uses ChatOpenAI pointed at Groq's OpenAI-compatible endpoint instead of
    langchain-groq, avoiding a package conflict between langchain-groq and
    the project's groq>=1.5.0 dependency.

    Args:
        results: Per-question pipeline outputs (from run_pipeline), ordered
                 to match golden.
        golden:  Loaded golden question list (from load_golden).

    Returns:
        Dict of metric_name → mean score.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.executor import RunConfig
    from ragas.metrics import (
        AnswerRelevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Groq's API rejects n>1; AnswerRelevancy defaults to strictness=3 which
    # sends n=3 in one call. Setting strictness=1 generates one question per
    # answer instead, keeping n=1 throughout.
    answer_relevancy = AnswerRelevancy(strictness=1)

    contexts = [r["retrieved_chunks"] for r in results]
    assert all(isinstance(c, list) for c in contexts), \
        "contexts must be list[list[str]]"

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": contexts,
        "ground_truth": [g["expected_answer"] for g in golden],
    })

    evaluator_llm = _build_ragas_llm()
    evaluator_embeddings = _build_ragas_embeddings()

    logger.info("Running RAGAS evaluation on %d samples", len(results))
    # max_workers=1 forces sequential scoring to avoid parallel Groq calls
    # that exhaust the free-tier rate limit. RAGAS 0.1.21 has no is_async
    # parameter — RunConfig is the correct way to control concurrency.
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1),
    )

    scores = dict(result)
    logger.info("RAGAS scores: %s", scores)
    return scores


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def save_results(
    results: list[dict],
    scores: dict,
    output_dir: str = "eval/results",
) -> None:
    """Persist per-question pipeline results and aggregate RAGAS scores to disk.

    Creates output_dir if it does not exist.  All filenames include a
    timestamp so repeated runs never overwrite each other.

    Files written:
      pipeline_results_<YYYYMMDD_HHMMSS>.json  — list of per-question results
      ragas_scores_<YYYYMMDD_HHMMSS>.json       — aggregate metric scores

    Args:
        results:    Per-question pipeline outputs.
        scores:     Aggregate RAGAS metric scores.
        output_dir: Directory to write into (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"pipeline_results_{timestamp}.json")
    scores_path = os.path.join(output_dir, f"ragas_scores_{timestamp}.json")

    # Serialise Enum members (e.g. CitationStatus) that json.dumps can't handle.
    serialisable_results = json.loads(json.dumps(results, default=str))

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable_results, fh, indent=2, ensure_ascii=False)
    logger.info("Pipeline results saved to '%s'", results_path)

    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=2, ensure_ascii=False)
    logger.info("RAGAS scores saved to '%s'", scores_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, load golden set, run the pipeline, evaluate with RAGAS,
    save results, and print a pass/fail summary table to stdout.

    CLI flags
    ---------
    --limit N        Run only the first N questions (0 = all, default: settings.eval_limit).
    --collection S   ChromaDB collection name (default: settings.default_collection).
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="RAG Hybrid eval runner")
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.eval_limit,
        help="Run only the first N questions (0 = all).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.default_collection,
        help="ChromaDB collection name to query.",
    )
    args = parser.parse_args()

    golden = load_golden(settings.golden_path)

    if args.limit > 0:
        golden = golden[: args.limit]
        logger.info("--limit %d: restricting to %d questions", args.limit, len(golden))

    logger.info("Running pipeline on %d questions", len(golden))
    results: list[dict] = []
    for entry in golden:
        logger.info(
            "Q%s [%s]: %s",
            entry["id"],
            entry["type"],
            entry["question"][:80],
        )
        result = run_pipeline(entry["question"], collection_name=args.collection)

        # Annotate with id/type for the summary table and saved output.
        if result:
            result["id"] = entry["id"]
            result["type"] = entry["type"]
        else:
            result = {
                "id": entry["id"],
                "type": entry["type"],
                "question": entry["question"],
                "answer": "",
                "citations": [],
                "support_score": 0.0,
                "retrieved_chunks": [],
                "retrieval_method": "hybrid",
            }

        results.append(result)
        # Groq free tier caps at ~30 req/min; sleep between questions to
        # avoid 429s from citation verification and generation calls.
        time.sleep(settings.eval_sleep_seconds)

    scores = evaluate_with_ragas(results, golden)
    save_results(results, scores)

    # Summary table.
    col_w = (14, 12, 14, 10)
    header = (
        f"{'Question ID':<{col_w[0]}} "
        f"{'Type':<{col_w[1]}} "
        f"{'Support Score':>{col_w[2]}} "
        f"{'Pass/Fail':>{col_w[3]}}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for result in results:
        qid = result.get("id", "?")
        qtype = result.get("type", "?")
        score = result.get("support_score", 0.0)
        verdict = "PASS" if score >= _PASS_THRESHOLD else "FAIL"
        print(
            f"{qid:<{col_w[0]}} "
            f"{qtype:<{col_w[1]}} "
            f"{score:>{col_w[2]}.3f} "
            f"{verdict:>{col_w[3]}}"
        )
    print(sep)

    if scores:
        print("\nRAGAS Scores:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
    print()


if __name__ == "__main__":
    main()
