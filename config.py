# config.py
"""Centralised settings for the RAG Hybrid project.

All tuneable values live here.  Modules read from the ``settings`` singleton
rather than calling ``os.environ`` or hard-coding defaults directly.

Environment variables are loaded from a ``.env`` file at the project root
(via pydantic-settings).  Any variable not present in ``.env`` falls back
to the default defined on the model.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide configuration loaded from environment / .env file."""

    # ------------------------------------------------------------------
    # LLM — Groq OpenAI-compatible endpoint
    # ------------------------------------------------------------------

    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY"),
    )
    """GROQ_API_KEY or LLM_API_KEY (both accepted)."""

    llm_api_base: str = "https://api.groq.com/openai/v1"
    """Base URL for the LLM API endpoint."""

    llm_model: str = "llama-3.1-8b-instant"
    """Model identifier sent to the LLM endpoint."""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    retrieval_top_k: int = 50
    """Number of candidates to fetch from each retriever before fusion."""

    fusion_top_k: int = 20
    """Number of results to keep after RRF fusion."""

    rerank_top_k: int = 5
    """Number of results to keep after cross-encoder reranking."""

    default_collection: str = "rag_hybrid"
    """ChromaDB collection name used by the pipeline."""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    eval_sleep_seconds: int = 3
    """Seconds to sleep between questions during eval to avoid Groq 429s."""

    eval_limit: int = 0
    """Maximum number of golden questions to evaluate. 0 = run all."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    chroma_path: str = "chroma_db"
    """Relative or absolute path to the ChromaDB persistence directory."""

    golden_path: str = "eval/golden/golden.json"
    """Path to the golden test set JSON file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
"""Module-level singleton — import and use directly: ``from config import settings``."""
