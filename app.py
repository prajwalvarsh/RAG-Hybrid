"""Streamlit demo UI for the RAG Hybrid Search system.

Connects to the FastAPI backend at BASE_URL and provides two tabs:
  - Upload & Ingest: upload a document, pick a collection and chunking strategy
  - Query: ask a natural-language question against any indexed collection
"""

import logging

import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_collections() -> list[str]:
    """Fetch collection names from GET /collections.

    Returns an empty list if the API is unreachable, so the UI degrades
    gracefully rather than crashing on startup.
    """
    try:
        resp = requests.get(f"{BASE_URL}/collections", timeout=5)
        resp.raise_for_status()
        return [entry["name"] for entry in resp.json()]
    except Exception as exc:
        logger.warning("Could not fetch collections: %s", exc)
        return []


def _post_ingest(
    uploaded_files: list,
    collection_name: str,
    strategy: str,
) -> dict:
    """POST /ingest with one or more files as a multipart form batch.

    Sends all files under the same 'file' multipart key so FastAPI
    receives them as List[UploadFile].  collection_name and strategy
    are shared across the batch.

    Args:
        uploaded_files:  List of Streamlit UploadedFile objects.
        collection_name: Target ChromaDB collection.
        strategy:        Chunking strategy value string (fixed/structural/semantic).

    Returns:
        Parsed JSON response dict from the API (IngestResponse shape).

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
    """
    files_param = [("files", (f.name, f.getvalue())) for f in uploaded_files]
    resp = requests.post(
        f"{BASE_URL}/ingest",
        files=files_param,
        data={"collection_name": collection_name, "strategy": strategy},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _post_query(question: str, collection_name: str) -> dict:
    """POST /query and return the parsed response dict.

    Args:
        question:        Natural-language question from the user.
        collection_name: ChromaDB collection to search.

    Returns:
        Parsed JSON response dict from the API.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
    """
    resp = requests.post(
        f"{BASE_URL}/query",
        json={"question": question, "collection_name": collection_name},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Hybrid Search", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("RAG Hybrid Search")
    st.markdown(
        "Production-grade hybrid retrieval (dense + sparse BM25) with "
        "reranking and grounded generation.\n\n"
        "**Model:** `llama-3.1-8b-instant` via Groq  \n"
        "**Embeddings:** `BAAI/bge-base-en-v1.5`  \n"
        "**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`"
    )

    st.divider()
    st.subheader("Live Collections")
    collections = _fetch_collections()
    if collections:
        for col in collections:
            st.markdown(f"- `{col}`")
    else:
        st.caption("No collections found (is the API running?)")

# --- Tabs ---
tab_ingest, tab_query = st.tabs(["Upload & Ingest", "Query"])


# ---------------------------------------------------------------------------
# Tab 1 — Upload & Ingest
# ---------------------------------------------------------------------------

with tab_ingest:
    st.header("Upload & Ingest Documents")

    st.info("Max 10 MB per file. Supported formats: PDF, TXT, MD.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        collection_name_input = st.text_input(
            "Collection name", value="rag_hybrid", help="ChromaDB collection to write into"
        )
    with col_right:
        strategy_label = st.selectbox(
            "Chunking strategy",
            options=["Fixed", "Structural", "Semantic"],
            help="How the document will be split into chunks",
        )

    if strategy_label in ("Structural", "Semantic"):
        st.warning(
            "Note: Structural and Semantic chunking require well-structured documents "
            "with clear headings. Flat PDFs may produce very few chunks."
        )

    _STRATEGY_MAP = {"Fixed": "fixed", "Structural": "structural", "Semantic": "semantic"}

    if st.button("Ingest", type="primary", disabled=not uploaded_files):
        if not collection_name_input.strip():
            st.error("Collection name cannot be empty.")
        else:
            with st.spinner(
                f"Ingesting {len(uploaded_files)} file(s) — this may take a minute…"
            ):
                try:
                    result = _post_ingest(
                        uploaded_files=uploaded_files,
                        collection_name=collection_name_input.strip(),
                        strategy=_STRATEGY_MAP[strategy_label],
                    )

                    file_results = result.get("files", [])
                    n_ok = sum(1 for r in file_results if r["status"] == "success")
                    n_err = sum(1 for r in file_results if r["status"] == "error")

                    # --- summary banner ---
                    if n_err == 0:
                        st.success(
                            f"All {n_ok} file(s) ingested successfully into "
                            f"`{collection_name_input.strip()}`."
                        )
                    else:
                        st.warning(
                            f"{n_ok} file(s) succeeded, {n_err} file(s) failed. "
                            "See table below for details."
                        )

                    # --- results table ---
                    table_rows = [
                        {
                            "filename": r["filename"],
                            "chunks": r["chunk_count"],
                            "time (ms)": f"{r['elapsed_ms']:.0f}",
                            "status": r["status"],
                            "error": r.get("error") or "",
                        }
                        for r in file_results
                    ]
                    st.dataframe(table_rows, use_container_width=True)

                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Ingest failed: {detail}")
                except requests.ConnectionError:
                    st.error(f"Could not connect to API at {BASE_URL}. Is the server running?")


# ---------------------------------------------------------------------------
# Tab 2 — Query
# ---------------------------------------------------------------------------

with tab_query:
    st.header("Ask a Question")

    query_collections = _fetch_collections()
    if not query_collections:
        st.info("No collections found. Upload a document first.")
    else:
        selected_collection = st.selectbox(
            "Collection", options=query_collections, help="ChromaDB collection to search"
        )

        question_input = st.text_input(
            "Question", placeholder="What did the document say about…?"
        )

        if st.button("Submit", type="primary", disabled=not question_input.strip()):
            with st.spinner("Running hybrid retrieval + generation…"):
                try:
                    result = _post_query(
                        question=question_input.strip(),
                        collection_name=selected_collection,
                    )

                    # --- Answer ---
                    st.subheader("Answer")
                    st.markdown(
                        f'<div style="background:#f0f4ff;padding:1rem;border-radius:8px;">'
                        f'{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # --- Metrics ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Support score", f"{result['support_score']:.2f}")
                    m2.metric("Retrieval method", result["retrieval_method"])
                    m3.metric("Latency", f"{result['latency_ms']:.0f} ms")

                    # --- Citations ---
                    with st.expander(f"Citations ({len(result['citations'])})"):
                        for cit in result["citations"]:
                            status_icon = "✓" if cit["supported"] == "supported" else "✗"
                            st.markdown(
                                f"**[{cit['citation_number']}]** {status_icon} "
                                f"*{cit['claim']}*  \n"
                                f"Chunk `{cit['chunk_id']}` — "
                                f"confidence {cit['confidence']:.2f}"
                            )

                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Query failed: {detail}")
                except requests.ConnectionError:
                    st.error(f"Could not connect to API at {BASE_URL}. Is the server running?")
