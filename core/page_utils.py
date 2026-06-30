"""
core/page_utils.py
------------------
Shared helpers for Streamlit sidebar pages (charts, results_table, human_validation).

Sidebar pages run in a separate session from app.py and have no access to
app.py's session_state. This module loads persisted results from disk so
all pages can display data without re-running the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from core.cache_manager import load_results
from core.search_engine import SearchResult


def _dict_to_result(d: Dict[str, Any]) -> SearchResult:
    """Reconstruct a SearchResult dataclass from a plain dict."""
    return SearchResult(
        doc_filename=d.get("doc_filename", ""),
        page=int(d.get("page", 0)),
        excerpt=d.get("excerpt", ""),
        theme=d.get("theme", ""),
        keyword_hit=bool(d.get("keyword_hit", False)),
        matched_keyword=d.get("matched_keyword", ""),
        sbert_score=float(d.get("sbert_score", 0.0)),
        openai_score=float(d.get("openai_score", 0.0)),
        cross_encoder_score=float(d.get("cross_encoder_score", 0.0)),
        final_score=float(d.get("final_score", 0.0)),
        eo_relevance_score=float(d.get("eo_relevance_score", 0.0)),
        context=d.get("context", ""),
        validation_category=d.get("validation_category", "UNSCORED"),
        human_label=d.get("human_label", ""),
        chunk_index=int(d.get("chunk_index", -1)),
    )


def get_results_for_page() -> List[SearchResult]:
    """
    Return results for a sidebar page.

    Priority:
    1. Already in session_state (user navigated from app.py in same session)
    2. Load from persisted disk file (different session, e.g. direct URL visit)
    3. Empty list — show "no results yet" message
    """
    # Check session_state first (same session as app.py)
    if st.session_state.get("results"):
        return st.session_state["results"]

    # Try loading from disk
    if "_page_results_loaded" not in st.session_state:
        payload = load_results()
        if payload and payload.get("results"):
            try:
                results = [_dict_to_result(r) for r in payload["results"]]
                st.session_state["results"] = results
                st.session_state["corpus_filenames"] = payload.get("corpus_filenames", [])
                st.session_state["analysis_done"] = True
                st.session_state["_page_results_loaded"] = True
                return results
            except Exception as exc:
                st.warning(f"Could not load saved results: {exc}")
        st.session_state["_page_results_loaded"] = True

    return st.session_state.get("results", [])


def no_results_message(page_key: str = "default"):
    """Standard message shown when no results are available.

    Args:
        page_key: unique string per page (e.g. 'results', 'charts',
                  'validation') so Streamlit never sees duplicate keys.
    """
    st.info(
        "No analysis results yet.  \n"
        "Go to the **main page**, upload PDFs or enter a folder path, "
        "and click **Run Analysis**."
    )
    if st.button("Go to main page", key=f"no_results_go_main_{page_key}"):
        st.switch_page("app.py")
