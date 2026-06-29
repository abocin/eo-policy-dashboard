"""
pages/human_validation.py
--------------------------
Lets the analyst review each evidence excerpt and assign a human label:
  - Valid evidence
  - Weak evidence
  - Not relevant
  - Needs review

Labels are stored in session_state["human_labels"] so they persist across
re-runs and are included in exports.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from core.search_engine import SearchResult
from core.exporters import results_to_dataframe


LABEL_OPTIONS = ["(not reviewed)", "Valid evidence", "Weak evidence", "Not relevant", "Needs review"]

LABEL_COLORS = {
    "Valid evidence": "#00b894",
    "Weak evidence": "#fdcb6e",
    "Not relevant": "#d63031",
    "Needs review": "#74b9ff",
    "(not reviewed)": "#636e72",
}


def _result_key(r: SearchResult) -> str:
    return f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"


def render_human_validation(results: List[SearchResult]):
    st.subheader("Human Validation")
    st.caption(
        "Review the machine-scored excerpts and assign your own judgement. "
        "Labels persist within this session and are included in all exports."
    )

    if not results:
        st.info("No results to review yet.")
        return

    if "human_labels" not in st.session_state:
        st.session_state["human_labels"] = {}

    df = results_to_dataframe(results)

    # ---- Filter controls ---------------------------------------------------
    col1, col2, col3 = st.columns(3)

    doc_filter = col1.selectbox(
        "Filter by document",
        ["All documents"] + sorted(df["Document"].unique().tolist()),
    )
    theme_filter = col2.selectbox(
        "Filter by theme",
        ["All themes"] + sorted(df["Theme"].unique().tolist()),
    )
    review_filter = col3.selectbox(
        "Show",
        ["All excerpts", "Not yet reviewed", "Reviewed only", "VALID EVIDENCE only"],
    )

    filtered_results = results

    if doc_filter != "All documents":
        filtered_results = [r for r in filtered_results if r.doc_filename == doc_filter]
    if theme_filter != "All themes":
        filtered_results = [r for r in filtered_results if r.theme == theme_filter]
    if review_filter == "Not yet reviewed":
        filtered_results = [
            r for r in filtered_results
            if st.session_state["human_labels"].get(_result_key(r), "(not reviewed)") == "(not reviewed)"
        ]
    elif review_filter == "Reviewed only":
        filtered_results = [
            r for r in filtered_results
            if st.session_state["human_labels"].get(_result_key(r), "(not reviewed)") != "(not reviewed)"
        ]
    elif review_filter == "VALID EVIDENCE only":
        filtered_results = [r for r in filtered_results if r.validation_category == "VALID EVIDENCE"]

    # Sort by final_score descending
    filtered_results = sorted(filtered_results, key=lambda x: x.final_score, reverse=True)

    st.caption(f"Reviewing **{len(filtered_results)}** of {len(results)} excerpts")
    st.divider()

    # ---- Progress tracker --------------------------------------------------
    reviewed_count = sum(
        1 for r in results
        if st.session_state["human_labels"].get(_result_key(r), "(not reviewed)") != "(not reviewed)"
    )
    st.progress(
        reviewed_count / len(results),
        text=f"Reviewed {reviewed_count} / {len(results)} excerpts",
    )
    st.divider()

    # ---- Per-excerpt review cards ------------------------------------------
    # Batch save shortcut
    batch_label = st.selectbox(
        "Quick-label ALL currently shown excerpts as:",
        ["— select to batch label —"] + LABEL_OPTIONS[1:],
        key="batch_label_select",
    )
    if st.button("Apply batch label", key="apply_batch") and batch_label != "— select to batch label —":
        for r in filtered_results:
            st.session_state["human_labels"][_result_key(r)] = batch_label
            r.human_label = batch_label
        st.success(f"Labelled {len(filtered_results)} excerpts as '{batch_label}'")

    st.divider()

    for i, r in enumerate(filtered_results[:100]):
        key = _result_key(r)
        current_label = st.session_state["human_labels"].get(key, "(not reviewed)")
        color = LABEL_COLORS.get(current_label, "#636e72")

        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 4px solid {color}; padding: 0.4rem 0.8rem;
                             background: #0e1117; border-radius: 4px;">
                  <span style="font-size:0.82rem; color:#aaa;">
                    <strong>{r.doc_filename}</strong> &nbsp;·&nbsp; p.{r.page}
                    &nbsp;·&nbsp; {r.theme}
                    &nbsp;·&nbsp; score: {r.final_score:.3f}
                    &nbsp;·&nbsp; auto: <em>{r.validation_category}</em>
                  </span>
                  <div style="margin-top:0.3rem; color:#e8f0f7; font-size:0.9rem;">
                    {r.excerpt}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            label = st.radio(
                "Your label",
                LABEL_OPTIONS,
                index=LABEL_OPTIONS.index(current_label),
                key=f"label_{key}_{i}",
                horizontal=True,
                label_visibility="collapsed",
            )

            if label != current_label:
                st.session_state["human_labels"][key] = label
                r.human_label = label

            st.markdown("---")

    if len(filtered_results) > 100:
        st.caption(
            f"Showing first 100 of {len(filtered_results)}. "
            "Use filters above to narrow down, or use batch labelling."
        )


# ---------------------------------------------------------------------------
# Standalone page execution (when accessed directly via sidebar URL)
# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:
    from core.page_utils import get_results_for_page, no_results_message
    _results = get_results_for_page()
    if not _results:
        no_results_message()
    else:
        render_human_validation(_results)
