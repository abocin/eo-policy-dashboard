"""
pages/charts.py  —  v1.3
------------------------
All visualisations for the dashboard.

v1.3 change: charts that iterate over all policy documents (heatmap,
stacked themes bar) now include a "Top N documents" selector above each
chart so the view can be scoped to the most-relevant subset without
re-running the analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from core.search_engine import SearchResult
from core.exporters import results_to_dataframe

THEME_COLOR_MAP = {
    "EO Downstream Skills": "#0984e3",
    "Space Industry Skills": "#6c5ce7",
    "Geospatial & GIS": "#00b894",
    "Copernicus & EU Space Services": "#e17055",
    "Digital Skills for Space": "#fdcb6e",
    "Skills Gaps & Workforce Development": "#a29bfe",
    "Policy Support for Downstream Applications": "#55efc4",
}

# ── helpers ────────────────────────────────────────────────────────────────

def _top_n_docs(df: pd.DataFrame, n: int, score_col: str = "Final Score") -> list[str]:
    """Return the full (untruncated) document names of the top-N by mean score."""
    return (
        df.groupby("Document")[score_col]
        .mean()
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )


def _top_n_selector(
    df: pd.DataFrame,
    key_suffix: str,
    score_col: str = "Final Score",
    default: int = 10,
) -> int:
    """
    Renders a compact radio that lets the user pick between Top 10, Top 20,
    or All documents.  Returns the chosen integer (or len(unique docs) for All).
    """
    total = df["Document"].nunique()
    options = [10, 20, total]
    labels  = ["Top 10", "Top 20", f"All ({total})"]
    # only show options that make sense
    shown = [(lbl, val) for lbl, val in zip(labels, options) if val <= total]
    if len(shown) <= 1:
        return total  # nothing to choose — just return all

    chosen_lbl = st.radio(
        "Documents shown",
        [lbl for lbl, _ in shown],
        index=0,
        horizontal=True,
        key=f"topn_{key_suffix}",
    )
    return dict(shown)[chosen_lbl]


def render_charts(results: List[SearchResult], taxonomy: Dict[str, Any]):
    if not results:
        st.info("No results to visualise yet.")
        return

    try:
        df = results_to_dataframe(results)
    except Exception as e:
        st.error(f"Could not build dataframe: {e}")
        return

    if df.empty:
        st.warning("Results dataframe is empty.")
        return

    relevant = df[df["Validation Category"].isin(["VALID EVIDENCE", "WEAK EVIDENCE"])].copy()

    if relevant.empty:
        st.warning(
            "No valid or weak evidence found — try lowering the thresholds in the sidebar "
            "and re-running the analysis."
        )
        return

    def short_name(name: str, n: int = 35) -> str:
        return name[:n] + "…" if len(name) > n else name

    relevant["Doc"] = relevant["Document"].apply(short_name)
    df["Doc"] = df["Document"].apply(short_name)

    col_left, col_right = st.columns(2)

    # ---- 1. Excerpts per policy ----------------------------------------
    with col_left:
        st.markdown("#### Evidence excerpts per document")
        try:
            counts = (
                relevant.groupby("Doc", as_index=False)
                .size()
                .rename(columns={"size": "Count"})
                .sort_values("Count", ascending=True)
            )
            fig1 = px.bar(
                counts, x="Count", y="Doc", orientation="h",
                color="Count", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig1.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, coloraxis_showscale=False,
                height=max(280, len(counts) * 28),
                yaxis_title="", xaxis_title="Excerpts",
            )
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    # ---- 2. Validation category donut ---------------------------------
    with col_right:
        st.markdown("#### Match confidence breakdown")
        try:
            cat_counts = df["Validation Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            color_map = {
                "VALID EVIDENCE": "#00b894",
                "WEAK EVIDENCE": "#fdcb6e",
                "NOT RELEVANT": "#d63031",
                "NEEDS REVIEW": "#74b9ff",
                "UNSCORED": "#636e72",
            }
            fig2 = px.pie(
                cat_counts, values="Count", names="Category", hole=0.5,
                color="Category", color_discrete_map=color_map,
                template="plotly_dark",
            )
            fig2.update_traces(textposition="outside", textinfo="percent+label")
            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    st.divider()

    # ---- 3. Score distribution histogram ------------------------------
    st.markdown("#### Score distribution")
    try:
        valid_t = taxonomy.get("thresholds", {}).get("valid_match", 0.50)
        weak_t  = taxonomy.get("thresholds", {}).get("weak_match",  0.35)

        fig3 = px.histogram(
            df, x="Final Score", nbins=40,
            color="Validation Category",
            color_discrete_map={
                "VALID EVIDENCE": "#00b894",
                "WEAK EVIDENCE": "#fdcb6e",
                "NOT RELEVANT": "#d63031",
                "UNSCORED": "#636e72",
            },
            template="plotly_dark", barmode="overlay", opacity=0.75,
        )
        fig3.add_vline(x=valid_t, line_dash="dash", line_color="#00b894",
                       annotation_text=f"Valid ≥{valid_t}",
                       annotation_position="top right")
        fig3.add_vline(x=weak_t, line_dash="dash", line_color="#fdcb6e",
                       annotation_text=f"Weak ≥{weak_t}",
                       annotation_position="top left")
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    st.divider()

    # ---- 4. Policy × Skills theme heatmap  (TOP-N) --------------------
    st.markdown("#### Policy × Skills theme heatmap")
    st.caption("Colour = max final score per cell. Ranked by mean score across themes.")

    try:
        _c_sel, _c_info = st.columns([3, 7])
        with _c_sel:
            n4 = _top_n_selector(relevant, key_suffix="heatmap")
        with _c_info:
            if relevant["Document"].nunique() > n4:
                st.caption(
                    f"Showing the {n4} documents with the highest mean relevance score. "
                    "Increase to see more, or export the full set as CSV from the Results page."
                )

        top_docs4 = _top_n_docs(relevant, n4)
        pivot = (
            relevant[relevant["Document"].isin(top_docs4)]
            .pivot_table(
                index="Doc", columns="Theme",
                values="Final Score", aggfunc="max",
            )
            .fillna(0)
        )
        # Re-order rows by mean score descending (most relevant at top)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]

        if not pivot.empty:
            fig4 = px.imshow(
                pivot, color_continuous_scale="Blues",
                aspect="auto", template="plotly_dark",
                zmin=0, zmax=1, text_auto=".2f",
            )
            fig4.update_layout(
                margin=dict(l=10, r=10, t=10, b=80),
                height=max(300, len(pivot) * 38),
                xaxis_title="Skill Theme", yaxis_title="Policy Document",
                coloraxis_colorbar_title="Score",
            )
            fig4.update_xaxes(tickangle=-35)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Not enough data for heatmap.")
    except Exception as e:
        st.error(f"Heatmap error: {e}")

    st.divider()

    # ---- 5. Skill themes per document stacked bar  (TOP-N) ------------
    st.markdown("#### Skill themes per document")
    st.caption("Ranked by total evidence excerpts.")

    try:
        _c_sel5, _c_info5 = st.columns([3, 7])
        with _c_sel5:
            n5 = _top_n_selector(relevant, key_suffix="themes_bar")
        with _c_info5:
            if relevant["Document"].nunique() > n5:
                st.caption(
                    f"Showing the {n5} documents with the most evidence excerpts."
                )

        # For this chart rank by excerpt count, not score
        top_docs5 = (
            relevant.groupby("Document")
            .size()
            .sort_values(ascending=False)
            .head(n5)
            .index.tolist()
        )
        theme_counts = (
            relevant[relevant["Document"].isin(top_docs5)]
            .groupby(["Doc", "Theme"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig5 = px.bar(
            theme_counts, x="Doc", y="Count", color="Theme",
            color_discrete_map=THEME_COLOR_MAP,
            template="plotly_dark", barmode="stack",
        )
        fig5.update_layout(
            margin=dict(l=10, r=10, t=10, b=120),
            height=380, xaxis_tickangle=-40,
            xaxis_title="", yaxis_title="Excerpts",
            legend=dict(orientation="h", yanchor="bottom", y=-0.6),
        )
        st.plotly_chart(fig5, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")


# ---------------------------------------------------------------------------
# Standalone page execution (when accessed directly via sidebar URL)
# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:
    from core.page_utils import get_results_for_page, no_results_message
    from core.taxonomy_loader import load_taxonomy
    _results = get_results_for_page()
    if not _results:
        no_results_message(page_key="charts")
    else:
        _taxonomy = load_taxonomy()
        render_charts(_results, _taxonomy)
