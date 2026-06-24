"""
pages/charts.py
---------------
All visualisations for the dashboard.
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

    # Truncate long document names for all charts
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
        weak_t = taxonomy.get("thresholds", {}).get("weak_match", 0.35)

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
                       annotation_text=f"Valid ≥{valid_t}", annotation_position="top right")
        fig3.add_vline(x=weak_t, line_dash="dash", line_color="#fdcb6e",
                       annotation_text=f"Weak ≥{weak_t}", annotation_position="top left")
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    st.divider()

    # ---- 4. Policy × Theme heatmap ------------------------------------
    st.markdown("#### Policy × Skills theme heatmap")
    st.caption("Colour = max final score per cell")
    try:
        pivot = relevant.pivot_table(
            index="Doc", columns="Theme",
            values="Final Score", aggfunc="max",
        ).fillna(0)

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

    # ---- 5. Themes per document stacked bar ---------------------------
    st.markdown("#### Skill themes per document")
    try:
        theme_counts = (
            relevant.groupby(["Doc", "Theme"], as_index=False)
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
