"""
pages/gap_analysis.py
---------------------
Policy Intelligence Dashboard — four analytical views:

  1. Gap Analysis        — which EO themes are covered / missing per policy
  2. Maturity Assessment — 1–6 EO maturity scale per policy document
  3. Commitment Strength — BINDING / STRONG / MODERATE / ASPIRATIONAL distribution
  4. Lifecycle Stage     — where in the EO capacity-building lifecycle each policy focuses

Reads results from session_state or disk (same as other sidebar pages).
No re-run required — all four analyses run on the cached SearchResult list.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.search_engine import SearchResult


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

COVERAGE_COLORS = {
    "STRONG":   "#00b894",
    "MODERATE": "#fdcb6e",
    "WEAK":     "#e17055",
    "MISSING":  "#2d3436",
}

MATURITY_COLORS = {
    1: "#636e72",
    2: "#b2bec3",
    3: "#74b9ff",
    4: "#0984e3",
    5: "#6c5ce7",
    6: "#00b894",
}

COMMITMENT_COLORS = {
    "BINDING":     "#00b894",
    "STRONG":      "#0984e3",
    "MODERATE":    "#fdcb6e",
    "ASPIRATIONAL":"#e17055",
}

LIFECYCLE_ORDER = [
    "Awareness", "Education", "Training", "Skills",
    "Innovation", "Entrepreneurship", "Adoption", "Sustainability",
]

LIFECYCLE_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_name(name: str, max_len: int = 30) -> str:
    return name[:max_len] + "…" if len(name) > max_len else name


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_gap_analysis(results: List[SearchResult], taxonomy: Dict[str, Any]):
    from core.gap_analyzer import per_doc_coverage, corpus_gap_report
    from core.maturity_scorer import score_maturity
    from core.commitment_analyzer import analyse_commitment
    from core.lifecycle_tagger import DEFAULT_STAGES

    st.title("🔬 Policy Intelligence")
    st.caption(
        "Four analytical lenses applied to your evidence excerpts. "
        "Re-run the analysis to refresh after taxonomy changes."
    )

    # ---- Run all four analyses --------------------------------------------
    coverage     = per_doc_coverage(results, taxonomy)
    gap_report   = corpus_gap_report(results, taxonomy)
    maturity_map = score_maturity(results, taxonomy)

    themes       = [t["label"] for t in taxonomy.get("themes", [])]
    docs         = list(coverage.keys())
    short_docs   = [_short_name(d) for d in docs]

    df_results = pd.DataFrame([
        {
            "Document": r.doc_filename,
            "Theme": r.theme,
            "Final Score": r.final_score,
            "Commitment Level": getattr(r, "commitment_level", "") or "ASPIRATIONAL",
            "Commitment Score": getattr(r, "commitment_score", 0.0),
            "Lifecycle Stage": getattr(r, "lifecycle_stage", "") or "Skills",
            "Validation Category": r.validation_category,
        }
        for r in results
    ])

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Gap Analysis",
        "🎯 Maturity Assessment",
        "💬 Commitment Strength",
        "🔄 Lifecycle Stage",
    ])

    # =======================================================================
    # TAB 1 — Gap Analysis
    # =======================================================================
    with tab1:
        st.subheader("Theme Coverage per Policy")

        # ---- Corpus summary metrics ----------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total themes", gap_report.total_themes)
        m2.metric("Covered (≥ MODERATE)", gap_report.covered_themes,
                  help="At least one document has MODERATE or STRONG coverage")
        m3.metric("Gap themes", len(gap_report.gap_themes),
                  delta=f"-{gap_report.gap_pct:.0f}%", delta_color="inverse",
                  help="Themes with no evidence across the full corpus")
        m4.metric("Partial themes", len(gap_report.partial_themes),
                  help="Themes with only WEAK evidence")

        st.divider()

        # ---- Coverage heatmap: docs × themes --------------------------------
        st.markdown("#### Coverage heatmap — all policies × all themes")

        # Build numeric matrix (0=MISSING, 1=WEAK, 2=MODERATE, 3=STRONG)
        level_to_num = {"MISSING": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
        matrix_data = []
        for doc in docs:
            row = [level_to_num.get(coverage[doc][t].coverage_level, 0) for t in themes]
            matrix_data.append(row)

        fig_heat = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=themes,
            y=short_docs,
            colorscale=[
                [0.0,  "#2d3436"],
                [0.33, "#e17055"],
                [0.66, "#fdcb6e"],
                [1.0,  "#00b894"],
            ],
            zmin=0, zmax=3,
            colorbar=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["Missing", "Weak", "Moderate", "Strong"],
                title="Coverage",
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Theme: %{x}<br>"
                "Coverage: %{text}<extra></extra>"
            ),
            text=[[
                coverage[doc][t].coverage_level for t in themes
            ] for doc in docs],
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            height=max(400, len(docs) * 28),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ---- Gap themes list -----------------------------------------------
        st.divider()
        col_gap, col_partial = st.columns(2)

        with col_gap:
            st.markdown("#### ❌ Gap themes (no coverage)")
            if gap_report.gap_themes:
                for gt in gap_report.gap_themes:
                    st.markdown(f"- {gt}")
            else:
                st.success("All themes covered in at least one document.")

        with col_partial:
            st.markdown("#### ⚠️ Partial themes (weak only)")
            if gap_report.partial_themes:
                for pt in gap_report.partial_themes:
                    st.markdown(f"- {pt}")
            else:
                st.success("No themes with only weak coverage.")

        # ---- Per-theme doc count bar chart --------------------------------
        st.divider()
        st.markdown("#### Themes ranked by number of covering policies")

        theme_doc_counts = {
            t: len(gap_report.per_theme_docs[t]) for t in themes
        }
        df_theme_cov = pd.DataFrame({
            "Theme": list(theme_doc_counts.keys()),
            "Policies Covering": list(theme_doc_counts.values()),
            "Avg Score": [gap_report.theme_scores[t] for t in themes],
        }).sort_values("Policies Covering", ascending=True)

        fig_bar = px.bar(
            df_theme_cov, x="Policies Covering", y="Theme",
            orientation="h",
            color="Avg Score",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            labels={"Policies Covering": "# Policies (MODERATE+)"},
        )
        fig_bar.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---- Per-doc gap table ---------------------------------------------
        st.divider()
        st.markdown("#### Per-policy gap detail")

        doc_sel = st.selectbox(
            "Select policy to inspect",
            docs,
            key="gap_doc_sel",
        )
        if doc_sel:
            doc_cov = coverage[doc_sel]
            rows = []
            for theme, rec in doc_cov.items():
                rows.append({
                    "Theme": theme,
                    "Coverage": rec.coverage_level,
                    "Excerpts": rec.hit_count,
                    "Max Score": rec.max_score,
                    "Avg Score": rec.avg_score,
                    "Keyword Hits": rec.keyword_hits,
                    "Semantic Hits": rec.semantic_hits,
                })
            df_doc = pd.DataFrame(rows).sort_values("Max Score", ascending=False)
            st.dataframe(
                df_doc,
                use_container_width=True,
                column_config={
                    "Max Score": st.column_config.ProgressColumn(
                        "Max Score", min_value=0, max_value=1, format="%.3f"
                    ),
                    "Avg Score": st.column_config.ProgressColumn(
                        "Avg Score", min_value=0, max_value=1, format="%.3f"
                    ),
                },
            )

    # =======================================================================
    # TAB 2 — Maturity Assessment
    # =======================================================================
    with tab2:
        st.subheader("EO Maturity Assessment per Policy")
        st.caption(
            "Scores each policy 1–6: "
            "1=Absent · 2=Indirect · 3=Recognised · 4=Targeted · 5=Funded · 6=Monitored"
        )

        if not maturity_map:
            st.info("No results to score.")
        else:
            # ---- Maturity level bar chart ----------------------------------
            mat_rows = [
                {
                    "Document": _short_name(doc),
                    "Level": rec.level,
                    "Label": rec.label,
                    "Evidence": rec.evidence_count,
                    "Max Score": rec.max_score,
                    "Funded": "✅" if rec.funded else "❌",
                    "Monitored": "✅" if rec.monitored else "❌",
                    "Notes": rec.notes,
                }
                for doc, rec in sorted(
                    maturity_map.items(), key=lambda x: x[1].level, reverse=True
                )
            ]
            df_mat = pd.DataFrame(mat_rows)

            fig_mat = px.bar(
                df_mat.sort_values("Level"),
                x="Level", y="Document",
                orientation="h",
                color="Level",
                color_continuous_scale=[
                    [0.0,  "#636e72"],
                    [0.2,  "#b2bec3"],
                    [0.4,  "#74b9ff"],
                    [0.6,  "#0984e3"],
                    [0.8,  "#6c5ce7"],
                    [1.0,  "#00b894"],
                ],
                range_color=[1, 6],
                template="plotly_dark",
                labels={"Level": "Maturity Level (1–6)"},
                hover_data=["Label", "Evidence", "Funded", "Monitored"],
            )
            fig_mat.update_layout(
                height=max(400, len(mat_rows) * 28),
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_colorbar=dict(
                    tickvals=[1, 2, 3, 4, 5, 6],
                    ticktext=["1 Absent", "2 Indirect", "3 Recognised",
                              "4 Targeted", "5 Funded", "6 Monitored"],
                    title="Maturity",
                ),
            )
            st.plotly_chart(fig_mat, use_container_width=True)

            # ---- Maturity distribution pie --------------------------------
            st.divider()
            col_pie, col_tbl = st.columns([1, 2])
            with col_pie:
                st.markdown("#### Distribution")
                level_counts = df_mat["Label"].value_counts().reset_index()
                level_counts.columns = ["Label", "Count"]
                fig_pie = px.pie(
                    level_counts, names="Label", values="Count",
                    template="plotly_dark",
                    color_discrete_sequence=list(MATURITY_COLORS.values()),
                )
                fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                fig_pie.update_layout(showlegend=False, margin=dict(l=5, r=5, t=5, b=5))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_tbl:
                st.markdown("#### Detail table")
                st.dataframe(
                    df_mat[["Document", "Level", "Label", "Evidence",
                             "Funded", "Monitored", "Notes"]],
                    use_container_width=True,
                    height=350,
                )

            # ---- Maturity level legend ------------------------------------
            st.divider()
            st.markdown("#### Maturity scale reference")
            for lvl, lbl in [
                (1, "Absent — EO never mentioned"),
                (2, "Indirect — EO implied (digital/spatial context)"),
                (3, "Recognised — EO explicitly mentioned, no actions"),
                (4, "Targeted — Specific EO actions/objectives defined"),
                (5, "Funded — Explicit budget or funding mechanism"),
                (6, "Monitored — KPIs, targets, monitoring framework"),
            ]:
                color = MATURITY_COLORS[lvl]
                st.markdown(
                    f"<span style='background:{color}; color:#fff; padding:2px 8px; "
                    f"border-radius:4px; font-size:0.85rem;'>{lvl}</span> &nbsp; {lbl}",
                    unsafe_allow_html=True,
                )

    # =======================================================================
    # TAB 3 — Commitment Strength
    # =======================================================================
    with tab3:
        st.subheader("Policy Commitment Strength")
        st.caption(
            "Classifies each excerpt by the strength of policy language: "
            "BINDING (shall/fund/allocate) → STRONG → MODERATE → ASPIRATIONAL (may/encourage)"
        )

        if df_results.empty:
            st.info("No results to analyse.")
        else:
            df_comm = df_results[df_results["Commitment Level"] != ""]

            # ---- Corpus-level commitment distribution ----------------------
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### Overall distribution")
                comm_counts = df_comm["Commitment Level"].value_counts().reset_index()
                comm_counts.columns = ["Level", "Count"]
                order = ["BINDING", "STRONG", "MODERATE", "ASPIRATIONAL"]
                comm_counts["Level"] = pd.Categorical(
                    comm_counts["Level"], categories=order, ordered=True
                )
                comm_counts = comm_counts.sort_values("Level")
                fig_comm = px.bar(
                    comm_counts, x="Level", y="Count",
                    color="Level",
                    color_discrete_map=COMMITMENT_COLORS,
                    template="plotly_dark",
                )
                fig_comm.update_layout(
                    showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                    height=320,
                )
                st.plotly_chart(fig_comm, use_container_width=True)

            with col_b:
                st.markdown("#### Avg commitment score per theme")
                theme_comm = (
                    df_comm.groupby("Theme")["Commitment Score"]
                    .mean()
                    .reset_index()
                    .sort_values("Commitment Score", ascending=True)
                )
                theme_comm.columns = ["Theme", "Avg Commitment Score"]
                fig_tcomm = px.bar(
                    theme_comm, x="Avg Commitment Score", y="Theme",
                    orientation="h",
                    color="Avg Commitment Score",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1],
                    template="plotly_dark",
                )
                fig_tcomm.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), height=320
                )
                st.plotly_chart(fig_tcomm, use_container_width=True)

            # ---- Per-doc commitment breakdown stacked bar ------------------
            st.divider()
            st.markdown("#### Commitment strength per policy")

            doc_comm = (
                df_comm.groupby(["Document", "Commitment Level"])
                .size()
                .reset_index(name="Count")
            )
            doc_comm["Document"] = doc_comm["Document"].apply(
                lambda x: _short_name(x, 35)
            )
            doc_comm["Commitment Level"] = pd.Categorical(
                doc_comm["Commitment Level"],
                categories=["ASPIRATIONAL", "MODERATE", "STRONG", "BINDING"],
                ordered=True,
            )
            fig_stack = px.bar(
                doc_comm, x="Count", y="Document",
                color="Commitment Level",
                orientation="h",
                color_discrete_map=COMMITMENT_COLORS,
                template="plotly_dark",
                barmode="stack",
            )
            fig_stack.update_layout(
                height=max(350, len(docs) * 28),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    # =======================================================================
    # TAB 4 — Lifecycle Stage
    # =======================================================================
    with tab4:
        st.subheader("Capacity Building Lifecycle Stage")
        st.caption(
            "Maps each excerpt to the stage of the EO capacity-building lifecycle "
            "it most strongly addresses."
        )

        if df_results.empty:
            st.info("No results to analyse.")
        else:
            df_lc = df_results[df_results["Lifecycle Stage"] != ""]

            # ---- Corpus overview sunburst ----------------------------------
            col_sun, col_bar = st.columns([1, 2])

            with col_sun:
                st.markdown("#### Corpus-wide stage distribution")
                lc_counts = df_lc["Lifecycle Stage"].value_counts().reset_index()
                lc_counts.columns = ["Stage", "Count"]
                fig_lc_pie = px.pie(
                    lc_counts, names="Stage", values="Count",
                    template="plotly_dark",
                    color_discrete_sequence=LIFECYCLE_COLORS,
                    hole=0.3,
                )
                fig_lc_pie.update_traces(textposition="outside", textinfo="percent+label")
                fig_lc_pie.update_layout(
                    showlegend=True, margin=dict(l=5, r=5, t=5, b=5)
                )
                st.plotly_chart(fig_lc_pie, use_container_width=True)

            with col_bar:
                st.markdown("#### Excerpts per stage (ordered by lifecycle)")
                present = [s for s in LIFECYCLE_ORDER if s in lc_counts["Stage"].values]
                lc_ordered = lc_counts.set_index("Stage").reindex(present).reset_index()
                lc_ordered = lc_ordered.dropna()
                fig_lc_bar = px.bar(
                    lc_ordered, x="Stage", y="Count",
                    color="Stage",
                    template="plotly_dark",
                    color_discrete_sequence=LIFECYCLE_COLORS,
                )
                fig_lc_bar.update_layout(
                    showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                    height=320,
                )
                st.plotly_chart(fig_lc_bar, use_container_width=True)

            # ---- Per-doc lifecycle heatmap ---------------------------------
            st.divider()
            st.markdown("#### Lifecycle stage focus per policy")

            doc_lc = (
                df_lc.groupby(["Document", "Lifecycle Stage"])
                .size()
                .reset_index(name="Count")
            )
            # Pivot to matrix
            lc_pivot = doc_lc.pivot_table(
                index="Document", columns="Lifecycle Stage",
                values="Count", fill_value=0
            ).reindex(columns=LIFECYCLE_ORDER, fill_value=0)

            short_idx = [_short_name(d, 35) for d in lc_pivot.index]

            fig_lc_heat = go.Figure(data=go.Heatmap(
                z=lc_pivot.values,
                x=list(lc_pivot.columns),
                y=short_idx,
                colorscale="Viridis",
                colorbar=dict(title="Excerpts"),
                hovertemplate=(
                    "<b>%{y}</b><br>Stage: %{x}<br>Excerpts: %{z}<extra></extra>"
                ),
            ))
            fig_lc_heat.update_layout(
                template="plotly_dark",
                height=max(400, len(docs) * 28),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig_lc_heat, use_container_width=True)

            # ---- Stage × Theme heatmap ------------------------------------
            st.divider()
            st.markdown("#### Which themes appear in which lifecycle stages?")

            theme_lc = (
                df_lc.groupby(["Theme", "Lifecycle Stage"])
                .size()
                .reset_index(name="Count")
            )
            theme_lc_pivot = theme_lc.pivot_table(
                index="Theme", columns="Lifecycle Stage",
                values="Count", fill_value=0
            ).reindex(columns=LIFECYCLE_ORDER, fill_value=0)

            fig_tl = go.Figure(data=go.Heatmap(
                z=theme_lc_pivot.values,
                x=list(theme_lc_pivot.columns),
                y=list(theme_lc_pivot.index),
                colorscale="Plasma",
                colorbar=dict(title="Excerpts"),
            ))
            fig_tl.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig_tl, use_container_width=True)


# ---------------------------------------------------------------------------
# Standalone page execution
# ---------------------------------------------------------------------------

if __name__ == "__main__" or True:
    from core.page_utils import get_results_for_page, no_results_message
    from core.taxonomy_loader import load_taxonomy
    _results = get_results_for_page()
    if not _results:
        no_results_message(page_key="gap_analysis")
    else:
        _taxonomy = load_taxonomy()
        render_gap_analysis(_results, _taxonomy)
