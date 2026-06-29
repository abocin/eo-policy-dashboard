"""
pages/results_table.py
----------------------
Filtered evidence table with:
  - Interactive st.dataframe (primary, handles thousands of rows)
  - Paginated card view — ALL results, 25 per page
  - Page navigation controls
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.search_engine import SearchResult
from core.exporters import results_to_dataframe

VALIDATION_COLORS = {
    "VALID EVIDENCE":  "#00b894",
    "WEAK EVIDENCE":   "#fdcb6e",
    "NOT RELEVANT":    "#d63031",
    "NEEDS REVIEW":    "#74b9ff",
    "UNSCORED":        "#636e72",
}

CARDS_PER_PAGE = 25


def render_results_table(results: List[SearchResult], taxonomy: Dict[str, Any]):
    st.subheader("Evidence Excerpts")

    if not results:
        st.info("No results yet. Upload PDFs and click Run Analysis.")
        return

    try:
        df = results_to_dataframe(results)
    except Exception as e:
        st.error(f"Could not build results table: {e}")
        return

    # ---- Check whether EO relevance scores are available -------------------
    has_eo_scores = "EO Relevance Score" in df.columns and df["EO Relevance Score"].max() > 0.0

    # ---- Filters -----------------------------------------------------------
    with st.expander("🔎 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        docs_available = sorted(df["Document"].unique().tolist())
        selected_docs = col1.multiselect("Document(s)", docs_available,
                                         default=docs_available)

        themes_available = sorted(df["Theme"].unique().tolist())
        selected_themes = col2.multiselect("Theme(s)", themes_available,
                                           default=themes_available)

        cats_available = sorted(df["Validation Category"].unique().tolist())
        default_cats = [c for c in cats_available
                        if c not in ("NOT RELEVANT", "UNSCORED")]
        selected_cats = col3.multiselect("Validation category", cats_available,
                                         default=default_cats)

        min_score = col4.slider("Minimum final score", 0.0, 1.0, 0.0, 0.01)
        keyword_filter = col5.text_input("Search in excerpts",
                                         placeholder="Type to filter…")

        # ---- EO Capacity-Building Relevance Filter (Stage 2) ---------------
        st.divider()
        eo_col1, eo_col2 = st.columns([2, 3])
        with eo_col1:
            eo_filter_on = st.toggle(
                "🛰️ EO Capacity Filter",
                value=False,
                disabled=not has_eo_scores,
                help=(
                    "Only show excerpts that score above the threshold on "
                    "EO capacity-building relevance (Stage-2 semantic re-rank). "
                    "Requires OPENAI_API_KEY to be set."
                    if has_eo_scores
                    else "Run analysis with OPENAI_API_KEY set to enable this filter."
                ),
            )
        with eo_col2:
            eo_min_score = 0.0
            if eo_filter_on and has_eo_scores:
                eo_min_score = st.slider(
                    "Min EO relevance score",
                    min_value=0.0, max_value=1.0,
                    value=0.35, step=0.01,
                    help=(
                        "Excerpts with EO relevance score below this threshold are hidden. "
                        "Recommended starting point: 0.35. Raise to 0.50+ for stricter filtering."
                    ),
                )

        if has_eo_scores:
            if eo_filter_on:
                above = (df["EO Relevance Score"] >= eo_min_score).sum()
                st.caption(
                    f"🛰️ EO Capacity Filter active — {above:,} of {len(df):,} excerpts "
                    f"pass the ≥ {eo_min_score:.2f} threshold."
                )
            else:
                avg_eo = df["EO Relevance Score"].mean()
                st.caption(
                    f"🛰️ EO relevance scores available (avg: {avg_eo:.3f}). "
                    "Enable the toggle above to filter by EO capacity relevance."
                )
        else:
            st.caption(
                "🛰️ EO Capacity Filter unavailable — "
                "set OPENAI_API_KEY and re-run analysis to activate."
            )

    # ---- Apply filters -----------------------------------------------------
    mask = (
        df["Document"].isin(selected_docs)
        & df["Theme"].isin(selected_themes)
        & df["Validation Category"].isin(selected_cats)
        & (df["Final Score"] >= min_score)
    )
    if keyword_filter.strip():
        mask &= df["Excerpt"].str.contains(keyword_filter.strip(),
                                           case=False, na=False)
    if eo_filter_on and has_eo_scores:
        mask &= df["EO Relevance Score"] >= eo_min_score

    filtered = df[mask].sort_values("Final Score", ascending=False).reset_index(drop=True)
    total = len(filtered)

    st.caption(f"Showing **{total:,}** of {len(df):,} excerpts")

    # ---- Primary view: interactive dataframe -------------------------------
    display_cols = ["Document", "Page", "Theme", "Final Score",
                    "Validation Category", "Keyword Hit", "Excerpt"]
    col_config = {
        "Final Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=1, format="%.3f"
        ),
        "Excerpt": st.column_config.TextColumn("Excerpt", width="large"),
        "Keyword Hit": st.column_config.CheckboxColumn("KW"),
    }

    # Show EO relevance score column when scores are available
    if has_eo_scores and "EO Relevance Score" in filtered.columns:
        display_cols = ["Document", "Page", "Theme", "Final Score",
                        "EO Relevance Score", "Validation Category",
                        "Keyword Hit", "Excerpt"]
        col_config["EO Relevance Score"] = st.column_config.ProgressColumn(
            "🛰️ EO Relevance",
            min_value=0, max_value=1, format="%.3f",
            help="Stage-2 EO capacity-building relevance score (0=not relevant, 1=highly relevant)",
        )

    st.dataframe(
        filtered[display_cols],
        width="stretch",
        height=480,
        column_config=col_config,
    )

    # ---- Download buttons (table section) ---------------------------------
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label=f"⬇️ Download filtered results ({total:,} rows)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="eo_results_filtered.csv",
            mime="text/csv",
            key="download_filtered_table",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label=f"⬇️ Download ALL results ({len(df):,} rows)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="eo_results_all.csv",
            mime="text/csv",
            key="download_all_results",
            use_container_width=True,
        )

    st.divider()

    # ---- Card view quick-filter bar ---------------------------------------
    st.divider()
    st.subheader("🃏 Card View")

    qf_col1, qf_col2, qf_col3 = st.columns([3, 2, 2])

    with qf_col1:
        # Quick category buttons — one-click presets
        cat_options = ["All"] + sorted(
            df["Validation Category"].unique().tolist(),
            key=lambda c: (
                0 if c == "HIGHLY RELEVANT" else
                1 if c == "RELEVANT" else
                2 if c == "UNSCORED" else
                3 if c == "NOT RELEVANT" else 4
            )
        )
        quick_cat = st.radio(
            "Quick category filter",
            cat_options,
            index=0,
            horizontal=True,
            key="card_quick_cat",
            help="One-click filter applied on top of the filters above.",
        )

    with qf_col2:
        card_sort = st.selectbox(
            "Sort cards by",
            ["Final Score ↓", "EO Relevance ↓", "Document A→Z", "Page ↑"],
            key="card_sort_by",
        )

    with qf_col3:
        cards_per_page_choice = st.selectbox(
            "Cards per page",
            [10, 25, 50, 100],
            index=1,
            key="cards_per_page_sel",
        )

    # Apply quick category filter on top of main filters
    card_df = filtered.copy()
    if quick_cat != "All":
        card_df = card_df[card_df["Validation Category"] == quick_cat]

    # Apply sort
    if card_sort == "Final Score ↓":
        card_df = card_df.sort_values("Final Score", ascending=False)
    elif card_sort == "EO Relevance ↓" and has_eo_scores:
        card_df = card_df.sort_values("EO Relevance Score", ascending=False)
    elif card_sort == "Document A→Z":
        card_df = card_df.sort_values(["Document", "Page"], ascending=True)
    elif card_sort == "Page ↑":
        card_df = card_df.sort_values(["Document", "Page"], ascending=True)
    card_df = card_df.reset_index(drop=True)

    card_total = len(card_df)
    CARDS_PER_PAGE = cards_per_page_choice

    st.caption(
        f"Showing **{card_total:,}** cards"
        + (f" (category: {quick_cat})" if quick_cat != "All" else "")
        + f" · sorted by {card_sort}"
    )

    if card_total == 0:
        st.info("No results match the current filters.")
    else:
        total_pages = max(1, (card_total + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE)

        # Page selector — keep in session state so filters don't reset it
        page_key = "card_page"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        # Reset to page 1 if filters change and current page is out of range
        if st.session_state[page_key] > total_pages:
            st.session_state[page_key] = 1

        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])

        with nav_col1:
            if st.button("⏮ First", width="stretch"):
                st.session_state[page_key] = 1
        with nav_col2:
            if st.button("◀ Prev", width="stretch",
                         disabled=st.session_state[page_key] <= 1):
                st.session_state[page_key] -= 1
        with nav_col3:
            st.markdown(
                f"<div style='text-align:center; padding-top:0.4rem; color:#aaa;'>"
                f"Page <strong>{st.session_state[page_key]}</strong> of "
                f"<strong>{total_pages}</strong> "
                f"({card_total:,} total results)</div>",
                unsafe_allow_html=True,
            )
        with nav_col4:
            if st.button("Next ▶", width="stretch",
                         disabled=st.session_state[page_key] >= total_pages):
                st.session_state[page_key] += 1
        with nav_col5:
            if st.button("Last ⏭", width="stretch"):
                st.session_state[page_key] = total_pages

        # Jump to page input
        jump = st.number_input(
            "Jump to page", min_value=1, max_value=total_pages,
            value=st.session_state[page_key], step=1, key="page_jump"
        )
        if jump != st.session_state[page_key]:
            st.session_state[page_key] = int(jump)

        st.divider()

        # Render current page
        start_idx = (st.session_state[page_key] - 1) * CARDS_PER_PAGE
        end_idx = min(start_idx + CARDS_PER_PAGE, card_total)
        page_df = card_df.iloc[start_idx:end_idx]

        for _, row in page_df.iterrows():
            color = VALIDATION_COLORS.get(row["Validation Category"], "#636e72")
            kw_badge = "🔑 keyword" if row["Keyword Hit"] else "🔍 semantic"
            matched_kw = (f" · <em>{row['Matched Keyword']}</em>"
                          if row.get("Matched Keyword") else "")
            eo_score_val = row.get("EO Relevance Score", 0.0)
            eo_badge = ""
            if has_eo_scores:
                eo_color = (
                    "#00b894" if eo_score_val >= 0.50
                    else "#fdcb6e" if eo_score_val >= 0.30
                    else "#b2bec3"
                )
                eo_badge = (
                    f" &nbsp;·&nbsp; "
                    f"<span style='color:{eo_color}; font-size:0.78rem;'>"
                    f"🛰️ EO: {eo_score_val:.2f}</span>"
                )
            st.markdown(
                f"""<div style="border-left:4px solid {color}; padding:0.5rem 1rem;
                    background:#0e1117; border-radius:4px; margin-bottom:0.6rem;">
                  <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px;">
                    <span style="font-weight:600; color:{color}; font-size:0.9rem;">
                      {row['Validation Category']}
                    </span>
                    <span style="color:#888; font-size:0.82rem;">
                      {row['Document'][:50]} &nbsp;·&nbsp; p.{int(row['Page'])}
                      &nbsp;·&nbsp; {kw_badge}{matched_kw}
                      &nbsp;·&nbsp; score: <strong>{row['Final Score']:.3f}</strong>{eo_badge}
                    </span>
                  </div>
                  <div style="color:#a29bfe; font-size:0.82rem; margin-top:0.25rem;">
                    {row['Theme']}
                  </div>
                  <div style="color:#e8f0f7; margin-top:0.35rem; font-size:0.9rem;
                              line-height:1.5;">
                    {row['Excerpt']}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.caption(
            f"Showing results {start_idx + 1}–{end_idx} of {card_total:,}. "
            f"Use filters above to narrow down, or export all results."
        )

    # ---- Download filtered cards as CSV -----------------------------------
    st.divider()
    csv_bytes = card_df.to_csv(index=False).encode("utf-8")
    quick_label = f"_{quick_cat.lower().replace(' ', '_')}" if quick_cat != "All" else ""
    st.download_button(
        label=f"⬇️ Download filtered cards as CSV ({card_total:,} rows)",
        data=csv_bytes,
        file_name=f"eo_results{quick_label}.csv",
        mime="text/csv",
        key="download_filtered_cards",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Standalone page execution (when accessed directly via sidebar URL)
# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:
    from core.page_utils import get_results_for_page, no_results_message
    from core.taxonomy_loader import load_taxonomy
    _results = get_results_for_page()
    if not _results:
        no_results_message(page_key="results")
    else:
        _taxonomy = load_taxonomy()
        render_results_table(_results, _taxonomy)
