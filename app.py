"""
app.py
------
Main entry point for the EO Policy Skills Dashboard.

Run locally:
    streamlit run app.py

The app is split into logical sections rendered on a single page:
  1. Sidebar  : upload PDFs, configure thresholds, choose taxonomy
  2. Main     : tabbed view — Results | Charts | Human Validation | Export
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ---- project imports -------------------------------------------------------
from core.taxonomy_loader import load_taxonomy, taxonomy_to_display
from core.pipeline import process_documents
from core.search_engine import SearchResult
from core.exporters import (
    to_csv_bytes,
    to_excel_bytes,
    to_excel_report_bytes,
    to_pdf_report_bytes,
    to_d3_json,
    to_markdown_report,
    results_to_dataframe,
)
from pages.charts import render_charts
from pages.results_table import render_results_table
from pages.human_validation import render_human_validation

# ---- logging ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- page config -----------------------------------------------------------
st.set_page_config(
    page_title="EO Policy Skills Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- custom CSS ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2a3a;
        color: #e8f0f7;
        border-radius: 6px 6px 0 0;
        padding: 0.4rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0a7ea4;
        color: white;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .excerpt-card {
        background: #0e1117;
        border-left: 4px solid #0a7ea4;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Session state initialisation
# ===========================================================================

def _result_key(r: SearchResult) -> str:
    return f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"


def init_session_state():
    defaults = {
        "results": [],          # List[SearchResult]
        "docs": [],             # List[DocumentContent]
        "taxonomy": None,
        "human_labels": {},     # {excerpt_hash: label_string}
        "analysis_done": False,
        "uploaded_file_names": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.title("🛰️ EO Policy Dashboard")
    st.caption("Detect space industry & EO downstream skills evidence in policy documents")
    st.divider()

    # -- PDF uploader --------------------------------------------------------
    st.subheader("1. Upload Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload the policy PDFs you want to analyse. Files are processed locally.",
    )

    # -- Taxonomy selector ---------------------------------------------------
    st.subheader("2. Taxonomy")
    taxonomy_source = st.radio(
        "Taxonomy source",
        ["Built-in (default)", "Upload custom YAML"],
        horizontal=True,
    )
    custom_taxonomy_file = None
    if taxonomy_source == "Upload custom YAML":
        custom_taxonomy_file = st.file_uploader(
            "Upload taxonomy YAML",
            type=["yaml", "yml"],
            help="Must follow the same schema as config/taxonomy.yaml",
        )

    # -- Threshold overrides -------------------------------------------------
    st.subheader("3. Thresholds")
    valid_threshold = st.slider(
        "Valid evidence threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Final score above this → VALID EVIDENCE",
    )
    weak_threshold = st.slider(
        "Weak evidence threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Final score above this → WEAK EVIDENCE",
    )

    # -- OpenAI optional -----------------------------------------------------
    st.subheader("4. Optional: OpenAI Re-ranking")
    openai_key_input = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        help="If provided, top results are re-ranked with text-embedding-3-small. "
             "Leave blank to use SBERT only (free, offline).",
    )
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input
    elif "OPENAI_API_KEY" not in os.environ:
        os.environ.pop("OPENAI_API_KEY", None)

    st.divider()

    # -- Run button ----------------------------------------------------------
    run_analysis = st.button(
        "🔍 Run Analysis",
        type="primary",
        disabled=not uploaded_files,
        width="stretch",
    )

    if not uploaded_files:
        st.info("Upload at least one PDF to begin.")

    # -- Reset button --------------------------------------------------------
    if st.button("🔄 Clear & Reset", width="stretch"):
        from core.cache_manager import clear_session_cache
        clear_session_cache()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ===========================================================================
# Load taxonomy
# ===========================================================================

@st.cache_data(show_spinner=False)
def _load_taxonomy_cached(yaml_bytes: Optional[bytes]) -> Dict[str, Any]:
    if yaml_bytes:
        import tempfile, yaml as _yaml
        data = _yaml.safe_load(yaml_bytes)
        return data
    return load_taxonomy()


taxonomy_bytes = custom_taxonomy_file.read() if custom_taxonomy_file else None
taxonomy = _load_taxonomy_cached(taxonomy_bytes)

# ── Search mode badge (shown in sidebar) ─────────────────────────────────────
_scfg = taxonomy.get("search", {})
_use_sbert = _scfg.get("use_sbert", True)
_use_oa_cfg = _scfg.get("use_openai", False)
_has_key = bool(os.environ.get("OPENAI_API_KEY") or "")
_use_ce = _scfg.get("use_cross_encoder", False)

if not _use_sbert and _has_key:
    _mode_label = "🟢 OpenAI-only"
    _mode_help = "Semantic search via OpenAI text-embedding-3-small. No local models loaded. (~$0.01–0.05 per 30-PDF run)"
elif _use_sbert and _use_ce and _has_key:
    _mode_label = "🔵 Hybrid (SBERT + CrossEncoder + OpenAI)"
    _mode_help = "Full pipeline — best quality, needs >2GB RAM."
elif _use_sbert and _has_key:
    _mode_label = "🔵 SBERT + OpenAI re-ranking"
    _mode_help = "Local SBERT encoding with OpenAI re-ranking."
elif _use_sbert:
    _mode_label = "🟡 SBERT-only (offline)"
    _mode_help = "Local semantic search — no API key required."
else:
    _mode_label = "🔴 Keyword-only"
    _mode_help = "use_sbert=false but OPENAI_API_KEY not set — only keyword matching active."

with st.sidebar:
    st.divider()
    st.markdown(f"**Search mode**")
    st.info(f"{_mode_label}  \n{_mode_help}")

    # -- Embedding cache stats -----------------------------------------------
    st.divider()
    st.markdown("**Embedding cache**")
    try:
        from core.cache_manager import cache_stats, clear_disk_cache, CACHE_DIR
        _cs = cache_stats()
        _persist_label = "🟢 Persistent (Railway volume)" if _cs["is_persistent"] else "🟡 Ephemeral (local .cache)"
        st.caption(
            f"{_persist_label}  \n"
            f"`{_cs['cache_dir']}`  \n"
            f"{_cs['unique_docs']} doc(s) cached · {_cs['cached_files']} file(s) · {_cs['total_size_mb']} MB"
        )
        if _cs["cached_files"] > 0:
            if st.button("🗑️ Clear disk cache", width="stretch"):
                n = clear_disk_cache()
                st.success(f"Cleared {n} cached embedding file(s).")
                st.rerun()
    except Exception as _e:
        st.caption(f"Cache info unavailable: {_e}")

# Override thresholds from sidebar sliders
taxonomy.setdefault("thresholds", {})
taxonomy["thresholds"]["valid_match"] = valid_threshold
taxonomy["thresholds"]["weak_match"] = weak_threshold

st.session_state["taxonomy"] = taxonomy


# ===========================================================================
# Run the pipeline when the user clicks Run Analysis
# ===========================================================================

if run_analysis and uploaded_files:
    # Read each file ONCE — UploadedFile buffers are exhausted after .read()
    file_pairs = [(f.name, f.read()) for f in uploaded_files]
    st.session_state["uploaded_file_names"] = [name for name, _ in file_pairs]

    progress_placeholder = st.empty()

    def update_status(msg: str):
        progress_placeholder.info(f"⏳ {msg}")

    with st.spinner("Running analysis pipeline…"):
        try:
            results, docs = process_documents(
                uploaded_files=file_pairs,
                taxonomy=taxonomy,
                status_callback=update_status,
            )
            # Apply human labels carried over from previous run
            for r in results:
                key = _result_key(r)
                r.human_label = st.session_state["human_labels"].get(key, "")

            st.session_state["results"] = results
            st.session_state["docs"] = docs
            st.session_state["analysis_done"] = True
            progress_placeholder.empty()
            st.success(
                f"✅ Analysis complete — {len(results)} evidence excerpts found "
                f"across {len(docs)} document(s)."
            )
        except Exception as exc:
            progress_placeholder.empty()
            st.error(f"Pipeline error: {exc}")
            logger.exception("Pipeline failed")


# ===========================================================================
# Main content area
# ===========================================================================

results: List[SearchResult] = st.session_state.get("results", [])
docs = st.session_state.get("docs", [])

if not st.session_state["analysis_done"]:
    # Welcome screen
    st.title("🛰️ EO Policy Skills Dashboard")
    st.markdown(
        """
        **Detect evidence of space industry and Earth Observation downstream skills needs
        in policy documents.**

        ### How to use
        1. Upload one or more PDF policy documents using the sidebar
        2. Choose or upload a skills taxonomy (default covers EO, Copernicus, GIS, digital skills)
        3. Adjust confidence thresholds if needed
        4. Click **Run Analysis**

        ### What this dashboard detects
        | Theme | Examples |
        |---|---|
        | EO Downstream Skills | earth observation, satellite imagery, Copernicus, remote sensing |
        | Space Industry Skills | space economy, aerospace, satellite operations |
        | Geospatial & GIS | GIS, spatial analysis, geoinformatics |
        | Digital Skills for Space | AI/ML for EO, cloud computing, Python |
        | Skills Gaps & Workforce | upskilling, reskilling, capacity building |
        | Policy Support for Downstream | smart specialisation, S3, downstream applications |

        > **Security note:** All processing runs locally. Your PDF contents never leave your machine.
        > The OpenAI API key field is optional — leave blank for 100% offline analysis.
        """
    )

    # Show taxonomy preview
    with st.expander("Preview active taxonomy"):
        for theme, kws in taxonomy_to_display(taxonomy).items():
            st.markdown(f"**{theme}**")
            st.caption(", ".join(kws[:8]) + ("…" if len(kws) > 8 else ""))

else:
    # ---- Summary metrics ---------------------------------------------------
    df_all = results_to_dataframe(results)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Documents", len(docs))
    col2.metric("Total Excerpts", len(results))
    col3.metric(
        "Valid Evidence",
        len(df_all[df_all["Validation Category"] == "VALID EVIDENCE"]),
    )
    col4.metric(
        "Weak Evidence",
        len(df_all[df_all["Validation Category"] == "WEAK EVIDENCE"]),
    )
    col5.metric(
        "Themes Detected",
        df_all["Theme"].nunique() if not df_all.empty else 0,
    )

    st.divider()

    # ---- Tabs --------------------------------------------------------------
    tab_results, tab_charts, tab_validate, tab_export = st.tabs(
        ["📋 Results", "📊 Charts", "🏷️ Human Validation", "📤 Export"]
    )

    with tab_results:
        render_results_table(results, taxonomy)

    with tab_charts:
        render_charts(results, taxonomy)

    with tab_validate:
        render_human_validation(results)

    with tab_export:
        st.subheader("Export Results")
        st.caption(
            "All exports include your current human validation labels. "
            "Re-run the analysis after changing thresholds to update scores."
        )
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')

        st.markdown("#### 📄 Evidence Reports")
        st.caption("Formatted reports grouping all valid and weak evidence by document and theme.")
        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            try:
                pdf_bytes = to_pdf_report_bytes(results)
                st.download_button(
                    label="⬇️ Evidence Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"eo_evidence_report_{ts}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            except Exception as e:
                st.warning(f"PDF export unavailable: {e}")

        with rep_col2:
            try:
                excel_report_bytes = to_excel_report_bytes(results)
                st.download_button(
                    label="⬇️ Evidence Report (Excel)",
                    data=excel_report_bytes,
                    file_name=f"eo_evidence_report_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
            except Exception as e:
                st.warning(f"Excel report export error: {e}")

        st.divider()
        st.markdown("#### 📊 Data Exports")
        st.caption("Raw data exports for Power BI, further analysis, or D3 visualisation.")
        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                label="⬇️ CSV (Power BI)",
                data=to_csv_bytes(results),
                file_name=f"eo_policy_evidence_{ts}.csv",
                mime="text/csv",
                width="stretch",
            )
            st.download_button(
                label="⬇️ Excel Data Workbook",
                data=to_excel_bytes(results),
                file_name=f"eo_policy_data_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

        with col_b:
            st.download_button(
                label="⬇️ D3 / Network JSON",
                data=to_d3_json(results),
                file_name=f"eo_policy_d3_{ts}.json",
                mime="application/json",
                width="stretch",
            )
            report_md = to_markdown_report(results)
            st.download_button(
                label="⬇️ Evidence Report (Markdown)",
                data=report_md.encode("utf-8"),
                file_name=f"eo_policy_report_{ts}.md",
                mime="text/markdown",
                width="stretch",
            )

        st.divider()
        with st.expander("Preview D3 JSON (first 50 lines)"):
            j = to_d3_json(results)
            st.code("\n".join(j.splitlines()[:50]), language="json")
