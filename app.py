"""
app.py  —  EO Policy Skills Dashboard  v1.4
--------------------------------------------
Main Streamlit entry point.

Input modes:
  1. Folder path (primary)   — reads PDFs directly from a server directory.
                               Recommended for large corpora (50–200 PDFs).
                               On Railway: mount a volume at /data, copy PDFs
                               to /data/pdfs, set PDF_FOLDER=/data/pdfs.
  2. File upload (secondary) — browser upload for small batches (<20 files).

Run locally:
    streamlit run app.py

Environment variables (all optional):
    OPENAI_API_KEY   — enables OpenAI text-embedding-3-small
    PDF_FOLDER       — pre-fills the folder path input (e.g. /data/pdfs)
    CACHE_DIR        — override cache directory (default: /data or .cache)
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
from core.cache_manager import (
    discover_pdfs,
    cache_stats,
    clear_disk_cache,
    clear_session_cache,
    save_output,
    list_outputs,
    save_results,
    OUTPUTS_DIR,
)
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
from pages.gap_analysis import render_gap_analysis

# ---- logging ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy loader — must be at module level for @st.cache_data to work
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_taxonomy_cached(yaml_bytes: Optional[bytes]) -> Dict[str, Any]:
    """Load taxonomy from YAML bytes, or fall back to built-in file."""
    if yaml_bytes:
        import yaml as _yaml  # type: ignore
        return _yaml.safe_load(yaml_bytes)
    return load_taxonomy()


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
# Session state
# ===========================================================================

def _result_key(r: SearchResult) -> str:
    return f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"


def init_session_state():
    defaults = {
        "results": [],
        "docs": [],
        "taxonomy": None,
        "human_labels": {},
        "analysis_done": False,
        "corpus_filenames": [],   # names of last-processed corpus
        "corpus_source": "",      # "folder" | "upload"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ---------------------------------------------------------------------------
# Auto-restore results from disk on fresh session (e.g. after reconnect)
# This means you never lose results just because the browser disconnected.
# ---------------------------------------------------------------------------
if not st.session_state.get("results") and not st.session_state.get("_app_restored"):
    from core.cache_manager import load_results as _load_results
    from core.page_utils import _dict_to_result
    _payload = _load_results()
    if _payload and _payload.get("results"):
        try:
            st.session_state["results"] = [_dict_to_result(r) for r in _payload["results"]]
            st.session_state["corpus_filenames"] = _payload.get("corpus_filenames", [])
            st.session_state["analysis_done"] = True
        except Exception:
            pass  # silently ignore corrupt cache; user can re-run
    st.session_state["_app_restored"] = True


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.title("🛰️ EO Policy Dashboard")
    st.caption("Detect space industry & EO downstream skills evidence in policy documents")
    st.divider()

    # ---- 1. Document input -------------------------------------------------
    st.subheader("1. Load Policy Documents")

    _default_folder = os.environ.get("PDF_FOLDER", "")

    input_mode = st.radio(
        "Input method",
        ["📁 Folder path", "⬆️ Upload files"],
        help=(
            "**Folder path** reads PDFs directly from disk — no browser upload, "
            "no timeout, recommended for 20+ files.\n\n"
            "**Upload files** works for small batches (<20 PDFs)."
        ),
    )

    uploaded_files = []
    folder_path_input = ""
    folder_pdfs: List[Path] = []
    corpus_warning = ""

    # ---- Folder path mode --------------------------------------------------
    if input_mode == "📁 Folder path":
        folder_path_input = st.text_input(
            "PDF folder path",
            value=_default_folder,
            placeholder="/data/pdfs",
            help=(
                "Absolute path to a directory of PDFs on the server.\n"
                "On Railway: mount a volume at `/data`, copy PDFs to `/data/pdfs`, "
                "then enter `/data/pdfs` here (or set `PDF_FOLDER=/data/pdfs` env var)."
            ),
        )

        recursive = st.checkbox(
            "Include subfolders",
            value=False,
            help="If checked, scans all subdirectories recursively.",
        )

        if folder_path_input:
            _fpath = Path(folder_path_input)
            if not _fpath.exists():
                st.error(f"❌ Folder not found: `{folder_path_input}`")
                corpus_warning = "not_found"
            elif not _fpath.is_dir():
                st.error(f"❌ Not a directory: `{folder_path_input}`")
                corpus_warning = "not_dir"
            else:
                try:
                    folder_pdfs = discover_pdfs(_fpath, recursive=recursive)
                    if not folder_pdfs:
                        st.warning("⚠️ No PDF files found in that folder.")
                        corpus_warning = "empty"
                    else:
                        st.success(f"✅ {len(folder_pdfs)} PDF(s) ready")
                        if len(folder_pdfs) > 100:
                            st.info(
                                f"ℹ️ Large corpus ({len(folder_pdfs)} files). "
                                "Processing will take several minutes. "
                                "Already-processed files will use the embedding cache."
                            )
                        with st.expander(f"Show file list ({len(folder_pdfs)} PDFs)"):
                            for p in folder_pdfs:
                                st.caption(f"• {p.name}")
                except PermissionError:
                    st.error(f"❌ Permission denied reading: `{folder_path_input}`")
                    corpus_warning = "permission"

    # ---- Upload mode -------------------------------------------------------
    else:
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Best for small batches. Use folder path for 20+ files.",
        )
        if uploaded_files:
            st.caption(f"📄 {len(uploaded_files)} file(s) selected")
            if len(uploaded_files) > 30:
                st.warning(
                    "⚠️ Large upload selected. For reliability with >30 files, "
                    "use the Folder path mode instead."
                )

    # ---- 2. Taxonomy -------------------------------------------------------
    st.divider()
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

    # ---- 3. Thresholds -----------------------------------------------------
    st.subheader("3. Thresholds")
    valid_threshold = st.slider(
        "Valid evidence threshold", 0.20, 0.90, 0.50, 0.05,
        help="Final score above this → VALID EVIDENCE",
    )
    weak_threshold = st.slider(
        "Weak evidence threshold", 0.10, 0.80, 0.35, 0.05,
        help="Final score above this → WEAK EVIDENCE",
    )

    # ---- 4. OpenAI ---------------------------------------------------------
    st.subheader("4. Optional: OpenAI Embeddings")
    openai_key_input = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        help=(
            "Enables OpenAI text-embedding-3-small for semantic search. "
            "Leave blank for 100% offline SBERT mode. "
            "Can also be set via OPENAI_API_KEY environment variable."
        ),
    )
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input
    elif "OPENAI_API_KEY" not in os.environ:
        os.environ.pop("OPENAI_API_KEY", None)

    st.divider()

    # ---- Run button --------------------------------------------------------
    _has_input = bool(folder_pdfs) or bool(uploaded_files)
    run_analysis = st.button(
        "🔍 Run Analysis",
        type="primary",
        disabled=not _has_input,
        width="stretch",
    )
    if not _has_input and not corpus_warning:
        st.caption("Select a folder or upload files to enable.")

    # ---- Reset button ------------------------------------------------------
    if st.button("🔄 Clear & Reset", width="stretch"):
        clear_session_cache()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.divider()

    # ---- Search mode badge -------------------------------------------------
    st.markdown("**Search mode**")

    taxonomy_bytes = custom_taxonomy_file.read() if custom_taxonomy_file else None
    taxonomy = _load_taxonomy_cached(taxonomy_bytes)

    _scfg = taxonomy.get("search", {})
    _use_sbert = _scfg.get("use_sbert", True)
    _has_key = bool(os.environ.get("OPENAI_API_KEY") or "")
    _use_ce = _scfg.get("use_cross_encoder", False)

    if not _use_sbert and _has_key:
        _mode_label = "🟢 OpenAI-only"
        _mode_help = "Semantic search via OpenAI text-embedding-3-small. (~$0.01–0.05 / 30-PDF run)"
    elif _use_sbert and _use_ce and _has_key:
        _mode_label = "🔵 Hybrid (SBERT + CrossEncoder + OpenAI)"
        _mode_help = "Full pipeline — best quality, needs >2 GB RAM."
    elif _use_sbert and _has_key:
        _mode_label = "🔵 SBERT + OpenAI re-ranking"
        _mode_help = "Local SBERT encoding with OpenAI re-ranking."
    elif _use_sbert:
        _mode_label = "🟡 SBERT-only (offline)"
        _mode_help = "Local semantic search — no API key required."
    else:
        _mode_label = "🔴 Keyword-only"
        _mode_help = "use_sbert=false and no OPENAI_API_KEY — keyword matching only."

    st.info(f"{_mode_label}  \n{_mode_help}")

    # ---- Embedding cache stats ---------------------------------------------
    st.divider()
    st.markdown("**Embedding cache**")
    try:
        _cs = cache_stats()
        _persist_icon = "🟢" if _cs["is_persistent"] else "🟡"
        _persist_label = "Persistent (Railway volume)" if _cs["is_persistent"] else "Ephemeral (local .cache)"
        st.caption(
            f"{_persist_icon} {_persist_label}  \n"
            f"`{_cs['base_dir']}`  \n"
            f"{_cs['unique_docs']} doc(s) · {_cs['cached_files']} embedding file(s) · "
            f"{_cs['total_size_mb']} MB"
        )
        if _cs["cached_files"] > 0:
            if st.button("🗑️ Clear embedding cache", width="stretch"):
                n = clear_disk_cache()
                st.success(f"Cleared {n} cached embedding file(s).")
                st.rerun()
    except Exception as _e:
        st.caption(f"Cache info unavailable: {_e}")


# Apply threshold overrides
taxonomy.setdefault("thresholds", {})
taxonomy["thresholds"]["valid_match"] = valid_threshold
taxonomy["thresholds"]["weak_match"] = weak_threshold
st.session_state["taxonomy"] = taxonomy


# ===========================================================================
# Run the pipeline
# ===========================================================================

if run_analysis:
    # Build list of (filename, path_or_bytes) from whichever input mode is active.
    # For folder mode: we pass paths directly — pipeline reads bytes on demand.
    # For upload mode: read bytes here (small batch, acceptable RAM use).
    file_pairs: List[tuple] = []

    if folder_pdfs:
        # Read each PDF from disk — one at a time inside the pipeline loop.
        # We read bytes here but the pipeline immediately discards them per-doc.
        file_pairs = [(p.name, p.read_bytes()) for p in folder_pdfs]
        st.session_state["corpus_source"] = "folder"
    elif uploaded_files:
        file_pairs = [(f.name, f.read()) for f in uploaded_files]
        st.session_state["corpus_source"] = "upload"

    if file_pairs:
        n = len(file_pairs)
        progress_bar = st.progress(0, text="Starting…")
        status_placeholder = st.empty()
        _processed_count = [0]

        def update_status(msg: str):
            status_placeholder.info(f"⏳ {msg}")
            # Estimate progress from the message format "[i/n]"
            try:
                part = msg.split("[")[1].split("]")[0]
                i, total = part.split("/")
                progress_bar.progress(int(i) / int(total), text=msg)
                _processed_count[0] = int(i)
            except Exception:
                pass

        try:
            results, docs = process_documents(
                uploaded_files=file_pairs,
                taxonomy=taxonomy,
                status_callback=update_status,
            )

            for r in results:
                key = _result_key(r)
                r.human_label = st.session_state["human_labels"].get(key, "")

            st.session_state["results"] = results
            st.session_state["docs"] = docs
            st.session_state["analysis_done"] = True
            st.session_state["corpus_filenames"] = [f for f, _ in file_pairs]

            progress_bar.progress(1.0, text="Complete")
            status_placeholder.empty()

            # Persist results to disk so sidebar pages can load them
            save_results(results, docs, st.session_state["corpus_filenames"])

            # Auto-save outputs to persistent directory
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
            try:
                save_output(f"eo_evidence_{ts}.csv", to_csv_bytes(results))
                save_output(f"eo_evidence_{ts}.xlsx", to_excel_bytes(results))
                logger.info("Auto-saved CSV and Excel to %s", OUTPUTS_DIR)
            except Exception as _save_exc:
                logger.warning("Auto-save failed: %s", _save_exc)

            st.success(
                f"✅ Analysis complete — **{len(results)}** evidence excerpts "
                f"across **{len(docs)}** document(s).  \n"
                f"Outputs auto-saved to `{OUTPUTS_DIR}`."
            )

        except MemoryError:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(
                "❌ Out of memory processing the corpus.  \n"
                "Try reducing the number of PDFs, or switch to OpenAI-only mode "
                "(set `use_sbert: false` in taxonomy.yaml) which uses ~5× less RAM."
            )
            logger.exception("MemoryError in pipeline")

        except Exception as exc:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"❌ Pipeline error: {exc}")
            logger.exception("Pipeline failed")


# ===========================================================================
# Main content
# ===========================================================================

results: List[SearchResult] = st.session_state.get("results", [])
docs = st.session_state.get("docs", [])

if not st.session_state["analysis_done"]:
    # ---- Welcome screen ----------------------------------------------------
    st.title("🛰️ EO Policy Skills Dashboard")
    st.markdown(
        """
        **Detect evidence of space industry and Earth Observation downstream skills
        needs in policy documents.**

        ### How to use

        **Option A — Folder path (recommended for large corpora)**
        1. Place your PDF files in a folder accessible to the server (e.g. `/data/pdfs` on Railway)
        2. Enter the folder path in the sidebar
        3. Click **Run Analysis**

        **Option B — File upload (small batches)**
        1. Select "Upload files" in the sidebar
        2. Upload up to ~20 PDFs
        3. Click **Run Analysis**

        ### What this dashboard detects
        | Theme | Examples |
        |---|---|
        | EO Downstream Skills | earth observation, satellite imagery, Copernicus, remote sensing |
        | Space Industry Skills | space economy, aerospace, satellite operations |
        | Geospatial & GIS | GIS, spatial analysis, geoinformatics |
        | Digital Skills for Space | AI/ML for EO, cloud computing, Python |
        | Skills Gaps & Workforce | upskilling, reskilling, capacity building |
        | Policy Support for Downstream | smart specialisation, S3, downstream applications |

        > **Security:** All processing runs on the server. PDF contents never leave your deployment.
        > The OpenAI API key field is optional — leave blank for 100% offline analysis.
        """
    )

    with st.expander("Preview active taxonomy"):
        for theme, kws in taxonomy_to_display(taxonomy).items():
            st.markdown(f"**{theme}**")
            st.caption(", ".join(kws[:8]) + ("…" if len(kws) > 8 else ""))

    # Show any previously auto-saved outputs
    _saved = list_outputs()
    if _saved:
        with st.expander(f"📁 Previously saved outputs ({len(_saved)} files)"):
            for out_path in _saved[:20]:
                col_a, col_b = st.columns([3, 1])
                col_a.caption(f"`{out_path.name}`")
                col_b.download_button(
                    "⬇️",
                    data=out_path.read_bytes(),
                    file_name=out_path.name,
                    key=f"dl_{out_path.name}",
                )

else:
    # ---- Summary metrics ---------------------------------------------------
    df_all = results_to_dataframe(results)
    corpus_src = st.session_state.get("corpus_source", "")
    corpus_names = st.session_state.get("corpus_filenames", [])

    if corpus_names:
        _src_icon = "📁" if corpus_src == "folder" else "⬆️"
        st.caption(
            f"{_src_icon} Corpus: {len(corpus_names)} document(s) — "
            + ", ".join(corpus_names[:5])
            + (f" … +{len(corpus_names)-5} more" if len(corpus_names) > 5 else "")
        )

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

    tab_results, tab_charts, tab_intel, tab_validate, tab_export = st.tabs(
        ["📋 Results", "📊 Charts", "🔬 Policy Intelligence", "🏷️ Human Validation", "📤 Export"]
    )

    with tab_results:
        render_results_table(results, taxonomy)

    with tab_charts:
        render_charts(results, taxonomy)

    with tab_intel:
        render_gap_analysis(results, taxonomy)

    with tab_validate:
        render_human_validation(results)

    with tab_export:
        st.subheader("Export Results")
        st.caption(
            "All exports include your current human validation labels. "
            "Re-run the analysis after changing thresholds to update scores."
        )
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

        st.markdown("#### 📄 Evidence Reports")
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
        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                label="⬇️ CSV (Power BI)",
                data=to_csv_bytes(results),
                file_name=f"eo_policy_evidence_{ts}.csv",
                mime="text/csv",
                width="stretch",
            )
            try:
                _excel_data = to_excel_bytes(results)
                st.download_button(
                    label="⬇️ Excel Data Workbook",
                    data=_excel_data,
                    file_name=f"eo_policy_data_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
            except Exception as _exc:
                st.error(f"Excel export error: {_exc}")
                logger.exception("to_excel_bytes failed")

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
        st.markdown(f"#### 📁 Server outputs  \n`{OUTPUTS_DIR}`")
        _saved = list_outputs()
        if _saved:
            st.caption(f"{len(_saved)} file(s) auto-saved to the server outputs directory.")
            for out_path in _saved[:20]:
                col_x, col_y = st.columns([3, 1])
                col_x.caption(f"`{out_path.name}`")
                col_y.download_button(
                    "⬇️",
                    data=out_path.read_bytes(),
                    file_name=out_path.name,
                    key=f"dl_out_{out_path.name}",
                )
        else:
            st.caption("No auto-saved outputs yet.")

        with st.expander("Preview D3 JSON (first 50 lines)"):
            j = to_d3_json(results)
            st.code("\n".join(j.splitlines()[:50]), language="json")
