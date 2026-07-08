"""
pages/99_Admin_Upload.py
------------------------
Temporary admin page for managing PDFs on the Railway volume.
Remove from repo after use with: git rm pages/99_Admin_Upload.py

Capabilities:
  - List all PDFs currently in /data/pdfs (or PDF_FOLDER env var)
  - Delete selected files individually or in bulk
  - Upload new PDFs (chunked, 200 MB per file limit)
  - Shows file sizes and total corpus size
"""

import os
from pathlib import Path

import streamlit as st

PDF_FOLDER = Path(os.environ.get("PDF_FOLDER", "/data/pdfs"))

st.set_page_config(page_title="PDF File Manager", page_icon="🗂️", layout="wide")
st.title("🗂️ PDF File Manager")
st.caption(f"Managing files in `{PDF_FOLDER}`")

# Ensure folder exists
PDF_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load file list (excluding macOS resource fork files)
# ---------------------------------------------------------------------------
all_files = sorted(
    p for p in PDF_FOLDER.glob("*.pdf")
    if not p.name.startswith("._") and not p.name.startswith(".")
)

total_size_mb = sum(f.stat().st_size for f in all_files) / 1_048_576

st.markdown(
    f"**{len(all_files)} PDF(s)** in corpus · "
    f"Total size: **{total_size_mb:.1f} MB**"
)

st.divider()

# ---------------------------------------------------------------------------
# TAB 1 — Manage (list + delete)
# TAB 2 — Upload
# ---------------------------------------------------------------------------
tab_manage, tab_upload = st.tabs(["🗑️ Manage Files", "⬆️ Upload Files"])

# ===========================================================================
# TAB 1 — Manage Files
# ===========================================================================
with tab_manage:
    if not all_files:
        st.info("No PDF files found in the corpus folder.")
    else:
        st.markdown("Select files to delete, then click **Delete selected**.")

        # ---- Select all toggle --------------------------------------------
        col_sel, col_del = st.columns([2, 1])
        with col_sel:
            select_all = st.checkbox("Select all", key="select_all")

        # ---- File list with checkboxes ------------------------------------
        to_delete = []
        for f in all_files:
            size_kb = f.stat().st_size / 1024
            size_str = (
                f"{size_kb:.0f} KB" if size_kb < 1024
                else f"{size_kb / 1024:.1f} MB"
            )
            checked = st.checkbox(
                f"{f.name}  —  {size_str}",
                value=select_all,
                key=f"file_{f.name}",
            )
            if checked:
                to_delete.append(f)

        st.divider()

        # ---- Delete button ------------------------------------------------
        if to_delete:
            with col_del:
                st.markdown(f"**{len(to_delete)}** file(s) selected")

            if st.button(
                f"🗑️ Delete {len(to_delete)} selected file(s)",
                type="primary",
                use_container_width=True,
            ):
                deleted = []
                errors = []
                for f in to_delete:
                    try:
                        f.unlink()
                        deleted.append(f.name)
                    except Exception as e:
                        errors.append(f"{f.name}: {e}")

                if deleted:
                    st.success(f"Deleted {len(deleted)} file(s):\n" + "\n".join(f"• {n}" for n in deleted))
                if errors:
                    st.error("Errors:\n" + "\n".join(errors))
                st.rerun()
        else:
            st.info("No files selected.")

# ===========================================================================
# TAB 2 — Upload Files
# ===========================================================================
with tab_upload:
    st.markdown(
        "Upload one or more PDF files. They will be saved directly to "
        f"`{PDF_FOLDER}`."
    )

    uploaded = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded:
        if st.button(
            f"⬆️ Save {len(uploaded)} file(s) to volume",
            type="primary",
            use_container_width=True,
        ):
            saved = []
            errors = []
            for uf in uploaded:
                dest = PDF_FOLDER / uf.name
                try:
                    dest.write_bytes(uf.getvalue())
                    saved.append(uf.name)
                except Exception as e:
                    errors.append(f"{uf.name}: {e}")

            if saved:
                st.success(
                    f"Saved {len(saved)} file(s):\n"
                    + "\n".join(f"• {n}" for n in saved)
                )
            if errors:
                st.error("Errors:\n" + "\n".join(errors))
            st.rerun()
    else:
        st.info("Drag and drop PDFs here, or click to browse.")
