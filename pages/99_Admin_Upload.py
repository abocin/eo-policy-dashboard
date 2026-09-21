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

from core.corpus_folders import (
    create_corpus_folder,
    folder_label,
    list_corpus_folders,
)

_DEFAULT_FOLDER = Path(os.environ.get("PDF_FOLDER", "/data/pdfs"))

st.set_page_config(page_title="PDF File Manager", page_icon="🗂️", layout="wide")
st.title("🗂️ PDF File Manager")

# Target folder is discovered from the volume root, exactly like the main page,
# so any number of corpora can be managed without a code change here.
_folders = list_corpus_folders()
_NEWF = "➕ New folder…"
_targets = {folder_label(f): f for f in _folders}
_choice = st.selectbox(
    "Target folder",
    list(_targets.keys()) + [_NEWF],
    key="admin_target_folder",
    help="Everything on this page — the file list, deletions, additions — "
         "applies to this folder only.",
)

if _choice == _NEWF:
    _nm = st.text_input("New folder name", key="admin_new_folder", placeholder="trends")
    if st.button("Create folder", key="admin_create_folder"):
        try:
            _c = create_corpus_folder(_nm)
            st.success(f"Created `{_c}` — select it above.")
        except (ValueError, OSError) as _exc:
            st.error(f"Could not create folder: {_exc}")
    st.stop()

PDF_FOLDER = _targets[_choice]

st.caption(f"Managing files in `{PDF_FOLDER}`")

# Ensure folder exists. If the volume is missing or not writable this must NOT
# raise: an uncaught error here aborts the whole script, so the upload widget
# below would never render and the page would look broken rather than explain
# itself.
try:
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
except OSError as _e:
    st.error(
        f"Cannot create or access `{PDF_FOLDER}`: {_e}\n\n"
        "On Railway, check that a volume is mounted at `/data`. "
        "You can still pick a different target folder above."
    )
    st.stop()

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
st.info(
    "This page manages **PDF documents only**. To change the taxonomy YAML, "
    "use **2. Taxonomy** in the sidebar of the main page.",
    icon="ℹ️",
)

tab_manage, tab_url, tab_copy, tab_upload = st.tabs(
    [
        "🗑️ Manage PDFs",
        "🌐 Add PDFs from URL",
        "📥 Copy PDFs between folders",
        "⬆️ Upload PDFs from computer",
    ]
)

# ===========================================================================
# TAB 1 — Manage Files
# ===========================================================================
with tab_manage:
    if not all_files:
        st.info("No PDF files found in the corpus folder.")
    else:
        st.markdown("Select files to delete, then click **Delete selected**.")

        # ---- Initialise selection state -----------------------------------
        for f in all_files:
            key = f"sel_{f.name}"
            if key not in st.session_state:
                st.session_state[key] = False

        # ---- Select all / Deselect all buttons ----------------------------
        def _select_all():
            for f in all_files:
                st.session_state[f"sel_{f.name}"] = True

        def _deselect_all():
            for f in all_files:
                st.session_state[f"sel_{f.name}"] = False

        col_a, col_b, col_c = st.columns([1, 1, 4])
        with col_a:
            st.button("Select all", width="stretch", on_click=_select_all)
        with col_b:
            st.button("Deselect all", width="stretch", on_click=_deselect_all)

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
                key=f"sel_{f.name}",
            )
            if checked:
                to_delete.append(f)

        st.divider()

        # ---- Delete button ------------------------------------------------
        if to_delete:
            st.markdown(f"**{len(to_delete)}** file(s) selected")
            if st.button(
                f"🗑️ Delete {len(to_delete)} selected file(s)",
                type="primary",
                width="stretch",
            ):
                deleted = []
                errors = []
                for f in to_delete:
                    try:
                        f.unlink()
                        # Clear session state for deleted file
                        st.session_state.pop(f"sel_{f.name}", None)
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
# Bulk import helpers
# ===========================================================================
#: Refuse absurd archives outright (zip-bomb guard).
_MAX_UNCOMPRESSED = 4 * 1024 ** 3   # 4 GB total
_MAX_RATIO = 200                    # compressed:uncompressed


def _normalise_share_url(url: str) -> str:
    """Turn common 'share page' links into direct-download links.

    Google Drive and Dropbox share URLs return an HTML viewer page, not the
    file, so downloading them verbatim saves a useless HTML blob. Rewrites
    them to their direct-download equivalents; other URLs pass through.
    """
    import re as _re

    m = _re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([\w-]{20,})", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in url:
        stripped = _re.sub(r"[?&]dl=[01]", "", url)
        # decide the separator from the STRIPPED url -- removing '?dl=0' can
        # leave no query string at all, in which case '&dl=1' is invalid.
        return stripped + ("&" if "?" in stripped else "?") + "dl=1"
    if "sharepoint.com" in url or "1drv.ms" in url:
        return url + ("&" if "?" in url else "?") + "download=1"
    return url


def _extract_pdfs_from_zip(zip_path: Path, dest: Path) -> tuple[list[str], list[str]]:
    """Extract every PDF inside ``zip_path`` into ``dest`` (flattened).

    Returns ``(added, skipped)``. Hardened against the usual archive traps:
    absolute/`..` member paths are never joined onto dest (only the basename is
    used), non-PDF members are ignored, and oversized or absurdly compressed
    archives are rejected before extraction.
    """
    import re as _re
    import zipfile

    added: list[str] = []
    skipped: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        total_unc = sum(i.file_size for i in infos)
        total_comp = max(1, sum(i.compress_size for i in infos))
        if total_unc > _MAX_UNCOMPRESSED:
            raise ValueError(
                f"Archive expands to {total_unc / 1024**3:.1f} GB — refusing."
            )
        if total_unc / total_comp > _MAX_RATIO:
            raise ValueError("Archive compression ratio looks like a zip bomb — refusing.")

        pdf_infos = [i for i in infos if i.filename.lower().endswith(".pdf")]
        if not pdf_infos:
            raise ValueError("No .pdf files found inside the archive.")

        for info in pdf_infos:
            # Use ONLY the basename: a member called '../../etc/x.pdf' or
            # '/abs/x.pdf' must never write outside dest.
            base = Path(info.filename.replace("\\", "/")).name
            base = _re.sub(r"[^\w\-. ]", "_", base).strip() or "document.pdf"
            if base.startswith("._"):      # macOS resource forks
                skipped.append(f"{base}: macOS resource fork")
                continue

            target = dest / base
            if target.exists():
                skipped.append(f"{base}: already present")
                continue

            with zf.open(info) as src:
                head = src.read(5)
                if head != b"%PDF-":
                    skipped.append(f"{base}: not a real PDF")
                    continue
                tmp = target.with_suffix(".part")
                with open(tmp, "wb") as out:
                    out.write(head)
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            tmp.rename(target)
            added.append(f"{base} ({info.file_size / 1_048_576:.1f} MB)")

    return added, skipped


# ===========================================================================
# Fetch from URL — server-side download, no file picker involved
# ===========================================================================
def _render_fetch_from_url(dest: Path) -> None:
    """Download PDFs straight onto the volume from public URLs."""
    import re as _re
    import urllib.parse as _urlparse

    import requests  # bundled with streamlit

    st.markdown(
        f"Paste links (one per line). The **server** downloads them into "
        f"`{dest}` — your browser is not involved, so no file dialog is needed."
    )
    st.markdown(
        "- A **`.zip` link** is the bulk path: every PDF inside is extracted "
        "into this folder in one go.\n"
        "- Google Drive, Dropbox and OneDrive/SharePoint share links are "
        "converted to direct downloads automatically.\n"
        "- Individual `.pdf` links also work."
    )
    raw = st.text_area(
        "Links (.zip or .pdf)",
        height=150,
        key="fetch_urls",
        placeholder="https://example.org/strategy.pdf\nhttps://example.org/roadmap.pdf",
    )
    urls = [u.strip() for u in raw.splitlines() if u.strip()]
    if urls:
        st.caption(f"{len(urls)} URL(s) ready.")
    if not st.button(
        f"🌐 Import {len(urls)} link(s)",
        disabled=not urls, type="primary",
        width="stretch", key="fetch_go",
    ):
        return

    prog = st.progress(0.0)
    ok, bad = [], []
    for i, url in enumerate(urls, 1):
        prog.progress(i / len(urls), text=f"{i}/{len(urls)} — {url[:60]}")
        try:
            # Derive a safe filename from the URL path, falling back to a
            # counter. Always force a .pdf suffix.
            _parts = [
                _urlparse.unquote(s)
                for s in _urlparse.urlparse(url).path.split("/") if s
            ]
            name = _parts[-1] if _parts else ""
            # Many document portals end the URL in a generic segment such as
            # ".../ST-11321-2023-INIT/en/pdf", which would yield "pdf.pdf".
            # Walk back to the last segment that actually identifies the file.
            if Path(name).stem.lower() in ("", "pdf", "download", "file", "en", "view"):
                for _seg in reversed(_parts[:-1]):
                    if Path(_seg).stem.lower() not in ("", "pdf", "download", "file", "en", "view"):
                        name = _seg
                        break
            name = _re.sub(r'[^\w\-. ]', "_", name).strip() or f"download_{i}"
            if not name.lower().endswith(".pdf"):
                name += ".pdf"

            target = dest / name
            if target.exists():
                bad.append(f"{name}: already present, skipped")
                continue

            url = _normalise_share_url(url)
            resp = requests.get(url, timeout=600, stream=True, headers={
                "User-Agent": "Mozilla/5.0 (compatible; eo-policy-dashboard)"
            })
            resp.raise_for_status()

            # ---- bulk archive branch ---------------------------------------
            _is_zip = (
                url.lower().split("?")[0].endswith(".zip")
                or "zip" in resp.headers.get("Content-Type", "").lower()
            )
            if _is_zip:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as _tf:
                    for chunk in resp.iter_content(1 << 20):
                        _tf.write(chunk)
                    _zp = Path(_tf.name)
                try:
                    _added, _skipped = _extract_pdfs_from_zip(_zp, dest)
                    ok.extend(_added)
                    bad.extend(_skipped)
                finally:
                    _zp.unlink(missing_ok=True)
                continue

            ctype = resp.headers.get("Content-Type", "")
            tmp = target.with_suffix(".part")
            size = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(65536):
                    fh.write(chunk)
                    size += len(chunk)

            # Validate it really is a PDF before keeping it (many bad links
            # return an HTML error page with status 200).
            with open(tmp, "rb") as fh:
                magic = fh.read(5)
            if magic != b"%PDF-":
                tmp.unlink(missing_ok=True)
                bad.append(f"{name}: not a PDF (Content-Type: {ctype or 'unknown'})")
                continue

            tmp.rename(target)
            ok.append(f"{name} ({size / 1_048_576:.1f} MB)")
        except Exception as exc:
            bad.append(f"{url[:60]}: {exc}")

    prog.empty()
    if ok:
        st.success(f"Downloaded {len(ok)} file(s):\n" + "\n".join(f"• {n}" for n in ok))
    if bad:
        st.warning("Not downloaded:\n" + "\n".join(f"• {b}" for b in bad))
    if ok:
        st.rerun()


# ===========================================================================
# Copy from another folder — needs no browser upload at all
# ===========================================================================
def _render_copy_from_folder(dest: Path) -> None:
    """Copy PDFs into `dest` from any other server-side folder.

    Browser uploads are the fragile path (size limits, timeouts, picker
    quirks). Everything already on the volume can be moved server-side.
    """
    import shutil

    st.markdown(
        f"Copy PDFs that are already on the server into `{dest}` — "
        "no browser upload involved."
    )

    src_options = [p for p in list_corpus_folders() if p != dest]
    src_label = st.selectbox(
        "Source folder",
        [str(p) for p in src_options] + ["Other path…"],
        key="copy_src_choice",
    )
    src = Path(
        st.text_input("Source path", value="", key="copy_src_custom")
        if src_label == "Other path…" else src_label
    )

    if not src or str(src) in ("", "."):
        st.caption("Enter a source path to continue.")
        return
    if not src.is_dir():
        st.warning(f"Not a directory: `{src}`")
        return

    available = sorted(
        p for p in src.glob("*.pdf")
        if p.is_file() and not p.name.startswith("._")
    )
    if not available:
        st.info(f"No PDFs found in `{src}`.")
        return

    picked = st.multiselect(
        f"Files in {src} ({len(available)} available)",
        [p.name for p in available],
        key="copy_pick",
        help="Leave empty and use 'Copy all' below to take everything.",
    )
    move = st.checkbox(
        "Move instead of copy (removes from source)", key="copy_move"
    )
    c1, c2 = st.columns(2)
    go_sel = c1.button(
        f"📥 {'Move' if move else 'Copy'} {len(picked)} selected",
        disabled=not picked, width="stretch", key="copy_go_sel",
    )
    go_all = c2.button(
        f"📥 {'Move' if move else 'Copy'} all {len(available)}",
        width="stretch", key="copy_go_all",
    )
    if not (go_sel or go_all):
        return

    names = picked if go_sel else [p.name for p in available]
    done, failed, skipped = [], [], []
    for name in names:
        target = dest / name
        if target.exists():
            skipped.append(name)
            continue
        try:
            shutil.move(str(src / name), str(target)) if move \
                else shutil.copy2(str(src / name), str(target))
            done.append(name)
        except OSError as exc:
            failed.append(f"{name}: {exc}")

    if done:
        st.success(f"{'Moved' if move else 'Copied'} {len(done)} file(s) to `{dest}`.")
    if skipped:
        st.info(f"Skipped {len(skipped)} already present: " + ", ".join(skipped[:8]))
    if failed:
        st.error("Failed:\n" + "\n".join(failed))
    if done:
        st.rerun()


# ===========================================================================
# TAB 2 — Upload Files
# ===========================================================================
with tab_upload:
    st.markdown(
        "Upload one or more PDF files. They will be saved directly to "
        f"`{PDF_FOLDER}`."
    )
    st.caption(
        "Use the **⬆ Upload** button below to open your file browser, or drag "
        "files straight onto it. Streamlit's uploader is a compact button in "
        "recent versions — it is not a large drop-zone."
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
            width="stretch",
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
        # NOTE: do not render a fake "drag and drop / click to browse" banner
        # here. Streamlit >=1.6x renders the uploader as a small "Upload"
        # button rather than the old large dashed drop-zone, so a separate
        # banner looks like the drop-zone but is inert -- users click it,
        # nothing happens, and the upload appears broken.
        st.caption("No files selected yet.")


# ===========================================================================
# TAB 3 — Copy from folder
# ===========================================================================
with tab_copy:
    _render_copy_from_folder(PDF_FOLDER)


# ===========================================================================
# TAB 4 — Fetch from URL
# ===========================================================================
with tab_url:
    _render_fetch_from_url(PDF_FOLDER)
