"""
core/cache_manager.py
---------------------
Two-layer caching + persistent directory management for the EO Policy Dashboard.

Layer 1 — Session cache (in-memory, Streamlit session_state)
  Stores processed (doc, results) pairs for the current browser session.
  Cleared on browser refresh or server restart.

Layer 2 — Disk cache (persistent .npy files)
  Stores OpenAI sentence embedding vectors, keyed by:
    <file_hash_16>_<theme_slug>.npy
  Survives server restarts and redeployments when a Railway volume is mounted.

Persistent directory layout (under BASE_DIR):
  BASE_DIR/
    embeddings/   ← OpenAI sentence vectors (.npy files)
    outputs/      ← Auto-saved CSV/Excel/JSON exports
    uploads/      ← (optional) browser-upload staging area

BASE_DIR resolution (priority order):
  1. CACHE_DIR env var       → e.g. /data/cache  (Railway volume)
  2. /data                   → Railway default volume mount (if writable)
  3. .cache                  → local ephemeral fallback

PDF_FOLDER env var:
  Absolute path to a folder containing PDFs on the server.
  On Railway: mount a volume at /data, put PDFs in /data/pdfs, set PDF_FOLDER=/data/pdfs.
  The Streamlit sidebar pre-fills this value automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory resolution — called once at import time
# ---------------------------------------------------------------------------

def _resolve_base_dir() -> Path:
    """
    Find the best available persistent base directory.

    Priority:
    1. CACHE_DIR env var  (explicit Railway volume path, e.g. /data/cache)
    2. /data              (Railway default volume mount — if exists and writable)
    3. .cache             (local / ephemeral fallback)
    """
    env_dir = os.environ.get("CACHE_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
            logger.info("Cache: using CACHE_DIR from environment: %s", p)
            return p
        except OSError as exc:
            logger.warning("CACHE_DIR %s not writable (%s) — falling back", p, exc)

    railway_vol = Path("/data")
    if railway_vol.exists():
        try:
            test = railway_vol / ".write_test"
            test.touch()
            test.unlink()
            logger.info("Cache: using Railway volume at /data")
            return railway_vol
        except OSError:
            pass

    p = Path(".cache")
    p.mkdir(parents=True, exist_ok=True)
    logger.info("Cache: using local .cache directory (ephemeral)")
    return p


BASE_DIR: Path = _resolve_base_dir()
EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
UPLOADS_DIR: Path = BASE_DIR / "uploads"

for _d in (EMBEDDINGS_DIR, OUTPUTS_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Legacy alias kept for imports that use CACHE_DIR
CACHE_DIR: Path = BASE_DIR


# ---------------------------------------------------------------------------
# File hash  (cache key)
# ---------------------------------------------------------------------------

def file_hash(file_bytes: bytes) -> str:
    """SHA-256 hex digest (first 16 chars) — used as cache key."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def file_hash_from_path(path: Path) -> str:
    """
    Compute SHA-256 of a file on disk without reading it all into RAM at once.
    Uses 64 KB chunks to stay memory-safe for large PDFs.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _theme_slug(theme_label: str) -> str:
    """Convert theme label to a safe filename component."""
    return "".join(c if c.isalnum() else "_" for c in theme_label).lower()


# ---------------------------------------------------------------------------
# Session cache  (in-memory, per Streamlit session)
# ---------------------------------------------------------------------------

def get_session_cache(key: str) -> Optional[Any]:
    return st.session_state.get(f"_cache_{key}")


def set_session_cache(key: str, value: Any) -> None:
    st.session_state[f"_cache_{key}"] = value


def clear_session_cache() -> None:
    keys = [k for k in st.session_state if k.startswith("_cache_")]
    for k in keys:
        del st.session_state[k]
    logger.info("Session cache cleared (%d entries)", len(keys))


# ---------------------------------------------------------------------------
# Disk embedding cache  (persistent .npy files)
# ---------------------------------------------------------------------------

def _embed_cache_path(fhash: str, theme_label: str) -> Path:
    return EMBEDDINGS_DIR / f"{fhash}_{_theme_slug(theme_label)}.npy"


def get_cached_embeddings(
    fhash: str,
    theme_label: str,
) -> Optional[np.ndarray]:
    """
    Return cached sentence embedding matrix for (file_hash, theme) if it exists.
    Returns None on cache miss or if the file is corrupt.
    """
    path = _embed_cache_path(fhash, theme_label)
    if path.exists():
        try:
            arr = np.load(str(path), allow_pickle=False)
            logger.debug("Embedding cache HIT: %s / %s", fhash, theme_label)
            return arr
        except Exception as exc:
            logger.warning("Corrupt embedding cache %s: %s — deleting", path.name, exc)
            path.unlink(missing_ok=True)
    return None


def store_cached_embeddings(
    fhash: str,
    theme_label: str,
    embeddings: np.ndarray,
) -> None:
    """Persist sentence embedding matrix to disk as a .npy file."""
    path = _embed_cache_path(fhash, theme_label)
    try:
        np.save(str(path), embeddings.astype(np.float32), allow_pickle=False)
        logger.debug("Embedding cache STORE: %s / %s → %s", fhash, theme_label, path.name)
    except Exception as exc:
        logger.warning("Failed to store embedding cache: %s", exc)


def embedding_cache_exists(fhash: str, theme_label: str) -> bool:
    return _embed_cache_path(fhash, theme_label).exists()


# ---------------------------------------------------------------------------
# Cache statistics  (shown in sidebar)
# ---------------------------------------------------------------------------

def cache_stats() -> Dict[str, Any]:
    """Return a summary dict for display in the Streamlit sidebar."""
    emb_files = list(EMBEDDINGS_DIR.glob("*.npy"))
    out_files = list(OUTPUTS_DIR.iterdir()) if OUTPUTS_DIR.exists() else []
    emb_bytes = sum(f.stat().st_size for f in emb_files)
    unique_docs = len({f.name.split("_")[0] for f in emb_files})

    is_persistent = (
        str(BASE_DIR).startswith("/data")
        or os.environ.get("CACHE_DIR", "").strip() != ""
    )

    return {
        "base_dir": str(BASE_DIR),
        "cache_dir": str(BASE_DIR),           # legacy key
        "is_persistent": is_persistent,
        "embeddings_dir": str(EMBEDDINGS_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "cached_files": len(emb_files),
        "unique_docs": unique_docs,
        "total_size_mb": round(emb_bytes / 1_048_576, 1),
        "output_files": len(out_files),
    }


def clear_disk_cache() -> int:
    """Delete all cached embedding files. Returns number deleted."""
    files = list(EMBEDDINGS_DIR.glob("*.npy"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.info("Disk embedding cache cleared: %d files deleted", len(files))
    return len(files)


# ---------------------------------------------------------------------------
# Outputs directory helpers
# ---------------------------------------------------------------------------

def save_output(filename: str, data: bytes) -> Path:
    """
    Save an export file (CSV, Excel, JSON) to the persistent outputs directory.
    Returns the path it was saved to.
    """
    dest = OUTPUTS_DIR / filename
    dest.write_bytes(data)
    logger.info("Output saved: %s (%d bytes)", dest.name, len(data))
    return dest


def list_outputs() -> List[Path]:
    """Return all files in the outputs directory, sorted by modification time (newest first)."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# PDF folder discovery
# ---------------------------------------------------------------------------

def discover_pdfs(folder: str | Path, recursive: bool = False) -> List[Path]:
    """
    Find all PDF files in a folder.

    Args:
        folder:    Absolute or relative path to the PDF directory.
        recursive: If True, also scan subdirectories.

    Returns:
        Sorted list of Path objects. Empty list if folder does not exist.

    Raises:
        PermissionError: If the folder exists but is not readable.
    """
    folder = Path(folder)

    if not folder.exists():
        logger.warning("PDF folder not found: %s", folder)
        return []

    if not folder.is_dir():
        logger.warning("PDF folder path is not a directory: %s", folder)
        return []

    pattern = "**/*.pdf" if recursive else "*.pdf"
    all_pdfs = folder.glob(pattern)

    # Filter out macOS resource fork files (._filename.pdf) and other
    # hidden files — these are created when zipping on macOS and are
    # not real PDFs.
    pdfs = sorted(
        p for p in all_pdfs
        if not p.name.startswith("._") and not p.name.startswith(".")
    )
    logger.info("Discovered %d PDF(s) in %s (recursive=%s)", len(pdfs), folder, recursive)
    return pdfs


# ---------------------------------------------------------------------------
# Upload staging (browser upload → disk, bypasses RAM accumulation)
# ---------------------------------------------------------------------------

def stage_upload(filename: str, file_bytes: bytes) -> Path:
    """
    Write uploaded PDF bytes to UPLOADS_DIR immediately.
    Bytes are discarded from RAM after this call.
    Filename is sanitised; existing files are overwritten.
    """
    safe_name = "".join(
        c if (c.isalnum() or c in "._- ") else "_" for c in filename
    )
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(file_bytes)
    logger.debug("Staged upload: %s (%d bytes)", dest.name, len(file_bytes))
    return dest


def list_staged_uploads() -> List[str]:
    """Return filenames of all PDFs in the upload staging area."""
    return sorted(p.name for p in UPLOADS_DIR.glob("*.pdf"))


def read_staged_upload(filename: str) -> Optional[bytes]:
    """Read bytes for a staged PDF. Returns None if not found."""
    path = UPLOADS_DIR / filename
    return path.read_bytes() if path.exists() else None


def clear_staged_uploads() -> int:
    """Delete all staged PDFs. Returns number deleted."""
    files = list(UPLOADS_DIR.glob("*.pdf"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.info("Staged uploads cleared: %d files", len(files))
    return len(files)


# ---------------------------------------------------------------------------
# Results persistence  (share results across Streamlit multipage sessions)
# ---------------------------------------------------------------------------

RESULTS_FILE: Path = BASE_DIR / "last_results.json"


def save_results(results: list, docs: list, corpus_filenames: list) -> None:
    """
    Serialise analysis results to disk so sidebar pages can load them
    without re-running the pipeline.  Called automatically after each run.
    """
    import dataclasses
    payload = {
        "corpus_filenames": corpus_filenames,
        "docs": [
            {"filename": d.filename, "page_count": getattr(d, "page_count", 0)}
            for d in docs
        ],
        "results": [
            dataclasses.asdict(r) if dataclasses.is_dataclass(r) else vars(r)
            for r in results
        ],
    }
    try:
        with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
            import json
            json.dump(payload, fh, ensure_ascii=False, default=str)
        logger.info("Results persisted to %s (%d excerpts)", RESULTS_FILE, len(results))
    except Exception as exc:
        logger.warning("Could not persist results: %s", exc)


def load_results() -> Optional[Dict[str, Any]]:
    """
    Load previously persisted results from disk.
    Returns None if no results file exists or it is corrupt.
    """
    if not RESULTS_FILE.exists():
        return None
    try:
        import json
        with open(RESULTS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not load persisted results: %s", exc)
        return None


def results_file_exists() -> bool:
    return RESULTS_FILE.exists()


# ---------------------------------------------------------------------------
# Legacy JSON helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_disk_cache(cache_name: str) -> Dict[str, Any]:
    path = BASE_DIR / f"{cache_name}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Corrupt JSON cache %s — resetting", cache_name)
    return {}


def save_disk_cache(cache_name: str, data: Dict[str, Any]) -> None:
    path = BASE_DIR / f"{cache_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


# Legacy aliases
load_embedding_cache = load_disk_cache
save_embedding_cache = save_disk_cache
