"""
core/cache_manager.py
---------------------
Two-layer caching for the EO Policy Dashboard:

Layer 1 — Session cache (in-memory, Streamlit session_state)
  Stores processed (doc, results) pairs for the current browser session.
  Cleared when the user refreshes or the server restarts.

Layer 2 — Disk cache (persistent JSON files)
  Stores OpenAI sentence embedding vectors keyed by:
    <file_hash>_<theme_slug>.json
  Survives server restarts and redeployments (when a Railway volume is mounted).

Cache directory resolution (in order of priority):
  1. CACHE_DIR environment variable  →  set this to /data on Railway volume mount
  2. /data  →  Railway default volume mount point (if exists and writable)
  3. .cache  →  local fallback (ephemeral on Railway without a volume)

Usage:
  from core.cache_manager import (
      file_hash,
      get_session_cache, set_session_cache, clear_session_cache,
      load_embedding_cache, save_embedding_cache,
      get_cached_embeddings, store_cached_embeddings,
      cache_stats,
  )
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
# Cache directory — resolved once at import time
# ---------------------------------------------------------------------------

def _resolve_cache_dir() -> Path:
    """
    Find the best available cache directory:
    1. CACHE_DIR env var (Railway volume mount)
    2. /data if it exists and is writable
    3. .cache (local / ephemeral fallback)
    """
    # Explicit override via env var
    env_dir = os.environ.get("CACHE_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        logger.info("Using CACHE_DIR from environment: %s", p)
        return p

    # Railway volume default mount point
    railway_vol = Path("/data")
    if railway_vol.exists():
        try:
            test = railway_vol / ".write_test"
            test.touch()
            test.unlink()
            logger.info("Using Railway volume at /data for persistent cache")
            return railway_vol
        except OSError:
            pass

    # Local fallback
    p = Path(".cache")
    p.mkdir(parents=True, exist_ok=True)
    logger.info("Using local .cache directory (ephemeral)")
    return p


CACHE_DIR: Path = _resolve_cache_dir()
EMBEDDINGS_DIR: Path = CACHE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR: Path = CACHE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# File hash  (cache key)
# ---------------------------------------------------------------------------

def file_hash(file_bytes: bytes) -> str:
    """SHA-256 hex digest (first 16 chars) — used as cache key."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


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
# Disk embedding cache  (persistent)
# ---------------------------------------------------------------------------

def _embed_cache_path(fhash: str, theme_label: str) -> Path:
    return EMBEDDINGS_DIR / f"{fhash}_{_theme_slug(theme_label)}.npy"


def get_cached_embeddings(
    fhash: str,
    theme_label: str,
) -> Optional[np.ndarray]:
    """
    Return cached sentence embedding matrix for (file_hash, theme) if it exists.
    Returns None on cache miss.
    """
    path = _embed_cache_path(fhash, theme_label)
    if path.exists():
        try:
            arr = np.load(str(path), allow_pickle=False)
            logger.debug("Embedding cache HIT: %s / %s", fhash, theme_label)
            return arr
        except Exception as exc:
            logger.warning("Corrupt embedding cache file %s: %s — deleting", path, exc)
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
        logger.debug("Embedding cache STORE: %s / %s (%s)", fhash, theme_label, path.name)
    except Exception as exc:
        logger.warning("Failed to store embedding cache: %s", exc)


def embedding_cache_exists(fhash: str, theme_label: str) -> bool:
    return _embed_cache_path(fhash, theme_label).exists()


# ---------------------------------------------------------------------------
# Cache statistics  (shown in sidebar)
# ---------------------------------------------------------------------------

def cache_stats() -> Dict[str, Any]:
    """Return a summary dict for display in the Streamlit sidebar."""
    files = list(EMBEDDINGS_DIR.glob("*.npy"))
    total_bytes = sum(f.stat().st_size for f in files)
    unique_docs = len({f.name.split("_")[0] for f in files})
    return {
        "cache_dir": str(CACHE_DIR),
        "is_persistent": str(CACHE_DIR).startswith("/data") or
                         os.environ.get("CACHE_DIR", "").strip() != "",
        "cached_files": len(files),
        "unique_docs": unique_docs,
        "total_size_mb": round(total_bytes / 1_048_576, 1),
    }


def clear_disk_cache() -> int:
    """Delete all cached embedding files. Returns number of files deleted."""
    files = list(EMBEDDINGS_DIR.glob("*.npy"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.info("Disk cache cleared: %d files deleted", len(files))
    return len(files)


# ---------------------------------------------------------------------------
# Upload staging  (write PDFs to disk so bytes are never held in RAM)
# ---------------------------------------------------------------------------

def stage_upload(filename: str, file_bytes: bytes) -> Path:
    """
    Write uploaded PDF bytes to UPLOADS_DIR and return the path.
    Filename is sanitised; existing files with the same name are overwritten.
    """
    safe_name = "".join(
        c if (c.isalnum() or c in "._- ") else "_" for c in filename
    )
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(file_bytes)
    logger.debug("Staged upload: %s (%d bytes)", dest.name, len(file_bytes))
    return dest


def list_staged_uploads() -> List[str]:
    """Return filenames of all PDFs currently in the upload staging area."""
    return sorted(p.name for p in UPLOADS_DIR.glob("*.pdf"))


def read_staged_upload(filename: str) -> Optional[bytes]:
    """Read bytes for a staged PDF. Returns None if the file doesn't exist."""
    path = UPLOADS_DIR / filename
    if path.exists():
        return path.read_bytes()
    return None


def clear_staged_uploads() -> int:
    """Delete all staged PDFs. Returns number of files deleted."""
    files = list(UPLOADS_DIR.glob("*.pdf"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.info("Staged uploads cleared: %d files deleted", len(files))
    return len(files)


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# ---------------------------------------------------------------------------

def load_disk_cache(cache_name: str) -> Dict[str, Any]:
    path = CACHE_DIR / f"{cache_name}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Corrupt JSON cache %s — resetting", cache_name)
    return {}


def save_disk_cache(cache_name: str, data: Dict[str, Any]) -> None:
    path = CACHE_DIR / f"{cache_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
