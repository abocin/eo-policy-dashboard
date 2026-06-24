"""
core/cache_manager.py
---------------------
Streamlit-aware caching helpers.

- Document embeddings are cached in the session so re-uploading the same
  file does not re-run the expensive SBERT encode step.
- A file-based disk cache (JSON) is used for OpenAI embeddings so they
  survive page refreshes.
- A hash of the file bytes is used as the cache key so renamed files are
  still recognised.

Note: no module-level side effects (no mkdir at import time) to avoid
      sys.modules KeyError during Streamlit hot-reload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

DISK_CACHE_DIR = Path(".cache")


def _ensure_cache_dir() -> None:
    """Create cache directory lazily — called inside functions, not at import."""
    DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------

def file_hash(file_bytes: bytes) -> str:
    """SHA-256 hex digest of file bytes — used as cache key."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Session-state cache (in-memory, per Streamlit session)
# ---------------------------------------------------------------------------

def get_session_cache(key: str) -> Optional[Any]:
    return st.session_state.get(f"_cache_{key}")


def set_session_cache(key: str, value: Any) -> None:
    st.session_state[f"_cache_{key}"] = value


def clear_session_cache() -> None:
    keys_to_remove = [k for k in st.session_state if k.startswith("_cache_")]
    for k in keys_to_remove:
        del st.session_state[k]


# ---------------------------------------------------------------------------
# Disk cache (JSON, for OpenAI embeddings)
# ---------------------------------------------------------------------------

def disk_cache_path(cache_name: str) -> Path:
    _ensure_cache_dir()
    return DISK_CACHE_DIR / f"{cache_name}.json"


def load_disk_cache(cache_name: str) -> Dict[str, Any]:
    path = disk_cache_path(cache_name)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Corrupted disk cache %s — resetting", cache_name)
            return {}
    return {}


def save_disk_cache(cache_name: str, data: Dict[str, Any]) -> None:
    path = disk_cache_path(cache_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def get_cached_embedding(cache_name: str, key: str) -> Optional[List[float]]:
    cache = load_disk_cache(cache_name)
    return cache.get(key)


def store_cached_embedding(
    cache_name: str, key: str, embedding: List[float]
) -> None:
    cache = load_disk_cache(cache_name)
    cache[key] = embedding
    save_disk_cache(cache_name, cache)
