"""
core/corpus_folders.py
======================

Single source of truth for **document corpus folders** on the storage volume.

Rationale
---------
The folder list used to be hard-coded in two separate places
(``app.py`` and ``pages/99_Admin_Upload.py``) as
``["pdfs (default)", "destine", "Custom path"]``. That meant:

* adding a third corpus required a code change in two files, and
* the two lists could (and did) drift apart.

This module discovers corpus folders dynamically instead, so an arbitrary
number of corpora can live side by side under the volume root::

    /data/pdfs/      <- default corpus
    /data/destine/   <- DestinE documents
    /data/trends/    <- trends documents
    /data/<...>/     <- any number of further corpora

Internal folders used by the cache layer (embeddings, outputs, cache,
taxonomies) are excluded so they never show up as if they were corpora.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

#: Volume root. Overridable for local development / tests.
CORPUS_ROOT: Path = Path(os.environ.get("CORPUS_ROOT", "/data"))

#: The corpus used when the user expresses no preference.
DEFAULT_CORPUS_NAME: str = "pdfs"

#: Folder names under the volume root that are NOT document corpora.
RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "cache",
        "embeddings",
        "outputs",
        "taxonomies",
        "logs",
        "tmp",
        "lost+found",
    }
)

#: Document types counted and listed as corpus content.
DOC_SUFFIXES: tuple[str, ...] = (".pdf",)


def _is_corpus_dir(p: Path) -> bool:
    """True if ``p`` looks like a user-facing corpus folder."""
    if not p.is_dir():
        return False
    name = p.name
    if name in RESERVED_NAMES:
        return False
    if name.startswith("."):  # hidden / bookkeeping
        return False
    return True


def list_corpus_folders(create_default: bool = True) -> List[Path]:
    """Return every corpus folder under :data:`CORPUS_ROOT`, sorted.

    The default corpus is always first when present. Missing or unmounted
    volumes yield an empty list rather than raising, so the UI can degrade
    gracefully during local development.
    """
    if create_default:
        try:
            (CORPUS_ROOT / DEFAULT_CORPUS_NAME).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.info("Cannot create default corpus folder: %s", exc)

    try:
        found = [p for p in CORPUS_ROOT.iterdir() if _is_corpus_dir(p)]
    except OSError as exc:
        logger.info("Corpus root %s not readable: %s", CORPUS_ROOT, exc)
        return []

    def sort_key(p: Path) -> tuple[int, str]:
        # default corpus first, then alphabetical
        return (0 if p.name == DEFAULT_CORPUS_NAME else 1, p.name.lower())

    return sorted(found, key=sort_key)


def sanitize_folder_name(raw: str) -> str:
    """Reduce user input to a safe single path segment.

    Strips directory separators and anything other than letters, digits,
    underscore, dash and dot, so a name can never escape the volume root.
    """
    name = (raw or "").strip().replace(" ", "_")
    name = re.sub(r"[^\w\-.]", "", name)
    name = name.strip(".-")  # no leading/trailing dots or dashes
    return name[:64]


def create_corpus_folder(raw_name: str) -> Path:
    """Create (or return) a corpus folder under the volume root.

    Raises
    ------
    ValueError
        If the sanitized name is empty or reserved.
    OSError
        If the folder cannot be created (e.g. volume not mounted).
    """
    name = sanitize_folder_name(raw_name)
    if not name:
        raise ValueError("Folder name is empty after removing unsafe characters.")
    if name in RESERVED_NAMES:
        raise ValueError(f"'{name}' is reserved for internal use — pick another name.")

    target = CORPUS_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Corpus folder ready: %s", target)
    return target


def count_docs(folder: Path) -> int:
    """Number of documents directly inside ``folder`` (non-recursive)."""
    try:
        return sum(
            1
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in DOC_SUFFIXES
        )
    except OSError:
        return 0


def folder_label(folder: Path) -> str:
    """Human-readable dropdown label, e.g. ``pdfs — 101 PDFs (default)``."""
    n = count_docs(folder)
    suffix = "  (default)" if folder.name == DEFAULT_CORPUS_NAME else ""
    return f"{folder.name} — {n} PDF{'s' if n != 1 else ''}{suffix}"
