"""
core/pipeline.py
----------------
Orchestrates the full analysis pipeline for a set of uploaded documents.

Input  : List of (filename, file_bytes) tuples + taxonomy config
Output : List of SearchResult objects (one per evidence excerpt found)

Memory strategy (OOM fix):
  - Process one document at a time — never accumulate sentences from all docs
  - Run the full search pipeline per document, then discard sentences
  - Merge results from all documents at the end
  - Peak memory = 1 doc sentences + SBERT model (~500MB), not all docs at once

This module is called from the Streamlit app and returns results that
are stored in session_state for display, filtering, and export.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _import_core_modules():
    """
    Import core submodules defensively, working around the Streamlit
    hot-reload issue where sys.modules entries can be evicted mid-import
    (manifests as KeyError: 'core.cache_manager').
    """
    for mod_name in [
        "core.cache_manager",
        "core.pdf_extractor",
        "core.chunker",
        "core.search_engine",
    ]:
        if mod_name in sys.modules and sys.modules[mod_name] is None:
            del sys.modules[mod_name]

    pdf_extractor = importlib.import_module("core.pdf_extractor")
    chunker = importlib.import_module("core.chunker")
    search_engine = importlib.import_module("core.search_engine")
    cache_manager = importlib.import_module("core.cache_manager")

    return (
        pdf_extractor.extract_document,
        pdf_extractor.DocumentContent,
        chunker.make_sentences,
        search_engine.run_search_pipeline,
        search_engine.SearchResult,
        cache_manager.file_hash,
        cache_manager.get_session_cache,
        cache_manager.set_session_cache,
        cache_manager.embedding_cache_exists,
    )


# Top-level imports for type hints and external callers
from core.search_engine import SearchResult  # noqa: E402
from core.pdf_extractor import DocumentContent  # noqa: E402


def process_documents(
    uploaded_files: List[Tuple[str, bytes]],
    taxonomy: Dict[str, Any],
    status_callback=None,  # optional callable(str) for Streamlit progress messages
) -> Tuple[List[SearchResult], List[DocumentContent]]:
    """
    Full pipeline — processes one document at a time to keep memory low.

    For each document:
      1. Extract text from PDF
      2. Split into sentences
      3. Run full hybrid search (keyword + SBERT + CrossEncoder) on that doc
      4. Discard sentences, keep only results

    Final results from all documents are merged and returned.

    Args:
        uploaded_files  : list of (filename, file_bytes)
        taxonomy        : loaded taxonomy dict
        status_callback : callable that accepts a status string (for st.spinner etc.)

    Returns:
        (results, docs) — SearchResult list and DocumentContent list
    """

    (
        extract_document,
        DocumentContent,
        make_sentences,
        run_search_pipeline,
        SearchResult,
        file_hash,
        get_session_cache,
        set_session_cache,
        embedding_cache_exists,
    ) = _import_core_modules()

    search_cfg = taxonomy.get("search", {})
    sbert_model = search_cfg.get("sbert_model", "all-MiniLM-L6-v2")
    cross_encoder = search_cfg.get(
        "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    top_k = search_cfg.get("top_k_sentences", 5)
    use_sbert = search_cfg.get("use_sbert", True)
    use_cross_encoder = search_cfg.get("use_cross_encoder", False)

    # OpenAI: taxonomy config can force it on, env var can enable it
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY") or "")
    use_openai_cfg = search_cfg.get("use_openai", False)
    use_openai = has_openai_key and (use_openai_cfg or not use_sbert)

    if not use_sbert and not has_openai_key:
        logger.warning(
            "use_sbert=false but OPENAI_API_KEY is not set — "
            "falling back to keyword-only search"
        )

    all_results: List[SearchResult] = []
    docs: List[DocumentContent] = []
    n_total = len(uploaded_files)

    # ---- Cache stats counters -----------------------------------------------
    session_hits = 0
    disk_hits = 0
    api_calls = 0

    for i, (fname, fbytes) in enumerate(uploaded_files, start=1):
        # ---- Compute file hash once per document ----------------------------
        fhash = file_hash(fbytes)

        # ---- Progress -------------------------------------------------------
        if status_callback:
            status_callback(
                f"[{i}/{n_total}] Processing {fname}…"
            )

        # ---- Check session cache for this document's results ---------------
        cache_key = f"results_{fhash}"
        cached = get_session_cache(cache_key)

        if cached is not None:
            doc, doc_results = cached
            session_hits += 1
            logger.info(
                "Session cache HIT for %s (%d results)", fname, len(doc_results)
            )
        else:
            # ---- Check whether ALL theme embeddings are already on disk ----
            themes = taxonomy.get("themes", [])
            all_themes_cached = use_openai and all(
                embedding_cache_exists(fhash, t["label"]) for t in themes
            )
            if all_themes_cached:
                disk_hits += 1
                logger.info(
                    "Disk embedding cache HIT for %s — %d theme(s) cached",
                    fname, len(themes),
                )
            else:
                api_calls += 1

            # ---- Extract text -----------------------------------------------
            doc = extract_document(fbytes, fname)
            sents = make_sentences(doc)
            logger.info("Extracted %s → %d sentences", fname, len(sents))

            if not sents:
                logger.warning("No sentences extracted from %s — skipping", fname)
                docs.append(doc)
                continue

            # ---- Search this document only ----------------------------------
            if status_callback:
                cache_note = " (disk cache)" if all_themes_cached else ""
                status_callback(
                    f"[{i}/{n_total}] Searching {fname} ({len(sents):,} sentences){cache_note}…"
                )

            doc_results = run_search_pipeline(
                sentences=sents,
                taxonomy=taxonomy,
                sbert_model_name=sbert_model,
                cross_encoder_name=cross_encoder,
                top_k_sentences=top_k,
                min_sbert_score=0.25,
                use_openai=use_openai,
                use_cross_encoder=use_cross_encoder,
                use_sbert=use_sbert,
                fhash=fhash,
            )

            logger.info(
                "Searched %s → %d evidence excerpts", fname, len(doc_results)
            )

            # ---- Cache results (not sentences) so re-upload is instant -----
            set_session_cache(cache_key, (doc, doc_results))

            # ---- Explicitly free sentence list before next doc --------------
            del sents

        docs.append(doc)
        all_results.extend(doc_results)

    logger.info(
        "Pipeline complete — %d documents, %d total evidence excerpts "
        "| session hits: %d, disk hits: %d, OpenAI calls: %d",
        len(docs),
        len(all_results),
        session_hits,
        disk_hits,
        api_calls,
    )
    return all_results, docs
