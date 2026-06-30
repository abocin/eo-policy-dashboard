"""
core/search_engine.py
---------------------
Hybrid search engine supporting three modes (set in config/taxonomy.yaml):

  Mode A — OpenAI-only  (use_sbert: false, use_openai: true)
    - No torch, no model download, ~200MB RAM
    - Keyword match + OpenAI text-embedding-3-small for semantic scoring
    - Recommended for Streamlit Cloud (1GB RAM limit)

  Mode B — SBERT-only  (use_sbert: true, use_openai: false)
    - Offline, no API key needed
    - Keyword match + local SBERT encode + optional CrossEncoder
    - ~600MB RAM (all-MiniLM-L6-v2)

  Mode C — Hybrid  (use_sbert: true, use_cross_encoder: true, use_openai: true)
    - Best quality, needs >2GB RAM — run locally

In all modes keyword matching always runs first as a zero-cost safety net.
"""

from __future__ import annotations

import os
import re
import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MAX_ENCODE_BATCH = 512   # sentences per SBERT encode batch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single evidence excerpt from one document."""
    doc_filename: str
    page: int
    excerpt: str
    theme: str
    keyword_hit: bool
    sbert_score: float
    cross_encoder_score: float = 0.0
    openai_score: float = 0.0
    final_score: float = 0.0
    validation_category: str = "UNSCORED"
    human_label: str = ""
    matched_keyword: str = ""
    chunk_index: int = -1
    eo_relevance_score: float = 0.0  # Stage-2 EO capacity-building re-rank score
    context: str = ""               # Surrounding sentences (N before + N after excerpt)
    commitment_score: float = 0.0   # 0.0 (aspirational) → 1.0 (binding/funded)
    commitment_level: str = ""      # BINDING | STRONG | MODERATE | ASPIRATIONAL
    lifecycle_stage: str = ""       # Awareness|Education|Training|Skills|Innovation|Entrepreneurship|Adoption|Sustainability


# ---------------------------------------------------------------------------
# Model loader (lazy, cached at module level)
# ---------------------------------------------------------------------------

_sbert_model = None
_sbert_model_name_loaded: Optional[str] = None
_cross_encoder = None


def _get_sbert(model_name: str = "all-MiniLM-L6-v2"):
    global _sbert_model, _sbert_model_name_loaded
    if _sbert_model is None or _sbert_model_name_loaded != model_name:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading SBERT model: %s", model_name)
        _sbert_model = SentenceTransformer(model_name)
        _sbert_model_name_loaded = model_name
    return _sbert_model


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # type: ignore
        logger.info("Loading CrossEncoder: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Keyword search  (always runs, language-agnostic)
# ---------------------------------------------------------------------------

def keyword_search(
    sentences,
    theme_label: str,
    keywords: List[str],
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for sent in sentences:
        for kw in keywords:
            if re.search(re.escape(kw), sent.text, re.IGNORECASE):
                results.append(SearchResult(
                    doc_filename=sent.doc_filename,
                    page=sent.page,
                    excerpt=sent.text,
                    theme=theme_label,
                    keyword_hit=True,
                    sbert_score=0.0,
                    matched_keyword=kw,
                ))
                break
    return results


# ---------------------------------------------------------------------------
# SBERT encoding  (Mode B / C only — skipped when use_sbert=False)
# ---------------------------------------------------------------------------

def encode_sentences(sent_texts: List[str], sbert_model_name: str) -> np.ndarray:
    """Encode all sentences into float32 matrix. Called ONCE per document."""
    model = _get_sbert(sbert_model_name)
    chunks = []
    for start in range(0, len(sent_texts), MAX_ENCODE_BATCH):
        vecs = model.encode(
            sent_texts[start: start + MAX_ENCODE_BATCH],
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True,
        )
        chunks.append(vecs)
    return np.vstack(chunks).astype(np.float32)


def semantic_search_with_vecs(
    sentences,
    sentence_vecs: np.ndarray,
    theme_queries: List[str],
    theme_label: str,
    sbert_model_name: str,
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[SearchResult]:
    """Semantic search for one theme reusing pre-computed sentence vectors."""
    model = _get_sbert(sbert_model_name)
    query_vecs = model.encode(
        theme_queries, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)

    sim_matrix = cosine_similarity(query_vecs, sentence_vecs)
    seen: set = set()
    results: List[SearchResult] = []

    for q_scores in sim_matrix:
        for idx in np.argsort(q_scores)[::-1][:top_k]:
            score = float(q_scores[idx])
            if score < min_score:
                continue
            if idx in seen:
                for r in results:
                    if r.excerpt == sentences[idx].text:
                        r.sbert_score = max(r.sbert_score, score)
                continue
            seen.add(idx)
            results.append(SearchResult(
                doc_filename=sentences[idx].doc_filename,
                page=sentences[idx].page,
                excerpt=sentences[idx].text,
                theme=theme_label,
                keyword_hit=False,
                sbert_score=score,
            ))

    del sim_matrix
    return results


# ---------------------------------------------------------------------------
# OpenAI semantic search  (Mode A / C — requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def openai_embed_sentences(
    sentences,
    fhash: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Embed all sentences ONCE via OpenAI text-embedding-3-small.
    Returns a float32 matrix (n_sentences x embedding_dim).

    Checks disk cache first keyed by fhash + "__sentences__".
    Stores result on a cache miss so subsequent runs skip the API call.
    Returns None if the API key is missing or the call fails.

    Call this ONCE per document before the theme loop, then pass the
    matrix to openai_semantic_search_with_vecs() for each theme.
    This reduces API calls from 7x per document to 1x.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        return None

    SENT_CACHE_KEY = "__sentences__"

    try:
        from openai import OpenAI  # type: ignore
        from core.cache_manager import get_cached_embeddings, store_cached_embeddings

        if fhash:
            cached = get_cached_embeddings(fhash, SENT_CACHE_KEY)
            if cached is not None:
                logger.info(
                    "Sentence embedding cache HIT -- %s (%d vecs)", fhash, len(cached)
                )
                return cached

        client = OpenAI(api_key=api_key)
        sent_texts = [s.text for s in sentences]
        all_vecs = []
        OPENAI_BATCH = 500
        for start in range(0, len(sent_texts), OPENAI_BATCH):
            batch = sent_texts[start: start + OPENAI_BATCH]
            resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
            all_vecs.extend([e.embedding for e in resp.data])

        arr = np.array(all_vecs, dtype=np.float32)
        logger.info(
            "Sentence embedding cache MISS -- %s  (called OpenAI, %d vecs)",
            fhash or "no-hash", len(arr),
        )

        if fhash:
            store_cached_embeddings(fhash, SENT_CACHE_KEY, arr)

        return arr

    except Exception as exc:
        logger.warning("openai_embed_sentences failed: %s", exc)
        return None


def openai_semantic_search_with_vecs(
    sentences,
    sent_vecs: np.ndarray,
    theme_queries: List[str],
    theme_label: str,
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[SearchResult]:
    """
    Score sentences against theme_queries using pre-computed sentence vectors.
    Only embeds the (tiny) query strings via OpenAI -- no sentence API call.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        return []
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)

        q_resp = client.embeddings.create(
            input=theme_queries, model="text-embedding-3-small"
        )
        query_vecs = np.array(
            [e.embedding for e in q_resp.data], dtype=np.float32
        )

        sim_matrix = cosine_similarity(query_vecs, sent_vecs)
        seen: set = set()
        results: List[SearchResult] = []

        for q_scores in sim_matrix:
            for idx in np.argsort(q_scores)[::-1][:top_k]:
                score = float(q_scores[idx])
                if score < min_score:
                    continue
                if idx in seen:
                    for r in results:
                        if r.excerpt == sentences[idx].text:
                            r.openai_score = max(r.openai_score, score)
                    continue
                seen.add(idx)
                results.append(SearchResult(
                    doc_filename=sentences[idx].doc_filename,
                    page=sentences[idx].page,
                    excerpt=sentences[idx].text,
                    theme=theme_label,
                    keyword_hit=False,
                    sbert_score=0.0,
                    openai_score=score,
                ))

        del sim_matrix
        logger.info(
            "OpenAI semantic search: %d results for theme '%s'",
            len(results), theme_label,
        )
        return results

    except Exception as exc:
        logger.warning("OpenAI semantic search failed for theme '%s': %s", theme_label, exc)
        return []


def openai_semantic_search(
    sentences,
    theme_queries: List[str],
    theme_label: str,
    top_k: int = 5,
    min_score: float = 0.25,
    fhash: Optional[str] = None,
) -> List[SearchResult]:
    """
    Legacy single-theme entry point kept for backward compatibility.
    Prefer calling openai_embed_sentences() once then
    openai_semantic_search_with_vecs() per theme.
    """
    sent_vecs = openai_embed_sentences(sentences, fhash=fhash)
    if sent_vecs is None:
        return []
    return openai_semantic_search_with_vecs(
        sentences, sent_vecs, theme_queries, theme_label,
        top_k=top_k, min_score=min_score,
    )


# ---------------------------------------------------------------------------
# CrossEncoder refinement  (Mode C only)
# ---------------------------------------------------------------------------

def refine_with_cross_encoder(
    results: List[SearchResult],
    theme_queries: List[str],
    model_name: str,
    valid_threshold: float,
    weak_threshold: float,
) -> List[SearchResult]:
    if not results:
        return results
    ce = _get_cross_encoder(model_name)
    query = theme_queries[0]
    scores = ce.predict([(query, r.excerpt) for r in results])
    for r, score in zip(results, scores):
        r.cross_encoder_score = float(score)
    return results


# ---------------------------------------------------------------------------
# Scoring & classification
# ---------------------------------------------------------------------------

def _score_and_classify(
    results: List[SearchResult],
    valid_threshold: float,
    weak_threshold: float,
    use_cross_encoder: bool,
    use_openai: bool,
) -> List[SearchResult]:
    for r in results:
        kw_boost = 0.05 if r.keyword_hit else 0.0

        if use_cross_encoder and r.cross_encoder_score > 0:
            # Cross-encoder score is most reliable
            r.final_score = round(
                0.6 * r.cross_encoder_score
                + 0.3 * max(r.sbert_score, r.openai_score)
                + kw_boost,
                4,
            )
        elif r.openai_score > 0:
            # OpenAI embedding cosine similarity
            r.final_score = round(r.openai_score + kw_boost, 4)
        elif r.sbert_score > 0:
            # SBERT cosine similarity
            r.final_score = round(r.sbert_score + kw_boost, 4)
        else:
            # Keyword-only hit
            r.final_score = round(0.40 + kw_boost, 4)

        r.validation_category = _classify(r.final_score, valid_threshold, weak_threshold)
    return results


def _classify(score: float, valid_threshold: float, weak_threshold: float) -> str:
    if score >= valid_threshold:
        return "VALID EVIDENCE"
    elif score >= weak_threshold:
        return "WEAK EVIDENCE"
    else:
        return "NOT RELEVANT"


def _attach_context(
    results: List[SearchResult],
    sentences,
    context_window: int = 2,
) -> None:
    """
    Populate result.context with the N sentences before and after the matched
    excerpt, taken from the same document and same page where possible.

    Mutates results in-place. No API calls, no extra cost.

    Args:
        results        : list of SearchResult (mutated)
        sentences      : full flat list of Sentence objects for this document
        context_window : number of sentences to include before and after (default 2)
    """
    if not results or not sentences:
        return

    # Build a lookup: excerpt text -> sentence_index (first match per doc/page)
    text_to_idx: dict = {}
    for s in sentences:
        if s.text not in text_to_idx:
            text_to_idx[s.text] = s.sentence_index

    n = len(sentences)
    for r in results:
        idx = text_to_idx.get(r.excerpt, -1)
        if idx < 0:
            r.context = r.excerpt  # fallback: just the excerpt itself
            continue

        start = max(0, idx - context_window)
        end = min(n, idx + context_window + 1)

        parts = [sentences[i].text for i in range(start, end)]
        r.context = " ".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_search_pipeline(
    sentences,
    taxonomy: dict,
    sbert_model_name: str,
    cross_encoder_name: str,
    top_k_sentences: int = 5,
    min_sbert_score: float = 0.25,
    use_openai: bool = False,
    use_cross_encoder: bool = False,
    use_sbert: bool = True,
    fhash: Optional[str] = None,
) -> List[SearchResult]:
    """
    Full search pipeline for sentences from ONE document.

    Mode is controlled by use_sbert / use_openai / use_cross_encoder flags
    (read from taxonomy.yaml by pipeline.py).

    In all modes:
      - Sentences are never re-encoded between themes
      - Embedding matrices are explicitly freed after use

    fhash (optional): SHA-256 hex digest of the source PDF bytes (first 16 chars).
      When provided, OpenAI sentence embeddings are cached to disk per theme,
      so subsequent runs skip the API call entirely for already-processed files.
    """
    valid_threshold = taxonomy.get("thresholds", {}).get("valid_match", 0.50)
    weak_threshold = taxonomy.get("thresholds", {}).get("weak_match", 0.35)
    themes = taxonomy.get("themes", [])

    if not sentences:
        return []

    doc_name = sentences[0].doc_filename if sentences else "unknown"

    # -- Mode A: OpenAI-only -------------------------------------------------
    if not use_sbert:
        logger.info(
            "Mode A (OpenAI-only): %d sentences from %s", len(sentences), doc_name
        )
        all_results: List[SearchResult] = []

        # Embed sentences ONCE for the whole document, reuse across all 7 themes.
        # This cuts OpenAI API calls from 7x down to 1x per document (~7x speedup).
        sent_vecs: Optional[np.ndarray] = None
        if use_openai:
            sent_vecs = openai_embed_sentences(sentences, fhash=fhash)
            if sent_vecs is None:
                logger.warning(
                    "openai_embed_sentences returned None for %s -- "
                    "falling back to keyword-only", doc_name
                )

        for theme in themes:
            label = theme["label"]
            keywords = theme.get("keywords", [])
            queries = theme.get("queries", [])

            kw_results = keyword_search(sentences, label, keywords)

            if use_openai and sent_vecs is not None:
                sem_results = openai_semantic_search_with_vecs(
                    sentences, sent_vecs, queries, label,
                    top_k=top_k_sentences, min_score=min_sbert_score,
                )
            else:
                sem_results = []

            excerpt_map: Dict[str, SearchResult] = {}
            for r in kw_results:
                key = f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"
                excerpt_map[key] = r
            for r in sem_results:
                key = f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"
                if key in excerpt_map:
                    excerpt_map[key].openai_score = max(
                        excerpt_map[key].openai_score, r.openai_score
                    )
                else:
                    excerpt_map[key] = r

            theme_results = _score_and_classify(
                list(excerpt_map.values()),
                valid_threshold, weak_threshold,
                use_cross_encoder=False, use_openai=use_openai,
            )
            all_results.extend(theme_results)

        del sent_vecs

        # ── Mode B / C: SBERT (+ optional CrossEncoder + optional OpenAI) ───────
    else:
        logger.info(
            "Mode B/C (SBERT): encoding %d sentences from %s once",
            len(sentences), doc_name,
        )
        sent_texts = [s.text for s in sentences]
        sentence_vecs = encode_sentences(sent_texts, sbert_model_name)
        all_results = []

        for theme in themes:
            label = theme["label"]
            keywords = theme.get("keywords", [])
            queries = theme.get("queries", [])

            kw_results = keyword_search(sentences, label, keywords)
            sem_results = semantic_search_with_vecs(
                sentences, sentence_vecs, queries, label,
                sbert_model_name, top_k=top_k_sentences, min_score=min_sbert_score,
            )

            excerpt_map = {}
            for r in kw_results:
                key = f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"
                excerpt_map[key] = r
            for r in sem_results:
                key = f"{r.doc_filename}||{r.page}||{r.excerpt[:60]}"
                if key in excerpt_map:
                    excerpt_map[key].sbert_score = max(
                        excerpt_map[key].sbert_score, r.sbert_score
                    )
                else:
                    excerpt_map[key] = r

            theme_results = list(excerpt_map.values())

            if use_cross_encoder:
                theme_results = refine_with_cross_encoder(
                    theme_results, queries, cross_encoder_name,
                    valid_threshold, weak_threshold,
                )

            theme_results = _score_and_classify(
                theme_results,
                valid_threshold, weak_threshold,
                use_cross_encoder=use_cross_encoder, use_openai=False,
            )
            all_results.extend(theme_results)

        del sentence_vecs
        gc.collect()

    # ── Global dedup: keep highest-scored result per excerpt ─────────────────
    global_map: Dict[str, SearchResult] = {}
    for r in all_results:
        key = f"{r.doc_filename}||{r.page}||{r.excerpt[:80]}"
        if key not in global_map or r.final_score > global_map[key].final_score:
            global_map[key] = r

    final = sorted(global_map.values(), key=lambda x: x.final_score, reverse=True)

    # Attach surrounding context sentences (2 before + 2 after) to each result.
    # Uses only the in-memory sentence list — no API calls, no extra cost.
    context_window = taxonomy.get("search", {}).get("context_window", 2)
    _attach_context(final, sentences, context_window=context_window)

    logger.info("Search complete for %s — %d evidence excerpts", doc_name, len(final))
    return final
