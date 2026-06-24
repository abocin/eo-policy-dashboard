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

def openai_semantic_search(
    sentences,
    theme_queries: List[str],
    theme_label: str,
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[SearchResult]:
    """
    Embed theme queries + all sentences via OpenAI text-embedding-3-small,
    return top_k most similar sentences per query.
    No torch required — pure HTTP + numpy.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        logger.warning("OpenAI semantic search requested but OPENAI_API_KEY not set")
        return []

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)

        # Embed queries
        q_resp = client.embeddings.create(
            input=theme_queries, model="text-embedding-3-small"
        )
        query_vecs = np.array(
            [e.embedding for e in q_resp.data], dtype=np.float32
        )

        # Embed sentences in batches of 500 (OpenAI max is 2048 inputs)
        sent_texts = [s.text for s in sentences]
        all_sent_vecs = []
        OPENAI_BATCH = 500
        for start in range(0, len(sent_texts), OPENAI_BATCH):
            batch = sent_texts[start: start + OPENAI_BATCH]
            resp = client.embeddings.create(
                input=batch, model="text-embedding-3-small"
            )
            all_sent_vecs.extend([e.embedding for e in resp.data])

        sent_vecs = np.array(all_sent_vecs, dtype=np.float32)
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

        del sent_vecs, sim_matrix
        logger.info(
            "OpenAI semantic search: %d results for theme '%s'",
            len(results), theme_label,
        )
        return results

    except Exception as exc:
        logger.warning("OpenAI semantic search failed for theme '%s': %s", theme_label, exc)
        return []


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
) -> List[SearchResult]:
    """
    Full search pipeline for sentences from ONE document.

    Mode is controlled by use_sbert / use_openai / use_cross_encoder flags
    (read from taxonomy.yaml by pipeline.py).

    In all modes:
      - Sentences are never re-encoded between themes
      - Embedding matrices are explicitly freed after use
    """
    valid_threshold = taxonomy.get("thresholds", {}).get("valid_match", 0.50)
    weak_threshold = taxonomy.get("thresholds", {}).get("weak_match", 0.35)
    themes = taxonomy.get("themes", [])

    if not sentences:
        return []

    doc_name = sentences[0].doc_filename if sentences else "unknown"

    # ── Mode A: OpenAI-only ─────────────────────────────────────────────────
    if not use_sbert:
        logger.info(
            "Mode A (OpenAI-only): %d sentences from %s", len(sentences), doc_name
        )
        all_results: List[SearchResult] = []

        for theme in themes:
            label = theme["label"]
            keywords = theme.get("keywords", [])
            queries = theme.get("queries", [])

            kw_results = keyword_search(sentences, label, keywords)

            sem_results = openai_semantic_search(
                sentences, queries, label,
                top_k=top_k_sentences, min_score=min_sbert_score,
            ) if use_openai else []

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
    logger.info("Search complete for %s — %d evidence excerpts", doc_name, len(final))
    return final
