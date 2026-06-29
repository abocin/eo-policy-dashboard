"""
core/eo_relevance_filter.py
---------------------------
Stage-2 EO Capacity-Building Relevance Filter.

After the Stage-1 taxonomy search produces evidence excerpts, this module
re-scores every excerpt against a set of "EO capacity building" reference
sentences using OpenAI embeddings (text-embedding-3-small).

The resulting `eo_relevance_score` (0.0–1.0) answers the question:
  "Does this excerpt discuss acquiring / building / using EO skills or
   capacities — not just mention EO in passing?"

Results below the user-chosen threshold can be hidden in the UI via the
"EO Capacity Filter" slider in results_table.py.

Design goals:
  - Zero RAM cost when OPENAI_API_KEY is absent (graceful no-op)
  - Re-uses existing sentence-level OpenAI embedding cache where possible
  - Reference sentences are stored in config/taxonomy.yaml under the
    `eo_relevance` key so they can be edited without code changes
  - Scores are cached alongside results in last_results.json
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default reference sentences (fallback if not in taxonomy.yaml)
# ---------------------------------------------------------------------------

DEFAULT_REFERENCE_SENTENCES: List[str] = [
    # English
    "training professionals to use Earth Observation data",
    "capacity building for Earth Observation downstream applications",
    "skills development for satellite data analysis",
    "uptake of Earth Observation data by end users",
    "educating practitioners in remote sensing and geospatial analysis",
    "building institutional capacity to exploit EO data",
    "policy support for Earth Observation data literacy",
    "workforce training in satellite imagery interpretation",
    "investing in human capital for EO downstream services",
    "fostering competencies in geospatial data and Earth Observation",
    # Dutch
    "opleiding voor het gebruik van aardobservatie data",
    "capaciteitsopbouw voor aardobservatie toepassingen",
    "vaardigheidsontwikkeling voor satellietgegevens analyse",
    # Greek
    "κατάρτιση για χρήση δεδομένων παρατήρησης γης",
    "ανάπτυξη ικανοτήτων για εφαρμογές παρατήρησης γης",
    # Portuguese
    "formação em observação da Terra e dados de satélite",
    "capacitação para aplicações de observação da Terra",
    # Italian
    "formazione nell'uso dei dati di osservazione della Terra",
    "sviluppo delle competenze per applicazioni EO downstream",
]


# ---------------------------------------------------------------------------
# Embedding helper (reuses OpenAI client, no torch dependency)
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str], api_key: str) -> Optional[np.ndarray]:
    """Embed a list of texts via OpenAI text-embedding-3-small."""
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        BATCH = 500
        all_vecs: List[List[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start: start + BATCH]
            resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
            all_vecs.extend([e.embedding for e in resp.data])
        return np.array(all_vecs, dtype=np.float32)
    except Exception as exc:
        logger.warning("EO relevance filter: embedding failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_eo_relevance(
    results,                       # List[SearchResult]
    taxonomy: dict,
    fhash: Optional[str] = None,   # reserved for future per-doc caching
) -> None:
    """
    Compute and set `eo_relevance_score` on every SearchResult in-place.

    If OPENAI_API_KEY is absent or the API call fails, all scores remain 0.0
    and the filter in the UI simply becomes a no-op (showing everything).

    Args:
        results  : list of SearchResult objects (mutated in-place)
        taxonomy : loaded taxonomy dict (may contain eo_relevance.sentences)
        fhash    : unused for now; kept for API consistency
    """
    if not results:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "EO relevance filter: OPENAI_API_KEY not set — skipping (all scores = 0.0)"
        )
        return

    # ---- Load reference sentences from taxonomy or fall back to defaults ---
    eo_cfg = taxonomy.get("eo_relevance", {})
    ref_sentences: List[str] = eo_cfg.get("sentences", DEFAULT_REFERENCE_SENTENCES)
    if not ref_sentences:
        ref_sentences = DEFAULT_REFERENCE_SENTENCES

    logger.info(
        "EO relevance filter: scoring %d excerpts against %d reference sentences",
        len(results), len(ref_sentences),
    )

    # ---- Embed reference sentences (small, ~20 sentences, cheap) -----------
    ref_vecs = _embed_texts(ref_sentences, api_key)
    if ref_vecs is None:
        return

    # ---- Embed all unique excerpts (batch, deduplicated) -------------------
    unique_excerpts: List[str] = list(dict.fromkeys(r.excerpt for r in results))

    excerpt_vecs = _embed_texts(unique_excerpts, api_key)
    if excerpt_vecs is None:
        return

    # Map excerpt text → row index in excerpt_vecs
    excerpt_index = {text: i for i, text in enumerate(unique_excerpts)}

    # ---- Cosine similarity: excerpt_vecs (N x D) vs ref_vecs (M x D) ------
    # sim_matrix shape: (N, M)
    sim_matrix = cosine_similarity(excerpt_vecs, ref_vecs)  # shape (N, M)

    # Score for each excerpt = max similarity across all reference sentences
    # This means "closest conceptual match to any EO capacity-building sentence"
    max_scores = sim_matrix.max(axis=1)  # shape (N,)

    # ---- Write scores back onto results ------------------------------------
    for r in results:
        idx = excerpt_index.get(r.excerpt, -1)
        if idx >= 0:
            r.eo_relevance_score = round(float(max_scores[idx]), 4)

    scored = sum(1 for r in results if r.eo_relevance_score > 0)
    avg = (
        sum(r.eo_relevance_score for r in results) / len(results)
        if results else 0.0
    )
    logger.info(
        "EO relevance filter complete — %d/%d scored, avg score %.3f",
        scored, len(results), avg,
    )
