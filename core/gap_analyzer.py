"""
core/gap_analyzer.py
---------------------
Gap analysis: which EO capacity-building themes are covered by each policy,
and which are missing or weakly covered?

Operates on a completed SearchResult list (post-pipeline).
No API calls. Pure aggregation.

Outputs (all as plain dicts, JSON-serialisable):

  per_doc_coverage(results, taxonomy)
    → Dict[doc_filename, Dict[theme_label, CoverageRecord]]

  corpus_gap_report(results, taxonomy)
    → CorpusGapReport

  CoverageRecord
    hit_count        : int   — number of evidence excerpts for this theme
    max_score        : float — highest final_score for this theme in this doc
    avg_score        : float — average final_score
    coverage_level   : str   — STRONG | MODERATE | WEAK | MISSING
    keyword_hits     : int   — keyword-triggered excerpts
    semantic_hits    : int   — semantic-only excerpts

  CorpusGapReport
    total_themes     : int
    covered_themes   : int   — at least one doc has MODERATE+ coverage
    gap_themes       : list  — theme labels with no or only WEAK corpus-wide coverage
    gap_pct          : float — % of themes that are gaps
    per_theme_docs   : Dict[theme, list of docs that cover it]
    theme_scores     : Dict[theme, corpus-wide avg max_score]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoverageRecord:
    hit_count: int = 0
    max_score: float = 0.0
    avg_score: float = 0.0
    coverage_level: str = "MISSING"
    keyword_hits: int = 0
    semantic_hits: int = 0


@dataclass
class CorpusGapReport:
    total_themes: int = 0
    covered_themes: int = 0
    gap_themes: List[str] = field(default_factory=list)
    partial_themes: List[str] = field(default_factory=list)
    gap_pct: float = 0.0
    per_theme_docs: Dict[str, List[str]] = field(default_factory=dict)
    theme_scores: Dict[str, float] = field(default_factory=dict)
    theme_hit_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coverage thresholds
# ---------------------------------------------------------------------------

def _coverage_level(hit_count: int, max_score: float) -> str:
    if hit_count == 0:
        return "MISSING"
    if max_score >= 0.65 or hit_count >= 5:
        return "STRONG"
    if max_score >= 0.45 or hit_count >= 2:
        return "MODERATE"
    return "WEAK"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def per_doc_coverage(
    results,
    taxonomy: Dict[str, Any],
) -> Dict[str, Dict[str, CoverageRecord]]:
    """
    Returns a nested dict:
      { doc_filename: { theme_label: CoverageRecord } }

    Every (doc, theme) pair is included even when hit_count == 0,
    so callers can immediately see which themes are missing per document.
    """
    themes = [t["label"] for t in taxonomy.get("themes", [])]
    docs = list(dict.fromkeys(r.doc_filename for r in results))

    # Initialise all cells to MISSING
    coverage: Dict[str, Dict[str, CoverageRecord]] = {
        doc: {theme: CoverageRecord() for theme in themes}
        for doc in docs
    }

    for r in results:
        if r.doc_filename not in coverage:
            continue
        if r.theme not in coverage[r.doc_filename]:
            continue
        rec = coverage[r.doc_filename][r.theme]
        rec.hit_count += 1
        rec.max_score = max(rec.max_score, r.final_score)
        # Running avg approximation (will be corrected below)
        rec.avg_score += r.final_score
        if r.keyword_hit:
            rec.keyword_hits += 1
        else:
            rec.semantic_hits += 1

    # Finalise avg_score and coverage_level
    for doc_rec in coverage.values():
        for rec in doc_rec.values():
            if rec.hit_count > 0:
                rec.avg_score = round(rec.avg_score / rec.hit_count, 4)
                rec.max_score = round(rec.max_score, 4)
            rec.coverage_level = _coverage_level(rec.hit_count, rec.max_score)

    return coverage


def corpus_gap_report(
    results,
    taxonomy: Dict[str, Any],
) -> CorpusGapReport:
    """
    Aggregates per-doc coverage into a corpus-wide gap report.

    A theme is:
      COVERED  — at least one document has STRONG or MODERATE coverage
      PARTIAL  — at least one document has WEAK coverage but none MODERATE+
      GAP      — no document covers this theme at all
    """
    themes = [t["label"] for t in taxonomy.get("themes", [])]
    coverage = per_doc_coverage(results, taxonomy)

    report = CorpusGapReport(total_themes=len(themes))

    per_theme_docs: Dict[str, List[str]] = {t: [] for t in themes}
    theme_scores: Dict[str, List[float]] = {t: [] for t in themes}
    theme_hits: Dict[str, int] = {t: 0 for t in themes}

    for doc, doc_rec in coverage.items():
        for theme, rec in doc_rec.items():
            if rec.coverage_level in ("STRONG", "MODERATE"):
                per_theme_docs[theme].append(doc)
                theme_scores[theme].append(rec.max_score)
                theme_hits[theme] += rec.hit_count
            elif rec.coverage_level == "WEAK":
                theme_scores[theme].append(rec.max_score)
                theme_hits[theme] += rec.hit_count

    gap_themes: List[str] = []
    partial_themes: List[str] = []
    covered_count = 0

    for theme in themes:
        if per_theme_docs[theme]:  # at least one MODERATE+ doc
            covered_count += 1
        elif theme_hits[theme] > 0:  # only WEAK hits
            partial_themes.append(theme)
        else:  # truly absent
            gap_themes.append(theme)

    report.covered_themes = covered_count
    report.gap_themes = gap_themes
    report.partial_themes = partial_themes
    report.gap_pct = round(
        (len(gap_themes) / len(themes) * 100) if themes else 0.0, 1
    )
    report.per_theme_docs = per_theme_docs
    report.theme_scores = {
        t: round(sum(v) / len(v), 4) if v else 0.0
        for t, v in theme_scores.items()
    }
    report.theme_hit_counts = theme_hits

    return report
