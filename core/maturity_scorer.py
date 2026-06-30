"""
core/maturity_scorer.py
-----------------------
Scores each policy document on the EO Maturity Scale (1–6).

  Level 1 — Absent       EO never mentioned; no relevant evidence
  Level 2 — Indirect     EO indirectly implied (digital/spatial/satellite context)
  Level 3 — Recognised   EO explicitly mentioned but no specific actions
  Level 4 — Targeted     Specific EO actions or objectives defined
  Level 5 — Funded       Explicit budget, funding mechanism, or financial commitment
  Level 6 — Monitored    KPIs, indicators, targets, monitoring framework for EO

Scoring logic:
  - Starts at Level 1.
  - Each successive level requires meeting the criteria of all lower levels.
  - Evidence from SearchResult list drives the assessment — no re-parsing.

Returns a dict:
  { doc_filename: MaturityRecord }

MaturityRecord fields:
  level           int     1–6
  label           str     e.g. "Targeted"
  evidence_count  int     number of relevant excerpts
  max_score       float   highest final_score across excerpts
  avg_score       float
  funded          bool    budget/funding signal detected
  monitored       bool    KPI/indicator signal detected
  notes           str     brief explanation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

# Level 3 — explicit EO/satellite/Copernicus mention
_EO_EXPLICIT = re.compile(
    r"\b(earth\s+observation|EO\b|satellite\s+(data|imagery|services?)"
    r"|Copernicus|remote\s+sensing|geospatial|GNSS|Galileo"
    r"|space\s+(data|services?|application))\b",
    re.IGNORECASE,
)

# Level 4 — specific actions: verbs + EO context
_TARGETED = re.compile(
    r"\b(develop|implement|establish|deploy|launch|creat|design|build)"
    r"\w*\b",
    re.IGNORECASE,
)

# Level 5 — funding / budget signals
_FUNDED = re.compile(
    r"(€\s?\d|£\s?\d|\$\s?\d|\d+\s*(million|billion|M€|Bn|MEUR)"
    r"|budget|fund\w+|financ\w+|co-financ\w+|grant|subsid\w+"
    r"|appropriat\w+|allocat\w+|invest\w+\s+in)",
    re.IGNORECASE,
)

# Level 6 — monitoring / KPI signals
_MONITORED = re.compile(
    r"\b(KPI|indicator\w*|target\w*|benchmark\w*|measur\w+"
    r"|track\w*|monitor\w*|evaluat\w+|report\w*\s+on"
    r"|review\w*\s+progress|annual\s+report|progress\s+report)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Maturity level definitions
# ---------------------------------------------------------------------------

_LEVELS = {
    1: "Absent",
    2: "Indirect",
    3: "Recognised",
    4: "Targeted",
    5: "Funded",
    6: "Monitored",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MaturityRecord:
    level: int = 1
    label: str = "Absent"
    evidence_count: int = 0
    max_score: float = 0.0
    avg_score: float = 0.0
    funded: bool = False
    monitored: bool = False
    targeted: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_maturity(
    results,
    taxonomy: Dict[str, Any],
) -> Dict[str, MaturityRecord]:
    """
    Returns { doc_filename: MaturityRecord } for every document in results.

    Documents not present in results still get a Level 1 record if
    passed via `all_docs` — but since we only have results here,
    docs with zero matches simply don't appear (callers handle this).
    """
    # Group results by document
    by_doc: Dict[str, list] = {}
    for r in results:
        by_doc.setdefault(r.doc_filename, []).append(r)

    records: Dict[str, MaturityRecord] = {}

    for doc, doc_results in by_doc.items():
        rec = MaturityRecord()
        rec.evidence_count = len(doc_results)

        scores = [r.final_score for r in doc_results]
        rec.max_score = round(max(scores), 4) if scores else 0.0
        rec.avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Combine all text for signal detection
        all_text = " ".join(
            (getattr(r, "context", "") or r.excerpt) for r in doc_results
        )

        has_eo_explicit = bool(_EO_EXPLICIT.search(all_text))
        has_targeted    = bool(_TARGETED.search(all_text)) and rec.evidence_count >= 2
        has_funded      = bool(_FUNDED.search(all_text))
        has_monitored   = bool(_MONITORED.search(all_text))

        rec.funded    = has_funded
        rec.monitored = has_monitored
        rec.targeted  = has_targeted

        # Determine level (cumulative)
        if rec.evidence_count == 0:
            level = 1
            notes = "No evidence excerpts found."
        elif not has_eo_explicit and rec.max_score < 0.40:
            level = 2
            notes = "EO indirectly implied; no explicit mention detected."
        elif not has_targeted:
            level = 3
            notes = "EO recognised but no specific actions identified."
        elif not has_funded:
            level = 4
            notes = "Specific EO actions identified; no funding signal found."
        elif not has_monitored:
            level = 5
            notes = "Funding or budget signal found; no KPI/monitoring found."
        else:
            level = 6
            notes = "Funding and monitoring/KPI signals both present."

        rec.level = level
        rec.label = _LEVELS[level]
        rec.notes = notes
        records[doc] = rec

    return records


def maturity_label(level: int) -> str:
    return _LEVELS.get(level, "Unknown")
