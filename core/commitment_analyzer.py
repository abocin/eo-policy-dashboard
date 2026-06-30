"""
core/commitment_analyzer.py
---------------------------
Analyses the strength of policy commitment in each excerpt.

Distinguishes between:
  BINDING      — "shall", "will", "must", "fund", "allocate"  (score 0.85–1.0)
  STRONG       — "commit", "ensure", "require", "establish"    (score 0.65–0.85)
  MODERATE     — "support", "promote", "develop", "invest"     (score 0.40–0.65)
  ASPIRATIONAL — "may", "could", "should", "encourage", "aim"  (score 0.0–0.40)

Also detects presence of actionability signals:
  - Budget mentions (€, million, billion, funding, budget)
  - KPI / indicator mentions
  - Timeline mentions (by 2025, by 2030, target date)
  - Governance mentions (responsible authority, designated body)

These boost the commitment score so a sentence that says
"will allocate €50 million by 2027" scores much higher than
"should consider encouraging".

No external API calls. Pure regex. Runs on any cached result set.
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Lexicons  (ordered: most binding → least binding)
# ---------------------------------------------------------------------------

_BINDING = re.compile(
    r"\b(shall|will\s+be|must|is\s+required\s+to|are\s+required\s+to"
    r"|legally\s+bound|mandated|obligat\w+|allocat\w+|fund\w+|financ\w+"
    r"|budget\w+|appropriat\w+|dedicate\w+|commit\w+\s+to)\b",
    re.IGNORECASE,
)

_STRONG = re.compile(
    r"\b(will|ensure\w*|require\w*|establish\w*|implement\w*|set\s+up"
    r"|create\w*|launch\w*|deploy\w*|invest\w*|priorit\w+|strengthen\w*"
    r"|mainst?ream\w*|integrate\w*|adopt\w*)\b",
    re.IGNORECASE,
)

_MODERATE = re.compile(
    r"\b(support\w*|promot\w*|develop\w*|foster\w*|facilitat\w*"
    r"|stimulat\w*|enhanc\w*|improv\w*|contribut\w*|aim\w*\s+to"
    r"|intend\w*|seek\w*\s+to|endeavour\w*|work\s+towards?)\b",
    re.IGNORECASE,
)

_ASPIRATIONAL = re.compile(
    r"\b(may|might|could|should|encourage\w*|consider\w*|explore\w*"
    r"|review\w*|assess\w*|monitor\w*|raise\s+awareness|sensiti\w+"
    r"|help\w*|assist\w*|guide\w*|recommend\w*|suggest\w*)\b",
    re.IGNORECASE,
)

# Actionability boosters
_BUDGET = re.compile(
    r"(€\s?\d|£\s?\d|\$\s?\d|\d+\s*(million|billion|m€|bn€|MEUR|BEUR)"
    r"|budget|appropriation|funding\s+of|co-financ\w+)",
    re.IGNORECASE,
)

_KPI = re.compile(
    r"\b(target\w*|indicator\w*|benchmark\w*|KPI\w*|metric\w*"
    r"|measur\w+|track\w*|monitor\w*|report\w*\s+on|evaluat\w+)\b",
    re.IGNORECASE,
)

_TIMELINE = re.compile(
    r"\b(by\s+20\d{2}|by\s+(end\s+of|the\s+end\s+of)"
    r"|deadline|within\s+\d+|no\s+later\s+than|roadmap|timetable"
    r"|phase\s+\d+|milestone\w*)\b",
    re.IGNORECASE,
)

_GOVERNANCE = re.compile(
    r"\b(designated\s+\w+|responsible\s+(authority|body|ministry)"
    r"|competent\s+authority|national\s+authority|focal\s+point"
    r"|steering\s+(committee|group)|governing\s+body)\b",
    re.IGNORECASE,
)


def _count(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def analyse_commitment(text: str) -> tuple[float, str]:
    """
    Returns (commitment_score, commitment_level) for a single text excerpt.

    Score is 0.0–1.0. Level is one of:
      BINDING | STRONG | MODERATE | ASPIRATIONAL
    """
    binding_n     = _count(_BINDING, text)
    strong_n      = _count(_STRONG, text)
    moderate_n    = _count(_MODERATE, text)
    aspirational_n = _count(_ASPIRATIONAL, text)

    # Base score from modal/commitment verb tier
    total = binding_n + strong_n + moderate_n + aspirational_n
    if total == 0:
        base = 0.15  # no commitment signals at all → low aspirational
    else:
        # Weighted average of tiers: binding=1.0, strong=0.75, mod=0.45, asp=0.15
        base = (
            binding_n * 1.0
            + strong_n * 0.75
            + moderate_n * 0.45
            + aspirational_n * 0.15
        ) / total

    # Actionability boosters (each adds up to +0.08, capped)
    booster = 0.0
    if _count(_BUDGET, text) > 0:
        booster += 0.08
    if _count(_KPI, text) > 0:
        booster += 0.05
    if _count(_TIMELINE, text) > 0:
        booster += 0.06
    if _count(_GOVERNANCE, text) > 0:
        booster += 0.04

    score = round(min(1.0, base + booster), 4)

    if score >= 0.80:
        level = "BINDING"
    elif score >= 0.60:
        level = "STRONG"
    elif score >= 0.35:
        level = "MODERATE"
    else:
        level = "ASPIRATIONAL"

    return score, level


def score_commitment(results) -> None:
    """
    Set commitment_score and commitment_level on every SearchResult in-place.
    Uses the context field if available (more text = better signal),
    falls back to excerpt only.
    No API calls. Pure regex. O(n * text_length).
    """
    for r in results:
        text = getattr(r, "context", "") or r.excerpt
        score, level = analyse_commitment(text)
        r.commitment_score = score
        r.commitment_level = level
