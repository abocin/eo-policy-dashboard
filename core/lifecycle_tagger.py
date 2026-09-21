"""
core/lifecycle_tagger.py
------------------------
Tags each SearchResult excerpt with the EO capacity-building lifecycle stage
it most strongly addresses.

Lifecycle stages (from the EO capacity building framework):

  1. Awareness       — promote, disseminate, raise awareness, outreach
  2. Education       — curricula, university, degree, academic, school
  3. Training        — courses, workshops, bootcamp, certification, MOOC
  4. Skills          — upskilling, reskilling, workforce, competence, talent
  5. Innovation      — R&D, living lab, testbed, prototype, pilot, research
  6. Entrepreneurship— startup, spin-off, venture, incubator, accelerator
  7. Adoption        — uptake, deployment, integration, operationalise
  8. Sustainability  — long-term, ecosystem, governance, institutionalise

Each excerpt is classified to the stage with the most keyword hits.
Ties are broken by stage order (earlier stage wins).

Taxonomy-driven: stage keywords are loaded from taxonomy.yaml under
`lifecycle.stages` if present, falling back to built-in defaults.
This means stages and keywords can be edited without code changes.

No API calls. Pure regex matching.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default stage definitions  (used if not in taxonomy.yaml)
# ---------------------------------------------------------------------------

DEFAULT_STAGES: List[Dict] = [
    {
        "stage": "Awareness",
        "keywords": [
            "awareness", "disseminat", "outreach", "promot",
            "communicat", "publicis", "inform", "sensitiz", "sensitise",
            "visibility", "advocacy", "campaign", "public engagement",
        ],
    },
    {
        "stage": "Education",
        "keywords": [
            "educat", "curricul", "universit", "degree", "academic",
            "school", "higher education", "bachelor", "master", "PhD",
            "doctorate", "faculty", "course credit", "study programme",
            "learning pathway", "formal education",
        ],
    },
    {
        "stage": "Training",
        "keywords": [
            "train", "course", "workshop", "bootcamp", "certificat",
            "MOOC", "e-learning", "online learning", "short course",
            "professional development", "instructor", "practical skills",
            "vocational", "apprentice",
        ],
    },
    {
        "stage": "Skills",
        "keywords": [
            "skill", "upskill", "reskill", "workforce", "competenc",
            "talent", "human capital", "labour", "capacity building",
            "capability", "proficienc", "expert", "digital literacy",
            "data literacy", "geospatial skills",
        ],
    },
    {
        "stage": "Innovation",
        "keywords": [
            "innovat", "research", "R&D", "living lab", "testbed",
            "prototype", "pilot", "demonstrat", "experiment", "co-creat",
            "open innovation", "technology transfer", "feasibility",
            "proof of concept",
        ],
    },
    {
        "stage": "Entrepreneurship",
        "keywords": [
            "startup", "start-up", "spin-off", "venture", "incubat",
            "accelerat", "entrepreneur", "business creation",
            "SME", "scale-up", "new business", "commercialis",
            "go-to-market", "market entry", "CASSINI",
        ],
    },
    {
        "stage": "Adoption",
        "keywords": [
            "uptake", "deploy", "integrat", "operationa", "implement",
            "mainstream", "adopt", "utiliz", "utilise", "use of EO",
            "apply", "embed", "roll out", "procure",
        ],
    },
    {
        "stage": "Sustainability",
        "keywords": [
            "sustain", "long-term", "ecosystem", "governance",
            "institutionali", "permanent", "embed in policy",
            "funding mechanism", "systemic", "structural change",
            "national strategy", "monitoring framework",
        ],
    },
]


#: Accepted spellings for a stage's display name. Custom taxonomies written
#: against the `themes:` section naturally use `label:`, which is the house
#: style, so requiring `stage:` here crashed the whole pipeline on a valid file.
_STAGE_NAME_KEYS = ("stage", "label", "name", "title")


def _stage_name(entry: Dict) -> str | None:
    """Return a stage's display name, accepting any known key spelling."""
    for key in _STAGE_NAME_KEYS:
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _build_patterns(stages: List[Dict]) -> List[Tuple[str, re.Pattern]]:
    """Compile a keyword regex per lifecycle stage.

    Skips malformed entries with a warning rather than raising: this runs at the
    very END of the pipeline, after all the (paid) embedding calls, so a typo in
    a taxonomy file must not discard an entire completed analysis.
    """
    compiled: List[Tuple[str, re.Pattern]] = []

    for i, s in enumerate(stages):
        if not isinstance(s, dict):
            logger.warning("Lifecycle stage #%d is not a mapping — skipped.", i + 1)
            continue

        name = _stage_name(s)
        if not name:
            logger.warning(
                "Lifecycle stage #%d has no name (expected one of %s) — skipped.",
                i + 1, ", ".join(_STAGE_NAME_KEYS),
            )
            continue

        kws = [str(k) for k in (s.get("keywords") or []) if str(k).strip()]
        if not kws:
            logger.warning("Lifecycle stage %r has no keywords — skipped.", name)
            continue

        pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in kws) + r")\w*", re.IGNORECASE
        )
        compiled.append((name, pattern))

    return compiled


def tag_lifecycle_stage(
    text: str,
    patterns: List[Tuple[str, re.Pattern]],
    default: str = "",
) -> str:
    """
    Return the lifecycle stage label with the most keyword matches.

    With no match, falls back to ``default`` — the caller passes the FIRST stage
    of the active taxonomy. Previously this was hard-coded to "Skills", which
    silently mislabelled every unmatched excerpt with a stage that does not
    exist in a custom taxonomy.
    """
    best_stage = default
    best_count = 0

    for stage_name, pattern in patterns:
        count = len(pattern.findall(text))
        if count > best_count:
            best_count = count
            best_stage = stage_name

    return best_stage


def tag_lifecycle(results, taxonomy: dict) -> None:
    """
    Set lifecycle_stage on every SearchResult in-place.

    Loads stage definitions from taxonomy.yaml `lifecycle.stages` if present,
    falls back to DEFAULT_STAGES.

    Uses the context field (more text = better classification),
    falls back to excerpt only.
    No API calls.
    """
    lc_cfg = taxonomy.get("lifecycle") or {}
    if not isinstance(lc_cfg, dict):
        logger.warning("taxonomy 'lifecycle' is not a mapping — using defaults.")
        lc_cfg = {}

    stage_defs = lc_cfg.get("stages")
    if not isinstance(stage_defs, list) or not stage_defs:
        stage_defs = DEFAULT_STAGES

    patterns = _build_patterns(stage_defs)
    if not patterns:
        logger.warning(
            "No usable lifecycle stages in the taxonomy — falling back to the "
            "%d built-in stages.", len(DEFAULT_STAGES),
        )
        patterns = _build_patterns(DEFAULT_STAGES)

    # Unmatched excerpts get the first stage of whichever set is in use.
    default_stage = patterns[0][0] if patterns else ""
    logger.info(
        "Lifecycle tagging with %d stage(s): %s",
        len(patterns), ", ".join(n for n, _ in patterns),
    )

    for r in results:
        text = getattr(r, "context", "") or r.excerpt
        r.lifecycle_stage = tag_lifecycle_stage(text, patterns, default_stage)
