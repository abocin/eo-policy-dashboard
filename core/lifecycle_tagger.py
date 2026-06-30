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

import re
from typing import Dict, List, Optional, Tuple


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


def _build_patterns(stages: List[Dict]) -> List[Tuple[str, re.Pattern]]:
    """Compile regex patterns for each stage."""
    compiled = []
    for s in stages:
        kws = [re.escape(k) for k in s["keywords"]]
        pattern = re.compile(r"\b(" + "|".join(kws) + r")\w*", re.IGNORECASE)
        compiled.append((s["stage"], pattern))
    return compiled


def tag_lifecycle_stage(
    text: str,
    patterns: List[Tuple[str, re.Pattern]],
) -> str:
    """
    Return the lifecycle stage label with the most keyword matches.
    Falls back to 'Skills' (most common) if nothing matches.
    """
    best_stage = "Skills"
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
    lc_cfg = taxonomy.get("lifecycle", {})
    stage_defs = lc_cfg.get("stages", DEFAULT_STAGES)
    patterns = _build_patterns(stage_defs)

    for r in results:
        text = getattr(r, "context", "") or r.excerpt
        r.lifecycle_stage = tag_lifecycle_stage(text, patterns)
