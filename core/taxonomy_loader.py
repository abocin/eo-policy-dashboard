"""
core/taxonomy_loader.py
-----------------------
Loads and validates the EO skills taxonomy YAML.
Falls back to a built-in minimal taxonomy if the file is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent / "config" / "taxonomy.yaml"


def load_taxonomy(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load taxonomy from a YAML file.  Returns a dict with keys:
      - themes: list of theme dicts
      - thresholds: dict with valid_match and weak_match
      - search: dict with model names and search parameters
    """
    target = Path(path) if path else DEFAULT_TAXONOMY_PATH

    try:
        import yaml  # type: ignore

        with open(target, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        _validate(data)
        logger.info("Loaded taxonomy from %s (%d themes)", target, len(data.get("themes", [])))
        return data

    except FileNotFoundError:
        logger.warning("Taxonomy file not found at %s — using minimal defaults", target)
        return _minimal_fallback()

    except Exception as exc:
        logger.error("Failed to load taxonomy: %s — using minimal defaults", exc)
        return _minimal_fallback()


def _validate(data: Dict[str, Any]) -> None:
    """Raise ValueError if taxonomy is missing required keys."""
    if "themes" not in data or not isinstance(data["themes"], list):
        raise ValueError("Taxonomy must have a 'themes' list")
    for theme in data["themes"]:
        if "label" not in theme:
            raise ValueError(f"Each theme must have a 'label'. Found: {theme}")
        if "queries" not in theme or not theme["queries"]:
            raise ValueError(f"Theme '{theme['label']}' must have at least one query")


def _minimal_fallback() -> Dict[str, Any]:
    return {
        "themes": [
            {
                "label": "EO Downstream Skills",
                "keywords": ["earth observation", "EO", "satellite", "Copernicus", "remote sensing"],
                "queries": [
                    "skills needed for earth observation and satellite data",
                    "earth observation workforce training and capacity building",
                ],
                "weight": 1.0,
            }
        ],
        "thresholds": {"valid_match": 0.50, "weak_match": 0.35},
        "search": {
            "sbert_model": "all-MiniLM-L6-v2",
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_sentences": 5,
            "chunk_size": 400,
            "chunk_overlap": 80,
        },
    }


def taxonomy_to_display(taxonomy: Dict[str, Any]) -> Dict[str, list]:
    """Returns a simplified dict for display: {label: [keywords]}."""
    return {
        t["label"]: t.get("keywords", [])
        for t in taxonomy.get("themes", [])
    }
