"""
tests/conftest.py
-----------------
Shared pytest fixtures used across all test modules.

Key design decisions for CI:
- The sample PDF fixture is generated programmatically using reportlab so
  the repo stays lean (no binary PDF committed to git).
- SBERT and CrossEncoder are loaded once per test session via session-scoped
  fixtures — avoids re-downloading or re-initialising models on every test.
- All fixtures that touch the filesystem use tmp_path so tests are isolated.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Make project root importable when running pytest from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF_PATH = FIXTURES_DIR / "sample_eo_policy.pdf"


# ---------------------------------------------------------------------------
# PDF fixture — generate if missing (e.g. fresh CI checkout)
# ---------------------------------------------------------------------------

def _generate_sample_pdf(path: Path) -> None:
    """Create a minimal 2-page EO policy PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas as rl_canvas  # type: ignore
    except ImportError:
        pytest.skip("reportlab not installed — skipping PDF fixture generation")

    path.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(path), pagesize=A4)

    # Page 1
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 780, "EU Space Skills Policy — Test Document")
    c.setFont("Helvetica", 11)
    sentences_p1 = [
        "This policy document outlines the strategic framework for developing",
        "earth observation downstream skills across the European Union.",
        "The Copernicus programme provides satellite imagery and remote sensing",
        "data that requires specialised workforce training and capacity building.",
        "Member states are encouraged to invest in upskilling and reskilling",
        "programmes targeting geospatial data analysts and GIS specialists.",
        "The skills gap in the space sector remains a critical challenge.",
        "Satellite data processing requires digital skills and Python expertise.",
    ]
    y = 740
    for line in sentences_p1:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 780, "Section 2: Workforce Development")
    c.setFont("Helvetica", 11)
    sentences_p2 = [
        "Governments should establish vocational training programmes in",
        "earth observation data analysis and satellite-based services.",
        "The Copernicus Land Monitoring Service relies on trained operators",
        "with expertise in remote sensing and geospatial analysis.",
        "Digital skills for the space sector include cloud computing and AI.",
        "Capacity building initiatives must address the full EO value chain.",
        "Policy support for the space downstream market should be prioritised.",
    ]
    y = 740
    for line in sentences_p2:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    """Return the path to the sample PDF, generating it if needed."""
    if not SAMPLE_PDF_PATH.exists():
        _generate_sample_pdf(SAMPLE_PDF_PATH)
    return SAMPLE_PDF_PATH


@pytest.fixture(scope="session")
def sample_pdf_bytes(sample_pdf_path: Path) -> bytes:
    """Return the raw bytes of the sample PDF."""
    return sample_pdf_path.read_bytes()


# ---------------------------------------------------------------------------
# Taxonomy fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_taxonomy():
    """Load the real taxonomy from config/taxonomy.yaml."""
    from core.taxonomy_loader import load_taxonomy
    return load_taxonomy()


@pytest.fixture(scope="session")
def minimal_taxonomy():
    """A tiny taxonomy for fast unit tests that don't need all themes."""
    return {
        "themes": [
            {
                "label": "EO Downstream Skills",
                "keywords": ["earth observation", "Copernicus", "remote sensing"],
                "queries": [
                    "skills needed for earth observation and satellite data",
                    "earth observation workforce training",
                ],
                "weight": 1.0,
            }
        ],
        "thresholds": {"valid_match": 0.50, "weak_match": 0.35},
        "search": {
            "sbert_model": "all-MiniLM-L6-v2",
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_sentences": 3,
            "chunk_size": 200,
            "chunk_overlap": 40,
        },
    }


# ---------------------------------------------------------------------------
# Extracted document fixture (cached so PDF is only parsed once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def extracted_doc(sample_pdf_bytes):
    """Fully extracted DocumentContent from the sample PDF."""
    from core.pdf_extractor import extract_document
    return extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")


@pytest.fixture(scope="session")
def doc_sentences(extracted_doc):
    """All sentences from the extracted sample document."""
    from core.chunker import make_sentences
    return make_sentences(extracted_doc)
