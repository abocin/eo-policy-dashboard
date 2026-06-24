"""
tests/test_pipeline.py
-----------------------
Unit tests for core pipeline components.
Run with: pytest tests/

Requires: pytest, sentence-transformers, pdfminer.six
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from core.pdf_extractor import _clean_text, PageContent, DocumentContent
from core.chunker import make_sentences, make_chunks, Sentence
from core.taxonomy_loader import load_taxonomy, _minimal_fallback
from core.exporters import results_to_dataframe
from core.search_engine import keyword_search, _classify


# ---------------------------------------------------------------------------
# Test: text cleaning
# ---------------------------------------------------------------------------

def test_clean_text_removes_null_bytes():
    raw = "Hello\x00World"
    assert "\x00" not in _clean_text(raw)


def test_clean_text_collapses_whitespace():
    raw = "Hello   world\t\there"
    result = _clean_text(raw)
    assert "  " not in result


def test_clean_text_empty_string():
    assert _clean_text("") == ""


# ---------------------------------------------------------------------------
# Test: sentence splitting
# ---------------------------------------------------------------------------

def test_make_sentences_basic():
    doc = DocumentContent(
        filename="test.pdf",
        pages=[
            PageContent(
                page_number=1,
                raw_text="Earth observation skills are in high demand. "
                         "Copernicus data training is essential for downstream users.",
            )
        ],
    )
    sents = make_sentences(doc)
    assert len(sents) >= 1
    assert all(isinstance(s, Sentence) for s in sents)
    assert all(s.page == 1 for s in sents)


def test_make_sentences_filters_short():
    doc = DocumentContent(
        filename="test.pdf",
        pages=[
            PageContent(page_number=1, raw_text="OK. Yes. This is a proper full sentence about EO skills.")
        ],
    )
    sents = make_sentences(doc)
    # Very short fragments ("OK", "Yes") should be filtered
    assert all(len(s.text) >= 25 for s in sents)


# ---------------------------------------------------------------------------
# Test: chunking
# ---------------------------------------------------------------------------

def test_make_chunks_returns_chunks():
    doc = DocumentContent(
        filename="test.pdf",
        pages=[
            PageContent(
                page_number=1,
                raw_text=" ".join(["word"] * 500),
            )
        ],
    )
    chunks = make_chunks(doc, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 110 for c in chunks)


# ---------------------------------------------------------------------------
# Test: keyword search
# ---------------------------------------------------------------------------

class _FakeSentence:
    def __init__(self, text, page=1, filename="test.pdf"):
        self.text = text
        self.page = page
        self.doc_filename = filename


def test_keyword_search_finds_match():
    sents = [
        _FakeSentence("This policy supports earth observation training programmes."),
        _FakeSentence("Nothing relevant here about agriculture."),
    ]
    results = keyword_search(sents, "EO Downstream Skills", ["earth observation", "Copernicus"])
    assert len(results) == 1
    assert results[0].keyword_hit is True
    assert results[0].matched_keyword == "earth observation"


def test_keyword_search_case_insensitive():
    sents = [_FakeSentence("EARTH OBSERVATION data is critical.")]
    results = keyword_search(sents, "EO", ["earth observation"])
    assert len(results) == 1


def test_keyword_search_no_false_positives():
    sents = [_FakeSentence("Agricultural subsidies and rural development policies.")]
    results = keyword_search(sents, "EO", ["earth observation", "Copernicus"])
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Test: classification thresholds
# ---------------------------------------------------------------------------

def test_classify_valid():
    assert _classify(0.55, 0.50, 0.35) == "VALID EVIDENCE"


def test_classify_weak():
    assert _classify(0.40, 0.50, 0.35) == "WEAK EVIDENCE"


def test_classify_not_relevant():
    assert _classify(0.20, 0.50, 0.35) == "NOT RELEVANT"


def test_classify_boundary_valid():
    assert _classify(0.50, 0.50, 0.35) == "VALID EVIDENCE"


# ---------------------------------------------------------------------------
# Test: taxonomy loader
# ---------------------------------------------------------------------------

def test_minimal_fallback_is_valid():
    tax = _minimal_fallback()
    assert "themes" in tax
    assert len(tax["themes"]) >= 1
    for t in tax["themes"]:
        assert "label" in t
        assert "queries" in t


def test_load_taxonomy_returns_dict():
    tax = load_taxonomy()
    assert isinstance(tax, dict)
    assert "themes" in tax


# ---------------------------------------------------------------------------
# Test: exporters
# ---------------------------------------------------------------------------

def test_results_to_dataframe_empty():
    df = results_to_dataframe([])
    assert len(df) == 0


def test_results_to_dataframe_columns():
    from core.search_engine import SearchResult
    r = SearchResult(
        doc_filename="test.pdf",
        page=1,
        excerpt="Earth observation is key.",
        theme="EO Downstream Skills",
        keyword_hit=True,
        sbert_score=0.7,
        cross_encoder_score=0.65,
        final_score=0.67,
        validation_category="VALID EVIDENCE",
    )
    df = results_to_dataframe([r])
    assert "Document" in df.columns
    assert "Final Score" in df.columns
    assert df.iloc[0]["Document"] == "test.pdf"
