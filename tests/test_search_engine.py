"""
tests/test_search_engine.py
----------------------------
Tests for core/search_engine.py.

Split into two groups:
  - Fast unit tests (no model loading) using mocked embeddings
  - Integration tests using the real SBERT model + sample PDF
    (marked with @pytest.mark.integration — skipped in fast mode via -m "not integration")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.search_engine import (
    SearchResult,
    _classify,
    keyword_search,
    refine_with_cross_encoder,
    run_search_pipeline,
    semantic_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSentence:
    """Minimal sentence object for unit tests."""
    def __init__(self, text: str, page: int = 1, filename: str = "test.pdf"):
        self.text = text
        self.page = page
        self.doc_filename = filename


def _make_sentences(texts: List[str]) -> List[_FakeSentence]:
    return [_FakeSentence(t, page=i + 1) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Classification threshold tests
# ---------------------------------------------------------------------------

class TestClassify:
    def test_valid_above_threshold(self):
        assert _classify(0.55, 0.50, 0.35) == "VALID EVIDENCE"

    def test_valid_at_exact_threshold(self):
        assert _classify(0.50, 0.50, 0.35) == "VALID EVIDENCE"

    def test_weak_between_thresholds(self):
        assert _classify(0.42, 0.50, 0.35) == "WEAK EVIDENCE"

    def test_weak_at_exact_lower_threshold(self):
        assert _classify(0.35, 0.50, 0.35) == "WEAK EVIDENCE"

    def test_not_relevant_below_weak(self):
        assert _classify(0.20, 0.50, 0.35) == "NOT RELEVANT"

    def test_zero_score_not_relevant(self):
        assert _classify(0.0, 0.50, 0.35) == "NOT RELEVANT"

    def test_perfect_score_valid(self):
        assert _classify(1.0, 0.50, 0.35) == "VALID EVIDENCE"

    @pytest.mark.parametrize("score,expected", [
        (0.49, "WEAK EVIDENCE"),
        (0.50, "VALID EVIDENCE"),
        (0.34, "NOT RELEVANT"),
        (0.35, "WEAK EVIDENCE"),
    ])
    def test_boundary_conditions(self, score, expected):
        assert _classify(score, 0.50, 0.35) == expected


# ---------------------------------------------------------------------------
# Keyword search unit tests
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_finds_exact_keyword(self):
        sents = _make_sentences([
            "Earth observation data is used for environmental monitoring.",
            "Agricultural subsidies and rural development funding.",
        ])
        results = keyword_search(sents, "EO Skills", ["earth observation"])
        assert len(results) == 1
        assert results[0].matched_keyword == "earth observation"

    def test_case_insensitive_match(self):
        sents = _make_sentences(["EARTH OBSERVATION skills are critical."])
        results = keyword_search(sents, "EO Skills", ["earth observation"])
        assert len(results) == 1

    def test_no_false_positive(self):
        sents = _make_sentences(["Agricultural policy and rural development."])
        results = keyword_search(sents, "EO Skills", ["earth observation", "Copernicus"])
        assert len(results) == 0

    def test_each_sentence_counted_once_per_theme(self):
        """A sentence matching multiple keywords should appear only once per theme."""
        sents = _make_sentences(["Copernicus earth observation satellite programme."])
        results = keyword_search(sents, "EO Skills", ["copernicus", "earth observation"])
        assert len(results) == 1

    def test_keyword_hit_flag_set(self):
        sents = _make_sentences(["Copernicus data skills are essential."])
        results = keyword_search(sents, "EO", ["Copernicus"])
        assert results[0].keyword_hit is True

    def test_result_preserves_page_number(self):
        sents = [_FakeSentence("Satellite remote sensing skills.", page=7)]
        results = keyword_search(sents, "EO", ["satellite"])
        assert results[0].page == 7

    def test_result_preserves_doc_filename(self):
        sents = [_FakeSentence("Earth observation.", filename="policy_abc.pdf")]
        results = keyword_search(sents, "EO", ["earth observation"])
        assert results[0].doc_filename == "policy_abc.pdf"

    def test_multiple_sentences_multiple_matches(self):
        sents = _make_sentences([
            "Copernicus services provide earth observation data.",
            "Remote sensing and geospatial analysis skills are needed.",
            "This sentence is about agriculture and food security.",
        ])
        results = keyword_search(sents, "EO", ["copernicus", "remote sensing"])
        assert len(results) == 2

    def test_empty_sentences_list(self):
        results = keyword_search([], "EO", ["copernicus"])
        assert results == []

    def test_empty_keywords_list(self):
        sents = _make_sentences(["Earth observation is important."])
        results = keyword_search(sents, "EO", [])
        assert results == []


# ---------------------------------------------------------------------------
# SearchResult dataclass tests
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_default_fields(self):
        r = SearchResult(
            doc_filename="doc.pdf",
            page=1,
            excerpt="Earth observation skills gap.",
            theme="EO Skills",
            keyword_hit=True,
            sbert_score=0.7,
        )
        assert r.cross_encoder_score == 0.0
        assert r.final_score == 0.0
        assert r.validation_category == "UNSCORED"
        assert r.human_label == ""

    def test_human_label_can_be_set(self):
        r = SearchResult(
            doc_filename="doc.pdf", page=1, excerpt="test",
            theme="EO", keyword_hit=False, sbert_score=0.5,
        )
        r.human_label = "Valid evidence"
        assert r.human_label == "Valid evidence"


# ---------------------------------------------------------------------------
# Semantic search — unit tests with mocked SBERT
# ---------------------------------------------------------------------------

class TestSemanticSearchUnit:
    """
    These tests mock the SBERT model so they run in milliseconds
    and require no GPU or model download.
    """

    def _make_fake_encode(self, n_dims: int = 384):
        """Returns a mock encode function that returns random unit vectors."""
        def fake_encode(texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            vecs = np.random.rand(n, n_dims).astype(np.float32)
            # Normalise to unit vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms
        return fake_encode

    def test_semantic_search_returns_list(self):
        sents = _make_sentences([
            "Earth observation and satellite remote sensing skills training.",
            "Agricultural subsidies in the EU rural development fund.",
            "Copernicus programme provides geospatial data services.",
        ])
        with patch("core.search_engine._get_sbert") as mock_sbert:
            mock_model = MagicMock()
            mock_model.encode.side_effect = self._make_fake_encode()
            mock_sbert.return_value = mock_model

            results = semantic_search(
                sents,
                theme_queries=["skills for earth observation"],
                theme_label="EO Skills",
                sbert_model_name="all-MiniLM-L6-v2",
                top_k=2,
                min_score=0.0,  # accept all scores in unit test
            )
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_semantic_search_empty_sentences(self):
        with patch("core.search_engine._get_sbert"):
            results = semantic_search(
                [],
                theme_queries=["earth observation skills"],
                theme_label="EO",
                sbert_model_name="all-MiniLM-L6-v2",
            )
        assert results == []

    def test_semantic_search_result_has_sbert_score(self):
        sents = _make_sentences(["Earth observation satellite skills training."])
        with patch("core.search_engine._get_sbert") as mock_sbert:
            mock_model = MagicMock()
            mock_model.encode.side_effect = self._make_fake_encode()
            mock_sbert.return_value = mock_model

            results = semantic_search(
                sents,
                theme_queries=["earth observation skills"],
                theme_label="EO",
                sbert_model_name="all-MiniLM-L6-v2",
                top_k=1,
                min_score=0.0,
            )

        if results:
            assert results[0].sbert_score >= 0.0
            assert results[0].keyword_hit is False


# ---------------------------------------------------------------------------
# CrossEncoder refinement unit tests
# ---------------------------------------------------------------------------

class TestCrossEncoderRefinement:
    def _make_results(self) -> List[SearchResult]:
        return [
            SearchResult(
                doc_filename="doc.pdf", page=1,
                excerpt="Copernicus earth observation skills for downstream users.",
                theme="EO Skills", keyword_hit=True, sbert_score=0.6,
            ),
            SearchResult(
                doc_filename="doc.pdf", page=2,
                excerpt="Agricultural subsidies and rural policy measures.",
                theme="EO Skills", keyword_hit=False, sbert_score=0.2,
            ),
        ]

    def test_refine_assigns_final_scores(self):
        results = self._make_results()
        with patch("core.search_engine._get_cross_encoder") as mock_ce:
            mock_ce.return_value.predict.return_value = np.array([0.72, 0.15])
            refined = refine_with_cross_encoder(
                results,
                theme_queries=["earth observation skills"],
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                valid_threshold=0.50,
                weak_threshold=0.35,
            )
        assert refined[0].final_score > refined[1].final_score

    def test_refine_assigns_validation_categories(self):
        results = self._make_results()
        with patch("core.search_engine._get_cross_encoder") as mock_ce:
            mock_ce.return_value.predict.return_value = np.array([0.80, 0.10])
            refined = refine_with_cross_encoder(
                results,
                theme_queries=["earth observation skills"],
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                valid_threshold=0.50,
                weak_threshold=0.35,
            )
        # First result has high CE score (0.80 + 0.6 SBERT blend → VALID)
        assert refined[0].validation_category in ("VALID EVIDENCE", "WEAK EVIDENCE")
        # Second result has low CE score (0.10) → NOT RELEVANT
        assert refined[1].validation_category == "NOT RELEVANT"

    def test_refine_empty_list(self):
        with patch("core.search_engine._get_cross_encoder"):
            result = refine_with_cross_encoder(
                [], ["query"], "model", 0.50, 0.35
            )
        assert result == []

    def test_keyword_boost_applied(self):
        """Keyword hits should score slightly higher than equivalent semantic-only results."""
        kw_result = SearchResult(
            doc_filename="doc.pdf", page=1, excerpt="EO skills.",
            theme="EO", keyword_hit=True, sbert_score=0.5,
        )
        sem_result = SearchResult(
            doc_filename="doc.pdf", page=2, excerpt="EO skills.",
            theme="EO", keyword_hit=False, sbert_score=0.5,
        )
        with patch("core.search_engine._get_cross_encoder") as mock_ce:
            mock_ce.return_value.predict.return_value = np.array([0.6, 0.6])
            refined = refine_with_cross_encoder(
                [kw_result, sem_result],
                ["earth observation skills"],
                "model",
                0.50,
                0.35,
            )
        # keyword_hit adds 0.05 boost
        assert refined[0].final_score > refined[1].final_score


# ---------------------------------------------------------------------------
# Full pipeline integration tests (use real models + sample PDF)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullPipeline:
    """
    End-to-end pipeline tests. These load real SBERT + CrossEncoder models.
    Run with: pytest -m integration
    Skipped by default in fast CI mode: pytest -m "not integration"
    """

    def test_pipeline_returns_results(self, doc_sentences, minimal_taxonomy):
        results = run_search_pipeline(
            sentences=doc_sentences,
            taxonomy=minimal_taxonomy,
            sbert_model_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k_sentences=3,
            min_sbert_score=0.2,
        )
        assert isinstance(results, list)
        # The sample PDF is about EO skills — we must find at least one result
        assert len(results) >= 1

    def test_pipeline_results_sorted_by_score(self, doc_sentences, minimal_taxonomy):
        results = run_search_pipeline(
            sentences=doc_sentences,
            taxonomy=minimal_taxonomy,
            sbert_model_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_pipeline_finds_eo_theme(self, doc_sentences, minimal_taxonomy):
        results = run_search_pipeline(
            sentences=doc_sentences,
            taxonomy=minimal_taxonomy,
            sbert_model_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            min_sbert_score=0.2,
        )
        themes_found = {r.theme for r in results}
        assert "EO Downstream Skills" in themes_found

    def test_pipeline_valid_evidence_found(self, doc_sentences, minimal_taxonomy):
        """The sample PDF has strong EO signals — at least one VALID result expected."""
        results = run_search_pipeline(
            sentences=doc_sentences,
            taxonomy=minimal_taxonomy,
            sbert_model_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            min_sbert_score=0.2,
        )
        valid = [r for r in results if r.validation_category == "VALID EVIDENCE"]
        assert len(valid) >= 1, (
            "No VALID EVIDENCE found — check thresholds or sample PDF content. "
            f"Top scores: {[r.final_score for r in results[:5]]}"
        )

    def test_pipeline_no_openai_by_default(self, doc_sentences, minimal_taxonomy):
        """Pipeline should complete without OPENAI_API_KEY set."""
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        results = run_search_pipeline(
            sentences=doc_sentences,
            taxonomy=minimal_taxonomy,
            sbert_model_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            use_openai=False,
        )
        assert all(r.openai_score == 0.0 for r in results)
