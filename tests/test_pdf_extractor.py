"""
tests/test_pdf_extractor.py
----------------------------
Integration tests for core/pdf_extractor.py using the sample PDF fixture.
These tests verify the real extraction pipeline end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pdf_extractor import (
    DocumentContent,
    PageContent,
    _clean_text,
    extract_document,
    extract_with_pypdf2,
    extract_with_pdfminer,
)


# ---------------------------------------------------------------------------
# Unit tests for _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_null_bytes(self):
        assert "\x00" not in _clean_text("Hello\x00World")

    def test_collapses_whitespace(self):
        result = _clean_text("Hello   world\t\there")
        assert "  " not in result

    def test_handles_empty_string(self):
        assert _clean_text("") == ""

    def test_removes_form_feed(self):
        result = _clean_text("Page one\fPage two")
        assert "\f" not in result

    def test_preserves_meaningful_text(self):
        text = "Earth observation skills are required for downstream EO applications."
        result = _clean_text(text)
        assert "earth observation" in result.lower()
        assert "downstream" in result.lower()

    def test_filters_short_lines(self):
        # Lines with only 1-2 letters after stripping non-word chars are dropped
        text = "A.\nB.\nThis is a proper sentence about satellite data."
        result = _clean_text(text)
        assert "satellite data" in result

    def test_collapses_multiple_newlines(self):
        text = "Para one.\n\n\n\n\nPara two."
        result = _clean_text(text)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# Unit tests for PageContent
# ---------------------------------------------------------------------------

class TestPageContent:
    def test_auto_cleans_text(self):
        p = PageContent(page_number=1, raw_text="Hello   World\x00")
        assert "\x00" not in p.clean_text
        assert "  " not in p.clean_text

    def test_page_number_stored(self):
        p = PageContent(page_number=3, raw_text="Some text here.")
        assert p.page_number == 3

    def test_empty_raw_text(self):
        p = PageContent(page_number=1, raw_text="")
        assert p.clean_text == ""


# ---------------------------------------------------------------------------
# Unit tests for DocumentContent
# ---------------------------------------------------------------------------

class TestDocumentContent:
    def _make_doc(self):
        return DocumentContent(
            filename="test.pdf",
            pages=[
                PageContent(page_number=1, raw_text="Earth observation skills training."),
                PageContent(page_number=2, raw_text="Copernicus data analysis workforce."),
            ],
        )

    def test_full_text_joins_pages(self):
        doc = self._make_doc()
        assert "earth observation" in doc.full_text.lower()
        assert "copernicus" in doc.full_text.lower()

    def test_page_count(self):
        doc = self._make_doc()
        assert doc.page_count == 2

    def test_text_with_page_markers(self):
        doc = self._make_doc()
        marked = doc.text_with_page_markers()
        assert "<<PAGE 1>>" in marked
        assert "<<PAGE 2>>" in marked

    def test_empty_doc_has_no_text(self):
        doc = DocumentContent(filename="empty.pdf", pages=[])
        assert doc.full_text == ""
        assert doc.page_count == 0


# ---------------------------------------------------------------------------
# Integration tests using the sample PDF fixture
# ---------------------------------------------------------------------------

class TestExtractDocument:
    def test_extract_document_returns_content(self, sample_pdf_bytes):
        doc = extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert isinstance(doc, DocumentContent)
        assert doc.filename == "sample_eo_policy.pdf"

    def test_extract_document_has_pages(self, sample_pdf_bytes):
        doc = extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert doc.page_count >= 1

    def test_extract_document_has_text(self, sample_pdf_bytes):
        doc = extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert len(doc.full_text) > 100

    def test_extract_document_contains_eo_keywords(self, sample_pdf_bytes):
        """The sample PDF is about EO skills — these keywords must appear."""
        doc = extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")
        text_lower = doc.full_text.lower()
        assert any(
            kw in text_lower
            for kw in ["earth observation", "copernicus", "satellite", "skills"]
        ), f"No EO keywords found in extracted text: {doc.full_text[:200]}"

    def test_extraction_method_recorded(self, sample_pdf_bytes):
        doc = extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert doc.extraction_method in ("pdfminer", "pypdf2")

    def test_no_error_on_valid_pdf(self, sample_pdf_bytes):
        doc = extract_document(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert doc.error is None

    def test_extract_with_bad_bytes_does_not_crash(self):
        """Corrupt bytes should return a DocumentContent with error set, not raise."""
        bad_bytes = b"this is not a pdf at all"
        doc = extract_document(bad_bytes, "corrupt.pdf")
        assert isinstance(doc, DocumentContent)
        assert doc.extraction_method == "failed"
        assert doc.error is not None

    def test_pdfminer_returns_doc(self, sample_pdf_bytes):
        doc = extract_with_pdfminer(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert isinstance(doc, DocumentContent)

    def test_pypdf2_returns_doc(self, sample_pdf_bytes):
        doc = extract_with_pypdf2(sample_pdf_bytes, "sample_eo_policy.pdf")
        assert isinstance(doc, DocumentContent)
        assert doc.page_count >= 1


# ---------------------------------------------------------------------------
# Integration tests for the chunker (depends on extraction)
# ---------------------------------------------------------------------------

class TestChunker:
    def test_make_sentences_returns_list(self, extracted_doc):
        from core.chunker import make_sentences
        sents = make_sentences(extracted_doc)
        assert isinstance(sents, list)
        assert len(sents) > 0

    def test_sentences_have_page_attribution(self, extracted_doc):
        from core.chunker import make_sentences
        sents = make_sentences(extracted_doc)
        assert all(s.page >= 1 for s in sents)

    def test_sentences_minimum_length(self, extracted_doc):
        from core.chunker import make_sentences
        sents = make_sentences(extracted_doc)
        assert all(len(s.text) >= 25 for s in sents)

    def test_sentences_reference_correct_doc(self, extracted_doc):
        from core.chunker import make_sentences
        sents = make_sentences(extracted_doc)
        assert all(s.doc_filename == "sample_eo_policy.pdf" for s in sents)

    def test_make_chunks_respects_size(self, extracted_doc):
        from core.chunker import make_chunks
        chunks = make_chunks(extracted_doc, chunk_size=100, overlap=20)
        assert all(len(c.text.split()) <= 115 for c in chunks)

    def test_make_chunks_page_range_valid(self, extracted_doc):
        from core.chunker import make_chunks
        chunks = make_chunks(extracted_doc, chunk_size=100, overlap=20)
        assert all(c.start_page <= c.end_page for c in chunks)
