"""
core/pdf_extractor.py
---------------------
PDF text extraction with per-page tracking.
Uses pdfminer.six as the primary engine (better layout parsing),
with PyPDF2 as a fallback for encrypted/scanned PDFs.
"""

from __future__ import annotations

import io
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Holds extracted text for a single PDF page."""
    page_number: int          # 1-indexed
    raw_text: str
    clean_text: str = field(default="")

    def __post_init__(self):
        if not self.clean_text:
            self.clean_text = _clean_text(self.raw_text)


@dataclass
class DocumentContent:
    """Full extracted content for one PDF document."""
    filename: str
    pages: List[PageContent] = field(default_factory=list)
    extraction_method: str = "unknown"   # 'pdfminer' | 'pypdf2' | 'failed'
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        """All pages joined into one string."""
        return " ".join(p.clean_text for p in self.pages if p.clean_text)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def text_with_page_markers(self) -> str:
        """Returns text with <<PAGE N>> markers for excerpt page attribution."""
        parts = []
        for p in self.pages:
            if p.clean_text.strip():
                parts.append(f"<<PAGE {p.page_number}>> {p.clean_text}")
        return " ".join(parts)


def _clean_text(text: str) -> str:
    """
    Normalise extracted text:
    - Collapse excessive whitespace / newlines
    - Remove null bytes and control characters
    - Preserve sentence boundaries
    """
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Replace form-feed page breaks with space
    text = text.replace("\f", " ")
    # Collapse multiple spaces / tabs to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to double newline (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are only punctuation / numbers (header/footer artefacts)
    lines = text.splitlines()
    lines = [ln for ln in lines if len(re.sub(r"[\W\d]+", "", ln)) > 2]
    return " ".join(lines).strip()


def extract_with_pdfminer(file_bytes: bytes, filename: str) -> DocumentContent:
    """Primary extraction method using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer

        doc = DocumentContent(filename=filename, extraction_method="pdfminer")
        file_obj = io.BytesIO(file_bytes)

        for page_num, page_layout in enumerate(extract_pages(file_obj), start=1):
            page_text_parts = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text_parts.append(element.get_text())
            raw = "\n".join(page_text_parts)
            doc.pages.append(PageContent(page_number=page_num, raw_text=raw))

        return doc

    except Exception as exc:
        logger.warning("pdfminer failed for %s: %s", filename, exc)
        return DocumentContent(
            filename=filename,
            extraction_method="failed",
            error=str(exc),
        )


def extract_with_pypdf2(file_bytes: bytes, filename: str) -> DocumentContent:
    """Fallback extraction using PyPDF2."""
    try:
        import PyPDF2  # type: ignore

        doc = DocumentContent(filename=filename, extraction_method="pypdf2")
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            doc.pages.append(PageContent(page_number=page_num, raw_text=raw))

        return doc

    except Exception as exc:
        logger.error("PyPDF2 also failed for %s: %s", filename, exc)
        return DocumentContent(
            filename=filename,
            extraction_method="failed",
            error=str(exc),
        )


def extract_document(file_bytes: bytes, filename: str) -> DocumentContent:
    """
    Main entry point. Tries pdfminer first, falls back to PyPDF2.
    Returns a DocumentContent object even on failure (with error field set).
    """
    doc = extract_with_pdfminer(file_bytes, filename)

    if doc.extraction_method == "failed" or not doc.full_text.strip():
        logger.info("Falling back to PyPDF2 for %s", filename)
        doc = extract_with_pypdf2(file_bytes, filename)

    total_chars = sum(len(p.clean_text) for p in doc.pages)
    logger.info(
        "Extracted %s: %d pages, %d chars via %s",
        filename, doc.page_count, total_chars, doc.extraction_method
    )
    return doc
