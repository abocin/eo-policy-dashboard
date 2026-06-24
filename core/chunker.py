"""
core/chunker.py
---------------
Splits a DocumentContent into:
  - Overlapping word-based chunks (for embedding)
  - Sentence-level segments (for excerpt extraction & CrossEncoder)
Page attribution is preserved for each chunk / sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from core.pdf_extractor import DocumentContent


@dataclass
class TextChunk:
    """A sliding-window text chunk with page reference."""
    doc_filename: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int


@dataclass
class Sentence:
    """A single sentence with its source page."""
    doc_filename: str
    sentence_index: int
    text: str
    page: int


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences; filter out very short fragments."""
    sents = _SENT_RE.split(text)
    return [s.strip() for s in sents if len(s.strip()) >= 25]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_sentences(doc: DocumentContent) -> List[Sentence]:
    """
    Returns all sentences from a document with page attribution.
    Works page-by-page so we can track which page a sentence came from.
    """
    sentences: List[Sentence] = []
    idx = 0
    for page in doc.pages:
        for sent in _split_into_sentences(page.clean_text):
            sentences.append(
                Sentence(
                    doc_filename=doc.filename,
                    sentence_index=idx,
                    text=sent,
                    page=page.page_number,
                )
            )
            idx += 1
    return sentences


def make_chunks(
    doc: DocumentContent,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[TextChunk]:
    """
    Splits the full document text into overlapping word-level chunks.
    Each chunk records which pages it spans.

    Args:
        doc        : DocumentContent instance
        chunk_size : target number of words per chunk
        overlap    : number of words to repeat at each boundary
    """
    chunks: List[TextChunk] = []

    # Build a list of (word, page_number) pairs across the whole document
    word_page_pairs: List[Tuple[str, int]] = []
    for page in doc.pages:
        words = page.clean_text.split()
        for w in words:
            word_page_pairs.append((w, page.page_number))

    if not word_page_pairs:
        return chunks

    total = len(word_page_pairs)
    step = max(1, chunk_size - overlap)
    chunk_idx = 0

    for start in range(0, total, step):
        end = min(start + chunk_size, total)
        slice_ = word_page_pairs[start:end]
        text = " ".join(w for w, _ in slice_)
        start_page = slice_[0][1]
        end_page = slice_[-1][1]

        if len(text.strip()) < 30:
            continue

        chunks.append(
            TextChunk(
                doc_filename=doc.filename,
                chunk_index=chunk_idx,
                text=text,
                start_page=start_page,
                end_page=end_page,
            )
        )
        chunk_idx += 1

        if end == total:
            break

    return chunks
