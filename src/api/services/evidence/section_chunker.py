"""
Section Chunker (evidence ingestion front-half)

Turns an ExtractedDocument (content_extractor.py) into retrieval chunks
addressed by heading, not by an arbitrary character offset — see the
architecture review's ingestion-pipeline notes: "What causes taste
changes?" and "What can I eat?" should be two focused retrieval units,
not three fixed-size windows that each blend part of both.

Parent/child: a section under CHILD_MAX_CHARS becomes one chunk as-is. A
longer section splits into overlapping child windows, each still tagged
with that section's heading and carrying parent_text (the section's full
text) so generation can recover surrounding context even though
retrieval matched a narrower child window.

Falls back to the previous fixed-window behavior when the extracted
document has no sections at all — always true for PDFs, since
content_extractor.py does not attempt to detect headings in a PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.api.services.evidence.content_extractor import ExtractedDocument

CHILD_MAX_CHARS = 1200
CHILD_OVERLAP = 150
FALLBACK_CHUNK_CHARS = 1800
FALLBACK_OVERLAP = 200


@dataclass
class Chunk:
    text: str  # what gets embedded: "<heading>\n<body>" for a sectioned chunk
    section_title: Optional[str]
    chunk_index: int
    parent_text: Optional[str] = None  # full section text, when this chunk is a split-out child


def _windows(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out


def _fixed_window_chunks(text: str) -> List[Chunk]:
    text = (text or "").strip()
    if not text:
        return []
    return [
        Chunk(text=w, section_title=None, chunk_index=i)
        for i, w in enumerate(_windows(text, FALLBACK_CHUNK_CHARS, FALLBACK_OVERLAP))
    ]


def chunk_document(doc: ExtractedDocument) -> List[Chunk]:
    if not doc.sections:
        return _fixed_window_chunks(doc.plain_text)

    chunks: List[Chunk] = []
    idx = 0
    for section in doc.sections:
        text = (section.text or "").strip()
        if not text:
            continue
        if len(text) <= CHILD_MAX_CHARS:
            chunks.append(Chunk(
                text=f"{section.heading}\n{text}", section_title=section.heading, chunk_index=idx,
            ))
            idx += 1
        else:
            for child_text in _windows(text, CHILD_MAX_CHARS, CHILD_OVERLAP):
                chunks.append(Chunk(
                    text=f"{section.heading}\n{child_text}", section_title=section.heading,
                    chunk_index=idx, parent_text=text,
                ))
                idx += 1

    if not chunks:
        return _fixed_window_chunks(doc.plain_text)
    return chunks
