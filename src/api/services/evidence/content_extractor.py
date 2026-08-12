"""
Content Extractor (evidence ingestion front-half)

HTML -> cleaned, heading-structured content. PDF -> plain text with page
markers. Strips navigation/footer/ads/cookie-banners/related-links/social
buttons so what gets embedded is the article body, not site chrome — see
the architecture review's ingestion-pipeline section on why raw HTML
should never be embedded directly.

Heuristic (pattern/structure matching), not a learned readability model —
the same technique clinical_inference.py and clinical_normalization.py
already use successfully elsewhere in this codebase, for the same reason:
fast, free, deterministic, and auditable. It will do worse than a
dedicated readability library on an adversarial page layout; that's an
acceptable trade against not adding a large, indirect dependency, given
this only ever runs against a curated allowlist of known, structurally
stable government/nonprofit sites (source_registry.py), not the open web.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

# Elements removed outright regardless of content.
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form", "iframe"}

# class/id substrings that mark chrome, not article content. Matched
# case-insensitively against the element's combined class list + id.
_CHROME_PATTERNS = re.compile(
    r"nav(?:igation)?|menu|footer|sidebar|cookie|banner|breadcrumb|social[-_]?share|"
    r"share[-_]?button|related[-_]?(?:links|articles|content)|advertisement|\bad[-_]|"
    r"promo|subscribe|newsletter|site[-_]?search|skip[-_]?to[-_]?content|back[-_]?to[-_]?top|"
    r"comment(?:s)?[-_]?section|print[-_]?(?:button|page)",
    re.IGNORECASE,
)

# Tried in order; first one that yields non-trivial text wins.
_CONTENT_ROOT_SELECTORS = [
    "main", "article",
    "#main-content", "#content", "#main",
    ".article-body", ".field--name-body", ".content-body", ".page-content",
    "[role='main']",
]

_MIN_CONTENT_CHARS = 200  # below this, we didn't find real article content


@dataclass
class ContentSection:
    level: int  # 1-4, matching h1-h4 (h1 rare in-body; usually the doc title)
    heading: str
    text: str  # this section's own paragraph/list/table text, not children's


@dataclass
class ExtractedDocument:
    title: str
    sections: List[ContentSection] = field(default_factory=list)
    plain_text: str = ""  # full document, sections joined — fallback for unstructured content

    def is_usable(self) -> bool:
        return len(self.plain_text.strip()) >= _MIN_CONTENT_CHARS


def _is_chrome(tag: Tag) -> bool:
    classes = " ".join(tag.get("class") or [])
    ident = tag.get("id") or ""
    return bool(_CHROME_PATTERNS.search(f"{classes} {ident}"))


def _strip_chrome(soup: BeautifulSoup) -> None:
    for tag_name in _STRIP_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()
    # A second pass for class/id-based chrome that isn't in a semantic tag
    # (e.g. a <div class="cookie-banner">). Collected first, then removed,
    # since decomposing while iterating find_all's live-ish result can
    # skip siblings.
    to_remove = [el for el in soup.find_all(True) if isinstance(el, Tag) and _is_chrome(el)]
    for el in to_remove:
        el.decompose()


def _find_content_root(soup: BeautifulSoup) -> Tag:
    for selector in _CONTENT_ROOT_SELECTORS:
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) >= _MIN_CONTENT_CHARS:
            return el
    return soup.body or soup


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}


def extract_html(html: bytes, source_url: str = "") -> ExtractedDocument:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text()) if title_tag else source_url or "Untitled"
    # Common pattern: "<Page title> | Site Name" or "<Page title> - Site Name"
    title = re.split(r"\s*[|–—-]\s*(?:National Cancer Institute|Cancer\.Net|"
                      r"American Cancer Society|MedlinePlus|NIH|NCI).*$", title)[0].strip() or title

    _strip_chrome(soup)
    root = _find_content_root(soup)

    sections: List[ContentSection] = []
    current_heading = title
    current_level = 1
    buffer: List[str] = []

    def flush():
        text = _clean_text("\n".join(buffer))
        if text:
            sections.append(ContentSection(level=current_level, heading=current_heading, text=text))
        buffer.clear()

    for el in root.descendants:
        if isinstance(el, NavigableString):
            continue
        if not isinstance(el, Tag):
            continue
        name = el.name.lower() if el.name else ""

        if name in _HEADING_LEVELS:
            heading_text = _clean_text(el.get_text())
            if not heading_text:
                continue
            flush()
            current_heading = heading_text
            current_level = _HEADING_LEVELS[name]
        elif name == "p":
            text = _clean_text(el.get_text())
            if text:
                buffer.append(text)
        elif name in ("li",):
            text = _clean_text(el.get_text())
            if text:
                buffer.append(f"- {text}")
        elif name == "table":
            # Flatten rows to "cell | cell | cell" lines rather than
            # dropping tables outright — dosing/reference-range tables
            # carry real content.
            for row in el.find_all("tr"):
                cells = [_clean_text(c.get_text()) for c in row.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    buffer.append(" | ".join(cells))

    flush()

    plain_text = "\n\n".join(
        (f"{'#' * min(s.level, 4)} {s.heading}\n{s.text}" if s.heading != title or i > 0 else s.text)
        for i, s in enumerate(sections)
    )

    return ExtractedDocument(title=title, sections=sections, plain_text=plain_text)


def extract_pdf(pdf_bytes: bytes) -> ExtractedDocument:
    """PDF -> plain text with page markers, using PyMuPDF (already a
    dependency for literature ingestion — see document_processor.py).
    No heading detection for PDFs in this first pass (drug labels and
    guideline PDFs vary too much in layout for a cheap heuristic to be
    reliable) — section_chunker.py falls back to fixed-window chunking
    for any ExtractedDocument with no sections, which this always is."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        title = (doc.metadata or {}).get("title") or "Untitled PDF"
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                parts.append(f"[page {i + 1}]\n{_clean_text(text)}")
        plain_text = "\n\n".join(parts)
    finally:
        doc.close()

    return ExtractedDocument(title=title, sections=[], plain_text=plain_text)


def extract(content: bytes, content_type: str, source_url: str = "") -> ExtractedDocument:
    if "pdf" in content_type:
        return extract_pdf(content)
    return extract_html(content, source_url=source_url)
