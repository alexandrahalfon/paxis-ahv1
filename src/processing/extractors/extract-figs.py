#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust PDF extractor with coverage + OCR fallback.

Requirements (install as needed):
  pip install pymupdf pdfminer.six pdfplumber pytesseract pillow numpy opencv-python

For table extraction (optional):
  pip install camelot-py[cv] ghostscript  # for lattice tables
  OR
  pip install tabula-py                   # requires Java

Usage:
  python pdf_extract_q76.py --pdf "/path/to/file.pdf" --out "/path/to/out.json" --coverage_csv "/path/to/coverage.csv" --threshold 0.95

Notes:
  - Figures/charts that are purely raster images will require OCR to capture embedded text (captions/axis labels).
  - This script keeps layout-aware reading order using PyMuPDF blocks and a simple two-column reflow heuristic.
"""

import json
import re
import sys
import math
import unicodedata
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Base deps
import fitz  # PyMuPDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LAParams

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Optional tables
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[\u0000-\u001F\u007F]", " ", s)
    s = re.sub(r"-\s*\n\s*", "", s)  # dehyphenate across linebreaks
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def shingle_chars(s: str, k: int = 5) -> List[str]:
    s = s.lower()
    return [s[i:i+k] for i in range(0, max(0, len(s)-k+1))]

def extract_pdf_pages_text_pdfminer(pdf_path: Path) -> List[str]:
    laparams = LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1)
    pages = []
    for page_layout in extract_pages(str(pdf_path), laparams=laparams):
        txt_parts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                txt_parts.append(element.get_text())
        pages.append(normalize_text("".join(txt_parts)))
    return pages

def reflow_blocks_two_column(blocks: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
    """
    Simple two-column heuristic: split by vertical midline and read top-to-bottom left col, then right col.
    """
    if not blocks:
        return []
    mid_x = page_width / 2.0
    left = [b for b in blocks if b["bbox"][0] < mid_x]
    right = [b for b in blocks if b["bbox"][0] >= mid_x]

    # sort within each col by y0 (top) then x0
    left_sorted = sorted(left, key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
    right_sorted = sorted(right, key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))

    # If content is clearly one column (e.g., one of the columns is almost empty), just return natural top-to-bottom order
    if len(left_sorted) == 0 or len(right_sorted) == 0:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1],1), round(b["bbox"][0],1)))

    return left_sorted + right_sorted

def page_blocks_text(page: fitz.Page) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Get text blocks with bounding boxes and a reflowed text string.
    """
    page_dict = page.get_text("dict")
    blocks = []
    for b in page_dict.get("blocks", []):
        if "lines" not in b:
            continue
        # Combine spans into one block string
        lines = []
        for ln in b["lines"]:
            spans_txt = " ".join([s.get("text","") for s in ln.get("spans", [])])
            lines.append(spans_txt)
        blk_text = normalize_text("\n".join(lines))
        if blk_text.strip():
            blocks.append({"text": blk_text, "bbox": b["bbox"]})

    # Decide if two-column reflow helps by comparing x-distribution
    width = page.rect.width
    reflowed = reflow_blocks_two_column(blocks, width)
    text_reflowed = normalize_text("\n".join([b["text"] for b in reflowed]))
    return text_reflowed, reflowed

def ocr_page(page: fitz.Page, dpi: int = 300, lang: str = "eng") -> str:
    if not OCR_AVAILABLE:
        return ""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cfg = "--oem 1 --psm 6"
    try:
        txt = pytesseract.image_to_string(img, lang=lang, config=cfg)
        return normalize_text(txt)
    except Exception:
        return ""

def extract_captions(text: str) -> List[Dict[str, Any]]:
    """
    Find Figure/Table captions by regex. Return list of dicts with kind/id/text.
    """
    out = []
    # Split into lines to find starts
    for line in re.split(r"\n+", text):
        line_clean = line.strip()
        m = re.match(r"^(figure|table)\s*([0-9ivxlcdm]+)[:\.\s-]*(.*)$", line_clean, re.I)
        if m:
            kind = m.group(1).title()
            ident = m.group(2)
            tail = m.group(3).strip()
            out.append({"kind": kind, "id": ident, "caption": line_clean if tail=="" else f"{kind} {ident}: {tail}"})
    return out

def extract_tables_pdfplumber(pdf_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    Try extracting tables with pdfplumber (stream or lattice). Returns dict page->list of tables.
    """
    if not PDFPLUMBER_AVAILABLE:
        return {}
    results = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            page_tables = []
            # try lattice first (works if ruling lines exist), else stream
            for flavor in ("lattice", "stream"):
                try:
                    tbls = p.extract_tables(table_settings={"vertical_strategy": "lines", "horizontal_strategy":"lines"}) if flavor=="lattice" else p.extract_tables()
                    for t in tbls or []:
                        # t is a list of rows; convert to rows of strings
                        rows = [[(cell or "").strip() for cell in row] for row in t]
                        if not rows:
                            continue
                        page_tables.append({"flavor": flavor, "rows": rows})
                    if page_tables:
                        break
                except Exception:
                    continue
            if page_tables:
                results[i] = page_tables
    return results

def coverage_against_corpus(page_texts: List[str], corpus_text: str, k: int = 5) -> List[Dict[str, Any]]:
    corpus_sh = set(shingle_chars(corpus_text, k=k)) if corpus_text else set()
    rows = []
    for idx, t in enumerate(page_texts, start=1):
        page_norm = t.lower()
        sh = set(shingle_chars(page_norm, k=k))
        total = len(sh)
        matched = len(sh & corpus_sh) if total else 0
        cov = (matched/total) if total else 0.0
        rows.append({
            "page_number": idx,
            "page_char_len": len(page_norm),
            "total_shingles": total,
            "matched_shingles": matched,
            "coverage_ratio": round(cov,4),
            "flag_below_0.95": cov < 0.95
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--out", required=True, help="Path to output JSON")
    ap.add_argument("--coverage_csv", default=None, help="Path to coverage CSV")
    ap.add_argument("--threshold", type=float, default=0.95, help="Coverage threshold for OCR fallback")
    ap.add_argument("--lang", default="eng", help="OCR language for Tesseract")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    cov_csv = Path(args.coverage_csv).expanduser().resolve() if args.coverage_csv else None

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    n_pages = doc.page_count

    # First pass: digital extraction with layout-aware reflow + captions + (optional) tables
    pages: List[Dict[str, Any]] = []
    for i in range(n_pages):
        page = doc.load_page(i)
        text_reflowed, blocks = page_blocks_text(page)
        captions = extract_captions(text_reflowed)

        pages.append({
            "page_number": i+1,
            "text": text_reflowed,
            "captions": captions,
            "blocks": [{"bbox": b["bbox"], "text": b["text"]} for b in blocks]
        })

    # Optional: table extraction via pdfplumber (fast, lightweight)
    tables_by_page = extract_tables_pdfplumber(pdf_path)

    # Build corpus and run coverage
    corpus_text = normalize_text(" ".join([p["text"] for p in pages]))
    cov_rows = coverage_against_corpus([p["text"] for p in pages], corpus_text, k=5)

    # If any page below threshold and OCR is available, re-OCR those pages and merge text
    any_ocr = False
    for row in cov_rows:
        if row["flag_below_0.95"] and OCR_AVAILABLE:
            pidx = row["page_number"] - 1
            ocr_txt = ocr_page(doc.load_page(pidx), dpi=300, lang=args.lang)
            if ocr_txt and len(ocr_txt) > 10:
                # Merge OCR text that isn't already present
                merged = normalize_text((pages[pidx]["text"] + " " + ocr_txt).strip())
                pages[pidx]["text"] = merged
                # Re-extract captions from merged text
                pages[pidx]["captions"] = extract_captions(merged)
                any_ocr = True

    # Recompute coverage if OCR added anything
    if any_ocr:
        corpus_text = normalize_text(" ".join([p["text"] for p in pages]))
        cov_rows = coverage_against_corpus([p["text"] for p in pages], corpus_text, k=5)

    # Attach tables into page records
    for p in pages:
        pnum = p["page_number"]
        if pnum in tables_by_page:
            p["tables"] = tables_by_page[pnum]

    # Heuristic Figure/Table presence
    figtab_counts = {
        "figure_mentions": len(re.findall(r"\bfigure\s*\d+\b", corpus_text.lower())),
        "table_mentions": len(re.findall(r"\btable\s*\d+\b", corpus_text.lower()))
    }

    result = {
        "source_pdf": str(pdf_path),
        "n_pages": n_pages,
        "figtab_counts": figtab_counts,
        "pages": pages,
        "coverage": cov_rows
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if cov_csv:
        import csv
        with cov_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()))
            writer.writeheader()
            for r in cov_rows:
                writer.writerow(r)

    # Console summary
    n_flagged = sum(1 for r in cov_rows if r["flag_below_0.95"])
    print(f"[Done] Pages: {n_pages}. Flagged (<{args.threshold*100:.0f}% coverage): {n_flagged}")
    if n_flagged:
        print("Flagged pages with coverage ratios:")
        for r in cov_rows:
            if r["flag_below_0.95"]:
                print(f"  - Page {r['page_number']}: coverage={r['coverage_ratio']}, chars={r['page_char_len']}")
    if not OCR_AVAILABLE:
        print("Note: pytesseract not available. Install it to OCR scanned/low-coverage pages.")
    if not PDFPLUMBER_AVAILABLE:
        print("Note: pdfplumber not available. Install it to improve table extraction.")

if __name__ == "__main__":
    main()
