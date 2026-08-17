#!/usr/bin/env python3
"""
Ingest NCI PDQ patient summaries from NCBI PubMed Books.

NCI's cancer.gov is a React SPA that requires JavaScript rendering.
The same PDQ content is mirrored on NCBI PubMed Books as server-rendered
HTML, which we can fetch with a plain HTTP client.

For each cancer type in our Qdrant/PG knowledge base, this script ingests
the PDQ patient treatment summary (and prevention/screening where available).

These go into the `oncology_patient_education` Qdrant collection via
ingest_document() — no code changes or redeployment needed.

Usage:
    python scripts/ingest_nci_cancer_types.py --seed-sources
    python scripts/ingest_nci_cancer_types.py
    python scripts/ingest_nci_cancer_types.py --dry-run
    python scripts/ingest_nci_cancer_types.py --category breast
    python scripts/ingest_nci_cancer_types.py --resume-from gi
"""

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx
from bs4 import BeautifulSoup


# ═══════════════════════════════════════════════════════════════════════
# NCI PDQ CONTENT MAPPING
# ═══════════════════════════════════════════════════════════════════════
# Each entry: (title, ncbi_book_id, cancer.gov_url_for_attribution)
# NCBI Book IDs from https://www.ncbi.nlm.nih.gov/books/n/pdqcis/
# These are the Patient Version PDQ summaries.

CATEGORY_TO_PDQ: Dict[str, List[Tuple[str, str, str]]] = {
    "breast": [
        ("Breast Cancer Treatment", "NBK65969",
         "https://www.cancer.gov/types/breast/patient/breast-treatment-pdq"),
        ("Breast Cancer Prevention", "NBK65979",
         "https://www.cancer.gov/types/breast/patient/breast-prevention-pdq"),
        ("Breast Cancer Screening", "NBK65811",
         "https://www.cancer.gov/types/breast/patient/breast-screening-pdq"),
    ],
    "lung": [
        ("Non-Small Cell Lung Cancer Treatment", "NBK65959",
         "https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq"),
        ("Small Cell Lung Cancer Treatment", "NBK65837",
         "https://www.cancer.gov/types/lung/patient/small-cell-lung-treatment-pdq"),
        ("Lung Cancer Prevention", "NBK65928",
         "https://www.cancer.gov/types/lung/patient/lung-prevention-pdq"),
        ("Lung Cancer Screening", "NBK65810",
         "https://www.cancer.gov/types/lung/patient/lung-screening-pdq"),
    ],
    "prostate": [
        ("Prostate Cancer Treatment", "NBK65915",
         "https://www.cancer.gov/types/prostate/patient/prostate-treatment-pdq"),
        ("Prostate Cancer Prevention", "NBK65925",
         "https://www.cancer.gov/types/prostate/patient/prostate-prevention-pdq"),
        ("Prostate Cancer Screening", "NBK65812",
         "https://www.cancer.gov/types/prostate/patient/prostate-screening-pdq"),
    ],
    "gi": [
        ("Colon Cancer Treatment", "NBK65782",
         "https://www.cancer.gov/types/colorectal/patient/colon-treatment-pdq"),
        ("Rectal Cancer Treatment", "NBK65830",
         "https://www.cancer.gov/types/colorectal/patient/rectal-treatment-pdq"),
        ("Pancreatic Cancer Treatment", "NBK65876",
         "https://www.cancer.gov/types/pancreatic/patient/pancreatic-treatment-pdq"),
        ("Liver Cancer Treatment (Adult)", "NBK65952",
         "https://www.cancer.gov/types/liver/patient/adult-liver-treatment-pdq"),
        ("Gastric (Stomach) Cancer Treatment", "NBK65847",
         "https://www.cancer.gov/types/stomach/patient/stomach-treatment-pdq"),
        ("Esophageal Cancer Treatment", "NBK65873",
         "https://www.cancer.gov/types/esophageal/patient/esophageal-treatment-pdq"),
        ("Gallbladder Cancer Treatment", "NBK65939",
         "https://www.cancer.gov/types/gallbladder/patient/gallbladder-treatment-pdq"),
        ("Colorectal Cancer Screening", "NBK65825",
         "https://www.cancer.gov/types/colorectal/patient/colorectal-screening-pdq"),
    ],
    "h_n": [
        ("Nasopharyngeal Cancer Treatment", "NBK66060",
         "https://www.cancer.gov/types/head-and-neck/patient/nasopharyngeal-treatment-pdq"),
        ("Oropharyngeal Cancer Treatment", "NBK65946",
         "https://www.cancer.gov/types/head-and-neck/patient/oropharyngeal-treatment-pdq"),
        ("Hypopharyngeal Cancer Treatment", "NBK65724",
         "https://www.cancer.gov/types/head-and-neck/patient/hypopharyngeal-treatment-pdq"),
        ("Laryngeal Cancer Treatment", "NBK65804",
         "https://www.cancer.gov/types/head-and-neck/patient/laryngeal-treatment-pdq"),
        ("Lip and Oral Cavity Cancer Treatment", "NBK65953",
         "https://www.cancer.gov/types/head-and-neck/patient/lip-mouth-treatment-pdq"),
        ("Salivary Gland Cancer Treatment", "NBK65893",
         "https://www.cancer.gov/types/head-and-neck/patient/salivary-gland-treatment-pdq"),
    ],
    "gyn": [
        ("Cervical Cancer Treatment", "NBK65763",
         "https://www.cancer.gov/types/cervical/patient/cervical-treatment-pdq"),
        ("Endometrial Cancer Treatment", "NBK65864",
         "https://www.cancer.gov/types/uterine/patient/endometrial-treatment-pdq"),
        ("Ovarian Epithelial Cancer Treatment", "NBK66007",
         "https://www.cancer.gov/types/ovarian/patient/ovarian-epithelial-treatment-pdq"),
        ("Cervical Cancer Screening", "NBK65808",
         "https://www.cancer.gov/types/cervical/patient/cervical-screening-pdq"),
    ],
    "gu": [
        ("Bladder Cancer Treatment", "NBK65876",
         "https://www.cancer.gov/types/bladder/patient/bladder-treatment-pdq"),
        ("Renal Cell Cancer Treatment", "NBK65909",
         "https://www.cancer.gov/types/kidney/patient/kidney-treatment-pdq"),
    ],
    "cns": [
        ("Adult CNS Tumors Treatment", "NBK65978",
         "https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq"),
    ],
    "cutaneous": [
        ("Melanoma Treatment", "NBK65880",
         "https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq"),
        ("Skin Cancer Treatment", "NBK65852",
         "https://www.cancer.gov/types/skin/patient/skin-treatment-pdq"),
        ("Skin Cancer Prevention", "NBK65877",
         "https://www.cancer.gov/types/skin/patient/skin-prevention-pdq"),
    ],
    "lymphoma": [
        ("Adult Hodgkin Lymphoma Treatment", "NBK65774",
         "https://www.cancer.gov/types/lymphoma/patient/adult-hodgkin-treatment-pdq"),
        ("Adult Non-Hodgkin Lymphoma Treatment", "NBK65799",
         "https://www.cancer.gov/types/lymphoma/patient/adult-nhl-treatment-pdq"),
    ],
    "leukemia": [
        ("Adult Acute Lymphoblastic Leukemia Treatment", "NBK66005",
         "https://www.cancer.gov/types/leukemia/patient/adult-all-treatment-pdq"),
        ("Adult Acute Myeloid Leukemia Treatment", "NBK65869",
         "https://www.cancer.gov/types/leukemia/patient/adult-aml-treatment-pdq"),
        ("Chronic Lymphocytic Leukemia Treatment", "NBK65888",
         "https://www.cancer.gov/types/leukemia/patient/cll-treatment-pdq"),
        ("Chronic Myelogenous Leukemia Treatment", "NBK65898",
         "https://www.cancer.gov/types/leukemia/patient/cml-treatment-pdq"),
    ],
    "myeloma": [
        ("Plasma Cell Neoplasms Treatment", "NBK65889",
         "https://www.cancer.gov/types/myeloma/patient/myeloma-treatment-pdq"),
    ],
    "thyroid": [
        ("Thyroid Cancer Treatment", "NBK65823",
         "https://www.cancer.gov/types/thyroid/patient/thyroid-treatment-pdq"),
    ],
    "sarcoma": [
        ("Adult Soft Tissue Sarcoma Treatment", "NBK65870",
         "https://www.cancer.gov/types/soft-tissue-sarcoma/patient/adult-soft-tissue-treatment-pdq"),
        ("Osteosarcoma Treatment", "NBK65855",
         "https://www.cancer.gov/types/bone/patient/osteosarcoma-treatment-pdq"),
    ],
    "peds": [
        ("Childhood Acute Lymphoblastic Leukemia Treatment", "NBK65763",
         "https://www.cancer.gov/types/leukemia/patient/child-all-treatment-pdq"),
        ("Childhood Brain Tumors Treatment", "NBK65834",
         "https://www.cancer.gov/types/brain/patient/child-brain-treatment-pdq"),
    ],
}


USER_AGENT = "PaxisEvidenceIngestion/1.0 (+https://paxis.health)"


def fetch_ncbi_content(book_id: str) -> Optional[str]:
    """Fetch PDQ content from NCBI PubMed Books (server-rendered HTML)."""
    url = f"https://www.ncbi.nlm.nih.gov/books/{book_id}/"
    try:
        with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
        if resp.status_code >= 400:
            return None
        soup = BeautifulSoup(resp.content, "html.parser")

        # NCBI Books: main content is in <div class="body-content">
        # or the article/main area
        content_div = (
            soup.select_one(".body-content") or
            soup.select_one("#mc") or
            soup.select_one("article") or
            soup.select_one("[role='main']")
        )
        if not content_div:
            content_div = soup.body

        if not content_div:
            return None

        # Remove nav, footer, sidebars
        for tag in content_div.find_all(["nav", "footer", "aside", "script", "style"]):
            tag.decompose()

        # Extract text with headings preserved
        lines = []
        for el in content_div.descendants:
            if not hasattr(el, 'name') or el.name is None:
                continue
            if el.name in ('h1', 'h2', 'h3', 'h4'):
                text = el.get_text(strip=True)
                if text:
                    level = int(el.name[1])
                    lines.append(f"\n{'#' * level} {text}\n")
            elif el.name == 'p':
                text = el.get_text(strip=True)
                if text and len(text) > 20:
                    lines.append(text)
            elif el.name == 'li':
                text = el.get_text(strip=True)
                if text:
                    lines.append(f"- {text}")

        full_text = "\n".join(lines)
        # Only return if we got meaningful content
        if len(full_text.strip()) < 500:
            return None
        return full_text.strip()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None


def build_url_list(categories=None, resume_from=None):
    """Build list of PDQ summaries to ingest."""
    items = []
    started = resume_from is None
    for category, pdq_list in CATEGORY_TO_PDQ.items():
        if not started:
            if category == resume_from:
                started = True
            else:
                continue
        if categories and category not in categories:
            continue
        for title, book_id, cancer_gov_url in pdq_list:
            items.append({
                "title": title,
                "book_id": book_id,
                "url": cancer_gov_url,
                "category": category,
            })
    return items


async def seed_sources():
    from src.api.services.evidence.source_registry import get_source_registry
    registry = get_source_registry()
    rows = await registry.seed_default_sources()
    print(f"Registered/updated {len(rows)} evidence sources:")
    for r in rows:
        print(f"   {r['source_key']:20s} {r['name']}")


async def ingest_single(item: Dict) -> Dict:
    """Fetch from NCBI and ingest via ingest_document()."""
    from src.api.services.evidence.evidence_ingestion_service import get_evidence_ingestion_service

    service = get_evidence_ingestion_service()

    # Fetch content from NCBI (server-rendered)
    text = await asyncio.to_thread(fetch_ncbi_content, item["book_id"])
    if not text:
        return {"status": "no_content", "title": item["title"]}

    try:
        result = await service.ingest_document(
            source_key="nci",
            doc_id=item["url"],  # Use cancer.gov URL as stable doc_id
            title=item["title"],
            raw_text=text,
            url=item["url"],
        )
        if result.get("skipped"):
            return {"status": "skipped", "title": item["title"], "reason": result.get("reason")}
        return {
            "status": "ok",
            "title": item["title"],
            "chunks": result["chunks_ingested"],
            "collection": result["collection"],
        }
    except Exception as e:
        return {"status": "error", "title": item["title"], "error": str(e)}


async def run_ingestion(categories=None, resume_from=None, dry_run=False, delay=1.0):
    items = build_url_list(categories=categories, resume_from=resume_from)
    print(f"\n{'=' * 70}")
    print(f"  NCI PDQ PATIENT SUMMARY INGESTION")
    print(f"{'=' * 70}")
    print(f"  Total summaries to process: {len(items)}")
    print(f"  Categories: {categories or 'ALL'}")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
    print(f"  Delay between requests: {delay}s")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'=' * 70}\n")

    if dry_run:
        current_category = None
        for item in items:
            if item["category"] != current_category:
                current_category = item["category"]
                print(f"\n  [{current_category.upper()}]")
            print(f"    {item['title']} (NCBI: {item['book_id']})")
        print(f"\n  Total: {len(items)} PDQ summaries across "
              f"{len(set(i['category'] for i in items))} categories")
        return

    stats = {"ok": 0, "skipped": 0, "no_content": 0, "error": 0}
    current_category = None
    t_start = time.time()

    for i, item in enumerate(items, 1):
        if item["category"] != current_category:
            current_category = item["category"]
            print(f"\n{'─' * 70}")
            print(f"  [{current_category.upper()}]")
            print(f"{'─' * 70}")
        print(f"  [{i}/{len(items)}] {item['title']} ", end="", flush=True)

        result = await ingest_single(item)
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1

        if status == "ok":
            print(f"OK {result['chunks']} chunks -> {result['collection']}")
        elif status == "skipped":
            print(f"SKIP (unchanged)")
        elif status == "no_content":
            print(f"-- (no content from NCBI)")
        else:
            print(f"ERR {result.get('error', 'unknown')[:80]}")

        if i < len(items):
            await asyncio.sleep(delay)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {elapsed:.1f}s elapsed")
    print(f"{'=' * 70}")
    print(f"  OK Ingested:      {stats['ok']}")
    print(f"  SKIP Unchanged:   {stats['skipped']}")
    print(f"  -- No content:    {stats['no_content']}")
    print(f"  ERR Errors:       {stats['error']}")
    print(f"{'=' * 70}\n")


async def main():
    parser = argparse.ArgumentParser(description="Ingest NCI PDQ patient summaries", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-sources", action="store_true", help="Register default source metadata first")
    parser.add_argument("--category", type=str, action="append", default=None, help="Only ingest specific category")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from this category")
    parser.add_argument("--dry-run", action="store_true", help="Show summaries without fetching")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests (default 1.5)")
    args = parser.parse_args()

    if args.seed_sources:
        await seed_sources()
        print()

    await run_ingestion(categories=args.category, resume_from=args.resume_from, dry_run=args.dry_run, delay=args.delay)


if __name__ == "__main__":
    asyncio.run(main())
