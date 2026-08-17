#!/usr/bin/env python3
"""
Ingest NCI PDQ supportive care, side effects, and general treatment summaries.

Companion to ingest_nci_cancer_types.py (which covers cancer-type-specific
treatment/prevention/screening). This script covers the cross-cutting topics:
- Supportive & palliative care (pain, fatigue, nausea, nutrition, etc.)
- Coping & emotional support (anxiety, depression, grief)
- General treatment information (side effects overview)

Same approach: fetches from NCBI PubMed Books (server-rendered HTML),
ingests via ingest_document() into oncology_patient_education.

Usage:
    python scripts/ingest_nci_supportive_care.py --dry-run
    python scripts/ingest_nci_supportive_care.py
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx
from bs4 import BeautifulSoup


# ═══════════════════════════════════════════════════════════════════════
# PDQ SUPPORTIVE CARE & GENERAL SUMMARIES
# ═══════════════════════════════════════════════════════════════════════
# (title, ncbi_book_id, cancer.gov_url_for_attribution)

SUPPORTIVE_CARE_PDQ: List[Tuple[str, str, str]] = [
    # ── Side Effects & Symptom Management ───────────────────────────
    ("Cancer Pain", "NBK65960",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/pain/pain-pdq"),
    ("Fatigue", "NBK65946",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/fatigue/fatigue-pdq"),
    ("Nausea and Vomiting", "NBK66056",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/nausea/nausea-pdq"),
    ("Oral Complications of Chemotherapy and Radiation", "NBK66016",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/mouth-throat/oral-complications-pdq"),
    ("Gastrointestinal Complications", "NBK65975",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/constipation/gi-complications-pdq"),
    ("Lymphedema", "NBK65893",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/lymphedema/lymphedema-pdq"),
    ("Pruritus (Itching)", "NBK66058",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/skin-nails/pruritus-pdq"),
    ("Hot Flashes and Night Sweats", "NBK65986",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/hot-flashes-pdq"),
    ("Sleep Disorders", "NBK65956",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/sleep-disorders-pdq"),
    ("Cardiopulmonary Syndromes", "NBK65899",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/cardiopulmonary-syndromes-pdq"),
    ("Delirium", "NBK65922",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/delirium-pdq"),
    ("Cognitive Impairment in Adults with Cancer", "NBK572188",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/cognitive-impairment-pdq"),
    ("Sexuality and Reproductive Issues", "NBK65958",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/sexuality-reproductive-issues-pdq"),
    ("Fever, Sweats, and Hot Flashes", "NBK65986",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/fever-sweats-pdq"),

    # ── Nutrition ───────────────────────────────────────────────────
    ("Nutrition in Cancer Care", "NBK65854",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/appetite-loss/nutrition-pdq"),
    ("Cancer Therapy Interactions With Foods and Dietary Supplements", "NBK572880",
     "https://www.cancer.gov/about-cancer/treatment/side-effects/appetite-loss/food-interactions-pdq"),

    # ── Coping & Emotional Support ──────────────────────────────────
    ("Adjustment to Cancer: Anxiety and Distress", "NBK65907",
     "https://www.cancer.gov/about-cancer/coping/feelings/anxiety-distress-pdq"),
    ("Depression", "NBK65891",
     "https://www.cancer.gov/about-cancer/coping/feelings/depression-pdq"),
    ("Cancer-Related Post-traumatic Stress", "NBK65900",
     "https://www.cancer.gov/about-cancer/coping/survivorship/post-traumatic-stress-pdq"),
    ("Grief, Bereavement, and Coping With Loss", "NBK65966",
     "https://www.cancer.gov/about-cancer/advanced-cancer/caregivers/planning/bereavement-pdq"),
    ("Communication in Cancer Care", "NBK65907",
     "https://www.cancer.gov/about-cancer/coping/adjusting-to-cancer/communication-pdq"),
    ("Informal Caregivers in Cancer", "NBK65802",
     "https://www.cancer.gov/about-cancer/coping/caregiver-support/caregivers-pdq"),
    ("Spirituality in Cancer Care", "NBK65876",
     "https://www.cancer.gov/about-cancer/coping/day-to-day/faith-and-spirituality/spirituality-pdq"),

    # ── End of Life / Advanced Cancer ───────────────────────────────
    ("Last Days of Life", "NBK66035",
     "https://www.cancer.gov/about-cancer/advanced-cancer/care-choices/last-days-pdq"),
    ("Planning the Transition to End-of-Life Care", "NBK65939",
     "https://www.cancer.gov/about-cancer/advanced-cancer/care-choices/planning-pdq"),

    # ── Pediatric Supportive Care ───────────────────────────────────
    ("Childhood Cancer Late Effects", "NBK65814",
     "https://www.cancer.gov/types/childhood-cancers/late-effects-pdq"),
]


USER_AGENT = "PaxisEvidenceIngestion/1.0 (+https://paxis.health)"


def fetch_ncbi_content(book_id: str):
    """Fetch PDQ content from NCBI PubMed Books (server-rendered HTML)."""
    url = f"https://www.ncbi.nlm.nih.gov/books/{book_id}/"
    try:
        with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
        if resp.status_code >= 400:
            return None
        soup = BeautifulSoup(resp.content, "html.parser")

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

        for tag in content_div.find_all(["nav", "footer", "aside", "script", "style"]):
            tag.decompose()

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
        if len(full_text.strip()) < 500:
            return None
        return full_text.strip()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None


async def ingest_single(item) -> Dict:
    from src.api.services.evidence.evidence_ingestion_service import get_evidence_ingestion_service

    service = get_evidence_ingestion_service()
    title, book_id, url = item

    text = await asyncio.to_thread(fetch_ncbi_content, book_id)
    if not text:
        return {"status": "no_content", "title": title}

    try:
        result = await service.ingest_document(
            source_key="nci",
            doc_id=url,
            title=title,
            raw_text=text,
            url=url,
        )
        if result.get("skipped"):
            return {"status": "skipped", "title": title}
        return {"status": "ok", "title": title, "chunks": result["chunks_ingested"], "collection": result["collection"]}
    except Exception as e:
        return {"status": "error", "title": title, "error": str(e)}


async def run_ingestion(dry_run=False, delay=1.5):
    items = SUPPORTIVE_CARE_PDQ
    print(f"\n{'=' * 70}")
    print(f"  NCI SUPPORTIVE CARE & GENERAL TOPICS INGESTION")
    print(f"{'=' * 70}")
    print(f"  Total summaries: {len(items)}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'=' * 70}\n")

    if dry_run:
        for title, book_id, url in items:
            print(f"  {title} (NCBI: {book_id})")
        print(f"\n  Total: {len(items)} PDQ summaries")
        return

    stats = {"ok": 0, "skipped": 0, "no_content": 0, "error": 0}
    t_start = time.time()

    for i, item in enumerate(items, 1):
        title = item[0]
        print(f"  [{i}/{len(items)}] {title} ", end="", flush=True)

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
    parser = argparse.ArgumentParser(description="Ingest NCI supportive care PDQ summaries")
    parser.add_argument("--dry-run", action="store_true", help="Show summaries without fetching")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    args = parser.parse_args()
    await run_ingestion(dry_run=args.dry_run, delay=args.delay)


if __name__ == "__main__":
    asyncio.run(main())
