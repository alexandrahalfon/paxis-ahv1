#!/usr/bin/env python3
"""
Ingest patient-education/medication content from approved sources into
the patient-facing Qdrant collections + evidence Postgres registry.

Usage:
    # Register the default source metadata (idempotent — safe to re-run)
    python scripts/ingest_evidence_source.py --seed-sources

    # Ingest specific URLs under a registered source
    python scripts/ingest_evidence_source.py --source nci --url https://www.cancer.gov/...

    # Ingest the small starter "golden corpus" defined below
    python scripts/ingest_evidence_source.py --golden-corpus

Every URL must belong to a source already registered in evidence_sources
(via --seed-sources or SourceRegistry.register_source) — this script
refuses to ingest from an unregistered source_key, which is what keeps
this from becoming a general-purpose crawler. See
src/api/services/evidence/source_registry.py.

GOLDEN_CORPUS below lists real, well-known URLs on cancer.gov/
medlineplus.gov as ingestion candidates — picked for topical relevance
(nutrition/taste changes, common symptoms, common oncology medications)
to match the flagship beta test scenarios. They have NOT been verified
reachable from this environment: this sandbox's egress proxy blocks
direct HTTP fetches to non-Anthropic domains (see source_fetcher.py's
docstring), so these URLs are unexecuted candidates, not a confirmed
ingested corpus. Check each resolves before relying on it, and expect to
prune/expand the list — it is intentionally small (starter-corpus
sized, not exhaustive) per the beta plan's "don't block RAG testing on
crawler engineering" guidance.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


GOLDEN_CORPUS = {
    "nci": [
        "https://www.cancer.gov/about-cancer/treatment/side-effects/appetite-loss",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/nausea",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/diarrhea",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/fatigue",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/hair-loss",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/mouth-throat",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/neuropathy",
        "https://www.cancer.gov/about-cancer/treatment/side-effects/skin-nail-changes",
    ],
    "medlineplus": [
        "https://medlineplus.gov/druginfo/meds/a695020.html",  # pembrolizumab
        "https://medlineplus.gov/druginfo/meds/a608043.html",  # oxaliplatin
        "https://medlineplus.gov/druginfo/meds/a682316.html",  # fluorouracil (topical entry; verify oncology entry separately)
    ],
    "cancer_net": [
        "https://www.cancer.net/coping-with-cancer/managing-emotions/anxiety",
    ],
}


async def seed_sources():
    from src.api.services.evidence.source_registry import get_source_registry

    registry = get_source_registry()
    rows = await registry.seed_default_sources()
    print(f"Registered/updated {len(rows)} evidence sources:")
    for r in rows:
        print(f"  {r['source_key']:20s} {r['name']}")


async def ingest_urls(source_key: str, urls: list[str]):
    from src.api.services.evidence.evidence_ingestion_service import get_evidence_ingestion_service
    from src.api.services.evidence.source_fetcher import FetchError

    service = get_evidence_ingestion_service()
    results = []
    for url in urls:
        print(f"\n[{source_key}] {url}")
        try:
            result = await service.ingest_url(source_key, url)
            if result.get("skipped"):
                print(f"  SKIPPED (unchanged): {result['reason']}")
            else:
                print(
                    f"  OK: {result['chunks_ingested']} chunks -> {result['collection']} "
                    f"(doc={result['document_id'][:8]}... version={result['version_id'][:8]}...)"
                )
            results.append({"url": url, **result})
        except FetchError as e:
            print(f"  FETCH FAILED: {e}")
            results.append({"url": url, "error": str(e)})
        except Exception as e:
            print(f"  INGESTION FAILED: {e}")
            results.append({"url": url, "error": str(e)})
    return results


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-sources", action="store_true", help="Register default source metadata first")
    parser.add_argument("--source", type=str, help="source_key to ingest under (see evidence_sources)")
    parser.add_argument("--url", action="append", default=[], help="URL to ingest (repeatable)")
    parser.add_argument("--golden-corpus", action="store_true", help="Ingest the starter corpus defined in this file")
    args = parser.parse_args()

    if args.seed_sources:
        await seed_sources()

    if args.golden_corpus:
        total_ok, total_skip, total_fail = 0, 0, 0
        for source_key, urls in GOLDEN_CORPUS.items():
            results = await ingest_urls(source_key, urls)
            for r in results:
                if "error" in r:
                    total_fail += 1
                elif r.get("skipped"):
                    total_skip += 1
                else:
                    total_ok += 1
        print(f"\n--- Golden corpus run complete: {total_ok} ingested, {total_skip} unchanged, {total_fail} failed ---")
        return

    if args.source and args.url:
        await ingest_urls(args.source, args.url)
        return

    if not args.seed_sources:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
