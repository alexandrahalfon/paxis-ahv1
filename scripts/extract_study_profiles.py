#!/usr/bin/env python3
"""
Extract Study Profiles from Processed Documents

This script extracts structured study profiles from processed document
directories and stores them in PostgreSQL for the study details display.

Usage:
    # Extract from a single document directory
    python scripts/extract_study_profiles.py --doc-dir processed_documents/my_study

    # Extract from all documents in a directory
    python scripts/extract_study_profiles.py --input-dir processed_documents/

    # Extract and save to JSON (without PostgreSQL)
    python scripts/extract_study_profiles.py --doc-dir processed_documents/my_study --json-only

    # Re-upsert existing rows (overwrite stale Postgres data)
    python scripts/extract_study_profiles.py --input-dir processed_documents/ --force

doc_id: the Postgres `studies.doc_id` column is written with the same
normalized + md5-suffixed id that Qdrant payloads use (see
`src/ingestion/doc_id.py`), so the two sides join cleanly.

Source: if `<doc_dir>/<doc_dir.name>_study_profile.json` already exists on
disk (produced during ingestion), it is loaded directly and no LLM call is
made. Otherwise the script falls back to `StudyProfileExtractor`.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.study_profile_extractor import StudyProfileExtractor
from src.api.services.study_profile_storage_service import get_study_profile_storage_service
from src.ingestion.doc_id import normalize_doc_id


def _load_profile_from_disk(doc_dir: Path) -> dict | None:
    """Return the ingestion-time profile JSON for this doc, or None."""
    json_path = doc_dir / f"{doc_dir.name}_study_profile.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠️  Ignoring unreadable {json_path.name}: {e}")
        return None


async def extract_single_document(
    doc_dir: Path,
    extractor: StudyProfileExtractor,
    save_to_postgres: bool = True,
    output_dir: Path = None,
    force: bool = False,
) -> dict:
    """Extract study profile from a single document directory.

    Prefers a pre-computed ``<dir>/<name>_study_profile.json`` on disk.
    Falls back to ``StudyProfileExtractor`` (LLM call) only if absent.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {doc_dir.name}")
    print(f"{'='*60}")

    result = _load_profile_from_disk(doc_dir)
    if result is not None:
        print(f"  📂 Loaded {doc_dir.name}_study_profile.json from disk")
    else:
        print(f"  🔄 No on-disk profile — running extractor")
        result = extractor.extract_from_processed_dir(doc_dir)

    if result.get("error"):
        print(f"  ❌ Extraction failed: {result['error']}")
        return result

    extracted_data = result.get("extracted_data")
    if not extracted_data:
        print(f"  ❌ No data extracted")
        return result

    # Save to JSON if output_dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{doc_dir.name}_profile.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  📄 Saved JSON: {json_path}")

    # Save to PostgreSQL
    if save_to_postgres:
        try:
            storage = get_study_profile_storage_service()

            # Match the normalized doc_id Qdrant writes into chunk payloads
            # so PG ↔ Qdrant joins on doc_id just work.
            doc_id = normalize_doc_id(doc_dir.name)

            study_id = await storage.store_study_profile(
                doc_id=doc_id,
                document_name=doc_dir.name,
                extracted_data=extracted_data,
                processing_duration=result.get("processing_duration_seconds"),
                api_usage=result.get("api_usage") or extractor.get_usage_summary(),
                force=force,
            )

            print(f"  💾 Stored in PostgreSQL: study_id={study_id} doc_id={doc_id}")
            result["study_id"] = study_id
            result["doc_id"] = doc_id

        except Exception as e:
            print(f"  ⚠️  PostgreSQL storage failed: {e}")
            result["storage_error"] = str(e)

    return result


async def extract_all_documents(
    input_dir: Path,
    save_to_postgres: bool = True,
    output_dir: Path = None,
    force: bool = False,
) -> dict:
    """Extract study profiles from all document directories."""
    print(f"\n{'='*60}")
    print(f"STUDY PROFILE EXTRACTION PIPELINE")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir or 'PostgreSQL only'}")
    print(f"{'='*60}\n")
    
    # Find document directories
    doc_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not doc_dirs:
        print(f"❌ No document directories found in {input_dir}")
        return {"error": "No documents found"}
    
    print(f"Found {len(doc_dirs)} document(s)\n")
    
    extractor = StudyProfileExtractor()
    results = []
    successful = 0
    failed = 0
    
    for i, doc_dir in enumerate(doc_dirs, 1):
        print(f"\n[{i}/{len(doc_dirs)}]", end="")
        
        result = await extract_single_document(
            doc_dir=doc_dir,
            extractor=extractor,
            save_to_postgres=save_to_postgres,
            output_dir=output_dir,
            force=force,
        )

        results.append(result)
        
        if result.get("extracted_data"):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total:      {len(doc_dirs)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"\nAPI Usage:")
    usage = extractor.get_usage_summary()
    print(f"  Tokens: {usage['total_tokens']:,}")
    print(f"  Cost:   ${usage['total_cost_usd']:.4f}")
    print(f"{'='*60}\n")
    
    return {
        "total": len(doc_dirs),
        "successful": successful,
        "failed": failed,
        "api_usage": usage,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract study profiles from processed documents"
    )
    parser.add_argument(
        "--doc-dir",
        type=Path,
        help="Single document directory to process"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing multiple document directories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for JSON files (optional)"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Save to JSON only, skip PostgreSQL"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upsert: delete and reinsert rows that already exist for this doc_id"
    )

    args = parser.parse_args()
    
    if not args.doc_dir and not args.input_dir:
        parser.error("Either --doc-dir or --input-dir is required")
    
    save_to_postgres = not args.json_only
    
    if args.doc_dir:
        if not args.doc_dir.exists():
            print(f"❌ Directory not found: {args.doc_dir}")
            sys.exit(1)
        
        extractor = StudyProfileExtractor()
        result = asyncio.run(extract_single_document(
            doc_dir=args.doc_dir,
            extractor=extractor,
            save_to_postgres=save_to_postgres,
            output_dir=args.output_dir,
            force=args.force,
        ))
        
        if result.get("error"):
            sys.exit(1)
    
    elif args.input_dir:
        if not args.input_dir.exists():
            print(f"❌ Directory not found: {args.input_dir}")
            sys.exit(1)
        
        summary = asyncio.run(extract_all_documents(
            input_dir=args.input_dir,
            save_to_postgres=save_to_postgres,
            output_dir=args.output_dir,
            force=args.force,
        ))
        
        if summary.get("failed", 0) > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
