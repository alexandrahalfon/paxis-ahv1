#!/usr/bin/env python3
"""
Normalize Biomarker Status — one-time migration script

Parses existing molecular_subtype, extraction_data, and inclusion_criteria into
a structured biomarker_status JSONB column on the studies table.

This enables exact structured matching instead of regex over free text.

The biomarker_status JSONB looks like:
    {"ER": "positive", "PR": "positive", "HER2": "negative",
     "genomic_assay": "Oncotype DX", "score_range": "11-25"}

Usage:
    python scripts/normalize_biomarker_status.py [--dry-run]

Flags:
    --dry-run   Print what would be updated without writing to DB
"""

import asyncio
import asyncpg
import json
import os
import re
import sys
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Biomarker parsing patterns (shared with study_profile_storage_service.py)
# ---------------------------------------------------------------------------

_RECEPTOR_PATTERNS = [
    (r'\ber\s*[\+]\s*|estrogen\s*receptor\s*positive', "ER", "positive"),
    (r'\ber\s*[\-]\s*|estrogen\s*receptor\s*negative', "ER", "negative"),
    (r'\bpr\s*[\+]\s*|progesterone\s*receptor\s*positive', "PR", "positive"),
    (r'\bpr\s*[\-]\s*|progesterone\s*receptor\s*negative', "PR", "negative"),
    (r'her.?2\s*[\+]|her.?2\s*positive|her.?2\s*amplified', "HER2", "positive"),
    (r'her.?2\s*[\-]|her.?2\s*negative|her.?2\s*non.amplified', "HER2", "negative"),
    (r'triple\s*negative|tnbc', "TNBC", "positive"),
    (r'egfr\s*mutant|egfr\s*mutation|egfr\s*positive|egfr\s*\+', "EGFR", "mutant"),
    (r'egfr\s*wild.?type|egfr\s*negative|egfr\s*wt', "EGFR", "wild-type"),
    (r'alk\s*positive|alk\s*\+|alk\s*rearrang|alk\s*fusion', "ALK", "positive"),
    (r'alk\s*negative|alk\s*\-', "ALK", "negative"),
    (r'pd.?l1\s*positive|pd.?l1\s*\+|pd.?l1\s*high', "PD-L1", "positive"),
    (r'pd.?l1\s*negative|pd.?l1\s*\-|pd.?l1\s*low', "PD-L1", "negative"),
    (r'kras\s*mutant|kras\s*mutation|kras\s*\+', "KRAS", "mutant"),
    (r'kras\s*wild.?type|kras\s*wt', "KRAS", "wild-type"),
    (r'braf\s*v600|braf\s*mutant|braf\s*mutation|braf\s*\+', "BRAF", "mutant"),
    (r'braf\s*wild.?type|braf\s*wt', "BRAF", "wild-type"),
    (r'msi.?h|microsatellite\s*instability.?high', "MSI", "high"),
    (r'mss|microsatellite\s*stable|msi.?l', "MSI", "stable"),
    (r'brca\s*mutant|brca\s*mutation|brca1|brca2', "BRCA", "mutant"),
    (r'hpv\s*positive|hpv\s*\+|p16\s*positive|p16\s*\+', "HPV", "positive"),
    (r'hpv\s*negative|hpv\s*\-|p16\s*negative|p16\s*\-', "HPV", "negative"),
]

_ASSAY_PATTERNS = [
    (r'oncotype\s*dx', "Oncotype DX"),
    (r'mammaprint', "MammaPrint"),
    (r'decipher', "Decipher"),
    (r'prolaris', "Prolaris"),
    (r'endopredict', "EndoPredict"),
    (r'prosigna|pam\s*50', "Prosigna"),
]


def parse_biomarker_status(
    molecular_subtype: str | None,
    extraction_data: dict | None,
) -> dict | None:
    """Parse molecular_subtype and extraction_data into normalized biomarker status."""
    result = {}

    # 1. Parse from molecular_subtype free text
    if molecular_subtype:
        mol_lower = molecular_subtype.lower()
        for pattern, name, status in _RECEPTOR_PATTERNS:
            if re.search(pattern, mol_lower):
                if name not in result:
                    result[name] = status

    if not extraction_data:
        return result if result else None

    # 2. Parse from biomarker_inclusion_criteria (if present from new extraction)
    bic = extraction_data.get("biomarker_inclusion_criteria", {})
    if isinstance(bic, dict):
        for req in bic.get("required_biomarkers", []):
            if isinstance(req, dict) and req.get("name") and req.get("status"):
                name = req["name"].upper().strip()
                status = req["status"].lower().strip()
                result[name] = status

        genomic_assay_data = bic.get("genomic_assay", {})
        if isinstance(genomic_assay_data, dict) and genomic_assay_data.get("value"):
            result["genomic_assay"] = genomic_assay_data["value"]

        score_range_data = bic.get("score_range", {})
        if isinstance(score_range_data, dict) and score_range_data.get("value"):
            result["score_range"] = score_range_data["value"]

    # 3. Parse from diagnosis.molecular_subtype in extraction_data
    diag = extraction_data.get("diagnosis", {})
    mol_data = diag.get("molecular_subtype", {})
    mol_text = None
    if isinstance(mol_data, dict):
        mol_text = mol_data.get("value")
    elif isinstance(mol_data, str):
        mol_text = mol_data

    if mol_text:
        for pattern, name, status in _RECEPTOR_PATTERNS:
            if re.search(pattern, mol_text.lower()):
                if name not in result:
                    result[name] = status

    # 4. Parse from inclusion criteria text
    patient_chars = extraction_data.get("patient_characteristics", {})
    for criterion in patient_chars.get("inclusion_criteria", []):
        crit_text = None
        if isinstance(criterion, dict):
            crit_text = criterion.get("criterion") or criterion.get("value")
        elif isinstance(criterion, str):
            crit_text = criterion

        if crit_text:
            for pattern, name, status in _RECEPTOR_PATTERNS:
                if re.search(pattern, crit_text.lower()):
                    if name not in result:
                        result[name] = status

    # 5. Check for genomic assay mentions
    if "genomic_assay" not in result:
        full_text = json.dumps(extraction_data).lower()
        for pattern, assay_name in _ASSAY_PATTERNS:
            if re.search(pattern, full_text):
                result["genomic_assay"] = assay_name
                score_match = re.search(
                    pattern + r'[^.]{0,80}(?:score|rs)\s*(?:of\s*)?(\d+\s*[-–]\s*\d+|\d+|[<>≤≥]\s*\d+)',
                    full_text
                )
                if score_match and "score_range" not in result:
                    result["score_range"] = score_match.group(1).strip()
                break

    return result if result else None


async def main():
    dry_run = "--dry-run" in sys.argv

    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "34.21.60.224"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DATABASE", "display-study-details"),
    )

    print("=" * 60)
    print(f"NORMALIZE BIOMARKER STATUS {'(DRY RUN)' if dry_run else ''}")
    print("=" * 60)

    # Ensure columns exist
    if not dry_run:
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='studies' AND column_name='biomarker_status') THEN
                    ALTER TABLE studies ADD COLUMN biomarker_status JSONB;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='studies' AND column_name='genomic_assay') THEN
                    ALTER TABLE studies ADD COLUMN genomic_assay TEXT;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='studies' AND column_name='genomic_score_range') THEN
                    ALTER TABLE studies ADD COLUMN genomic_score_range TEXT;
                END IF;
            END $$;
        """)
        # Ensure GIN index exists
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_biomarker_status_gin ON studies USING GIN (biomarker_status);
            CREATE INDEX IF NOT EXISTS idx_studies_genomic_assay ON studies(genomic_assay);
        """)
        print("Schema updated.\n")

    # Fetch all studies
    rows = await conn.fetch("""
        SELECT study_id, document_name, molecular_subtype, extraction_data
        FROM studies
        ORDER BY study_id
    """)
    print(f"Found {len(rows)} studies to process.\n")

    updated = 0
    skipped = 0
    with_biomarkers = 0

    for row in rows:
        study_id = row["study_id"]
        mol_subtype = row["molecular_subtype"]
        extraction_data = row["extraction_data"]

        # Parse extraction_data from JSON if it's a string
        if isinstance(extraction_data, str):
            try:
                extraction_data = json.loads(extraction_data)
            except (json.JSONDecodeError, TypeError):
                extraction_data = None

        biomarker_status = parse_biomarker_status(mol_subtype, extraction_data)

        if biomarker_status:
            with_biomarkers += 1
            genomic_assay = biomarker_status.get("genomic_assay")
            genomic_score_range = biomarker_status.get("score_range")

            if dry_run:
                print(f"  [{study_id}] {row['document_name'][:50]}")
                print(f"    mol_subtype: {mol_subtype}")
                print(f"    -> biomarker_status: {json.dumps(biomarker_status)}")
                if genomic_assay:
                    print(f"    -> genomic_assay: {genomic_assay}, score_range: {genomic_score_range}")
            else:
                await conn.execute("""
                    UPDATE studies
                    SET biomarker_status = $2::jsonb,
                        genomic_assay = $3,
                        genomic_score_range = $4
                    WHERE study_id = $1
                """, study_id, json.dumps(biomarker_status), genomic_assay, genomic_score_range)

            updated += 1
        else:
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total studies:     {len(rows)}")
    print(f"  With biomarkers:   {with_biomarkers}")
    print(f"  Updated:           {updated}")
    print(f"  Skipped (no data): {skipped}")

    if not dry_run and updated > 0:
        # Show sample results
        print(f"\nSample normalized biomarker_status:")
        samples = await conn.fetch("""
            SELECT document_name, molecular_subtype, biomarker_status, genomic_assay
            FROM studies
            WHERE biomarker_status IS NOT NULL
            LIMIT 10
        """)
        for s in samples:
            print(f"  {s['document_name'][:40]}")
            print(f"    mol: {s['molecular_subtype']}")
            print(f"    bio: {s['biomarker_status']}")
            if s['genomic_assay']:
                print(f"    assay: {s['genomic_assay']}")

    if dry_run:
        print(f"\n(Dry run — no changes written. Remove --dry-run to apply.)")

    await conn.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
