#!/usr/bin/env python3
"""
Audit biomarker_status JSONB column for RAG Pipeline Consolidation (Phase 0).

Validates Requirement 1.4:
  Sample 50 rows from studies.biomarker_status, check key spelling consistency
  (e.g., "EGFR" vs "egfr" vs "Egfr"), and log findings.

Output is saved to tests/fixtures/consolidation_baselines/biomarker_audit.json

Usage:
    python tests/fixtures/consolidation_baselines/audit_biomarker_status.py
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Expected canonical keys — the "correct" casing for each biomarker
# ---------------------------------------------------------------------------

CANONICAL_KEYS = {
    "ER", "PR", "HER2", "EGFR", "ALK", "KRAS", "BRAF", "BRCA",
    "BRCA1", "BRCA2", "PD-L1", "MSI", "TMB", "HPV", "TNBC",
    "ROS1", "NTRK", "MET", "RET", "FGFR", "PIK3CA",
    "genomic_assay", "score_range",
}

# Map lowercased variants → canonical form for inconsistency detection
_CANONICAL_LOOKUP = {k.lower(): k for k in CANONICAL_KEYS}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_pg_dsn() -> str:
    """Build a PostgreSQL DSN from environment / settings."""
    from dotenv import load_dotenv
    load_dotenv()

    host = os.getenv("POSTGRES_HOST", "34.21.60.224")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DATABASE", "display-study-details")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


async def _sample_biomarker_rows(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch `limit` rows from studies where biomarker_status IS NOT NULL.

    Returns a list of dicts, each with keys: doc_id, biomarker_status.
    """
    import asyncpg

    dsn = _get_pg_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT doc_id, biomarker_status
            FROM studies
            WHERE biomarker_status IS NOT NULL
            ORDER BY random()
            LIMIT $1
            """,
            limit,
        )
        results = []
        for row in rows:
            bm = row["biomarker_status"]
            # asyncpg returns JSONB as a string or dict depending on version
            if isinstance(bm, str):
                bm = json.loads(bm)
            results.append({
                "doc_id": row["doc_id"],
                "biomarker_status": bm,
            })
        return results
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def _group_keys_by_canonical(all_keys: List[str]) -> Dict[str, List[str]]:
    """Group observed keys by their lowercased canonical form.

    Returns a dict mapping canonical_lower → list of observed spellings.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in all_keys:
        groups[key.lower()].append(key)
    return dict(groups)


def _find_inconsistencies(
    key_groups: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Detect spelling inconsistencies across observed keys.

    An inconsistency is when the same biomarker appears with different
    casings or hyphenation variants (e.g., "EGFR" vs "egfr" vs "Egfr").
    """
    inconsistencies = []
    for lower_key, variants in key_groups.items():
        unique_variants = sorted(set(variants))
        if len(unique_variants) > 1:
            canonical = _CANONICAL_LOOKUP.get(lower_key, unique_variants[0])
            inconsistencies.append({
                "canonical": canonical,
                "variants_found": unique_variants,
                "variant_count": len(unique_variants),
                "total_occurrences": len(variants),
            })
    return inconsistencies


def audit_biomarker_keys(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full audit on sampled biomarker_status rows.

    Returns the audit result dict suitable for JSON serialization.
    """
    all_keys: List[str] = []
    key_frequency: Dict[str, int] = defaultdict(int)
    rows_with_keys: int = 0
    sample_entries: List[Dict[str, Any]] = []

    for row in rows:
        bm = row.get("biomarker_status")
        if not isinstance(bm, dict):
            continue
        rows_with_keys += 1
        keys = list(bm.keys())
        all_keys.extend(keys)
        for k in keys:
            key_frequency[k] += 1
        sample_entries.append({
            "doc_id": row["doc_id"],
            "keys": keys,
        })

    unique_keys = sorted(set(all_keys))
    key_groups = _group_keys_by_canonical(all_keys)
    inconsistencies = _find_inconsistencies(key_groups)

    # Check which keys are not in the canonical set
    non_canonical = []
    for key in unique_keys:
        if key not in CANONICAL_KEYS:
            expected = _CANONICAL_LOOKUP.get(key.lower())
            non_canonical.append({
                "observed": key,
                "expected_canonical": expected,
                "is_case_mismatch": expected is not None and key != expected,
                "is_unknown": expected is None,
            })

    return {
        "audit_summary": {
            "rows_sampled": len(rows),
            "rows_with_biomarker_keys": rows_with_keys,
            "total_key_occurrences": len(all_keys),
            "unique_keys_observed": unique_keys,
            "unique_key_count": len(unique_keys),
        },
        "key_frequency": dict(sorted(
            key_frequency.items(), key=lambda x: -x[1]
        )),
        "inconsistencies": inconsistencies,
        "inconsistency_count": len(inconsistencies),
        "non_canonical_keys": non_canonical,
        "sample_entries": sample_entries[:10],  # first 10 for reference
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

AUDIT_OUTPUT_PATH = Path(__file__).parent / "biomarker_audit.json"


def save_audit(audit_result: Dict[str, Any]) -> Path:
    """Save audit results to JSON."""
    AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_OUTPUT_PATH, "w") as f:
        json.dump(audit_result, f, indent=2, default=str)
    print(f"[BiomarkerAudit] Saved audit → {AUDIT_OUTPUT_PATH}")
    return AUDIT_OUTPUT_PATH


def load_audit() -> Optional[Dict[str, Any]]:
    """Load existing audit results from JSON."""
    if not AUDIT_OUTPUT_PATH.exists():
        return None
    with open(AUDIT_OUTPUT_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_audit(sample_size: int = 50) -> Dict[str, Any]:
    """Run the full biomarker_status audit against the live database.

    Connects to PostgreSQL, samples rows, analyzes key consistency,
    and saves results to biomarker_audit.json.
    """
    print("=" * 70)
    print("  Phase 0 — Biomarker Status JSONB Audit")
    print("=" * 70)

    print(f"\n[BiomarkerAudit] Sampling {sample_size} rows from studies.biomarker_status...")
    try:
        rows = await _sample_biomarker_rows(limit=sample_size)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[BiomarkerAudit] DB connection failed: {e}")
        print("[BiomarkerAudit] Using placeholder audit results.")
        return _placeholder_audit()

    if not rows:
        print("[BiomarkerAudit] No rows with biomarker_status found.")
        return _placeholder_audit()

    print(f"[BiomarkerAudit] Fetched {len(rows)} rows, analyzing keys...")
    audit_result = audit_biomarker_keys(rows)

    # Print summary
    summary = audit_result["audit_summary"]
    print(f"  Rows with biomarker keys: {summary['rows_with_biomarker_keys']}/{summary['rows_sampled']}")
    print(f"  Unique keys observed: {summary['unique_key_count']}")
    print(f"  Key frequency: {audit_result['key_frequency']}")

    if audit_result["inconsistencies"]:
        print(f"\n  INCONSISTENCIES FOUND ({audit_result['inconsistency_count']}):")
        for inc in audit_result["inconsistencies"]:
            print(f"    {inc['canonical']}: {inc['variants_found']}")
    else:
        print("\n  No key spelling inconsistencies found.")

    if audit_result["non_canonical_keys"]:
        print(f"\n  NON-CANONICAL KEYS ({len(audit_result['non_canonical_keys'])}):")
        for nc in audit_result["non_canonical_keys"]:
            label = "case mismatch" if nc["is_case_mismatch"] else "unknown key"
            print(f"    {nc['observed']} ({label}, expected: {nc['expected_canonical']})")

    save_audit(audit_result)
    print("\n" + "=" * 70)
    return audit_result


def _placeholder_audit() -> Dict[str, Any]:
    """Return a placeholder audit when DB is unavailable."""
    return {
        "audit_summary": {
            "rows_sampled": 0,
            "rows_with_biomarker_keys": 0,
            "total_key_occurrences": 0,
            "unique_keys_observed": [],
            "unique_key_count": 0,
            "note": "Placeholder — no live DB connection available",
        },
        "key_frequency": {},
        "inconsistencies": [],
        "inconsistency_count": 0,
        "non_canonical_keys": [],
        "sample_entries": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit biomarker_status JSONB key spelling consistency"
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of rows to sample (default: 50)",
    )
    parser.add_argument(
        "--load", action="store_true",
        help="Load and print existing audit results instead of running live",
    )
    args = parser.parse_args()

    if args.load:
        result = load_audit()
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("No existing audit found. Run without --load to generate.")
    else:
        asyncio.run(run_audit(sample_size=args.sample_size))
