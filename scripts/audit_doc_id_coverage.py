#!/usr/bin/env python3
"""
Audit coverage / parity of doc_ids between Qdrant and Postgres `studies`.

Both sides use the same normalized doc_id (see ``src/ingestion/doc_id.py``),
so these modes join cleanly on ``doc_id``.

Modes
-----
  coverage                     Report Qdrant ↔ Postgres doc_id intersection
                               and which docs live in only one side.
  diff <doc_id>                Show Postgres row vs. a few Qdrant chunks for
                               one doc, highlighting mismatches in the
                               headline keyword / criteria fields.
  llm-audit [--sample N ...]   Thin wrapper around
                               scripts/audit_qdrant_postgres_consistency.py
                               for content-level verification via LLM.

Examples
--------
  python scripts/audit_doc_id_coverage.py coverage
  python scripts/audit_doc_id_coverage.py coverage --out coverage.json
  python scripts/audit_doc_id_coverage.py diff my_study_abcd1234
  python scripts/audit_doc_id_coverage.py llm-audit --sample 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from src.core.config import settings
from src.api.services.study_profile_storage_service import (
    get_study_profile_storage_service,
)


# ── Qdrant helpers ────────────────────────────────────────────────────────

def _qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def _scroll_all_doc_ids(client: QdrantClient, batch_size: int = 1000) -> Set[str]:
    """Return the set of unique `doc_id`s present in the configured collection."""
    ids: Set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            did = (p.payload or {}).get("doc_id")
            if did:
                ids.add(did)
        if offset is None:
            break
    return ids


def _scroll_chunks_for_doc(
    client: QdrantClient, doc_id: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Pull a handful of chunk payloads for a given doc_id."""
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            )]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [dict(p.payload or {}) for p in points]


# ── Mode A: coverage ──────────────────────────────────────────────────────

async def _mode_coverage(out_path: Optional[Path]) -> int:
    print("Scrolling Qdrant for doc_ids…")
    client = _qdrant_client()
    qdrant_ids = _scroll_all_doc_ids(client)
    print(f"  Qdrant: {len(qdrant_ids)} unique doc_ids")

    storage = get_study_profile_storage_service()
    pg_rows = await storage.list_studies_by_doc_ids(list(qdrant_ids))
    pg_ids_in_qdrant = set(pg_rows.keys())

    # Also collect PG doc_ids that are NOT in Qdrant (to list "pg_only")
    db = storage
    # Cheap way: fetch all distinct PG doc_ids via a raw query through the pool.
    from src.api.services.account_db import get_account_db
    pool = await get_account_db().get_pool()
    async with pool.acquire() as conn:
        pg_all_rows = await conn.fetch(
            "SELECT doc_id FROM studies WHERE doc_id IS NOT NULL"
        )
    pg_ids_all = {r["doc_id"] for r in pg_all_rows}

    intersection = qdrant_ids & pg_ids_all
    qdrant_only = sorted(qdrant_ids - pg_ids_all)
    pg_only = sorted(pg_ids_all - qdrant_ids)

    print()
    print(f"  Postgres studies:     {len(pg_ids_all)}")
    print(f"  Intersection:         {len(intersection)}")
    print(f"  Qdrant only (no PG):  {len(qdrant_only)}")
    print(f"  PG only (no Qdrant):  {len(pg_only)}")
    print()
    if qdrant_only:
        print("Qdrant-only sample (up to 10):")
        for d in qdrant_only[:10]:
            print(f"  - {d}")
    if pg_only:
        print("PG-only sample (up to 10):")
        for d in pg_only[:10]:
            print(f"  - {d}")

    if out_path:
        out_path.write_text(json.dumps({
            "qdrant_count": len(qdrant_ids),
            "pg_count": len(pg_ids_all),
            "intersection_count": len(intersection),
            "qdrant_only": qdrant_only,
            "pg_only": pg_only,
        }, indent=2))
        print(f"\nWrote {out_path}")
    return 0


# ── Mode B: diff ──────────────────────────────────────────────────────────

_DIFF_FIELDS = [
    # (label, PG column, list of Qdrant payload keys to consult)
    ("title",         "study_name",         ["doc_meta.title"]),
    ("cancer_type",   "cancer_type",        ["doc_level_cancer_types", "metadata.cancer_types_detected"]),
    ("biomarkers",    "biomarker_status",   ["doc_level_biomarkers", "metadata.biomarkers_detected"]),
    ("drugs",         None,                 ["doc_level_drugs", "metadata.drugs_detected"]),
    ("study_phase",   "study_phase",        ["doc_level_study_phase"]),
    ("metastatic",    "metastatic_status",  ["doc_level_metastatic_status"]),
    ("year",          "publish_date",       ["doc_meta.year"]),
    ("doi",           "doi",                ["doc_meta.doi"]),
]


def _deep_get(payload: Dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _aggregate_qdrant(chunks: List[Dict[str, Any]], keys: List[str]) -> Any:
    """Collect non-empty values for any of ``keys`` across chunks."""
    seen: List[Any] = []
    for ch in chunks:
        for k in keys:
            val = _deep_get(ch, k)
            if val in (None, "", [], {}):
                continue
            if isinstance(val, list):
                for v in val:
                    if v not in seen:
                        seen.append(v)
            else:
                if val not in seen:
                    seen.append(val)
    return seen if len(seen) != 1 else seen[0]


async def _mode_diff(doc_id: str) -> int:
    client = _qdrant_client()
    chunks = _scroll_chunks_for_doc(client, doc_id, limit=5)
    if not chunks:
        print(f"⚠️  No Qdrant chunks for doc_id={doc_id}")

    storage = get_study_profile_storage_service()
    pg_row = await storage.get_study_by_doc_id(doc_id)
    if not pg_row:
        print(f"⚠️  No Postgres row for doc_id={doc_id}")

    print(f"doc_id: {doc_id}")
    print(f"Qdrant chunks sampled: {len(chunks)}")
    print(f"Postgres row: {'found' if pg_row else 'missing'}")
    print()
    print(f"{'field':<14} {'postgres':<40} qdrant")
    print("-" * 100)
    for label, pg_col, qdrant_keys in _DIFF_FIELDS:
        pg_val = pg_row.get(pg_col) if (pg_row and pg_col) else None
        q_val = _aggregate_qdrant(chunks, qdrant_keys) if chunks else None
        pg_s = _fmt(pg_val)
        q_s = _fmt(q_val)
        marker = "  " if _loose_eq(pg_val, q_val) else "⚠ "
        print(f"{marker}{label:<12} {pg_s:<40} {q_s}")
    return 0


def _fmt(val: Any) -> str:
    if val is None:
        return "(none)"
    if isinstance(val, (dict, list)):
        s = json.dumps(val, ensure_ascii=False)
    else:
        s = str(val)
    return s if len(s) <= 38 else s[:35] + "…"


def _loose_eq(a: Any, b: Any) -> bool:
    """Both empty-ish → equal. Both containers → any overlap → equal."""
    empty_a = a in (None, "", [], {})
    empty_b = b in (None, "", [], {})
    if empty_a and empty_b:
        return True
    if empty_a or empty_b:
        return False
    if isinstance(a, list) and isinstance(b, list):
        return any(x in b for x in a)
    if isinstance(a, list):
        return b in a
    if isinstance(b, list):
        return a in b
    return str(a).strip().lower() == str(b).strip().lower()


# ── Mode C: LLM audit (delegates) ─────────────────────────────────────────

def _mode_llm_audit(forwarded: List[str]) -> int:
    target = Path(__file__).parent / "audit_qdrant_postgres_consistency.py"
    if not target.exists():
        print(f"❌ {target} not found — cannot run llm-audit")
        return 2
    cmd = [sys.executable, str(target), *forwarded]
    print(f"↳ Delegating to: {' '.join(cmd)}")
    return subprocess.call(cmd)


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    cov = sub.add_parser("coverage", help="Qdrant vs Postgres doc_id coverage")
    cov.add_argument("--out", type=Path, help="Write JSON report to this path")

    diff = sub.add_parser("diff", help="Side-by-side PG vs Qdrant for one doc_id")
    diff.add_argument("doc_id", help="Normalized doc_id (as stored in Qdrant payloads)")

    sub.add_parser(
        "llm-audit",
        help="Delegate to audit_qdrant_postgres_consistency.py",
        add_help=False,
    )

    args, rest = parser.parse_known_args()

    if args.mode == "coverage":
        return asyncio.run(_mode_coverage(out_path=args.out))
    if args.mode == "diff":
        return asyncio.run(_mode_diff(doc_id=args.doc_id))
    if args.mode == "llm-audit":
        return _mode_llm_audit(forwarded=rest)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
