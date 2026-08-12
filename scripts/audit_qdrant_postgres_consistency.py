#!/usr/bin/env python3
"""
Layer C — Qdrant ↔ Postgres data consistency audit.

For a sample of `doc_id`s, this script:

  1. Fetches the Postgres row for each doc_id from the `studies` table
     (the same table the structured study matcher queries).
  2. Pulls every Qdrant chunk for the same doc_id and concatenates
     the chunk text into a single "study text window".
  3. Asks `gpt-4o-mini` to extract a strict 6-field structured profile
     from the chunk text (cancer_type, histology, stage_requirement,
     biomarkers, prior_therapies, treatment_modality).
  4. Diffs the LLM extraction against the Postgres row field-by-field.
  5. Emits a Markdown report listing every mismatch and gap.

Purpose: a data-quality audit, not a matcher test. If Postgres is
missing fields that Qdrant clearly discusses, that's an ingestion /
upsert bug and the matcher is innocent. If Postgres and Qdrant disagree
on a populated field, that's a more serious ingestion bug.

Usage
-----

    # sample 20 random doc_ids
    python scripts/audit_qdrant_postgres_consistency.py --sample 20 --report consistency_report.md

    # or target specific doc_ids
    python scripts/audit_qdrant_postgres_consistency.py \
        --doc-ids doc_id_abc doc_id_xyz \
        --report consistency_report.md

    # limit to one cancer site
    python scripts/audit_qdrant_postgres_consistency.py --sample 15 --site head_neck

Cost: ~$0.02/doc at gpt-4o-mini rates, so 20 docs is about $0.40.

Credentials come from `src.core.config.settings` (same env vars the
/rag service uses). Read-only on both Qdrant and Postgres.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import asyncpg

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient  # noqa: E402
import qdrant_client.models as qm  # noqa: E402

from src.core.config import settings  # noqa: E402


# ─── Extraction prompt ────────────────────────────────────────────────────
# Mirrors the 5-field eligibility schema from
# patient_eligibility_boost_service.py:652, but for STUDY side (what the
# study reports about its own enrolment population).

EXTRACTION_SYSTEM_PROMPT = (
    "You are a clinical-trial data extractor. From the study text "
    "provided, extract ONLY fields that are explicitly stated. "
    "If a field is not stated in the text, set its value to null. "
    "Never invent values. Respond with a single JSON object only."
)

EXTRACTION_USER_TEMPLATE = """Extract the following fields from the study text below:

  - cancer_type:        canonical cancer type (e.g. "head and neck cancer", "breast cancer", "non-small-cell lung cancer")
  - histology:          histopathologic type (e.g. "squamous cell carcinoma", "adenocarcinoma", "ductal carcinoma")
  - stage_requirement:  enrolment stage criterion (e.g. "stage II-IVA", "metastatic only", "any stage")
  - biomarkers:         dict of marker name -> status (e.g. {{"PD-L1": "high", "HER2": "negative"}})
  - prior_therapies:    prior-treatment requirement (e.g. "treatment-naive", "must have progressed on anti-PD-1")
  - treatment_modality: arm/intervention modality (e.g. "chemoradiation", "immunotherapy", "surgery")

Return JSON with exactly those six keys. Use null for unknowns, never an empty string.

STUDY TEXT:
---
{study_text}
---

JSON:"""


# ─── Qdrant helpers ────────────────────────────────────────────────────────

def _get_qdrant_client() -> QdrantClient:
    """Build a Qdrant client with a generous timeout — same pattern as
    tumor_board/retrieval.py."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )


def _fetch_chunks_for_doc_id(
    client: QdrantClient,
    collection: str,
    doc_id: str,
    max_chunks: int = 30,
) -> List[Dict[str, Any]]:
    """Scroll all chunks for a single doc_id. Returns a list of payload dicts."""
    scroll_filter = qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
    )
    chunks: List[Dict[str, Any]] = []
    offset = None
    pages = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=50,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            chunks.append(payload)
            if len(chunks) >= max_chunks:
                return chunks
        offset = next_offset
        pages += 1
        if offset is None or pages > 10:
            break
    return chunks


def _build_study_text_window(chunks: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Concatenate chunk text up to max_chars, preferring sections that
    discuss methods / inclusion / diagnosis over references / acknowledgements."""
    # Rank-biased section order: inclusion/methods first, then results,
    # then anything else, then skip references/funding.
    SECTION_PRIORITY = {
        "methods": 10, "method": 10, "inclusion": 10, "eligibility": 10,
        "patients": 9, "diagnosis": 9, "design": 9,
        "results": 7, "outcomes": 7, "endpoints": 7,
        "introduction": 5, "background": 5,
        "discussion": 3, "conclusion": 3,
        "references": -5, "acknowledg": -5, "funding": -5,
    }

    def _rank(c):
        sec = (c.get("section") or "").lower()
        for k, v in SECTION_PRIORITY.items():
            if k in sec:
                return -v  # negate so higher priority sorts first
        return 0

    sorted_chunks = sorted(chunks, key=_rank)

    parts: List[str] = []
    total = 0
    for c in sorted_chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        sec = c.get("section") or ""
        block = f"[{sec}]\n{text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


# ─── Postgres helpers ──────────────────────────────────────────────────────

POSTGRES_COLUMNS = [
    "doc_id",
    "study_name",
    "cancer_type",
    "cancer_location",
    "histopathologic_type",
    "biomarker_status",
    "molecular_subtype",
    "extraction_data",
    "age_range",
    "median_age",
    "number_of_patients",
    "performance_status",
]


async def _fetch_pg_row(conn, doc_id: str) -> Optional[Dict[str, Any]]:
    cols = ", ".join(POSTGRES_COLUMNS)
    query = f"SELECT {cols} FROM studies WHERE doc_id = $1 LIMIT 1"
    row = await conn.fetchrow(query, doc_id)
    if row is None:
        return None
    out = dict(row)
    # asyncpg returns JSONB as strings by default unless codec registered;
    # try to json.loads biomarker_status + extraction_data for diffing.
    for col in ("biomarker_status", "extraction_data"):
        val = out.get(col)
        if isinstance(val, str):
            try:
                out[col] = json.loads(val)
            except json.JSONDecodeError:
                pass
    return out


async def _fetch_sample_doc_ids(conn, sample: int, site_filter: Optional[str]) -> List[str]:
    """Pick a random sample of doc_ids from the studies table."""
    if site_filter:
        query = (
            "SELECT doc_id FROM studies "
            "WHERE doc_id IS NOT NULL AND cancer_location ILIKE $1 "
            f"ORDER BY random() LIMIT {sample}"
        )
        rows = await conn.fetch(query, f"%{site_filter}%")
    else:
        query = (
            "SELECT doc_id FROM studies "
            "WHERE doc_id IS NOT NULL "
            f"ORDER BY random() LIMIT {sample}"
        )
        rows = await conn.fetch(query)
    return [r["doc_id"] for r in rows]


# ─── LLM extraction ────────────────────────────────────────────────────────

async def _llm_extract(openai_client, study_text: str) -> Dict[str, Any]:
    """Call gpt-4o-mini for one extraction. Returns the parsed JSON dict or {}."""
    if not study_text.strip():
        return {}
    try:
        response = await openai_client.chat.completions.create(
            model=settings.openai_mini_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(
                    study_text=study_text
                )},
            ],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        return {"_error": str(e)}


# ─── Diff logic ────────────────────────────────────────────────────────────

def _diff_cancer_type(llm: Optional[str], pg_row: Dict[str, Any]) -> Dict[str, Any]:
    pg_type = (pg_row.get("cancer_type") or "").lower()
    pg_loc = (pg_row.get("cancer_location") or "").lower()
    if llm is None:
        return {"llm": None, "pg": pg_type or pg_loc or None,
                "verdict": "qdrant_silent"}
    if not pg_type and not pg_loc:
        return {"llm": llm, "pg": None, "verdict": "postgres_null_qdrant_stated"}
    l = llm.lower()
    if l in pg_type or l in pg_loc or any(tok in pg_type for tok in l.split()):
        return {"llm": llm, "pg": pg_type or pg_loc, "verdict": "match"}
    return {"llm": llm, "pg": pg_type or pg_loc, "verdict": "mismatch"}


def _diff_histology(llm: Optional[str], pg_row: Dict[str, Any]) -> Dict[str, Any]:
    pg_hist = (pg_row.get("histopathologic_type") or "").lower()
    pg_mol = (pg_row.get("molecular_subtype") or "").lower()
    if llm is None:
        return {"llm": None, "pg": pg_hist or pg_mol or None,
                "verdict": "qdrant_silent"}
    if not pg_hist and not pg_mol:
        return {"llm": llm, "pg": None, "verdict": "postgres_null_qdrant_stated"}
    l = llm.lower()
    if l in pg_hist or l in pg_mol:
        return {"llm": llm, "pg": pg_hist or pg_mol, "verdict": "match"}
    return {"llm": llm, "pg": pg_hist or pg_mol, "verdict": "mismatch"}


def _diff_biomarkers(llm: Any, pg_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-marker diff. LLM returns a dict like {"PD-L1": "high", "HER2": "negative"}."""
    pg_bm = pg_row.get("biomarker_status") or {}
    if not isinstance(pg_bm, dict):
        pg_bm = {}
    if not isinstance(llm, dict):
        llm = {}
    all_keys = set(llm.keys()) | set(pg_bm.keys())
    rows = []
    for k in sorted(all_keys):
        l = llm.get(k)
        p = pg_bm.get(k)
        if l is None and p is not None:
            verdict = "qdrant_silent"
        elif l is not None and p is None:
            verdict = "postgres_null_qdrant_stated"
        elif l is not None and p is not None:
            verdict = "match" if str(l).lower() == str(p).lower() else "mismatch"
        else:
            verdict = "both_null"
        rows.append({"marker": k, "llm": l, "pg": p, "verdict": verdict})
    return rows


# ─── Report generation ────────────────────────────────────────────────────

def format_report(audits: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Qdrant ↔ Postgres Data Consistency Audit")
    lines.append("")
    lines.append(f"Doc IDs audited: **{len(audits)}**")

    # Summary stats
    stats = defaultdict(int)
    for a in audits:
        for field, diff in a.get("diffs", {}).items():
            if field == "biomarkers":
                for bm in diff:
                    stats[bm["verdict"]] += 1
            else:
                stats[diff.get("verdict", "unknown")] += 1

    lines.append("")
    lines.append("## Aggregate verdict counts")
    lines.append("")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")
    lines.append("> Verdicts:")
    lines.append("> - `match`: Qdrant and Postgres agree")
    lines.append("> - `mismatch`: Both stated a value but they disagree (serious)")
    lines.append("> - `postgres_null_qdrant_stated`: Qdrant chunks state the field, Postgres is null (ingestion gap)")
    lines.append("> - `qdrant_silent`: Neither source explicitly stated the field")
    lines.append("> - `both_null`: Both agree the field is not stated")
    lines.append("")
    lines.append("---")
    lines.append("")

    for a in audits:
        doc_id = a["doc_id"]
        lines.append(f"## `{doc_id}`")
        if a.get("error"):
            lines.append(f"**ERROR**: {a['error']}")
            lines.append("")
            continue
        if a.get("postgres_missing"):
            lines.append("**ERROR**: no Postgres row found for this doc_id")
            lines.append("")
            continue
        if a.get("qdrant_empty"):
            lines.append("**ERROR**: no Qdrant chunks found for this doc_id")
            lines.append("")
            continue

        title = a.get("study_name", "(untitled)")
        lines.append(f"_{title}_")
        lines.append("")
        lines.append(f"- Qdrant chunks: {a.get('qdrant_chunk_count', 0)}")
        lines.append(f"- Postgres row: `{a.get('cancer_location', '?')}` "
                     f"({a.get('number_of_patients', '?')} pts)")
        lines.append("")

        diffs = a.get("diffs", {})

        lines.append("| Field | Qdrant (LLM extraction) | Postgres | Verdict |")
        lines.append("|---|---|---|---|")
        for field in ("cancer_type", "histology", "stage_requirement",
                      "prior_therapies", "treatment_modality"):
            d = diffs.get(field, {})
            llm_val = str(d.get("llm"))[:50]
            pg_val = str(d.get("pg"))[:50]
            verdict = d.get("verdict", "?")
            lines.append(f"| {field} | {llm_val} | {pg_val} | {verdict} |")
        lines.append("")

        bm_diff = diffs.get("biomarkers", [])
        if bm_diff:
            lines.append("**Biomarker diff**:")
            lines.append("")
            lines.append("| Marker | Qdrant | Postgres | Verdict |")
            lines.append("|---|---|---|---|")
            for r in bm_diff:
                lines.append(f"| {r['marker']} | {r['llm']} | {r['pg']} | {r['verdict']} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── Main audit loop ───────────────────────────────────────────────────────

async def audit_one_doc_id(
    doc_id: str,
    qdrant: QdrantClient,
    pg_conn,
    openai_client,
    collection: str,
) -> Dict[str, Any]:
    # 1. Fetch Postgres row
    try:
        pg_row = await _fetch_pg_row(pg_conn, doc_id)
    except Exception as e:
        return {"doc_id": doc_id, "error": f"postgres fetch failed: {e}"}
    if pg_row is None:
        return {"doc_id": doc_id, "postgres_missing": True}

    # 2. Fetch Qdrant chunks
    try:
        chunks = await asyncio.to_thread(
            _fetch_chunks_for_doc_id, qdrant, collection, doc_id
        )
    except Exception as e:
        return {"doc_id": doc_id, "error": f"qdrant fetch failed: {e}"}
    if not chunks:
        return {"doc_id": doc_id, "qdrant_empty": True}

    study_text = _build_study_text_window(chunks)

    # 3. LLM extraction
    llm_data = await _llm_extract(openai_client, study_text)
    if llm_data.get("_error"):
        return {"doc_id": doc_id, "error": f"llm failed: {llm_data['_error']}"}

    # 4. Diff
    diffs: Dict[str, Any] = {}
    diffs["cancer_type"] = _diff_cancer_type(llm_data.get("cancer_type"), pg_row)
    diffs["histology"] = _diff_histology(llm_data.get("histology"), pg_row)
    diffs["stage_requirement"] = {
        "llm": llm_data.get("stage_requirement"),
        "pg": "(extraction_data, not diffed)",
        "verdict": "match" if llm_data.get("stage_requirement") else "qdrant_silent",
    }
    diffs["prior_therapies"] = {
        "llm": llm_data.get("prior_therapies"),
        "pg": "(extraction_data, not diffed)",
        "verdict": "match" if llm_data.get("prior_therapies") else "qdrant_silent",
    }
    diffs["treatment_modality"] = {
        "llm": llm_data.get("treatment_modality"),
        "pg": "(extraction_data, not diffed)",
        "verdict": "match" if llm_data.get("treatment_modality") else "qdrant_silent",
    }
    diffs["biomarkers"] = _diff_biomarkers(llm_data.get("biomarkers"), pg_row)

    return {
        "doc_id": doc_id,
        "study_name": pg_row.get("study_name"),
        "cancer_location": pg_row.get("cancer_location"),
        "number_of_patients": pg_row.get("number_of_patients"),
        "qdrant_chunk_count": len(chunks),
        "diffs": diffs,
    }


async def main_async(args: argparse.Namespace) -> int:
    # Validate settings
    if not settings.qdrant_url or not settings.openai_api_key:
        print("ERROR: QDRANT_URL and OPENAI_API_KEY must be set in .env",
              file=sys.stderr)
        return 2

    # Connect to Postgres
    try:
        pg_conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database,
            timeout=10,
        )
    except Exception as e:
        print(f"ERROR: Postgres connection failed: {e}", file=sys.stderr)
        return 2

    # Select doc_ids
    if args.doc_ids:
        doc_ids = args.doc_ids
        print(f"Auditing {len(doc_ids)} explicit doc_ids", file=sys.stderr)
    else:
        doc_ids = await _fetch_sample_doc_ids(pg_conn, args.sample, args.site)
        print(f"Sampled {len(doc_ids)} random doc_ids "
              f"(site_filter={args.site!r})", file=sys.stderr)

    if not doc_ids:
        print("No doc_ids to audit — exiting", file=sys.stderr)
        await pg_conn.close()
        return 1

    # Qdrant client
    qdrant = _get_qdrant_client()

    # OpenAI client
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    except ImportError:
        print("ERROR: openai package not available", file=sys.stderr)
        await pg_conn.close()
        return 2

    # Run audit with concurrency cap
    semaphore = asyncio.Semaphore(args.concurrency)

    async def _guarded(did):
        async with semaphore:
            if not args.quiet:
                print(f"  auditing {did[:40]}...", file=sys.stderr)
            return await audit_one_doc_id(
                did, qdrant, pg_conn, openai_client, settings.qdrant_collection
            )

    audits = await asyncio.gather(*[_guarded(d) for d in doc_ids])

    await pg_conn.close()

    # Format report
    report = format_report(audits)
    if args.report:
        Path(args.report).write_text(report)
        print(f"Report written to {args.report}", file=sys.stderr)
    else:
        print(report)

    # Exit code: nonzero if any mismatch was found
    has_mismatch = False
    for a in audits:
        for field, diff in a.get("diffs", {}).items():
            if field == "biomarkers":
                if any(bm["verdict"] == "mismatch" for bm in diff):
                    has_mismatch = True
            elif diff.get("verdict") == "mismatch":
                has_mismatch = True
    return 1 if has_mismatch else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit data consistency between Qdrant and Postgres for "
                    "the same doc_id (data quality audit, not a matcher test)."
    )
    parser.add_argument("--sample", type=int, default=15,
                        help="Number of random doc_ids to audit (default: 15)")
    parser.add_argument("--doc-ids", nargs="+",
                        help="Explicit doc_ids to audit (overrides --sample)")
    parser.add_argument("--site",
                        help="Filter random sample to studies whose "
                             "cancer_location ILIKE %%site%% (e.g. 'breast', 'lung')")
    parser.add_argument("--report", help="Write report to this file (default: stdout)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent audits (default: 4)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
