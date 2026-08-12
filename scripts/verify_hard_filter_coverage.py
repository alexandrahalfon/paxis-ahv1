#!/usr/bin/env python3
"""
Layer D — Hard-filter coverage verification.

Probes the LLM-based PatientEligibility hard-filter layer to verify
that all FIVE clinical criteria are actually being evaluated and
applied end-to-end:

  1. Cancer type
  2. Histology
  3. Stage / disease status
  4. Prior therapies / treatment lines
  5. Molecular biomarkers

The list is declared at `patient_eligibility_boost_service.py:31` as
`HARD_FILTER_CRITERIA`, but we have never verified that:

  (a) The criteria actually end up in `active_criteria` (line 590)
      when the patient context clearly has them
  (b) The LLM evaluates each criterion rather than returning
      NOT_AVAILABLE by default
  (c) At least one MISMATCH is being detected per criterion across
      a non-trivial study pool (if no mismatches ever fire, the
      filter is functionally inactive)

This script feeds a set of curated raw clinical case strings through:

  1. `extract_patient_context_from_query(raw_text)` — build patient_context
  2. Live Qdrant retrieval — fetch a batch of candidate chunks for
     the case (via `tumor_board/retrieval.py::lightweight_search`,
     which already has a tuned timeout)
  3. `check_patient_eligibility_for_studies(query, patient_context,
     doc_ids, openai_client)` — the LLM judge
  4. Per-criterion coverage stats from the returned verdicts

Writes a Markdown report documenting:

  - Which of the 5 criteria are in `active_criteria` for each case
  - Per-criterion: evaluated_count, match_count, mismatch_count,
    not_available_count
  - Any criterion that's "active" but has zero MISMATCHes across
    the whole dataset (functionally inactive)
  - Any criterion that's "present in patient_context" but NOT in
    active_criteria (a wiring bug in the extractor)

Usage
-----

    python scripts/verify_hard_filter_coverage.py --report hard_filter_report.md
    python scripts/verify_hard_filter_coverage.py --case canonical_hn_scc
    python scripts/verify_hard_filter_coverage.py --limit 15  # chunks per case

Read-only. Uses gpt-4o-mini for the LLM judge (same model the prod
pipeline uses).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.patient_eligibility_boost_service import (  # noqa: E402
    HARD_FILTER_CRITERIA,
    check_patient_eligibility_for_studies,
    extract_patient_context_from_query,
)
from src.api.services.tumor_board.retrieval import (  # noqa: E402
    LightweightStudy,
    lightweight_search,
)
from src.core.config import settings  # noqa: E402


# ─── Curated case library ──────────────────────────────────────────────────
# Small, hand-picked set of raw clinical narratives that together
# exercise all 5 hard-filter criteria. Each case should clearly
# state at least 3 of the 5 so we can detect filter gaps.

CASE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "canonical_hn_scc": {
        "description": "80 y.o. recurrent oral tongue SCC, CPS 100, post-glossectomy, progressing on pembrolizumab",
        "raw_text": (
            "80 y.o. male non-smoker with a PMH HTN, Hep C, BPH, CKD, latent syphilis, "
            "transverse colon adenocarcinoma complicated by LBO s/p (6/16/21) diagnostic "
            "lap, ex lap with extended right hemicolectomy 6/2021 and ileostomy reversal "
            "10/6/2021, and initial Stage II (pT2pN0M0R0, DOI 5.1 mm, PNI-, LVSI-) "
            "squamous cell carcinoma of the left oral tongue, status post left partial "
            "glossectomy, left neck dissection levels I-III, and radial forearm free flap "
            "reconstruction. In August 2025, he developed a recurrent lesion in the left "
            "level I neck, biopsy-proven recurrent SCC with a CPS score of 100, started "
            "on pembrolizumab and is progressing on systemic therapy."
        ),
        "retrieval_hint": "head and neck SCC recurrent pembrolizumab",
        "expected_criteria_active": [
            "cancer_type", "histology", "stage", "biomarkers", "prior_therapies"
        ],
    },
    "stage_iv_nsclc_egfr": {
        "description": "Stage IV NSCLC with EGFR L858R, treatment-naive",
        "raw_text": (
            "65 y.o. female never-smoker with newly diagnosed stage IV lung adenocarcinoma. "
            "EGFR L858R mutation, PD-L1 TPS 30%. No prior systemic therapy. "
            "ECOG 1. Looking for first-line treatment options."
        ),
        "retrieval_hint": "stage IV NSCLC EGFR L858R first-line",
        "expected_criteria_active": [
            "cancer_type", "histology", "stage", "biomarkers", "prior_therapies"
        ],
    },
    "metastatic_tnbc": {
        "description": "Metastatic triple-negative breast cancer, post-taxane",
        "raw_text": (
            "52 y.o. female with metastatic triple-negative breast cancer "
            "(ER-, PR-, HER2-). Previously received neoadjuvant AC-T. "
            "Now with biopsy-confirmed liver and lung metastases. "
            "PD-L1 CPS 15, BRCA wild-type."
        ),
        "retrieval_hint": "metastatic triple-negative breast cancer post taxane",
        "expected_criteria_active": [
            "cancer_type", "histology", "stage", "biomarkers", "prior_therapies"
        ],
    },
    "biomarker_only_kras": {
        "description": "KRAS G12C — no stage or prior therapy stated",
        "raw_text": (
            "What are the treatment options for KRAS G12C-mutant colorectal cancer?"
        ),
        "retrieval_hint": "KRAS G12C colorectal cancer",
        # Intentionally minimal — expect only 1-2 criteria active
        "expected_criteria_active": ["cancer_type", "biomarkers"],
    },
}


# ─── Retrieval helper ──────────────────────────────────────────────────────

async def _retrieve_candidate_chunks(
    retrieval_hint: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Use the lightweight Qdrant helper to pull candidate chunks for
    a case. Returns a chunks list in the shape PatientEligibility expects
    (dict-like, not LightweightStudy dataclass)."""
    studies: List[LightweightStudy] = await lightweight_search(
        retrieval_hint, limit_points=limit, max_chunks_per_study=2
    )
    chunks_out: List[Dict[str, Any]] = []
    for s in studies:
        # PatientEligibility expects chunk dicts with doc_id, title, text, score
        for c in s.chunks:
            chunks_out.append({
                "doc_id": s.doc_id,
                "title": s.title,
                "text": c.get("text", ""),
                "score": c.get("score", s.rerank_score),
                "doc_meta": {"title": s.title, "citation": s.citation, "year": s.year},
            })
    return chunks_out


# ─── Per-case probe ────────────────────────────────────────────────────────

async def probe_case(name: str, case: Dict[str, Any], limit: int) -> Dict[str, Any]:
    raw_text = case["raw_text"]

    # 1. Extract patient_context
    try:
        patient_context = extract_patient_context_from_query(raw_text)
    except Exception as e:
        return {"name": name, "error": f"extract failed: {e}"}
    if not patient_context:
        return {
            "name": name,
            "error": "extract_patient_context_from_query returned None",
            "raw_text": raw_text[:200],
        }

    # 2. Build active_criteria the same way patient_eligibility_boost_service does
    active_criteria = []
    if patient_context.get("cancer_type"):
        active_criteria.append("cancer_type")
    if patient_context.get("histology"):
        active_criteria.append("histology")
    if patient_context.get("stage") or patient_context.get("tnm"):
        active_criteria.append("stage")
    if patient_context.get("treatment_history"):
        active_criteria.append("prior_therapies")
    if patient_context.get("biomarkers"):
        active_criteria.append("biomarkers")

    # Compare against the case's expectation
    expected = set(case.get("expected_criteria_active", []))
    got = set(active_criteria)
    missing_from_active = expected - got
    extra_in_active = got - expected

    # 3. Retrieve candidate chunks
    try:
        chunks = await _retrieve_candidate_chunks(
            case["retrieval_hint"], limit=limit
        )
    except Exception as e:
        return {
            "name": name,
            "error": f"qdrant retrieval failed: {e}",
            "patient_context": patient_context,
            "active_criteria": active_criteria,
        }
    if not chunks:
        return {
            "name": name,
            "error": "qdrant retrieval returned zero chunks",
            "patient_context": patient_context,
            "active_criteria": active_criteria,
        }

    # 4. Collect unique doc_ids from the chunks
    seen: set = set()
    doc_ids = []
    for c in chunks:
        did = c.get("doc_id")
        if did and did not in seen:
            seen.add(did)
            doc_ids.append(did)
    if not doc_ids:
        return {
            "name": name,
            "error": "no doc_ids on retrieved chunks",
            "patient_context": patient_context,
            "active_criteria": active_criteria,
        }

    # 5. Call the LLM eligibility check
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)
    except ImportError:
        return {"name": name, "error": "openai package not available"}

    try:
        eligibility_results = await check_patient_eligibility_for_studies(
            query=raw_text,
            patient_context=patient_context,
            doc_ids=doc_ids,
            openai_client=openai_client,
        )
    except Exception as e:
        return {
            "name": name,
            "error": f"eligibility LLM check failed: {e}",
            "patient_context": patient_context,
            "active_criteria": active_criteria,
        }

    # 6. Compute per-criterion stats
    verdict_counts: Dict[str, Counter] = {
        criterion: Counter() for criterion in HARD_FILTER_CRITERIA
    }
    per_study_verdicts: List[Dict[str, Any]] = []
    for doc_id, result in eligibility_results.items():
        verdicts = result.get("criteria_verdicts", {})
        per_study_verdicts.append({
            "doc_id": doc_id,
            "status": result.get("status"),
            "verdicts": verdicts,
            "reason": result.get("reason", ""),
        })
        for criterion in HARD_FILTER_CRITERIA:
            v = verdicts.get(criterion, "NOT_AVAILABLE").upper()
            if v not in ("MATCH", "MISMATCH", "NOT_AVAILABLE"):
                v = "NOT_AVAILABLE"
            verdict_counts[criterion][v] += 1

    # 7. Flag suspicious states
    flags: List[str] = []
    for criterion in HARD_FILTER_CRITERIA:
        counts = verdict_counts[criterion]
        total = sum(counts.values())
        if criterion in active_criteria:
            if counts["MATCH"] == 0 and counts["MISMATCH"] == 0:
                flags.append(
                    f"`{criterion}`: active but LLM returned NOT_AVAILABLE for all {total} studies "
                    f"— the LLM is not actually using this criterion for this case."
                )
            if counts["MISMATCH"] == 0 and total >= 5:
                flags.append(
                    f"`{criterion}`: active but ZERO mismatches across {total} studies — "
                    f"filter is functionally inactive for this case."
                )
        else:
            # Criterion NOT in active_criteria — was it clearly present
            # in the raw text even if the extractor missed it?
            raw_lower = raw_text.lower()
            if criterion == "cancer_type" and any(
                kw in raw_lower for kw in ("cancer", "carcinoma", "tumor", "neoplasm")
            ):
                flags.append(
                    f"`cancer_type`: raw text mentions cancer but active_criteria "
                    f"list does not include it — wiring bug in extractor."
                )
            if criterion == "histology" and any(
                kw in raw_lower for kw in ("scc", "squamous", "adenocarcinoma", "ductal")
            ):
                flags.append(
                    f"`histology`: raw text mentions histology but not in active_criteria."
                )
    if missing_from_active:
        flags.append(
            f"expected_criteria_active minus actual: {sorted(missing_from_active)} "
            f"(criteria you expected to be active but aren't)"
        )
    if extra_in_active:
        flags.append(
            f"active_criteria minus expected: {sorted(extra_in_active)} "
            f"(criteria activated that weren't in the expected set)"
        )

    return {
        "name": name,
        "description": case.get("description", ""),
        "raw_text": raw_text,
        "patient_context": patient_context,
        "active_criteria": active_criteria,
        "expected_criteria_active": sorted(expected),
        "chunk_count": len(chunks),
        "doc_id_count": len(doc_ids),
        "verdict_counts": {k: dict(v) for k, v in verdict_counts.items()},
        "per_study_verdicts": per_study_verdicts,
        "flags": flags,
    }


# ─── Report formatting ────────────────────────────────────────────────────

def format_report(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Hard-Filter Coverage Verification Report")
    lines.append("")
    lines.append(f"Cases probed: **{len(results)}**")
    lines.append("")
    lines.append(f"Hard-filter criteria (from `HARD_FILTER_CRITERIA` in "
                 f"`patient_eligibility_boost_service.py:31`):")
    for c in HARD_FILTER_CRITERIA:
        lines.append(f"- `{c}`")
    lines.append("")

    # Aggregate per-criterion stats across all cases
    agg: Dict[str, Counter] = {c: Counter() for c in HARD_FILTER_CRITERIA}
    active_in_cases: Dict[str, int] = {c: 0 for c in HARD_FILTER_CRITERIA}
    total_cases = 0
    for r in results:
        if r.get("error"):
            continue
        total_cases += 1
        for c in HARD_FILTER_CRITERIA:
            if c in r.get("active_criteria", []):
                active_in_cases[c] += 1
            for verdict, count in r.get("verdict_counts", {}).get(c, {}).items():
                agg[c][verdict] += count

    lines.append("## Aggregate per-criterion coverage")
    lines.append("")
    lines.append("| Criterion | Active in N cases | Total MATCH | Total MISMATCH | Total NOT_AVAILABLE |")
    lines.append("|---|---|---|---|---|")
    for c in HARD_FILTER_CRITERIA:
        counts = agg[c]
        lines.append(
            f"| {c} | {active_in_cases[c]}/{total_cases} "
            f"| {counts.get('MATCH', 0)} "
            f"| {counts.get('MISMATCH', 0)} "
            f"| {counts.get('NOT_AVAILABLE', 0)} |"
        )
    lines.append("")

    # Dead criteria — those with 0 MISMATCHes across all cases
    dead = [
        c for c in HARD_FILTER_CRITERIA
        if agg[c].get("MISMATCH", 0) == 0 and active_in_cases[c] > 0
    ]
    if dead:
        lines.append("**⚠ Functionally inactive criteria** (active in at least one case "
                     "but ZERO mismatches ever detected):")
        for c in dead:
            lines.append(f"- `{c}`")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Per-case details
    for r in results:
        name = r["name"]
        lines.append(f"## `{name}`")
        if r.get("description"):
            lines.append(f"_{r['description']}_")
        lines.append("")

        if r.get("error"):
            lines.append(f"**ERROR**: {r['error']}")
            lines.append("")
            continue

        pc = r.get("patient_context", {})
        lines.append("**Detected patient_context**:")
        lines.append("```json")
        lines.append(json.dumps(pc, indent=2, default=str))
        lines.append("```")
        lines.append("")

        lines.append(f"**active_criteria**: `{r.get('active_criteria', [])}`")
        lines.append(f"**expected_criteria_active**: `{r.get('expected_criteria_active', [])}`")
        lines.append(f"**Studies evaluated**: {r.get('doc_id_count', 0)}")
        lines.append("")

        lines.append("**Per-criterion verdicts**:")
        lines.append("")
        lines.append("| Criterion | MATCH | MISMATCH | NOT_AVAILABLE |")
        lines.append("|---|---|---|---|")
        for c in HARD_FILTER_CRITERIA:
            counts = r.get("verdict_counts", {}).get(c, {})
            lines.append(
                f"| {c} | {counts.get('MATCH', 0)} "
                f"| {counts.get('MISMATCH', 0)} "
                f"| {counts.get('NOT_AVAILABLE', 0)} |"
            )
        lines.append("")

        flags = r.get("flags") or []
        if flags:
            lines.append("**⚠ Flags**:")
            for f in flags:
                lines.append(f"- {f}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── Main ──────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> int:
    if args.case:
        if args.case not in CASE_LIBRARY:
            print(f"ERROR: unknown case {args.case!r}. Available: "
                  f"{list(CASE_LIBRARY.keys())}", file=sys.stderr)
            return 2
        cases = {args.case: CASE_LIBRARY[args.case]}
    else:
        cases = CASE_LIBRARY

    results = []
    for name, case in cases.items():
        if not args.quiet:
            print(f"Probing {name!r}...", file=sys.stderr)
        r = await probe_case(name, case, limit=args.limit)
        results.append(r)

    report = format_report(results)
    if args.report:
        Path(args.report).write_text(report)
        print(f"Report written to {args.report}", file=sys.stderr)
    else:
        print(report)

    # Exit code: nonzero if any dead criteria found
    dead_any = False
    for r in results:
        if r.get("flags"):
            dead_any = True
    return 1 if dead_any else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe the LLM hard-filter layer for per-criterion "
                    "coverage across a curated set of clinical cases."
    )
    parser.add_argument("--case", help="Run only this case name")
    parser.add_argument("--report", help="Write report to this file (default: stdout)")
    parser.add_argument("--limit", type=int, default=15,
                        help="Max chunks per case from Qdrant (default: 15)")
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
