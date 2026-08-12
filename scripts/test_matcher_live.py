#!/usr/bin/env python3
"""
Layer B — Live-Postgres integration test for the structured study matcher.

Runs every golden `QueryStructure` fixture from
`tests/fixtures/matcher_golden_queries.py` against the real Postgres
`studies` table and emits a Markdown report summarising each query's:

  - Inputs (QueryStructure dict, pretty-printed)
  - `present_criteria` and `dynamic_weights` (from the matcher's own logs)
  - Top-10 returned studies with normalized scores
  - precision@10 and recall@50 against the optional ground-truth
    doc_ids in `tests/fixtures/matcher_ground_truth.json`
  - Any `must_not_return` doc_ids that actually came back (hard fail)
  - Warnings: zero-result queries, suspicious empty biomarker parses,
    short-circuit returns, and known-broken inputs

Usage
-----

    python scripts/test_matcher_live.py                          # runs all fixtures
    python scripts/test_matcher_live.py --query hn_scc_multi_axis
    python scripts/test_matcher_live.py --report matcher_report.md
    python scripts/test_matcher_live.py --limit 25               # top-K cap
    python scripts/test_matcher_live.py --quiet                  # report-only

Requires: the same env vars the regular /rag service uses
(POSTGRES_HOST, POSTGRES_PASSWORD, etc.). Credentials come from
`src.core.config.settings`. Read-only — issues SELECTs only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.structured_study_matcher import (  # noqa: E402
    StructuredMatchResult,
    match_studies_by_structure,
)
from tests.fixtures.matcher_golden_queries import GOLDEN_FIXTURES  # noqa: E402


# ─── Stdout capture ────────────────────────────────────────────────────────
# The matcher logs its present_criteria, dynamic_weights, and SQL stats
# to stdout via print(). Redirect stdout temporarily so we can capture
# those lines per query and embed them in the report.

@contextmanager
def capture_stdout():
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old


# ─── Report helpers ────────────────────────────────────────────────────────

def _extract_log_lines(log_text: str, prefixes: List[str]) -> List[str]:
    """Pull only matcher log lines that start with any of the given prefixes."""
    out = []
    for line in log_text.splitlines():
        for p in prefixes:
            if p in line:
                out.append(line.strip())
                break
    return out


def _format_query_structure(qs: Dict[str, Any]) -> str:
    """Pretty-print a QueryStructure dict for the report."""
    return json.dumps(qs, indent=2, default=str)


def _precision_at_k(returned: List[str], relevant: List[str], k: int = 10) -> float:
    if not returned[:k]:
        return 0.0
    hits = sum(1 for d in returned[:k] if d in relevant)
    return hits / min(k, len(returned))


def _recall_at_k(returned: List[str], relevant: List[str], k: int = 50) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in relevant if d in returned[:k])
    return hits / len(relevant)


def _count_must_not_violations(returned: List[str], must_not: List[str]) -> List[str]:
    return [d for d in returned if d in must_not]


# ─── Per-query runner ──────────────────────────────────────────────────────

async def run_one_fixture(
    name: str,
    query_struct: Dict[str, Any],
    ground_truth: Dict[str, Any],
    limit: int = 50,
) -> Dict[str, Any]:
    """Run the matcher on one fixture, return a report dict."""
    with capture_stdout() as buf:
        t0 = time.perf_counter()
        try:
            result: StructuredMatchResult = await match_studies_by_structure(
                query_struct, limit=limit
            )
        except Exception as e:
            return {
                "name": name,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }
        elapsed_ms = (time.perf_counter() - t0) * 1000

    log_text = buf.getvalue()

    # Order doc_ids by score (match_scores is unordered)
    sorted_docs = sorted(
        result.match_scores.items(), key=lambda x: x[1], reverse=True
    )
    returned_doc_ids = [did for did, _ in sorted_docs]

    relevant = ground_truth.get("expected_doc_ids_relevant", [])
    must_not = ground_truth.get("expected_doc_ids_must_not_return", [])

    warnings: List[str] = []
    if not result.doc_ids:
        warnings.append(
            "Matcher returned ZERO studies — check log for 'No criteria detected' "
            "(the biomarker-only short-circuit bug) or 'No matching studies found'."
        )
    # Biomarker parsing sanity check: verify the parser normalised each
    # input biomarker to a known canonical name (not a raw uppercase
    # passthrough), which would indicate a parser gap. The numeric /
    # variant patterns ('CPS score of 100', 'EGFR L858R', 'KRAS G12C',
    # etc.) are now handled by the staged parser as of commit 36e81b3.
    biomarkers = query_struct.get("cancer", {}).get("biomarkers", []) or []
    for m in biomarkers:
        try:
            from src.api.services.structured_study_matcher import (
                CANONICAL_BIOMARKERS,
                _parse_biomarker_query,
            )
            canonical, _status = _parse_biomarker_query(m)
            if canonical and canonical not in CANONICAL_BIOMARKERS.values():
                # Uppercase passthrough — parser couldn't canonicalise
                warnings.append(
                    f"Biomarker {m!r} parsed to non-canonical {canonical!r} "
                    f"— may not match any biomarker_status JSONB key."
                )
        except Exception:
            pass

    return {
        "name": name,
        "description": ground_truth.get("description", ""),
        "notes": ground_truth.get("notes", ""),
        "query_structure": query_struct,
        "log": log_text,
        "conditions_used": result.conditions_used,
        "dynamic_weights": (
            # dynamic_weights are attached per-doc in match_details
            next(iter(result.match_details.values())).get("dynamic_weights", {})
            if result.match_details else {}
        ),
        "returned_count": len(returned_doc_ids),
        "top_10": sorted_docs[:10],
        "top_50_ids": returned_doc_ids[:50],
        "match_details": {
            did: result.match_details.get(did, {}) for did, _ in sorted_docs[:10]
        },
        "precision_at_10": _precision_at_k(returned_doc_ids, relevant, 10),
        "recall_at_50": _recall_at_k(returned_doc_ids, relevant, 50),
        "must_not_violations": _count_must_not_violations(returned_doc_ids, must_not),
        "has_ground_truth": bool(relevant),
        "warnings": warnings,
        "elapsed_ms": elapsed_ms,
    }


# ─── Report formatting ─────────────────────────────────────────────────────

def format_report(per_query: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# PostgreSQL Structured Matcher — Live Integration Report")
    lines.append("")
    lines.append(f"Fixtures run: **{len(per_query)}**")
    lines.append("")

    # Summary stats
    with_gt = [r for r in per_query if r.get("has_ground_truth")]
    if with_gt:
        avg_prec = sum(r["precision_at_10"] for r in with_gt) / len(with_gt)
        avg_rec = sum(r["recall_at_50"] for r in with_gt) / len(with_gt)
        lines.append(f"- Mean precision@10 (over {len(with_gt)} fixtures with ground truth): **{avg_prec:.2f}**")
        lines.append(f"- Mean recall@50: **{avg_rec:.2f}**")
    zero_return = sum(1 for r in per_query if r.get("returned_count", 0) == 0 and not r.get("error"))
    if zero_return:
        lines.append(f"- Fixtures that returned ZERO studies: **{zero_return}**")
    total_violations = sum(len(r.get("must_not_violations", [])) for r in per_query)
    if total_violations:
        lines.append(f"- Total must-not-return violations: **{total_violations}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in per_query:
        lines.append(f"## `{r['name']}`")
        if r.get("description"):
            lines.append(f"_{r['description']}_")
        lines.append("")

        if r.get("error"):
            lines.append(f"**ERROR**: {r['error']}")
            lines.append("")
            continue

        lines.append(f"**Elapsed**: {r['elapsed_ms']:.1f}ms | "
                     f"**Returned**: {r['returned_count']} studies")
        lines.append("")

        # QueryStructure
        lines.append("**Input QueryStructure**:")
        lines.append("```json")
        lines.append(_format_query_structure(r['query_structure']))
        lines.append("```")
        lines.append("")

        # Dynamic weights
        dw = r.get("dynamic_weights") or {}
        if dw:
            lines.append(f"**Dynamic weights** (total={sum(dw.values())}): `{dw}`")
        lines.append(f"**Criteria used**: `{r.get('conditions_used', [])}`")
        lines.append("")

        # Warnings
        if r.get("warnings"):
            lines.append("**⚠ Warnings**:")
            for w in r["warnings"]:
                lines.append(f"- {w}")
            lines.append("")

        # Must-not-return violations (hard fail)
        if r.get("must_not_violations"):
            lines.append("**🚨 MUST-NOT-RETURN violations**:")
            for did in r["must_not_violations"]:
                lines.append(f"- `{did}`")
            lines.append("")

        # Top 10
        lines.append(f"**Top 10 studies** "
                     f"(precision@10 = {r['precision_at_10']:.2f}, "
                     f"recall@50 = {r['recall_at_50']:.2f}):")
        lines.append("")
        if not r.get("top_10"):
            lines.append("_(no studies returned)_")
        else:
            lines.append("| # | Score | doc_id | site | title | n |")
            lines.append("|---|-------|--------|------|-------|---|")
            for i, (did, score) in enumerate(r["top_10"], 1):
                detail = r["match_details"].get(did, {})
                title = (detail.get("study_name") or "")[:60]
                site = (detail.get("cancer_location") or "")[:25]
                n = detail.get("num_patients", "?")
                lines.append(f"| {i} | {score:.0%} | `{did[:30]}` | {site} | {title} | {n} |")
        lines.append("")

        # Matcher log excerpt (present criteria, weights, exclusions)
        log_excerpt = _extract_log_lines(
            r.get("log", ""),
            ["[PG Criteria]", "[PG Weights]", "[PG Query]", "[PG Exclusion]", "[PG Results]"],
        )
        if log_excerpt:
            lines.append("**Matcher log** (filtered):")
            lines.append("```")
            for line in log_excerpt:
                lines.append(line)
            lines.append("```")
            lines.append("")

        if r.get("notes"):
            lines.append(f"**Notes**: {r['notes']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── CLI ────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> int:
    # Load ground truth
    gt_path = REPO_ROOT / "tests" / "fixtures" / "matcher_ground_truth.json"
    try:
        with open(gt_path) as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {gt_path} not found — precision/recall will be 0",
              file=sys.stderr)
        ground_truth = {}
    # Drop the _readme key
    ground_truth = {k: v for k, v in ground_truth.items() if not k.startswith("_")}

    if args.query:
        if args.query not in GOLDEN_FIXTURES:
            print(f"ERROR: unknown fixture {args.query!r}. Available: "
                  f"{list(GOLDEN_FIXTURES.keys())}", file=sys.stderr)
            return 2
        fixtures_to_run = {args.query: GOLDEN_FIXTURES[args.query]}
    else:
        fixtures_to_run = GOLDEN_FIXTURES

    per_query: List[Dict[str, Any]] = []
    for name, qs in fixtures_to_run.items():
        if not args.quiet:
            print(f"Running {name!r}...", file=sys.stderr)
        report_entry = await run_one_fixture(
            name, qs, ground_truth.get(name, {}), limit=args.limit,
        )
        per_query.append(report_entry)

    report = format_report(per_query)

    if args.report:
        Path(args.report).write_text(report)
        print(f"Report written to {args.report}", file=sys.stderr)
    else:
        print(report)

    # Exit code: nonzero if any fixture hit a must-not violation or errored
    exit_code = 0
    for r in per_query:
        if r.get("error") or r.get("must_not_violations"):
            exit_code = 1
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run golden fixtures against the live PostgreSQL matcher"
    )
    parser.add_argument("--query", help="Run only this fixture name")
    parser.add_argument("--report", help="Write Markdown report to this file (default: stdout)")
    parser.add_argument("--limit", type=int, default=50, help="Max studies per query (default: 50)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
