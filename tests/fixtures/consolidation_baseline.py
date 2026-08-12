"""
Baseline capture for RAG Pipeline Consolidation (Phase 0).

Captures pipeline behavior for the 3 canonical test cases BEFORE any
consolidation changes, so we can measure improvement and detect regressions.

Canonical test cases:
  (a) 43F cT4dN2M0 TNBC post-NAC
  (b) 80M recurrent oral tongue SCC ICI-refractory
  (c) 55F pT1cN1mi ER+/HER2- RS22

Usage:
  # Capture baselines (requires live DB/Qdrant):
  python -m tests.fixtures.consolidation_baseline --capture

  # Load existing baselines from JSON fixtures:
  from tests.fixtures.consolidation_baseline import load_baselines
  baselines = load_baselines()
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import datetime


# ---------------------------------------------------------------------------
# Canonical test case queries
# ---------------------------------------------------------------------------

CANONICAL_QUERIES = {
    "tnbc_post_nac": (
        "43-year-old female with cT4dN2M0 triple-negative breast cancer "
        "(TNBC), ER-/PR-/HER2-, status post neoadjuvant AC-T with "
        "residual disease. What are the best treatment options?"
    ),
    "oral_tongue_scc_ici_refractory": (
        "80-year-old male with recurrent squamous cell carcinoma of the "
        "oral tongue, PD-L1 CPS 100, previously treated with "
        "pembrolizumab (ICI-refractory), no longer a surgical candidate. "
        "What is the best next-line systemic therapy?"
    ),
    "er_pos_her2_neg_rs22": (
        "55-year-old female with pT1cN1mi ER+/HER2- invasive ductal "
        "carcinoma of the breast, Oncotype DX recurrence score 22. "
        "What is the recommended adjuvant systemic therapy?"
    ),
}


# ---------------------------------------------------------------------------
# BaselineCapture dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaselineCapture:
    """Snapshot of pipeline behavior for a single canonical test case.

    Fields:
        query: The raw query text sent to the pipeline.
        query_structure: Dict from structure_query_fast() — site, biomarkers,
            histology, stage, filter_category, query_type, etc.
        retrieval_doc_ids: Ordered list of doc_ids returned by retrieval.
        eligibility_verdicts: Per-study eligibility verdicts from
            PatientEligibility (e.g. {"doc_123": "MATCH", ...}).
        final_study_count: Number of studies in the final response bundle.
        pipeline_metrics: Timing and source-lane metrics from the pipeline
            (e.g. qdrant_count, pg_count, pto_count, latency_ms).
        inference_terms_count: Number of inference terms expanded by
            ClinicalInference for this query.
        biomarker_status_sample: Sample of biomarker_status JSONB keys from
            PG studies table (for spelling audit).
        timestamp: ISO-8601 timestamp of when the baseline was captured.
    """

    query: str = ""
    query_structure: Dict[str, Any] = field(default_factory=dict)
    retrieval_doc_ids: List[str] = field(default_factory=list)
    eligibility_verdicts: Dict[str, str] = field(default_factory=dict)
    final_study_count: int = 0
    pipeline_metrics: Dict[str, Any] = field(default_factory=dict)
    inference_terms_count: int = 0
    biomarker_status_sample: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineCapture":
        """Deserialize from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Fixture I/O helpers
# ---------------------------------------------------------------------------

BASELINES_DIR = Path(__file__).parent / "consolidation_baselines"


def _ensure_baselines_dir() -> Path:
    """Create the baselines directory if it doesn't exist."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINES_DIR


def save_baseline(case_id: str, baseline: BaselineCapture) -> Path:
    """Save a single baseline capture to JSON.

    Args:
        case_id: One of the CANONICAL_QUERIES keys
            (e.g. "tnbc_post_nac").
        baseline: The captured baseline data.

    Returns:
        Path to the written JSON file.
    """
    out_dir = _ensure_baselines_dir()
    path = out_dir / f"{case_id}.json"
    with open(path, "w") as f:
        json.dump(baseline.to_dict(), f, indent=2, default=str)
    print(f"[BaselineCapture] Saved {case_id} → {path}")
    return path


def load_baseline(case_id: str) -> Optional[BaselineCapture]:
    """Load a single baseline capture from JSON.

    Returns None if the fixture file doesn't exist.
    """
    path = BASELINES_DIR / f"{case_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return BaselineCapture.from_dict(data)


def load_baselines() -> Dict[str, BaselineCapture]:
    """Load all baseline captures from the fixtures directory.

    Returns:
        Dict mapping case_id → BaselineCapture for each JSON file found.
    """
    baselines: Dict[str, BaselineCapture] = {}
    if not BASELINES_DIR.exists():
        return baselines
    for path in sorted(BASELINES_DIR.glob("*.json")):
        case_id = path.stem
        if case_id.startswith("_"):
            continue  # skip metadata files
        baseline = load_baseline(case_id)
        if baseline:
            baselines[case_id] = baseline
    return baselines


# ---------------------------------------------------------------------------
# Live capture (requires running pipeline with DB/Qdrant)
# ---------------------------------------------------------------------------

async def capture_baseline(case_id: str, query: str) -> BaselineCapture:
    """Run a query through the current pipeline and capture the baseline.

    This requires a live environment with DB and Qdrant connections.
    When those aren't available, use the pre-captured JSON fixtures instead.
    """
    try:
        from src.api.services.comprehensive_retrieval import ComprehensiveRetrieval
        from src.api.services.query_structuring_service import structure_query_fast
    except ImportError as e:
        print(f"[BaselineCapture] Cannot import pipeline modules: {e}")
        return BaselineCapture(
            query=query,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )

    baseline = BaselineCapture(
        query=query,
        timestamp=datetime.datetime.utcnow().isoformat(),
    )

    # 1. Capture query structure
    try:
        qs = structure_query_fast(query)
        baseline.query_structure = {
            "site": getattr(qs, "site", None),
            "filter_category": getattr(qs, "filter_category", None),
            "biomarkers": getattr(qs, "biomarkers", []),
            "histology": getattr(qs, "histology", None),
            "stage": getattr(qs, "stage", None),
            "query_type": getattr(qs, "query_type", None),
            "question_focus": getattr(qs, "question_focus", None),
        }
    except Exception as e:
        print(f"[BaselineCapture] query_structure failed: {e}")

    # 2. Capture retrieval results
    try:
        cr = ComprehensiveRetrieval()
        result = await cr.retrieve_comprehensive(query)
        baseline.retrieval_doc_ids = [
            s.get("doc_id", "") for s in getattr(result, "studies", [])
        ]
        baseline.final_study_count = len(baseline.retrieval_doc_ids)
        baseline.pipeline_metrics = {
            "qdrant_count": getattr(result, "qdrant_count", 0),
            "pg_count": getattr(result, "pg_count", 0),
            "pto_count": getattr(result, "pto_count", 0),
            "total_candidates": getattr(result, "total_candidates", 0),
        }
    except Exception as e:
        print(f"[BaselineCapture] retrieval failed: {e}")

    # 3. Capture eligibility verdicts
    try:
        verdicts = {}
        for s in getattr(result, "studies", []):
            doc_id = s.get("doc_id", "")
            verdict = s.get("eligibility_verdict", "UNKNOWN")
            if doc_id:
                verdicts[doc_id] = verdict
        baseline.eligibility_verdicts = verdicts
    except Exception:
        pass

    # 4. Capture inference terms count
    try:
        from src.api.services.clinical_inference import run_inference
        inf_result = run_inference(query, {})
        baseline.inference_terms_count = len(
            getattr(inf_result, "inferred_terms", [])
        )
    except Exception:
        pass

    return baseline


async def capture_all_baselines() -> Dict[str, BaselineCapture]:
    """Capture baselines for all 3 canonical test cases."""
    results = {}
    for case_id, query in CANONICAL_QUERIES.items():
        print(f"[BaselineCapture] Capturing {case_id}...")
        baseline = await capture_baseline(case_id, query)
        save_baseline(case_id, baseline)
        results[case_id] = baseline
    print(f"[BaselineCapture] All baselines captured: {list(results.keys())}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Capture or load RAG pipeline consolidation baselines"
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Run live capture against current pipeline (requires DB/Qdrant)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load and print existing baseline fixtures",
    )
    args = parser.parse_args()

    if args.capture:
        asyncio.run(capture_all_baselines())
    elif args.load:
        baselines = load_baselines()
        for case_id, bl in baselines.items():
            print(f"\n=== {case_id} ===")
            print(json.dumps(bl.to_dict(), indent=2))
    else:
        parser.print_help()
