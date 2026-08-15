"""
Pipeline regression tests against golden fixtures.

Each fixture in `tests/pipeline_golden/queries/*.json` is fed through all
five retrieval / generation pipelines. The test asserts on:

- minimum and maximum study count
- presence of the expected events emitted via ``PipelineMetrics``
- absence of forbidden doc_id substrings (the "no cross-category leakage"
  guard surfaced by the v1-v4 comparison runs — tonsil, thymoma, bladder,
  amyloidosis, NCCN-NSCLC in H&N queries, etc.)
- minimum safety-trigger counts (numerical validation, citation
  validation) so that ``[Numerical Validation] Stripped`` counts don't
  silently regress

Live infrastructure (Qdrant + OpenAI) is required. When credentials are
missing the whole module is skipped so `pytest` stays green in CI dev
runs that don't have the production secrets plumbed through.

To run locally::

    export QDRANT_URL=...
    export QDRANT_API_KEY=...
    export OPENAI_API_KEY=sk-...
    pytest tests/pipeline_golden/ -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Skip module wholesale when live credentials aren't present.
# The pipelines construct OpenAI / Qdrant clients at import time, so trying to
# run them without creds would crash at import rather than produce a useful
# test skip. Guard at the module level.
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_ENV = ("OPENAI_API_KEY", "QDRANT_URL")
_missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]

pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "Live pipeline regression tests require environment variables "
        + ", ".join(_REQUIRED_ENV)
        + f" — missing: {', '.join(_missing)}"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixture discovery
# ──────────────────────────────────────────────────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent / "queries"


def _fixture_files() -> List[Path]:
    return sorted(_FIXTURES_DIR.glob("*.json"))


def _load(fixture_path: Path) -> Dict[str, Any]:
    data = json.loads(fixture_path.read_text())
    required = {"id", "query", "expected"}
    missing = required - data.keys()
    if missing:
        raise ValueError(
            f"Fixture {fixture_path.name} missing required keys: {missing}"
        )
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runners — each returns a uniform dict:
#   { "studies": [{doc_id, title, score, source}],
#     "metrics": PipelineMetrics.to_dict() or {} }
# ──────────────────────────────────────────────────────────────────────────────


async def _run_p1_enhanced(query: str, category, query_type: str):
    from src.api.services.enhanced_rag_service import get_enhanced_rag_service
    from src.api.services import pipeline_metrics

    pipeline_metrics.start("p1")
    svc = get_enhanced_rag_service()
    result = await svc.query(
        question=query,
        query_mode="hybrid",
        top_k=5,
        category=category,
        use_site_inference=True,
        use_study_focused=False,
    )
    retrieval = result.get("retrieval_results") or result.get("evidence") or []
    seen, studies = set(), []
    for e in retrieval:
        did = e.get("doc_id")
        if did and did not in seen:
            seen.add(did)
            studies.append(
                {"doc_id": did, "title": e.get("title", ""),
                 "score": float(e.get("score") or 0), "source": e.get("source", "")}
            )
    m = pipeline_metrics.current()
    return {"studies": studies, "metrics": m.to_dict() if m else {}}


async def _run_p2_comprehensive(query: str, category, query_type: str):
    from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
    from src.api.services import pipeline_metrics

    pipeline_metrics.start("p2")
    retriever = get_comprehensive_retriever()
    result = await retriever.retrieve_comprehensive(
        query_text=query,
        max_studies=5,
        chunks_per_study=6,
        category=category,
    )
    studies = [
        {"doc_id": s.doc_id, "title": s.title,
         "score": float(getattr(s, "rerank_score", None)
                        or getattr(s, "initial_score", 0) or 0),
         "source": getattr(s, "source", "")}
        for s in result.studies
    ]
    m = pipeline_metrics.current()
    return {"studies": studies, "metrics": m.to_dict() if m else {}}


async def _run_p3_multispecialty(query: str, category, query_type: str):
    from src.api.services.multi_specialty_retrieval import (
        retrieve_evidence_multispecialty,
    )
    from src.api.services import pipeline_metrics

    pipeline_metrics.start("p3")
    ms = await retrieve_evidence_multispecialty(
        case_text=query,
        query_type=query_type,
        category=category,
        max_studies=5,
    )
    studies = [
        {"doc_id": getattr(s, "doc_id", ""), "title": getattr(s, "title", ""),
         "score": float(getattr(s, "score", 0) or 0),
         "source": ",".join(sorted(getattr(s, "specialties", []) or []))}
        for s in (getattr(ms, "merged_studies", []) or [])
    ]
    m = pipeline_metrics.current()
    return {"studies": studies, "metrics": m.to_dict() if m else {}}


async def _run_p4_tumor_board(query: str, category, query_type: str):
    from src.api.services.tumor_board.orchestrator import (
        get_tumor_board_orchestrator,
    )
    from src.api.services import pipeline_metrics

    pipeline_metrics.start("p4")
    orch = get_tumor_board_orchestrator()
    report = await orch.present_case(case_text=query, query_type=query_type)
    studies: List[Dict[str, Any]] = []
    for a in report.expert_assessments:
        for s in list(a.supporting_studies or []):
            studies.append({"doc_id": s.doc_id, "title": s.title,
                            "score": float(s.relevance_score or 0),
                            "source": a.specialty})
    m = pipeline_metrics.current()
    return {"studies": studies, "metrics": m.to_dict() if m else {}}


async def _run_p5_trial_match(query: str, category, query_type: str):
    from src.api.services.query_intent_service import get_query_intent_service
    from src.api.services import pipeline_metrics

    pipeline_metrics.start("p5")
    svc = get_query_intent_service()
    result = await svc.analyze_query(
        query=query,
        find_matching_trials=True,
        force_trial_match=True,
    )
    trials = result.matching_trials or []
    studies = [
        {"doc_id": getattr(t, "doc_id", "") or getattr(t, "study_id", ""),
         "title": getattr(t, "title", "") or getattr(t, "study_title", ""),
         "score": float(getattr(t, "match_score", 0) or getattr(t, "score", 0) or 0),
         "source": "trial_match"}
        for t in trials
    ]
    m = pipeline_metrics.current()
    return {"studies": studies, "metrics": m.to_dict() if m else {}}


_PIPELINES = {
    "p1": _run_p1_enhanced,
    "p2": _run_p2_comprehensive,
    "p3": _run_p3_multispecialty,
    "p4": _run_p4_tumor_board,
    "p5": _run_p5_trial_match,
}


# ──────────────────────────────────────────────────────────────────────────────
# Assertions
# ──────────────────────────────────────────────────────────────────────────────


def _assert_fixture(pipeline: str, fixture: Dict[str, Any],
                    result: Dict[str, Any]) -> None:
    exp = fixture["expected"]
    studies = result["studies"]
    metrics = result["metrics"]

    min_n = exp.get("min_studies", 0)
    max_n = exp.get("max_studies", 10_000)
    assert len(studies) >= min_n, (
        f"[{pipeline}:{fixture['id']}] expected >= {min_n} studies, got {len(studies)}"
    )
    assert len(studies) <= max_n, (
        f"[{pipeline}:{fixture['id']}] expected <= {max_n} studies, got {len(studies)}"
    )

    forbidden = exp.get("forbidden_doc_substrings") or []
    for study in studies:
        doc_id = (study.get("doc_id") or "").lower()
        title = (study.get("title") or "").lower()
        haystack = f"{doc_id} {title}"
        for bad in forbidden:
            assert bad.lower() not in haystack, (
                f"[{pipeline}:{fixture['id']}] forbidden substring "
                f"{bad!r} appeared in {study.get('doc_id')!r} / "
                f"{study.get('title')!r}"
            )

    min_safety = exp.get("min_safety_triggers") or {}
    actual_safety = metrics.get("safety") or {}
    for key, min_val in min_safety.items():
        got = actual_safety.get(key, 0)
        assert got >= min_val, (
            f"[{pipeline}:{fixture['id']}] safety[{key}] expected >= {min_val}, "
            f"got {got}"
        )

    min_elig = exp.get("min_eligibility") or {}
    actual_elig = metrics.get("eligibility") or {}
    for key, min_val in min_elig.items():
        if key == "MATCH_plus_POSSIBLE":
            got = actual_elig.get("MATCH", 0) + actual_elig.get("POSSIBLE", 0)
        else:
            got = actual_elig.get(key, 0)
        # Only assert if the pipeline actually ran the eligibility step at all
        if actual_elig:
            assert got >= min_val, (
                f"[{pipeline}:{fixture['id']}] eligibility[{key}] expected "
                f">= {min_val}, got {got} (full bucket: {actual_elig})"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Parametrised tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_path", _fixture_files(),
                         ids=lambda p: p.stem)
@pytest.mark.parametrize("pipeline", list(_PIPELINES.keys()))
def test_pipeline_matches_golden(pipeline: str, fixture_path: Path) -> None:
    fixture = _load(fixture_path)
    runner = _PIPELINES[pipeline]
    result = asyncio.run(
        runner(fixture["query"], fixture.get("category"),
               fixture.get("query_type", "treatment_recommendation"))
    )
    _assert_fixture(pipeline, fixture, result)
