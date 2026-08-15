"""
Multi-Specialty Evidence Retrieval

Runs the SAME case-bundle + specialty-agent fan-out + lightweight Qdrant
retrieval that the tumor board uses, but stops BEFORE the LLM expert
assessment step. The output is a flat (and per-specialty) bundle of
retrieved studies that the trial-match, patient-matching, and treatment-
comparison pipelines can convert into their own response shapes.

Pipeline (mirrors `TumorBoardOrchestrator.present_case` up to the cut-line):

    1. build_case_bundle(case_text)            ← regex + LLM 8-axis + ontology
    2. for each SpecialtyAgent on the panel:
         a. relevance_filter(bundle)            ← skip if not applicable
         b. build_sub_queries(bundle)           ← specialty-aware queries
         c. lightweight_search(...) per query   ← direct Qdrant, capped concurrency
         d. _merge_studies()                    ← dedupe by doc_id
    3. cross-specialty merge with multi-specialty boost + provenance tracking

The result is a `MultiSpecialtyEvidence` that exposes:

  - `bundle`             — the PatientCaseBundle (so callers can re-use the
                            extracted axes / inferred terms / summary lines)
  - `merged_studies`     — flat list of studies, deduped across specialties,
                            sorted by best score, with `source` /
                            `sections_covered` / `specialties` populated so
                            they can be passed to existing converters that
                            expect `StudyEvidence`-shaped objects
  - `per_specialty`      — `Dict[specialty, List[LightweightStudy]]`
  - `study_specialties`  — `Dict[doc_id, List[specialty]]`
  - `sub_queries_used`   — `Dict[specialty, List[str]]`
  - `skipped`            — `Dict[specialty, reason]`
  - `metadata`           — timings + counts

NOTE: this module deliberately does NOT call the LLM expert synthesis
step. Callers that need the LLM verdict should use
`TumorBoardOrchestrator.present_case` instead.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.api.services.tumor_board.agents import ALL_AGENT_CLASSES
from src.api.services.tumor_board.base_agent import SpecialtyAgent
from src.api.services.tumor_board.case_bundle import (
    PatientCaseBundle,
    build_case_bundle,
)
from src.api.services.tumor_board.retrieval import (
    LightweightStudy,
    lightweight_search,
)


# ─── Result dataclass ──────────────────────────────────────────────────────


@dataclass
class MultiSpecialtyEvidence:
    """Output of `retrieve_evidence_multispecialty`."""

    bundle: PatientCaseBundle
    merged_studies: List[LightweightStudy]
    per_specialty: Dict[str, List[LightweightStudy]] = field(default_factory=dict)
    study_specialties: Dict[str, List[str]] = field(default_factory=dict)
    sub_queries_used: Dict[str, List[str]] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_summary": self.bundle.summary_lines(),
            "merged_studies": [
                {
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "year": s.year,
                    "rerank_score": s.rerank_score,
                    "source": s.source,
                    "specialties": s.specialties,
                    "chunks": len(s.chunks),
                }
                for s in self.merged_studies
            ],
            "per_specialty_counts": {
                k: len(v) for k, v in self.per_specialty.items()
            },
            "study_specialties": self.study_specialties,
            "sub_queries_used": self.sub_queries_used,
            "skipped": self.skipped,
            "metadata": self.metadata,
        }


# ─── Singleton agent panel (stateless — safe to reuse) ────────────────────


_PANEL: Optional[List[SpecialtyAgent]] = None


def _get_panel() -> List[SpecialtyAgent]:
    """Lazy-construct the 6-agent panel. Agents are stateless so we can
    keep one instance per process."""
    global _PANEL
    if _PANEL is None:
        _PANEL = [cls() for cls in ALL_AGENT_CLASSES]
    return _PANEL


# ─── Per-specialty merge helper ────────────────────────────────────────────


def _merge_within_specialty(
    retrieval_results: List[Any],
) -> List[LightweightStudy]:
    """
    Dedupe studies returned by a single specialty's sub-queries by `doc_id`,
    keeping the highest-scoring observation per study.

    Mirrors `SpecialtyAgent._merge_studies` so the per-specialty dedup is
    identical to what the tumor board does.
    """
    merged: Dict[str, LightweightStudy] = {}
    for res in retrieval_results:
        if isinstance(res, Exception) or res is None:
            continue
        if not isinstance(res, list):
            continue
        for study in res:
            doc_id = getattr(study, "doc_id", None)
            if not doc_id:
                continue
            score = float(getattr(study, "rerank_score", 0) or 0)
            existing = merged.get(doc_id)
            if existing is None or score > float(
                getattr(existing, "rerank_score", 0) or 0
            ):
                merged[doc_id] = study
    return sorted(
        merged.values(),
        key=lambda s: (
            float(getattr(s, "rerank_score", 0) or 0),
            float(getattr(s, "initial_score", 0) or 0),
        ),
        reverse=True,
    )


# ─── Per-agent fan-out (one specialty's worth of queries) ──────────────────


async def _run_one_specialty(
    agent: SpecialtyAgent,
    bundle: PatientCaseBundle,
    category: Optional[str],
    force: bool,
) -> Tuple[str, List[LightweightStudy], List[str], Optional[str]]:
    """
    Returns (specialty, studies, sub_queries, skip_reason_or_None).

    `force=True` bypasses `relevance_filter` (used by treatment-comparison
    where a treatment-only query lacks rich patient context but we still
    want every specialty to take a shot at the literature).
    """
    if not force:
        skip = agent.relevance_filter(bundle)
        if skip:
            return agent.specialty, [], [], skip

    sub_queries = [
        q for q in agent.build_sub_queries(bundle) if q and q.strip()
    ]
    sub_queries = sub_queries[: agent.max_sub_queries]
    if not sub_queries:
        return (
            agent.specialty,
            [],
            [],
            "no specialty sub-queries could be built from the extracted case",
        )

    # Fan out the agent's sub-queries through lightweight_search. The
    # tumor_board.retrieval module already enforces a global asyncio
    # semaphore (QDRANT_CONCURRENCY=6), so kicking off many gather()
    # calls here will queue politely instead of stampeding Qdrant.
    tasks = [
        lightweight_search(
            q,
            limit_points=agent.points_per_query,
            max_chunks_per_study=agent.max_chunks_per_study,
            category=category,
        )
        for q in sub_queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged = _merge_within_specialty(results)
    print(
        f"[MultiSpecialty:{agent.specialty}] {len(merged)} unique studies "
        f"from {len(sub_queries)} sub-queries"
    )
    return agent.specialty, merged, sub_queries, None


# ─── Main entry point ──────────────────────────────────────────────────────


async def retrieve_evidence_multispecialty(
    case_text: str,
    query_type: str = "treatment_recommendation",
    category: Optional[str] = None,
    max_studies: int = 15,
    bundle: Optional[PatientCaseBundle] = None,
    force_all_agents: bool = False,
) -> MultiSpecialtyEvidence:
    """
    Run the tumor-board specialty fan-out + per-specialty merge, stopping
    BEFORE the LLM expert assessment step.

    Args:
        case_text: Raw clinical narrative or query string. If `bundle` is
            provided this is only used as the fallback `bundle.raw_text`.
        query_type: Pre-classified query type passed to the case bundle
            extractor. Defaults to "treatment_recommendation".
        category: Optional Qdrant category filter (e.g. "head_neck"). When
            provided it is forwarded into every per-agent
            `lightweight_search` call.
        max_studies: Cap on the number of studies in `merged_studies`.
        bundle: Optional pre-built `PatientCaseBundle`. Useful when callers
            already extracted patient axes upstream and don't want to re-run
            the LLM extraction inside this module.
        force_all_agents: When True, every agent's `relevance_filter` is
            bypassed. Used by the treatment-comparison pipeline where a
            treatment-name query has no rich patient context but we still
            want every specialty to retrieve evidence.

    Returns:
        `MultiSpecialtyEvidence` with merged + per-specialty results.
    """
    t_start = time.perf_counter()

    try:
        from src.api.services import pipeline_metrics as _pm
        if _pm.current() is None:
            _pm.start("p3")
    except Exception:
        pass

    if bundle is None:
        if not case_text or not case_text.strip():
            raise ValueError("case_text or bundle must be provided")
        t_bundle = time.perf_counter()
        bundle = await build_case_bundle(case_text, query_type=query_type)
        t_bundle_ms = (time.perf_counter() - t_bundle) * 1000
    else:
        t_bundle_ms = 0.0

    if bundle.has_patient_context:
        try:
            from src.api.services import pipeline_metrics as _pm
            if _pm.current() is not None:
                _pm.current().event("has_patient_context")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("  MULTI-SPECIALTY EVIDENCE RETRIEVAL (no LLM synthesis)")
    print("=" * 80)
    print(
        f"  Case ({len(case_text)} chars): "
        f"{case_text[:160]}{'...' if len(case_text) > 160 else ''}"
    )
    print(
        f"  Bundle: has_patient_context={bundle.has_patient_context}, "
        f"force_all_agents={force_all_agents}, category={category}"
    )

    panel = _get_panel()
    print(f"  Panel: {[a.display_name for a in panel]}")

    # Fan out across all specialties in parallel. Each agent itself fans out
    # its 1-5 specialty sub-queries through `lightweight_search`, and the
    # global semaphore inside that helper caps concurrent Qdrant calls.
    tasks = [
        _run_one_specialty(
            agent=agent,
            bundle=bundle,
            category=category,
            force=force_all_agents,
        )
        for agent in panel
    ]
    pair_results = await asyncio.gather(*tasks, return_exceptions=True)

    per_specialty: Dict[str, List[LightweightStudy]] = {}
    sub_queries_used: Dict[str, List[str]] = {}
    skipped: Dict[str, str] = {}

    for agent, pr in zip(panel, pair_results):
        if isinstance(pr, Exception):
            print(
                f"  [MultiSpecialty] {agent.specialty} raised: "
                f"{type(pr).__name__}: {pr}"
            )
            skipped[agent.specialty] = f"agent error: {pr}"
            continue
        specialty, studies, sub_queries, skip_reason = pr
        if skip_reason:
            skipped[specialty] = skip_reason
            print(f"  [MultiSpecialty] {specialty:<22} SKIPPED — {skip_reason}")
            continue
        per_specialty[specialty] = studies
        sub_queries_used[specialty] = sub_queries

    # ── Cross-specialty merge ────────────────────────────────────────────
    cross_merged: Dict[str, LightweightStudy] = {}
    study_specialties: Dict[str, List[str]] = defaultdict(list)

    for specialty, studies in per_specialty.items():
        for s in studies:
            doc_id = s.doc_id
            if not doc_id:
                continue
            study_specialties[doc_id].append(specialty)
            existing = cross_merged.get(doc_id)
            if existing is None:
                cross_merged[doc_id] = s
            else:
                # Keep the chunks/metadata from the highest-scoring observation
                if (s.rerank_score or 0) > (existing.rerank_score or 0):
                    cross_merged[doc_id] = s

    # ── Multi-specialty consensus boost ──────────────────────────────────
    # A study that surfaces independently from 2+ specialties is more likely
    # to be clinically relevant than one that only one agent found. Apply a
    # small multiplicative boost (5% per extra specialty, capped at +20%).
    for doc_id, study in cross_merged.items():
        n = len(set(study_specialties[doc_id]))
        if n >= 2:
            boost = min(0.20, 0.05 * (n - 1))
            study.rerank_score = float(study.rerank_score or 0.0) * (1.0 + boost)

    # ── Populate provenance fields on each LightweightStudy so the existing
    #    converters (which expect StudyEvidence-shaped objects) work as-is.
    for doc_id, study in cross_merged.items():
        specs = sorted(set(study_specialties[doc_id]))
        study.specialties = specs
        if specs:
            study.source = "tumor_board:" + "+".join(specs)
        else:
            study.source = "tumor_board"
        try:
            study.sections_covered = {
                c.get("section") for c in (study.chunks or [])
                if c.get("section")
            }
        except Exception:
            study.sections_covered = set()

    sorted_studies = sorted(
        cross_merged.values(),
        key=lambda s: (s.rerank_score or 0.0, s.initial_score or 0.0),
        reverse=True,
    )[:max_studies]

    # ── Hard eligibility filter ──────────────────────────────────────────
    # Remove studies that clearly don't match the patient on cancer type,
    # histology, stage, prior therapies, or biomarkers. Uses the same
    # gpt-4o-mini-based filter as the standard RAG pipeline.
    pre_elig = len(sorted_studies)
    try:
        from src.api.services.patient_eligibility_boost_service import (
            run_patient_eligibility_check,
        )
        from openai import OpenAI as _OpenAI
        from src.core.config import settings as _settings

        # Convert LightweightStudy → flat chunk dicts for the service
        elig_chunks = []
        for s in sorted_studies:
            best_text = ""
            for c in (s.chunks or [])[:3]:
                best_text += " " + (c.get("text") or "")
            elig_chunks.append({
                "doc_id": s.doc_id,
                "title": s.title,
                "text": best_text.strip()[:1500],
                "score": s.rerank_score or 0.0,
            })

        elig_client = _OpenAI(api_key=_settings.openai_api_key)
        filtered_elig, elig_meta = await run_patient_eligibility_check(
            query=case_text,
            chunks=elig_chunks,
            openai_client=elig_client,
        )

        if elig_meta.get("patient_context_detected"):
            surviving_ids = {c.get("doc_id") for c in filtered_elig}
            sorted_studies = [
                s for s in sorted_studies if s.doc_id in surviving_ids
            ]
            removed = pre_elig - len(sorted_studies)
            if removed:
                print(
                    f"  [MultiSpecialty:HardEligibility] Removed {removed} "
                    f"studies ({pre_elig} → {len(sorted_studies)})"
                )
            else:
                print(
                    f"  [MultiSpecialty:HardEligibility] All "
                    f"{len(sorted_studies)} studies passed"
                )
        else:
            print(
                f"  [MultiSpecialty:HardEligibility] No patient context "
                f"— skipped"
            )
    except Exception as e:
        print(
            f"  [MultiSpecialty:HardEligibility] Failed (continuing "
            f"without): {e}"
        )

    total_ms = (time.perf_counter() - t_start) * 1000

    print(
        f"  [MultiSpecialty] Done in {total_ms:.0f}ms "
        f"(bundle={t_bundle_ms:.0f}ms): "
        f"{len(sorted_studies)} merged studies, "
        f"{len(per_specialty)} specialties contributed, "
        f"{len(skipped)} skipped"
    )
    for s in sorted_studies[:5]:
        print(
            f"    - [{s.rerank_score:.3f}] {(s.title or '')[:60]} "
            f"({'+'.join(s.specialties)})"
        )

    metadata = {
        "total_elapsed_ms": total_ms,
        "case_bundle_elapsed_ms": t_bundle_ms,
        "used_llm_extraction": bundle.used_llm_extraction,
        "agent_count": len(panel),
        "agents_run": list(per_specialty.keys()),
        "agents_skipped": list(skipped.keys()),
        "trajectory_flags": bundle.trajectory_flags,
        "metastatic_sites": bundle.metastatic_sites,
        "surgical_candidate": bundle.surgical_candidate,
        "force_all_agents": force_all_agents,
        "category_filter": category,
    }

    try:
        from src.api.services import pipeline_metrics as _pm
        _pm_cur = _pm.current()
        if _pm_cur is not None:
            print(_pm_cur.summary_line())
    except Exception:
        pass

    return MultiSpecialtyEvidence(
        bundle=bundle,
        merged_studies=sorted_studies,
        per_specialty=per_specialty,
        study_specialties=dict(study_specialties),
        sub_queries_used=sub_queries_used,
        skipped=skipped,
        metadata=metadata,
    )


# ─── Convenience: convert merged studies to the chunk shape used by the
#    legacy patient-matching validator + match builder ─────────────────────


def studies_to_validator_chunks(
    studies: List[LightweightStudy],
) -> List[Dict[str, Any]]:
    """
    Convert `LightweightStudy` objects into the chunk dict shape consumed by
    `SimplePatientMatchingService._validate_matches_semantically` /
    `_normalize_scores` / `_build_match_data` so the existing patient-
    matching response shape can be reused unchanged.
    """
    chunks: List[Dict[str, Any]] = []
    for study in studies:
        # Pick the highest-scored chunk as the representative
        best = None
        best_score = float("-inf")
        for c in study.chunks or []:
            cs = float(c.get("score") or 0.0)
            if cs > best_score:
                best_score = cs
                best = c
        if best is None:
            best = {"text": "", "section": None, "doc_meta": {}}
            best_score = float(study.rerank_score or 0.0)

        doc_meta = dict(best.get("doc_meta") or {})
        if study.title and not doc_meta.get("title"):
            doc_meta["title"] = study.title
        if study.citation and not doc_meta.get("citation"):
            doc_meta["citation"] = study.citation
        if study.year and not doc_meta.get("year"):
            doc_meta["year"] = study.year

        payload = {
            "doc_id": study.doc_id,
            "doc_meta": doc_meta,
            "text": best.get("text", ""),
            "section": best.get("section"),
            "chunk_type": best.get("chunk_type"),
            # Forward doc_level_* metadata so patient_match_scorer can
            # score each study without re-fetching from Qdrant. The
            # source field is populated by tumor_board/retrieval.py
            # (commit aligning Find Trials with main RAG fixes).
            "metadata": best.get("metadata", {}) or {},
        }

        chunks.append({
            "point_id": best.get("chunk_id") or study.doc_id,
            "score_dense": best_score,
            "score_rerank": float(study.rerank_score or best_score or 0.0),
            "payload": payload,
            "_source": study.source,
            "_specialties": list(study.specialties),
        })
    return chunks
