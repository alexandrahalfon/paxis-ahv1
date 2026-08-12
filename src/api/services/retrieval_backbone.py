"""
Unified retrieval backbone.

`retrieve_evidence(mode=...)` is the single entry point callers should use
for all study retrieval. It dispatches to the existing pipelines
(multispecialty / comprehensive / fast) and converts each one's native
result into a shared `EvidenceBundle`.

The goal: downstream code (router, routes, patient-matching, trial-match)
talks to ONE shape instead of three, so cut-over to a single retrieval
backbone is mechanical rather than risky.

Mode map:
    "multispecialty"  → tumor_board fan-out + per-specialty merge
                        (multi_specialty_retrieval.retrieve_evidence_multispecialty)
    "comprehensive"   → ComprehensiveRetriever.retrieve_comprehensive
    "fast"            → EnhancedRAGService.query retrieval subset
                        (no generation / synthesis step)

The `EvidenceBundle` holds:
    studies          : List[EvidenceStudy]     — canonical study dicts
    chunks           : List[Dict]              — flattened chunks
    extracted_axes   : Dict                    — from query structuring
    eligibility      : Dict                    — per-doc verdicts (if any)
    source_provenance: Dict[doc_id, str]       — qdrant/pg/pto/both
    metadata         : Dict                    — timings, flags, etc.

`EvidenceStudy` is deliberately smaller than the per-pipeline dataclasses
(StudyEvidence / LightweightStudy) — only the fields every caller needs.
Internal per-pipeline bookkeeping stays on the original objects and is
discarded at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Canonical per-study dataclass ──────────────────────────────────────────


@dataclass
class EvidenceStudy:
    doc_id: str
    title: str
    citation: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    score: float = 0.0                       # rerank/cross-encoder score if set, else initial
    source: str = "qdrant"                   # "qdrant" / "postgres" / "pto" / "both"
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    sections_covered: List[str] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)   # only populated for multispecialty
    match_score: Optional[float] = None                    # PG match score 0-100
    match_breakdown: Optional[Dict[str, Any]] = None       # PGMatchBreakdown.to_dict()
    axis_mismatches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "citation": self.citation,
            "year": self.year,
            "category": self.category,
            "score": self.score,
            "source": self.source,
            "sections_covered": list(self.sections_covered),
            "specialties": list(self.specialties),
            "chunk_count": len(self.chunks),
        }
        if self.match_score is not None:
            d["match_score"] = self.match_score
        if self.match_breakdown is not None:
            d["match_breakdown"] = self.match_breakdown
        if self.axis_mismatches:
            d["axis_mismatches"] = list(self.axis_mismatches)
        return d


@dataclass
class EvidenceBundle:
    """Single canonical shape returned by every retrieval mode."""

    mode: str
    studies: List[EvidenceStudy] = field(default_factory=list)
    extracted_axes: Dict[str, Any] = field(default_factory=dict)
    eligibility: Dict[str, Any] = field(default_factory=dict)
    source_provenance: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        """Flattened list of chunks across all studies.

        Each chunk dict is augmented with its study's identifiers so
        downstream consumers (numerical validator, citation checker)
        can trace it back.
        """
        flat: List[Dict[str, Any]] = []
        for s in self.studies:
            for c in s.chunks:
                flat.append({
                    **c,
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "citation": s.citation,
                    "year": s.year,
                    "category": s.category,
                    "_study_source": s.source,
                })
        return flat

    def doc_ids(self) -> List[str]:
        return [s.doc_id for s in self.studies]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "studies": [s.to_dict() for s in self.studies],
            "extracted_axes": self.extracted_axes,
            "eligibility": self.eligibility,
            "source_provenance": self.source_provenance,
            "metadata": self.metadata,
        }


# ─── Entry point ────────────────────────────────────────────────────────────


async def retrieve_evidence(
    query_text: str,
    *,
    mode: str = "comprehensive",
    category: Optional[str] = None,
    max_studies: int = 15,
    chunks_per_study: int = 8,
    query_type: str = "treatment_recommendation",
    force_all_agents: bool = False,
    bundle: Optional[Any] = None,
) -> EvidenceBundle:
    """Run retrieval under one of three backbones and return a unified bundle.

    Args:
        query_text: Raw clinical narrative or question.
        mode: "multispecialty" | "comprehensive" | "fast".
        category: Optional Qdrant category filter (e.g. "head_neck").
        max_studies: Cap on studies returned.
        chunks_per_study: Per-study chunk cap (respected by comprehensive / fast).
        query_type: Passed to pipelines that do query-type routing.
        force_all_agents: Only used by mode="multispecialty". Bypasses each
            agent's `relevance_filter` so a treatment-name query still fans
            out to every specialty.
        bundle: Optional pre-built `PatientCaseBundle` for mode="multispecialty"
            so callers that already did LLM extraction don't pay for it twice.
    """
    if mode == "multispecialty":
        return await _retrieve_multispecialty(
            query_text=query_text,
            category=category,
            max_studies=max_studies,
            query_type=query_type,
            force_all_agents=force_all_agents,
            bundle=bundle,
        )
    if mode == "comprehensive":
        return await _retrieve_comprehensive(
            query_text=query_text,
            category=category,
            max_studies=max_studies,
            chunks_per_study=chunks_per_study,
            query_type=query_type,
        )
    if mode == "fast":
        return await _retrieve_fast(
            query_text=query_text,
            category=category,
            max_studies=max_studies,
            chunks_per_study=chunks_per_study,
            query_type=query_type,
        )
    raise ValueError(
        f"retrieve_evidence: unknown mode {mode!r}; "
        f"expected 'multispecialty' | 'comprehensive' | 'fast'"
    )


# ─── Mode adapters ──────────────────────────────────────────────────────────


async def _retrieve_multispecialty(
    query_text: str,
    category: Optional[str],
    max_studies: int,
    query_type: str,
    force_all_agents: bool,
    bundle: Optional[Any],
) -> EvidenceBundle:
    from src.api.services.multi_specialty_retrieval import (
        retrieve_evidence_multispecialty,
    )
    ms = await retrieve_evidence_multispecialty(
        case_text=query_text,
        query_type=query_type,
        category=category,
        max_studies=max_studies,
        bundle=bundle,
        force_all_agents=force_all_agents,
    )

    studies: List[EvidenceStudy] = []
    provenance: Dict[str, str] = {}
    for s in ms.merged_studies:
        studies.append(EvidenceStudy(
            doc_id=getattr(s, "doc_id", "") or "",
            title=getattr(s, "title", "") or "",
            citation=getattr(s, "citation", None),
            year=getattr(s, "year", None),
            category=getattr(s, "category", None),
            score=float(getattr(s, "rerank_score", 0) or 0),
            source=getattr(s, "source", "qdrant") or "qdrant",
            chunks=list(getattr(s, "chunks", []) or []),
            sections_covered=list(getattr(s, "sections_covered", []) or []),
            specialties=list(getattr(s, "specialties", []) or []),
        ))
        provenance[studies[-1].doc_id] = studies[-1].source

    bundle_obj = ms.bundle
    axes = {
        "has_patient_context": bundle_obj.has_patient_context,
        "trajectory_flags": list(bundle_obj.trajectory_flags),
        "metastatic_sites": list(bundle_obj.metastatic_sites),
        "surgical_candidate": bundle_obj.surgical_candidate,
        "category": bundle_obj.category,
    }
    return EvidenceBundle(
        mode="multispecialty",
        studies=studies,
        extracted_axes=axes,
        source_provenance=provenance,
        metadata={
            "per_specialty_counts": {
                k: len(v) for k, v in (ms.per_specialty or {}).items()
            },
            "skipped": dict(ms.skipped or {}),
            **dict(ms.metadata or {}),
        },
    )


async def _retrieve_comprehensive(
    query_text: str,
    category: Optional[str],
    max_studies: int,
    chunks_per_study: int,
    query_type: str,
) -> EvidenceBundle:
    from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
    retriever = get_comprehensive_retriever()
    result = await retriever.retrieve_comprehensive(
        query_text=query_text,
        category=category,
        max_studies=max_studies,
        chunks_per_study=chunks_per_study,
        max_guidelines=5,
    )

    studies: List[EvidenceStudy] = []
    provenance: Dict[str, str] = {}
    for s in result.studies:
        studies.append(EvidenceStudy(
            doc_id=s.doc_id,
            title=s.title,
            citation=s.citation,
            year=s.year,
            category=s.category,
            score=float(s.rerank_score or s.initial_score or 0),
            source=s.source or "qdrant",
            chunks=list(s.chunks or []),
            sections_covered=list(s.sections_covered or []),
            match_score=getattr(s, "match_score", None),
            match_breakdown=getattr(s, "match_breakdown", None),
            axis_mismatches=list(getattr(s, "axis_mismatches", []) or []),
        ))
        provenance[s.doc_id] = s.source or "qdrant"

    return EvidenceBundle(
        mode="comprehensive",
        studies=studies,
        extracted_axes=dict(result.query_structure or {}),
        source_provenance=provenance,
        metadata={
            "total_chunks": result.total_chunks,
            "retrieval_time_ms": result.retrieval_time_ms,
            "phase1_qdrant_docs": result.phase1_qdrant_docs,
            "phase1_postgres_docs": result.phase1_postgres_docs,
            "phase2_docs_searched": result.phase2_docs_searched,
            "expanded_query": result.expanded_query,
        },
    )


async def _retrieve_fast(
    query_text: str,
    category: Optional[str],
    max_studies: int,
    chunks_per_study: int,
    query_type: str,
) -> EvidenceBundle:
    """Fast retrieval: ComprehensiveRetriever with tight caps.

    P1's historical "fast" path and its comprehensive path share the same
    Qdrant + PG + PTO backbone — they differ only in size caps and
    chunks-per-study. Running ComprehensiveRetriever with a tight budget
    is both cheaper than the full comprehensive path and keeps the
    retrieval wiring in one place (fewer code paths to maintain).
    """
    # Halve budgets for a "fast" run; callers who want full depth use
    # mode="comprehensive" explicitly.
    fast_max_studies = max(1, min(max_studies, 5))
    fast_chunks = max(1, min(chunks_per_study, 4))

    bundle = await _retrieve_comprehensive(
        query_text=query_text,
        category=category,
        max_studies=fast_max_studies,
        chunks_per_study=fast_chunks,
        query_type=query_type,
    )
    bundle.mode = "fast"
    return bundle
