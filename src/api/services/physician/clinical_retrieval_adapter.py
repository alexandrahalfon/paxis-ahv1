"""
Legacy Clinical Retrieval Adapter (2026-08-12 convergence Sprint C item
15)

"Your existing retrieval stack is one of the strongest parts of Paxis"
— per the convergence plan, Sprint C wraps it, it does not replace it.
comprehensive_retrieval.py's dense + lexical hybrid scoring + cross-
encoder gate + study-profile/PG matching + PTO stays exactly as it is;
this module is the one-directional adapter that reshapes its output
(StudyEvidence, ComprehensiveRetrievalResult — CLAUDE.md's own "do not
change" list) into EvidenceCandidate (Sprint A item 1), so the physician
path can eventually flow through the same downstream applicability/
hierarchy/packet/claim-validation machinery the patient path already
does, without reprocessing the KB or rewriting a single line of the
retriever itself.

Granularity: StudyEvidence is a STUDY (one doc_id, many chunks);
EvidenceCandidate is a CHUNK (text/section/point-level, matching
multi_corpus_retriever's own granularity — see evidence_candidate.py).
adapt_legacy_study() expands one study's chunks into one
EvidenceCandidate per chunk, mirroring exactly how multi_corpus_
retriever preserves distinct sections of the same document as distinct
candidates rather than collapsing them.

Deliberately duck-typed, not importing StudyEvidence itself: comprehensive_
retrieval.py pulls in a heavy dependency graph (the cross-encoder model,
etc.) that a module whose whole job is a one-way field adapter shouldn't
need at import time — matching the inline-import convention every other
evidence-layer module in this codebase already uses for exactly this
reason. Fields are read via getattr()/dict .get() with safe defaults,
so a StudyEvidence instance (or anything else shaped like one) works
without a hard import dependency.

Two fields this codebase genuinely doesn't have an equivalent for, left
honest rather than faked (see EvidenceCandidate's own docstring for the
same convention applied to source_governance.py's collection_target/
acquisition_mode gap):
  - version_id: the legacy clinical corpus predates evidence_ingestion_
    service.py's is_current/superseded_by version tracking entirely
    (architecture review item 13, "existing clinical literature still
    uses a different provenance model") — always None here.
  - rrf_score: this pipeline fuses dense + lexical scores via a weighted
    sum (_apply_hybrid_scoring's dense_weight/lexical_weight), not
    literal Reciprocal Rank Fusion — always None; bm25_score is the
    closest available signal (a Jaccard-like lexical overlap score, not
    literal BM25 either, but it fills the same "lexical match" role).

Nothing calls this yet — Sprint C item 20's physician orchestrator is
what will, after running the existing retriever unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.api.services.evidence.evidence_candidate import EvidenceCandidate


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Reads an attribute off a StudyEvidence-shaped object OR a plain
    dict with the same key, so this adapter works whether the caller
    passes a real dataclass instance or its .to_dict() form."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def adapt_legacy_chunk(study: Any, chunk: Dict[str, Any]) -> EvidenceCandidate:
    """Adapts one chunk dict from StudyEvidence.chunks (the shape
    comprehensive_retrieval.py's _phase3_document_search() /
    _apply_hybrid_scoring() actually produce — point_id/score_dense/
    score_lexical/score_crossencoder_gate/doc_meta/section/chunk_id/
    text) into one EvidenceCandidate, carrying the parent study's
    citation/year/category/match-scoring metadata along with it."""
    doc_meta = chunk.get("doc_meta") or {}
    point_id = chunk.get("point_id")
    year = _get(study, "year")

    return EvidenceCandidate(
        qdrant_point_id=str(point_id) if point_id is not None else None,
        document_id=chunk.get("doc_id") or _get(study, "doc_id"),
        version_id=None,  # see module docstring
        corpus="clinical_literature",
        source_key=None,  # legacy corpus predates source_registry.py adoption
        source_name=doc_meta.get("source_name") or doc_meta.get("journal"),
        authority_class=None,
        authority_score=None,
        title=doc_meta.get("title") or _get(study, "title") or "",
        section=chunk.get("section"),
        chunk_index=chunk.get("chunk_id"),
        url=doc_meta.get("url"),
        publication_date=str(year) if year else None,
        text=chunk.get("text") or "",
        semantic_score=chunk.get("score_dense"),
        bm25_score=chunk.get("score_lexical"),
        rrf_score=None,  # see module docstring
        cross_encoder_score=chunk.get("score_crossencoder_gate"),
        # Deliberately left unset -- clinical applicability against a
        # SPECIFIC patient's state is the physician applicability
        # scorer's job (Sprint C item 16), not this adapter's; it only
        # reshapes what the retriever already computed.
        applicability_score=None,
        applicability_components={},
        incompatibility_reasons=[],
        metadata={
            "citation": _get(study, "citation"),
            "year": year,
            "category": _get(study, "category"),
            "study_profile": {
                "match_score": _get(study, "match_score"),
                "match_breakdown": _get(study, "match_breakdown"),
                "axis_mismatches": _get(study, "axis_mismatches") or [],
                "soft_score_normalized": _get(study, "soft_score_normalized"),
                "patient_match_score": _get(study, "patient_match_score"),
                "patient_match_breakdown": _get(study, "patient_match_breakdown"),
                "evidence_type": _get(study, "evidence_type"),
                "source": _get(study, "source"),
            },
        },
    )


def adapt_legacy_study(study: Any) -> List[EvidenceCandidate]:
    """Expands one StudyEvidence's chunks into a list of
    EvidenceCandidate — see adapt_legacy_chunk() for the per-chunk
    mapping. A chunk with no text is skipped, matching multi_corpus_
    retriever._search_collection()'s own "no text, no candidate" rule."""
    chunks = _get(study, "chunks") or []
    return [
        adapt_legacy_chunk(study, chunk)
        for chunk in chunks
        if (chunk.get("text") or "").strip()
    ]


def adapt_legacy_results(studies: List[Any]) -> List[EvidenceCandidate]:
    """Adapts a full ComprehensiveRetrievalResult.studies list (or any
    list of StudyEvidence-shaped objects) into EvidenceCandidates."""
    out: List[EvidenceCandidate] = []
    for study in studies:
        out.extend(adapt_legacy_study(study))
    return out
