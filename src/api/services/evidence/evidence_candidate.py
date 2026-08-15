"""
EvidenceCandidate (2026-08-12 patient/physician convergence, Sprint A
item 1)

The canonical shape every retriever is meant to converge on — patient's
multi_corpus_retriever today, the legacy physician clinical retriever
once Sprint C's adapter (clinical_retrieval_adapter.py) lands, and any
corpus added after that. Per the convergence plan: "freeze the shared
contracts first" before building the physician integration on top of
them.

This module is deliberately NOT a rewrite of multi_corpus_retriever.py,
applicability_scorer.py, or evidence_packet_builder.py — those three
already work, are tested, and operate on plain dicts. Forcing them onto
a dataclass now would be exactly the "another redesign" the convergence
plan explicitly says not to do. Instead:

  - `EvidenceCandidate.from_multi_corpus_dict()` adapts the patient
    path's existing dict shape into this typed contract — read this to
    see exactly which legacy field maps to which canonical field; the
    physician-side legacy retriever adapter (Sprint C item 15) follows
    the same pattern for a very differently-shaped old hit.
  - `EvidenceCandidate.to_dict()` goes the other way, producing the
    exact dict shape applicability_scorer.score_candidate() and
    evidence_packet_builder.build_packet() already consume — so an
    EvidenceCandidate built by a NEW retriever can be dropped into the
    existing, tested patient-side scoring/packet pipeline unchanged.

Net effect: today, nothing calls this module yet and the patient path is
completely unaffected. It exists so Sprint C has a stable target to
adapt the physician retriever into, and so a future migration of the
patient path itself onto typed candidates is a mechanical swap rather
than a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceCandidate:
    qdrant_point_id: Optional[str] = None
    document_id: Optional[str] = None
    version_id: Optional[str] = None
    corpus: str = ""
    source_key: Optional[str] = None
    source_name: Optional[str] = None
    authority_class: Optional[str] = None
    authority_score: Optional[float] = None
    title: str = ""
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    url: Optional[str] = None
    publication_date: Optional[str] = None
    text: str = ""
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    applicability_score: Optional[float] = None
    applicability_components: Dict[str, Any] = field(default_factory=dict)
    incompatibility_reasons: List[Any] = field(default_factory=list)
    # Citation string, raw year, applicability tag bundle
    # (symptoms/regimens/drugs/cancer_types/treatment_phases/
    # treatment_modalities — see applicability_scorer.py), and anything
    # else a specific corpus/retriever wants to carry through without
    # earning its own top-level field yet.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def identity(self) -> str:
        """Dedup identity — deliberately the same precedence order as
        multi_corpus_retriever._candidate_identity (point id > doc+
        section+chunk composite > text prefix). Kept as a literal mirror
        rather than a shared import so this module has no import-time
        dependency on multi_corpus_retriever and vice versa; if this
        ordering changes, change it in both places (each docstring
        cross-references the other)."""
        if self.qdrant_point_id:
            return f"point:{self.qdrant_point_id}"
        if self.document_id:
            return f"doc:{self.document_id}|section:{self.section}|chunk:{self.chunk_index}"
        return f"text:{(self.text or '')[:80]}"

    def to_dict(self) -> Dict[str, Any]:
        """Legacy dict shape consumed today by
        applicability_scorer.score_candidate() and
        evidence_packet_builder.build_packet(). Scoring fields
        (applicability_score/components/incompatibility_reasons/bm25_
        score/rrf_score/cross_encoder_score) are included only when
        already set, matching how those dicts look before vs. after
        applicability_scorer.rank() has run on them.

        Emits the semantic score under BOTH `semantic_score` (the raw
        field score_candidate() reads as its input) and
        `semantic_relevance` (the field score_candidate() itself writes
        back, which build_packet() reads as its input) — the two real
        consumers read different keys depending on pipeline stage, and
        a candidate built fresh (not yet scored) doesn't know which
        stage it's headed for."""
        out: Dict[str, Any] = {
            "qdrant_point_id": self.qdrant_point_id,
            "doc_id": self.document_id,
            "version_id": self.version_id,
            "collection": self.corpus,
            "source_key": self.source_key,
            "source_name": self.source_name,
            "authority_class": self.authority_class,
            "title": self.title,
            "section_title": self.section,
            "chunk_index": self.chunk_index,
            "url": self.url,
            "text": self.text,
            "semantic_score": self.semantic_score if self.semantic_score is not None else 0.0,
            "semantic_relevance": self.semantic_score,
            "applicability_meta": self.metadata.get("applicability_meta") or {},
            "citation": self.metadata.get("citation"),
            "year": self.metadata.get("year", self.publication_date),
        }
        if self.authority_score is not None:
            out["source_authority"] = self.authority_score
        if self.applicability_score is not None:
            out["applicability_score"] = self.applicability_score
        if self.applicability_components:
            out["components"] = self.applicability_components
        if self.incompatibility_reasons:
            out["incompatibility_reasons"] = self.incompatibility_reasons
        if self.bm25_score is not None:
            out["bm25_score"] = self.bm25_score
        if self.rrf_score is not None:
            out["rrf_score"] = self.rrf_score
        if self.cross_encoder_score is not None:
            out["cross_encoder_score"] = self.cross_encoder_score
        return out

    @classmethod
    def from_multi_corpus_dict(cls, d: Dict[str, Any]) -> "EvidenceCandidate":
        """Adapts a dict from multi_corpus_retriever.search() (pre-score)
        or applicability_scorer.rank() (post-score) into the canonical
        shape. This is the reference mapping Sprint C's legacy physician
        retrieval adapter (clinical_retrieval_adapter.py) follows for the
        old clinical retriever's differently-shaped hit."""
        return cls(
            qdrant_point_id=d.get("qdrant_point_id"),
            document_id=d.get("doc_id"),
            version_id=d.get("version_id"),
            corpus=d.get("collection") or "",
            source_key=d.get("source_key"),
            source_name=d.get("source_name"),
            authority_class=d.get("authority_class"),
            authority_score=d.get("source_authority"),
            title=d.get("title") or "",
            section=d.get("section_title"),
            chunk_index=d.get("chunk_index"),
            url=d.get("url"),
            publication_date=str(d["year"]) if d.get("year") else None,
            text=d.get("text") or "",
            semantic_score=(
                d["semantic_relevance"] if d.get("semantic_relevance") is not None
                else d.get("semantic_score")
            ),
            bm25_score=d.get("bm25_score"),
            rrf_score=d.get("rrf_score"),
            cross_encoder_score=d.get("cross_encoder_score") or d.get("score_crossencoder_gate"),
            applicability_score=d.get("applicability_score"),
            applicability_components=d.get("components") or {},
            incompatibility_reasons=d.get("incompatibility_reasons") or [],
            metadata={
                "citation": d.get("citation"),
                "year": d.get("year"),
                "applicability_meta": d.get("applicability_meta") or {},
            },
        )
