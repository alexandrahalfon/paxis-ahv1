"""
Tests for clinical_retrieval_adapter.py (2026-08-12 convergence Sprint C
item 15): adapts comprehensive_retrieval.py's StudyEvidence (real chunk/
scoring shape from _phase3_document_search()/_apply_hybrid_scoring())
into EvidenceCandidate, without importing comprehensive_retrieval.py
itself (duck-typed) and without touching the existing retriever.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.api.services.physician.clinical_retrieval_adapter import (
    adapt_legacy_chunk,
    adapt_legacy_results,
    adapt_legacy_study,
)
from src.api.services.evidence.applicability_scorer import score_candidate
from src.api.services.evidence.evidence_packet_builder import build_packet
from src.api.services.evidence.retrieval_planner import RetrievalPlan


def _study(**overrides):
    base = dict(
        doc_id="doc-1", title="Adagrasib in KRAS G12C NSCLC", citation="Smith et al., NEJM 2024",
        year=2024, category="lung", match_score=85, match_breakdown={"biomarker": "match"},
        axis_mismatches=[], soft_score_normalized=72.0, patient_match_score=90,
        patient_match_breakdown={"biomarker": 1.0}, evidence_type="trial", source="qdrant",
        chunks=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _chunk(**overrides):
    base = dict(
        point_id="point-1", score_dense=0.82, doc_id="doc-1",
        text="Adagrasib demonstrated a PFS benefit in KRAS G12C-mutant NSCLC.",
        section="Results", chunk_type="paragraph", chunk_id=3, section_window_idx=1,
        doc_meta={"title": "Adagrasib in KRAS G12C NSCLC", "source_name": "NEJM",
                  "url": "https://example.org/study", "journal": "NEJM"},
        category="lung", score_lexical=0.4, score=0.68, score_crossencoder_gate=0.71,
    )
    base.update(overrides)
    return base


class TestAdaptLegacyChunk:
    def test_maps_provenance_fields(self):
        candidate = adapt_legacy_chunk(_study(), _chunk())
        assert candidate.qdrant_point_id == "point-1"
        assert candidate.document_id == "doc-1"
        assert candidate.corpus == "clinical_literature"
        assert candidate.section == "Results"
        assert candidate.chunk_index == 3
        assert candidate.url == "https://example.org/study"
        assert candidate.title == "Adagrasib in KRAS G12C NSCLC"
        assert candidate.text == "Adagrasib demonstrated a PFS benefit in KRAS G12C-mutant NSCLC."

    def test_maps_scoring_fields(self):
        candidate = adapt_legacy_chunk(_study(), _chunk())
        assert candidate.semantic_score == 0.82
        assert candidate.bm25_score == 0.4
        assert candidate.cross_encoder_score == 0.71
        # Neither exists in this pipeline -- see module docstring.
        assert candidate.rrf_score is None
        assert candidate.version_id is None

    def test_publication_date_from_study_year(self):
        candidate = adapt_legacy_chunk(_study(year=2019), _chunk())
        assert candidate.publication_date == "2019"

    def test_no_year_leaves_publication_date_none(self):
        candidate = adapt_legacy_chunk(_study(year=None), _chunk())
        assert candidate.publication_date is None

    def test_study_profile_metadata_carried_through(self):
        candidate = adapt_legacy_chunk(_study(), _chunk())
        profile = candidate.metadata["study_profile"]
        assert profile["match_score"] == 85
        assert profile["patient_match_score"] == 90
        assert profile["evidence_type"] == "trial"
        assert profile["source"] == "qdrant"
        assert candidate.metadata["citation"] == "Smith et al., NEJM 2024"
        assert candidate.metadata["year"] == 2024

    def test_missing_point_id_leaves_qdrant_point_id_none(self):
        candidate = adapt_legacy_chunk(_study(), _chunk(point_id=None))
        assert candidate.qdrant_point_id is None

    def test_works_with_a_dict_shaped_study_too(self):
        """to_dict()'d StudyEvidence should work identically to the real
        dataclass instance -- _get() dispatches on isinstance(dict)."""
        study_dict = {"doc_id": "doc-2", "title": "T", "citation": "C", "year": 2020,
                      "category": None, "match_score": None, "match_breakdown": None,
                      "axis_mismatches": [], "soft_score_normalized": None,
                      "patient_match_score": None, "patient_match_breakdown": None,
                      "evidence_type": "guideline", "source": "postgres"}
        candidate = adapt_legacy_chunk(study_dict, _chunk(doc_id=None))
        assert candidate.document_id == "doc-2"
        assert candidate.metadata["study_profile"]["evidence_type"] == "guideline"


class TestAdaptLegacyStudy:
    def test_one_candidate_per_chunk(self):
        study = _study(chunks=[
            _chunk(point_id="point-1", section="Results"),
            _chunk(point_id="point-2", section="Methods", text="Patients were randomized 1:1."),
        ])
        candidates = adapt_legacy_study(study)
        assert len(candidates) == 2
        sections = {c.section for c in candidates}
        assert sections == {"Results", "Methods"}

    def test_empty_text_chunk_is_skipped(self):
        study = _study(chunks=[_chunk(text=""), _chunk(point_id="point-2", text="   ")])
        assert adapt_legacy_study(study) == []

    def test_no_chunks_yields_empty_list(self):
        assert adapt_legacy_study(_study(chunks=[])) == []


class TestAdaptLegacyResults:
    def test_flattens_multiple_studies(self):
        studies = [
            _study(doc_id="doc-1", chunks=[_chunk(point_id="p1", doc_id="doc-1")]),
            _study(doc_id="doc-2", chunks=[_chunk(point_id="p2", doc_id="doc-2")]),
        ]
        candidates = adapt_legacy_results(studies)
        assert len(candidates) == 2
        assert {c.document_id for c in candidates} == {"doc-1", "doc-2"}

    def test_empty_studies_list_yields_empty_candidates(self):
        assert adapt_legacy_results([]) == []


class TestRoundTripsThroughSharedScoringAndPacketPipeline:
    """The load-bearing property, same as EvidenceCandidate's own tests:
    an adapted legacy chunk must be a genuine drop-in for the existing
    applicability_scorer/evidence_packet_builder pipeline the patient
    path already uses."""

    def test_adapted_candidate_scores_and_packets_without_error(self):
        candidate = adapt_legacy_chunk(_study(), _chunk())
        plan = RetrievalPlan(intent="treatment_explainer", collections=[], patient_values={})

        scored = score_candidate(candidate.to_dict(), plan)
        assert scored["applicability_score"] >= 0

        packet = build_packet("What are the options for KRAS G12C NSCLC?", None, [scored])
        assert len(packet["evidence"]) == 1
        entry = packet["evidence"][0]
        assert entry["qdrant_point_id"] == "point-1"
        assert entry["text"] == "Adagrasib demonstrated a PFS benefit in KRAS G12C-mutant NSCLC."


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
