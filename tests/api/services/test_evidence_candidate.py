"""
Tests for EvidenceCandidate (2026-08-12 convergence Sprint A item 1): the
canonical shape every retriever is meant to converge on. The critical
property under test is drop-in compatibility -- an EvidenceCandidate
built from the patient path's existing dict shape, or built directly by
a new (future physician) retriever, must round-trip through
applicability_scorer.score_candidate() and evidence_packet_builder.
build_packet() with IDENTICAL results to passing the legacy dict
directly. If that ever breaks, the "adapter, not rewrite" promise this
module's docstring makes is broken too.
"""

from __future__ import annotations

from src.api.services.evidence.evidence_candidate import EvidenceCandidate
from src.api.services.evidence.retrieval_planner import RetrievalPlan
from src.api.services.evidence.applicability_scorer import score_candidate
from src.api.services.evidence.evidence_packet_builder import build_packet
from src.api.services.evidence.patient_context_service import INTENT_MEDICATION


def _plan(**overrides):
    base = dict(
        intent=INTENT_MEDICATION,
        collections=["oncology_medication_knowledge"],
        boost_terms=[],
        patient_values={"drugs": ["pembrolizumab"]},
    )
    base.update(overrides)
    return RetrievalPlan(**base)


def _pre_score_dict(**overrides):
    """Shape multi_corpus_retriever.search() actually produces, before
    applicability_scorer has touched it."""
    base = {
        "qdrant_point_id": "point-1",
        "doc_id": "doc-1",
        "version_id": "version-1",
        "section_title": "Dosing",
        "chunk_index": 2,
        "url": "https://www.fda.gov/pembrolizumab",
        "source_name": "FDA",
        "title": "Pembrolizumab prescribing information",
        "text": "Pembrolizumab is dosed at 200mg every 3 weeks.",
        "citation": "FDA label",
        "year": 2024,
        "collection": "oncology_medication_knowledge",
        "semantic_score": 0.82,
        "applicability_meta": {"drugs": ["pembrolizumab"]},
        "source_key": "fda",
        "authority_class": "A",
    }
    base.update(overrides)
    return base


class TestFromMultiCorpusDictRoundTripsThroughScoring:
    def test_scoring_result_identical_whether_built_via_candidate_or_dict_directly(self):
        raw = _pre_score_dict()
        plan = _plan()

        direct = score_candidate(raw, plan)

        candidate = EvidenceCandidate.from_multi_corpus_dict(raw)
        via_candidate = score_candidate(candidate.to_dict(), plan)

        assert direct["applicability_score"] == via_candidate["applicability_score"]
        assert direct["components"] == via_candidate["components"]
        assert direct["semantic_relevance"] == via_candidate["semantic_relevance"]

    def test_all_provenance_fields_survive_the_round_trip(self):
        raw = _pre_score_dict()
        rt = EvidenceCandidate.from_multi_corpus_dict(raw).to_dict()

        assert rt["qdrant_point_id"] == raw["qdrant_point_id"]
        assert rt["doc_id"] == raw["doc_id"]
        assert rt["version_id"] == raw["version_id"]
        assert rt["section_title"] == raw["section_title"]
        assert rt["chunk_index"] == raw["chunk_index"]
        assert rt["url"] == raw["url"]
        assert rt["source_name"] == raw["source_name"]
        assert rt["source_key"] == raw["source_key"]
        assert rt["authority_class"] == raw["authority_class"]
        assert rt["title"] == raw["title"]
        assert rt["text"] == raw["text"]
        assert rt["citation"] == raw["citation"]
        assert rt["year"] == raw["year"]
        assert rt["collection"] == raw["collection"]


class TestFromMultiCorpusDictAfterScoring:
    """The other real shape this adapter must handle: a dict that has
    already been through applicability_scorer.rank() (evidence_packet_
    builder's actual input)."""

    def test_scored_fields_carry_through(self):
        scored = score_candidate(_pre_score_dict(), _plan())
        candidate = EvidenceCandidate.from_multi_corpus_dict(scored)

        assert candidate.applicability_score == scored["applicability_score"]
        assert candidate.applicability_components == scored["components"]
        assert candidate.semantic_score == scored["semantic_relevance"]

    def test_to_dict_after_scoring_feeds_build_packet_identically(self):
        scored = score_candidate(_pre_score_dict(), _plan())
        candidate = EvidenceCandidate.from_multi_corpus_dict(scored)

        direct_packet = build_packet("What's the dose?", None, [scored])
        via_candidate_packet = build_packet("What's the dose?", None, [candidate.to_dict()])

        assert direct_packet["evidence"] == via_candidate_packet["evidence"]

    def test_incompatibility_reasons_carry_through_when_present(self):
        scored = score_candidate(
            _pre_score_dict(applicability_meta={"treatment_modalities": ["targeted_therapy"]}),
            _plan(patient_values={"treatment_modalities": ["chemotherapy"]}),
        )
        assert scored["incompatibility_reasons"]  # sanity: conflict actually triggered

        candidate = EvidenceCandidate.from_multi_corpus_dict(scored)
        assert candidate.incompatibility_reasons == scored["incompatibility_reasons"]
        assert "incompatibility_reasons" in candidate.to_dict()


class TestToDictOmitsUnsetScoringFields:
    def test_unscored_candidate_omits_scoring_keys(self):
        candidate = EvidenceCandidate(
            qdrant_point_id="point-1", document_id="doc-1", corpus="c", text="x", title="t",
        )
        d = candidate.to_dict()
        assert "applicability_score" not in d
        assert "components" not in d
        assert "incompatibility_reasons" not in d
        assert "bm25_score" not in d
        assert "rrf_score" not in d
        assert "cross_encoder_score" not in d

    def test_scored_candidate_includes_scoring_keys(self):
        candidate = EvidenceCandidate(
            qdrant_point_id="point-1", document_id="doc-1", corpus="c", text="x", title="t",
            applicability_score=0.8, applicability_components={"semantic": 0.8},
            incompatibility_reasons=["cancer_type_mismatch"],
            bm25_score=1.2, rrf_score=0.05, cross_encoder_score=0.6,
        )
        d = candidate.to_dict()
        assert d["applicability_score"] == 0.8
        assert d["components"] == {"semantic": 0.8}
        assert d["incompatibility_reasons"] == ["cancer_type_mismatch"]
        assert d["bm25_score"] == 1.2
        assert d["rrf_score"] == 0.05
        assert d["cross_encoder_score"] == 0.6


class TestIdentityMatchesMultiCorpusRetrieverPrecedence:
    def test_point_id_preferred(self):
        c = EvidenceCandidate(qdrant_point_id="point-1", document_id="doc-1", section="S", chunk_index=0)
        assert c.identity() == "point:point-1"

    def test_falls_back_to_doc_section_chunk_composite(self):
        c = EvidenceCandidate(document_id="doc-1", section="Dosing", chunk_index=2)
        assert c.identity() == "doc:doc-1|section:Dosing|chunk:2"

    def test_falls_back_to_text_prefix_as_last_resort(self):
        c = EvidenceCandidate(text="Some passage with no identifiers at all, quite long indeed.")
        assert c.identity() == "text:Some passage with no identifiers at all, quite long indeed."

    def test_matches_multi_corpus_retriever_candidate_identity_for_the_same_input(self):
        from src.api.services.evidence.multi_corpus_retriever import _candidate_identity

        raw = _pre_score_dict()
        legacy_identity = _candidate_identity(raw)
        candidate_identity = EvidenceCandidate.from_multi_corpus_dict(raw).identity()
        assert legacy_identity == candidate_identity == "point:point-1"

        raw_no_point = _pre_score_dict(qdrant_point_id=None)
        assert (
            _candidate_identity(raw_no_point)
            == EvidenceCandidate.from_multi_corpus_dict(raw_no_point).identity()
        )


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
