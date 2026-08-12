"""
Tests for retrieval_debug_trace.py's provenance fields (2026-08-12 beta
audit items 6/7): TraceBuilder now records the same qdrant_point_id/
document_id/version_id/section/chunk_index/url/source_name/
score_components/incompatibility_reasons that multi_corpus_retriever.py
and evidence_packet_builder.py attach to a candidate, so a trace alone is
enough to audit which exact Qdrant point a candidate/citation came from.
"""

from __future__ import annotations

from src.api.services.evidence.retrieval_debug_trace import TraceBuilder


def _candidate(**overrides):
    base = {
        "qdrant_point_id": "point-1",
        "doc_id": "doc-1",
        "version_id": "version-1",
        "collection": "oncology_patient_education",
        "title": "Nutrition During Cancer Treatment",
        "section_title": "Taste changes",
        "chunk_index": 0,
        "source_key": "nci",
        "source_name": "National Cancer Institute",
        "url": "https://www.cancer.gov/nutrition",
        "authority_class": "A",
        "semantic_relevance": 0.8,
        "clinical_applicability": 0.7,
        "source_authority": 1.0,
        "applicability_score": 0.85,
        "components": {"semantic": 0.8},
        "incompatibility_reasons": [],
        "text": "Foods may taste metallic during chemotherapy.",
    }
    base.update(overrides)
    return base


class TestCandidateProvenanceRecorded:
    def test_set_candidates_records_full_provenance(self):
        trace = TraceBuilder(patient_profile_id="p1", question="What can I eat?")
        trace.set_candidates([_candidate()])
        recorded = trace.candidates[0]

        assert recorded["qdrant_point_id"] == "point-1"
        assert recorded["doc_id"] == "doc-1"
        assert recorded["version_id"] == "version-1"
        assert recorded["chunk_index"] == 0
        assert recorded["source_name"] == "National Cancer Institute"
        assert recorded["url"] == "https://www.cancer.gov/nutrition"
        assert recorded["incompatibility_reasons"] == []

    def test_set_ranked_records_full_provenance(self):
        trace = TraceBuilder(patient_profile_id="p1", question="What can I eat?")
        trace.set_ranked([_candidate(incompatibility_reasons=["modality_mismatch: x"])])
        recorded = trace.ranked[0]
        assert recorded["incompatibility_reasons"] == ["modality_mismatch: x"]


class TestPacketProvenanceRecorded:
    def test_set_packet_records_provenance_per_evidence_entry(self):
        trace = TraceBuilder(patient_profile_id="p1", question="What can I eat?")
        packet = {
            "evidence": [{
                "source": "nci",
                "qdrant_point_id": "point-1",
                "document_id": "doc-1",
                "version_id": "version-1",
                "section": "Taste changes",
                "url": "https://www.cancer.gov/nutrition",
                "title": "Nutrition During Cancer Treatment",
                "role": "oncology_patient_education",
                "authority": "A",
                "semantic_score": 0.8,
                "applicability_score": 0.85,
                "score_components": {"semantic": 0.8},
                "incompatibility_reasons": [],
                "citation": "NCI",
                "text": "Foods may taste metallic.",
            }],
            "patient_context": {},
            "safety": {"category": "general", "red_flags": []},
        }
        trace.set_packet(packet)
        entry = trace.packet_summary["evidence"][0]

        assert entry["qdrant_point_id"] == "point-1"
        assert entry["document_id"] == "doc-1"
        assert entry["version_id"] == "version-1"
        assert entry["section"] == "Taste changes"
        assert entry["url"] == "https://www.cancer.gov/nutrition"
        assert entry["score_components"] == {"semantic": 0.8}
        assert entry["incompatibility_reasons"] == []

    def test_to_dict_includes_provenance(self):
        trace = TraceBuilder(patient_profile_id="p1", question="What can I eat?")
        trace.set_candidates([_candidate()])
        d = trace.to_dict()
        assert d["candidates"][0]["qdrant_point_id"] == "point-1"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
