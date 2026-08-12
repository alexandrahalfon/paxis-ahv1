"""
Tests for evidence_packet_builder.py's provenance + content-hash dedup
(2026-08-12 beta audit item 7): each packet entry now carries the full
provenance chain (qdrant_point_id/document_id/version_id/source_key/
source_name/section/chunk_index/url/score_components/
incompatibility_reasons), and a duplicate PASSAGE (same text surfaced
twice, e.g. from two collections) is deduped at packet-construction time
-- a different, later dedup pass than multi_corpus_retriever's per-chunk
identity dedup, which deliberately does NOT collapse two distinct
sections of the same document.
"""

from __future__ import annotations

from src.api.services.evidence.evidence_packet_builder import (
    build_packet, summarize_context, to_prompt_block, to_sources,
)


def _candidate(**overrides):
    base = {
        "doc_id": "doc-1",
        "qdrant_point_id": "point-1",
        "version_id": "version-1",
        "section_title": "Taste changes",
        "chunk_index": 0,
        "url": "https://www.cancer.gov/nutrition",
        "source_key": "nci",
        "source_name": "National Cancer Institute",
        "title": "Nutrition During Cancer Treatment",
        "collection": "oncology_patient_education",
        "authority_class": "A",
        "semantic_relevance": 0.8,
        "applicability_score": 0.9,
        "components": {"semantic": 0.8, "symptom": 1.0},
        "incompatibility_reasons": [],
        "text": "Foods may taste metallic during chemotherapy.",
        "citation": "NCI",
        "year": 2024,
    }
    base.update(overrides)
    return base


class TestProvenanceFieldsPresent:
    def test_packet_entry_carries_full_provenance(self):
        packet = build_packet("What can I eat?", None, [_candidate()])
        entry = packet["evidence"][0]

        assert entry["qdrant_point_id"] == "point-1"
        assert entry["document_id"] == "doc-1"
        assert entry["version_id"] == "version-1"
        assert entry["source_key"] == "nci"
        assert entry["source_name"] == "National Cancer Institute"
        assert entry["section"] == "Taste changes"
        assert entry["chunk_index"] == 0
        assert entry["url"] == "https://www.cancer.gov/nutrition"
        assert entry["semantic_score"] == 0.8
        assert entry["score_components"] == {"semantic": 0.8, "symptom": 1.0}
        assert entry["incompatibility_reasons"] == []
        # Original fields are preserved, not replaced.
        assert entry["title"] == "Nutrition During Cancer Treatment"
        assert entry["role"] == "oncology_patient_education"
        assert entry["authority"] == "A"
        assert entry["citation"] == "NCI"
        assert entry["year"] == 2024
        assert entry["text"] == "Foods may taste metallic during chemotherapy."

    def test_incompatibility_reasons_carried_through_when_present(self):
        packet = build_packet("Question", None, [
            _candidate(incompatibility_reasons=["modality_mismatch: patient=['chemotherapy'] chunk=['targeted_therapy']"])
        ])
        assert packet["evidence"][0]["incompatibility_reasons"] == [
            "modality_mismatch: patient=['chemotherapy'] chunk=['targeted_therapy']"
        ]


class TestDistinctSectionsOfSameDocumentBothSurvive:
    """multi_corpus_retriever already preserves these as two candidates
    (see test_multi_corpus_retriever.py); this confirms build_packet()
    doesn't re-introduce the old doc_id-collapsing bug at this layer."""

    def test_two_different_passages_from_the_same_doc_id_both_appear(self):
        taste = _candidate(
            qdrant_point_id="point-taste", section_title="Taste changes",
            text="Foods may taste metallic.",
        )
        appetite = _candidate(
            qdrant_point_id="point-appetite", section_title="Appetite loss",
            text="Eat small frequent meals instead of three large ones.",
        )
        packet = build_packet("What can I eat?", None, [taste, appetite])
        assert len(packet["evidence"]) == 2
        sections = {e["section"] for e in packet["evidence"]}
        assert sections == {"Taste changes", "Appetite loss"}


class TestContentDedup:
    def test_identical_text_from_two_different_points_collapses_to_one(self):
        first = _candidate(qdrant_point_id="point-a", text="Some passage repeated verbatim.")
        second = _candidate(qdrant_point_id="point-b", text="Some passage repeated verbatim.")
        packet = build_packet("Question", None, [first, second])
        assert len(packet["evidence"]) == 1

    def test_highest_ranked_duplicate_is_kept(self):
        """ranked_evidence arrives already sorted by applicability_score
        descending (applicability_scorer.rank()'s contract); dedup must
        keep the first occurrence, i.e. the higher-scored one."""
        higher = _candidate(qdrant_point_id="point-a", source_key="nci", applicability_score=0.95, text="Duplicate text.")
        lower = _candidate(qdrant_point_id="point-b", source_key="acs", applicability_score=0.40, text="Duplicate text.")
        packet = build_packet("Question", None, [higher, lower])
        assert len(packet["evidence"]) == 1
        assert packet["evidence"][0]["source_key"] == "nci"

    def test_dedup_is_case_and_whitespace_insensitive(self):
        a = _candidate(qdrant_point_id="point-a", text="Foods  may   taste metallic.")
        b = _candidate(qdrant_point_id="point-b", text="foods may taste metallic.")
        packet = build_packet("Question", None, [a, b])
        assert len(packet["evidence"]) == 1

    def test_different_text_is_never_deduped(self):
        a = _candidate(qdrant_point_id="point-a", text="Foods may taste metallic.")
        b = _candidate(qdrant_point_id="point-b", text="Eat small frequent meals.")
        packet = build_packet("Question", None, [a, b])
        assert len(packet["evidence"]) == 2


class TestSummarizeContextCareTeamInstructions:
    """2026-08-12 beta audit item 5: care_team_instructions were stored
    and read into patient state, but summarize_context() -- the function
    that decides what the evidence packet's patient_context contains --
    dropped them entirely, so nothing downstream could even see that they
    existed."""

    def test_active_instructions_are_surfaced(self):
        context = {
            "state": {
                "care_team_instructions": [
                    {"text": "Avoid grapefruit while on this regimen.", "type": "dietary"},
                    {"text": "Call us if your temperature is above 100.4F.", "type": "monitoring"},
                ],
            },
            "retrieval_features": {},
        }
        summary = summarize_context(context)
        assert summary["care_team_instructions"] == [
            {"text": "Avoid grapefruit while on this regimen.", "type": "dietary"},
            {"text": "Call us if your temperature is above 100.4F.", "type": "monitoring"},
        ]

    def test_absent_when_no_instructions(self):
        context = {"state": {"care_team_instructions": []}, "retrieval_features": {}}
        assert "care_team_instructions" not in summarize_context(context)

    def test_absent_when_state_missing_entirely(self):
        assert "care_team_instructions" not in summarize_context({})
        assert "care_team_instructions" not in summarize_context(None)

    def test_entries_without_text_are_dropped(self):
        context = {
            "state": {"care_team_instructions": [{"text": "", "type": "other"}, {"type": "other"}]},
        }
        assert "care_team_instructions" not in summarize_context(context)

    def test_surfaced_in_build_packet_patient_context(self):
        context = {
            "state": {
                "care_team_instructions": [{"text": "No NSAIDs on this regimen.", "type": "medication"}],
            },
        }
        packet = build_packet("Can I take ibuprofen?", context, [_candidate()])
        assert packet["patient_context"]["care_team_instructions"] == [
            {"text": "No NSAIDs on this regimen.", "type": "medication"}
        ]


class TestSharedPacketContractFields:
    """2026-08-12 convergence Sprint A item 2: build_packet()'s output
    expands toward the full shared EvidencePacket shape (audience,
    query_analysis, patient_snapshot_id, selected_patient_context,
    retrieval_plan, pto_frame, safety_policy, interpretation_policies),
    all additive and defaulted so existing callers are unaffected."""

    def test_defaults_when_nothing_new_is_passed(self):
        packet = build_packet("Question", None, [_candidate()])
        assert packet["audience"] == "patient"
        assert packet["query_analysis"] == {}
        assert packet["patient_snapshot_id"] is None
        assert packet["retrieval_plan"] is None
        assert packet["pto_frame"] is None
        assert packet["safety_policy"] == {}
        assert packet["interpretation_policies"] == {}
        # selected_patient_context defaults to the same value as
        # patient_context when nothing overrides it.
        assert packet["selected_patient_context"] == packet["patient_context"]

    def test_explicit_values_are_carried_through(self):
        packet = build_packet(
            "Question", None, [_candidate()],
            audience="physician",
            query_analysis={"intent": "therapy_selection"},
            patient_snapshot_id="profile-123",
            pto_frame={"axis": "biomarker"},
            safety_policy={"tier": "strict"},
            interpretation_policies={"anc": "exact_value_and_trend_only"},
        )
        assert packet["audience"] == "physician"
        assert packet["query_analysis"] == {"intent": "therapy_selection"}
        assert packet["patient_snapshot_id"] == "profile-123"
        assert packet["pto_frame"] == {"axis": "biomarker"}
        assert packet["safety_policy"] == {"tier": "strict"}
        assert packet["interpretation_policies"] == {"anc": "exact_value_and_trend_only"}

    def test_selected_patient_context_override_is_independent_of_patient_context(self):
        packet = build_packet(
            "Question", {"state": {"active_diagnosis": {"cancer_site": "lung"}}}, [_candidate()],
            selected_patient_context={"only": "what physician context selector picked"},
        )
        assert packet["selected_patient_context"] == {"only": "what physician context selector picked"}
        assert packet["patient_context"] == {"cancer_type": "lung"}
        assert packet["selected_patient_context"] != packet["patient_context"]

    def test_retrieval_plan_dataclass_is_serialized(self):
        from src.api.services.evidence.retrieval_planner import RetrievalPlan
        plan = RetrievalPlan(intent="medication_explainer", collections=["c1"], boost_terms=["x"])
        packet = build_packet("Question", None, [_candidate()], retrieval_plan=plan)
        assert packet["retrieval_plan"] == {
            "intent": "medication_explainer", "collections": ["c1"], "boost_terms": ["x"],
            "hard_constraints": {}, "patient_values": {},
        }

    def test_retrieval_plan_dict_passes_through_unchanged(self):
        packet = build_packet("Question", None, [_candidate()], retrieval_plan={"collections": ["c1"]})
        assert packet["retrieval_plan"] == {"collections": ["c1"]}

    def test_existing_fields_unaffected(self):
        """The pre-existing question/patient_context/evidence/safety
        fields must be byte-identical to before this expansion."""
        packet = build_packet("Question", None, [_candidate()], safety_category="distress")
        assert packet["question"] == "Question"
        assert packet["safety"] == {"category": "distress", "red_flags": []}
        assert len(packet["evidence"]) == 1


class TestBackwardCompatibleHelpers:
    """to_prompt_block()/to_sources() are read by patient_chat_service.py
    and the legacy patient_query.py route -- must keep working unchanged."""

    def test_to_prompt_block_still_works(self):
        packet = build_packet("Question", None, [_candidate()])
        block = to_prompt_block(packet)
        assert "[1]" in block
        assert "Nutrition During Cancer Treatment" in block
        assert "Foods may taste metallic" in block

    def test_to_sources_still_works(self):
        packet = build_packet("Question", None, [_candidate()])
        sources = to_sources(packet)
        assert len(sources) == 1
        assert sources[0]["title"] == "Nutrition During Cancer Treatment"
        assert sources[0]["citation"] == "NCI"
        assert sources[0]["year"] == 2024
        assert sources[0]["source_type"] == "oncology_patient_education"
        assert sources[0]["authority"] == "A"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
