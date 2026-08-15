"""
Tests for QueryAnalysis (2026-08-12 convergence Sprint C item 12): the
adapter from the existing, cheap (regex-based, no LLM) legacy
classifiers -- query_structuring_service.structure_query_fast() and
clinical_entity_extractor.ClinicalEntityExtractor().extract() for
audience="physician", patient_context_service.classify_intent() (via
the caller) for audience="patient" -- into one typed, audience-neutral
shape. Runs the real extractors (both synchronous, no I/O) rather than
mocking them, since the whole point of this adapter is faithfully
reshaping their REAL output.
"""

from __future__ import annotations

from src.api.services.evidence.query_analysis import (
    QueryAnalysis,
    _dedup,
    from_patient_message,
    from_physician_query,
)


class TestDedupHelper:
    def test_flattens_and_dedups_case_insensitively(self):
        assert _dedup(["EGFR", "egfr", "KRAS"]) == ["EGFR", "KRAS"]

    def test_drops_none_and_empty_values(self):
        assert _dedup([None, "", "lung", None]) == ["lung"]

    def test_single_non_list_value_is_wrapped(self):
        assert _dedup("lung") == ["lung"]

    def test_none_alone_yields_empty_list(self):
        assert _dedup([None]) == []

    def test_strips_whitespace(self):
        assert _dedup(["  lung  "]) == ["lung"]

    def test_preserves_first_occurrence_order(self):
        assert _dedup(["b", "a", "b", "c"]) == ["b", "a", "c"]


class TestFromPhysicianQuery:
    QUERY = (
        "65 year old male with stage III lung adenocarcinoma, EGFR positive, "
        "s/p chemotherapy and radiation, now with progression"
    )

    def test_audience_is_physician(self):
        result = from_physician_query(self.QUERY)
        assert result.audience == "physician"

    def test_cancer_type_and_histology_extracted(self):
        result = from_physician_query(self.QUERY)
        assert "lung" in result.cancer_types
        assert "adenocarcinoma" in result.histologies

    def test_stage_extracted_from_both_sources(self):
        result = from_physician_query(self.QUERY)
        # Both structure_query_fast (bare "III") and ClinicalEntityExtractor
        # (a longer "stage III " phrase) contribute -- they're different
        # literal strings, not deduped against each other semantically.
        assert "III" in result.stages

    def test_biomarker_extracted_and_deduped_across_both_sources(self):
        result = from_physician_query(self.QUERY)
        assert result.biomarkers.count("EGFR mutant") == 1

    def test_drugs_and_regimens_both_read_the_same_treatments_field(self):
        result = from_physician_query(self.QUERY)
        assert result.drugs == result.regimens
        assert "chemotherapy" in result.drugs
        assert "radiation" in result.drugs

    def test_labs_and_symptoms_are_always_empty(self):
        """Neither adapted source extracts these from free text -- see
        module docstring for why this is left honest rather than faked."""
        result = from_physician_query(self.QUERY)
        assert result.labs == []
        assert result.symptoms == []

    def test_patient_specific_true_when_structure_detects_patient_context(self):
        result = from_physician_query(self.QUERY)
        assert result.patient_specific is True

    def test_intent_defaults_to_question_focus(self):
        result = from_physician_query(self.QUERY)
        assert result.intent  # non-empty -- exact value depends on question_focus
        assert result.intent != "general"  # this query has a detected focus

    def test_explicit_intent_overrides_question_focus_default(self):
        result = from_physician_query(self.QUERY, intent="trial_eligibility")
        assert result.intent == "trial_eligibility"

    def test_empty_query_degrades_to_sensible_defaults(self):
        result = from_physician_query("")
        assert result.audience == "physician"
        assert result.intent == "general"
        assert result.cancer_types == []
        assert result.patient_specific is False

    def test_returns_a_real_query_analysis_instance(self):
        result = from_physician_query(self.QUERY)
        assert isinstance(result, QueryAnalysis)
        d = result.to_dict()
        assert d["audience"] == "physician"
        assert isinstance(d["cancer_types"], list)


class TestFromPatientMessage:
    def test_audience_is_patient(self):
        result = from_patient_message("What are the side effects?", intent="medication_explainer")
        assert result.audience == "patient"

    def test_intent_passed_through_unchanged(self):
        result = from_patient_message("What are the side effects?", intent="medication_explainer")
        assert result.intent == "medication_explainer"

    def test_patient_specific_always_true(self):
        result = from_patient_message("Thanks!", intent="general")
        assert result.patient_specific is True

    def test_clinical_axis_fields_are_all_empty(self):
        """Structured patient context comes from patient_state_service
        via a separate EvidencePacket field, not from this adapter --
        see module docstring."""
        result = from_patient_message(
            "I have stage III lung cancer and take pembrolizumab", intent="medication_explainer",
        )
        assert result.cancer_types == []
        assert result.histologies == []
        assert result.stages == []
        assert result.biomarkers == []
        assert result.drugs == []
        assert result.regimens == []


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
