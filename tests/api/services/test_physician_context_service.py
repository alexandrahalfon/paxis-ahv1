"""
Tests for physician_context_service.py (2026-08-12 convergence Sprint C
item 13): select_physician_context() filters a full PatientState down to
just what a given intent needs -- "don't dump the complete longitudinal
record into every physician prompt."
"""

from __future__ import annotations

from src.api.services.physician.physician_context_service import (
    PHYSICIAN_CONTEXT_POLICY,
    THERAPY_SELECTION,
    TOXICITY_MANAGEMENT,
    TRIAL_ELIGIBILITY,
    select_physician_context,
)


def _full_state():
    return {
        "demographics": {"age": 65, "sex": "male"},
        "active_diagnosis": {"cancer_site": "lung", "histology": "adenocarcinoma", "stage": "III"},
        "active_diagnoses": [{"cancer_site": "lung", "histology": "adenocarcinoma", "stage": "III"}],
        "tumor_profile": {"grade": "2"},
        "biomarkers": [{"biomarker_name": "EGFR", "value": "positive"}],
        "active_treatment": [{"regimen": "carboplatin/pemetrexed", "modality": "chemotherapy"}],
        "active_medications": [{"name": "pembrolizumab"}],
        "active_symptoms": [{"name": "fatigue", "severity": "moderate"}],
        "nutrition": {"weight_change_30d_pct": -3},
        "recent_labs": {"anc": {"value": 1.4, "unit": "10^9/L"}},
        "labs": [{"canonical_test": "anc", "latest": {"value": 1.4}, "allowed_interpretation": "exact_value_only"}],
        "comorbidities": ["CKD"],
        "allergies": ["penicillin"],
        "intolerances": [],
        "care_team_instructions": [{"text": "No NSAIDs.", "type": "medication"}],
        "has_care_team": True,
    }


class TestTherapySelectionPolicy:
    def test_includes_the_policy_fields(self):
        selected = select_physician_context(_full_state(), THERAPY_SELECTION)
        assert "active_diagnosis" in selected or "active_diagnoses" in selected
        assert "tumor_profile" in selected
        assert "biomarkers" in selected
        assert "active_treatment" in selected
        assert "comorbidities" in selected
        assert "allergies" in selected
        assert "care_team_instructions" in selected

    def test_excludes_fields_not_in_the_policy(self):
        selected = select_physician_context(_full_state(), THERAPY_SELECTION)
        # nutrition and active_symptoms aren't in THERAPY_SELECTION's
        # field list -- must not leak through.
        assert "nutrition" not in selected
        assert "active_symptoms" not in selected


class TestToxicityManagementPolicy:
    def test_includes_symptoms_and_medications_not_diagnoses(self):
        selected = select_physician_context(_full_state(), TOXICITY_MANAGEMENT)
        assert "active_symptoms" in selected
        assert "active_medications" in selected
        assert "active_treatment" in selected  # current_treatment maps here
        assert "comorbidities" in selected
        # diagnoses/tumor_profile/allergies/care_team_instructions aren't
        # in TOXICITY_MANAGEMENT's field list.
        assert "active_diagnosis" not in selected
        assert "tumor_profile" not in selected
        assert "allergies" not in selected


class TestTrialEligibilityPolicy:
    def test_stage_and_histology_pull_in_diagnoses_not_separate_fields(self):
        """stage/histology live nested inside diagnosis entries -- the
        policy names them separately (matching the plan's spec) but they
        resolve to the same state key as 'diagnoses'."""
        selected = select_physician_context(_full_state(), TRIAL_ELIGIBILITY)
        assert "active_diagnosis" in selected or "active_diagnoses" in selected
        assert "biomarkers" in selected
        assert "active_treatment" in selected  # prior_treatments maps here today


class TestEmptyValuesAreExcluded:
    def test_empty_list_field_is_not_included(self):
        state = _full_state()
        state["biomarkers"] = []
        selected = select_physician_context(state, THERAPY_SELECTION)
        assert "biomarkers" not in selected

    def test_none_field_is_not_included(self):
        state = _full_state()
        state["active_diagnosis"] = None
        selected = select_physician_context(state, THERAPY_SELECTION)
        assert "active_diagnosis" not in selected
        # active_diagnoses (the plural list) still carries the same info.
        assert "active_diagnoses" in selected

    def test_missing_state_field_is_silently_absent_not_an_error(self):
        selected = select_physician_context({}, THERAPY_SELECTION)
        assert selected == {}


class TestUnknownIntentUsesDefaultNotFullDump:
    def test_unrecognized_intent_gets_the_small_default_set(self):
        selected = select_physician_context(_full_state(), "some_intent_not_in_the_policy")
        # Default policy is diagnoses/treatment_history/biomarkers/
        # performance_status -- performance_status maps to nothing today,
        # so only three keys are expected.
        assert set(selected.keys()) <= {"active_diagnosis", "active_diagnoses", "active_treatment", "biomarkers"}
        # Must NOT include fields outside the default (e.g. nutrition,
        # allergies) -- confirms this isn't silently falling back to
        # "select everything".
        assert "nutrition" not in selected
        assert "allergies" not in selected
        assert "active_symptoms" not in selected

    def test_general_intent_also_gets_the_default(self):
        selected = select_physician_context(_full_state(), "general")
        assert "active_treatment" in selected


class TestPerformanceStatusAndTreatmentCyclesGapsAreHonest:
    """These are named in PHYSICIAN_CONTEXT_POLICY (matching the
    convergence plan's spec) but have no backing PatientState field
    today -- confirms the selector doesn't fabricate a value for them."""

    def test_performance_status_selects_nothing_extra(self):
        selected_therapy = select_physician_context(_full_state(), THERAPY_SELECTION)
        # Every key present must correspond to a REAL field in the
        # policy's other entries (diagnoses/tumor_profiles/biomarkers/
        # treatment_history/labs/comorbidities/allergies/
        # care_team_instructions) -- performance_status contributes none
        # of its own.
        assert "performance_status" not in selected_therapy


class TestPolicyCoversAllFourNamedIntents(object):
    def test_all_four_intents_are_defined(self):
        assert set(PHYSICIAN_CONTEXT_POLICY.keys()) == {
            "therapy_selection", "treatment_sequencing", "toxicity_management", "trial_eligibility",
        }


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
