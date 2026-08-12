"""
Tests for patient_state_service.py's removal of hard-coded lab-derived
risk labels (2026-08-12 convergence Sprint A item 3):
neutropenia_risk/thrombocytopenia_risk/renal_function_context used to be
computed in _derive_retrieval_features() from arbitrary thresholds and
handed to generation/retrieval as named clinical conclusions the system
never validated. They must never appear in retrieval_features again,
regardless of what recent_labs/labs contain.
"""

from __future__ import annotations

from src.api.services.patient.patient_state_service import PatientStateService


def _state_with_labs(recent_labs):
    return {
        "active_diagnoses": [], "active_diagnosis": None, "active_treatment": [],
        "biomarkers": [], "comorbidities": [], "care_team_instructions": [],
        "nutrition": {}, "recent_labs": recent_labs,
    }


class TestHardcodedRiskLabelsNeverProduced:
    def test_low_anc_no_longer_produces_neutropenia_risk(self):
        state = _state_with_labs({"anc": {"value": 0.7, "unit": "10^9/L"}})
        features = PatientStateService()._derive_retrieval_features(state)
        assert "neutropenia_risk" not in features

    def test_low_platelets_no_longer_produces_thrombocytopenia_risk(self):
        state = _state_with_labs({"platelets": {"value": 40, "unit": "10^9/L"}})
        features = PatientStateService()._derive_retrieval_features(state)
        assert "thrombocytopenia_risk" not in features

    def test_high_creatinine_no_longer_produces_renal_function_context(self):
        state = _state_with_labs({"creatinine": {"value": 2.1, "unit": "mg/dL"}})
        features = PatientStateService()._derive_retrieval_features(state)
        assert "renal_function_context" not in features

    def test_all_three_absent_together_with_every_threshold_triggered(self):
        state = _state_with_labs({
            "anc": {"value": 0.5}, "platelets": {"value": 30}, "creatinine": {"value": 3.0},
        })
        features = PatientStateService()._derive_retrieval_features(state)
        assert "neutropenia_risk" not in features
        assert "thrombocytopenia_risk" not in features
        assert "renal_function_context" not in features

    def test_other_retrieval_features_are_unaffected(self):
        """Removing the lab-derived fields must not be a wholesale
        rewrite -- everything else _derive_retrieval_features computes
        still works the same."""
        state = _state_with_labs({})
        state["nutrition"] = {"assessed_nutrition_risk": "high"}
        state["care_team_instructions"] = [{"text": "x", "type": "other"}]
        features = PatientStateService()._derive_retrieval_features(state)
        assert features["nutrition_risk"] == "high"
        assert features["has_active_care_instructions"] is True


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
