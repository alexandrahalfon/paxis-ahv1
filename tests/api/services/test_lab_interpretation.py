"""
Tests for lab_interpretation.py (2026-08-12 convergence Sprint A item 3):
allowed_interpretation_for() and interpretation_policy_summary(), which
replace the hard-coded neutropenia_risk/thrombocytopenia_risk/
renal_function_context labels patient_state_service.py used to derive
from arbitrary thresholds.
"""

from __future__ import annotations

from src.api.services.patient.lab_interpretation import (
    ALL_LEVELS,
    CLINICIAN_INTERPRETED,
    EXACT_VALUE_AND_TREND_ONLY,
    EXACT_VALUE_ONLY,
    VALIDATED_RULE_INTERPRETATION,
    allowed_interpretation_for,
    interpretation_policy_summary,
)


class TestAllowedInterpretationFor:
    def test_exact_value_only_when_no_previous_reading(self):
        latest = {"value": 1.4, "unit": "10^9/L", "collected_at": "2026-08-01"}
        assert allowed_interpretation_for(latest, None) == EXACT_VALUE_ONLY

    def test_exact_value_only_when_previous_has_no_value(self):
        latest = {"value": 1.4}
        previous = {"value": None, "unit": None}
        assert allowed_interpretation_for(latest, previous) == EXACT_VALUE_ONLY

    def test_exact_value_and_trend_when_previous_reading_exists(self):
        latest = {"value": 1.4, "unit": "10^9/L"}
        previous = {"value": 2.4, "unit": "10^9/L"}
        assert allowed_interpretation_for(latest, previous) == EXACT_VALUE_AND_TREND_ONLY

    def test_never_returns_a_reserved_unimplemented_level(self):
        """VALIDATED_RULE_INTERPRETATION and CLINICIAN_INTERPRETED are
        reserved for future work this codebase doesn't do yet -- nothing
        should ever actually be assigned either of them."""
        for latest, previous in [
            ({"value": 1.4}, None),
            ({"value": 1.4}, {"value": 2.4}),
        ]:
            result = allowed_interpretation_for(latest, previous)
            assert result not in (VALIDATED_RULE_INTERPRETATION, CLINICIAN_INTERPRETED)
            assert result in ALL_LEVELS


class TestInterpretationPolicySummary:
    def test_builds_canonical_test_to_level_mapping(self):
        labs = [
            {"canonical_test": "anc", "allowed_interpretation": "exact_value_and_trend_only"},
            {"canonical_test": "creatinine", "allowed_interpretation": "exact_value_only"},
        ]
        assert interpretation_policy_summary(labs) == {
            "anc": "exact_value_and_trend_only",
            "creatinine": "exact_value_only",
        }

    def test_empty_when_no_labs(self):
        assert interpretation_policy_summary([]) == {}
        assert interpretation_policy_summary(None) == {}

    def test_skips_malformed_entries(self):
        labs = [
            {"canonical_test": "anc", "allowed_interpretation": "exact_value_only"},
            {"canonical_test": None, "allowed_interpretation": "exact_value_only"},
            {"allowed_interpretation": "exact_value_only"},
            "not_a_dict",
        ]
        assert interpretation_policy_summary(labs) == {"anc": "exact_value_only"}


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
