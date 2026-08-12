"""
Trivial test that imports the score weight constants and asserts their
expected values as specified in bugfix.md Score Weights section.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6, 2.7**
"""

from src.api.services.patient_eligibility_boost_service import (
    CORE_MISMATCH_PENALTY,
    CORE_AXIS_WEIGHTS,
    SECONDARY_AXIS_WEIGHTS,
)


def test_core_mismatch_penalty_value():
    assert CORE_MISMATCH_PENALTY == 10


def test_core_axis_weights_keys():
    expected_axes = {"cancer_type", "histology", "stage", "prior_therapies", "biomarkers"}
    assert set(CORE_AXIS_WEIGHTS.keys()) == expected_axes


def test_core_axis_weights_values():
    for axis in CORE_AXIS_WEIGHTS:
        w = CORE_AXIS_WEIGHTS[axis]
        assert w["MATCH"] == 15, f"{axis} MATCH should be 15"
        assert w["COMPATIBLE"] == 10, f"{axis} COMPATIBLE should be 10"
        assert w["NOT_AVAILABLE"] == 0, f"{axis} NOT_AVAILABLE should be 0"
        assert w["MISMATCH"] == 0, f"{axis} MISMATCH should be 0"


def test_secondary_axis_weights_keys():
    expected_axes = {
        "performance_status", "age_range", "modality", "metastatic_sites",
        "comorbidity_compatibility", "gender", "study_phase",
        "landmark_trial_status", "recency",
    }
    assert set(SECONDARY_AXIS_WEIGHTS.keys()) == expected_axes


def test_secondary_axis_weights_match_values():
    expected_match = {
        "performance_status": 8,
        "age_range": 6,
        "modality": 5,
        "metastatic_sites": 5,
        "comorbidity_compatibility": 4,
        "gender": 4,
        "study_phase": 3,
        "landmark_trial_status": 3,
        "recency": 2,
    }
    for axis, expected in expected_match.items():
        assert SECONDARY_AXIS_WEIGHTS[axis]["MATCH"] == expected, (
            f"{axis} MATCH should be {expected}"
        )


def test_secondary_axis_weights_mismatch_values():
    expected_mismatch = {
        "performance_status": -4,
        "age_range": -3,
        "modality": 0,
        "metastatic_sites": -2,
        "comorbidity_compatibility": -4,
        "gender": -2,
        "study_phase": 0,
        "landmark_trial_status": 0,
        "recency": 0,
    }
    for axis, expected in expected_mismatch.items():
        assert SECONDARY_AXIS_WEIGHTS[axis]["MISMATCH"] == expected, (
            f"{axis} MISMATCH should be {expected}"
        )


def test_secondary_axis_weights_neutral_values():
    """COMPATIBLE and NOT_AVAILABLE are always 0 for secondary axes."""
    for axis in SECONDARY_AXIS_WEIGHTS:
        w = SECONDARY_AXIS_WEIGHTS[axis]
        assert w["COMPATIBLE"] == 0, f"{axis} COMPATIBLE should be 0"
        assert w["NOT_AVAILABLE"] == 0, f"{axis} NOT_AVAILABLE should be 0"
