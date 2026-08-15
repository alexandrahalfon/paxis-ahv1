"""
Unit tests for _biomarker_hard_exclusion() and check_biomarker_hard_exclusion().

Validates Requirements 7.1 and 7.4 from the RAG Pipeline Consolidation spec.
"""

import os
import pytest
from unittest.mock import patch

from src.api.services.biomarker_canonicalizer import CanonicalBiomarker
from src.api.services.patient_eligibility_boost_service import (
    _biomarker_hard_exclusion,
    check_biomarker_hard_exclusion,
)


def _make_biomarker(canonical_id: str, polarity: str = None) -> CanonicalBiomarker:
    return CanonicalBiomarker(
        canonical_id=canonical_id,
        polarity=polarity,
        metric=None,
        metric_value=None,
        raw_text="",
        source="test",
    )


# ---------------------------------------------------------------------------
# _biomarker_hard_exclusion (core logic)
# ---------------------------------------------------------------------------

class TestBiomarkerHardExclusion:
    """Validates: Requirement 7.1"""

    def test_contradictory_mutant_vs_wildtype_excludes(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "wild-type"}
        assert _biomarker_hard_exclusion(patient, study) is True

    def test_contradictory_wildtype_vs_mutant_excludes(self):
        patient = [_make_biomarker("EGFR", "wild-type")]
        study = {"EGFR": "mutant"}
        assert _biomarker_hard_exclusion(patient, study) is True

    def test_contradictory_positive_vs_negative_excludes(self):
        patient = [_make_biomarker("HER2", "positive")]
        study = {"HER2": "negative"}
        assert _biomarker_hard_exclusion(patient, study) is True

    def test_contradictory_negative_vs_positive_excludes(self):
        patient = [_make_biomarker("HER2", "negative")]
        study = {"HER2": "positive"}
        assert _biomarker_hard_exclusion(patient, study) is True

    def test_matching_polarity_does_not_exclude(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "mutant"}
        assert _biomarker_hard_exclusion(patient, study) is False

    def test_study_has_no_biomarker_data_does_not_exclude(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {}
        assert _biomarker_hard_exclusion(patient, study) is False

    def test_patient_has_no_polarity_does_not_exclude(self):
        patient = [_make_biomarker("EGFR", None)]
        study = {"EGFR": "wild-type"}
        assert _biomarker_hard_exclusion(patient, study) is False

    def test_study_missing_specific_biomarker_does_not_exclude(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"HER2": "positive"}
        assert _biomarker_hard_exclusion(patient, study) is False

    def test_empty_patient_biomarkers_does_not_exclude(self):
        assert _biomarker_hard_exclusion([], {"EGFR": "mutant"}) is False

    def test_multiple_biomarkers_one_contradiction_excludes(self):
        patient = [
            _make_biomarker("EGFR", "mutant"),
            _make_biomarker("HER2", "positive"),
        ]
        study = {"EGFR": "mutant", "HER2": "negative"}
        assert _biomarker_hard_exclusion(patient, study) is True

    def test_multiple_biomarkers_no_contradiction(self):
        patient = [
            _make_biomarker("EGFR", "mutant"),
            _make_biomarker("HER2", "positive"),
        ]
        study = {"EGFR": "mutant", "HER2": "positive"}
        assert _biomarker_hard_exclusion(patient, study) is False


# ---------------------------------------------------------------------------
# check_biomarker_hard_exclusion (feature-flag wrapper)
# ---------------------------------------------------------------------------

class TestCheckBiomarkerHardExclusionWrapper:
    """Validates: Requirement 7.4"""

    def test_flag_off_skips_check(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "wild-type"}
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = False
            result = check_biomarker_hard_exclusion(patient, study, study_id="S1")
        assert result is False

    def test_flag_on_detects_contradiction(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "wild-type"}
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_biomarker_hard_exclusion(patient, study, study_id="S1")
        assert result is True

    def test_flag_on_no_contradiction_passes(self):
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "mutant"}
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_biomarker_hard_exclusion(patient, study, study_id="S1")
        assert result is False
