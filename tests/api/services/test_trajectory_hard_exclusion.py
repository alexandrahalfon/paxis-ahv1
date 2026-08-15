"""
Unit tests for _trajectory_hard_exclusion() and check_trajectory_hard_exclusion().

Validates Requirements 7.2 and 7.5 from the RAG Pipeline Consolidation spec.
"""

import pytest
from unittest.mock import patch

from src.api.services.patient_eligibility_boost_service import (
    _trajectory_hard_exclusion,
    check_trajectory_hard_exclusion,
)


# ---------------------------------------------------------------------------
# _trajectory_hard_exclusion (core logic)
# ---------------------------------------------------------------------------

class TestTrajectoryHardExclusion:
    """Core trajectory contradiction detection."""

    # -- Contradictions that SHOULD exclude --

    def test_naive_patient_vs_second_line_study_excludes(self):
        assert _trajectory_hard_exclusion("treatment_naive", "second_line") is True

    def test_naive_patient_vs_later_line_study_excludes(self):
        assert _trajectory_hard_exclusion("treatment_naive", "later_line") is True

    def test_naive_patient_vs_refractory_study_excludes(self):
        assert _trajectory_hard_exclusion("treatment_naive", "refractory") is True

    def test_first_line_patient_vs_second_line_study_excludes(self):
        assert _trajectory_hard_exclusion("first_line", "second_line") is True

    def test_first_line_patient_vs_refractory_study_excludes(self):
        assert _trajectory_hard_exclusion("first_line", "refractory") is True

    def test_second_line_patient_vs_naive_study_excludes(self):
        assert _trajectory_hard_exclusion("second_line", "treatment_naive") is True

    def test_refractory_patient_vs_first_line_study_excludes(self):
        assert _trajectory_hard_exclusion("refractory", "first_line") is True

    def test_later_line_patient_vs_treatment_naive_study_excludes(self):
        assert _trajectory_hard_exclusion("later_line", "treatment_naive") is True

    # -- Non-contradictions that should NOT exclude --

    def test_same_trajectory_does_not_exclude(self):
        assert _trajectory_hard_exclusion("treatment_naive", "treatment_naive") is False

    def test_both_experienced_does_not_exclude(self):
        assert _trajectory_hard_exclusion("second_line", "refractory") is False

    def test_both_naive_group_does_not_exclude(self):
        assert _trajectory_hard_exclusion("treatment_naive", "first_line") is False

    def test_adjuvant_vs_naive_does_not_exclude(self):
        assert _trajectory_hard_exclusion("adjuvant", "treatment_naive") is False

    def test_neoadjuvant_vs_second_line_does_not_exclude(self):
        assert _trajectory_hard_exclusion("neoadjuvant", "second_line") is False

    def test_adjuvant_vs_refractory_does_not_exclude(self):
        assert _trajectory_hard_exclusion("adjuvant", "refractory") is False

    # -- Empty / missing trajectories --

    def test_empty_patient_trajectory_does_not_exclude(self):
        assert _trajectory_hard_exclusion("", "second_line") is False

    def test_empty_study_trajectory_does_not_exclude(self):
        assert _trajectory_hard_exclusion("treatment_naive", "") is False

    def test_both_empty_does_not_exclude(self):
        assert _trajectory_hard_exclusion("", "") is False

    # -- Case insensitivity --

    def test_case_insensitive_match(self):
        assert _trajectory_hard_exclusion("Treatment_Naive", "Second_Line") is True

    def test_whitespace_trimmed(self):
        assert _trajectory_hard_exclusion("  treatment_naive  ", "second_line") is True


# ---------------------------------------------------------------------------
# check_trajectory_hard_exclusion (feature-flag wrapper)
# ---------------------------------------------------------------------------

class TestCheckTrajectoryHardExclusion:
    """Feature-flag-gated wrapper tests."""

    def test_flag_disabled_skips_check(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = False
            result = check_trajectory_hard_exclusion(
                "treatment_naive", "second_line", study_id="S1"
            )
        assert result is False

    def test_flag_enabled_detects_contradiction(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_trajectory_hard_exclusion(
                "treatment_naive", "second_line", study_id="S1"
            )
        assert result is True

    def test_flag_enabled_no_contradiction_passes(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_trajectory_hard_exclusion(
                "treatment_naive", "first_line", study_id="S1"
            )
        assert result is False

    def test_flag_enabled_logs_exclusion(self, capsys):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            check_trajectory_hard_exclusion(
                "treatment_naive", "refractory", study_id="STUDY_42"
            )
        captured = capsys.readouterr()
        assert "[HardGate]" in captured.out
        assert "STUDY_42" in captured.out
        assert "trajectory_contradiction" in captured.out
