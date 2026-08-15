"""
Unit tests for _check_stage_subsumption() and check_stage_subsumption().

Validates Requirements 8.1 and 8.2 from the RAG Pipeline Consolidation spec.
"""

import pytest
from unittest.mock import patch

from src.api.services.patient_eligibility_boost_service import (
    _check_stage_subsumption,
    check_stage_subsumption,
)


# ---------------------------------------------------------------------------
# _check_stage_subsumption (core logic)
# ---------------------------------------------------------------------------

class TestCheckStageSubsumption:
    """Core stage subsumption logic."""

    # -- Exact matches → MATCH --

    def test_exact_match_same_stage(self):
        assert _check_stage_subsumption("III", "III") == "MATCH"

    def test_exact_match_with_suffix(self):
        assert _check_stage_subsumption("IIIA", "IIIA") == "MATCH"

    def test_exact_match_case_insensitive(self):
        assert _check_stage_subsumption("iiia", "IIIA") == "MATCH"

    def test_exact_match_with_stage_prefix(self):
        assert _check_stage_subsumption("Stage III", "Stage III") == "MATCH"

    def test_exact_match_mixed_prefix(self):
        assert _check_stage_subsumption("Stage II", "II") == "MATCH"

    # -- Sub-stage subsumption → COMPATIBLE (Req 8.1) --

    def test_substage_iiia_matches_parent_iii(self):
        """Patient IIIA is subsumed by study III."""
        assert _check_stage_subsumption("IIIA", "III") == "COMPATIBLE"

    def test_substage_iiib_matches_parent_iii(self):
        assert _check_stage_subsumption("IIIB", "III") == "COMPATIBLE"

    def test_substage_iiic_matches_parent_iii(self):
        assert _check_stage_subsumption("IIIC", "III") == "COMPATIBLE"

    def test_substage_iia_matches_parent_ii(self):
        assert _check_stage_subsumption("IIA", "II") == "COMPATIBLE"

    def test_substage_iib_matches_parent_ii(self):
        assert _check_stage_subsumption("IIB", "II") == "COMPATIBLE"

    def test_substage_ia_matches_parent_i(self):
        assert _check_stage_subsumption("IA", "I") == "COMPATIBLE"

    def test_substage_ib_matches_parent_i(self):
        assert _check_stage_subsumption("IB", "I") == "COMPATIBLE"

    def test_substage_iva_matches_parent_iv(self):
        assert _check_stage_subsumption("IVA", "IV") == "COMPATIBLE"

    def test_substage_ivb_matches_parent_iv(self):
        assert _check_stage_subsumption("IVB", "IV") == "COMPATIBLE"

    # -- Study sub-stage, patient parent → COMPATIBLE --

    def test_patient_parent_matches_study_substage(self):
        """Patient III matches study IIIA (study is more specific)."""
        assert _check_stage_subsumption("III", "IIIA") == "COMPATIBLE"

    # -- Range containment → COMPATIBLE (Req 8.2) --

    def test_stage_ii_in_range_ii_iii(self):
        assert _check_stage_subsumption("II", "II-III") == "COMPATIBLE"

    def test_stage_iii_in_range_ii_iii(self):
        assert _check_stage_subsumption("III", "II-III") == "COMPATIBLE"

    def test_stage_i_in_range_i_ii(self):
        assert _check_stage_subsumption("I", "I-II") == "COMPATIBLE"

    def test_stage_ii_in_range_i_ii(self):
        assert _check_stage_subsumption("II", "I-II") == "COMPATIBLE"

    def test_stage_iii_in_range_iii_iv(self):
        assert _check_stage_subsumption("III", "III-IV") == "COMPATIBLE"

    def test_stage_iv_in_range_iii_iv(self):
        assert _check_stage_subsumption("IV", "III-IV") == "COMPATIBLE"

    def test_substage_iiia_in_range_ii_iii(self):
        """Sub-stage IIIA base is III, which falls in II-III range."""
        assert _check_stage_subsumption("IIIA", "II-III") == "COMPATIBLE"

    def test_substage_iib_in_range_i_iii(self):
        """Sub-stage IIB base is II, which falls in I-III range."""
        assert _check_stage_subsumption("IIB", "I-III") == "COMPATIBLE"

    # -- Clear mismatches → MISMATCH --

    def test_stage_iv_vs_stage_ii_mismatch(self):
        assert _check_stage_subsumption("IV", "II") == "MISMATCH"

    def test_stage_i_vs_stage_iii_mismatch(self):
        assert _check_stage_subsumption("I", "III") == "MISMATCH"

    def test_stage_iiia_vs_stage_ii_mismatch(self):
        """IIIA base is III, which doesn't match II."""
        assert _check_stage_subsumption("IIIA", "II") == "MISMATCH"

    def test_stage_i_outside_range_ii_iii(self):
        assert _check_stage_subsumption("I", "II-III") == "MISMATCH"

    def test_stage_iv_outside_range_i_ii(self):
        assert _check_stage_subsumption("IV", "I-II") == "MISMATCH"

    # -- Empty / missing stages → MISMATCH --

    def test_empty_patient_stage(self):
        assert _check_stage_subsumption("", "III") == "MISMATCH"

    def test_empty_study_stage(self):
        assert _check_stage_subsumption("III", "") == "MISMATCH"

    def test_both_empty(self):
        assert _check_stage_subsumption("", "") == "MISMATCH"


# ---------------------------------------------------------------------------
# check_stage_subsumption (feature-flag wrapper)
# ---------------------------------------------------------------------------

class TestCheckStageSubsumptionFlagGated:
    """Feature-flag-gated wrapper tests."""

    def test_flag_disabled_exact_match_returns_match(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = False
            result = check_stage_subsumption("III", "III")
        assert result == "MATCH"

    def test_flag_disabled_substage_returns_mismatch(self):
        """When flag is off, subsumption is NOT applied — strict exact match."""
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = False
            result = check_stage_subsumption("IIIA", "III")
        assert result == "MISMATCH"

    def test_flag_enabled_substage_returns_compatible(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_stage_subsumption("IIIA", "III", study_id="S1")
        assert result == "COMPATIBLE"

    def test_flag_enabled_range_returns_compatible(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_stage_subsumption("II", "II-III", study_id="S1")
        assert result == "COMPATIBLE"

    def test_flag_enabled_mismatch_returns_mismatch(self):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_stage_subsumption("IV", "II", study_id="S1")
        assert result == "MISMATCH"

    def test_flag_enabled_logs_compatible(self, capsys):
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            check_stage_subsumption("IIIA", "III", study_id="STUDY_99")
        captured = capsys.readouterr()
        assert "[HardGate]" in captured.out
        assert "STUDY_99" in captured.out
        assert "stage_subsumption" in captured.out
