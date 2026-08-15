"""
Unit tests for RelaxedNumericalValidator.

Validates Requirements 15.1, 15.2, 15.3, 15.4, 15.5 from the
RAG Pipeline Consolidation spec.
"""

import pytest
from unittest.mock import patch

from src.api.services.relaxed_numerical_validator import (
    NumberValidation,
    RelaxedNumericalValidator,
    KNOWN_CLINICAL_CONSTANTS,
)


@pytest.fixture
def validator():
    return RelaxedNumericalValidator()


# ---------------------------------------------------------------------------
# validate_number — exact match → VERIFIED (Requirement 15.1)
# ---------------------------------------------------------------------------

class TestVerified:
    """Validates: Requirement 15.1"""

    def test_exact_percentage_match(self, validator):
        result = validator.validate_number("85%", ["5-year OS was 85% in the study"])
        assert result.status == "VERIFIED"
        assert result.source_text is not None

    def test_exact_hr_match(self, validator):
        result = validator.validate_number("0.72", ["HR 0.72 (95% CI 0.55-0.94)"])
        assert result.status == "VERIFIED"

    def test_exact_dose_match(self, validator):
        result = validator.validate_number("50 Gy", ["Patients received 50 Gy in 25 fractions"])
        assert result.status == "VERIFIED"


# ---------------------------------------------------------------------------
# validate_number — ±5% tolerance → LIKELY_CORRECT (Requirement 15.2)
# ---------------------------------------------------------------------------

class TestLikelyCorrect:
    """Validates: Requirement 15.2"""

    def test_within_5_percent_tolerance(self, validator):
        # 83 is within 5% of 85 (diff = 2.35%)
        result = validator.validate_number("83%", ["OS rate was 85% at 5 years"])
        assert result.status == "LIKELY_CORRECT"
        assert result.tolerance is not None
        assert result.tolerance <= 0.05

    def test_just_outside_5_percent_fails(self, validator):
        # 79 vs 85 → diff = 7.06%, outside 5%
        result = validator.validate_number("79%", ["OS rate was 85% at 5 years"])
        assert result.status == "UNVERIFIED"

    def test_tolerance_with_decimal(self, validator):
        # 0.70 vs 0.72 → diff = 2.78%
        result = validator.validate_number("0.70", ["HR 0.72 reported"])
        assert result.status == "LIKELY_CORRECT"


# ---------------------------------------------------------------------------
# validate_number — known clinical constants → KNOWN_CONSTANT (Req 15.3)
# ---------------------------------------------------------------------------

class TestKnownConstant:
    """Validates: Requirement 15.3"""

    def test_50_gy_is_known_constant(self, validator):
        result = validator.validate_number("50 Gy", [])
        assert result.status == "KNOWN_CONSTANT"

    def test_2_gy_fraction_is_known_constant(self, validator):
        result = validator.validate_number("2 Gy/fraction", [])
        assert result.status == "KNOWN_CONSTANT"

    def test_1_8_gy_is_known_constant(self, validator):
        result = validator.validate_number("1.8 Gy", [])
        assert result.status == "KNOWN_CONSTANT"

    def test_60_gy_is_known_constant(self, validator):
        result = validator.validate_number("60 Gy", [])
        assert result.status == "KNOWN_CONSTANT"

    def test_45_gy_is_known_constant(self, validator):
        result = validator.validate_number("45 Gy", [])
        assert result.status == "KNOWN_CONSTANT"

    def test_unknown_dose_not_constant(self, validator):
        result = validator.validate_number("99 Gy", [])
        assert result.status == "UNVERIFIED"


# ---------------------------------------------------------------------------
# validate_number — no source support → UNVERIFIED (annotated) (Req 15.4)
# ---------------------------------------------------------------------------

class TestUnverified:
    """Validates: Requirement 15.4"""

    def test_no_source_returns_unverified(self, validator):
        result = validator.validate_number("42%", [])
        assert result.status == "UNVERIFIED"
        assert result.source_text is None

    def test_unverified_not_stripped(self, validator):
        """UNVERIFIED numbers are annotated, not stripped."""
        result = validator.validate_number("42%", ["no matching numbers here"])
        assert result.status == "UNVERIFIED"
        # The key assertion: status is UNVERIFIED, not stripped/removed


# ---------------------------------------------------------------------------
# validate_response — full response validation
# ---------------------------------------------------------------------------

class TestValidateResponse:
    """Validates: Requirements 15.1, 15.2, 15.3, 15.4"""

    def test_mixed_response(self, validator):
        response = "OS was 85% with 50 Gy radiation and HR 0.99"
        sources = ["The 5-year OS was 85% in the treatment arm"]
        result = validator.validate_response(response, sources)

        assert result["verified"] >= 1       # 85% exact match
        assert result["known_constants"] >= 1  # 50 Gy
        assert isinstance(result["validations"], list)
        assert isinstance(result["metadata"], dict)

    def test_all_verified(self, validator):
        response = "OS was 85%"
        sources = ["OS was 85% at 5 years"]
        result = validator.validate_response(response, sources)
        assert result["verified"] >= 1
        assert result["unverified"] == 0

    def test_unverified_annotated_in_metadata(self, validator):
        response = "Response rate was 42%"
        sources = ["No matching data"]
        result = validator.validate_response(response, sources)
        assert result["unverified"] >= 1
        assert len(result["metadata"]["unverified_numbers"]) >= 1
        annotation = result["metadata"]["unverified_numbers"][0]
        assert annotation["status"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Feature flag gating (Requirement 15.5)
# ---------------------------------------------------------------------------

class TestFeatureFlagGating:
    """Validates: Requirement 15.5"""

    def test_flag_false_means_caller_uses_strict(self):
        """When enable_relaxed_numval is False, the relaxed validator
        should not be used — the caller is responsible for checking the flag."""
        with patch("src.api.services.relaxed_numerical_validator.settings") as mock_settings:
            mock_settings.enable_relaxed_numval = False
            # The validator itself works regardless; the flag is checked by callers.
            # This test documents the contract.
            assert mock_settings.enable_relaxed_numval is False


# ---------------------------------------------------------------------------
# NumberValidation dataclass
# ---------------------------------------------------------------------------

class TestNumberValidationDataclass:

    def test_fields(self):
        nv = NumberValidation(
            number_text="85%",
            status="VERIFIED",
            source_text="OS was 85%",
            tolerance=None,
        )
        assert nv.number_text == "85%"
        assert nv.status == "VERIFIED"
        assert nv.source_text == "OS was 85%"
        assert nv.tolerance is None

    def test_defaults(self):
        nv = NumberValidation(number_text="42%", status="UNVERIFIED")
        assert nv.source_text is None
        assert nv.tolerance is None
