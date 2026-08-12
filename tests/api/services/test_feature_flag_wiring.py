"""
Feature flag wiring verification tests.

Verifies that each RAG pipeline consolidation phase respects its feature flag:
- When the flag is False, the phase's code path is skipped (passthrough/neutral).
- When the flag is True, the phase's code path executes real logic.

**Validates: Requirements 18.1, 18.2**
"""

import pytest
from unittest.mock import patch

from src.api.services.biomarker_canonicalizer import BiomarkerCanonicalizer
from src.api.services.patient_eligibility_boost_service import (
    check_biomarker_hard_exclusion,
)
from src.api.services.biomarker_canonicalizer import CanonicalBiomarker
from src.api.services.soft_scorer import SoftScorer
from src.api.services.query_decomposer import QueryDecomposer
from src.core.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# 1. BiomarkerCanonicalizer — enable_canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalizationFlag:
    """Validates: Requirements 18.1, 18.2 — canonicalization flag wiring."""

    def test_flag_off_returns_passthrough(self):
        """enable_canonicalization=False → resolve() returns original name unchanged."""
        with patch("src.api.services.biomarker_canonicalizer.settings") as mock_settings:
            mock_settings.enable_canonicalization = False
            canon = BiomarkerCanonicalizer()
            result = canon.resolve("HER-2", polarity="amplified", raw_text="HER-2 amplified")

        # Passthrough: canonical_id should be the raw input, not resolved
        assert result.canonical_id == "HER-2"
        assert result.polarity == "amplified"

    def test_flag_on_returns_canonical_form(self):
        """enable_canonicalization=True → resolve() maps synonym to canonical ID."""
        with patch("src.api.services.biomarker_canonicalizer.settings") as mock_settings:
            mock_settings.enable_canonicalization = True
            canon = BiomarkerCanonicalizer()
            result = canon.resolve("HER-2", polarity="amplified", raw_text="HER-2 amplified")

        assert result.canonical_id == "HER2"
        assert result.polarity == "positive"


# ---------------------------------------------------------------------------
# 2. Hard gate — enable_hard_gate
# ---------------------------------------------------------------------------

class TestHardGateFlag:
    """Validates: Requirements 18.1, 18.2 — hard gate flag wiring."""

    def test_flag_off_returns_false(self):
        """enable_hard_gate=False → check_biomarker_hard_exclusion() returns False."""
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "wild-type"}
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = False
            result = check_biomarker_hard_exclusion(patient, study, study_id="TEST")
        assert result is False

    def test_flag_on_detects_contradiction(self):
        """enable_hard_gate=True → check_biomarker_hard_exclusion() detects contradictions."""
        patient = [_make_biomarker("EGFR", "mutant")]
        study = {"EGFR": "wild-type"}
        with patch("src.api.services.patient_eligibility_boost_service.settings") as mock_settings:
            mock_settings.enable_hard_gate = True
            result = check_biomarker_hard_exclusion(patient, study, study_id="TEST")
        assert result is True


# ---------------------------------------------------------------------------
# 3. Soft scorer — enable_soft_scorer
# ---------------------------------------------------------------------------

class TestSoftScorerFlag:
    """Validates: Requirements 18.1, 18.2 — soft scorer flag wiring."""

    def test_flag_off_returns_neutral_result(self):
        """enable_soft_scorer=False → SoftScorer.score() returns neutral 50.0 result."""
        with patch("src.api.services.soft_scorer.settings") as mock_settings:
            mock_settings.enable_soft_scorer = False
            scorer = SoftScorer()
            result = scorer.score(
                study_id="NCT00001",
                axis_verdicts={"histology": "MATCH", "biomarkers": "MISMATCH"},
            )

        assert result.normalized == 50.0
        # All axes should have score 0.5 (neutral)
        for axis_score in result.axis_scores:
            assert axis_score.score == 0.5

    def test_flag_on_returns_actual_scores(self):
        """enable_soft_scorer=True → SoftScorer.score() returns real graduated scores."""
        with patch("src.api.services.soft_scorer.settings") as mock_settings:
            mock_settings.enable_soft_scorer = True
            scorer = SoftScorer()
            result = scorer.score(
                study_id="NCT00001",
                axis_verdicts={"histology": "MATCH", "biomarkers": "MISMATCH"},
            )

        # With MATCH on histology and MISMATCH on biomarkers, score should differ from neutral
        assert result.normalized != 50.0
        # histology MATCH → 1.0, biomarkers MISMATCH → -0.2
        hist_score = next(s for s in result.axis_scores if s.axis == "histology")
        bio_score = next(s for s in result.axis_scores if s.axis == "biomarkers")
        assert hist_score.score == 1.0
        assert bio_score.score == -0.2


# ---------------------------------------------------------------------------
# 4. Query decomposition — enable_query_decomposition
# ---------------------------------------------------------------------------

class TestQueryDecompositionFlag:
    """Validates: Requirements 18.1, 18.2 — query decomposition flag wiring."""

    @pytest.mark.asyncio
    async def test_flag_off_returns_original_query(self):
        """enable_query_decomposition=False → decompose() returns original query unchanged."""
        with patch("src.api.services.query_decomposer.settings") as mock_settings:
            mock_settings.enable_query_decomposition = False
            decomposer = QueryDecomposer()
            result = await decomposer.decompose("pembrolizumab vs nivolumab for NSCLC")

        assert result.is_decomposed is False
        assert result.sub_queries == ["pembrolizumab vs nivolumab for NSCLC"]
        assert result.original == "pembrolizumab vs nivolumab for NSCLC"


# ---------------------------------------------------------------------------
# 5. All flags default to False in Settings
# ---------------------------------------------------------------------------

class TestAllFlagsDefaultFalse:
    """Validates: Requirement 18.2 — all flags default to False."""

    def test_all_consolidation_flags_default_false(self):
        """All RAG pipeline consolidation feature flags default to False."""
        s = Settings()
        assert s.enable_canonicalization is False
        assert s.enable_hard_gate is False
        assert s.enable_soft_scorer is False
        assert s.enable_pto_retrieval is False
        assert s.enable_query_decomposition is False
        assert s.enable_relaxed_numval is False
        assert s.enable_perf_optimizations is False
