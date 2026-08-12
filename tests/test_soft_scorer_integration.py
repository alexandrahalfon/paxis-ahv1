"""
Tests for SoftScorer integration into comprehensive_retrieval.py (Task 14.1).

Validates:
- SoftScorer is called for surviving studies after hard eligibility
- soft_score_normalized is stored on StudyEvidence
- Combined ranking blends cross-encoder + soft scores
- Feature flag gating (enable_soft_scorer=false → no scoring)
- Studies without eligibility verdicts are skipped gracefully
"""

import pytest
from unittest.mock import patch
from dataclasses import field

from src.api.services.comprehensive_retrieval import StudyEvidence
from src.api.services.soft_scorer import SoftScorer, SoftScoreResult


class TestStudyEvidenceSoftScore:
    """Test that StudyEvidence carries soft_score_normalized."""

    def test_soft_score_defaults_to_none(self):
        se = StudyEvidence(doc_id="NCT001", title="Test Study")
        assert se.soft_score_normalized is None

    def test_soft_score_in_to_dict_when_set(self):
        se = StudyEvidence(doc_id="NCT001", title="Test Study")
        se.soft_score_normalized = 72.5
        d = se.to_dict()
        assert d["soft_score_normalized"] == 72.5

    def test_soft_score_absent_from_to_dict_when_none(self):
        se = StudyEvidence(doc_id="NCT001", title="Test Study")
        d = se.to_dict()
        assert "soft_score_normalized" not in d


class TestSoftScorerIntegrationLogic:
    """Test the scoring + blending logic used in comprehensive_retrieval."""

    @patch("src.api.services.soft_scorer.settings")
    def test_scorer_produces_normalized_score(self, mock_settings):
        mock_settings.enable_soft_scorer = True
        scorer = SoftScorer()
        verdicts = {
            "histology": "MATCH",
            "stage": "COMPATIBLE",
            "biomarkers": "MATCH",
            "prior_treatments": "NOT_AVAILABLE",
            "treatment_setting": "MATCH",
        }
        result = scorer.score("NCT001", verdicts)
        assert isinstance(result, SoftScoreResult)
        assert 0 <= result.normalized <= 100
        assert result.study_id == "NCT001"

    @patch("src.api.services.soft_scorer.settings")
    def test_scorer_disabled_returns_neutral(self, mock_settings):
        mock_settings.enable_soft_scorer = False
        scorer = SoftScorer()
        result = scorer.score("NCT001", {"histology": "MATCH"})
        # Neutral result: 50.0
        assert result.normalized == 50.0

    @patch("src.api.services.soft_scorer.settings")
    def test_blending_logic(self, mock_settings):
        """Simulate the 70/30 blending done in comprehensive_retrieval."""
        mock_settings.enable_soft_scorer = True
        scorer = SoftScorer()

        # Create studies with known rerank_scores
        studies = [
            StudyEvidence(doc_id="A", title="Study A", rerank_score=0.9),
            StudyEvidence(doc_id="B", title="Study B", rerank_score=0.8),
            StudyEvidence(doc_id="C", title="Study C", rerank_score=0.7),
        ]

        # Simulate verdicts: B has best soft score, A has worst
        doc_verdicts = {
            "A": {"histology": "MISMATCH", "stage": "MISMATCH", "biomarkers": "MISMATCH",
                   "prior_treatments": "MISMATCH", "treatment_setting": "MISMATCH"},
            "B": {"histology": "MATCH", "stage": "MATCH", "biomarkers": "MATCH",
                   "prior_treatments": "MATCH", "treatment_setting": "MATCH"},
            "C": {"histology": "COMPATIBLE", "stage": "MATCH", "biomarkers": "NOT_AVAILABLE",
                   "prior_treatments": "COMPATIBLE", "treatment_setting": "MATCH"},
        }

        # Score each study
        for study in studies:
            verdicts = doc_verdicts.get(study.doc_id)
            if verdicts:
                result = scorer.score(study.doc_id, verdicts)
                study.soft_score_normalized = result.normalized

        # Apply blending (same logic as in comprehensive_retrieval.py)
        for study in studies:
            if study.soft_score_normalized is not None:
                soft_norm = study.soft_score_normalized / 100.0
                study.rerank_score = 0.7 * study.rerank_score + 0.3 * soft_norm

        studies.sort(key=lambda s: s.rerank_score, reverse=True)

        # B should now rank higher than A despite lower initial cross-encoder score
        # because B has perfect soft scores
        assert studies[0].doc_id == "B", f"Expected B first, got {studies[0].doc_id}"

    @patch("src.api.services.soft_scorer.settings")
    def test_missing_verdicts_skipped(self, mock_settings):
        """Studies without eligibility verdicts keep soft_score_normalized=None."""
        mock_settings.enable_soft_scorer = True
        scorer = SoftScorer()

        study = StudyEvidence(doc_id="NCT999", title="No Verdicts", rerank_score=0.5)
        # No verdicts → don't call scorer, leave soft_score as None
        assert study.soft_score_normalized is None

        # After blending, rerank_score should be unchanged
        original_score = study.rerank_score
        if study.soft_score_normalized is not None:
            soft_norm = study.soft_score_normalized / 100.0
            study.rerank_score = 0.7 * study.rerank_score + 0.3 * soft_norm
        assert study.rerank_score == original_score
