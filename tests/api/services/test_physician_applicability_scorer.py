"""
Tests for physician_applicability_scorer.py (2026-08-12 convergence
Sprint C item 16): "is this passage relevant?" (already answered by the
existing retriever) vs. "does this apply to THIS patient?" -- the
physician-side counterpart to evidence/applicability_scorer.py.
"""

from __future__ import annotations

import pytest

from src.api.services.physician.physician_applicability_scorer import (
    WEIGHTS_BY_PHYSICIAN_INTENT,
    _COMPONENT_NAMES,
    _DEFAULT_WEIGHTS,
    _freshness_score,
    rank,
    score_candidate,
)
from src.api.services.physician.physician_context_service import (
    THERAPY_SELECTION, TOXICITY_MANAGEMENT, TRIAL_ELIGIBILITY,
)


def _candidate(**overrides):
    base = {
        "text": "Adagrasib demonstrated a progression-free survival benefit in KRAS G12C-mutant NSCLC.",
        "title": "Adagrasib in KRAS G12C NSCLC", "authority_class": "A",
        "publication_date": "2024", "metadata": {"applicability_meta": {}},
        "collection": "oncology_clinical_guidelines",
    }
    base.update(overrides)
    return base


class TestWeightTablesAreComplete:
    @pytest.mark.parametrize("intent", list(WEIGHTS_BY_PHYSICIAN_INTENT.keys()))
    def test_sums_to_one(self, intent):
        assert sum(WEIGHTS_BY_PHYSICIAN_INTENT[intent].values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("intent", list(WEIGHTS_BY_PHYSICIAN_INTENT.keys()))
    def test_covers_exactly_the_component_set(self, intent):
        assert set(WEIGHTS_BY_PHYSICIAN_INTENT[intent].keys()) == set(_COMPONENT_NAMES)

    def test_default_weights_sum_to_one_and_cover_the_component_set(self):
        assert sum(_DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
        assert set(_DEFAULT_WEIGHTS.keys()) == set(_COMPONENT_NAMES)


class TestFreshnessScore:
    def test_recent_publication_scores_full_credit(self):
        assert _freshness_score("2024", as_of_year=2026) == 1.0

    def test_very_old_publication_scores_the_floor(self):
        assert _freshness_score("2005", as_of_year=2026) == 0.2

    def test_missing_date_is_neutral(self):
        assert _freshness_score(None, as_of_year=2026) == 0.5

    def test_mid_range_interpolates_between_full_and_floor(self):
        score = _freshness_score("2018", as_of_year=2026)  # 8 years old
        assert 0.2 < score < 1.0

    def test_unparseable_date_does_not_crash(self):
        assert _freshness_score("unknown", as_of_year=2026) == 0.5


class TestScoreCandidateComponentMatching:
    def test_unspecified_patient_axis_is_neutral(self):
        scored = score_candidate(_candidate(), intent=THERAPY_SELECTION, patient_values={})
        assert scored["components"]["biomarker"] == 0.5

    def test_named_match_via_structured_tag(self):
        candidate = _candidate(metadata={"applicability_meta": {"biomarkers": ["KRAS G12C"]}})
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]},
        )
        assert scored["components"]["biomarker"] == 1.0

    def test_named_match_via_text_fallback_when_untagged(self):
        candidate = _candidate(text="Efficacy data for KRAS G12C-mutant NSCLC patients.")
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["kras g12c"]},
        )
        assert scored["components"]["biomarker"] == 1.0

    def test_named_mismatch_with_no_text_corroboration_scores_zero(self):
        candidate = _candidate(
            text="EGFR-targeted therapy outcomes.",
            metadata={"applicability_meta": {"biomarkers": ["EGFR"]}},
        )
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]},
        )
        assert scored["components"]["biomarker"] == 0.0

    def test_candidate_dict_is_not_mutated(self):
        candidate = _candidate()
        original = dict(candidate)
        score_candidate(candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]})
        assert candidate == original


class TestBiomarkerConflictPenalty:
    def test_conflict_populates_incompatibility_reasons(self):
        candidate = _candidate(
            text="EGFR-targeted therapy outcomes in EGFR-mutant NSCLC.",
            metadata={"applicability_meta": {"biomarkers": ["EGFR"]}},
        )
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]},
        )
        assert scored["incompatibility_reasons"]
        assert "biomarker_mismatch" in scored["incompatibility_reasons"][0]
        assert scored["components"]["biomarker_conflict"] is True

    def test_conflict_penalizes_the_combined_score(self):
        conflicting = _candidate(
            text="EGFR-targeted therapy outcomes in EGFR-mutant NSCLC.",
            metadata={"applicability_meta": {"biomarkers": ["EGFR"]}},
        )
        neutral = _candidate(
            text="General oncology overview.",
            metadata={"applicability_meta": {}},
        )
        conflicting_score = score_candidate(
            conflicting, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]},
        )["applicability_score"]
        neutral_score = score_candidate(
            neutral, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]},
        )["applicability_score"]
        assert conflicting_score < neutral_score

    def test_no_conflict_when_patient_has_no_recorded_biomarker(self):
        candidate = _candidate(metadata={"applicability_meta": {"biomarkers": ["EGFR"]}})
        scored = score_candidate(candidate, intent=THERAPY_SELECTION, patient_values={})
        assert scored["incompatibility_reasons"] == []


class TestEvidenceTypeUsesEvidenceHierarchy:
    def test_guideline_collection_scores_higher_evidence_type_than_unrecognized(self):
        guideline = _candidate(collection="oncology_clinical_guidelines")
        # authority_class=None too -- infer_evidence_type() also falls
        # back to GUIDELINE for authority_class="A" regardless of
        # collection (by design, evidence_hierarchy.py), so isolating
        # the collection's own effect means clearing that fallback too.
        unrecognized = _candidate(collection="some_random_corpus", authority_class=None)
        g = score_candidate(guideline, intent=THERAPY_SELECTION, patient_values={})
        u = score_candidate(unrecognized, intent=THERAPY_SELECTION, patient_values={})
        assert g["components"]["evidence_type"] > u["components"]["evidence_type"]


class TestAuthorityComponent:
    def test_known_authority_class_a_scores_full(self):
        scored = score_candidate(_candidate(authority_class="A"), intent=THERAPY_SELECTION, patient_values={})
        assert scored["components"]["authority"] == 1.0

    def test_unknown_authority_class_is_mildly_penalized_not_neutral(self):
        scored = score_candidate(
            _candidate(authority_class=None), intent=THERAPY_SELECTION, patient_values={},
        )
        assert scored["components"]["authority"] == 0.6


class TestIntentSpecificWeightsProduceDifferentScores:
    def test_same_candidate_scores_differently_under_different_intents(self):
        candidate = _candidate(
            text="Adagrasib demonstrated a PFS benefit in KRAS G12C NSCLC.",
            metadata={"applicability_meta": {"biomarkers": ["KRAS G12C"]}},
        )
        patient_values = {"biomarkers": ["KRAS G12C"]}
        therapy = score_candidate(candidate, intent=THERAPY_SELECTION, patient_values=patient_values)
        toxicity = score_candidate(candidate, intent=TOXICITY_MANAGEMENT, patient_values=patient_values)
        assert therapy["applicability_score"] != toxicity["applicability_score"]

    def test_unrecognized_intent_falls_back_to_default_weights_without_crashing(self):
        scored = score_candidate(_candidate(), intent="some_unrecognized_intent", patient_values={})
        assert 0 <= scored["applicability_score"] <= 1


class TestRank:
    def test_sorts_descending_by_applicability_score(self):
        candidates = [
            _candidate(text="EGFR-targeted therapy.", metadata={"applicability_meta": {"biomarkers": ["EGFR"]}}),
            _candidate(text="KRAS G12C targeted therapy.", metadata={"applicability_meta": {"biomarkers": ["KRAS G12C"]}}),
        ]
        ranked = rank(candidates, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]})
        assert ranked[0]["applicability_score"] >= ranked[1]["applicability_score"]
        assert "KRAS G12C" in ranked[0]["text"]

    def test_respects_limit(self):
        candidates = [_candidate() for _ in range(5)]
        ranked = rank(candidates, intent=TRIAL_ELIGIBILITY, patient_values={}, limit=2)
        assert len(ranked) == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
