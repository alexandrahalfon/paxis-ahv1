"""
Tests for evidence_hierarchy.py (2026-08-12 convergence Sprint A item 5):
per-audience/intent authority ordering, applied as a PRIOR on top of
relevance -- never a replacement for it.
"""

from __future__ import annotations

from src.api.services.evidence.evidence_hierarchy import (
    DOSE_MODIFICATION,
    GENERIC_EDUCATION,
    GUIDELINE,
    HIERARCHY_POLICIES,
    MEDICATION_AUTHORITY,
    PATIENT_EDUCATION,
    PATIENT_SELF_CARE,
    PHYSICIAN_STANDARD_OF_CARE,
    REGULATORY_LABEL,
    apply_authority_prior,
    authority_prior,
    infer_evidence_type,
    select_hierarchy,
)


class TestSelectHierarchy:
    def test_physician_default_is_standard_of_care(self):
        assert select_hierarchy("physician") == HIERARCHY_POLICIES[PHYSICIAN_STANDARD_OF_CARE]

    def test_patient_default_is_self_care(self):
        assert select_hierarchy("patient") == HIERARCHY_POLICIES[PATIENT_SELF_CARE]

    def test_dose_modification_intent_overrides_physician_audience(self):
        assert select_hierarchy("physician", intent="dose_modification") == HIERARCHY_POLICIES[DOSE_MODIFICATION]

    def test_dose_modification_intent_overrides_patient_audience(self):
        assert select_hierarchy("patient", intent="medication_explainer") == HIERARCHY_POLICIES[DOSE_MODIFICATION]

    def test_unrelated_intent_does_not_trigger_dose_modification_policy(self):
        assert select_hierarchy("patient", intent="nutrition") == HIERARCHY_POLICIES[PATIENT_SELF_CARE]


class TestInferEvidenceType:
    def test_maps_known_collections(self):
        assert infer_evidence_type({"collection": "oncology_clinical_guidelines"}) == GUIDELINE
        assert infer_evidence_type({"collection": "oncology_patient_education"}) == PATIENT_EDUCATION
        assert infer_evidence_type({"collection": "oncology_medication_knowledge"}) == MEDICATION_AUTHORITY

    def test_reads_corpus_field_too(self):
        """EvidenceCandidate uses `corpus`, multi_corpus_retriever dicts
        use `collection` -- both must work."""
        assert infer_evidence_type({"corpus": "oncology_clinical_guidelines"}) == GUIDELINE

    def test_authority_class_a_without_a_known_collection_maps_to_guideline(self):
        assert infer_evidence_type({"collection": "some_other_corpus", "authority_class": "A"}) == GUIDELINE

    def test_unknown_falls_back_to_generic_education(self):
        assert infer_evidence_type({"collection": "unrecognized"}) == GENERIC_EDUCATION
        assert infer_evidence_type({}) == GENERIC_EDUCATION


class TestAuthorityPrior:
    def test_first_tier_scores_highest(self):
        tiers = [[GUIDELINE], [REGULATORY_LABEL]]
        assert authority_prior(GUIDELINE, tiers) > authority_prior(REGULATORY_LABEL, tiers)

    def test_types_in_the_same_tier_score_equally(self):
        tiers = HIERARCHY_POLICIES[DOSE_MODIFICATION]
        assert authority_prior(REGULATORY_LABEL, tiers) == authority_prior(GUIDELINE, tiers)

    def test_untagged_type_gets_neutral_prior_not_lowest(self):
        # A full 6-tier policy, so the bottom tier's position-based score
        # is genuinely below the neutral untagged value -- being
        # EXPLICITLY ranked last is worse than having no opinion at all.
        tiers = HIERARCHY_POLICIES[PHYSICIAN_STANDARD_OF_CARE]
        untagged = authority_prior("something_not_in_any_tier", tiers)
        assert authority_prior(GENERIC_EDUCATION, tiers) < untagged < authority_prior(GUIDELINE, tiers)

    def test_tiers_beyond_the_defined_score_table_do_not_error(self):
        long_tiers = [[f"type_{i}"] for i in range(20)]
        # Must not raise, and must stay non-increasing.
        scores = [authority_prior(f"type_{i}", long_tiers) for i in range(20)]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestApplyAuthorityPrior:
    def test_evidence_type_and_authority_prior_recorded_on_each_candidate(self):
        candidates = [{"applicability_score": 0.5, "collection": "oncology_clinical_guidelines"}]
        out = apply_authority_prior(candidates, audience="physician")
        assert out[0]["evidence_type"] == GUIDELINE
        assert out[0]["authority_prior"] > 0

    def test_does_not_mutate_the_input_dicts(self):
        original = {"applicability_score": 0.5, "collection": "oncology_clinical_guidelines"}
        candidates = [original]
        apply_authority_prior(candidates, audience="physician")
        assert "evidence_type" not in original

    def test_reorders_by_blended_score(self):
        candidates = [
            {"applicability_score": 0.60, "collection": "unrecognized", "id": "low_authority"},
            {"applicability_score": 0.58, "collection": "oncology_clinical_guidelines", "id": "high_authority"},
        ]
        out = apply_authority_prior(candidates, audience="physician", prior_weight=0.15)
        assert [c["id"] for c in out] == ["high_authority", "low_authority"]

    def test_a_much_more_relevant_low_authority_candidate_still_wins(self):
        """Authority is a prior, not a substitute for relevance -- the
        default weight must not let a low-relevance guideline beat a
        highly relevant study."""
        candidates = [
            {"applicability_score": 0.95, "collection": "unrecognized", "id": "very_relevant"},
            {"applicability_score": 0.20, "collection": "oncology_clinical_guidelines", "id": "high_authority_low_relevance"},
        ]
        out = apply_authority_prior(candidates, audience="physician")
        assert out[0]["id"] == "very_relevant"

    def test_blended_score_formula(self):
        candidates = [{"applicability_score": 0.6, "collection": "oncology_clinical_guidelines"}]
        out = apply_authority_prior(candidates, audience="physician", prior_weight=0.2)
        prior = out[0]["authority_prior"]
        expected = round((1 - 0.2) * 0.6 + 0.2 * prior, 4)
        assert out[0]["applicability_score_with_authority_prior"] == expected

    def test_custom_score_key_is_respected(self):
        candidates = [{"my_score": 0.7, "collection": "oncology_patient_education"}]
        out = apply_authority_prior(candidates, audience="patient", score_key="my_score")
        assert "my_score_with_authority_prior" in out[0]


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
