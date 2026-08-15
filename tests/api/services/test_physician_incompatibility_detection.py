"""
Tests for typed incompatibility detection (2026-08-12 convergence
Sprint C item 17): extends physician_applicability_scorer.py's single
biomarker-only penalty (item 16) into the full typed taxonomy --
{type, severity, patient, evidence, reason} dicts distinguishing hard
incompatibility (penalizes score), soft mismatch (informational only),
and unknown (a numeric eligibility bound the evidence simply doesn't
report, surfaced only for trial_eligibility scoring).
"""

from __future__ import annotations

from src.api.services.physician.physician_applicability_scorer import (
    SEVERITY_HARD,
    SEVERITY_SOFT,
    SEVERITY_UNKNOWN,
    score_candidate,
)
from src.api.services.physician.physician_context_service import (
    THERAPY_SELECTION, TRIAL_ELIGIBILITY,
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


def _by_type(details, type_name):
    return [d for d in details if d["type"] == type_name]


class TestAxisMismatchesAreTypedAndSeverityLabeled:
    def test_biomarker_mismatch_is_hard(self):
        candidate = _candidate(
            text="EGFR-targeted therapy outcomes.",
            metadata={"applicability_meta": {"biomarkers": ["EGFR"]}},
        )
        scored = score_candidate(candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]})
        hits = _by_type(scored["incompatibility_details"], "biomarker_mismatch")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_HARD
        assert hits[0]["patient"] == "KRAS G12C"
        assert hits[0]["evidence"] == "EGFR"

    def test_cancer_type_mismatch_is_hard(self):
        candidate = _candidate(
            text="Outcomes in breast cancer patients.",
            metadata={"applicability_meta": {"cancer_types": ["breast"]}},
        )
        scored = score_candidate(candidate, intent=THERAPY_SELECTION, patient_values={"cancer_types": ["lung"]})
        hits = _by_type(scored["incompatibility_details"], "cancer_type_mismatch")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_HARD

    def test_histology_mismatch_is_soft(self):
        candidate = _candidate(
            text="Outcomes in adenocarcinoma patients.",
            metadata={"applicability_meta": {"histologies": ["adenocarcinoma"]}},
        )
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"histologies": ["squamous cell carcinoma"]},
        )
        hits = _by_type(scored["incompatibility_details"], "histology_mismatch")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_SOFT

    def test_prior_therapy_requirement_missing_is_soft(self):
        candidate = _candidate(
            text="Outcomes after platinum-based chemotherapy.",
            metadata={"applicability_meta": {"prior_treatments": ["platinum chemotherapy"]}},
        )
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"prior_treatments": ["surgery only"]},
        )
        hits = _by_type(scored["incompatibility_details"], "prior_therapy_requirement_missing")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_SOFT

    def test_organ_function_incompatible_is_hard(self):
        candidate = _candidate(
            text="Requires normal renal function.",
            metadata={"applicability_meta": {"organ_functions": ["normal renal function"]}},
        )
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"organ_functions": ["renal impairment"]},
        )
        hits = _by_type(scored["incompatibility_details"], "organ_function_incompatible")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_HARD

    def test_no_mismatch_when_axes_are_unspecified(self):
        scored = score_candidate(_candidate(), intent=THERAPY_SELECTION, patient_values={})
        assert scored["incompatibility_details"] == []


class TestIncompatibilityReasonsStaysInSyncWithDetails:
    def test_reasons_list_derived_from_details(self):
        candidate = _candidate(
            text="EGFR-targeted therapy outcomes.",
            metadata={"applicability_meta": {"biomarkers": ["EGFR"]}},
        )
        scored = score_candidate(candidate, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]})
        assert scored["incompatibility_reasons"] == [d["reason"] for d in scored["incompatibility_details"]]


class TestHardIncompatibilityPenalizesScoreSoftDoesNot:
    def test_hard_mismatch_reduces_the_combined_score(self):
        conflicting = _candidate(
            text="EGFR-targeted therapy outcomes.",
            metadata={"applicability_meta": {"biomarkers": ["EGFR"]}},
        )
        neutral = _candidate(text="General oncology overview.")
        c = score_candidate(conflicting, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]})
        n = score_candidate(neutral, intent=THERAPY_SELECTION, patient_values={"biomarkers": ["KRAS G12C"]})
        assert c["applicability_score"] < n["applicability_score"]
        assert c["components"]["hard_incompatibility"] is True

    def test_soft_only_mismatch_does_not_trigger_the_hard_penalty(self):
        """A histology-only mismatch (soft) must not multiply the score
        down the way a hard mismatch does."""
        candidate = _candidate(
            text="Outcomes in adenocarcinoma patients.",
            metadata={"applicability_meta": {"histologies": ["adenocarcinoma"]}},
        )
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"histologies": ["squamous cell carcinoma"]},
        )
        assert scored["components"]["hard_incompatibility"] is False


class TestFirstLineOnlyVsPreviouslyTreated:
    def test_detected_when_patient_is_subsequent_line_and_evidence_is_first_line_only(self):
        candidate = _candidate(metadata={"applicability_meta": {"treatment_lines": ["first_line_only"]}})
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"treatment_lines": ["second_line"]},
        )
        hits = _by_type(scored["incompatibility_details"], "first_line_only_vs_previously_treated")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_HARD

    def test_not_detected_when_evidence_allows_any_line(self):
        candidate = _candidate(metadata={"applicability_meta": {"treatment_lines": []}})
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"treatment_lines": ["second_line"]},
        )
        assert _by_type(scored["incompatibility_details"], "first_line_only_vs_previously_treated") == []


class TestTrialAgeAndEcogOnlyForTrialEligibilityIntent:
    def test_age_outside_range_is_hard(self):
        candidate = _candidate(metadata={"applicability_meta": {"age_range": {"min": 18, "max": 65}}})
        scored = score_candidate(candidate, intent=TRIAL_ELIGIBILITY, patient_values={"age": 72})
        hits = _by_type(scored["incompatibility_details"], "trial_age_incompatible")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_HARD

    def test_age_within_range_is_not_flagged(self):
        candidate = _candidate(metadata={"applicability_meta": {"age_range": {"min": 18, "max": 65}}})
        scored = score_candidate(candidate, intent=TRIAL_ELIGIBILITY, patient_values={"age": 45})
        assert _by_type(scored["incompatibility_details"], "trial_age_incompatible") == []

    def test_age_given_but_evidence_reports_no_range_is_unknown(self):
        candidate = _candidate()  # no age_range in metadata
        scored = score_candidate(candidate, intent=TRIAL_ELIGIBILITY, patient_values={"age": 45})
        hits = _by_type(scored["incompatibility_details"], "trial_age_incompatible")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_UNKNOWN

    def test_no_patient_age_means_no_age_finding_at_all(self):
        scored = score_candidate(_candidate(), intent=TRIAL_ELIGIBILITY, patient_values={})
        assert _by_type(scored["incompatibility_details"], "trial_age_incompatible") == []

    def test_ecog_exceeding_ceiling_is_hard(self):
        candidate = _candidate(metadata={"applicability_meta": {"ecog_max": 1}})
        scored = score_candidate(candidate, intent=TRIAL_ELIGIBILITY, patient_values={"ecog": 3})
        hits = _by_type(scored["incompatibility_details"], "ECOG_incompatible")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_HARD

    def test_ecog_within_ceiling_is_not_flagged(self):
        candidate = _candidate(metadata={"applicability_meta": {"ecog_max": 1}})
        scored = score_candidate(candidate, intent=TRIAL_ELIGIBILITY, patient_values={"ecog": 1})
        assert _by_type(scored["incompatibility_details"], "ECOG_incompatible") == []

    def test_ecog_given_but_evidence_reports_no_ceiling_is_unknown(self):
        candidate = _candidate()
        scored = score_candidate(candidate, intent=TRIAL_ELIGIBILITY, patient_values={"ecog": 1})
        hits = _by_type(scored["incompatibility_details"], "ECOG_incompatible")
        assert len(hits) == 1
        assert hits[0]["severity"] == SEVERITY_UNKNOWN

    def test_age_and_ecog_checks_are_skipped_for_non_trial_eligibility_intents(self):
        """An 'unknown eligibility' note on every therapy_selection
        candidate would just be noise -- these checks are scoped to
        trial_eligibility only."""
        candidate = _candidate()
        scored = score_candidate(
            candidate, intent=THERAPY_SELECTION, patient_values={"age": 90, "ecog": 4},
        )
        assert _by_type(scored["incompatibility_details"], "trial_age_incompatible") == []
        assert _by_type(scored["incompatibility_details"], "ECOG_incompatible") == []


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
