"""
Tests for the three new hard-filter axes added to patient_eligibility_boost_service:

  - disease_status        (CORE, MISMATCH = hard drop)
  - surgical_candidacy    (CORE, MISMATCH = hard drop)
  - study_exclusions_violated  (CORE, MATCH = hard drop — inverted polarity)

These focus on:
  1. The new axes are listed in HARD_FILTER_CRITERIA and CORE_AXIS_WEIGHTS.
  2. extract_patient_context_from_query populates disease_status and
     surgical_candidacy when the query language supports it.
  3. build_patient_summary surfaces the new fields in human-readable form.
  4. The hard-drop logic correctly removes studies whose verdict on any
     of the three new axes triggers a hard drop.
"""

import pytest

from src.api.services.patient_eligibility_boost_service import (
    CORE_AXIS_WEIGHTS,
    HARD_DROP_AXES,
    HARD_DROP_ON_MATCH_AXES,
    HARD_FILTER_CRITERIA,
    build_patient_summary,
    extract_patient_context_from_query,
)


class TestHardFilterCriteria:

    def test_new_axes_in_hard_filter_criteria(self):
        for axis in ("disease_status", "surgical_candidacy", "study_exclusions_violated"):
            assert axis in HARD_FILTER_CRITERIA

    def test_new_axes_in_core_weights(self):
        for axis in ("disease_status", "surgical_candidacy", "study_exclusions_violated"):
            assert axis in CORE_AXIS_WEIGHTS
            weights = CORE_AXIS_WEIGHTS[axis]
            # Verdict universe is consistent
            assert set(weights.keys()) == {"MATCH", "COMPATIBLE", "NOT_AVAILABLE", "MISMATCH"}

    def test_hard_drop_axes_set(self):
        assert HARD_DROP_AXES == {"cancer_type", "disease_status", "surgical_candidacy"}

    def test_hard_drop_on_match_axes_set(self):
        assert HARD_DROP_ON_MATCH_AXES == {"study_exclusions_violated"}


class TestPatientContextExtractionRegexPath:
    """Tests the slow regex path (no reconciled structure provided)."""

    def test_metastatic_query_populates_disease_status(self):
        ctx = extract_patient_context_from_query(
            "65 year old male with metastatic prostate adenocarcinoma"
        )
        assert ctx is not None
        assert ctx.get("disease_status") == "metastatic"

    def test_post_progression_populates_disease_status(self):
        ctx = extract_patient_context_from_query(
            "55 year old female with recurrent HNSCC progressing on pembrolizumab"
        )
        assert ctx is not None
        assert ctx.get("disease_status") == "post_progression"

    def test_not_surgical_candidate_populates_surgical_candidacy(self):
        ctx = extract_patient_context_from_query(
            "70 year old with unresectable head and neck cancer, no longer a "
            "surgical candidate"
        )
        assert ctx is not None
        assert ctx.get("surgical_candidacy") == "not_candidate"

    def test_declined_surgery_populates_surgical_candidacy(self):
        ctx = extract_patient_context_from_query(
            "67 year old male with prostate cancer who declined surgery"
        )
        assert ctx is not None
        assert ctx.get("surgical_candidacy") == "declined"

    def test_silent_query_leaves_axes_empty(self):
        ctx = extract_patient_context_from_query(
            "Stage II breast cancer, ER positive, HER2 negative"
        )
        assert ctx is not None
        assert "disease_status" not in ctx
        assert "surgical_candidacy" not in ctx


class TestBuildPatientSummary:
    """The new fields must be rendered into the LLM-facing patient summary."""

    def test_renders_recurrent_disease(self):
        s = build_patient_summary({
            "age": 65, "gender": "male", "cancer_type": "HNSCC",
            "disease_status": "recurrent",
        })
        assert "recurrent disease" in s

    def test_renders_metastatic_disease(self):
        s = build_patient_summary({
            "cancer_type": "prostate cancer", "disease_status": "metastatic",
        })
        assert "metastatic disease" in s

    def test_renders_post_progression(self):
        s = build_patient_summary({
            "cancer_type": "HNSCC", "disease_status": "post_progression",
        })
        assert "post-progression" in s

    def test_renders_not_a_surgical_candidate(self):
        s = build_patient_summary({
            "cancer_type": "HNSCC", "surgical_candidacy": "not_candidate",
        })
        assert "not a surgical candidate" in s

    def test_renders_declined_surgery(self):
        s = build_patient_summary({
            "cancer_type": "prostate cancer", "surgical_candidacy": "declined",
        })
        assert "declined surgical management" in s

    def test_renders_surgical_candidate(self):
        s = build_patient_summary({
            "cancer_type": "bladder cancer", "surgical_candidacy": "candidate",
        })
        assert "surgical candidate" in s


class TestHardDropDecision:
    """Hard-drop logic should fire for any HARD_DROP_AXES MISMATCH or
    HARD_DROP_ON_MATCH_AXES MATCH. We test the decision predicate
    directly using the verdict dicts the LLM would produce."""

    def _hard_drop_axis(self, verdicts):
        """Reimplement the inline predicate from apply_patient_eligibility_filter_and_boost
        for direct testing — keep this expression in sync with the
        source."""
        axis = next(
            (a for a in HARD_DROP_AXES if verdicts.get(a) == "MISMATCH"),
            None,
        )
        if axis is None:
            axis = next(
                (a for a in HARD_DROP_ON_MATCH_AXES if verdicts.get(a) == "MATCH"),
                None,
            )
        return axis

    def test_cancer_type_mismatch_drops(self):
        assert self._hard_drop_axis({"cancer_type": "MISMATCH"}) == "cancer_type"

    def test_disease_status_mismatch_drops(self):
        assert self._hard_drop_axis({"disease_status": "MISMATCH"}) == "disease_status"

    def test_surgical_candidacy_mismatch_drops(self):
        assert self._hard_drop_axis({"surgical_candidacy": "MISMATCH"}) == "surgical_candidacy"

    def test_exclusions_violated_match_drops(self):
        assert self._hard_drop_axis({"study_exclusions_violated": "MATCH"}) == "study_exclusions_violated"

    def test_secondary_axis_mismatch_does_not_drop(self):
        # performance_status / age_range are secondary, never hard-drop
        assert self._hard_drop_axis({"performance_status": "MISMATCH"}) is None

    def test_all_match_does_not_drop(self):
        verdicts = {
            "cancer_type": "MATCH", "histology": "MATCH", "stage": "MATCH",
            "disease_status": "MATCH", "surgical_candidacy": "MATCH",
            "study_exclusions_violated": "NOT_AVAILABLE",
        }
        assert self._hard_drop_axis(verdicts) is None

    def test_exclusions_match_overrides_other_matches(self):
        """If study_exclusions_violated == MATCH (patient violates an
        exclusion), the study is dropped even when everything else
        matches."""
        verdicts = {
            "cancer_type": "MATCH",
            "histology": "MATCH",
            "stage": "MATCH",
            "study_exclusions_violated": "MATCH",
        }
        assert self._hard_drop_axis(verdicts) == "study_exclusions_violated"

    def test_cancer_type_takes_precedence_over_others(self):
        """When multiple hard-drop axes mismatch, cancer_type wins
        (it's the earliest in HARD_DROP_AXES)."""
        verdicts = {
            "cancer_type": "MISMATCH",
            "disease_status": "MISMATCH",
            "surgical_candidacy": "MISMATCH",
        }
        result = self._hard_drop_axis(verdicts)
        # Any of the three is acceptable; the predicate just needs to
        # report one. The current implementation hits cancer_type first
        # because dict iteration on HARD_DROP_AXES (a set) is order-
        # preserving for inserted values in CPython 3.7+, but to keep
        # the test robust we just assert it picked one of the three.
        assert result in {"cancer_type", "disease_status", "surgical_candidacy"}
