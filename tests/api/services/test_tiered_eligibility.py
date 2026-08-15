"""
Property-based tests for tiered eligibility model.

Feature: patient-study-match-scoring

Tests the following properties:
- Property 9: Cancer type MISMATCH causes hard removal
- Property 10: Secondary mismatch penalty formula
- Property 11: No-mismatch boost preservation

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.api.services.patient_eligibility_boost_service import (
    apply_patient_eligibility_filter_and_boost,
)


# ======================================================================
# Shared constants and strategies
# ======================================================================

#: The five eligibility axes evaluated by PatientEligibility.
ELIGIBILITY_AXES = ["cancer_type", "histology", "stage", "biomarkers", "prior_therapies"]

#: Secondary axes (everything except cancer_type).
SECONDARY_AXES = ["histology", "stage", "biomarkers", "prior_therapies"]

#: Possible verdict values for eligibility axes.
ALL_VERDICTS = ["MATCH", "MISMATCH", "NOT_AVAILABLE", "POSSIBLE"]

#: Non-mismatch verdicts (used for Property 11).
NON_MISMATCH_VERDICTS = ["MATCH", "NOT_AVAILABLE"]

#: Boost-eligible verdicts and their expected boost values.
BOOST_VALUES = {"MATCH": 0.25, "POSSIBLE": 0.10}


def _make_chunk(doc_id: str, score: float = 0.8) -> dict:
    """Build a minimal chunk dict for testing."""
    return {"doc_id": doc_id, "score": score, "final_score": score}


def _make_eligibility_result(
    verdicts: dict,
    has_hard_mismatch: bool = False,
    status: str = "MATCH",
    boost: float = 0.0,
) -> dict:
    """Build a minimal eligibility result dict for testing."""
    return {
        "status": status,
        "criteria_verdicts": dict(verdicts),
        "has_hard_mismatch": has_hard_mismatch,
        "boost": boost,
        "reason": "",
    }


# ======================================================================
# Property 9: Cancer type MISMATCH causes hard removal
# ======================================================================


@st.composite
def cancer_type_mismatch_scenario(draw):
    """
    Generate a random eligibility result where cancer_type is MISMATCH.
    Secondary axes get random verdicts. The study should be hard-removed
    regardless of secondary axis verdicts.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))

    # cancer_type is always MISMATCH
    verdicts = {"cancer_type": "MISMATCH"}

    # Random verdicts for secondary axes
    for axis in SECONDARY_AXES:
        verdicts[axis] = draw(st.sampled_from(ALL_VERDICTS))

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _make_eligibility_result(
        verdicts=verdicts,
        has_hard_mismatch=True,
        status="NO_MATCH",
        boost=0.0,
    )

    return chunk, doc_id, eligibility_result


# Feature: patient-study-match-scoring, Property 9: Cancer type MISMATCH causes hard removal
@settings(max_examples=150)
@given(data=cancer_type_mismatch_scenario())
def test_cancer_type_mismatch_causes_hard_removal(data):
    """
    Property 9: Cancer type MISMATCH causes hard removal.

    For any study where the PatientEligibility cancer_type verdict is
    MISMATCH (with both patient and study cancer types declared), the
    tiered eligibility model SHALL hard-remove that study from the result
    set.

    **Validates: Requirements 4.1**

    Tag: Feature: patient-study-match-scoring, Property 9: Cancer type MISMATCH causes hard removal
    """
    chunk, doc_id, eligibility_result = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=True,
    )

    # The study with cancer_type MISMATCH must be removed
    kept_doc_ids = {
        c.get("doc_id") or c.get("payload", {}).get("doc_id")
        for c in kept
    }
    removed_doc_ids = {
        c.get("doc_id") or c.get("payload", {}).get("doc_id")
        for c in removed
    }

    assert doc_id not in kept_doc_ids, (
        f"Study {doc_id} with cancer_type MISMATCH should be removed from kept, "
        f"but was found in kept list"
    )
    assert doc_id in removed_doc_ids, (
        f"Study {doc_id} with cancer_type MISMATCH should appear in removed list"
    )

    # Verify the removed chunk is annotated as hard-filtered
    removed_chunk = [c for c in removed if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id][0]
    pe = removed_chunk.get("patient_eligibility", {})
    assert pe.get("hard_filtered") is True, (
        f"Removed study should be marked hard_filtered=True, got {pe.get('hard_filtered')}"
    )


# ======================================================================
# Property 10: Secondary mismatch penalty formula
# ======================================================================


@st.composite
def secondary_mismatch_penalty_scenario(draw):
    """
    Generate a random scenario where cancer_type is MATCH (not a hard
    removal) and N secondary axes (0-5) have MISMATCH verdicts.
    The expected penalty is min(N * 0.15, 0.45).
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False))

    # cancer_type is MATCH (no hard removal)
    verdicts = {"cancer_type": "MATCH"}

    # Pick how many secondary axes will be MISMATCH (0-4, since there are 4 secondary axes)
    # But the task says 0-5 for the count range — we cap at 4 since there are only 4 secondary axes
    num_mismatches = draw(st.integers(min_value=0, max_value=len(SECONDARY_AXES)))

    # Randomly choose which secondary axes are MISMATCH
    mismatch_axes = draw(
        st.lists(
            st.sampled_from(SECONDARY_AXES),
            min_size=num_mismatches,
            max_size=num_mismatches,
            unique=True,
        )
    )

    for axis in SECONDARY_AXES:
        if axis in mismatch_axes:
            verdicts[axis] = "MISMATCH"
        else:
            # Non-mismatched secondary axes get a random non-MISMATCH verdict
            verdicts[axis] = draw(st.sampled_from(["MATCH", "NOT_AVAILABLE", "POSSIBLE"]))

    chunk = _make_chunk(doc_id, score)

    # In the tiered model, cancer_type MATCH means no hard mismatch
    eligibility_result = _make_eligibility_result(
        verdicts=verdicts,
        has_hard_mismatch=False,
        status="MATCH" if num_mismatches == 0 else "POSSIBLE",
        boost=0.25 if num_mismatches == 0 else 0.0,
    )

    expected_penalty = min(num_mismatches * 0.15, 0.45)

    return chunk, doc_id, eligibility_result, num_mismatches, expected_penalty


# Feature: patient-study-match-scoring, Property 10: Secondary mismatch penalty formula
@settings(max_examples=150)
@given(data=secondary_mismatch_penalty_scenario())
def test_secondary_mismatch_penalty_formula(data):
    """
    Property 10: Secondary mismatch penalty formula.

    For any study where the cancer_type verdict is MATCH or NOT_AVAILABLE
    and N secondary axes (histology, stage, prior_therapies, biomarkers)
    have MISMATCH verdicts, the tiered eligibility model SHALL retain the
    study with a score penalty of min(N * 0.15, 0.45), and SHALL annotate
    the study with per-axis verdicts and the penalty applied.

    **Validates: Requirements 4.2, 4.3, 4.4, 4.6**

    Tag: Feature: patient-study-match-scoring, Property 10: Secondary mismatch penalty formula
    """
    chunk, doc_id, eligibility_result, num_mismatches, expected_penalty = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=True,
    )

    # Study should be RETAINED (not removed) since cancer_type is MATCH
    kept_doc_ids = {
        c.get("doc_id") or c.get("payload", {}).get("doc_id")
        for c in kept
    }
    assert doc_id in kept_doc_ids, (
        f"Study {doc_id} with cancer_type MATCH should be retained, "
        f"but was not found in kept list (num_secondary_mismatches={num_mismatches})"
    )

    # Find the kept chunk and check its annotation
    kept_chunk = [c for c in kept if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id][0]
    pe = kept_chunk.get("patient_eligibility", {})

    # Verify penalty annotation
    penalty_applied = pe.get("penalty_applied", 0.0)
    assert abs(penalty_applied - expected_penalty) < 1e-9, (
        f"Expected penalty {expected_penalty} for {num_mismatches} secondary mismatches, "
        f"got {penalty_applied}"
    )

    # Verify per-axis verdicts are present in the annotation
    criteria_verdicts = pe.get("criteria_verdicts", {})
    assert "cancer_type" in criteria_verdicts, (
        "Annotation must include cancer_type verdict"
    )
    for axis in SECONDARY_AXES:
        assert axis in criteria_verdicts, (
            f"Annotation must include {axis} verdict"
        )

    # Verify the study is not hard-filtered
    assert pe.get("hard_filtered") is not True, (
        f"Study with cancer_type MATCH should not be hard_filtered"
    )

    # If there are mismatches, verify the score was penalized
    if num_mismatches > 0:
        original_score = chunk["score"]
        final_score = kept_chunk.get("final_score", original_score)
        # The final score should be reduced by the penalty
        assert final_score <= original_score + 0.01, (
            f"Score should not increase when there are secondary mismatches. "
            f"Original: {original_score}, Final: {final_score}, Penalty: {expected_penalty}"
        )


# ======================================================================
# Property 11: No-mismatch boost preservation
# ======================================================================


@st.composite
def no_mismatch_boost_scenario(draw):
    """
    Generate a random eligibility result where ALL axes are MATCH or
    NOT_AVAILABLE (no mismatches at all). The existing boost logic should
    apply and penalty should be 0.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(min_value=0.1, max_value=0.75, allow_nan=False, allow_infinity=False))

    # All axes are MATCH or NOT_AVAILABLE
    verdicts = {}
    for axis in ELIGIBILITY_AXES:
        verdicts[axis] = draw(st.sampled_from(NON_MISMATCH_VERDICTS))

    # Determine the expected boost based on the overall status
    # If any axis is MATCH, the study gets a MATCH boost (+0.25)
    # If all are NOT_AVAILABLE, there's no boost
    has_any_match = any(v == "MATCH" for v in verdicts.values())

    if has_any_match:
        status = "MATCH"
        boost = 0.25
    else:
        # All NOT_AVAILABLE — no boost
        status = "NOT_AVAILABLE"
        boost = 0.0

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _make_eligibility_result(
        verdicts=verdicts,
        has_hard_mismatch=False,
        status=status,
        boost=boost,
    )

    return chunk, doc_id, eligibility_result, boost


# Feature: patient-study-match-scoring, Property 11: No-mismatch boost preservation
@settings(max_examples=150)
@given(data=no_mismatch_boost_scenario())
def test_no_mismatch_boost_preservation(data):
    """
    Property 11: No-mismatch boost preservation.

    For any study where all axes are MATCH or NOT_AVAILABLE, the tiered
    eligibility model SHALL apply the existing boost logic (MATCH: +0.25,
    POSSIBLE: +0.10) without modification and with zero penalty.

    **Validates: Requirements 4.5**

    Tag: Feature: patient-study-match-scoring, Property 11: No-mismatch boost preservation
    """
    chunk, doc_id, eligibility_result, expected_boost = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=True,
    )

    # Study should be retained (no mismatches)
    kept_doc_ids = {
        c.get("doc_id") or c.get("payload", {}).get("doc_id")
        for c in kept
    }
    assert doc_id in kept_doc_ids, (
        f"Study {doc_id} with all MATCH/NOT_AVAILABLE should be retained"
    )

    # Find the kept chunk
    kept_chunk = [c for c in kept if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id][0]
    pe = kept_chunk.get("patient_eligibility", {})

    # Penalty must be 0 (no mismatches)
    penalty_applied = pe.get("penalty_applied", 0.0)
    assert abs(penalty_applied - 0.0) < 1e-9, (
        f"Expected zero penalty for all-MATCH/NOT_AVAILABLE study, got {penalty_applied}"
    )

    # Verify boost was applied correctly
    if expected_boost > 0:
        boost_applied = pe.get("boost_applied", 0.0)
        assert abs(boost_applied - expected_boost) < 1e-9, (
            f"Expected boost {expected_boost}, got {boost_applied}"
        )

        # Verify the final score reflects the boost
        original_score = chunk["score"]
        final_score = kept_chunk.get("final_score", original_score)
        expected_final = min(1.0, original_score + expected_boost)
        assert abs(final_score - expected_final) < 1e-9, (
            f"Expected final_score {expected_final} (original {original_score} + boost {expected_boost}), "
            f"got {final_score}"
        )

    # Verify not hard-filtered
    assert pe.get("hard_filtered") is not True, (
        "Study with no mismatches should not be hard_filtered"
    )
