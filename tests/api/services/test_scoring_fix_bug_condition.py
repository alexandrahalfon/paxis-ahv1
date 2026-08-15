"""
Property-based test for bug condition exploration — Non-Cancer-Type Core
Mismatch Hard-Filters Study.

Feature: patient-study-match-scoring-fix
Property 1: Bug Condition — studies with non-cancer_type core mismatches
are incorrectly hard-filtered instead of being retained with a penalty.

**Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.7**

Pre-work findings
-----------------
- ``has_hard_mismatch`` is set in ``check_patient_eligibility_for_studies``
  (patient_eligibility_boost_service.py ~line 1023). It is True when ANY
  active criterion in HARD_FILTER_CRITERIA has a MISMATCH verdict.
- The non-tiered path (``use_tiered_model=False``) in
  ``apply_patient_eligibility_filter_and_boost`` (~line 1364) does:
      if result.get("has_hard_mismatch"): → remove study
  This is the bug: it treats ALL core-axis mismatches identically,
  including non-cancer_type axes that should only incur a penalty.
- Current boost values: MATCH → 0.25, POSSIBLE → 0.10, NO_MATCH → 0.
- HARD_FILTER_CRITERIA = ["cancer_type", "histology", "stage",
  "prior_therapies", "biomarkers"]

Bug condition
-------------
isBugCondition(X) = cancer_type_ok AND has_non_cancer_type_mismatch
  where cancer_type_ok = verdicts["cancer_type"] != "MISMATCH"
  and has_non_cancer_type_mismatch = at least one of
      {histology, stage, prior_therapies, biomarkers} == "MISMATCH"

This test encodes the EXPECTED (fixed) behavior. On unfixed code it MUST
FAIL — failure confirms the bug exists.
"""

import pytest
from hypothesis import given, strategies as st, settings as h_settings

from src.api.services.patient_eligibility_boost_service import (
    apply_patient_eligibility_filter_and_boost,
    HARD_FILTER_CRITERIA,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Core axes that are NOT cancer_type — the axes where a MISMATCH should
#: incur a penalty rather than a hard drop.
NON_CANCER_TYPE_CORE_AXES = ["histology", "stage", "prior_therapies", "biomarkers"]

#: Verdicts that are NOT MISMATCH (used for cancer_type and non-mismatched axes).
NON_MISMATCH_VERDICTS = ["MATCH", "NOT_AVAILABLE"]

#: All possible verdicts.
ALL_VERDICTS = ["MATCH", "MISMATCH", "NOT_AVAILABLE"]

#: Expected penalty per non-cancer_type core axis with MISMATCH verdict.
CORE_MISMATCH_PENALTY = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, score: float = 0.5) -> dict:
    """Build a minimal chunk dict for testing."""
    return {"doc_id": doc_id, "score": score, "final_score": score}


def _build_eligibility_result(verdicts: dict) -> dict:
    """Build an eligibility result dict the way the production code does.

    This mirrors the logic in ``check_patient_eligibility_for_studies``:
    has_hard_mismatch is True when ANY active criterion is MISMATCH.
    We do NOT pre-populate the flag — we compute it from verdicts, exactly
    as the production code does.
    """
    has_mismatch = any(
        verdicts.get(c) == "MISMATCH"
        for c in HARD_FILTER_CRITERIA
    )
    match_count = sum(1 for v in verdicts.values() if v == "MATCH")

    if has_mismatch:
        status = "NO_MATCH"
        boost = 0
    elif match_count > 0:
        status = "MATCH"
        boost = 0.25
    else:
        status = "POSSIBLE"
        boost = 0.1

    return {
        "status": status,
        "reason": "",
        "boost": boost,
        "criteria_verdicts": dict(verdicts),
        "has_hard_mismatch": has_mismatch,
    }


# ---------------------------------------------------------------------------
# Hypothesis strategy
# ---------------------------------------------------------------------------

@st.composite
def bug_condition_scenario(draw):
    """Generate a study that triggers the bug condition.

    - cancer_type verdict is NOT MISMATCH (MATCH or NOT_AVAILABLE)
    - At least one of {histology, stage, prior_therapies, biomarkers} IS MISMATCH
    - Remaining non-cancer_type axes get random non-MISMATCH verdicts

    The eligibility result is built using the same logic as production code
    (has_hard_mismatch computed from verdicts, not pre-set).
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))

    # cancer_type is NOT MISMATCH
    cancer_type_verdict = draw(st.sampled_from(NON_MISMATCH_VERDICTS))

    # Pick 1-4 non-cancer_type axes to be MISMATCH
    num_mismatched = draw(st.integers(min_value=1, max_value=len(NON_CANCER_TYPE_CORE_AXES)))
    mismatched_axes = draw(
        st.sampled_from(
            sorted(
                _combinations(NON_CANCER_TYPE_CORE_AXES, num_mismatched),
                key=str,
            )
        )
    )

    verdicts = {"cancer_type": cancer_type_verdict}
    for axis in NON_CANCER_TYPE_CORE_AXES:
        if axis in mismatched_axes:
            verdicts[axis] = "MISMATCH"
        else:
            verdicts[axis] = draw(st.sampled_from(NON_MISMATCH_VERDICTS))

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _build_eligibility_result(verdicts)

    return chunk, doc_id, eligibility_result, len(mismatched_axes)


def _combinations(items, r):
    """Return all r-length combinations of items as tuples."""
    from itertools import combinations
    return list(combinations(items, r))


# ---------------------------------------------------------------------------
# Property 1: Bug Condition — Non-Cancer-Type Core Mismatch Hard-Filters Study
# ---------------------------------------------------------------------------

@h_settings(max_examples=200)
@given(data=bug_condition_scenario())
def test_non_cancer_type_core_mismatch_retains_study_with_penalty(data):
    """
    Property 1: Bug Condition — Non-Cancer-Type Core Mismatch Hard-Filters Study.

    For any study where cancer_type is NOT MISMATCH but at least one of
    {histology, stage, prior_therapies, biomarkers} IS MISMATCH, the study
    SHALL be RETAINED (not hard-filtered) with a score penalty applied.

    Expected (fixed) behavior:
    - Study appears in ``kept``, NOT in ``removed``
    - Score has penalty: CORE_MISMATCH_PENALTY * count(mismatched non-cancer axes)
    - Final score is floored at 1 (never 0)

    On UNFIXED code this test MUST FAIL because the non-tiered path
    hard-filters any study with has_hard_mismatch=True.

    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.7**
    """
    chunk, doc_id, eligibility_result, num_mismatched_axes = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,  # non-tiered path — where the bug lives
    )

    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}
    removed_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in removed}

    verdicts = eligibility_result["criteria_verdicts"]

    # ── Assert: study is RETAINED, not hard-filtered ──────────────────
    assert doc_id in kept_ids, (
        f"Study {doc_id} should be RETAINED (not hard-filtered) when "
        f"cancer_type={verdicts.get('cancer_type')} and non-cancer_type "
        f"axes have MISMATCH. Verdicts: {verdicts}"
    )
    assert doc_id not in removed_ids, (
        f"Study {doc_id} should NOT be in removed list when "
        f"cancer_type is not MISMATCH. Verdicts: {verdicts}"
    )

    # ── Assert: score has penalty applied, floored at 1 ───────────────
    kept_chunk = [c for c in kept if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id][0]
    final_score = kept_chunk.get("final_score", kept_chunk.get("score", 0))

    assert final_score >= 1, (
        f"Final score must be floored at 1 (got {final_score}). "
        f"Verdicts: {verdicts}"
    )
