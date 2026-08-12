"""
Property-based test for secondary-boost ranking exploration — Studies with
more secondary matches should rank higher.

Feature: patient-study-match-scoring-fix
Property 2: Secondary-boost Checking — Studies with more secondary matches
rank higher than studies with fewer secondary matches when core verdicts
are identical.

**Validates: Requirements 1.4, 1.5, 2.4, 2.5**

Pre-work findings
-----------------
- The non-tiered path (``use_tiered_model=False``) in
  ``apply_patient_eligibility_filter_and_boost`` (~line 1364) applies:
      MATCH → +0.25 boost, POSSIBLE → +0.10, NO_MATCH → 0
  Secondary axes are NOT currently scored — two studies with identical
  core verdicts but different secondary match counts get the same score.
- HARD_FILTER_CRITERIA = ["cancer_type", "histology", "stage",
  "prior_therapies", "biomarkers"]

Secondary-boost condition
-------------------------
For two studies X1, X2 with identical core verdicts (all MATCH or
COMPATIBLE, no mismatches), if X1 has MATCH on K1 secondary axes and
X2 has MATCH on K2 secondary axes (K1 > K2), then score(X1) > score(X2).

This test encodes the EXPECTED (fixed) behavior. On unfixed code it MUST
FAIL — failure confirms secondary scoring is missing.
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

#: Core axes — all set to MATCH or COMPATIBLE (no mismatches) for this test.
CORE_AXES = ["cancer_type", "histology", "stage", "prior_therapies", "biomarkers"]

#: Secondary axes from bugfix.md — these are the axes whose MATCH verdicts
#: should provide additive score boosts.
SECONDARY_AXES = [
    "performance_status",
    "age_range",
    "gender",
    "modality",
    "metastatic_sites",
    "comorbidity_compatibility",
    "study_phase",
    "landmark_trial_status",
    "recency",
]

#: Secondary axis weights from bugfix.md (MATCH values).
#: Used to compute the expected score delta between X1 and X2.
SECONDARY_AXIS_WEIGHTS = {
    "performance_status": 8,
    "age_range": 6,
    "modality": 5,
    "metastatic_sites": 5,
    "comorbidity_compatibility": 4,
    "gender": 4,
    "study_phase": 3,
    "landmark_trial_status": 3,
    "recency": 2,
}

#: Non-MISMATCH verdicts for core axes (both studies share these).
CORE_OK_VERDICTS = ["MATCH", "COMPATIBLE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, score: float = 0.5) -> dict:
    """Build a minimal chunk dict for testing."""
    return {"doc_id": doc_id, "score": score, "final_score": score}


def _build_eligibility_result(verdicts: dict) -> dict:
    """Build an eligibility result dict the way the production code does.

    Mirrors the logic in ``check_patient_eligibility_for_studies``:
    has_hard_mismatch is True when ANY active criterion in
    HARD_FILTER_CRITERIA is MISMATCH. We compute it from verdicts,
    exactly as the production code does.
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


def _get_final_score(chunks: list, doc_id: str) -> float:
    """Extract the final_score for a given doc_id from a list of chunks."""
    for c in chunks:
        cid = c.get("doc_id") or c.get("payload", {}).get("doc_id")
        if cid == doc_id:
            return c.get("final_score", c.get("score", 0))
    raise ValueError(f"doc_id {doc_id} not found in chunks")


# ---------------------------------------------------------------------------
# Hypothesis strategy
# ---------------------------------------------------------------------------

@st.composite
def secondary_boost_pair(draw):
    """Generate a pair of studies (X1, X2) for secondary-boost comparison.

    Both studies share identical core-axis verdicts (all MATCH or
    COMPATIBLE, no mismatches). X1 has MATCH on K1 secondary axes,
    X2 has MATCH on K2 secondary axes, with K1 > K2. All non-matching
    secondary axes are NOT_AVAILABLE (neutral).

    Returns (chunk1, chunk2, doc_id1, doc_id2, elig1, elig2,
             k1, k2, extra_match_axes)
    where extra_match_axes are the secondary axes that X1 has MATCH on
    but X2 does not.
    """
    base_id = draw(st.integers(min_value=1, max_value=4999))
    doc_id1 = f"study_{base_id:04d}_x1"
    doc_id2 = f"study_{base_id:04d}_x2"

    # Both studies start with the same base score
    base_score = draw(st.floats(
        min_value=0.3, max_value=0.8,
        allow_nan=False, allow_infinity=False,
    ))

    # Core verdicts: identical for both, all MATCH or COMPATIBLE (no MISMATCH)
    core_verdicts = {}
    for axis in CORE_AXES:
        core_verdicts[axis] = draw(st.sampled_from(CORE_OK_VERDICTS))

    # Pick K2 (fewer secondary matches) and K1 (more secondary matches)
    # K2 can be 0 (no secondary matches), K1 must be > K2
    k2 = draw(st.integers(min_value=0, max_value=len(SECONDARY_AXES) - 1))
    k1 = draw(st.integers(min_value=k2 + 1, max_value=len(SECONDARY_AXES)))

    # Draw which secondary axes get MATCH for X2 (the smaller set)
    x2_match_axes = draw(
        st.sampled_from(
            sorted(
                _combinations(SECONDARY_AXES, k2),
                key=str,
            )
        )
    ) if k2 > 0 else ()

    # X1 gets all of X2's matches PLUS additional ones to reach K1
    remaining_axes = [a for a in SECONDARY_AXES if a not in x2_match_axes]
    extra_count = k1 - k2
    extra_match_axes = draw(
        st.sampled_from(
            sorted(
                _combinations(remaining_axes, extra_count),
                key=str,
            )
        )
    )
    x1_match_axes = set(x2_match_axes) | set(extra_match_axes)

    # Build verdicts for X1
    verdicts1 = dict(core_verdicts)
    for axis in SECONDARY_AXES:
        verdicts1[axis] = "MATCH" if axis in x1_match_axes else "NOT_AVAILABLE"

    # Build verdicts for X2
    verdicts2 = dict(core_verdicts)
    for axis in SECONDARY_AXES:
        verdicts2[axis] = "MATCH" if axis in x2_match_axes else "NOT_AVAILABLE"

    chunk1 = _make_chunk(doc_id1, base_score)
    chunk2 = _make_chunk(doc_id2, base_score)
    elig1 = _build_eligibility_result(verdicts1)
    elig2 = _build_eligibility_result(verdicts2)

    return (chunk1, chunk2, doc_id1, doc_id2, elig1, elig2,
            k1, k2, extra_match_axes)


def _combinations(items, r):
    """Return all r-length combinations of items as tuples."""
    from itertools import combinations
    return list(combinations(items, r))


# ---------------------------------------------------------------------------
# Property 2: Secondary-boost Checking
# ---------------------------------------------------------------------------

@h_settings(max_examples=200)
@given(data=secondary_boost_pair())
def test_secondary_boost_ranking(data):
    """
    Property 2: Secondary-boost Checking — Studies with more secondary
    matches rank higher.

    For any pair of studies (X1, X2) where:
    - Both have identical core-axis verdicts (all MATCH or COMPATIBLE)
    - X1 has MATCH on K1 secondary axes, X2 has MATCH on K2, K1 > K2
    - Non-matching secondary axes are NOT_AVAILABLE

    Expected (fixed) behavior:
    - Both studies are in ``kept`` (no hard-filter)
    - score(X1) > score(X2)

    On UNFIXED code this test MUST FAIL because secondary axes are not
    scored — both studies receive the same boost regardless of secondary
    match count.

    **Validates: Requirements 1.4, 1.5, 2.4, 2.5**
    """
    (chunk1, chunk2, doc_id1, doc_id2, elig1, elig2,
     k1, k2, extra_match_axes) = data

    eligibility_results = {
        doc_id1: elig1,
        doc_id2: elig2,
    }
    chunks = [chunk1, chunk2]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,  # non-tiered path — where the bug lives
    )

    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}

    verdicts1 = elig1["criteria_verdicts"]
    verdicts2 = elig2["criteria_verdicts"]

    # ── Assert: both studies are RETAINED ─────────────────────────────
    assert doc_id1 in kept_ids, (
        f"Study X1 ({doc_id1}) should be RETAINED (core verdicts have no "
        f"MISMATCH). Verdicts: {verdicts1}"
    )
    assert doc_id2 in kept_ids, (
        f"Study X2 ({doc_id2}) should be RETAINED (core verdicts have no "
        f"MISMATCH). Verdicts: {verdicts2}"
    )

    # ── Assert: score(X1) > score(X2) ────────────────────────────────
    score1 = _get_final_score(kept, doc_id1)
    score2 = _get_final_score(kept, doc_id2)

    assert score1 > score2, (
        f"Study X1 with {k1} secondary MATCHes should score higher than "
        f"X2 with {k2} secondary MATCHes, but got score(X1)={score1} vs "
        f"score(X2)={score2}. Extra MATCH axes in X1: {extra_match_axes}. "
        f"Secondary axes are not differentiating rank — secondary scoring "
        f"is missing. X1 verdicts: {verdicts1}, X2 verdicts: {verdicts2}"
    )
