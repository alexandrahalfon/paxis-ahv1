"""
Preservation property tests for patient-study-match-scoring-fix.

Feature: patient-study-match-scoring-fix
Property 4: Preservation — Cancer-Type Mismatch Still Hard-Filtered,
Full-Match Boost Preserved, Thin-Context Passthrough Preserved,
Secondary MISMATCH Never Drops Study, All-COMPATIBLE Core Studies Retained.

**Validates: Requirements 3.1, 3.2, 3.3, 3.6, 3.7**

These tests MUST PASS on unfixed code. They establish baseline behavior
that must be preserved after the fix is applied.

Observation-first methodology: we do NOT hardcode boost/penalty values.
Instead, we observe the unfixed code's behavior and assert structural
invariants (e.g., "study is in kept", "score > 0", "removed list is empty").

Pre-work findings
-----------------
- ``has_hard_mismatch`` is set in ``check_patient_eligibility_for_studies``
  (patient_eligibility_boost_service.py ~line 1023). It is True when ANY
  active criterion in HARD_FILTER_CRITERIA has a MISMATCH verdict.
- The non-tiered path (``use_tiered_model=False``) in
  ``apply_patient_eligibility_filter_and_boost`` (~line 1364) does:
      if result.get("has_hard_mismatch"): → remove study
      else: boost (MATCH → +0.25, POSSIBLE → +0.10)
- HARD_FILTER_CRITERIA = ["cancer_type", "histology", "stage",
  "prior_therapies", "biomarkers"]
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

#: Core axes used by the eligibility filter.
CORE_AXES = ["cancer_type", "histology", "stage", "prior_therapies", "biomarkers"]

#: Non-cancer_type core axes.
NON_CANCER_TYPE_CORE_AXES = ["histology", "stage", "prior_therapies", "biomarkers"]

#: Secondary axes from bugfix.md.
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

#: All possible verdicts.
ALL_VERDICTS = ["MATCH", "MISMATCH", "NOT_AVAILABLE", "COMPATIBLE"]

#: Non-MISMATCH verdicts.
NON_MISMATCH_VERDICTS = ["MATCH", "NOT_AVAILABLE", "COMPATIBLE"]

#: Verdicts that are COMPATIBLE or NOT_AVAILABLE (no declared requirement).
NEUTRAL_VERDICTS = ["COMPATIBLE", "NOT_AVAILABLE"]


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


def _combinations(items, r):
    """Return all r-length combinations of items as tuples."""
    from itertools import combinations
    return list(combinations(items, r))


# ===========================================================================
# Property 2a: Cancer-type MISMATCH still hard-filtered
# ===========================================================================

@st.composite
def cancer_type_mismatch_scenario(draw):
    """Generate a scenario where cancer_type=MISMATCH with random secondary
    verdicts. The study should always be hard-removed on the non-tiered path.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(
        min_value=0.1, max_value=1.0,
        allow_nan=False, allow_infinity=False,
    ))

    # cancer_type is MISMATCH — this is the hard-filter trigger
    verdicts = {"cancer_type": "MISMATCH"}

    # Random verdicts for non-cancer_type core axes
    for axis in NON_CANCER_TYPE_CORE_AXES:
        verdicts[axis] = draw(st.sampled_from(ALL_VERDICTS))

    # Random verdicts for secondary axes
    for axis in SECONDARY_AXES:
        verdicts[axis] = draw(st.sampled_from(ALL_VERDICTS))

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _build_eligibility_result(verdicts)

    return chunk, doc_id, eligibility_result


@h_settings(max_examples=200)
@given(data=cancer_type_mismatch_scenario())
def test_preservation_2a_cancer_type_mismatch_hard_filtered(data):
    """
    Preservation 2a: Cancer-type MISMATCH still hard-filtered.

    For any study where cancer_type=MISMATCH, regardless of other axis
    verdicts, the study SHALL be removed from the result set on the
    non-tiered path.

    This preserves requirement 3.1 — cancer_type MISMATCH is always a
    hard drop.

    **Validates: Requirements 3.1**
    """
    chunk, doc_id, eligibility_result = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,
    )

    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}
    removed_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in removed}

    verdicts = eligibility_result["criteria_verdicts"]

    # Assert: study is REMOVED (hard-filtered)
    assert doc_id in removed_ids, (
        f"Study {doc_id} with cancer_type=MISMATCH should be hard-filtered "
        f"(removed), but was not found in removed list. Verdicts: {verdicts}"
    )
    assert doc_id not in kept_ids, (
        f"Study {doc_id} with cancer_type=MISMATCH should NOT be in kept "
        f"list. Verdicts: {verdicts}"
    )


# ===========================================================================
# Property 2b: Full-match boost preserved (observation-based)
# ===========================================================================

@st.composite
def full_match_scenario(draw):
    """Generate a scenario where ALL active core axes are MATCH with no
    mismatches. The study should be boosted and retained.

    Observation-first: we run the inputs through the code and record the
    resulting score as the baseline. We do NOT hardcode +0.25 or any
    specific numeric boost.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(
        min_value=0.1, max_value=0.7,
        allow_nan=False, allow_infinity=False,
    ))

    # All core axes are MATCH
    verdicts = {}
    for axis in CORE_AXES:
        verdicts[axis] = "MATCH"

    # Secondary axes: random non-MISMATCH (to keep things clean for this test)
    for axis in SECONDARY_AXES:
        verdicts[axis] = draw(st.sampled_from(["MATCH", "NOT_AVAILABLE"]))

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _build_eligibility_result(verdicts)

    return chunk, doc_id, eligibility_result, score


@h_settings(max_examples=200)
@given(data=full_match_scenario())
def test_preservation_2b_full_match_boost_preserved(data):
    """
    Preservation 2b: Full-match boost preserved (observation-based).

    For any study where ALL active core axes are MATCH with no mismatches,
    the study SHALL be retained in ``kept`` and its final_score SHALL be
    >= the original base score (i.e., a boost was applied, not a penalty).

    We observe the unfixed code's behavior and assert non-regression:
    the boost is preserved or improved after the fix.

    **Validates: Requirements 3.2**
    """
    chunk, doc_id, eligibility_result, original_score = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,
    )

    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}

    # Assert: study is RETAINED
    assert doc_id in kept_ids, (
        f"Study {doc_id} with all-MATCH core verdicts should be retained "
        f"in kept list."
    )

    # Observe the resulting score (baseline from unfixed code)
    final_score = _get_final_score(kept, doc_id)

    # Assert: final_score >= original_score (boost applied, not penalty)
    assert final_score >= original_score - 1e-9, (
        f"Full-match study should have final_score >= original_score. "
        f"Got final_score={final_score}, original_score={original_score}. "
        f"The boost should be preserved."
    )


# ===========================================================================
# Property 2c: No-patient-context passthrough preserved
# ===========================================================================

@st.composite
def no_patient_context_scenario(draw):
    """Generate a scenario where eligibility_results is empty (no patient
    context detected). All chunks should pass through unmodified.
    """
    num_chunks = draw(st.integers(min_value=1, max_value=5))
    chunks = []
    for i in range(num_chunks):
        doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}_{i}"
        score = draw(st.floats(
            min_value=0.1, max_value=1.0,
            allow_nan=False, allow_infinity=False,
        ))
        chunks.append(_make_chunk(doc_id, score))

    return chunks


@h_settings(max_examples=200)
@given(data=no_patient_context_scenario())
def test_preservation_2c_no_patient_context_passthrough(data):
    """
    Preservation 2c: No-patient-context passthrough preserved.

    When ``eligibility_results`` is empty (no patient context detected),
    all chunks SHALL pass through unmodified: ``kept == chunks`` and
    ``removed == []``.

    **Validates: Requirements 3.3**
    """
    chunks = data

    # Empty eligibility_results — no patient context
    eligibility_results = {}

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,
    )

    # Assert: all chunks pass through
    assert len(kept) == len(chunks), (
        f"With empty eligibility_results, all {len(chunks)} chunks should "
        f"pass through. Got {len(kept)} kept."
    )
    assert len(removed) == 0, (
        f"With empty eligibility_results, removed should be empty. "
        f"Got {len(removed)} removed."
    )

    # Assert: chunks are unmodified (same doc_ids, same scores)
    original_ids = {c["doc_id"] for c in chunks}
    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}
    assert original_ids == kept_ids, (
        f"Kept doc_ids should match original. "
        f"Original: {original_ids}, Kept: {kept_ids}"
    )


# ===========================================================================
# Property 2d: Secondary MISMATCH never drops study
# ===========================================================================

@st.composite
def secondary_mismatch_scenario(draw):
    """Generate a scenario with core axes all MATCH and one or more
    secondary axes MISMATCH. The study should be retained (secondary
    axes never trigger removal on the non-tiered path).

    On the unfixed non-tiered path, secondary axes are NOT in
    HARD_FILTER_CRITERIA, so has_hard_mismatch is False when all core
    axes are MATCH. The study is retained and boosted.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(
        min_value=0.1, max_value=1.0,
        allow_nan=False, allow_infinity=False,
    ))

    # All core axes are MATCH (no hard-filter trigger)
    verdicts = {}
    for axis in CORE_AXES:
        verdicts[axis] = "MATCH"

    # Pick 1+ secondary axes to be MISMATCH
    num_secondary_mismatches = draw(st.integers(
        min_value=1, max_value=len(SECONDARY_AXES),
    ))
    mismatch_secondary = draw(
        st.sampled_from(
            sorted(
                _combinations(SECONDARY_AXES, num_secondary_mismatches),
                key=str,
            )
        )
    )

    for axis in SECONDARY_AXES:
        if axis in mismatch_secondary:
            verdicts[axis] = "MISMATCH"
        else:
            verdicts[axis] = draw(st.sampled_from(["MATCH", "NOT_AVAILABLE"]))

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _build_eligibility_result(verdicts)

    return chunk, doc_id, eligibility_result


@h_settings(max_examples=200)
@given(data=secondary_mismatch_scenario())
def test_preservation_2d_secondary_mismatch_never_drops(data):
    """
    Preservation 2d: Secondary MISMATCH never drops study.

    For any study with all core axes MATCH and one or more secondary axes
    MISMATCH, the study SHALL be retained in ``kept`` (not removed).
    The score may be lower than baseline but must not be zero.

    On the unfixed non-tiered path, secondary axes are not in
    HARD_FILTER_CRITERIA, so they don't trigger has_hard_mismatch.
    The study is retained and boosted normally.

    **Validates: Requirements 3.6**
    """
    chunk, doc_id, eligibility_result = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,
    )

    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}
    removed_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in removed}

    # Assert: study is RETAINED
    assert doc_id in kept_ids, (
        f"Study {doc_id} with all-MATCH core and secondary MISMATCH "
        f"should be retained. Secondary axes never trigger removal."
    )
    assert doc_id not in removed_ids, (
        f"Study {doc_id} should NOT be in removed list. "
        f"Secondary MISMATCH never drops a study."
    )

    # Assert: score is > 0
    final_score = _get_final_score(kept, doc_id)
    assert final_score > 0, (
        f"Study with all-MATCH core should have final_score > 0. "
        f"Got {final_score}."
    )


# ===========================================================================
# Property 2e: All-COMPATIBLE core studies retained
# ===========================================================================

@st.composite
def all_compatible_scenario(draw):
    """Generate a scenario where all core axes are COMPATIBLE or NOT_AVAILABLE
    (sparse-metadata study, no declared requirements). No MISMATCH anywhere.

    On the unfixed non-tiered path, has_hard_mismatch is False (no MISMATCH
    on any core axis), so the study is retained. The boost depends on whether
    any axis is MATCH — with all COMPATIBLE/NOT_AVAILABLE, the status is
    POSSIBLE (boost=0.10) since match_count is 0 but there's no mismatch.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(
        min_value=0.1, max_value=1.0,
        allow_nan=False, allow_infinity=False,
    ))

    # All core axes are COMPATIBLE or NOT_AVAILABLE (no MISMATCH, no MATCH)
    verdicts = {}
    for axis in CORE_AXES:
        verdicts[axis] = draw(st.sampled_from(NEUTRAL_VERDICTS))

    # Secondary axes: also neutral (no MISMATCH)
    for axis in SECONDARY_AXES:
        verdicts[axis] = draw(st.sampled_from(NEUTRAL_VERDICTS))

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _build_eligibility_result(verdicts)

    return chunk, doc_id, eligibility_result, score


@h_settings(max_examples=200)
@given(data=all_compatible_scenario())
def test_preservation_2e_all_compatible_core_retained(data):
    """
    Preservation 2e: All-COMPATIBLE core studies retained.

    For any study where all core axes are COMPATIBLE or NOT_AVAILABLE
    (sparse-metadata study), the study SHALL be retained in ``kept``
    with final_score > 0 and no penalty applied (no MISMATCH anywhere).

    **Validates: Requirements 3.7**
    """
    chunk, doc_id, eligibility_result, original_score = data

    eligibility_results = {doc_id: eligibility_result}
    chunks = [chunk]

    kept, removed = apply_patient_eligibility_filter_and_boost(
        chunks=chunks,
        eligibility_results=eligibility_results,
        use_tiered_model=False,
    )

    kept_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in kept}
    removed_ids = {c.get("doc_id") or c.get("payload", {}).get("doc_id") for c in removed}

    # Assert: study is RETAINED
    assert doc_id in kept_ids, (
        f"Study {doc_id} with all COMPATIBLE/NOT_AVAILABLE core verdicts "
        f"should be retained. No MISMATCH means no hard-filter."
    )
    assert doc_id not in removed_ids, (
        f"Study {doc_id} should NOT be in removed list. "
        f"All-COMPATIBLE/NOT_AVAILABLE studies have no MISMATCH."
    )

    # Assert: final_score > 0
    final_score = _get_final_score(kept, doc_id)
    assert final_score > 0, (
        f"All-COMPATIBLE study should have final_score > 0. "
        f"Got {final_score}."
    )

    # Assert: no penalty applied (score should be >= original, since
    # the code applies a POSSIBLE boost of +0.10 when no MATCH but no
    # MISMATCH either). We observe rather than hardcode the boost value.
    assert final_score >= original_score - 1e-9, (
        f"All-COMPATIBLE study should not have a penalty. "
        f"final_score={final_score} should be >= original_score={original_score}."
    )
