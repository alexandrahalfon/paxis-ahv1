"""
Property-based test for biomarkers-when-declared exploration — Silent studies
don't trigger biomarker penalty.

Feature: patient-study-match-scoring-fix
Property 3: Biomarkers-when-declared Checking — When a study has no declared
biomarker requirement, the biomarkers axis verdict should be NOT_AVAILABLE
regardless of the patient's biomarker profile.

**Validates: Requirements 1.6, 2.6**

Pre-work findings
-----------------
- ``has_hard_mismatch`` is set in ``check_patient_eligibility_for_studies``
  (patient_eligibility_boost_service.py ~line 1023). It is True when ANY
  active criterion in HARD_FILTER_CRITERIA has a MISMATCH verdict.
- The non-tiered path (``use_tiered_model=False``) in
  ``apply_patient_eligibility_filter_and_boost`` (~line 1364) does:
      if result.get("has_hard_mismatch"): → remove study
  This is the bug: it treats ALL core-axis mismatches identically,
  including biomarker mismatches on studies that are silent on biomarkers.
- Current boost values: MATCH → 0.25, POSSIBLE → 0.10, NO_MATCH → 0.
- HARD_FILTER_CRITERIA = ["cancer_type", "histology", "stage",
  "prior_therapies", "biomarkers"]

Biomarkers-when-declared condition
----------------------------------
When a study has NO declared biomarker requirement (study metadata is silent
on biomarkers) AND the patient has one or more declared biomarkers, the
biomarkers axis verdict should be NOT_AVAILABLE (not MISMATCH, not MATCH).
The biomarkers axis should contribute 0 to the final score (no penalty, no
boost), and the study should be retained in ``kept``.

On UNFIXED code, the system may produce MISMATCH on the biomarkers axis
when the patient has a biomarker the study doesn't mention, because the
eligibility layer does not distinguish between "study declared a biomarker
requirement that contradicts" and "study is silent on biomarkers". This
causes the study to be hard-filtered via ``has_hard_mismatch=True``.

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

#: Patient biomarker examples — realistic oncology biomarkers.
PATIENT_BIOMARKERS = [
    "BRCA2-mutant",
    "EGFR exon 19 del",
    "HER2-positive",
    "PD-L1 high (CPS >= 10)",
    "KRAS G12C",
    "ALK-rearranged",
    "BRAF V600E",
    "MSI-high",
    "TMB-high",
    "ER-positive",
]

#: Non-MISMATCH verdicts for core axes (used for axes other than biomarkers).
NON_MISMATCH_VERDICTS = ["MATCH", "NOT_AVAILABLE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, score: float = 0.5) -> dict:
    """Build a minimal chunk dict for testing."""
    return {"doc_id": doc_id, "score": score, "final_score": score}


def _build_eligibility_result_silent_biomarker(verdicts: dict) -> dict:
    """Build an eligibility result for a study that is SILENT on biomarkers.

    This simulates what the UNFIXED production code produces: when the
    patient has biomarkers and the study is silent, the LLM may return
    MISMATCH on the biomarkers axis (or the post-processing may treat
    it as active). The unfixed code computes has_hard_mismatch from ALL
    active criteria including biomarkers, even when the study didn't
    declare a biomarker requirement.

    We build the result exactly as the production code does — computing
    has_hard_mismatch from verdicts. On unfixed code, if biomarkers is
    MISMATCH, has_hard_mismatch will be True, causing hard-filter.
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
        "has_declared_biomarker_requirement": False,
    }


def _build_eligibility_result_no_biomarker(verdicts: dict) -> dict:
    """Build an eligibility result for a baseline study (no patient biomarker).

    Same verdicts as the silent-biomarker study, but biomarkers is
    NOT_AVAILABLE (as it would be when the patient has no biomarkers).
    Used to compute the expected baseline score for comparison.
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
def biomarker_silent_study_scenario(draw):
    """Generate a scenario where a study is silent on biomarkers but the
    patient has declared biomarkers.

    - Study has no declared biomarker requirement (simulated by the fact
      that the study metadata is silent on biomarkers)
    - Patient has one or more declared biomarkers
    - All core axes except biomarkers are MATCH (best case for the study)
    - The biomarkers verdict is set to MISMATCH to simulate what the
      UNFIXED code produces when the patient has biomarkers and the study
      is silent (the LLM or post-processing may incorrectly flag this)

    Returns:
        (chunk, doc_id, eligibility_result_with_biomarker_mismatch,
         baseline_verdicts, patient_biomarkers_list)
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(
        min_value=0.3, max_value=0.9,
        allow_nan=False, allow_infinity=False,
    ))

    # Patient has 1-3 biomarkers
    num_biomarkers = draw(st.integers(min_value=1, max_value=3))
    patient_biomarkers_list = draw(
        st.lists(
            st.sampled_from(PATIENT_BIOMARKERS),
            min_size=num_biomarkers,
            max_size=num_biomarkers,
            unique=True,
        )
    )

    # All core axes except biomarkers are MATCH
    core_axes_no_bio = ["cancer_type", "histology", "stage", "prior_therapies"]
    verdicts = {}
    for axis in core_axes_no_bio:
        verdicts[axis] = "MATCH"

    # The bug: on unfixed code, biomarkers gets MISMATCH even though
    # the study is silent on biomarkers. This is the scenario we're testing.
    verdicts["biomarkers"] = "MISMATCH"

    # Build the baseline verdicts (what the score SHOULD look like when
    # biomarkers is correctly NOT_AVAILABLE)
    baseline_verdicts = dict(verdicts)
    baseline_verdicts["biomarkers"] = "NOT_AVAILABLE"

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _build_eligibility_result_silent_biomarker(verdicts)

    return (chunk, doc_id, eligibility_result, baseline_verdicts,
            patient_biomarkers_list)


# ---------------------------------------------------------------------------
# Property 3: Biomarkers-when-declared Checking
# ---------------------------------------------------------------------------

@h_settings(max_examples=200)
@given(data=biomarker_silent_study_scenario())
def test_silent_biomarker_study_not_penalized(data):
    """
    Property 3: Biomarkers-when-declared Checking — Silent studies don't
    trigger biomarker penalty.

    For any study where:
    - The study has no declared biomarker requirement (study is silent)
    - The patient has one or more declared biomarkers
    - All core axes except biomarkers are MATCH

    Expected (fixed) behavior:
    - The biomarkers verdict should be NOT_AVAILABLE (not MISMATCH)
    - The biomarkers axis contributes 0 to the final score
    - The study is retained in ``kept`` (not hard-filtered)
    - The study's score equals the score a study with identical verdicts
      but biomarkers=NOT_AVAILABLE would receive

    On UNFIXED code this test MUST FAIL because:
    - The eligibility layer sets biomarkers=MISMATCH when the patient has
      biomarkers and the study is silent
    - has_hard_mismatch becomes True (any MISMATCH on active criteria)
    - The non-tiered path hard-filters the study entirely

    **Validates: Requirements 1.6, 2.6**
    """
    (chunk, doc_id, eligibility_result, baseline_verdicts,
     patient_biomarkers_list) = data

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
        f"Study {doc_id} should be RETAINED when the study is silent on "
        f"biomarkers (no declared biomarker requirement). Patient has "
        f"biomarkers {patient_biomarkers_list} but the study doesn't "
        f"declare any biomarker requirement. Verdicts: {verdicts}. "
        f"The biomarkers axis should be NOT_AVAILABLE, not MISMATCH."
    )
    assert doc_id not in removed_ids, (
        f"Study {doc_id} should NOT be in removed list when the study "
        f"is silent on biomarkers. Patient biomarkers: "
        f"{patient_biomarkers_list}. Verdicts: {verdicts}"
    )

    # ── Assert: biomarkers verdict should be NOT_AVAILABLE ────────────
    # On fixed code, the scoring layer should override the biomarkers
    # verdict to NOT_AVAILABLE when the study has no declared requirement.
    kept_chunk = [
        c for c in kept
        if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id
    ][0]
    chunk_verdicts = kept_chunk.get("patient_eligibility", {}).get(
        "criteria_verdicts", verdicts
    )
    assert chunk_verdicts.get("biomarkers") == "NOT_AVAILABLE", (
        f"Biomarkers verdict should be NOT_AVAILABLE when the study is "
        f"silent on biomarkers, but got '{chunk_verdicts.get('biomarkers')}'. "
        f"Patient biomarkers: {patient_biomarkers_list}. "
        f"Study has no declared biomarker requirement."
    )

    # ── Assert: score equals baseline (no biomarker effect) ───────────
    # Run the same study through with biomarkers=NOT_AVAILABLE to get
    # the baseline score. The silent-biomarker study should score the same.
    baseline_chunk = _make_chunk(f"{doc_id}_baseline", chunk["score"])
    baseline_elig = _build_eligibility_result_no_biomarker(baseline_verdicts)
    baseline_kept, _ = apply_patient_eligibility_filter_and_boost(
        chunks=[baseline_chunk],
        eligibility_results={f"{doc_id}_baseline": baseline_elig},
        use_tiered_model=False,
    )

    if baseline_kept:
        baseline_score = baseline_kept[0].get(
            "final_score", baseline_kept[0].get("score", 0)
        )
        actual_score = kept_chunk.get(
            "final_score", kept_chunk.get("score", 0)
        )
        assert abs(actual_score - baseline_score) < 0.001, (
            f"Silent-biomarker study score ({actual_score}) should equal "
            f"baseline score ({baseline_score}) — the patient's biomarker "
            f"should have no scoring effect on a study that is silent on "
            f"biomarkers. Patient biomarkers: {patient_biomarkers_list}"
        )
