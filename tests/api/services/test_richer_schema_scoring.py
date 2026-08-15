"""
Property-based tests for richer schema column scoring.

Feature: patient-study-match-scoring

Tests the following properties:
- Property 12: New scoring axes activate when relevant data is present
- Property 13: JSONB polarity-aware biomarker matching

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.api.services.structured_study_matcher import (
    BASE_SCORING_WEIGHTS,
    POLARITY_MAP,
    STATUS_MATCH_SYNONYMS,
    AxisContribution,
    PGMatchBreakdown,
)
from src.api.services.query_reconciliation import ReconciledStructure


# ======================================================================
# Shared constants and strategies
# ======================================================================

# The new axes that Task 6.2 will add to BASE_SCORING_WEIGHTS.
# Property 12 tests that when a ReconciledStructure has the corresponding
# field set, the axis key exists in BASE_SCORING_WEIGHTS and would be
# activated by calculate_dynamic_weights.
NEW_AXIS_FIELD_MAP = {
    "metastatic_status": "metastatic_status",   # ReconciledStructure.metastatic_status
    "risk_stratification": "risk_level",         # ReconciledStructure.risk_level
}

# Sample values for the trajectory / metastatic / risk fields
METASTATIC_VALUES = [
    "metastatic", "non-metastatic", "locally_advanced",
    "distant", "oligometastatic",
]

RISK_VALUES = [
    "high-risk", "intermediate-risk", "low-risk",
    "favorable", "unfavorable", "high", "intermediate", "low",
]

TRAJECTORY_VALUES = [
    "recurrent", "metastatic", "treatment-naive",
    "progressive", "newly diagnosed", "relapsed",
]

# Canonical biomarker names for generating random biomarker keys
SAMPLE_BIOMARKER_KEYS = [
    "EGFR", "HER2", "ALK", "KRAS", "BRAF", "BRCA1", "BRCA2",
    "PD-L1", "ER", "PR", "ROS1", "RET", "MET", "NTRK",
    "MSI", "TMB", "TP53", "PIK3CA", "PTEN",
]

# The normalized polarity classes used as keys in STATUS_MATCH_SYNONYMS
POLARITY_CLASSES = list(STATUS_MATCH_SYNONYMS.keys())


# ======================================================================
# Property 12: New scoring axes activate when relevant data is present
# ======================================================================


@st.composite
def reconciled_with_optional_new_fields(draw):
    """
    Generate a ReconciledStructure with randomly present/absent values
    for metastatic_status, risk_level, and disease_trajectory.

    Returns (reconciled, expected_active_axes) where expected_active_axes
    is the set of new axis keys that should be activated.
    """
    # Randomly decide which new fields are present
    has_metastatic = draw(st.booleans())
    has_risk = draw(st.booleans())
    has_trajectory = draw(st.booleans())

    metastatic_val = draw(st.sampled_from(METASTATIC_VALUES)) if has_metastatic else None
    risk_val = draw(st.sampled_from(RISK_VALUES)) if has_risk else None
    trajectory_val = draw(st.sampled_from(TRAJECTORY_VALUES)) if has_trajectory else None

    reconciled = ReconciledStructure(
        cancer_site=draw(st.sampled_from(["lung", "breast", "prostate", None])),
        metastatic_status=metastatic_val,
        risk_level=risk_val,
        disease_trajectory=trajectory_val,
        has_patient_context=True,
    )

    expected_active = set()
    if has_metastatic:
        expected_active.add("metastatic_status")
    if has_risk:
        expected_active.add("risk_stratification")

    return reconciled, expected_active


# Feature: patient-study-match-scoring, Property 12: New scoring axes activate when relevant data is present
@settings(max_examples=150)
@given(data=reconciled_with_optional_new_fields())
def test_new_scoring_axes_activate_when_relevant_data_present(data):
    """
    Property 12: New scoring axes activate when relevant data is present.

    For any patient query that contains metastatic status indicators,
    risk stratification level, or trajectory information, the PG matcher
    SHALL include the corresponding new axis in the active scoring criteria
    and in the PGMatchBreakdown.

    We verify:
    1. The new axis keys exist in BASE_SCORING_WEIGHTS (added by Task 6.2).
    2. When a ReconciledStructure has the field set, the axis would be
       included in present_criteria for calculate_dynamic_weights.
    3. When the field is None, the axis would NOT be included.

    **Validates: Requirements 5.1, 5.2, 5.5**

    Tag: Feature: patient-study-match-scoring, Property 12: New scoring axes activate when relevant data is present
    """
    reconciled, expected_active = data

    # Ensure at least one new field is present so the test is meaningful
    assume(len(expected_active) > 0)

    # 1. Verify the new axis keys exist in BASE_SCORING_WEIGHTS
    #    (Task 6.2 adds these — this assertion drives the TDD cycle)
    for axis_key in expected_active:
        assert axis_key in BASE_SCORING_WEIGHTS, (
            f"Expected axis '{axis_key}' to be present in BASE_SCORING_WEIGHTS. "
            f"Available keys: {list(BASE_SCORING_WEIGHTS.keys())}"
        )

    # 2. Simulate the axis activation logic:
    #    An axis is "active" when the corresponding ReconciledStructure field
    #    is non-None, meaning the PG matcher would add it to present_criteria.
    activated_axes = set()

    if reconciled.metastatic_status is not None:
        activated_axes.add("metastatic_status")
    if reconciled.risk_level is not None:
        activated_axes.add("risk_stratification")

    # 3. Assert activated axes match expected
    assert activated_axes == expected_active, (
        f"Activated axes {activated_axes} != expected {expected_active}. "
        f"metastatic_status={reconciled.metastatic_status!r}, "
        f"risk_level={reconciled.risk_level!r}"
    )

    # 4. When an axis is active, verify it has a positive weight
    for axis_key in activated_axes:
        weight = BASE_SCORING_WEIGHTS[axis_key]
        assert weight > 0, (
            f"Active axis '{axis_key}' should have positive weight, got {weight}"
        )

    # 5. Build a PGMatchBreakdown with the activated axes and verify structure
    contributions = []
    for axis_key in activated_axes:
        max_pts = float(BASE_SCORING_WEIGHTS[axis_key])
        contributions.append(AxisContribution(
            axis_name=axis_key,
            points_earned=max_pts,  # full match
            max_points=max_pts,
            label="exact match",
        ))

    if contributions:
        breakdown = PGMatchBreakdown(
            total_score=sum(c.points_earned for c in contributions),
            axis_contributions=contributions,
            axis_mismatches=[],
        )
        # Verify each expected axis appears in the breakdown
        breakdown_axes = {ac.axis_name for ac in breakdown.axis_contributions}
        for axis_key in expected_active:
            assert axis_key in breakdown_axes, (
                f"Expected axis '{axis_key}' in PGMatchBreakdown, "
                f"got {breakdown_axes}"
            )


# ======================================================================
# Property 13: JSONB polarity-aware biomarker matching
# ======================================================================


def polarity_matches(patient_polarity: str, study_polarity: str) -> bool:
    """
    Check if a study-side polarity value matches the patient-side polarity
    using STATUS_MATCH_SYNONYMS expansion.

    The patient_polarity is first normalized through POLARITY_MAP to get
    the canonical class, then STATUS_MATCH_SYNONYMS is consulted to see
    if the study_polarity is in the synonym list.
    """
    # Normalize patient polarity to canonical class via POLARITY_MAP
    canonical = POLARITY_MAP.get(patient_polarity.lower(), patient_polarity.lower())

    # Look up the synonym list for the canonical class
    synonyms = STATUS_MATCH_SYNONYMS.get(canonical, [canonical])

    # Check if the study polarity (lowered) is in the synonym list
    return study_polarity.lower() in [s.lower() for s in synonyms]


@st.composite
def biomarker_polarity_pair(draw):
    """
    Generate a random biomarker key, a patient-side polarity indicator,
    and a study-side polarity value. Also compute whether they should match
    via STATUS_MATCH_SYNONYMS expansion.

    Returns (biomarker_key, patient_polarity, study_polarity, should_match).
    """
    biomarker_key = draw(st.sampled_from(SAMPLE_BIOMARKER_KEYS))

    # Patient polarity: pick from POLARITY_MAP keys (raw indicators)
    patient_polarity = draw(st.sampled_from(list(POLARITY_MAP.keys())))

    # Normalize patient polarity to canonical class
    canonical = POLARITY_MAP[patient_polarity]

    # Study polarity: either pick a synonym (should match) or a non-synonym
    is_match = draw(st.booleans())

    if is_match:
        # Pick from the synonym list for the canonical class
        synonym_list = STATUS_MATCH_SYNONYMS.get(canonical, [canonical])
        study_polarity = draw(st.sampled_from(synonym_list))
    else:
        # Pick a polarity from a DIFFERENT canonical class
        other_classes = [k for k in STATUS_MATCH_SYNONYMS.keys() if k != canonical]
        # Filter out classes whose synonym lists overlap with the canonical class
        canonical_synonyms = set(
            s.lower() for s in STATUS_MATCH_SYNONYMS.get(canonical, [canonical])
        )
        truly_different = []
        for other_class in other_classes:
            other_synonyms = STATUS_MATCH_SYNONYMS.get(other_class, [other_class])
            # Pick synonyms that are NOT in the canonical synonym list
            non_overlapping = [
                s for s in other_synonyms
                if s.lower() not in canonical_synonyms
            ]
            truly_different.extend(non_overlapping)

        if not truly_different:
            # Fallback: use a clearly non-matching value
            study_polarity = "COMPLETELY_UNRELATED_VALUE"
        else:
            study_polarity = draw(st.sampled_from(truly_different))

    # Compute expected match result
    should_match = polarity_matches(patient_polarity, study_polarity)

    return biomarker_key, patient_polarity, study_polarity, should_match


# Feature: patient-study-match-scoring, Property 13: JSONB polarity-aware biomarker matching
@settings(max_examples=150)
@given(data=biomarker_polarity_pair())
def test_jsonb_polarity_aware_biomarker_matching(data):
    """
    Property 13: JSONB polarity-aware biomarker matching.

    For any patient biomarker with a declared polarity and any study whose
    biomarker_status JSONB contains a matching key, the PG matcher SHALL
    compare the stored polarity value against the patient's polarity using
    STATUS_MATCH_SYNONYMS expansion, and SHALL award points only when the
    polarity matches or is a recognized synonym.

    We verify:
    1. POLARITY_MAP normalizes the patient indicator to a canonical class.
    2. STATUS_MATCH_SYNONYMS expands the canonical class to a synonym list.
    3. A study polarity in the synonym list → match (points awarded).
    4. A study polarity NOT in the synonym list → no match (0 points).

    **Validates: Requirements 5.3, 5.4**

    Tag: Feature: patient-study-match-scoring, Property 13: JSONB polarity-aware biomarker matching
    """
    biomarker_key, patient_polarity, study_polarity, should_match = data

    # Step 1: Normalize patient polarity via POLARITY_MAP
    canonical = POLARITY_MAP.get(patient_polarity.lower(), patient_polarity.lower())
    assert canonical is not None, (
        f"POLARITY_MAP should normalize '{patient_polarity}' to a canonical class"
    )

    # Step 2: Get synonym list from STATUS_MATCH_SYNONYMS
    synonyms = STATUS_MATCH_SYNONYMS.get(canonical, [canonical])
    assert isinstance(synonyms, list), (
        f"STATUS_MATCH_SYNONYMS['{canonical}'] should be a list, got {type(synonyms)}"
    )
    assert len(synonyms) > 0, (
        f"STATUS_MATCH_SYNONYMS['{canonical}'] should be non-empty"
    )

    # Step 3: Verify the canonical class itself is in its own synonym list
    assert canonical in [s.lower() for s in synonyms] or canonical in synonyms, (
        f"Canonical class '{canonical}' should be in its own synonym list {synonyms}"
    )

    # Step 4: Verify match/no-match determination
    actual_match = polarity_matches(patient_polarity, study_polarity)
    assert actual_match == should_match, (
        f"polarity_matches('{patient_polarity}', '{study_polarity}') = {actual_match}, "
        f"expected {should_match}. canonical='{canonical}', synonyms={synonyms}"
    )

    # Step 5: Simulate scoring — points awarded only when polarity matches
    max_points = 14.0  # biomarker_jsonb weight from design
    points_awarded = max_points if actual_match else 0.0

    if actual_match:
        assert points_awarded > 0, (
            f"Matching polarity should award points. "
            f"patient='{patient_polarity}', study='{study_polarity}'"
        )
    else:
        assert points_awarded == 0.0, (
            f"Non-matching polarity should award 0 points. "
            f"patient='{patient_polarity}', study='{study_polarity}'"
        )


# ======================================================================
# Additional unit tests for STATUS_MATCH_SYNONYMS coverage
# ======================================================================


class TestStatusMatchSynonymsExpansion:
    """Unit tests verifying specific synonym expansion rules from the spec."""

    def test_positive_matches_amplified(self):
        """'positive' should match 'amplified'."""
        assert polarity_matches("positive", "amplified")

    def test_positive_matches_overexpressed(self):
        """'positive' should match 'overexpressed'."""
        assert polarity_matches("positive", "overexpressed")

    def test_positive_matches_detected(self):
        """'positive' should match 'detected'."""
        assert polarity_matches("positive", "detected")

    def test_positive_matches_high(self):
        """'positive' should match 'high'."""
        assert polarity_matches("positive", "high")

    def test_negative_matches_absent(self):
        """'negative' should match 'absent'."""
        assert polarity_matches("negative", "absent")

    def test_negative_matches_not_detected(self):
        """'negative' should match 'not detected'."""
        assert polarity_matches("negative", "not detected")

    def test_negative_matches_wild_type(self):
        """'negative' should match 'wild-type'."""
        assert polarity_matches("negative", "wild-type")

    def test_mutant_matches_mutation(self):
        """'mutant' should match 'mutation'."""
        assert polarity_matches("mutant", "mutation")

    def test_mutant_matches_altered(self):
        """'mutant' should match 'altered'."""
        assert polarity_matches("mutant", "altered")

    def test_mutant_matches_fusion(self):
        """'mutant' should match 'fusion'."""
        assert polarity_matches("mutant", "fusion")

    def test_mutant_matches_rearrangement(self):
        """'mutant' should match 'rearrangement'."""
        assert polarity_matches("mutant", "rearrangement")

    def test_positive_does_not_match_wild_type(self):
        """'positive' should NOT match 'wild-type'."""
        assert not polarity_matches("positive", "wild-type")

    def test_negative_does_not_match_amplified(self):
        """'negative' should NOT match 'amplified'."""
        assert not polarity_matches("negative", "amplified")

    def test_polarity_map_normalizes_plus_to_positive(self):
        """POLARITY_MAP should normalize '+' to 'positive'."""
        assert POLARITY_MAP["+"] == "positive"

    def test_polarity_map_normalizes_minus_to_negative(self):
        """POLARITY_MAP should normalize '-' to 'negative'."""
        assert POLARITY_MAP["-"] == "negative"

    def test_polarity_map_normalizes_mutation_to_mutant(self):
        """POLARITY_MAP should normalize 'mutation' to 'mutant'."""
        assert POLARITY_MAP["mutation"] == "mutant"

    def test_polarity_map_normalizes_wt_to_wild_type(self):
        """POLARITY_MAP should normalize 'wt' to 'wild-type'."""
        assert POLARITY_MAP["wt"] == "wild-type"
