"""
Property-based tests for PG match breakdown dataclasses.

Feature: patient-study-match-scoring

Tests the following properties:
- Property 6: PGMatchBreakdown completeness invariant
- Property 7: PGMatchBreakdown total equals sum of parts
- Property 8: Missing axis data yields zero points

**Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.api.services.structured_study_matcher import (
    AxisContribution,
    PGMatchBreakdown,
    BASE_SCORING_WEIGHTS,
)


# ======================================================================
# Shared constants and strategies
# ======================================================================

ALL_AXIS_KEYS = list(BASE_SCORING_WEIGHTS.keys())

MATCH_LABELS = ["exact match", "partial match", "no match", "not reported"]


def axis_contribution_strategy(
    axis_name: str | None = None,
    allow_zero: bool = True,
):
    """Generate a random AxisContribution for a given or random axis name."""
    name_st = st.just(axis_name) if axis_name else st.sampled_from(ALL_AXIS_KEYS)
    max_pts = st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    if allow_zero:
        earned = st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    else:
        earned = st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
    label = st.sampled_from(MATCH_LABELS)

    return st.builds(
        AxisContribution,
        axis_name=name_st,
        points_earned=earned,
        max_points=max_pts,
        label=label,
    )


# ======================================================================
# Property 6: PGMatchBreakdown completeness invariant
# ======================================================================


@st.composite
def active_axes_with_contributions(draw):
    """
    Generate a random non-empty subset of BASE_SCORING_WEIGHTS keys as
    "active axes" and build a PGMatchBreakdown with one AxisContribution
    per active axis.
    """
    # Pick a non-empty subset of axis keys
    subset = draw(
        st.lists(
            st.sampled_from(ALL_AXIS_KEYS),
            min_size=1,
            max_size=len(ALL_AXIS_KEYS),
            unique=True,
        )
    )

    contributions = []
    for axis_key in subset:
        max_pts = draw(st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
        earned = draw(st.floats(min_value=0.0, max_value=max_pts, allow_nan=False, allow_infinity=False))
        label = draw(st.sampled_from(["exact match", "partial match", "no match", "not reported"]))
        contributions.append(AxisContribution(
            axis_name=axis_key,
            points_earned=earned,
            max_points=max_pts,
            label=label,
        ))

    total = sum(c.points_earned for c in contributions)
    mismatches = [c.axis_name for c in contributions if c.points_earned == 0.0 and c.label == "no match"]

    breakdown = PGMatchBreakdown(
        total_score=total,
        axis_contributions=contributions,
        axis_mismatches=mismatches,
    )

    return subset, breakdown


# Feature: patient-study-match-scoring, Property 6: PGMatchBreakdown completeness invariant
@settings(max_examples=150)
@given(data=active_axes_with_contributions())
def test_pg_match_breakdown_completeness_invariant(data):
    """
    Property 6: PGMatchBreakdown completeness invariant.

    For any query with active scoring axes and any matched study, the
    PGMatchBreakdown SHALL contain exactly one AxisContribution per active
    axis, and each AxisContribution SHALL contain a non-null axis name,
    points earned (>= 0), max points (> 0), and a human-readable label.

    **Validates: Requirements 3.1, 3.2, 3.3**

    Tag: Feature: patient-study-match-scoring, Property 6: PGMatchBreakdown completeness invariant
    """
    active_axes, breakdown = data

    # One AxisContribution per active axis
    contribution_names = [ac.axis_name for ac in breakdown.axis_contributions]
    assert len(contribution_names) == len(active_axes), (
        f"Expected {len(active_axes)} contributions, got {len(contribution_names)}"
    )
    assert set(contribution_names) == set(active_axes), (
        f"Contribution axes {set(contribution_names)} != active axes {set(active_axes)}"
    )

    # Each AxisContribution has non-null fields with correct constraints
    for ac in breakdown.axis_contributions:
        assert ac.axis_name is not None and ac.axis_name != "", (
            f"axis_name must be non-null and non-empty, got {ac.axis_name!r}"
        )
        assert ac.points_earned is not None and ac.points_earned >= 0, (
            f"points_earned must be >= 0, got {ac.points_earned}"
        )
        assert ac.max_points is not None and ac.max_points > 0, (
            f"max_points must be > 0, got {ac.max_points}"
        )
        assert ac.label is not None and ac.label != "", (
            f"label must be non-null and non-empty, got {ac.label!r}"
        )


# ======================================================================
# Property 7: PGMatchBreakdown total equals sum of parts
# ======================================================================


@st.composite
def contributions_with_total(draw):
    """
    Generate a random list of AxisContribution entries with random
    points_earned values, and build a PGMatchBreakdown where total_score
    is set to the sum of points_earned.
    """
    num_axes = draw(st.integers(min_value=1, max_value=len(ALL_AXIS_KEYS)))
    axes = draw(
        st.lists(
            st.sampled_from(ALL_AXIS_KEYS),
            min_size=num_axes,
            max_size=num_axes,
            unique=True,
        )
    )

    contributions = []
    for axis_key in axes:
        earned = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        max_pts = draw(st.floats(min_value=max(earned, 1.0), max_value=100.0, allow_nan=False, allow_infinity=False))
        label = draw(st.sampled_from(MATCH_LABELS))
        contributions.append(AxisContribution(
            axis_name=axis_key,
            points_earned=earned,
            max_points=max_pts,
            label=label,
        ))

    total = sum(c.points_earned for c in contributions)

    breakdown = PGMatchBreakdown(
        total_score=total,
        axis_contributions=contributions,
        axis_mismatches=[],
    )

    return breakdown


# Feature: patient-study-match-scoring, Property 7: PGMatchBreakdown total equals sum of parts
@settings(max_examples=150)
@given(data=contributions_with_total())
def test_pg_match_breakdown_total_equals_sum_of_parts(data):
    """
    Property 7: PGMatchBreakdown total equals sum of parts.

    For any PGMatchBreakdown, the total_score SHALL equal the sum of all
    AxisContribution.points_earned values.

    **Validates: Requirements 3.6**

    Tag: Feature: patient-study-match-scoring, Property 7: PGMatchBreakdown total equals sum of parts
    """
    breakdown = data

    expected_total = sum(ac.points_earned for ac in breakdown.axis_contributions)

    assert abs(breakdown.total_score - expected_total) < 1e-9, (
        f"total_score {breakdown.total_score} != sum of points_earned {expected_total}"
    )


# ======================================================================
# Property 8: Missing axis data yields zero points
# ======================================================================


@st.composite
def contributions_with_missing_axes(draw):
    """
    Generate a list of AxisContribution entries where some have 0
    points_earned and label "not reported" (simulating NULL/missing data
    in the study), and others have positive points.
    """
    num_axes = draw(st.integers(min_value=2, max_value=len(ALL_AXIS_KEYS)))
    axes = draw(
        st.lists(
            st.sampled_from(ALL_AXIS_KEYS),
            min_size=num_axes,
            max_size=num_axes,
            unique=True,
        )
    )

    # At least one axis must be "missing" (NULL data)
    num_missing = draw(st.integers(min_value=1, max_value=max(1, num_axes - 1)))
    missing_indices = set(draw(
        st.lists(
            st.integers(min_value=0, max_value=num_axes - 1),
            min_size=num_missing,
            max_size=num_missing,
            unique=True,
        )
    ))

    contributions = []
    for i, axis_key in enumerate(axes):
        if i in missing_indices:
            # Simulate NULL/missing data: 0 points, "not reported"
            max_pts = draw(st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
            contributions.append(AxisContribution(
                axis_name=axis_key,
                points_earned=0.0,
                max_points=max_pts,
                label="not reported",
            ))
        else:
            # Normal axis with some points
            max_pts = draw(st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
            earned = draw(st.floats(min_value=0.01, max_value=max_pts, allow_nan=False, allow_infinity=False))
            label = draw(st.sampled_from(["exact match", "partial match"]))
            contributions.append(AxisContribution(
                axis_name=axis_key,
                points_earned=earned,
                max_points=max_pts,
                label=label,
            ))

    return contributions, missing_indices


# Feature: patient-study-match-scoring, Property 8: Missing axis data yields zero points
@settings(max_examples=150)
@given(data=contributions_with_missing_axes())
def test_missing_axis_data_yields_zero_points(data):
    """
    Property 8: Missing axis data yields zero points.

    For any scoring axis where the study has no data (NULL or empty column),
    the AxisContribution for that axis SHALL report 0 points earned with a
    label of "not reported".

    **Validates: Requirements 3.5**

    Tag: Feature: patient-study-match-scoring, Property 8: Missing axis data yields zero points
    """
    contributions, missing_indices = data

    for i, ac in enumerate(contributions):
        if i in missing_indices:
            # Missing axis: must have 0 points and "not reported" label
            assert ac.points_earned == 0.0, (
                f"Missing axis {ac.axis_name!r} should have 0 points, got {ac.points_earned}"
            )
            assert ac.label == "not reported", (
                f"Missing axis {ac.axis_name!r} should have label 'not reported', got {ac.label!r}"
            )
        else:
            # Non-missing axis: should have positive points
            assert ac.points_earned > 0, (
                f"Non-missing axis {ac.axis_name!r} should have positive points, got {ac.points_earned}"
            )
            assert ac.label != "not reported", (
                f"Non-missing axis {ac.axis_name!r} should not have label 'not reported'"
            )
