"""
Unit tests for the notebook criteria scorer's category comparison logic.

The notebook ``pipeline_backbone_test.ipynb`` contains a ``score_matching_criteria()``
function that compares ``filter_category`` against ``study.category``.  The original
code uses exact string equality (``filter_category.lower() == str(study.category).lower()``),
which fails when the two strings refer to the same cancer site but use different
naming conventions (e.g. ``"head_neck"`` vs ``"h&n_processed_documents"``).

This module extracts the category-matching logic into a standalone
``score_category_match()`` function that uses ``normalize_category()`` from
``comprehensive_retrieval.py`` — the same alias map used by the Phase3Gate fix.

**Validates: Requirements 2.9**
"""

import pytest

from src.api.services.comprehensive_retrieval import normalize_category


# ── Standalone testable function ──────────────────────────────────────
# This mirrors the category-comparison branch inside the notebook's
# ``score_matching_criteria()``, but uses ``normalize_category()`` so
# that alias pairs (h&n ↔ head_neck) and suffix differences
# (_processed_documents) are resolved before comparison.

_CATEGORY_WEIGHT = 0.20  # same weight as in the notebook


def score_category_match(
    filter_category: str | None,
    study_category: str | None,
) -> float:
    """Return the category score component (0.0 or ``_CATEGORY_WEIGHT``).

    Uses ``normalize_category()`` so that ``"head_neck"`` matches
    ``"h&n_processed_documents"`` and ``"prostate"`` matches
    ``"prostate_processed_documents"``.
    """
    if not filter_category or not study_category:
        return 0.0
    if normalize_category(filter_category) == normalize_category(str(study_category)):
        return _CATEGORY_WEIGHT
    return 0.0


# ======================================================================
# Tests — same-site pairs must produce a non-zero category score
# ======================================================================

class TestSameSiteCategoryScore:
    """Pairs that refer to the same cancer site must score > 0."""

    def test_head_neck_vs_hn_processed_documents(self):
        """head_neck vs h&n_processed_documents → non-zero."""
        score = score_category_match("head_neck", "h&n_processed_documents")
        assert score > 0, (
            f"Expected non-zero category score for head_neck vs "
            f"h&n_processed_documents, got {score}"
        )
        assert score == pytest.approx(_CATEGORY_WEIGHT)

    def test_prostate_vs_prostate_processed_documents(self):
        """prostate vs prostate_processed_documents → non-zero."""
        score = score_category_match("prostate", "prostate_processed_documents")
        assert score > 0, (
            f"Expected non-zero category score for prostate vs "
            f"prostate_processed_documents, got {score}"
        )
        assert score == pytest.approx(_CATEGORY_WEIGHT)

    def test_lung_vs_lung_processed_documents(self):
        """lung vs lung_processed_documents → non-zero."""
        score = score_category_match("lung", "lung_processed_documents")
        assert score > 0
        assert score == pytest.approx(_CATEGORY_WEIGHT)


# ======================================================================
# Tests — genuine cross-site pairs must produce zero category score
# ======================================================================

class TestCrossSiteCategoryScore:
    """Pairs that refer to different cancer sites must score 0."""

    def test_lung_vs_prostate_processed_documents(self):
        """lung vs prostate_processed_documents → zero."""
        score = score_category_match("lung", "prostate_processed_documents")
        assert score == 0.0

    def test_head_neck_vs_lung_processed_documents(self):
        """head_neck vs lung_processed_documents → zero."""
        score = score_category_match("head_neck", "lung_processed_documents")
        assert score == 0.0

    def test_breast_vs_prostate(self):
        """breast vs prostate → zero."""
        score = score_category_match("breast", "prostate")
        assert score == 0.0


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """None / empty inputs must produce zero score without errors."""

    def test_none_filter_category(self):
        assert score_category_match(None, "prostate_processed_documents") == 0.0

    def test_none_study_category(self):
        assert score_category_match("prostate", None) == 0.0

    def test_both_none(self):
        assert score_category_match(None, None) == 0.0

    def test_empty_string_filter_category(self):
        assert score_category_match("", "prostate_processed_documents") == 0.0

    def test_empty_string_study_category(self):
        assert score_category_match("prostate", "") == 0.0
