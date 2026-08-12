"""
Unit tests for normalize_category() helper in comprehensive_retrieval.py.

Tests cover:
- Suffix stripping (_processed_documents, _docs, _documents)
- Known alias resolution (h&n ↔ head_neck)
- Same-site pairs normalize to equal values
- Genuine cross-site pairs normalize to different values
- Edge cases: None, empty string, already-normalized values

**Validates: Requirements 2.5, 2.9, 3.4**
"""

import pytest

from src.api.services.comprehensive_retrieval import normalize_category
from tests.fixtures.backbone_golden_queries import CATEGORY_ALIAS_PAIRS


# ======================================================================
# Basic suffix stripping
# ======================================================================

class TestSuffixStripping:
    """normalize_category strips _processed_documents (and variants)."""

    def test_strips_processed_documents_suffix(self):
        assert normalize_category("prostate_processed_documents") == normalize_category("prostate")

    def test_strips_docs_suffix(self):
        assert normalize_category("lung_docs") == normalize_category("lung")

    def test_strips_documents_suffix(self):
        assert normalize_category("breast_documents") == normalize_category("breast")


# ======================================================================
# Known alias resolution
# ======================================================================

class TestAliasResolution:
    """h&n ↔ head_neck is the tricky alias pair."""

    def test_head_neck_equals_hn_processed_documents(self):
        assert normalize_category("head_neck") == normalize_category("h&n_processed_documents")

    def test_hn_equals_head_neck(self):
        assert normalize_category("h&n") == normalize_category("head_neck")

    def test_prostate_equals_prostate_processed_documents(self):
        assert normalize_category("prostate") == normalize_category("prostate_processed_documents")


# ======================================================================
# All golden alias pairs
# ======================================================================

class TestAllGoldenAliasPairs:
    """Every pair from CATEGORY_ALIAS_PAIRS must normalize to the same value."""

    @pytest.mark.parametrize("short,long", CATEGORY_ALIAS_PAIRS)
    def test_alias_pair_normalizes_equal(self, short, long):
        assert normalize_category(short) == normalize_category(long), (
            f"normalize_category({short!r}) = {normalize_category(short)!r} "
            f"!= normalize_category({long!r}) = {normalize_category(long)!r}"
        )


# ======================================================================
# Genuine cross-site pairs must NOT match
# ======================================================================

class TestGenuineCrossSite:
    """Different cancer sites must normalize to different values."""

    def test_lung_not_equal_prostate(self):
        assert normalize_category("lung") != normalize_category("prostate_processed_documents")

    def test_breast_not_equal_lung(self):
        assert normalize_category("breast") != normalize_category("lung_processed_documents")

    def test_head_neck_not_equal_gi(self):
        assert normalize_category("head_neck") != normalize_category("gi_processed_documents")


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """None, empty string, already-normalized values."""

    def test_none_returns_empty_string(self):
        assert normalize_category(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert normalize_category("") == ""

    def test_whitespace_returns_empty_string(self):
        assert normalize_category("  ") == ""

    def test_already_normalized_value_unchanged(self):
        assert normalize_category("lung") == "lung"

    def test_mixed_case_normalized(self):
        assert normalize_category("PROSTATE") == normalize_category("prostate")

    def test_leading_trailing_whitespace_stripped(self):
        assert normalize_category("  lung  ") == normalize_category("lung")
