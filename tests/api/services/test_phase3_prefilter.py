"""
Tests for Phase 3 site-category pre-filter (Fix 5 / Task 7).

The pre-filter skips off-site candidates BEFORE dispatching to expensive
cross-encoder + LLM calls, reducing wasted computation for queries with
a known category.

Validates: Requirements 2.6
"""

import pytest

from src.api.services.comprehensive_retrieval import (
    normalize_category,
    should_skip_phase3_candidate,
)


class TestShouldSkipPhase3Candidate:
    """Unit tests for should_skip_phase3_candidate()."""

    # ── Off-site candidates should be skipped ──────────────────────────

    def test_skip_off_site_candidate_different_sites(self):
        """When query is head_neck and doc is lung, skip."""
        assert should_skip_phase3_candidate("head_neck", "lung") is True

    def test_skip_off_site_candidate_with_suffix(self):
        """When query is prostate and doc is lung_processed_documents, skip."""
        assert should_skip_phase3_candidate("prostate", "lung_processed_documents") is True

    def test_skip_off_site_candidate_hn_vs_lung(self):
        """When query is head_neck and doc is lung_processed_documents, skip."""
        assert should_skip_phase3_candidate("head_neck", "lung_processed_documents") is True

    def test_skip_off_site_candidate_lung_vs_prostate(self):
        """When query is lung and doc is prostate_processed_documents, skip."""
        assert should_skip_phase3_candidate("lung", "prostate_processed_documents") is True

    # ── On-site candidates must NOT be skipped ─────────────────────────

    def test_on_site_same_string(self):
        """Same category string → not skipped."""
        assert should_skip_phase3_candidate("prostate", "prostate") is False

    def test_on_site_with_suffix(self):
        """prostate vs prostate_processed_documents → same site, not skipped."""
        assert should_skip_phase3_candidate("prostate", "prostate_processed_documents") is False

    def test_on_site_hn_alias(self):
        """head_neck vs h&n_processed_documents → same site via alias, not skipped."""
        assert should_skip_phase3_candidate("head_neck", "h&n_processed_documents") is False

    def test_on_site_lung_with_suffix(self):
        """lung vs lung_processed_documents → same site, not skipped."""
        assert should_skip_phase3_candidate("lung", "lung_processed_documents") is False

    # ── No category → never skip (no false rejections) ─────────────────

    def test_no_query_category_none(self):
        """When query has no category (None), never skip."""
        assert should_skip_phase3_candidate(None, "lung_processed_documents") is False

    def test_no_query_category_empty(self):
        """When query has empty category, never skip."""
        assert should_skip_phase3_candidate("", "lung_processed_documents") is False

    def test_no_doc_category_none(self):
        """When doc has no category (None), never skip."""
        assert should_skip_phase3_candidate("head_neck", None) is False

    def test_no_doc_category_empty(self):
        """When doc has empty category, never skip."""
        assert should_skip_phase3_candidate("head_neck", "") is False

    def test_both_categories_none(self):
        """When both are None, never skip."""
        assert should_skip_phase3_candidate(None, None) is False

    def test_both_categories_empty(self):
        """When both are empty, never skip."""
        assert should_skip_phase3_candidate("", "") is False

    # ── When query has no known category, all candidates pass ──────────

    def test_no_query_category_all_pass(self):
        """With no query category, every doc category passes through."""
        doc_categories = [
            "lung_processed_documents",
            "prostate_processed_documents",
            "h&n_processed_documents",
            "breast",
            None,
            "",
        ]
        for doc_cat in doc_categories:
            assert should_skip_phase3_candidate(None, doc_cat) is False, (
                f"Expected not to skip doc_cat={doc_cat!r} when query_cat is None"
            )
