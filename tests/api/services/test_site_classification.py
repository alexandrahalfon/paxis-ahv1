"""
Site Classification Unit Tests — Fix 2: Lung site misclassification.

Tests that primary tumor sites win over metastatic site mentions,
and that single-site queries without metastatic mentions still work.

**Validates: Requirements 2.3, 3.3**
"""

import pytest

from src.api.services.query_structuring_service import structure_query_fast


# ======================================================================
# Primary + metastatic combination tests
# ======================================================================

class TestPrimaryOverMetastaticSite:
    """
    When a query mentions both a primary tumor site and metastatic sites,
    the primary site must win.

    **Validates: Requirements 2.3**
    """

    def test_lung_primary_with_liver_mets(self):
        result = structure_query_fast(
            "adenocarcinoma of the right lower lobe, liver metastases, "
            "leptomeningeal spread"
        )
        assert result.cancer.site == "lung", (
            f"Expected site='lung', got site='{result.cancer.site}'"
        )

    def test_lung_primary_with_brain_mets(self):
        result = structure_query_fast(
            "lung adenocarcinoma with brain metastases"
        )
        assert result.cancer.site == "lung", (
            f"Expected site='lung', got site='{result.cancer.site}'"
        )

    def test_lung_primary_with_bone_mets(self):
        result = structure_query_fast(
            "NSCLC with spread to bone"
        )
        assert result.cancer.site == "lung", (
            f"Expected site='lung', got site='{result.cancer.site}'"
        )

    def test_prostate_primary_with_bone_mets(self):
        result = structure_query_fast(
            "prostate adenocarcinoma with bone metastases"
        )
        assert result.cancer.site == "prostate", (
            f"Expected site='prostate', got site='{result.cancer.site}'"
        )


# ======================================================================
# Metastatic-context detection tests
# ======================================================================

class TestMetastaticContextDetection:
    """
    Metastatic language near a site mention should demote that site.

    **Validates: Requirements 2.3**
    """

    def test_liver_metastases_demoted(self):
        result = structure_query_fast(
            "lung cancer with liver metastases"
        )
        assert result.cancer.site == "lung", (
            f"'liver metastases' should be demoted; got site='{result.cancer.site}'"
        )

    def test_brain_mets_demoted(self):
        result = structure_query_fast(
            "lung cancer with brain mets"
        )
        assert result.cancer.site == "lung", (
            f"'brain mets' should be demoted; got site='{result.cancer.site}'"
        )

    def test_spread_to_bone_demoted(self):
        result = structure_query_fast(
            "prostate cancer, spread to bone"
        )
        assert result.cancer.site == "prostate", (
            f"'spread to bone' should be demoted; got site='{result.cancer.site}'"
        )


# ======================================================================
# Backbone query test (the specific failing case from Bug 3)
# ======================================================================

class TestBackboneLungQuery:
    """
    The exact backbone query that triggered Bug 3.

    **Validates: Requirements 2.3**
    """

    def test_backbone_adenocarcinoma_rll_liver_mets(self):
        result = structure_query_fast(
            "adenocarcinoma of the right lower lobe, liver metastases, "
            "leptomeningeal spread"
        )
        assert result.cancer.site == "lung", (
            f"Expected site='lung', got site='{result.cancer.site}'"
        )


# ======================================================================
# Preservation: single-site queries must still work
# ======================================================================

class TestSingleSitePreservation:
    """
    Single-site queries without metastatic mentions must continue to
    classify correctly on first pattern match.

    **Validates: Requirements 3.3**
    """

    def test_breast_cancer(self):
        result = structure_query_fast("breast cancer")
        assert result.cancer.site == "breast", (
            f"Expected site='breast', got site='{result.cancer.site}'"
        )

    def test_prostate_adenocarcinoma(self):
        result = structure_query_fast("prostate adenocarcinoma")
        assert result.cancer.site == "prostate", (
            f"Expected site='prostate', got site='{result.cancer.site}'"
        )

    def test_lung_nsclc(self):
        result = structure_query_fast("lung NSCLC")
        assert result.cancer.site == "lung", (
            f"Expected site='lung', got site='{result.cancer.site}'"
        )

    def test_head_and_neck_scc(self):
        result = structure_query_fast("head and neck SCC")
        assert result.cancer.site == "head_neck", (
            f"Expected site='head_neck', got site='{result.cancer.site}'"
        )

    def test_colon_cancer(self):
        result = structure_query_fast("colon cancer")
        assert result.cancer.site == "gi_lower", (
            f"Expected site='gi_lower', got site='{result.cancer.site}'"
        )

    def test_cervical_cancer(self):
        result = structure_query_fast("cervical cancer")
        assert result.cancer.site == "gyn", (
            f"Expected site='gyn', got site='{result.cancer.site}'"
        )
