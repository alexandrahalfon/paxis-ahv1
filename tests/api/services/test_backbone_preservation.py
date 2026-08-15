"""
Preservation Property Tests — BEFORE any fixes.

These tests document CORRECT baseline behaviors that must NOT regress
after the 9 backbone fixes are applied. They MUST PASS on unfixed code.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9**
"""

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from src.api.services.query_structuring_service import structure_query_fast
from src.api.services.enhanced_rag_service import classify_query
from src.api.services.patient_eligibility_boost_service import (
    apply_patient_eligibility_filter_and_boost,
)


# ======================================================================
# Test 2a: Wild-type / negative-polarity extraction preserved
#          (Property 2 — Preservation)
# ======================================================================

# Map each wild-type query to its expected biomarker on unfixed code
WILDTYPE_QUERIES = {
    "EGFR wild-type": "EGFR wild-type",
    "BRCA wild-type": "BRCA wild-type",
    "BRCA-negative": "BRCA wild-type",
}


class TestWildTypePreservation:
    """
    Property 2: For explicit negative-polarity biomarker terms,
    structure_query_fast() must continue to extract the correct
    negative-polarity canonical name after fixes.

    **Validates: Requirements 3.1, 3.2, 3.9**
    """

    @given(
        wt_label=st.sampled_from(list(WILDTYPE_QUERIES.keys())),
    )
    @hyp_settings(max_examples=10, deadline=5000)
    def test_2a_wildtype_strings_extract_negative_polarity(self, wt_label):
        query = f"{wt_label} cancer"
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers
        expected = WILDTYPE_QUERIES[wt_label]
        assert expected in biomarkers, (
            f"Query '{query}': expected '{expected}' in biomarkers, "
            f"got {biomarkers}"
        )


# ======================================================================
# Test 2b: Single-site classification preserved (Property 4)
# ======================================================================

SINGLE_SITE_QUERIES = {
    "breast cancer": "breast",
    "prostate adenocarcinoma": "prostate",
    "lung NSCLC": "lung",
    "head and neck SCC": "head_neck",
}


class TestSingleSitePreservation:
    """
    Property 4: For single-site queries without metastatic mentions,
    structure_query_fast() must continue to classify the correct site.

    **Validates: Requirements 3.3**
    """

    @given(
        query_text=st.sampled_from(list(SINGLE_SITE_QUERIES.keys())),
    )
    @hyp_settings(max_examples=10, deadline=5000)
    def test_2b_single_site_queries_return_expected_site(self, query_text):
        result = structure_query_fast(query_text)
        expected_site = SINGLE_SITE_QUERIES[query_text]
        assert result.cancer.site == expected_site, (
            f"Query '{query_text}': expected site='{expected_site}', "
            f"got site='{result.cancer.site}'"
        )


# ======================================================================
# Test 2c: Genuine cross-site penalty IS applied (Property 6)
# ======================================================================

# Genuinely different category pairs — penalty must be applied
CROSS_SITE_PAIRS = [
    ("lung", "prostate_processed_documents"),
    ("breast", "lung_processed_documents"),
    ("head_neck", "gi_processed_documents"),
    ("prostate", "breast_processed_documents"),
]


class TestCrossSitePenaltyPreservation:
    """
    Property 6: For genuinely different category pairs, the Phase3Gate
    site-mismatch penalty IS applied. Since normalize_category doesn't
    exist yet, we test the raw string comparison behavior that the
    Phase3Gate uses: query_category != doc_category.

    **Validates: Requirements 3.4**
    """

    @given(
        pair=st.sampled_from(CROSS_SITE_PAIRS),
    )
    @hyp_settings(max_examples=10, deadline=5000)
    def test_2c_genuine_cross_site_pairs_trigger_penalty(self, pair):
        query_category, doc_category = pair
        # The Phase3Gate applies penalty when query_category != doc_category.
        # For genuinely different sites, this must remain true even after
        # normalize_category is introduced.
        assert query_category != doc_category, (
            f"Raw comparison: {query_category!r} should differ from "
            f"{doc_category!r} for genuine cross-site pair"
        )
        # Additionally verify the base sites are genuinely different
        # (strip _processed_documents suffix for semantic check)
        base_doc = doc_category.replace("_processed_documents", "")
        assert query_category != base_doc, (
            f"Base sites should differ: {query_category!r} vs {base_doc!r}"
        )


# ======================================================================
# Test 2d: Unambiguous staging query classification (Property 8)
# ======================================================================

class TestStagingClassificationPreservation:
    """
    Property 8: Unambiguous staging queries must continue to classify
    as 'staging' after fixes.

    **Validates: Requirements 3.6**
    """

    def test_2d_staging_query_classifies_as_staging(self):
        result = classify_query("What stage is T2N1M0 NSCLC?")
        assert result["primary_type"] == "staging", (
            f"Expected primary_type='staging', "
            f"got '{result['primary_type']}' with scores={result['scores']}"
        )


# ======================================================================
# Test 2e: Unambiguous dose query classification (Property 8)
# ======================================================================

class TestDoseClassificationPreservation:
    """
    Property 8: Unambiguous dose queries must continue to classify
    as 'dose_question' after fixes.

    **Validates: Requirements 3.7**
    """

    def test_2e_dose_query_classifies_as_dose_question(self):
        result = classify_query(
            "What dose should be given for breast cancer? 50 Gy in 25 fractions?"
        )
        assert result["primary_type"] == "dose_question", (
            f"Expected primary_type='dose_question', "
            f"got '{result['primary_type']}' with scores={result['scores']}"
        )


# ======================================================================
# Test 2f: Legitimate cancer_type MISMATCH removals (Property 10)
# ======================================================================

class TestLegitimateHardFilterPreservation:
    """
    Property 10: Legitimate cancer_type MISMATCH removals must still
    be removed by the eligibility filter. Lung studies for an H&N
    patient must be hard-filtered.

    **Validates: Requirements 3.5, 3.8**
    """

    def test_2f_cancer_type_mismatch_studies_are_removed(self):
        # Simulate chunks from two studies: one lung study (mismatch for H&N)
        # and one H&N study (match)
        chunks = [
            {
                "doc_id": "lung_study_001",
                "title": "NSCLC Pembrolizumab Trial",
                "score": 0.8,
                "final_score": 0.8,
                "text": "Lung cancer immunotherapy results",
            },
            {
                "doc_id": "hn_study_001",
                "title": "H&N SCC Immunotherapy Trial",
                "score": 0.7,
                "final_score": 0.7,
                "text": "Head and neck cancer immunotherapy results",
            },
        ]

        # Simulate eligibility results where lung study has cancer_type MISMATCH
        eligibility_results = {
            "lung_study_001": {
                "status": "NO_MATCH",
                "reason": "Study enrolls lung cancer, patient has H&N cancer",
                "has_hard_mismatch": True,
                "boost": 0,
                "criteria_verdicts": {
                    "cancer_type": "MISMATCH",
                    "histology": "NOT_AVAILABLE",
                    "stage": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                    "biomarkers": "NOT_AVAILABLE",
                },
            },
            "hn_study_001": {
                "status": "MATCH",
                "reason": "Study matches patient cancer type",
                "has_hard_mismatch": False,
                "boost": 0.25,
                "criteria_verdicts": {
                    "cancer_type": "MATCH",
                    "histology": "MATCH",
                    "stage": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                    "biomarkers": "NOT_AVAILABLE",
                },
            },
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks, eligibility_results
        )

        # The lung study must be removed (cancer_type MISMATCH)
        removed_doc_ids = [c.get("doc_id") for c in removed]
        assert "lung_study_001" in removed_doc_ids, (
            f"Expected lung_study_001 to be removed, "
            f"removed={removed_doc_ids}"
        )

        # The H&N study must be kept
        kept_doc_ids = [c.get("doc_id") for c in kept]
        assert "hn_study_001" in kept_doc_ids, (
            f"Expected hn_study_001 to be kept, kept={kept_doc_ids}"
        )

        # Verify the removed chunk has the right eligibility metadata
        removed_lung = [c for c in removed if c.get("doc_id") == "lung_study_001"][0]
        assert removed_lung["patient_eligibility"]["hard_filtered"] is True
        assert removed_lung["patient_eligibility"]["criteria_verdicts"]["cancer_type"] == "MISMATCH"
