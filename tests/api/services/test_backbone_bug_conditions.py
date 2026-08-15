"""
Bug Condition Exploration Tests — BEFORE any fixes.

These tests encode the EXPECTED (correct) behavior for each bug.
They are designed to FAIL on unfixed code, confirming the bugs exist.
After fixes are applied, they should PASS.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.8, 1.9, 2.1, 2.2, 2.3, 2.5, 2.8, 2.9**
"""

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from src.api.services.query_structuring_service import structure_query_fast
from src.api.services.enhanced_rag_service import classify_query


# ── Biomarker polarity gene/qualifier sets for property test ──────────
POSITIVE_POLARITY_GENES = ["EGFR", "BRCA1", "BRCA2", "KRAS", "BRAF"]
POSITIVE_POLARITY_QUALIFIERS = ["mutant", "mutation", "mutated"]

# Canonical positive-polarity names expected per gene
GENE_TO_CANONICAL = {
    "EGFR": "EGFR mutant",
    "BRCA1": "BRCA mutant",
    "BRCA2": "BRCA mutant",
    "KRAS": "KRAS mutant",
    "BRAF": "BRAF mutant",
}

# Negative-polarity canonical names that must NOT appear
GENE_TO_WILDTYPE = {
    "EGFR": "EGFR wild-type",
    "BRCA1": "BRCA wild-type",
    "BRCA2": "BRCA wild-type",
    "KRAS": "KRAS wild-type",
    "BRAF": "BRAF wild-type",
}


# ======================================================================
# Test 1a: EGFR polarity flip (Bug 1, Property 1)
# ======================================================================

class TestEGFRPolarityFlip:
    """
    Bug 1: 'EGFR-mutant (exon 19 del)' extracts 'EGFR wild-type'
    instead of 'EGFR mutant' because the hyphen matches the wild-type
    pattern's bare `-` alternative.

    **Validates: Requirements 2.1**
    """

    def test_1a_egfr_mutant_extracts_positive_polarity(self):
        result = structure_query_fast(
            "EGFR-mutant (exon 19 del) lung adenocarcinoma"
        )
        biomarkers = result.cancer.biomarkers
        assert "EGFR mutant" in biomarkers, (
            f"Expected 'EGFR mutant' in biomarkers, got {biomarkers}"
        )
        assert "EGFR wild-type" not in biomarkers, (
            f"'EGFR wild-type' should NOT be in biomarkers, got {biomarkers}"
        )


# ======================================================================
# Test 1b: BRCA polarity flip (Bug 2, Property 1)
# ======================================================================

class TestBRCAPolarityFlip:
    """
    Bug 2: 'BRCA2-mutated' extracts 'BRCA wild-type' instead of
    'BRCA mutant' due to the same hyphen-as-negation bug.

    **Validates: Requirements 2.2**
    """

    def test_1b_brca2_mutated_extracts_positive_polarity(self):
        result = structure_query_fast("BRCA2-mutated prostate cancer")
        biomarkers = result.cancer.biomarkers
        assert "BRCA mutant" in biomarkers, (
            f"Expected 'BRCA mutant' in biomarkers, got {biomarkers}"
        )
        assert "BRCA wild-type" not in biomarkers, (
            f"'BRCA wild-type' should NOT be in biomarkers, got {biomarkers}"
        )


# ======================================================================
# Test 1c: Hypothesis property — hyphenated positive-polarity biomarkers
#          (Bugs 1-2, Property 1)
# ======================================================================

class TestBiomarkerPolarityProperty:
    """
    Property 1: For all genes in {EGFR, BRCA1, BRCA2, KRAS, BRAF} and
    qualifiers in {mutant, mutation, mutated}, structure_query_fast()
    must extract the positive-polarity canonical name and must NOT
    extract the wild-type canonical name.

    **Validates: Requirements 2.1, 2.2**
    """

    @given(
        gene=st.sampled_from(POSITIVE_POLARITY_GENES),
        qualifier=st.sampled_from(POSITIVE_POLARITY_QUALIFIERS),
    )
    @hyp_settings(max_examples=15, deadline=5000)
    def test_1c_hyphenated_positive_polarity_extracts_correct_canonical(
        self, gene, qualifier
    ):
        query = f"{gene}-{qualifier} cancer"
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers

        expected_canonical = GENE_TO_CANONICAL[gene]
        wildtype_canonical = GENE_TO_WILDTYPE[gene]

        assert expected_canonical in biomarkers, (
            f"Query '{query}': expected '{expected_canonical}' in "
            f"biomarkers, got {biomarkers}"
        )
        assert wildtype_canonical not in biomarkers, (
            f"Query '{query}': '{wildtype_canonical}' should NOT be in "
            f"biomarkers, got {biomarkers}"
        )


# ======================================================================
# Test 1d: Lung site misclassification (Bug 3, Property 3)
# ======================================================================

class TestSiteMisclassification:
    """
    Bug 3: 'adenocarcinoma of the right lower lobe, liver metastases,
    leptomeningeal spread' classifies as gi_hepatobiliary instead of
    lung because 'liver' matches gi_hepatobiliary before 'lobe' can
    match lung.

    **Validates: Requirements 2.3**
    """

    def test_1d_lung_primary_with_liver_mets_classifies_as_lung(self):
        result = structure_query_fast(
            "adenocarcinoma of the right lower lobe, liver metastases, "
            "leptomeningeal spread"
        )
        assert result.cancer.site == "lung", (
            f"Expected site='lung', got site='{result.cancer.site}'"
        )


# ======================================================================
# Test 1e: Category normalization — normalize_category doesn't exist yet
#          (Bug 4, Property 5)
# ======================================================================

class TestCategoryNormalization:
    """
    Bug 4: The cross-encoder gate compares 'prostate' vs
    'prostate_processed_documents' with exact string equality, applying
    a penalty to same-site documents.

    normalize_category() doesn't exist yet — this test will fail with
    ImportError or AssertionError until it's implemented.

    **Validates: Requirements 2.5**
    """

    def test_1e_prostate_category_normalization(self):
        try:
            from src.api.services.comprehensive_retrieval import (
                normalize_category,
            )
        except ImportError:
            pytest.fail(
                "normalize_category() not yet implemented in "
                "comprehensive_retrieval.py — Bug 4 unresolved"
            )

        assert normalize_category("prostate") == normalize_category(
            "prostate_processed_documents"
        ), (
            "normalize_category('prostate') should equal "
            "normalize_category('prostate_processed_documents')"
        )


# ======================================================================
# Test 1f: Classifier tie-breaking (Bug 5, Property 7)
# ======================================================================

class TestClassifierTieBreaking:
    """
    Bug 5: 'What is the best next-line systemic therapy for head and
    neck squamous cell carcinoma' classifies as 'staging' instead of
    'treatment_recommendation' because the PRIORITY list ranks staging
    above treatment_recommendation.

    **Validates: Requirements 2.8**
    """

    def test_1f_treatment_query_classifies_as_treatment_recommendation(self):
        result = classify_query(
            "What is the best next-line systemic therapy for "
            "head and neck squamous cell carcinoma"
        )
        assert result["primary_type"] == "treatment_recommendation", (
            f"Expected primary_type='treatment_recommendation', "
            f"got '{result['primary_type']}'"
        )


# ======================================================================
# Test 1g: Category scorer — head_neck vs h&n_processed_documents
#          (Bug 7, Property 5)
# ======================================================================

class TestCategoryScorerNormalization:
    """
    Bug 7: The notebook criteria scorer compares 'head_neck' vs
    'h&n_processed_documents' with exact string equality, producing
    0% category match for same-site documents.

    normalize_category() doesn't exist yet — this test will fail with
    ImportError or AssertionError until it's implemented.

    **Validates: Requirements 2.9**
    """

    def test_1g_head_neck_category_scorer_normalized_match(self):
        try:
            from src.api.services.comprehensive_retrieval import (
                normalize_category,
            )
        except ImportError:
            pytest.fail(
                "normalize_category() not yet implemented in "
                "comprehensive_retrieval.py — Bug 7 unresolved"
            )

        assert normalize_category("head_neck") == normalize_category(
            "h&n_processed_documents"
        ), (
            "normalize_category('head_neck') should equal "
            "normalize_category('h&n_processed_documents')"
        )
