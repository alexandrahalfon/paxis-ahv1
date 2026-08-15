"""
Classifier Tie-Breaking Unit Tests

Tests that classify_query() correctly returns "treatment_recommendation"
for treatment-seeking queries, and preserves correct classification for
unambiguous staging and dose queries.

**Validates: Requirements 2.8, 3.6, 3.7**
"""

import pytest

from src.api.services.enhanced_rag_service import classify_query


# ======================================================================
# Treatment-seeking queries that must classify as treatment_recommendation
# ======================================================================

class TestTreatmentQueryClassification:
    """
    Treatment-seeking queries containing language like "best next-line
    systemic therapy" must classify as 'treatment_recommendation', not
    'staging' or any other type.

    **Validates: Requirements 2.8**
    """

    def test_hn_scc_treatment_query(self):
        """H&N SCC treatment query classifies as treatment_recommendation."""
        result = classify_query(
            "What is the best next-line systemic therapy for "
            "head and neck squamous cell carcinoma"
        )
        assert result["primary_type"] == "treatment_recommendation", (
            f"Expected 'treatment_recommendation', "
            f"got '{result['primary_type']}' with scores={result['scores']}"
        )

    def test_egfr_lung_treatment_query(self):
        """EGFR-mutant lung adenocarcinoma treatment query classifies as treatment_recommendation."""
        result = classify_query(
            "What is the best next-line systemic therapy for "
            "EGFR-mutant lung adenocarcinoma"
        )
        assert result["primary_type"] == "treatment_recommendation", (
            f"Expected 'treatment_recommendation', "
            f"got '{result['primary_type']}' with scores={result['scores']}"
        )

    def test_brca2_prostate_treatment_query(self):
        """BRCA2-mutated prostate cancer treatment query classifies as treatment_recommendation."""
        result = classify_query(
            "What is the best next-line systemic therapy for "
            "BRCA2-mutated prostate cancer"
        )
        assert result["primary_type"] == "treatment_recommendation", (
            f"Expected 'treatment_recommendation', "
            f"got '{result['primary_type']}' with scores={result['scores']}"
        )

    def test_treatment_queries_confidence_above_threshold(self):
        """Multiple treatment-seeking queries return confidence > 0.5."""
        queries = [
            "What is the best next-line systemic therapy for head and neck squamous cell carcinoma",
            "What is the best next-line systemic therapy for EGFR-mutant lung adenocarcinoma",
            "What is the best next-line systemic therapy for BRCA2-mutated prostate cancer",
        ]
        for query in queries:
            result = classify_query(query)
            assert result["primary_type"] == "treatment_recommendation", (
                f"Query '{query[:50]}...': expected 'treatment_recommendation', "
                f"got '{result['primary_type']}'"
            )
            assert result["confidence"] > 0.5, (
                f"Query '{query[:50]}...': expected confidence > 0.5, "
                f"got {result['confidence']}"
            )


# ======================================================================
# Preservation: unambiguous staging queries must still return "staging"
# ======================================================================

class TestStagingPreservation:
    """
    Unambiguous staging queries must continue to classify as 'staging'
    after the tie-breaking fix.

    **Validates: Requirements 3.6**
    """

    def test_unambiguous_staging_query(self):
        """Staging query with TNM notation classifies as staging."""
        result = classify_query("What stage is T2N1M0 NSCLC?")
        assert result["primary_type"] == "staging", (
            f"Expected 'staging', got '{result['primary_type']}' "
            f"with scores={result['scores']}"
        )


# ======================================================================
# Preservation: unambiguous dose queries must still return "dose_question"
# ======================================================================

class TestDosePreservation:
    """
    Unambiguous dose queries must continue to classify as 'dose_question'
    after the tie-breaking fix.

    **Validates: Requirements 3.7**
    """

    def test_unambiguous_dose_query(self):
        """Dose query with Gy notation classifies as dose_question."""
        result = classify_query(
            "What dose should be given for breast cancer? 50 Gy in 25 fractions?"
        )
        assert result["primary_type"] == "dose_question", (
            f"Expected 'dose_question', got '{result['primary_type']}' "
            f"with scores={result['scores']}"
        )
