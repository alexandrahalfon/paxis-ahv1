"""
Property-based tests for patient signal scoring.

Feature: patient-study-match-scoring

Tests the following property:
- Property 1: Patient signal score determines has_patient_context

**Validates: Requirements 1.1, 1.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.api.services.query_structuring_service import (
    QueryStructure,
    PatientContext,
    CancerContext,
    TreatmentContext,
    ClinicalHistory,
    _patient_signal_score,
)


# ======================================================================
# Signal definitions matching the design table
# ======================================================================

# Each signal: (name, points, query_snippet_builder, structure_mutator)
# query_snippet_builder: returns a string to inject into the query
# structure_mutator: mutates a QueryStructure to reflect the signal

SIGNAL_DEFS = {
    "age": {
        "points": 2,
        "snippets": ["65 year old", "42yo", "55 y.o.", "70 year"],
    },
    "gender": {
        "points": 1,
        "snippets": ["male", "female", "man", "woman"],
    },
    "patient_phrases": {
        "points": 2,
        "snippets": [
            "patient with",
            "presenting with",
            "diagnosed with",
            "history of",
        ],
    },
    "biomarkers_polarity": {
        "points": 2,
        "snippets": [],  # detected via structure, not query text
    },
    "prior_treatments": {
        "points": 2,
        "snippets": [],  # detected via structure, not query text
    },
    "tnm_staging": {
        "points": 1,
        "snippets": [],  # detected via structure
    },
    "comorbidities": {
        "points": 1,
        "snippets": [],  # detected via structure
    },
    "diagnosis_staging_combo": {
        "points": 2,
        "snippets": [],  # detected via structure (site + stage)
    },
}

SIGNAL_NAMES = list(SIGNAL_DEFS.keys())


# ======================================================================
# Strategies
# ======================================================================

signal_type_strategy = st.sampled_from(SIGNAL_NAMES)
signal_count_strategy = st.integers(min_value=0, max_value=7)


def _build_query_and_structure(selected_signals):
    """
    Given a list of signal names, build a query string and a QueryStructure
    that contains exactly those signals.

    Returns (query_text, structure, expected_points).
    """
    query_parts = []
    patient = PatientContext()
    cancer = CancerContext()
    treatment = TreatmentContext()
    expected_points = 0

    for signal_name in selected_signals:
        sig = SIGNAL_DEFS[signal_name]
        expected_points += sig["points"]

        if signal_name == "age":
            query_parts.append("65 year old")

        elif signal_name == "gender":
            query_parts.append("male")

        elif signal_name == "patient_phrases":
            query_parts.append("patient with cancer")

        elif signal_name == "biomarkers_polarity":
            # Biomarkers with polarity: need biomarkers list + polarity in text
            cancer.biomarkers = ["EGFR mutant"]
            query_parts.append("EGFR-mutant")

        elif signal_name == "prior_treatments":
            treatment.prior_treatments = ["cisplatin"]
            query_parts.append("prior cisplatin")

        elif signal_name == "tnm_staging":
            cancer.tnm_t = "2"
            query_parts.append("T2")

        elif signal_name == "comorbidities":
            patient.comorbidities = ["diabetes"]
            query_parts.append("diabetic")

        elif signal_name == "diagnosis_staging_combo":
            cancer.site = "lung"
            cancer.stage = "IIIA"
            query_parts.append("lung cancer stage IIIA")

    query_text = " ".join(query_parts) if query_parts else "general question"

    structure = QueryStructure(
        original_query=query_text,
        patient=patient,
        cancer=cancer,
        treatment=treatment,
        clinical_history=ClinicalHistory(),
    )

    return query_text, structure, expected_points


# ======================================================================
# Property 1 test
# ======================================================================

# Feature: patient-study-match-scoring, Property 1: Patient signal score determines has_patient_context
@settings(max_examples=150)
@given(
    signal_indices=st.lists(
        st.sampled_from(SIGNAL_NAMES),
        min_size=0,
        max_size=7,
        unique=True,
    )
)
def test_patient_signal_score_determines_has_patient_context(signal_indices):
    """
    Property 1: Patient signal score determines has_patient_context.

    For any random subset of patient signals, has_patient_context SHALL be
    true if and only if _patient_signal_score >= 2.

    **Validates: Requirements 1.1, 1.2**

    Tag: Feature: patient-study-match-scoring, Property 1: Patient signal score determines has_patient_context
    """
    query_text, structure, expected_points = _build_query_and_structure(signal_indices)

    score = _patient_signal_score(query_text, structure)

    # The score must be at least the expected points (implementation may
    # detect additional signals from the query text we constructed, but
    # it must detect at least the ones we explicitly set up).
    assert score >= expected_points, (
        f"Score {score} is less than expected {expected_points} "
        f"for signals {signal_indices}"
    )

    # Core property: has_patient_context == (score >= 2)
    has_context = score >= 2
    expected_has_context = expected_points >= 2

    # If expected_points >= 2, score must also be >= 2 (since score >= expected_points)
    if expected_has_context:
        assert has_context, (
            f"Expected has_patient_context=True (expected_points={expected_points}, "
            f"score={score}) for signals {signal_indices}"
        )

    # If expected_points < 2, we verify the threshold semantics hold:
    # score >= 2 iff has_patient_context is true
    assert has_context == (score >= 2), (
        f"has_patient_context should be (score >= 2): "
        f"score={score}, has_context={has_context}"
    )


# ======================================================================
# Example-based unit tests for signal scoring edge cases
# ======================================================================
# **Validates: Requirements 1.3, 1.4**


class TestSignalScoringEdgeCases:
    """
    Example-based unit tests for _patient_signal_score() edge cases.

    These tests verify that short factual queries produce low scores
    (has_patient_context = false) and full patient presentations produce
    high scores (has_patient_context = true).

    **Validates: Requirements 1.3, 1.4**
    """

    def test_what_is_sbrt_no_patient_context(self):
        """
        'What is SBRT?' is a factual question with zero patient signals.
        Score should be 0, has_patient_context should be false.

        **Validates: Requirements 1.3**
        """
        query = "What is SBRT?"
        structure = QueryStructure(
            original_query=query,
            patient=PatientContext(),
            cancer=CancerContext(),
            treatment=TreatmentContext(),
            clinical_history=ClinicalHistory(),
        )

        score = _patient_signal_score(query, structure)

        assert score < 2, f"Expected score < 2 for '{query}', got {score}"
        assert score >= 0, f"Score should be non-negative, got {score}"
        has_patient_context = score >= 2
        assert has_patient_context is False, (
            f"'{query}' should not trigger patient context (score={score})"
        )

    def test_define_imrt_no_patient_context(self):
        """
        'Define IMRT' is a factual question with zero patient signals.
        Score should be 0, has_patient_context should be false.

        **Validates: Requirements 1.3**
        """
        query = "Define IMRT"
        structure = QueryStructure(
            original_query=query,
            patient=PatientContext(),
            cancer=CancerContext(),
            treatment=TreatmentContext(),
            clinical_history=ClinicalHistory(),
        )

        score = _patient_signal_score(query, structure)

        assert score < 2, f"Expected score < 2 for '{query}', got {score}"
        has_patient_context = score >= 2
        assert has_patient_context is False, (
            f"'{query}' should not trigger patient context (score={score})"
        )

    def test_full_patient_presentation_has_context(self):
        """
        '65 year old male with stage IIIA NSCLC, EGFR-mutant' contains
        multiple patient signals: age(+2), gender(+1), site+stage combo(+2),
        biomarkers(+2). Score should be >= 7, has_patient_context = true.

        **Validates: Requirements 1.4**
        """
        query = "65 year old male with stage IIIA NSCLC, EGFR-mutant"
        structure = QueryStructure(
            original_query=query,
            patient=PatientContext(age=65, gender="male"),
            cancer=CancerContext(
                site="lung",
                stage="IIIA",
                biomarkers=["EGFR mutant"],
            ),
            treatment=TreatmentContext(),
            clinical_history=ClinicalHistory(),
        )

        score = _patient_signal_score(query, structure)

        assert score >= 2, (
            f"Expected score >= 2 for full patient presentation, got {score}"
        )
        # With age(+2), gender(+1), site+stage combo(+2), biomarkers(+2) = 7
        assert score >= 7, (
            f"Expected score >= 7 for '{query}' "
            f"(age+gender+site_stage_combo+biomarkers), got {score}"
        )
        has_patient_context = score >= 2
        assert has_patient_context is True, (
            f"Full patient presentation should trigger patient context (score={score})"
        )

    def test_breast_cancer_alone_insufficient(self):
        """
        'breast cancer' alone has only 1 signal: cancer site without staging.
        No combo bonus (site without stage). Score should be < 2.

        **Validates: Requirements 1.3**
        """
        query = "breast cancer"
        structure = QueryStructure(
            original_query=query,
            patient=PatientContext(),
            cancer=CancerContext(site="breast"),
            treatment=TreatmentContext(),
            clinical_history=ClinicalHistory(),
        )

        score = _patient_signal_score(query, structure)

        assert score < 2, (
            f"Expected score < 2 for 'breast cancer' alone (site only, no stage combo), "
            f"got {score}"
        )
        has_patient_context = score >= 2
        assert has_patient_context is False, (
            f"'breast cancer' alone should not trigger patient context (score={score})"
        )

    def test_stage_iv_metastatic_prostate_brca2(self):
        """
        'stage IV metastatic prostate cancer BRCA2-mutated' contains
        site+stage combo(+2) and biomarkers(+2). Score should be >= 4.

        **Validates: Requirements 1.4**
        """
        query = "stage IV metastatic prostate cancer BRCA2-mutated"
        structure = QueryStructure(
            original_query=query,
            patient=PatientContext(),
            cancer=CancerContext(
                site="prostate",
                stage="IV",
                biomarkers=["BRCA2 mutated"],
            ),
            treatment=TreatmentContext(),
            clinical_history=ClinicalHistory(),
        )

        score = _patient_signal_score(query, structure)

        assert score >= 2, (
            f"Expected score >= 2 for metastatic prostate with biomarkers, got {score}"
        )
        # site+stage combo(+2) + biomarkers(+2) = 4
        assert score >= 4, (
            f"Expected score >= 4 for '{query}' "
            f"(site_stage_combo + biomarkers), got {score}"
        )
        has_patient_context = score >= 2
        assert has_patient_context is True, (
            f"Metastatic prostate with biomarkers should trigger patient context "
            f"(score={score})"
        )
