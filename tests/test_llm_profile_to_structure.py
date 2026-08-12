"""
Tests for LLM → ClinicalProfile → QueryStructure normalization wiring.

These pin the behaviour of two new helpers that are about to be used in
``EnhancedRAGService.query()``:

1. ``clinical_extractor.build_profile_from_llm_result()``  — takes an
   already-computed 8-axis LLM dict (from ``structure_query_with_llm``)
   and produces a ``ClinicalProfile`` normalized via ``SynonymIndex``.
   Crucially: no second LLM call.

2. ``clinical_extractor.apply_profile_to_structure()`` — fills gaps in a
   regex-produced ``QueryStructure`` using the canonical values from a
   ``ClinicalProfile``. Never overrides a strong regex extraction with a
   contradictory LLM value; only fills when the regex left a field empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.api.services.clinical_extractor import (
    ClinicalProfile,
    apply_profile_to_structure,
    build_profile_from_llm_result,
)
from src.api.services.query_structuring_service import (
    structure_query_fast,
)


# ── build_profile_from_llm_result ───────────────────────────────────────

def test_llm_primary_cancer_lung_resolves_to_canonical():
    """LLM says 'adenocarcinoma of the lung' → ClinicalProfile gets
    cancer_type_label='Lung Cancer' via SynonymIndex."""
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={"primary_cancer": "adenocarcinoma of the lung"},
    )
    assert profile.cancer_type_label == "Lung Cancer"
    assert profile.cancer_type_key == "lung"


def test_llm_primary_cancer_breast_resolves_to_canonical():
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={"primary_cancer": "invasive ductal carcinoma of the left breast"},
    )
    assert profile.cancer_type_label == "Breast Cancer"


def test_llm_primary_cancer_hnscc_resolves_to_h_n():
    """HNSCC abbreviation should resolve to Head and Neck Cancer."""
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={"primary_cancer": "HNSCC of the oral tongue"},
    )
    assert profile.cancer_type_label == "Head and Neck Cancer"


def test_llm_primary_cancer_unknown_leaves_cancer_type_none():
    """If LLM gives something the synonym index can't resolve, don't
    invent a cancer type."""
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={"primary_cancer": "gibberish entity"},
    )
    assert profile.cancer_type_label is None


def test_llm_histology_resolves_through_primary_cancer():
    """'SCC' in primary_cancer resolves to canonical histology."""
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={"primary_cancer": "SCC of the oral tongue"},
    )
    assert "Squamous cell carcinoma" in profile.histologies


def test_llm_biomarker_profile_resolves_canonicals():
    """LLM says 'CPS 20, PD-L1 positive, HPV negative' → canonical biomarker names."""
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={"biomarker_profile": "CPS score of 20 and PD-L1 positive, HPV negative"},
    )
    # Should include CPS and PD-L1; HPV status goes to biomarker_expressions
    names = {b for b in profile.biomarkers}
    assert "CPS" in names
    assert "PD-L1" in names


def test_empty_llm_result_returns_empty_profile():
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result={},
    )
    assert profile.cancer_type_label is None
    assert profile.histologies == []
    assert profile.biomarkers == []


def test_none_llm_result_returns_empty_profile():
    profile = build_profile_from_llm_result(
        query_text="placeholder",
        llm_result=None,
    )
    assert profile.cancer_type_label is None


# ── apply_profile_to_structure ──────────────────────────────────────────

def test_apply_profile_fills_cancer_site_when_regex_missed_it():
    """The bread-and-butter case: regex couldn't extract site from a
    complex query; LLM profile has a canonical label. apply should
    populate cancer.site + filter_category."""
    structure = structure_query_fast("patient with hepatic metastases and bone mets", "general")
    assert structure.cancer.site is None  # regex-only, no primary found

    profile = ClinicalProfile(
        cancer_type_label="Lung Cancer",
        cancer_type_key="lung",
    )
    updated = apply_profile_to_structure(structure, profile)
    assert updated.cancer.site == "lung"
    assert updated.filter_category == "lung"


def test_apply_profile_does_not_override_existing_regex_site():
    """If regex already extracted cancer.site, the LLM profile must not
    override it with a contradictory value."""
    structure = structure_query_fast("lung cancer with liver metastases", "general")
    assert structure.cancer.site == "lung"

    profile = ClinicalProfile(
        cancer_type_label="Breast Cancer",  # contradictory to regex
        cancer_type_key="breast",
    )
    updated = apply_profile_to_structure(structure, profile)
    assert updated.cancer.site == "lung"  # regex wins


def test_apply_profile_fills_histology_when_regex_missed_it():
    """Histology isn't always extracted by regex; profile can fill it."""
    structure = structure_query_fast("patient with lung tumor", "general")
    # Regex might or might not set histology; after applying profile
    # with a populated histology, it should be set.
    profile = ClinicalProfile(
        histologies=["Adenocarcinoma"],
    )
    updated = apply_profile_to_structure(structure, profile)
    # If regex didn't set it, profile fills it
    if not structure_query_fast("patient with lung tumor", "general").cancer.histology:
        assert updated.cancer.histology in ("Adenocarcinoma", "adenocarcinoma")


def test_apply_profile_merges_biomarkers_without_clobbering():
    """If regex got HER2 and profile has CPS, result should have both."""
    structure = structure_query_fast("HER2+ breast cancer", "general")
    original_bio = list(structure.cancer.biomarkers)

    profile = ClinicalProfile(biomarkers=["CPS", "PD-L1"])
    updated = apply_profile_to_structure(structure, profile)
    # Everything that was there is still there
    for b in original_bio:
        assert b in updated.cancer.biomarkers
    # New canonicals are added
    assert "CPS" in updated.cancer.biomarkers
    assert "PD-L1" in updated.cancer.biomarkers


def test_apply_empty_profile_is_noop():
    """Passing in an empty profile doesn't change anything."""
    structure = structure_query_fast("lung cancer treatment", "general")
    snapshot = (
        structure.cancer.site,
        structure.filter_category,
        tuple(structure.cancer.biomarkers),
    )
    updated = apply_profile_to_structure(structure, ClinicalProfile())
    assert (
        updated.cancer.site,
        updated.filter_category,
        tuple(updated.cancer.biomarkers),
    ) == snapshot


# ── End-to-end integration ──────────────────────────────────────────────

def test_regression_mediastinal_mass_with_llm_result_produces_lung_profile():
    """The exact field-test scenario, validated end-to-end:
    regex extracts site via Tier A fix (mediastinal mass → lung),
    LLM returns primary_cancer='adenocarcinoma', profile resolution
    agrees with regex, apply_profile_to_structure preserves regex."""
    query = (
        "A 65-year old smoker was found to have an asymptomatic mediastinal "
        "mass on screening low dose CT imaging that was biopsy proven to be "
        "an adenocarcinoma. Imaging showed numerous hepatic metastases."
    )
    structure = structure_query_fast(query, "general")
    assert structure.cancer.site == "lung"  # Tier A fix

    # Simulate what structure_query_with_llm + merge_llm_extraction
    # would have produced in parallel:
    llm_result = {
        "primary_cancer": "adenocarcinoma",
        "metastatic_concern": "numerous hepatic metastases",
        "patient_factors": "65-year old smoker",
    }
    profile = build_profile_from_llm_result(query, llm_result)
    # 'adenocarcinoma' alone doesn't resolve cancer_type (good — no
    # cancer can be uniquely inferred from histology alone)
    assert profile.cancer_type_label is None
    # But it DOES resolve histology
    assert "Adenocarcinoma" in profile.histologies

    updated = apply_profile_to_structure(structure, profile)
    assert updated.cancer.site == "lung"  # unchanged — regex already right


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
