"""
Tests for the primary-vs-metastatic site precedence fix in
src/api/services/query_structuring_service.py.

Background
----------
An earlier version of the structurer used "first pattern match wins" to
assign ``cancer.site``. That treated "hepatic metastases" and "brain
metastasis" as candidate primary sites, which silently misrouted queries
whose metastatic descriptions happened to appear before (or in place of)
their primary-tumor description.

These tests pin the corrected behaviour:

  - A site match in a metastatic context window is NOT assigned to
    ``cancer.site``; it goes into ``cancer.metastatic_sites_detected``.
  - If every site match is metastatic, ``cancer.site`` stays ``None``.
  - ``merge_llm_extraction`` recovers the primary site from the LLM's
    ``primary_cancer`` axis when the regex pass left site unset.
  - The regression case from the field test — "mediastinal mass... adenocarcinoma...
    hepatic metastases" — resolves to lung after regex+LLM merge, not
    ``gi_hepatobiliary``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.api.services.query_structuring_service import (
    CANCER_SITE_PATTERNS,
    METASTATIC_CONTEXT_PATTERN,
    merge_llm_extraction,
    structure_query_fast,
)


# ── Regression: the exact field-test query ──────────────────────────────

_MEDIASTINAL_ADENO_QUERY = (
    "A 65-year old smoker was found to have an asymptomatic mediastinal "
    "mass on screening low dose CT imaging that was biopsy proven to be "
    "an adenocarcinoma. Imaging showed numerous hepatic metastases and "
    "no brain metastasis. What is the recommended next step in management?"
)


def test_mediastinal_mass_with_hepatic_mets_resolves_to_lung_after_llm_merge():
    """The exact regression scenario from the field test.

    Regex alone: `mediastinal mass` triggers the lung pattern → site=lung.
    Even without the LLM, the query should NOT be routed to GI/hepatobiliary.
    """
    structure = structure_query_fast(_MEDIASTINAL_ADENO_QUERY, "general")
    assert structure.cancer.site == "lung", (
        f"Expected site='lung' for a mediastinal-mass adenocarcinoma query, "
        f"got site={structure.cancer.site!r}"
    )
    assert structure.filter_category == "lung"
    # Hepatic and brain sites should be recorded as metastatic, not primary
    assert "gi_hepatobiliary" in structure.cancer.metastatic_sites_detected
    assert "cns" in structure.cancer.metastatic_sites_detected


def test_simple_primary_query_still_works():
    """Baseline: a plain 'lung cancer treatment' query still resolves to lung."""
    structure = structure_query_fast("lung cancer treatment", "general")
    assert structure.cancer.site == "lung"
    assert structure.filter_category == "lung"
    assert structure.cancer.metastatic_sites_detected == []


def test_mets_only_query_leaves_primary_site_none_from_regex():
    """A query describing only metastatic sites (no primary mention) must
    leave cancer.site = None so the LLM merge step can recover the primary.

    (The LLM merge recovery path is exercised by
    ``test_llm_primary_cancer_recovers_site``.)
    """
    q = "patient with hepatic metastases and brain metastasis, progression on chemo"
    structure = structure_query_fast(q, "general")
    assert structure.cancer.site is None
    assert structure.filter_category is None
    # Both matches were correctly categorized as metastatic
    assert "gi_hepatobiliary" in structure.cancer.metastatic_sites_detected
    assert "cns" in structure.cancer.metastatic_sites_detected


def test_llm_primary_cancer_recovers_site_when_regex_left_none():
    """When regex produces no primary site but the LLM extractor supplies a
    primary_cancer string naming a site, merge_llm_extraction must set
    cancer.site and filter_category from that string.
    """
    q = "patient with hepatic metastases and bone mets"
    structure = structure_query_fast(q, "general")
    assert structure.cancer.site is None  # regex-only: no primary

    merged = merge_llm_extraction(
        structure,
        {"primary_cancer": "adenocarcinoma of the lung"},
    )
    assert merged.cancer.site == "lung"
    assert merged.filter_category == "lung"


def test_llm_primary_cancer_does_not_override_existing_regex_site():
    """If regex already extracted a primary site, the LLM-merge recovery
    must NOT overwrite it. (Regex is the ground truth when it fires.)"""
    structure = structure_query_fast(
        "lung cancer with liver metastases", "general"
    )
    assert structure.cancer.site == "lung"
    merged = merge_llm_extraction(
        structure,
        {"primary_cancer": "prostate adenocarcinoma"},  # contradictory; ignore
    )
    assert merged.cancer.site == "lung"


def test_primary_site_wins_when_query_names_both_primary_and_mets():
    """'Lung cancer with brain metastases' — lung is primary, CNS is mets."""
    structure = structure_query_fast(
        "lung cancer with brain metastases", "general"
    )
    assert structure.cancer.site == "lung"
    assert "cns" in structure.cancer.metastatic_sites_detected


def test_lung_pattern_covers_mediastinal_mass():
    """Fix 2: the lung regex must fire on 'mediastinal mass' even without
    the word 'lung' appearing in the query."""
    assert CANCER_SITE_PATTERNS["lung"]["pattern"].search(
        "an asymptomatic mediastinal mass on CT"
    )
    assert CANCER_SITE_PATTERNS["lung"]["pattern"].search("hilar mass")
    assert CANCER_SITE_PATTERNS["lung"]["pattern"].search("pulmonary nodule")
    assert CANCER_SITE_PATTERNS["lung"]["pattern"].search("left upper lobe mass")


def test_metastatic_trailing_pattern_matches_common_shapes():
    """The trailing pattern is scoped to site+mets shapes that appear
    right after a site mention. Leading constructs ("spread to the
    liver", "distant disease") are intentionally NOT handled yet —
    adding them requires a separate leading-window pass that can be
    done later without touching this test."""
    for phrase in (
        "metastases",
        "metastasis",
        "metastatic",
        "mets",
        "metastasized",
    ):
        assert METASTATIC_CONTEXT_PATTERN.search(phrase), (
            f"METASTATIC_CONTEXT_PATTERN failed to match {phrase!r}"
        )


def test_lobe_in_primary_context_still_assigns_lung():
    """'Left upper lobe adenocarcinoma' is a PRIMARY lung lesion, not mets."""
    structure = structure_query_fast(
        "left upper lobe adenocarcinoma, T2N0M0", "general"
    )
    assert structure.cancer.site == "lung"
    assert structure.cancer.metastatic_sites_detected == []


def test_brain_mets_from_prostate_does_not_become_cns_primary():
    """Ensure CNS mets never overrides a prostate primary."""
    structure = structure_query_fast(
        "metastatic prostate cancer with brain metastases",
        "general",
    )
    assert structure.cancer.site == "prostate"
    assert "cns" in structure.cancer.metastatic_sites_detected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
