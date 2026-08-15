"""
Tests for the two downstream fixes on top of the primary-vs-mets
Tier A work:

(a) ``query_structure.filter_category`` as a fallback for the Qdrant
    dense category filter when the caller doesn't pass an explicit one.
    This is tested at the structurer level — the actual Qdrant filter
    construction lives in EnhancedRAGService.query() and is wired to
    consume ``query_structure.filter_category`` when ``category is None``.

(b) ``filter_metastatic_site_canonicals()`` — strip canonical site labels
    whose only occurrences in the query sit in a metastatic-context
    window. Prevents Resolver Expansion from injecting "Liver" /
    "Mediastinum" into the embedding of a lung-primary query with
    hepatic mets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.api.services.query_structuring_service import (
    filter_metastatic_site_canonicals,
    structure_query_fast,
)


# ── (a) filter_category is populated for use as Qdrant filter fallback ──

def test_structure_exposes_filter_category_for_obvious_query():
    """'lung cancer treatment' → filter_category='lung'. Callers can
    use this as a Qdrant category filter when no explicit one is set."""
    s = structure_query_fast("lung cancer treatment", "general")
    assert s.filter_category == "lung"


def test_structure_exposes_filter_category_for_mediastinal_mass_query():
    """Tier A regression: mediastinal-mass query must surface
    filter_category='lung' for downstream Qdrant filtering."""
    q = (
        "A 65-year old smoker was found to have an asymptomatic mediastinal "
        "mass ... biopsy proven adenocarcinoma. Numerous hepatic metastases."
    )
    s = structure_query_fast(q, "general")
    assert s.filter_category == "lung"


def test_structure_filter_category_none_for_mets_only_query():
    """Mets-only query → filter_category None (the hepatic mets mention
    must NOT make this a 'gi' query)."""
    s = structure_query_fast("patient with hepatic metastases", "general")
    assert s.filter_category is None


# ── (b) Resolver-expansion metastatic filter ────────────────────────────

def test_filter_metastatic_hepatic_mets_drops_liver_canonical():
    """Query says 'hepatic metastases' → resolver would emit 'Liver'
    as a canonical site. The filter drops it because the only
    occurrence is in a metastatic-context window."""
    canonicals = ["Liver", "Lung"]
    query = "lung primary with hepatic metastases"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    assert "Liver" not in filtered
    # Lung appears as primary (no trailing mets word) → preserved
    assert "Lung" in filtered


def test_filter_metastatic_preserves_primary_liver_cancer():
    """'hepatocellular carcinoma of the liver' → 'Liver' is primary,
    must not be dropped."""
    canonicals = ["Liver"]
    query = "patient with hepatocellular carcinoma of the liver"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    assert "Liver" in filtered


def test_filter_metastatic_brain_mets_drops_brain_canonical():
    """'brain metastasis' → drop 'Brain'."""
    canonicals = ["Brain", "Lung"]
    query = "NSCLC with brain metastasis"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    assert "Brain" not in filtered
    assert "Lung" in filtered


def test_filter_metastatic_keeps_all_when_no_mets_context():
    """No mets words in the query → all canonicals preserved."""
    canonicals = ["Lung", "Liver", "Brain"]
    query = "comparison of pulmonary vs hepatic vs cerebral primary tumors"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    assert filtered == canonicals


def test_filter_metastatic_drops_all_in_mets_only_query():
    """'hepatic mets and brain metastasis' → all metastatic → drop all."""
    canonicals = ["Liver", "Brain"]
    query = "hepatic mets and brain metastasis"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    assert filtered == []


def test_filter_metastatic_preserves_sites_not_mentioned():
    """'Pancreas' in canonicals but not in query → preserved (we only
    strip if a variant appears in a mets context, not based on absence)."""
    canonicals = ["Pancreas", "Liver"]
    query = "patient with hepatic metastases"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    # Liver mentioned via "hepatic" in mets context → dropped
    assert "Liver" not in filtered
    # Pancreas not in query at all → preserved (variant never appeared)
    assert "Pancreas" in filtered


def test_filter_metastatic_empty_input_returns_empty():
    assert filter_metastatic_site_canonicals("anything", []) == []
    assert filter_metastatic_site_canonicals("", ["Lung"]) == ["Lung"]


def test_filter_metastatic_mixed_primary_and_mets_mentions_keep_canonical():
    """If a canonical appears BOTH as primary and mets in the same
    query, we keep it — the user IS talking about the site in a
    primary sense somewhere."""
    canonicals = ["Lung"]
    query = "primary lung tumor with pulmonary metastases in other lobe"
    filtered = filter_metastatic_site_canonicals(query, canonicals)
    # 'lung' in 'primary lung tumor' has no trailing mets word → primary
    assert "Lung" in filtered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
