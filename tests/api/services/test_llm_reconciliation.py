"""
LLM Reconciliation Unit Tests — Fix 3: Reconcile regex vs LLM extraction.

Tests that when regex and LLM disagree on biomarkers, the merged result
prefers LLM output for biomarkers. When they agree, the merged result is
unchanged. Site extraction prefers regex output (after Fix 2). The
reconciliation is gated behind a flag (use_llm_primary_extraction=True).

**Validates: Requirements 2.4**
"""

import pytest
from copy import deepcopy

from src.api.services.query_structuring_service import (
    merge_llm_extraction,
    QueryStructure,
    CancerContext,
    PatientContext,
    TreatmentContext,
    ClinicalHistory,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_structure(
    biomarkers: list[str] | None = None,
    site: str | None = None,
    filter_category: str | None = None,
) -> QueryStructure:
    """Build a minimal QueryStructure with the given regex-extracted fields."""
    s = QueryStructure(original_query="test query")
    s.cancer = CancerContext(
        site=site,
        biomarkers=list(biomarkers) if biomarkers else [],
    )
    s.filter_category = filter_category
    s.boost_terms = []
    return s


# ======================================================================
# Biomarker disagreement: LLM wins
# ======================================================================

class TestBiomarkerDisagreementLLMWins:
    """
    When regex extracts a wrong-polarity biomarker but LLM extracts the
    correct one, the merged result must use the LLM biomarker output.

    **Validates: Requirements 2.4**
    """

    def test_egfr_wildtype_regex_vs_mutant_llm(self):
        """Regex says EGFR wild-type, LLM says EGFR mutant → use LLM."""
        structure = _make_structure(
            biomarkers=["EGFR wild-type"],
            site="lung",
            filter_category="lung",
        )
        llm_result = {
            "biomarker_profile": "EGFR mutant (exon 19 del)",
            "primary_cancer": "lung adenocarcinoma",
        }

        merged = merge_llm_extraction(structure, llm_result)

        # The merged biomarkers must contain the LLM's positive-polarity marker
        assert any("EGFR" in b and "mutant" in b.lower() for b in merged.cancer.biomarkers), (
            f"Expected EGFR mutant from LLM in biomarkers, got {merged.cancer.biomarkers}"
        )
        # The wrong-polarity regex marker must be removed
        assert "EGFR wild-type" not in merged.cancer.biomarkers, (
            f"EGFR wild-type should have been replaced by LLM output, "
            f"got {merged.cancer.biomarkers}"
        )

    def test_brca_wildtype_regex_vs_mutant_llm(self):
        """Regex says BRCA wild-type, LLM says BRCA2 mutant → use LLM."""
        structure = _make_structure(
            biomarkers=["BRCA wild-type"],
            site="prostate",
            filter_category="prostate",
        )
        llm_result = {
            "biomarker_profile": "BRCA2 mutant",
            "primary_cancer": "prostate adenocarcinoma",
        }

        merged = merge_llm_extraction(structure, llm_result)

        assert any("BRCA" in b and "mutant" in b.lower() for b in merged.cancer.biomarkers), (
            f"Expected BRCA mutant from LLM in biomarkers, got {merged.cancer.biomarkers}"
        )
        assert "BRCA wild-type" not in merged.cancer.biomarkers, (
            f"BRCA wild-type should have been replaced by LLM output, "
            f"got {merged.cancer.biomarkers}"
        )


# ======================================================================
# Agreement: merged result unchanged
# ======================================================================

class TestAgreementPreservation:
    """
    When regex and LLM agree on biomarkers, the merged result must be
    unchanged (no duplication, no removal).

    **Validates: Requirements 2.4**
    """

    def test_both_agree_egfr_mutant(self):
        """Both regex and LLM say EGFR mutant → merged keeps EGFR mutant."""
        structure = _make_structure(
            biomarkers=["EGFR mutant"],
            site="lung",
            filter_category="lung",
        )
        llm_result = {
            "biomarker_profile": "EGFR mutant",
            "primary_cancer": "lung adenocarcinoma",
        }

        merged = merge_llm_extraction(structure, llm_result)

        assert "EGFR mutant" in merged.cancer.biomarkers, (
            f"Expected 'EGFR mutant' preserved, got {merged.cancer.biomarkers}"
        )
        # Should not have duplicates
        egfr_count = sum(1 for b in merged.cancer.biomarkers if "EGFR" in b and "mutant" in b.lower())
        assert egfr_count == 1, (
            f"Expected exactly 1 EGFR mutant entry, got {egfr_count} in {merged.cancer.biomarkers}"
        )

    def test_both_agree_brca_mutant(self):
        """Both regex and LLM say BRCA mutant → merged keeps BRCA mutant."""
        structure = _make_structure(
            biomarkers=["BRCA mutant"],
            site="prostate",
            filter_category="prostate",
        )
        llm_result = {
            "biomarker_profile": "BRCA2 mutant",
            "primary_cancer": "prostate adenocarcinoma",
        }

        merged = merge_llm_extraction(structure, llm_result)

        assert any("BRCA" in b and "mutant" in b.lower() for b in merged.cancer.biomarkers), (
            f"Expected BRCA mutant preserved, got {merged.cancer.biomarkers}"
        )


# ======================================================================
# Site: regex wins (after Fix 2)
# ======================================================================

class TestSiteRegexPreferred:
    """
    Site extraction must prefer the regex output (after Fix 2 made regex
    site extraction correct) over LLM for site.

    **Validates: Requirements 2.4**
    """

    def test_regex_site_preserved_over_llm(self):
        """Regex says lung, LLM primary_cancer says 'hepatic adenocarcinoma' → keep lung."""
        structure = _make_structure(
            biomarkers=["EGFR mutant"],
            site="lung",
            filter_category="lung",
        )
        llm_result = {
            "primary_cancer": "hepatic adenocarcinoma",
            "biomarker_profile": "EGFR mutant",
        }

        merged = merge_llm_extraction(structure, llm_result)

        assert merged.cancer.site == "lung", (
            f"Expected site='lung' (regex), got site='{merged.cancer.site}'"
        )
        assert merged.filter_category == "lung", (
            f"Expected filter_category='lung', got '{merged.filter_category}'"
        )

    def test_regex_site_not_overwritten_by_llm_primary_cancer(self):
        """LLM primary_cancer should populate raw_text but not override site."""
        structure = _make_structure(
            site="prostate",
            filter_category="prostate",
        )
        llm_result = {
            "primary_cancer": "prostate adenocarcinoma with bone involvement",
        }

        merged = merge_llm_extraction(structure, llm_result)

        assert merged.cancer.site == "prostate", (
            f"Expected site='prostate', got site='{merged.cancer.site}'"
        )


# ======================================================================
# Flag gating: use_llm_primary_extraction
# ======================================================================

class TestReconciliationFlagGating:
    """
    Reconciliation must be gated behind use_llm_primary_extraction flag.
    When False, regex wins for everything (old behavior).

    **Validates: Requirements 2.4**
    """

    def test_flag_true_llm_biomarkers_win(self):
        """With flag=True (default), LLM biomarkers override regex on disagreement."""
        structure = _make_structure(
            biomarkers=["EGFR wild-type"],
            site="lung",
            filter_category="lung",
        )
        llm_result = {
            "biomarker_profile": "EGFR mutant",
            "primary_cancer": "lung adenocarcinoma",
        }

        merged = merge_llm_extraction(
            structure, llm_result, use_llm_primary_extraction=True
        )

        assert any("EGFR" in b and "mutant" in b.lower() for b in merged.cancer.biomarkers), (
            f"With flag=True, expected LLM EGFR mutant, got {merged.cancer.biomarkers}"
        )
        assert "EGFR wild-type" not in merged.cancer.biomarkers

    def test_flag_false_regex_biomarkers_win(self):
        """With flag=False, regex biomarkers are kept even when LLM disagrees."""
        structure = _make_structure(
            biomarkers=["EGFR wild-type"],
            site="lung",
            filter_category="lung",
        )
        llm_result = {
            "biomarker_profile": "EGFR mutant",
            "primary_cancer": "lung adenocarcinoma",
        }

        merged = merge_llm_extraction(
            structure, llm_result, use_llm_primary_extraction=False
        )

        # With flag=False, regex wins — EGFR wild-type stays
        assert "EGFR wild-type" in merged.cancer.biomarkers, (
            f"With flag=False, expected regex 'EGFR wild-type' preserved, "
            f"got {merged.cancer.biomarkers}"
        )
