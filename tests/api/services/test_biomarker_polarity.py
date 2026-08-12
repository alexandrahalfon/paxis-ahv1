"""
Biomarker Polarity Unit Tests — Fix 1

Tests that hyphenated positive-polarity biomarker terms extract the correct
positive-polarity canonical name, and that explicit negative-polarity terms
continue to extract the correct negative-polarity canonical name.

**Validates: Requirements 2.1, 2.2, 3.1, 3.2, 3.9**
"""

import pytest
from src.api.services.query_structuring_service import structure_query_fast


# ======================================================================
# Positive-polarity: hyphenated variants MUST extract positive canonical
# ======================================================================

class TestHyphenatedPositivePolarity:
    """
    Each hyphenated positive-polarity variant must extract the correct
    positive-polarity canonical name and must NOT extract wild-type.

    **Validates: Requirements 2.1, 2.2**
    """

    @pytest.mark.parametrize("query,expected_canonical,excluded_canonical", [
        ("EGFR-mutant lung cancer", "EGFR mutant", "EGFR wild-type"),
        ("EGFR-mutation lung cancer", "EGFR mutant", "EGFR wild-type"),
        ("EGFR-mutated lung cancer", "EGFR mutant", "EGFR wild-type"),
        ("BRCA2-mutated prostate cancer", "BRCA mutant", "BRCA wild-type"),
        ("BRCA1-mutated breast cancer", "BRCA mutant", "BRCA wild-type"),
        ("KRAS-mutant colorectal cancer", "KRAS mutant", "KRAS wild-type"),
        ("BRAF-mutated melanoma", "BRAF mutant", "BRAF wild-type"),
    ])
    def test_hyphenated_positive_polarity_extracts_correct_canonical(
        self, query, expected_canonical, excluded_canonical
    ):
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers
        assert expected_canonical in biomarkers, (
            f"Query '{query}': expected '{expected_canonical}' in "
            f"biomarkers, got {biomarkers}"
        )
        assert excluded_canonical not in biomarkers, (
            f"Query '{query}': '{excluded_canonical}' should NOT be in "
            f"biomarkers, got {biomarkers}"
        )

    def test_her2_positive_extracts_positive(self):
        # "HER2 positive" (no hyphen) should work via RECEPTOR_PATTERNS
        result = structure_query_fast("HER2 positive breast cancer")
        receptor = result.cancer.receptor_status or ""
        biomarkers = result.cancer.biomarkers
        assert "HER2+" in receptor or any(
            "HER2" in b and ("+" in b or "positive" in b.lower() or "amplified" in b.lower())
            for b in biomarkers
        ), (
            f"Expected HER2-positive marker, got biomarkers={biomarkers}, "
            f"receptor_status={receptor}"
        )

    def test_pdl1_cps_100_positive(self):
        result = structure_query_fast("PD-L1 CPS 100 head and neck cancer")
        biomarkers = result.cancer.biomarkers
        # Should extract PD-L1 related positive marker
        assert any("PD-L1" in b or "CPS" in b for b in biomarkers), (
            f"Expected PD-L1/CPS marker, got biomarkers={biomarkers}"
        )

    def test_egfr_mutant_no_hyphen_positive(self):
        result = structure_query_fast("EGFR mutant lung cancer")
        biomarkers = result.cancer.biomarkers
        assert "EGFR mutant" in biomarkers, (
            f"Expected 'EGFR mutant' in biomarkers, got {biomarkers}"
        )
        assert "EGFR wild-type" not in biomarkers, (
            f"'EGFR wild-type' should NOT be in biomarkers, got {biomarkers}"
        )

    def test_egfr_plus_positive(self):
        result = structure_query_fast("EGFR+ lung cancer")
        biomarkers = result.cancer.biomarkers
        assert "EGFR mutant" in biomarkers, (
            f"Expected 'EGFR mutant' in biomarkers, got {biomarkers}"
        )


# ======================================================================
# Negative-polarity: explicit wild-type/negative MUST still work
# ======================================================================

class TestNegativePolarityPreservation:
    """
    Explicit negative-polarity terms must continue to extract the correct
    negative-polarity canonical name after the fix.

    **Validates: Requirements 3.1, 3.2, 3.9**
    """

    @pytest.mark.parametrize("query,expected_canonical", [
        ("EGFR wild-type lung cancer", "EGFR wild-type"),
        ("EGFR wt lung cancer", "EGFR wild-type"),
        ("EGFR-negative lung cancer", "EGFR wild-type"),
        ("BRCA wild-type breast cancer", "BRCA wild-type"),
        ("BRCA-negative breast cancer", "BRCA wild-type"),
        ("HER2-negative breast cancer", None),  # HER2- handled by RECEPTOR_PATTERNS
    ])
    def test_negative_polarity_extracts_correct_canonical(
        self, query, expected_canonical
    ):
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers
        if expected_canonical is not None:
            assert expected_canonical in biomarkers, (
                f"Query '{query}': expected '{expected_canonical}' in "
                f"biomarkers, got {biomarkers}"
            )
        else:
            # HER2-negative is handled by receptor status, not BIOMARKER_PATTERNS
            receptor = result.cancer.receptor_status or ""
            assert "HER2-" in receptor or any("HER2" in b and "-" in b for b in biomarkers), (
                f"Query '{query}': expected HER2-negative marker, "
                f"got biomarkers={biomarkers}, receptor={receptor}"
            )


# ======================================================================
# HPV / p16 polarity — historically broken by the bare-"-" alternative
# ======================================================================

class TestHpvP16Polarity:
    """
    The original HPV/p16 regex had ``\\s*(?:negative|-)`` as the negative
    pattern. The bare ``-`` alternative matched ANY hyphen after HPV/p16,
    so "p16-positive" / "HPV-positive" / "HPV-related" were all wrongly
    tagged as NEGATIVE. Clinically dangerous: HPV/p16 status drives
    treatment de-escalation for OPSCC.

    Fixed by:
      - Both patterns use ``[\\s-]*`` so hyphen is a connector
      - Negative pattern requires bare ``-`` to be word-bounded
        (followed by whitespace, end-of-string, or punctuation)
    """

    @pytest.mark.parametrize("query,expected_tag,wrong_tag", [
        # The exact bug — hyphen-connected positive variants
        ("p16-positive oropharyngeal SCC",  "HPV+", "HPV-"),
        ("HPV-positive HNSCC",              "HPV+", "HPV-"),
        ("p16+ OPSCC",                      "HPV+", "HPV-"),
        ("HPV+ tumor",                      "HPV+", "HPV-"),
        # Whitespace-separated variants (should still work)
        ("p16 positive HNSCC",              "HPV+", "HPV-"),
        ("HPV positive cancer",             "HPV+", "HPV-"),
    ])
    def test_p16_hpv_positive_polarity(self, query, expected_tag, wrong_tag):
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers
        assert expected_tag in biomarkers, (
            f"Query '{query}': expected '{expected_tag}' in biomarkers, "
            f"got {biomarkers}"
        )
        assert wrong_tag not in biomarkers, (
            f"Query '{query}': '{wrong_tag}' should NOT be in biomarkers "
            f"(polarity flipped), got {biomarkers}"
        )

    @pytest.mark.parametrize("query,expected_tag", [
        ("p16-negative OPSCC",  "HPV-"),
        ("HPV-negative HNSCC",  "HPV-"),
        ("p16 negative tumor",  "HPV-"),
        ("HPV negative cancer", "HPV-"),
        # Bare "-" shorthand at end-of-token still tags negative
        ("p16- tumor",          "HPV-"),
        ("HPV-, ER+ disease",   "HPV-"),
    ])
    def test_p16_hpv_negative_polarity(self, query, expected_tag):
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers
        assert expected_tag in biomarkers, (
            f"Query '{query}': expected '{expected_tag}' in biomarkers, "
            f"got {biomarkers}"
        )
        assert "HPV+" not in biomarkers, (
            f"Query '{query}': 'HPV+' should NOT be in biomarkers, "
            f"got {biomarkers}"
        )

    @pytest.mark.parametrize("query", [
        # Descriptive hyphenated phrases — NOT polarity markers
        "HPV-related oropharyngeal cancer",
        "HPV-associated head and neck cancer",
        "HPV-driven OPSCC",
        "p16-associated tumor",
    ])
    def test_descriptive_hyphenated_phrases_do_not_tag_polarity(self, query):
        """
        "HPV-related" was being wrongly tagged as HPV- (negative).
        These descriptive uses of HPV-/p16- as compound-word prefixes
        should produce NEITHER polarity tag.
        """
        result = structure_query_fast(query)
        biomarkers = result.cancer.biomarkers
        assert "HPV+" not in biomarkers, (
            f"Query '{query}': 'HPV+' should NOT be in biomarkers "
            f"(descriptive phrase, not polarity), got {biomarkers}"
        )
        assert "HPV-" not in biomarkers, (
            f"Query '{query}': 'HPV-' should NOT be in biomarkers "
            f"(descriptive phrase, not polarity), got {biomarkers}"
        )
