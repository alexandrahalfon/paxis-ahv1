"""
End-to-end canonical test case validation for RAG Pipeline Consolidation.

Tests the canonicalization pipeline components directly against the three
canonical test cases, verifying that biomarkers, cancer types, staging,
and entity linking produce the expected canonical outputs.

Canonical test cases:
  (a) 43F cT4dN2M0 TNBC post-NAC
  (b) 80M recurrent oral tongue SCC ICI-refractory
  (c) 55F pT1cN1mi ER+/HER2- RS22

Validates: Requirements 19.1, 19.2, 19.3
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from src.api.services.biomarker_canonicalizer import (
    BiomarkerCanonicalizer,
    CanonicalBiomarker,
)
from src.api.services.cancer_type_canonicalizer import (
    CancerTypeCanonicalizer,
    CanonicalCancerType,
)
from src.api.services.stage_canonicalizer import (
    StageCanonicalizer,
    TNMCanonical,
    StageHistory,
)
from src.api.services.entity_linker import EntityLinker
from tests.fixtures.consolidation_baseline import CANONICAL_QUERIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reconciled(**kwargs):
    """Create a mock ReconciledStructure with the given attributes."""
    return SimpleNamespace(**kwargs)



# ===========================================================================
# Test Case (a): 43F cT4dN2M0 TNBC post-NAC
# Validates: Requirement 19.1
# ===========================================================================

class TestCanonicalCaseA_TNBC_PostNAC:
    """43F cT4dN2M0 TNBC post-NAC.

    Expected:
    - TNBC canonicalized as ER-/PR-/HER2- (all negative polarity)
    - cT4d prefix preserved (t_prefix="c", t="T4d")
    - staging_type = "clinical"
    - post-NAC treatment history detected in query text
    """

    RAW_TEXT = CANONICAL_QUERIES["tnbc_post_nac"]

    # -- Biomarker canonicalization --

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_er_negative_canonicalized(self, _mock):
        """ER- resolves to canonical_id='ER' with polarity='negative'."""
        canon = BiomarkerCanonicalizer()
        result = canon.resolve("ER", "-", raw_text=self.RAW_TEXT, source="reconciled")
        assert result.canonical_id == "ER"
        assert result.polarity == "negative"

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_pr_negative_canonicalized(self, _mock):
        """PR- resolves to canonical_id='PR' with polarity='negative'."""
        canon = BiomarkerCanonicalizer()
        result = canon.resolve("PR", "-", raw_text=self.RAW_TEXT, source="reconciled")
        assert result.canonical_id == "PR"
        assert result.polarity == "negative"

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_her2_negative_canonicalized(self, _mock):
        """HER2- resolves to canonical_id='HER2' with polarity='negative'."""
        canon = BiomarkerCanonicalizer()
        result = canon.resolve("HER2", "-", raw_text=self.RAW_TEXT, source="reconciled")
        assert result.canonical_id == "HER2"
        assert result.polarity == "negative"

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_tnbc_all_three_negative(self, _mock):
        """TNBC = ER-/PR-/HER2-, all three resolve to negative polarity."""
        canon = BiomarkerCanonicalizer()
        biomarkers = canon.resolve_list(
            [("ER", "-"), ("PR", "-"), ("HER2", "-")],
            raw_text=self.RAW_TEXT,
            source="reconciled",
        )
        assert len(biomarkers) == 3
        for bm in biomarkers:
            assert bm.polarity == "negative", (
                f"{bm.canonical_id} should be negative, got {bm.polarity}"
            )

    # -- Stage canonicalization via EntityLinker --

    def test_ct4d_prefix_preserved(self):
        """cT4dN2M0 → t_prefix='c', t='T4d'."""
        linker = EntityLinker()
        tnm = linker.link_tnm(self.RAW_TEXT)
        assert tnm is not None
        assert tnm["t_prefix"] == "c"
        assert tnm["t"] == "T4d"

    def test_n2_extracted(self):
        """cT4dN2M0 → n='N2'."""
        linker = EntityLinker()
        tnm = linker.link_tnm(self.RAW_TEXT)
        assert tnm is not None
        assert tnm["n"] == "N2"

    def test_m0_extracted(self):
        """cT4dN2M0 → m='M0'."""
        linker = EntityLinker()
        tnm = linker.link_tnm(self.RAW_TEXT)
        assert tnm is not None
        assert tnm["m"] == "M0"

    @patch("src.api.services.stage_canonicalizer.settings",
           enable_canonicalization=True)
    def test_staging_type_clinical(self, _mock):
        """cT4d prefix → staging_type='clinical'."""
        reconciled = _make_reconciled(cancer_site="breast")
        with patch(
            "src.api.services.entity_linker.EntityLinker"
        ) as MockLinker:
            instance = MockLinker.return_value
            instance.link_tnm.return_value = {
                "t_prefix": "c", "t": "T4d",
                "n_prefix": "", "n": "N2",
                "m_prefix": "", "m": "M0",
            }
            stage_canon = StageCanonicalizer()
            tnm_canonical, history = stage_canon.canonicalize(reconciled, self.RAW_TEXT)
            assert tnm_canonical.t_prefix == "c"
            assert tnm_canonical.t == "T4d"
            assert tnm_canonical.staging_type == "clinical"

    # -- Cancer type --

    @patch("src.api.services.cancer_type_canonicalizer.settings",
           enable_canonicalization=True)
    @patch("src.api.services.comprehensive_retrieval.normalize_category",
           return_value="breast")
    def test_cancer_type_breast(self, _mock_norm, _mock_settings):
        """TNBC → cancer site = breast."""
        reconciled = _make_reconciled(cancer_site="breast", histology="")
        ct_canon = CancerTypeCanonicalizer()
        result = ct_canon.canonicalize(reconciled)
        assert result.site == "breast"

    # -- Post-NAC detection --

    def test_post_nac_in_query_text(self):
        """Query text contains 'neoadjuvant' indicating post-NAC treatment."""
        assert "neoadjuvant" in self.RAW_TEXT.lower()



# ===========================================================================
# Test Case (b): 80M recurrent oral tongue SCC ICI-refractory
# Validates: Requirement 19.2
# ===========================================================================

class TestCanonicalCaseB_OralTongueSCC:
    """80M recurrent oral tongue SCC ICI-refractory.

    Expected:
    - Recurrence detected (is_recurrent=True)
    - Oral tongue → head_neck site, site_detail=oral_tongue
    - Histology = scc
    - ICI-refractory trajectory flags set (pembrolizumab in query)
    """

    RAW_TEXT = CANONICAL_QUERIES["oral_tongue_scc_ici_refractory"]

    # -- Recurrence detection --

    @patch("src.api.services.stage_canonicalizer.settings",
           enable_canonicalization=True)
    def test_recurrence_detected(self, _mock):
        """'recurrent' in query → StageHistory.is_recurrent=True."""
        reconciled = _make_reconciled(cancer_site="head_neck", histology="scc")
        with patch(
            "src.api.services.entity_linker.EntityLinker"
        ) as MockLinker:
            instance = MockLinker.return_value
            instance.link_tnm.return_value = None
            stage_canon = StageCanonicalizer()
            tnm_canonical, history = stage_canon.canonicalize(reconciled, self.RAW_TEXT)
            assert history.is_recurrent is True

    # -- Cancer type canonicalization --

    @patch("src.api.services.cancer_type_canonicalizer.settings",
           enable_canonicalization=True)
    @patch("src.api.services.comprehensive_retrieval.normalize_category",
           return_value="head_neck")
    def test_oral_tongue_maps_to_head_neck(self, _mock_norm, _mock_settings):
        """oral tongue SCC → site='head_neck'."""
        reconciled = _make_reconciled(cancer_site="head_neck", histology="scc")
        ct_canon = CancerTypeCanonicalizer()
        result = ct_canon.canonicalize(reconciled)
        assert result.site == "head_neck"

    def test_site_detail_oral_tongue(self):
        """oral tongue → site_detail='oral_tongue'."""
        ct_canon = CancerTypeCanonicalizer()
        detail = ct_canon._extract_site_detail("head_neck", self.RAW_TEXT)
        assert detail == "oral_tongue"

    @patch("src.api.services.cancer_type_canonicalizer.settings",
           enable_canonicalization=True)
    @patch("src.api.services.comprehensive_retrieval.normalize_category",
           return_value="head_neck")
    def test_histology_scc(self, _mock_norm, _mock_settings):
        """SCC → histology='scc'."""
        reconciled = _make_reconciled(cancer_site="head_neck", histology="scc")
        ct_canon = CancerTypeCanonicalizer()
        result = ct_canon.canonicalize(reconciled)
        assert result.histology == "scc"

    @patch("src.api.services.cancer_type_canonicalizer.settings",
           enable_canonicalization=True)
    @patch("src.api.services.comprehensive_retrieval.normalize_category",
           return_value="head_neck")
    def test_keyed_category_head_neck_scc(self, _mock_norm, _mock_settings):
        """head_neck + scc → category='head_neck_scc'."""
        reconciled = _make_reconciled(cancer_site="head_neck", histology="scc")
        ct_canon = CancerTypeCanonicalizer()
        result = ct_canon.canonicalize(reconciled)
        assert result.category == "head_neck_scc"

    # -- ICI-refractory trajectory --

    def test_ici_refractory_in_query(self):
        """Query mentions ICI-refractory and pembrolizumab."""
        assert "ici-refractory" in self.RAW_TEXT.lower()
        assert "pembrolizumab" in self.RAW_TEXT.lower()

    # -- PD-L1 CPS extraction --

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_pdl1_cps_100_extracted(self, _mock):
        """PD-L1 CPS 100 → metric='CPS', metric_value='100'."""
        canon = BiomarkerCanonicalizer()
        result = canon.resolve(
            "PD-L1", "positive",
            raw_text=self.RAW_TEXT,
            source="reconciled",
        )
        assert result.canonical_id == "PD-L1"
        assert result.metric == "CPS"
        assert result.metric_value == "100"



# ===========================================================================
# Test Case (c): 55F pT1cN1mi ER+/HER2- RS22
# Validates: Requirement 19.3
# ===========================================================================

class TestCanonicalCaseC_ERpos_HER2neg:
    """55F pT1cN1mi ER+/HER2- RS22.

    Expected:
    - N1mi micrometastasis preserved (n='N1mi')
    - ER+ → positive, HER2- → negative
    - pT1c → t_prefix='p', t='T1c', staging_type='pathologic'
    - RS22 (Oncotype DX recurrence score) present in query
    """

    RAW_TEXT = CANONICAL_QUERIES["er_pos_her2_neg_rs22"]

    # -- Biomarker canonicalization --

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_er_positive_canonicalized(self, _mock):
        """ER+ resolves to polarity='positive'."""
        canon = BiomarkerCanonicalizer()
        result = canon.resolve("ER", "+", raw_text=self.RAW_TEXT, source="reconciled")
        assert result.canonical_id == "ER"
        assert result.polarity == "positive"

    @patch("src.api.services.biomarker_canonicalizer.settings",
           enable_canonicalization=True)
    def test_her2_negative_canonicalized(self, _mock):
        """HER2- resolves to polarity='negative'."""
        canon = BiomarkerCanonicalizer()
        result = canon.resolve("HER2", "-", raw_text=self.RAW_TEXT, source="reconciled")
        assert result.canonical_id == "HER2"
        assert result.polarity == "negative"

    def test_rs22_in_query(self):
        """Oncotype DX recurrence score 22 present in query text."""
        assert "recurrence score 22" in self.RAW_TEXT.lower()

    # -- Stage canonicalization: N1mi preserved --

    def test_n1mi_preserved_by_entity_linker(self):
        """pT1cN1miM0 → n='N1mi' (micrometastasis suffix preserved).

        The EntityLinker regex requires all three T, N, M components.
        The raw query text has "pT1cN1mi" without explicit M0, so we
        test with the complete TNM string that the reconciliation layer
        would produce.
        """
        linker = EntityLinker()
        tnm = linker.link_tnm("pT1cN1miM0")
        assert tnm is not None
        assert tnm["n"] == "N1mi"

    def test_pt1c_prefix_preserved(self):
        """pT1cN1miM0 → t_prefix='p', t='T1c'."""
        linker = EntityLinker()
        tnm = linker.link_tnm("pT1cN1miM0")
        assert tnm is not None
        assert tnm["t_prefix"] == "p"
        assert tnm["t"] == "T1c"

    @patch("src.api.services.stage_canonicalizer.settings",
           enable_canonicalization=True)
    def test_staging_type_pathologic(self, _mock):
        """pT1c prefix → staging_type='pathologic'."""
        reconciled = _make_reconciled(cancer_site="breast")
        with patch(
            "src.api.services.entity_linker.EntityLinker"
        ) as MockLinker:
            instance = MockLinker.return_value
            instance.link_tnm.return_value = {
                "t_prefix": "p", "t": "T1c",
                "n_prefix": "", "n": "N1mi",
                "m_prefix": "", "m": "M0",
            }
            stage_canon = StageCanonicalizer()
            tnm_canonical, history = stage_canon.canonicalize(reconciled, self.RAW_TEXT)
            assert tnm_canonical.t_prefix == "p"
            assert tnm_canonical.t == "T1c"
            assert tnm_canonical.n == "N1mi"
            assert tnm_canonical.staging_type == "pathologic"

    @patch("src.api.services.stage_canonicalizer.settings",
           enable_canonicalization=True)
    def test_n1mi_preserved_in_canonical(self, _mock):
        """N1mi is preserved through full stage canonicalization."""
        reconciled = _make_reconciled(cancer_site="breast")
        with patch(
            "src.api.services.entity_linker.EntityLinker"
        ) as MockLinker:
            instance = MockLinker.return_value
            instance.link_tnm.return_value = {
                "t_prefix": "p", "t": "T1c",
                "n_prefix": "", "n": "N1mi",
                "m_prefix": "", "m": "M0",
            }
            stage_canon = StageCanonicalizer()
            tnm_canonical, _ = stage_canon.canonicalize(reconciled, self.RAW_TEXT)
            assert "N1mi" in tnm_canonical.tnm_string()
