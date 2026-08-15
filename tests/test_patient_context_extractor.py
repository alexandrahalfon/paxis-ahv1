"""
Tests for patient_eligibility_boost_service.extract_patient_context_from_query.

Focus: the site-aware disambiguation fix for multi-cancer patients. Before
this fix, the extractor used first-match-wins + break semantics for
histology and surgery, which caused patients with more than one cancer in
their history to be mislabeled — e.g. a patient with a cured transverse
colon adenocarcinoma in their PMH and a recurrent oral-tongue squamous
cell carcinoma as the active disease would be tagged histology=
adenocarcinoma and surgery=colorectal surgery, causing legitimate H&N SCC
studies to be hard-filtered out by the downstream eligibility check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.patient_eligibility_boost_service import (  # noqa: E402
    extract_patient_context_from_query,
)
from src.api.services.ontology_loader import (  # noqa: E402
    cancer_type_label_to_site_key,
    is_histology_plausible_for_site,
    is_surgery_plausible_for_site,
    get_plausible_histologies_for_site,
    get_plausible_surgeries_for_site,
)


# ─── Canonical case from the live audit log ─────────────────────────────

CANONICAL_MULTI_CANCER_CASE = (
    "80 y.o. male non-smoker with a PMH HTN, Hep C, BPH, CKD, latent syphilis, "
    "transverse colon adenocarcinoma complicated by LBO s/p (6/16/21) diagnostic "
    "lap, ex lap with extended right hemicolectomy 6/2021 and ileostomy reversal "
    "10/6/2021, and initial Stage II (pT2pN0M0R0, DOI 5.1 mm, PNI-, LVSI-) "
    "squamous cell carcinoma of the left oral tongue, status post left partial "
    "glossectomy, left neck dissection levels I-III, and radial forearm free "
    "flap reconstruction, and left STSG performed at Bellevue Hospital on "
    "12/2/2024 with Dr. Moses. In August 2025, he developed a recurrent lesion "
    "in the left level I neck associated with a multiloculated left sub-lingual "
    "collection, which was biopsy-proven recurrent SCC with a CPS score of 100, "
    "started on pembrolizumab (declined combination with chemotherapy) and is "
    "no longer a surgical candidate following significant locoregional "
    "progression on ICI with radiographic concern for metastatic disease to "
    "the right ventricle and progressing on systemic therapy."
)


class TestCanonicalMultiCancerFix:
    """The exact regression from the live log. Must not break."""

    def test_cancer_type_is_head_and_neck(self):
        ctx = extract_patient_context_from_query(CANONICAL_MULTI_CANCER_CASE)
        assert ctx is not None
        assert ctx["cancer_type"] == "head and neck cancer"

    def test_histology_is_scc_not_adenocarcinoma(self):
        """Regression guard: adenocarcinoma from the colon-cancer PMH must
        not override squamous cell carcinoma from the active H&N cancer."""
        ctx = extract_patient_context_from_query(CANONICAL_MULTI_CANCER_CASE)
        assert ctx["histology"] == "squamous cell carcinoma"
        assert ctx["histology"] != "adenocarcinoma"

    def test_surgery_is_hn_not_colorectal(self):
        """Regression guard: the colectomy in the PMH must not set
        surgery=colorectal surgery when the active cancer is H&N."""
        ctx = extract_patient_context_from_query(CANONICAL_MULTI_CANCER_CASE)
        assert ctx["surgery"] in {
            "partial glossectomy",
            "glossectomy",
            "neck dissection",
            "radial forearm free flap",
            "free flap reconstruction",
        }
        assert "colectomy" not in ctx["surgery"]
        assert "colorectal" not in ctx["surgery"]

    def test_demographics_still_extracted(self):
        ctx = extract_patient_context_from_query(CANONICAL_MULTI_CANCER_CASE)
        assert ctx["age"] == 80
        assert ctx["gender"] == "male"
        assert ctx["stage"] == "II"


class TestSingleCancerBaseline:
    """Single-cancer queries must continue to work unchanged."""

    @pytest.mark.parametrize(
        "query,cancer_type,histology",
        [
            (
                "62 year old male with metastatic colon cancer, adenocarcinoma, s/p hemicolectomy",
                "colorectal cancer",
                "adenocarcinoma",
            ),
            (
                "68 year old male with stage IV lung adenocarcinoma, EGFR-mutant, s/p lobectomy",
                "lung cancer",
                "adenocarcinoma",
            ),
            (
                "70 year old male with stage IV NSCLC, squamous cell carcinoma, pembrolizumab",
                "lung cancer",
                "squamous cell carcinoma",
            ),
            (
                "58 year old male with high-grade glioblastoma s/p craniotomy",
                "brain cancer",
                "glioblastoma",
            ),
            (
                "72 year old female with metastatic melanoma BRAF V600E",
                "melanoma",
                "melanoma",
            ),
            (
                "58 year old male with stage III oropharyngeal squamous cell carcinoma, HPV-positive",
                "head and neck cancer",
                "squamous cell carcinoma",
            ),
            (
                "60 year old male with esophageal squamous cell carcinoma",
                "esophageal cancer",
                "squamous cell carcinoma",
            ),
            (
                "70 year old male with metastatic colorectal adenocarcinoma KRAS wild-type",
                "colorectal cancer",
                "adenocarcinoma",
            ),
        ],
    )
    def test_single_cancer(self, query, cancer_type, histology):
        ctx = extract_patient_context_from_query(query)
        assert ctx is not None
        assert ctx.get("cancer_type") == cancer_type
        assert ctx.get("histology") == histology


class TestHistoricalSignalSuppression:
    """Cancer mentioned under "PMH" / "history of" / "remote" should NOT
    become the primary cancer_type when an active cancer is also present."""

    def test_remote_breast_lung_active(self):
        query = (
            "65 year old female with stage IV lung adenocarcinoma and remote "
            "history of breast cancer s/p mastectomy"
        )
        ctx = extract_patient_context_from_query(query)
        assert ctx["cancer_type"] == "lung cancer"
        assert ctx["histology"] == "adenocarcinoma"

    def test_pmh_breast_active_lung(self):
        query = (
            "55 year old female with PMH of breast cancer s/p lumpectomy 2015, "
            "now with stage IV lung adenocarcinoma"
        )
        ctx = extract_patient_context_from_query(query)
        assert ctx["cancer_type"] == "lung cancer"
        assert ctx["histology"] == "adenocarcinoma"

    def test_history_of_prostate_active_colon(self):
        query = (
            "60 year old male with history of prostate cancer in remission, "
            "now with newly diagnosed colon cancer"
        )
        ctx = extract_patient_context_from_query(query)
        assert ctx["cancer_type"] == "colorectal cancer"


class TestOntologyLookups:
    """Ontology-backed plausibility helpers."""

    def test_cancer_type_label_maps_to_site_key(self):
        assert cancer_type_label_to_site_key("head and neck cancer") == "h_n"
        assert cancer_type_label_to_site_key("lung cancer") == "lung"
        assert cancer_type_label_to_site_key("colorectal cancer") == "gi"
        assert cancer_type_label_to_site_key("pancreatic cancer") == "gi"
        assert cancer_type_label_to_site_key(None) is None
        assert cancer_type_label_to_site_key("unknown cancer") is None

    def test_adenocarcinoma_not_plausible_for_hn(self):
        """The headline fix: adenocarcinoma must NOT be tagged as
        plausible for an H&N patient."""
        assert not is_histology_plausible_for_site("adenocarcinoma", "h_n")

    def test_scc_plausible_for_hn(self):
        assert is_histology_plausible_for_site("squamous cell carcinoma", "h_n")

    def test_adenocarcinoma_plausible_for_lung_and_gi(self):
        assert is_histology_plausible_for_site("adenocarcinoma", "lung")
        assert is_histology_plausible_for_site("adenocarcinoma", "gi")

    def test_scc_plausible_for_lung_and_gi(self):
        """SCC is esophageal in GI and squamous NSCLC in lung — both real."""
        assert is_histology_plausible_for_site("squamous cell carcinoma", "lung")
        assert is_histology_plausible_for_site("squamous cell carcinoma", "gi")

    def test_permissive_fallback_on_unknown_site(self):
        """Unknown site → return True (don't over-filter)."""
        assert is_histology_plausible_for_site("adenocarcinoma", None)
        assert is_histology_plausible_for_site("adenocarcinoma", "radiopharm")

    def test_colorectal_surgery_not_plausible_for_hn(self):
        assert not is_surgery_plausible_for_site("colectomy", "h_n")
        assert not is_surgery_plausible_for_site("hemicolectomy", "h_n")

    def test_hn_surgeries_plausible_for_hn(self):
        assert is_surgery_plausible_for_site("neck dissection", "h_n")
        assert is_surgery_plausible_for_site("partial glossectomy", "h_n")
        assert is_surgery_plausible_for_site("radial forearm free flap", "h_n")

    def test_site_histology_map_is_lowercased(self):
        hist_hn = get_plausible_histologies_for_site("h_n")
        assert all(h == h.lower() for h in hist_hn)
        assert "squamous cell carcinoma" in hist_hn

    def test_site_surgery_map_populated(self):
        surg_hn = get_plausible_surgeries_for_site("h_n")
        assert "neck dissection" in surg_hn
        assert "glossectomy" in surg_hn
        surg_gi = get_plausible_surgeries_for_site("gi")
        assert "colectomy" in surg_gi
        assert "hemicolectomy" in surg_gi
