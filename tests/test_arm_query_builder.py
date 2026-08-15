"""
Offline tests for `build_patient_summary_for_arm_queries` (RF-4).

Purpose: lock in the invariant that the patient summary used to build
each treatment arm's retrieval query contains every clinical axis we
care about (site_detail, histology, stage, TNM, biomarkers), so arm
queries cannot silently drop context (as observed in the v4 live run
where the Immunotherapy arm dropped "maxilla" / "oral cavity" and
caused OCAT to be false-rejected downstream).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.arm_query_builder import (  # noqa: E402
    build_patient_summary_for_arm_queries,
)
from src.api.services.query_structuring_service import (  # noqa: E402
    CancerContext,
    PatientContext,
    QueryStructure,
    TreatmentContext,
)


class TestBuildPatientSummaryForArmQueries:
    def test_maxilla_scc_case_carries_all_clinical_axes(self):
        """The 68yo maxilla SCC case must carry site_detail, histology,
        stage, TNM, and grade through to every arm query."""
        qs = QueryStructure(
            original_query="68yo M poorly differentiated SCC of the maxilla pT4N0",
            cancer=CancerContext(
                site="head_neck",
                site_detail="maxilla",
                histology="scc",
                stage="IV",
                tnm_t="4",
                tnm_n="0",
                grade="poorly differentiated",
            ),
            patient=PatientContext(age=68),
        )
        summary = build_patient_summary_for_arm_queries(qs)
        assert "maxilla" in summary.lower()
        assert "scc" in summary.lower()
        assert "stage iv" in summary.lower()
        assert "t4" in summary.lower()
        assert "n0" in summary.lower()
        assert "68" in summary

    def test_biomarkers_are_included(self):
        qs = QueryStructure(
            original_query="colorectal MSI-H KRAS wild-type metastatic",
            cancer=CancerContext(
                site="gi",
                site_detail="colon",
                histology="adenocarcinoma",
                biomarkers=["MSI-H", "KRAS wild-type"],
            ),
        )
        summary = build_patient_summary_for_arm_queries(qs)
        assert "colon" in summary.lower()
        assert "adenocarcinoma" in summary.lower()
        assert "msi" in summary.lower()
        assert "kras" in summary.lower()

    def test_empty_structure_returns_empty_string(self):
        qs = QueryStructure(original_query="random non-clinical text")
        # no cancer / patient / treatment data → empty summary
        assert build_patient_summary_for_arm_queries(qs) == ""

    def test_none_input_returns_empty_string(self):
        assert build_patient_summary_for_arm_queries(None) == ""

    def test_dedup_preserves_order(self):
        qs = QueryStructure(
            original_query="breast cancer ER+ breast",
            cancer=CancerContext(
                site="breast",
                site_detail="breast",  # duplicate with site
                receptor_status="ER+",
                biomarkers=["ER+"],  # duplicate with receptor_status
            ),
        )
        summary = build_patient_summary_for_arm_queries(qs)
        # "breast" should only appear once
        assert summary.lower().count("breast") == 1
        # ER+ should only appear once
        assert summary.count("ER+") == 1

    def test_prior_treatments_rendered_as_spst(self):
        qs = QueryStructure(
            original_query="s/p pembrolizumab and chemo",
            cancer=CancerContext(site="head_neck", histology="scc"),
            treatment=TreatmentContext(
                prior_treatments=["pembrolizumab", "cisplatin"]
            ),
        )
        summary = build_patient_summary_for_arm_queries(qs)
        assert "s/p" in summary.lower()
        assert "pembrolizumab" in summary.lower()
