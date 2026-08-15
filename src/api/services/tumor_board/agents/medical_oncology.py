"""Medical Oncology specialty agent."""

from __future__ import annotations

from typing import List, Optional

from ..base_agent import SpecialtyAgent
from ..case_bundle import PatientCaseBundle
from ..prompts import MEDICAL_ONCOLOGY_PROMPT


CISPLATIN_CONTRAINDICATIONS = (
    "ckd", "chronic kidney", "renal", "kidney", "creatinine", "dialysis",
)


class MedicalOncologyAgent(SpecialtyAgent):
    specialty = "medical_oncology"
    display_name = "Medical Oncology"
    system_prompt = MEDICAL_ONCOLOGY_PROMPT

    def relevance_filter(self, bundle: PatientCaseBundle) -> Optional[str]:
        # Medical Oncology weighs in on virtually every cancer case with
        # patient context. Skip only if there is literally no patient.
        if not bundle.has_patient_context:
            return "no patient context extracted"
        return None

    def build_sub_queries(self, bundle: PatientCaseBundle) -> List[str]:
        c = bundle.cancer
        site = (c.site or "").replace("_", " ")
        site_detail = (c.site_detail or "").replace("_", " ")
        histology = c.histology or ""
        histology_disp = histology.upper() if histology == "scc" else histology
        primary_ident = " ".join(p for p in [site_detail, site, histology_disp] if p).strip()
        if not primary_ident:
            primary_ident = "cancer"

        queries: List[str] = []

        # 1. Primary-cancer systemic baseline (always)
        queries.append(
            f"{primary_ident} systemic therapy outcomes overall survival "
            f"progression-free survival response rate"
        )

        # 2. ICI-refractory salvage, if the case shows ICI failure
        flags = set(bundle.trajectory_flags)
        if "ici_refractory" in flags or "progressing_on_ici" in flags:
            queries.append(
                f"{primary_ident} ICI-refractory anti-PD1 failure second-line "
                f"salvage checkpoint inhibitor resistant cetuximab"
            )

        # 3. Biomarker-directed options
        biomarkers = [b.upper() for b in bundle.biomarkers]
        has_cps = any("CPS" in b or "PD-L1" in b or "PDL1" in b for b in biomarkers)
        if has_cps:
            queries.append(
                f"{primary_ident} CPS PD-L1 high pembrolizumab response "
                f"primary resistance mechanisms"
            )
        targetable = [
            t for t in ("HER2", "EGFR", "PIK3CA", "KRAS", "BRAF", "NTRK", "FGFR")
            if any(t in b for b in biomarkers)
        ]
        if targetable:
            queries.append(
                f"{primary_ident} targeted therapy {' '.join(targetable)} "
                f"molecularly matched treatment"
            )

        # 4. Comorbidity-constrained regimen choices
        comorbid_lower = " ".join(bundle.comorbidities).lower()
        raw_lower = bundle.raw_text.lower()
        if any(k in comorbid_lower or k in raw_lower
               for k in CISPLATIN_CONTRAINDICATIONS):
            queries.append(
                f"{primary_ident} cisplatin ineligible carboplatin renal impairment "
                f"CKD dose modification non-nephrotoxic systemic therapy"
            )

        # 5. Recurrent / metastatic context fallback
        if not any("ici" in q.lower() or "biomarker" in q.lower() for q in queries[1:]):
            if c.stage == "IV" or bundle.metastatic_sites:
                queries.append(
                    f"{primary_ident} metastatic recurrent systemic therapy "
                    f"R/M outcomes second-line"
                )

        # De-dupe while preserving order
        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped
