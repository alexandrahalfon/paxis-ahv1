"""Palliative Care specialty agent."""

from __future__ import annotations

from typing import List, Optional

from ..base_agent import SpecialtyAgent
from ..case_bundle import PatientCaseBundle
from ..prompts import PALLIATIVE_CARE_PROMPT


class PalliativeCareAgent(SpecialtyAgent):
    specialty = "palliative_care"
    display_name = "Palliative Care"
    system_prompt = PALLIATIVE_CARE_PROMPT

    def relevance_filter(self, bundle: PatientCaseBundle) -> Optional[str]:
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

        # 1. Prognosis in the exact disease state (recurrent / metastatic /
        #    post-ICI failure drives hospice timing)
        is_advanced = (
            c.stage == "IV"
            or bundle.metastatic_sites
            or "ici_refractory" in bundle.trajectory_flags
            or "progressing_on_ici" in bundle.trajectory_flags
            or bundle.surgical_candidate is False
        )
        if is_advanced:
            queries.append(
                f"{primary_ident} prognosis median overall survival recurrent "
                f"metastatic post-immunotherapy failure hospice"
            )
        else:
            queries.append(
                f"{primary_ident} prognosis survival quality of life"
            )

        # 2. Symptom burden — tailor to likely symptoms for this site
        raw_lower = bundle.raw_text.lower()
        if "oral" in (c.site_detail or "") or "tongue" in (c.site_detail or "") \
                or "head" in (c.site or "") or "neck" in raw_lower:
            queries.append(
                "head and neck cancer symptom management dysphagia pain "
                "aspiration airway palliative care"
            )

        # 3. Early palliative care integration evidence
        queries.append(
            f"{primary_ident} early palliative care integration "
            f"quality of life outcomes trial"
        )

        # 4. Hospice eligibility / goals of care
        if is_advanced:
            queries.append(
                f"{primary_ident} hospice eligibility goals of care "
                f"advance care planning end of life"
            )

        # 5. Comorbidity + geriatric burden
        age = bundle.patient.age
        if (age and age >= 75) or bundle.comorbidities:
            queries.append(
                "older adult advanced cancer palliative care geriatric "
                "oncology comorbidity symptom burden"
            )

        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped
