"""Surgical Oncology specialty agent."""

from __future__ import annotations

from typing import List, Optional

from ..base_agent import SpecialtyAgent
from ..case_bundle import PatientCaseBundle
from ..prompts import SURGICAL_ONCOLOGY_PROMPT


class SurgicalOncologyAgent(SpecialtyAgent):
    specialty = "surgical_oncology"
    display_name = "Surgical Oncology"
    system_prompt = SURGICAL_ONCOLOGY_PROMPT

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

        # 1. Salvage vs. primary resection question
        raw_lower = bundle.raw_text.lower()
        recurrent = (
            "recurrent" in raw_lower
            or "recurrence" in raw_lower
            or (bundle.clinical_history and
                (bundle.clinical_history.recurrence_info or ""))
        )
        if recurrent:
            queries.append(
                f"{primary_ident} salvage surgery recurrent outcomes "
                f"R0 resection disease-free survival 30-day mortality"
            )
        else:
            queries.append(
                f"{primary_ident} primary surgical resection outcomes margins "
                f"negative resection disease-free survival"
            )

        # 2. Unresectable / non-surgical pathway
        if bundle.surgical_candidate is False:
            queries.append(
                f"{primary_ident} unresectable inoperable non-surgical "
                f"locoregional advanced management"
            )

        # 3. Palliative / debulking procedures
        if bundle.metastatic_sites or c.stage == "IV" or bundle.surgical_candidate is False:
            queries.append(
                f"{primary_ident} palliative debulking symptom control airway "
                f"feeding tube PEG tracheostomy"
            )

        # 4. Reirradiated / reconstructed field surgery (adds huge morbidity)
        if any(k in raw_lower for k in
               ("flap", "reconstruction", "radiation", "radiotherapy",
                "chemoradiation", "imrt", "vmat")):
            queries.append(
                f"{primary_ident} surgery after radiation reconstructed field "
                f"flap failure wound healing morbidity salvage"
            )

        # 5. Lymph node dissection question (if neck / nodal disease)
        if "neck" in raw_lower or "nodal" in raw_lower or c.tnm_n:
            queries.append(
                f"{primary_ident} neck dissection levels recurrent cervical "
                f"lymphadenopathy regional control"
            )

        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped
