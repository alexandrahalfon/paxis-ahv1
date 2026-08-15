"""Radiation Oncology specialty agent."""

from __future__ import annotations

from typing import List, Optional

from ..base_agent import SpecialtyAgent
from ..case_bundle import PatientCaseBundle
from ..prompts import RADIATION_ONCOLOGY_PROMPT


def _has_prior_rt(bundle: PatientCaseBundle) -> bool:
    raw = bundle.raw_text.lower()
    prior_tx = " ".join(bundle.treatment.prior_treatments or []).lower()
    rt_indicators = ("radiation", "radiotherapy", "xrt ", " rt ", "imrt", "vmat",
                     "chemoradiation", "crt", "sbrt", "brachy")
    return any(k in raw for k in rt_indicators) or any(k in prior_tx for k in rt_indicators)


class RadiationOncologyAgent(SpecialtyAgent):
    specialty = "radiation_oncology"
    display_name = "Radiation Oncology"
    system_prompt = RADIATION_ONCOLOGY_PROMPT

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

        # 1. Reirradiation question if prior RT — always the decisive
        #    radiation-oncology question in a recurrent case.
        if _has_prior_rt(bundle):
            queries.append(
                f"{primary_ident} reirradiation salvage recurrent dose "
                f"constraints cumulative toxicity late effects"
            )

        # 2. Palliative RT for local symptom control when disease is
        #    unresectable / metastatic
        if bundle.surgical_candidate is False or bundle.metastatic_sites or c.stage == "IV":
            queries.append(
                f"{primary_ident} palliative radiation therapy symptom control "
                f"hemostatic pain control fungating tumor hypofractionation"
            )

        # 3. SBRT for oligometastasis to unusual sites
        met_sites = bundle.metastatic_sites
        if met_sites:
            met_str = " ".join(met_sites)
            queries.append(
                f"SBRT stereotactic body radiation {met_str} metastasis "
                f"oligometastatic disease outcomes"
            )

        # 4. Site-specific dose constraints (carotid blowout, organ at risk)
        if "head" in (c.site or "") or "oral" in (c.site_detail or "") or \
                "neck" in bundle.raw_text.lower():
            queries.append(
                "head and neck reirradiation carotid blowout risk "
                "osteoradionecrosis mandible cervical spinal cord dose constraint"
            )

        # 5. Concurrent RT + systemic therapy
        if "ici_refractory" in bundle.trajectory_flags or \
                "progressing_on_ici" in bundle.trajectory_flags:
            queries.append(
                f"{primary_ident} radiation immunotherapy checkpoint inhibitor "
                f"combination abscopal salvage palliative"
            )

        # Fallback — keep the panel balanced
        if not queries:
            queries.append(
                f"{primary_ident} radiation therapy outcomes dose fractionation "
                f"definitive adjuvant"
            )

        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped
