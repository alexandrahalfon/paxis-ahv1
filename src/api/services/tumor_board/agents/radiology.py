"""Radiology specialty agent."""

from __future__ import annotations

from typing import List, Optional

from ..base_agent import SpecialtyAgent
from ..case_bundle import PatientCaseBundle
from ..prompts import RADIOLOGY_PROMPT


class RadiologyAgent(SpecialtyAgent):
    specialty = "radiology"
    display_name = "Radiology"
    system_prompt = RADIOLOGY_PROMPT

    def relevance_filter(self, bundle: PatientCaseBundle) -> Optional[str]:
        if not bundle.has_patient_context:
            return "no patient context extracted"
        # Radiology is most useful when imaging findings or metastatic
        # concerns are mentioned. If the narrative has none, we can still
        # contribute (staging workup), but flag it.
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

        # 1. Metastatic site work-up / differential — if a specific distant
        #    site is suspected this is THE question the radiologist carries
        met_sites = bundle.metastatic_sites
        if met_sites:
            met_str = " ".join(met_sites)
            queries.append(
                f"{met_str} metastasis imaging appearance differential diagnosis "
                f"cardiac MRI PET CT biopsy confirmation"
            )

        # 2. Cardiac-specific differential if right ventricle / cardiac
        #    mention exists
        raw_lower = bundle.raw_text.lower()
        if "ventricle" in raw_lower or "cardiac" in raw_lower or "heart" in raw_lower:
            queries.append(
                "cardiac metastasis right ventricle tumor thrombus differential "
                "cardiac MRI echocardiography endomyocardial biopsy"
            )

        # 3. Local staging accuracy in the post-surgical / post-flap bed
        if any(k in raw_lower for k in ("flap", "reconstruction", "glossectomy",
                                        "dissection", "resection")):
            queries.append(
                f"{primary_ident} recurrence post-surgical post-flap MRI PET CT "
                f"accuracy detection cervical neck"
            )

        # 4. Carotid / vascular encasement assessment
        if "neck" in raw_lower or "carotid" in raw_lower or "oral" in (c.site_detail or ""):
            queries.append(
                "head neck cancer carotid artery encasement imaging criteria "
                "unresectable prevertebral fascia invasion"
            )

        # 5. Response assessment on ICI (iRECIST, pseudo-progression)
        if any(f in bundle.trajectory_flags for f in
               ("ici_refractory", "progressing_on_ici")):
            queries.append(
                f"{primary_ident} immunotherapy response assessment iRECIST "
                f"pseudo-progression hyperprogression imaging"
            )

        # 6. Generic staging workup fallback
        if not queries:
            queries.append(
                f"{primary_ident} staging CT MRI PET FDG imaging workup "
                f"recommendations"
            )

        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped
