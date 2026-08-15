"""Pathology / Molecular specialty agent."""

from __future__ import annotations

from typing import List, Optional

from ..base_agent import SpecialtyAgent
from ..case_bundle import PatientCaseBundle
from ..prompts import PATHOLOGY_MOLECULAR_PROMPT


class PathologyMolecularAgent(SpecialtyAgent):
    specialty = "pathology_molecular"
    display_name = "Pathology / Molecular"
    system_prompt = PATHOLOGY_MOLECULAR_PROMPT

    def relevance_filter(self, bundle: PatientCaseBundle) -> Optional[str]:
        # Pathology always weighs in when there is a confirmed histology.
        if not bundle.has_patient_context:
            return "no patient context extracted"
        if not (bundle.cancer.histology or bundle.cancer.biomarkers):
            return "no histology or biomarker data to interpret"
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

        biomarkers = [b.upper() for b in bundle.biomarkers]

        # 1. NGS / molecular profiling discovery query
        queries.append(
            f"{primary_ident} next-generation sequencing NGS molecular profiling "
            f"actionable alterations targetable mutations HER2 EGFR PIK3CA"
        )

        # 2. CPS / PD-L1 interpretation and primary resistance
        if any("CPS" in b or "PD-L1" in b or "PDL1" in b for b in biomarkers):
            queries.append(
                f"{primary_ident} CPS PD-L1 high pembrolizumab primary resistance "
                f"biology IFN gamma antigen presentation loss"
            )

        # 3. HPV / p16 stratification (especially H&N)
        if any("HPV" in b or "P16" in b for b in biomarkers) or \
                "oropharynx" in (c.site_detail or "") or \
                "tongue" in (c.site_detail or ""):
            queries.append(
                f"{primary_ident} HPV p16 status prognostic predictive "
                f"stratification outcomes"
            )

        # 4. Biomarker gap — what tests are MISSING that could unlock therapy
        queries.append(
            f"{primary_ident} HER2 expression amplification PIK3CA mutation "
            f"EGFR FGFR NTRK fusion actionable"
        )

        # 5. Histology-specific molecular atlas
        if histology_disp:
            queries.append(
                f"{histology_disp} {site or 'cancer'} molecular landscape "
                f"genomic alterations frequency actionability"
            )

        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped
