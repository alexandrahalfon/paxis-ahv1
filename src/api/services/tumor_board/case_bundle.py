"""
PatientCaseBundle — the shared, immutable patient case passed to every
specialty agent. Built once from the existing query structuring + clinical
inference pipeline so all six agents see exactly the same extracted facts.

No agent talks to the retriever through this object; it is pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api.services.query_structuring_service import (
    QueryStructure,
    structure_query_with_llm_if_needed,
)
from src.api.services.clinical_inference import (
    apply_inference_to_query_structure,
)


@dataclass
class PatientCaseBundle:
    """
    Immutable bundle of extracted patient facts. Built once and passed to
    every specialty agent.

    Attributes are intentionally flat so that agent `build_sub_queries()`
    implementations can read them directly without knowing the regex /
    LLM extraction plumbing.
    """

    # Original raw narrative
    raw_text: str

    # Full QueryStructure (has patient / cancer / treatment / clinical_history)
    structure: QueryStructure

    # Inferred axes produced by clinical_inference.apply_inference_to_query_structure()
    #
    # Keys:
    #   expanded_axes      : Dict[str, str] — original axes + inferred terms appended
    #   trajectory_flags   : List[str]      — e.g. ["ici_refractory", "progressing_on_ici"]
    #   metastatic_sites   : List[str]      — e.g. ["right ventricle", "cardiac"]
    #   surgical_candidate : Optional[bool] — True / False / None
    #   inferred_terms     : Dict[str, List[str]]  — per-axis inferred vocabulary
    inferred_axes: Dict[str, Any] = field(default_factory=dict)

    # Whether the LLM 8-axis extractor ran (informational)
    used_llm_extraction: bool = False

    # ─── Convenience accessors ────────────────────────────────────────────

    @property
    def patient(self):
        return self.structure.patient

    @property
    def cancer(self):
        return self.structure.cancer

    @property
    def treatment(self):
        return self.structure.treatment

    @property
    def clinical_history(self):
        return self.structure.clinical_history

    @property
    def trajectory_flags(self) -> List[str]:
        return list(self.inferred_axes.get("trajectory_flags", []) or [])

    @property
    def metastatic_sites(self) -> List[str]:
        return list(self.inferred_axes.get("metastatic_sites", []) or [])

    @property
    def surgical_candidate(self) -> Optional[bool]:
        return self.inferred_axes.get("surgical_candidate")

    @property
    def expanded_axes(self) -> Dict[str, str]:
        return dict(self.inferred_axes.get("expanded_axes", {}) or {})

    @property
    def inferred_terms(self) -> Dict[str, List[str]]:
        return dict(self.inferred_axes.get("inferred_terms", {}) or {})

    @property
    def has_patient_context(self) -> bool:
        return bool(self.structure.has_patient_context)

    @property
    def biomarkers(self) -> List[str]:
        return list(self.cancer.biomarkers or [])

    @property
    def comorbidities(self) -> List[str]:
        return list(self.patient.comorbidities or [])

    @property
    def category(self) -> Optional[str]:
        """Qdrant category filter derived from the extracted site / histology.

        Threaded into every agent's `lightweight_search` call so per-specialty
        sub-queries can't match literature from unrelated cancer categories
        (e.g. a pathology NGS sub-query embedding-matching NCCN-NSCLC content
        for a head-and-neck case).
        """
        site = (self.cancer.site or "").lower()
        site_detail = (self.cancer.site_detail or "").lower()
        histology = (self.cancer.histology or "").lower()
        return _resolve_site_to_category(site, site_detail, histology)

    # ─── Summarisation for prompts / API responses ────────────────────────

    def summary_lines(self) -> List[str]:
        """Human-readable bullet summary used in LLM prompts and in the API
        response's `case_summary` field. Only lines with real content."""
        lines: List[str] = []

        p = self.patient
        demo_bits: List[str] = []
        if p.age is not None:
            demo_bits.append(f"{p.age} y.o.")
        if p.gender:
            demo_bits.append(p.gender)
        if p.smoking_status:
            demo_bits.append(p.smoking_status)
        if p.performance_status:
            demo_bits.append(p.performance_status)
        if demo_bits:
            lines.append("Patient: " + ", ".join(demo_bits))

        if p.comorbidities:
            lines.append("Comorbidities: " + ", ".join(p.comorbidities))

        c = self.cancer
        cancer_bits: List[str] = []
        if c.site:
            cancer_bits.append(c.site.replace("_", " "))
        if c.site_detail:
            cancer_bits.append(c.site_detail.replace("_", " "))
        if c.histology:
            cancer_bits.append(c.histology.upper() if c.histology == "scc" else c.histology)
        if c.stage:
            cancer_bits.append(f"stage {c.stage}")
        tnm = c.get_tnm_string() if hasattr(c, "get_tnm_string") else None
        if tnm:
            cancer_bits.append(tnm)
        if cancer_bits:
            lines.append("Cancer: " + ", ".join(cancer_bits))

        path_bits: List[str] = []
        if c.doi:
            path_bits.append(f"DOI {c.doi}")
        if c.pni:
            path_bits.append(f"PNI {c.pni}")
        if c.lvi:
            path_bits.append(f"LVI {c.lvi}")
        if c.margins:
            path_bits.append(f"margins {c.margins}")
        if path_bits:
            lines.append("Pathology: " + ", ".join(path_bits))

        if c.biomarkers:
            lines.append("Biomarkers: " + ", ".join(c.biomarkers))

        t = self.treatment
        if t.prior_treatments:
            lines.append("Prior treatment: " + ", ".join(t.prior_treatments))
        if t.raw_text:
            lines.append("Treatment detail: " + t.raw_text)

        ch = self.clinical_history
        if ch.recurrence_info:
            lines.append("Disease trajectory: " + ch.recurrence_info)
        if ch.imaging_findings:
            lines.append("Imaging / metastatic concern: " + ch.imaging_findings)

        if self.trajectory_flags:
            lines.append("Inferred trajectory flags: " + ", ".join(self.trajectory_flags))
        if self.metastatic_sites:
            lines.append("Inferred metastatic sites: " + ", ".join(self.metastatic_sites))
        if self.surgical_candidate is False:
            lines.append("Inferred surgical candidacy: NOT a candidate (unresectable/inoperable)")
        elif self.surgical_candidate is True:
            lines.append("Inferred surgical candidacy: candidate")

        return lines

    def summary_text(self) -> str:
        """One-shot text summary used inside agent LLM prompts."""
        lines = self.summary_lines()
        if not lines:
            return self.raw_text[:500]
        return "\n".join(f"- {line}" for line in lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "raw_text": self.raw_text,
            "used_llm_extraction": self.used_llm_extraction,
            "structure": self.structure.to_dict(),
            "inferred_axes": {
                "trajectory_flags": self.trajectory_flags,
                "metastatic_sites": self.metastatic_sites,
                "surgical_candidate": self.surgical_candidate,
                "inferred_terms": self.inferred_terms,
                "expanded_axes": self.expanded_axes,
            },
            "summary_lines": self.summary_lines(),
        }


async def build_case_bundle(
    case_text: str,
    query_type: str = "treatment_recommendation",
) -> PatientCaseBundle:
    """
    Build a PatientCaseBundle from a raw clinical narrative.

    Pipeline:
      1. structure_query_with_llm_if_needed() — regex + (optional) LLM 8-axis
         extraction. Produces a QueryStructure.
      2. apply_inference_to_query_structure() — maps implicit narrative to
         explicit clinical labels (unresectable, ICI-refractory, cardiac met).
      3. Package everything into PatientCaseBundle.

    This function is the ONLY entry point the orchestrator uses. Agents do
    not re-extract anything.
    """
    structure, used_llm = await structure_query_with_llm_if_needed(
        case_text, query_type=query_type
    )

    inferred_axes = apply_inference_to_query_structure(structure, case_text)

    return PatientCaseBundle(
        raw_text=case_text,
        structure=structure,
        inferred_axes=inferred_axes,
        used_llm_extraction=used_llm,
    )


# ─── site → Qdrant category resolver ──────────────────────────────────────
# Mirrors the mapping in patient_matching_service_simple._infer_category_from_site
# but returns the short form accepted by build_category_match_variants.

_SITE_CATEGORY_MAP: Dict[str, str] = {
    # Head & neck
    "maxilla": "head_neck", "mandible": "head_neck", "oral_cavity": "head_neck",
    "oral cavity": "head_neck", "tongue": "head_neck", "oral_tongue": "head_neck",
    "gingiva": "head_neck", "hard_palate": "head_neck", "soft_palate": "head_neck",
    "buccal_mucosa": "head_neck", "floor_of_mouth": "head_neck",
    "oropharynx": "head_neck", "nasopharynx": "head_neck",
    "hypopharynx": "head_neck", "larynx": "head_neck",
    "tonsil": "head_neck", "base_of_tongue": "head_neck",
    "pharynx": "head_neck", "neck": "head_neck",
    "salivary_gland": "head_neck", "parotid": "head_neck",
    # Lung
    "lung": "lung", "bronchus": "lung",
    # Breast
    "breast": "breast",
    # GYN
    "cervix": "gyn", "uterus": "gyn", "ovary": "gyn",
    "endometrium": "gyn", "vulva": "gyn", "vagina": "gyn",
    # GI
    "anus": "gi", "rectum": "gi", "colon": "gi", "esophagus": "gi",
    "stomach": "gi", "liver": "gi", "pancreas": "gi",
    # GU
    "bladder": "gu", "kidney": "gu",
    # Prostate (own category)
    "prostate": "prostate",
    # CNS
    "brain": "cns",
    # Skin
    "skin": "cutaneous",
}


def _resolve_site_to_category(
    site: str,
    site_detail: str,
    histology: str,
) -> Optional[str]:
    for candidate in (site_detail, site):
        if not candidate:
            continue
        key = candidate.strip().lower().replace(" ", "_")
        if key in _SITE_CATEGORY_MAP:
            return _SITE_CATEGORY_MAP[key]
        for site_key, cat in _SITE_CATEGORY_MAP.items():
            if site_key in key:
                return cat
    h = (histology or "").lower()
    if "nsclc" in h or "sclc" in h:
        return "lung"
    if "melanoma" in h:
        return "cutaneous"
    if "glioma" in h or "glioblastoma" in h:
        return "cns"
    return None
