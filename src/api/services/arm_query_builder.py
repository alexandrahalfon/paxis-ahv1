"""
Deterministic per-arm query builder (RF-4).

Builds a compact patient-summary string that is concatenated with each
treatment arm's label to form that arm's retrieval query. The goal is
that EVERY arm's query carries the full clinical context — site_detail,
histology, stage, TNM, grade, biomarkers, and prior treatments — so
downstream layers (PatientEligibility, PG matcher, Qdrant pre-filter)
always see a consistent view of the patient, regardless of which arm
label is prepended.

This module lives under `services/` (not `routes/`) so it can be unit
tested without pulling in the FastAPI / auth-dependency chain.
"""

from __future__ import annotations

from typing import List


def build_patient_summary_for_arm_queries(query_structure) -> str:
    """
    Produce a compact patient summary string suitable for
    `{arm_label} {patient_summary} outcomes` retrieval queries.

    Accepts any duck-typed object with `.cancer`, `.patient`, and
    `.treatment` attributes (as produced by
    `query_structuring_service.structure_query_fast`). Returns an empty
    string if `query_structure` is None or has no extracted patient
    context, in which case the caller should fall back to the raw
    query.
    """
    parts: List[str] = []
    if query_structure is None:
        return ""

    cancer = getattr(query_structure, "cancer", None)
    if cancer is not None:
        site_detail = getattr(cancer, "site_detail", None)
        site = getattr(cancer, "site", None)
        histology = getattr(cancer, "histology", None)
        stage = getattr(cancer, "stage", None)
        tnm = (
            cancer.get_tnm_string()
            if hasattr(cancer, "get_tnm_string")
            else None
        )
        grade = getattr(cancer, "grade", None)
        biomarkers = getattr(cancer, "biomarkers", None) or []
        receptor_status = getattr(cancer, "receptor_status", None)

        if site_detail:
            parts.append(site_detail)
        elif site:
            parts.append(site.replace("_", " "))
        if histology:
            # Upper-case the well-known histology abbreviations so they
            # match the spelling used in study titles.
            upper_abbr = {"scc", "nsclc", "sclc", "hcc", "crc"}
            if histology.lower() in upper_abbr:
                parts.append(histology.upper())
            else:
                parts.append(histology)
        if stage:
            parts.append(f"stage {stage}")
        if tnm:
            parts.append(tnm)
        if grade:
            parts.append(f"grade {grade}")
        if biomarkers:
            parts.extend(biomarkers[:4])
        if receptor_status and receptor_status not in biomarkers:
            parts.append(receptor_status)

    patient = getattr(query_structure, "patient", None)
    if patient is not None:
        age = getattr(patient, "age", None)
        if age:
            parts.append(f"{age} year old")

    treatment = getattr(query_structure, "treatment", None)
    if treatment is not None:
        prior = getattr(treatment, "prior_treatments", None) or []
        if prior:
            parts.append("s/p " + ", ".join(prior[:3]))

    # Deduplicate while preserving order (case-insensitive)
    seen: set = set()
    deduped: List[str] = []
    for p in parts:
        s = str(p).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    return " ".join(deduped)
