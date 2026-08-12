"""
Golden QueryStructure fixtures for the PostgreSQL structured study matcher.

Each fixture is a dict matching `QueryStructure.to_dict()` shape — the
exact input the matcher consumes at `match_studies_by_structure()` in
`src/api/services/structured_study_matcher.py:451`. Keeping them in
this file (instead of hardcoded inside the tests) lets the integration
script (`scripts/test_matcher_live.py`) re-use the same dict against
the live Postgres, so offline and live runs exercise identical inputs.

Fixtures intentionally span:

- simple_lung_adeno              — basic site+histology+biomarker flow
- hn_scc_multi_axis              — the canonical 80 y.o. case from the
                                   live audit log (full cancer context)
- triple_negative_breast         — TNBC special-case and HER2- exclusion
- prostate_crpc                  — prostate site with castration-resistant
                                   disease descriptor
- colorectal_msi_high            — MSI JSONB Tier-1 path + GI site
- breast_her2_neg_liver_mets     — validates the hard site filter stays
                                   on %breast% even when the patient has
                                   liver mets (current behaviour)
- empty_query                    — no criteria at all, must return
                                   empty without touching the DB
- biomarker_only                 — biomarkers but no cancer_site, no
                                   hard filter applied
- cps_score_of_100_raw           — the literal string from the audit
                                   log, documents the parsing gap
"""

from __future__ import annotations

from typing import Any, Dict


# ────────────────────────────────────────────────────────────────────────────
# 1. Simple lung adenocarcinoma with EGFR
# ────────────────────────────────────────────────────────────────────────────

SIMPLE_LUNG_ADENO: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "lung",
        "histology": "adenocarcinoma",
        "stage": "IV",
        "biomarkers": ["EGFR mutant"],
    },
    "patient": {},
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# 2. Canonical 80 y.o. H&N SCC (the audit-log case)
# ────────────────────────────────────────────────────────────────────────────

HN_SCC_MULTI_AXIS: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "head_neck",
        "site_detail": "oral_cavity",
        "histology": "scc",
        "stage": "II",
        "tnm_n": "0",
        # Raw biomarker string as it appears in the audit log — we want
        # the tests to capture what the matcher does with this literal
        "biomarkers": ["CPS score of 100"],
    },
    "patient": {
        "age": 80,
        "gender": "male",
    },
    "treatment": {
        "modality": "chemotherapy",   # note: upstream polarity bug
        "setting": "salvage",
    },
}


# ────────────────────────────────────────────────────────────────────────────
# 3. Triple-negative breast cancer
# ────────────────────────────────────────────────────────────────────────────

TRIPLE_NEGATIVE_BREAST: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "breast",
        "histology": "ductal",
        "stage": "III",
        "receptor_status": "triple negative",
        "biomarkers": ["triple negative", "ER-", "PR-", "HER2-"],
    },
    "patient": {
        "age": 55,
        "gender": "female",
    },
    "treatment": {
        "setting": "neoadjuvant",
    },
}


# ────────────────────────────────────────────────────────────────────────────
# 4. Prostate castration-resistant
# ────────────────────────────────────────────────────────────────────────────

PROSTATE_CRPC: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "prostate",
        "histology": "adenocarcinoma",
        "stage": "IV",
        "disease_descriptor": "metastatic",
        "biomarkers": ["BRCA mutant"],
    },
    "patient": {
        "age": 72,
        "gender": "male",
    },
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# 5. Colorectal MSI-high
# ────────────────────────────────────────────────────────────────────────────

COLORECTAL_MSI_HIGH: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "gi",
        "site_detail": "colon",
        "histology": "adenocarcinoma",
        "stage": "IV",
        "disease_descriptor": "metastatic",
        "biomarkers": ["MSI-H", "KRAS wild-type"],
    },
    "patient": {
        "age": 62,
    },
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# 6. Breast HER2-neg with liver mets (stresses the hard site filter)
# ────────────────────────────────────────────────────────────────────────────

BREAST_HER2_NEG_LIVER_METS: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "breast",
        "histology": "ductal",
        "stage": "IV",
        "disease_descriptor": "metastatic",
        "receptor_status": "ER+/PR+/HER2-",
        "biomarkers": ["ER+", "PR+", "HER2-"],
    },
    "patient": {
        "age": 58,
        "gender": "female",
    },
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# 7. Empty query
# ────────────────────────────────────────────────────────────────────────────

EMPTY_QUERY: Dict[str, Any] = {
    "query_type": "general",
    "has_patient_context": False,
    "cancer": {},
    "patient": {},
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# 8. Biomarker only (no cancer_site → no hard filter)
# ────────────────────────────────────────────────────────────────────────────

BIOMARKER_ONLY: Dict[str, Any] = {
    "query_type": "trial_results",
    "has_patient_context": True,
    "cancer": {
        "biomarkers": ["KRAS G12C"],
    },
    "patient": {},
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# 9. "CPS score of 100" raw string — bug record
# ────────────────────────────────────────────────────────────────────────────

CPS_SCORE_OF_100_RAW: Dict[str, Any] = {
    "query_type": "treatment_recommendation",
    "has_patient_context": True,
    "cancer": {
        "site": "head_neck",
        "histology": "scc",
        "biomarkers": ["CPS score of 100"],
    },
    "patient": {},
    "treatment": {},
}


# ────────────────────────────────────────────────────────────────────────────
# Registry — iterate over this in parameterized tests
# ────────────────────────────────────────────────────────────────────────────

GOLDEN_FIXTURES: Dict[str, Dict[str, Any]] = {
    "simple_lung_adeno": SIMPLE_LUNG_ADENO,
    "hn_scc_multi_axis": HN_SCC_MULTI_AXIS,
    "triple_negative_breast": TRIPLE_NEGATIVE_BREAST,
    "prostate_crpc": PROSTATE_CRPC,
    "colorectal_msi_high": COLORECTAL_MSI_HIGH,
    "breast_her2_neg_liver_mets": BREAST_HER2_NEG_LIVER_METS,
    "empty_query": EMPTY_QUERY,
    "biomarker_only": BIOMARKER_ONLY,
    "cps_score_of_100_raw": CPS_SCORE_OF_100_RAW,
}
