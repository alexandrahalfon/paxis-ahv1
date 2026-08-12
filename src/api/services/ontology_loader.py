"""
Ontology Data Loader

Loads keyword and cancer-type data from the JSON files in data/ at startup,
providing structured access for inference, sub-query generation, and expansion.

Files loaded:
  - data/keywords/extractor_keywords.json     → keyword categories per clinical axis
  - data/ajcc_staging_tables.json             → cancer type aliases + staging tables
  - data/ontology/cancer_type_ontology.json   → rich cancer-type synonyms, keywords, drugs, subtypes
  - data/ontology/clinical_trial_ontology.json → clinical trial vocabulary (biomarkers, treatment, outcomes, ICI)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set


# ── File paths ────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_KEYWORDS_PATH = _DATA_DIR / "keywords" / "extractor_keywords.json"
_AJCC_PATH = _DATA_DIR / "ajcc_staging_tables.json"
_CANCER_ONTOLOGY_PATH = _DATA_DIR / "ontology" / "cancer_type_ontology.json"
_TRIAL_ONTOLOGY_PATH = _DATA_DIR / "ontology" / "clinical_trial_ontology.json"


# ── Loaders (cached singletons) ──────────────────────────────────────────────


def _load_json(path: Path, label: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[OntologyLoader] Failed to load {label}: {e}")
        return {}


@lru_cache(maxsize=1)
def load_extractor_keywords() -> dict:
    """Load extractor_keywords.json. Returns raw dict."""
    return _load_json(_KEYWORDS_PATH, "extractor_keywords.json")


@lru_cache(maxsize=1)
def load_ajcc_staging() -> dict:
    """Load ajcc_staging_tables.json. Returns raw dict."""
    return _load_json(_AJCC_PATH, "ajcc_staging_tables.json")


@lru_cache(maxsize=1)
def load_cancer_type_ontology() -> dict:
    """Load cancer_type_ontology.json (synonyms, keywords, drugs, subtypes per cancer type)."""
    return _load_json(_CANCER_ONTOLOGY_PATH, "cancer_type_ontology.json")


@lru_cache(maxsize=1)
def load_clinical_trial_ontology() -> dict:
    """Load clinical_trial_ontology.json (biomarkers, treatment, outcomes, ICI vocabulary)."""
    return _load_json(_TRIAL_ONTOLOGY_PATH, "clinical_trial_ontology.json")


# ── Cancer-type synonyms (merged AJCC + cancer_type_ontology) ─────────────────


@lru_cache(maxsize=1)
def get_cancer_type_synonyms() -> Dict[str, List[str]]:
    """
    Build a cancer-type synonym map from AJCC aliases AND
    cancer_type_ontology.json (synonyms + keywords + subtypes).

    Returns:
        Dict mapping each alias/keyword (lowercased) to the full combined
        synonym list for that cancer type.
    """
    synonym_map: Dict[str, List[str]] = {}

    # Layer 1: AJCC aliases
    ajcc = load_ajcc_staging()
    for cancer_key, entry in ajcc.items():
        if cancer_key == "_metadata":
            continue
        aliases = entry.get("aliases", [])
        if not aliases:
            continue
        for alias in aliases:
            synonym_map[alias.lower()] = aliases

    # Layer 2: cancer_type_ontology.json — richer synonyms, keywords, subtypes
    ct_ontology = load_cancer_type_ontology()
    for cancer_key, entry in ct_ontology.items():
        if not isinstance(entry, dict):
            continue
        all_terms: List[str] = []
        all_terms.extend(entry.get("synonyms", []))
        all_terms.extend(entry.get("keywords", []))
        all_terms.extend(entry.get("subtypes", []))
        if not all_terms:
            continue
        # Merge with existing AJCC entries when there's overlap
        for term in list(all_terms):
            key = term.lower()
            if key in synonym_map:
                # Merge: union of both lists, deduplicated
                merged = list(dict.fromkeys(synonym_map[key] + all_terms))
                for t in all_terms:
                    synonym_map[t.lower()] = merged
                break
        else:
            # No overlap with AJCC — add as new entries
            for term in all_terms:
                synonym_map[term.lower()] = all_terms

    return synonym_map


@lru_cache(maxsize=1)
def get_cancer_type_to_key() -> Dict[str, str]:
    """
    Map every alias (lowercased) to its canonical AJCC key.

    E.g. "tongue" → "head_neck_oral_cavity"
    """
    ajcc = load_ajcc_staging()
    mapping: Dict[str, str] = {}
    for cancer_key, entry in ajcc.items():
        if cancer_key == "_metadata":
            continue
        for alias in entry.get("aliases", []):
            mapping[alias.lower()] = cancer_key
    return mapping


@lru_cache(maxsize=1)
def get_cancer_drugs(cancer_key: str) -> List[str]:
    """Return drug list for a cancer type from cancer_type_ontology.json."""
    ct = load_cancer_type_ontology()
    entry = ct.get(cancer_key, {})
    return entry.get("drugs", []) if isinstance(entry, dict) else []


@lru_cache(maxsize=1)
def get_biomarker_keywords() -> Dict[str, List[str]]:
    """
    Return biomarker keyword categories merged from extractor_keywords.json
    AND clinical_trial_ontology.json → biomarkers section.

    Keys include: protein_biomarkers, hormone_receptors, her2,
    genetic_mutations, immunotherapy_markers, viral_markers, msi_mmr, tmb,
    plus trial-ontology keys: PD_L1, MSI_MMR (trial), TMB (trial), etc.
    """
    result: Dict[str, List[str]] = {}

    # Base: extractor_keywords.json
    kw = load_extractor_keywords()
    for k, v in kw.get("biomarker_keywords", {}).items():
        result[k] = list(v)

    # Merge: clinical_trial_ontology.json → biomarkers
    trial = load_clinical_trial_ontology()
    trial_bio = trial.get("biomarkers", {})
    for section_key, section_val in trial_bio.items():
        if isinstance(section_val, list):
            # Flat list (e.g. protein_IHC, assays)
            existing = result.get(section_key, [])
            merged = list(dict.fromkeys(existing + _clean_terms(section_val)))
            result[section_key] = merged
        elif isinstance(section_val, dict):
            # Nested dict (e.g. immunotherapy_markers → PD_L1, MSI_MMR, TMB)
            for sub_key, sub_val in section_val.items():
                if isinstance(sub_val, list):
                    flat_key = f"{section_key}_{sub_key}"
                    existing = result.get(flat_key, [])
                    merged = list(dict.fromkeys(existing + _clean_terms(sub_val)))
                    result[flat_key] = merged

    return result


@lru_cache(maxsize=1)
def get_biomarker_terms_flat() -> Set[str]:
    """Flat set of all biomarker terms (lowercased) for pattern matching."""
    terms: Set[str] = set()
    for category_terms in get_biomarker_keywords().values():
        for t in category_terms:
            terms.add(t.lower())
    return terms


@lru_cache(maxsize=1)
def get_ici_resistance_terms() -> List[str]:
    """Return ICI resistance/refractory terms from clinical_trial_ontology.json."""
    trial = load_clinical_trial_ontology()
    ici = trial.get("immunotherapy_specific", {})
    return _clean_terms(ici.get("resistance", []))


@lru_cache(maxsize=1)
def get_metastatic_pattern_terms() -> List[str]:
    """Return metastatic pattern terms from clinical_trial_ontology.json."""
    trial = load_clinical_trial_ontology()
    staging = trial.get("staging_and_risk", {})
    return _clean_terms(staging.get("metastatic_patterns", []))


@lru_cache(maxsize=1)
def get_treatment_setting_terms() -> List[str]:
    """Return treatment setting terms from clinical_trial_ontology.json."""
    trial = load_clinical_trial_ontology()
    treatment = trial.get("treatment", {})
    return _clean_terms(treatment.get("setting", []))


def _clean_terms(terms: List[str]) -> List[str]:
    """Clean ontology terms: replace underscores with spaces, strip whitespace."""
    return [t.replace("_", " ").strip() for t in terms if t]


@lru_cache(maxsize=1)
def get_staging_keywords() -> Dict[str, List[str]]:
    """
    Return cancer staging keyword categories.

    Keys: clinical_stage, pathologic_stage, tnm_components, stage_groups
    """
    kw = load_extractor_keywords()
    return kw.get("cancer_staging_keywords", {})


@lru_cache(maxsize=1)
def get_treatment_keywords() -> Dict[str, List[str]]:
    """
    Return treatment keyword categories.

    Keys: setting, modality, duration, response
    """
    kw = load_extractor_keywords()
    return kw.get("treatment_keywords", {})


@lru_cache(maxsize=1)
def get_treatment_technique_keywords() -> Dict[str, List[str]]:
    """
    Return treatment technique keywords.

    Keys: radiation, surgery, chemotherapy
    """
    kw = load_extractor_keywords()
    return kw.get("treatment_technique_keywords", {})


@lru_cache(maxsize=1)
def get_outcome_keywords() -> Dict[str, List[str]]:
    """
    Return outcome keyword categories.

    Keys: survival_metrics, response_metrics, qol_metrics, safety_metrics, ...
    """
    kw = load_extractor_keywords()
    return kw.get("outcome_keywords", {})


@lru_cache(maxsize=1)
def get_tumor_keywords() -> Dict[str, List[str]]:
    """
    Return tumor characteristics keywords.

    Keys: invasion, margins, differentiation_grade, biomarkers, ...
    """
    kw = load_extractor_keywords()
    return kw.get("tumor_characteristics_keywords", {})


@lru_cache(maxsize=1)
def get_patient_keywords() -> Dict[str, List[str]]:
    """
    Return patient characteristics keywords.

    Keys: age, performance_status, comorbidities, eligibility, ...
    """
    kw = load_extractor_keywords()
    return kw.get("patient_characteristics_keywords", {})


# ── Axis-specific keyword sets for sub-query enrichment ───────────────────────

def get_axis_keywords(axis_name: str) -> List[str]:
    """
    Return a flat list of keywords relevant to a specific clinical axis.

    Merges data from extractor_keywords.json AND clinical_trial_ontology.json
    to provide the richest possible term set for each axis.
    """
    axis_to_categories = {
        "primary_cancer": lambda: (
            get_tumor_keywords().get("cell_type", [])
            + get_tumor_keywords().get("location", [])[:6]
            + load_extractor_keywords().get("cancer_diagnosis_keywords", {}).get("cancer_type", [])
            + _clean_terms(load_clinical_trial_ontology().get("diagnosis", {}).get("histology", []))[:10]
        ),
        "tnm_pathology": lambda: (
            get_staging_keywords().get("tnm_components", [])
            + get_staging_keywords().get("stage_groups", [])
            + get_tumor_keywords().get("invasion", [])
            + get_tumor_keywords().get("margins", [])
            + get_tumor_keywords().get("differentiation_grade", [])
            + _clean_terms(load_clinical_trial_ontology().get("staging_and_risk", {}).get("tnm_components", {}).get("prefixes", []))
        ),
        "prior_definitive_treatment": lambda: (
            get_treatment_technique_keywords().get("surgery", [])
            + get_treatment_keywords().get("setting", [])
            + _clean_terms(load_clinical_trial_ontology().get("treatment", {}).get("surgery", []))[:15]
        ),
        "current_treatment": lambda: (
            get_treatment_keywords().get("modality", [])
            + get_treatment_keywords().get("response", [])
            + get_treatment_technique_keywords().get("chemotherapy", [])[:8]
            + get_treatment_setting_terms()[:10]
        ),
        "biomarker_profile": lambda: (
            get_biomarker_keywords().get("immunotherapy_markers", [])
            + get_biomarker_keywords().get("hormone_receptors", [])
            + get_biomarker_keywords().get("genetic_mutations", [])[:10]
            + get_biomarker_keywords().get("viral_markers", [])
            + get_biomarker_keywords().get("msi_mmr", [])
            # Trial ontology: CPS thresholds, PD-L1 scoring
            + get_biomarker_keywords().get("immunotherapy_markers_PD_L1", [])
            + get_biomarker_keywords().get("immunotherapy_markers_MSI_MMR", [])
        ),
        "disease_trajectory": lambda: (
            get_tumor_keywords().get("distant_spread", [])
            + get_treatment_keywords().get("response", [])
            + get_ici_resistance_terms()
        ),
        "metastatic_concern": lambda: (
            get_tumor_keywords().get("distant_spread", [])
            + get_tumor_keywords().get("lymph_nodes", [])
            + get_metastatic_pattern_terms()
        ),
        "patient_factors": lambda: (
            get_patient_keywords().get("comorbidities", [])
            + get_patient_keywords().get("performance_status", [])
            + get_patient_keywords().get("eligibility", [])
            + _clean_terms(load_clinical_trial_ontology().get("patient_characteristics", {}).get("comorbidities", []))
        ),
    }

    factory = axis_to_categories.get(axis_name)
    if factory:
        return factory()
    return []


def expand_cancer_site_synonyms(site_text: str) -> List[str]:
    """
    Given a cancer site string (e.g. "oral tongue", "nsclc"), return
    all known synonyms for that cancer type from both AJCC aliases
    and cancer_type_ontology.json (synonyms + keywords + subtypes).

    Returns empty list if no match found.
    """
    if not site_text:
        return []
    site_lower = site_text.lower().strip()
    synonyms = get_cancer_type_synonyms()

    # Direct match
    if site_lower in synonyms:
        return synonyms[site_lower]

    # Substring match: check if any alias is contained in the site text
    for alias, all_aliases in synonyms.items():
        if alias in site_lower or site_lower in alias:
            return all_aliases

    return []


def get_cancer_type_context(site_text: str) -> Dict[str, List[str]]:
    """
    Given a cancer site string, return the full context from
    cancer_type_ontology.json: synonyms, keywords, drugs, subtypes.

    Returns empty dict if no match.
    """
    if not site_text:
        return {}
    site_lower = site_text.lower().strip()
    ct = load_cancer_type_ontology()

    for cancer_key, entry in ct.items():
        if not isinstance(entry, dict):
            continue
        all_terms = (
            entry.get("synonyms", [])
            + entry.get("keywords", [])
        )
        for term in all_terms:
            if term.lower() in site_lower or site_lower in term.lower():
                return {
                    "synonyms": entry.get("synonyms", []),
                    "keywords": entry.get("keywords", []),
                    "drugs": entry.get("drugs", []),
                    "subtypes": entry.get("subtypes", []),
                }

    return {}


# ── Site-aware histology / surgery lookups ───────────────────────────────────
#
# These are used by patient_eligibility_boost_service.extract_patient_context_from_query
# to disambiguate histology / surgery mentions for multi-cancer patients. Before
# this layer existed, the extractor walked a flat regex list with first-match-wins
# semantics — so a patient with "transverse colon ADENOCARCINOMA ... and ... oral
# tongue SQUAMOUS CELL CARCINOMA" would be labeled histology=adenocarcinoma and
# downstream eligibility filters would hard-reject the correct H&N SCC studies.

#: Map the cancer_type strings that `extract_patient_context_from_query` produces
#: (e.g. "head and neck cancer") to the cancer_type_ontology.json site key
#: (e.g. "h_n"). Used to look up plausible histologies/surgeries per site.
CANCER_TYPE_LABEL_TO_SITE_KEY: Dict[str, str] = {
    "breast cancer": "breast",
    "lung cancer": "lung",
    "prostate cancer": "prostate",
    "colorectal cancer": "gi",
    "head and neck cancer": "h_n",
    "brain cancer": "cns",
    "melanoma": "cutaneous",
    "pancreatic cancer": "gi",
    "ovarian cancer": "gyn",
    "cervical cancer": "gyn",
    "esophageal cancer": "gi",
    "gastric cancer": "gi",
    "liver cancer": "gi",
    "renal cancer": "gu",
    "bladder cancer": "gu",
}


@lru_cache(maxsize=1)
def get_site_histology_map() -> Dict[str, Set[str]]:
    """Return a dict mapping each ontology site key to the set of canonical
    histology strings plausible for that site.

    All strings are lowercased for case-insensitive comparison.
    """
    ct = load_cancer_type_ontology()
    result: Dict[str, Set[str]] = {}
    for site_key, entry in ct.items():
        if not isinstance(entry, dict):
            continue
        histologies = entry.get("histologies", [])
        if histologies:
            result[site_key] = {h.lower() for h in histologies if isinstance(h, str)}
    return result


@lru_cache(maxsize=1)
def get_site_surgery_map() -> Dict[str, Set[str]]:
    """Return a dict mapping each ontology site key to the set of canonical
    surgery strings plausible for that site."""
    ct = load_cancer_type_ontology()
    result: Dict[str, Set[str]] = {}
    for site_key, entry in ct.items():
        if not isinstance(entry, dict):
            continue
        surgeries = entry.get("surgeries", [])
        if surgeries:
            result[site_key] = {s.lower() for s in surgeries if isinstance(s, str)}
    return result


def get_plausible_histologies_for_site(site_key: str) -> Set[str]:
    """Return the set of canonical histology strings plausible for a site.
    Empty set if the site is unknown or has no registered histologies."""
    return get_site_histology_map().get(site_key, set())


def get_plausible_surgeries_for_site(site_key: str) -> Set[str]:
    """Return the set of canonical surgery strings plausible for a site.
    Empty set if the site is unknown or has no registered surgeries."""
    return get_site_surgery_map().get(site_key, set())


def cancer_type_label_to_site_key(cancer_type_label: Optional[str]) -> Optional[str]:
    """Map a cancer_type label emitted by the patient extractor (e.g.
    "head and neck cancer") to its ontology site key (e.g. "h_n").

    Returns None if no mapping exists.
    """
    if not cancer_type_label:
        return None
    return CANCER_TYPE_LABEL_TO_SITE_KEY.get(cancer_type_label.lower().strip())


def is_histology_plausible_for_site(histology: str, site_key: Optional[str]) -> bool:
    """Return True if the given canonical histology string is registered as
    plausible for the given site_key. Returns True (permissive) when:
      - site_key is None (no site information to check against)
      - the site has no registered histologies (not enumerated in the ontology)
    """
    if not site_key or not histology:
        return True
    plausible = get_plausible_histologies_for_site(site_key)
    if not plausible:
        # Ontology has no enumeration for this site — don't over-filter
        return True
    return histology.lower() in plausible


def is_surgery_plausible_for_site(surgery: str, site_key: Optional[str]) -> bool:
    """Return True if the given canonical surgery string is registered as
    plausible for the given site_key. Permissive fallback same as
    `is_histology_plausible_for_site`."""
    if not site_key or not surgery:
        return True
    plausible = get_plausible_surgeries_for_site(site_key)
    if not plausible:
        return True
    return surgery.lower() in plausible


# ---------------------------------------------------------------------------
# Unified site-keyword map + phrase → site_key resolver
# ---------------------------------------------------------------------------
# Bridge between the two site_key naming conventions used in the codebase:
#   - ontology JSON keys (lowercase, e.g. "h_n", "breast")
#   - SITE_KEYWORDS capitalized keys (e.g. "H&N", "Breast") from
#     enhanced_rag_service.py
SITE_KEY_CAP_TO_ONTOLOGY: Dict[str, str] = {
    "Breast":     "breast",
    "Lung":       "lung",
    "Prostate":   "prostate",
    "GI":         "gi",
    "H&N":        "h_n",
    "CNS":        "cns",
    "Cutaneous":  "cutaneous",
    "GYN":        "gyn",
    "GU":         "gu",
    "Sarcoma":    "sarcoma",
    "Lymphoma":   "lymphoma",
    "Thyroid":    "thyroid",
    "Peds":       "peds",
}


@lru_cache(maxsize=1)
def get_site_keyword_map() -> Dict[str, Set[str]]:
    """Return ontology site_key → union of keywords / subtypes / synonyms /
    label / SITE_KEYWORDS phrases (all lowercased). Used by
    `resolve_phrase_to_site_key` and `is_same_site_family` to decide whether
    two free-text cancer phrases belong to the same ontology family.
    """
    ct = load_cancer_type_ontology()
    result: Dict[str, Set[str]] = {}
    for site_key, entry in ct.items():
        if not isinstance(entry, dict):
            continue
        terms: Set[str] = set()
        for field in ("keywords", "subtypes", "synonyms"):
            for val in entry.get(field, []) or []:
                if isinstance(val, str) and val.strip():
                    terms.add(val.lower().strip())
        label = entry.get("label")
        if isinstance(label, str) and label.strip():
            terms.add(label.lower().strip())
        if terms:
            result[site_key] = terms

    # Augment with SITE_KEYWORDS from enhanced_rag_service for subsites the
    # ontology file doesn't enumerate (e.g. "oral cavity", "maxilla").
    try:
        from src.api.services.enhanced_rag_service import SITE_KEYWORDS
        for cap_key, kws in SITE_KEYWORDS:
            ontology_key = SITE_KEY_CAP_TO_ONTOLOGY.get(cap_key)
            if not ontology_key:
                continue
            bucket = result.setdefault(ontology_key, set())
            for kw in kws or []:
                if isinstance(kw, str) and kw.strip():
                    bucket.add(kw.lower().strip())
    except Exception:
        pass
    return result


def resolve_phrase_to_site_key(phrase: Optional[str]) -> Optional[str]:
    """Resolve a free-text cancer phrase to its ontology site_key.

    Tries, in order: exact cancer_type label lookup → longest keyword match
    against `get_site_keyword_map()`. Returns None when nothing plausibly
    matches — callers should treat None as "unknown", not as "matches".
    """
    if not phrase:
        return None
    phrase_lower = phrase.lower().strip()
    direct = cancer_type_label_to_site_key(phrase_lower)
    if direct:
        return direct
    best_site: Optional[str] = None
    best_len = 0
    for site_key, keywords in get_site_keyword_map().items():
        for kw in keywords:
            if not kw:
                continue
            if kw == phrase_lower or kw in phrase_lower:
                if len(kw) > best_len:
                    best_len = len(kw)
                    best_site = site_key
    return best_site


def is_same_site_family(phrase_a: Optional[str], phrase_b: Optional[str]) -> bool:
    """True iff both phrases resolve to the same ontology site_key.

    Used by the eligibility hard-filter to detect that an LLM-reported
    cancer_type MISMATCH is really a subsite-hierarchy artifact (patient
    "oral cavity SCC" vs. study "head and neck" umbrella cohort — both
    resolve to h_n). Returns False when either side fails to resolve, so
    cross-organ mismatches (lung vs. h_n) still hard-filter correctly.
    """
    sa = resolve_phrase_to_site_key(phrase_a)
    sb = resolve_phrase_to_site_key(phrase_b)
    return bool(sa and sb and sa == sb)


# ── Expansion table access (for per-axis expansion) ──────────────────────────

_expansion_tables_cache: Dict[str, dict] = {}


def get_expansion_tables() -> Dict[str, dict]:
    """
    Lazily load the four expansion dicts from enhanced_rag_service.py.

    Returns dict with keys: "oncology", "reverse", "staging", "clinical"
    mapping to ONCOLOGY_EXPANSIONS, REVERSE_EXPANSIONS, STAGING_SYNONYMS,
    CLINICAL_SYNONYMS respectively.

    Uses a module-level cache and deferred import to avoid pulling in
    qdrant_client and other heavy dependencies at import time.
    """
    if _expansion_tables_cache:
        return _expansion_tables_cache

    try:
        from src.api.services.enhanced_rag_service import (
            ONCOLOGY_EXPANSIONS,
            REVERSE_EXPANSIONS,
            STAGING_SYNONYMS,
            CLINICAL_SYNONYMS,
        )
        _expansion_tables_cache["oncology"] = ONCOLOGY_EXPANSIONS
        _expansion_tables_cache["reverse"] = REVERSE_EXPANSIONS
        _expansion_tables_cache["staging"] = STAGING_SYNONYMS
        _expansion_tables_cache["clinical"] = CLINICAL_SYNONYMS
    except Exception as e:
        print(f"[OntologyLoader] Failed to load expansion tables: {e}")
        _expansion_tables_cache["oncology"] = {}
        _expansion_tables_cache["reverse"] = {}
        _expansion_tables_cache["staging"] = {}
        _expansion_tables_cache["clinical"] = {}

    return _expansion_tables_cache
