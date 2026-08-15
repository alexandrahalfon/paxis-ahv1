"""
Comprehensive Query Expansion

Centralised expansion layer that enriches a query string with ALL known
synonyms, abbreviations, ontology terms, brand names, staging variants,
and clinical-context expansions **before** the embedding step.  The
expanded text is concatenated with the original query so the embedding
vector sits in the right region of the latent space regardless of which
vocabulary the query or the documents happen to use.

Data sources (loaded once at import time):

  data/ontology/cancer_type_ontology.json
      22 cancer categories, each with: synonyms, keywords, drugs,
      subtypes, histologies, surgeries

  data/ontology/clinical_trial_ontology.json
      Controlled trial vocabulary: study designs, endpoints, phase
      terms, biomarker vocabulary, cooperative group names

  data/keywords/extractor_keywords.json
      1200+ extraction keywords grouped by category (tumour
      characteristics, staging, biomarkers, treatment, outcomes, etc.)

  data/ajcc_staging_tables.json
      35 cancer-type staging tables with aliases, TNM → stage-group
      mappings, and T/N/M definitions

Plus hard-coded tables for:

  - Drug brand-name ↔ generic ↔ mechanism class
  - Clinical context / disease status synonyms
  - Full staging notation expansion (Stage III ↔ stage 3 ↔ III ↔
    T3N1M0 ↔ cT3N1 ↔ pT3pN1 etc.)
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# ── Locate data/ directory ──────────────────────────────────────────────────
# Works whether the process cwd is the repo root or a subdirectory.

_THIS_DIR = Path(__file__).resolve().parent          # src/api/services
_REPO_ROOT = _THIS_DIR.parent.parent.parent          # repo root
_DATA_DIR = _REPO_ROOT / "data"


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD JSON DATA FILES
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(relpath: str) -> dict:
    p = _DATA_DIR / relpath
    if not p.exists():
        print(f"[QueryExpansion] WARNING: {p} not found — skipping")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


_CANCER_ONTOLOGY:    dict = _load_json("ontology/cancer_type_ontology.json")
_TRIAL_ONTOLOGY:     dict = _load_json("ontology/clinical_trial_ontology.json")
_EXTRACTOR_KEYWORDS: dict = _load_json("keywords/extractor_keywords.json")
_AJCC_STAGING:       dict = _load_json("ajcc_staging_tables.json")


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD FLAT LOOKUP TABLES FROM THE LOADED JSON
# ═══════════════════════════════════════════════════════════════════════════

# 2a. Cancer-type synonym map: trigger → set of expansion terms
#     e.g. "breast carcinoma" → {"breast cancer", "mammary carcinoma", ...}
_CANCER_SYNONYM_MAP: Dict[str, Set[str]] = {}

for _cat, _data in _CANCER_ONTOLOGY.items():
    if not isinstance(_data, dict):
        continue
    label = (_data.get("label") or "").strip()
    synonyms = list(_data.get("synonyms") or [])
    keywords = list(_data.get("keywords") or [])
    drugs    = list(_data.get("drugs") or [])
    subtypes = list(_data.get("subtypes") or [])
    histologies = list(_data.get("histologies") or [])

    # Every synonym / keyword / label can trigger the full set
    all_terms = set()
    if label:
        all_terms.add(label)
    all_terms.update(s for s in synonyms if s)
    all_terms.update(k for k in keywords if k)
    all_terms.update(d for d in drugs if d)
    all_terms.update(st for st in subtypes if st)
    all_terms.update(h for h in histologies if h)

    for trigger in list(all_terms):
        trigger_lower = trigger.lower().strip()
        if trigger_lower and len(trigger_lower) >= 2:
            _CANCER_SYNONYM_MAP[trigger_lower] = all_terms


# 2b. Drug brand-name ↔ generic ↔ mechanism class
#     Each group is a set of interchangeable names.  When any member
#     appears in the query, ALL other members are appended.
DRUG_SYNONYM_GROUPS: List[Set[str]] = [
    # Checkpoint inhibitors — anti-PD1
    {"pembrolizumab", "keytruda", "MK-3475", "anti-PD1", "anti-PD-1",
     "immune checkpoint inhibitor", "ICI", "CPI"},
    {"nivolumab", "opdivo", "BMS-936558", "anti-PD1", "anti-PD-1",
     "immune checkpoint inhibitor", "ICI", "CPI"},
    {"cemiplimab", "libtayo", "anti-PD1", "anti-PD-1", "ICI"},
    # Checkpoint inhibitors — anti-PD-L1
    {"atezolizumab", "tecentriq", "anti-PD-L1", "ICI", "CPI"},
    {"durvalumab", "imfinzi", "anti-PD-L1", "ICI", "CPI"},
    {"avelumab", "bavencio", "anti-PD-L1", "ICI"},
    # Checkpoint inhibitors — anti-CTLA4
    {"ipilimumab", "yervoy", "anti-CTLA4", "anti-CTLA-4", "ICI"},
    {"tremelimumab", "imjudo", "anti-CTLA4", "ICI"},
    # HER2-targeted
    {"trastuzumab", "herceptin", "anti-HER2", "HER2-targeted"},
    {"pertuzumab", "perjeta", "anti-HER2", "HER2-targeted"},
    {"trastuzumab deruxtecan", "T-DXd", "enhertu", "DS-8201", "anti-HER2 ADC"},
    {"trastuzumab emtansine", "T-DM1", "kadcyla", "anti-HER2 ADC"},
    {"lapatinib", "tykerb", "HER2 TKI"},
    {"tucatinib", "tukysa", "HER2 TKI"},
    # EGFR-targeted
    {"cetuximab", "erbitux", "anti-EGFR"},
    {"panitumumab", "vectibix", "anti-EGFR"},
    {"osimertinib", "tagrisso", "EGFR TKI", "3rd-gen EGFR inhibitor"},
    {"erlotinib", "tarceva", "EGFR TKI"},
    {"gefitinib", "iressa", "EGFR TKI"},
    # VEGF / angiogenesis
    {"bevacizumab", "avastin", "anti-VEGF"},
    {"ramucirumab", "cyramza", "anti-VEGFR2"},
    {"lenvatinib", "lenvima", "multi-kinase inhibitor", "VEGFR TKI"},
    {"cabozantinib", "cabometyx", "cometriq", "multi-kinase inhibitor"},
    {"sunitinib", "sutent", "VEGFR TKI", "multi-kinase inhibitor"},
    # PARP inhibitors
    {"olaparib", "lynparza", "PARP inhibitor", "PARPi"},
    {"niraparib", "zejula", "PARP inhibitor", "PARPi"},
    {"rucaparib", "rubraca", "PARP inhibitor", "PARPi"},
    {"talazoparib", "talzenna", "PARP inhibitor", "PARPi"},
    # CDK4/6 inhibitors
    {"palbociclib", "ibrance", "CDK4/6 inhibitor"},
    {"ribociclib", "kisqali", "CDK4/6 inhibitor"},
    {"abemaciclib", "verzenio", "CDK4/6 inhibitor"},
    # Endocrine / hormonal
    {"tamoxifen", "nolvadex", "SERM", "endocrine therapy", "hormonal therapy"},
    {"anastrozole", "arimidex", "aromatase inhibitor", "AI", "endocrine therapy"},
    {"letrozole", "femara", "aromatase inhibitor", "AI", "endocrine therapy"},
    {"exemestane", "aromasin", "aromatase inhibitor", "AI"},
    {"enzalutamide", "xtandi", "androgen receptor inhibitor", "AR inhibitor"},
    {"abiraterone", "zytiga", "CYP17 inhibitor", "androgen biosynthesis inhibitor"},
    {"apalutamide", "erleada", "androgen receptor inhibitor"},
    {"darolutamide", "nubeqa", "androgen receptor inhibitor"},
    # Classic chemo
    {"cisplatin", "CDDP", "platinum", "platinum-based"},
    {"carboplatin", "paraplatin", "platinum", "platinum-based"},
    {"oxaliplatin", "eloxatin", "platinum", "platinum-based"},
    {"paclitaxel", "taxol", "taxane"},
    {"docetaxel", "taxotere", "taxane"},
    {"nab-paclitaxel", "abraxane", "taxane"},
    {"doxorubicin", "adriamycin", "anthracycline"},
    {"epirubicin", "ellence", "anthracycline"},
    {"cyclophosphamide", "cytoxan", "alkylating agent"},
    {"ifosfamide", "ifex", "alkylating agent"},
    {"gemcitabine", "gemzar"},
    {"capecitabine", "xeloda", "oral fluoropyrimidine", "5-FU prodrug"},
    {"5-FU", "fluorouracil", "5-fluorouracil", "fluoropyrimidine"},
    {"irinotecan", "camptosar", "topoisomerase I inhibitor"},
    {"topotecan", "hycamtin", "topoisomerase I inhibitor"},
    {"etoposide", "VP-16", "toposar", "topoisomerase II inhibitor"},
    {"vincristine", "oncovin", "vinca alkaloid"},
    {"vinorelbine", "navelbine", "vinca alkaloid"},
    {"temozolomide", "temodar", "TMZ", "alkylating agent"},
    {"methotrexate", "MTX", "trexall", "antifolate"},
    {"pemetrexed", "alimta", "antifolate"},
    # Regimen abbreviations
    {"FOLFOX", "5-FU leucovorin oxaliplatin"},
    {"FOLFIRI", "5-FU leucovorin irinotecan"},
    {"FOLFIRINOX", "5-FU leucovorin irinotecan oxaliplatin"},
    {"TCHP", "docetaxel carboplatin trastuzumab pertuzumab"},
    {"AC-T", "doxorubicin cyclophosphamide paclitaxel"},
    {"AC", "doxorubicin cyclophosphamide", "adriamycin cyclophosphamide"},
    {"TPF", "docetaxel cisplatin 5-FU", "induction chemotherapy"},
    {"XELOX", "capecitabine oxaliplatin"},
    {"CAPOX", "capecitabine oxaliplatin"},
    {"R-CHOP", "rituximab cyclophosphamide doxorubicin vincristine prednisone"},
    {"ABVD", "doxorubicin bleomycin vinblastine dacarbazine"},
    # Radiopharmaceuticals
    {"lutetium-177", "Lu-177", "177Lu-PSMA", "Pluvicto", "PSMA-targeted radionuclide"},
    {"radium-223", "Ra-223", "xofigo", "alpha emitter"},
    {"I-131", "radioiodine", "RAI", "sodium iodide I-131"},
]

# Build a fast lookup: lowercase drug name → set of all synonyms in its group
_DRUG_SYNONYM_LOOKUP: Dict[str, Set[str]] = {}
for _group in DRUG_SYNONYM_GROUPS:
    lower_group = {d.lower() for d in _group}
    for drug in _group:
        _DRUG_SYNONYM_LOOKUP[drug.lower()] = _group


# 2c. Clinical context / disease status synonyms
CLINICAL_CONTEXT_SYNONYMS: Dict[str, List[str]] = {
    "metastatic": [
        "metastatic", "metastases", "metastasis", "distant disease",
        "stage IV", "M1", "systemic disease", "disseminated",
        "widespread", "distant spread",
    ],
    "locally advanced": [
        "locally advanced", "regional", "locally extensive",
        "unresectable", "borderline resectable", "T4", "bulky",
    ],
    "unresectable": [
        "unresectable", "inoperable", "non-resectable",
        "not amenable to surgery", "surgically unresectable",
        "non-surgical", "non-surgical candidate",
    ],
    "recurrent": [
        "recurrent", "recurrence", "relapsed", "relapse",
        "disease recurrence", "locoregional recurrence",
        "local recurrence", "regional recurrence",
    ],
    "refractory": [
        "refractory", "resistant", "non-responsive",
        "treatment-resistant", "progressive", "progressing",
        "progression on", "failure", "failed",
    ],
    "oligometastatic": [
        "oligometastatic", "oligomet", "limited metastatic",
        "few metastases", "1-5 metastases", "oligo-recurrent",
    ],
    "de novo metastatic": [
        "de novo metastatic", "presentation with metastatic disease",
        "newly diagnosed metastatic", "initial metastatic",
    ],
    "early stage": [
        "early stage", "early-stage", "stage I", "stage II",
        "localised", "localized", "limited disease",
    ],
    "advanced": [
        "advanced", "late-stage", "stage III", "stage IV",
        "locally advanced", "metastatic", "unresectable",
    ],
    "salvage": [
        "salvage", "rescue", "re-treatment", "retreatment",
        "2nd-line", "second-line", "post-progression",
    ],
    "neoadjuvant": [
        "neoadjuvant", "preoperative", "pre-operative",
        "induction", "pre-surgical", "before surgery",
    ],
    "adjuvant": [
        "adjuvant", "postoperative", "post-operative",
        "post-surgical", "after surgery", "additional",
    ],
    "palliative": [
        "palliative", "symptom control", "comfort care",
        "supportive", "best supportive care", "BSC",
        "end of life", "hospice",
    ],
    "curative": [
        "curative", "curative-intent", "definitive",
        "radical", "curative treatment",
    ],
}

# Build fast lookup: lowercase trigger → set of all synonyms in its group
_CLINICAL_CONTEXT_LOOKUP: Dict[str, Set[str]] = {}
for _key, _terms in CLINICAL_CONTEXT_SYNONYMS.items():
    lower_terms = {t.lower() for t in _terms}
    for t in _terms:
        _CLINICAL_CONTEXT_LOOKUP[t.lower()] = set(_terms)


# 2d. Staging notation expansion — comprehensive T/N/M/Stage variants
#     This supplements the existing STAGING_SYNONYMS dict in enhanced_rag_service.py
#     with Roman ↔ Arabic ↔ clinical/pathologic prefix variants.

_STAGE_ROMAN_ARABIC = {
    "I": "1", "II": "2", "III": "3", "IV": "4",
    "1": "I", "2": "II", "3": "III", "4": "IV",
}

_STAGE_SUBSTAGES = {
    "I": ["IA", "IB", "IA1", "IA2", "IB1", "IB2"],
    "II": ["IIA", "IIB", "IIA1", "IIA2", "IIB1", "IIB2", "IIC"],
    "III": ["IIIA", "IIIB", "IIIC", "IIIC1", "IIIC2"],
    "IV": ["IVA", "IVB", "IVC"],
}


def _build_staging_expansions(query_text: str) -> Set[str]:
    """
    Extract staging terms from query and return ALL variant spellings.

    For "T4N0" returns: T4, pT4, cT4, ypT4, yT4, N0, pN0, cN0, ypN0, yN0,
    T4N0M0, pT4pN0, cT4cN0, Stage IV, stage 4, Stage IVA, etc.

    For "Stage III" returns: Stage III, stage 3, stage III, Stage IIIA,
    Stage IIIB, Stage IIIC, T3, T4, etc.
    """
    expansions: Set[str] = set()
    ql = query_text

    # ── T stage variants ────────────────────────────────────────────────
    t_match = re.findall(r'(?:c|p|yp|y)?T(\d[a-d]?(?:is)?)', ql, re.IGNORECASE)
    for t_val in t_match:
        base = f"T{t_val}"
        for prefix in ("", "p", "c", "yp", "y"):
            expansions.add(f"{prefix}{base}")
            expansions.add(f"{prefix.upper()}{base}" if prefix else base)
        expansions.add(f"T{t_val} disease")
        expansions.add(f"T-stage {t_val}")

    # ── N stage variants ────────────────────────────────────────────────
    n_match = re.findall(r'(?:c|p|yp|y)?N(\d[a-c]?(?:mi)?)', ql, re.IGNORECASE)
    for n_val in n_match:
        base = f"N{n_val}"
        for prefix in ("", "p", "c", "yp", "y"):
            expansions.add(f"{prefix}{base}")
            expansions.add(f"{prefix.upper()}{base}" if prefix else base)
        if n_val.startswith("0"):
            expansions.update(["node negative", "N0 disease", "no nodal involvement"])
        elif n_val.startswith("1"):
            expansions.update(["node positive", "N+ disease", "nodal involvement",
                               "N1 disease", "lymph node metastasis"])
        elif n_val.startswith("2") or n_val.startswith("3"):
            expansions.update(["node positive", "N+ disease", "advanced nodal disease",
                               f"N{n_val} disease"])

    # ── M stage variants ────────────────────────────────────────────────
    m_match = re.findall(r'(?:c|p)?M([01x])', ql, re.IGNORECASE)
    for m_val in m_match:
        for prefix in ("", "p", "c"):
            expansions.add(f"{prefix}M{m_val}")
        if m_val == "0":
            expansions.update(["no distant metastasis", "non-metastatic", "M0"])
        elif m_val == "1":
            expansions.update(["metastatic", "distant metastasis", "M1",
                               "stage IV", "metastatic disease"])

    # ── Stage group variants ────────────────────────────────────────────
    stage_match = re.findall(
        r'[Ss]tage\s*(I{1,3}V?|IV|[1-4])\s*([A-Ca-c]?\d?)',
        ql,
    )
    for roman_or_arabic, substage in stage_match:
        # Normalise to Roman
        roman = _STAGE_ROMAN_ARABIC.get(roman_or_arabic, roman_or_arabic).upper()
        arabic = _STAGE_ROMAN_ARABIC.get(roman, roman)
        full = f"{roman}{substage.upper()}" if substage else roman
        full_arabic = f"{arabic}{substage.upper()}" if substage else arabic

        expansions.add(f"Stage {full}")
        expansions.add(f"stage {full}")
        expansions.add(f"Stage {full_arabic}")
        expansions.add(f"stage {full_arabic}")
        expansions.add(full)

        # Add sub-stages
        for sub in _STAGE_SUBSTAGES.get(roman, []):
            expansions.add(f"Stage {sub}")
            expansions.add(sub)

    # ── AJCC staging table TNM ↔ stage-group mapping ────────────────────
    # If we found a T+N value, look up what stage group it maps to
    if t_match and n_match:
        t_norm = t_match[0].lower()
        n_norm = n_match[0].lower()
        m_norm = m_match[0].lower() if m_match else "0"
        for _cancer, _table in _AJCC_STAGING.items():
            if _cancer.startswith("_"):
                continue
            staging_table = _table.get("staging_table", {})
            for stage_group, tnm_combos in staging_table.items():
                for combo in tnm_combos:
                    if len(combo) >= 3:
                        ct, cn, cm = combo[0].lower(), combo[1].lower(), combo[2].lower()
                        if ct == t_norm and cn == n_norm and cm == m_norm:
                            expansions.add(f"Stage {stage_group}")
                            expansions.add(f"stage {stage_group}")
                            expansions.add(stage_group)

    return expansions


# ═══════════════════════════════════════════════════════════════════════════
# 2e. RANGE VALUE EXPANSION
# ═══════════════════════════════════════════════════════════════════════════
#
# Clinical literature encodes eligibility ranges in many notations:
#   T2-3, T2-T4, cT2-3N+, Stage II-III, ECOG 0-1, age ≥65, CPS ≥20,
#   age 18-75, PS ≤2, KPS ≥70
#
# When a patient's profile contains a specific value (e.g. T3), the
# embedding needs to include ALL range notations that encompass it, AND
# the value itself in all common spellings. This function detects range
# patterns in the query text and expands them into individual values.

# T-stage ordinal values for range expansion (0-4, with sub-stages)
_T_STAGES = ["0", "is", "1", "1a", "1b", "1c", "1mi", "2", "2a", "2b",
             "3", "3a", "3b", "4", "4a", "4b", "4c", "4d"]
_T_STAGE_ORDER = {s: i for i, s in enumerate(_T_STAGES)}

# N-stage ordinal values
_N_STAGES = ["0", "0i+", "1", "1a", "1b", "1c", "1mi", "2", "2a", "2b",
             "2c", "3", "3a", "3b", "3c"]
_N_STAGE_ORDER = {s: i for i, s in enumerate(_N_STAGES)}

# ECOG performance status values
_ECOG_VALUES = [0, 1, 2, 3, 4, 5]

# CPS thresholds commonly used in clinical trials
_CPS_THRESHOLDS = [1, 5, 10, 20, 50, 80, 100]


def _expand_integer_range(low: int, high: int, cap: int = 20) -> List[int]:
    """Expand an integer range [low, high] inclusive, capped at `cap` values."""
    if low > high:
        low, high = high, low
    return list(range(low, min(high + 1, low + cap)))


def _expand_tnm_range(prefix: str, val: str, stage_order: Dict[str, int]) -> Set[str]:
    """
    Expand a TNM range notation into all individual values.

    Examples:
        _expand_tnm_range("T", "2-3", _T_STAGE_ORDER) → {"T2", "T2a", "T2b", "T3", "T3a", "T3b"}
        _expand_tnm_range("N", "0-1", _N_STAGE_ORDER) → {"N0", "N1", "N1a", "N1b", "N1c", "N1mi"}
    """
    result: Set[str] = set()

    # Handle range: "2-3", "2-T3", "1-4"
    range_match = re.match(r'(\d[a-d]?(?:is|mi)?)\s*[-–]\s*(?:' + prefix + r')?(\d[a-d]?(?:is|mi)?)', val, re.IGNORECASE)
    if range_match:
        low_str = range_match.group(1).lower()
        high_str = range_match.group(2).lower()
        low_ord = stage_order.get(low_str)
        high_ord = stage_order.get(high_str)
        if low_ord is not None and high_ord is not None:
            if low_ord > high_ord:
                low_ord, high_ord = high_ord, low_ord
            stages = [s for s, o in stage_order.items() if low_ord <= o <= high_ord]
            for s in stages:
                result.add(f"{prefix}{s}")
                result.add(f"{prefix}{s.upper()}" if s.isalpha() else f"{prefix}{s}")
                # Add prefix variants
                for pfx in ("", "p", "c", "yp"):
                    result.add(f"{pfx}{prefix}{s}")
            return result

    # Handle ≥/>/≤/< operators: "≥2", ">1", "≤3"
    op_match = re.match(r'([≥>≤<]=?)\s*(\d[a-d]?(?:is|mi)?)', val)
    if op_match:
        op = op_match.group(1)
        threshold_str = op_match.group(2).lower()
        threshold_ord = stage_order.get(threshold_str)
        if threshold_ord is not None:
            for s, o in stage_order.items():
                include = False
                if op in ("≥", ">="):
                    include = o >= threshold_ord
                elif op == ">":
                    include = o > threshold_ord
                elif op in ("≤", "<="):
                    include = o <= threshold_ord
                elif op == "<":
                    include = o < threshold_ord
                if include:
                    result.add(f"{prefix}{s}")
                    for pfx in ("", "p", "c", "yp"):
                        result.add(f"{pfx}{prefix}{s}")
            return result

    # Handle "+" meaning "and above": "N+"  = N1, N2, N3
    if val.endswith("+") and len(val) >= 2:
        base = val[:-1].lower()
        base_ord = stage_order.get(base)
        if base_ord is not None:
            for s, o in stage_order.items():
                if o >= base_ord:
                    result.add(f"{prefix}{s}")
            # Also add "node positive", "N+ disease" for N stages
            if prefix == "N":
                result.update(["node positive", "N+ disease", "N+",
                               "nodal involvement", "lymph node metastasis"])
            return result

    # Single value: just return it with all prefix variants
    val_lower = val.lower().rstrip("+")
    if val_lower in stage_order:
        for pfx in ("", "p", "c", "yp", "y"):
            result.add(f"{pfx}{prefix}{val_lower}")
        result.add(f"{prefix}{val_lower} disease")

    return result


def _expand_ecog_range(val: str) -> Set[str]:
    """Expand ECOG PS range. E.g. "0-1" → {"ECOG 0", "ECOG 1", "PS 0", "PS 1"}."""
    result: Set[str] = set()

    range_match = re.match(r'(\d)\s*[-–]\s*(\d)', val)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        for v in _expand_integer_range(low, high, cap=6):
            result.update([f"ECOG {v}", f"PS {v}", f"ECOG PS {v}",
                           f"performance status {v}"])
        return result

    op_match = re.match(r'([≥>≤<]=?)\s*(\d)', val)
    if op_match:
        op, threshold = op_match.group(1), int(op_match.group(2))
        for v in _ECOG_VALUES:
            include = False
            if op in ("≥", ">="): include = v >= threshold
            elif op == ">": include = v > threshold
            elif op in ("≤", "<="): include = v <= threshold
            elif op == "<": include = v < threshold
            if include:
                result.update([f"ECOG {v}", f"PS {v}", f"performance status {v}"])
        return result

    # Single value
    try:
        v = int(val.strip())
        result.update([f"ECOG {v}", f"PS {v}", f"performance status {v}"])
    except ValueError:
        pass
    return result


def _expand_age_range(val: str) -> Set[str]:
    """Expand age ranges. E.g. "≥65" → {"65", "70", "75", "80", "elderly", "older"}."""
    result: Set[str] = set()

    range_match = re.match(r'(\d+)\s*[-–]\s*(\d+)', val)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        # Add decade markers within range
        for decade in range(low // 10 * 10, high + 10, 10):
            if low <= decade <= high:
                result.add(str(decade))
        result.add(f"age {low}-{high}")
        result.add(f"aged {low}-{high}")
        if low <= 40:
            result.add("young adult")
        if high >= 65 or low >= 65:
            result.update(["elderly", "older adult", "geriatric"])
        return result

    op_match = re.match(r'([≥>≤<]=?)\s*(\d+)', val)
    if op_match:
        op, threshold = op_match.group(1), int(op_match.group(2))
        if op in ("≥", ">="):
            result.update([str(threshold), f"≥{threshold}", f">={threshold}",
                           f"age {threshold} or older", f"over {threshold}"])
            if threshold >= 65:
                result.update(["elderly", "older adult", "geriatric"])
            if threshold <= 18:
                result.update(["adult", "adults"])
        elif op in ("≤", "<="):
            result.update([str(threshold), f"≤{threshold}", f"<={threshold}",
                           f"age {threshold} or younger", f"under {threshold}"])
        return result

    # Single age
    try:
        age = int(val.strip())
        result.add(str(age))
        decade = (age // 10) * 10
        result.add(str(decade))
        if age >= 65:
            result.update(["elderly", "older adult"])
        elif age < 40:
            result.update(["young adult"])
    except ValueError:
        pass
    return result


def _expand_cps_value(val: str) -> Set[str]:
    """Expand CPS/PD-L1 threshold. E.g. "100" → {"CPS ≥ 1", "CPS ≥ 20", ..., "CPS 100", "PD-L1 high"}."""
    result: Set[str] = set()

    # Extract numeric value
    num_match = re.search(r'(\d+)', val)
    if not num_match:
        return result
    cps_val = int(num_match.group(1))

    # A patient with CPS X qualifies for all thresholds ≤ X
    for threshold in _CPS_THRESHOLDS:
        if cps_val >= threshold:
            result.add(f"CPS ≥ {threshold}")
            result.add(f"CPS >= {threshold}")
            result.add(f"CPS ≥{threshold}")
    result.add(f"CPS {cps_val}")
    result.add(f"CPS score {cps_val}")

    if cps_val >= 50:
        result.update(["PD-L1 high expression", "PD-L1 high", "high CPS"])
    if cps_val >= 1:
        result.update(["PD-L1 positive", "CPS positive", "PD-L1 expression"])

    return result


def _expand_stage_group_range(val: str) -> Set[str]:
    """Expand stage group ranges. E.g. "II-III" → {"Stage II", "Stage IIA", ..., "Stage IIIC"}."""
    result: Set[str] = set()
    _STAGE_ORDER = {"I": 1, "II": 2, "III": 3, "IV": 4,
                    "1": 1, "2": 2, "3": 3, "4": 4}

    # Range: "II-III", "Stage 2-3", "II-IV"
    range_match = re.match(
        r'(?:Stage\s*)?(I{1,3}V?|IV|[1-4])\s*[-–]\s*(?:Stage\s*)?(I{1,3}V?|IV|[1-4])',
        val, re.IGNORECASE
    )
    if range_match:
        low_str = range_match.group(1).upper()
        high_str = range_match.group(2).upper()
        low_ord = _STAGE_ORDER.get(low_str, _STAGE_ORDER.get(
            _STAGE_ROMAN_ARABIC.get(low_str, ""), 0))
        high_ord = _STAGE_ORDER.get(high_str, _STAGE_ORDER.get(
            _STAGE_ROMAN_ARABIC.get(high_str, ""), 0))
        if low_ord > high_ord:
            low_ord, high_ord = high_ord, low_ord

        for roman, ordinal in [("I", 1), ("II", 2), ("III", 3), ("IV", 4)]:
            if low_ord <= ordinal <= high_ord:
                arabic = _STAGE_ROMAN_ARABIC.get(roman, roman)
                result.add(f"Stage {roman}")
                result.add(f"stage {roman}")
                result.add(f"Stage {arabic}")
                result.add(roman)
                # Sub-stages
                for sub in _STAGE_SUBSTAGES.get(roman, []):
                    result.add(f"Stage {sub}")
                    result.add(sub)
        return result

    return result


def _build_range_expansions(query_text: str) -> Set[str]:
    """
    Detect range patterns in the query and expand them into all
    inclusive individual values. Handles:

    - TNM ranges: T2-3, T2-T4, cT2-3, N0-1, N+, ≥T2
    - Stage group ranges: Stage II-III, Stage 2-4
    - ECOG/PS ranges: ECOG 0-1, PS ≤2, ECOG ≥0
    - Age ranges: age 18-75, ≥65, age ≥ 18
    - CPS/PD-L1 values: CPS 100, CPS ≥20, PD-L1 ≥50
    """
    expansions: Set[str] = set()
    ql = query_text

    # ── TNM T-stage ranges ────────────────────────────────────────────
    for m in re.finditer(
        r'(?:c|p|yp)?T(\d[a-d]?(?:is|mi)?\s*[-–]\s*(?:T)?\d[a-d]?(?:is|mi)?'
        r'|[≥>≤<]=?\s*\d[a-d]?(?:is|mi)?'
        r'|\d[a-d]?(?:is|mi)?\+)',
        ql, re.IGNORECASE
    ):
        expansions.update(_expand_tnm_range("T", m.group(1) if m.group(1) else m.group(0).lstrip("T"), _T_STAGE_ORDER))

    # More targeted T-range extraction
    for m in re.finditer(r'(?:c|p|yp)?T(\d[a-d]?)\s*[-–]\s*(?:T)?(\d[a-d]?)', ql, re.IGNORECASE):
        expansions.update(_expand_tnm_range("T", f"{m.group(1)}-{m.group(2)}", _T_STAGE_ORDER))
    for m in re.finditer(r'(?:c|p|yp)?T([≥>≤<]=?\s*\d[a-d]?)', ql, re.IGNORECASE):
        expansions.update(_expand_tnm_range("T", m.group(1), _T_STAGE_ORDER))

    # ── Single T-stage values (pT4, cT3, T2a, etc.) ──────────────────
    for m in re.finditer(r'(?:c|p|yp|y)?T(\d[a-d]?(?:is|mi)?)\b', ql, re.IGNORECASE):
        val = m.group(1).lower()
        if val in _T_STAGE_ORDER:
            for pfx in ("", "p", "c", "yp", "y"):
                expansions.add(f"{pfx}T{val}")
            expansions.add(f"T{val} disease")

    # ── TNM N-stage ranges ────────────────────────────────────────────
    for m in re.finditer(r'(?:c|p|yp)?N(\d[a-c]?)\s*[-–]\s*(?:N)?(\d[a-c]?)', ql, re.IGNORECASE):
        expansions.update(_expand_tnm_range("N", f"{m.group(1)}-{m.group(2)}", _N_STAGE_ORDER))
    for m in re.finditer(r'(?:c|p|yp)?N([≥>≤<]=?\s*\d[a-c]?)', ql, re.IGNORECASE):
        expansions.update(_expand_tnm_range("N", m.group(1), _N_STAGE_ORDER))
    # N+ shorthand
    if re.search(r'\bN\+', ql):
        expansions.update(_expand_tnm_range("N", "1+", _N_STAGE_ORDER))

    # ── Single N-stage values (pN0, cN1, N2a, etc.) ──────────────────
    for m in re.finditer(r'(?:c|p|yp|y)?N(\d[a-c]?(?:mi)?)\b', ql, re.IGNORECASE):
        val = m.group(1).lower()
        if val in _N_STAGE_ORDER:
            for pfx in ("", "p", "c", "yp", "y"):
                expansions.add(f"{pfx}N{val}")
            if val.startswith("0"):
                expansions.update(["node negative", "N0 disease"])
            else:
                expansions.update(["node positive", "N+ disease"])

    # ── Stage group ranges ────────────────────────────────────────────
    for m in re.finditer(
        r'[Ss]tage\s*(I{1,3}V?|IV|[1-4])\s*[-–]\s*(?:[Ss]tage\s*)?(I{1,3}V?|IV|[1-4])',
        ql
    ):
        expansions.update(_expand_stage_group_range(f"{m.group(1)}-{m.group(2)}"))

    # ── ECOG / PS ranges ─────────────────────────────────────────────
    for m in re.finditer(r'(?:ECOG|PS|performance\s+status)\s+(\d(?:\s*[-–]\s*\d)?)', ql, re.IGNORECASE):
        expansions.update(_expand_ecog_range(m.group(1).strip()))
    for m in re.finditer(r'(?:ECOG|PS)\s*([≥>≤<]=?\s*\d)', ql, re.IGNORECASE):
        expansions.update(_expand_ecog_range(m.group(1).strip()))

    # ── Age ranges ────────────────────────────────────────────────────
    for m in re.finditer(r'(?:age[ds]?\s*|aged?\s+)([≥>≤<]=?\s*\d+(?:\s*[-–]\s*\d+)?)', ql, re.IGNORECASE):
        expansions.update(_expand_age_range(m.group(1).strip()))
    # "≥65 years" without "age" prefix
    for m in re.finditer(r'([≥>≤<]=?\d+)\s*(?:years?|yo|y\.o\.)', ql, re.IGNORECASE):
        expansions.update(_expand_age_range(m.group(1).strip()))
    # Bare "X year old" / "X y.o."
    for m in re.finditer(r'(\d{2})\s*(?:year[- ]old|y\.?o\.?)', ql, re.IGNORECASE):
        expansions.update(_expand_age_range(m.group(1).strip()))

    # ── CPS / PD-L1 values ───────────────────────────────────────────
    for m in re.finditer(r'CPS\s*(?:score\s*)?(?:of\s*)?([≥>≤<]=?\s*\d+|\d+)', ql, re.IGNORECASE):
        expansions.update(_expand_cps_value(m.group(1).strip()))
    for m in re.finditer(r'PD-?L1\s*(?:TPS\s*)?([≥>≤<]=?\s*\d+%?)', ql, re.IGNORECASE):
        expansions.update(_expand_cps_value(m.group(1).replace("%", "").strip()))

    return expansions


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def expand_query_comprehensive(query: str) -> str:
    """
    Expand a query string with ALL available synonyms, abbreviations,
    ontology terms, brand names, staging variants, and clinical-context
    expansions.  The original query text is preserved verbatim and the
    expansion terms are appended after it.

    This function is intended to be called BEFORE embedding so the
    resulting vector sits in the right neighbourhood regardless of
    vocabulary mismatch between the query and the documents.

    It complements (and should be called AFTER) the existing
    `enhanced_rag_service.expand_query()` which handles the core
    ONCOLOGY_EXPANSIONS / REVERSE_EXPANSIONS / STAGING_SYNONYMS /
    CLINICAL_SYNONYMS dicts.  This function adds:

      - Cancer-type ontology synonyms (data/ontology/cancer_type_ontology.json)
      - Drug brand-name ↔ generic ↔ mechanism class
      - Clinical context / disease status synonyms
      - Comprehensive staging notation expansion (Roman ↔ Arabic ↔
        clinical/pathologic prefix variants ↔ AJCC table lookup)

    Returns the original query + appended expansion terms.
    """
    if not query or not query.strip():
        return query

    ql = query.lower()
    added: Set[str] = set()
    original_words = set(ql.split())

    # ── Cancer-type ontology synonyms ────────────────────────────────────
    for trigger, terms in _CANCER_SYNONYM_MAP.items():
        if len(trigger) <= 3:
            # Short triggers require word-boundary match
            if re.search(rf'\b{re.escape(trigger)}\b', ql):
                added.update(t for t in terms if t.lower() not in ql)
        else:
            if trigger in ql:
                added.update(t for t in terms if t.lower() not in ql)

    # ── Drug brand-name / generic / mechanism expansion ──────────────────
    for trigger, group in _DRUG_SYNONYM_LOOKUP.items():
        if len(trigger) <= 3:
            if re.search(rf'\b{re.escape(trigger)}\b', ql):
                added.update(d for d in group if d.lower() not in ql)
        else:
            if trigger in ql:
                added.update(d for d in group if d.lower() not in ql)

    # ── Clinical context / disease status expansion ──────────────────────
    for trigger, group in _CLINICAL_CONTEXT_LOOKUP.items():
        if trigger in ql:
            added.update(t for t in group if t.lower() not in ql)

    # ── Staging notation expansion ───────────────────────────────────────
    staging_terms = _build_staging_expansions(query)
    added.update(t for t in staging_terms if t.lower() not in ql)

    # ── Range value expansion (TNM, stage, ECOG, age, CPS) ───────────────
    # Expands range notations (T2-3, ECOG 0-1, ≥65, CPS 100) into all
    # individual values inclusive. This ensures the embedding vector covers
    # every value a study might have used for the same range.
    range_terms = _build_range_expansions(query)
    added.update(t for t in range_terms if t.lower() not in ql)

    # ── Deduplicate against original query words ─────────────────────────
    deduped = []
    seen: Set[str] = set()
    for term in sorted(added):
        term_lower = term.lower().strip()
        if not term_lower or term_lower in seen:
            continue
        # Skip single words already in the original query
        if term_lower in original_words:
            continue
        seen.add(term_lower)
        deduped.append(term)

    if deduped:
        # Cap expansion to avoid exceeding embedding model token limits
        # text-embedding-3-large supports 8191 tokens; we reserve ~2000
        # for the original query and allow ~4000 tokens of expansion.
        expansion_text = " ".join(deduped)
        if len(expansion_text) > 12000:
            expansion_text = expansion_text[:12000]
        return f"{query} {expansion_text}"

    return query
