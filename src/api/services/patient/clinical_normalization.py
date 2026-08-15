"""
Clinical Normalization (Phase 1 finalization / Phase 5 groundwork)

Pure entity normalization — mapping free text a patient or an extractor
wrote to a canonical term. Deliberately separate from clinical_inference.py
(which maps implicit clinical *status* to explicit labels for retrieval,
e.g. "no longer a surgical candidate" -> "unresectable") and from
patient_safety_service.py (triage). This module answers "what is the
standard name for what was written", not "what does it imply" or "is this
urgent" — see the Phase 5 checklist item this anticipates
("Separate normalization from clinical inference").

Honest scope, read before extending this file:

* This is a curated static lookup table covering common oncology terms —
  it is NOT a licensed medical coding database. Real ICD-O-3 (site/
  histology), SNOMED CT, and RxNorm require UMLS/NLM licenses and (for
  RxNorm) a live API to be authoritative and complete. Nothing here
  should be presented as a certified code; `*_code` fields stay None
  unless this file's static tables happen to have one, and are written
  that way deliberately (see normalize_drug_name).
* Matching is substring/keyword-based on lowercased text, same technique
  clinical_inference.py already uses successfully for this domain. It
  will miss phrasings not in the table and should fail to a null/raw
  result rather than guess — a wrong normalization is worse than none.
* Extending the tables (more sites, more genes, more drugs/regimens) is
  expected and safe; each is an independent dict entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedTerm:
    raw: str
    canonical: Optional[str]
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"raw": self.raw, "canonical": self.canonical, "code": self.code}


# ── Cancer site ──────────────────────────────────────────────────────────
# canonical name -> patterns that should resolve to it. Deliberately not
# reversed (pattern -> canonical) because several patterns often map to
# one canonical site (subsites of the same organ).
CANCER_SITES: Dict[str, List[str]] = {
    "oral tongue": ["oral tongue", "tongue, oral", "mobile tongue", "anterior tongue"],
    "oropharynx": ["oropharynx", "oropharyngeal", "tonsil", "base of tongue", "soft palate"],
    "larynx": ["larynx", "laryngeal", "glottis", "glottic", "supraglottic", "subglottic"],
    "hypopharynx": ["hypopharynx", "hypopharyngeal", "pyriform sinus"],
    "nasopharynx": ["nasopharynx", "nasopharyngeal"],
    "breast": ["breast", "mammary"],
    "lung": ["lung", "pulmonary", "bronchus", "bronchial"],
    "colon": ["colon", "colonic", "cecum", "ascending colon", "transverse colon",
              "descending colon", "sigmoid"],
    "rectum": ["rectum", "rectal"],
    "pancreas": ["pancreas", "pancreatic"],
    "stomach": ["stomach", "gastric"],
    "esophagus": ["esophagus", "esophageal", "oesophag"],
    "liver": ["liver", "hepatic", "hepatocellular"],
    "kidney": ["kidney", "renal cell", "renal"],
    "bladder": ["bladder", "urothelial"],
    "prostate": ["prostate", "prostatic"],
    "ovary": ["ovary", "ovarian"],
    "cervix": ["cervix", "cervical"],
    "uterus": ["uterus", "uterine", "endometrial", "endometrium"],
    "thyroid": ["thyroid"],
    "skin (melanoma)": ["melanoma", "cutaneous melanoma"],
    "brain": ["brain", "glioblastoma", "glioma", "astrocytoma"],
    "lymph node (lymphoma)": ["lymphoma", "hodgkin", "non-hodgkin"],
    "bone marrow (leukemia)": ["leukemia", "leukaemia", "aml", "all", "cll", "cml"],
}

# ── Histology ────────────────────────────────────────────────────────────
HISTOLOGIES: Dict[str, List[str]] = {
    "squamous cell carcinoma": ["squamous cell carcinoma", "scc", "squamous carcinoma"],
    "adenocarcinoma": ["adenocarcinoma"],
    "small cell carcinoma": ["small cell carcinoma", "small cell lung"],
    "non-small cell carcinoma": ["non-small cell", "nsclc"],
    "ductal carcinoma": ["ductal carcinoma", "idc", "dcis"],
    "lobular carcinoma": ["lobular carcinoma", "ilc"],
    "clear cell carcinoma": ["clear cell carcinoma", "clear cell rcc"],
    "transitional cell carcinoma": ["transitional cell", "urothelial carcinoma"],
    "hepatocellular carcinoma": ["hepatocellular carcinoma", "hcc"],
    "melanoma": ["melanoma"],
    "glioblastoma": ["glioblastoma", "gbm"],
    "diffuse large B-cell lymphoma": ["dlbcl", "diffuse large b-cell"],
    "acute myeloid leukemia": ["aml", "acute myeloid leukemia"],
}

# ── Metastatic sites ─────────────────────────────────────────────────────
METASTATIC_SITE_PATTERNS: List[tuple] = [
    (r"\bright ventricl|\bcardiac\b|\bheart\b", "cardiac"),
    (r"\blung metastas|\bpulmonary met", "lung"),
    (r"\bliver metastas|\bhepatic met", "liver"),
    (r"\bbone metastas|\bosseous met|\bskeletal met", "bone"),
    (r"\bbrain metastas|\bcerebral met|\bintracranial met", "brain"),
    (r"\badrenal metastas", "adrenal"),
    (r"\bperitoneal (?:carcinomatosis|metastas)", "peritoneum"),
    (r"\blymph node metastas|\bnodal metastas", "lymph nodes"),
]

# ── Genes / biomarkers ───────────────────────────────────────────────────
# canonical symbol -> patterns. category is used to populate
# patient_biomarker_results.biomarker_category.
GENE_BIOMARKERS: List[Dict[str, Any]] = [
    {"canonical": "ER", "category": "receptor_status", "patterns": [r"\ber[\s-]?(?:status|receptor)\b", r"\bestrogen receptor\b"]},
    {"canonical": "PR", "category": "receptor_status", "patterns": [r"\bpr[\s-]?(?:status|receptor)\b", r"\bprogesterone receptor\b"]},
    {"canonical": "HER2", "category": "receptor_status", "patterns": [r"\bher2\b", r"\bher[\s-]?2/neu\b", r"\berbb2\b"]},
    {"canonical": "PD-L1", "category": "pdl1", "patterns": [r"\bpd[\s-]?l1\b", r"\bcps\b", r"\btps\b"]},
    {"canonical": "MSI", "category": "msi_mmr", "patterns": [r"\bmsi\b", r"\bmicrosatellite instab"]},
    {"canonical": "MMR", "category": "msi_mmr", "patterns": [r"\bmmr\b", r"\bmismatch repair\b", r"\bdmmr\b", r"\bpmmr\b"]},
    {"canonical": "TMB", "category": "tmb", "patterns": [r"\btmb\b", r"\btumor mutational burden\b"]},
    {"canonical": "EGFR", "category": "variant", "patterns": [r"\begfr\b"]},
    {"canonical": "ALK", "category": "variant", "patterns": [r"\balk\b(?!aline)"]},
    {"canonical": "ROS1", "category": "variant", "patterns": [r"\bros[\s-]?1\b"]},
    {"canonical": "KRAS", "category": "variant", "patterns": [r"\bkras\b"]},
    {"canonical": "BRAF", "category": "variant", "patterns": [r"\bbraf\b"]},
    {"canonical": "BRCA1", "category": "variant", "patterns": [r"\bbrca[\s-]?1\b"]},
    {"canonical": "BRCA2", "category": "variant", "patterns": [r"\bbrca[\s-]?2\b"]},
    {"canonical": "NTRK", "category": "variant", "patterns": [r"\bntrk\b"]},
    {"canonical": "PIK3CA", "category": "variant", "patterns": [r"\bpik3ca\b"]},
    {"canonical": "TP53", "category": "variant", "patterns": [r"\btp53\b", r"\bp53\b"]},
    {"canonical": "KIT", "category": "variant", "patterns": [r"\bc-?kit\b", r"\bcd117\b"]},
    {"canonical": "IDH1", "category": "variant", "patterns": [r"\bidh1\b"]},
]

# ── Symptoms ─────────────────────────────────────────────────────────────
# Common oncology/treatment-related symptoms. Same honest-scope caveat as
# everything else in this file: a starting curated vocabulary, not a
# terminology standard (MedDRA/CTCAE would be the licensed equivalent).
SYMPTOM_PATTERNS: List[tuple] = [
    (r"\bnause", "nausea"),
    (r"\bvomit|\bthrowing up", "vomiting"),
    (r"\bdiarrhea|\bloose stool", "diarrhea"),
    (r"\bconstipat", "constipation"),
    (r"\bfatigu|\bexhaust|\btired all the time", "fatigue"),
    (r"\bneuropath|\btingling|\bnumbness in (?:my |the )?(?:hands|feet|fingers|toes)", "peripheral neuropathy"),
    (r"\bmetallic taste|\btaste chang|\bdysgeusia|\beverything tastes", "dysgeusia"),
    (r"\bmucositis|\bmouth sores|\bmouth ulcers", "mucositis"),
    (r"\brash\b|\bskin irritation", "rash"),
    (r"\bshort(?:ness)? of breath|\bdyspnea|\btrouble breathing", "dyspnea"),
    (r"\bpain\b", "pain"),
    (r"\bfever\b", "fever"),
    (r"\bloss of appetite|\banorexia|\bdon'?t want to eat|\bnot hungry", "anorexia"),
    (r"\bhair loss|\balopecia", "alopecia"),
    (r"\binsomnia|\bcan'?t sleep|\btrouble sleeping", "insomnia"),
    (r"\banxious|\banxiety", "anxiety"),
    (r"\bdepress", "depression"),
    (r"\bdry mouth|\bxerostomia", "xerostomia"),
    (r"\bconstant itching|\bpruritus|\bitchy skin", "pruritus"),
    (r"\bswelling|\bedema", "edema"),
]


def normalize_symptom(text: str) -> NormalizedTerm:
    text = (text or "").strip()
    for pattern, canonical in SYMPTOM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return NormalizedTerm(raw=text, canonical=canonical)
    return NormalizedTerm(raw=text, canonical=None)


# ── Drug names: brand -> generic, plus reverse aliasing ─────────────────
# rxnorm_code stays None throughout — see module docstring.
DRUG_ALIASES: Dict[str, Dict[str, Any]] = {
    "pembrolizumab": {"brands": ["keytruda"], "rxnorm_code": None},
    "nivolumab": {"brands": ["opdivo"], "rxnorm_code": None},
    "atezolizumab": {"brands": ["tecentriq"], "rxnorm_code": None},
    "durvalumab": {"brands": ["imfinzi"], "rxnorm_code": None},
    "ipilimumab": {"brands": ["yervoy"], "rxnorm_code": None},
    "cetuximab": {"brands": ["erbitux"], "rxnorm_code": None},
    "trastuzumab": {"brands": ["herceptin"], "rxnorm_code": None},
    "bevacizumab": {"brands": ["avastin"], "rxnorm_code": None},
    "oxaliplatin": {"brands": ["eloxatin"], "rxnorm_code": None},
    "fluorouracil": {"brands": ["5-fu", "adrucil"], "rxnorm_code": None},
    "leucovorin": {"brands": ["folinic acid", "wellcovorin"], "rxnorm_code": None},
    "irinotecan": {"brands": ["camptosar"], "rxnorm_code": None},
    "cisplatin": {"brands": ["platinol"], "rxnorm_code": None},
    "carboplatin": {"brands": ["paraplatin"], "rxnorm_code": None},
    "paclitaxel": {"brands": ["taxol", "abraxane"], "rxnorm_code": None},
    "docetaxel": {"brands": ["taxotere"], "rxnorm_code": None},
    "gemcitabine": {"brands": ["gemzar"], "rxnorm_code": None},
    "doxorubicin": {"brands": ["adriamycin"], "rxnorm_code": None},
    "cyclophosphamide": {"brands": ["cytoxan"], "rxnorm_code": None},
    "vincristine": {"brands": ["oncovin"], "rxnorm_code": None},
    "rituximab": {"brands": ["rituxan"], "rxnorm_code": None},
    "prednisone": {"brands": ["deltasone", "rayos"], "rxnorm_code": None},
    "dexamethasone": {"brands": ["decadron"], "rxnorm_code": None},
    "ondansetron": {"brands": ["zofran"], "rxnorm_code": None},
    "metformin": {"brands": ["glucophage"], "rxnorm_code": None},
    "warfarin": {"brands": ["coumadin"], "rxnorm_code": None},
    "apixaban": {"brands": ["eliquis"], "rxnorm_code": None},
    "osimertinib": {"brands": ["tagrisso"], "rxnorm_code": None},
    "erlotinib": {"brands": ["tarceva"], "rxnorm_code": None},
    "sunitinib": {"brands": ["sutent"], "rxnorm_code": None},
    "imatinib": {"brands": ["gleevec"], "rxnorm_code": None},
}
_BRAND_TO_GENERIC: Dict[str, str] = {
    brand: generic for generic, info in DRUG_ALIASES.items() for brand in info["brands"]
}

# regimen name -> component generic agent names
REGIMEN_EXPANSIONS: Dict[str, List[str]] = {
    "folfox": ["fluorouracil", "leucovorin", "oxaliplatin"],
    "folfiri": ["fluorouracil", "leucovorin", "irinotecan"],
    "folfirinox": ["fluorouracil", "leucovorin", "irinotecan", "oxaliplatin"],
    "xelox": ["capecitabine", "oxaliplatin"],
    "capox": ["capecitabine", "oxaliplatin"],
    "chop": ["cyclophosphamide", "doxorubicin", "vincristine", "prednisone"],
    "r-chop": ["rituximab", "cyclophosphamide", "doxorubicin", "vincristine", "prednisone"],
    "abvd": ["doxorubicin", "bleomycin", "vinblastine", "dacarbazine"],
    "ac": ["doxorubicin", "cyclophosphamide"],
    "tc": ["docetaxel", "cyclophosphamide"],
    "tch": ["docetaxel", "carboplatin", "trastuzumab"],
    "tchp": ["docetaxel", "carboplatin", "trastuzumab", "pertuzumab"],
    "ec": ["epirubicin", "cyclophosphamide"],
    "gemcarbo": ["gemcitabine", "carboplatin"],
    "gemcis": ["gemcitabine", "cisplatin"],
}


def _match(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def normalize_cancer_site(text: str) -> NormalizedTerm:
    text = (text or "").strip()
    for canonical, patterns in CANCER_SITES.items():
        if _match(text, patterns):
            return NormalizedTerm(raw=text, canonical=canonical)
    return NormalizedTerm(raw=text, canonical=None)


def normalize_histology(text: str) -> NormalizedTerm:
    text = (text or "").strip()
    for canonical, patterns in HISTOLOGIES.items():
        if _match(text, patterns):
            return NormalizedTerm(raw=text, canonical=canonical)
    return NormalizedTerm(raw=text, canonical=None)


def normalize_metastatic_sites(text: str) -> List[str]:
    """Scans free text for every metastatic site mentioned (a patient can
    have several), returns the canonical list, deduplicated, in the order
    first mentioned."""
    text = text or ""
    found: List[str] = []
    for pattern, canonical in METASTATIC_SITE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and canonical not in found:
            found.append(canonical)
    return found


# AJCC applies to almost all solid tumors; a handful of cancers use their
# own system instead. Keyed by normalized cancer site.
_STAGE_SYSTEM_OVERRIDES: Dict[str, str] = {
    "cervix": "FIGO", "uterus": "FIGO", "ovary": "FIGO",
    "lymph node (lymphoma)": "Ann Arbor / Lugano",
    "bone marrow (leukemia)": "not applicable (staged by disease-specific criteria)",
}


def normalize_stage_system(cancer_site_canonical: Optional[str]) -> Optional[str]:
    """Best-effort guess at which staging system applies, from the
    already-normalized cancer site. Not a substitute for what the
    pathology/staging report itself states — raw_text on patient_diagnoses
    always wins when the two would disagree."""
    if not cancer_site_canonical:
        return None
    return _STAGE_SYSTEM_OVERRIDES.get(cancer_site_canonical, "AJCC")


def normalize_gene(text: str) -> NormalizedTerm:
    text = (text or "").strip()
    for entry in GENE_BIOMARKERS:
        if _match(text, entry["patterns"]):
            return NormalizedTerm(raw=text, canonical=entry["canonical"])
    return NormalizedTerm(raw=text, canonical=None)


def biomarker_category_for(biomarker_name: str) -> str:
    """'receptor_status' | 'pdl1' | 'msi_mmr' | 'tmb' | 'variant' | 'other'"""
    text = (biomarker_name or "").strip()
    for entry in GENE_BIOMARKERS:
        if _match(text, entry["patterns"]):
            return entry["category"]
    return "other"


def normalize_drug_name(text: str) -> Dict[str, Any]:
    """Brand -> generic (and back), with rxnorm_code left None unless this
    file's static table happens to carry one (see module docstring — no
    live RxNorm API integration exists in this codebase)."""
    raw = (text or "").strip()
    key = raw.lower()
    if key in DRUG_ALIASES:
        info = DRUG_ALIASES[key]
        return {
            "raw": raw, "canonical": key, "rxnorm_code": info["rxnorm_code"],
            "aliases": list(info["brands"]),
        }
    if key in _BRAND_TO_GENERIC:
        generic = _BRAND_TO_GENERIC[key]
        info = DRUG_ALIASES[generic]
        return {
            "raw": raw, "canonical": generic, "rxnorm_code": info["rxnorm_code"],
            "aliases": [b for b in info["brands"] if b != key] + [key],
        }
    return {"raw": raw, "canonical": None, "rxnorm_code": None, "aliases": []}


def expand_regimen(name: str) -> List[str]:
    """Component generic agent names for a known regimen, [] if the name
    isn't in REGIMEN_EXPANSIONS. Case/whitespace/hyphen-insensitive
    ('FOLFOX', 'folfox', 'FOLF-OX' style variants are not all covered —
    only the literal keys above and their case-folded form)."""
    key = re.sub(r"[\s\-]", "", (name or "").strip().lower())
    return list(REGIMEN_EXPANSIONS.get(key, []))
