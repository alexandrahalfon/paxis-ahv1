"""
Fast Query Structuring Service

Extracts structured information from user queries using regex patterns only.
No LLM calls - designed for zero latency overhead.

This service identifies:
- Patient context (age, gender, stage, TNM, etc.)
- Cancer site and histology
- Treatment modality being asked about
- Question focus (dose, survival, indication, etc.)
- Boost terms for retrieval optimization
- Raw text spans for semantic search fallback

For complex queries (>200 chars, multiple commas, pathology terms), an optional
LLM extraction can run IN PARALLEL with embedding generation to extract
additional structured text spans for semantic search.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class PatientContext:
    """Extracted patient information from query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    performance_status: Optional[str] = None
    ethnicity: Optional[str] = None
    smoking_status: Optional[str] = None
    comorbidities: List[str] = field(default_factory=list)
    # Raw text span for semantic search
    raw_text: Optional[str] = None
    
    def has_data(self) -> bool:
        return any([self.age, self.gender, self.performance_status, 
                    self.ethnicity, self.smoking_status, self.comorbidities])
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != []:
                result[k] = v
        return result


@dataclass
class CancerContext:
    """Extracted cancer-specific information"""
    site: Optional[str] = None  # breast, lung, prostate, etc.
    site_detail: Optional[str] = None  # oral cavity, oropharynx, etc.
    histology: Optional[str] = None  # SCC, adenocarcinoma, etc.
    stage: Optional[str] = None  # I, II, III, IV
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    grade: Optional[str] = None
    receptor_status: Optional[str] = None  # ER+, HER2+, etc.
    biomarkers: List[str] = field(default_factory=list)  # EGFR+, KRAS G12C, ALK+, PD-L1 high, etc.
    # Sites explicitly described as metastatic ("liver metastases",
    # "brain mets"). Recorded separately from `site` (primary) so
    # downstream scoring can reason about disease extent without
    # treating mets as primaries.
    metastatic_sites_detected: List[str] = field(default_factory=list)
    # Pathology details
    doi: Optional[str] = None  # Depth of invasion
    lvi: Optional[str] = None  # Lymphovascular invasion
    pni: Optional[str] = None  # Perineural invasion
    margins: Optional[str] = None
    lymph_nodes: Optional[str] = None
    # Raw text span for semantic search
    raw_text: Optional[str] = None
    
    def has_data(self) -> bool:
        return any([self.site, self.histology, self.stage, 
                    self.tnm_t, self.tnm_n, self.receptor_status,
                    self.biomarkers,
                    self.doi, self.lvi, self.pni, self.margins])
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result
    
    def get_tnm_string(self) -> Optional[str]:
        """Get formatted TNM string if available"""
        parts = []
        if self.tnm_t:
            parts.append(f"T{self.tnm_t}")
        if self.tnm_n:
            parts.append(f"N{self.tnm_n}")
        if self.tnm_m:
            parts.append(f"M{self.tnm_m}")
        return "".join(parts) if parts else None


@dataclass
class TreatmentContext:
    """Extracted treatment-related information"""
    modality: Optional[str] = None  # RT, chemo, surgery, immunotherapy
    setting: Optional[str] = None  # adjuvant, neoadjuvant, definitive, palliative
    prior_treatments: List[str] = field(default_factory=list)
    # Raw text span for semantic search
    raw_text: Optional[str] = None
    
    def has_data(self) -> bool:
        return any([self.modality, self.setting, self.prior_treatments])
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.modality:
            result['modality'] = self.modality
        if self.setting:
            result['setting'] = self.setting
        if self.prior_treatments:
            result['prior_treatments'] = self.prior_treatments
        if self.raw_text:
            result['raw_text'] = self.raw_text
        return result


@dataclass
class ClinicalHistory:
    """Extracted clinical history and imaging findings"""
    imaging_findings: Optional[str] = None
    recurrence_info: Optional[str] = None
    # Disease status axis: one of "primary", "recurrent", "metastatic",
    # "post_progression". Used by patient_eligibility_boost_service to
    # hard-filter studies whose enrolled population describes a different
    # disease state (e.g. an "N0 primary oral cancer" study for a
    # recurrent post-treatment patient).
    disease_status: Optional[str] = None
    # Surgical candidacy axis: one of "candidate", "not_candidate",
    # "declined". Distinct from `treatment.prior_treatments` (which
    # records past surgeries) — this captures whether the patient is
    # currently eligible for surgical management. Hard-filters studies
    # whose population requires (or excludes) surgical candidates.
    surgical_candidacy: Optional[str] = None
    raw_text: Optional[str] = None

    def has_data(self) -> bool:
        return any([
            self.imaging_findings, self.recurrence_info,
            self.disease_status, self.surgical_candidacy, self.raw_text,
        ])

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result


@dataclass
class QueryStructure:
    """Complete structured breakdown of user query"""
    
    # Original query
    original_query: str = ""
    
    # Query classification (from existing classify_query)
    query_type: str = "general"
    has_explicit_question: bool = True
    
    # Extracted contexts
    patient: PatientContext = field(default_factory=PatientContext)
    cancer: CancerContext = field(default_factory=CancerContext)
    treatment: TreatmentContext = field(default_factory=TreatmentContext)
    clinical_history: ClinicalHistory = field(default_factory=ClinicalHistory)
    
    # What's being asked
    question_focus: Optional[str] = None  # dose, survival, indication, toxicity, etc.
    
    # Retrieval optimization
    boost_terms: List[str] = field(default_factory=list)
    filter_category: Optional[str] = None
    
    # Flags
    has_patient_context: bool = False
    used_llm_extraction: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        result = {
            "query_type": self.query_type,
            "has_explicit_question": self.has_explicit_question,
            "has_patient_context": self.has_patient_context,
            "question_focus": self.question_focus,
            "patient": self.patient.to_dict() if self.patient.has_data() else None,
            "cancer": self.cancer.to_dict() if self.cancer.has_data() else None,
            "treatment": self.treatment.to_dict() if self.treatment.has_data() else None,
            "boost_terms": self.boost_terms if self.boost_terms else None,
            "filter_category": self.filter_category,
        }
        
        # Add clinical history if present
        if self.clinical_history.has_data():
            result["clinical_history"] = self.clinical_history.to_dict()
        
        # Add LLM extraction flag if used
        if self.used_llm_extraction:
            result["used_llm_extraction"] = True
        
        return result


# ============================================
# REGEX PATTERNS
# ============================================

# Patient demographics
PATIENT_PATTERNS = {
    "age": re.compile(r'(\d{1,3})[\s\-]*(?:year[\s\-]*old|yo|y[./]?o\.?|yr[\s\-]*old|years?\s*of\s*age)', re.I),
    "gender_male": re.compile(r'\b(male|man|gentleman|boy)\b', re.I),
    "gender_female": re.compile(r'\b(female|woman|lady|girl)\b', re.I),
    "ecog": re.compile(r'\bECOG\s*(?:PS\s*)?(\d)', re.I),
    "kps": re.compile(r'\bKPS\s*(\d{2,3})', re.I),
}

# Cancer sites - maps to Qdrant categories
CANCER_SITE_PATTERNS = {
    "breast": {
        "pattern": re.compile(r'\b(breast|mammary|dcis|lcis|mastectomy|lumpectomy)\b', re.I),
        "category": "breast",
    },
    "lung": {
        # Anatomic signals like "mediastinal mass", "hilar mass", and
        # "lobe"-based descriptions are strong primary-lung cues that
        # real clinical queries lean on ("mediastinal mass biopsy proven
        # adenocarcinoma" is a lung primary, not a mediastinal primary).
        "pattern": re.compile(
            r'\b(lung|nsclc|sclc|pulmonary|bronchogenic|'
            r'mediastinal\s*mass|hilar\s*mass|'
            r'lung\s*nodule|pulmonary\s*nodule|'
            r'(?:(?:upper|middle|lower|right|left)\s+)?lobe)\b',
            re.I,
        ),
        "category": "lung",
    },
    "prostate": {
        "pattern": re.compile(r'\b(prostate|prostatic|psa)\b', re.I),
        "category": "prostate",
    },
    "head_neck": {
        "pattern": re.compile(r'\b(head\s*(?:and|&)\s*neck|oral\s*cavity|tongue|larynx|pharynx|hnscc|oropharynx|nasopharynx|hypopharynx|tonsil|base\s*of\s*tongue|glottis|supraglottic|subglottic|salivary|parotid|maxilla|mandible|lip|floor\s*of\s*mouth|buccal|palate)\b', re.I),
        "category": "head_neck",
    },
    "gi_upper": {
        "pattern": re.compile(r'\b(esophagus|esophageal|gastric|stomach|gej|gastroesophageal)\b', re.I),
        "category": "gi",
    },
    "gi_lower": {
        "pattern": re.compile(r'\b(colon|rectal|rectum|colorectal|crc|anal|anus)\b', re.I),
        "category": "gi",
    },
    "gi_hepatobiliary": {
        "pattern": re.compile(r'\b(liver|hepatocellular|hcc|hepatic|cholangiocarcinoma|bile\s*duct|gallbladder|pancreas|pancreatic)\b', re.I),
        "category": "gi",
    },
    "gyn": {
        "pattern": re.compile(r'\b(cervix|cervical|uterus|uterine|endometrial|ovary|ovarian|vulva|vulvar|vagina|vaginal)\b', re.I),
        "category": "gyn",
    },
    "gu": {
        "pattern": re.compile(r'\b(bladder|kidney|renal|rcc|urothelial|testis|testicular|seminoma|penile)\b', re.I),
        "category": "gu",
    },
    "cns": {
        "pattern": re.compile(r'\b(brain|glioma|gbm|glioblastoma|meningioma|astrocytoma|medulloblastoma|ependymoma|craniopharyngioma|pituitary|acoustic|schwannoma|spine|spinal\s*cord)\b', re.I),
        "category": "cns",
    },
    "lymphoma": {
        "pattern": re.compile(r'\b(lymphoma|hodgkin|non-hodgkin|dlbcl|follicular|mantle\s*cell|marginal\s*zone|burkitt)\b', re.I),
        "category": "lymphoma",
    },
    "sarcoma": {
        "pattern": re.compile(r'\b(sarcoma|soft\s*tissue|osteosarcoma|ewing|rhabdomyosarcoma|liposarcoma|leiomyosarcoma|synovial)\b', re.I),
        "category": "sarcoma",
    },
    "skin": {
        "pattern": re.compile(r'\b(melanoma|basal\s*cell|bcc|squamous\s*cell\s*skin|merkel|cutaneous)\b', re.I),
        "category": "skin",
    },
    "thyroid": {
        "pattern": re.compile(r'\b(thyroid|papillary|follicular\s*thyroid|medullary\s*thyroid|anaplastic\s*thyroid)\b', re.I),
        "category": "thyroid",
    },
    "pediatric": {
        "pattern": re.compile(r'\b(pediatric|childhood|neuroblastoma|wilms|retinoblastoma)\b', re.I),
        "category": "pediatric",
    },
}

# Metastatic-context signals.
#
# The rule: a site match is metastatic iff one of these words appears
# IMMEDIATELY AFTER it ("hepatic metastases", "brain mets", "bone
# metastasis"). We deliberately do NOT treat pre-site modifiers as
# demoters — "metastatic prostate cancer" describes the disease state,
# not the site's role, and prostate is still the primary. A trailing-
# only window is the smallest change that handles the common clinical
# phrasings without demoting primaries by association.
METASTATIC_TRAILING_PATTERN = re.compile(
    r'\b(?:metastasis|metastases|metastatic|mets|metastasized)\b',
    re.I,
)

# Number of characters scanned to the right of a site match to decide
# whether that match is a metastatic site. 15 is tight enough to not
# leak past "X cancer with Y mets" (lung → no mets in trailing 15;
# prostate in "metastatic prostate cancer with brain mets" → no mets
# in trailing 15), but wide enough to catch "hepatic metastases",
# "brain mets", "liver metastasis in …" verbatim.
_METASTATIC_TRAILING_WINDOW = 15

# Back-compat alias for code that imported the old name.
METASTATIC_CONTEXT_PATTERN = METASTATIC_TRAILING_PATTERN

# Primary-tumor lead-in patterns: "<site> cancer", "adenocarcinoma of
# the <site>", etc. Kept for downstream callers from landmark that may
# import this — not consumed in this module today.
PRIMARY_TUMOR_PATTERN = re.compile(
    r'(?:adenocarcinoma\s+of\s+(?:the\s+)?|carcinoma\s+of\s+(?:the\s+)?|'
    r'SCC\s+of\s+(?:the\s+)?)',
    re.I,
)


# ---------------------------------------------------------------------------
# Disease-status patterns (recurrent / metastatic / post-progression / primary)
# ---------------------------------------------------------------------------
#
# Each entry: (pattern, canonical_status). Order matters: post_progression
# checks run before "recurrent" because a patient progressing on ICI is
# more specifically post-progression than just generically recurrent.
#
# Outputs feed `ClinicalHistory.disease_status` which is then used as a
# hard-filter axis by patient_eligibility_boost_service so that studies
# whose enrolled population is for a different disease state (e.g.
# "elective neck dissection in N0 primary" applied to a recurrent
# post-ICI patient) get filtered out, not just penalised.

DISEASE_STATUS_PATTERNS = [
    # Post-progression: explicit progression on a named therapy or
    # "refractory to" language. Most specific — check first.
    (
        re.compile(
            r'\b(?:progress(?:ing|ion|ed)\s+on|refractory\s+to|failed\s+on|'
            r'post[-\s]?progression|post[-\s]?ICI|post[-\s]?immunotherapy)\b',
            re.I,
        ),
        "post_progression",
    ),
    # Metastatic: explicit M1 / distant metastases / "metastatic <X>".
    (
        re.compile(
            r'\b(?:metastatic\s+(?:disease|cancer|carcinoma|adenocarcinoma|'
            r'sccs?|tumor|melanoma|prostate|breast|lung|colon|colorectal)|'
            r'distant\s+metast|stage\s*(?:IV|4)\b|\bM1\b|\bcM1\b|\bpM1\b)',
            re.I,
        ),
        "metastatic",
    ),
    # Recurrent: biopsy-proven recurrence, local recurrence,
    # locoregional progression. Comes after post_progression so the
    # more specific label wins when both apply.
    (
        re.compile(
            r'\b(?:biopsy[-\s]?proven\s+recurrent|recurrent\s+(?:lesion|disease|'
            r'tumor|carcinoma|cancer|sccs?|adenocarcinoma|melanoma)|'
            r'(?:local|locoregional)\s+recurrence|recurrence\s+(?:of|in)\b|'
            r'salvage\s+setting|R/M\s+(?:HNSCC|disease))',
            re.I,
        ),
        "recurrent",
    ),
]


# ---------------------------------------------------------------------------
# Surgical-candidacy patterns
# ---------------------------------------------------------------------------
#
# Each entry: (pattern, canonical_status). Distinct from prior surgical
# history. Captures the patient's *current* eligibility for surgical
# management, which often differs from whether they have had surgery in
# the past. A patient post-glossectomy who is "no longer a surgical
# candidate" should still hard-filter studies that require salvage
# surgery enrolment.

SURGICAL_CANDIDACY_PATTERNS = [
    (
        re.compile(
            r'\b(?:no\s+longer\s+(?:a\s+)?surgical\s+candidate|'
            r'not\s+(?:a\s+)?surgical\s+candidate|'
            r'unresectable|inoperable|'
            r'(?:beyond|exceeds)\s+(?:surgical|resection)\s+(?:salvage|consideration)|'
            r'salvage\s+(?:surgery|resection)\s+not\s+(?:feasible|possible|advisable))\b',
            re.I,
        ),
        "not_candidate",
    ),
    (
        re.compile(
            r'\b(?:declined\s+(?:surgery|resection|surgical)|'
            r'refused\s+(?:surgery|resection|surgical)|'
            r'patient\s+refusal\s+of\s+(?:surgery|resection))\b',
            re.I,
        ),
        "declined",
    ),
    # Explicit affirmation of candidacy. Only emit when the language
    # is clearly future-looking ("will undergo", "planned for") — past
    # surgery doesn't make someone a current candidate.
    (
        re.compile(
            r'\b(?:will\s+(?:undergo|proceed\s+(?:with|to))\s+'
            r'(?:surgery|resection|cystectomy|prostatectomy|mastectomy|lobectomy)|'
            r'planned\s+for\s+(?:surgery|resection)|'
            r'surgical\s+candidate\b(?!\s+for))',
            re.I,
        ),
        "candidate",
    ),
]


def extract_disease_status(query: str) -> Optional[str]:
    """Return the canonical disease_status label for a query, or None.

    First-match-wins on DISEASE_STATUS_PATTERNS (which are ordered from
    most specific to least). Returns None if no pattern fires so the
    eligibility check sees NOT_AVAILABLE rather than a guessed status.
    """
    for pattern, label in DISEASE_STATUS_PATTERNS:
        if pattern.search(query):
            return label
    return None


def extract_surgical_candidacy(query: str) -> Optional[str]:
    """Return "not_candidate" / "declined" / "candidate" or None.

    "not_candidate" is checked first so that "no longer a surgical
    candidate" wins over any incidental "surgical candidate" mention
    nearby. None when nothing matches — never inferred by default.
    """
    for pattern, label in SURGICAL_CANDIDACY_PATTERNS:
        if pattern.search(query):
            return label
    return None



# ── Metastatic site-canonical filter for Resolver Expansion ─────────────
#
# ``query_token_resolver.resolve_query_tokens`` returns every canonical
# site that appears (by variant) in the raw query text — including
# sites mentioned only as metastatic destinations ("hepatic mets" →
# canonical "Liver"). Injecting those canonicals into embedding input
# pollutes the dense retrieval with off-target studies (liver cancer
# studies for a lung-primary query).
#
# ``filter_metastatic_site_canonicals`` takes a list of canonicals and
# returns only those that appear as PRIMARY in the query — i.e., have
# at least one variant occurrence that is NOT immediately followed by
# a metastatic word. Uses the same trailing-window check as the
# structurer's site-detection loop, so the demotion logic is consistent
# across the pipeline.


def _site_canonical_has_primary_mention(query_text: str, canonical: str) -> bool:
    """True iff at least one occurrence of a variant of ``canonical``
    in ``query_text`` has no metastatic word in the trailing window.

    If the canonical doesn't appear in the query at all, returns True
    (we're filtering, not selecting — absence is not a demotion).
    """
    try:
        from src.ingestion.keyword_tagger import SITE_SYNONYM_GROUPS
    except Exception:
        return True

    # Build the variant list for this canonical
    variants: List[str] = []
    for group in SITE_SYNONYM_GROUPS:
        if group.get("canonical") == canonical:
            variants.append(canonical)
            variants.extend(group.get("variants", []) or [])
            break
    if not variants:
        return True  # unknown canonical — don't filter it

    lowered = query_text.lower()
    found_any = False
    for variant in variants:
        v_lower = variant.lower()
        start = 0
        while True:
            pos = lowered.find(v_lower, start)
            if pos < 0:
                break
            end = pos + len(v_lower)
            # Require variant to sit on word boundaries
            if pos > 0 and lowered[pos - 1].isalnum():
                start = pos + 1
                continue
            if end < len(lowered) and lowered[end].isalnum():
                start = pos + 1
                continue
            found_any = True
            trailing = query_text[end: end + _METASTATIC_TRAILING_WINDOW]
            if not METASTATIC_TRAILING_PATTERN.search(trailing):
                return True  # at least one primary occurrence
            start = end

    # Variant not found at all → don't filter (absence != metastatic)
    return not found_any or False


def filter_metastatic_site_canonicals(
    query_text: str,
    canonicals: List[str],
) -> List[str]:
    """Drop site canonicals whose ONLY occurrences in the query are
    metastatic-context.

    Used by enhanced_rag_service's Resolver Expansion block to avoid
    injecting wrong-site labels ("Liver", "Mediastinum") into the
    embedding for a query whose liver / mediastinal mentions are all
    metastatic descriptors.

    Args:
        query_text: The raw user query.
        canonicals: Site canonicals emitted by the resolver.

    Returns:
        A new list containing only canonicals that either (a) don't
        appear in the query at all, or (b) have at least one primary-
        context mention in the query.
    """
    if not canonicals:
        return []
    if not query_text:
        return list(canonicals)
    return [
        c for c in canonicals
        if _site_canonical_has_primary_mention(query_text, c)
    ]


# Site detail patterns (more specific anatomical locations)
SITE_DETAIL_PATTERNS = {
    "oral_cavity": re.compile(r'\b(oral\s*cavity|tongue|floor\s*of\s*mouth|buccal|palate|gingiva|lip|alveolar)\b', re.I),
    "oropharynx": re.compile(r'\b(oropharynx|oropharyngeal|tonsil|base\s*of\s*tongue|soft\s*palate)\b', re.I),
    "nasopharynx": re.compile(r'\b(nasopharynx|nasopharyngeal|npc)\b', re.I),
    "hypopharynx": re.compile(r'\b(hypopharynx|hypopharyngeal|pyriform)\b', re.I),
    "larynx": re.compile(r'\b(larynx|laryngeal|glottis|glottic|supraglottic|subglottic)\b', re.I),
    "maxilla": re.compile(r'\b(maxilla|maxillary|maxillectomy)\b', re.I),
}

# Histology patterns
HISTOLOGY_PATTERNS = {
    # Each pattern has a `(?<!non[- ])(?<!non)` negative lookbehind so a query
    # mentioning "non-small cell lung cancer" doesn't trigger the small_cell
    # histology (NSCLC ≠ SCLC — completely different disease). Same for
    # "non-squamous", "non-adenocarcinoma", "non-clear cell".
    "scc": re.compile(r'(?<!non[- ])(?<!non)\b(scc|squamous\s*cell\s*carcinoma|squamous\s*cell|epidermoid)\b', re.I),
    "adenocarcinoma": re.compile(r'(?<!non[- ])(?<!non)\b(adeno|adenocarcinoma)\b', re.I),
    "dcis": re.compile(r'\b(dcis|ductal\s*carcinoma\s*in\s*situ)\b', re.I),
    "lcis": re.compile(r'\b(lcis|lobular\s*carcinoma\s*in\s*situ)\b', re.I),
    "idc": re.compile(r'\b(idc|invasive\s*ductal|infiltrating\s*ductal)\b', re.I),
    "ilc": re.compile(r'\b(ilc|invasive\s*lobular|infiltrating\s*lobular)\b', re.I),
    "small_cell": re.compile(r'(?<!non[- ])(?<!non)\b(small\s*cell|sclc|oat\s*cell)\b', re.I),
    "large_cell": re.compile(r'(?<!non[- ])(?<!non)\b(large\s*cell)\b', re.I),
    "seminoma": re.compile(r'\b(seminoma|pure\s*seminoma)\b', re.I),
    "nonseminoma": re.compile(r'\b(non-?seminoma|nsgct|embryonal|yolk\s*sac|choriocarcinoma|teratoma)\b', re.I),
    "clear_cell": re.compile(r'(?<!non[- ])(?<!non)\b(clear\s*cell)\b', re.I),
    "transitional": re.compile(r'\b(transitional\s*cell|urothelial)\b', re.I),
}

# Staging patterns
STAGING_PATTERNS = {
    "stage_roman": re.compile(r'\bstage\s*([IViv]+[ABCabc]?)\b', re.I),
    "stage_numeric": re.compile(r'\bstage\s*([1-4][ABCabc]?)\b', re.I),
    "metastatic": re.compile(r'\b(metastatic|mets|m1|distant\s*metast|advanced\s*(?:stage)?)\b', re.I),
    "locally_advanced": re.compile(r'\b(locally\s*advanced|LA|unresectable)\b', re.I),
    # TNM patterns - handle both spaced (pT4 N0) and concatenated (pT4N0) formats
    "tnm_t": re.compile(r'[cyp]?T([0-4][a-d]?(?:is)?)(?=\s|N|M|$|[^a-zA-Z0-9])'),  # Lookahead for N, M, or end
    "tnm_n": re.compile(r'[cyp]?N([0-3][a-c]?(?:mi)?)(?=\s|M|$|[^a-zA-Z0-9])'),  # Lookahead for M or end
    "tnm_m": re.compile(r'[cyp]?M([01][a-c]?)(?=\s|$|[^a-zA-Z0-9])'),
    "grade": re.compile(r'\bgrade\s*([1-3]|I{1,3})\b', re.I),
}

# Receptor status (breast cancer)
RECEPTOR_PATTERNS = {
    "er_positive": re.compile(r'\b(ER\+|ER\s*positive|estrogen\s*receptor\s*positive|ER\s*\+)\b', re.I),
    "er_negative": re.compile(r'\b(ER-|ER\s*negative|estrogen\s*receptor\s*negative|ER\s*-)\b', re.I),
    "pr_positive": re.compile(r'\b(PR\+|PR\s*positive|progesterone\s*receptor\s*positive|PR\s*\+)\b', re.I),
    "pr_negative": re.compile(r'\b(PR-|PR\s*negative|progesterone\s*receptor\s*negative|PR\s*-)\b', re.I),
    "her2_positive": re.compile(r'\b(HER2\+|HER2\s*positive|HER2\s*amplified|ERBB2\+|HER2\s*\+)\b', re.I),
    "her2_negative": re.compile(r'\b(HER2-|HER2\s*negative|HER2\s*-)\b', re.I),
    "triple_negative": re.compile(r'\b(triple\s*negative|tnbc)\b', re.I),
    # Also match slash-separated format like ER+/PR+/HER2-
    "er_pos_slash": re.compile(r'\bER\+/', re.I),
    "er_neg_slash": re.compile(r'\bER-/', re.I),
    "pr_pos_slash": re.compile(r'/PR\+', re.I),
    "pr_neg_slash": re.compile(r'/PR-', re.I),
    "her2_pos_slash": re.compile(r'/HER2\+', re.I),
    "her2_neg_slash": re.compile(r'/HER2-', re.I),
}

# Biomarker patterns — oncology molecular markers beyond breast receptor status
# Each entry: (pattern, canonical_name) so we can normalize to a consistent label
BIOMARKER_PATTERNS = [
    # Lung / pan-cancer actionable mutations
    # NOTE: Positive-polarity (mutant) patterns MUST appear BEFORE negative-polarity
    # (wild-type) patterns for each gene. The [\s-]* allows optional hyphen so that
    # "EGFR-mutant" matches the mutant pattern. Wild-type patterns must NOT have a
    # bare `-` alternative — require explicit wild-type/negative language.
    (re.compile(r'\bEGFR[\s-]*(?:mutant|mutation|mutated|\+|positive|exon\s*\d+|L858R|T790M|del\s*19)', re.I), "EGFR mutant"),
    (re.compile(r'\bEGFR[\s-]*(?:wild[\s-]*type|wt|negative)\b', re.I), "EGFR wild-type"),
    (re.compile(r'\bALK\s*(?:positive|\+|rearranged|rearrangement|fusion|translocation)', re.I), "ALK+"),
    (re.compile(r'\bALK[\s-]*(?:negative)\b', re.I), "ALK-"),
    (re.compile(r'\bROS1\s*(?:positive|\+|rearranged|rearrangement|fusion)', re.I), "ROS1+"),
    (re.compile(r'\bKRAS[\s-]*(?:mutant|mutation|mutated|\+|G12C|G12D|G12V)', re.I), "KRAS mutant"),
    (re.compile(r'\bKRAS[\s-]*(?:wild[\s-]*type|wt|negative)\b', re.I), "KRAS wild-type"),
    (re.compile(r'\bBRAF[\s-]*(?:mutant|mutation|mutated|\+|V600E|V600)', re.I), "BRAF mutant"),
    (re.compile(r'\bBRAF[\s-]*(?:wild[\s-]*type|wt|negative)\b', re.I), "BRAF wild-type"),
    (re.compile(r'\bRET\s*(?:fusion|rearrangement|positive|\+|mutant|mutation)', re.I), "RET+"),
    (re.compile(r'\bMET\s*(?:exon\s*14|amplification|amplified|overexpression)', re.I), "MET altered"),
    (re.compile(r'\bNTRK\s*(?:fusion|positive|\+|rearrangement)', re.I), "NTRK fusion"),
    (re.compile(r'\bHER2\s*(?:mutant|mutation|mutated)\b', re.I), "HER2 mutant"),  # distinct from HER2+ amplification
    # PD-L1 expression
    (re.compile(r'\bPD-?L1\s*(?:positive|\+|high|>?\s*50%?|TPS\s*[>≥]\s*\d+)', re.I), "PD-L1 high"),
    (re.compile(r'\bPD-?L1\s*(?:negative|-|low|<?\s*1%?|TPS\s*[<≤]\s*\d+)', re.I), "PD-L1 low"),
    (re.compile(r'\bPD-?L1\s*(?:expression|status|score|TPS|CPS)', re.I), "PD-L1"),
    # CPS score (Combined Positive Score) — critical for ICI eligibility
    (re.compile(r'\bCPS\s*(?:score\s*(?:of\s*)?)?\s*(?:=\s*)?100\b', re.I), "CPS 100"),
    (re.compile(r'\bCPS\s*(?:score\s*(?:of\s*)?)?\s*[≥>]=?\s*\d+', re.I), "CPS high"),
    (re.compile(r'\bCPS\s*(?:score\s*(?:of\s*)?)?\s*\d+', re.I), "CPS positive"),
    # MSI / MMR
    (re.compile(r'\b(?:MSI-?H|microsatellite\s*instability[\s-]*high)', re.I), "MSI-H"),
    (re.compile(r'\b(?:MSS|MSI-?L|microsatellite\s*stable)', re.I), "MSS"),
    (re.compile(r'\b(?:dMMR|mismatch\s*repair\s*deficient)', re.I), "dMMR"),
    (re.compile(r'\b(?:pMMR|mismatch\s*repair\s*proficient)', re.I), "pMMR"),
    # TMB
    (re.compile(r'\bTMB[\s-]*(?:high|H)\b', re.I), "TMB-H"),
    (re.compile(r'\btumor\s*mutational\s*burden[\s-]*high', re.I), "TMB-H"),
    # BRCA
    (re.compile(r'\bBRCA[12]?[\s-]*(?:mutant|mutation|mutated|\+|positive|pathogenic)', re.I), "BRCA mutant"),
    (re.compile(r'\bBRCA[12]?[\s-]*(?:wild[\s-]*type|wt|negative)\b', re.I), "BRCA wild-type"),
    # HPV / p16
    # Hyphen in "p16-positive" / "HPV-positive" needs to be treated as a
    # connector (like whitespace), not as the negative polarity marker.
    # Previous patterns used `\s*(?:...|-)` for negative, so the bare `-`
    # alternative matched ANY hyphen — including the connector in
    # "p16-positive", flipping it to NEGATIVE. Clinically dangerous
    # because HPV/p16 status drives treatment de-escalation decisions
    # for OPSCC. Also matched "HPV-related" as NEGATIVE (false positive).
    # Fix: both patterns use [\s-]* so hyphen acts as a connector; the
    # bare `-` polarity marker (still needed for shorthand "p16-")
    # requires a word boundary (whitespace, end-of-string, or
    # punctuation) immediately after it.
    (re.compile(r'\b(?:HPV|p16)[\s-]*(?:positive|\+)', re.I), "HPV+"),
    (re.compile(r'\b(?:HPV|p16)(?:[\s-]*negative|\s*-(?=\s|$|[,.;:]))', re.I), "HPV-"),
    # Prostate-specific
    (re.compile(r'\bPSA\s*(?:>|≥|>=)\s*[\d.]+', re.I), "PSA elevated"),
    # GI-specific
    (re.compile(r'\bHER2\s*(?:amplified|amplification|overexpressed|overexpression|IHC\s*3\+|FISH\s*positive)', re.I), "HER2 amplified"),
    # IDH (glioma)
    (re.compile(r'\bIDH[12]?\s*(?:mutant|mutation|mutated|\+)', re.I), "IDH mutant"),
    (re.compile(r'\bIDH\s*(?:wild[\s-]*type|wt)\b', re.I), "IDH wild-type"),
    # MGMT (glioma)
    (re.compile(r'\bMGMT\s*(?:methylated|methylation|promoter\s*methylat)', re.I), "MGMT methylated"),
    (re.compile(r'\bMGMT\s*(?:unmethylated|un-?methylated)', re.I), "MGMT unmethylated"),
    # 1p/19q (oligodendroglioma)
    (re.compile(r'\b1p/?19q\s*(?:co-?deleted|co-?deletion|loss)', re.I), "1p/19q co-deleted"),
    # PIK3CA
    (re.compile(r'\bPIK3CA\s*(?:mutant|mutation|mutated|\+)', re.I), "PIK3CA mutant"),
    # FGFR
    (re.compile(r'\bFGFR[1-4]?\s*(?:alteration|altered|fusion|mutation|amplification)', re.I), "FGFR altered"),
]

# Treatment modality patterns
TREATMENT_MODALITY_PATTERNS = {
    "radiation": re.compile(r'\b(rt|radiation|radiotherapy|xrt|ebrt|imrt|vmat|sbrt|srs|brachytherapy|proton|chemoradiation|chemoradiotherapy|crt)\b', re.I),
    "chemotherapy": re.compile(r'\b(chemo|chemotherapy|ctx|systemic\s*therapy|cisplatin|carboplatin|docetaxel|paclitaxel|5-?fu|fluorouracil)\b', re.I),
    "surgery": re.compile(r'\b(surgery|surgical|resection|excision|mastectomy|lumpectomy|prostatectomy|nephrectomy|lobectomy|colectomy)\b', re.I),
    "immunotherapy": re.compile(r'\b(immunotherapy|checkpoint|pd-?1|pd-?l1|ctla-?4|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab)\b', re.I),
    "targeted": re.compile(r'\b(targeted|tki|egfr\s*inhibitor|her2\s*targeted|trastuzumab|pertuzumab|lapatinib)\b', re.I),
    "hormonal": re.compile(r'\b(hormonal|hormone|endocrine|tamoxifen|letrozole|anastrozole|adt|androgen\s*deprivation)\b', re.I),
}

# Treatment setting patterns
TREATMENT_SETTING_PATTERNS = {
    "adjuvant": re.compile(r'\b(adjuvant|postoperative|post-op)\b', re.I),
    "neoadjuvant": re.compile(r'\b(neoadjuvant|preoperative|pre-op|induction)\b', re.I),
    "definitive": re.compile(r'\b(definitive|curative|radical)\b', re.I),
    "palliative": re.compile(r'\b(palliative|symptom|comfort)\b', re.I),
    "salvage": re.compile(r'\b(salvage|recurrent|relapsed)\b', re.I),
    "concurrent": re.compile(r'\b(concurrent|chemoradiation|chemoradiotherapy|crt)\b', re.I),
}

# Prior treatment patterns
PRIOR_TREATMENT_PATTERNS = re.compile(
    r'\b(?:s/p|status\s*post|after|following|prior|previous)\s+'
    r'(surgery|resection|mastectomy|lumpectomy|chemotherapy|chemo|radiation|rt|'
    r'prostatectomy|nephrectomy|orchiectomy|maxillectomy|laryngectomy|'
    r'immunotherapy|targeted\s*therapy)\b',
    re.I
)

# Question focus patterns
QUESTION_FOCUS_PATTERNS = {
    "dose": re.compile(r'\b(dose|dosing|gy|gray|fractionation|fraction|constraint|v\d+|d\d+)\b', re.I),
    "survival": re.compile(r'\b(survival|os|pfs|dfs|outcome|prognosis|mortality)\b', re.I),
    "indication": re.compile(r'\b(indication|indicated|when\s*should|who\s*should|criteria|candidate|appropriate)\b', re.I),
    "toxicity": re.compile(r'\b(toxicity|side\s*effect|adverse|complication|acute|late\s*effect)\b', re.I),
    "comparison": re.compile(r'\b(compare|comparison|versus|vs|difference|better|superior|inferior)\b', re.I),
    "technique": re.compile(r'\b(technique|approach|method|protocol|field|volume|target)\b', re.I),
    "staging": re.compile(r'\b(stage|staging|tnm|ajcc|workup)\b', re.I),
}

# Question markers (to detect explicit questions)
QUESTION_MARKERS = re.compile(
    r'(\?|^what\s|^how\s|^why\s|^when\s|^where\s|^which\s|^who\s|'
    r'^is\s|^are\s|^can\s|^could\s|^should\s|^would\s|^will\s|'
    r'^do\s|^does\s|^did\s|^has\s|^have\s|'
    r'\bcompare\b|\bexplain\b|\bdescribe\b|\blist\b|\bshow\b)',
    re.I | re.M
)


# ============================================
# PATIENT SIGNAL SCORING
# ============================================

# Regex patterns for patient signal detection
_AGE_SIGNAL_RE = re.compile(r'\d{1,3}\s*(?:year|yo|y\.?o\.?)', re.I)
_GENDER_SIGNAL_RE = re.compile(r'\b(male|female|man|woman|he|she)\b', re.I)
_PATIENT_PHRASE_RE = re.compile(
    r'(?:patient with|presenting with|diagnosed with|history of)', re.I
)
_BIOMARKER_POLARITY_KEYWORDS = {
    "mutant", "wild-type", "wildtype", "wild type", "positive", "negative",
    "amplified", "overexpressed", "mutated", "methylated", "unmethylated",
    "co-deleted", "rearranged", "fusion",
}


def _patient_signal_score(raw_text: str, extracted: QueryStructure) -> int:
    """
    Count patient-indicative signals from raw query text and extracted structure.

    Returns an integer score. The threshold for has_patient_context is >= 2.

    Signal table:
        Age mention          +2   regex on raw_text
        Gender mention       +1   regex on raw_text
        Patient phrases      +2   regex on raw_text
        Biomarkers w/ polar  +2   extracted.cancer.biomarkers with polarity keyword
        Prior treatments     +2   extracted.treatment.prior_treatments
        TNM staging          +1   extracted.cancer.tnm_t is not None
        Comorbidities        +1   extracted.patient.comorbidities non-empty
        Diagnosis+staging    +2   extracted.cancer.site and extracted.cancer.stage
    """
    score = 0
    text_lower = raw_text.lower()

    # Age mention (+2)
    if _AGE_SIGNAL_RE.search(raw_text):
        score += 2

    # Gender mention (+1)
    if _GENDER_SIGNAL_RE.search(raw_text):
        score += 1

    # Patient phrases (+2)
    if _PATIENT_PHRASE_RE.search(raw_text):
        score += 2

    # Biomarkers with polarity (+2)
    if extracted.cancer.biomarkers:
        has_polarity = False
        for marker in extracted.cancer.biomarkers:
            marker_lower = marker.lower()
            for kw in _BIOMARKER_POLARITY_KEYWORDS:
                if kw in marker_lower:
                    has_polarity = True
                    break
            if has_polarity:
                break
        if has_polarity:
            score += 2

    # Prior treatments (+2)
    if extracted.treatment.prior_treatments:
        score += 2

    # TNM staging (+1)
    if extracted.cancer.tnm_t is not None:
        score += 1

    # Comorbidities (+1)
    if extracted.patient.comorbidities:
        score += 1

    # Diagnosis + staging combo (+2)
    if extracted.cancer.site and extracted.cancer.stage:
        score += 2

    return score


# ============================================
# MAIN STRUCTURING FUNCTION
# ============================================

def structure_query(query: str, query_type: str = "general") -> QueryStructure:
    """
    Extract structured information from a user query using regex patterns.

    This is designed to be fast (no LLM calls) and provide:
    - Patient context (age, gender, performance status)
    - Cancer context (site, histology, stage, TNM)
    - Treatment context (modality, setting, prior treatments)
    - Question focus (what's being asked)
    - Boost terms for retrieval optimization

    Args:
        query: The user's query string
        query_type: Pre-classified query type from classify_query()

    Returns:
        QueryStructure with all extracted information
    """
    print(f"    [QueryStructuring] Parsing query ({len(query)} chars) with regex patterns...")
    structure = QueryStructure(
        original_query=query,
        query_type=query_type,
    )

    query_lower = query.lower()
    
    # ===================================================================
    # 1. Detect if this is an explicit question
    # ===================================================================
    structure.has_explicit_question = bool(QUESTION_MARKERS.search(query))
    
    # ===================================================================
    # 2. Extract patient demographics
    # ===================================================================
    patient = PatientContext()
    
    # Age
    age_match = PATIENT_PATTERNS["age"].search(query)
    if age_match:
        try:
            patient.age = int(age_match.group(1))
        except ValueError:
            pass
    
    # Gender
    if PATIENT_PATTERNS["gender_male"].search(query):
        patient.gender = "male"
    elif PATIENT_PATTERNS["gender_female"].search(query):
        patient.gender = "female"
    
    # Performance status
    ecog_match = PATIENT_PATTERNS["ecog"].search(query)
    if ecog_match:
        patient.performance_status = f"ECOG {ecog_match.group(1)}"
    else:
        kps_match = PATIENT_PATTERNS["kps"].search(query)
        if kps_match:
            patient.performance_status = f"KPS {kps_match.group(1)}"
    
    # Capture raw patient demographics text (for semantic search)
    # Pattern: "XX year old [ethnicity] [gender], [smoking status], [comorbidities]"
    patient_raw_pattern = re.compile(
        r'(\d{1,3}\s*(?:year|yr|y/?o)[\s-]*old\s*'  # Age
        r'(?:[a-z]+\s+)?'  # Optional ethnicity (asian, african american, etc.)
        r'(?:male|female|man|woman|m|f)'  # Gender
        r'(?:[,\s]+(?:non-?smoker|smoker|former\s*smoker|never\s*smoked|'  # Smoking
        r'diabetic|hypertensive|obese|with\s+[^,\.]+))*)',  # Comorbidities
        re.I
    )
    patient_raw_match = patient_raw_pattern.search(query)
    if patient_raw_match:
        patient.raw_text = patient_raw_match.group(0).strip()
    
    structure.patient = patient
    
    # ===================================================================
    # 3. Extract cancer context
    # ===================================================================
    cancer = CancerContext()

    # Cancer site (and category for filtering).
    #
    # Primary-vs-metastatic precedence: a pattern that matches inside a
    # metastatic-context window (e.g. "hepatic metastases", "brain mets")
    # is NOT a primary tumor and must not be assigned to `cancer.site`.
    # The old "first match wins" loop silently used metastatic matches
    # as primaries, which caused e.g. "mediastinal mass... adenocarcinoma...
    # hepatic metastases" to be classified as a GI/hepatobiliary primary.
    #
    # We collect every pattern hit, bucket each one as primary vs.
    # metastatic based on nearby words, prefer the first primary match,
    # and leave `cancer.site = None` if every match was metastatic. The
    # LLM extractor (see merge_llm_extraction) can recover the primary
    # from its `primary_cancer` axis in that case.
    primary_site = None
    primary_category = None
    metastatic_sites: List[str] = []
    for site_name, site_info in CANCER_SITE_PATTERNS.items():
        m = site_info["pattern"].search(query)
        if not m:
            continue
        # Trailing-only window: only demote if a mets word comes right
        # AFTER the site mention ("hepatic metastases"), not before
        # it ("metastatic prostate cancer" leaves prostate as primary).
        trailing = query[m.end(): m.end() + _METASTATIC_TRAILING_WINDOW]
        if METASTATIC_TRAILING_PATTERN.search(trailing):
            if site_name not in metastatic_sites:
                metastatic_sites.append(site_name)
        elif primary_site is None:
            primary_site = site_name
            primary_category = site_info["category"]

    if primary_site is not None:
        cancer.site = primary_site
        structure.filter_category = primary_category
    # else: leave cancer.site = None. Downstream (merge_llm_extraction)
    # will try to recover a primary from the LLM's primary_cancer axis.

    # Record the metastatic sites so downstream scoring / eligibility
    # layers can reason about disease extent without treating mets as
    # primaries.
    if metastatic_sites:
        cancer.metastatic_sites_detected = metastatic_sites
    
    # Site detail (more specific location)
    for detail_name, detail_pattern in SITE_DETAIL_PATTERNS.items():
        if detail_pattern.search(query):
            cancer.site_detail = detail_name
            break
    
    # Histology
    for hist_name, hist_pattern in HISTOLOGY_PATTERNS.items():
        if hist_pattern.search(query):
            cancer.histology = hist_name
            break
    
    # Stage - check explicit stage first, then infer from metastatic/advanced
    stage_match = STAGING_PATTERNS["stage_roman"].search(query)
    if stage_match:
        cancer.stage = stage_match.group(1).upper()
    else:
        stage_match = STAGING_PATTERNS["stage_numeric"].search(query)
        if stage_match:
            # Convert numeric to roman
            num_to_roman = {"1": "I", "2": "II", "3": "III", "4": "IV"}
            stage_num = stage_match.group(1)
            cancer.stage = num_to_roman.get(stage_num[0], stage_num[0]) + stage_num[1:].upper()
        elif STAGING_PATTERNS["metastatic"].search(query):
            # Infer stage IV from "metastatic" or "advanced"
            cancer.stage = "IV"
        elif STAGING_PATTERNS["locally_advanced"].search(query):
            # Infer stage III from "locally advanced"
            cancer.stage = "III"
    
    # TNM staging
    t_match = STAGING_PATTERNS["tnm_t"].search(query)
    if t_match:
        cancer.tnm_t = t_match.group(1)
    
    n_match = STAGING_PATTERNS["tnm_n"].search(query)
    if n_match:
        cancer.tnm_n = n_match.group(1)
    
    m_match = STAGING_PATTERNS["tnm_m"].search(query)
    if m_match:
        cancer.tnm_m = m_match.group(1)
    
    # Grade
    grade_match = STAGING_PATTERNS["grade"].search(query)
    if grade_match:
        cancer.grade = grade_match.group(1)
    
    # Receptor status (breast cancer)
    receptor_parts = []
    if RECEPTOR_PATTERNS["triple_negative"].search(query):
        cancer.receptor_status = "triple negative"
        # Populate biomarkers with the individual negative markers so the
        # structured matcher can do polarity-aware matching and exclusion
        cancer.biomarkers.extend(["ER-", "PR-", "HER2-", "triple negative"])
    else:
        # Check both standalone and slash-separated formats
        if RECEPTOR_PATTERNS["er_positive"].search(query) or RECEPTOR_PATTERNS["er_pos_slash"].search(query):
            receptor_parts.append("ER+")
        elif RECEPTOR_PATTERNS["er_negative"].search(query) or RECEPTOR_PATTERNS["er_neg_slash"].search(query):
            receptor_parts.append("ER-")
        
        if RECEPTOR_PATTERNS["pr_positive"].search(query) or RECEPTOR_PATTERNS["pr_pos_slash"].search(query):
            receptor_parts.append("PR+")
        elif RECEPTOR_PATTERNS["pr_negative"].search(query) or RECEPTOR_PATTERNS["pr_neg_slash"].search(query):
            receptor_parts.append("PR-")
        
        if RECEPTOR_PATTERNS["her2_positive"].search(query) or RECEPTOR_PATTERNS["her2_pos_slash"].search(query):
            receptor_parts.append("HER2+")
        elif RECEPTOR_PATTERNS["her2_negative"].search(query) or RECEPTOR_PATTERNS["her2_neg_slash"].search(query):
            receptor_parts.append("HER2-")
        
        if receptor_parts:
            cancer.receptor_status = "/".join(receptor_parts)
            # Also add individual receptor markers to biomarkers list so the
            # structured matcher can do polarity-aware matching
            for part in receptor_parts:
                if part not in cancer.biomarkers:
                    cancer.biomarkers.append(part)

    # Biomarkers (molecular markers beyond receptor status)
    # IMPORTANT: append to existing list — receptor_status markers (ER+, HER2-, etc.)
    # were already added above and must NOT be overwritten.
    existing_upper = {b.upper() for b in cancer.biomarkers}
    for pattern, canonical_name in BIOMARKER_PATTERNS:
        if pattern.search(query):
            if canonical_name.upper() not in existing_upper:
                cancer.biomarkers.append(canonical_name)
                existing_upper.add(canonical_name.upper())
    
    structure.cancer = cancer
    
    # ===================================================================
    # 4. Extract treatment context
    # ===================================================================
    treatment = TreatmentContext()
    
    # Treatment modality
    for modality_name, modality_pattern in TREATMENT_MODALITY_PATTERNS.items():
        if modality_pattern.search(query):
            treatment.modality = modality_name
            break
    
    # Treatment setting
    for setting_name, setting_pattern in TREATMENT_SETTING_PATTERNS.items():
        if setting_pattern.search(query):
            treatment.setting = setting_name
            break
    
    # Prior treatments
    prior_matches = PRIOR_TREATMENT_PATTERNS.findall(query)
    if prior_matches:
        treatment.prior_treatments = list(set(m.lower() for m in prior_matches))
    
    structure.treatment = treatment
    
    # ===================================================================
    # 5. Determine question focus
    # ===================================================================
    for focus_name, focus_pattern in QUESTION_FOCUS_PATTERNS.items():
        if focus_pattern.search(query):
            structure.question_focus = focus_name
            break
    
    # If no focus detected, infer from query_type
    if not structure.question_focus:
        type_to_focus = {
            "dose_question": "dose",
            "treatment_recommendation": "treatment",
            "indication_question": "indication",
            "side_effects": "toxicity",
            "trial_results": "survival",
            "staging": "staging",
        }
        structure.question_focus = type_to_focus.get(query_type)
    
    # ===================================================================
    # 6. Determine if we have patient context
    # ===================================================================
    structure.has_patient_context = _patient_signal_score(query, structure) >= 2
    
    # ===================================================================
    # 7. Build boost terms for retrieval
    # ===================================================================
    boost_terms = []
    
    # Add cancer site terms
    if cancer.site:
        boost_terms.append(cancer.site.replace("_", " "))
    if cancer.site_detail:
        boost_terms.append(cancer.site_detail.replace("_", " "))
    
    # Add histology
    if cancer.histology:
        boost_terms.append(cancer.histology.upper() if cancer.histology == "scc" else cancer.histology)
    
    # Add staging terms
    if cancer.stage:
        boost_terms.append(f"stage {cancer.stage}")
    if cancer.tnm_t:
        boost_terms.append(f"T{cancer.tnm_t}")
    if cancer.tnm_n:
        boost_terms.append(f"N{cancer.tnm_n}")
    if cancer.tnm_m:
        boost_terms.append(f"M{cancer.tnm_m}")
    
    # Add receptor status
    if cancer.receptor_status:
        boost_terms.append(cancer.receptor_status)
    
    # Add biomarkers
    for marker in cancer.biomarkers:
        boost_terms.append(marker)
    
    # Add treatment modality
    if treatment.modality:
        boost_terms.append(treatment.modality)
    
    # Add treatment setting
    if treatment.setting:
        boost_terms.append(treatment.setting)
    
    # Add question focus terms
    if structure.question_focus == "dose":
        boost_terms.extend(["dose", "Gy", "fractionation"])
    elif structure.question_focus == "survival":
        boost_terms.extend(["survival", "outcome", "OS", "PFS"])
    elif structure.question_focus == "indication":
        boost_terms.extend(["indication", "recommended", "criteria"])
    elif structure.question_focus == "toxicity":
        boost_terms.extend(["toxicity", "side effect", "complication"])
    
    structure.boost_terms = boost_terms

    # Summary log
    extracted = []
    if patient.age: extracted.append(f"age={patient.age}")
    if patient.gender: extracted.append(f"gender={patient.gender}")
    if patient.performance_status: extracted.append(f"ps={patient.performance_status}")
    if patient.comorbidities: extracted.append(f"comorbidities={patient.comorbidities}")
    if cancer.site: extracted.append(f"site={cancer.site}")
    if cancer.site_detail: extracted.append(f"site_detail={cancer.site_detail}")
    if cancer.histology: extracted.append(f"histology={cancer.histology}")
    if cancer.stage: extracted.append(f"stage={cancer.stage}")
    if cancer.tnm_t: extracted.append(f"T={cancer.tnm_t}")
    if cancer.tnm_n: extracted.append(f"N={cancer.tnm_n}")
    if cancer.tnm_m: extracted.append(f"M={cancer.tnm_m}")
    if cancer.biomarkers: extracted.append(f"biomarkers={cancer.biomarkers}")
    if cancer.receptor_status: extracted.append(f"receptor={cancer.receptor_status}")
    if cancer.doi: extracted.append(f"DOI={cancer.doi}")
    if cancer.pni: extracted.append(f"PNI={cancer.pni}")
    if cancer.lvi: extracted.append(f"LVI={cancer.lvi}")
    if treatment.modality: extracted.append(f"tx_modality={treatment.modality}")
    if treatment.setting: extracted.append(f"tx_setting={treatment.setting}")
    if treatment.prior_treatments: extracted.append(f"prior_tx={treatment.prior_treatments}")
    if structure.question_focus: extracted.append(f"focus={structure.question_focus}")
    if structure.filter_category: extracted.append(f"category={structure.filter_category}")

    # Disease status + surgical candidacy axes (regex-based; LLM
    # extraction may also populate via merge_llm_extraction). These
    # land on clinical_history so patient_eligibility_boost_service
    # can verdict them as hard-filter axes.
    disease_status = extract_disease_status(query)
    if disease_status and not structure.clinical_history.disease_status:
        structure.clinical_history.disease_status = disease_status
        extracted.append(f"disease_status={disease_status}")

    surgical_candidacy = extract_surgical_candidacy(query)
    if surgical_candidacy and not structure.clinical_history.surgical_candidacy:
        structure.clinical_history.surgical_candidacy = surgical_candidacy
        extracted.append(f"surgical_candidacy={surgical_candidacy}")

    print(f"    [QueryStructuring] has_patient_context: {structure.has_patient_context}")
    print(f"    [QueryStructuring] Extracted {len(extracted)} fields: {', '.join(extracted) if extracted else 'none'}")
    if boost_terms:
        print(f"    [QueryStructuring] Boost terms ({len(boost_terms)}): {boost_terms[:10]}{'...' if len(boost_terms) > 10 else ''}")

    return structure


def get_retrieval_boost_filter(structure: QueryStructure) -> Dict[str, Any]:
    """
    Generate retrieval boost parameters from query structure.
    
    Returns a dict with:
    - boost_terms: Terms to boost in lexical scoring
    - category_filter: Category to filter by (if confident)
    - must_match_terms: Terms that must appear (for strict filtering)
    """
    result = {
        "boost_terms": structure.boost_terms,
        "category_filter": structure.filter_category,
        "must_match_terms": [],
    }
    
    # Add must-match terms for very specific queries
    if structure.cancer.tnm_t and structure.cancer.tnm_n:
        # If we have specific TNM, these are important
        result["must_match_terms"].append(f"T{structure.cancer.tnm_t}")
        result["must_match_terms"].append(f"N{structure.cancer.tnm_n}")
    
    return result


# ============================================
# SINGLETON
# ============================================

_query_structuring_service = None

def get_query_structuring_service():
    """Get singleton instance (for consistency, though this is stateless)"""
    global _query_structuring_service
    if _query_structuring_service is None:
        _query_structuring_service = True  # Stateless, just mark as initialized
    return True


def structure_query_fast(query: str, query_type: str = "general") -> QueryStructure:
    """
    Convenience function for fast query structuring.
    This is the main entry point for the service.
    """
    return structure_query(query, query_type)


# ============================================
# LLM-BASED EXTRACTION (for complex queries)
# ============================================

# Patterns that indicate a complex query needing LLM extraction
COMPLEX_QUERY_INDICATORS = {
    "pathology_terms": re.compile(
        r'\b(s/p|status\s*post|PMH|DOI|LVI|PNI|margins?|lymph\s*nodes?|'
        r'poorly\s*differentiated|well\s*differentiated|moderately\s*differentiated|'
        r'recurrence|recurrent|metastatic|metastasis|imaging|CT|MRI|PET|'
        r'enhancement|lesion|lymphadenopathy|nodal|dissection)\b',
        re.I
    ),
    "date_patterns": re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}'),
    "measurement_patterns": re.compile(r'\d+\.?\d*\s*(cm|mm|x)\s*\d'),
}

# LLM extraction template - 8 clinical axes
LLM_EXTRACTION_TEMPLATE = """{
  "primary_cancer": "",
  "tnm_pathology": "",
  "prior_definitive_treatment": "",
  "current_treatment": "",
  "biomarker_profile": "",
  "disease_trajectory": "",
  "metastatic_concern": "",
  "patient_factors": ""
}"""

LLM_EXTRACTION_PROMPT = """Extract structured text spans from this clinical query into 8 clinical axes. Fill each field with EXACT text from the query - do not paraphrase or summarize. Leave fields empty ("") if not mentioned.

Axis definitions:
- primary_cancer: Histology, cancer site, sub-site (e.g., "SCC left oral tongue", "adenocarcinoma of the transverse colon")
- tnm_pathology: TNM staging, DOI, PNI, LVSI, margin status, differentiation grade, lymph node counts (e.g., "pT2pN0M0R0, DOI 5.1 mm, PNI-, LVSI-")
- prior_definitive_treatment: All prior surgeries with dates, reconstruction, adjuvant therapy (e.g., "left partial glossectomy, left neck dissection levels I-III, radial forearm free flap reconstruction 12/2/2024")
- current_treatment: Current/recent agent, dose, line of therapy, response vs. progression (e.g., "started on pembrolizumab, declined combination with chemotherapy")
- biomarker_profile: CPS score, PD-L1 expression, HPV/p16 status, MSI, EGFR, ALK, receptor status (e.g., "CPS score of 100")
- disease_trajectory: Recurrence pattern, time to recurrence, progression on therapy, ICI-refractory status (e.g., "biopsy-proven recurrent SCC, locoregional progression on ICI")
- metastatic_concern: Suspected or confirmed metastatic sites, imaging findings suggesting distant disease (e.g., "radiographic concern for metastatic disease to the right ventricle")
- patient_factors: Surgical candidacy, performance status, relevant comorbidities, smoking status (e.g., "no longer a surgical candidate, PMH HTN, Hep C, CKD")

Query: {query}

Template:
{template}

Return ONLY the filled JSON, no explanation."""


def _needs_llm_extraction(query: str) -> bool:
    """
    Detect if a query is complex enough to benefit from LLM extraction.
    
    Complex queries typically have:
    - Length > 200 characters
    - Multiple commas (>5) indicating detailed patient history
    - Pathology/imaging terms
    - Date patterns
    - Measurement patterns
    
    Returns True if LLM extraction would add value.
    """
    # Length check
    if len(query) < 200:
        return False
    
    # Comma count (indicates detailed history)
    comma_count = query.count(',')
    if comma_count < 5:
        return False
    
    # Check for complex indicators
    indicators_found = 0
    for name, pattern in COMPLEX_QUERY_INDICATORS.items():
        if pattern.search(query):
            indicators_found += 1
    
    # Need at least 2 indicators for LLM extraction
    return indicators_found >= 2


async def structure_query_with_llm(query: str) -> Optional[Dict[str, str]]:
    """
    Extract structured text spans from a complex query using GPT-4o-mini.

    This is designed to run IN PARALLEL with embedding generation.
    Returns raw text spans (not normalized) for semantic search fallback.

    Args:
        query: The user's query string

    Returns:
        Dict with 8 clinical axes: primary_cancer, tnm_pathology,
        prior_definitive_treatment, current_treatment, biomarker_profile,
        disease_trajectory, metastatic_concern, patient_factors.
        Returns None if extraction fails.
    """
    try:
        from openai import AsyncOpenAI
        from src.core.config import settings

        print(f"    [LLM Extraction] Calling {settings.openai_mini_model} for 8-axis extraction...")
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        prompt = LLM_EXTRACTION_PROMPT.format(
            query=query,
            template=LLM_EXTRACTION_TEMPLATE
        )

        import time as _time
        _t0 = _time.perf_counter()
        response = await client.chat.completions.create(
            model=settings.openai_mini_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        _elapsed = (_time.perf_counter() - _t0) * 1000
        print(f"    [LLM Extraction] LLM responded in {_elapsed:.0f}ms")
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)

        # Filter out empty strings
        filtered = {k: v for k, v in result.items() if v and v.strip()}
        print(f"    [LLM Extraction] Extracted {len(filtered)} non-empty axes:")
        for k, v in filtered.items():
            print(f"      {k}: {v[:80]}{'...' if len(v) > 80 else ''}")
        return filtered

    except Exception as e:
        print(f"    [LLM Extraction] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# Known gene families for polarity reconciliation
_GENE_FAMILIES = {
    "EGFR": ["EGFR"],
    "BRCA": ["BRCA", "BRCA1", "BRCA2"],
    "KRAS": ["KRAS"],
    "BRAF": ["BRAF"],
    "IDH": ["IDH", "IDH1", "IDH2"],
    "ALK": ["ALK"],
    "ROS1": ["ROS1"],
}

_POSITIVE_POLARITY_WORDS = {"mutant", "mutation", "mutated", "positive", "pathogenic", "amplified"}
_NEGATIVE_POLARITY_WORDS = {"wild-type", "wildtype", "wt", "negative", "wild type"}


def _detect_polarity(text: str) -> Optional[str]:
    """Return 'positive', 'negative', or None for a biomarker string."""
    lower = text.lower()
    for w in _POSITIVE_POLARITY_WORDS:
        if w in lower:
            return "positive"
    for w in _NEGATIVE_POLARITY_WORDS:
        if w in lower:
            return "negative"
    return None


def _gene_family(marker: str) -> Optional[str]:
    """Return the gene family key for a biomarker string, or None."""
    upper = marker.upper()
    for family, members in _GENE_FAMILIES.items():
        for member in members:
            if upper.startswith(member):
                return family
    return None


def _reconcile_biomarkers(
    regex_biomarkers: List[str],
    merged_biomarkers: List[str],
    llm_bio_text: str,
) -> List[str]:
    """
    Reconcile regex vs LLM biomarker outputs.

    When the LLM biomarker_profile text contains a positive-polarity marker
    for a gene but the regex extracted a negative-polarity (wild-type) marker
    for the same gene, replace the regex marker with the LLM-derived one.

    Returns the reconciled biomarker list.
    """
    import re as _re

    # Parse LLM biomarker tokens with polarity
    llm_tokens: List[str] = []
    for token in _re.findall(
        r'(?:EGFR|BRCA[12]?|KRAS|BRAF|ALK|ROS1|IDH[12]?|HER2|NTRK|RET|MET|FGFR[1-4]?)'
        r'[\s-]*(?:mutant|mutation|mutated|wild[\s-]*type|wt|negative|positive|\+|-|'
        r'pathogenic|amplified|fusion|rearrangement|exon\s*\d+[\s]*(?:del(?:etion)?)?|'
        r'L858R|T790M|V600E?|G12[CDV])',
        llm_bio_text, _re.IGNORECASE,
    ):
        llm_tokens.append(token.strip())

    # Build a map: gene_family → (polarity, token) from LLM
    llm_gene_polarity: Dict[str, Tuple[str, str]] = {}
    for token in llm_tokens:
        family = _gene_family(token)
        polarity = _detect_polarity(token)
        if family and polarity:
            llm_gene_polarity[family] = (polarity, token)

    if not llm_gene_polarity:
        return merged_biomarkers

    # Walk through merged biomarkers and reconcile
    reconciled: List[str] = []
    replaced_families: set = set()

    for marker in merged_biomarkers:
        family = _gene_family(marker)
        regex_polarity = _detect_polarity(marker)

        if family and family in llm_gene_polarity:
            llm_polarity, llm_token = llm_gene_polarity[family]
            if regex_polarity and llm_polarity != regex_polarity:
                # Disagreement — LLM wins, drop the regex marker
                print(
                    f"    [Reconciliation] biomarkers: regex=['{marker}'] "
                    f"llm=['{llm_token}'] → using llm"
                )
                if family not in replaced_families:
                    # Find the canonical name from BIOMARKER_PATTERNS for the LLM token
                    canonical = _resolve_canonical(llm_token)
                    if canonical and canonical not in reconciled:
                        reconciled.append(canonical)
                    replaced_families.add(family)
                continue  # skip the regex marker
            else:
                # Agreement or no polarity info — keep as-is
                if marker not in reconciled:
                    reconciled.append(marker)
        else:
            if marker not in reconciled:
                reconciled.append(marker)

    # Add any LLM-only families that weren't in regex at all
    for family, (polarity, token) in llm_gene_polarity.items():
        if family not in replaced_families:
            canonical = _resolve_canonical(token)
            if canonical and canonical not in reconciled:
                # Only add if not already present
                reconciled.append(canonical)

    return reconciled


def _resolve_canonical(token: str) -> Optional[str]:
    """Resolve a raw biomarker token to its canonical name via BIOMARKER_PATTERNS."""
    for pattern, canonical_name in BIOMARKER_PATTERNS:
        if pattern.search(token):
            return canonical_name
    # Fallback: return the token itself
    return token


def merge_llm_extraction(
    structure: QueryStructure,
    llm_result: Dict[str, str],
    use_llm_primary_extraction: bool = True,
) -> QueryStructure:
    """
    Merge LLM extraction results (8 clinical axes) into the regex-based QueryStructure.

    LLM results provide raw_text spans for semantic search fallback.
    Regex results provide structured fields for filtering/boosting.

    When ``use_llm_primary_extraction`` is True (default), a reconciliation
    step runs after the standard merge:
      - **Biomarkers**: if the LLM extracted biomarker data that disagrees
        with the regex biomarkers (e.g. polarity flip), the LLM output wins.
      - **Site**: regex output is always preferred (Fix 2 made it correct).
      - Disagreements are logged for observability.

    When the flag is False, the existing behaviour is preserved (regex wins
    for all structured fields; LLM only populates raw_text / boost_terms).

    The 8 axes map to QueryStructure as follows:
      primary_cancer          → cancer.raw_text (primary identity)
      tnm_pathology           → cancer.raw_text (appended), boost_terms
      prior_definitive_treatment → treatment.raw_text
      current_treatment       → treatment.raw_text (appended)
      biomarker_profile       → cancer.biomarkers enrichment, boost_terms
      disease_trajectory      → clinical_history.recurrence_info, boost_terms
      metastatic_concern      → clinical_history.imaging_findings, boost_terms
      patient_factors         → patient.raw_text

    Args:
        structure: QueryStructure from regex extraction
        llm_result: Dict from LLM extraction (8 clinical axes)
        use_llm_primary_extraction: When True (default), reconcile biomarker
            disagreements in favour of the LLM. When False, keep regex output.

    Returns:
        Updated QueryStructure with raw_text fields populated
    """
    if not llm_result:
        return structure

    # Store the full 8-axis dict on the structure for downstream consumers
    # (inference layer, sub-query generation) that read axes directly
    structure._llm_axes = llm_result  # type: ignore[attr-defined]

    # Snapshot regex biomarkers BEFORE merge for reconciliation logging
    regex_biomarkers_before = list(structure.cancer.biomarkers or [])

    # Axis 1: Primary cancer → cancer.raw_text
    cancer_parts = []
    if llm_result.get("primary_cancer"):
        cancer_parts.append(llm_result["primary_cancer"])

        # If the regex pass left `cancer.site` empty (e.g. because every
        # site match in the raw query was inside a metastatic-context
        # window), re-run the site regex on the LLM's primary_cancer
        # string alone. That string is scoped to the primary tumor by
        # construction, so metastatic-context demotion can't fire and
        # we recover the primary site instead of leaving the pipeline
        # unscoped.
        if structure.cancer.site is None:
            primary_text = llm_result["primary_cancer"]
            for site_name, site_info in CANCER_SITE_PATTERNS.items():
                if site_info["pattern"].search(primary_text):
                    structure.cancer.site = site_name
                    structure.filter_category = site_info["category"]
                    print(
                        f"    [Structure] Recovered cancer.site={site_name!r} "
                        f"from LLM primary_cancer (regex had only mets matches)"
                    )
                    break

    # Axis 2: TNM + pathology → append to cancer.raw_text + boost_terms
    if llm_result.get("tnm_pathology"):
        cancer_parts.append(llm_result["tnm_pathology"])
        structure.boost_terms.append(llm_result["tnm_pathology"])
    if cancer_parts:
        structure.cancer.raw_text = " ".join(cancer_parts)

    # Axis 3: Prior definitive treatment → treatment.raw_text
    treatment_parts = []
    if llm_result.get("prior_definitive_treatment"):
        treatment_parts.append(llm_result["prior_definitive_treatment"])

    # Axis 4: Current/recent treatment → append to treatment.raw_text + boost_terms
    if llm_result.get("current_treatment"):
        treatment_parts.append(llm_result["current_treatment"])
        structure.boost_terms.append(llm_result["current_treatment"])
    if treatment_parts:
        structure.treatment.raw_text = " ".join(treatment_parts)

    # Axis 5: Biomarker profile → enrich cancer.biomarkers + boost_terms
    if llm_result.get("biomarker_profile"):
        bio_text = llm_result["biomarker_profile"]
        structure.boost_terms.append(bio_text)
        # Parse known biomarker tokens into the biomarkers list if not already present
        import re as _re
        existing = {b.upper() for b in (structure.cancer.biomarkers or [])}
        for token in _re.findall(
            r'CPS\s*(?:score\s*(?:of\s*)?)?\d+|PD-?L1[^,;]*|HPV[+-]?|p16[+-]?|'
            r'MSI-?[HL]|EGFR[+-]?|ALK[+-]?|ER[+-]|PR[+-]|HER2[+-]?',
            bio_text, _re.IGNORECASE,
        ):
            if token.strip().upper() not in existing:
                structure.cancer.biomarkers.append(token.strip())
                existing.add(token.strip().upper())

    # ------------------------------------------------------------------
    # Reconciliation: resolve regex vs LLM biomarker disagreements
    # ------------------------------------------------------------------
    if use_llm_primary_extraction and llm_result.get("biomarker_profile"):
        structure.cancer.biomarkers = _reconcile_biomarkers(
            regex_biomarkers=regex_biomarkers_before,
            merged_biomarkers=structure.cancer.biomarkers,
            llm_bio_text=llm_result["biomarker_profile"],
        )

    # Site: always keep regex output (Fix 2 made regex site correct).
    # cancer.site and filter_category are NOT overwritten by LLM.

    # Axis 6: Disease trajectory → clinical_history.recurrence_info + boost_terms
    if llm_result.get("disease_trajectory"):
        traj_text = llm_result["disease_trajectory"]
        structure.clinical_history.recurrence_info = traj_text
        structure.boost_terms.append(traj_text)

    # Axis 7: Metastatic concern → clinical_history.imaging_findings + boost_terms
    if llm_result.get("metastatic_concern"):
        met_text = llm_result["metastatic_concern"]
        structure.clinical_history.imaging_findings = met_text
        structure.boost_terms.append(met_text)

    # Axis 8: Patient factors → patient.raw_text
    if llm_result.get("patient_factors"):
        structure.patient.raw_text = llm_result["patient_factors"]

    return structure


async def structure_query_with_llm_if_needed(
    query: str, 
    query_type: str = "general"
) -> Tuple[QueryStructure, bool]:
    """
    Structure a query using regex, and optionally LLM for complex queries.
    
    This is the async entry point that can be called in parallel with embedding.
    
    Args:
        query: The user's query string
        query_type: Pre-classified query type
        
    Returns:
        Tuple of (QueryStructure, used_llm: bool)
    """
    # Always do regex extraction first (fast)
    structure = structure_query(query, query_type)
    
    # Check if LLM extraction would help
    if not _needs_llm_extraction(query):
        return structure, False
    
    # Run LLM extraction
    print(f"[QueryStructuring] Complex query detected, running LLM extraction...")
    llm_result = await structure_query_with_llm(query)
    
    if llm_result:
        structure = merge_llm_extraction(structure, llm_result)
        print(f"[QueryStructuring] LLM extraction added raw text spans")
        return structure, True
    
    return structure, False


# ============================================
# ACCUMULATED CONTEXT MERGING
# ============================================

def merge_query_structures(
    accumulated: Optional[Dict[str, Any]], 
    new_structure: QueryStructure
) -> QueryStructure:
    """
    Merge a new query structure into accumulated context from conversation.
    
    Rules:
    - New non-None values override old values
    - Lists (boost_terms, prior_treatments, comorbidities) are merged (union)
    - raw_text fields are concatenated with " | " separator
    - filter_category uses new if provided, else keeps old
    
    Args:
        accumulated: Previous accumulated context dict (from API response)
        new_structure: Newly extracted QueryStructure
        
    Returns:
        Merged QueryStructure with accumulated context
    """
    if not accumulated:
        return new_structure
    
    # Start with the new structure
    merged = new_structure
    
    # Helper to merge two values
    def merge_value(old_val, new_val, is_list=False):
        if new_val is not None and new_val != [] and new_val != "":
            if is_list and old_val:
                # Merge lists, keeping unique values
                return list(set(old_val + new_val))
            return new_val
        return old_val
    
    def merge_raw_text(old_text: Optional[str], new_text: Optional[str]) -> Optional[str]:
        if new_text and old_text:
            # Avoid duplicates
            if new_text in old_text:
                return old_text
            return f"{old_text} | {new_text}"
        return new_text or old_text
    
    # Merge patient context
    if accumulated.get("patient"):
        old_patient = accumulated["patient"]
        merged.patient.age = merge_value(old_patient.get("age"), merged.patient.age)
        merged.patient.gender = merge_value(old_patient.get("gender"), merged.patient.gender)
        merged.patient.performance_status = merge_value(
            old_patient.get("performance_status"), merged.patient.performance_status
        )
        merged.patient.ethnicity = merge_value(old_patient.get("ethnicity"), merged.patient.ethnicity)
        merged.patient.smoking_status = merge_value(
            old_patient.get("smoking_status"), merged.patient.smoking_status
        )
        merged.patient.comorbidities = merge_value(
            old_patient.get("comorbidities", []), merged.patient.comorbidities, is_list=True
        )
        merged.patient.raw_text = merge_raw_text(
            old_patient.get("raw_text"), merged.patient.raw_text
        )
    
    # Merge cancer context
    if accumulated.get("cancer"):
        old_cancer = accumulated["cancer"]
        merged.cancer.site = merge_value(old_cancer.get("site"), merged.cancer.site)
        merged.cancer.site_detail = merge_value(old_cancer.get("site_detail"), merged.cancer.site_detail)
        merged.cancer.histology = merge_value(old_cancer.get("histology"), merged.cancer.histology)
        merged.cancer.stage = merge_value(old_cancer.get("stage"), merged.cancer.stage)
        merged.cancer.tnm_t = merge_value(old_cancer.get("tnm_t"), merged.cancer.tnm_t)
        merged.cancer.tnm_n = merge_value(old_cancer.get("tnm_n"), merged.cancer.tnm_n)
        merged.cancer.tnm_m = merge_value(old_cancer.get("tnm_m"), merged.cancer.tnm_m)
        merged.cancer.grade = merge_value(old_cancer.get("grade"), merged.cancer.grade)
        merged.cancer.receptor_status = merge_value(
            old_cancer.get("receptor_status"), merged.cancer.receptor_status
        )
        merged.cancer.doi = merge_value(old_cancer.get("doi"), merged.cancer.doi)
        merged.cancer.lvi = merge_value(old_cancer.get("lvi"), merged.cancer.lvi)
        merged.cancer.pni = merge_value(old_cancer.get("pni"), merged.cancer.pni)
        merged.cancer.margins = merge_value(old_cancer.get("margins"), merged.cancer.margins)
        merged.cancer.lymph_nodes = merge_value(old_cancer.get("lymph_nodes"), merged.cancer.lymph_nodes)
        merged.cancer.biomarkers = merge_value(
            old_cancer.get("biomarkers", []), merged.cancer.biomarkers, is_list=True
        )
        merged.cancer.raw_text = merge_raw_text(
            old_cancer.get("raw_text"), merged.cancer.raw_text
        )
    
    # Merge treatment context
    if accumulated.get("treatment"):
        old_treatment = accumulated["treatment"]
        merged.treatment.modality = merge_value(
            old_treatment.get("modality"), merged.treatment.modality
        )
        merged.treatment.setting = merge_value(
            old_treatment.get("setting"), merged.treatment.setting
        )
        merged.treatment.prior_treatments = merge_value(
            old_treatment.get("prior_treatments", []), merged.treatment.prior_treatments, is_list=True
        )
        merged.treatment.raw_text = merge_raw_text(
            old_treatment.get("raw_text"), merged.treatment.raw_text
        )
    
    # Merge clinical history
    if accumulated.get("clinical_history"):
        old_history = accumulated["clinical_history"]
        merged.clinical_history.imaging_findings = merge_value(
            old_history.get("imaging_findings"), merged.clinical_history.imaging_findings
        )
        merged.clinical_history.recurrence_info = merge_value(
            old_history.get("recurrence_info"), merged.clinical_history.recurrence_info
        )
        merged.clinical_history.disease_status = merge_value(
            old_history.get("disease_status"), merged.clinical_history.disease_status
        )
        merged.clinical_history.surgical_candidacy = merge_value(
            old_history.get("surgical_candidacy"), merged.clinical_history.surgical_candidacy
        )
        merged.clinical_history.raw_text = merge_raw_text(
            old_history.get("raw_text"), merged.clinical_history.raw_text
        )
    
    # Merge boost terms (union)
    old_boost = accumulated.get("boost_terms", [])
    if old_boost:
        merged.boost_terms = list(set(old_boost + merged.boost_terms))
    
    # Keep filter_category from new if set, else use old
    if not merged.filter_category and accumulated.get("filter_category"):
        merged.filter_category = accumulated["filter_category"]
    
    # Update has_patient_context flag
    merged.has_patient_context = (
        merged.patient.has_data() or 
        merged.cancer.has_data() or 
        merged.treatment.prior_treatments or
        merged.treatment.setting is not None
    )
    
    print(f"[QueryStructuring] Merged accumulated context, boost_terms: {len(merged.boost_terms)}")
    
    return merged
