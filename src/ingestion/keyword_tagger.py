"""
Keyword matching and tagging for document chunks.

Enhanced, ambiguity-aware tagger that scans against:
  - data/keywords/extractor_keywords.json (1200+ extraction keywords)
  - data/ontology/cancer_type_ontology.json (22 cancer types with synonyms,
    subtypes, drugs, histologies, surgeries, related_diseases, related_pathways)
  - data/ontology/clinical_trial_ontology.json (biomarkers, genomic alterations,
    treatments, outcomes, 400+ controlled terms)
  - data/ajcc_staging_tables.json (staging aliases, T/N/M definitions)

Key improvements over the naive scanner:
  1. Cancer tagging is weighted by field specificity AND ambiguity.
     Terms shared across many cancers (e.g. "adenocarcinoma", "pembrolizumab")
     contribute only a fractional score per cancer; a cancer is tagged only if
     its total score exceeds a threshold OR it has a highly specific hit
     (synonym / related_disease / label).
  2. Biomarkers, drugs, and genomic alterations are surfaced into dedicated
     metadata slots (not just buried in generic ontology_tags).
  3. Short acronyms (<=3 chars) require word-boundary matching.

Used by:
  - ColabIngestionPipeline._tag_chunk_keywords() during initial ingestion
  - metadata_reupsert.py during post-hoc metadata updates
  - section_upsert.py during section-level re-chunking
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import defaultdict


# ─── Locate data directory ───────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_DATA_DIR = _REPO_ROOT / "data"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Per-field base weights for cancer ontology terms ────────────────────
# Higher = more specific evidence for that cancer type.
_CANCER_FIELD_WEIGHT = {
    "label":             5.0,   # "Lung Cancer", "Breast Cancer" — near-certain
    "synonyms":          4.0,   # cancer-name variants (e.g. "NSCLC")
    "related_diseases":  3.0,   # specific disease entities
    "subtypes":          2.0,
    "surgeries":         2.0,
    "histologies":       2.0,   # diagnostic pathology — boosted
    "keywords":          1.5,
    "related_pathways":  1.0,
    "drugs":             1.0,   # often shared across cancers
}

# Cancer tag decision: a cancer is tagged if EITHER
#   (a) a highly specific field matched (label / synonyms / related_diseases), OR
#   (b) the total weighted score meets _CANCER_SCORE_THRESHOLD.
# Either way, at least one non-generic term must have matched. Generic
# terms like "adenocarcinoma" alone cannot justify a cancer tag.
_CANCER_SCORE_THRESHOLD = 2.5
_CANCER_SPECIFIC_FIELDS = {"label", "synonyms", "related_diseases"}

# Terms that are so generic they should NEVER tag a cancer alone, even if
# listed in that cancer's ontology (they still contribute to the score).
_GENERIC_AMBIGUOUS_TERMS = {
    "adenocarcinoma", "carcinoma", "squamous cell carcinoma",
    "squamous cell", "tumor", "malignancy", "cancer",
    "ade",  # ambiguous acronym ("Adriamycin-Daunorubicin-Etoposide")
}


# ─── NegEx-style negation detection ──────────────────────────────────────
# Negation cues appearing BEFORE a term (within ~30 chars, same sentence)
_NEG_BEFORE_CUES = re.compile(
    r'\b('
    r'no|not|without|exclud(?:e|ed|es|ing|able)?|'
    r'negative\s+for|absence\s+of|lack(?:ing|s)?\s+of|'
    r'no\s+history\s+of|no\s+evidence\s+of|no\s+signs\s+of|no\s+known|'
    r'denies|denied|never\s+had|free\s+of|non-|'
    r'ruled\s+out|rule\s+out|ruling\s+out|'
    r'prior\s+[^.;]{0,25}\s+(?:excluded|ruled\s+out)'
    r')\b',
    re.IGNORECASE,
)
# Negation cues appearing AFTER a term. Anchored: the cue must appear
# IMMEDIATELY after the term (allowing at most a copula like was/were/is/are)
# so that "bronchus, EGFR L858R, ALK-negative" doesn't read "negative" 25
# chars later as negating "bronchus".
_NEG_AFTER_CUES = re.compile(
    r'^[\s,;:]*(?:(?:was|were|is|are|has\s+been|have\s+been)\s+)?'
    r'(?:ruled\s+out|excluded|absent|negative|'
    r'not\s+(?:present|found|seen|detected))\b',
    re.IGNORECASE,
)
_SENT_BOUNDARY = re.compile(r'[.;!?]')

# Window (characters) for negation cue proximity
_NEG_WINDOW_BEFORE = 60
_NEG_WINDOW_AFTER = 35


def _is_negated(
    text_lower: str, start: int, end: int, check_after: bool = True
) -> bool:
    """
    Simple NegEx-style negation detection: look for negation cues within
    a small character window on either side of the term match, but stop
    at sentence boundaries to avoid cross-sentence false positives.

    `check_after=False` disables the post-cue check — useful when the
    matched term itself contains a status word (like "triple-negative")
    where a trailing "negative" elsewhere would otherwise trigger.
    """
    # Window BEFORE the match, clipped to the nearest sentence boundary
    before = text_lower[max(0, start - _NEG_WINDOW_BEFORE):start]
    last_boundary = -1
    for bm in _SENT_BOUNDARY.finditer(before):
        last_boundary = bm.end()
    if last_boundary > 0:
        before = before[last_boundary:]
    if _NEG_BEFORE_CUES.search(before):
        return True

    if not check_after:
        return False

    # Window AFTER the match, clipped to the nearest sentence boundary.
    # The regex is anchored (^), so the cue must appear immediately after
    # the term. This avoids "bronchus, EGFR L858R, ALK-negative" wrongly
    # suppressing "bronchus" because "negative" happens to be within 35
    # chars but clearly applies to a different concept.
    after = text_lower[end:end + _NEG_WINDOW_AFTER]
    b = _SENT_BOUNDARY.search(after)
    if b:
        after = after[:b.start()]
    if _NEG_AFTER_CUES.match(after):
        return True

    return False


def _dedupe_preferring_canonical(items) -> List[str]:
    """
    Deduplicate a collection of terms case-insensitively, preferring the
    entry that has uppercase letters (canonical acronyms like "ER", "MSI-H")
    over all-lowercase variants from ontology loading.
    """
    # Canonical preference: prefer mixed-case (has uppercase) over all-lowercase
    best: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.lower()
        existing = best.get(key)
        if existing is None:
            best[key] = item
        else:
            # Prefer entry with uppercase letters (canonical acronym form)
            if any(c.isupper() for c in item) and not any(c.isupper() for c in existing):
                best[key] = item
    return sorted(best.values(), key=lambda s: s.lower())


# ─── Biomarker / alteration polarity detection ───────────────────────────
# For biomarkers, imaging, serum markers, and genomic alterations, a
# trailing "+/-/positive/negative/high/low/mutated/wild-type" is a STATUS,
# not a negation of the concept. We record these as status, not suppress.

_STATUS_PATTERNS = [
    # Amplified (before "positive" so it wins on "HER2 amplified")
    (re.compile(r'[\s\-/](?:amplified|amplification|amp)\b', re.IGNORECASE), "amplified"),
    # Mutated / mutation
    (re.compile(r'[\s\-/](?:mutated|mutation|mutant|methylated|methylation)\b', re.IGNORECASE), "mutated"),
    # Wild-type / wt (must come before "negative" so it wins)
    (re.compile(r'[\s\-/](?:wild[\s\-]?type|wt|unmutated)\b', re.IGNORECASE), "wild-type"),
    # Loss / lost
    (re.compile(r'[\s\-/](?:lost|loss|deleted|deletion)\b', re.IGNORECASE), "loss"),
    # Positive — accept space, slash, or hyphen as separator; also bare "+"
    # ("HER2+", "HER2-positive", "ER-positive", "HER2+/ER-positive")
    (re.compile(r'(?:[\s\-/](?:positive|pos|intact|detected|elevated|rising|increased|expressed|overexpressed|strongly\s+positive|present)\b|\+(?![\w]))', re.IGNORECASE), "positive"),
    # Negative — accept space, slash, or hyphen as separator; also bare "-"
    # ("HER2-", "HER2-negative", "ER-/PR-")
    (re.compile(r'(?:[\s\-/](?:negative|neg|absent|not\s+(?:expressed|detected|present))\b|-(?![\w]))', re.IGNORECASE), "negative"),
    # High / low status (literal words only — no bare "H" or "L")
    (re.compile(r'[\s\-/](?:high)\b', re.IGNORECASE), "high"),
    (re.compile(r'[\s\-/](?:low|ultra[\s\-]?low)\b', re.IGNORECASE), "low"),
]

# Prefix cues — status before the term
_PREFIX_STATUS = [
    (re.compile(r'\b(?:elevated|rising|high|positive|detected|expressed|overexpressed|increased|amplified|strong)\s*$', re.IGNORECASE), "positive"),
    (re.compile(r'\b(?:no|not|without|absent|no\s+evidence\s+of|negative\s+for|lack\s+of|low|weak)\s*$', re.IGNORECASE), "negative"),
]

# Numeric value patterns — tag the literal number
_VALUE_RE = re.compile(
    r'[\s:=]+(\d+(?:\.\d+)?)\s*(%|ng/ml|mg/ml|u/ml|iu/ml)?',
    re.IGNORECASE,
)


def _detect_status(text_lower: str, start: int, end: int, canonical: str) -> Optional[str]:
    """
    Detect polarity / status for a biomarker match at text[start:end].

    Looks immediately after (primary) and before (fallback) for cues like
    +/-, positive/negative, high/low, mutated/wild-type, or a numeric
    value. Returns one of:
      "positive" / "negative" / "high" / "low" / "mutated" / "amplified"
      / "value:<num>" / None
    """
    # If the canonical explicitly encodes a status (HER2-low, MSI-H,
    # TMB-H, MSI-L), inherit it. Careful: names like "PD-L1" also end in
    # "-l1" so the check must use word-boundary matches, not substring.
    c_low = canonical.lower()
    if re.search(r'-(?:high|h)\b', c_low) or c_low.endswith("-high"):
        return "high"
    if re.search(r'-(?:low|l)\b', c_low) or c_low.endswith("-low"):
        return "low"

    # Look at up to 20 chars immediately after the match
    after = text_lower[end:end + 20]
    # Stop at sentence boundary
    b = _SENT_BOUNDARY.search(after)
    if b:
        after = after[:b.start()]

    # Numeric value right after
    m = _VALUE_RE.match(after)
    if m:
        unit = m.group(2) or ""
        return f"value:{m.group(1)}{unit}".strip()

    # Trailing +/- / positive / negative / high / low
    for pat, status in _STATUS_PATTERNS:
        if pat.match(after):
            return status

    # Look backward up to 30 chars (same sentence only)
    before = text_lower[max(0, start - 30):start]
    last_boundary = -1
    for bm in _SENT_BOUNDARY.finditer(before):
        last_boundary = bm.end()
    if last_boundary > 0:
        before = before[last_boundary:]

    for pat, status in _PREFIX_STATUS:
        if pat.search(before):
            return status

    return None


def _dedupe_status_map(status_map: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    Deduplicate status map keys case-insensitively, preferring keys with
    uppercase letters (canonical acronym forms). Merges statuses from
    case-variant duplicates.
    """
    merged: Dict[str, Tuple[str, Set[str]]] = {}  # lower_key → (display_key, statuses)
    for key, statuses in status_map.items():
        if not isinstance(key, str) or not statuses:
            continue
        lk = key.lower()
        if lk in merged:
            existing_display, existing_statuses = merged[lk]
            existing_statuses.update(statuses)
            # Prefer mixed-case (canonical) form
            if any(c.isupper() for c in key) and not any(c.isupper() for c in existing_display):
                merged[lk] = (key, existing_statuses)
        else:
            merged[lk] = (key, set(statuses))
    return {
        display_key: sorted(statuses)
        for display_key, statuses in merged.values()
    }


def _find_positions(term: str, text_lower: str, word_bounded: bool) -> List[int]:
    """Return all start positions of `term` within `text_lower`."""
    positions: List[int] = []
    if word_bounded:
        for m in re.finditer(rf'\b{re.escape(term)}\b', text_lower):
            positions.append(m.start())
    else:
        pos = text_lower.find(term)
        while pos != -1:
            positions.append(pos)
            pos = text_lower.find(term, pos + 1)
    return positions


# ─── Ontology term variant expansion at load time ────────────────────────
def _term_variants(term: str) -> List[str]:
    """
    Generate common text variants of an ontology term so it matches real
    paper prose regardless of punctuation style.

    Examples:
      microsatellite_instability_high → [same, "microsatellite instability high",
                                         "microsatellite-instability-high"]
      CPS_≥1 → [same, "CPS ≥1", "CPS >=1", "CPS >= 1", "CPS 1"]
      HER2/neu → [same, "her2 neu", "her2-neu"]
    """
    t = term.strip().lower()
    variants = {t}
    if '_' in t:
        variants.add(t.replace('_', ' '))
        variants.add(t.replace('_', '-'))
    if '≥' in t:
        variants.add(t.replace('≥', '>='))
        variants.add(t.replace('≥', ' '))
        variants.add(t.replace('_≥', ' '))
        variants.add(t.replace('_≥', '≥'))
    if '/' in t and len(t) > 4:
        variants.add(t.replace('/', ' '))
        variants.add(t.replace('/', '-'))
    # Drop variants shorter than 2 chars
    return [v for v in variants if len(v) >= 2]


# ─── Biomarker / imaging / test synonym groups ───────────────────────────
# When ANY variant in a group matches the text, the canonical form is
# added to biomarkers_detected. This handles the "text says
# 'estrogen receptor' but ontology has 'ER'" problem bidirectionally.
BIOMARKER_SYNONYM_GROUPS: List[Dict[str, Any]] = [
    # ── Protein / IHC biomarkers ─────────────────────────────────────
    # NOTE: +/- variants intentionally omitted — the bare form matches via
    # word boundary, and keeping +/-/positive/negative as variants confuses
    # negation detection (ER-negative would count as a positive ER hit
    # via the "er-negative" variant but as negated via bare "er").
    {"canonical": "ER",     "variants": ["er", "estrogen receptor", "oestrogen receptor"]},
    {"canonical": "PR",     "variants": ["pr", "progesterone receptor"]},
    {"canonical": "HER2",   "variants": ["her2", "erbb2", "her2/neu", "her-2", "her 2", "c-erbB-2"]},
    {"canonical": "AR",     "variants": ["androgen receptor"]},
    {"canonical": "Ki-67",  "variants": ["ki-67", "ki67", "mib-1", "proliferation index"]},
    # ── Immunotherapy markers ─────────────────────────────────────────
    {"canonical": "PD-L1",  "variants": ["pd-l1", "pdl1", "pd l1", "pd-l1 expression", "programmed death-ligand 1", "programmed death ligand 1", "programmed death-ligand-1"]},
    {"canonical": "PD-1",   "variants": ["pd-1", "pd1", "programmed death-1", "programmed death 1"]},
    {"canonical": "MSI-H",  "variants": ["msi", "msi-h", "msi high", "msi-high", "microsatellite instability", "microsatellite instability high", "microsatellite-high", "msi positive"]},
    {"canonical": "MSS",    "variants": ["mss", "microsatellite stable", "msi-l", "msi low"]},
    {"canonical": "dMMR",   "variants": ["dmmr", "deficient mismatch repair", "deficient mmr", "mmr deficient", "mmr-deficient", "mmr-d"]},
    {"canonical": "pMMR",   "variants": ["pmmr", "proficient mismatch repair", "proficient mmr", "mmr proficient", "mmr-proficient", "mmr-p"]},
    {"canonical": "TMB-H",  "variants": ["tmb", "tmb-h", "tmb high", "tumor mutational burden", "tumor mutation burden", "high mutational burden"]},
    {"canonical": "HRD",    "variants": ["hrd", "homologous recombination deficiency", "homologous recombination-deficient"]},
    {"canonical": "CPS",    "variants": ["cps", "combined positive score", "cps ≥1", "cps >=1", "cps >= 1", "cps100", "cps 100", "cps≥1"]},
    {"canonical": "TPS",    "variants": ["tps", "tumor proportion score"]},
    # ── Genomic alterations (also populate genomic_alterations slot) ─
    {"canonical": "EGFR",          "variants": ["egfr", "epidermal growth factor receptor"], "slot": "alteration"},
    {"canonical": "EGFR L858R",    "variants": ["l858r", "egfr l858r", "exon 21 l858r"], "slot": "alteration"},
    {"canonical": "EGFR exon 19 del", "variants": ["egfr exon19del", "exon 19 deletion", "exon 19 del", "ex19del"], "slot": "alteration"},
    {"canonical": "EGFR T790M",    "variants": ["t790m", "egfr t790m"], "slot": "alteration"},
    {"canonical": "ALK fusion",    "variants": ["alk", "alk fusion", "alk rearrangement", "eml4-alk", "anaplastic lymphoma kinase"], "slot": "alteration"},
    {"canonical": "ROS1 fusion",   "variants": ["ros1", "ros1 fusion", "ros1 rearrangement"], "slot": "alteration"},
    {"canonical": "RET fusion",    "variants": ["ret fusion", "ret rearrangement"], "slot": "alteration"},
    {"canonical": "NTRK fusion",   "variants": ["ntrk", "ntrk1", "ntrk2", "ntrk3", "ntrk fusion", "ntrk-fusion", "trk fusion"], "slot": "alteration"},
    {"canonical": "BRAF V600E",    "variants": ["braf v600e", "v600e", "braf mutation"], "slot": "alteration"},
    {"canonical": "KRAS G12C",     "variants": ["kras g12c", "g12c"], "slot": "alteration"},
    {"canonical": "KRAS G12D",     "variants": ["kras g12d", "g12d"], "slot": "alteration"},
    {"canonical": "KRAS",          "variants": ["kras", "k-ras"], "slot": "alteration"},
    {"canonical": "MET exon 14",   "variants": ["met exon 14", "met ex14", "metex14", "met exon14"], "slot": "alteration"},
    {"canonical": "HER2 amplification", "variants": ["her2 amplification", "erbb2 amplification", "her2-amplified"], "slot": "alteration"},
    {"canonical": "BRCA",          "variants": ["brca", "brca1", "brca2", "brca1/2", "brca 1", "brca 2", "brca1/brca2", "breast cancer gene"], "slot": "alteration"},
    {"canonical": "MGMT methylation", "variants": ["mgmt", "mgmt promoter methylation", "mgmt methylated", "mgmt-methylated", "mgmt promoter"], "slot": "alteration"},
    {"canonical": "IDH1",           "variants": ["idh1", "idh 1"], "slot": "alteration"},
    {"canonical": "IDH2",           "variants": ["idh2", "idh 2"], "slot": "alteration"},
    {"canonical": "IDH",            "variants": ["idh", "idh mutation", "idh mutated", "idh wild-type", "idh wildtype"], "slot": "alteration"},
    {"canonical": "1p/19q codeletion", "variants": ["1p/19q", "1p 19q", "1p19q codeletion", "1p/19q co-deletion", "1p/19q codeletion", "1p 19q codeletion"], "slot": "alteration"},
    {"canonical": "H3 K27M",        "variants": ["h3 k27m", "h3k27m", "k27m"], "slot": "alteration"},
    {"canonical": "TERT promoter",  "variants": ["tert", "tert promoter", "tert promoter mutation"], "slot": "alteration"},
    {"canonical": "NF2",            "variants": ["nf2", "nf-2"], "slot": "alteration"},
    {"canonical": "NF1",            "variants": ["nf1", "nf-1", "neurofibromin 1"], "slot": "alteration"},
    {"canonical": "Lynch syndrome", "variants": ["lynch syndrome", "lynch-like syndrome"], "slot": "alteration"},
    # ── Sarcoma-specific alterations ─────────────────────────────────
    {"canonical": "MDM2 amplification",   "variants": ["mdm2 amplification", "mdm2 amplified", "mdm2-amplified"], "slot": "alteration"},
    {"canonical": "CDK4 amplification",   "variants": ["cdk4 amplification", "cdk4 amplified"], "slot": "alteration"},
    {"canonical": "EWSR1-FLI1 fusion",    "variants": ["ewsr1-fli1", "ewsr1 fli1", "ewsr1-fli1 fusion", "ews-fli1"], "slot": "alteration"},
    {"canonical": "EWSR1 rearrangement",  "variants": ["ewsr1 rearrangement", "ewsr1 fusion", "ewsr1-"], "slot": "alteration"},
    {"canonical": "SS18-SSX fusion",      "variants": ["ss18-ssx", "ss18 ssx", "syt-ssx", "ss18-ssx1", "ss18-ssx2"], "slot": "alteration"},
    {"canonical": "KIT mutation",         "variants": ["kit mutation", "kit exon 11", "kit exon 9", "kit exon 13", "kit exon 17", "c-kit mutation"], "slot": "alteration"},
    {"canonical": "PDGFRA mutation",      "variants": ["pdgfra", "pdgfra d842v", "pdgfra mutation"], "slot": "alteration"},
    {"canonical": "MYOD1 mutation",       "variants": ["myod1 mutation", "myod1 p.l122r"], "slot": "alteration"},
    {"canonical": "ATRX loss",            "variants": ["atrx loss", "atrx mutation", "atrx-deficient"], "slot": "alteration"},
    {"canonical": "COL1A1-PDGFB fusion",  "variants": ["col1a1-pdgfb", "col1a1 pdgfb fusion"], "slot": "alteration"},
    # ── Prostate-specific alterations ────────────────────────────────
    {"canonical": "TMPRSS2-ERG fusion",   "variants": ["tmprss2-erg", "tmprss2 erg", "tmprss2-erg fusion"], "slot": "alteration"},
    {"canonical": "SPOP mutation",        "variants": ["spop mutation", "spop"], "slot": "alteration"},
    {"canonical": "PTEN loss",            "variants": ["pten loss", "pten deletion", "pten-null", "pten mutation"], "slot": "alteration"},
    {"canonical": "AR-V7",                "variants": ["ar-v7", "ar v7", "androgen receptor splice variant 7"], "slot": "alteration"},
    {"canonical": "AR amplification",     "variants": ["ar amplification", "androgen receptor amplification"], "slot": "alteration"},
    {"canonical": "ATM mutation",         "variants": ["atm mutation", "atm loss", "atm deficient"], "slot": "alteration"},
    {"canonical": "RB1 loss",             "variants": ["rb1 loss", "rb1 mutation", "rb1 deletion", "rb-deficient"], "slot": "alteration"},
    # ── Pediatric-specific alterations ───────────────────────────────
    {"canonical": "MYCN amplification",   "variants": ["mycn amplification", "mycn-amplified", "mycn amplified", "n-myc amplification"], "slot": "alteration"},
    {"canonical": "H3 K27M",              "variants": ["h3 k27m", "h3k27m", "h3-k27m", "h3 k27m-mutant"], "slot": "alteration"},
    {"canonical": "H3 G34",               "variants": ["h3 g34r", "h3 g34v", "h3g34"], "slot": "alteration"},
    {"canonical": "WT1 mutation",         "variants": ["wt1 mutation", "wt1"], "slot": "alteration"},
    {"canonical": "CTNNB1 mutation",      "variants": ["ctnnb1 mutation", "ctnnb1", "beta-catenin mutation"], "slot": "alteration"},
    {"canonical": "ETV6-RUNX1 fusion",    "variants": ["etv6-runx1", "etv6 runx1", "etv6-runx1 fusion", "tel-aml1"], "slot": "alteration"},
    {"canonical": "KMT2A rearrangement",  "variants": ["kmt2a", "kmt2a rearrangement", "mll", "mll rearrangement"], "slot": "alteration"},
    {"canonical": "NPM1 mutation",        "variants": ["npm1 mutation", "npm1", "nucleophosmin"], "slot": "alteration"},
    # ── Lymphoma/leukemia alterations ────────────────────────────────
    {"canonical": "BCL2 translocation",   "variants": ["bcl2 translocation", "bcl-2 translocation", "t(14;18)", "bcl2 rearrangement"], "slot": "alteration"},
    {"canonical": "BCL6 translocation",   "variants": ["bcl6 translocation", "bcl-6 translocation", "bcl6 rearrangement"], "slot": "alteration"},
    {"canonical": "MYC translocation",    "variants": ["myc translocation", "myc rearrangement", "t(8;14)", "c-myc rearrangement"], "slot": "alteration"},
    {"canonical": "Double-hit lymphoma",  "variants": ["double-hit lymphoma", "double hit lymphoma", "triple-hit lymphoma"], "slot": "alteration"},
    {"canonical": "t(11;14)",             "variants": ["t(11;14)", "ccnd1 translocation", "ccnd1 igh"], "slot": "alteration"},
    {"canonical": "MYD88 mutation",       "variants": ["myd88 mutation", "myd88 l265p"], "slot": "alteration"},
    {"canonical": "EZH2 mutation",        "variants": ["ezh2 mutation", "ezh2 y646"], "slot": "alteration"},
    {"canonical": "BCR-ABL",              "variants": ["bcr-abl", "bcr abl", "bcr-abl1", "philadelphia chromosome", "t(9;22)"], "slot": "alteration"},
    {"canonical": "FLT3-ITD",             "variants": ["flt3-itd", "flt3 itd", "flt3 tkd", "flt3 mutation"], "slot": "alteration"},
    {"canonical": "IDH1 mutation",        "variants": ["idh1 mutation", "idh1 r132h", "idh1 r132"], "slot": "alteration"},
    {"canonical": "IDH2 mutation",        "variants": ["idh2 mutation", "idh2 r140", "idh2 r172"], "slot": "alteration"},
    # ── Lung alterations (additions) ─────────────────────────────────
    {"canonical": "KRAS G12V",            "variants": ["kras g12v"], "slot": "alteration"},
    {"canonical": "MET amplification",    "variants": ["met amplification", "met-amplified", "met amplified"], "slot": "alteration"},
    {"canonical": "NRG1 fusion",          "variants": ["nrg1 fusion", "nrg1 rearrangement", "neuregulin 1 fusion"], "slot": "alteration"},
    {"canonical": "BAP1 loss",            "variants": ["bap1 loss", "bap1 mutation", "bap1-deficient"], "slot": "alteration"},
    {"canonical": "EGFR exon 20 insertion", "variants": ["egfr exon 20 insertion", "egfr ex20ins", "egfr exon 20 ins"], "slot": "alteration"},
    # ── H&N alterations ──────────────────────────────────────────────
    {"canonical": "NOTCH1 mutation",      "variants": ["notch1 mutation", "notch1 loss-of-function"], "slot": "alteration"},
    {"canonical": "HRAS mutation",        "variants": ["hras mutation", "h-ras mutation", "hras g12", "hras q61"], "slot": "alteration"},
    {"canonical": "MYB-NFIB fusion",      "variants": ["myb-nfib", "myb nfib fusion"], "slot": "alteration"},
    {"canonical": "PIK3CA mutation",      "variants": ["pik3ca mutation", "pik3ca h1047r", "pik3ca e545k"], "slot": "alteration"},
    # ── Gyn alterations ──────────────────────────────────────────────
    {"canonical": "TP53 mutation",        "variants": ["tp53 mutation", "p53 mutation", "tp53 loss-of-function"], "slot": "alteration"},
    {"canonical": "ARID1A mutation",      "variants": ["arid1a mutation", "arid1a loss", "baf250a loss"], "slot": "alteration"},
    {"canonical": "STK11 mutation",       "variants": ["stk11 mutation", "lkb1 mutation"], "slot": "alteration"},
    {"canonical": "PALB2 mutation",       "variants": ["palb2 mutation", "palb2 germline"], "slot": "alteration"},
    # ── GU alterations ───────────────────────────────────────────────
    {"canonical": "FGFR3 mutation",       "variants": ["fgfr3 mutation", "fgfr3 alteration", "fgfr3 s249c", "fgfr3-tacc3"], "slot": "alteration"},
    {"canonical": "FGFR2 fusion",         "variants": ["fgfr2 fusion", "fgfr2-bicc1"], "slot": "alteration"},
    {"canonical": "TERT promoter mutation", "variants": ["tert promoter mutation", "tert promoter", "tert c228t", "tert c250t"], "slot": "alteration"},
    {"canonical": "VHL mutation",         "variants": ["vhl mutation", "vhl loss", "von hippel-lindau mutation"], "slot": "alteration"},
    {"canonical": "PBRM1 mutation",       "variants": ["pbrm1 mutation", "pbrm1 loss"], "slot": "alteration"},
    {"canonical": "SETD2 mutation",       "variants": ["setd2 mutation", "setd2 loss"], "slot": "alteration"},
    # ── GI alterations ───────────────────────────────────────────────
    {"canonical": "NRAS mutation",        "variants": ["nras mutation", "nras q61", "nras g12"], "slot": "alteration"},
    {"canonical": "CDKN2A loss",          "variants": ["cdkn2a loss", "cdkn2a deletion", "cdkn2a-null", "p16 loss"], "slot": "alteration"},
    {"canonical": "SMAD4 loss",           "variants": ["smad4 loss", "smad4 deletion", "dpc4 loss"], "slot": "alteration"},
    # ── Cutaneous alterations ────────────────────────────────────────
    {"canonical": "NF1 mutation",         "variants": ["nf1 mutation", "nf1 loss", "neurofibromatosis 1 mutation"], "slot": "alteration"},
    {"canonical": "NRAS Q61",             "variants": ["nras q61", "nras q61l", "nras q61r", "nras q61k"], "slot": "alteration"},
    {"canonical": "PTCH1 mutation",       "variants": ["ptch1 mutation", "patched-1 mutation"], "slot": "alteration"},
    {"canonical": "SMO mutation",         "variants": ["smo mutation", "smoothened mutation"], "slot": "alteration"},
    # ── CNS alterations ──────────────────────────────────────────────
    {"canonical": "1p/19q co-deletion",   "variants": ["1p/19q co-deletion", "1p/19q codeletion", "1p 19q codeletion", "1p19q codeletion"], "slot": "alteration"},
    {"canonical": "NF2 mutation",         "variants": ["nf2 mutation", "nf2 loss", "merlin loss"], "slot": "alteration"},
    {"canonical": "EGFR amplification",   "variants": ["egfr amplification", "egfr-amplified", "egfrviii", "egfr vIII"], "slot": "alteration"},
    # ── Breast alterations ───────────────────────────────────────────
    {"canonical": "ESR1 mutation",        "variants": ["esr1 mutation", "esr1 d538g", "esr1 y537", "estrogen receptor mutation"], "slot": "alteration"},
    {"canonical": "CDH1 mutation",        "variants": ["cdh1 mutation", "e-cadherin loss", "e-cadherin mutation"], "slot": "alteration"},
    # ── Benign-associated alterations ────────────────────────────────
    {"canonical": "GNAS mutation",        "variants": ["gnas mutation", "gnas r201"], "slot": "alteration"},
    {"canonical": "MED12 mutation",       "variants": ["med12 mutation"], "slot": "alteration"},
    {"canonical": "APC mutation",         "variants": ["apc mutation", "apc loss", "familial adenomatous polyposis mutation"], "slot": "alteration"},
    # ── Assays / testing platforms ────────────────────────────────────
    {"canonical": "IHC",   "variants": ["ihc", "immunohistochemistry", "immuno-histochemistry", "immunostaining"]},
    {"canonical": "FISH",  "variants": ["fish", "fluorescence in situ hybridization", "fluorescent in situ hybridization"]},
    {"canonical": "ISH",   "variants": ["ish", "in situ hybridization"]},
    {"canonical": "NGS",   "variants": ["ngs", "next generation sequencing", "next-generation sequencing"]},
    {"canonical": "WES",   "variants": ["wes", "whole exome sequencing", "whole-exome sequencing"]},
    # NOTE: bare "pcr" omitted because papers use "pCR" for pathologic
    # complete response, which is NOT the PCR assay. Only long-form matches.
    {"canonical": "PCR",   "variants": ["polymerase chain reaction", "rt-pcr", "qpcr", "quantitative pcr"]},
    # ── Imaging ───────────────────────────────────────────────────────
    {"canonical": "CT",    "variants": ["ct scan", "computed tomography", "ct imaging"], "slot": "imaging"},
    {"canonical": "MRI",   "variants": ["mri", "magnetic resonance imaging", "mri scan"], "slot": "imaging"},
    {"canonical": "PET",   "variants": ["pet scan", "positron emission tomography", "pet-ct", "pet ct", "fdg pet", "fdg-pet", "fdg-pet/ct"], "slot": "imaging"},
    {"canonical": "PSMA-PET", "variants": ["psma pet", "psma-pet", "prostate specific membrane antigen pet", "68ga-psma", "psma-11"], "slot": "imaging"},
    {"canonical": "DOTATATE-PET", "variants": ["dotatate pet", "dotatate-pet", "68ga-dotatate"], "slot": "imaging"},
    {"canonical": "Bone scan", "variants": ["bone scan", "bone scintigraphy", "99mtc-mdp"], "slot": "imaging"},
    {"canonical": "Ultrasound", "variants": ["ultrasound", "sonography", "us imaging"], "slot": "imaging"},
    # ── Serum markers ─────────────────────────────────────────────────
    {"canonical": "PSA",   "variants": ["psa", "prostate specific antigen", "prostate-specific antigen"], "slot": "serum_marker"},
    {"canonical": "CEA",   "variants": ["cea", "carcinoembryonic antigen"], "slot": "serum_marker"},
    {"canonical": "AFP",   "variants": ["afp", "alpha fetoprotein", "alpha-fetoprotein", "α-fetoprotein"], "slot": "serum_marker"},
    {"canonical": "CA 19-9", "variants": ["ca19-9", "ca 19-9", "ca19.9", "carbohydrate antigen 19-9"], "slot": "serum_marker"},
    {"canonical": "CA-125", "variants": ["ca-125", "ca125", "ca 125", "cancer antigen 125"], "slot": "serum_marker"},
    {"canonical": "CA 15-3", "variants": ["ca15-3", "ca 15-3", "cancer antigen 15-3"], "slot": "serum_marker"},
    {"canonical": "HCG",   "variants": ["hcg", "β-hcg", "beta hcg", "beta-hcg", "human chorionic gonadotropin"], "slot": "serum_marker"},
    {"canonical": "LDH",   "variants": ["ldh", "lactate dehydrogenase"], "slot": "serum_marker"},
    {"canonical": "CA 27-29", "variants": ["ca27-29", "ca 27-29", "cancer antigen 27-29", "ca27.29"], "slot": "serum_marker"},
    {"canonical": "CYFRA 21-1", "variants": ["cyfra 21-1", "cyfra21-1", "cytokeratin 19 fragment", "cytokeratin-19 fragment"], "slot": "serum_marker"},
    {"canonical": "Pro-GRP", "variants": ["pro-grp", "progrp", "pro-gastrin releasing peptide", "pro gastrin-releasing peptide", "progastrin-releasing peptide"], "slot": "serum_marker"},
    {"canonical": "NSE",   "variants": ["nse", "neuron specific enolase", "neuron-specific enolase"], "slot": "serum_marker"},
    {"canonical": "Chromogranin A", "variants": ["chromogranin a", "cga", "cgr-a", "chromogranin-a"], "slot": "serum_marker"},
    {"canonical": "Thyroglobulin", "variants": ["thyroglobulin", "tg antibody", "anti-thyroglobulin"], "slot": "serum_marker"},
    {"canonical": "Calcitonin", "variants": ["calcitonin", "serum calcitonin"], "slot": "serum_marker"},
    {"canonical": "HE4",   "variants": ["he4", "human epididymis protein 4"], "slot": "serum_marker"},
    {"canonical": "Inhibin", "variants": ["inhibin a", "inhibin b", "inhibin-a", "inhibin-b"], "slot": "serum_marker"},
    {"canonical": "PTH",   "variants": ["pth", "parathyroid hormone", "serum pth"], "slot": "serum_marker"},
    {"canonical": "Prolactin", "variants": ["prolactin", "serum prolactin"], "slot": "serum_marker"},
    {"canonical": "CRP",   "variants": ["crp", "c-reactive protein"], "slot": "serum_marker"},
    {"canonical": "Beta-2 microglobulin", "variants": ["beta-2 microglobulin", "b2m", "β2-microglobulin", "β2m", "beta2 microglobulin"], "slot": "serum_marker"},
    {"canonical": "N-telopeptide", "variants": ["n-telopeptide", "ntx", "urinary n-telopeptide"], "slot": "serum_marker"},
    {"canonical": "HVA",   "variants": ["hva", "homovanillic acid"], "slot": "serum_marker"},
    {"canonical": "VMA",   "variants": ["vma", "vanillylmandelic acid"], "slot": "serum_marker"},
    {"canonical": "EBV DNA", "variants": ["ebv dna", "ebv-dna", "epstein-barr virus dna", "ebv viral load"], "slot": "serum_marker"},
    {"canonical": "Free PSA", "variants": ["free psa", "fpsa", "%free psa", "free-to-total psa"], "slot": "serum_marker"},
    {"canonical": "PHI",   "variants": ["phi", "prostate health index"], "slot": "serum_marker"},
    {"canonical": "4Kscore", "variants": ["4kscore", "4k score"], "slot": "serum_marker"},
    {"canonical": "PCA3",  "variants": ["pca3", "prostate cancer antigen 3", "pca3 mrna"], "slot": "serum_marker"},
    # ── Additional IHC / histopathologic markers ─────────────────────
    {"canonical": "S100",  "variants": ["s100", "s-100 protein", "s100 protein"]},
    {"canonical": "HMB-45", "variants": ["hmb-45", "hmb45"]},
    {"canonical": "Melan-A", "variants": ["melan-a", "melan a", "mart-1", "mart1"]},
    {"canonical": "CD99",  "variants": ["cd99", "mic2"]},
    {"canonical": "CD20",  "variants": ["cd20", "ms4a1"]},
    {"canonical": "CD19",  "variants": ["cd19"]},
    {"canonical": "CD30",  "variants": ["cd30", "tnfrsf8"]},
    {"canonical": "CD15",  "variants": ["cd15"]},
    {"canonical": "CD22",  "variants": ["cd22"]},
    {"canonical": "CD33",  "variants": ["cd33"]},
    {"canonical": "CD3",   "variants": ["cd3"]},
    {"canonical": "CD5",   "variants": ["cd5"]},
    {"canonical": "CD10",  "variants": ["cd10"]},
    {"canonical": "CD34",  "variants": ["cd34"]},
    {"canonical": "CD38",  "variants": ["cd38"]},
    {"canonical": "CD138", "variants": ["cd138", "syndecan-1"]},
    {"canonical": "GFAP",  "variants": ["gfap", "glial fibrillary acidic protein"]},
    {"canonical": "Synaptophysin", "variants": ["synaptophysin", "syp"]},
    {"canonical": "Chromogranin", "variants": ["chromogranin"]},
    {"canonical": "Vimentin", "variants": ["vimentin"]},
    {"canonical": "Beta-catenin", "variants": ["beta-catenin", "β-catenin", "beta catenin", "nuclear beta-catenin", "ctnnb1 protein"]},
    {"canonical": "Cyclin D1", "variants": ["cyclin d1", "ccnd1 protein", "bcl-1 protein"]},
    {"canonical": "SMA",   "variants": ["sma", "smooth muscle actin", "α-sma", "alpha smooth muscle actin"]},
    {"canonical": "Desmin", "variants": ["desmin"]},
    {"canonical": "Myogenin", "variants": ["myogenin", "myod-related"]},
    {"canonical": "MyoD1", "variants": ["myod1", "myod 1"]},
    {"canonical": "MCPyV", "variants": ["mcpyv", "merkel cell polyomavirus"]},
    {"canonical": "p16",   "variants": ["p16", "p16ink4a", "p16 ihc"]},
    {"canonical": "p53",   "variants": ["p53 ihc", "p53 protein"]},
    # ── Receptor / surface targets ──────────────────────────────────
    {"canonical": "Nectin-4", "variants": ["nectin-4", "nectin 4", "pvrl4"]},
    {"canonical": "Claudin 18.2", "variants": ["claudin 18.2", "claudin18.2", "cldn18.2"]},
    {"canonical": "Folate receptor alpha", "variants": ["folate receptor alpha", "folate receptor α", "frα", "fr alpha", "fra"]},
    {"canonical": "Trop-2", "variants": ["trop-2", "trop2", "trophoblast cell-surface antigen 2"]},
    {"canonical": "GD2",   "variants": ["gd2", "disialoganglioside gd2"]},
    {"canonical": "MAGE-A4", "variants": ["mage-a4", "magea4", "mage a4"]},
    {"canonical": "gp100", "variants": ["gp100", "pmel17"]},
    {"canonical": "BCMA",  "variants": ["bcma", "b-cell maturation antigen", "tnfrsf17"]},
    {"canonical": "SSTR",  "variants": ["sstr", "sstr2", "sstr5", "somatostatin receptor"]},
    {"canonical": "PSMA",  "variants": ["psma", "prostate-specific membrane antigen", "folh1"]},
    # ── Liquid-biopsy / monitoring biomarkers ───────────────────────
    {"canonical": "ctDNA", "variants": ["ctdna", "circulating tumor dna", "cell-free dna", "cfdna", "liquid biopsy"]},
    {"canonical": "CTCs",  "variants": ["ctc", "ctcs", "circulating tumor cells"]},
    {"canonical": "MRD",   "variants": ["mrd", "minimal residual disease", "measurable residual disease"]},
    # ── Pharmacogenomic markers ─────────────────────────────────────
    {"canonical": "DPYD",  "variants": ["dpyd", "dihydropyrimidine dehydrogenase"]},
    {"canonical": "UGT1A1", "variants": ["ugt1a1", "uridine diphosphate glucuronosyltransferase"]},
    # ── HER2-low / HER2-ultralow (for ADC eligibility) ──────────────
    {"canonical": "HER2-low", "variants": ["her2-low", "her2 low", "her2 1+", "her2 2+ ish negative", "her2-low positive"]},
    {"canonical": "HER2-ultralow", "variants": ["her2-ultralow", "her2 ultralow", "her2 0 with membrane staining"]},
]


# ─── Histology synonym groups (→ histologies_detected) ───────────────────
# Matches the "Cancer type and histology" hard filter. Each match tags
# the canonical histology label. These are DIFFERENT from cancer types
# (Lung Cancer) — they're the pathology's histologic classification.
HISTOLOGY_SYNONYM_GROUPS: List[Dict[str, Any]] = [
    {"canonical": "Adenocarcinoma", "variants": ["adenocarcinoma", "adeno", "adenocarcinomatous"]},
    {"canonical": "Squamous cell carcinoma", "variants": ["squamous cell carcinoma", "squamous-cell carcinoma", "scc", "epidermoid carcinoma"]},
    # Bare "small cell" omitted — it's a substring of "non-small cell"
    # and causes SCLC to false-tag on NSCLC text. Long forms only.
    {"canonical": "Small cell carcinoma", "variants": ["small cell carcinoma", "small-cell carcinoma", "sclc", "small cell lung cancer", "small-cell lung cancer"]},
    {"canonical": "Large cell neuroendocrine carcinoma", "variants": ["large cell neuroendocrine carcinoma", "lcnec", "large cell neuroendocrine"]},
    {"canonical": "Non-small cell carcinoma", "variants": ["non-small cell", "non-small-cell", "non small cell", "nsclc"]},
    {"canonical": "Neuroendocrine tumor", "variants": ["neuroendocrine tumor", "neuroendocrine tumour", "neuroendocrine carcinoma", "net", "nec", "carcinoid", "carcinoid tumor"]},
    {"canonical": "Ductal carcinoma (invasive)", "variants": ["invasive ductal carcinoma", "idc", "ductal carcinoma", "infiltrating ductal carcinoma", "invasive breast carcinoma of no special type", "nst carcinoma"]},
    {"canonical": "Lobular carcinoma (invasive)", "variants": ["invasive lobular carcinoma", "ilc", "lobular carcinoma", "infiltrating lobular carcinoma"]},
    {"canonical": "DCIS", "variants": ["ductal carcinoma in situ", "dcis"]},
    {"canonical": "LCIS", "variants": ["lobular carcinoma in situ", "lcis"]},
    {"canonical": "Mucinous carcinoma", "variants": ["mucinous carcinoma", "mucinous adenocarcinoma", "colloid carcinoma"]},
    {"canonical": "Signet ring cell carcinoma", "variants": ["signet ring cell carcinoma", "signet-ring cell carcinoma", "signet ring"]},
    {"canonical": "Medullary carcinoma", "variants": ["medullary carcinoma"]},
    {"canonical": "Papillary carcinoma", "variants": ["papillary carcinoma", "papillary adenocarcinoma"]},
    {"canonical": "Follicular carcinoma", "variants": ["follicular carcinoma", "follicular thyroid carcinoma"]},
    {"canonical": "Anaplastic carcinoma", "variants": ["anaplastic carcinoma", "undifferentiated carcinoma"]},
    {"canonical": "Hurthle cell carcinoma", "variants": ["hurthle cell carcinoma", "hürthle cell carcinoma", "oncocytic carcinoma"]},
    {"canonical": "Clear cell carcinoma", "variants": ["clear cell carcinoma", "clear cell renal cell carcinoma", "ccRCC"]},
    {"canonical": "Renal cell carcinoma", "variants": ["renal cell carcinoma", "rcc", "renal carcinoma"]},
    {"canonical": "Urothelial carcinoma", "variants": ["urothelial carcinoma", "transitional cell carcinoma", "tcc"]},
    {"canonical": "Hepatocellular carcinoma", "variants": ["hepatocellular carcinoma", "hcc", "hepatoma"]},
    {"canonical": "Cholangiocarcinoma", "variants": ["cholangiocarcinoma", "bile duct cancer", "intrahepatic cholangiocarcinoma", "extrahepatic cholangiocarcinoma", "icc"]},
    {"canonical": "Pancreatic ductal adenocarcinoma", "variants": ["pancreatic ductal adenocarcinoma", "pdac"]},
    {"canonical": "Glioblastoma", "variants": ["glioblastoma", "gbm", "glioblastoma multiforme"]},
    {"canonical": "Glioma", "variants": ["glioma", "low-grade glioma", "lgg", "high-grade glioma", "hgg"]},
    {"canonical": "Astrocytoma", "variants": ["astrocytoma", "anaplastic astrocytoma", "pilocytic astrocytoma"]},
    {"canonical": "Oligodendroglioma", "variants": ["oligodendroglioma", "anaplastic oligodendroglioma"]},
    {"canonical": "Meningioma", "variants": ["meningioma"]},
    {"canonical": "Melanoma", "variants": ["melanoma", "malignant melanoma", "cutaneous melanoma"]},
    {"canonical": "Sarcoma", "variants": ["sarcoma", "soft tissue sarcoma", "sts", "bone sarcoma"]},
    {"canonical": "Leiomyosarcoma", "variants": ["leiomyosarcoma", "lms"]},
    {"canonical": "Liposarcoma", "variants": ["liposarcoma", "dedifferentiated liposarcoma", "ddls", "myxoid liposarcoma"]},
    {"canonical": "Osteosarcoma", "variants": ["osteosarcoma", "osteogenic sarcoma"]},
    {"canonical": "Ewing sarcoma", "variants": ["ewing sarcoma", "ewing's sarcoma"]},
    {"canonical": "Rhabdomyosarcoma", "variants": ["rhabdomyosarcoma", "rms"]},
    {"canonical": "GIST", "variants": ["gastrointestinal stromal tumor", "gist"]},
    {"canonical": "Hodgkin lymphoma", "variants": ["hodgkin lymphoma", "hodgkin's lymphoma", "hl", "classical hodgkin lymphoma", "chl"]},
    {"canonical": "Diffuse large B-cell lymphoma", "variants": ["diffuse large b-cell lymphoma", "dlbcl"]},
    {"canonical": "Follicular lymphoma", "variants": ["follicular lymphoma", "fl"]},
    {"canonical": "Mantle cell lymphoma", "variants": ["mantle cell lymphoma", "mcl"]},
    {"canonical": "Marginal zone lymphoma", "variants": ["marginal zone lymphoma", "mzl", "malt lymphoma"]},
    {"canonical": "Multiple myeloma", "variants": ["multiple myeloma", "mm", "plasma cell myeloma"]},
    {"canonical": "Acute lymphoblastic leukemia", "variants": ["acute lymphoblastic leukemia", "acute lymphoblastic leukaemia"]},
    {"canonical": "Acute myeloid leukemia", "variants": ["acute myeloid leukemia", "acute myeloid leukaemia", "aml"]},
    {"canonical": "Chronic lymphocytic leukemia", "variants": ["chronic lymphocytic leukemia", "chronic lymphocytic leukaemia", "cll"]},
    {"canonical": "Chronic myeloid leukemia", "variants": ["chronic myeloid leukemia", "chronic myeloid leukaemia", "cml"]},
    {"canonical": "Basal cell carcinoma", "variants": ["basal cell carcinoma", "bcc"]},
    {"canonical": "Merkel cell carcinoma", "variants": ["merkel cell carcinoma", "mcc"]},
    # ── Breast-specific variants ─────────────────────────────────────
    {"canonical": "Inflammatory breast cancer", "variants": ["inflammatory breast cancer", "inflammatory breast carcinoma", "ibc"]},
    {"canonical": "Phyllodes tumor", "variants": ["phyllodes tumor", "phyllodes tumour", "cystosarcoma phyllodes"]},
    {"canonical": "Paget disease of the breast", "variants": ["paget disease of the breast", "paget's disease of the breast", "mammary paget"]},
    {"canonical": "Tubular carcinoma", "variants": ["tubular carcinoma"]},
    {"canonical": "Metaplastic breast carcinoma", "variants": ["metaplastic breast carcinoma", "metaplastic carcinoma of the breast"]},
    # ── Lung-specific variants ───────────────────────────────────────
    {"canonical": "Large cell carcinoma", "variants": ["large cell carcinoma", "large-cell carcinoma"]},
    {"canonical": "Mesothelioma", "variants": ["mesothelioma", "pleural mesothelioma", "peritoneal mesothelioma", "malignant mesothelioma"]},
    # ── Prostate-specific variants ───────────────────────────────────
    {"canonical": "Acinar prostate adenocarcinoma", "variants": ["acinar adenocarcinoma", "acinar prostate adenocarcinoma"]},
    {"canonical": "Ductal prostate adenocarcinoma", "variants": ["ductal prostate adenocarcinoma", "ductal adenocarcinoma of the prostate"]},
    {"canonical": "Neuroendocrine prostate carcinoma", "variants": ["neuroendocrine prostate cancer", "nepc", "small cell prostate carcinoma"]},
    # ── H&N specific variants ────────────────────────────────────────
    {"canonical": "Mucoepidermoid carcinoma", "variants": ["mucoepidermoid carcinoma", "mec"]},
    {"canonical": "Adenoid cystic carcinoma", "variants": ["adenoid cystic carcinoma", "acc"]},
    {"canonical": "Papillary thyroid carcinoma", "variants": ["papillary thyroid carcinoma", "ptc"]},
    {"canonical": "Follicular thyroid carcinoma", "variants": ["follicular thyroid carcinoma", "ftc"]},
    {"canonical": "Medullary thyroid carcinoma", "variants": ["medullary thyroid carcinoma", "mtc"]},
    {"canonical": "Anaplastic thyroid carcinoma", "variants": ["anaplastic thyroid carcinoma", "atc"]},
    {"canonical": "Nasopharyngeal carcinoma", "variants": ["nasopharyngeal carcinoma", "npc"]},
    # ── GI / Urothelial specific variants ────────────────────────────
    {"canonical": "Anal squamous cell carcinoma", "variants": ["anal squamous cell carcinoma", "anal scc"]},
    {"canonical": "Chromophobe RCC", "variants": ["chromophobe renal cell carcinoma", "chromophobe rcc", "chromophobe"]},
    {"canonical": "Papillary RCC", "variants": ["papillary renal cell carcinoma", "papillary rcc", "type 1 papillary rcc", "type 2 papillary rcc"]},
    {"canonical": "Collecting duct carcinoma", "variants": ["collecting duct carcinoma", "bellini duct carcinoma"]},
    {"canonical": "Seminoma", "variants": ["seminoma"]},
    {"canonical": "Non-seminomatous germ cell tumor", "variants": ["nsgct", "non-seminomatous germ cell tumor", "nonseminoma", "mixed germ cell tumor"]},
    {"canonical": "Embryonal carcinoma", "variants": ["embryonal carcinoma"]},
    {"canonical": "Yolk sac tumor", "variants": ["yolk sac tumor", "yolk sac tumour", "endodermal sinus tumor"]},
    {"canonical": "Choriocarcinoma", "variants": ["choriocarcinoma"]},
    {"canonical": "Teratoma", "variants": ["teratoma", "mature teratoma", "immature teratoma"]},
    # ── Gyn specific variants ────────────────────────────────────────
    {"canonical": "High-grade serous carcinoma", "variants": ["high-grade serous carcinoma", "hgsc", "high grade serous", "serous ovarian carcinoma"]},
    {"canonical": "Low-grade serous carcinoma", "variants": ["low-grade serous carcinoma", "lgsc", "low grade serous"]},
    {"canonical": "Endometrioid carcinoma", "variants": ["endometrioid carcinoma", "endometrioid adenocarcinoma"]},
    {"canonical": "Cervical squamous cell carcinoma", "variants": ["cervical squamous cell carcinoma", "squamous cell carcinoma of the cervix"]},
    {"canonical": "Germ cell tumor", "variants": ["germ cell tumor", "germ cell tumour", "gct"]},
    {"canonical": "Granulosa cell tumor", "variants": ["granulosa cell tumor", "granulosa cell tumour"]},
    # ── CNS specific variants ────────────────────────────────────────
    {"canonical": "Ependymoma", "variants": ["ependymoma", "anaplastic ependymoma"]},
    {"canonical": "Medulloblastoma", "variants": ["medulloblastoma"]},
    {"canonical": "Diffuse midline glioma", "variants": ["diffuse midline glioma", "dmg", "h3 k27m-altered"]},
    {"canonical": "DIPG", "variants": ["dipg", "diffuse intrinsic pontine glioma"]},
    # ── Lymphoma specific variants ───────────────────────────────────
    {"canonical": "Burkitt lymphoma", "variants": ["burkitt lymphoma", "burkitt's lymphoma", "bl"]},
    {"canonical": "MALT lymphoma", "variants": ["malt lymphoma", "mucosa-associated lymphoid tissue lymphoma", "extranodal marginal zone lymphoma"]},
    {"canonical": "Nodular lymphocyte-predominant Hodgkin", "variants": ["nodular lymphocyte-predominant hodgkin", "nlphl"]},
    {"canonical": "Classical Hodgkin lymphoma", "variants": ["classical hodgkin lymphoma", "chl"]},
    {"canonical": "Mycosis fungoides", "variants": ["mycosis fungoides", "mf", "cutaneous t-cell lymphoma", "ctcl"]},
    {"canonical": "Sezary syndrome", "variants": ["sezary syndrome", "sézary syndrome"]},
    {"canonical": "Waldenstrom macroglobulinemia", "variants": ["waldenstrom macroglobulinemia", "waldenström macroglobulinaemia", "lymphoplasmacytic lymphoma", "lpl"]},
    {"canonical": "T-cell lymphoma", "variants": ["peripheral t-cell lymphoma", "ptcl", "anaplastic large cell lymphoma", "alcl"]},
    # ── Sarcoma specific variants ────────────────────────────────────
    {"canonical": "Chondrosarcoma", "variants": ["chondrosarcoma"]},
    {"canonical": "Angiosarcoma", "variants": ["angiosarcoma"]},
    {"canonical": "Synovial sarcoma", "variants": ["synovial sarcoma"]},
    {"canonical": "MPNST", "variants": ["mpnst", "malignant peripheral nerve sheath tumor", "malignant peripheral nerve sheath tumour"]},
    {"canonical": "Undifferentiated pleomorphic sarcoma", "variants": ["undifferentiated pleomorphic sarcoma", "ups", "malignant fibrous histiocytoma", "mfh"]},
    {"canonical": "Myxofibrosarcoma", "variants": ["myxofibrosarcoma"]},
    {"canonical": "Epithelioid sarcoma", "variants": ["epithelioid sarcoma"]},
    {"canonical": "Alveolar soft part sarcoma", "variants": ["alveolar soft part sarcoma", "aspscr1"]},
    {"canonical": "Fibrosarcoma", "variants": ["fibrosarcoma"]},
    {"canonical": "Pleomorphic rhabdomyosarcoma", "variants": ["pleomorphic rhabdomyosarcoma"]},
    {"canonical": "Alveolar rhabdomyosarcoma", "variants": ["alveolar rhabdomyosarcoma", "ARMS"]},
    {"canonical": "Embryonal rhabdomyosarcoma", "variants": ["embryonal rhabdomyosarcoma", "ERMS"]},
    # ── Cutaneous specific variants ──────────────────────────────────
    {"canonical": "DFSP", "variants": ["dfsp", "dermatofibrosarcoma protuberans"]},
    {"canonical": "Sebaceous carcinoma", "variants": ["sebaceous carcinoma"]},
    {"canonical": "Kaposi sarcoma", "variants": ["kaposi sarcoma", "kaposi's sarcoma"]},
    # ── Pediatric specific variants ──────────────────────────────────
    {"canonical": "Wilms tumor", "variants": ["wilms tumor", "wilms' tumor", "wilms tumour", "nephroblastoma"]},
    {"canonical": "Neuroblastoma", "variants": ["neuroblastoma"]},
    {"canonical": "Retinoblastoma", "variants": ["retinoblastoma"]},
    {"canonical": "Hepatoblastoma", "variants": ["hepatoblastoma"]},
    # ── Benign tumors ────────────────────────────────────────────────
    {"canonical": "Nevus", "variants": ["nevus", "benign nevus", "melanocytic nevus"]},
    {"canonical": "Fibroadenoma", "variants": ["fibroadenoma"]},
    {"canonical": "Leiomyoma", "variants": ["leiomyoma", "uterine fibroid", "fibroid"]},
    {"canonical": "Tubular adenoma", "variants": ["tubular adenoma", "colonic tubular adenoma"]},
    {"canonical": "Osteoma", "variants": ["osteoma"]},
    {"canonical": "Lipoma", "variants": ["lipoma"]},
    {"canonical": "Parathyroid adenoma", "variants": ["parathyroid adenoma"]},
    {"canonical": "Pituitary adenoma", "variants": ["pituitary adenoma", "prolactinoma"]},
    {"canonical": "Fibrous dysplasia", "variants": ["fibrous dysplasia"]},
]


# ─── Disease status synonym groups (→ disease_status_detected) ───────────
# Matches the "Tumor stage or disease status" hard filter.
DISEASE_STATUS_SYNONYM_GROUPS: List[Dict[str, Any]] = [
    {"canonical": "Metastatic",             "variants": ["metastatic", "metastases", "metastasis", "distant disease", "m1 disease", "stage iv"]},
    {"canonical": "De novo metastatic",     "variants": ["de novo metastatic", "newly diagnosed metastatic", "presenting with metastatic disease", "initial metastatic"]},
    {"canonical": "Oligometastatic",        "variants": ["oligometastatic", "oligomet", "limited metastatic", "oligo-recurrent", "oligoprogressive"]},
    {"canonical": "Locally advanced",       "variants": ["locally advanced", "locoregionally advanced", "locally extensive", "borderline resectable"]},
    {"canonical": "Unresectable",           "variants": ["unresectable", "inoperable", "non-resectable", "surgically unresectable", "not amenable to surgery"]},
    {"canonical": "Recurrent",              "variants": ["recurrent", "recurrence", "relapsed", "relapse", "disease recurrence", "locoregional recurrence", "recurrent/metastatic", "r/m"]},
    {"canonical": "Refractory",             "variants": ["refractory", "treatment-resistant", "non-responsive", "resistant disease"]},
    {"canonical": "Progressive disease",    "variants": ["progressive disease", "disease progression", "progression on", "progressing on", "radiographic progression"]},
    {"canonical": "Early stage",            "variants": ["early stage", "early-stage", "localised", "localized"]},
    {"canonical": "Advanced",               "variants": ["advanced disease", "late-stage"]},
    {"canonical": "Castration-resistant",   "variants": ["castration-resistant", "castration resistant", "crpc", "mcrpc"]},
    {"canonical": "Castration-sensitive",   "variants": ["castration-sensitive", "castration sensitive", "cspc", "mcspc", "hormone-sensitive prostate cancer", "hormone sensitive prostate cancer"]},
    {"canonical": "Hormone receptor positive", "variants": ["hormone receptor positive", "hormone-receptor positive", "hr+", "hr positive"]},
    {"canonical": "Triple negative",        "variants": ["triple negative", "triple-negative", "tnbc"]},
    {"canonical": "ICI-refractory",         "variants": ["ici-refractory", "ici refractory", "checkpoint inhibitor refractory", "post-ici progression", "post-immunotherapy progression", "anti-pd1 failure", "anti-pd-1 failure"]},
    {"canonical": "Platinum-refractory",    "variants": ["platinum-refractory", "platinum refractory", "platinum-resistant", "platinum resistant"]},
    {"canonical": "Platinum-sensitive",     "variants": ["platinum-sensitive", "platinum sensitive"]},
]


# ─── Treatment line / setting synonym groups (→ treatment_lines_detected) ─
# Matches the "Prior therapies and treatment lines" hard filter.
TREATMENT_LINE_SYNONYM_GROUPS: List[Dict[str, Any]] = [
    {"canonical": "First-line",        "variants": ["first-line", "first line", "1l", "1st-line", "1st line", "frontline", "front-line", "front line", "initial treatment", "treatment-naive", "treatment naive", "newly diagnosed"]},
    {"canonical": "Second-line",       "variants": ["second-line", "second line", "2l", "2nd-line", "2nd line", "post-progression", "post progression"]},
    {"canonical": "Third-line",        "variants": ["third-line", "third line", "3l", "3rd-line", "3rd line"]},
    {"canonical": "Later-line",        "variants": ["later-line", "later line", "fourth-line", "4l", "subsequent line"]},
    # NOTE: "induction" omitted here — in leukemia it means induction
    # chemotherapy (a distinct concept); kept separately in the "Induction"
    # canonical below.
    {"canonical": "Neoadjuvant",       "variants": ["neoadjuvant", "preoperative", "pre-operative", "pre operative", "pre-surgical", "before surgery"]},
    {"canonical": "Adjuvant",          "variants": ["adjuvant", "postoperative", "post-operative", "post operative", "post-surgical", "after surgery"]},
    {"canonical": "Perioperative",     "variants": ["perioperative"]},
    {"canonical": "Consolidation",     "variants": ["consolidation", "consolidation therapy"]},
    {"canonical": "Maintenance",       "variants": ["maintenance", "maintenance therapy"]},
    {"canonical": "Induction",         "variants": ["induction", "induction chemotherapy", "induction therapy"]},
    {"canonical": "Salvage",           "variants": ["salvage", "salvage therapy", "re-treatment", "retreatment", "rescue therapy"]},
    {"canonical": "Palliative",        "variants": ["palliative", "palliative intent", "best supportive care", "bsc", "symptom control"]},
    {"canonical": "Definitive",        "variants": ["definitive", "curative", "curative-intent", "curative intent", "radical"]},
    {"canonical": "Post-ICI",          "variants": ["post-ici", "after immunotherapy", "post-immunotherapy", "after checkpoint inhibitor"]},
]


# ─── Targeted therapy drugs (→ drugs_detected) ───────────────────────────
# FDA-approved / clinically relevant targeted therapies grouped by canonical
# generic name with brand names and ASCO/FDA aliases. Populated into the
# same drugs_detected slot the cancer ontology feeds.
DRUG_SYNONYM_GROUPS: List[Dict[str, Any]] = [
    # ── Sarcoma / GIST / connective ────────────────────────────────
    {"canonical": "Avapritinib",        "variants": ["avapritinib", "ayvakit"], "slot": "drug"},
    {"canonical": "Ripretinib",         "variants": ["ripretinib", "qinlock"], "slot": "drug"},
    {"canonical": "Imatinib",           "variants": ["imatinib", "gleevec", "glivec"], "slot": "drug"},
    {"canonical": "Sunitinib",          "variants": ["sunitinib", "sutent"], "slot": "drug"},
    {"canonical": "Pazopanib",          "variants": ["pazopanib", "votrient"], "slot": "drug"},
    {"canonical": "Nirogacestat",       "variants": ["nirogacestat", "ogsiveo"], "slot": "drug"},
    {"canonical": "Afamitresgene autoleucel", "variants": ["afamitresgene autoleucel", "afami-cel", "tecelra"], "slot": "drug"},
    # ── Radiopharmaceuticals ────────────────────────────────────────
    {"canonical": "Lutetium Lu 177 vipivotide tetraxetan", "variants": ["lutetium lu 177 vipivotide tetraxetan", "pluvicto", "177lu-psma-617", "lu-177 psma", "lu177-psma-617"], "slot": "drug"},
    {"canonical": "Lutetium Lu 177 dotatate", "variants": ["lutetium lu 177 dotatate", "lutathera", "177lu-dotatate", "lu-177 dotatate"], "slot": "drug"},
    {"canonical": "Iodine-131",         "variants": ["iodine-131", "131i", "i-131", "radioactive iodine", "rai"], "slot": "drug"},
    {"canonical": "Radium-223",         "variants": ["radium-223", "ra-223", "223ra", "xofigo"], "slot": "drug"},
    {"canonical": "PNT2003",            "variants": ["pnt2003"], "slot": "drug"},
    # ── Prostate ────────────────────────────────────────────────────
    {"canonical": "Olaparib",           "variants": ["olaparib", "lynparza"], "slot": "drug"},
    {"canonical": "Niraparib",          "variants": ["niraparib", "zejula"], "slot": "drug"},
    {"canonical": "Rucaparib",          "variants": ["rucaparib", "rubraca"], "slot": "drug"},
    {"canonical": "Talazoparib",        "variants": ["talazoparib", "talzenna"], "slot": "drug"},
    {"canonical": "Darolutamide",       "variants": ["darolutamide", "nubeqa"], "slot": "drug"},
    {"canonical": "Enzalutamide",       "variants": ["enzalutamide", "xtandi"], "slot": "drug"},
    {"canonical": "Apalutamide",        "variants": ["apalutamide", "erleada"], "slot": "drug"},
    {"canonical": "Abiraterone",        "variants": ["abiraterone", "zytiga", "yonsa"], "slot": "drug"},
    {"canonical": "Niraparib + Abiraterone", "variants": ["akeega", "niraparib plus abiraterone", "niraparib and abiraterone"], "slot": "drug"},
    # ── Pediatric ───────────────────────────────────────────────────
    {"canonical": "Revumenib",          "variants": ["revumenib", "revuforj"], "slot": "drug"},
    {"canonical": "Larotrectinib",      "variants": ["larotrectinib", "vitrakvi", "loxo-101"], "slot": "drug"},
    {"canonical": "Entrectinib",        "variants": ["entrectinib", "rozlytrek"], "slot": "drug"},
    {"canonical": "Crizotinib",         "variants": ["crizotinib", "xalkori"], "slot": "drug"},
    {"canonical": "Dinutuximab",        "variants": ["dinutuximab", "unituxin", "ch14.18"], "slot": "drug"},
    {"canonical": "Selumetinib",        "variants": ["selumetinib", "koselugo"], "slot": "drug"},
    # ── Lymphoma / leukemia / myeloma ───────────────────────────────
    {"canonical": "Acalabrutinib",      "variants": ["acalabrutinib", "calquence"], "slot": "drug"},
    {"canonical": "Zanubrutinib",       "variants": ["zanubrutinib", "brukinsa"], "slot": "drug"},
    {"canonical": "Ibrutinib",          "variants": ["ibrutinib", "imbruvica"], "slot": "drug"},
    {"canonical": "Venetoclax",         "variants": ["venetoclax", "venclexta"], "slot": "drug"},
    {"canonical": "Epcoritamab",        "variants": ["epcoritamab", "epkinly"], "slot": "drug"},
    {"canonical": "Glofitamab",         "variants": ["glofitamab", "columvi"], "slot": "drug"},
    {"canonical": "Mosunetuzumab",      "variants": ["mosunetuzumab", "lunsumio"], "slot": "drug"},
    {"canonical": "Tazemetostat",       "variants": ["tazemetostat", "tazverik"], "slot": "drug"},
    {"canonical": "Tafasitamab",        "variants": ["tafasitamab", "monjuvi"], "slot": "drug"},
    {"canonical": "Polatuzumab vedotin", "variants": ["polatuzumab vedotin", "polivy"], "slot": "drug"},
    {"canonical": "Brentuximab vedotin", "variants": ["brentuximab vedotin", "adcetris"], "slot": "drug"},
    {"canonical": "Rituximab",          "variants": ["rituximab", "rituxan"], "slot": "drug"},
    {"canonical": "Obinutuzumab",       "variants": ["obinutuzumab", "gazyva"], "slot": "drug"},
    # ── Lung (NSCLC / SCLC) ─────────────────────────────────────────
    {"canonical": "Osimertinib",        "variants": ["osimertinib", "tagrisso"], "slot": "drug"},
    {"canonical": "Erlotinib",          "variants": ["erlotinib", "tarceva"], "slot": "drug"},
    {"canonical": "Gefitinib",          "variants": ["gefitinib", "iressa"], "slot": "drug"},
    {"canonical": "Afatinib",           "variants": ["afatinib", "gilotrif"], "slot": "drug"},
    {"canonical": "Dacomitinib",        "variants": ["dacomitinib", "vizimpro"], "slot": "drug"},
    {"canonical": "Amivantamab",        "variants": ["amivantamab", "rybrevant"], "slot": "drug"},
    {"canonical": "Sotorasib",          "variants": ["sotorasib", "lumakras"], "slot": "drug"},
    {"canonical": "Adagrasib",          "variants": ["adagrasib", "krazati"], "slot": "drug"},
    {"canonical": "Zongertinib",        "variants": ["zongertinib"], "slot": "drug"},
    {"canonical": "Taletrectinib",      "variants": ["taletrectinib", "ibtrozi"], "slot": "drug"},
    {"canonical": "Alectinib",          "variants": ["alectinib", "alecensa"], "slot": "drug"},
    {"canonical": "Ensartinib",         "variants": ["ensartinib", "ensacove"], "slot": "drug"},
    {"canonical": "Brigatinib",         "variants": ["brigatinib", "alunbrig"], "slot": "drug"},
    {"canonical": "Lorlatinib",         "variants": ["lorlatinib", "lorbrena"], "slot": "drug"},
    {"canonical": "Selpercatinib",      "variants": ["selpercatinib", "retevmo"], "slot": "drug"},
    {"canonical": "Pralsetinib",        "variants": ["pralsetinib", "gavreto"], "slot": "drug"},
    {"canonical": "Capmatinib",         "variants": ["capmatinib", "tabrecta"], "slot": "drug"},
    {"canonical": "Tepotinib",          "variants": ["tepotinib", "tepmetko"], "slot": "drug"},
    {"canonical": "Trastuzumab deruxtecan", "variants": ["trastuzumab deruxtecan", "enhertu", "t-dxd", "tdxd", "ds-8201"], "slot": "drug"},
    # ── H&N ─────────────────────────────────────────────────────────
    {"canonical": "Cetuximab",          "variants": ["cetuximab", "erbitux"], "slot": "drug"},
    {"canonical": "Pembrolizumab",      "variants": ["pembrolizumab", "keytruda", "mk-3475"], "slot": "drug"},
    {"canonical": "Nivolumab",          "variants": ["nivolumab", "opdivo", "bms-936558"], "slot": "drug"},
    {"canonical": "Tipifarnib",         "variants": ["tipifarnib", "zarnestra"], "slot": "drug"},
    # ── Gyn ─────────────────────────────────────────────────────────
    {"canonical": "Mirvetuximab soravtansine", "variants": ["mirvetuximab soravtansine", "elahere", "mirv"], "slot": "drug"},
    {"canonical": "Bevacizumab",        "variants": ["bevacizumab", "avastin"], "slot": "drug"},
    {"canonical": "Relacorilant",       "variants": ["relacorilant"], "slot": "drug"},
    # ── GU ──────────────────────────────────────────────────────────
    {"canonical": "Erdafitinib",        "variants": ["erdafitinib", "balversa"], "slot": "drug"},
    {"canonical": "Enfortumab vedotin", "variants": ["enfortumab vedotin", "padcev"], "slot": "drug"},
    {"canonical": "Belzutifan",         "variants": ["belzutifan", "welireg"], "slot": "drug"},
    {"canonical": "Cabozantinib",       "variants": ["cabozantinib", "cabometyx", "cometriq"], "slot": "drug"},
    {"canonical": "Lenvatinib",         "variants": ["lenvatinib", "lenvima"], "slot": "drug"},
    {"canonical": "Sacituzumab govitecan", "variants": ["sacituzumab govitecan", "trodelvy"], "slot": "drug"},
    {"canonical": "Atezolizumab",       "variants": ["atezolizumab", "tecentriq"], "slot": "drug"},
    {"canonical": "Durvalumab",         "variants": ["durvalumab", "imfinzi"], "slot": "drug"},
    {"canonical": "Avelumab",           "variants": ["avelumab", "bavencio"], "slot": "drug"},
    # ── GI ──────────────────────────────────────────────────────────
    {"canonical": "Encorafenib",        "variants": ["encorafenib", "braftovi"], "slot": "drug"},
    {"canonical": "Panitumumab",        "variants": ["panitumumab", "vectibix"], "slot": "drug"},
    {"canonical": "Zolbetuximab",       "variants": ["zolbetuximab", "vyloy"], "slot": "drug"},
    {"canonical": "Trastuzumab",        "variants": ["trastuzumab", "herceptin"], "slot": "drug"},
    {"canonical": "Pertuzumab",         "variants": ["pertuzumab", "perjeta"], "slot": "drug"},
    {"canonical": "Fruquintinib",       "variants": ["fruquintinib", "fruzaqla"], "slot": "drug"},
    {"canonical": "Regorafenib",        "variants": ["regorafenib", "stivarga"], "slot": "drug"},
    # ── Cutaneous / Melanoma ────────────────────────────────────────
    {"canonical": "Dabrafenib",         "variants": ["dabrafenib", "tafinlar"], "slot": "drug"},
    {"canonical": "Trametinib",         "variants": ["trametinib", "mekinist"], "slot": "drug"},
    {"canonical": "Vemurafenib",        "variants": ["vemurafenib", "zelboraf"], "slot": "drug"},
    {"canonical": "Cobimetinib",        "variants": ["cobimetinib", "cotellic"], "slot": "drug"},
    {"canonical": "Vismodegib",         "variants": ["vismodegib", "erivedge"], "slot": "drug"},
    {"canonical": "Sonidegib",          "variants": ["sonidegib", "odomzo"], "slot": "drug"},
    {"canonical": "Cemiplimab",         "variants": ["cemiplimab", "libtayo"], "slot": "drug"},
    {"canonical": "Cosibelimab",        "variants": ["cosibelimab", "unloxcyt"], "slot": "drug"},
    {"canonical": "Tebentafusp",        "variants": ["tebentafusp", "kimmtrak"], "slot": "drug"},
    {"canonical": "Ipilimumab",         "variants": ["ipilimumab", "yervoy"], "slot": "drug"},
    # ── CNS ─────────────────────────────────────────────────────────
    {"canonical": "Vorasidenib",        "variants": ["vorasidenib", "voranigo"], "slot": "drug"},
    {"canonical": "Tucatinib",          "variants": ["tucatinib", "tukysa"], "slot": "drug"},
    {"canonical": "Temozolomide",       "variants": ["temozolomide", "temodar"], "slot": "drug"},
    # ── Breast ──────────────────────────────────────────────────────
    {"canonical": "Alpelisib",          "variants": ["alpelisib", "piqray"], "slot": "drug"},
    {"canonical": "Inavolisib",         "variants": ["inavolisib", "itovebi"], "slot": "drug"},
    {"canonical": "Elacestrant",        "variants": ["elacestrant", "orserdu"], "slot": "drug"},
    {"canonical": "Capivasertib",       "variants": ["capivasertib", "truqap"], "slot": "drug"},
    {"canonical": "Palbociclib",        "variants": ["palbociclib", "ibrance"], "slot": "drug"},
    {"canonical": "Ribociclib",         "variants": ["ribociclib", "kisqali"], "slot": "drug"},
    {"canonical": "Abemaciclib",        "variants": ["abemaciclib", "verzenio"], "slot": "drug"},
    {"canonical": "Letrozole",          "variants": ["letrozole", "femara"], "slot": "drug"},
    {"canonical": "Anastrozole",        "variants": ["anastrozole", "arimidex"], "slot": "drug"},
    {"canonical": "Exemestane",         "variants": ["exemestane", "aromasin"], "slot": "drug"},
    {"canonical": "Tamoxifen",          "variants": ["tamoxifen", "nolvadex"], "slot": "drug"},
    {"canonical": "Fulvestrant",        "variants": ["fulvestrant", "faslodex"], "slot": "drug"},
    # ── Cytotoxic chemotherapy backbones (common across cancers) ─
    {"canonical": "Cisplatin",          "variants": ["cisplatin"], "slot": "drug"},
    {"canonical": "Carboplatin",        "variants": ["carboplatin"], "slot": "drug"},
    {"canonical": "Oxaliplatin",        "variants": ["oxaliplatin"], "slot": "drug"},
    {"canonical": "Paclitaxel",         "variants": ["paclitaxel", "taxol"], "slot": "drug"},
    {"canonical": "Docetaxel",          "variants": ["docetaxel", "taxotere"], "slot": "drug"},
    {"canonical": "Nab-paclitaxel",     "variants": ["nab-paclitaxel", "abraxane"], "slot": "drug"},
    {"canonical": "Gemcitabine",        "variants": ["gemcitabine"], "slot": "drug"},
    {"canonical": "5-Fluorouracil",     "variants": ["5-fluorouracil", "5-fu", "fluorouracil"], "slot": "drug"},
    {"canonical": "Capecitabine",       "variants": ["capecitabine", "xeloda"], "slot": "drug"},
    {"canonical": "Irinotecan",         "variants": ["irinotecan", "camptosar"], "slot": "drug"},
    {"canonical": "Etoposide",          "variants": ["etoposide", "vp-16"], "slot": "drug"},
    {"canonical": "Doxorubicin",        "variants": ["doxorubicin", "adriamycin"], "slot": "drug"},
    {"canonical": "Cyclophosphamide",   "variants": ["cyclophosphamide", "cytoxan"], "slot": "drug"},
    {"canonical": "Vincristine",        "variants": ["vincristine", "oncovin"], "slot": "drug"},
    {"canonical": "FOLFOX",             "variants": ["folfox", "folfox-4", "folfox-6", "mfolfox6"], "slot": "drug"},
    {"canonical": "FOLFIRI",            "variants": ["folfiri"], "slot": "drug"},
    {"canonical": "FOLFIRINOX",         "variants": ["folfirinox", "mfolfirinox"], "slot": "drug"},
    {"canonical": "Pemetrexed",         "variants": ["pemetrexed", "alimta"], "slot": "drug"},
]


# ─── Anatomical sites (→ sites_detected) ─────────────────────────────────
# Primary sites and anatomic locations where tumors arise or spread to.
# Matching is substring for multi-word terms, word-bounded for short ones.
SITE_SYNONYM_GROUPS: List[Dict[str, Any]] = [
    # ── Breast ──────────────────────────────────────────────────────
    {"canonical": "Breast",             "variants": ["breast", "mammary gland", "mammary tissue"]},
    {"canonical": "Breast duct",        "variants": ["breast duct", "mammary duct", "ducts of the breast"]},
    {"canonical": "Breast lobule",      "variants": ["breast lobule", "mammary lobule", "lobules"]},
    {"canonical": "Nipple",             "variants": ["nipple", "nipple-areolar complex"]},
    # ── Lung ────────────────────────────────────────────────────────
    {"canonical": "Lung",               "variants": ["lung", "pulmonary parenchyma"]},
    {"canonical": "Bronchus",           "variants": ["bronchus", "bronchi", "bronchial tree", "bronchial"]},
    {"canonical": "Alveolus",           "variants": ["alveolus", "alveoli", "alveolar"]},
    {"canonical": "Trachea",            "variants": ["trachea", "tracheal"]},
    {"canonical": "Pleura",             "variants": ["pleura", "pleural"]},
    {"canonical": "Mediastinum",        "variants": ["mediastinum", "mediastinal"]},
    # ── Prostate / GU ───────────────────────────────────────────────
    {"canonical": "Prostate",           "variants": ["prostate", "prostate gland", "prostatic"]},
    {"canonical": "Seminal vesicle",    "variants": ["seminal vesicle", "seminal vesicles"]},
    {"canonical": "Bladder",            "variants": ["bladder", "urinary bladder"]},
    {"canonical": "Kidney",             "variants": ["kidney", "renal", "renal parenchyma"]},
    {"canonical": "Ureter",             "variants": ["ureter", "ureteral"]},
    {"canonical": "Urethra",            "variants": ["urethra", "urethral"]},
    {"canonical": "Testis",             "variants": ["testis", "testes", "testicle", "testicles", "testicular"]},
    {"canonical": "Penis",              "variants": ["penis", "penile"]},
    {"canonical": "Adrenal gland",      "variants": ["adrenal", "adrenal gland", "suprarenal"]},
    # ── Gyn ─────────────────────────────────────────────────────────
    {"canonical": "Uterus",             "variants": ["uterus", "uterine", "endometrium", "endometrial"]},
    {"canonical": "Cervix",             "variants": ["cervix", "cervical", "uterine cervix"]},
    {"canonical": "Ovary",              "variants": ["ovary", "ovaries", "ovarian"]},
    {"canonical": "Fallopian tube",     "variants": ["fallopian tube", "fallopian tubes", "tubal"]},
    {"canonical": "Vagina",             "variants": ["vagina", "vaginal"]},
    {"canonical": "Vulva",              "variants": ["vulva", "vulvar"]},
    {"canonical": "Peritoneum",         "variants": ["peritoneum", "peritoneal", "peritoneal cavity"]},
    # ── GI ──────────────────────────────────────────────────────────
    {"canonical": "Esophagus",          "variants": ["esophagus", "oesophagus", "esophageal", "oesophageal"]},
    {"canonical": "Stomach",            "variants": ["stomach", "gastric"]},
    {"canonical": "Gastroesophageal junction", "variants": ["gastroesophageal junction", "gej", "oesophagogastric junction", "goj"]},
    {"canonical": "Small intestine",    "variants": ["small intestine", "jejunum", "ileum", "duodenum", "duodenal", "jejunal", "ileal"]},
    {"canonical": "Colon",              "variants": ["colon", "colonic", "large intestine", "sigmoid", "ascending colon", "descending colon", "transverse colon"]},
    {"canonical": "Rectum",             "variants": ["rectum", "rectal"]},
    {"canonical": "Anus",               "variants": ["anus", "anal", "anal canal"]},
    {"canonical": "Liver",              "variants": ["liver", "hepatic", "hepatobiliary"]},
    {"canonical": "Pancreas",           "variants": ["pancreas", "pancreatic"]},
    {"canonical": "Biliary tract",      "variants": ["biliary tract", "bile duct", "biliary", "intrahepatic biliary", "extrahepatic biliary"]},
    {"canonical": "Gallbladder",        "variants": ["gallbladder", "gallbladder cancer"]},
    # ── Head & Neck ─────────────────────────────────────────────────
    {"canonical": "Oral cavity",        "variants": ["oral cavity", "oral tongue", "floor of mouth", "buccal mucosa", "hard palate", "gingiva", "alveolar ridge", "retromolar trigone"]},
    {"canonical": "Oropharynx",         "variants": ["oropharynx", "oropharyngeal", "base of tongue", "tonsil", "tonsillar", "soft palate", "posterior pharyngeal wall"]},
    {"canonical": "Nasopharynx",        "variants": ["nasopharynx", "nasopharyngeal"]},
    {"canonical": "Hypopharynx",        "variants": ["hypopharynx", "hypopharyngeal", "pyriform sinus", "pyriform fossa"]},
    {"canonical": "Larynx",             "variants": ["larynx", "laryngeal", "glottis", "supraglottis", "subglottis", "glottic", "supraglottic", "subglottic"]},
    {"canonical": "Paranasal sinuses",  "variants": ["paranasal sinus", "paranasal sinuses", "maxillary sinus", "ethmoid sinus", "sphenoid sinus", "frontal sinus"]},
    {"canonical": "Nasal cavity",       "variants": ["nasal cavity", "nasal"]},
    {"canonical": "Salivary gland",     "variants": ["salivary gland", "parotid", "submandibular gland", "sublingual gland", "minor salivary gland"]},
    {"canonical": "Thyroid",            "variants": ["thyroid", "thyroid gland"]},
    {"canonical": "Parathyroid",        "variants": ["parathyroid", "parathyroid gland"]},
    # ── CNS ─────────────────────────────────────────────────────────
    {"canonical": "Brain",              "variants": ["brain", "cerebrum", "cerebral", "cerebellum", "brainstem"]},
    {"canonical": "Spinal cord",        "variants": ["spinal cord", "spinal"]},
    {"canonical": "Meninges",           "variants": ["meninges", "meningeal", "dura", "dural", "leptomeningeal"]},
    {"canonical": "Cranial nerve",      "variants": ["cranial nerve", "cranial nerves", "optic nerve", "acoustic nerve"]},
    {"canonical": "Pituitary",          "variants": ["pituitary", "pituitary gland", "hypothalamic"]},
    # ── Lymphatic / hematopoietic ───────────────────────────────────
    {"canonical": "Lymph node",         "variants": ["lymph node", "lymph nodes", "nodal"]},
    {"canonical": "Spleen",             "variants": ["spleen", "splenic"]},
    {"canonical": "Thymus",             "variants": ["thymus", "thymic"]},
    {"canonical": "Bone marrow",        "variants": ["bone marrow", "marrow", "myeloid compartment", "blood and marrow"]},
    {"canonical": "Waldeyer's ring",    "variants": ["waldeyer ring", "waldeyer's ring", "tonsils"]},
    # ── Sarcoma / connective ────────────────────────────────────────
    {"canonical": "Bone",               "variants": ["bone", "skeletal", "osseous"]},
    {"canonical": "Cartilage",          "variants": ["cartilage", "cartilaginous"]},
    {"canonical": "Soft tissue",        "variants": ["soft tissue", "adipose tissue", "fat", "muscle", "skeletal muscle", "smooth muscle"]},
    {"canonical": "Blood vessel",       "variants": ["blood vessel", "blood vessels", "vascular", "vessel wall"]},
    {"canonical": "Peripheral nerve",   "variants": ["peripheral nerve", "peripheral nerves", "nerve sheath"]},
    {"canonical": "Tendon",             "variants": ["tendon", "tendinous"]},
    {"canonical": "Retroperitoneum",    "variants": ["retroperitoneum", "retroperitoneal"]},
    {"canonical": "Extremity",          "variants": ["extremity", "extremities", "upper extremity", "lower extremity", "limb"]},
    {"canonical": "Chest wall",         "variants": ["chest wall", "thoracic wall"]},
    {"canonical": "Abdominal wall",     "variants": ["abdominal wall"]},
    {"canonical": "Mesentery",          "variants": ["mesentery", "mesenteric"]},
    # ── Skin / cutaneous ────────────────────────────────────────────
    {"canonical": "Skin",               "variants": ["skin", "cutaneous", "epidermis", "dermis", "epidermal", "dermal"]},
    {"canonical": "Melanocyte",         "variants": ["melanocyte", "melanocytes", "melanocytic"]},
    {"canonical": "Hair follicle",      "variants": ["hair follicle", "hair follicles", "follicular epithelium"]},
    # ── Other sites ─────────────────────────────────────────────────
    {"canonical": "Eye",                "variants": ["eye", "ocular", "intraocular", "retina", "retinal", "uvea", "uveal", "choroid", "conjunctiva"]},
    {"canonical": "Heart",              "variants": ["heart", "cardiac", "myocardium", "myocardial", "right ventricle", "left ventricle", "ventricular", "atrium", "atrial"]},
    {"canonical": "Pericardium",        "variants": ["pericardium", "pericardial"]},
    {"canonical": "Neuroendocrine system", "variants": ["neuroendocrine system", "neuroendocrine cells", "enterochromaffin"]},
    {"canonical": "Pelvic lymph node",  "variants": ["pelvic lymph node", "pelvic lymph nodes", "pelvic nodes", "obturator node", "external iliac node", "internal iliac node"]},
]


# ─── Regex patterns for structured clinical fields ───────────────────────
# TNM: T/N/M axes can appear adjacent (cT2aN1bM0), so no word boundaries
# in the middle. We require a word boundary only BEFORE the first token of
# a cluster, and the optional prefix (yp/c/p/...) is only consumed when
# preceded by a word boundary.
# Two patterns: one for the first token of a cluster (allows prefix and
# requires a word-boundary before it), one for subsequent TNM tokens in the
# same cluster (no prefix, no word boundary required — they sit right
# after the previous token's stage digit or suffix).
_TNM_CLUSTER_START_RE = re.compile(
    r'\b(?P<prefix>yp|yc|cp|c|p|r|a|u)?'
    r'(?P<axis>[TNM])'
    r'(?P<stage>is|[0-4x])'
    r'(?P<suffix>mi|sn|i\+|[abcd])?'
    r'(?=[TNM]|[^A-Za-z]|$)',
    re.IGNORECASE,
)
# Used recursively after a start-token match to grab the next adjacent axis
_TNM_NEXT_AXIS_RE = re.compile(
    r'(?P<axis>[NM])'
    r'(?P<stage>is|[0-4x])'
    r'(?P<suffix>mi|sn|i\+|[abcd])?'
    r'(?=[TNM]|[^A-Za-z]|$)',
    re.IGNORECASE,
)

# Stage: "Stage I/II/III/IV" with optional A/B/C/0
_STAGE_RE = re.compile(
    r'\bstage\s+(0|I{1,3}V?|IV)\s*([ABC]|A1|A2|B1|B2|C1|C2)?\b',
    re.IGNORECASE,
)
# Also match bare "Stage 1/2/3/4"
_STAGE_ARABIC_RE = re.compile(
    r'\bstage\s+([1-4])\s*([abc])?\b',
    re.IGNORECASE,
)

# Tumor grade
_GRADE_GX_RE = re.compile(r'\bg(?:rade)?\s?([1-4])\b(?!\s*\+)', re.IGNORECASE)
_GLEASON_RE = re.compile(
    r'\bgleason\s*(?:score\s*)?(\d)\s*\+\s*(\d)(?:\s*=\s*(\d{1,2}))?\b',
    re.IGNORECASE,
)
_ISUP_RE = re.compile(r'\bisup\s+(?:grade\s+)?group\s+([1-5])\b', re.IGNORECASE)
_WHO_GRADE_RE = re.compile(r'\bwho\s+grade\s+(I{1,3}V?|IV|[1-4])\b', re.IGNORECASE)
_DIFFERENTIATION_RE = re.compile(
    r'\b(well|moderately|poorly|un)[\s-]differentiated\b',
    re.IGNORECASE,
)

# Pathologic vs clinical staging qualifier
_STAGING_QUALIFIER_RE = re.compile(
    r'\b(pathologic(?:al)?\s+stage|clinical\s+stage|pre-?treatment\s+stage|post-?neoadjuvant\s+stage)\b',
    re.IGNORECASE,
)


# ─── Patient demographics regex patterns ────────────────────────────────
# Age: range, median, min, max, and individual-patient age.
_AGE_RANGE_RE = re.compile(
    r'\b(?:aged?|age\s+range|between|from)\s+(\d{1,3})\s*(?:to|-|–|—)\s*(\d{1,3})\s*(?:years?|yrs?|yo|y)\b',
    re.IGNORECASE,
)
_AGE_MEDIAN_RE = re.compile(
    r'\bmedian\s+age\s+(?:of\s+|was\s+)?(\d{1,3}(?:\.\d)?)\s*(?:\([^)]*\))?\s*(?:years?|yrs?|yo|y)?\b',
    re.IGNORECASE,
)
_AGE_INDIVIDUAL_RE = re.compile(
    r'\b(\d{1,3})[\s\-]?year[\s\-]?old\b',
    re.IGNORECASE,
)
_AGE_MIN_RE = re.compile(
    r'\b(?:aged?|age)\s*[≥>=]+\s*(\d{1,3})\s*(?:years?|yrs?|yo|y)?\b',
    re.IGNORECASE,
)
_AGE_MAX_RE = re.compile(
    r'\b(?:aged?|age)\s*[≤<=]+\s*(\d{1,3})\s*(?:years?|yrs?|yo|y)?\b',
    re.IGNORECASE,
)

# Gender / sex
_GENDER_PCT_RE = re.compile(
    r'\b(\d{1,3}(?:\.\d)?)\s*%\s*(male|men|female|women|m|f)\b',
    re.IGNORECASE,
)
_GENDER_COUNT_RE = re.compile(
    r'\b(\d+)\s*(?:\([^)]*\))?\s*(males?|men|females?|women)\b',
    re.IGNORECASE,
)
_GENDER_KEYWORDS_RE = re.compile(
    r'\b(male[s]?\s+and\s+female[s]?|men\s+and\s+women|'
    r'pediatric|paediatric|adult[s]?|postmenopausal|premenopausal)\b',
    re.IGNORECASE,
)

# Performance status: ECOG (0-4), KPS/Karnofsky (0-100), Lansky (0-100)
_ECOG_RE = re.compile(
    r'\becog(?:\s+(?:ps|performance\s+status))?\s*(?:of\s+)?[≤<=]*\s*(\d)(?:\s*(?:-|to|–)\s*(\d))?\b',
    re.IGNORECASE,
)
_KPS_RE = re.compile(
    r'\b(?:kps|karnofsky(?:\s+performance\s+(?:score|status))?)\s*[≥>=]*\s*(\d{2,3})\b',
    re.IGNORECASE,
)
_LANSKY_RE = re.compile(
    r'\blansky\s*[≥>=]*\s*(\d{2,3})\b',
    re.IGNORECASE,
)


# ─── Histology → Histopathologic family ──────────────────────────────────
# Maps canonical histology labels (from HISTOLOGY_SYNONYM_GROUPS) to the
# high-level family used by the trial matcher's diagnostic classification.
_HISTOPATHOLOGIC_FAMILY: Dict[str, str] = {
    # Carcinoma family
    "Adenocarcinoma":                      "Carcinoma",
    "Squamous cell carcinoma":             "Carcinoma",
    "Small cell carcinoma":                "Carcinoma",
    "Large cell neuroendocrine carcinoma": "Carcinoma",
    "Non-small cell carcinoma":            "Carcinoma",
    "Ductal carcinoma (invasive)":         "Carcinoma",
    "Lobular carcinoma (invasive)":        "Carcinoma",
    "DCIS":                                "Carcinoma",
    "LCIS":                                "Carcinoma",
    "Mucinous carcinoma":                  "Carcinoma",
    "Signet ring cell carcinoma":          "Carcinoma",
    "Medullary carcinoma":                 "Carcinoma",
    "Papillary carcinoma":                 "Carcinoma",
    "Follicular carcinoma":                "Carcinoma",
    "Anaplastic carcinoma":                "Carcinoma",
    "Hurthle cell carcinoma":              "Carcinoma",
    "Clear cell carcinoma":                "Carcinoma",
    "Renal cell carcinoma":                "Carcinoma",
    "Urothelial carcinoma":                "Carcinoma",
    "Hepatocellular carcinoma":            "Carcinoma",
    "Cholangiocarcinoma":                  "Carcinoma",
    "Pancreatic ductal adenocarcinoma":    "Carcinoma",
    "Basal cell carcinoma":                "Carcinoma",
    "Merkel cell carcinoma":               "Carcinoma",
    "Acinar prostate adenocarcinoma":      "Carcinoma",
    "High-grade serous carcinoma":         "Carcinoma",
    # Sarcoma family
    "Sarcoma":                             "Sarcoma",
    "Leiomyosarcoma":                      "Sarcoma",
    "Liposarcoma":                         "Sarcoma",
    "Osteosarcoma":                        "Sarcoma",
    "Ewing sarcoma":                       "Sarcoma",
    "Rhabdomyosarcoma":                    "Sarcoma",
    "GIST":                                "Sarcoma",
    # Lymphoma family
    "Hodgkin lymphoma":                    "Lymphoma",
    "Diffuse large B-cell lymphoma":       "Lymphoma",
    "Follicular lymphoma":                 "Lymphoma",
    "Mantle cell lymphoma":                "Lymphoma",
    "Marginal zone lymphoma":              "Lymphoma",
    # Leukemia family
    "Acute lymphoblastic leukemia":        "Leukemia",
    "Acute myeloid leukemia":              "Leukemia",
    "Chronic lymphocytic leukemia":        "Leukemia",
    "Chronic myeloid leukemia":            "Leukemia",
    # Myeloma
    "Multiple myeloma":                    "Myeloma",
    # Melanoma
    "Melanoma":                            "Melanoma",
    # Glioma/CNS
    "Glioblastoma":                        "Glioma",
    "Glioma":                              "Glioma",
    "Astrocytoma":                         "Glioma",
    "Oligodendroglioma":                   "Glioma",
    "Meningioma":                          "Meningioma",
    # Neuroendocrine
    "Neuroendocrine tumor":                "Neuroendocrine",
}


class KeywordTagger:
    """
    Tag document chunks with keywords, ontology terms, biomarkers, drugs,
    genomic alterations, and expanded synonyms.

    Loads all data files once at construction time and provides fast
    per-chunk scanning via tag_chunk().
    """

    def __init__(
        self,
        keyword_json_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        self.data_dir = data_dir or _DATA_DIR
        self.keyword_json_path = keyword_json_path or (
            self.data_dir / "keywords" / "extractor_keywords.json"
        )

        # ── Extractor keywords: term → {categories} ──────────────────
        self.term_to_categories: Dict[str, Set[str]] = defaultdict(set)

        # ── Cancer ontology: term → [(cancer_label, field, weight)] ──
        # Each term can contribute to multiple cancers (e.g. "pembrolizumab"
        # contributes to Lung + Breast + HNSCC + ...). We track field and
        # ambiguity-damped weight per (term, cancer) pair.
        self.cancer_term_evidence: Dict[
            str, List[Tuple[str, str, float]]
        ] = defaultdict(list)

        # ── Trial ontology tags: term → {tags} (e.g. "trial:biomarkers:...")
        self.trial_tags: Dict[str, Set[str]] = defaultdict(set)

        # ── Dedicated structured slots ────────────────────────────────
        # term → dedicated slot name: biomarker / drug / alteration
        self.biomarker_terms: Set[str] = set()     # lowercase
        self.drug_terms: Set[str] = set()          # lowercase
        self.alteration_terms: Set[str] = set()    # lowercase

        # ── Synonym group index: variant → (canonical, slot) ──────────
        # slot ∈ {"biomarker", "alteration", "imaging", "serum_marker"}
        self.synonym_variant_to_canonical: Dict[str, Tuple[str, str]] = {}

        # ── AJCC staging aliases ──────────────────────────────────────
        self.ajcc_tags: Dict[str, Set[str]] = defaultdict(set)

        self._load_all()

    # ────────────────────────────────────────────────────────────────────
    # LOAD
    # ────────────────────────────────────────────────────────────────────
    def _load_all(self) -> None:
        self._load_extractor_keywords()
        self._load_cancer_ontology()
        self._load_trial_ontology()
        self._load_ajcc()
        self._load_synonym_groups()

        n_cats = len({
            c for cats in self.term_to_categories.values() for c in cats
        })
        n_cancer_terms = len(self.cancer_term_evidence)
        print(
            f"[KeywordTagger] Loaded {len(self.term_to_categories)} "
            f"extractor keywords across {n_cats} categories | "
            f"{n_cancer_terms} cancer-ontology terms | "
            f"{len(self.trial_tags)} trial-ontology terms | "
            f"{len(self.biomarker_terms)} biomarkers | "
            f"{len(self.drug_terms)} drugs | "
            f"{len(self.alteration_terms)} alterations | "
            f"{len(self.synonym_variant_to_canonical)} synonym variants"
        )

    def _load_synonym_groups(self) -> None:
        """Build variant → (canonical, slot) index from all synonym groups."""
        for group, default_slot in [
            (BIOMARKER_SYNONYM_GROUPS,        "biomarker"),
            (HISTOLOGY_SYNONYM_GROUPS,        "histology"),
            (DISEASE_STATUS_SYNONYM_GROUPS,   "disease_status"),
            (TREATMENT_LINE_SYNONYM_GROUPS,   "treatment_line"),
            (SITE_SYNONYM_GROUPS,             "site"),
            (DRUG_SYNONYM_GROUPS,             "drug"),
        ]:
            for g in group:
                canonical = g["canonical"]
                slot = g.get("slot", default_slot)
                for variant in g["variants"]:
                    vl = variant.lower().strip()
                    if not vl:
                        continue
                    for v in _term_variants(vl):
                        self.synonym_variant_to_canonical.setdefault(v, (canonical, slot))

    def _load_extractor_keywords(self) -> None:
        kw_json = _load_json(self.keyword_json_path)

        def _flatten(obj: Any, category: str):
            if isinstance(obj, dict):
                for sub_val in obj.values():
                    _flatten(sub_val, category)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and len(item) >= 2:
                        self.term_to_categories[item.lower()].add(category)
            elif isinstance(obj, str) and len(obj) >= 2:
                self.term_to_categories[obj.lower()].add(category)

        for cat, content in kw_json.items():
            _flatten(content, cat)

    def _load_cancer_ontology(self) -> None:
        cancer_json = _load_json(
            self.data_dir / "ontology" / "cancer_type_ontology.json"
        )

        # Pass 1: count how many cancers share each term per field so we can
        # damp ambiguous terms' contribution.
        term_cancers: Dict[str, Set[str]] = defaultdict(set)
        raw_records: List[Tuple[str, str, str]] = []  # (term, cancer_label, field)

        ALL_FIELDS = (
            "synonyms", "keywords", "drugs", "subtypes",
            "histologies", "surgeries", "related_diseases", "related_pathways",
        )

        for cancer_key, cancer_data in cancer_json.items():
            if not isinstance(cancer_data, dict):
                continue
            label = cancer_data.get("label", cancer_key)

            # Add the label itself as a self-match ("Lung Cancer", etc.)
            if isinstance(label, str) and len(label) >= 3:
                for variant in _term_variants(label):
                    raw_records.append((variant, label, "label"))
                    term_cancers[variant].add(label)

            for field in ALL_FIELDS:
                for term in (cancer_data.get(field) or []):
                    if isinstance(term, str) and len(term) >= 2:
                        for variant in _term_variants(term):
                            raw_records.append((variant, label, field))
                            term_cancers[variant].add(label)

        # Pass 2: build evidence records with ambiguity-damped weights.
        for term, cancer_label, field in raw_records:
            base_w = _CANCER_FIELD_WEIGHT.get(field, 1.0)
            n_sharing = max(1, len(term_cancers[term]))
            # Damp by ambiguity: term in 5 cancers → 1/5 contribution each.
            # But specific fields (label, synonyms, related_diseases) are
            # trusted at closer to full weight even when technically shared.
            if field in _CANCER_SPECIFIC_FIELDS:
                weight = base_w / max(1.0, n_sharing ** 0.5)
            else:
                weight = base_w / n_sharing
            self.cancer_term_evidence[term].append((cancer_label, field, weight))

    def _load_trial_ontology(self) -> None:
        trial_json = _load_json(
            self.data_dir / "ontology" / "clinical_trial_ontology.json"
        )

        def _flatten_trial(obj: Any, prefix: str):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    nested = f"{prefix}:{k}" if prefix else k
                    _flatten_trial(v, nested)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and len(item) >= 2:
                        for variant in _term_variants(item):
                            self._register_trial_term(variant, prefix)
            elif isinstance(obj, str) and len(obj) >= 2:
                for variant in _term_variants(obj):
                    self._register_trial_term(variant, prefix)

        for top_key, top_val in trial_json.items():
            if top_key == "meta":
                continue
            _flatten_trial(top_val, top_key)

    # Ambiguous short terms that are in the ontology but collide with other
    # common clinical abbreviations when case-folded. Kept in generic
    # trial_tags for coverage but NOT promoted to dedicated slots.
    _AMBIGUOUS_TRIAL_TERMS = {
        "pcr",   # "pCR" = pathologic complete response (clinical endpoint),
                 # NOT PCR (polymerase chain reaction assay)
        "all",   # "ALL" = Acute Lymphoblastic Leukemia, but "all" is common
        "or",    # "OR" = objective response, but "or" is a conjunction
        "cr",    # "CR" = complete response, but collides with other acronyms
        "pd",    # "PD" = progressive disease, but collides with PD-1/PD-L1
        "sd",    # "SD" = stable disease, but collides with standard deviation
        "pr",    # "PR" = partial response OR progesterone receptor — keep in
                 # biomarkers via synonym group, not trial slot
    }

    def _register_trial_term(self, term: str, prefix: str) -> None:
        """Register a trial-ontology term into generic tags + dedicated slots."""
        self.trial_tags[term].add(f"trial:{prefix}")

        if term in self._AMBIGUOUS_TRIAL_TERMS:
            return  # don't promote to dedicated slots

        # Dedicated slot routing by prefix path:
        p = prefix.lower()
        if p.startswith("biomarkers"):
            if "genomic_alterations" in p or "fusion" in p or "snv" in p \
                    or "cnv" in p or "methylation" in p:
                self.alteration_terms.add(term)
            else:
                self.biomarker_terms.add(term)
        elif p.startswith("treatment:systemic_therapy") \
                or p.startswith("treatment:radiopharmaceuticals"):
            self.drug_terms.add(term)

    def _load_ajcc(self) -> None:
        ajcc_json = _load_json(self.data_dir / "ajcc_staging_tables.json")
        for cancer_key, cancer_table in ajcc_json.items():
            if cancer_key.startswith("_") or not isinstance(cancer_table, dict):
                continue
            for alias in (cancer_table.get("aliases") or []):
                if isinstance(alias, str) and len(alias) >= 2:
                    self.ajcc_tags[alias.lower()].add(f"ajcc:{cancer_key}")
            for def_type in ("t_definitions", "n_definitions", "m_definitions"):
                defs = cancer_table.get(def_type) or {}
                for _code, desc in defs.items():
                    if isinstance(desc, str) and len(desc) >= 3:
                        key = desc.lower()[:100]
                        self.ajcc_tags[key].add(f"ajcc:{cancer_key}:{def_type}")

        # Cross-link cancer-ontology drugs into dedicated drug slot as well.
        for term, evidence in self.cancer_term_evidence.items():
            for _cancer, field, _w in evidence:
                if field == "drugs":
                    self.drug_terms.add(term)
                    break

    # ────────────────────────────────────────────────────────────────────
    # SCAN
    # ────────────────────────────────────────────────────────────────────
    def _word_bounded(self, term: str) -> bool:
        """Whether a term needs word-boundary matching.

        Single-word acronyms/names (no whitespace) always use word boundary
        so that "sclc" doesn't match inside "nsclc", "all" doesn't match
        in "small", "er" doesn't match in "other", etc. Multi-word terms
        use substring matching.
        """
        if ' ' in term:
            return False  # multi-word term — substring match is fine
        return True

    def _match(self, term: str, text_lower: str) -> bool:
        """Match a term against lowered text, word-bounded for short/acronym-like terms."""
        if self._word_bounded(term):
            return bool(re.search(rf'\b{re.escape(term)}\b', text_lower))
        return term in text_lower

    def _positions(self, term: str, text_lower: str) -> List[int]:
        """Return start positions of `term` in `text_lower` (may be empty)."""
        return _find_positions(term, text_lower, self._word_bounded(term))

    def _has_non_negated_occurrence(self, term: str, text_lower: str) -> bool:
        """True if `term` appears at least once without a nearby negation cue."""
        positions = self._positions(term, text_lower)
        if not positions:
            return False
        for pos in positions:
            if not _is_negated(text_lower, pos, pos + len(term)):
                return True
        return False

    def scan_text(self, text: str) -> Tuple[
        Dict[str, List[str]],  # keyword_matches: category -> [terms]
        List[str],             # keywords_flat
        List[str],             # ontology_tags
        List[str],             # expanded_synonyms
    ]:
        """
        Scan text against all keyword + ontology tables and expand synonyms.

        Returns:
            (keyword_matches, keywords_flat, ontology_matches, expanded_synonyms)

        For additional structured data (cancer types, biomarkers, drugs,
        alterations), call scan_text_detailed() instead.
        """
        detailed = self.scan_text_detailed(text)
        return (
            detailed["keyword_matches"],
            detailed["keywords_flat"],
            detailed["ontology_tags"],
            detailed["expanded_synonyms"],
        )

    def scan_text_detailed(self, text: str) -> Dict[str, Any]:
        """Full scan producing structured output, with NegEx-style negation
        and bidirectional synonym-group expansion for biomarkers/imaging/tests.
        """
        text_lower = (text or "").lower()

        category_to_terms: Dict[str, List[str]] = defaultdict(list)
        flat_terms: List[str] = []
        ont_matches: Set[str] = set()
        biomarkers_hit: Set[str] = set()
        drugs_hit: Set[str] = set()
        alterations_hit: Set[str] = set()
        imaging_hit: Set[str] = set()
        serum_markers_hit: Set[str] = set()
        ajcc_hit: Set[str] = set()
        # New slots for hard-filter axes
        histologies_hit: Set[str] = set()
        disease_status_hit: Set[str] = set()
        treatment_lines_hit: Set[str] = set()
        sites_hit: Set[str] = set()
        stages_hit: Set[str] = set()
        tnm_hit: Set[str] = set()
        grades_hit: Set[str] = set()
        staging_qualifiers_hit: Set[str] = set()
        negated_mentions: List[str] = []  # for debugging
        # Biomarker / alteration / serum-marker / imaging polarity detection.
        # Maps canonical → set of statuses (positive/negative/high/low/
        # mutated/amplified/value:<num>). "STATUS_BEARING_SLOTS" defines
        # which slots participate.
        status_map: Dict[str, Set[str]] = defaultdict(set)
        STATUS_BEARING_SLOTS = {"biomarker", "alteration", "serum_marker", "imaging"}

        def _route_to_slot(canonical: str, slot: str):
            if slot == "alteration":
                alterations_hit.add(canonical)
            elif slot == "imaging":
                imaging_hit.add(canonical)
            elif slot == "serum_marker":
                serum_markers_hit.add(canonical)
            elif slot == "histology":
                histologies_hit.add(canonical)
            elif slot == "disease_status":
                disease_status_hit.add(canonical)
            elif slot == "treatment_line":
                treatment_lines_hit.add(canonical)
            elif slot == "site":
                sites_hit.add(canonical)
            elif slot == "drug":
                drugs_hit.add(canonical)
            else:
                biomarkers_hit.add(canonical)

        # ── Extractor keywords (NOT negation-checked — too noisy, fast path) ─
        for term, categories in self.term_to_categories.items():
            if self._match(term, text_lower):
                for cat in categories:
                    category_to_terms[cat].append(term)
                flat_terms.append(term)

        # ── Cancer ontology scoring (negation-checked) ─────────────────
        cancer_scores: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "terms": set(), "fields": set()}
        )
        for term, evidence in self.cancer_term_evidence.items():
            # Cheap existence gate first
            if not self._match(term, text_lower):
                continue
            # Negation gate: require at least one non-negated occurrence
            if not self._has_non_negated_occurrence(term, text_lower):
                negated_mentions.append(term)
                continue
            for cancer_label, field, weight in evidence:
                rec = cancer_scores[cancer_label]
                rec["score"] += weight
                rec["terms"].add(term)
                rec["fields"].add(field)

        cancer_types_detected: List[str] = []
        for cancer_label, rec in cancer_scores.items():
            specific_hit = bool(rec["fields"] & _CANCER_SPECIFIC_FIELDS)
            non_generic_terms = {t for t in rec["terms"]
                                 if t not in _GENERIC_AMBIGUOUS_TERMS}
            has_non_generic = len(non_generic_terms) >= 1

            if not has_non_generic:
                continue

            passes_threshold = rec["score"] >= _CANCER_SCORE_THRESHOLD
            if specific_hit or passes_threshold:
                cancer_types_detected.append(cancer_label)
                ont_matches.add(f"cancer:{cancer_label}")

        # ── Trial ontology ─────────────────────────────────────────────
        # Biomarker / alteration terms record status; other tags suppress
        # when negated. Drugs use concept-level negation (a "not given"
        # drug shouldn't be attributed to this patient).
        for term, tags in self.trial_tags.items():
            if not self._match(term, text_lower):
                continue

            is_biomarker = term in self.biomarker_terms
            is_alteration = term in self.alteration_terms
            is_drug = term in self.drug_terms

            if is_biomarker or is_alteration:
                # Always record; capture status per occurrence.
                for pos in self._positions(term, text_lower):
                    status = _detect_status(
                        text_lower, pos, pos + len(term), term
                    )
                    if status:
                        status_map[term].add(status)
                ont_matches.update(tags)
                if is_biomarker:
                    biomarkers_hit.add(term)
                if is_alteration:
                    alterations_hit.add(term)
            else:
                # Concept-level negation for other tags
                if not self._has_non_negated_occurrence(term, text_lower):
                    negated_mentions.append(term)
                    continue
                ont_matches.update(tags)
                if is_drug:
                    drugs_hit.add(term)

        # ── Synonym groups (biomarker/imaging/serum_marker/alteration/
        # histology/disease_status/treatment_line/site/drug) ──────────
        # Behavior splits by slot type:
        #  - STATUS_BEARING_SLOTS (biomarker, alteration, imaging,
        #    serum_marker): always tag the canonical even if preceded/
        #    followed by "negative" — "HER2-" is DATA, not a negation of
        #    the concept. Polarity is recorded in status_map.
        #  - Other slots (histology, disease_status, treatment_line,
        #    site, drug): keep negation suppression — "no history of
        #    breast cancer" truly means suppress Breast.
        seen_canonicals: Set[Tuple[str, str]] = set()
        for variant, (canonical, slot) in self.synonym_variant_to_canonical.items():
            if (canonical, slot) in seen_canonicals:
                continue
            if not self._match(variant, text_lower):
                continue

            positions = self._positions(variant, text_lower)
            if not positions:
                continue

            if slot in STATUS_BEARING_SLOTS:
                # Always tag; detect status per occurrence.
                for pos in positions:
                    status = _detect_status(
                        text_lower, pos, pos + len(variant), canonical
                    )
                    if status:
                        status_map[canonical].add(status)
                _route_to_slot(canonical, slot)
                seen_canonicals.add((canonical, slot))
            else:
                # Concept-level negation: suppress if ALL occurrences
                # are negated (e.g., "no history of breast cancer").
                # EXCEPTION: if the variant itself contains a status word
                # ("triple-negative", "ICI-refractory", "castration-resistant"),
                # a trailing "negative"/"refractory" would wrongly trigger
                # suppression — for these, only check the BEFORE cue.
                has_embedded_status = bool(re.search(
                    r'\b(positive|negative|refractory|resistant|sensitive|'
                    r'advanced|unresectable|recurrent|progressive|metastatic)\b',
                    variant, re.IGNORECASE,
                ))
                non_negated = [
                    p for p in positions
                    if not _is_negated(
                        text_lower, p, p + len(variant),
                        check_after=not has_embedded_status,
                    )
                ]
                if not non_negated:
                    negated_mentions.append(f"{variant}→{canonical}")
                    continue
                _route_to_slot(canonical, slot)
                seen_canonicals.add((canonical, slot))

        # ── Suppress ontology-level variants when the canonical form
        # has been tagged via a synonym group. E.g. if "MSI-H" is tagged
        # as canonical, remove "msi", "msi-h", "microsatellite_instability_high"
        # from biomarkers_hit so the output is clean.
        slot_map = {
            "biomarker":      biomarkers_hit,
            "alteration":     alterations_hit,
            "imaging":        imaging_hit,
            "serum_marker":   serum_markers_hit,
            "histology":      histologies_hit,
            "disease_status": disease_status_hit,
            "treatment_line": treatment_lines_hit,
            "site":           sites_hit,
            "drug":           drugs_hit,
        }
        for variant, (canonical, slot) in self.synonym_variant_to_canonical.items():
            target_set = slot_map.get(slot, biomarkers_hit)
            # Is the canonical already in this slot (case-insensitive)?
            canonical_lower = canonical.lower()
            if any(h.lower() == canonical_lower for h in target_set):
                target_set.discard(variant)
                target_set.discard(variant.lower())

        # ── Cancer-ontology drugs contribute to drugs_hit too ──────────
        for term in self.drug_terms:
            if term in drugs_hit:
                continue
            if not self._match(term, text_lower):
                continue
            if not self._has_non_negated_occurrence(term, text_lower):
                negated_mentions.append(term)
                continue
            drugs_hit.add(term)

        # ── AJCC staging ───────────────────────────────────────────────
        for term, tags in self.ajcc_tags.items():
            if self._match(term, text_lower):
                ajcc_hit.update(tags)
                ont_matches.update(tags)

        # ── Structured clinical extractors (regex) ─────────────────────
        # Stage I/II/III/IV (Roman) and Stage 1-4 (Arabic), with substage
        for m in _STAGE_RE.finditer(text):
            roman = m.group(1).upper()
            sub = (m.group(2) or "").upper()
            if _is_negated(text_lower, m.start(), m.end()):
                negated_mentions.append(f"stage {roman}{sub}")
                continue
            stages_hit.add(f"Stage {roman}{sub}" if sub else f"Stage {roman}")
        for m in _STAGE_ARABIC_RE.finditer(text):
            num = m.group(1)
            sub = (m.group(2) or "").upper()
            roman = {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(num, num)
            if _is_negated(text_lower, m.start(), m.end()):
                negated_mentions.append(f"stage {roman}{sub}")
                continue
            stages_hit.add(f"Stage {roman}{sub}" if sub else f"Stage {roman}")

        # TNM tokens: cT2aN1bM0, pT3N0Mx, ypT1 etc.
        # Match the first token with optional prefix, then chase adjacent
        # N/M tokens that share the same prefix by convention.
        for m in _TNM_CLUSTER_START_RE.finditer(text):
            # Skip cluster starts that look like stray words (e.g. "a"
            # matched by the prefix regex inside a word). Require that the
            # match begin at a word boundary.
            if m.start() > 0 and text[m.start() - 1].isalnum():
                continue
            if _is_negated(text_lower, m.start(), m.end()):
                negated_mentions.append(f"tnm:{m.group(0)}")
                continue
            prefix = (m.group("prefix") or "").lower()
            axis = m.group("axis").upper()
            stage = (m.group("stage") or "").lower()
            suffix = (m.group("suffix") or "").lower()
            tnm_hit.add(f"{prefix}{axis}{stage}{suffix}")

            # Chase adjacent N and M tokens (no prefix; inherit from start)
            pos = m.end()
            while pos < len(text):
                nxt = _TNM_NEXT_AXIS_RE.match(text, pos)
                if not nxt:
                    break
                n_axis = nxt.group("axis").upper()
                n_stage = (nxt.group("stage") or "").lower()
                n_suffix = (nxt.group("suffix") or "").lower()
                tnm_hit.add(f"{prefix}{n_axis}{n_stage}{n_suffix}")
                pos = nxt.end()

        # Pathologic vs clinical staging qualifier
        for m in _STAGING_QUALIFIER_RE.finditer(text):
            qualifier = re.sub(r'\s+', ' ', m.group(1).strip().lower())
            # Normalize: "pathologic stage" / "pathological stage" / "clinical stage"
            if qualifier.startswith("pathologic") or qualifier.startswith("pathological"):
                staging_qualifiers_hit.add("pathologic")
            elif qualifier.startswith("clinical"):
                staging_qualifiers_hit.add("clinical")
            elif "pre-treatment" in qualifier or "pretreatment" in qualifier:
                staging_qualifiers_hit.add("pre-treatment")
            elif "post-neoadjuvant" in qualifier or "post neoadjuvant" in qualifier:
                staging_qualifiers_hit.add("post-neoadjuvant")

        # Tumor grade G1-G4
        for m in _GRADE_GX_RE.finditer(text):
            if _is_negated(text_lower, m.start(), m.end()):
                continue
            grades_hit.add(f"G{m.group(1)}")
        # Gleason score
        for m in _GLEASON_RE.finditer(text):
            a, b, total = m.group(1), m.group(2), m.group(3)
            total = total or str(int(a) + int(b))
            grades_hit.add(f"Gleason {a}+{b}={total}")
        # ISUP grade group
        for m in _ISUP_RE.finditer(text):
            grades_hit.add(f"ISUP grade group {m.group(1)}")
        # WHO grade
        for m in _WHO_GRADE_RE.finditer(text):
            g = m.group(1).upper()
            grades_hit.add(f"WHO grade {g}")
        # Differentiation
        for m in _DIFFERENTIATION_RE.finditer(text):
            grades_hit.add(f"{m.group(1).lower()}-differentiated")

        # ── Patient demographics (age / gender / performance status) ──
        demographics: Dict[str, Any] = {
            "age": [],
            "gender": [],
            "performance_status": [],
        }
        # Age range
        for m in _AGE_RANGE_RE.finditer(text):
            demographics["age"].append(f"range:{m.group(1)}-{m.group(2)}")
        # Age median
        for m in _AGE_MEDIAN_RE.finditer(text):
            demographics["age"].append(f"median:{m.group(1)}")
        # Individual patient age
        for m in _AGE_INDIVIDUAL_RE.finditer(text):
            demographics["age"].append(f"individual:{m.group(1)}")
        # Age min / max (eligibility thresholds)
        for m in _AGE_MIN_RE.finditer(text):
            demographics["age"].append(f"min:{m.group(1)}")
        for m in _AGE_MAX_RE.finditer(text):
            demographics["age"].append(f"max:{m.group(1)}")

        # Gender: percentages ("60% male")
        for m in _GENDER_PCT_RE.finditer(text):
            pct, gender = m.group(1), m.group(2).lower()
            g = "male" if gender in ("male", "men", "m") else "female"
            demographics["gender"].append(f"{g}:{pct}%")
        # Gender: counts ("25 males, 30 females")
        for m in _GENDER_COUNT_RE.finditer(text):
            count, gender = m.group(1), m.group(2).lower()
            g = "male" if gender.startswith(("male", "men")) else "female"
            demographics["gender"].append(f"{g}:n={count}")
        # Gender: descriptive keywords
        for m in _GENDER_KEYWORDS_RE.finditer(text):
            demographics["gender"].append(m.group(1).lower())

        # ECOG performance status
        for m in _ECOG_RE.finditer(text):
            lo = m.group(1)
            hi = m.group(2)
            if hi:
                demographics["performance_status"].append(f"ECOG {lo}-{hi}")
            else:
                demographics["performance_status"].append(f"ECOG {lo}")
        # KPS / Karnofsky
        for m in _KPS_RE.finditer(text):
            demographics["performance_status"].append(f"KPS {m.group(1)}")
        # Lansky (pediatric)
        for m in _LANSKY_RE.finditer(text):
            demographics["performance_status"].append(f"Lansky {m.group(1)}")

        # Dedupe, preserve order
        demographics = {
            k: sorted(set(v)) for k, v in demographics.items() if v
        }

        # ── Histopathologic type family (Carcinoma / Sarcoma / etc.) ──
        histopathologic_types: Set[str] = set()
        for hist in histologies_hit:
            fam = _HISTOPATHOLOGIC_FAMILY.get(hist)
            if fam:
                histopathologic_types.add(fam)
        # Fallback: scan for bare family words in text if histologies empty
        if not histopathologic_types:
            for fam in ("Carcinoma", "Sarcoma", "Lymphoma", "Leukemia",
                        "Myeloma", "Melanoma", "Glioma"):
                if re.search(rf'\b{fam.lower()}\b', text_lower):
                    histopathologic_types.add(fam)

        # ── Synonym expansion (query-side expansion for output metadata)
        expanded_terms = self._expand_synonyms(text_lower, flat_terms)

        return {
            "keyword_matches": {
                k: sorted(set(v)) for k, v in category_to_terms.items()
            },
            "keywords_flat": sorted(set(flat_terms)),
            "ontology_tags": sorted(ont_matches),
            "expanded_synonyms": sorted(set(expanded_terms)),
            # Dedicated structured slots (case-insensitively deduplicated,
            # preferring canonical casing from BIOMARKER_SYNONYM_GROUPS).
            "cancer_types_detected": sorted(set(cancer_types_detected)),
            "histologies_detected": _dedupe_preferring_canonical(histologies_hit),
            "histopathologic_types": sorted(histopathologic_types),
            "sites_detected": _dedupe_preferring_canonical(sites_hit),
            "patient_demographics": demographics,
            "biomarkers_detected": _dedupe_preferring_canonical(biomarkers_hit),
            "drugs_detected": _dedupe_preferring_canonical(drugs_hit),
            "genomic_alterations": _dedupe_preferring_canonical(alterations_hit),
            "imaging_detected": _dedupe_preferring_canonical(imaging_hit),
            "serum_markers_detected": _dedupe_preferring_canonical(serum_markers_hit),
            "disease_status_detected": _dedupe_preferring_canonical(disease_status_hit),
            "treatment_lines_detected": _dedupe_preferring_canonical(treatment_lines_hit),
            "stages_detected": sorted(stages_hit),
            "tnm_detected": sorted(tnm_hit),
            "grades_detected": sorted(grades_hit),
            "staging_qualifier": sorted(staging_qualifiers_hit),
            "ajcc_tags": sorted(ajcc_hit),
            # Biomarker / alteration / marker polarity: canonical → sorted list
            # of statuses ("positive"/"negative"/"high"/"low"/"value:N%"/etc.)
            "biomarker_status": _dedupe_status_map(status_map),
            # Debug
            "_cancer_scoring": {
                label: {
                    "score": round(rec["score"], 2),
                    "terms": sorted(rec["terms"]),
                    "fields": sorted(rec["fields"]),
                }
                for label, rec in cancer_scores.items()
            },
            "_negated_mentions": sorted(set(negated_mentions)),
        }

    def _expand_synonyms(
        self, text_lower: str, flat_terms: List[str]
    ) -> List[str]:
        expanded_terms: List[str] = []
        try:
            from src.api.services.query_expansion import (
                _DRUG_SYNONYM_LOOKUP,
                _CANCER_SYNONYM_MAP,
                CLINICAL_CONTEXT_SYNONYMS,
            )
            detected_lower = set(t.lower() for t in flat_terms)

            for term in list(detected_lower):
                group = _DRUG_SYNONYM_LOOKUP.get(term)
                if group:
                    for syn in group:
                        if syn.lower() not in detected_lower:
                            expanded_terms.append(syn)
                            detected_lower.add(syn.lower())

            for term in list(flat_terms):
                if term.lower() in _CANCER_SYNONYM_MAP:
                    for syn in _CANCER_SYNONYM_MAP[term.lower()]:
                        if syn.lower() not in detected_lower and len(syn) > 2:
                            expanded_terms.append(syn)
                            detected_lower.add(syn.lower())

            for concept, synonyms in CLINICAL_CONTEXT_SYNONYMS.items():
                for syn in synonyms:
                    if syn.lower() in text_lower:
                        for other in synonyms:
                            if other.lower() not in detected_lower:
                                expanded_terms.append(other)
                                detected_lower.add(other.lower())
                        break
        except ImportError:
            pass
        return expanded_terms

    # ────────────────────────────────────────────────────────────────────
    # CHUNK TAGGING
    # ────────────────────────────────────────────────────────────────────
    def tag_chunk(
        self,
        chunk: Dict[str, Any],
        add_to_text_for_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Tag a chunk with keywords, ontology terms, and expanded synonyms.

        Populates metadata with:
            keyword_matches, keywords_flat, ontology_tags, expanded_synonyms,
            cancer_types_detected, biomarkers_detected, drugs_detected,
            genomic_alterations, ajcc_tags.
        """
        text = chunk.get("text", "") or ""
        metadata = chunk.setdefault("metadata", {})

        detailed = self.scan_text_detailed(text)

        metadata["keyword_matches"] = detailed["keyword_matches"]
        metadata["keywords_flat"] = detailed["keywords_flat"]
        metadata["ontology_tags"] = detailed["ontology_tags"]
        metadata["expanded_synonyms"] = detailed["expanded_synonyms"]
        # Cancer-type / histology / sites axis
        metadata["cancer_types_detected"] = detailed["cancer_types_detected"]
        metadata["histologies_detected"] = detailed["histologies_detected"]
        metadata["histopathologic_types"] = detailed["histopathologic_types"]
        metadata["sites_detected"] = detailed["sites_detected"]
        # Patient demographics (age / gender / performance status)
        metadata["patient_demographics"] = detailed["patient_demographics"]
        # Stage / TNM / grade axis (hard filter: tumor stage)
        metadata["stages_detected"] = detailed["stages_detected"]
        metadata["tnm_detected"] = detailed["tnm_detected"]
        metadata["grades_detected"] = detailed["grades_detected"]
        metadata["staging_qualifier"] = detailed["staging_qualifier"]
        metadata["disease_status_detected"] = detailed["disease_status_detected"]
        # Treatment / line of therapy axis (hard filter: prior therapies)
        metadata["treatment_lines_detected"] = detailed["treatment_lines_detected"]
        metadata["drugs_detected"] = detailed["drugs_detected"]
        # Biomarker axis (hard filter: molecular biomarkers)
        metadata["biomarkers_detected"] = detailed["biomarkers_detected"]
        metadata["biomarker_status"] = detailed["biomarker_status"]
        # ─── Flat parallel lists for Qdrant KEYWORD indexing ─────────
        # Qdrant can't index nested dicts, so every Dict[str, List[str]]
        # field gets a flat "key:value" list version that IS indexable.

        # biomarker_status: {"HER2": ["positive"]} → ["HER2:positive", ...]
        metadata["biomarker_status_flat"] = [
            f"{canonical}:{status}"
            for canonical, statuses in (detailed["biomarker_status"] or {}).items()
            for status in (statuses or [])
        ]
        # patient_demographics: {"age": ["individual:58"]} → ["age:individual:58"]
        metadata["patient_demographics_flat"] = [
            f"{axis}:{value}"
            for axis, values in (detailed["patient_demographics"] or {}).items()
            for value in (values or [])
        ]
        # keyword_matches: {"biomarkers": ["her2"]} → ["biomarkers:her2"]
        metadata["keyword_matches_flat"] = [
            f"{category}:{term}"
            for category, terms in (detailed["keyword_matches"] or {}).items()
            for term in (terms or [])
        ]
        metadata["genomic_alterations"] = detailed["genomic_alterations"]
        metadata["imaging_detected"] = detailed["imaging_detected"]
        metadata["serum_markers_detected"] = detailed["serum_markers_detected"]
        # Other
        metadata["ajcc_tags"] = detailed["ajcc_tags"]

        if add_to_text_for_embedding and detailed["keywords_flat"]:
            chunk["text_for_embedding"] = (
                text + "\n\nKeywords: "
                + ", ".join(detailed["keywords_flat"])
            )
        else:
            chunk["text_for_embedding"] = text

        return chunk

    # ── Backward-compatible aliases ──────────────────────────────────

    def tag_chunk_keywords(
        self,
        chunk: Dict[str, Any],
        add_to_text_for_embedding: bool = False,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for tag_chunk()."""
        return self.tag_chunk(chunk, add_to_text_for_embedding)

    def match_keywords_in_text(
        self, text: str
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Backward-compatible: returns just (keyword_matches, flat_terms)."""
        detailed = self.scan_text_detailed(text)
        return detailed["keyword_matches"], detailed["keywords_flat"]
