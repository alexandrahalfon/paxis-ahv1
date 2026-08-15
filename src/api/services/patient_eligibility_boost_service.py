"""
Patient Eligibility Filter & Boost Service

Performs rule-based hard filtering AND score boosting for retrieved studies
based on how well they match the patient described in the query.

Hard filter criteria (study is REMOVED if it explicitly contradicts):
  1. Cancer type and histology
  2. Tumor stage or disease status
  3. Prior therapies and treatment lines
  4. Molecular biomarkers (only when the study explicitly reports them)

Key principle: a study is NOT removed if it simply doesn't report a criterion.
Only confirmed mismatches (the study explicitly enrolled a *different* population)
trigger removal.
"""

import asyncio
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

from src.core.config import settings
from src.api.services.query_reconciliation import ReconciledStructure
from src.api.services.biomarker_canonicalizer import CanonicalBiomarker


# Shared Qdrant client for eligibility-context scrolls. Previously a
# fresh QdrantClient was constructed on every eligibility check (one per
# query), which pays TLS/setup overhead each time — the same per-call-
# client mistake the study-details service already documented and fixed.
_shared_qdrant_client = None


def _get_shared_qdrant_client():
    global _shared_qdrant_client
    if _shared_qdrant_client is None:
        from qdrant_client import QdrantClient
        _shared_qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
    return _shared_qdrant_client


# ---------------------------------------------------------------------------
# Hard-filter criteria definitions
# ---------------------------------------------------------------------------

HARD_FILTER_CRITERIA = [
    "cancer_type",
    "histology",
    "stage",
    "prior_therapies",
    "biomarkers",
    "disease_status",
    "surgical_candidacy",
    "study_exclusions_violated",
]

#: Axes whose verdict=="MISMATCH" causes immediate hard removal of the
#: study (study is excluded from the result set). Verdict=="MATCH" on
#: `study_exclusions_violated` is also a hard drop — see eligibility
#: prompt for the inverted polarity.
#:
#: Includes the four oncologist-consensus axes (cancer_type, histology,
#: stage, prior_therapies) plus disease_status and surgical_candidacy
#: which are already structurally required for clinical eligibility.
#: Biomarkers stay SOFT because the spec explicitly carves out
#: "if available in the trials" — older studies often don't report
#: molecular subtype.
#:
#: MISMATCH false-positive protection lives in three places:
#:   - LLM prompt rules 6, 8, 12 (strict MISMATCH usage; COMPATIBLE for
#:     range / subsumption)
#:   - _check_stage_subsumption() post-processes stage verdicts to demote
#:     range-matching MISMATCH → COMPATIBLE
#:   - subsite-hierarchy demotion catches cancer_type over-eagerness
#: When a study survives all three and still has a hard-drop axis
#: MISMATCH, the rejection is clinically warranted.
HARD_DROP_AXES = {
    "cancer_type",
    "histology",
    "stage",
    "prior_therapies",
    "disease_status",
    "surgical_candidacy",
}
HARD_DROP_ON_MATCH_AXES = {"study_exclusions_violated"}


def _was_hard_dropped_for_axes(chunk: Dict[str, Any]) -> bool:
    """Return True if this chunk was removed because of a HARD_DROP axis.

    Used by the surgical Path A restoration logic to decide which removed
    chunks are eligible for restoration (none of these are) vs which
    represent overly-aggressive filtering that should be restored at
    reduced score when the bundle is otherwise empty.

    A chunk is considered hard-dropped if ANY of its eligibility verdicts
    hits one of:
      - HARD_DROP_AXES with verdict == "MISMATCH"
      - HARD_DROP_ON_MATCH_AXES with verdict == "MATCH" (inverted polarity)
    """
    verdicts = (chunk.get("patient_eligibility", {}) or {}).get("criteria_verdicts", {}) or {}
    for axis in HARD_DROP_AXES:
        if verdicts.get(axis) == "MISMATCH":
            return True
    for axis in HARD_DROP_ON_MATCH_AXES:
        if verdicts.get(axis) == "MATCH":
            return True
    return False

#: Minimum number of studies in a bundle before it's considered clinically
#: useful. When the eligibility filter + category_routing_suspect guard
#: leaves fewer than this many studies, the fallback restores the
#: least-mismatched removed studies at reduced scores.
MIN_STUDIES_FLOOR = 3


# ---------------------------------------------------------------------------
# Score weight constants (bugfix: patient-study-match-scoring-fix)
# ---------------------------------------------------------------------------

#: Penalty per non-cancer_type core axis with verdict MISMATCH.
#: Total penalty = CORE_MISMATCH_PENALTY × count(mismatched non-cancer_type core axes).
#: Final score is floored at 1 (never 0) after penalty application.
CORE_MISMATCH_PENALTY = 10

#: Per-axis score contributions for core eligibility axes.
#: Keys: verdict → points awarded.  cancer_type MISMATCH is a hard drop
#: (handled separately), so its MISMATCH entry is 0 here for completeness.
CORE_AXIS_WEIGHTS: Dict[str, Dict[str, int]] = {
    "cancer_type":       {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "histology":         {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "stage":             {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "prior_therapies":   {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "biomarkers":        {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    # disease_status MISMATCH is a hard drop (handled separately).
    # MATCH/COMPATIBLE values follow the standard 15/10 schedule so a
    # correctly-aligned disease state still boosts the study's score.
    "disease_status":    {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    # surgical_candidacy MISMATCH is a hard drop. Studies that don't
    # restrict on surgical candidacy → NOT_AVAILABLE → 0 points (no
    # penalty, no boost). A study requiring surgical candidates that
    # the patient cannot meet → MISMATCH → hard drop.
    "surgical_candidacy": {"MATCH": 15, "COMPATIBLE": 10, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    # study_exclusions_violated has INVERTED polarity: MATCH means
    # patient violates an exclusion criterion → hard drop. Verdict
    # values: MATCH (violation found), NOT_AVAILABLE (no exclusion
    # criteria documented, or none apply to this patient).
    "study_exclusions_violated": {"MATCH": 0, "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": 0},
}

#: Valid verdict values for per-axis eligibility verdicts.
VALID_ELIGIBILITY_VERDICTS = {"MATCH", "COMPATIBLE", "MISMATCH", "NOT_AVAILABLE"}

#: Per-axis score contributions for secondary (ranking-only) axes.
#: Keys: verdict → points awarded.  Secondary axes never trigger removal.
SECONDARY_AXIS_WEIGHTS: Dict[str, Dict[str, int]] = {
    "performance_status":        {"MATCH": 8,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": -4},
    "age_range":                 {"MATCH": 6,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": -3},
    "modality":                  {"MATCH": 5,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "metastatic_sites":          {"MATCH": 5,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": -2},
    "comorbidity_compatibility": {"MATCH": 4,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": -4},
    "gender":                    {"MATCH": 4,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": -2},
    "study_phase":               {"MATCH": 3,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "landmark_trial_status":     {"MATCH": 3,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": 0},
    "recency":                   {"MATCH": 2,  "COMPATIBLE": 0, "NOT_AVAILABLE": 0, "MISMATCH": 0},
}


#: Subsite enumeration caps keyed by SITE_KEYWORDS capitalized site_key.
#: These phrases are the coarse umbrella term for a site or a post-treatment
#: / procedural term — neither is a meaningful tumor subsite. Including
#: them in `cancer_subsite` would either duplicate the `cancer_type` label
#: or surface surgery history where tumor location is expected. Stripped
#: during subsite extraction.
_UMBRELLA_TERMS_BY_SITE: Dict[str, set] = {
    "Breast":    {"breast", "breast cancer", "mastectomy", "lumpectomy"},
    "Lung":      {"lung", "lung cancer", "pulmonary", "lobectomy"},
    "Prostate":  {"prostate", "prostate cancer", "prostatic", "prostatectomy"},
    "GI":        {"colorectal", "colon", "rectal", "rectum", "stomach",
                  "hemicolectomy", "colectomy", "gastrectomy"},
    "H&N":       {"head and neck", "h&n", "hnscc", "scchn", "hncscc",
                  # procedural / post-treatment terms — not tumor subsites
                  "glossectomy", "laryngectomy", "neck dissection",
                  "radial forearm free flap", "rfff", "alt flap", "pec flap",
                  "maxillectomy", "mandibulectomy"},
    "CNS":       {"brain", "cns", "central nervous system", "craniotomy"},
    "Cutaneous": {"skin", "melanoma", "cutaneous"},
    "GYN":       {"gynecologic", "gyn", "hysterectomy"},
    "GU":        {"bladder", "gu", "genitourinary", "cystectomy", "nephrectomy"},
}


#: Inverse of _SITE_KEY_TO_CANCER_TYPE_LABEL (defined below after the dict).
_CANCER_TYPE_LABEL_TO_SITE_KEY_CAP: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Patient context extraction (unchanged from original)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical regex tables — exported so site-aware disambiguation helpers can
# iterate them with re.finditer() and record all match positions, not just
# the first one.
# ---------------------------------------------------------------------------

#: Ordered list of (compiled_pattern, cancer_type_label) — cancer_type
#: output uses human-readable labels that map to ontology site keys via
#: `ontology_loader.CANCER_TYPE_LABEL_TO_SITE_KEY`.
#:
#: Patterns accept "<site> cancer", "<site> carcinoma", "<site> adenocarcinoma",
#: "<site> tumor", "<site> malignancy" AND short-form tokens (NSCLC, SCLC,
#: GBM, HCC, etc.) so they catch phrases like "lung adenocarcinoma" or
#: "high-grade glioblastoma" that the old `(lung)\s*cancer` form missed.
#: We still require a cancer-context word to avoid false positives on
#: common English words like "lung" or "colon" standing alone.
_CT_TAIL = r'(?:cancer|carcinoma|tumou?r|malignancy|neoplasm|adenocarcinoma)'


#: Fallback map from `infer_site_key()` return values (the site_key strings
#: used by enhanced_rag_service.SITE_KEYWORDS) → canonical cancer_type
#: labels used throughout patient_eligibility_boost_service.
#:
#: Used by `_synthesize_cancer_type_from_site()` when the direct
#: CANCER_TYPE_PATTERNS regex loop doesn't find an explicit "<site> cancer"
#: phrase but the raw query mentions an anatomical subsite that
#: infer_site_key() recognizes (e.g. "maxilla" → "H&N" → "head and neck cancer").
#:
#: Keep these labels in lockstep with the strings produced by
#: CANCER_TYPE_PATTERNS above — cancer_type_label_to_site_key() and the
#: downstream ontology lookups expect the same canonical form.
_SITE_KEY_TO_CANCER_TYPE_LABEL: Dict[str, str] = {
    "Breast":    "breast cancer",
    "Lung":      "lung cancer",
    "Prostate":  "prostate cancer",
    "GI":        "colorectal cancer",
    "H&N":       "head and neck cancer",
    "CNS":       "brain cancer",
    "Cutaneous": "melanoma",
    "GYN":       "gynecologic cancer",
    "GU":        "bladder cancer",
    "Sarcoma":   "sarcoma",
    "Lymphoma":  "lymphoma",
    "Thyroid":   "thyroid cancer",
    "Peds":      "pediatric cancer",
    # NOTE: "Radiotherapy&Oncology" is the generic fallback category —
    # don't synthesize a cancer_type from it, that would wrongly imply
    # we detected a specific site when we didn't.
}

# Populate the inverse lookup now that the forward map is defined.
_CANCER_TYPE_LABEL_TO_SITE_KEY_CAP.update(
    {v: k for k, v in _SITE_KEY_TO_CANCER_TYPE_LABEL.items()}
)


# ---------------------------------------------------------------------------
# Biomarker hard-exclusion (Phase 3 — Hard Eligibility Gate)
# ---------------------------------------------------------------------------

#: Contradictory polarity pairs. When the patient has one polarity and the
#: study explicitly requires the other, the study is hard-excluded.
_BIOMARKER_CONTRADICTIONS = {
    ("mutant", "wild-type"), ("wild-type", "mutant"),
    ("positive", "negative"), ("negative", "positive"),
}


def _biomarker_hard_exclusion(
    patient_biomarkers: List[CanonicalBiomarker],
    study_biomarker_status: Dict[str, str],
) -> bool:
    """Return True if study should be hard-excluded based on biomarker mismatch.

    Only triggers when BOTH sides have explicit polarity declarations
    and they are contradictory (mutant vs wild-type, positive vs negative).

    Args:
        patient_biomarkers: Canonical biomarkers extracted from the patient query.
        study_biomarker_status: Mapping of biomarker canonical_id -> polarity
            from the study's metadata (e.g. {"EGFR": "wild-type"}).

    Returns:
        True if a contradictory polarity pair is found, False otherwise.
    """
    for pb in patient_biomarkers:
        if not pb.polarity:
            continue
        study_polarity = study_biomarker_status.get(pb.canonical_id)
        if not study_polarity:
            continue
        if (pb.polarity, study_polarity) in _BIOMARKER_CONTRADICTIONS:
            return True
    return False


def check_biomarker_hard_exclusion(
    patient_biomarkers: List[CanonicalBiomarker],
    study_biomarker_status: Dict[str, str],
    study_id: str = "",
) -> bool:
    """Feature-flag-gated wrapper around ``_biomarker_hard_exclusion``.

    When ``settings.enable_hard_gate`` is False the check is skipped and
    the function returns False (no exclusion).  Exclusions are logged with
    the ``[HardGate]`` prefix per observability requirements.

    Args:
        patient_biomarkers: Canonical biomarkers from the patient query.
        study_biomarker_status: Biomarker polarity map from the study.
        study_id: Optional study identifier for logging.

    Returns:
        True if the study should be excluded, False otherwise.
    """
    if not settings.enable_hard_gate:
        return False

    try:
        excluded = _biomarker_hard_exclusion(patient_biomarkers, study_biomarker_status)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[HardGate] Error in biomarker hard-exclusion for study={study_id}: {e}")
        return False

    if excluded:
        # Build a short reason string for the log line
        reasons = []
        for pb in patient_biomarkers:
            if not pb.polarity:
                continue
            sp = study_biomarker_status.get(pb.canonical_id)
            if sp and (pb.polarity, sp) in _BIOMARKER_CONTRADICTIONS:
                reasons.append(f"{pb.canonical_id} patient={pb.polarity} study={sp}")
        reason_str = "; ".join(reasons) if reasons else "contradictory polarity"
        print(f"[HardGate] study={study_id} verdict=EXCLUDED reason=biomarker_contradiction ({reason_str})")

    return excluded


# ---------------------------------------------------------------------------
# Trajectory hard-exclusion (Phase 3 — Hard Eligibility Gate)
# ---------------------------------------------------------------------------

#: Trajectory categories grouped by treatment experience.
#: "naive" = patient has NOT received prior systemic therapy for this disease.
#: "experienced" = patient HAS received prior systemic therapy.
#: "adjuvant" / "neoadjuvant" are specific treatment settings that don't
#: directly contradict either group — they are excluded from contradiction
#: checks because a patient who received adjuvant therapy may still be
#: treatment-naive in the metastatic setting.
_TRAJECTORY_NAIVE: frozenset = frozenset({"treatment_naive", "first_line"})
_TRAJECTORY_EXPERIENCED: frozenset = frozenset({"second_line", "later_line", "refractory"})


def _trajectory_hard_exclusion(
    patient_trajectory: str,
    study_trajectory: str,
) -> bool:
    """Return True if the patient's disease trajectory contradicts the study's.

    A contradiction exists when one side is treatment-naive/first-line and
    the other requires second-line/later-line/refractory, or vice versa.

    Adjuvant/neoadjuvant trajectories are NOT considered contradictory with
    either group — they represent specific treatment settings rather than
    lines of therapy.

    Args:
        patient_trajectory: Normalized trajectory tag for the patient
            (e.g. "treatment_naive", "second_line", "refractory").
        study_trajectory: Normalized trajectory tag required by the study.

    Returns:
        True if the trajectories are contradictory, False otherwise.
    """
    if not patient_trajectory or not study_trajectory:
        return False

    pt = patient_trajectory.strip().lower()
    st = study_trajectory.strip().lower()

    # Same category → no contradiction
    if pt == st:
        return False

    patient_is_naive = pt in _TRAJECTORY_NAIVE
    patient_is_experienced = pt in _TRAJECTORY_EXPERIENCED
    study_is_naive = st in _TRAJECTORY_NAIVE
    study_is_experienced = st in _TRAJECTORY_EXPERIENCED

    # Contradiction: naive patient vs experienced-only study
    if patient_is_naive and study_is_experienced:
        return True

    # Contradiction: experienced patient vs naive-only study
    if patient_is_experienced and study_is_naive:
        return True

    return False


def check_trajectory_hard_exclusion(
    patient_trajectory: str,
    study_trajectory: str,
    study_id: str = "",
) -> bool:
    """Feature-flag-gated wrapper around ``_trajectory_hard_exclusion``.

    When ``settings.enable_hard_gate`` is False the check is skipped and
    the function returns False (no exclusion).  Exclusions are logged with
    the ``[HardGate]`` prefix per observability requirements.

    Args:
        patient_trajectory: Normalized trajectory tag for the patient.
        study_trajectory: Normalized trajectory tag required by the study.
        study_id: Optional study identifier for logging.

    Returns:
        True if the study should be excluded, False otherwise.
    """
    if not settings.enable_hard_gate:
        return False

    try:
        excluded = _trajectory_hard_exclusion(patient_trajectory, study_trajectory)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[HardGate] Error in trajectory hard-exclusion for study={study_id}: {e}")
        return False

    if excluded:
        print(
            f"[HardGate] study={study_id} verdict=EXCLUDED "
            f"reason=trajectory_contradiction "
            f"(patient={patient_trajectory} study={study_trajectory})"
        )

    return excluded


# ---------------------------------------------------------------------------
# Stage subsumption (Phase 3 — Over-Aggressive Stage Filtering Fix)
# ---------------------------------------------------------------------------

#: Ordered Roman numeral stages for range comparison.
_ROMAN_STAGE_ORDER: List[str] = ["I", "II", "III", "IV"]

#: Map from Roman numeral to ordinal index for range checks.
_ROMAN_TO_INDEX: Dict[str, int] = {s: i for i, s in enumerate(_ROMAN_STAGE_ORDER)}

#: Pattern to parse a stage string into its base Roman numeral and optional suffix.
_STAGE_PARSE_RE = re.compile(
    r'^(?:stage\s*)?'           # optional "Stage " prefix
    r'(I{1,3}V?|IV)'           # base Roman numeral: I, II, III, IV
    r'([A-Ca-c]?)$',           # optional sub-stage suffix: A, B, C
    re.IGNORECASE,
)

#: Pattern to detect a stage range like "II-III" or "Stage I-II".
_STAGE_RANGE_RE = re.compile(
    r'^(?:stage\s*)?'
    r'(I{1,3}V?|IV)'           # start of range
    r'\s*[-–]\s*'              # dash separator
    r'(I{1,3}V?|IV)$',        # end of range
    re.IGNORECASE,
)


def _parse_stage(stage_str: str) -> Optional[Tuple[str, str]]:
    """Parse a stage string into (base_roman, suffix).

    Returns None if the string is not a recognizable stage.

    Examples:
        "IIIA" → ("III", "A")
        "II"   → ("II", "")
        "Stage IV" → ("IV", "")
    """
    if not stage_str:
        return None
    m = _STAGE_PARSE_RE.match(stage_str.strip())
    if not m:
        return None
    return (m.group(1).upper(), m.group(2).upper())


def _parse_stage_range(stage_str: str) -> Optional[Tuple[str, str]]:
    """Parse a stage range string into (start_roman, end_roman).

    Returns None if the string is not a recognizable range.

    Examples:
        "II-III" → ("II", "III")
        "Stage I-II" → ("I", "II")
    """
    if not stage_str:
        return None
    m = _STAGE_RANGE_RE.match(stage_str.strip())
    if not m:
        return None
    return (m.group(1).upper(), m.group(2).upper())


def _check_stage_subsumption(patient_stage: str, study_stage: str) -> str:
    """Determine stage compatibility between patient and study.

    Supports three kinds of match:
    1. Exact match (including case-insensitive): "III" vs "III" → "MATCH"
    2. Sub-stage subsumption: patient "IIIA" is subsumed by study "III" → "COMPATIBLE"
    3. Range containment: patient "II" falls within study "II-III" → "COMPATIBLE"

    Returns one of: "MATCH", "COMPATIBLE", "MISMATCH".

    Args:
        patient_stage: The patient's stage string (e.g., "IIIA", "II", "Stage IV").
        study_stage: The study's stage requirement (e.g., "III", "II-III", "Stage IVA").
    """
    if not patient_stage or not study_stage:
        return "MISMATCH"

    ps = patient_stage.strip()
    ss = study_stage.strip()

    # Normalize for case-insensitive exact match
    if ps.upper() == ss.upper():
        return "MATCH"

    # Strip optional "Stage " prefix for comparison
    ps_clean = re.sub(r'^stage\s*', '', ps, flags=re.IGNORECASE).strip()
    ss_clean = re.sub(r'^stage\s*', '', ss, flags=re.IGNORECASE).strip()

    if ps_clean.upper() == ss_clean.upper():
        return "MATCH"

    # --- Check if study_stage is a range (e.g., "II-III") ---
    range_parsed = _parse_stage_range(ss)
    if range_parsed:
        range_start, range_end = range_parsed
        start_idx = _ROMAN_TO_INDEX.get(range_start)
        end_idx = _ROMAN_TO_INDEX.get(range_end)
        if start_idx is not None and end_idx is not None:
            # Parse patient stage to get its base Roman numeral
            patient_parsed = _parse_stage(ps)
            if patient_parsed:
                patient_base = patient_parsed[0]
                patient_idx = _ROMAN_TO_INDEX.get(patient_base)
                if patient_idx is not None and start_idx <= patient_idx <= end_idx:
                    return "COMPATIBLE"
        return "MISMATCH"

    # --- Check sub-stage subsumption ---
    patient_parsed = _parse_stage(ps)
    study_parsed = _parse_stage(ss)

    if patient_parsed and study_parsed:
        patient_base, patient_suffix = patient_parsed
        study_base, study_suffix = study_parsed

        # Patient sub-stage subsumed by study parent stage:
        # e.g., patient "IIIA" matches study "III" (patient has suffix, study doesn't)
        if patient_base == study_base and patient_suffix and not study_suffix:
            return "COMPATIBLE"

        # Study sub-stage subsumed by patient parent stage:
        # e.g., patient "III" matches study "IIIA" (study has suffix, patient doesn't)
        if patient_base == study_base and not patient_suffix and study_suffix:
            return "COMPATIBLE"

    return "MISMATCH"


def check_stage_subsumption(
    patient_stage: str,
    study_stage: str,
    study_id: str = "",
) -> str:
    """Feature-flag-gated wrapper around ``_check_stage_subsumption``.

    When ``settings.enable_hard_gate`` is False the check is skipped and
    the function returns "MISMATCH" (existing strict behavior).

    Args:
        patient_stage: The patient's stage string.
        study_stage: The study's stage requirement.
        study_id: Optional study identifier for logging.

    Returns:
        "MATCH", "COMPATIBLE", or "MISMATCH".
    """
    if not settings.enable_hard_gate:
        # Fall back to strict exact-match behavior
        if patient_stage and study_stage:
            ps = re.sub(r'^stage\s*', '', patient_stage.strip(), flags=re.IGNORECASE).upper()
            ss = re.sub(r'^stage\s*', '', study_stage.strip(), flags=re.IGNORECASE).upper()
            return "MATCH" if ps == ss else "MISMATCH"
        return "MISMATCH"

    try:
        result = _check_stage_subsumption(patient_stage, study_stage)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[HardGate] Error in stage subsumption for study={study_id}: {e}")
        return "MISMATCH"

    if result == "COMPATIBLE":
        print(
            f"[HardGate] study={study_id} verdict=COMPATIBLE "
            f"reason=stage_subsumption "
            f"(patient={patient_stage} study={study_stage})"
        )

    return result


def _extract_cancer_subsite(query: str, cancer_type_label: Optional[str]) -> Optional[str]:
    """Scan the raw query for fine-grained subsite phrases registered in
    SITE_KEYWORDS for the patient's cancer site. Returns a ' / '-joined
    list of the most specific matches, or None when no subsite signal is
    found.

    The coarse cancer_type label (e.g. "head and neck cancer") throws away
    subsite detail that the literature uses for eligibility ("oral cavity
    only", "laryngeal SCC only"). Preserving it here lets the downstream
    LLM eligibility prompt — and the keyword-flat Qdrant filter — actually
    discriminate between subsite-specific and umbrella-cohort studies.
    """
    if not query or not cancer_type_label:
        return None
    site_key_cap = _CANCER_TYPE_LABEL_TO_SITE_KEY_CAP.get(cancer_type_label)
    if not site_key_cap:
        return None
    try:
        from src.api.services.enhanced_rag_service import SITE_KEYWORDS
    except Exception:
        return None
    umbrella = _UMBRELLA_TERMS_BY_SITE.get(site_key_cap, set())
    keywords: List[str] = []
    for key, kws in SITE_KEYWORDS:
        if key == site_key_cap:
            keywords = list(kws or [])
            break
    if not keywords:
        return None

    ql = query.lower()
    hits: List[str] = []
    for kw in keywords:
        kw_l = kw.lower().strip()
        if not kw_l or kw_l in umbrella:
            continue
        # Word-boundary match so "lip" doesn't fire on "lipoma".
        if re.search(r'\b' + re.escape(kw_l) + r'\b', ql):
            hits.append(kw_l)

    if not hits:
        return None

    # Deduplicate and prefer more specific phrases (longer or containing
    # more-general matches). E.g. if both "tongue" and "oral tongue"
    # matched, keep only "oral tongue".
    unique = sorted(set(hits), key=len, reverse=True)
    kept: List[str] = []
    for term in unique:
        if any(term != t and term in t for t in kept):
            continue
        kept.append(term)
    # Cap at 3 for summary readability.
    return " / ".join(kept[:3])


def _synthesize_cancer_type_from_site(query: str) -> Optional[str]:
    """
    Fallback cancer_type detection when CANCER_TYPE_PATTERNS doesn't fire.

    Calls `infer_site_key()` (which has a much richer subsite keyword
    table than the narrow <site>+TAIL regex patterns) and maps the
    returned site_key to a canonical cancer_type label.

    Returns None on:
      - Empty / trivial query
      - Generic "Radiotherapy&Oncology" fallback (not a real site match)
      - Unknown site_key
      - Import failure (defensive — never crash the extractor)
    """
    if not query or len(query) < 3:
        return None
    try:
        from src.api.services.enhanced_rag_service import infer_site_key
    except Exception:
        return None
    try:
        inferred_site = infer_site_key(query)
    except Exception:
        return None
    if not inferred_site or inferred_site == "Radiotherapy&Oncology":
        return None
    return _SITE_KEY_TO_CANCER_TYPE_LABEL.get(inferred_site)


CANCER_TYPE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(rf'\b(?:breast|mammary)\s+{_CT_TAIL}\b', re.I), "breast cancer"),
    (re.compile(
        rf'\b(?:lung|pulmonary)\s+{_CT_TAIL}\b'
        r'|\blung\s+(?:adeno|squamous|small[-\s]?cell)\w*\b'
        r'|\bNSCLC\b|\bSCLC\b|\bnon[-\s]?small[-\s]?cell\s+lung\b',
        re.I,
    ), "lung cancer"),
    (re.compile(rf'\b(?:prostate|prostatic)\s+{_CT_TAIL}\b', re.I), "prostate cancer"),
    (re.compile(
        rf'\b(?:colorectal|colon|rectal)\s+{_CT_TAIL}\b'
        r'|\bCRC\b',
        re.I,
    ), "colorectal cancer"),
    (re.compile(
        # ── Umbrella terms ───────────────────────────────────────────
        r'\bhead\s*and\s*neck\b|\bh&n\b|\bHNSCC\b|\bHNCSCC\b|\bSCCHN\b'
        r'|\bOPSCC\b|\bNPSCC\b|\bLSCC\b|\bCSCC\b'
        r'|\boropharyngeal\b|\blaryngeal\b|\bnasopharyngeal\b|\bhypopharyngeal\b'
        r'|\boral\s+(?:tongue|cavity)\b|\boral\s+cancer\b|\bNPC\b'
        # ── Oral cavity subsites ─────────────────────────────────────
        r'|\bmaxilla(?:ry)?\b|\bmaxillectomy\b'
        r'|\bmandible\b|\bmandibular\b|\bmandibulectomy\b'
        r'|\bbuccal(?:\s+mucosa)?\b'
        r'|\bhard\s+palate\b|\bsoft\s+palate\b'
        r'|\bfloor\s+of\s+mouth\b|\bFOM\b'
        r'|\bretromolar\b|\bgingiva\b|\balveolar\s+ridge\b'
        r'|\bbase\s+of\s+tongue\b|\btongue\s+base\b|\bBOT\b'
        # ── Pharyngeal / laryngeal / salivary ────────────────────────
        r'|\bpy?riform(?:\s+sinus)?\b|\bpostcricoid\b'
        r'|\bglottic\b|\bglottis\b|\bsupraglottic\b|\bsubglottic\b'
        r'|\bvocal\s+(?:cord|fold)\b|\baryepiglottic\b|\bepiglott\w*\b'
        r'|\bparotid\b|\bsubmandibular\s+gland\b|\bsublingual\s+gland\b'
        r'|\bmucoepidermoid\b|\badenoid\s+cystic\b'
        # ── Sinonasal ───────────────────────────────────────────────
        r'|\bmaxillary\s+sinus\b|\bethmoid(?:\s+sinus)?\b'
        r'|\bsphenoid\s+sinus\b|\bnasal\s+cavity\b|\bnasal\s+septum\b'
        # ── H&N-specific procedures ─────────────────────────────────
        r'|\bglossectomy\b|\blaryngectomy\b|\bneck\s+dissection\b'
        r'|\bradial\s+forearm\s+free\s+flap\b|\bRFFF\b',
        re.I,
    ), "head and neck cancer"),
    (re.compile(
        r'\bglioblastoma\b|\bGBM\b|\bglioma\b'
        r'|\bbrain\s+(?:cancer|tumou?r|malignancy)\b'
        r'|\bastrocytoma\b|\boligodendroglioma\b|\bmedulloblastoma\b',
        re.I,
    ), "brain cancer"),
    (re.compile(r'\bmelanoma\b|\bskin\s+cancer\b', re.I), "melanoma"),
    (re.compile(
        rf'\b(?:pancreatic|pancreas)\s+{_CT_TAIL}\b'
        r'|\bPDAC\b',
        re.I,
    ), "pancreatic cancer"),
    (re.compile(rf'\b(?:ovarian|ovary)\s+{_CT_TAIL}\b', re.I), "ovarian cancer"),
    (re.compile(rf'\b(?:cervical|cervix)\s+{_CT_TAIL}\b', re.I), "cervical cancer"),
    (re.compile(
        rf'\b(?:esophageal|esophagus|oesophageal)\s+{_CT_TAIL}\b'
        r'|\besophageal\s+(?:squamous|adeno)\w*\b',
        re.I,
    ), "esophageal cancer"),
    (re.compile(rf'\b(?:gastric|stomach)\s+{_CT_TAIL}\b', re.I), "gastric cancer"),
    (re.compile(
        rf'\b(?:hepatocellular|liver)\s+{_CT_TAIL}\b'
        r'|\bHCC\b',
        re.I,
    ), "liver cancer"),
    (re.compile(
        rf'\b(?:renal|kidney)\s+{_CT_TAIL}\b'
        r'|\bRCC\b',
        re.I,
    ), "renal cancer"),
    (re.compile(rf'\b(?:bladder|urothelial)\s+{_CT_TAIL}\b', re.I), "bladder cancer"),
]

#: Ordered list of (compiled_pattern, canonical_histology). Canonical names
#: must line up with the strings registered in cancer_type_ontology.json
#: `histologies` arrays so site-aware disambiguation can cross-check them
#: via `ontology_loader.is_histology_plausible_for_site`.
HISTOLOGY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Check MORE-SPECIFIC patterns first so "invasive ductal" wins over "ductal"
    (re.compile(r'\binvasive\s+ductal\s+carcinoma\b', re.I), "invasive ductal carcinoma"),
    (re.compile(r'\binvasive\s+lobular\s+carcinoma\b', re.I), "invasive lobular carcinoma"),
    (re.compile(r'\bhepatocellular\s+carcinoma\b|\bhcc\b', re.I), "hepatocellular carcinoma"),
    (re.compile(r'\bpancreatic\s+ductal\s+adenocarcinoma\b|\bpdac\b', re.I), "pancreatic ductal adenocarcinoma"),
    (re.compile(r'\bmucoepidermoid\s+carcinoma\b', re.I), "mucoepidermoid carcinoma"),
    (re.compile(r'\badenoid\s+cystic\s+carcinoma\b', re.I), "adenoid cystic carcinoma"),
    (re.compile(r'\burothelial\s+carcinoma\b|\btransitional\s+cell\s+carcinoma\b', re.I), "urothelial carcinoma"),
    (re.compile(r'\bclear\s+cell\s+(?:renal|rcc)\b', re.I), "clear cell renal cell carcinoma"),
    (re.compile(r'\bpapillary\s+thyroid\s+carcinoma\b', re.I), "papillary thyroid carcinoma"),
    (re.compile(r'\bmerkel\s+cell\s+carcinoma\b', re.I), "merkel cell carcinoma"),
    (re.compile(r'\bbasal\s+cell\s+carcinoma\b|\bbcc\b', re.I), "basal cell carcinoma"),
    (re.compile(r'\bnon[-\s]?small[-\s]?cell\b', re.I), "non-small cell"),
    (re.compile(r'\bsmall\s+cell\s+carcinoma\b', re.I), "small cell carcinoma"),
    (re.compile(r'\bsmall\s+cell\b', re.I), "small cell"),
    (re.compile(r'\blarge\s+cell\s+carcinoma\b', re.I), "large cell carcinoma"),
    (re.compile(r'\bsquamous\s*cell\s+carcinoma\b|\bscchn\b|\bscc\b|\bsquamous\s*cell\b', re.I), "squamous cell carcinoma"),
    (re.compile(r'\badenocarcinoma\b|\badeno\b', re.I), "adenocarcinoma"),
    (re.compile(r'\bmelanoma\b', re.I), "melanoma"),
    (re.compile(r'\bductal\s+carcinoma(?:\s+in\s+situ)?\b|\bdcis\b', re.I), "ductal carcinoma"),
    (re.compile(r'\blobular\s+carcinoma(?:\s+in\s+situ)?\b|\blcis\b', re.I), "lobular carcinoma"),
    (re.compile(r'\bglioblastoma\b|\bgbm\b', re.I), "glioblastoma"),
    (re.compile(r'\bastrocytoma\b', re.I), "astrocytoma"),
    (re.compile(r'\boligodendroglioma\b', re.I), "oligodendroglioma"),
    (re.compile(r'\bseminoma\b', re.I), "seminoma"),
    (re.compile(r'\bdlbcl\b|\bdiffuse\s+large\s+b[- ]cell\s+lymphoma\b', re.I), "diffuse large b-cell lymphoma"),
    (re.compile(r'\bfollicular\s+lymphoma\b', re.I), "follicular lymphoma"),
    (re.compile(r'\bhodgkin\s+lymphoma\b', re.I), "hodgkin lymphoma"),
    (re.compile(r'\bgist\b|\bgastrointestinal\s+stromal\s+tumou?r\b', re.I), "gist"),
    # Gynecologic histologies
    (re.compile(r'\bhigh[-\s]grade\s+serous\s+(?:carcinoma|cancer)?\b', re.I), "high-grade serous carcinoma"),
    (re.compile(r'\blow[-\s]grade\s+serous\s+(?:carcinoma|cancer)?\b', re.I), "low-grade serous carcinoma"),
    (re.compile(r'\bserous\s+(?:carcinoma|cancer|ovarian|endometrial)\b', re.I), "serous carcinoma"),
    (re.compile(r'\bendometrioid\s+(?:carcinoma|cancer|adenocarcinoma)?\b', re.I), "endometrioid carcinoma"),
    (re.compile(r'\bclear\s+cell\s+carcinoma\b', re.I), "clear cell carcinoma"),
]

#: Ordered list of (compiled_pattern, canonical_surgery).
SURGERY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Head & neck
    (re.compile(r'\bpartial\s+glossectomy\b', re.I), "partial glossectomy"),
    (re.compile(r'\btotal\s+glossectomy\b', re.I), "total glossectomy"),
    (re.compile(r'\bglossectomy\b', re.I), "glossectomy"),
    (re.compile(r'\bradical\s+neck\s+dissection\b', re.I), "radical neck dissection"),
    (re.compile(r'\bmodified\s+radical\s+neck\s+dissection\b', re.I), "modified radical neck dissection"),
    (re.compile(r'\bselective\s+neck\s+dissection\b', re.I), "selective neck dissection"),
    (re.compile(r'\bneck\s+dissection\b', re.I), "neck dissection"),
    (re.compile(r'\bmaxillectomy\b', re.I), "maxillectomy"),
    (re.compile(r'\bmandibulectomy\b', re.I), "mandibulectomy"),
    (re.compile(r'\btotal\s+laryngectomy\b', re.I), "total laryngectomy"),
    (re.compile(r'\blaryngectomy\b', re.I), "laryngectomy"),
    (re.compile(r'\bparotidectomy\b', re.I), "parotidectomy"),
    (re.compile(r'\bradial\s+forearm\s+free\s+flap\b|\brfff\b', re.I), "radial forearm free flap"),
    (re.compile(r'\banterolateral\s+thigh\s+flap\b|\balt\s+flap\b', re.I), "anterolateral thigh flap"),
    (re.compile(r'\bfibula\s+free\s+flap\b', re.I), "fibula free flap"),
    (re.compile(r'\bfree\s+flap\s+reconstruction\b', re.I), "free flap reconstruction"),
    (re.compile(r'\bstsg\b|\bsplit[-\s]thickness\s+skin\s+graft\b', re.I), "stsg"),
    # Breast
    (re.compile(r'\bradical\s+mastectomy\b', re.I), "radical mastectomy"),
    (re.compile(r'\bmastectomy\b', re.I), "mastectomy"),
    (re.compile(r'\blumpectomy\b', re.I), "lumpectomy"),
    (re.compile(r'\bbreast[- ]conserving\s+(?:surgery|therapy)\b|\bbcs\b', re.I), "breast-conserving surgery"),
    (re.compile(r'\bsentinel\s+lymph\s+node\s+biopsy\b|\bslnb\b', re.I), "sentinel lymph node biopsy"),
    (re.compile(r'\baxillary\s+lymph\s+node\s+dissection\b|\balnd\b', re.I), "axillary lymph node dissection"),
    # Lung
    (re.compile(r'\blobectomy\b', re.I), "lobectomy"),
    (re.compile(r'\bpneumonectomy\b', re.I), "pneumonectomy"),
    (re.compile(r'\bwedge\s+resection\b', re.I), "wedge resection"),
    (re.compile(r'\bsegmentectomy\b', re.I), "segmentectomy"),
    (re.compile(r'\bvats\b|\bvideo[- ]assisted\s+thoracoscopic\s+surgery\b', re.I), "vats"),
    # GI
    (re.compile(r'\bextended\s+right\s+hemicolectomy\b', re.I), "extended right hemicolectomy"),
    (re.compile(r'\bright\s+hemicolectomy\b', re.I), "right hemicolectomy"),
    (re.compile(r'\bleft\s+hemicolectomy\b', re.I), "left hemicolectomy"),
    (re.compile(r'\bhemicolectomy\b', re.I), "hemicolectomy"),
    (re.compile(r'\bcolectomy\b', re.I), "colectomy"),
    (re.compile(r'\bsigmoidectomy\b', re.I), "sigmoidectomy"),
    (re.compile(r'\blow\s+anterior\s+resection\b|\blar\b', re.I), "low anterior resection"),
    (re.compile(r'\babdominoperineal\s+resection\b|\bapr\b', re.I), "abdominoperineal resection"),
    (re.compile(r'\bproctectomy\b', re.I), "proctectomy"),
    (re.compile(r'\btotal\s+gastrectomy\b', re.I), "total gastrectomy"),
    (re.compile(r'\bsubtotal\s+gastrectomy\b', re.I), "subtotal gastrectomy"),
    (re.compile(r'\bgastrectomy\b', re.I), "gastrectomy"),
    (re.compile(r'\besophagectomy\b', re.I), "esophagectomy"),
    (re.compile(r'\bwhipple\b|\bpancreaticoduodenectomy\b', re.I), "whipple"),
    (re.compile(r'\bdistal\s+pancreatectomy\b', re.I), "distal pancreatectomy"),
    (re.compile(r'\bileostomy\s+reversal\b', re.I), "ileostomy reversal"),
    # GU
    (re.compile(r'\bradical\s+cystectomy\b', re.I), "radical cystectomy"),
    (re.compile(r'\bcystectomy\b', re.I), "cystectomy"),
    (re.compile(r'\bradical\s+nephrectomy\b', re.I), "radical nephrectomy"),
    (re.compile(r'\bpartial\s+nephrectomy\b', re.I), "partial nephrectomy"),
    (re.compile(r'\bnephrectomy\b', re.I), "nephrectomy"),
    (re.compile(r'\borchiectomy\b', re.I), "orchiectomy"),
    (re.compile(r'\bturbt\b', re.I), "turbt"),
    # Prostate
    (re.compile(r'\bradical\s+prostatectomy\b', re.I), "radical prostatectomy"),
    (re.compile(r'\bprostatectomy\b', re.I), "prostatectomy"),
    # Gyn
    (re.compile(r'\bradical\s+hysterectomy\b', re.I), "radical hysterectomy"),
    (re.compile(r'\bhysterectomy\b', re.I), "hysterectomy"),
    (re.compile(r'\bbilateral\s+salpingo[- ]oophorectomy\b|\bbso\b', re.I), "bilateral salpingo-oophorectomy"),
    (re.compile(r'\bdebulking\b|\bcytoreductive\s+surgery\b', re.I), "debulking"),
    # CNS
    (re.compile(r'\bcraniotomy\b', re.I), "craniotomy"),
]


#: Phrases that indicate a cancer mentioned AFTER them is a historical /
#: cured / remote finding, not the subject of the current case. Used by
#: `_pick_primary_cancer_type` to demote cancer_type matches that appear
#: inside a PMH / "history of" / "remote" segment.
HISTORICAL_CANCER_SIGNALS = re.compile(
    r'\bPMH\b'
    r'|\bpast\s+medical\s+history\b'
    r'|\bprior\s+(?:history|diagnosis|cancer|malignancy)\b'
    r'|\bhistory\s+of\b'
    r'|\bremote\b'
    r'|\bpreviously\s+(?:had|diagnosed|treated|cured)\b'
    r'|\bin\s+remission\b',
    re.I,
)

#: Window (in characters) before a cancer_type match in which we look for a
#: historical-signal phrase. 150 characters comfortably covers a typical PMH
#: list like "PMH HTN, Hep C, BPH, CKD, latent syphilis, transverse colon…"
_HISTORICAL_LOOKBACK_CHARS = 150


def _pick_primary_cancer_type(query: str) -> Tuple[Optional[str], Optional[int]]:
    """Find the cancer_type that is most likely the SUBJECT of the query.

    Unlike a simple first-match-wins scan, this:

      1. Collects every cancer_type pattern hit in the query (not just
         the first one).
      2. Flags each hit as historical if the 150 characters preceding
         it contain a PMH / "history of" / "remote" phrase.
      3. Prefers non-historical hits, then picks the LAST one in the
         text (clinical narratives tend to describe the active problem
         after the past-medical-history block).
      4. Falls back to the latest historical hit if no non-historical
         match exists — better to return something than nothing.

    Example — the canonical failing case from the audit log:

        "... PMH HTN, Hep C, BPH, CKD, latent syphilis, transverse colon
         adenocarcinoma ... and initial Stage II ... squamous cell
         carcinoma of the left oral tongue ..."

      • "colon adenocarcinoma"  → matches "colorectal cancer", HISTORICAL
        (preceded by "PMH" within 150 chars)
      • "oral tongue"           → matches "head and neck cancer", NOT historical

      → picks "head and neck cancer"
    """
    all_hits: List[Tuple[str, int, bool]] = []
    for pat, label in CANCER_TYPE_PATTERNS:
        for m in pat.finditer(query):
            pos = m.start()
            preceding = query[max(0, pos - _HISTORICAL_LOOKBACK_CHARS):pos]
            is_historical = bool(HISTORICAL_CANCER_SIGNALS.search(preceding))
            all_hits.append((label, pos, is_historical))

    if not all_hits:
        return None, None

    # Prefer non-historical hits
    non_hist = [(label, pos) for label, pos, is_hist in all_hits if not is_hist]
    if non_hist:
        # Among non-historical, the LATEST in the text wins (active cancer
        # is typically described after the PMH block in clinical narratives)
        label, pos = max(non_hist, key=lambda x: x[1])
        return label, pos

    # All matches were in historical segments — return the latest one
    label, pos, _ = max(all_hits, key=lambda x: x[1])
    return label, pos


def _find_all_matches(
    patterns: List[Tuple[re.Pattern, str]],
    query: str,
) -> List[Tuple[str, int]]:
    """Walk `patterns` against `query` and return every (canonical, position)
    match found. Dedups canonical names by keeping the earliest occurrence
    of each. Unlike first-match-wins, this records *every* distinct canonical
    name present in the text — the disambiguation layer then picks the right
    one using site plausibility + proximity."""
    seen: Dict[str, int] = {}
    for pat, canonical in patterns:
        for m in pat.finditer(query):
            # Keep the earliest position for each canonical name
            if canonical not in seen or m.start() < seen[canonical]:
                seen[canonical] = m.start()
    return [(name, pos) for name, pos in seen.items()]


def _disambiguate_to_site(
    candidates: List[Tuple[str, int]],
    site_key: Optional[str],
    anchor_pos: Optional[int],
    plausibility_check,
) -> Optional[str]:
    """Given a list of (canonical, position) candidates, pick the one most
    consistent with `site_key`.

    Strategy:
      1. If there's only one candidate, return it.
      2. Filter to candidates whose canonical name is plausible for the
         site (via `plausibility_check(canonical, site_key)`).
      3. If filtering leaves exactly one, return it — the unambiguous win.
      4. If multiple remain (e.g. lung SCC vs lung adeno), pick the one
         whose position is nearest to `anchor_pos` (usually the cancer_type
         mention), so we prefer the histology written next to the site.
      5. If the filter eliminates everything, fall back to the first
         candidate in the original list (backwards-compatible behaviour).
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    # Step 2: filter to site-plausible candidates
    site_ok = [
        (name, pos) for name, pos in candidates
        if plausibility_check(name, site_key)
    ]

    if not site_ok:
        # Ontology rejected everything — keep old behavior (first by position)
        return min(candidates, key=lambda x: x[1])[0]

    if len(site_ok) == 1:
        return site_ok[0][0]

    # Step 4: multiple plausible — pick the one nearest the anchor.
    if anchor_pos is not None:
        site_ok.sort(key=lambda x: abs(x[1] - anchor_pos))
    else:
        site_ok.sort(key=lambda x: x[1])
    return site_ok[0][0]


def extract_patient_context_from_query(
    query: str,
    resolver_hints: Optional[Any] = None,
    reconciled: Optional[ReconciledStructure] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract patient characteristics from the query using regex patterns,
    with **site-aware disambiguation** for multi-cancer patients.

    When *reconciled* (a ``ReconciledStructure`` from
    ``query_reconciliation.reconcile``) is supplied and the
    ``USE_RECONCILED_STRUCTURE`` feature flag is true, patient context
    fields are populated directly from the reconciled structure instead
    of re-parsing the query text.

    When *resolver_hints* (a ``ResolvedQueryTokens`` from
    ``query_token_resolver.resolve_query_tokens``) is supplied, its canonical
    detections are used as a fallback: biomarkers and stage are lifted in
    only when the regex path returned nothing for those slots, and
    ``biomarker_status`` is always surfaced so ``build_patient_summary`` can
    render phrases like "CPS 100" or "PD-L1 high" that the regex extractor
    does not capture.

    The previous implementation used first-match-wins + break semantics,
    which silently mislabeled patients with more than one cancer in their
    history. For example, a patient with "transverse colon adenocarcinoma
    ... and ... oral tongue squamous cell carcinoma" would be tagged
    histology=adenocarcinoma (because adeno was checked first) and
    surgery=colorectal (because colectomy was checked first), causing
    legitimate H&N SCC studies to be hard-filtered out as "MISMATCH".

    The new logic:
      1. Scans the query for ALL histology and surgery canonical names
         (not just the first one) via ``re.finditer``.
      2. Uses ``cancer_type_ontology.json`` to check which candidates are
         *plausible* for the detected cancer_type (via
         ``ontology_loader.is_histology_plausible_for_site``).
      3. When multiple plausible candidates remain, picks the one whose
         position in the text is nearest to the cancer_type mention —
         the histology written right next to "oral tongue" wins over
         the adeno buried in the PMH.
      4. Falls back to the legacy first-match-wins behaviour when
         neither cancer_type nor ontology data is available, so existing
         calls with simpler single-cancer queries are unaffected.
    """
    # ── Fast path: populate from ReconciledStructure when provided ────
    if reconciled is not None:
        patient_context: Dict[str, Any] = {}

        if reconciled.cancer_site:
            patient_context["cancer_type"] = reconciled.cancer_site
        if reconciled.histology:
            patient_context["histology"] = reconciled.histology
        if reconciled.stage:
            patient_context["stage"] = reconciled.stage
        if reconciled.tnm_t is not None and reconciled.tnm_n is not None and reconciled.tnm_m is not None:
            patient_context["tnm"] = {
                "t": reconciled.tnm_t,
                "n": reconciled.tnm_n,
                "m": reconciled.tnm_m,
            }
        if reconciled.biomarkers:
            biomarker_strings = []
            for bm in reconciled.biomarkers:
                if bm.polarity:
                    biomarker_strings.append(f"{bm.name} {bm.polarity}")
                else:
                    biomarker_strings.append(bm.name)
            patient_context["biomarkers"] = biomarker_strings
        if reconciled.prior_treatments:
            # Map to treatment_history format expected downstream
            patient_context["treatment_history"] = "previously_treated"
        if reconciled.age is not None:
            patient_context["age"] = reconciled.age
        if reconciled.gender:
            patient_context["gender"] = reconciled.gender
        if reconciled.performance_status:
            patient_context["ecog"] = reconciled.performance_status
        if reconciled.receptor_status:
            patient_context["receptor_status"] = reconciled.receptor_status

        # Disease status / surgical candidacy — populated via the
        # regex extractors in query_structuring_service. The reconciled
        # structure doesn't carry these (they live on clinical_history),
        # so we re-extract from the raw query as a fallback path.
        from src.api.services.query_structuring_service import (
            extract_disease_status,
            extract_surgical_candidacy,
        )
        ds = extract_disease_status(query)
        if ds:
            patient_context["disease_status"] = ds
        sc = extract_surgical_candidacy(query)
        if sc:
            patient_context["surgical_candidacy"] = sc

        return patient_context if patient_context else None
    from src.api.services.ontology_loader import (
        cancer_type_label_to_site_key,
        is_histology_plausible_for_site,
        is_surgery_plausible_for_site,
    )

    patient_context: Dict[str, Any] = {}
    query_lower = query.lower()

    # ── Demographics ──────────────────────────────────────────────────────
    age_match = re.search(r'(\d{1,3})\s*(?:year[s]?\s*old|yo|y\.?o\.?|years?\s*of\s*age)', query_lower)
    if age_match:
        patient_context["age"] = int(age_match.group(1))

    if re.search(r'\b(male|man|gentleman|he|his)\b', query_lower):
        patient_context["gender"] = "male"
    elif re.search(r'\b(female|woman|lady|she|her)\b', query_lower):
        patient_context["gender"] = "female"

    # ── Stage + TNM ───────────────────────────────────────────────────────
    stage_match = re.search(r'stage\s*([IViv]{1,4}[ABCabc]?|[1-4][ABCabc]?)', query, re.IGNORECASE)
    if stage_match:
        patient_context["stage"] = stage_match.group(1).upper()

    tnm_match = re.search(r'[pcy]?T([0-4](?:is|a|b|c)?)\s*N([0-3](?:a|b|c|mi)?)\s*M([01])', query, re.IGNORECASE)
    if tnm_match:
        patient_context["tnm"] = {
            "t": tnm_match.group(1),
            "n": tnm_match.group(2),
            "m": tnm_match.group(3),
        }

    # ── Cancer type (historical-signal-aware primary selection) ──────────
    chosen_cancer_type, cancer_type_pos = _pick_primary_cancer_type(query)
    if chosen_cancer_type:
        patient_context["cancer_type"] = chosen_cancer_type
    else:
        # Fallback: query didn't contain an explicit "<site> cancer" phrase
        # but may contain anatomical subsite tokens (maxilla, glottic,
        # buccal, etc.) or disease abbreviations (NSCLC, HNSCC, RCC).
        # Delegate to infer_site_key() which has the full subsite keyword
        # table, then synthesize a canonical cancer_type label from the site.
        #
        # This fixes the class of bug where arm queries like
        #   "adjuvant chemoradiotherapy poorly differentiated SCC maxilla pT4N0"
        # produced NO cancer_type, which in turn excluded cancer_type from
        # `active_criteria` and let wrong-cancer-type studies pass the
        # PatientEligibility hard filter. See RF-2 / RF-3 audit notes.
        synthesized = _synthesize_cancer_type_from_site(query)
        if synthesized:
            patient_context["cancer_type"] = synthesized
            cancer_type_pos = 0  # no explicit span — use start-of-query

    # Resolve the ontology site key once — used for both histology and surgery
    site_key = cancer_type_label_to_site_key(patient_context.get("cancer_type"))

    # ── Cancer subsite (fine-grained, preserved for eligibility discrimination) ─
    subsite = _extract_cancer_subsite(query, patient_context.get("cancer_type"))
    if subsite:
        patient_context["cancer_subsite"] = subsite

    # ── Histology (site-aware disambiguation) ────────────────────────────
    histology_candidates = _find_all_matches(HISTOLOGY_PATTERNS, query)
    chosen_histology = _disambiguate_to_site(
        candidates=histology_candidates,
        site_key=site_key,
        anchor_pos=cancer_type_pos,
        plausibility_check=is_histology_plausible_for_site,
    )
    if chosen_histology:
        patient_context["histology"] = chosen_histology

    # ── Surgery (site-aware disambiguation) ──────────────────────────────
    surgery_candidates = _find_all_matches(SURGERY_PATTERNS, query)
    chosen_surgery = _disambiguate_to_site(
        candidates=surgery_candidates,
        site_key=site_key,
        anchor_pos=cancer_type_pos,
        plausibility_check=is_surgery_plausible_for_site,
    )
    if chosen_surgery:
        patient_context["surgery"] = chosen_surgery

    # ── Treatment history ────────────────────────────────────────────────
    if re.search(r'(previously\s*treated|prior\s*treatment|failed|refractory|resistant)', query_lower):
        patient_context["treatment_history"] = "previously_treated"
    elif re.search(r'(treatment\s*naive|untreated|newly\s*diagnosed|first\s*line)', query_lower):
        patient_context["treatment_history"] = "treatment_naive"

    # ── Performance status ───────────────────────────────────────────────
    ecog_match = re.search(r'ecog\s*(?:ps|performance\s*status)?\s*[=:]?\s*([0-4])', query_lower)
    if ecog_match:
        patient_context["ecog"] = int(ecog_match.group(1))

    kps_match = re.search(r'kps\s*[=:]?\s*(\d{2,3})', query_lower)
    if kps_match:
        patient_context["kps"] = int(kps_match.group(1))

    # ── Biomarkers ───────────────────────────────────────────────────────
    try:
        from src.api.services.enhanced_rag_service import _extract_biomarkers_from_text
        biomarkers = _extract_biomarkers_from_text(query)
    except ImportError:
        biomarkers = []
    if biomarkers:
        patient_context["biomarkers"] = biomarkers

    # ── Resolver fallback (Fix H) ────────────────────────────────────────
    # When the regex slots are empty but the JSON-ontology-driven resolver
    # detected canonical entities, lift them in. Never overwrite regex
    # findings — resolver is strictly a backfill for what regex missed.
    if resolver_hints is not None:
        if not patient_context.get("biomarkers"):
            resolver_biomarkers = sorted(getattr(resolver_hints, "biomarkers", ()) or ())
            if resolver_biomarkers:
                patient_context["biomarkers"] = resolver_biomarkers

        if not patient_context.get("stage"):
            resolver_stages = getattr(resolver_hints, "stages", ()) or ()
            # Prefer the canonical "Stage IV"-style emission from the tagger;
            # strip the "Stage " prefix so downstream slot matches "IV"/"IVA".
            for s in sorted(resolver_stages):
                m = re.match(r"^\s*stage\s+([IViv\d]+[ABCabc]?)\s*$", s, re.IGNORECASE)
                if m:
                    patient_context["stage"] = m.group(1).upper()
                    break

        # biomarker_status is always additive — it carries the qualifier
        # ("positive"/"high"/"CPS 100") that raw biomarker names lack.
        bm_status = sorted(getattr(resolver_hints, "biomarker_status", ()) or ())
        if bm_status:
            patient_context["biomarker_status"] = bm_status

    # ── Genomic / recurrence score ───────────────────────────────────────
    score_match = re.search(
        r'(?:21[- ]?gene|oncotype|recurrence)\s*(?:dx\s*)?(?:recurrence\s*)?score\s*(?:of\s*)?(\d+)',
        query_lower,
    )
    if score_match:
        patient_context["recurrence_score"] = int(score_match.group(1))

    # ── Disease status + surgical candidacy ───────────────────────────────
    # Both axes feed the eligibility hard-filter. Imported lazily to
    # avoid circular import with query_structuring_service.
    from src.api.services.query_structuring_service import (
        extract_disease_status,
        extract_surgical_candidacy,
    )
    ds = extract_disease_status(query)
    if ds:
        patient_context["disease_status"] = ds
    sc = extract_surgical_candidacy(query)
    if sc:
        patient_context["surgical_candidacy"] = sc

    return patient_context if patient_context else None


def build_patient_summary(patient_context: Dict[str, Any]) -> str:
    """Build a concise patient summary string from extracted context."""
    parts = []

    if patient_context.get("age"):
        parts.append(f"{patient_context['age']} year old")
    if patient_context.get("gender"):
        parts.append(patient_context["gender"])
    if patient_context.get("cancer_subsite") and patient_context.get("cancer_type"):
        parts.append(
            f"with {patient_context['cancer_subsite']} "
            f"({patient_context['cancer_type']})"
        )
    elif patient_context.get("cancer_type"):
        parts.append(f"with {patient_context['cancer_type']}")
    if patient_context.get("histology"):
        parts.append(f"({patient_context['histology']})")
    if patient_context.get("stage"):
        parts.append(f"stage {patient_context['stage']}")
    if patient_context.get("tnm"):
        tnm = patient_context["tnm"]
        parts.append(f"T{tnm['t']}N{tnm['n']}M{tnm['m']}")
    if patient_context.get("ecog") is not None:
        parts.append(f"ECOG {patient_context['ecog']}")
    if patient_context.get("treatment_history"):
        parts.append(patient_context["treatment_history"].replace("_", " "))
    if patient_context.get("biomarkers"):
        parts.append(", ".join(patient_context["biomarkers"]))
    # biomarker_status (from resolver) carries qualifier phrases like
    # "HER2:positive" or "PD-L1:high" — split back to human-readable form.
    if patient_context.get("biomarker_status"):
        rendered = []
        for entry in patient_context["biomarker_status"]:
            canonical, _, status = entry.partition(":")
            if canonical and status:
                rendered.append(f"{canonical} {status}")
            elif canonical:
                rendered.append(canonical)
        if rendered:
            parts.append(", ".join(rendered))
    if patient_context.get("recurrence_score") is not None:
        parts.append(f"recurrence score {patient_context['recurrence_score']}")
    if patient_context.get("surgery"):
        parts.append(f"s/p {patient_context['surgery']}")
    # Disease status and surgical candidacy — clinically critical
    # qualifiers that the eligibility LLM uses for the new hard-drop
    # axes. Render in human-readable form (e.g. "recurrent disease",
    # "not a surgical candidate") so the prompt can quote them back.
    ds = patient_context.get("disease_status")
    if ds:
        _ds_label = {
            "primary": "primary disease",
            "recurrent": "recurrent disease",
            "metastatic": "metastatic disease",
            "post_progression": "post-progression on prior systemic therapy",
        }.get(ds, ds.replace("_", " "))
        parts.append(_ds_label)
    sc = patient_context.get("surgical_candidacy")
    if sc:
        _sc_label = {
            "candidate": "surgical candidate",
            "not_candidate": "not a surgical candidate",
            "declined": "declined surgical management",
        }.get(sc, sc.replace("_", " "))
        parts.append(_sc_label)

    return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Core eligibility check with structured per-criterion verdicts
# ---------------------------------------------------------------------------

async def check_patient_eligibility_for_studies(
    query: str,
    patient_context: Dict[str, Any],
    doc_ids: List[str],
    openai_client: OpenAI,
) -> Dict[str, Dict[str, Any]]:
    """
    Check if the patient described in the query matches the enrolled patients
    in each study.  Returns a dict mapping doc_id -> eligibility result with
    per-criterion verdicts (MATCH / MISMATCH / NOT_AVAILABLE).
    """
    if not doc_ids or not patient_context:
        return {}

    doc_ids_to_check = doc_ids[:10]

    patient_summary = build_patient_summary(patient_context)
    if not patient_summary:
        return {}

    print(f"[PatientEligibility] Checking {len(doc_ids_to_check)} studies for patient: {patient_summary}")

    try:
        qdrant = _get_shared_qdrant_client()

        study_eligibility_info = {}
        for doc_id in doc_ids_to_check:
            from qdrant_client import models as qm
            # Threaded — qdrant.scroll() is a synchronous network call;
            # calling it directly inside this async function blocks the
            # event loop (and every other in-flight request) for each of
            # the up-to-10 sequential scrolls.
            results = await asyncio.to_thread(
                qdrant.scroll,
                collection_name=settings.qdrant_collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
                    ]
                ),
                limit=5,
                with_payload=True,
            )

            if results and results[0]:
                # Score each chunk so the highest-quality eligibility
                # content surfaces first and references-only chunks get
                # pushed down. A purely numeric/citation chunk (the kind
                # PDF parsers extract from reference sections) has no
                # information the eligibility LLM can verdict against,
                # and lets unrelated studies survive surgical_candidacy
                # checks with NOT_AVAILABLE.
                #
                # Score components (higher = preferred):
                #   +6 per eligibility/inclusion/exclusion hit
                #   +3 per population/study-design term hit
                #   -5 if the chunk looks like a references list
                #     (≥5 lines starting with "N." or "[N]" or just "N")
                #   -3 if chunk has > 40% non-alphanumeric characters
                ELIG_KW = (
                    "eligib", "inclusion", "exclusion", "enroll",
                    "criteria", "selection", "unresectable", "inoperable",
                    "resectable", "surgical candidate", "salvage surgery",
                    "treatment-naive", "previously untreated",
                )
                METHOD_KW = (
                    "patient", "population", "study design", "phase",
                    "received", "treated", "randomized", "underwent",
                    "histology", "diagnosis",
                )

                def _score_chunk(t: str) -> int:
                    tl = t.lower()
                    score = 0
                    for kw in ELIG_KW:
                        if kw in tl:
                            score += 6
                    for kw in METHOD_KW:
                        if kw in tl:
                            score += 3
                    # Detect reference-list chunks.
                    import re as _re
                    ref_lines = _re.findall(
                        r"^\s*(?:\d+\.|\[\d+\]|\d+\s)\s",
                        t,
                        flags=_re.MULTILINE,
                    )
                    if len(ref_lines) >= 5:
                        score -= 5
                    if t and (
                        sum(1 for c in t if not c.isalnum() and not c.isspace())
                        / max(len(t), 1)
                    ) > 0.4:
                        score -= 3
                    return score

                scored = []
                title = None
                for point in results[0]:
                    payload = point.payload or {}
                    text = payload.get("text", "")
                    if not title:
                        title = payload.get("doc_meta", {}).get("title", doc_id)
                    scored.append((_score_chunk(text), text))

                # Highest-scored chunks first; fall back to original order
                # for ties so the existing behaviour is preserved when
                # nothing distinguishes the candidates.
                scored.sort(key=lambda p: p[0], reverse=True)
                texts = [
                    (t[:500] if i == 0 else t[:300])
                    for i, (_, t) in enumerate(scored)
                ]

                study_eligibility_info[doc_id] = {
                    "title": title,
                    "text": " ".join(texts[:3])[:1500]
                }

        if not study_eligibility_info:
            return {}

        # ── Structured inclusion/exclusion criteria from Postgres ──────
        # The Qdrant-chunk text often does not contain the full enrolment
        # criteria. Pull them from `display-study-details` (the same DB
        # the study details panel reads from) and graft them onto each
        # study's prompt input. Missing studies just fall back to the
        # chunk-derived text.
        #
        # IMPORTANT: this is on PostgresStudyDetailsService, which uses
        # settings.postgres_database (= display-study-details). The
        # similarly named StudyProfileStorageService uses the cache DB
        # (exueed_cache) — it does NOT hold inclusion/exclusion rows.
        structured_criteria: Dict[str, Dict[str, List[str]]] = {}
        try:
            from src.api.services.postgres_study_details_service import (
                get_postgres_study_details_service,
            )
            # Use the shared singleton — a fresh service per request
            # leaks pools and exhausts display-study-details
            # max_connections in a few queries.
            details_service = get_postgres_study_details_service()
            doc_ids_for_fetch = list(study_eligibility_info.keys())
            structured_criteria = await details_service.get_eligibility_criteria_by_doc_ids(
                doc_ids_for_fetch
            )
            n_with_excl = sum(
                1 for c in structured_criteria.values() if c.get("exclusion")
            )
            n_with_incl = sum(
                1 for c in structured_criteria.values() if c.get("inclusion")
            )
            print(
                f"[PatientEligibility] Fetched structured criteria from "
                f"display-study-details: {len(structured_criteria)}/{len(doc_ids_for_fetch)} "
                f"studies present in DB; {n_with_incl} have inclusion criteria, "
                f"{n_with_excl} have exclusion criteria"
            )
        except Exception as e:
            print(f"[PatientEligibility] structured criteria fetch failed (continuing without): {e}")
            structured_criteria = {}

        # ---------------------------------------------------------------
        # Build the criteria-aware prompt
        # ---------------------------------------------------------------
        # Determine which criteria are actually present in the patient context
        active_criteria = []
        if patient_context.get("cancer_type"):
            active_criteria.append("cancer_type")
        if patient_context.get("histology"):
            active_criteria.append("histology")
            # SAFETY: if the patient has a histology but the extractor
            # couldn't determine cancer_type from the raw text, a
            # cancer_type=MISMATCH verdict from the LLM would be silently
            # downgraded to a pass. Force cancer_type into active_criteria
            # whenever histology is active so studies about the wrong
            # cancer type still get hard-filtered. See RF-3 audit notes
            # for the real-world failure this addresses (68yo maxilla SCC
            # case returning anal / glioma / chordoma / prostate studies).
            if "cancer_type" not in active_criteria:
                active_criteria.append("cancer_type")
        if patient_context.get("stage") or patient_context.get("tnm"):
            active_criteria.append("stage")
        if patient_context.get("treatment_history"):
            active_criteria.append("prior_therapies")
        if patient_context.get("biomarkers"):
            active_criteria.append("biomarkers")
        # New hard-filter axes — only activate when the patient profile
        # carries the information. Studies whose text is silent on these
        # axes will get NOT_AVAILABLE, not MISMATCH (per CRITICAL RULES).
        if patient_context.get("disease_status"):
            active_criteria.append("disease_status")
        if patient_context.get("surgical_candidacy"):
            active_criteria.append("surgical_candidacy")
        # study_exclusions_violated always fires when we have any
        # exclusion criteria pulled from Postgres for any study in the
        # batch — even if patient profile is thin, the LLM may still
        # find a violation against the patient_summary text.
        if any(c.get("exclusion") for c in structured_criteria.values()):
            active_criteria.append("study_exclusions_violated")

        # Include recurrence score as part of biomarkers for eligibility evaluation
        biomarker_parts = list(patient_context.get("biomarkers", []))
        if patient_context.get("recurrence_score") is not None:
            biomarker_parts.append(f"21-gene recurrence score {patient_context['recurrence_score']}")
            if "biomarkers" not in active_criteria:
                active_criteria.append("biomarkers")

        # Render the patient-side criteria block. Each line is a single
        # axis the LLM must verdict; "N/A" means absent from patient
        # profile (LLM should mark NOT_AVAILABLE for that axis).
        criteria_lines = [
            f"  - cancer_type: {patient_context.get('cancer_type', 'N/A')}",
            f"  - histology: {patient_context.get('histology', 'N/A')}",
            f"  - stage: {patient_context.get('stage', 'N/A')}"
            + (f" (TNM: T{patient_context['tnm']['t']}N{patient_context['tnm']['n']}M{patient_context['tnm']['m']})" if patient_context.get('tnm') else ""),
            f"  - prior_therapies: {patient_context.get('treatment_history', 'N/A')}",
            f"  - biomarkers: {', '.join(biomarker_parts) if biomarker_parts else 'N/A'}",
            f"  - disease_status: {patient_context.get('disease_status', 'N/A')}",
            f"  - surgical_candidacy: {patient_context.get('surgical_candidacy', 'N/A')}",
        ]
        criteria_description = "\n".join(criteria_lines)

        # Per-study text block. Append structured inclusion/exclusion
        # criteria from Postgres when available — these are typically
        # cleaner and more complete than free-text retrieval from chunks.
        def _study_block(i: int, doc_id: str, info: Dict[str, str]) -> str:
            block = (
                f"Study {i+1} (ID: {doc_id}):\n"
                f"Title: {info['title']}\n"
                f"Content: {info['text'][:800]}..."
            )
            crit = structured_criteria.get(doc_id)
            if crit:
                if crit.get("inclusion"):
                    inc = "; ".join(crit["inclusion"][:8])
                    block += f"\nInclusion criteria: {inc}"
                if crit.get("exclusion"):
                    exc = "; ".join(crit["exclusion"][:8])
                    block += f"\nExclusion criteria: {exc}"
            return block

        studies_text = "\n\n".join(
            _study_block(i, doc_id, info)
            for i, (doc_id, info) in enumerate(study_eligibility_info.items())
        )

        prompt = f"""You are a clinical trial eligibility expert. For each study below, evaluate whether the patient matches the study's enrolled population on EACH of the following criteria.

Patient Profile: {patient_summary}
Active criteria to evaluate:
{criteria_description}

Studies to evaluate:
{studies_text}

For EACH study, evaluate EACH criterion independently and assign one of these verdicts:
- "MATCH" — the study's enrolled population clearly includes this patient on this criterion
- "COMPATIBLE" — the patient is not an exact match but falls within an acceptable range (e.g., patient stage IIIA vs study requiring stage III; patient histology is a subtype of what the study accepts; study accepts a broader category that includes the patient)
- "MISMATCH" — the study explicitly enrolled a DIFFERENT population on this criterion (e.g., study is for lung cancer but patient has breast cancer; study is for HER2+ but patient is HER2-; study enrolled only stage IV but patient is stage II; study required treatment-naive but patient was previously treated)
- "NOT_AVAILABLE" — the study does NOT explicitly report or restrict on this criterion (the information is simply absent from the study text)

CRITICAL RULES:
1. "NOT_AVAILABLE" means the study text does not mention or restrict on that criterion AT ALL. This is DIFFERENT from MISMATCH.
2. A study that does not mention biomarkers at all → biomarkers = NOT_AVAILABLE (do NOT mark as MISMATCH).
3. A study that does not mention histology → histology = NOT_AVAILABLE.
4. A study that does not restrict by stage → stage = NOT_AVAILABLE.
5. A study that does not mention prior therapy requirements → prior_therapies = NOT_AVAILABLE.
6. Be STRICT about MISMATCH: only use it when the study EXPLICITLY contradicts the patient's profile.
7. Be STRICT about biomarker polarity: ER+ ≠ ER-, HER2+ ≠ HER2-, "amplified"/"overexpressed"/"mutant" = positive, "non-amplified"/"wild-type" = negative.
8. Range matching: a study for "stage II-III" is COMPATIBLE with a stage III patient; a study for "score < 25" is COMPATIBLE with score 22.
9. SAME ORGAN SYSTEM ≠ SAME CANCER: cancers in the same organ system are DIFFERENT cancers. Prostate cancer ≠ bladder cancer ≠ renal cancer (all GU). Colon cancer ≠ rectal cancer ≠ gastric cancer ≠ esophageal cancer (all GI). Cervical cancer ≠ ovarian cancer ≠ endometrial cancer (all GYN). If the study explicitly enrolls patients with a DIFFERENT specific cancer from the same organ system, mark cancer_type as MISMATCH.
9a. EXCLUSION CRITERIA DO NOT REDEFINE CANCER_TYPE: a study that ENROLLS head-and-neck cancer patients but EXCLUDES "history of head and neck cancer" is still a head-and-neck cancer study — cancer_type is MATCH for an H&N patient. The exclusion is about prior unrelated cancers in the PMH, not about the index diagnosis being treated. Read the INCLUSION criteria (or the title / study description) to determine what cancer the study is FOR. Exclusion criteria phrased as "history of X cancer" or "prior X cancer" go under study_exclusions_violated (with the index-cancer carve-out from rule 24), NEVER under cancer_type. Stamping cancer_type=MISMATCH because of a "history of [same cancer]" exclusion is a recurring false positive that filters out legitimately matching studies.
10. ORTHOGONAL AXES ARE INDEPENDENT: Evaluate each criterion ONLY on its own axis. Nodal status (N+/N-) is a STAGING criterion — it is NOT a biomarker. Molecular subtype (TNBC, HER2+, ER+) is a BIOMARKER criterion — it is NOT staging. A study enrolling "node-positive breast cancer" is compatible with a TNBC patient who is node-positive. Do NOT confuse staging eligibility with biomarker eligibility. Similarly, treatment setting (neoadjuvant vs adjuvant) is NOT the same as treatment modality (chemotherapy vs immunotherapy).
11. INCLUSIVE studies: Many studies enroll "all comers" within a cancer type regardless of subtype. A breast cancer study that does not restrict by receptor status is NOT a MISMATCH for a TNBC patient — it is NOT_AVAILABLE for biomarkers.
12. Use COMPATIBLE for subsumption: patient sub-stage vs study parent stage (IIIA vs III), patient subtype vs study umbrella type, or study accepts a range that includes the patient.

DISEASE STATUS AXIS (disease_status):
13. Patient values: "primary" (newly diagnosed, treatment-naive at diagnosis), "recurrent" (disease returned after prior definitive treatment), "metastatic" (M1 / distant disease), "post_progression" (currently progressing on a named systemic therapy such as ICI/pembrolizumab).
14. Study compatibility: a study whose enrolled population is "newly diagnosed primary disease" is MISMATCH for a recurrent or metastatic patient. A study for "recurrent/metastatic head and neck cancer" is COMPATIBLE with both recurrent and metastatic patients. A study restricted to "post-checkpoint progression" MATCHES a post_progression patient and is MISMATCH for treatment-naive primary patients.
15. When the study text does not specify what disease state it enrolled → NOT_AVAILABLE.

SURGICAL CANDIDACY AXIS (surgical_candidacy):
16. Patient values: "candidate" (planned for or eligible for surgical management), "not_candidate" (unresectable, inoperable, no longer a surgical candidate), "declined" (refused surgery).
17. Study compatibility: a study whose enrolment requires patients undergoing surgery, salvage surgery, or resection is MISMATCH for a "not_candidate" or "declined" patient. A study enrolling unresectable / inoperable patients is MATCH for "not_candidate" and MISMATCH for an explicit "candidate".
18. CRITICAL EXAMPLES — these are common false-positive traps:
    a. A study that defines "criteria for unresectability" or enrols "unresectable head and neck cancer" patients (e.g. Adelstein 2003 Intergroup CRT trial, full-dose reirradiation for unresectable HNSCC) is MATCH for a "not_candidate" patient, NOT MISMATCH. The study is for the same population as the patient.
    b. A study that enrols "resectable" or "operable" disease (e.g. OCAT for "resectable oral cavity SCC", D'Cruz 2015 for early-stage operable oral cancer with elective neck dissection) is MISMATCH for "not_candidate".
    c. A study that requires "salvage surgery" or "after surgery" or "postoperative" (e.g. Janot 2008 postoperative reirradiation after salvage surgery) is MISMATCH for "not_candidate" — the patient cannot undergo the required salvage operation.
    d. A study about "trimodality" or "combined modality" including required surgery is MISMATCH for "not_candidate".
19. Studies that do not specify surgical status → NOT_AVAILABLE.

STUDY EXCLUSIONS VIOLATED AXIS (study_exclusions_violated) — INVERTED POLARITY:
20. Evaluate each item in the study's "Exclusion criteria:" line against the patient profile.
21. Respond MATCH only if the patient CLEARLY violates one or more exclusion criteria (i.e. the study would EXCLUDE this patient). Quote the violated criterion in `reasoning`.
22. NOT_AVAILABLE when no exclusion criteria are listed, or when none are violated.
23. NEVER respond MISMATCH on this axis — it has inverted polarity.
24. CRITICAL: exclusion criteria phrased as "history of X cancer", "prior X cancer", "previous X cancer", or "second primary X cancer" refer to a PRIOR UNRELATED cancer — NOT the patient's current/index diagnosis that's the reason they're being evaluated. If a study excludes "history of head and neck cancer" and the patient's index cancer IS head and neck cancer, that is NOT a violation — the study is treating the patient's H&N cancer, not excluding it. Only flag as MATCH (violation) if the patient has a SEPARATE prior cancer in their PMH that the exclusion would catch.
25. Similarly: "prior chemotherapy / radiation / surgery" exclusions refer to treatment for an unrelated cancer or a separate disease, not the patient's index-cancer-directed therapy. The patient's adjuvant chemo for their index cancer does NOT violate a "no prior chemo" exclusion — the study is recruiting AT that point in the patient's treatment course.

HARD-DROP AXES (be CONSERVATIVE — wrong MISMATCH here drops the study from results entirely):
26. The following axes are HARD-DROP: cancer_type, histology, stage, prior_therapies, disease_status, surgical_candidacy. MISMATCH on any of these REMOVES the study from the result set.
27. Use MISMATCH ONLY when the study has an EXPLICIT inclusion or eligibility constraint that the patient FAILS. Examples that ARE MISMATCH:
    - Study eligibility says "stage II-III" and patient is stage IV → stage MISMATCH
    - Study enrols "treatment-naive" patients and patient has had ≥1 prior systemic therapy → prior_therapies MISMATCH
    - Study enrols "adenocarcinoma" and patient is "squamous cell carcinoma" → histology MISMATCH
28. Use COMPATIBLE (not MISMATCH) when:
    - The study's range INCLUDES the patient (stage III patient + study for "stage II-III" → COMPATIBLE per rule 8)
    - Patient sub-stage falls under the study's parent stage (IIIA patient + study for "stage III" → COMPATIBLE per rule 12)
    - Study cancer-type umbrella includes the patient (patient "oral cavity SCC" + study for "head and neck SCC" → COMPATIBLE)
29. Use NOT_AVAILABLE when the study does NOT restrict on that axis. A breast cancer study that doesn't mention HR status is NOT_AVAILABLE on biomarkers — not MISMATCH. A H&N study that doesn't specify stage requirements is NOT_AVAILABLE on stage — not MISMATCH.
30. When in doubt between MISMATCH and COMPATIBLE on a hard-drop axis, choose COMPATIBLE. False MISMATCH on a hard-drop axis is the most damaging error this evaluator can make.

Respond in EXACTLY this JSON format (no other text). Return a JSON object with one key per study number. Each study value must contain an "overall_verdict", an "axis_verdicts" object with per-axis verdicts, and a "reasoning" string. Include the new axes (disease_status, surgical_candidacy, study_exclusions_violated) inside `axis_verdicts` for every study:
{{
  "1": {{
    "overall_verdict": "MATCH or COMPATIBLE or MISMATCH or NOT_AVAILABLE",
    "axis_verdicts": {{
      "cancer_type": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "histology": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "stage": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "biomarkers": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "prior_therapies": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "disease_status": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "surgical_candidacy": "MATCH/COMPATIBLE/MISMATCH/NOT_AVAILABLE",
      "study_exclusions_violated": "MATCH/NOT_AVAILABLE"
    }},
    "reasoning": "Brief explanation; quote any violated exclusion criterion"
  }},
  "2": {{ "overall_verdict": "...", "axis_verdicts": {{...}}, "reasoning": "..." }}
}}

The overall_verdict should reflect the worst axis: if ANY hard-drop axis (cancer_type, histology, stage, prior_therapies, disease_status, surgical_candidacy) is MISMATCH, overall is MISMATCH. If study_exclusions_violated is MATCH (patient violates an exclusion), overall is MISMATCH. If no MISMATCH but any COMPATIBLE, overall is COMPATIBLE. If all evaluated axes are MATCH or NOT_AVAILABLE, overall is MATCH."""

        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=settings.openai_mini_model,
            messages=[
                {"role": "system", "content": "You are a clinical trial eligibility expert. Respond ONLY with the requested JSON. Use MATCH, COMPATIBLE, MISMATCH, or NOT_AVAILABLE for each axis. Be strict about confirmed mismatches but never confuse missing information with a mismatch."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            # max_tokens bumped from 1200 → 2500 because the structured
            # response format is ~140 tokens per study; 10 studies × 140 =
            # 1400 already overruns the old budget and the response was
            # silently truncating mid-string, cascading through legacy
            # parser fallbacks that also fail on truncated tails. 2500
            # covers ~17 studies comfortably.
            max_tokens=2500,
            # response_format guarantees the model returns syntactically
            # valid JSON. Without this the model occasionally wraps the
            # JSON in prose or emits subtle malformations (trailing
            # commas, smart quotes) that defeat strict parsing. Both the
            # system message and user prompt already mention "json", so
            # the mode's prompt-content requirement is satisfied.
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        print(f"[PatientEligibility] LLM response: {result_text[:300]}...")

        # Parse JSON response — try structured format first, fall back to legacy
        eligibility_results = {}
        doc_id_list = list(study_eligibility_info.keys())

        parsed = _parse_structured_eligibility_json(result_text, len(doc_id_list))
        if not parsed:
            # Fallback to legacy flat-format parser (regex-based extraction)
            print("[PatientEligibility] Structured JSON parse failed, falling back to legacy parser")
            parsed = _parse_eligibility_json(result_text, len(doc_id_list))

        for i, doc_id in enumerate(doc_id_list):
            study_key = str(i + 1)
            if study_key in parsed:
                study_data = parsed[study_key]

                # Extract verdicts — support both structured and legacy formats
                if "axis_verdicts" in study_data:
                    # New structured format
                    verdicts = study_data["axis_verdicts"]
                    reason = study_data.get("reasoning", "")[:100]
                else:
                    # Legacy flat format
                    verdicts = study_data
                    reason = verdicts.get("reason", "")[:100]

                # Determine per-criterion verdicts (only for active criteria)
                criteria_verdicts = {}
                has_mismatch = False
                match_count = 0
                compatible_count = 0
                for criterion in HARD_FILTER_CRITERIA:
                    verdict = verdicts.get(criterion, "NOT_AVAILABLE").upper()
                    # Normalize: accept MATCH, COMPATIBLE, MISMATCH, NOT_AVAILABLE
                    if verdict not in ("MATCH", "COMPATIBLE", "MISMATCH", "NOT_AVAILABLE"):
                        verdict = "NOT_AVAILABLE"
                    criteria_verdicts[criterion] = verdict
                    if verdict == "MISMATCH" and criterion in active_criteria:
                        has_mismatch = True
                    if verdict == "MATCH":
                        match_count += 1
                    if verdict == "COMPATIBLE":
                        compatible_count += 1

                # Biomarkers-when-declared: the study has a declared
                # biomarker requirement only when the LLM returned
                # MATCH or MISMATCH for biomarkers (meaning the study
                # text explicitly mentions biomarker criteria).  When
                # the LLM returns NOT_AVAILABLE, the study is silent.
                biomarker_verdict = criteria_verdicts.get("biomarkers", "NOT_AVAILABLE")
                has_declared_biomarker_req = biomarker_verdict in ("MATCH", "MISMATCH")

                # If the study is silent on biomarkers, force the
                # verdict to NOT_AVAILABLE and exclude it from
                # has_mismatch computation.
                if not has_declared_biomarker_req and biomarker_verdict != "NOT_AVAILABLE":
                    criteria_verdicts["biomarkers"] = "NOT_AVAILABLE"
                    # Recompute has_mismatch without biomarkers
                    has_mismatch = any(
                        criteria_verdicts.get(c) == "MISMATCH"
                        for c in HARD_FILTER_CRITERIA
                        if c in active_criteria
                    )

                # Determine overall status
                if has_mismatch:
                    overall = "NO_MATCH"
                    boost = 0
                elif match_count > 0 or compatible_count > 0:
                    overall = "MATCH" if match_count > 0 and compatible_count == 0 else "POSSIBLE"
                    boost = 0.25 if overall == "MATCH" else 0.15
                else:
                    overall = "POSSIBLE"
                    boost = 0.1

                eligibility_results[doc_id] = {
                    "status": overall,
                    "reason": reason,
                    "boost": boost,
                    "criteria_verdicts": criteria_verdicts,
                    "has_hard_mismatch": has_mismatch,
                    "has_declared_biomarker_requirement": has_declared_biomarker_req,
                }
            else:
                eligibility_results[doc_id] = {
                    "status": "UNKNOWN",
                    "reason": "Could not parse",
                    "boost": 0,
                    "criteria_verdicts": {},
                    "has_hard_mismatch": False,
                }

        matched = sum(1 for r in eligibility_results.values() if r["status"] == "MATCH")
        possible = sum(1 for r in eligibility_results.values() if r["status"] == "POSSIBLE")
        no_match = sum(1 for r in eligibility_results.values() if r["status"] == "NO_MATCH")
        print(f"[PatientEligibility] Results: {matched} MATCH, {possible} POSSIBLE, {no_match} NO_MATCH (hard-filtered)")
        try:
            from src.api.services import pipeline_metrics
            pipeline_metrics.incr("eligibility", "MATCH", matched)
            pipeline_metrics.incr("eligibility", "POSSIBLE", possible)
            pipeline_metrics.incr("eligibility", "NO_MATCH", no_match)
        except Exception:
            pass

        # Log per-study verdicts for debugging
        for doc_id, result in eligibility_results.items():
            title = study_eligibility_info.get(doc_id, {}).get("title", doc_id)[:60]
            verdicts = result.get("criteria_verdicts", {})
            verdict_str = ", ".join(f"{k}={v}" for k, v in verdicts.items() if v != "NOT_AVAILABLE")
            if not verdict_str:
                verdict_str = "all NOT_AVAILABLE"
            print(f"[PatientEligibility]   {result['status']:10s} '{title}' — {verdict_str} — {result.get('reason', '')[:80]}")

        return eligibility_results

    except Exception as e:
        print(f"[PatientEligibility] Error checking eligibility: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _parse_structured_eligibility_json(text: str, num_studies: int) -> Dict[str, Dict[str, Any]]:
    """Parse the structured JSON response with per-axis verdicts from the LLM.

    Expected format per study:
    {
      "1": {
        "overall_verdict": "MATCH",
        "axis_verdicts": {
          "cancer_type": "MATCH",
          "histology": "COMPATIBLE",
          "stage": "MATCH",
          "biomarkers": "MATCH",
          "prior_therapies": "NOT_AVAILABLE"
        },
        "reasoning": "Brief explanation"
      }
    }

    Returns empty dict if the response doesn't match the structured format,
    signaling the caller to fall back to the legacy parser.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    # Validate that at least one study entry has the structured format
    result: Dict[str, Dict[str, Any]] = {}
    valid_verdicts = {"MATCH", "COMPATIBLE", "MISMATCH", "NOT_AVAILABLE"}

    for key, value in parsed.items():
        if not isinstance(value, dict):
            continue

        # Check for structured format (has axis_verdicts)
        if "axis_verdicts" in value and isinstance(value["axis_verdicts"], dict):
            axis_verdicts = value["axis_verdicts"]
            # Normalize verdict values
            normalized_axes = {}
            for axis, verdict in axis_verdicts.items():
                v = str(verdict).upper().strip()
                normalized_axes[axis] = v if v in valid_verdicts else "NOT_AVAILABLE"

            result[key] = {
                "overall_verdict": str(value.get("overall_verdict", "")).upper().strip(),
                "axis_verdicts": normalized_axes,
                "reasoning": str(value.get("reasoning", "")),
            }

    # Only return if we successfully parsed at least one structured entry
    if not result:
        return {}

    print(f"[PatientEligibility] Parsed {len(result)} studies from structured JSON format")
    return result


def _parse_eligibility_json(text: str, num_studies: int) -> Dict[str, Dict[str, str]]:
    """Parse the JSON response from the LLM, with fallback for malformed output."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback 1: try to extract per-study blocks from malformed JSON
    print(f"[PatientEligibility] JSON parse failed, trying regex block extraction")
    result = {}
    for i in range(1, num_studies + 1):
        pattern = rf'"{i}"\s*:\s*\{{([^}}]+)\}}'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                block = json.loads("{" + m.group(1) + "}")
                result[str(i)] = block
            except json.JSONDecodeError:
                pass

    # Fallback 2: line-by-line parsing for non-JSON responses
    if not result:
        print(f"[PatientEligibility] Block extraction failed, trying line-by-line parse")
        for i in range(1, num_studies + 1):
            pattern = rf"{i}:\s*(MATCH|POSSIBLE|NO_MATCH)\s*[-–]?\s*(.+?)(?=\n\d+:|$)"
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                status = m.group(1).upper()
                reason = m.group(2).strip()[:100]
                # Map old-style status to new per-criterion format
                all_match = "MATCH" if status == "MATCH" else ("MISMATCH" if status == "NO_MATCH" else "NOT_AVAILABLE")
                result[str(i)] = {
                    "cancer_type": all_match,
                    "histology": "NOT_AVAILABLE",
                    "stage": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                    "biomarkers": "NOT_AVAILABLE",
                    "reason": reason,
                }

    return result


# ---------------------------------------------------------------------------
# Apply hard filter + boost
# ---------------------------------------------------------------------------

def _demote_subsite_hierarchy_mismatch(
    chunk: Dict[str, Any],
    verdicts: Dict[str, str],
    patient_context: Optional[Dict[str, Any]],
) -> bool:
    """If a `cancer_type: MISMATCH` verdict is actually a subsite-hierarchy
    artifact (patient 'oral cavity SCC' vs. study 'head and neck' umbrella,
    or patient 'h_n' vs. study tagged with a sibling h_n subsite), demote
    it in-place to `"POSSIBLE"` and return True. Returns False when the
    mismatch is a true cross-organ MISMATCH or we can't resolve either
    side to an ontology site_key.

    Guard: do NOT demote when `disease_status` or `surgical_candidacy`
    also returned MISMATCH. A study that's in the same subsite but a
    different disease state (primary vs recurrent) or with different
    surgical requirements is not a "subsite hierarchy artifact" — it's
    a real population mismatch that should keep its hard-drop status.
    Without this guard, an N0-primary oral cancer study would survive
    the cancer_type MISMATCH for a recurrent post-progression patient
    just because both are oral_cavity SCC.
    """
    if not patient_context or verdicts.get("cancer_type") != "MISMATCH":
        return False
    if verdicts.get("disease_status") == "MISMATCH":
        return False
    if verdicts.get("surgical_candidacy") == "MISMATCH":
        return False
    try:
        from src.api.services.ontology_loader import (
            is_same_site_family,
            resolve_phrase_to_site_key,
        )
    except Exception:
        return False

    patient_site = patient_context.get("cancer_type")
    patient_subsite = patient_context.get("cancer_subsite") or ""
    patient_site_key = resolve_phrase_to_site_key(patient_site)
    if not patient_site_key:
        return False

    # Source 1: study's keyword_flat payload tags (ingestion-time metadata).
    payload = chunk.get("payload") or {}
    meta = payload.get("metadata") or chunk.get("metadata") or {}
    keywords_flat = meta.get("keywords_flat") or []
    # Source 2: chunk title + text as a last-resort signal.
    title = chunk.get("title") or payload.get("doc_meta", {}).get("title", "")
    text = (chunk.get("text") or "") + " " + str(title)
    tag_blob = " ".join(str(k) for k in keywords_flat).lower() + " " + text.lower()

    if not tag_blob.strip():
        return False

    # Check 1: does the study carry the patient's subsite keyword directly?
    if patient_subsite:
        for term in patient_subsite.lower().split(" / "):
            term = term.strip()
            if term and term in tag_blob:
                verdicts["cancer_type"] = "POSSIBLE"
                return True

    # Check 2: does any study tag resolve to the same ontology family?
    for candidate in list(keywords_flat)[:50]:
        if not isinstance(candidate, str):
            continue
        if is_same_site_family(patient_site, candidate):
            verdicts["cancer_type"] = "POSSIBLE"
            return True
    return False


def _recompute_hard_mismatch(
    verdicts: Dict[str, str],
    active_criteria: List[str],
) -> bool:
    """Recompute whether any active criterion still carries a MISMATCH
    verdict after in-place demotions (see `_demote_subsite_hierarchy_mismatch`).
    """
    return any(verdicts.get(c) == "MISMATCH" for c in active_criteria)


def _count_hard_mismatches(chunk: Dict[str, Any]) -> int:
    """Count the number of MISMATCH verdicts on a removed chunk.

    Fewer mismatches → the study is less severely mismatched and is a
    better candidate for fallback restoration.
    """
    verdicts = chunk.get("patient_eligibility", {}).get("criteria_verdicts", {})
    return sum(1 for v in verdicts.values() if v == "MISMATCH")


def _apply_eligibility_fallback(
    kept: List[Dict[str, Any]],
    removed: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Restore the least-mismatched removed studies at heavily reduced
    scores when the bundle is below ``MIN_STUDIES_FLOOR``.

    This is the fallback for the ``category_routing_suspect`` guard:
    when the guard fires AND the surviving bundle is clinically useless
    (< 3 studies), we restore enough removed studies to reach the floor.

    Restored studies are:
      - Sorted by mismatch severity (fewest hard mismatches first)
      - Scored at ``original_score * 0.3`` (heavily penalized)
      - Marked with ``restored_at_reduced_score=True`` so downstream
        generation knows they are lower-confidence

    Returns a new list containing the original kept studies plus any
    restored studies, sorted by final_score descending.
    """
    if len(kept) >= MIN_STUDIES_FLOOR or not removed:
        return list(kept)

    need = MIN_STUDIES_FLOOR - len(kept)

    # Sort removed by mismatch severity ascending (least mismatched first),
    # then by original score descending as tiebreaker.
    candidates = sorted(
        removed,
        key=lambda c: (_count_hard_mismatches(c), -(c.get("score", 0))),
    )

    restored = []
    for chunk in candidates[:need]:
        original_score = chunk.get("score", chunk.get("final_score", 0.1))
        chunk["final_score"] = original_score * 0.3
        chunk["restored_at_reduced_score"] = True
        restored.append(chunk)

    if restored:
        print(
            f"[Eligibility Fallback] Restoring {len(restored)} studies "
            f"at reduced scores (bundle below floor)"
        )

    result = list(kept) + restored
    result.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return result


def apply_patient_eligibility_filter_and_boost(
    chunks: List[Dict[str, Any]],
    eligibility_results: Dict[str, Dict[str, Any]],
    patient_context: Optional[Dict[str, Any]] = None,
    active_criteria: Optional[List[str]] = None,
    use_tiered_model: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply hard filtering AND score boosting based on eligibility results.

    Hard filter: remove chunks whose doc_id has a confirmed MISMATCH on any
    active hard-filter criterion (cancer_type, histology, stage, prior_therapies,
    biomarkers).

    Subsite hierarchy guard: when `patient_context` is provided, a
    `cancer_type: MISMATCH` that is really a subsite artifact (patient
    'oral cavity' vs. study 'head and neck' umbrella) is demoted in-place
    to `POSSIBLE` before the hard-filter decision. This prevents the
    downstream Fix-2B `category_routing_suspect` guard from locking in
    false positives caused by coarse umbrella-vs-subsite labeling.

    Boost: MATCH studies get +0.25, POSSIBLE get +0.10.

    Tiered model (when ``use_tiered_model=True``):
      - Tier 1: cancer_type MISMATCH → hard removal (unchanged)
      - Tier 2: secondary axis MISMATCH → retain with penalty
        (0.15 per mismatch, capped at 0.45)
      - Each study is annotated with per-axis verdicts, penalty_applied,
        boost_applied, hard_filtered, and tier.

    Returns:
        (filtered_and_boosted_chunks, removed_chunks)
    """
    if not eligibility_results:
        return chunks, []

    kept = []
    removed = []
    boosted_count = 0
    demoted_count = 0
    penalized_count = 0
    active = active_criteria or HARD_FILTER_CRITERIA

    #: Secondary axes for tiered penalty calculation.
    secondary_axes = [a for a in active if a != "cancer_type"]

    for chunk in chunks:
        doc_id = chunk.get("doc_id") or chunk.get("payload", {}).get("doc_id")

        if doc_id and doc_id in eligibility_results:
            result = eligibility_results[doc_id]
            verdicts = result.get("criteria_verdicts") or {}

            # Subsite-hierarchy demotion (Fix B) — MUST run before the
            # has_hard_mismatch branch below so a demoted MISMATCH can
            # avoid the hard filter on this chunk.
            if _demote_subsite_hierarchy_mismatch(chunk, verdicts, patient_context):
                demoted_count += 1
                result["criteria_verdicts"] = verdicts
                result["has_hard_mismatch"] = _recompute_hard_mismatch(verdicts, active)

            # ── Biomarkers-when-declared override ─────────────────
            # If the study explicitly has no declared biomarker requirement
            # (has_declared_biomarker_requirement is False), force the
            # biomarkers verdict to NOT_AVAILABLE regardless of what the
            # LLM returned.  This prevents silent-biomarker studies from
            # being penalized on the biomarkers axis.
            # NOTE: when the field is absent we do NOT override — the
            # caller is responsible for setting it.  This preserves
            # backward compatibility with eligibility results that
            # predate this field.
            if result.get("has_declared_biomarker_requirement") is False:
                if verdicts.get("biomarkers") and verdicts["biomarkers"] != "NOT_AVAILABLE":
                    verdicts["biomarkers"] = "NOT_AVAILABLE"
                    result["criteria_verdicts"] = verdicts
                    # Recompute has_hard_mismatch after the override
                    has_hard = any(
                        verdicts.get(c) == "MISMATCH"
                        for c in (active_criteria or HARD_FILTER_CRITERIA)
                    )
                    result["has_hard_mismatch"] = has_hard
                    # Recompute status and boost to reflect corrected verdicts
                    match_count = sum(1 for v in verdicts.values() if v == "MATCH")
                    if has_hard:
                        result["status"] = "NO_MATCH"
                        result["boost"] = 0
                    elif match_count > 0:
                        result["status"] = "MATCH"
                        result["boost"] = 0.25
                    else:
                        result["status"] = "POSSIBLE"
                        result["boost"] = 0.1

            if use_tiered_model:
                # ── Tiered eligibility model ──────────────────────────
                # Tier 1: hard-drop axes. Any of cancer_type /
                # disease_status / surgical_candidacy returning MISMATCH
                # removes the study; study_exclusions_violated returning
                # MATCH (inverted polarity — patient violates an
                # exclusion criterion) also removes the study.
                hard_drop_axis = next(
                    (a for a in HARD_DROP_AXES if verdicts.get(a) == "MISMATCH"),
                    None,
                )
                if hard_drop_axis is None:
                    hard_drop_axis = next(
                        (a for a in HARD_DROP_ON_MATCH_AXES if verdicts.get(a) == "MATCH"),
                        None,
                    )
                if hard_drop_axis is not None:
                    chunk["patient_eligibility"] = {
                        "hard_filtered": True,
                        "criteria_verdicts": dict(verdicts),
                        "penalty_applied": 0.0,
                        "boost_applied": 0.0,
                        "tier": "hard_filter",
                        "hard_drop_axis": hard_drop_axis,
                    }
                    removed.append(chunk)
                    title = chunk.get("title") or chunk.get("payload", {}).get("doc_meta", {}).get("title", "unknown")
                    reason = result.get("reason", "")
                    print(
                        f"[PatientEligibility] HARD FILTER (tiered) removed "
                        f"'{str(title)[:60]}' — {hard_drop_axis} "
                        f"{'MATCH' if hard_drop_axis in HARD_DROP_ON_MATCH_AXES else 'MISMATCH'}"
                        + (f" — {reason[:80]}" if reason else "")
                    )
                    continue

                # Tier 2: count secondary mismatches and apply penalty
                secondary_mismatches = sum(
                    1 for axis in secondary_axes
                    if verdicts.get(axis) == "MISMATCH"
                )

                if secondary_mismatches > 0:
                    # Penalty: 0.15 per secondary mismatch, capped at 0.45
                    penalty = min(secondary_mismatches * 0.15, 0.45)
                    current_score = chunk.get("final_score", chunk.get("score", 0.5))
                    chunk["final_score"] = max(0.0, current_score - penalty)
                    penalized_count += 1

                    chunk["patient_eligibility"] = {
                        "status": result.get("status", "POSSIBLE"),
                        "criteria_verdicts": dict(verdicts),
                        "penalty_applied": penalty,
                        "boost_applied": 0.0,
                        "hard_filtered": False,
                        "tier": "penalty",
                    }
                else:
                    # No mismatches — apply existing boost logic
                    boost = result.get("boost", 0)
                    if boost > 0:
                        current_score = chunk.get("final_score", chunk.get("score", 0.5))
                        chunk["final_score"] = min(1.0, current_score + boost)
                        boosted_count += 1

                    chunk["patient_eligibility"] = {
                        "status": result.get("status", "MATCH"),
                        "criteria_verdicts": dict(verdicts),
                        "penalty_applied": 0.0,
                        "boost_applied": boost,
                        "hard_filtered": False,
                        "tier": "boost",
                    }
            else:
                # ── Non-tiered behavior (with bugfix scoring) ─────────
                # Tier 1: hard-drop axes — cancer_type / disease_status /
                # surgical_candidacy MISMATCH, plus study_exclusions_violated
                # MATCH (inverted polarity).
                hard_drop_axis = next(
                    (a for a in HARD_DROP_AXES if verdicts.get(a) == "MISMATCH"),
                    None,
                )
                if hard_drop_axis is None:
                    hard_drop_axis = next(
                        (a for a in HARD_DROP_ON_MATCH_AXES if verdicts.get(a) == "MATCH"),
                        None,
                    )
                if hard_drop_axis is not None:
                    chunk["patient_eligibility"] = {
                        "status": "NO_MATCH",
                        "reason": result.get("reason", ""),
                        "criteria_verdicts": verdicts,
                        "penalty_applied": 0,
                        "boost_applied": 0,
                        "hard_filtered": True,
                        "tier": "hard_filtered",
                        "hard_drop_axis": hard_drop_axis,
                    }
                    removed.append(chunk)
                    title = chunk.get("title") or chunk.get("payload", {}).get("doc_meta", {}).get("title", "unknown")
                    mismatched = [k for k, v in verdicts.items() if v == "MISMATCH"]
                    print(
                        f"[PatientEligibility] HARD FILTER removed "
                        f"'{str(title)[:60]}' — {hard_drop_axis} "
                        f"{'MATCH' if hard_drop_axis in HARD_DROP_ON_MATCH_AXES else 'MISMATCH'} "
                        f"(all MISMATCH axes: {mismatched})"
                    )
                    continue

                # Tier 2: non-cancer_type core axis MISMATCH → retain
                # with penalty (CORE_MISMATCH_PENALTY per mismatch),
                # floor final score at 1.
                _NON_CANCER_CORE = ("histology", "stage", "prior_therapies", "biomarkers")
                non_cancer_mismatches = sum(
                    1 for axis in _NON_CANCER_CORE
                    if verdicts.get(axis) == "MISMATCH"
                )

                # ── Secondary axis scoring (additive to core score) ──
                secondary_boost = 0
                for axis, weights in SECONDARY_AXIS_WEIGHTS.items():
                    v = verdicts.get(axis)
                    if v and v in weights:
                        secondary_boost += weights[v]

                if non_cancer_mismatches > 0:
                    penalty = CORE_MISMATCH_PENALTY * non_cancer_mismatches
                    current_score = chunk.get("final_score", chunk.get("score", 0.5))
                    chunk["final_score"] = max(1, current_score - penalty + secondary_boost)
                    penalized_count += 1

                    chunk["patient_eligibility"] = {
                        "status": result.get("status", "POSSIBLE"),
                        "reason": result.get("reason", ""),
                        "criteria_verdicts": verdicts,
                        "penalty_applied": penalty,
                        "boost_applied": secondary_boost,
                        "hard_filtered": False,
                        "tier": "core_match_with_penalty",
                    }
                    title = chunk.get("title") or chunk.get("payload", {}).get("doc_meta", {}).get("title", "unknown")
                    mismatched = [a for a in _NON_CANCER_CORE if verdicts.get(a) == "MISMATCH"]
                    print(f"[PatientEligibility] PENALTY retained '{str(title)[:60]}' — non-cancer_type mismatch on: {mismatched}, penalty={penalty}, secondary_boost={secondary_boost}")
                else:
                    # No mismatches — boost matching studies
                    boost = result.get("boost", 0)
                    if boost > 0:
                        current_score = chunk.get("final_score", chunk.get("score", 0.5))
                        chunk["final_score"] = min(1.0, current_score + boost)
                        boosted_count += 1

                    # Add secondary boost on top of core boost
                    if secondary_boost != 0:
                        current_score = chunk.get("final_score", chunk.get("score", 0.5))
                        chunk["final_score"] = max(1, current_score + secondary_boost)

                    # Determine tier: "full_match" when all core axes
                    # are MATCH, "partial_match" when some are
                    # COMPATIBLE / NOT_AVAILABLE (no mismatches either way).
                    _CORE_AXES = ("cancer_type", "histology", "stage", "prior_therapies", "biomarkers")
                    all_core_match = all(
                        verdicts.get(axis) == "MATCH" for axis in _CORE_AXES
                    )
                    tier = "full_match" if all_core_match else "partial_match"

                    chunk["patient_eligibility"] = {
                        "status": result["status"],
                        "reason": result.get("reason", ""),
                        "criteria_verdicts": verdicts,
                        "penalty_applied": 0,
                        "boost_applied": (boost if boost > 0 else 0) + secondary_boost,
                        "hard_filtered": False,
                        "tier": tier,
                    }

        kept.append(chunk)

    if boosted_count > 0:
        print(f"[PatientEligibility] Boosted {boosted_count} chunks based on patient eligibility")

    if penalized_count > 0:
        print(f"[PatientEligibility] Penalized {penalized_count} chunks (tiered secondary mismatch)")

    if demoted_count > 0:
        print(f"[PatientEligibility] DEMOTED {demoted_count} cancer_type MISMATCH "
              f"verdicts to POSSIBLE (subsite hierarchy artifact)")
        try:
            from src.api.services import pipeline_metrics
            pipeline_metrics.incr("eligibility", "DEMOTED", demoted_count)
        except Exception:
            pass

    if removed:
        print(f"[PatientEligibility] Hard-filtered {len(removed)} chunks (confirmed mismatch)")

    # Re-sort kept chunks by final_score
    kept.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return kept, removed


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (old boost-only behavior removed)
# ---------------------------------------------------------------------------

def apply_patient_eligibility_boost(
    chunks: List[Dict[str, Any]],
    eligibility_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper — now delegates to filter_and_boost."""
    kept, _ = apply_patient_eligibility_filter_and_boost(chunks, eligibility_results)
    return kept


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_patient_eligibility_check(
    query: str,
    chunks: List[Dict[str, Any]],
    openai_client: OpenAI,
    resolver_hints: Optional[Any] = None,
    has_patient_context: Optional[bool] = None,
    reconciled: Optional[ReconciledStructure] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point: Extract patient context, check eligibility,
    hard-filter mismatched studies, and boost matching ones.

    When *has_patient_context* is explicitly ``False`` (set by the
    upstream ``_patient_signal_score`` threshold), the eligibility
    pipeline is skipped entirely and the studies are returned unmodified.

    When *reconciled* is provided (a ``ReconciledStructure`` from the
    query reconciliation module), patient context is populated from the
    reconciled structure instead of re-parsing the query text.

    Returns:
        Tuple of (filtered_and_boosted_chunks, eligibility_metadata)
    """
    # Early exit: upstream signal scoring determined there is insufficient
    # patient context to run eligibility checks (score < 2).
    if has_patient_context is False:
        print("[PatientEligibility] has_patient_context is False — skipping eligibility check")
        return chunks, {"patient_context_detected": False}

    patient_context = extract_patient_context_from_query(
        query, resolver_hints=resolver_hints, reconciled=reconciled,
    )

    if not patient_context:
        print(f"[PatientEligibility] No patient context detected in query")
        return chunks, {"patient_context_detected": False}

    print(f"[PatientEligibility] Detected patient context: {patient_context}")

    # Get unique doc_ids from chunks
    doc_ids = []
    seen = set()
    for chunk in chunks:
        doc_id = chunk.get("doc_id") or chunk.get("payload", {}).get("doc_id")
        if doc_id and doc_id not in seen:
            doc_ids.append(doc_id)
            seen.add(doc_id)

    if not doc_ids:
        return chunks, {"patient_context_detected": True, "doc_ids_checked": 0}

    # Check eligibility with per-criterion verdicts
    eligibility_results = await check_patient_eligibility_for_studies(
        query=query,
        patient_context=patient_context,
        doc_ids=doc_ids,
        openai_client=openai_client,
    )

    # Apply hard filter + boost. Passing patient_context enables the
    # subsite-hierarchy demotion guard (Fix B) so umbrella-vs-subsite
    # MISMATCH verdicts don't hard-filter legitimate evidence.
    filtered_chunks, removed_chunks = apply_patient_eligibility_filter_and_boost(
        chunks, eligibility_results, patient_context=patient_context,
        use_tiered_model=settings.use_reconciled_structure,
    )

    # If the majority of removed chunks were rejected on cancer_type specifically,
    # that indicates upstream category routing went wrong (wrong Qdrant bucket)
    # rather than genuinely strict multi-axis eligibility. Restoring those chunks
    # would fabricate confident answers over cancer-type-mismatched studies, so
    # instead accept the sparse result and flag the routing as suspect.
    category_routing_suspect = False
    if removed_chunks:
        cancer_type_mismatches = 0
        for rc in removed_chunks:
            verdicts = rc.get("patient_eligibility", {}).get("criteria_verdicts", {})
            if verdicts.get("cancer_type") == "MISMATCH":
                cancer_type_mismatches += 1
        if cancer_type_mismatches / len(removed_chunks) >= 0.8:
            category_routing_suspect = True
            print(
                f"[PatientEligibility] {cancer_type_mismatches}/{len(removed_chunks)} "
                f"removed chunks were cancer_type MISMATCH — suspect category routing, "
                f"refusing to restore. Returning {len(filtered_chunks)} accepted chunks."
            )

    # Safety: preserve a minimum number of evidence chunks.
    # If the hard filter is too aggressive (removes most/all results),
    # restore removed chunks at reduced scores rather than returning empty.
    # Skip this restoration when the removals look like a cancer-type routing
    # failure (see category_routing_suspect above).
    min_evidence_threshold = max(3, len(chunks) // 2)  # keep at least 3 or half
    if category_routing_suspect:
        # Path A: Honour the routing-suspect veto unconditionally.
        # Previously this branch restored "least-mismatched" wrong-cancer
        # studies when the bundle fell below MIN_STUDIES_FLOOR, on the
        # theory that some evidence beats none. For medical RAG the
        # opposite is true — restoring wrong-cancer evidence (cervical
        # cancer studies for a male H&N CUP patient, the canonical
        # failure mode logged earlier) silently fabricates confident
        # answers over the wrong literature. The clinically safe default
        # when retrieval misroutes is "no relevant evidence found".
        # Callers / downstream layers can detect the empty bundle and
        # surface that to the user. Future work: a rescue-retrieval
        # layer that re-queries with the corrected category instead of
        # accepting the sparse result here.
        if len(filtered_chunks) < MIN_STUDIES_FLOOR:
            print(
                f"[PatientEligibility] Bundle below floor ({len(filtered_chunks)}/"
                f"{MIN_STUDIES_FLOOR}) AND routing is suspect — NOT restoring "
                f"wrong-cancer chunks. Returning sparse/empty bundle to caller."
            )
        # else: keep the sparse result; caller will see the flag
    elif len(filtered_chunks) < min_evidence_threshold and len(chunks) >= min_evidence_threshold:
        # Surgical Path A: when the safety net fires, restore ONLY chunks
        # that were NOT rejected on a hard-drop axis. Hard-drop-rejected
        # chunks stay gone regardless of bundle size — these are studies
        # the patient is structurally ineligible for (wrong cancer / wrong
        # histology / wrong stage / wrong prior-therapy line / wrong disease
        # status / patient not a surgical candidate when surgery is required).
        # Surfacing them as "possible matches" fabricates evidence.
        # This generalizes the prior cancer_type-only protection: any axis
        # in HARD_DROP_AXES (or HARD_DROP_ON_MATCH_AXES) gets the same veto.
        hard_drop_rejected = [
            c for c in removed_chunks if _was_hard_dropped_for_axes(c)
        ]
        other_rejected = [
            c for c in removed_chunks if not _was_hard_dropped_for_axes(c)
        ]
        print(
            f"[PatientEligibility] Hard filter too aggressive ({len(filtered_chunks)}/{len(chunks)} "
            f"left, need {min_evidence_threshold}) — restoring {len(other_rejected)} non-hard-drop "
            f"removals at reduced scores; keeping {len(hard_drop_rejected)} hard-drop "
            f"rejects out (Path A)"
        )
        for chunk in other_rejected:
            chunk["final_score"] = chunk.get("final_score", chunk.get("score", 0.1)) * 0.3
        filtered_chunks = filtered_chunks + other_rejected
        filtered_chunks.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        removed_chunks = hard_drop_rejected
    elif len(filtered_chunks) == 0 and len(chunks) > 0 and not category_routing_suspect:
        # Absolute safety: never return empty if we had evidence — UNLESS
        # category routing is suspect, in which case "empty" is the
        # clinically safe answer (returning wrong-cancer chunks at
        # reduced scores would silently fabricate evidence over the
        # wrong literature; see Path A note in the suspect-routing
        # branch above).
        #
        # Even on this "absolute safety" branch, restore only chunks NOT
        # rejected on a hard-drop axis. If every chunk was hard-drop-rejected,
        # the bundle stays empty — better empty than misleading.
        hard_drop_rejected = [
            c for c in removed_chunks if _was_hard_dropped_for_axes(c)
        ]
        other_rejected = [
            c for c in removed_chunks if not _was_hard_dropped_for_axes(c)
        ]
        print(
            f"[PatientEligibility] Hard filter removed ALL {len(chunks)} chunks — restoring "
            f"{len(other_rejected)} non-hard-drop rejects at reduced scores; "
            f"keeping {len(hard_drop_rejected)} hard-drop rejects out (Path A)"
        )
        for chunk in other_rejected:
            chunk["final_score"] = chunk.get("final_score", chunk.get("score", 0.1)) * 0.3
        filtered_chunks = other_rejected
        filtered_chunks.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        removed_chunks = hard_drop_rejected
    elif len(filtered_chunks) == 0 and len(chunks) > 0 and category_routing_suspect:
        print(
            f"[PatientEligibility] Hard filter removed ALL {len(chunks)} chunks "
            f"AND routing is suspect — returning empty bundle "
            f"(refusing to fabricate evidence over wrong-cancer studies)."
        )

    metadata = {
        "patient_context_detected": True,
        "patient_context": patient_context,
        "patient_summary": build_patient_summary(patient_context),
        "doc_ids_checked": len(eligibility_results),
        "matches": sum(1 for r in eligibility_results.values() if r["status"] == "MATCH"),
        "possible_matches": sum(1 for r in eligibility_results.values() if r["status"] == "POSSIBLE"),
        "hard_filtered": len(removed_chunks),
        "penalized": sum(1 for r in eligibility_results.values() if r.get("has_hard_mismatch") and r["status"] != "NO_MATCH"),
        "eligibility_results": eligibility_results,
        "category_routing_suspect": category_routing_suspect,
    }

    return filtered_chunks, metadata
