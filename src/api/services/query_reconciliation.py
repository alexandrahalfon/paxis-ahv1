"""
Query reconciliation module.

Merges regex-extracted QueryStructure and LLM-extracted 8-axis dict into a
single authoritative ReconciledStructure for all downstream consumers
(PG_Matcher, PatientEligibility, Phase3Gate).

Feature: patient-study-match-scoring
"""

import re
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple

# Eagerly attempt to import normalize_category so the cost is paid at
# module-load time rather than inside the first reconcile() call (which
# would blow the Hypothesis deadline on cold start).
try:
    from src.api.services.comprehensive_retrieval import normalize_category as _ext_normalize_category
except ImportError:
    _ext_normalize_category = None  # type: ignore[assignment]


def _normalize_category_safe(value: Optional[str]) -> str:
    """Thin wrapper around the external normalize_category.

    Falls back to a simple lowercase-strip if the import failed.
    """
    if _ext_normalize_category is not None:
        return _ext_normalize_category(value)
    # Minimal fallback
    if not value or not value.strip():
        return ""
    return value.strip().lower()


class Source(Enum):
    """Tracks which extraction pipeline produced a reconciled field value."""
    REGEX = "regex"
    LLM = "llm"
    INFERRED = "inferred"
    AGREED = "agreed"


@dataclass
class Biomarker:
    """A molecular biomarker with optional polarity and provenance."""
    name: str
    polarity: Optional[str] = None  # "mutant", "wild-type", "positive", "negative"
    source: Source = Source.REGEX


@dataclass
class ReconciledStructure:
    """
    Single authoritative patient profile produced by merging regex-extracted
    and LLM-extracted fields.  All downstream consumers (PG_Matcher,
    PatientEligibility, Phase3Gate) read from this structure when the
    USE_RECONCILED_STRUCTURE feature flag is true.
    """

    # Cancer context
    cancer_site: Optional[str] = None
    cancer_site_source: Source = Source.REGEX
    histology: Optional[str] = None
    histology_source: Source = Source.REGEX
    stage: Optional[str] = None
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    biomarkers: List[Biomarker] = field(default_factory=list)
    receptor_status: Optional[str] = None

    # Patient context
    age: Optional[int] = None
    gender: Optional[str] = None
    performance_status: Optional[str] = None

    # Treatment context
    prior_treatments: List[str] = field(default_factory=list)
    treatment_setting: Optional[str] = None
    treatment_modality: Optional[str] = None

    # LLM 8-axis raw text spans (preserved for semantic search)
    llm_axes: Dict[str, str] = field(default_factory=dict)

    # Metadata
    filter_category: Optional[str] = None
    has_patient_context: bool = False
    disagreements: List[Dict[str, Any]] = field(default_factory=list)

    # Trajectory fields for PG scoring
    disease_trajectory: Optional[str] = None  # "recurrent", "metastatic", "treatment-naive"
    metastatic_status: Optional[str] = None   # for PG metastatic_status column matching
    risk_level: Optional[str] = None          # "high", "intermediate", "low"

    def pg_metastatic_status(self) -> Optional[str]:
        """Return the value to match against studies.metastatic_status column."""
        return self.metastatic_status

    def to_query_structure_dict(self) -> Dict[str, Any]:
        """
        Convert to dict compatible with PG matcher and PatientEligibility.

        Produces the same nested structure as ``QueryStructure.to_dict()``
        so that ``match_studies_by_structure()`` can consume it without
        changes.
        """
        # Build biomarker strings in the format the PG matcher expects
        # e.g. ["EGFR mutant", "HER2 positive"]
        biomarker_strings: List[str] = []
        for bm in self.biomarkers:
            if bm.polarity:
                biomarker_strings.append(f"{bm.name} {bm.polarity}")
            else:
                biomarker_strings.append(bm.name)

        cancer: Dict[str, Any] = {}
        if self.cancer_site is not None:
            cancer["site"] = self.cancer_site
        if self.histology is not None:
            cancer["histology"] = self.histology
        if self.stage is not None:
            cancer["stage"] = self.stage
        if self.tnm_t is not None:
            cancer["tnm_t"] = self.tnm_t
        if self.tnm_n is not None:
            cancer["tnm_n"] = self.tnm_n
        if self.tnm_m is not None:
            cancer["tnm_m"] = self.tnm_m
        if biomarker_strings:
            cancer["biomarkers"] = biomarker_strings
        if self.receptor_status is not None:
            cancer["receptor_status"] = self.receptor_status

        patient: Dict[str, Any] = {}
        if self.age is not None:
            patient["age"] = self.age
        if self.gender is not None:
            patient["gender"] = self.gender
        if self.performance_status is not None:
            patient["performance_status"] = self.performance_status

        treatment: Dict[str, Any] = {}
        if self.prior_treatments:
            treatment["prior_treatments"] = self.prior_treatments
        if self.treatment_setting is not None:
            treatment["setting"] = self.treatment_setting
        if self.treatment_modality is not None:
            treatment["modality"] = self.treatment_modality

        result: Dict[str, Any] = {
            "has_patient_context": self.has_patient_context,
        }
        if cancer:
            result["cancer"] = cancer
        if patient:
            result["patient"] = patient
        if treatment:
            result["treatment"] = treatment
        if self.filter_category is not None:
            result["filter_category"] = self.filter_category

        return result


def _parse_biomarker_string(raw: str) -> List[Tuple[str, Optional[str]]]:
    """Parse a biomarker string like ``"EGFR mutant, HER2 positive"`` into
    a list of ``(name, polarity)`` tuples.

    Handles formats:
    - ``"EGFR mutant"``
    - ``"EGFR-mutant"``
    - ``"EGFR"`` (no polarity)
    - ``"EGFR mutant, HER2 positive"`` (comma-separated)
    """
    if not raw or not raw.strip():
        return []

    known_polarities = {
        "mutant", "wild-type", "wildtype", "wt",
        "positive", "negative",
        "amplified", "overexpressed",
        "high", "low",
    }

    results: List[Tuple[str, Optional[str]]] = []
    # Split on comma to handle multiple biomarkers
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        # Try splitting on space or hyphen
        # e.g. "EGFR mutant" or "EGFR-mutant" or "PD-L1 positive"
        # Handle hyphenated biomarker names like PD-L1 by checking if the
        # last token is a known polarity
        parts = re.split(r"[\s]+", segment)
        if len(parts) >= 2 and parts[-1].lower() in known_polarities:
            name = " ".join(parts[:-1]).strip()
            polarity = parts[-1].lower()
            # Normalize wildtype variants
            if polarity in ("wildtype", "wt"):
                polarity = "wild-type"
            results.append((name, polarity))
        elif len(parts) == 1 and "-" in segment:
            # Try splitting on last hyphen: "EGFR-mutant"
            last_hyphen = segment.rfind("-")
            candidate_pol = segment[last_hyphen + 1:].lower()
            if candidate_pol in known_polarities:
                name = segment[:last_hyphen].strip()
                polarity = candidate_pol
                if polarity in ("wildtype", "wt"):
                    polarity = "wild-type"
                results.append((name, polarity))
            else:
                results.append((segment, None))
        else:
            # No polarity detected — just a name
            results.append((segment, None))

    return results


def _parse_demographics(demographics_str: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract age and gender from an LLM demographics string.

    Handles formats like ``"55 year old male"``, ``"65yo female"``, etc.
    Returns ``(age, gender)``.
    """
    age: Optional[int] = None
    gender: Optional[str] = None

    if not demographics_str or not demographics_str.strip():
        return age, gender

    text = demographics_str.strip().lower()

    # Extract age
    age_match = re.search(r"(\d{1,3})\s*(?:year|yo|y\.?o\.?)", text)
    if age_match:
        try:
            age = int(age_match.group(1))
        except ValueError:
            pass

    # Extract gender
    gender_match = re.search(r"\b(male|female|man|woman)\b", text)
    if gender_match:
        g = gender_match.group(1)
        if g in ("male", "man"):
            gender = "male"
        elif g in ("female", "woman"):
            gender = "female"

    return age, gender


def _detect_trajectory(
    llm_dict: Dict[str, str],
    regex_stage: Optional[str],
    regex_tnm_m: Optional[str],
) -> Optional[str]:
    """Detect disease trajectory from LLM data and regex extraction.

    Returns one of: ``"recurrent"``, ``"metastatic"``, ``"treatment-naive"``,
    ``"progressive"``, ``"locally_advanced"``, or ``None``.
    """
    trajectory_indicators = {
        "recurrent": ["recurrent", "recurrence", "relapsed"],
        "metastatic": ["metastatic", "metastases", "distant metastases"],
        "treatment-naive": ["treatment-naive", "treatment naive", "newly diagnosed", "untreated"],
        "progressive": ["progressive", "progression"],
        "locally_advanced": ["locally advanced"],
    }

    # Check all LLM axis values for trajectory keywords
    combined_text = " ".join(v for v in llm_dict.values() if v).lower()

    for trajectory, keywords in trajectory_indicators.items():
        for kw in keywords:
            if kw in combined_text:
                return trajectory

    # Infer from stage: stage IV or M1 → metastatic
    if regex_stage and regex_stage.upper().startswith("IV"):
        return "metastatic"
    if regex_tnm_m == "1":
        return "metastatic"

    return None


def _detect_metastatic_status(
    trajectory: Optional[str],
    regex_stage: Optional[str],
    regex_tnm_m: Optional[str],
    llm_dict: Dict[str, str],
) -> Optional[str]:
    """Detect metastatic status from trajectory, stage, or LLM data."""
    if trajectory == "metastatic":
        return "metastatic"

    # Check M1 in TNM
    if regex_tnm_m == "1":
        return "metastatic"

    # Check stage IV
    if regex_stage and regex_stage.upper().startswith("IV"):
        return "metastatic"

    # Check LLM text for explicit mentions
    combined_text = " ".join(v for v in llm_dict.values() if v).lower()
    if "non-metastatic" in combined_text or "non metastatic" in combined_text:
        return "non-metastatic"
    if "metastatic" in combined_text or "metastases" in combined_text:
        return "metastatic"
    if "locally advanced" in combined_text:
        return "locally_advanced"

    return None


def _detect_risk_level(llm_dict: Dict[str, str]) -> Optional[str]:
    """Detect risk level from LLM data."""
    combined_text = " ".join(v for v in llm_dict.values() if v).lower()

    risk_patterns = [
        ("high-risk", ["high-risk", "high risk"]),
        ("intermediate-risk", ["intermediate-risk", "intermediate risk"]),
        ("low-risk", ["low-risk", "low risk"]),
        ("favorable", ["favorable"]),
        ("unfavorable", ["unfavorable"]),
    ]

    for risk_level, keywords in risk_patterns:
        for kw in keywords:
            if kw in combined_text:
                return risk_level

    return None


def _log_disagreement(
    disagreements: List[Dict[str, Any]],
    field_name: str,
    regex_value: Any,
    llm_value: Any,
    winner: str,
) -> None:
    """Record a disagreement between regex and LLM values."""
    disagreements.append({
        "field": field_name,
        "regex_value": regex_value,
        "llm_value": llm_value,
        "winner": winner,
    })
    print(
        f"[QueryReconciliation] Disagreement on '{field_name}': "
        f"regex='{regex_value}', llm='{llm_value}', winner='{winner}'"
    )


def reconcile_if_enabled(
    regex_structure: Any,
    llm_dict: Dict[str, str],
) -> Optional["ReconciledStructure"]:
    """Gate reconciliation behind the USE_RECONCILED_STRUCTURE feature flag.

    When the flag is false (default), returns None so that downstream
    consumers continue to use the original QueryStructure unchanged.
    When the flag is true, delegates to ``reconcile()`` and returns the
    resulting ``ReconciledStructure``.
    """
    from src.core.config import settings

    if not settings.use_reconciled_structure:
        return None
    return reconcile(regex_structure, llm_dict)


def reconcile(
    regex_structure: Any,
    llm_dict: Dict[str, str],
) -> "ReconciledStructure":
    """Merge regex-extracted QueryStructure and LLM 8-axis dict.

    Priority rules per the design document:
    - Site: regex > LLM
    - Histology: LLM > regex
    - Biomarkers: LLM wins on polarity
    - Stage: regex for explicit, LLM for inferred
    - Trajectory: inference engine authoritative
    - Demographics (age, gender): regex
    - Category: recomputed from reconciled site via normalize_category()
    """
    if llm_dict is None:
        llm_dict = {}

    disagreements: List[Dict[str, Any]] = []

    # ── Collect LLM axes (preserve all non-empty values) ──────────────
    llm_axes: Dict[str, str] = {}
    for key, value in llm_dict.items():
        if value and value.strip():
            llm_axes[key] = value

    # ── Extract regex fields ──────────────────────────────────────────
    regex_site = getattr(regex_structure.cancer, "site", None)
    regex_histology = getattr(regex_structure.cancer, "histology", None)
    regex_stage = getattr(regex_structure.cancer, "stage", None)
    regex_tnm_t = getattr(regex_structure.cancer, "tnm_t", None)
    regex_tnm_n = getattr(regex_structure.cancer, "tnm_n", None)
    regex_tnm_m = getattr(regex_structure.cancer, "tnm_m", None)
    regex_biomarkers_raw: List[str] = getattr(regex_structure.cancer, "biomarkers", []) or []
    regex_receptor_status = getattr(regex_structure.cancer, "receptor_status", None)
    regex_age = getattr(regex_structure.patient, "age", None)
    regex_gender = getattr(regex_structure.patient, "gender", None)
    regex_perf_status = getattr(regex_structure.patient, "performance_status", None)
    regex_prior_tx: List[str] = getattr(regex_structure.treatment, "prior_treatments", []) or []
    regex_modality = getattr(regex_structure.treatment, "modality", None)
    regex_setting = getattr(regex_structure.treatment, "setting", None)
    regex_filter_category = getattr(regex_structure, "filter_category", None)
    regex_has_patient_context = getattr(regex_structure, "has_patient_context", False)

    # ── Extract LLM fields ────────────────────────────────────────────
    llm_site = llm_dict.get("cancer_type", "").strip() or None
    llm_histology = llm_dict.get("histology", "").strip() or None
    llm_stage = llm_dict.get("stage", "").strip() or None
    llm_biomarkers_raw = llm_dict.get("biomarkers", "").strip() or None
    llm_prior_tx = llm_dict.get("prior_treatments", "").strip() or None
    llm_setting = llm_dict.get("treatment_setting", "").strip() or None
    llm_demographics = llm_dict.get("demographics", "").strip() or None

    # ── Site: regex > LLM ─────────────────────────────────────────────
    cancer_site: Optional[str] = None
    cancer_site_source = Source.REGEX

    if regex_site and llm_site:
        if regex_site.lower() == llm_site.lower():
            cancer_site = regex_site
            cancer_site_source = Source.AGREED
        else:
            # Regex wins
            cancer_site = regex_site
            cancer_site_source = Source.REGEX
            _log_disagreement(disagreements, "site", regex_site, llm_site, "regex")
    elif regex_site:
        cancer_site = regex_site
        cancer_site_source = Source.REGEX
    elif llm_site:
        cancer_site = llm_site
        cancer_site_source = Source.LLM

    # ── Histology: LLM > regex ────────────────────────────────────────
    histology: Optional[str] = None
    histology_source = Source.REGEX

    if regex_histology and llm_histology:
        if regex_histology.lower() == llm_histology.lower():
            histology = regex_histology
            histology_source = Source.AGREED
        else:
            # LLM wins
            histology = llm_histology
            histology_source = Source.LLM
            _log_disagreement(disagreements, "histology", regex_histology, llm_histology, "llm")
    elif regex_histology:
        histology = regex_histology
        histology_source = Source.REGEX
    elif llm_histology:
        histology = llm_histology
        histology_source = Source.LLM

    # ── Stage: regex for explicit, LLM for inferred ───────────────────
    stage: Optional[str] = None

    if regex_stage and llm_stage:
        if regex_stage.upper() == llm_stage.upper():
            stage = regex_stage
        else:
            # Regex wins for explicit staging
            stage = regex_stage
            _log_disagreement(disagreements, "stage", regex_stage, llm_stage, "regex")
    elif regex_stage:
        stage = regex_stage
    elif llm_stage:
        stage = llm_stage

    # ── TNM: regex authoritative ──────────────────────────────────────
    tnm_t = regex_tnm_t
    tnm_n = regex_tnm_n
    tnm_m = regex_tnm_m

    # ── Biomarkers: LLM wins on polarity ──────────────────────────────
    regex_parsed = _parse_biomarker_string(", ".join(regex_biomarkers_raw))
    llm_parsed = _parse_biomarker_string(llm_biomarkers_raw or "")

    # Build lookup by uppercase name for LLM biomarkers
    llm_bm_map: Dict[str, Optional[str]] = {}
    for name, pol in llm_parsed:
        llm_bm_map[name.upper()] = pol

    reconciled_biomarkers: List[Biomarker] = []
    seen_names: set = set()

    for name, regex_pol in regex_parsed:
        upper_name = name.upper()
        seen_names.add(upper_name)
        llm_pol = llm_bm_map.get(upper_name)

        if llm_pol is not None and regex_pol is not None and llm_pol != regex_pol:
            # LLM wins on polarity disagreement
            reconciled_biomarkers.append(
                Biomarker(name=name, polarity=llm_pol, source=Source.LLM)
            )
            _log_disagreement(
                disagreements,
                f"biomarker_{name}",
                f"{name} {regex_pol}",
                f"{name} {llm_pol}",
                "llm",
            )
        elif llm_pol is not None and regex_pol is not None and llm_pol == regex_pol:
            reconciled_biomarkers.append(
                Biomarker(name=name, polarity=regex_pol, source=Source.AGREED)
            )
        else:
            # Only one side has polarity, or neither does
            polarity = llm_pol if llm_pol is not None else regex_pol
            source = Source.LLM if llm_pol is not None and regex_pol is None else Source.REGEX
            reconciled_biomarkers.append(
                Biomarker(name=name, polarity=polarity, source=source)
            )

    # Add LLM-only biomarkers not seen in regex
    for name, pol in llm_parsed:
        if name.upper() not in seen_names:
            seen_names.add(name.upper())
            reconciled_biomarkers.append(
                Biomarker(name=name, polarity=pol, source=Source.LLM)
            )

    # ── Demographics: regex wins ──────────────────────────────────────
    llm_age, llm_gender = _parse_demographics(llm_demographics or "")

    age = regex_age if regex_age is not None else llm_age
    gender = regex_gender if regex_gender is not None else llm_gender

    if regex_age is not None and llm_age is not None and regex_age != llm_age:
        _log_disagreement(disagreements, "age", regex_age, llm_age, "regex")
    if regex_gender is not None and llm_gender is not None and regex_gender != llm_gender:
        _log_disagreement(disagreements, "gender", regex_gender, llm_gender, "regex")

    # ── Treatment context ─────────────────────────────────────────────
    prior_treatments = regex_prior_tx if regex_prior_tx else []
    if not prior_treatments and llm_prior_tx:
        prior_treatments = [t.strip() for t in llm_prior_tx.split(",") if t.strip()]

    treatment_setting = regex_setting if regex_setting else llm_setting
    treatment_modality = regex_modality

    # ── Receptor status ───────────────────────────────────────────────
    receptor_status = regex_receptor_status

    # ── Performance status ────────────────────────────────────────────
    performance_status = regex_perf_status

    # ── Trajectory: inference engine authoritative ─────────────────────
    disease_trajectory = _detect_trajectory(llm_dict, regex_stage, regex_tnm_m)

    # ── Metastatic status ─────────────────────────────────────────────
    metastatic_status = _detect_metastatic_status(
        disease_trajectory, regex_stage, regex_tnm_m, llm_dict
    )

    # ── Risk level ────────────────────────────────────────────────────
    risk_level = _detect_risk_level(llm_dict)

    # ── Category: recomputed from reconciled site ─────────────────────
    filter_category = regex_filter_category
    if cancer_site:
        try:
            normalized = _normalize_category_safe(cancer_site)
            if normalized:
                filter_category = normalized
        except Exception:
            # Fallback to regex filter_category
            pass

    # ── has_patient_context ───────────────────────────────────────────
    has_patient_context = regex_has_patient_context

    return ReconciledStructure(
        cancer_site=cancer_site,
        cancer_site_source=cancer_site_source,
        histology=histology,
        histology_source=histology_source,
        stage=stage,
        tnm_t=tnm_t,
        tnm_n=tnm_n,
        tnm_m=tnm_m,
        biomarkers=reconciled_biomarkers,
        receptor_status=receptor_status,
        age=age,
        gender=gender,
        performance_status=performance_status,
        prior_treatments=prior_treatments,
        treatment_setting=treatment_setting,
        treatment_modality=treatment_modality,
        llm_axes=llm_axes,
        filter_category=filter_category,
        has_patient_context=has_patient_context,
        disagreements=disagreements,
        disease_trajectory=disease_trajectory,
        metastatic_status=metastatic_status,
        risk_level=risk_level,
    )
