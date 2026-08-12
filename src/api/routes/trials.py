"""
API routes for external ClinicalTrials.gov search.
"""

from typing import Dict, Any, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from openai import OpenAI

from src.core.config import settings
from src.api.models.trial_search_models import (
    TrialSearchRequest,
    TrialSearchResponse,
    TrialResult,
)
from src.api.services.literature_search_service import LiteratureSearchService
from src.api.services.unstructured_patient_extractor import extract_patient_profile
from src.api.services.staging_inference_validator import validate_and_correct_staging

router = APIRouter(prefix="/trials", tags=["Trials"])
search_service = LiteratureSearchService()


def _merge_profile(
    extracted: Optional[Dict[str, Any]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    if extracted:
        profile.update({k: v for k, v in extracted.items() if v is not None})
    if override:
        profile.update({k: v for k, v in override.items() if v is not None})
    return profile


def _age_to_clinical_term(age: Any) -> Optional[str]:
    """Map a numeric age to a clinical term that ClinicalTrials.gov understands."""
    try:
        age_val = int(age)
    except (TypeError, ValueError):
        return None
    if age_val < 18:
        return "pediatric"
    if age_val >= 65:
        return "elderly"
    return None  # 18-64 is the default adult range — no extra term needed


def _build_condition_and_terms(
    profile: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    """
    Split the extracted profile into a *condition* string (for ``query.cond``)
    and remaining *other_terms* (for ``query.term``).

    ClinicalTrials.gov produces far better results when the disease/condition
    goes into ``query.cond`` and supplementary clinical details (markers, stage,
    histology, demographics) go into ``query.term``.
    """
    # --- condition ---
    cond_parts: List[str] = []
    cancer_type = profile.get("cancer_type")
    anatomical_site = profile.get("anatomical_site")
    if cancer_type:
        cond_parts.append(str(cancer_type))
    if anatomical_site and (not cancer_type or str(anatomical_site).lower() not in str(cancer_type).lower()):
        cond_parts.append(str(anatomical_site))
    condition = " ".join(cond_parts) if cond_parts else None

    # --- other terms (clinical + demographic) ---
    terms: List[str] = []

    # Clinical characteristics
    histology = profile.get("histology")
    stage = profile.get("cancer_stage")
    markers = profile.get("molecular_markers") or []

    if histology:
        terms.append(str(histology))
    if stage:
        s = str(stage).strip()
        terms.append(s if s.upper().startswith("STAGE") else f"Stage {s}")
    for marker in markers:
        if marker:
            terms.append(str(marker))

    # Patient demographics / characteristics
    age = profile.get("age")
    age_term = _age_to_clinical_term(age)
    if age_term:
        terms.append(age_term)

    gender = profile.get("gender")
    if gender:
        terms.append(str(gender))

    ecog = profile.get("performance_status")
    if ecog is not None and str(ecog).strip():
        terms.append(f"ECOG {ecog}")

    smoking = profile.get("smoking_status")
    if smoking:
        terms.append(f"{smoking} smoker" if smoking.lower() != "never" else "never smoker")

    prior_tx = profile.get("prior_treatments") or []
    if isinstance(prior_tx, str):
        prior_tx = [t.strip() for t in prior_tx.split(",") if t.strip()]
    for tx in prior_tx[:3]:  # Limit to avoid over-stuffing the query
        if tx:
            terms.append(str(tx))

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for t in terms:
        norm = t.strip().lower()
        if norm and norm not in seen:
            deduped.append(t.strip())
            seen.add(norm)
    return condition, deduped


def _normalize_phase_list(phases: Any) -> List[str]:
    phase_map = {
        "EARLY_PHASE1": "I",
        "PHASE1": "I",
        "PHASE2": "II",
        "PHASE3": "III",
        "PHASE4": "IV",
    }
    if isinstance(phases, str):
        phases = [phases]
    normalized = []
    for phase in phases or []:
        key = str(phase).replace(" ", "").upper()
        mapped = phase_map.get(key, None)
        if mapped:
            normalized.append(mapped)
        else:
            normalized.append(str(phase))
    return normalized


def _format_locations(locations: List[Dict[str, Any]]) -> List[str]:
    formatted = []
    for loc in locations or []:
        facility = (loc.get("facility") or "").strip()
        city = (loc.get("city") or "").strip()
        state = (loc.get("state") or "").strip()
        country = (loc.get("country") or "").strip()
        parts = [p for p in [facility, city, state, country] if p]
        if parts:
            formatted.append(", ".join(parts))
    return formatted


def _extract_trial(study: Dict[str, Any]) -> Dict[str, Any]:
    protocol = study.get("protocolSection", {}) or {}
    identification = protocol.get("identificationModule", {}) or {}
    status = protocol.get("statusModule", {}) or {}
    conditions = protocol.get("conditionsModule", {}) or {}
    design = protocol.get("designModule", {}) or {}
    arms = protocol.get("armsInterventionsModule", {}) or {}
    description = protocol.get("descriptionModule", {}) or {}
    eligibility = protocol.get("eligibilityModule", {}) or {}
    contacts = protocol.get("contactsLocationsModule", {}) or {}

    nct_id = identification.get("nctId") or ""
    interventions = [
        i.get("name")
        for i in (arms.get("interventions") or [])
        if i.get("name")
    ]

    return {
        "title": identification.get("briefTitle") or "Untitled trial",
        "nct_id": nct_id,
        "status": status.get("overallStatus"),
        "phase": _normalize_phase_list(design.get("phases")),
        "conditions": conditions.get("conditions") or [],
        "interventions": interventions,
        "locations": _format_locations(contacts.get("locations") or []),
        "summary": description.get("briefSummary") or description.get("detailedDescription"),
        "eligibility_criteria": eligibility.get("eligibilityCriteria"),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
    }


def _score_trial(trial: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score how well a ClinicalTrials.gov result matches the patient profile.

    Weights (max 1.0):
      Disease / tumour:
        cancer_type        0.30
        anatomical_site    0.10
        histology          0.08
        stage              0.08
        molecular_markers  0.04 each (up to ~0.12)
      Patient characteristics:
        age (range fit)    0.08
        gender             0.05
        performance_status 0.05
        smoking_status     0.04
        prior_treatments   0.03 each (up to ~0.09)
    """
    import re as _re

    if not profile:
        return 0.0, []

    text_blob = " ".join(
        [
            trial.get("title") or "",
            trial.get("summary") or "",
            trial.get("eligibility_criteria") or "",
            " ".join(trial.get("conditions") or []),
            " ".join(trial.get("interventions") or []),
        ]
    ).lower()

    score = 0.0
    reasons: List[str] = []

    def _add(term: Optional[str], weight: float, label: str):
        nonlocal score
        if term and term.lower() in text_blob:
            score += weight
            reasons.append(label)

    # --- Disease / tumour ---
    _add(profile.get("cancer_type"), 0.30, "Cancer type match")
    _add(profile.get("anatomical_site"), 0.10, "Anatomical site match")
    _add(profile.get("histology"), 0.08, "Histology match")

    stage = profile.get("cancer_stage")
    if stage:
        stage_term = f"stage {str(stage).lower().strip()}"
        if stage_term in text_blob:
            score += 0.08
            reasons.append("Stage match")

    markers = profile.get("molecular_markers") or []
    for marker in markers:
        if marker and marker.lower() in text_blob:
            score += 0.04
            reasons.append(f"Marker: {marker}")

    # --- Patient characteristics ---
    # Age — check if patient's age falls within any age range mentioned in
    # the eligibility text (e.g. "18 years to 75 years")
    age = profile.get("age")
    if age is not None:
        try:
            age_val = int(age)
            # Look for patterns like "≥ 18" or "18 Years" or "age 18-75"
            age_fits = False
            # Simple heuristic: find all numbers near "year" in eligibility
            eligibility = (trial.get("eligibility_criteria") or "").lower()
            # Check min/max age from ClinicalTrials.gov eligibility module
            nums = [int(n) for n in _re.findall(r"(\d{1,3})\s*(?:years?|yrs?)", eligibility)]
            if len(nums) >= 2:
                lo, hi = min(nums), max(nums)
                if lo <= age_val <= hi:
                    age_fits = True
            elif len(nums) == 1:
                # Single bound — likely a minimum
                if age_val >= nums[0]:
                    age_fits = True
            # Also match clinical terms
            age_term = _age_to_clinical_term(age_val)
            if age_term and age_term in text_blob:
                age_fits = True
            if age_fits:
                score += 0.08
                reasons.append(f"Age {age_val} within range")
        except (TypeError, ValueError):
            pass

    gender = profile.get("gender")
    if gender:
        g = gender.lower()
        # Match "male", "female", "men", "women", "man", "woman"
        gender_synonyms = {"male": ["male", "men", "man"], "female": ["female", "women", "woman"]}
        for syn in gender_synonyms.get(g, [g]):
            if syn in text_blob:
                score += 0.05
                reasons.append(f"Gender: {gender}")
                break

    ecog = profile.get("performance_status")
    if ecog is not None and str(ecog).strip():
        ecog_str = str(ecog).strip()
        # Match "ECOG 0-1", "ECOG ≤ 2", "ECOG 0 or 1", "ECOG performance status 0-1"
        if f"ecog" in text_blob and ecog_str in text_blob:
            score += 0.05
            reasons.append(f"ECOG {ecog_str} eligible")

    smoking = profile.get("smoking_status")
    if smoking:
        smoking_terms = {
            "never": ["never smok", "non-smok", "nonsmoking", "never-smok"],
            "former": ["former smok", "ex-smok", "previously smok"],
            "current": ["current smok", "active smok"],
        }
        for term in smoking_terms.get(smoking.lower(), [smoking.lower()]):
            if term in text_blob:
                score += 0.04
                reasons.append(f"Smoking: {smoking}")
                break

    prior_tx = profile.get("prior_treatments") or []
    if isinstance(prior_tx, str):
        prior_tx = [t.strip() for t in prior_tx.split(",") if t.strip()]
    for tx in prior_tx[:3]:
        if tx and tx.lower() in text_blob:
            score += 0.03
            reasons.append(f"Prior treatment: {tx}")

    return min(score, 1.0), reasons


def _matches_filter(value: Optional[str], filters: Optional[List[str]]) -> bool:
    if not filters:
        return True
    if not value:
        return False
    normalized = value.strip().lower()
    return any(normalized == f.strip().lower() for f in filters if f)


def _phase_matches(phases: List[str], filters: Optional[List[str]]) -> bool:
    if not filters:
        return True
    if not phases:
        return False
    normalized_phases = {p.strip().upper() for p in phases if p}
    for f in filters:
        if not f:
            continue
        if f.strip().upper() in normalized_phases:
            return True
    return False


@router.post("/search", response_model=TrialSearchResponse)
async def search_trials(request: TrialSearchRequest):
    if not request.query and not request.patient_profile:
        raise HTTPException(
            status_code=400,
            detail="Provide a case description or structured patient profile.",
        )

    extracted_profile = None
    if request.query:
        openai_client = OpenAI(api_key=settings.openai_api_key)
        extracted_profile = extract_patient_profile(request.query, openai_client)
        
        # Validate and correct staging using AJCC rules
        if extracted_profile:
            extracted_profile = validate_and_correct_staging(extracted_profile, request.query)

    override_profile = (
        request.patient_profile.dict(exclude_none=True)
        if request.patient_profile
        else None
    )
    profile = _merge_profile(extracted_profile, override_profile)

    # Build structured params for ClinicalTrials.gov
    condition, other_terms_list = _build_condition_and_terms(profile)
    other_terms = " ".join(other_terms_list) if other_terms_list else None

    # Assemble a flat list for the response so the caller can see what we searched
    query_terms_display: List[str] = []
    if condition:
        query_terms_display.append(f"Condition: {condition}")
    if other_terms_list:
        query_terms_display.extend(other_terms_list)

    raw_studies = search_service.search_clinical_trials_structured(
        condition=condition,
        other_terms=other_terms,
        max_results=request.max_results,
    )

    trials: List[TrialResult] = []
    for study in raw_studies:
        normalized = _extract_trial(study)

        if request.recruiting_status and not _matches_filter(
            normalized.get("status"), request.recruiting_status
        ):
            continue
        if request.phase and not _phase_matches(
            normalized.get("phase") or [], request.phase
        ):
            continue
        if request.location:
            location_term = request.location.strip().lower()
            if location_term:
                locations_blob = " ".join(normalized.get("locations") or []).lower()
                if location_term not in locations_blob:
                    continue

        score, reasons = _score_trial(normalized, profile)
        normalized["match_score"] = score
        normalized["match_reasons"] = reasons
        trials.append(TrialResult(**normalized))

    trials.sort(key=lambda t: t.match_score, reverse=True)

    return TrialSearchResponse(
        trials=trials,
        total_results=len(trials),
        extracted_profile=extracted_profile,
        query_terms=query_terms_display,
    )
