"""
Physician Applicability Scorer (2026-08-12 convergence Sprint C item 16)

The physician-side counterpart to evidence/applicability_scorer.py,
same conceptual shape: "is this passage relevant to the query?" (already
answered by comprehensive_retrieval.py's dense/lexical/cross-encoder
scoring — see clinical_retrieval_adapter.py, Sprint C item 15) is a
DIFFERENT question from "does this evidence apply to THIS patient?",
which is what this module answers, run AFTER the existing retriever, not
instead of it.

Fifteen components (per the convergence plan's own list): cancer,
histology, stage, biomarker, treatment_line, prior_treatment,
active_treatment, drug_regimen, performance_status, organ_function,
study_population, outcome, evidence_type, authority, freshness. Combined
with intent-specific weights (WEIGHTS_BY_PHYSICIAN_INTENT, keyed by
physician_context_service.py's four named intents) rather than one fixed
blend — a therapy_selection question weights biomarker/cancer/line/
prior_treatment heaviest; toxicity_management weights drug_regimen/
active_treatment/authority (standing in for "regulatory/guideline
source") heaviest; trial_eligibility weights biomarker/stage/
performance_status/organ_function/prior_treatment/study_population
heaviest. Every weight set sums to 1.0 (enforced in tests, matching
applicability_scorer.py's own convention).

Reuses applicability_scorer._set_match() directly — the same neutral-
when-unspecified / 0.75-for-general-token / 1.0-named-match /
0.0-named-mismatch semantics, rather than a second, potentially
drifting reimplementation of the same matching rule.

evidence_type's score comes from evidence_hierarchy.py (Sprint A item
5) directly: infer_evidence_type() + authority_prior() against
select_hierarchy(audience="physician", intent=...) — reused as this
one component's scoring mechanism rather than duplicating tier logic.
This is a SEPARATE, smaller use of evidence_hierarchy.py than
apply_authority_prior()'s own intended post-scoring pass; a caller can
still layer apply_authority_prior() on top of this scorer's output for
an additional authority tilt, since evidence_hierarchy.py is designed
to compose with whatever relevance/applicability score preceded it.

Structured per-axis tags (candidate.metadata["applicability_meta"]) are
not populated by clinical_retrieval_adapter.py today — the legacy corpus
predates that kind of metadata classification entirely (same documented
gap as that module's version_id/rrf_score). Every component therefore
falls back to the SAME text-containment check applicability_scorer.py's
_set_match() already does for an unspecified/untagged axis: does a
patient axis TERM the record already asserts appear literally in the
chunk text. This is honest given the corpus's current state, not a
placeholder pretending to be real classification.

Incompatibility (Sprint C item 17 builds the full typed taxonomy on top
of this): only ONE hard-mismatch penalty exists here so far —
biomarker, mirroring applicability_scorer.py's own single modality-
conflict penalty exactly (a named, non-overlapping, textually-
uncorroborated mismatch on the single axis most likely to cause real
clinical harm if ignored — recommending an EGFR-targeted therapy passage
for a documented KRAS G12C patient). incompatibility_reasons here is a
plain string list, same shape as the patient scorer's; C17 is what adds
typed/structured reasons (hard vs. soft vs. unknown) on top.

Nothing calls score_candidate()/rank() here yet — same as every other
Sprint C piece landing before Sprint C item 20's physician orchestrator
wires it in, after clinical_retrieval_adapter.py's output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.api.services.evidence.applicability_scorer import _set_match
from src.api.services.evidence import evidence_hierarchy
from src.api.services.physician.physician_context_service import (
    THERAPY_SELECTION, TREATMENT_SEQUENCING, TOXICITY_MANAGEMENT, TRIAL_ELIGIBILITY,
)

_GENERAL_CANCER_TOKEN = "all"

# Applied multiplicatively to the combined score on a detected biomarker
# conflict — see module docstring for why biomarker, specifically,
# mirrors applicability_scorer.py's single modality-conflict penalty.
_BIOMARKER_CONFLICT_PENALTY = 0.35

_COMPONENT_NAMES = (
    "cancer", "histology", "stage", "biomarker", "treatment_line",
    "prior_treatment", "active_treatment", "drug_regimen",
    "performance_status", "organ_function", "study_population",
    "outcome", "evidence_type", "authority", "freshness",
)

# Every row sums to 1.0 -- see module docstring for the "heavier" fields
# named per the convergence plan; the remaining components fill out a
# sensible baseline rather than being zeroed.
WEIGHTS_BY_PHYSICIAN_INTENT: Dict[str, Dict[str, float]] = {
    THERAPY_SELECTION: {
        "cancer": 0.12, "histology": 0.05, "stage": 0.05, "biomarker": 0.18,
        "treatment_line": 0.13, "prior_treatment": 0.12, "active_treatment": 0.05,
        "drug_regimen": 0.05, "performance_status": 0.03, "organ_function": 0.02,
        "study_population": 0.03, "outcome": 0.05, "evidence_type": 0.05,
        "authority": 0.05, "freshness": 0.02,
    },
    TREATMENT_SEQUENCING: {
        "cancer": 0.10, "histology": 0.05, "stage": 0.05, "biomarker": 0.12,
        "treatment_line": 0.18, "prior_treatment": 0.18, "active_treatment": 0.05,
        "drug_regimen": 0.05, "performance_status": 0.04, "organ_function": 0.03,
        "study_population": 0.03, "outcome": 0.05, "evidence_type": 0.03,
        "authority": 0.03, "freshness": 0.01,
    },
    TOXICITY_MANAGEMENT: {
        "cancer": 0.04, "histology": 0.02, "stage": 0.02, "biomarker": 0.03,
        "treatment_line": 0.04, "prior_treatment": 0.04, "active_treatment": 0.15,
        "drug_regimen": 0.20, "performance_status": 0.08, "organ_function": 0.10,
        "study_population": 0.02, "outcome": 0.05, "evidence_type": 0.05,
        "authority": 0.13, "freshness": 0.03,
    },
    TRIAL_ELIGIBILITY: {
        "cancer": 0.08, "histology": 0.05, "stage": 0.14, "biomarker": 0.16,
        "treatment_line": 0.03, "prior_treatment": 0.10, "active_treatment": 0.03,
        "drug_regimen": 0.02, "performance_status": 0.12, "organ_function": 0.12,
        "study_population": 0.10, "outcome": 0.02, "evidence_type": 0.01,
        "authority": 0.01, "freshness": 0.01,
    },
}

# Fallback for a physician intent not in the table above (e.g. "general")
# -- an even-ish spread rather than raising or defaulting to one named
# intent's specialization.
_DEFAULT_WEIGHTS: Dict[str, float] = {name: round(1.0 / len(_COMPONENT_NAMES), 6) for name in _COMPONENT_NAMES}
# Rounding 1/15 fifteen times won't sum to exactly 1.0 -- correct the
# last component so weight normalization downstream can rely on it.
_DEFAULT_WEIGHTS[_COMPONENT_NAMES[-1]] += 1.0 - sum(_DEFAULT_WEIGHTS.values())


def _freshness_score(publication_date: Optional[str], as_of_year: int) -> float:
    """Simple recency decay: full credit within 3 years, floor at 15+
    years old. A candidate with no publication_date is neutral (0.5) --
    unknown age is not the same as old."""
    if not publication_date:
        return 0.5
    try:
        year = int(str(publication_date)[:4])
    except (ValueError, TypeError):
        return 0.5
    age = max(0, as_of_year - year)
    if age <= 3:
        return 1.0
    if age >= 15:
        return 0.2
    # Linear decay between 3 and 15 years old, from 1.0 down to 0.2.
    return round(1.0 - (age - 3) / 12 * 0.8, 4)


def score_candidate(
    candidate: Dict[str, Any],
    *,
    intent: str,
    patient_values: Optional[Dict[str, List[str]]] = None,
    as_of_year: int = 2026,
) -> Dict[str, Any]:
    """Scores one EvidenceCandidate-shaped dict (candidate.to_dict()'s
    output, or clinical_retrieval_adapter's adapted candidates) against
    a physician's patient_values -- the SAME dict-of-lists-per-axis
    shape retrieval_planner.RetrievalPlan.patient_values already uses on
    the patient side, keyed here by: cancer_types, histologies, stages,
    biomarkers, treatment_lines, prior_treatments, active_treatments,
    drugs_regimens, performance_statuses, organ_functions,
    study_populations, outcomes.

    Returns a NEW dict (candidate is not mutated) with the original keys
    preserved plus applicability_score/components/incompatibility_reasons,
    matching applicability_scorer.score_candidate()'s own contract."""
    patient_values = patient_values or {}
    text_lower = (candidate.get("text") or "").lower()
    meta = (candidate.get("metadata") or {}).get("applicability_meta") or {}

    cancer = _set_match(
        patient_values.get("cancer_types"), meta.get("cancer_types"), text_lower,
        general_token=_GENERAL_CANCER_TOKEN,
    )
    histology = _set_match(patient_values.get("histologies"), meta.get("histologies"), text_lower)
    stage = _set_match(patient_values.get("stages"), meta.get("stages"), text_lower)
    biomarker = _set_match(patient_values.get("biomarkers"), meta.get("biomarkers"), text_lower)
    treatment_line = _set_match(patient_values.get("treatment_lines"), meta.get("treatment_lines"), text_lower)
    prior_treatment = _set_match(patient_values.get("prior_treatments"), meta.get("prior_treatments"), text_lower)
    active_treatment = _set_match(patient_values.get("active_treatments"), meta.get("active_treatments"), text_lower)
    drug_regimen = _set_match(patient_values.get("drugs_regimens"), meta.get("drugs_regimens"), text_lower)
    performance_status = _set_match(
        patient_values.get("performance_statuses"), meta.get("performance_statuses"), text_lower,
    )
    organ_function = _set_match(patient_values.get("organ_functions"), meta.get("organ_functions"), text_lower)
    study_population = _set_match(
        patient_values.get("study_populations"), meta.get("study_populations"), text_lower,
    )
    outcome = _set_match(patient_values.get("outcomes"), meta.get("outcomes"), text_lower)

    hierarchy = evidence_hierarchy.select_hierarchy("physician", intent)
    evidence_type_score = evidence_hierarchy.authority_prior(
        evidence_hierarchy.infer_evidence_type(candidate), hierarchy,
    )

    authority_class = candidate.get("authority_class")
    authority = (
        {"A": 1.0, "B": 0.75, "C": 0.5}.get(authority_class, 0.6)
        if authority_class else 0.6  # unknown source, mild penalty rather than neutral -- physician-facing evidence with no traceable authority signal is a real gap, not a non-signal
    )

    freshness = _freshness_score(candidate.get("publication_date"), as_of_year)

    patient_biomarkers = {str(b).strip().lower() for b in (patient_values.get("biomarkers") or []) if b}
    candidate_biomarkers = {str(b).strip().lower() for b in (meta.get("biomarkers") or []) if b}
    biomarker_conflict = bool(
        patient_biomarkers and candidate_biomarkers
        and not (patient_biomarkers & candidate_biomarkers)
        and not any(b in text_lower for b in patient_biomarkers)
    )
    incompatibility_reasons: List[str] = []
    if biomarker_conflict:
        incompatibility_reasons.append(
            f"biomarker_mismatch: patient={sorted(patient_biomarkers)} "
            f"chunk={sorted(candidate_biomarkers)}"
        )

    components = {
        "cancer": round(cancer, 4), "histology": round(histology, 4), "stage": round(stage, 4),
        "biomarker": round(biomarker, 4), "treatment_line": round(treatment_line, 4),
        "prior_treatment": round(prior_treatment, 4), "active_treatment": round(active_treatment, 4),
        "drug_regimen": round(drug_regimen, 4), "performance_status": round(performance_status, 4),
        "organ_function": round(organ_function, 4), "study_population": round(study_population, 4),
        "outcome": round(outcome, 4), "evidence_type": round(evidence_type_score, 4),
        "authority": round(authority, 4), "freshness": round(freshness, 4),
        "biomarker_conflict": biomarker_conflict,
    }

    weights = WEIGHTS_BY_PHYSICIAN_INTENT.get(intent, _DEFAULT_WEIGHTS)
    combined = sum(weights[name] * components[name] for name in _COMPONENT_NAMES)
    if biomarker_conflict:
        combined *= (1 - _BIOMARKER_CONFLICT_PENALTY)

    out = dict(candidate)
    out.update({
        "applicability_score": round(combined, 4),
        "components": components,
        "incompatibility_reasons": incompatibility_reasons,
    })
    return out


def rank(
    candidates: List[Dict[str, Any]],
    *,
    intent: str,
    patient_values: Optional[Dict[str, List[str]]] = None,
    as_of_year: int = 2026,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    scored = [
        score_candidate(c, intent=intent, patient_values=patient_values, as_of_year=as_of_year)
        for c in candidates
    ]
    scored.sort(key=lambda c: c["applicability_score"], reverse=True)
    return scored[:limit]
