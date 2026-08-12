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

Incompatibility (2026-08-12, Sprint C item 17 — extends item 16's single
biomarker-only penalty into the full typed taxonomy): incompatibility_
details is a list of {type, severity, patient, evidence, reason} dicts,
distinguishing hard incompatibility (a real exclusionary mismatch —
penalizes the combined score, same 0.35 multiplicative penalty item 16
introduced, now triggered by ANY hard finding rather than biomarker
specifically), soft mismatch (informational, does not penalize the
score — e.g. histology or prior-therapy differences that often still
inform practice even when they don't match exactly), and unknown
(trial_eligibility scoring specifically flags when a numeric eligibility
bound — age range, ECOG ceiling — simply isn't reported by the evidence
at all, which is a real "we can't confirm eligibility" gap worth
surfacing, not silent neutrality). incompatibility_reasons (a flat
string list, item 16's original shape) is now DERIVED from
incompatibility_details rather than computed separately, so both stay
in sync and both reflect the full detected set, not just biomarker.

Detection mechanisms:
  - Five axes (cancer_type_mismatch/hard, histology_mismatch/soft,
    biomarker_mismatch/hard, prior_therapy_requirement_missing/soft,
    organ_function_incompatible/hard) reuse the exact 0.0 signal
    _set_match() already produces for a named, non-overlapping,
    textually-uncorroborated mismatch on that component — no new
    matching logic, just labeling what item 16 already detects.
  - first_line_only_vs_previously_treated is a term-based heuristic over
    patient_values["treatment_lines"]/meta["treatment_lines"] (e.g. the
    patient's line is tagged "second_line"/"previously_treated" while
    the evidence is tagged "first_line_only"). No real corpus tagging
    exists yet to validate this against (documented, not pretended
    otherwise) — same honesty convention as clinical_retrieval_adapter.py's
    version_id/rrf_score gaps.
  - trial_age_incompatible and ECOG_incompatible are dedicated numeric-
    range checks: patient_values["age"]/["ecog"] (single values, not
    lists — the one exception to every other axis's list-of-terms shape)
    against meta["age_range"] ({"min","max"}) / meta["ecog_max"].

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

# Applied multiplicatively to the combined score whenever ANY hard
# incompatibility is detected (Sprint C item 17 generalizes this from
# item 16's biomarker-only trigger) — see module docstring.
_HARD_INCOMPATIBILITY_PENALTY = 0.35

SEVERITY_HARD = "hard"
SEVERITY_SOFT = "soft"
SEVERITY_UNKNOWN = "unknown"

# axis (a _COMPONENT_NAMES entry with a real _set_match() score) ->
# (incompatibility type name, severity). The patient_values/meta dict
# key for each axis is the axis name itself with an "s" plurally
# adjusted where needed -- see _AXIS_TO_VALUES_KEY below.
_AXIS_INCOMPATIBILITY: Dict[str, tuple] = {
    "cancer": ("cancer_type_mismatch", SEVERITY_HARD),
    "histology": ("histology_mismatch", SEVERITY_SOFT),
    "biomarker": ("biomarker_mismatch", SEVERITY_HARD),
    "prior_treatment": ("prior_therapy_requirement_missing", SEVERITY_SOFT),
    "organ_function": ("organ_function_incompatible", SEVERITY_HARD),
}

_AXIS_TO_VALUES_KEY: Dict[str, str] = {
    "cancer": "cancer_types", "histology": "histologies", "biomarker": "biomarkers",
    "prior_treatment": "prior_treatments", "organ_function": "organ_functions",
}

# Term-based heuristic for first_line_only_vs_previously_treated -- see
# module docstring for why this is a heuristic, not real corpus tagging.
_SUBSEQUENT_LINE_TERMS = frozenset({
    "second_line", "third_line", "subsequent_line", "previously_treated",
    "post_progression", "2l", "3l", "relapsed", "refractory",
})
_FIRST_LINE_ONLY_TERMS = frozenset({
    "first_line_only", "treatment_naive_only", "1l_only", "frontline_only",
})

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


def _detect_incompatibilities(
    patient_values: Dict[str, Any],
    meta: Dict[str, Any],
    text_lower: str,
    components: Dict[str, Any],
    intent: str,
) -> List[Dict[str, Any]]:
    """Returns the full typed incompatibility list (Sprint C item 17) —
    see module docstring for each detection mechanism."""
    details: List[Dict[str, Any]] = []

    # ── Axis mismatches: reuse the 0.0 signal _set_match() already
    # computed for each component above, just label what it means. ────
    for axis, (type_name, severity) in _AXIS_INCOMPATIBILITY.items():
        if components.get(axis) != 0.0:
            continue
        values_key = _AXIS_TO_VALUES_KEY[axis]
        patient_terms = sorted({str(t) for t in (patient_values.get(values_key) or []) if t})
        evidence_terms = sorted({str(t) for t in (meta.get(values_key) or []) if t})
        details.append({
            "type": type_name,
            "severity": severity,
            "patient": ", ".join(patient_terms) or None,
            "evidence": ", ".join(evidence_terms) or None,
            "reason": f"{type_name}: patient={patient_terms} evidence={evidence_terms}",
        })

    # ── Line-of-therapy direction heuristic ─────────────────────────────
    patient_lines = {str(t).strip().lower() for t in (patient_values.get("treatment_lines") or []) if t}
    evidence_lines = {str(t).strip().lower() for t in (meta.get("treatment_lines") or []) if t}
    if (patient_lines & _SUBSEQUENT_LINE_TERMS) and (evidence_lines & _FIRST_LINE_ONLY_TERMS):
        details.append({
            "type": "first_line_only_vs_previously_treated",
            "severity": SEVERITY_HARD,
            "patient": ", ".join(sorted(patient_lines & _SUBSEQUENT_LINE_TERMS)),
            "evidence": ", ".join(sorted(evidence_lines & _FIRST_LINE_ONLY_TERMS)),
            "reason": (
                "first_line_only_vs_previously_treated: patient is previously "
                "treated but evidence population is first-line only"
            ),
        })

    # ── Numeric eligibility bounds: age, ECOG. Only meaningful for
    # trial_eligibility scoring -- an "unknown eligibility" note on every
    # other intent would just be noise. ─────────────────────────────────
    if intent == TRIAL_ELIGIBILITY:
        patient_age = patient_values.get("age")
        age_range = meta.get("age_range")
        if patient_age is not None and isinstance(age_range, dict) and (
            "min" in age_range or "max" in age_range
        ):
            lo, hi = age_range.get("min"), age_range.get("max")
            if (lo is not None and patient_age < lo) or (hi is not None and patient_age > hi):
                details.append({
                    "type": "trial_age_incompatible",
                    "severity": SEVERITY_HARD,
                    "patient": str(patient_age),
                    "evidence": f"{lo}-{hi}",
                    "reason": f"trial_age_incompatible: patient age {patient_age} outside {lo}-{hi}",
                })
        elif patient_age is not None and age_range is None:
            details.append({
                "type": "trial_age_incompatible", "severity": SEVERITY_UNKNOWN,
                "patient": str(patient_age), "evidence": None,
                "reason": "trial_age_incompatible: evidence does not report an age eligibility range",
            })

        patient_ecog = patient_values.get("ecog")
        ecog_max = meta.get("ecog_max")
        if patient_ecog is not None and ecog_max is not None:
            if patient_ecog > ecog_max:
                details.append({
                    "type": "ECOG_incompatible", "severity": SEVERITY_HARD,
                    "patient": str(patient_ecog), "evidence": f"<= {ecog_max}",
                    "reason": f"ECOG_incompatible: patient ECOG {patient_ecog} exceeds {ecog_max}",
                })
        elif patient_ecog is not None and ecog_max is None:
            details.append({
                "type": "ECOG_incompatible", "severity": SEVERITY_UNKNOWN,
                "patient": str(patient_ecog), "evidence": None,
                "reason": "ECOG_incompatible: evidence does not report an ECOG eligibility ceiling",
            })

    return details


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

    components = {
        "cancer": round(cancer, 4), "histology": round(histology, 4), "stage": round(stage, 4),
        "biomarker": round(biomarker, 4), "treatment_line": round(treatment_line, 4),
        "prior_treatment": round(prior_treatment, 4), "active_treatment": round(active_treatment, 4),
        "drug_regimen": round(drug_regimen, 4), "performance_status": round(performance_status, 4),
        "organ_function": round(organ_function, 4), "study_population": round(study_population, 4),
        "outcome": round(outcome, 4), "evidence_type": round(evidence_type_score, 4),
        "authority": round(authority, 4), "freshness": round(freshness, 4),
    }

    incompatibility_details = _detect_incompatibilities(
        patient_values, meta, text_lower, components, intent,
    )
    hard_incompatibility = any(d["severity"] == SEVERITY_HARD for d in incompatibility_details)
    # Kept for any lightweight consumer that just wants a summary line
    # per finding — now derived from incompatibility_details rather than
    # computed separately, so the two never drift out of sync.
    incompatibility_reasons: List[str] = [d["reason"] for d in incompatibility_details]
    components["hard_incompatibility"] = hard_incompatibility

    weights = WEIGHTS_BY_PHYSICIAN_INTENT.get(intent, _DEFAULT_WEIGHTS)
    combined = sum(weights[name] * components[name] for name in _COMPONENT_NAMES)
    if hard_incompatibility:
        combined *= (1 - _HARD_INCOMPATIBILITY_PENALTY)

    out = dict(candidate)
    out.update({
        "applicability_score": round(combined, 4),
        "components": components,
        "incompatibility_reasons": incompatibility_reasons,
        "incompatibility_details": incompatibility_details,
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
