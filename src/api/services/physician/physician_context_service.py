"""
Physician Context Selector (2026-08-12 convergence Sprint C item 13)

The patient context selector (evidence/patient_context_service.py) took
"an intent selects only the state fields that matter" as its founding
shape; this module gives the physician path the same shape, per the
convergence plan: "don't dump the complete longitudinal record into
every physician prompt."

select_physician_context() takes a PatientState dict — the exact same
shape patient_state_service.build_state()/get_context() already produce
(active_diagnosis/active_diagnoses/tumor_profile/biomarkers/
active_treatment/active_medications/active_symptoms/nutrition/
recent_labs/labs/comorbidities/allergies/care_team_instructions/...) —
and an intent, and returns only the subset PHYSICIAN_CONTEXT_POLICY says
that intent needs. This is a pure, synchronous, in-memory filter; it
does not fetch or build state itself, so it composes directly with
whatever already produced the state (patient_state_service.build_state(),
a cached snapshot, or a future physician-specific state builder).

PHYSICIAN_CONTEXT_POLICY's field names are written to match the
convergence plan's own spec verbatim (diagnoses, tumor_profiles,
performance_status, current_treatment, treatment_cycles, ...), NOT
PatientState's actual dict keys, which differ (active_diagnoses,
tumor_profile singular, active_treatment, ...) — _FIELD_TO_STATE_KEYS
below is the mapping between the two, kept as its own table so the
policy itself stays readable against the plan it implements.

Two honest gaps this surfaces rather than papers over: performance_status
(ECOG/KPS) and treatment_cycles are not distinct top-level fields on
PatientState today — nothing in patient_state_service.py captures them
as structured data yet. They're still listed in the policy below
(matching the plan's spec, and ready to activate the moment those
fields exist), but currently select nothing, since
_FIELD_TO_STATE_KEYS has no entry for them. stage/histology (used by the
trial_eligibility policy) aren't separate top-level fields either — they
live nested inside each entry of active_diagnosis/active_diagnoses, so
those two names map onto the SAME state keys "diagnoses" already does;
selecting "diagnoses" already carries stage/histology, there's nothing
extra to add.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Named per the convergence plan's own example, using its field
# vocabulary (not PatientState's dict keys — see module docstring).
THERAPY_SELECTION = "therapy_selection"
TREATMENT_SEQUENCING = "treatment_sequencing"
TOXICITY_MANAGEMENT = "toxicity_management"
TRIAL_ELIGIBILITY = "trial_eligibility"

PHYSICIAN_CONTEXT_POLICY: Dict[str, List[str]] = {
    THERAPY_SELECTION: [
        "diagnoses", "tumor_profiles", "biomarkers", "treatment_history",
        "performance_status", "labs", "comorbidities", "allergies",
        "care_team_instructions",
    ],
    TREATMENT_SEQUENCING: [
        "diagnoses", "biomarkers", "treatment_history", "performance_status",
    ],
    TOXICITY_MANAGEMENT: [
        "current_treatment", "treatment_cycles", "symptoms", "medications",
        "labs", "comorbidities",
    ],
    TRIAL_ELIGIBILITY: [
        "diagnoses", "stage", "histology", "biomarkers", "prior_treatments",
        "performance_status", "labs",
    ],
}

# Used for any intent not explicitly in PHYSICIAN_CONTEXT_POLICY above
# (including "general") -- a small, generically useful baseline rather
# than silently falling back to "select everything", which would
# reintroduce exactly the full-record-dump problem this module exists
# to avoid.
_DEFAULT_FIELDS: List[str] = ["diagnoses", "treatment_history", "biomarkers", "performance_status"]

# Policy field name -> PatientState dict key(s) that currently hold that
# information. A field name present here with an empty list, or absent
# entirely, selects nothing -- see module docstring for
# performance_status/treatment_cycles specifically.
_FIELD_TO_STATE_KEYS: Dict[str, List[str]] = {
    "diagnoses": ["active_diagnosis", "active_diagnoses"],
    "stage": ["active_diagnosis", "active_diagnoses"],
    "histology": ["active_diagnosis", "active_diagnoses"],
    "tumor_profiles": ["tumor_profile"],
    "biomarkers": ["biomarkers"],
    "treatment_history": ["active_treatment"],
    "current_treatment": ["active_treatment"],
    # PatientState only tracks ACTIVE treatment episodes -- a distinct
    # "completed/prior treatment" list isn't a separate top-level field
    # today (treatment_service.list_episodes() returns every status, but
    # build_state() only surfaces the active ones into `state`). Mapped
    # to the same field as the closest available approximation rather
    # than left silently empty.
    "prior_treatments": ["active_treatment"],
    "medications": ["active_medications"],
    "symptoms": ["active_symptoms"],
    "labs": ["recent_labs", "labs"],
    "comorbidities": ["comorbidities"],
    "allergies": ["allergies", "intolerances"],
    "care_team_instructions": ["care_team_instructions"],
}


def select_physician_context(state: Dict[str, Any], intent: str) -> Dict[str, Any]:
    """Returns the subset of `state` PHYSICIAN_CONTEXT_POLICY says
    `intent` needs. A state key is included only when it's both selected
    by the policy AND actually present with a non-empty value in
    `state` -- this never pads the output with empty placeholders for
    fields the patient simply doesn't have data for."""
    field_names = PHYSICIAN_CONTEXT_POLICY.get(intent, _DEFAULT_FIELDS)
    selected: Dict[str, Any] = {}
    for field_name in field_names:
        for state_key in _FIELD_TO_STATE_KEYS.get(field_name, []):
            if state_key in selected:
                continue
            value = state.get(state_key)
            if value not in (None, [], {}, ""):
                selected[state_key] = value
    return selected
