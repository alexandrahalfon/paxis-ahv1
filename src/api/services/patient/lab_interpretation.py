"""
Lab interpretation policy (2026-08-12 convergence Sprint A item 3).

Governs how far downstream generation is allowed to go when discussing a
lab value, separate from the raw fact itself. patient_state_service.py
used to derive named clinical risk labels straight from hard-coded
thresholds — `neutropenia_risk` from ANC < 1.5, `thrombocytopenia_risk`
from platelets < 100, `renal_function_context: "elevated_creatinine"`
from creatinine > 1.3 — and put them in retrieval_features where
generation could reach for them directly. That let generation state a
named clinical conclusion ("this patient is neutropenic") the system
never actually validated: whether a single ANC value below one cutoff
means "neutropenic" in a clinically meaningful sense depends on trend,
cause, the other cell lines, and context this system does not have.
Those three fields have been removed from
patient_state_service._derive_retrieval_features() entirely, rather than
left in place for something downstream to eventually lean on.

What replaces them: every lab in state["labs"] (see
patient_state_service.py) carries an allowed_interpretation level from
this module. Generation, and any future claim validator (Sprint A item
4), are expected to respect it — state the exact value and, if a prior
reading exists, whether it moved up or down, and nothing beyond that,
unless the level explicitly says more is warranted.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# A lab with only one reading on file: generation may state the value,
# unit, and collection date. No trend, no named clinical conclusion.
EXACT_VALUE_ONLY = "exact_value_only"

# A lab with a prior reading to compare against: generation may
# additionally say whether the value went up or down (and by how much),
# still never naming a clinical condition ("neutropenia", "renal
# impairment") the system has not validated.
EXACT_VALUE_AND_TREND_ONLY = "exact_value_and_trend_only"

# Reserved, not produced by anything in this codebase yet: a lab whose
# derived label has gone through an actual validated clinical rule (e.g.
# CTCAE toxicity grading against a cited version), rather than an
# ad hoc threshold picked when this code was written. Building that rule
# engine is future work — see the module docstring for why this codebase
# does not skip straight to it.
VALIDATED_RULE_INTERPRETATION = "validated_rule_interpretation"

# Reserved, not produced by anything in this codebase yet: a lab a
# clinician has explicitly reviewed and annotated with their own
# interpretation. Ties to a future clinician-review field on
# lab_results that does not exist yet either.
CLINICIAN_INTERPRETED = "clinician_interpreted"

ALL_LEVELS = (
    EXACT_VALUE_ONLY,
    EXACT_VALUE_AND_TREND_ONLY,
    VALIDATED_RULE_INTERPRETATION,
    CLINICIAN_INTERPRETED,
)


def interpretation_policy_summary(labs: Optional[Any]) -> Dict[str, str]:
    """canonical_test -> allowed_interpretation, for EvidencePacket's
    interpretation_policies field (Sprint A item 2) — the form
    generation/claim-validation is expected to consult when deciding how
    far it's allowed to go in describing a specific lab."""
    return {
        lab["canonical_test"]: lab["allowed_interpretation"]
        for lab in (labs or [])
        if isinstance(lab, dict) and lab.get("canonical_test") and lab.get("allowed_interpretation")
    }


def allowed_interpretation_for(
    latest: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]]
) -> str:
    """Every lab produced by this codebase today gets exactly one of the
    first two levels — there is no validated-rule engine or
    clinician-annotation field yet (see module docstring). A lab with a
    previous reading to compare against may state the direction of
    change; one with only a single reading may state only the value
    itself."""
    if previous and previous.get("value") is not None:
        return EXACT_VALUE_AND_TREND_ONLY
    return EXACT_VALUE_ONLY
