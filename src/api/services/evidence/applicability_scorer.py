"""
Applicability Scorer (Phase 4, rewritten 2026-08-12)

Scores a retrieved candidate against the patient's actual state instead of
ranking on semantic similarity alone — see the architecture review,
section 18: an NCI generic dysgeusia page and an ONS FOLFOX-specific page
can have close semantic scores but very different applicability to a
specific patient, and the second should usually win.

Eight components, each visible in the trace (retrieval_debug_trace.py
reads candidate["components"]) instead of one blended
"clinical_applicability" number:

  semantic    vector similarity score from the corpus search
  symptom     patient's active_symptoms vs. the chunk's tagged symptoms
  modality    patient's active treatment_modalities vs. the chunk's
  regimen     patient's regimen(s) vs. the chunk's tagged regimens
  drug        patient's active_agents vs. the chunk's tagged drugs
  cancer      patient's cancer_type(s) vs. the chunk's tagged cancer_types
  phase       patient's care phase vs. the chunk's tagged treatment_phases
  authority   evidence_sources.authority_class, or a fixed value for the
              already-curated literature corpus

Combined with intent-specific weights (WEIGHTS_BY_INTENT) rather than one
fixed blend — a medication question should weight drug/modality over
symptom; a symptom question the reverse.

Match semantics per component (see _set_match): unspecified on EITHER
side (the patient's state doesn't say, or the chunk wasn't tagged for
that axis) is neutral (0.5), never a mismatch — a patient with no
recorded symptoms isn't "wrong" for a symptom page. A candidate's
general/"all" token (cancer_types defaults to ["all"] for content that
names no specific cancer type — see metadata_classifier.py) is 0.75:
plausibly applicable, but not as specific as a named match (1.0). A named
non-overlap with no textual corroboration is a real mismatch (0.0).

Modality gets one more thing beyond its own component score: an explicit
multiplicative incompatibility penalty when the patient's active modality
and the chunk's tagged modality are both known, named, and don't overlap
— e.g. a chemotherapy patient's question served a targeted-therapy-only
chunk. A single 0.0 among eight weighted components can still leave a
wrong-modality chunk with a deceptively high blended score; this is the
separately-flagged penalty needed for that case (the reference Colab
notebook's scoring is what surfaced the need for it), not just another
data point folded into the average.

No cross-encoder here: Phase 4 already ran an LLM-cheap classification
(intent) and this scorer is meant to be fast enough to run over every
candidate from every corpus on every patient message. If ranking quality
turns out to need it, a cross-encoder gate can be added the same way
comprehensive_retrieval.py's Phase 3 gate was — this module's
score_candidate is the natural place to add it without touching callers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.api.services.evidence.retrieval_planner import RetrievalPlan
from src.api.services.evidence.patient_context_service import (
    INTENT_MEDICATION, INTENT_SYMPTOM, INTENT_NUTRITION,
    INTENT_TREATMENT, INTENT_DIAGNOSIS, INTENT_GENERAL,
)

_AUTHORITY_CLASS_SCORE = {"A": 1.0, "B": 0.75, "C": 0.5}
_LITERATURE_DEFAULT_AUTHORITY = 0.8  # already-curated peer-reviewed corpus
_GENERAL_CANCER_TOKEN = "all"

# Applied multiplicatively to the combined score on a detected modality
# conflict (see module docstring). 0.35 = combined score reduced by 35%
# — enough to reliably drop a wrong-modality chunk below general content
# on the same topic without ever forcing it to exactly zero (a chunk
# that's wrong on modality can still be the least-bad option when
# nothing else was retrieved at all).
_MODALITY_CONFLICT_PENALTY = 0.35

# Every weight set must sum to 1.0 — enforced in tests, not at runtime,
# so a typo here fails loudly in CI rather than silently under-weighting
# the whole scorer.
WEIGHTS_BY_INTENT: Dict[str, Dict[str, float]] = {
    INTENT_MEDICATION: {
        "semantic": 0.30, "symptom": 0.05, "modality": 0.15, "regimen": 0.10,
        "drug": 0.20, "cancer": 0.05, "phase": 0.05, "authority": 0.10,
    },
    INTENT_SYMPTOM: {
        "semantic": 0.25, "symptom": 0.25, "modality": 0.15, "regimen": 0.05,
        "drug": 0.05, "cancer": 0.05, "phase": 0.10, "authority": 0.10,
    },
    INTENT_NUTRITION: {
        "semantic": 0.30, "symptom": 0.20, "modality": 0.10, "regimen": 0.05,
        "drug": 0.05, "cancer": 0.05, "phase": 0.15, "authority": 0.10,
    },
    INTENT_TREATMENT: {
        "semantic": 0.25, "symptom": 0.05, "modality": 0.20, "regimen": 0.15,
        "drug": 0.05, "cancer": 0.15, "phase": 0.05, "authority": 0.10,
    },
    INTENT_DIAGNOSIS: {
        "semantic": 0.30, "symptom": 0.05, "modality": 0.05, "regimen": 0.05,
        "drug": 0.05, "cancer": 0.30, "phase": 0.05, "authority": 0.15,
    },
    INTENT_GENERAL: {
        "semantic": 0.35, "symptom": 0.10, "modality": 0.10, "regimen": 0.05,
        "drug": 0.05, "cancer": 0.10, "phase": 0.05, "authority": 0.20,
    },
}

_APPLICABILITY_AXES = ("symptom", "modality", "regimen", "drug", "cancer", "phase")


def _set_match(
    patient_terms: Optional[List[str]],
    candidate_tags: Optional[List[str]],
    text_lower: str,
    general_token: Optional[str] = None,
) -> float:
    """One component's match score. See module docstring for the
    unspecified/general/named semantics this implements."""
    patient_lower = {str(t).strip().lower() for t in (patient_terms or []) if t}
    if not patient_lower:
        return 0.5  # patient state doesn't specify this axis
    tag_lower = {str(t).strip().lower() for t in (candidate_tags or []) if t}
    if general_token and general_token in tag_lower:
        return 0.75
    if tag_lower and (patient_lower & tag_lower):
        return 1.0
    # No structured tag overlap — fall back to whether the patient's own
    # known term literally appears in the chunk text. This is safe (it
    # checks for a term the patient's record already asserts, never
    # invents specificity the way trusting an ungrounded model output
    # would) and preserves recall against chunks that predate — or
    # weren't reachable by — metadata classification.
    if any(t in text_lower for t in patient_lower):
        return 1.0
    if tag_lower:
        # Chunk is explicitly tagged with other, non-overlapping,
        # non-general values for this axis — a real mismatch signal.
        return 0.0
    # Chunk carries no tags for this axis at all: no evidence either way.
    return 0.5


def score_candidate(candidate: Dict[str, Any], plan: RetrievalPlan) -> Dict[str, Any]:
    semantic = float(candidate.get("semantic_score") or 0.0)
    text_lower = (candidate.get("text") or "").lower()
    meta = candidate.get("applicability_meta") or {}
    values = plan.patient_values or {}

    symptom = _set_match(values.get("symptoms"), meta.get("symptoms"), text_lower)
    regimen = _set_match(values.get("regimens"), meta.get("regimens"), text_lower)
    drug = _set_match(values.get("drugs"), meta.get("drugs"), text_lower)
    cancer = _set_match(
        values.get("cancer_types"), meta.get("cancer_types"), text_lower,
        general_token=_GENERAL_CANCER_TOKEN,
    )
    phase = _set_match(values.get("treatment_phase"), meta.get("treatment_phases"), text_lower)
    modality = _set_match(values.get("treatment_modalities"), meta.get("treatment_modalities"), text_lower)

    patient_modalities = {str(m).strip().lower() for m in (values.get("treatment_modalities") or []) if m}
    candidate_modalities = {str(m).strip().lower() for m in (meta.get("treatment_modalities") or []) if m}
    modality_conflict = bool(
        patient_modalities and candidate_modalities
        and not (patient_modalities & candidate_modalities)
        and not any(m in text_lower for m in patient_modalities)
    )
    # Human-readable form of the conflict, carried through to the
    # evidence packet/debug trace (see evidence_packet_builder.py) —
    # "score dropped" alone doesn't tell a reviewer why; this does.
    incompatibility_reasons: List[str] = []
    if modality_conflict:
        incompatibility_reasons.append(
            f"modality_mismatch: patient={sorted(patient_modalities)} "
            f"chunk={sorted(candidate_modalities)}"
        )

    authority_class = candidate.get("authority_class")
    authority = (
        _AUTHORITY_CLASS_SCORE.get(authority_class, 0.6)
        if authority_class else _LITERATURE_DEFAULT_AUTHORITY
    )

    components = {
        "semantic": round(semantic, 4),
        "symptom": round(symptom, 4),
        "modality": round(modality, 4),
        "regimen": round(regimen, 4),
        "drug": round(drug, 4),
        "cancer": round(cancer, 4),
        "phase": round(phase, 4),
        "authority": round(authority, 4),
        "modality_conflict": modality_conflict,
    }

    weights = WEIGHTS_BY_INTENT.get(plan.intent, WEIGHTS_BY_INTENT[INTENT_GENERAL])
    combined = sum(weights[k] * components[k] for k in weights)
    if modality_conflict:
        combined *= (1 - _MODALITY_CONFLICT_PENALTY)

    # Retained for callers/telemetry that pre-date the itemized
    # components (evidence_packet_builder.py, older trace consumers):
    # the mean of the six patient-specific axes, excluding semantic and
    # source authority which aren't "applicability to this patient" in
    # the same sense.
    clinical_applicability = sum(components[k] for k in _APPLICABILITY_AXES) / len(_APPLICABILITY_AXES)

    out = dict(candidate)
    out.update({
        "semantic_relevance": round(semantic, 4),
        "clinical_applicability": round(clinical_applicability, 4),
        "source_authority": round(authority, 4),
        "applicability_score": round(combined, 4),
        "components": components,
        "incompatibility_reasons": incompatibility_reasons,
    })
    return out


def rank(
    candidates: List[Dict[str, Any]], plan: RetrievalPlan, limit: int = 5
) -> List[Dict[str, Any]]:
    scored = [score_candidate(c, plan) for c in candidates]
    scored.sort(key=lambda c: c["applicability_score"], reverse=True)
    return scored[:limit]
