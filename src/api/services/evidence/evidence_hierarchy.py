"""
Evidence Hierarchy (2026-08-12 convergence Sprint A item 5)

Different questions warrant different authority orderings. A physician
asking a standard-of-care question should see a current guideline ahead
of a single-arm study; a patient asking a self-care question should see
approved patient education ahead of a comparative trial written for
oncologists; a dose-modification question should weight the drug label
and professional guideline far above patient education, not just
slightly. One universal ranking policy can't express all three — this
module is that per-context policy, applied as a PRIOR on top of
applicability_scorer.py's existing relevance score, never a replacement
for it: "authority is a prior, not a substitute for relevance." A highly
relevant single-arm study should still usually beat an irrelevant
guideline; this just tilts the scale between comparably-relevant
candidates toward the source type this context actually wants.

Deliberately additive and standalone — nothing in applicability_scorer.py
or evidence_packet_builder.py calls this yet. apply_authority_prior() is
provided as the integration point for whichever caller wants to use it
(patient path today via authority_class already does something similar
inline in applicability_scorer.py; this is the more expressive
successor, ready for physician convergence to adopt directly rather than
re-derive).

Evidence-type tagging: today's candidates aren't tagged with a "current
guideline" vs. "single-arm study" label anywhere in ingestion — that
level of metadata classification doesn't exist yet. infer_evidence_type()
is a best-effort mapping from what candidates ARE already tagged with
(collection name from settings.qdrant_*_collection, authority_class) onto
the canonical evidence-type vocabulary below, so this hierarchy is usable
immediately rather than blocked on new ingestion metadata. A real
per-document evidence-type classifier is future work this function is
designed to be swapped out for without changing the hierarchy policies
or apply_authority_prior() themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Canonical evidence-type vocabulary ──────────────────────────────────
GUIDELINE = "guideline"
REGULATORY_LABEL = "regulatory_label"
COMPARATIVE_TRIAL = "comparative_trial"
SINGLE_ARM_STUDY = "single_arm_study"
REVIEW = "review"
PATIENT_GUIDELINE = "patient_guideline"
PATIENT_EDUCATION = "patient_education"
MEDICATION_AUTHORITY = "medication_authority"
CLINICAL_STUDY = "clinical_study"
GENERIC_EDUCATION = "generic_education"

# ── Named hierarchy policies ─────────────────────────────────────────────
# Each policy is an ordered list of TIERS (list of lists), highest
# priority first. Types within the same tier are treated as roughly
# equal — see dose_modification, where the convergence plan's own
# notation ("current label + current professional guideline >> patient
# education") is a big gap between two tiers, not three strictly ordered
# types.
PHYSICIAN_STANDARD_OF_CARE = "physician_standard_of_care"
PATIENT_SELF_CARE = "patient_self_care"
DOSE_MODIFICATION = "dose_modification"

HIERARCHY_POLICIES: Dict[str, List[List[str]]] = {
    PHYSICIAN_STANDARD_OF_CARE: [
        [GUIDELINE],
        [REGULATORY_LABEL],
        [COMPARATIVE_TRIAL],
        [SINGLE_ARM_STUDY],
        [REVIEW],
        [GENERIC_EDUCATION],
    ],
    PATIENT_SELF_CARE: [
        [PATIENT_GUIDELINE],
        [PATIENT_EDUCATION],
        [MEDICATION_AUTHORITY],
        [GUIDELINE],
        [CLINICAL_STUDY],
    ],
    DOSE_MODIFICATION: [
        [REGULATORY_LABEL, GUIDELINE],
        [PATIENT_EDUCATION],
    ],
}

# Intents that route to the dose_modification hierarchy regardless of
# audience — a patient asking "can I take a lower dose" deserves the
# same label/guideline-first weighting a physician's dose-modification
# question does, not the general patient_self_care ordering.
_DOSE_MODIFICATION_INTENTS = frozenset({"dose_modification", "medication_explainer"})

# Position-based prior score by tier index; anything beyond the last
# defined tier gets the final (lowest) value rather than continuing to
# shrink indefinitely.
_TIER_SCORES = [1.0, 0.85, 0.7, 0.55, 0.45, 0.35]

# An evidence_type this policy has no opinion on (not found in any tier)
# is neutral, not penalized — being untagged is not the same as being
# wrong, matching applicability_scorer._set_match's same convention for
# an unspecified axis.
_UNTAGGED_PRIOR = 0.5

_COLLECTION_TO_EVIDENCE_TYPE = {
    # Matches settings.qdrant_guideline_collection /
    # qdrant_patient_education_collection / qdrant_medication_collection /
    # qdrant_collection (src/core/config.py) — kept as literal strings
    # rather than importing settings so this module has no import-time
    # dependency on config, matching multi_corpus_retriever/evidence_
    # candidate's existing no-cross-import convention for peer evidence
    # modules.
    "oncology_clinical_guidelines": GUIDELINE,
    "oncology_patient_education": PATIENT_EDUCATION,
    "oncology_medication_knowledge": MEDICATION_AUTHORITY,
    "exueed_kb_latest": CLINICAL_STUDY,
}


def select_hierarchy(audience: str, intent: Optional[str] = None) -> List[List[str]]:
    """Which named policy applies to this audience/intent. Physician
    dose-modification questions and (today) patient medication questions
    both route to the tighter dose_modification policy; everything else
    falls back to the audience's general policy."""
    if intent in _DOSE_MODIFICATION_INTENTS:
        return HIERARCHY_POLICIES[DOSE_MODIFICATION]
    if audience == "physician":
        return HIERARCHY_POLICIES[PHYSICIAN_STANDARD_OF_CARE]
    return HIERARCHY_POLICIES[PATIENT_SELF_CARE]


def infer_evidence_type(candidate: Dict[str, Any]) -> str:
    """Best-effort mapping from what a candidate is already tagged with
    today (corpus/collection name, authority_class) to the canonical
    evidence-type vocabulary — see module docstring for why this exists
    instead of a real per-document classifier."""
    collection = candidate.get("corpus") or candidate.get("collection")
    if collection in _COLLECTION_TO_EVIDENCE_TYPE:
        return _COLLECTION_TO_EVIDENCE_TYPE[collection]
    if candidate.get("authority_class") == "A":
        return GUIDELINE
    return GENERIC_EDUCATION


def authority_prior(evidence_type: str, tiers: List[List[str]]) -> float:
    """Position-based prior for one evidence_type within a hierarchy —
    the raw number apply_authority_prior() blends with relevance."""
    for i, tier in enumerate(tiers):
        if evidence_type in tier:
            return _TIER_SCORES[min(i, len(_TIER_SCORES) - 1)]
    return _UNTAGGED_PRIOR


def apply_authority_prior(
    candidates: List[Dict[str, Any]],
    *,
    audience: str,
    intent: Optional[str] = None,
    score_key: str = "applicability_score",
    prior_weight: float = 0.15,
) -> List[Dict[str, Any]]:
    """Blends each candidate's existing relevance score with this
    context's authority prior: `(1 - prior_weight) * relevance +
    prior_weight * authority_prior`. Deliberately a small weight by
    default — this is a TILT between comparably-relevant candidates, not
    a re-ranking that can make an irrelevant guideline beat a highly
    relevant study. Returns NEW dicts (candidates are not mutated),
    re-sorted descending by the blended score, with the evidence_type and
    authority_prior recorded on each entry for auditability (the trace/
    packet layer can surface why one candidate outranked another).

    Does not touch applicability_scorer.py — this is meant to run AFTER
    it, as an optional additional step a caller opts into, not a
    replacement for its relevance scoring."""
    tiers = select_hierarchy(audience, intent)
    out = []
    for c in candidates:
        evidence_type = infer_evidence_type(c)
        prior = authority_prior(evidence_type, tiers)
        relevance = float(c.get(score_key) or 0.0)
        blended = (1 - prior_weight) * relevance + prior_weight * prior
        updated = dict(c)
        updated["evidence_type"] = evidence_type
        updated["authority_prior"] = prior
        updated[f"{score_key}_with_authority_prior"] = round(blended, 4)
        out.append(updated)
    out.sort(key=lambda c: c[f"{score_key}_with_authority_prior"], reverse=True)
    return out
