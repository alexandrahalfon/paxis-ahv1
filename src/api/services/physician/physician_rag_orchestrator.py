"""
Physician RAG Orchestrator (2026-08-12 convergence Sprint C item 20)

The single callable pipeline every Sprint C piece has been landing
pieces for: QueryAnalysis (item 12) -> physician context selector
(item 13) -> verified authorization (item 14) -> legacy retrieval
adapter (item 15) -> physician applicability scorer (item 16/17) ->
physician answer generator (item 18) -> grounding gate (item 19).
Matches the convergence plan's own pseudocode for this step:

    async def answer_physician_query(user, question, patient_profile_id=None):
        analysis = analyze_query(question)
        if patient_profile_id:
            authorize(...)
            state = get_patient_state(...)
            context = select_physician_context(state, analysis.intent)
        else:
            context = None
        legacy_results = await enhanced_rag.retrieve(...)
        candidates = adapt_legacy_results(legacy_results)
        extra_candidates = await retrieve_other_corpora(...)
        ranked = physician_applicability.rank(candidates + extra_candidates, context, analysis)
        packet = build_evidence_packet(...)
        draft = physician_generator.generate(packet)
        grounded = await claim_validator.validate(draft, packet)
        return finalize(grounded)

"Your existing retrieval stack is one of the strongest parts of Paxis" --
the legacy retriever (comprehensive_retrieval.ComprehensiveRetriever,
CLAUDE.md's own "do not change" list) runs completely unchanged here,
through the exact same clinical_retrieval_adapter.py (item 15) that
already reshapes its StudyEvidence output without touching a single line
of the retriever itself.

Authorization is a hard gate, not a filter: `patient_profile_id` given
with a physician_user_id that authorize_physician_patient_access()
(item 14) rejects returns ACCESS_DENIED_RESPONSE immediately -- this
function never falls through to "answer without patient context" on a
denied authorization, since that would silently leak the fact that a
patient_profile_id resolves to a real patient at all. A `patient_
profile_id=None` call (no specific patient in view) is not gated at
all, matching physician_context_service.py's own framing: general
questions do not need a linked patient.

Extra-corpora search (retrieve_other_corpora in the plan's pseudocode)
is real -- multi_corpus_retriever.search() against the same patient/
medication/guideline collections the patient path already searches,
audience="physician" so source_governance.py's physician-facing rules
apply (Sprint B item 8) -- but OFF by default
(include_extra_corpora=False). Two honest reasons, not an oversight:
(1) retrieval_planner.build_plan()'s _INTENT_COLLECTIONS table is keyed
by the PATIENT intent vocabulary (medication/symptom/nutrition/...);
none of physician_context_service.py's four intent names are in it, so
every physician call falls through to the same generic default
(patient education + the main literature collection) regardless of
which physician intent was actually detected -- a real result, just not
yet the differentiated-by-intent behavior the patient side gets. (2) it
is a live Qdrant + embedding network call with no injection seam yet
(unlike the legacy retriever below, which accepts `retriever=` for
exactly this reason) -- turned on by default it would make every
orchestrator unit test dependent on live network access. A caller that
wants the supplemental signal today can still pass
include_extra_corpora=True; failures there are caught and logged
(fail-open), never surfaced as an orchestrator-level error.

Patient state -> physician_applicability_scorer.py's patient_values
shape (_patient_values_for_scoring below) surfaces three more honest
gaps the same way clinical_retrieval_adapter.py and physician_context_
service.py already do for their own fields: performance_statuses,
study_populations, and outcomes have no equivalent PatientState field
yet and are left as empty lists, not fabricated. organ_functions reuses
comorbidities as the closest available proxy (no distinct organ-function
field exists either) -- clinical_inference.py already treats renal/
hepatic comorbidity terms as organ-function-relevant for the identical
reason. prior_treatments reuses active_treatment, matching physician_
context_service._FIELD_TO_STATE_KEYS's own documented approximation
(PatientState tracks only ACTIVE episodes today, not a distinct
completed/prior list).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Deliberately generic -- must not confirm or deny that patient_profile_id
# corresponds to a real patient to a physician who isn't authorized for
# it. Distinct wording from grounding_validator.SAFE_FALLBACK_RESPONSE so
# a caller (and any eval harness) can tell an authorization denial apart
# from a grounding failure.
ACCESS_DENIED_RESPONSE = (
    "You are not authorized to view this patient's record. If you believe "
    "this is an error, ask the patient to confirm your access on their "
    "care team, or contact your system administrator."
)


@dataclass
class PhysicianAnswer:
    answer: str
    sources_valid: bool
    # False only when patient_profile_id was given and authorization was
    # denied -- distinguishes "no patient in view" (authorized=True,
    # patient_profile_id is None) from "denied access" (authorized=False)
    # so a caller can render the right UI state rather than treating both
    # as an ordinary ungrounded answer.
    authorized: bool = True
    query_analysis: Optional[Dict[str, Any]] = None
    packet: Optional[Dict[str, Any]] = None
    retried_mechanical: bool = False
    grounding_result: Optional[Dict[str, Any]] = None
    claim_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Same shape/spirit as patient_chat_service.ChatResult.to_dict()
        -- the route layer (Sprint C item 21) returns this directly.
        `packet` is intentionally excluded from the default dict: it
        carries full evidence text/scores meant for the trace/debug
        surface, not the top-level API response body every caller gets
        back. A caller that wants it can still read `.packet` off the
        dataclass directly (e.g. a debug route)."""
        return {
            "answer": self.answer,
            "sources_valid": self.sources_valid,
            "authorized": self.authorized,
            "query_analysis": self.query_analysis,
            "sources": (self.packet or {}).get("evidence", []),
            "retried_mechanical": self.retried_mechanical,
        }


async def _get_patient_state(patient_profile_id: str) -> Dict[str, Any]:
    """Fetches PatientState directly by patient_profile_id, using the
    same deterministic freshness check evidence/patient_context_service.
    get_context() uses (Sprint B item 7: rebuild whenever the snapshot's
    source_revision doesn't match the profile's CURRENT state_revision) --
    but keyed straight from patient_profile_id rather than resolved from
    a patient_user_id first, since the physician side already has the
    profile id authorized (item 14) by the time this is called. Returns
    {} on any lookup failure -- same fail-open convention as get_context()
    itself: a state-lookup hiccup degrades to an ungrounded-by-state
    answer, not a hard error for an already-authorized physician."""
    try:
        from src.api.services.patient.patient_profile_service import (
            get_patient_profile_service,
        )
        from src.api.services.patient.patient_state_service import (
            get_patient_state_service,
        )

        profile = await get_patient_profile_service().get_by_id(patient_profile_id)
        if not profile:
            return {}

        state_service = get_patient_state_service()
        snapshot = await state_service.get_latest_snapshot(patient_profile_id)
        current_revision = profile.get("state_revision")
        snapshot_revision = snapshot.get("source_revision") if snapshot else None
        if snapshot is None or snapshot_revision != current_revision:
            built = await state_service.build_state(patient_profile_id)
            return built["state"]
        return snapshot.get("state", {}) or {}
    except Exception:
        logger.warning(
            "[PhysicianOrchestrator] patient state lookup failed for profile "
            "%s, continuing without state", patient_profile_id, exc_info=True,
        )
        return {}


def _patient_values_for_scoring(state: Dict[str, Any]) -> Dict[str, Any]:
    """Maps a PatientState dict onto physician_applicability_scorer.
    score_candidate()'s patient_values shape. See module docstring for
    which axes have no PatientState equivalent yet and are left honestly
    empty rather than fabricated."""
    diagnoses = state.get("active_diagnoses") or (
        [state["active_diagnosis"]] if state.get("active_diagnosis") else []
    )
    active_treatment = state.get("active_treatment") or []
    biomarkers = state.get("biomarkers") or []

    regimens_and_agents: List[str] = []
    for t in active_treatment:
        if t.get("regimen"):
            regimens_and_agents.append(t["regimen"])
        regimens_and_agents.extend(t.get("agents") or [])

    return {
        "cancer_types": [
            d.get("canonical_cancer_type") or d.get("cancer_site")
            for d in diagnoses if d.get("canonical_cancer_type") or d.get("cancer_site")
        ],
        "histologies": [d.get("histology") for d in diagnoses if d.get("histology")],
        "stages": [d.get("stage") for d in diagnoses if d.get("stage")],
        "biomarkers": [
            (b.get("biomarker_name") if isinstance(b, dict) else str(b))
            for b in biomarkers if b
        ],
        "treatment_lines": [
            t.get("line_of_therapy") for t in active_treatment if t.get("line_of_therapy")
        ],
        # See module docstring -- reuses active_treatment, the closest
        # available approximation (no distinct prior/completed list yet).
        "prior_treatments": [t.get("regimen") for t in active_treatment if t.get("regimen")],
        "active_treatments": [t.get("regimen") for t in active_treatment if t.get("regimen")],
        "drugs_regimens": regimens_and_agents,
        "performance_statuses": [],  # no PatientState field yet -- see module docstring
        "organ_functions": [c for c in (state.get("comorbidities") or []) if c],
        "study_populations": [],  # no PatientState field yet -- see module docstring
        "outcomes": [],  # no PatientState field yet -- see module docstring
        "age": (state.get("demographics") or {}).get("age"),
        "ecog": None,  # no PatientState field yet -- see module docstring
    }


async def _retrieve_extra_corpora(
    question: str, intent: str, state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Best-effort supplemental search of the patient/medication/
    guideline collections via multi_corpus_retriever.search(), audience=
    "physician" -- see module docstring for why this is opt-in and what
    its current limitations are. Always fails open: any exception (a
    network hiccup, an unconfigured Qdrant client in a given deployment)
    returns [] rather than failing the whole orchestrated answer."""
    try:
        from src.api.services.evidence.retrieval_planner import build_plan
        from src.api.services.evidence import multi_corpus_retriever

        # No physician-specific retrieval_features exist yet (that shape
        # is patient_state_service._derive_retrieval_features()'s own
        # patient-side output) -- an empty dict makes build_plan() fall
        # through to its unmatched-intent default rather than guessing at
        # a shape this function doesn't actually have.
        plan = build_plan(intent, {})
        return await multi_corpus_retriever.search(question, plan, audience="physician")
    except Exception:
        logger.info(
            "[PhysicianOrchestrator] extra-corpora search skipped (failed open)",
            exc_info=True,
        )
        return []


async def answer_physician_query(
    physician_user_id: str,
    question: str,
    patient_profile_id: Optional[str] = None,
    intent: Optional[str] = None,
    *,
    client: Any = None,
    model: Optional[str] = None,
    retriever: Any = None,
    max_studies: int = 8,
    chunks_per_study: int = 6,
    include_extra_corpora: bool = False,
) -> PhysicianAnswer:
    """Runs the full physician pipeline end to end. `retriever` accepts
    a comprehensive_retrieval.ComprehensiveRetriever-shaped object (must
    expose an async retrieve_comprehensive(...)) for dependency-injected
    testing, defaulting to the real singleton
    (comprehensive_retrieval.get_comprehensive_retriever()) exactly the
    way `client`/`model` already do for the LLM calls further down this
    pipeline."""
    from src.api.services.evidence.query_analysis import from_physician_query

    analysis = from_physician_query(question, intent=intent)
    effective_intent = analysis.intent

    state: Dict[str, Any] = {}
    selected_context: Optional[Dict[str, Any]] = None

    if patient_profile_id:
        # Imported locally so tests can monkeypatch the module attribute
        # and have it take effect here -- same reason physician_
        # grounding_gate.py imports claim_grounding_validator locally.
        from src.api.services.patient.patient_care_team_service import (
            authorize_physician_patient_access,
        )

        authorized = await authorize_physician_patient_access(
            physician_user_id, patient_profile_id,
        )
        if not authorized:
            return PhysicianAnswer(
                answer=ACCESS_DENIED_RESPONSE,
                sources_valid=False,
                authorized=False,
                query_analysis=analysis.to_dict(),
            )

        state = await _get_patient_state(patient_profile_id)
        from src.api.services.physician.physician_context_service import (
            select_physician_context,
        )
        selected_context = select_physician_context(state, effective_intent)

    # ── Legacy retrieval, unchanged (Sprint C item 15's whole point) ────
    from src.api.services.physician.clinical_retrieval_adapter import (
        adapt_legacy_results,
    )

    if retriever is None:
        from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
        retriever = get_comprehensive_retriever()

    legacy_result = await retriever.retrieve_comprehensive(
        query_text=question, max_studies=max_studies, chunks_per_study=chunks_per_study,
    )
    candidates = [c.to_dict() for c in adapt_legacy_results(legacy_result.studies)]

    if include_extra_corpora:
        candidates.extend(await _retrieve_extra_corpora(question, effective_intent, state))

    # ── Applicability scoring (Sprint C item 16/17) ──────────────────────
    from src.api.services.physician.physician_applicability_scorer import rank

    patient_values = _patient_values_for_scoring(state) if state else {}
    ranked = rank(
        candidates, intent=effective_intent, patient_values=patient_values, limit=max_studies,
    )

    # ── Evidence packet (Sprint A item 2) ────────────────────────────────
    from src.api.services.evidence.evidence_packet_builder import build_packet
    from src.api.services.patient.lab_interpretation import interpretation_policy_summary

    packet = build_packet(
        question,
        {"state": state} if state else None,
        ranked,
        audience="physician",
        query_analysis=analysis.to_dict(),
        patient_snapshot_id=patient_profile_id,
        selected_patient_context=selected_context,
        interpretation_policies=(
            interpretation_policy_summary(state.get("labs")) if state else {}
        ),
    )

    # ── Generation + grounding gate (Sprint C item 18/19) ────────────────
    from src.api.services.physician.physician_grounding_gate import (
        generate_grounded_physician_answer,
    )

    grounded = await generate_grounded_physician_answer(
        question, packet, effective_intent, client=client, model=model,
    )

    return PhysicianAnswer(
        answer=grounded.answer,
        sources_valid=grounded.sources_valid,
        authorized=True,
        query_analysis=analysis.to_dict(),
        packet=packet,
        retried_mechanical=grounded.retried_mechanical,
        grounding_result=grounded.grounding_result,
        claim_result=grounded.claim_result,
    )
