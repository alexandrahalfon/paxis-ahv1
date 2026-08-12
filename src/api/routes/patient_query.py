"""
Patient-facing Q&A endpoint (legacy, unauthenticated).

Predates the patient portal (patient_portal/patient_chat_service.py +
/api/patient-portal/chat) and has no login, no patient_profile_id, and no
linked clinical record — a caller is just "someone asking a question,"
with nothing this endpoint can key a longitudinal record on. That is a
structural difference from the portal chat, not a bug: it means this
endpoint can never do patient-state-aware personalization (retrieval
boosted by a specific patient's regimen/symptoms/labs), because there is
no patient identity here to look that state up for.

What CAN be shared, and — per the 2026-08-12 beta audit's "do not
maintain two independent patient RAG behaviors" finding — now IS shared,
is the evidence pipeline itself: this endpoint retrieves through the same
multi_corpus_retriever -> applicability_scorer -> evidence_packet_builder
-> grounding_validator path evidence/ services the authenticated portal
chat uses (evidence/patient_context_service.classify_intent() +
evidence/retrieval_planner.build_plan() with an empty patient_values dict
— which applicability_scorer.py's "unspecified is neutral" semantics
already handle correctly for a caller with no known patient facts), so a
question answered here and the same question answered through the portal
draw on identically-scored evidence, not two different retrieval stacks
that can silently disagree. Only the patient-specific personalization
layer is necessarily absent.

Kept in place rather than removed (an active, if legacy, integration
point — see the beta audit's "either remove or make it call the new
service" framing, resolved here as the latter) and left unauthenticated
on purpose: turning it into an authenticated endpoint would be a product
decision (merging it into the portal chat, or requiring login) beyond
what this fix is scoped to make unilaterally.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["Patient QA"])

# Shared OpenAI client. A fresh one was previously constructed on every
# request, paying connection setup each time.
_openai_client = None


def _get_shared_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        from src.core.config import settings
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


_PATIENT_SYSTEM_PROMPT = """You are a patient-friendly oncology assistant. A patient is asking \
about their prescribed treatment plan or a treatment they have been told about.

Your role:
- Explain in clear, simple, compassionate, non-technical language.
- Describe what the treatment involves, how it generally works, what to expect, and common side effects to be aware of.
- Do NOT recommend new treatments or suggest changes to their current plan.
- Do NOT contradict or second-guess advice from their care team.
- When relevant, mention what side effects or symptoms should prompt them to contact their care team.
- Always end by reminding them to discuss any concerns with their doctor or nurse.
- If you reference a study or source, explain its findings in plain terms — avoid jargon.
- If you genuinely do not have information on something, say so honestly rather than guessing.

Tone: warm, clear, reassuring. Avoid medical abbreviations unless you immediately explain them."""


class PatientQueryRequest(BaseModel):
    question: str = Field(..., description="Patient's question about their treatment", min_length=1)
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous turns as [{role: 'user'|'assistant', content: '...'}]"
    )


class PatientQueryResponse(BaseModel):
    answer: str = Field(..., description="Plain-language answer for the patient")
    disclaimer: str = Field(
        default=(
            "This information is for educational purposes only and does not replace "
            "medical advice from your care team. Always consult your doctor or nurse "
            "before making any decisions about your treatment."
        )
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting sources from the knowledge base"
    )


@router.post("/patient-query", response_model=PatientQueryResponse)
async def patient_query(request: PatientQueryRequest):
    """
    Answer a patient's plain-language question about their prescribed treatment.

    Retrieves relevant evidence from the knowledge base, then generates a
    patient-friendly explanation. Does not recommend new treatments or
    contradict the care team.
    """
    # Safety triage first. This endpoint predates the patient portal and
    # is unauthenticated, so it had no guardrails at all: a patient
    # describing chest pain got a chatty literature answer. Emergencies
    # are now answered from a fixed template without calling the model.
    try:
        from src.api.services.patient_portal import patient_safety_service as _safety
        _tri = _safety.triage(request.question)
        if _tri.blocks_answer:
            return PatientQueryResponse(
                answer=_safety.emergency_response(_tri),
                sources=[],
            )
    except Exception:
        # Never let the guardrail itself break the endpoint.
        _tri = None

    # Computed unconditionally (pure regex, no I/O) so it's available for
    # the hard grounding gate below even if retrieval itself fails.
    from src.api.services.evidence import patient_context_service as evidence_patient_context_service
    intent = evidence_patient_context_service.classify_intent(request.question)

    try:
        from src.core.config import settings

        openai_client = _get_shared_openai_client()

        # --- Retrieve relevant evidence: same pipeline the authenticated
        # portal chat uses (see module docstring), not the old clinician
        # literature backbone this endpoint used to call directly. ---
        evidence_block, sources_out = "", []
        try:
            from src.api.services.evidence.retrieval_planner import build_plan
            from src.api.services.evidence import multi_corpus_retriever
            from src.api.services.evidence.applicability_scorer import rank as rank_evidence
            from src.api.services.evidence.evidence_packet_builder import (
                build_packet, to_prompt_block, to_sources,
            )

            # No patient identity here (see module docstring) -- an empty
            # retrieval_features dict yields empty patient_values on the
            # plan, which applicability_scorer.py's "unspecified is
            # neutral" semantics score correctly: nothing is boosted or
            # penalized on axes this endpoint has no data for.
            plan = build_plan(intent, {})
            candidates = await multi_corpus_retriever.search(request.question, plan)
            ranked = rank_evidence(candidates, plan)
            packet = build_packet(request.question, None, ranked, safety_category="general")
            evidence_block = to_prompt_block(packet)
            sources_out = to_sources(packet)
        except Exception as e:
            logger.warning("[patient-query] multi-corpus retrieval failed, falling back: %s", e)

        # Fall back to the clinician literature backbone only when the
        # shared evidence pipeline found nothing -- same fallback shape
        # patient_chat_service.answer() uses, so an empty result degrades
        # identically on both patient-facing paths rather than differently.
        if not sources_out:
            from src.api.services.enhanced_rag_service import get_enhanced_rag_service

            rag_service = get_enhanced_rag_service()
            result = await rag_service.query(
                question=request.question,
                query_mode="hybrid",
                top_k=8,
                category=None,
                use_site_inference=False,
                conversation_history=request.conversation_history,
                conversation_context=[],
            )
            evidence = result.get("evidence", [])
            for i, e in enumerate(evidence[:6], 1):
                title = e.get("title", "Untitled")
                text = (e.get("text") or "")[:400]
                citation = e.get("citation", "")
                evidence_block += f"\n[{i}] {title}\n{text}\nSource: {citation}\n"
                sources_out.append({
                    "title": title,
                    "citation": citation,
                    "doi": e.get("doi"),
                    "pmid": e.get("pmid"),
                    "year": e.get("year"),
                    "source_type": e.get("source_type", "kb"),
                })

        # Hard grounding gate (2026-08-12 beta audit) -- same rule as
        # patient_chat_service.answer(): a factual medication/symptom/
        # nutrition/treatment/diagnosis question with zero usable evidence
        # after every fallback above must not be answered from the
        # model's own memory. `_tri` can be None if triage itself errored;
        # treat that the same as GENERAL (fail toward gating a factual
        # question rather than silently generating from memory).
        tri_category = _tri.category if _tri is not None else evidence_patient_context_service.GENERAL_SAFETY_CATEGORY
        if (
            not sources_out
            and intent in evidence_patient_context_service.FACTUAL_INTENTS
            and tri_category == evidence_patient_context_service.GENERAL_SAFETY_CATEGORY
        ):
            return PatientQueryResponse(
                answer=evidence_patient_context_service.NO_EVIDENCE_RESPONSE,
                sources=[],
            )

        # Build conversation messages
        messages = [{"role": "system", "content": _PATIENT_SYSTEM_PROMPT}]

        # Include prior turns
        for turn in request.conversation_history[-6:]:  # Last 3 exchanges
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        # Current question with evidence
        user_content = f"Patient question: {request.question}"
        if evidence_block.strip():
            user_content += (
                f"\n\nRelevant information from medical literature "
                f"(use this to inform your answer, but explain it in simple terms):\n"
                f"{evidence_block}"
                "\n\nWhen you make a claim drawn from the numbered passages above, cite "
                "it inline using its number in brackets, e.g. [1] or [2] -- matching the "
                "numbers shown, not a new numbering of your own."
            )
        messages.append({"role": "user", "content": user_content})

        # Threaded: a direct call blocks the event loop for the whole
        # generation, stalling every other in-flight request on this
        # single-worker process.
        import asyncio
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=settings.openai_model,
            temperature=0.4,
            max_tokens=800,
            messages=messages,
        )

        answer = response.choices[0].message.content.strip()

        # Grounding check -- same non-blocking rollout stage as
        # patient_chat_service.answer(): logged for now, not yet a hard
        # gate. See that module's docstring for the promotion plan.
        try:
            from src.api.services.evidence.grounding_validator import validate as validate_grounding
            grounding_result = validate_grounding(answer, {"evidence": sources_out})
            if not grounding_result.valid:
                logger.warning(
                    "[patient-query] grounding validation failed: %s", grounding_result.reasons
                )
        except Exception:
            logger.warning("[patient-query] grounding validation errored (continuing)", exc_info=True)

        return PatientQueryResponse(
            answer=answer,
            sources=sources_out,
        )

    except Exception:
        # Never surface internals here. This endpoint is patient-facing
        # and unauthenticated; it previously returned a full traceback,
        # which leaked file paths and connection details to the browser.
        logger.exception("[patient-query] failed")
        raise HTTPException(
            status_code=503,
            detail="I couldn't answer that just now. Please try again in a moment.",
        )
