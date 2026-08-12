"""
Patient-facing Q&A endpoint.

Allows patients to ask plain-language questions about their prescribed treatment plan.
Uses the knowledge base for relevant evidence but generates answers in accessible,
non-clinical language. Does NOT recommend new treatments or changes to care.
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

    try:
        from src.core.config import settings
        from src.api.services.enhanced_rag_service import get_enhanced_rag_service

        openai_client = _get_shared_openai_client()

        # --- Retrieve relevant KB evidence ---
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

        # Build a condensed evidence block for the prompt (abstracts, not full text)
        evidence_block = ""
        sources_out = []
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
