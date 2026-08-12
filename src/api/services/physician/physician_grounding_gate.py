"""
Physician Grounding Gate (2026-08-12 convergence Sprint C item 19)

Wires the two grounding layers already built this sprint — mechanical
citation checking (evidence/grounding_validator.py) and claim-level
entailment (evidence/claim_grounding_validator.py, Sprint A item 4) —
around physician_answer_generator.generate() (Sprint C item 18),
mirroring exactly the retry/repair/fallback sequence patient_chat_
service.answer() already runs for the patient path (that sequence's own
Sprint A item 8 / Sprint B item 11 predecessors). Factored out as its
own reusable function here, rather than inlined the way the patient
path's is, because the physician side has no single long-lived
answer()-style method to embed this logic into yet — Sprint C item 20's
orchestrator is what will call generate_grounded_physician_answer()
directly.

Sequence:
    generate draft
      -> mechanical grounding fails (and there was real evidence to
         cite)? retry once with a stricter prompt
      -> still fails? fall back to SAFE_FALLBACK_RESPONSE, sources_valid
         becomes False
      -> mechanical grounding passed (first or second try) AND this
         intent is in claim_grounding_validator.STRICT_VALIDATION_
         INTENTS (requires_strict_validation())?
           -> claim-level check
           -> a claim needs_repair? mechanically narrow/remove it
              (repair_answer())
           -> nothing left to salvage after repair? fall back to
              SAFE_FALLBACK_RESPONSE, sources_valid becomes False

Both grounding calls (draft generation, the mechanical retry, the claim
check) accept the same optional `client`/`model` injection
physician_answer_generator.generate() and claim_grounding_validator.
validate_claims() already support, for dependency-injected testing
without touching the real OpenAI client.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.api.services.evidence.grounding_validator import (
    validate as validate_grounding,
    RETRY_INSTRUCTION,
    SAFE_FALLBACK_RESPONSE,
)
from src.api.services.physician.physician_answer_generator import (
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundedAnswer:
    answer: str
    # False once the answer fell back to SAFE_FALLBACK_RESPONSE at
    # either gate -- callers should treat the packet's evidence as NOT
    # actually reflected in the returned answer in that case (matching
    # patient_chat_service.answer()'s `sources = []` on the same event).
    sources_valid: bool
    retried_mechanical: bool = False
    grounding_result: Optional[Dict[str, Any]] = None
    claim_result: Optional[Dict[str, Any]] = None


def _default_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


async def _call_model(
    messages: List[Dict[str, str]], *, client: Any = None, model: Optional[str] = None,
) -> str:
    def _call():
        c = client or _default_client()
        resp = c.chat.completions.create(
            model=model or settings.openai_mini_model or "gpt-4o-mini",
            temperature=0.2,
            max_tokens=900,
            messages=messages,
        )
        return resp.choices[0].message.content
    return await asyncio.to_thread(_call)


async def generate_grounded_physician_answer(
    question: str,
    packet: Dict[str, Any],
    intent: Optional[str] = None,
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> GroundedAnswer:
    # Imported locally (not at module load time) so that tests can
    # monkeypatch claim_grounding_validator.validate_claims and have this
    # function see the patched version -- matching patient_chat_service.
    # answer()'s own established pattern for the identical reason.
    from src.api.services.evidence.claim_grounding_validator import (
        validate_claims,
        repair_answer,
        requires_strict_validation,
    )

    sources = packet.get("evidence") or []

    messages = [
        {"role": "system", "content": build_system_prompt(packet)},
        {"role": "user", "content": build_user_message(question, packet)},
    ]
    draft = await _call_model(messages, client=client, model=model)

    grounding_result = validate_grounding(draft, packet)
    answer = draft
    retried_mechanical = False

    # Only a hard gate when there WAS evidence to ground in -- validate()
    # itself also flags a genuinely empty packet as invalid (a defense
    # against an upstream bug), but that's not this gate's job to act on;
    # see grounding_validator.py's own docstring.
    if not grounding_result.valid and sources:
        retried_mechanical = True
        retry_answer, retry_result = None, None
        try:
            retry_messages = messages + [
                {"role": "assistant", "content": draft},
                {"role": "user", "content": RETRY_INSTRUCTION},
            ]
            retry_answer = await _call_model(retry_messages, client=client, model=model)
            retry_result = validate_grounding(retry_answer, packet)
        except Exception:
            logger.warning(
                "[PhysicianGroundingGate] mechanical retry generation failed, "
                "falling back to safe response", exc_info=True,
            )

        if retry_result and retry_result.valid:
            answer, grounding_result = retry_answer, retry_result
        else:
            if retry_result:
                logger.warning(
                    "[PhysicianGroundingGate] mechanical grounding failed again "
                    "after retry: %s — falling back to safe response", retry_result.reasons,
                )
            return GroundedAnswer(
                answer=SAFE_FALLBACK_RESPONSE,
                sources_valid=False,
                retried_mechanical=True,
                grounding_result=(retry_result or grounding_result).to_dict(),
            )

    claim_result_dict: Optional[Dict[str, Any]] = None
    if sources and requires_strict_validation(intent):
        try:
            claim_result_obj = await validate_claims(answer, packet, client=client, model=model)
            claim_result_dict = claim_result_obj.to_dict()
            if claim_result_obj.needs_repair:
                repaired = repair_answer(answer, claim_result_obj)
                if repaired.strip() and repaired != answer:
                    logger.info(
                        "[PhysicianGroundingGate] claim validation repaired "
                        "unsupported claim(s) for intent=%s", intent,
                    )
                    answer = repaired
                else:
                    logger.warning(
                        "[PhysicianGroundingGate] claim validation found "
                        "unsupported claim(s) with nothing to mechanically "
                        "repair, falling back to safe response for intent=%s", intent,
                    )
                    return GroundedAnswer(
                        answer=SAFE_FALLBACK_RESPONSE,
                        sources_valid=False,
                        retried_mechanical=retried_mechanical,
                        grounding_result=grounding_result.to_dict(),
                        claim_result=claim_result_dict,
                    )
        except Exception:
            logger.warning(
                "[PhysicianGroundingGate] claim validation errored (continuing "
                "with the mechanically-grounded answer)", exc_info=True,
            )

    return GroundedAnswer(
        answer=answer,
        sources_valid=True,
        retried_mechanical=retried_mechanical,
        grounding_result=grounding_result.to_dict(),
        claim_result=claim_result_dict,
    )
