"""
Claim Grounding Validator (2026-08-12 convergence Sprint A item 4)

The most important shared missing component per the convergence plan:
grounding_validator.py (mechanical) checks that an answer cites SOME
evidence and that every citation number resolves to a real packet
entry — cheap, fast, and already a hard retry/fail gate on the patient
path. What it explicitly does not check (see its own docstring) is
whether a cited passage actually SUPPORTS the specific claim sitting
next to it: "Adagrasib improved progression-free survival [4]" mechanically
passes if evidence #4 exists at all, even if #4 only reports that PFS was
measured and says nothing about improvement.

This module is that check. Audience-agnostic on purpose — the same
draft -> extract claims -> match each claim to its cited passage(s) ->
judge supported/partially_supported/unsupported -> narrow/remove pipeline
applies whether the draft came from the patient or (future) physician
generator, so it lives in evidence/ rather than under either audience's
package.

Not every question needs this. The convergence plan splits validation
level by intent: strict claim validation for therapy_selection,
treatment_sequencing, dose_modification, drug_interaction,
toxicity_management, trial_eligibility, prognosis, and lab_interpretation
— the questions where an unsupported specific claim is a real safety
issue. Mechanical grounding (grounding_validator.py) is sufficient for
definitions, terminology explanations, and basic educational questions —
running an extra LLM pass on every "what is chemotherapy" would just add
latency for no safety benefit. requires_strict_validation() is the
shared policy function so callers don't each duplicate this list.

Building the shared service is this module's whole job. WIRING it into
the patient path (retry/repair/regenerate on an unsupported claim) is
Sprint B item 11; wiring it into the physician path is Sprint C item 19.
Neither exists yet — nothing calls validate_claims() today, matching how
evidence_candidate.py and the EvidencePacket expansion landed before
anything consumed them either.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings

logger = logging.getLogger(__name__)

# Per the convergence plan: an unsupported claim in one of these areas is
# a real clinical-safety issue, not a style nit — these get the full
# claim-level entailment pass. Everything else (definitions, basic
# education, general conversation) is left to the mechanical citation
# check in grounding_validator.py, which is fast enough to run on every
# turn and catches the coarse failure mode (no citation at all).
STRICT_VALIDATION_INTENTS = frozenset({
    "therapy_selection",
    "treatment_sequencing",
    "dose_modification",
    "drug_interaction",
    "toxicity_management",
    "trial_eligibility",
    "prognosis",
    "lab_interpretation",
})

SUPPORTED = "supported"
PARTIALLY_SUPPORTED = "partially_supported"
UNSUPPORTED = "unsupported"
NOT_A_CLAIM = "not_a_claim"  # connective/conversational text, not a factual assertion

_VALID_SUPPORT_LEVELS = {SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, NOT_A_CLAIM}

_SYSTEM_PROMPT = """You are a clinical fact-checker. You will be given a \
draft answer and a numbered list of evidence passages it was written from. \
Break the draft into its individual factual claims (ignore purely \
connective or conversational sentences -- mark those "not_a_claim") and, \
for each claim, judge whether the passages it cites actually support it.

Reply with JSON only, in this exact shape:
{"claims": [
  {"claim": "<the claim, quoted verbatim from the draft>",
   "citations": [<int, ...>],
   "support_level": "supported" | "partially_supported" | "unsupported" | "not_a_claim",
   "reason": "<one sentence: what the cited passage(s) actually say, and how that compares to the claim>",
   "rewrite": "<a narrower version of the claim that IS fully supported, or null if support_level is not partially_supported>"}
]}

support_level definitions:
- supported: the cited passage(s) directly state what the claim asserts.
- partially_supported: the cited passage(s) are relevant but the claim \
overstates, generalizes, or adds a comparison/conclusion the passage(s) \
don't make (e.g. claiming "improved survival" when the passage only \
reports that survival was measured, or claiming superiority over another \
option the passage never compares against).
- unsupported: the citation doesn't exist, or the passage(s) say nothing \
that supports the claim.
- not_a_claim: the sentence carries no independently checkable factual \
assertion (greetings, offers to help, "let's discuss this together", etc).

Quote each "claim" as a verbatim substring of the draft so it can be \
located mechanically. Do not invent claims that aren't in the draft."""


@dataclass
class ClaimAssessment:
    claim: str
    citations: List[int] = field(default_factory=list)
    support_level: str = UNSUPPORTED
    reason: str = ""
    rewrite: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "citations": self.citations,
            "support_level": self.support_level,
            "reason": self.reason,
            "rewrite": self.rewrite,
        }


@dataclass
class ClaimValidationResult:
    claims: List[ClaimAssessment] = field(default_factory=list)
    # None = the validator itself could not run (LLM/network failure,
    # malformed response) -- distinct from False (it ran and found a
    # real problem). Callers decide their own fail-open/fail-closed
    # policy for None; this module doesn't decide it for them.
    ran: bool = True
    error: Optional[str] = None

    @property
    def overall_valid(self) -> Optional[bool]:
        if not self.ran:
            return None
        return not any(c.support_level == UNSUPPORTED for c in self.claims)

    @property
    def needs_repair(self) -> List[ClaimAssessment]:
        return [c for c in self.claims if c.support_level in (UNSUPPORTED, PARTIALLY_SUPPORTED)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "ran": self.ran,
            "error": self.error,
            "overall_valid": self.overall_valid,
        }


def requires_strict_validation(intent: Optional[str]) -> bool:
    """Shared policy: does this intent warrant the full claim-level
    entailment pass, or is mechanical citation checking
    (grounding_validator.py) enough? See module docstring."""
    return bool(intent) and intent in STRICT_VALIDATION_INTENTS


def _format_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    block = ""
    for i, e in enumerate(evidence, 1):
        title = e.get("title") or "Source"
        text = (e.get("text") or "")[:1200]
        block += f"\n[{i}] {title}\n{text}\n"
    return block


def _default_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


def _parse_response(raw: str) -> List[ClaimAssessment]:
    data = json.loads(raw or "{}")
    claims_in = data.get("claims") or []
    out: List[ClaimAssessment] = []
    for c in claims_in:
        if not isinstance(c, dict) or not c.get("claim"):
            continue
        level = c.get("support_level")
        if level not in _VALID_SUPPORT_LEVELS:
            level = UNSUPPORTED
        citations = [int(n) for n in (c.get("citations") or []) if isinstance(n, (int, float))]
        out.append(ClaimAssessment(
            claim=str(c["claim"]),
            citations=citations,
            support_level=level,
            reason=str(c.get("reason") or ""),
            rewrite=c.get("rewrite") if level == PARTIALLY_SUPPORTED else None,
        ))
    return out


async def validate_claims(
    answer: str,
    packet: Dict[str, Any],
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> ClaimValidationResult:
    """Claim-level entailment check: does each factual claim in `answer`
    actually follow from the evidence it cites?

    `packet` is an EvidencePacket dict (see evidence_packet_builder.
    build_packet()) — only packet["evidence"] is read. Does not consult
    packet["query_analysis"]/intent itself; that's what
    requires_strict_validation() is for, applied by the caller before
    deciding whether to call this function at all (an LLM pass on every
    turn is not free).

    Never raises: an LLM/network failure returns a ClaimValidationResult
    with ran=False and error set, rather than propagating and costing
    the caller their answer. Callers decide their own policy for that
    case (this module doesn't have an opinion on fail-open vs.
    fail-closed)."""
    evidence = packet.get("evidence") or []
    if not evidence or not (answer or "").strip():
        # Nothing to check a claim against, or nothing to check --
        # mirrors grounding_validator.validate()'s "empty packet" signal
        # rather than silently declaring success.
        return ClaimValidationResult(claims=[], ran=True)

    import asyncio

    def _call():
        c = client or _default_client()
        resp = c.chat.completions.create(
            model=model or settings.openai_mini_model or "gpt-4o-mini",
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Evidence passages:\n{_format_evidence_block(evidence)}\n\n"
                    f"Draft answer:\n{answer}"
                )},
            ],
        )
        return resp.choices[0].message.content

    try:
        raw = await asyncio.to_thread(_call)
        claims = _parse_response(raw)
        return ClaimValidationResult(claims=claims, ran=True)
    except Exception as e:
        logger.warning("[ClaimGroundingValidator] validate_claims failed: %s", e, exc_info=True)
        return ClaimValidationResult(claims=[], ran=False, error=str(e))


def repair_answer(answer: str, result: ClaimValidationResult) -> str:
    """Best-effort MECHANICAL repair: drops each verbatim 'unsupported'
    claim from the answer and substitutes each 'partially_supported'
    claim's suggested narrower rewrite in place. A claim the model
    paraphrased rather than quoted verbatim won't be found and is
    silently left as-is -- this is a safety net for the common case, not
    a guarantee. The complete fix described in the convergence plan's
    pipeline is an LLM-driven regenerate-and-revalidate loop, which is
    the caller's responsibility (Sprint B item 11 / Sprint C item 19),
    not this function's."""
    if not result.ran:
        return answer
    repaired = answer
    for c in result.claims:
        if c.support_level == UNSUPPORTED and c.claim in repaired:
            repaired = repaired.replace(c.claim, "")
        elif c.support_level == PARTIALLY_SUPPORTED and c.rewrite and c.claim in repaired:
            repaired = repaired.replace(c.claim, c.rewrite)
    return " ".join(repaired.split())
