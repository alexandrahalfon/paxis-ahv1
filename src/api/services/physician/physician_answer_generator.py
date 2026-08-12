"""
Physician Answer Generator (2026-08-12 convergence Sprint C item 18)

"Shared packet does not mean shared prompt" — per the convergence plan,
the patient and physician paths converge on one EvidencePacket
(Sprint A item 2) but generate from it with different prompts and
different response shapes. The patient generator stays exactly where it
is (embedded in patient_chat_service.answer(), already shipped and
tested — extracting it into its own module is a refactor "beta does not
require" per the plan's own framing, and this task's actual job is the
physician side, which doesn't exist yet). This module is that physician
side: register/clinical-decision-support register instead of plain
language, emphasizing clinical synthesis, patient applicability
(how the evidence applies to THIS patient, not just what it says in
general), guideline position, efficacy AND toxicity together (not just
efficacy), population match, limitations, and missing patient variables
that limit confidence — per the plan's own list for this generator.

Same evidence-citation contract as the patient path: build_user_message()
reuses evidence_packet_builder.to_prompt_block()'s [n] numbering exactly,
so grounding_validator.py's mechanical citation check (Sprint A item 4's
predecessor) and claim_grounding_validator.py (Sprint A item 4) both work
unchanged against a physician-generated answer — Sprint C item 19 wires
the latter in, not this module.

Precedence (architecture review section 14, "implement care-team
instruction precedence globally"): PRECEDENCE_ORDER is the explicit,
testable ranking this system uses when two sources of guidance could
conflict — deterministic safety policy and regulatory contraindications
outrank a validated care-team instruction, which still outranks a
current guideline or patient education (matching the patient side's own
rule for the layer beneath it — see patient_chat_service.
_care_team_instructions_block). build_precedence_directive() renders
this ranking, plus whatever policy content the packet actually carries
(safety_policy, care_team_instructions surfaced via selected_
patient_context, interpretation_policies), into an explicit instruction
block every physician generation includes.

detect_policy_conflicts() is intentionally a stub returning [] today —
"record conflicts in the packet" as stated in the review means detecting
that, say, a care-team instruction and a guideline recommendation
actually contradict each other, which needs real semantic judgment (an
LLM pass, or clinician-authored conflict rules) this module doesn't
attempt to fake with string matching. The precedence ORDERING above is
this commit's real mechanism — it tells the model how to resolve a
conflict it encounters, rather than trying to pre-detect one
deterministically. Automatic conflict detection is documented future
work, not silently skipped.

Nothing calls generate() yet — Sprint C item 20's physician orchestrator
is what will, after physician_applicability_scorer.py ranks candidates
and evidence_packet_builder.build_packet() assembles the packet.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.core.config import settings

SYSTEM_PROMPT = """You are a clinical decision-support assistant for oncologists.

Your role:
- Synthesize the evidence provided into a clinically useful answer, not a \
list of abstracts.
- State how the evidence applies to THIS specific patient (population \
match) -- age, biomarker, stage, treatment history, organ function -- not \
just what the evidence found in general.
- When the evidence includes a guideline-tier source, note the study's \
position relative to current guideline recommendations.
- Report efficacy AND toxicity/safety findings together when both are \
present in the cited evidence -- never efficacy alone when toxicity data \
was also provided.
- Explicitly state limitations: population mismatches between the patient \
and the evidence's enrolled population, and any patient variable (ECOG, \
organ function, exact biomarker status, prior therapy detail) that is \
missing or unconfirmed and limits how confidently the evidence applies.
- Cite every evidence-based claim inline using its bracket number (e.g. \
[1], [2]), matching the numbers shown in the evidence list -- do not \
renumber them. Never state a claim beyond what the cited passage \
actually supports.
- This is decision SUPPORT, not a directive: the treating clinician makes \
the final decision. Do not phrase recommendations as instructions to be \
followed without clinical judgment.
"""

# See module docstring. Highest precedence first.
PRECEDENCE_ORDER = (
    "deterministic_safety_policy",
    "regulatory_contraindication",
    "validated_care_team_instruction",
    "current_guideline",
    "patient_education",
)


def build_precedence_directive(packet: Dict[str, Any]) -> str:
    """Renders PRECEDENCE_ORDER plus whatever policy content the packet
    actually carries into an explicit instruction block. Always returns
    a non-empty string with the ordering rule -- the model should know
    the rule even for a packet carrying none of the optional content."""
    lines = ["PRECEDENCE (when sources of guidance could conflict, follow this order, highest first):"]
    lines.extend(f"{i + 1}. {name.replace('_', ' ')}" for i, name in enumerate(PRECEDENCE_ORDER))

    selected_context = packet.get("selected_patient_context") or packet.get("patient_context") or {}
    instructions = selected_context.get("care_team_instructions") or []
    rendered_instructions = [
        (i.get("text") if isinstance(i, dict) else str(i))
        for i in instructions
    ]
    rendered_instructions = [t for t in rendered_instructions if t]
    if rendered_instructions:
        lines.append("")
        lines.append("VALIDATED CARE-TEAM INSTRUCTIONS for this patient (rank 3 above):")
        lines.extend(f"- {t}" for t in rendered_instructions)

    safety_policy = packet.get("safety_policy") or {}
    if safety_policy:
        lines.append("")
        lines.append(f"DETERMINISTIC SAFETY POLICY (rank 1 above): {safety_policy}")

    interpretation_policies = packet.get("interpretation_policies") or {}
    if interpretation_policies:
        lines.append("")
        lines.append(
            "LAB INTERPRETATION LIMITS -- for each lab below, state only what "
            "the allowed level permits (exact_value_only = value/unit/date "
            "only; exact_value_and_trend_only = also whether it moved up or "
            "down); never name a clinical condition beyond what the level "
            "allows:"
        )
        lines.extend(f"- {test}: {level}" for test, level in interpretation_policies.items())

    return "\n".join(lines)


def detect_policy_conflicts(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stub -- see module docstring for why automatic semantic conflict
    detection isn't attempted here yet. Returns [] unconditionally."""
    return []


def build_system_prompt(packet: Dict[str, Any]) -> str:
    return SYSTEM_PROMPT + "\n\n" + build_precedence_directive(packet)


def build_user_message(question: str, packet: Dict[str, Any]) -> str:
    """Evidence block using evidence_packet_builder.to_prompt_block()'s
    exact [n] numbering, so the SAME grounding checks that already work
    for patient answers work unchanged here."""
    from src.api.services.evidence.evidence_packet_builder import to_prompt_block

    block = to_prompt_block(packet)
    message = f"Clinical question: {question}"
    if block.strip():
        message += (
            "\n\nEvidence, numbered for citation:\n" + block +
            "\n\nCite the number in brackets ([1], [2], ...) for every claim you "
            "draw from these passages."
        )
    return message


def _default_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


async def generate(
    question: str,
    packet: Dict[str, Any],
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> str:
    """Generates a physician-facing draft answer from an EvidencePacket
    (audience='physician'). Does not validate grounding itself -- see
    module docstring for why that's Sprint C item 19's job, wired into
    the orchestrator (item 20), not this function's."""
    def _call():
        c = client or _default_client()
        resp = c.chat.completions.create(
            model=model or settings.openai_mini_model or "gpt-4o-mini",
            temperature=0.2,
            max_tokens=900,
            messages=[
                {"role": "system", "content": build_system_prompt(packet)},
                {"role": "user", "content": build_user_message(question, packet)},
            ],
        )
        return resp.choices[0].message.content

    return await asyncio.to_thread(_call)
