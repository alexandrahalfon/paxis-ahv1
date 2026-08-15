"""
Grounding Validator (Phase 4 hardening)

Post-generation check that an answer actually cites the evidence it was
given, rather than trusting the generation prompt's instructions alone.
This is the exact regression the reference Colab notebook's v1 -> v2 fix
was built around: a "v1 grounding failure" where the model answered
without citing anything, or used phrasing like "no direct evidence
sources provided" or "general oncology guidance" to talk around missing
grounding instead of citing a real passage.

What this checks (see validate()):
  1. If the evidence packet is non-empty, the answer must cite at least
     one [n] reference.
  2. Every [n] cited must resolve to an actual packet entry (1..N).
  3. None of FORBIDDEN_PHRASES appear — specific observed ungrounded-
     language patterns, not a general style critique.
  4. A packet with zero evidence should never have reached generation at
     all (evidence_packet_builder / the hard gate in
     patient_chat_service.answer() are supposed to prevent that) — this
     is checked anyway so a bypassed gate fails loudly instead of
     silently.

What this does NOT check: that a cited passage actually supports the
specific sentence sitting next to it. That's claim-level entailment —
matching each generated sentence to the passage(s) it draws from, most
likely via a second LLM/NLI pass — and is real, separate work. Flagged
here as the documented next step rather than half-built now: verifying
"citation [2] exists" is cheap and mechanical; verifying "this sentence
is actually supported by [2]" needs its own design and its own eval set.

Retry/fallback (2026-08-12 beta audit item 8): this module used to be
consulted after generation and only logged on failure — the code
literally said "recorded and logged, not yet a hard gate", so a patient
could receive an answer this validator itself judged ungrounded. Callers
are now expected to use RETRY_INSTRUCTION and SAFE_FALLBACK_RESPONSE
(see patient_chat_service.answer(), step 6b) to enforce:

    generate -> validate() fails -> regenerate once with RETRY_INSTRUCTION
    -> validate() fails again -> return SAFE_FALLBACK_RESPONSE, not the
       ungrounded answer

Both are defined here rather than at each call site so every caller that
promotes validate() to a hard gate uses the same retry framing and the
same fallback wording — a caller reaching for "not grounded" text should
find it here first rather than writing a slightly different version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# Specific phrasings the model has used, in practice, to talk around
# missing grounding instead of citing a real passage. Each entry here is
# a failure mode that already happened once — not a general style
# preference, so keep this list evidence-driven (add a phrase when you
# actually see it produced) rather than speculative.
FORBIDDEN_PHRASES: List[str] = [
    "no direct evidence sources provided",
    "no specific evidence was provided",
    "information comes from typical",
    "general oncology guidance",
    "based on general knowledge",
    "without specific sources",
]

# Sent back to the model as an extra user turn, alongside its own
# (failed) prior answer, when validate() rejects a first attempt that
# had non-empty evidence to work with. Mirrors FORBIDDEN_PHRASES/the
# citation check above so the instruction the model receives matches
# exactly what will be re-checked.
RETRY_INSTRUCTION = (
    "Your previous answer did not properly cite the evidence you were given. "
    "Rewrite it: every specific factual claim drawn from the numbered passages "
    "above must include its bracket citation (e.g. [1], [2]), matching the "
    "numbers already shown -- do not renumber them. Do not use phrases like "
    "'general oncology guidance', 'based on general knowledge', or similar "
    "language to talk around a missing citation; either cite the passage that "
    "supports the claim or leave the claim out. Keep the same warm, plain-"
    "language tone and the same level of helpfulness."
)

# Returned instead of the answer when a second, retried attempt still
# fails validation. Deliberately does not repeat any of the ungrounded
# content from either attempt -- the point of the gate is that neither
# attempt is trustworthy enough to show the patient.
SAFE_FALLBACK_RESPONSE = (
    "I found some information that may be relevant, but I wasn't able to "
    "confirm my answer was properly grounded in it, so I don't want to risk "
    "giving you something inaccurate. Your care team can help confirm the "
    "details here."
)


@dataclass
class GroundingValidationResult:
    valid: bool
    evidence_count: int
    citations_used: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    forbidden_phrases_found: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)  # empty iff valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "evidence_count": self.evidence_count,
            "citations_used": self.citations_used,
            "invalid_citations": self.invalid_citations,
            "forbidden_phrases_found": self.forbidden_phrases_found,
            "reasons": self.reasons,
        }


def validate(answer: str, packet: Dict[str, Any]) -> GroundingValidationResult:
    evidence_count = len(packet.get("evidence") or [])
    cited = sorted({int(x) for x in _CITATION_PATTERN.findall(answer or "")})
    invalid = [n for n in cited if n < 1 or n > evidence_count]
    lowered = (answer or "").lower()
    forbidden_found = [p for p in FORBIDDEN_PHRASES if p in lowered]

    reasons: List[str] = []

    if evidence_count == 0:
        reasons.append("evidence packet is empty — generation should not have run")
    elif not cited:
        reasons.append("answer contains no [n] evidence citations despite a non-empty evidence packet")

    if invalid:
        reasons.append(f"answer cites out-of-range evidence numbers: {invalid}")

    if forbidden_found:
        reasons.append(f"answer contains ungrounded-language pattern(s): {forbidden_found}")

    return GroundingValidationResult(
        valid=not reasons,
        evidence_count=evidence_count,
        citations_used=cited,
        invalid_citations=invalid,
        forbidden_phrases_found=forbidden_found,
        reasons=reasons,
    )
