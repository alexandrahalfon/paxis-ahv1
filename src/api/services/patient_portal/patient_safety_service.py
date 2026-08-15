"""
Patient message safety triage.

Runs on every patient message *before* an answer is generated. A physician
using Paxis can sanity-check a bad answer; a frightened patient at 11pm
cannot. This module is the guardrail for that difference.

Design: rules first, model second.

The unambiguous emergencies are matched with regex, not an LLM, because
regex is faster, deterministic, free, and cannot be talked out of firing
by unusual phrasing. An LLM is only consulted for the ambiguous middle,
and even then its answer can only *escalate* the category, never lower it.
Anything unrecognised falls through to a normal answer.

Categories
----------
``emergency``
    Possible medical emergency. Do not answer the question as asked.
    Show urgent-care instructions immediately.
``clinical_decision``
    A question only their care team can answer (dose changes, stopping
    treatment, prognosis, "is this right for me"). Explain generally,
    then offer to route it to the physician.
``distress``
    Emotional distress without an emergency signal. Answer warmly and
    surface support options.
``general``
    Everything else. Answer normally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Category constants ─────────────────────────────────────────────────────

EMERGENCY = "emergency"
CLINICAL_DECISION = "clinical_decision"
DISTRESS = "distress"
GENERAL = "general"


# ── Emergency patterns ─────────────────────────────────────────────────────
# Deliberately broad. A false positive costs the patient one extra
# reassurance step; a false negative can cost far more. Each entry is
# (pattern, short reason shown in logs / to the care team).

_EMERGENCY_PATTERNS: List[tuple] = [
    # Cardiac / respiratory
    (r"\bchest (?:pain|pressure|tightness)\b", "chest pain"),
    (r"\b(?:can'?t|cannot|trouble|difficulty|hard to|struggling to)\s+breath\w*", "breathing difficulty"),
    (r"\bshort(?:ness)? of breath\b", "shortness of breath"),
    (r"\bgasping\b|\bsuffocat\w+", "breathing difficulty"),

    # Neutropenic fever: a genuine oncology emergency that patients
    # routinely do not know is urgent. Any fever mentioned alongside
    # chemotherapy is treated as an emergency.
    (r"\bfever\b(?:(?!\?).){0,80}\b(?:chemo\w*|neutropen\w+|treatment)\b", "possible neutropenic fever"),
    (r"\b(?:chemo\w*|neutropen\w+)\b(?:(?!\?).){0,80}\bfever\b", "possible neutropenic fever"),
    (r"\b(?:temperature|temp|fever)\s*(?:of\s*)?(?:1[0-9]{2}(?:\.\d)?)\s*(?:f\b|degrees)?", "high fever"),
    (r"\b(?:38|39|40|41)(?:\.\d)?\s*(?:c\b|celsius|degrees c)", "high fever"),

    # Bleeding
    (r"\b(?:bleeding|blood)\b(?:(?!\?).){0,40}\b(?:won'?t stop|not stopping|heavily|uncontrolled)\b", "uncontrolled bleeding"),
    (r"\b(?:coughing|throwing|vomiting|spitting)\s+up\s+blood\b", "bleeding"),
    (r"\bblood in (?:my )?(?:stool|urine|vomit)\b", "bleeding"),

    # Neurological / stroke
    (r"\b(?:slurred speech|face (?:is )?droop\w*|can'?t move (?:my )?(?:arm|leg|side))\b", "possible stroke"),
    (r"\b(?:sudden|severe)\s+(?:confusion|weakness|numbness)\b", "possible stroke"),
    (r"\bworst headache\b", "severe headache"),
    (r"\b(?:seizure|convuls\w+)\b", "seizure"),
    (r"\b(?:passed out|fainted|unconscious|unresponsive)\b", "loss of consciousness"),

    # Self-harm. Handled with the highest priority.
    (_SELF_HARM_SRC := (
        r"\b(?:kill(?:ing|ed)?\s+myself|end(?:ing)?\s+(?:my\s+)?life|suicid\w+|"
        r"tak(?:e|ing)\s+my\s+own\s+life|don'?t\s+want\s+to\s+(?:live|be\s+here|wake\s+up)|"
        r"want\s+to\s+die|wish\s+i\s+(?:was|were)\s+dead|better\s+off\s+dead|"
        r"harm(?:ing)?\s+myself|hurt(?:ing)?\s+myself|self[-\s]?harm)\b"
    ), "self-harm risk"),
]

# Self-harm gets a distinct response track from physical emergencies.
# Compiled from the same source string used in _EMERGENCY_PATTERNS so the
# two can never drift apart. Covers inflections ("killing myself") because
# an exact-phrase-only pattern silently missed those.
_SELF_HARM_PATTERN = re.compile(_SELF_HARM_SRC, re.IGNORECASE)


# ── Clinical-decision patterns ─────────────────────────────────────────────
# Questions Paxis must not answer as if it were the care team.

_CLINICAL_PATTERNS: List[tuple] = [
    (r"\bshould i\b(?:(?!\?).){0,60}\b(?:stop|quit|skip|pause|halt|continue|take|switch|change)\b", "treatment change"),
    (r"\bcan i (?:stop|skip|pause|halt|reduce|double|change)\b", "treatment change"),
    (r"\b(?:stop|skip|miss(?:ing)?|double)\s+(?:a\s+)?(?:dose|treatment|cycle|infusion)\b", "dosing decision"),
    (r"\bis (?:this|that|it) (?:the )?(?:right|best|correct)\b(?:(?!\?).){0,40}\b(?:for me|treatment|option|choice)\b", "treatment suitability"),
    (r"\bshould i (?:get|have|do|try)\b", "treatment suitability"),
    (r"\bwhat (?:should|would) (?:i|you) do\b", "advice seeking"),
    (r"\b(?:change|adjust|increase|decrease|lower|raise)\s+(?:my\s+)?(?:dose|dosage|medication)\b", "dosing decision"),
    (r"\bdo i (?:need|have to)\b(?:(?!\?).){0,40}\b(?:surgery|chemo\w*|radiation|treatment)\b", "treatment suitability"),
]

# Prognosis: common, deeply human, and never appropriate for an AI to
# answer from population statistics. Always routed, always gently.
_PROGNOSIS_PATTERN = re.compile(
    r"\b(?:how long (?:do|have) i(?:'ve)? got|how long (?:do i|will i) (?:have|live)|"
    r"life expectancy|how long to live|am i (?:going to |gonna )?(?:die|survive)|"
    r"is (?:this|it) terminal|what are my (?:odds|chances)|survival rate for me|"
    r"is (?:this|it) curable for me)\b",
    re.IGNORECASE,
)


# ── Distress patterns ──────────────────────────────────────────────────────

_DISTRESS_PATTERN = re.compile(
    # Intensifier is open-ended ("so", "really", "very", "absolutely",
    # "kind of") because a fixed list silently missed most real phrasings.
    r"\b(?:i'?m\s+(?:\w+\s+){0,2}(?:scared|terrified|frightened|afraid|anxious|"
    r"panicking|panicked|overwhelmed|struggling|devastated)|"
    r"i\s+feel\s+(?:\w+\s+){0,2}(?:scared|hopeless|alone|lost|helpless|numb)|"
    r"i can'?t cope|can'?t handle this|feel hopeless|feel(?:ing)? alone|"
    r"losing hope|giving up|so depressed|keep crying|can'?t stop crying|"
    r"freaking out|falling apart)\b",
    re.IGNORECASE,
)


@dataclass
class TriageResult:
    category: str = GENERAL
    reasons: List[str] = field(default_factory=list)
    self_harm: bool = False
    prognosis: bool = False
    matched_terms: List[str] = field(default_factory=list)

    @property
    def blocks_answer(self) -> bool:
        """True when the normal answer must not be generated."""
        return self.category == EMERGENCY

    @property
    def should_offer_escalation(self) -> bool:
        return self.category in (CLINICAL_DECISION, DISTRESS) or self.prognosis

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "reasons": self.reasons,
            "self_harm": self.self_harm,
            "prognosis": self.prognosis,
            "blocks_answer": self.blocks_answer,
            "should_offer_escalation": self.should_offer_escalation,
        }


def triage(message: str) -> TriageResult:
    """Classify one patient message. Pure, fast, no network calls."""
    result = TriageResult()
    if not message or not message.strip():
        return result

    text = message.strip()

    # 1. Self-harm first, so it always wins the category.
    if _SELF_HARM_PATTERN.search(text):
        result.category = EMERGENCY
        result.self_harm = True
        result.reasons.append("self-harm risk")
        return result

    # 2. Physical emergencies.
    for pattern, reason in _EMERGENCY_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result.category = EMERGENCY
            if reason not in result.reasons:
                result.reasons.append(reason)
            result.matched_terms.append(m.group(0)[:60])
    if result.category == EMERGENCY:
        return result

    # 3. Prognosis, then other clinical decisions.
    if _PROGNOSIS_PATTERN.search(text):
        result.category = CLINICAL_DECISION
        result.prognosis = True
        result.reasons.append("prognosis question")
        return result

    for pattern, reason in _CLINICAL_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result.category = CLINICAL_DECISION
            if reason not in result.reasons:
                result.reasons.append(reason)
            result.matched_terms.append(m.group(0)[:60])
    if result.category == CLINICAL_DECISION:
        return result

    # 4. Emotional distress.
    if _DISTRESS_PATTERN.search(text):
        result.category = DISTRESS
        result.reasons.append("emotional distress")
        return result

    return result


# ── Canned responses ───────────────────────────────────────────────────────
# Emergencies are answered from a fixed template, never by the model, so
# the wording cannot drift and cannot be influenced by the message.

_EMERGENCY_RESPONSE = (
    "**Please get medical help right now.**\n\n"
    "What you've described can be serious and needs a person, not an app. "
    "Please call your care team's urgent line straight away. If you can't "
    "reach them, or if this feels severe, call your local emergency number "
    "or go to your nearest emergency department.\n\n"
    "If you have a 24-hour oncology triage number from your hospital, that "
    "is usually the fastest route.\n\n"
    "I'm not able to help with something urgent like this, and I don't want "
    "to slow you down by trying."
)

_SELF_HARM_RESPONSE = (
    "I'm really glad you told me, and I'm sorry you're carrying this.\n\n"
    "I'm not the right kind of help for this, but people are available right "
    "now who are. If you're in the United States you can call or text **988** "
    "to reach the Suicide and Crisis Lifeline, any time. If you're somewhere "
    "else, your local emergency number can connect you.\n\n"
    "If you're in immediate danger, please call your emergency services.\n\n"
    "Please also tell someone you trust, and your care team. They will want "
    "to know, and they can help."
)


def emergency_response(result: TriageResult) -> str:
    return _SELF_HARM_RESPONSE if result.self_harm else _EMERGENCY_RESPONSE


def system_prompt_additions(result: TriageResult) -> str:
    """Extra system-prompt guidance for non-blocking categories."""
    if result.category == CLINICAL_DECISION:
        base = (
            "\n\nIMPORTANT: This patient is asking something only their care team "
            "can decide. Explain what is generally known so they understand the "
            "topic, but do NOT tell them what to do, and do NOT suggest they "
            "change, stop, delay, or start anything. Finish by encouraging them "
            "to put this exact question to their care team, and let them know "
            "you can send it over for them."
        )
        if result.prognosis:
            base += (
                "\n\nThis is a question about prognosis or survival. Do NOT give "
                "statistics, percentages, or timelines. Numbers from studies "
                "describe groups, never an individual, and giving them here would "
                "be misleading and frightening. Acknowledge the question with "
                "warmth, explain briefly why their own doctor is the only person "
                "who can speak to their situation, and offer to pass it on."
            )
        return base

    if result.category == DISTRESS:
        return (
            "\n\nIMPORTANT: This patient sounds distressed. Lead with warmth and "
            "acknowledge how they're feeling before any information. Keep the "
            "answer short and gentle. Mention that their care team can connect "
            "them with support, and that many cancer centres have counsellors "
            "and support groups."
        )
    return ""
