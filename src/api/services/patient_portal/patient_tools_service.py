"""
Patient tools: medication explainer and report explainer.

Both reuse the chat service's retrieval (corpus first, PubMed fallback)
and its safety triage. They differ from the chat only in prompt and
output shape.

Framing is the whole safety story for these two:

* **Medication.** Describes what a drug is and what studies found *in
  general*. Never says what will happen to this person, never gives
  survival numbers, never suggests starting or stopping anything.

* **Report.** Explains what the words mean. Deliberately does NOT say
  whether results are good or bad, what they imply for prognosis, or what
  should happen next. That is interpretation, it belongs to the care
  team, and it is the line that keeps this an educational tool.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.api.services.patient_portal import patient_safety_service as safety

logger = logging.getLogger(__name__)


MEDICATION_PROMPT = """You are explaining a cancer medication to a patient or \
their family, in plain language.

Structure your answer with these short sections, using markdown headings:

**What it is** - the drug class and what it is generally used for.
**How it works** - the mechanism, in everyday terms. Avoid jargon; if you
must use a term, explain it in the same sentence.
**What the research looks at** - what kinds of studies have been done and
what they generally measured. Describe findings as things researchers
observed in groups of people.
**Common side effects** - what people frequently report, and which ones
should prompt a call to the care team.
**Good questions for your care team** - three or four specific questions
worth asking about this drug.

Hard rules:
- Do NOT state or imply what will happen to this individual.
- Do NOT give survival statistics, response rates, or percentages for
  outcomes. Numbers from studies describe groups, never a person, and
  quoting them here would be misleading and frightening.
- Do NOT recommend starting, stopping, changing, or delaying anything.
- Do NOT compare this drug to alternatives as if advising a choice.
- If the evidence provided does not cover something, say so plainly.

Tone: warm, calm, roughly an eighth-grade reading level. Short paragraphs."""


REPORT_PROMPT = """You are helping a patient understand the words in a medical \
report (pathology, imaging, or lab). Many people receive these before they can \
speak to anyone, and the language is impenetrable.

Your job is to explain TERMINOLOGY, not to interpret results.

Do this:
- Pick out the medical terms, abbreviations, and measurements that appear,
  and explain what each one means in general.
- Explain what the type of test is and what it generally looks at.
- Note where a term is simply descriptive rather than a finding.
- Finish with two or three questions they could ask their care team about
  this report.

Do NOT do this, under any circumstances:
- Do NOT say whether a result is good, bad, reassuring, or concerning.
- Do NOT say what the findings mean for their diagnosis, stage, outlook,
  or treatment.
- Do NOT speculate about what the care team will do next.
- Do NOT give any prognosis, survival, or probability information.
- Do NOT diagnose anything.

If they ask what it means for them, say honestly that the same words can
mean very different things depending on the whole picture, and that their
care team is the only one who can put it in context for them. Be warm
about it, not evasive.

Tone: calm and steady, roughly an eighth-grade reading level. This is
often read by someone who is frightened."""


@dataclass
class ToolResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    safety_category: str = safety.GENERAL
    used_web_search: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "safety_category": self.safety_category,
            "used_web_search": self.used_web_search,
        }


class PatientToolsService:
    def _chat(self):
        from src.api.services.patient_portal.patient_chat_service import (
            get_patient_chat_service,
        )
        return get_patient_chat_service()

    async def _generate(self, system: str, user: str, max_tokens: int = 900) -> str:
        chat = self._chat()
        resp = await asyncio.to_thread(
            chat._client().chat.completions.create,
            model=settings.openai_mini_model or "gpt-4o-mini",
            temperature=0.3,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    # ── Medication explainer ──────────────────────────────────────────

    async def explain_medication(
        self, medication: str, patient_user_id: Optional[str] = None
    ) -> ToolResult:
        med = (medication or "").strip()
        if not med:
            raise ValueError("Please enter a medication name.")

        chat = self._chat()
        query = f"{med} treatment outcomes side effects mechanism"

        # Medication lookup prefers the medication-knowledge corpus
        # (DailyMed/FDA/MedlinePlus/Chemocare, once ingested — Phase 3)
        # over the general literature collection, per the architecture
        # review section 36. Falls back to the plain literature search
        # the same way patient_chat_service.answer() does when that
        # corpus is empty or the import fails.
        block, sources = "", []
        try:
            from src.api.services.evidence.retrieval_planner import build_plan
            from src.api.services.evidence.patient_context_service import INTENT_MEDICATION
            from src.api.services.evidence import multi_corpus_retriever
            from src.api.services.evidence.applicability_scorer import rank as rank_evidence
            from src.api.services.evidence.evidence_packet_builder import (
                build_packet, to_prompt_block, to_sources,
            )

            plan = build_plan(INTENT_MEDICATION, {})
            candidates = await multi_corpus_retriever.search(query, plan)
            ranked = rank_evidence(candidates, plan)
            packet = build_packet(query, {}, ranked)
            block = to_prompt_block(packet)
            sources = to_sources(packet)
        except Exception as e:
            logger.warning("[PatientTools] medication multi-corpus retrieval failed: %s", e)

        if not sources:
            corpus = await chat._retrieve(query, top_k=6)
            block, sources = chat._evidence_block(corpus)

        used_web = False
        if not sources:
            web = await chat._retrieve_web(query, limit=5)
            if web:
                block, sources = chat._web_block(web)
                used_web = True

        # Personalise lightly when we know their diagnosis, without ever
        # implying the evidence applies to them specifically.
        context = ""
        if patient_user_id:
            facts = await chat.known_facts_for(patient_user_id)
            if facts.get("cancer_type"):
                context = (
                    f"\n\nFor context, this person is being treated for "
                    f"{facts['cancer_type']}. You may mention that this drug is "
                    f"commonly used in that setting IF the evidence supports it, "
                    f"but do not draw any conclusion about their own outcome."
                )

        user = f"Medication the patient asked about: {med}{context}"
        if block.strip():
            origin = "recent published research" if used_web else "the medical literature"
            user += (
                f"\n\nRelevant information from {origin}. Translate it into plain "
                f"language, and describe findings as things seen in groups of "
                f"people rather than predictions:\n{block}"
            )
        else:
            user += (
                "\n\nNo specific studies were retrieved. Explain what is generally "
                "known about this drug and be clear about the limits of what you "
                "can say."
            )

        answer = await self._generate(MEDICATION_PROMPT, user)
        return ToolResult(answer=answer, sources=sources, used_web_search=used_web)

    # ── Report explainer ──────────────────────────────────────────────

    async def explain_report(self, report_text: str) -> ToolResult:
        text = (report_text or "").strip()
        if len(text) < 20:
            raise ValueError("Please paste a bit more of the report so I can help.")

        # Triage the report text too. People paste a report and add "am I
        # dying?" at the bottom, and that should be handled as the
        # question it is rather than swallowed by the explainer.
        tri = safety.triage(text)
        if tri.blocks_answer:
            return ToolResult(
                answer=safety.emergency_response(tri),
                safety_category=tri.category,
            )

        # Truncated so an enormous paste cannot blow the context window.
        excerpt = text[:6000]
        system = REPORT_PROMPT + safety.system_prompt_additions(tri)
        user = (
            "Here is the report the patient received. Explain the terminology "
            "in it. Remember: explain what the words mean, never whether the "
            "results are good or bad.\n\n"
            f"---\n{excerpt}\n---"
        )
        answer = await self._generate(system, user, max_tokens=1200)
        return ToolResult(answer=answer, safety_category=tri.category)

    # ── Appointment question prep ─────────────────────────────────────

    async def prepare_questions(self, topics: List[str]) -> ToolResult:
        """Turn what someone has been asking about into questions to bring
        to their next appointment."""
        joined = "; ".join(t for t in topics if t)[:2000]
        if not joined:
            raise ValueError("Ask a few questions first and I'll turn them into a list.")

        system = (
            "You help patients get more out of short appointments. Given what "
            "someone has been wondering about, write a focused list of questions "
            "they could ask their care team.\n\n"
            "Rules:\n"
            "- 5 to 8 questions, each one sentence, in plain language.\n"
            "- Specific and answerable, not vague.\n"
            "- Group under two or three short headings.\n"
            "- Do NOT include questions that assume a diagnosis or outcome.\n"
            "- Do NOT give any medical advice or answers, only questions.\n"
            "- Open with one short encouraging line, then the list."
        )
        user = f"Things this person has been asking about:\n{joined}"
        answer = await self._generate(system, user, max_tokens=700)
        return ToolResult(answer=answer)


_service: Optional[PatientToolsService] = None


def get_patient_tools_service() -> PatientToolsService:
    global _service
    if _service is None:
        _service = PatientToolsService()
    return _service
