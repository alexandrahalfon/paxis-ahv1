"""
Patient chat.

Orchestrates one patient message end to end:

    triage -> (block if emergency) -> vocabulary expansion -> retrieval
    -> plain-language answer -> optional single follow-up -> persist

Reuses the existing retrieval pipeline unchanged. The only differences
from the clinician path are the prompt, the safety layer, and the
gap-filling follow-up.

Three sources of what we know about the patient, in priority order:

1. Their linked clinical record, if a physician connected them. Then we
   never interrogate, we confirm.
2. Facts gathered earlier in this conversation.
3. Whatever is in the current message.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.api.services.patient_db import get_patient_db
from src.api.services.patient_portal import patient_safety_service as safety
from src.api.services.patient_portal.patient_vocab import (
    GAP_QUESTIONS,
    detect_gaps,
    expand_patient_language,
)

logger = logging.getLogger(__name__)


BASE_SYSTEM_PROMPT = """You are a patient-friendly oncology assistant, talking \
directly to a patient or their family member about cancer care.

Your role:
- Explain in clear, simple, compassionate, non-technical language.
- Describe what things mean, how treatments generally work, what to expect,
  and common side effects to be aware of.
- Do NOT recommend new treatments or suggest changes to their current plan.
- Do NOT diagnose, and do NOT contradict or second-guess their care team.
- Do NOT give statistics about survival or prognosis for this individual.
- When relevant, mention which symptoms should prompt them to contact their
  care team.
- If you reference a study, explain what it found in plain terms and never
  imply it applies directly to this specific person.
- If you genuinely do not know, say so plainly rather than guessing.

Style:
- Warm, calm, and direct. Short paragraphs. No medical abbreviations unless
  you immediately explain them.
- Write at roughly an eighth-grade reading level.
- Never open with "I'm sorry to hear" more than once in a conversation.
- Do not end every message with the same disclaimer; the interface already
  shows one. A closing nudge to speak with their care team is enough when
  it is actually relevant.

When someone opens by describing their situation rather than asking a
question ("they said it spread and I'm scared", "I just got diagnosed"):
- Acknowledge it briefly and warmly first. Do not lead with information.
- Explain what you can about what they described, in everyday words.
- Then offer two or three specific things you could help with next, phrased
  as an invitation rather than a menu. Many people do not know what they are
  allowed to ask.
- Never treat a vague opening as a failure to give you enough detail.

Many people using this will not have a doctor connected here, and may not
know their exact diagnosis, drug names, or stage. That is completely normal.
Work with whatever they give you, be useful at the level of detail you have,
and never imply they should have known more or come better prepared.
"""


@dataclass
class ChatResult:
    answer: str
    safety_category: str = safety.GENERAL
    followup_question: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    offer_escalation: bool = False
    conversation_id: Optional[str] = None
    known_facts: Dict[str, Any] = field(default_factory=dict)
    retrieval_used: bool = False
    used_web_search: bool = False
    # Bare UUID, safe to log/return — the trace content itself lives only
    # in query_debug_traces, never here. See retrieval_debug_trace.py.
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "safety_category": self.safety_category,
            "followup_question": self.followup_question,
            "sources": self.sources,
            "offer_escalation": self.offer_escalation,
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "known_facts": self.known_facts,
            "retrieval_used": self.retrieval_used,
            "used_web_search": self.used_web_search,
        }


class PatientChatService:
    def __init__(self):
        self._openai = None

    def _client(self):
        """Shared OpenAI client. The previous patient endpoint built a new
        one on every request."""
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=settings.openai_api_key)
        return self._openai

    # ── Known facts ───────────────────────────────────────────────────

    async def known_facts_for(
        self, patient_user_id: str, conversation_facts: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Merge the linked clinical record with facts from this conversation."""
        facts: Dict[str, Any] = dict(conversation_facts or {})
        try:
            from src.api.services.patient_portal.patient_link_service import (
                get_patient_link_service,
            )
            record = await get_patient_link_service().get_linked_record(patient_user_id)
            if not record:
                return facts

            facts["linked"] = True
            facts["patient_record_id"] = record["patient_id"]
            facts["physician_id"] = record["physician_id"]

            from src.api.services.patient_service import get_patient_service
            full = await get_patient_service().get_patient_full(
                record["patient_id"], record["physician_id"]
            )
            if full:
                dx = full.get("diagnosis") or {}
                if dx.get("cancer_site"):
                    facts["cancer_type"] = dx["cancer_site"]
                if dx.get("histology"):
                    facts["histology"] = dx["histology"]
                if dx.get("stage"):
                    facts["stage"] = dx["stage"]
                bios = full.get("biomarkers") or []
                if bios:
                    facts["biomarkers"] = [
                        f"{b.get('biomarker_name')} {b.get('value') or ''}".strip()
                        for b in bios if b.get("biomarker_name")
                    ]
                tx = full.get("treatment_history") or []
                if tx:
                    facts["treatment"] = ", ".join(
                        t.get("regimen") or t.get("treatment_type") or ""
                        for t in tx if (t.get("regimen") or t.get("treatment_type"))
                    ).strip(", ")
        except Exception as e:
            logger.warning("[PatientChat] known-facts lookup failed: %s", e)
        return facts

    # ── Retrieval ─────────────────────────────────────────────────────

    async def _retrieve(self, expanded_query: str, top_k: int = 6):
        """Lightweight semantic search over the corpus.

        Deliberately NOT the clinician backbone. Even its "fast" mode runs
        the full machinery: three-source intake (Qdrant + Postgres + PTO),
        a per-document cross-encoder gate, a priority queue, and an LLM
        eligibility check. That exists to match a complex patient profile
        against trial eligibility, which is a genuinely hard problem.

        A patient asking "what is pembrolizumab" or "what side effects
        should I expect" does not need any of it, and paying for it made
        the chat take tens of seconds. Plain embedding + vector search
        answers those well and returns in a fraction of the time.

        Complex profile matching stays on the clinician side, where it
        belongs.
        """
        try:
            from src.api.services.comprehensive_retrieval import (
                get_comprehensive_retriever,
            )
            r = get_comprehensive_retriever()
            vec = await r._embed_async(expanded_query)
            resp = await r._qdrant_query(
                query=vec,
                limit=top_k * 3,      # over-fetch, then dedupe by document
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(resp, "points", resp) or []

            seen, out = set(), []
            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id or doc_id in seen:
                    continue
                text = (payload.get("text") or "").strip()
                if not text:
                    continue
                seen.add(doc_id)
                meta = payload.get("doc_meta") or {}
                out.append({
                    "title": meta.get("title") or payload.get("title") or "Study",
                    "text": text,
                    "citation": meta.get("citation") or meta.get("citation_string"),
                    "year": meta.get("year"),
                })
                if len(out) >= top_k:
                    break
            return out
        except Exception as e:
            logger.warning("[PatientChat] retrieval failed, answering without: %s", e)
            return []

    async def _retrieve_web(self, query: str, limit: int = 4):
        """PubMed fallback when the internal corpus has nothing useful.

        The ingested corpus does not cover everything a patient might ask
        about, especially newly approved drugs. Without this, a reasonable
        question about something not yet ingested gets an answer with no
        evidence behind it at all.

        Same source the clinician path uses (LiteratureSearchService), so
        there is no second search stack to maintain. Synchronous, so it is
        threaded. Always fails soft: no network means we answer from the
        model's own knowledge with the usual caveats rather than erroring.
        """
        try:
            from src.api.services.literature_search_service import (
                LiteratureSearchService,
            )
            svc = LiteratureSearchService()
            articles = await asyncio.to_thread(
                svc.search_pubmed, query, max_results=limit
            )
            out = []
            for a in articles or []:
                abstract = (a.get("abstract") or "").strip()
                if not abstract:
                    continue
                # literature_search_service stores authors as
                # "LastName ForeName" (see its Author parsing), so the
                # surname is the FIRST token. Taking the last token yields
                # the given name: "Burtness Barbara" -> "Barbara et al."
                authors = [x for x in (a.get("authors") or []) if x]
                author_str = f"{authors[0].split()[0]} et al." if authors else None
                citation = (
                    f"{author_str}, {a.get('journal', '')}, {a.get('year', '')}"
                    if author_str else (a.get("journal") or "PubMed")
                )
                out.append({
                    "title": a.get("title", "") or "PubMed article",
                    "text": abstract,
                    "citation": citation,
                    "year": a.get("year"),
                    "source_type": "pubmed",
                })
            if out:
                logger.info("[PatientChat] web fallback returned %d articles", len(out))
            return out
        except Exception as e:
            logger.warning("[PatientChat] web fallback failed (continuing): %s", e)
            return []

    @staticmethod
    def _web_block(articles, limit: int = 4):
        """Same shape as _evidence_block, for PubMed results."""
        if not articles:
            return "", []
        block, sources = "", []
        for i, a in enumerate(articles[:limit], 1):
            block += f"\n[{i}] {a['title']}\n{a['text'][:500]}\n"
            sources.append({
                "title": a["title"],
                "citation": a.get("citation"),
                "year": a.get("year"),
                "source_type": "pubmed",
            })
        return block, sources

    @staticmethod
    def _care_team_instructions_block(context: Optional[Dict[str, Any]]) -> str:
        """Renders active care_team_instructions (Phase 1 finalization —
        clinician-entered guidance like "no NSAIDs while on this
        regimen") into an explicit system-prompt directive.

        Deliberately independent of the facts.get("linked")/context
        personalization branching in answer(): those are the OLD
        physician-linked-record summary vs. the NEW patient_profile
        summary, and a patient can have either, both, or neither.
        Care-team instructions live only in the new patient_profile model
        (care_team_instructions table), so this is called unconditionally
        against `context` rather than nested inside either branch —
        otherwise a patient with a legacy physician link (facts["linked"]
        True, which short-circuits the `elif context:` branch) but who
        ALSO has instructions recorded against their patient_profile
        would silently never see them.

        2026-08-12 beta audit item 5: "care-team-specific instruction >
        generic education" was one of the architecture's central rules
        but had no downstream consumer — `evidence_packet_builder.
        summarize_context()` didn't surface the instruction text and
        neither prompt branch below read it. This is that consumer.
        """
        state = (context or {}).get("state") or {}
        instructions = [i for i in (state.get("care_team_instructions") or []) if i.get("text")]
        if not instructions:
            return ""
        lines = "\n".join(
            f"- [{i.get('type') or 'other'}] {i['text']}" for i in instructions
        )
        return (
            "\n\nCARE TEAM INSTRUCTIONS FOR THIS PATIENT (recorded directly in their "
            "chart by their own care team):\n" + lines +
            "\n\nThese are specific to this patient and take precedence over generic "
            "education: if a generic answer would conflict with one of these "
            "instructions, follow the instruction instead of giving the generic "
            "advice. If the patient's question is already answered by one of these "
            "instructions, lead with it rather than a generic explanation. Do not "
            "invent a reason for an instruction beyond what is stated here."
        )

    @staticmethod
    def _evidence_block(results, limit: int = 5):
        """Same shape as _web_block, for corpus results."""
        if not results:
            return "", []
        block, sources = "", []
        for i, s in enumerate(results[:limit], 1):
            block += f"\n[{i}] {s['title']}\n{s['text'][:500]}\n"
            sources.append({
                "title": s["title"],
                "citation": s.get("citation"),
                "year": s.get("year"),
            })
        return block, sources

    # ── Main entry point ──────────────────────────────────────────────

    async def answer(
        self,
        message: str,
        patient_user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_facts: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        persist: bool = True,
    ) -> ChatResult:
        history = conversation_history or []

        # 1. Safety first, before anything else touches the message.
        tri = safety.triage(message)

        if tri.blocks_answer:
            result = ChatResult(
                answer=safety.emergency_response(tri),
                safety_category=tri.category,
                offer_escalation=False,   # emergencies go to humans, not a queue
                conversation_id=conversation_id,
            )
            if persist:
                result.conversation_id = await self._persist(
                    patient_user_id, conversation_id, message, result
                )
            return result

        # 2. What do we already know about them?
        facts = await self.known_facts_for(patient_user_id, conversation_facts)

        # Debug trace, accumulated across every stage below and persisted
        # at the end regardless of which retrieval path was taken — see
        # evidence/retrieval_debug_trace.py. Resolution failure (no
        # profile yet, DB unreachable) degrades to no trace, never to a
        # failed answer.
        trace = None
        try:
            from src.api.services.patient.patient_profile_service import (
                get_patient_profile_service,
            )
            profile = await get_patient_profile_service().get_by_user(patient_user_id)
            if profile:
                from src.api.services.evidence.retrieval_debug_trace import TraceBuilder
                trace = TraceBuilder(patient_profile_id=profile["id"], question=message)
        except Exception:
            logger.warning("[PatientChat] trace setup failed (continuing without)", exc_info=True)

        # 3. Expand their words into clinical vocabulary for retrieval.
        vocab = expand_patient_language(message)
        query = vocab.expanded_query

        # Computed unconditionally (pure regex, no I/O) so it's available
        # for the hard grounding gate below even if multi-corpus retrieval
        # itself fails before reaching its own classify_intent() call.
        from src.api.services.evidence import patient_context_service as evidence_patient_context_service
        intent = evidence_patient_context_service.classify_intent(message)

        # 4. Patient-state-aware multi-corpus retrieval (Phase 4). Replaces
        #    the old "concatenate known facts onto the query string" seed
        #    below with structured selection: an intent label picks which
        #    corpora to search (patient education / medication / guideline
        #    / literature) and the patient's actual regimen/agents/symptoms
        #    become soft boosts, not extra embedding text. See
        #    evidence/retrieval_planner.py section 17 of the architecture
        #    review for the worked example this replaced.
        #
        #    context carries patient_profile state independent of whether
        #    a physician is linked — see patient/patient_state_service.py
        #    (Phase 0/1) — so an unlinked patient who has entered or
        #    uploaded their own data still gets personalized retrieval.
        context: Dict[str, Any] = {}
        evidence_block, sources, used_web = "", [], False
        try:
            from src.api.services.evidence.patient_context_service import (
                get_patient_context_service,
            )
            from src.api.services.evidence.retrieval_planner import build_plan
            from src.api.services.evidence import multi_corpus_retriever
            from src.api.services.evidence.applicability_scorer import rank as rank_evidence
            from src.api.services.evidence.evidence_packet_builder import (
                build_packet, to_prompt_block, to_sources,
            )

            context = await get_patient_context_service().get_context(patient_user_id)
            plan = build_plan(intent, context.get("retrieval_features", {}))
            candidates = await multi_corpus_retriever.search(query, plan)
            ranked = rank_evidence(candidates, plan)
            from src.api.services.patient.lab_interpretation import interpretation_policy_summary
            packet = build_packet(
                message, context, ranked, safety_category=tri.category,
                audience="patient",
                retrieval_plan=plan,
                # Stand-in for a real state-revision-scoped snapshot id
                # until B7 (deterministic state freshness via revision
                # counters) lands -- patient_profile_id is the closest
                # thing that exists today to "which patient's state this
                # packet was built from".
                patient_snapshot_id=context.get("patient_profile_id"),
                interpretation_policies=interpretation_policy_summary(
                    context.get("state", {}).get("labs")
                ),
            )
            evidence_block = to_prompt_block(packet)
            sources = to_sources(packet)

            if trace:
                trace.set_intent(intent, {"boost_terms": plan.boost_terms})
                trace.set_routing(plan.collections)
                trace.set_context(context.get("retrieval_features", {}))
                trace.set_candidates(candidates)
                trace.set_ranked(ranked)
                trace.set_packet(packet)
        except Exception as e:
            logger.warning("[PatientChat] multi-corpus retrieval failed, falling back: %s", e)

        # Fall back to the plain literature-only search this replaced.
        # Covers both an outright failure above and a genuine empty result
        # — the new patient-education/medication/guideline collections
        # start unpopulated in most deployments of this change (Phase 3),
        # so this is the common path today, not a rare one. Seeding with
        # known facts here (as the old code always did) still helps a
        # vague question land somewhere useful for a linked patient.
        if not sources:
            seed = " ".join(
                str(facts.get(k)) for k in ("cancer_type", "histology", "stage", "treatment")
                if facts.get(k)
            )
            fallback_query = f"{seed} {query}".strip() if seed else query
            corpus = await self._retrieve(fallback_query)
            evidence_block, sources = self._evidence_block(corpus)

        # PubMed stays the last resort, tried only after every internal
        # corpus has had a chance — patient education, medication,
        # guideline, then literature — per the fallback ordering in the
        # architecture review section 27 ("don't use PubMed as the main
        # patient-chat fallback").
        if not sources:
            web = await self._retrieve_web(query)
            if web:
                evidence_block, sources = self._web_block(web)
                used_web = True

        # 4b. Hard grounding gate (2026-08-12 beta audit, "make the
        # evidence packet a true hard boundary"): a factual medication/
        # symptom/nutrition/treatment/diagnosis question with zero usable
        # evidence after every fallback above must not be answered from
        # the model's own memory — the evidence packet is empty, so any
        # specifics in the answer would be ungrounded. Skips generation
        # entirely rather than generating and then discarding, so this
        # never costs an OpenAI call. Distress/clinical-decision/
        # emergency categories are already handled without needing
        # retrieved evidence (see patient_safety_service.
        # system_prompt_additions), so this only fires for the plain
        # "general" safety category — see FACTUAL_INTENTS' docstring in
        # patient_context_service.py for the full rationale.
        if (
            not sources
            and intent in evidence_patient_context_service.FACTUAL_INTENTS
            and tri.category == safety.GENERAL
        ):
            result = ChatResult(
                answer=evidence_patient_context_service.NO_EVIDENCE_RESPONSE,
                safety_category=tri.category,
                sources=[],
                offer_escalation=tri.should_offer_escalation and bool(facts.get("linked")),
                conversation_id=conversation_id,
                known_facts={k: v for k, v in facts.items() if k != "physician_id"},
                retrieval_used=False,
                used_web_search=used_web,
            )
            if trace:
                trace.set_answer(result.answer)
                trace.set_grounding({"gated": True, "reason": "no_usable_evidence_for_factual_intent"})
                result.trace_id = await trace.save()
            if persist:
                result.conversation_id = await self._persist(
                    patient_user_id, conversation_id, message, result
                )
            return result

        # 5. Build the prompt.
        system = BASE_SYSTEM_PROMPT + safety.system_prompt_additions(tri)
        if facts.get("linked"):
            summary = ", ".join(
                f"{k}: {v}" for k, v in facts.items()
                if k in ("cancer_type", "histology", "stage", "treatment", "biomarkers") and v
            )
            if summary:
                system += (
                    f"\n\nWhat their care team has on record for this patient: {summary}. "
                    "Use this so your answer is about their situation rather than generic. "
                    "Do not ask them to repeat information that is already listed here. "
                    "You may confirm it naturally in passing, but never present it as a "
                    "new finding or interpret it as a diagnosis."
                )
        elif context:
            # Not linked to a physician, but this account has its own
            # patient_profile data — self-entered or from a confirmed
            # document upload (Phase 0/2). Same personalization, phrased
            # as their own record rather than "their care team has".
            try:
                from src.api.services.evidence.evidence_packet_builder import summarize_context
                ctx_summary = summarize_context(context)
            except Exception:
                ctx_summary = {}
            if ctx_summary:
                summary = ", ".join(
                    f"{k}: {v}" for k, v in ctx_summary.items()
                    if v and k != "care_team_instructions"
                )
                if summary:
                    system += (
                        f"\n\nWhat this patient has recorded about their own situation: {summary}. "
                        "Use this so your answer is about their situation rather than generic. "
                        "Do not ask them to repeat information already listed here. Never present "
                        "it as a new finding or interpret it as a diagnosis."
                    )

        # Independent of the linked/unlinked branching above — see the
        # method's docstring for why (item 5).
        system += self._care_team_instructions_block(context)

        messages = [{"role": "system", "content": system}]
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        user_content = f"Patient message: {message}"
        if evidence_block.strip():
            origin = (
                "recent published research (PubMed abstracts, written for "
                "clinicians)" if used_web else "the medical literature"
            )
            user_content += (
                f"\n\nRelevant information from {origin}. Use it to inform your "
                "answer, but translate it into plain language and do not quote "
                "statistics at this individual:\n" + evidence_block +
                "\n\nWhen you make a claim drawn from the numbered passages above, "
                "cite it inline using its number in brackets, e.g. [1] or [2] — "
                "matching the numbers shown, not a new numbering of your own. Do not "
                "cite a number for something drawn only from what the patient told you "
                "or from their record; citations are for the passages above only."
            )
        messages.append({"role": "user", "content": user_content})

        # 6. Generate. Threaded: the previous endpoint blocked the event
        #    loop for the whole call, stalling every other user.
        try:
            resp = await asyncio.to_thread(
                self._client().chat.completions.create,
                model=settings.openai_mini_model or "gpt-4o-mini",
                temperature=0.4,
                max_tokens=700,
                messages=messages,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("[PatientChat] generation failed")
            raise RuntimeError("generation_failed") from e

        # 6b. Grounding validation — hard retry/fail gate (2026-08-12 beta
        # audit item 8, promoted from log-only): did the answer actually
        # cite the evidence it was given, per
        # evidence/grounding_validator.py.
        #
        # Only enforced as a hard gate when `sources` is non-empty —
        # validate() itself treats an EMPTY evidence packet as invalid
        # too (a defense against the pregen hard gate in step 4b being
        # bypassed), but reaching this point with empty sources is the
        # legitimate, expected case for conversational/non-factual
        # messages and non-GENERAL safety categories, which step 4b
        # deliberately lets through without evidence. Gating on that
        # here would block ordinary "thank you" replies, so `and
        # sources` scopes the retry/fallback to exactly the case item 8
        # describes: a packet existed and the answer failed to ground
        # itself in it.
        grounding_result = None
        retried = False
        try:
            from src.api.services.evidence.grounding_validator import (
                validate as validate_grounding,
                RETRY_INSTRUCTION,
                SAFE_FALLBACK_RESPONSE,
            )
            grounding_result = validate_grounding(answer, {"evidence": sources})

            if not grounding_result.valid and sources:
                logger.warning(
                    "[PatientChat] grounding validation failed on first attempt: %s "
                    "— retrying once with a stricter prompt", grounding_result.reasons,
                )
                retried = True
                try:
                    retry_messages = messages + [
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": RETRY_INSTRUCTION},
                    ]
                    resp2 = await asyncio.to_thread(
                        self._client().chat.completions.create,
                        model=settings.openai_mini_model or "gpt-4o-mini",
                        temperature=0.2,
                        max_tokens=700,
                        messages=retry_messages,
                    )
                    retry_answer = resp2.choices[0].message.content.strip()
                    retry_result = validate_grounding(retry_answer, {"evidence": sources})
                except Exception:
                    logger.warning(
                        "[PatientChat] grounding retry generation failed, "
                        "falling back to safe response", exc_info=True,
                    )
                    retry_answer, retry_result = None, None

                if retry_result and retry_result.valid:
                    answer, grounding_result = retry_answer, retry_result
                else:
                    if retry_result:
                        logger.warning(
                            "[PatientChat] grounding validation failed again after "
                            "retry: %s — falling back to safe response", retry_result.reasons,
                        )
                    answer = SAFE_FALLBACK_RESPONSE
                    sources = []
                    grounding_result = retry_result or grounding_result
        except Exception:
            logger.warning("[PatientChat] grounding validation errored (continuing)", exc_info=True)

        trace_id = None
        if trace:
            trace.set_answer(answer)
            if grounding_result:
                trace_grounding = grounding_result.to_dict()
                trace_grounding["retried"] = retried
                trace.set_grounding(trace_grounding)
            trace_id = await trace.save()

        # 7. At most one follow-up, and only when it would change the answer.
        followup = None
        if tri.category == safety.GENERAL and not facts.get("linked"):
            asked = " ".join(
                t.get("content", "") for t in history if t.get("role") == "assistant"
            )
            for gap in detect_gaps(message, known=facts):
                q = GAP_QUESTIONS[gap]
                # Don't repeat a question already asked in this conversation.
                if q[:40] not in asked:
                    followup = q
                    break

        result = ChatResult(
            answer=answer,
            safety_category=tri.category,
            followup_question=followup,
            sources=sources,
            offer_escalation=tri.should_offer_escalation and bool(facts.get("linked")),
            conversation_id=conversation_id,
            known_facts={k: v for k, v in facts.items() if k != "physician_id"},
            retrieval_used=bool(sources),
            used_web_search=used_web,
            trace_id=trace_id,
        )

        if persist:
            result.conversation_id = await self._persist(
                patient_user_id, conversation_id, message, result
            )
        return result

    # ── Persistence ───────────────────────────────────────────────────

    async def _persist(
        self,
        patient_user_id: str,
        conversation_id: Optional[str],
        user_message: str,
        result: ChatResult,
    ) -> Optional[str]:
        """Save the exchange. Never raises: a storage failure must not
        cost the patient their answer."""
        try:
            import json
            db = get_patient_db()
            await db.ensure_schema()
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                # Ownership check. conversation_id arrives from the client,
                # so without this a patient could pass someone else's id and
                # have their messages appended to that person's conversation.
                # An id that isn't theirs is discarded rather than rejected,
                # which starts a fresh conversation and leaks nothing about
                # whether the id exists.
                if conversation_id:
                    owner = await conn.fetchval(
                        "SELECT patient_user_id FROM patient_conversations WHERE id = $1",
                        conversation_id,
                    )
                    if owner is None or str(owner) != str(patient_user_id):
                        logger.warning(
                            "[PatientChat] discarding conversation_id not owned by caller"
                        )
                        conversation_id = None

                if not conversation_id:
                    conversation_id = str(uuid.uuid4())
                    await conn.execute(
                        """
                        INSERT INTO patient_conversations
                            (id, patient_user_id, patient_record_id, title)
                        VALUES ($1, $2, $3, $4)
                        """,
                        conversation_id,
                        patient_user_id,
                        result.known_facts.get("patient_record_id"),
                        (user_message or "")[:80],
                    )
                else:
                    await conn.execute(
                        "UPDATE patient_conversations SET updated_at = now() WHERE id = $1",
                        conversation_id,
                    )
                await conn.execute(
                    """
                    INSERT INTO patient_messages
                        (id, conversation_id, role, content, safety_category)
                    VALUES ($1, $2, 'patient', $3, $4)
                    """,
                    str(uuid.uuid4()), conversation_id, user_message, result.safety_category,
                )
                await conn.execute(
                    """
                    INSERT INTO patient_messages
                        (id, conversation_id, role, content, safety_category, sources)
                    VALUES ($1, $2, 'assistant', $3, $4, $5::jsonb)
                    """,
                    str(uuid.uuid4()), conversation_id, result.answer,
                    result.safety_category, json.dumps(result.sources),
                )
            return conversation_id
        except Exception as e:
            logger.warning("[PatientChat] persist failed (answer still returned): %s", e)
            return conversation_id


_service: Optional[PatientChatService] = None


def get_patient_chat_service() -> PatientChatService:
    global _service
    if _service is None:
        _service = PatientChatService()
    return _service
