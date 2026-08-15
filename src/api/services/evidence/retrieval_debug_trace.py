"""
Retrieval Debug Trace (Phase 4 hardening)

Records every stage of one patient-chat query — intent, ontology hits,
routed corpora, selected patient context, raw retrieval candidates, score
components, final ranking, the evidence packet, the generated answer, and
the grounding-validation result — as a single row in query_debug_traces,
so a bad answer can be diagnosed without reproducing it blind. This is
the exact "you won't know whether the failure came from patient-state
extraction, ontology normalization, retrieval, filtering, applicability
scoring, source ranking, or generation" gap the architecture review
flags, and the exact fields the reference Colab notebook's debug-trace
cell (20) writes.

PHI handling: a trace is patient-derived content (it embeds the
query-specific patient context and the generated answer), so it is
stored in its own access-controlled table — the same CASCADE-on-profile-
delete behavior every other patient-owned table gets — never in a
generic application log line. Code elsewhere in this codebase (chat
service, routes) should log only the trace_id (a bare UUID, meaningless
without DB access) for correlation, never the trace content itself. This
module enforces that boundary by design: build_trace()/save() never call
logger.info/warning with the trace body, only with the id.

Usage — a builder accumulated across the pipeline, not a single call,
since the pipeline's intermediate values are produced at different
points in patient_chat_service.answer():

    trace = TraceBuilder(patient_profile_id=..., question=...)
    trace.set_intent(intent, ontology_hits)
    trace.set_routing(collections)
    trace.set_context(context_summary)
    trace.set_candidates(candidates)
    trace.set_ranked(ranked)
    trace.set_packet(packet)
    trace.set_answer(answer)
    trace.set_grounding(grounding_result.to_dict())
    trace_id = await trace.save()

Shared schema (2026-08-12 convergence Sprint A item 6): to_dict() now
also emits the audience-neutral field names from the convergence plan's
shared trace vocabulary (audience, query_analysis, patient_snapshot, pto,
corpora, raw_candidates, ranked_candidates, evidence_packet, draft,
grounding_validation, claim_validation, final_answer) so a patient trace
and a future physician trace are directly comparable in the same
observability tool, per the plan's own framing ("that enables you to
compare patient vs physician ... using the same observability
vocabulary"). Every one of these is either a genuinely new field
(audience, query_analysis, pto, claim_validation, draft — none of which
anything sets yet, matching how every other Sprint A shared-contract
piece landed unwired) or an ALIAS of an existing field in to_dict()'s
output (corpora=routed_collections, raw_candidates=candidates,
ranked_candidates=ranked, final_answer=answer,
grounding_validation=grounding, patient_snapshot=patient_snapshot_id or
patient_profile_id) — every pre-existing key stays exactly as it was, so
no current reader of a trace dict breaks.

audience defaults to "patient" — every trace built by this codebase
today is a patient trace; nothing constructs one with audience=
"physician" yet (that's Sprint C's job). patient_profile_id stays a
required field on TraceBuilder and save()'s target table still requires
one — a physician trace for a free-text question with no linked patient
is a real future gap this comment flags but does not solve; that's for
whoever wires the physician orchestrator (Sprint C item 20) to resolve,
since it may need its own nullable-patient trace path rather than a
change to this shared dataclass.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap on how much evidence text and how many candidates get persisted per
# trace — a trace is for diagnosing routing/ranking/grounding behavior,
# not for archiving the full corpus text a second time (that's what
# evidence_chunk_registry + Qdrant already do, addressable by point_id).
_MAX_CANDIDATES_RECORDED = 20
_MAX_EVIDENCE_TEXT_CHARS = 600


def _trim_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        # Provenance (2026-08-12 beta audit items 6/7) — the same fields
        # multi_corpus_retriever.py now attaches to every candidate and
        # evidence_packet_builder.py carries into the packet; recorded
        # here too so a trace alone is enough to audit which exact
        # Qdrant point/document/version a candidate came from, without
        # needing to cross-reference a separate log line.
        "qdrant_point_id": c.get("qdrant_point_id"),
        "doc_id": c.get("doc_id"),
        "version_id": c.get("version_id"),
        "collection": c.get("collection"),
        "title": c.get("title"),
        "section_title": c.get("section_title"),
        "chunk_index": c.get("chunk_index"),
        "source_key": c.get("source_key"),
        "source_name": c.get("source_name"),
        "url": c.get("url"),
        "authority_class": c.get("authority_class"),
        "semantic_relevance": c.get("semantic_relevance", c.get("semantic_score")),
        "clinical_applicability": c.get("clinical_applicability"),
        "source_authority": c.get("source_authority"),
        "applicability_score": c.get("applicability_score"),
        "components": c.get("components"),  # itemized scorer components, when present
        "incompatibility_reasons": c.get("incompatibility_reasons"),
        "text_preview": (c.get("text") or "")[:_MAX_EVIDENCE_TEXT_CHARS],
    }


@dataclass
class TraceBuilder:
    patient_profile_id: str
    question: str
    # Shared-schema fields (Sprint A item 6) — see module docstring.
    audience: str = "patient"
    query_analysis: Dict[str, Any] = field(default_factory=dict)
    patient_snapshot_id: Optional[str] = None
    pto: Dict[str, Any] = field(default_factory=dict)
    claim_validation: Dict[str, Any] = field(default_factory=dict)
    draft: str = ""
    intent: Optional[str] = None
    ontology_hits: Dict[str, Any] = field(default_factory=dict)
    routed_collections: List[str] = field(default_factory=list)
    selected_context: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    ranked: List[Dict[str, Any]] = field(default_factory=list)
    packet_summary: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    grounding: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def set_audience(self, audience: str) -> None:
        self.audience = audience

    def set_query_analysis(self, query_analysis: Dict[str, Any]) -> None:
        self.query_analysis = query_analysis or {}

    def set_patient_snapshot_id(self, patient_snapshot_id: Optional[str]) -> None:
        self.patient_snapshot_id = patient_snapshot_id

    def set_pto(self, pto: Dict[str, Any]) -> None:
        self.pto = pto or {}

    def set_draft(self, draft: str) -> None:
        """The FIRST generated answer, before any grounding retry/claim-
        repair rewrote it. set_answer() continues to record the final
        one — call both when a repair happened so the trace shows what
        changed, or just set_answer() alone (as every call site does
        today) when there was only ever one draft."""
        self.draft = draft or ""

    def set_claim_validation(self, claim_validation: Dict[str, Any]) -> None:
        self.claim_validation = claim_validation or {}

    def set_intent(self, intent: str, ontology_hits: Optional[Dict[str, Any]] = None) -> None:
        self.intent = intent
        self.ontology_hits = ontology_hits or {}

    def set_routing(self, collections: List[str]) -> None:
        self.routed_collections = list(collections)

    def set_context(self, context_summary: Dict[str, Any]) -> None:
        self.selected_context = context_summary or {}

    def set_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        self.candidates = [_trim_candidate(c) for c in candidates[:_MAX_CANDIDATES_RECORDED]]

    def set_ranked(self, ranked: List[Dict[str, Any]]) -> None:
        self.ranked = [_trim_candidate(c) for c in ranked[:_MAX_CANDIDATES_RECORDED]]

    def set_packet(self, packet: Dict[str, Any]) -> None:
        self.packet_summary = {
            "evidence_count": len(packet.get("evidence") or []),
            "patient_context": packet.get("patient_context"),
            "safety": packet.get("safety"),
            "evidence": [
                {
                    "n": i + 1,
                    "source": e.get("source"),
                    "qdrant_point_id": e.get("qdrant_point_id"),
                    "document_id": e.get("document_id"),
                    "version_id": e.get("version_id"),
                    "section": e.get("section"),
                    "url": e.get("url"),
                    "title": e.get("title"),
                    "role": e.get("role"),
                    "authority": e.get("authority"),
                    "semantic_score": e.get("semantic_score"),
                    "applicability_score": e.get("applicability_score"),
                    "score_components": e.get("score_components"),
                    "incompatibility_reasons": e.get("incompatibility_reasons"),
                    "citation": e.get("citation"),
                    "text_preview": (e.get("text") or "")[:_MAX_EVIDENCE_TEXT_CHARS],
                }
                for i, e in enumerate(packet.get("evidence") or [])
            ],
        }

    def set_answer(self, answer: str) -> None:
        self.answer = answer or ""

    def set_grounding(self, grounding: Dict[str, Any]) -> None:
        self.grounding = grounding or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "ontology_hits": self.ontology_hits,
            "routed_collections": self.routed_collections,
            "selected_context": self.selected_context,
            "candidate_count": len(self.candidates),
            "candidates": self.candidates,
            "ranked": self.ranked,
            "evidence_packet": self.packet_summary,
            "answer": self.answer,
            "grounding_validation": self.grounding,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            # Shared schema aliases (Sprint A item 6) — see module
            # docstring. Every key above is unchanged; these are
            # additions, several of them pointing at the same underlying
            # value under the convergence plan's audience-neutral name.
            "audience": self.audience,
            "query_analysis": self.query_analysis,
            "patient_snapshot": self.patient_snapshot_id or self.patient_profile_id,
            "pto": self.pto,
            "corpora": self.routed_collections,
            "raw_candidates": self.candidates,
            "ranked_candidates": self.ranked,
            "draft": self.draft or self.answer,
            "claim_validation": self.claim_validation,
            "final_answer": self.answer,
        }

    async def save(self) -> Optional[str]:
        """Persists the trace and returns its id, or None on failure.
        Never raises — a trace-write failure must not cost the patient
        their answer, matching every other best-effort write in this
        codebase (see patient_state_service, patient_link_service)."""
        try:
            from src.api.services.patient_db import get_patient_db
            db = get_patient_db()
            await db.ensure_schema()
            pool = await db.get_pool()
            trace_id = str(uuid.uuid4())
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO query_debug_traces (id, patient_profile_id, trace)
                    VALUES ($1, $2, $3::jsonb)
                    """,
                    trace_id, self.patient_profile_id, json.dumps(self.to_dict(), default=str),
                )
            # Only the id is ever logged — see module docstring.
            logger.info("[RetrievalTrace] saved trace_id=%s intent=%s", trace_id, self.intent)
            return trace_id
        except Exception:
            logger.warning("[RetrievalTrace] failed to save trace (answer still returned)", exc_info=True)
            return None


async def get_trace(trace_id: str, patient_profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch one trace by id. When patient_profile_id is given, scopes
    the lookup to that profile — used by the patient-facing path (if any
    ever needs it). The admin debug route passes None deliberately, since
    an operator debugging a report needs to look up a trace without
    already knowing which patient it belongs to."""
    from src.api.services.patient_db import get_patient_db
    db = get_patient_db()
    await db.ensure_schema()
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        if patient_profile_id:
            row = await conn.fetchrow(
                "SELECT * FROM query_debug_traces WHERE id = $1 AND patient_profile_id = $2",
                trace_id, patient_profile_id,
            )
        else:
            row = await conn.fetchrow("SELECT * FROM query_debug_traces WHERE id = $1", trace_id)
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    d["patient_profile_id"] = str(d["patient_profile_id"])
    d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
    if isinstance(d.get("trace"), str):
        try:
            d["trace"] = json.loads(d["trace"])
        except (TypeError, ValueError):
            pass
    return d


async def list_traces_for_profile(patient_profile_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    from src.api.services.patient_db import get_patient_db
    db = get_patient_db()
    await db.ensure_schema()
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, created_at, trace->>'intent' AS intent, trace->>'question' AS question
              FROM query_debug_traces
             WHERE patient_profile_id = $1
             ORDER BY created_at DESC
             LIMIT $2
            """,
            patient_profile_id, limit,
        )
    return [
        {"id": str(r["id"]), "created_at": r["created_at"].isoformat() if r["created_at"] else None,
         "intent": r["intent"], "question": r["question"]}
        for r in rows
    ]
