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
        "doc_id": c.get("doc_id"),
        "collection": c.get("collection"),
        "title": c.get("title"),
        "section_title": c.get("section_title"),
        "source_key": c.get("source_key"),
        "authority_class": c.get("authority_class"),
        "semantic_relevance": c.get("semantic_relevance", c.get("semantic_score")),
        "clinical_applicability": c.get("clinical_applicability"),
        "source_authority": c.get("source_authority"),
        "applicability_score": c.get("applicability_score"),
        "components": c.get("components"),  # itemized scorer components, when present
        "text_preview": (c.get("text") or "")[:_MAX_EVIDENCE_TEXT_CHARS],
    }


@dataclass
class TraceBuilder:
    patient_profile_id: str
    question: str
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
                    "title": e.get("title"),
                    "role": e.get("role"),
                    "authority": e.get("authority"),
                    "applicability_score": e.get("applicability_score"),
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
