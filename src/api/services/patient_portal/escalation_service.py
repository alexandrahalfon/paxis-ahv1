"""
Escalation queue: patient question -> physician inbox -> answer back.

The design point that makes this save a physician time rather than add to
their workload: they never receive a bare question. Each item carries the
draft answer Paxis would have given plus a short summary of what it knows
about the patient, so the physician approves, edits, or overrides instead
of composing from scratch.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db

logger = logging.getLogger(__name__)

OPEN = "open"
ANSWERED = "answered"
CLOSED = "closed"


def _summarize_facts(facts: Dict[str, Any]) -> str:
    """One-line context summary for the physician."""
    order = ("cancer_type", "histology", "stage", "biomarkers", "treatment")
    bits = []
    for k in order:
        v = facts.get(k)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        bits.append(f"{k.replace('_', ' ')}: {v}")
    return " | ".join(bits)


class EscalationService:
    async def create(
        self,
        patient_user_id: str,
        physician_id: str,
        question: str,
        ai_draft_answer: Optional[str] = None,
        patient_record_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        facts: Optional[Dict[str, Any]] = None,
        urgency: str = "routine",
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        esc_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO patient_escalations
                    (id, patient_user_id, patient_record_id, physician_id,
                     conversation_id, question, ai_draft_answer,
                     context_summary, urgency)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                esc_id, patient_user_id, patient_record_id, physician_id,
                conversation_id, question, ai_draft_answer,
                _summarize_facts(facts or {}), urgency,
            )
        return {"escalation_id": esc_id, "status": OPEN}

    async def list_for_physician(
        self, physician_id: str, status: str = OPEN, limit: int = 100
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_user_id, patient_record_id, conversation_id,
                       question, ai_draft_answer, context_summary, urgency,
                       status, created_at
                  FROM patient_escalations
                 WHERE physician_id = $1 AND status = $2
                 ORDER BY CASE urgency WHEN 'urgent' THEN 0
                                       WHEN 'soon'   THEN 1
                                       ELSE 2 END,
                          created_at DESC
                 LIMIT $3
                """,
                physician_id, status, limit,
            )
        return [
            {
                "id": str(r["id"]),
                "patient_user_id": str(r["patient_user_id"]),
                "patient_record_id": str(r["patient_record_id"]) if r["patient_record_id"] else None,
                "conversation_id": str(r["conversation_id"]) if r["conversation_id"] else None,
                "question": r["question"],
                "ai_draft_answer": r["ai_draft_answer"],
                "context_summary": r["context_summary"],
                "urgency": r["urgency"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    async def respond(
        self, escalation_id: str, physician_id: str, response_text: str
    ) -> Dict[str, Any]:
        """Physician answers. The reply is written back into the patient's
        conversation, attributed to the physician rather than to Paxis."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT id, physician_id, conversation_id, status
                      FROM patient_escalations WHERE id = $1
                    """,
                    escalation_id,
                )
                if not row:
                    raise ValueError("Escalation not found")
                if str(row["physician_id"]) != str(physician_id):
                    raise PermissionError("Not your escalation")
                if row["status"] != OPEN:
                    raise ValueError(f"Already {row['status']}")

                await conn.execute(
                    """
                    UPDATE patient_escalations
                       SET physician_response = $2, status = 'answered',
                           answered_at = now()
                     WHERE id = $1
                    """,
                    escalation_id, response_text,
                )
                if row["conversation_id"]:
                    await conn.execute(
                        """
                        INSERT INTO patient_messages
                            (id, conversation_id, role, content)
                        VALUES ($1, $2, 'physician', $3)
                        """,
                        str(uuid.uuid4()), row["conversation_id"], response_text,
                    )
        return {"escalation_id": escalation_id, "status": ANSWERED}

    async def list_for_patient(
        self, patient_user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, question, status, physician_response,
                       created_at, answered_at
                  FROM patient_escalations
                 WHERE patient_user_id = $1
                 ORDER BY created_at DESC LIMIT $2
                """,
                patient_user_id, limit,
            )
        return [
            {
                "id": str(r["id"]),
                "question": r["question"],
                "status": r["status"],
                "physician_response": r["physician_response"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "answered_at": r["answered_at"].isoformat() if r["answered_at"] else None,
            }
            for r in rows
        ]


_service: Optional[EscalationService] = None


def get_escalation_service() -> EscalationService:
    global _service
    if _service is None:
        _service = EscalationService()
    return _service
