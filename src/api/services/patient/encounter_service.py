"""Encounter (appointment) service (Phase 1) — appointments as a
first-class object, so "what changed at my last visit" has somewhere to
live and feeds the timeline, per the architecture review section 11.

Phase 1 finalization: explicit newly_ordered_tests / next_steps /
questions_for_next_visit columns rather than everything folded into
structured_changes, so a visit recap and the "questions for next
appointment" list (architecture review Phase 24) can be built directly.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


class EncounterService:
    async def add_encounter(
        self,
        patient_profile_id: str,
        encounter_date: Optional[str] = None,
        encounter_type: Optional[str] = None,
        provider_name: Optional[str] = None,
        organization: Optional[str] = None,
        patient_summary: Optional[str] = None,
        clinician_note: Optional[str] = None,
        structured_changes: Optional[Dict[str, Any]] = None,
        newly_ordered_tests: Optional[List[str]] = None,
        next_steps: Optional[str] = None,
        questions_for_next_visit: Optional[List[str]] = None,
        source_document_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        encounter_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO encounters
                        (id, patient_profile_id, encounter_date, encounter_type,
                         provider_name, organization, patient_summary, clinician_note,
                         structured_changes, newly_ordered_tests, next_steps,
                         questions_for_next_visit, source_document_id)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb,$11,$12::jsonb,$13)
                    RETURNING *
                    """,
                    encounter_id, patient_profile_id, encounter_date, encounter_type,
                    provider_name, organization, patient_summary, clinician_note,
                    json.dumps(structured_changes or {}), json.dumps(newly_ordered_tests or []),
                    next_steps, json.dumps(questions_for_next_visit or []), source_document_id,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "encounter",
                    {
                        "encounter_type": encounter_type, "provider_name": provider_name,
                        "patient_summary": patient_summary, "next_steps": next_steps,
                    },
                    created_by=created_by, event_date=encounter_date,
                )
        return row_to_dict(row)

    async def list_encounters(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM encounters
                 WHERE patient_profile_id = $1
                 ORDER BY encounter_date DESC NULLS LAST, created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def upcoming_questions(self, patient_profile_id: str, limit: int = 20) -> List[str]:
        """Flattened questions_for_next_visit across recent encounters —
        what patient_tools_service.prepare_questions can seed from,
        alongside a patient's chat history."""
        encounters = await self.list_encounters(patient_profile_id)
        out: List[str] = []
        for e in encounters[:5]:
            for q in (e.get("questions_for_next_visit") or []):
                if q not in out:
                    out.append(q)
                if len(out) >= limit:
                    return out
        return out


_service: Optional[EncounterService] = None


def get_encounter_service() -> EncounterService:
    global _service
    if _service is None:
        _service = EncounterService()
    return _service
