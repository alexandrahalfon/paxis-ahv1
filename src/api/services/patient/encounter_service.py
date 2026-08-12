"""Encounter (appointment) service (Phase 1) — appointments as a
first-class object, so "what changed at my last visit" has somewhere to
live and feeds the timeline, per the architecture review section 11.
"""

from __future__ import annotations

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
        source_document_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        import json

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
                         structured_changes, source_document_id)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    RETURNING *
                    """,
                    encounter_id, patient_profile_id, encounter_date, encounter_type,
                    provider_name, organization, patient_summary, clinician_note,
                    json.dumps(structured_changes or {}), source_document_id,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "encounter",
                    {
                        "encounter_type": encounter_type, "provider_name": provider_name,
                        "patient_summary": patient_summary,
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


_service: Optional[EncounterService] = None


def get_encounter_service() -> EncounterService:
    global _service
    if _service is None:
        _service = EncounterService()
    return _service
