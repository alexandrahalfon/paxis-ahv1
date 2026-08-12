"""Medication exposure service (Phase 1) — non-cancer medications matter
for retrieval (interaction / eligibility questions) as much as the active
regimen, so they get their own table rather than living only inside a
treatment_episode. See medication_exposures in patient_db.py.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


class MedicationService:
    async def add_medication(
        self,
        patient_profile_id: str,
        generic_name: str,
        brand_name: Optional[str] = None,
        rxnorm_code: Optional[str] = None,
        dose: Optional[str] = None,
        route: Optional[str] = None,
        frequency: Optional[str] = None,
        indication: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: str = "active",
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        med_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO medication_exposures
                        (id, patient_profile_id, generic_name, brand_name, rxnorm_code,
                         dose, route, frequency, indication, start_date, end_date, status,
                         source_type, source_document_id, verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                    RETURNING *
                    """,
                    med_id, patient_profile_id, generic_name, brand_name, rxnorm_code,
                    dose, route, frequency, indication, start_date, end_date, status,
                    source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "medication_started",
                    {"generic_name": generic_name, "indication": indication},
                    created_by=created_by, event_date=start_date, source=source_type,
                )
        return row_to_dict(row)

    async def update_status(
        self, medication_id: str, patient_profile_id: str, status: str,
        end_date: Optional[str] = None, created_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE medication_exposures
                       SET status = $3, end_date = COALESCE($4, end_date)
                     WHERE id = $1 AND patient_profile_id = $2
                    RETURNING *
                    """,
                    medication_id, patient_profile_id, status, end_date,
                )
                if row and status == "stopped":
                    await append_profile_timeline_event(
                        conn, patient_profile_id, "medication_stopped",
                        {"generic_name": row["generic_name"]},
                        created_by=created_by, event_date=end_date,
                    )
        return row_to_dict(row) if row else None

    async def list_medications(
        self, patient_profile_id: str, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        query = "SELECT * FROM medication_exposures WHERE patient_profile_id = $1"
        if active_only:
            query += " AND status = 'active'"
        query += " ORDER BY start_date DESC NULLS LAST, created_at DESC"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, patient_profile_id)
        return [row_to_dict(r) for r in rows]


_service: Optional[MedicationService] = None


def get_medication_service() -> MedicationService:
    global _service
    if _service is None:
        _service = MedicationService()
    return _service
