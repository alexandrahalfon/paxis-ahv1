"""Comorbidity + allergy service (Phase 1).

Comorbidities feed treatment-eligibility inference downstream (e.g. CKD ->
cisplatin ineligible, see clinical_inference.py's inference map) so they
get their own table rather than being buried in free text.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


class ConditionsService:
    async def add_comorbidity(
        self,
        patient_profile_id: str,
        condition_name: str,
        status: str = "active",
        onset_date: Optional[str] = None,
        raw_text: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        cid = str(uuid.uuid4())
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_comorbidities
                        (id, patient_profile_id, condition_name, status, onset_date,
                         raw_text, source_type, source_document_id, verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    RETURNING *
                    """,
                    cid, patient_profile_id, condition_name, status, onset_date,
                    raw_text, source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "comorbidity_added",
                    {"condition_name": condition_name}, created_by=created_by,
                    event_date=onset_date, source=source_type,
                )
        return row_to_dict(row)

    async def list_comorbidities(
        self, patient_profile_id: str, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        query = "SELECT * FROM patient_comorbidities WHERE patient_profile_id = $1"
        if active_only:
            query += " AND status = 'active'"
        query += " ORDER BY created_at DESC"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, patient_profile_id)
        return [row_to_dict(r) for r in rows]

    async def add_allergy(
        self,
        patient_profile_id: str,
        allergen: str,
        reaction: Optional[str] = None,
        severity: Optional[str] = None,
        allergy_type: str = "allergy",
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if allergy_type not in ("allergy", "intolerance"):
            allergy_type = "allergy"

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        aid = str(uuid.uuid4())
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_allergies
                        (id, patient_profile_id, allergen, reaction, severity, allergy_type,
                         source_type, source_document_id, verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    RETURNING *
                    """,
                    aid, patient_profile_id, allergen, reaction, severity, allergy_type,
                    source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "allergy_added",
                    {"allergen": allergen, "severity": severity}, created_by=created_by,
                    source=source_type,
                )
        return row_to_dict(row)

    async def list_allergies(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_allergies
                 WHERE patient_profile_id = $1 AND status = 'active'
                 ORDER BY created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]


_service: Optional[ConditionsService] = None


def get_conditions_service() -> ConditionsService:
    global _service
    if _service is None:
        _service = ConditionsService()
    return _service
