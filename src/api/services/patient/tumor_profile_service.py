"""Tumor profile service (Phase 1 finalization).

grade/size/receptor-status/molecular-subtype as their own entity, so a
re-biopsy or new stain updates the tumor profile without implying the
diagnosis itself changed. See tumor_profiles in patient_schema.py.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


class TumorProfileService:
    async def add_profile(
        self,
        patient_profile_id: str,
        diagnosis_id: Optional[str] = None,
        grade: Optional[str] = None,
        tumor_size_mm: Optional[float] = None,
        molecular_subtype: Optional[str] = None,
        receptor_status: Optional[Dict[str, str]] = None,
        specimen_date: Optional[str] = None,
        specimen_site: Optional[str] = None,
        raw_text: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        profile_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO tumor_profiles
                        (id, patient_profile_id, diagnosis_id, grade, tumor_size_mm,
                         molecular_subtype, receptor_status, specimen_date, specimen_site,
                         raw_text, source_type, source_document_id, verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13)
                    RETURNING *
                    """,
                    profile_id, patient_profile_id, diagnosis_id, grade, tumor_size_mm,
                    molecular_subtype, json.dumps(receptor_status or {}), specimen_date,
                    specimen_site, raw_text, source_type, source_document_id,
                    verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "tumor_profile_added",
                    {"grade": grade, "molecular_subtype": molecular_subtype,
                     "receptor_status": receptor_status or {}},
                    created_by=created_by, event_date=specimen_date, source=source_type,
                )
        return row_to_dict(row)

    async def list_profiles(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM tumor_profiles
                 WHERE patient_profile_id = $1
                 ORDER BY specimen_date DESC NULLS LAST, created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def get_latest(self, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        profiles = await self.list_profiles(patient_profile_id)
        return profiles[0] if profiles else None


_service: Optional[TumorProfileService] = None


def get_tumor_profile_service() -> TumorProfileService:
    global _service
    if _service is None:
        _service = TumorProfileService()
    return _service
