"""Diagnosis + biomarker service (Phase 1), keyed by patient_profile_id.

Counterpart to the legacy patient_service.add_diagnosis/add_biomarker,
which stay keyed by the physician-owned patients.id and untouched.
patient_state_service merges both when a legacy care-team link exists.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


class DiagnosisService:
    async def add_diagnosis(
        self,
        patient_profile_id: str,
        cancer_site: Optional[str] = None,
        histology: Optional[str] = None,
        stage: Optional[str] = None,
        tnm_t: Optional[str] = None,
        tnm_n: Optional[str] = None,
        tnm_m: Optional[str] = None,
        diagnosis_date: Optional[str] = None,
        raw_text: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        diagnosis_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_diagnoses
                        (id, patient_profile_id, cancer_site, histology, stage,
                         tnm_t, tnm_n, tnm_m, diagnosis_date, raw_text,
                         source_type, source_document_id, verification_status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING *
                    """,
                    diagnosis_id, patient_profile_id, cancer_site, histology, stage,
                    tnm_t, tnm_n, tnm_m, diagnosis_date, raw_text,
                    source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "diagnosis_added",
                    {"cancer_site": cancer_site, "histology": histology, "stage": stage},
                    created_by=created_by, event_date=diagnosis_date, source=source_type,
                )
        return row_to_dict(row)

    async def get_latest_diagnosis(self, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM patient_diagnoses
                 WHERE patient_profile_id = $1
                 ORDER BY created_at DESC LIMIT 1
                """,
                patient_profile_id,
            )
        return row_to_dict(row) if row else None

    async def list_diagnoses(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_diagnoses
                 WHERE patient_profile_id = $1
                 ORDER BY created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def add_biomarker(
        self,
        patient_profile_id: str,
        biomarker_name: str,
        value: Optional[str] = None,
        measured_date: Optional[str] = None,
        raw_text: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        biomarker_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_biomarker_results
                        (id, patient_profile_id, biomarker_name, value, measured_date,
                         raw_text, source_type, source_document_id, verification_status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING *
                    """,
                    biomarker_id, patient_profile_id, biomarker_name, value,
                    measured_date, raw_text, source_type, source_document_id,
                    verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "biomarker_result",
                    {"biomarker_name": biomarker_name, "value": value},
                    created_by=created_by, event_date=measured_date, source=source_type,
                )
        return row_to_dict(row)

    async def list_biomarkers(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_biomarker_results
                 WHERE patient_profile_id = $1
                 ORDER BY created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]


_service: Optional[DiagnosisService] = None


def get_diagnosis_service() -> DiagnosisService:
    global _service
    if _service is None:
        _service = DiagnosisService()
    return _service
