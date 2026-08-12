"""Diagnosis + biomarker service (Phase 1), keyed by patient_profile_id.

Counterpart to the legacy patient_service.add_diagnosis/add_biomarker,
which stay keyed by the physician-owned patients.id and untouched.
patient_state_service merges both when a legacy care-team link exists.

Phase 1 finalization: supports multiple primaries / recurrence /
progression / remission as first-class (diagnosis_type, status,
related_diagnosis_id chaining a recurrence back to what it followed), and
auto-populates normalization columns via clinical_normalization.py at
write time — callers never need to normalize anything themselves.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event
from src.api.services.patient.clinical_normalization import (
    normalize_cancer_site, normalize_histology, normalize_stage_system,
    normalize_metastatic_sites, normalize_gene, biomarker_category_for,
)

VALID_DIAGNOSIS_TYPES = {"primary", "second_primary", "recurrence", "progression", "remission"}
VALID_DIAGNOSIS_STATUSES = {"active", "remission", "resolved", "historical"}


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
        diagnosis_type: str = "primary",
        status: str = "active",
        effective_date: Optional[str] = None,
        related_diagnosis_id: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if diagnosis_type not in VALID_DIAGNOSIS_TYPES:
            diagnosis_type = "primary"
        if status not in VALID_DIAGNOSIS_STATUSES:
            status = "active"

        site_norm = normalize_cancer_site(cancer_site or raw_text or "")
        histology_norm = normalize_histology(histology or raw_text or "")
        stage_system = normalize_stage_system(site_norm.canonical)
        met_sites = normalize_metastatic_sites(raw_text or "")

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
                         diagnosis_type, status, effective_date, canonical_cancer_type,
                         canonical_histology, stage_system, metastatic_sites,
                         related_diagnosis_id, source_type, source_document_id,
                         verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
                            $17::jsonb,$18,$19,$20,$21)
                    RETURNING *
                    """,
                    diagnosis_id, patient_profile_id, cancer_site, histology, stage,
                    tnm_t, tnm_n, tnm_m, diagnosis_date, raw_text,
                    diagnosis_type, status, effective_date or diagnosis_date,
                    site_norm.canonical, histology_norm.canonical, stage_system,
                    json.dumps(met_sites), related_diagnosis_id,
                    source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "diagnosis_added",
                    {
                        "cancer_site": cancer_site, "histology": histology, "stage": stage,
                        "diagnosis_type": diagnosis_type, "metastatic_sites": met_sites,
                    },
                    created_by=created_by, event_date=diagnosis_date, source=source_type,
                )
        from src.api.services.patient.patient_state_service import invalidate_patient_state
        await invalidate_patient_state(patient_profile_id)
        return row_to_dict(row)

    async def get_latest_diagnosis(self, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        """Single most-recent diagnosis row, for callers that only want
        one. Prefer get_active_diagnoses for anything that needs to
        reflect multiple primaries or a recurrence/progression history."""
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

    async def get_active_diagnoses(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        """Every diagnosis currently marked active/remission — supports a
        patient with multiple concurrent primaries, unlike
        get_latest_diagnosis which can only ever return one row."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_diagnoses
                 WHERE patient_profile_id = $1 AND status IN ('active', 'remission')
                 ORDER BY diagnosis_type = 'primary' DESC, created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

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
        specimen_date: Optional[str] = None,
        specimen_site: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        gene_norm = normalize_gene(biomarker_name)
        category = biomarker_category_for(biomarker_name)

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
                         raw_text, specimen_date, specimen_site, biomarker_category,
                         canonical_gene, source_type, source_document_id, verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    RETURNING *
                    """,
                    biomarker_id, patient_profile_id, biomarker_name, value,
                    measured_date, raw_text, specimen_date, specimen_site, category,
                    gene_norm.canonical, source_type, source_document_id,
                    verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "biomarker_result",
                    {"biomarker_name": biomarker_name, "value": value, "category": category},
                    created_by=created_by, event_date=measured_date, source=source_type,
                )
        from src.api.services.patient.patient_state_service import invalidate_patient_state
        await invalidate_patient_state(patient_profile_id)
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
