"""Symptom observation service (Phase 1 finalization).

The patient_profile-keyed symptom table Phase 1 calls for — preserves the
patient's own words, normalizes to a canonical symptom term (best-effort,
see clinical_normalization.py), tracks severity/onset/resolution/status/
frequency, and an optional "possibly related" treatment episode link that
is deliberately named to avoid implying confirmed causality.

Distinct from the legacy patient_symptom_entries table
(patient_portal/symptom_service.py), which stays exactly as it is and
keeps serving the existing /portal/symptoms endpoints and physician-chart
symptom sharing. patient_state_service prefers this table for new data
and falls back to the legacy one when a profile has no observations here
yet — see that module's symptom-loading section.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event
from src.api.services.patient.clinical_normalization import normalize_symptom


class SymptomObservationService:
    async def add_observation(
        self,
        patient_profile_id: str,
        raw_text: str,
        severity: Optional[int] = None,
        onset_date: Optional[str] = None,
        frequency: Optional[str] = None,
        possibly_related_treatment_episode_id: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        canonical = normalize_symptom(raw_text).canonical

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        obs_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO symptom_observations
                        (id, patient_profile_id, raw_text, canonical_symptom, severity,
                         onset_date, status, frequency, possibly_related_treatment_episode_id,
                         source_type, source_document_id, verification_status)
                    VALUES ($1,$2,$3,$4,$5,$6,'active',$7,$8,$9,$10,$11)
                    RETURNING *
                    """,
                    obs_id, patient_profile_id, raw_text, canonical, severity,
                    onset_date, frequency, possibly_related_treatment_episode_id,
                    source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "symptom_logged",
                    {"raw_text": raw_text, "canonical_symptom": canonical, "severity": severity},
                    created_by=created_by, event_date=onset_date, source=source_type,
                )
        from src.api.services.patient.patient_state_service import invalidate_patient_state
        await invalidate_patient_state(patient_profile_id)
        return row_to_dict(row)

    async def resolve_observation(
        self, observation_id: str, patient_profile_id: str,
        resolved_date: Optional[str] = None, created_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        from datetime import date as _date

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE symptom_observations
                       SET status = 'resolved', resolved_date = COALESCE($3, CURRENT_DATE)
                     WHERE id = $1 AND patient_profile_id = $2
                    RETURNING *
                    """,
                    observation_id, patient_profile_id, resolved_date,
                )
                if row:
                    await append_profile_timeline_event(
                        conn, patient_profile_id, "symptom_resolved",
                        {"canonical_symptom": row["canonical_symptom"]},
                        created_by=created_by,
                        event_date=resolved_date or _date.today().isoformat(),
                    )
        if row:
            from src.api.services.patient.patient_state_service import invalidate_patient_state
            await invalidate_patient_state(patient_profile_id)
        return row_to_dict(row) if row else None

    async def list_observations(
        self, patient_profile_id: str, active_only: bool = False, limit: int = 100
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        query = "SELECT * FROM symptom_observations WHERE patient_profile_id = $1"
        if active_only:
            query += " AND status = 'active'"
        query += " ORDER BY onset_date DESC NULLS LAST, created_at DESC LIMIT $2"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, patient_profile_id, limit)
        return [row_to_dict(r) for r in rows]


_service: Optional[SymptomObservationService] = None


def get_symptom_observation_service() -> SymptomObservationService:
    global _service
    if _service is None:
        _service = SymptomObservationService()
    return _service
