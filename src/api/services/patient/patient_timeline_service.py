"""Patient Timeline Service (Phase 1) — read side.

The write side is append_profile_timeline_event() in _common.py, called by
every domain service (diagnosis, treatment, medication, lab, encounter,
document) after each mutation, exactly like the legacy
patient_service._append_timeline_event pattern. This module is the reader,
plus the vocabulary of event_type values the UI/timeline widget can expect:

    diagnosis_added, biomarker_result, treatment_started, cycle_received,
    treatment_stopped, medication_started, medication_stopped, lab_result,
    symptom_logged, encounter, document_uploaded, comorbidity_added,
    allergy_added, care_team_member_added
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict


class PatientTimelineService:
    async def get_timeline(
        self, patient_profile_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_profile_id, event_type, event_date,
                       recorded_at, payload, source, created_by
                  FROM patient_profile_timeline_events
                 WHERE patient_profile_id = $1
                 ORDER BY recorded_at DESC
                 LIMIT $2
                """,
                patient_profile_id, limit,
            )
        out = []
        for r in rows:
            d = row_to_dict(r)
            if isinstance(d.get("payload"), str):
                try:
                    d["payload"] = json.loads(d["payload"])
                except (TypeError, ValueError):
                    pass
            out.append(d)
        return out


_service: Optional[PatientTimelineService] = None


def get_patient_timeline_service() -> PatientTimelineService:
    global _service
    if _service is None:
        _service = PatientTimelineService()
    return _service
