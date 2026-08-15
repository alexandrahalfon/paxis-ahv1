"""Shared helpers for the patient_profile-keyed service layer (Phase 1+).

Mirrors the row-serialization and append-only-timeline conventions
established in the legacy patient_service.py, so the two layers read the
same way even though they're keyed differently (patient_profile_id here
vs. the physician-owned patients.id there).
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from typing import Any, Dict, Optional


def row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            d[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


async def append_profile_timeline_event(
    conn,
    patient_profile_id: str,
    event_type: str,
    payload: Dict[str, Any],
    created_by: Optional[str] = None,
    event_date: Optional[str] = None,
    source: str = "manual",
) -> None:
    """Append using an already-acquired connection, ideally inside the
    same transaction as the write it's recording. Append-only: no update
    or delete path exists on purpose, matching patient_timeline_events."""
    await conn.execute(
        """
        INSERT INTO patient_profile_timeline_events
            (id, patient_profile_id, event_type, event_date, payload, source, created_by)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        str(uuid.uuid4()),
        patient_profile_id,
        event_type,
        event_date,
        json.dumps(payload, default=str),
        source,
        created_by,
    )
