"""
Patient Care Team Service (Phase 0 / Phase 6)

Manages patient_care_team_links: the many-clinicians-per-patient
relationship that replaces the single physician_id ownership column on
the legacy `patients` table. A patient_profile can have zero, one, or
several active links (medical oncologist, radiation oncologist, surgeon,
PCP, ...), each optionally tied to a physician-owned chart row
(legacy_patient_record_id) via the pre-existing invite/request flow in
patient_link_service.py.

patient_link_service.py is left doing exactly what it did before — it
still links a user_id to a physician-owned `patients` row. What changes
is that claim_invite/approve_link_request now also call
sync_legacy_link() here, so the same event additionally produces (or
updates) a patient_profile and a care-team link. A patient linked before
this file existed is backfilled the first time sync_legacy_link runs for
them (e.g. next time they open the portal), not retroactively.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient.patient_profile_service import get_patient_profile_service

VALID_ROLES = {
    "oncologist", "radiation_oncologist", "surgeon", "primary_care",
    "nutritionist", "palliative_care", "other",
}


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            d[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


class PatientCareTeamService:
    async def add_member(
        self,
        patient_profile_id: str,
        physician_id: str,
        role: str = "oncologist",
        is_primary: bool = False,
        legacy_patient_record_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add or reactivate a care-team link. Reconnecting the same
        clinician updates the existing active row (role, legacy record)
        rather than creating a duplicate — see the partial unique index
        on (patient_profile_id, physician_id) WHERE status='active'."""
        if role not in VALID_ROLES:
            role = "other"

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        link_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO patient_care_team_links
                    (id, patient_profile_id, physician_id, role,
                     legacy_patient_record_id, is_primary, status)
                VALUES ($1, $2, $3, $4, $5, $6, 'active')
                ON CONFLICT (patient_profile_id, physician_id) WHERE status = 'active'
                DO UPDATE SET
                    role = EXCLUDED.role,
                    legacy_patient_record_id = COALESCE(
                        EXCLUDED.legacy_patient_record_id,
                        patient_care_team_links.legacy_patient_record_id
                    ),
                    is_primary = EXCLUDED.is_primary OR patient_care_team_links.is_primary,
                    updated_at = now()
                RETURNING *
                """,
                link_id, patient_profile_id, physician_id, role,
                legacy_patient_record_id, is_primary,
            )
        return _row_to_dict(row)

    async def list_care_team(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_care_team_links
                WHERE patient_profile_id = $1 AND status = 'active'
                ORDER BY is_primary DESC, created_at
                """,
                patient_profile_id,
            )
        return [_row_to_dict(r) for r in rows]

    async def revoke_member(self, patient_profile_id: str, physician_id: str) -> bool:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE patient_care_team_links
                   SET status = 'revoked', updated_at = now()
                 WHERE patient_profile_id = $1 AND physician_id = $2 AND status = 'active'
                """,
                patient_profile_id, physician_id,
            )
        return not result.endswith("0")

    async def sync_legacy_link(
        self,
        patient_user_id: str,
        physician_id: str,
        legacy_patient_record_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        date_of_birth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bridge: called after patient_link_service establishes a legacy
        (user_id <-> physician-owned patients row) link, so the same event
        also produces a patient_profile + care-team link. Never raises —
        the legacy link is the one that matters for existing features
        (chat known_facts, escalations); this is additive enrichment."""
        profile = await get_patient_profile_service().ensure_profile(
            user_id=patient_user_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
        )
        link = await self.add_member(
            patient_profile_id=profile["id"],
            physician_id=physician_id,
            role="oncologist",
            is_primary=True,
            legacy_patient_record_id=legacy_patient_record_id,
        )
        return {"profile": profile, "link": link}


_service: Optional[PatientCareTeamService] = None


def get_patient_care_team_service() -> PatientCareTeamService:
    global _service
    if _service is None:
        _service = PatientCareTeamService()
    return _service
