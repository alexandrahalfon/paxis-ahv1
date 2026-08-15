"""
Patient Profile Service (Phase 0)

CRUD for patient_profiles — the consumer-owned anchor for a patient's
longitudinal record. One row per user_id, created automatically at
patient registration (see ensure_profile, called from
src/api/routes/auth.py register_patient) so an unlinked consumer can
start building a record before — or without ever — connecting a clinician.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, Optional

from src.api.services.patient_db import get_patient_db


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            d[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


class PatientProfileService:
    async def ensure_profile(
        self,
        user_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        date_of_birth: Optional[str] = None,
        sex: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get this user's profile, creating it if it doesn't exist yet.

        Idempotent by design: called both at registration (where it always
        creates) and defensively from anywhere else a profile is assumed
        to exist (where it's almost always a no-op) so a user who somehow
        reached the app without one — e.g. an account created before this
        table existed — is repaired transparently rather than 404ing.
        """
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM patient_profiles WHERE user_id = $1", user_id
            )
            if existing:
                return _row_to_dict(existing)

            profile_id = str(uuid.uuid4())
            row = await conn.fetchrow(
                """
                INSERT INTO patient_profiles
                    (id, user_id, first_name, last_name, date_of_birth, sex)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id) DO UPDATE SET user_id = EXCLUDED.user_id
                RETURNING *
                """,
                profile_id, user_id, first_name, last_name, date_of_birth, sex,
            )
        return _row_to_dict(row)

    async def get_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM patient_profiles WHERE user_id = $1", user_id
            )
        return _row_to_dict(row) if row else None

    async def get_by_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM patient_profiles WHERE id = $1", profile_id
            )
        return _row_to_dict(row) if row else None

    async def update_profile(self, user_id: str, **fields) -> Optional[Dict[str, Any]]:
        allowed = {"first_name", "last_name", "date_of_birth", "sex", "preferred_language", "timezone"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return await self.get_by_user(user_id)

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        set_clauses, params, idx = [], [], 1
        for k, v in updates.items():
            set_clauses.append(f"{k} = ${idx}")
            params.append(v)
            idx += 1
        set_clauses.append("updated_at = now()")

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE patient_profiles SET {", ".join(set_clauses)}
                WHERE user_id = ${idx}
                RETURNING *
                """,
                *params, user_id,
            )
        return _row_to_dict(row) if row else None


_service: Optional[PatientProfileService] = None


def get_patient_profile_service() -> PatientProfileService:
    global _service
    if _service is None:
        _service = PatientProfileService()
    return _service
