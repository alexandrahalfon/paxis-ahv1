"""
Patient <-> physician linking.

Two ways a patient account becomes connected to a clinical record:

1. **Invite (preferred).** The physician generates a single-use code from
   an existing patient record. The patient enters it at signup and the
   link is established immediately, because the physician initiated it.

2. **Request (fallback).** A patient signs up without a code and picks
   their physician from a list. That creates a *pending request*, not a
   link. The physician approves it and chooses which of their patient
   records it belongs to (or creates one).

The second path is deliberately not automatic. Letting anyone attach
themselves to any physician by selecting a name would let a stranger
assert a clinical relationship and push messages into that physician's
inbox. Selecting a name is a request; only the physician can grant it.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db

logger = logging.getLogger(__name__)


# Unambiguous alphabet: no 0/O, no 1/I/L. Patients read these off a phone
# screen or a printed sheet and type them by hand.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LENGTH = 8


def generate_invite_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


def _normalize_code(code: str) -> str:
    """Uppercase and strip separators so 'abcd-1234' matches 'ABCD1234'."""
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


class PatientLinkService:
    # ── Physician side ────────────────────────────────────────────────

    async def create_invite(self, patient_id: str, physician_id: str) -> Dict[str, Any]:
        """Generate (or regenerate) a single-use invite code for a record.

        Ownership-scoped: a physician can only invite for their own
        patients. Regenerating replaces any previous unused code, which
        is the desired behaviour if a patient lost the first one.
        """
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            owner = await conn.fetchval(
                "SELECT physician_id FROM patients WHERE id = $1", patient_id
            )
            if owner is None:
                raise ValueError("Patient not found")
            if str(owner) != str(physician_id):
                raise PermissionError("Not your patient")

            already_linked = await conn.fetchval(
                "SELECT user_id FROM patients WHERE id = $1", patient_id
            )
            if already_linked is not None:
                raise ValueError("This patient already has a linked account")

            # Retry on the (vanishingly unlikely) code collision.
            for _ in range(5):
                code = generate_invite_code()
                exists = await conn.fetchval(
                    "SELECT 1 FROM patients WHERE invite_code = $1", code
                )
                if not exists:
                    break
            else:
                raise RuntimeError("Could not allocate an invite code")

            await conn.execute(
                """
                UPDATE patients
                   SET invite_code = $2,
                       invite_created_at = now(),
                       link_status = 'invited',
                       updated_at = now()
                 WHERE id = $1
                """,
                patient_id, code,
            )
        return {"patient_id": patient_id, "invite_code": code}

    async def list_link_requests(
        self, physician_id: str, status: str = "pending"
    ) -> List[Dict[str, Any]]:
        """Pending connection requests for this physician to approve."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_user_id, patient_first_name, patient_last_name,
                       date_of_birth, note, status, created_at
                  FROM patient_link_requests
                 WHERE physician_id = $1 AND status = $2
                 ORDER BY created_at DESC
                """,
                physician_id, status,
            )
        return [
            {
                "id": str(r["id"]),
                "patient_user_id": str(r["patient_user_id"]),
                "first_name": r["patient_first_name"],
                "last_name": r["patient_last_name"],
                "date_of_birth": r["date_of_birth"].isoformat() if r["date_of_birth"] else None,
                "note": r["note"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    async def approve_link_request(
        self, request_id: str, physician_id: str, patient_record_id: str
    ) -> Dict[str, Any]:
        """Approve a request and bind the patient account to a record."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                req = await conn.fetchrow(
                    """
                    SELECT id, patient_user_id, physician_id, status
                      FROM patient_link_requests
                     WHERE id = $1
                    """,
                    request_id,
                )
                if not req:
                    raise ValueError("Request not found")
                if str(req["physician_id"]) != str(physician_id):
                    raise PermissionError("Not your request")
                if req["status"] != "pending":
                    raise ValueError(f"Request already {req['status']}")

                owner = await conn.fetchval(
                    "SELECT physician_id FROM patients WHERE id = $1", patient_record_id
                )
                if owner is None:
                    raise ValueError("Patient record not found")
                if str(owner) != str(physician_id):
                    raise PermissionError("Not your patient")

                taken = await conn.fetchval(
                    "SELECT 1 FROM patients WHERE id = $1 AND user_id IS NOT NULL",
                    patient_record_id,
                )
                if taken:
                    raise ValueError("That record is already linked to an account")

                await conn.execute(
                    """
                    UPDATE patients
                       SET user_id = $2, link_status = 'linked',
                           linked_at = now(), invite_code = NULL, updated_at = now()
                     WHERE id = $1
                    """,
                    patient_record_id, req["patient_user_id"],
                )
                await conn.execute(
                    """
                    UPDATE patient_link_requests
                       SET status = 'approved', resolved_at = now(),
                           resolved_by = $2, linked_patient_id = $3
                     WHERE id = $1
                    """,
                    request_id, physician_id, patient_record_id,
                )
                patient_user_id = str(req["patient_user_id"])

        # See claim_invite: same additive, non-fatal patient_profile /
        # care-team sync, outside the transaction.
        try:
            from src.api.services.patient.patient_care_team_service import (
                get_patient_care_team_service,
            )
            await get_patient_care_team_service().sync_legacy_link(
                patient_user_id=patient_user_id,
                physician_id=str(physician_id),
                legacy_patient_record_id=str(patient_record_id),
            )
        except Exception:
            logger.warning(
                "[PatientLink] patient_profile sync failed after approve_link_request "
                "for patient_user_id=%s (legacy link still succeeded)", patient_user_id,
                exc_info=True,
            )

        return {"request_id": request_id, "patient_id": patient_record_id, "status": "approved"}

    async def decline_link_request(self, request_id: str, physician_id: str) -> bool:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE patient_link_requests
                   SET status = 'declined', resolved_at = now(), resolved_by = $2
                 WHERE id = $1 AND physician_id = $2 AND status = 'pending'
                """,
                request_id, physician_id,
            )
        return result.endswith("1")

    # ── Patient side ──────────────────────────────────────────────────

    async def claim_invite(self, invite_code: str, patient_user_id: str) -> Dict[str, Any]:
        """Bind a patient account to the record the code was issued for."""
        code = _normalize_code(invite_code)
        if not code:
            raise ValueError("Invite code is required")

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT id, physician_id, user_id
                      FROM patients
                     WHERE invite_code = $1
                    """,
                    code,
                )
                if not row:
                    raise ValueError("That invite code isn't valid. Check with your care team.")
                if row["user_id"] is not None:
                    raise ValueError("That invite code has already been used.")

                existing = await conn.fetchval(
                    "SELECT id FROM patients WHERE user_id = $1", patient_user_id
                )
                if existing:
                    raise ValueError("This account is already connected to a care team.")

                await conn.execute(
                    """
                    UPDATE patients
                       SET user_id = $2, link_status = 'linked', linked_at = now(),
                           invite_code = NULL, updated_at = now()
                     WHERE id = $1
                    """,
                    row["id"], patient_user_id,
                )

        # Additive: also produces/updates a patient_profile + care-team
        # link (Phase 0) alongside the legacy link above. Outside the
        # transaction and non-fatal — the legacy link is the one existing
        # features depend on; this is enrichment, not a requirement for
        # claim_invite to succeed.
        try:
            from src.api.services.patient.patient_care_team_service import (
                get_patient_care_team_service,
            )
            await get_patient_care_team_service().sync_legacy_link(
                patient_user_id=patient_user_id,
                physician_id=str(row["physician_id"]),
                legacy_patient_record_id=str(row["id"]),
            )
        except Exception:
            logger.warning(
                "[PatientLink] patient_profile sync failed after claim_invite for "
                "patient_user_id=%s (legacy link still succeeded)", patient_user_id,
                exc_info=True,
            )

        return {
            "patient_id": str(row["id"]),
            "physician_id": str(row["physician_id"]),
            "status": "linked",
        }

    async def request_link(
        self,
        patient_user_id: str,
        physician_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        date_of_birth: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a pending connection request. Grants no access by itself."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            already = await conn.fetchval(
                "SELECT id FROM patients WHERE user_id = $1", patient_user_id
            )
            if already:
                raise ValueError("This account is already connected to a care team.")

            dup = await conn.fetchval(
                """
                SELECT id FROM patient_link_requests
                 WHERE patient_user_id = $1 AND physician_id = $2 AND status = 'pending'
                """,
                patient_user_id, physician_id,
            )
            if dup:
                return {"request_id": str(dup), "status": "pending", "duplicate": True}

            request_id = str(uuid.uuid4())
            dob = None
            if date_of_birth:
                from datetime import date
                try:
                    dob = date.fromisoformat(str(date_of_birth)[:10])
                except Exception:
                    dob = None

            await conn.execute(
                """
                INSERT INTO patient_link_requests
                    (id, patient_user_id, physician_id, patient_first_name,
                     patient_last_name, date_of_birth, note)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                request_id, patient_user_id, physician_id,
                first_name, last_name, dob, note,
            )
        return {"request_id": request_id, "status": "pending", "duplicate": False}

    async def get_linked_record(self, patient_user_id: str) -> Optional[Dict[str, Any]]:
        """The clinical record this patient account is linked to, if any."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, physician_id, first_name, last_name,
                       date_of_birth, sex, link_status, linked_at
                  FROM patients
                 WHERE user_id = $1
                """,
                patient_user_id,
            )
        if not row:
            return None
        return {
            "patient_id": str(row["id"]),
            "physician_id": str(row["physician_id"]),
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "date_of_birth": row["date_of_birth"].isoformat() if row["date_of_birth"] else None,
            "sex": row["sex"],
            "link_status": row["link_status"],
            "linked_at": row["linked_at"].isoformat() if row["linked_at"] else None,
        }

    async def pending_request_for(self, patient_user_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, physician_id, created_at
                  FROM patient_link_requests
                 WHERE patient_user_id = $1 AND status = 'pending'
                 ORDER BY created_at DESC LIMIT 1
                """,
                patient_user_id,
            )
        if not row:
            return None
        return {
            "request_id": str(row["id"]),
            "physician_id": str(row["physician_id"]),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }


_service: Optional[PatientLinkService] = None


def get_patient_link_service() -> PatientLinkService:
    global _service
    if _service is None:
        _service = PatientLinkService()
    return _service
