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

Verified physician authorization (2026-08-12 convergence Sprint C item
14): `status` ('active'/'revoked') has always governed this row's own
lifecycle, but nothing distinguished a patient SELF-REPORTING a
physician_id (see patient_records.add_care_team_member()'s own
docstring: "for a care-team member who has no chart on their end at
all" — literally any UUID the patient types in) from a physician's OWN
account actually confirming the relationship. Both created an
identical-looking 'active' row. `link_status` ('invited' | 'verified')
is the new, independent axis that closes that gap:
authorize_physician_patient_access() — the gate any future physician
route MUST call before reading a patient's canonical state (Sprint C
item 20/21) — requires BOTH status='active' AND link_status='verified'
AND the CALLING physician's own authenticated id to equal
verified_physician_user_id, not just any physician_id string a patient
happened to enter. The existing legacy invite/claim flow
(patient_link_service.py, bridged here via sync_legacy_link()) already
performs real identity confirmation on both sides, so links created
through that path are marked verified automatically — see
sync_legacy_link()'s call to add_member() below. Only the direct
patient-self-report path (add_member() called with its defaults) starts
at 'invited' and stays unauthorized until verify_link() is called by the
named physician's own account.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient.patient_profile_service import get_patient_profile_service

VALID_ROLES = {
    "oncologist", "radiation_oncologist", "surgeon", "primary_care",
    "nutritionist", "palliative_care", "other",
}

# link_status values -- see module docstring. No "pending" state is
# actually produced by this service today (nothing here implements a
# two-step "physician was notified, awaiting their response" flow yet);
# it's included in the vocabulary because the convergence plan names it
# explicitly and a future verification-request flow may need it, but
# every link currently starts at INVITED and moves directly to VERIFIED
# (or stays INVITED forever if never confirmed).
LINK_STATUS_INVITED = "invited"
LINK_STATUS_PENDING = "pending"
LINK_STATUS_VERIFIED = "verified"
# No separate link_status "revoked" value: this table's pre-existing
# `status` column already owns that lifecycle transition (see
# revoke_member() below) -- duplicating it onto link_status would create
# two ways to express "this link no longer applies" that could disagree
# with each other.


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
        *,
        link_status: str = LINK_STATUS_INVITED,
        verified_physician_user_id: Optional[str] = None,
        granted_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add or reactivate a care-team link. Reconnecting the same
        clinician updates the existing active row (role, legacy record)
        rather than creating a duplicate — see the partial unique index
        on (patient_profile_id, physician_id) WHERE status='active'.

        Defaults to an UNVERIFIED (link_status='invited') link — the
        caller entering a physician_id is not, by itself, proof that
        physician's own account has confirmed anything (see module
        docstring). sync_legacy_link() below is the one caller that
        passes link_status=LINK_STATUS_VERIFIED, since that path already
        goes through real identity confirmation on both sides.

        Reactivating an already-verified link never downgrades it back
        to invited — see the ON CONFLICT clause's CASE — and verification
        fields are only ever set, never cleared, by a call that doesn't
        pass them."""
        if role not in VALID_ROLES:
            role = "other"
        if link_status not in (LINK_STATUS_INVITED, LINK_STATUS_PENDING, LINK_STATUS_VERIFIED):
            link_status = LINK_STATUS_INVITED

        verified_at = datetime.now(timezone.utc) if link_status == LINK_STATUS_VERIFIED else None
        if granted_by is None:
            granted_by = "legacy_invite_flow" if link_status == LINK_STATUS_VERIFIED else "patient_self_report"

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        link_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO patient_care_team_links
                    (id, patient_profile_id, physician_id, role,
                     legacy_patient_record_id, is_primary, status,
                     link_status, verified_physician_user_id, verified_at, granted_by)
                VALUES ($1, $2, $3, $4, $5, $6, 'active', $7, $8, $9, $10)
                ON CONFLICT (patient_profile_id, physician_id) WHERE status = 'active'
                DO UPDATE SET
                    role = EXCLUDED.role,
                    legacy_patient_record_id = COALESCE(
                        EXCLUDED.legacy_patient_record_id,
                        patient_care_team_links.legacy_patient_record_id
                    ),
                    is_primary = EXCLUDED.is_primary OR patient_care_team_links.is_primary,
                    -- Never downgrade an already-verified link back to
                    -- invited/pending on reactivation; only "upgrade".
                    link_status = CASE
                        WHEN EXCLUDED.link_status = 'verified' THEN 'verified'
                        ELSE patient_care_team_links.link_status
                    END,
                    verified_physician_user_id = COALESCE(
                        EXCLUDED.verified_physician_user_id,
                        patient_care_team_links.verified_physician_user_id
                    ),
                    verified_at = COALESCE(EXCLUDED.verified_at, patient_care_team_links.verified_at),
                    granted_by = COALESCE(EXCLUDED.granted_by, patient_care_team_links.granted_by),
                    updated_at = now()
                RETURNING *
                """,
                link_id, patient_profile_id, physician_id, role,
                legacy_patient_record_id, is_primary,
                link_status, verified_physician_user_id, verified_at, granted_by,
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
            # The legacy invite/claim flow already confirms identity on
            # both sides before this bridge ever runs -- see module
            # docstring -- so this link is verified immediately rather
            # than starting at 'invited' like a direct patient
            # self-report.
            link_status=LINK_STATUS_VERIFIED,
            verified_physician_user_id=physician_id,
            granted_by="legacy_invite_flow",
        )
        return {"profile": profile, "link": link}

    async def verify_link(
        self, patient_profile_id: str, physician_id: str, verifying_physician_user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Promotes a self-reported ('invited') link to 'verified' —
        called from the physician's OWN authenticated session confirming
        this patient relationship (the future physician-facing
        confirmation flow — Sprint C item 21 — is what will actually
        expose this; nothing calls it yet). Only succeeds when
        verifying_physician_user_id matches the physician_id already on
        the link: a physician can only verify a link claiming to be
        THEM, never someone else's. Returns None (no row updated) if no
        matching active link exists or the ids don't match — this is not
        an error, just "nothing to verify"."""
        if verifying_physician_user_id != physician_id:
            return None
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE patient_care_team_links
                   SET link_status = 'verified',
                       verified_physician_user_id = $3,
                       verified_at = now(),
                       granted_by = COALESCE(granted_by, 'physician_verified'),
                       updated_at = now()
                 WHERE patient_profile_id = $1 AND physician_id = $2 AND status = 'active'
             RETURNING *
                """,
                patient_profile_id, physician_id, verifying_physician_user_id,
            )
        return _row_to_dict(row) if row else None


_service: Optional[PatientCareTeamService] = None


def get_patient_care_team_service() -> PatientCareTeamService:
    global _service
    if _service is None:
        _service = PatientCareTeamService()
    return _service


async def authorize_physician_patient_access(physician_user_id: str, patient_profile_id: str) -> bool:
    """The gate (2026-08-12 convergence Sprint C item 14): any future
    physician route that reads a patient's canonical state MUST call
    this before doing so. Returns True only when a
    patient_care_team_links row exists that is simultaneously:
      - status = 'active' (not revoked)
      - link_status = 'verified' (not just a patient's self-report)
      - verified_physician_user_id = physician_user_id -- the CALLING
        physician's own authenticated identity, not merely "some
        physician_id string matches" (a patient entering someone else's
        UUID must never authorize that someone else).

    A patient with no patient_profile row, a physician with no link at
    all, a revoked link, or an invited-but-never-verified link all
    correctly return False. Never raises on a missing row -- absence is
    the normal "not authorized" case, not an error; only a genuine DB
    failure propagates, matching this codebase's existing convention
    (get_source/get_document/etc.) that read-only lookups return None/
    False, not swallow a real connectivity failure into a false
    "authorized"."""
    db = get_patient_db()
    await db.ensure_schema()
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1 FROM patient_care_team_links
             WHERE patient_profile_id = $1
               AND verified_physician_user_id = $2
               AND status = 'active'
               AND link_status = 'verified'
             LIMIT 1
            """,
            patient_profile_id, physician_user_id,
        )
    return row is not None
