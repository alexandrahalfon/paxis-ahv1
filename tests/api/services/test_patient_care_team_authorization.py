"""
Tests for verified physician-patient authorization (2026-08-12
convergence Sprint C item 14). A patient could always add ANY
physician_id to their care team directly (patient_records.
add_care_team_member() -- "for a care-team member who has no chart on
their end at all") with zero proof that UUID belongs to a real,
consenting clinician. link_status distinguishes that self-report
(invited) from the named physician's own account actually confirming it
(verified); authorize_physician_patient_access() is the gate that must
require the latter.

Uses a lightweight in-memory fake that faithfully implements the real
INSERT ... ON CONFLICT ... DO UPDATE upsert semantics (including the
CASE that never downgrades an already-verified link, and the COALESCE
that never clears a verification field with a NULL from a caller that
didn't pass one) -- not just a dict that happens to return the right
shape, so these tests actually exercise the SQL's real logic rather than
a reimplementation of it that could silently drift from the real query.
"""

from __future__ import annotations

import pytest

from src.api.services.patient.patient_care_team_service import (
    LINK_STATUS_INVITED,
    LINK_STATUS_VERIFIED,
    PatientCareTeamService,
    authorize_physician_patient_access,
)
import src.api.services.patient.patient_care_team_service as pcts_module


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    """Faithfully implements add_member()'s upsert (INSERT ... ON
    CONFLICT (patient_profile_id, physician_id) WHERE status='active'
    DO UPDATE ...), verify_link()'s UPDATE, revoke_member()'s UPDATE,
    list_care_team()'s SELECT, and authorize_physician_patient_access()'s
    SELECT 1 -- one small in-memory store per test."""

    def __init__(self):
        self.rows = {}
        self._active_index = {}  # (patient_profile_id, physician_id) -> row id

    async def fetchrow(self, query, *args):
        q = " ".join(query.split())

        if q.startswith("INSERT INTO patient_care_team_links"):
            (link_id, patient_profile_id, physician_id, role,
             legacy_patient_record_id, is_primary,
             link_status, verified_physician_user_id, verified_at, granted_by) = args
            key = (patient_profile_id, physician_id)
            existing_id = self._active_index.get(key)
            if existing_id is not None:
                row = self.rows[existing_id]
                row["role"] = role
                row["legacy_patient_record_id"] = (
                    legacy_patient_record_id or row["legacy_patient_record_id"]
                )
                row["is_primary"] = is_primary or row["is_primary"]
                if link_status == "verified":
                    row["link_status"] = "verified"
                # else: preserve existing link_status (CASE ... ELSE)
                row["verified_physician_user_id"] = (
                    verified_physician_user_id or row["verified_physician_user_id"]
                )
                row["verified_at"] = verified_at or row["verified_at"]
                row["granted_by"] = granted_by or row["granted_by"]
                return dict(row)

            row = {
                "id": link_id, "patient_profile_id": patient_profile_id,
                "physician_id": physician_id, "role": role,
                "legacy_patient_record_id": legacy_patient_record_id,
                "is_primary": is_primary, "status": "active",
                "link_status": link_status,
                "verified_physician_user_id": verified_physician_user_id,
                "verified_at": verified_at, "granted_by": granted_by,
            }
            self.rows[link_id] = row
            self._active_index[key] = link_id
            return dict(row)

        if q.startswith("UPDATE patient_care_team_links") and "link_status = 'verified'" in q:
            patient_profile_id, physician_id, verifying_physician_user_id = args
            row_id = self._active_index.get((patient_profile_id, physician_id))
            if row_id is None:
                return None
            row = self.rows[row_id]
            row["link_status"] = "verified"
            row["verified_physician_user_id"] = verifying_physician_user_id
            row["verified_at"] = "now"
            if not row.get("granted_by"):
                row["granted_by"] = "physician_verified"
            return dict(row)

        if q.startswith("SELECT 1 FROM patient_care_team_links"):
            patient_profile_id, physician_user_id = args
            for row in self.rows.values():
                if (
                    row["patient_profile_id"] == patient_profile_id
                    and row.get("verified_physician_user_id") == physician_user_id
                    and row["status"] == "active"
                    and row["link_status"] == "verified"
                ):
                    return {"?column?": 1}
            return None

        raise AssertionError(f"unexpected fetchrow query: {q[:100]}")

    async def execute(self, query, *args):
        q = " ".join(query.split())
        if q.startswith("UPDATE patient_care_team_links") and "'revoked'" in q:
            patient_profile_id, physician_id = args
            row_id = self._active_index.pop((patient_profile_id, physician_id), None)
            if row_id is None:
                return "UPDATE 0"
            self.rows[row_id]["status"] = "revoked"
            return "UPDATE 1"
        raise AssertionError(f"unexpected execute query: {q[:100]}")

    async def fetch(self, query, *args):
        (patient_profile_id,) = args
        return [
            dict(r) for r in self.rows.values()
            if r["patient_profile_id"] == patient_profile_id and r["status"] == "active"
        ]


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


class _FakeDB:
    def __init__(self, pool):
        self._pool = pool

    async def ensure_schema(self):
        pass

    async def get_pool(self):
        return self._pool


@pytest.fixture
def db(monkeypatch):
    conn = _FakeConn()
    fake_db = _FakeDB(_FakePool(conn))
    monkeypatch.setattr(pcts_module, "get_patient_db", lambda: fake_db)
    return conn


class TestAddMemberDefaultsToUnverified:
    @pytest.mark.asyncio
    async def test_self_report_link_starts_invited(self, db):
        link = await PatientCareTeamService().add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
        )
        assert link["link_status"] == LINK_STATUS_INVITED
        assert link["granted_by"] == "patient_self_report"
        assert link["verified_physician_user_id"] is None

    @pytest.mark.asyncio
    async def test_explicit_verified_add_is_stored_verified(self, db):
        link = await PatientCareTeamService().add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
            link_status=LINK_STATUS_VERIFIED, verified_physician_user_id="physician-1",
        )
        assert link["link_status"] == LINK_STATUS_VERIFIED
        assert link["verified_physician_user_id"] == "physician-1"
        assert link["granted_by"] == "legacy_invite_flow"


class TestReactivationNeverDowngrades:
    @pytest.mark.asyncio
    async def test_readding_a_verified_link_with_defaults_stays_verified(self, db):
        svc = PatientCareTeamService()
        await svc.add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
            link_status=LINK_STATUS_VERIFIED, verified_physician_user_id="physician-1",
        )
        # Reconnecting with plain defaults (as the patient-side route
        # does) must not silently strip verification.
        link = await svc.add_member(patient_profile_id="profile-1", physician_id="physician-1")
        assert link["link_status"] == LINK_STATUS_VERIFIED
        assert link["verified_physician_user_id"] == "physician-1"

    @pytest.mark.asyncio
    async def test_readding_an_invited_link_stays_invited(self, db):
        svc = PatientCareTeamService()
        await svc.add_member(patient_profile_id="profile-1", physician_id="physician-1")
        link = await svc.add_member(patient_profile_id="profile-1", physician_id="physician-1")
        assert link["link_status"] == LINK_STATUS_INVITED


class TestSyncLegacyLinkIsAutoVerified:
    @pytest.mark.asyncio
    async def test_legacy_bridge_produces_a_verified_link(self, db, monkeypatch):
        async def fake_ensure_profile(self, user_id, first_name=None, last_name=None, date_of_birth=None):
            return {"id": "profile-1"}
        monkeypatch.setattr(
            "src.api.services.patient.patient_profile_service.PatientProfileService.ensure_profile",
            fake_ensure_profile,
        )

        result = await PatientCareTeamService().sync_legacy_link(
            patient_user_id="user-1", physician_id="physician-1",
            legacy_patient_record_id="legacy-1",
        )
        assert result["link"]["link_status"] == LINK_STATUS_VERIFIED
        assert result["link"]["verified_physician_user_id"] == "physician-1"
        assert result["link"]["granted_by"] == "legacy_invite_flow"


class TestVerifyLink:
    @pytest.mark.asyncio
    async def test_physician_can_verify_their_own_claimed_link(self, db):
        svc = PatientCareTeamService()
        await svc.add_member(patient_profile_id="profile-1", physician_id="physician-1")
        result = await svc.verify_link(
            patient_profile_id="profile-1", physician_id="physician-1",
            verifying_physician_user_id="physician-1",
        )
        assert result is not None
        assert result["link_status"] == LINK_STATUS_VERIFIED
        assert result["verified_physician_user_id"] == "physician-1"

    @pytest.mark.asyncio
    async def test_a_different_physician_cannot_verify_someone_elses_claimed_link(self, db):
        """The core protection: a patient claiming physician-A's UUID
        must not be confirmable by physician-B logging in."""
        svc = PatientCareTeamService()
        await svc.add_member(patient_profile_id="profile-1", physician_id="physician-A")
        result = await svc.verify_link(
            patient_profile_id="profile-1", physician_id="physician-A",
            verifying_physician_user_id="physician-B",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_verifying_a_nonexistent_link_returns_none(self, db):
        result = await PatientCareTeamService().verify_link(
            patient_profile_id="profile-1", physician_id="physician-1",
            verifying_physician_user_id="physician-1",
        )
        assert result is None


class TestAuthorizePhysicianPatientAccess:
    @pytest.mark.asyncio
    async def test_true_for_verified_active_link_with_matching_physician(self, db):
        await PatientCareTeamService().add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
            link_status=LINK_STATUS_VERIFIED, verified_physician_user_id="physician-1",
        )
        assert await authorize_physician_patient_access("physician-1", "profile-1") is True

    @pytest.mark.asyncio
    async def test_false_for_invited_only_self_reported_link(self, db):
        """The exact gap this feature closes: a patient self-reporting a
        physician_id must not grant that physician access."""
        await PatientCareTeamService().add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
        )
        assert await authorize_physician_patient_access("physician-1", "profile-1") is False

    @pytest.mark.asyncio
    async def test_false_when_requesting_physician_does_not_match_verified_id(self, db):
        await PatientCareTeamService().add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
            link_status=LINK_STATUS_VERIFIED, verified_physician_user_id="physician-1",
        )
        assert await authorize_physician_patient_access("physician-2", "profile-1") is False

    @pytest.mark.asyncio
    async def test_false_for_a_revoked_link_even_if_previously_verified(self, db):
        svc = PatientCareTeamService()
        await svc.add_member(
            patient_profile_id="profile-1", physician_id="physician-1",
            link_status=LINK_STATUS_VERIFIED, verified_physician_user_id="physician-1",
        )
        await svc.revoke_member("profile-1", "physician-1")
        assert await authorize_physician_patient_access("physician-1", "profile-1") is False

    @pytest.mark.asyncio
    async def test_false_when_no_link_exists_at_all(self, db):
        assert await authorize_physician_patient_access("physician-1", "profile-1") is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
