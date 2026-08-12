"""
Tests for deterministic patient-state freshness via revision counters
(2026-08-12 convergence Sprint B item 7): before this, get_context()
(patient_context_service.py) rebuilt a snapshot only when none existed
at all -- a rebuild attempt that failed once after a later canonical
write left every subsequent read trusting a stale snapshot FOREVER.
patient_profiles.state_revision (bumped by invalidate_patient_state()
before it attempts a rebuild) and patient_state_snapshots.source_revision
(stamped by build_state() onto whatever it persists) let get_context()
compare the two and retry the rebuild on every read where they don't
match, not just once.
"""

from __future__ import annotations

import pytest

from src.api.services.patient import patient_state_service as pss
from src.api.services.patient.patient_state_service import PatientStateService


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self):
        self.fetchval_calls = []
        self.execute_calls = []
        self._next_fetchval = None

    def set_next_fetchval(self, value):
        self._next_fetchval = value

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return self._next_fetchval

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))


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


class TestIncrementStateRevision:
    @pytest.mark.asyncio
    async def test_returns_the_new_revision_and_issues_an_atomic_update(self, monkeypatch):
        conn = _FakeConn()
        conn.set_next_fetchval(5)
        monkeypatch.setattr(pss, "get_patient_db", lambda: _FakeDB(_FakePool(conn)))

        result = await PatientStateService()._increment_state_revision("profile-1")

        assert result == 5
        assert len(conn.fetchval_calls) == 1
        query, args = conn.fetchval_calls[0]
        assert "state_revision = state_revision + 1" in " ".join(query.split())
        assert "RETURNING state_revision" in " ".join(query.split())
        assert args == ("profile-1",)


class TestInvalidatePatientStateBumpsRevisionBeforeRebuilding:
    @pytest.mark.asyncio
    async def test_increment_called_before_build_state(self, monkeypatch):
        order = []

        class FakeService:
            async def _increment_state_revision(self, patient_profile_id):
                order.append(("increment", patient_profile_id))
                return 1

            async def build_state(self, patient_profile_id):
                order.append(("build_state", patient_profile_id))

        monkeypatch.setattr(pss, "get_patient_state_service", lambda: FakeService())

        await pss.invalidate_patient_state("profile-1")

        assert order == [("increment", "profile-1"), ("build_state", "profile-1")]

    @pytest.mark.asyncio
    async def test_increment_failure_does_not_prevent_the_rebuild(self, monkeypatch):
        """The exact gap this feature closes: a failed revision bump
        must not stop the rebuild attempt itself -- both are best-effort,
        independently."""
        build_state_calls = []

        class FakeService:
            async def _increment_state_revision(self, patient_profile_id):
                raise RuntimeError("db unreachable")

            async def build_state(self, patient_profile_id):
                build_state_calls.append(patient_profile_id)

        monkeypatch.setattr(pss, "get_patient_state_service", lambda: FakeService())

        # Must not raise.
        await pss.invalidate_patient_state("profile-1")
        assert build_state_calls == ["profile-1"]

    @pytest.mark.asyncio
    async def test_build_state_failure_after_successful_increment_does_not_raise(self, monkeypatch):
        class FakeService:
            async def _increment_state_revision(self, patient_profile_id):
                return 1

            async def build_state(self, patient_profile_id):
                raise RuntimeError("db unreachable")

        monkeypatch.setattr(pss, "get_patient_state_service", lambda: FakeService())

        # Must not raise -- the revision was still bumped, so the NEXT
        # get_context() read will detect the mismatch and retry.
        await pss.invalidate_patient_state("profile-1")


class TestPersistSnapshotIncludesSourceRevision:
    @pytest.mark.asyncio
    async def test_source_revision_is_passed_to_the_insert(self, monkeypatch):
        conn = _FakeConn()
        monkeypatch.setattr(pss, "get_patient_db", lambda: _FakeDB(_FakePool(conn)))

        await PatientStateService()._persist_snapshot(
            "profile-1", {"some": "state"}, {"some": "features"}, source_revision=7,
        )

        assert len(conn.execute_calls) == 1
        query, args = conn.execute_calls[0]
        assert "source_revision" in " ".join(query.split())
        # args: (id, patient_profile_id, state_json, features_json, source_revision)
        assert args[-1] == 7

    @pytest.mark.asyncio
    async def test_source_revision_defaults_to_none(self, monkeypatch):
        conn = _FakeConn()
        monkeypatch.setattr(pss, "get_patient_db", lambda: _FakeDB(_FakePool(conn)))

        await PatientStateService()._persist_snapshot("profile-1", {}, {})

        _, args = conn.execute_calls[0]
        assert args[-1] is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
