"""
Tests for patient_context_service.get_context()'s deterministic
freshness check (2026-08-12 convergence Sprint B item 7): rebuild
whenever the latest snapshot's source_revision doesn't match the
profile's current state_revision, not just when no snapshot exists.
"""

from __future__ import annotations

import pytest

from src.api.services.evidence.patient_context_service import PatientContextService


def _patch_profile(monkeypatch, profile):
    async def get_by_user(self, user_id):
        return profile
    monkeypatch.setattr(
        "src.api.services.patient.patient_profile_service.PatientProfileService.get_by_user",
        get_by_user,
    )


def _patch_state_service(monkeypatch, snapshot, build_calls):
    class FakeStateService:
        async def get_latest_snapshot(self, patient_profile_id):
            return snapshot

        async def build_state(self, patient_profile_id):
            build_calls.append(patient_profile_id)
            return {"state": {"rebuilt": True}, "retrieval_features": {"rebuilt": True}}

    monkeypatch.setattr(
        "src.api.services.patient.patient_state_service.get_patient_state_service",
        lambda: FakeStateService(),
    )


class TestNoSnapshotAlwaysRebuilds:
    @pytest.mark.asyncio
    async def test_rebuilds_when_no_snapshot_exists(self, monkeypatch):
        _patch_profile(monkeypatch, {"id": "profile-1", "state_revision": 3})
        build_calls = []
        _patch_state_service(monkeypatch, None, build_calls)

        context = await PatientContextService().get_context("user-1")

        assert build_calls == ["profile-1"]
        assert context["state"] == {"rebuilt": True}


class TestRevisionMatchUsesCachedSnapshot:
    @pytest.mark.asyncio
    async def test_matching_revision_does_not_rebuild(self, monkeypatch):
        _patch_profile(monkeypatch, {"id": "profile-1", "state_revision": 3})
        build_calls = []
        _patch_state_service(
            monkeypatch,
            {"state": {"cached": True}, "retrieval_features": {}, "source_revision": 3},
            build_calls,
        )

        context = await PatientContextService().get_context("user-1")

        assert build_calls == []
        assert context["state"] == {"cached": True}


class TestRevisionMismatchRebuilds:
    @pytest.mark.asyncio
    async def test_stale_snapshot_revision_triggers_rebuild(self, monkeypatch):
        """The core fix: a snapshot that exists but is behind the
        profile's current revision -- e.g. because its own rebuild
        attempt failed after a later write -- must be rebuilt, not
        trusted just because it exists."""
        _patch_profile(monkeypatch, {"id": "profile-1", "state_revision": 5})
        build_calls = []
        _patch_state_service(
            monkeypatch,
            {"state": {"cached": True, "stale": True}, "retrieval_features": {}, "source_revision": 3},
            build_calls,
        )

        context = await PatientContextService().get_context("user-1")

        assert build_calls == ["profile-1"]
        assert context["state"] == {"rebuilt": True}

    @pytest.mark.asyncio
    async def test_pre_migration_snapshot_with_null_source_revision_triggers_one_rebuild(self, monkeypatch):
        """A snapshot written before this feature shipped has
        source_revision = NULL. Post-migration, state_revision defaults
        to 0 (NOT NULL DEFAULT 0), so NULL != 0 correctly forces exactly
        one rebuild to backfill a real source_revision -- never silently
        treated as fresh."""
        _patch_profile(monkeypatch, {"id": "profile-1", "state_revision": 0})
        build_calls = []
        _patch_state_service(
            monkeypatch,
            {"state": {"cached": True}, "retrieval_features": {}, "source_revision": None},
            build_calls,
        )

        context = await PatientContextService().get_context("user-1")

        assert build_calls == ["profile-1"]


class TestRepeatedFailedRebuildsKeepRetrying:
    @pytest.mark.asyncio
    async def test_every_read_retries_while_revisions_stay_mismatched(self, monkeypatch):
        """The exact scenario this feature exists to fix: a rebuild that
        keeps failing (or a snapshot that's persistently behind) must be
        retried on EVERY subsequent get_context() call, not given up on
        after the first attempt."""
        _patch_profile(monkeypatch, {"id": "profile-1", "state_revision": 9})
        build_calls = []
        _patch_state_service(
            monkeypatch,
            {"state": {}, "retrieval_features": {}, "source_revision": 2},
            build_calls,
        )

        service = PatientContextService()
        await service.get_context("user-1")
        await service.get_context("user-1")
        await service.get_context("user-1")

        assert build_calls == ["profile-1", "profile-1", "profile-1"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
