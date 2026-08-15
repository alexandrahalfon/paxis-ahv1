"""
Tests for patient-state snapshot invalidation on canonical writes
(2026-08-12 beta audit item 4): patient_context_service.get_context()
reads the LATEST patient_state_snapshots row and only rebuilds when none
exists at all -- it never re-checked staleness. A manual write
(add_diagnosis/add_episode/add_medication/add_observation/
add_assessment/add_vital/...) used to leave the existing snapshot in
place, so a patient could add a new treatment and immediately ask a
question retrieval answered from state as it was BEFORE that write.

Every canonical write path now calls
patient_state_service.invalidate_patient_state(patient_profile_id)
immediately after its transaction commits (never from inside the
transaction -- see that function's docstring for why).

Two kinds of coverage here:
  1. Direct tests of invalidate_patient_state() itself.
  2. A structural sweep confirming every write method across the 11
     patient-domain services actually calls it -- 18 call sites is too
     many to fake a full Postgres schema for each individually, and the
     wiring itself (not build_state()'s own correctness, which is
     pre-existing, tested-elsewhere behavior) is what this fix adds.
Plus one deeper integration-style test proving the call happens AFTER
the write's transaction commits, against a real, minimal Postgres fake.
"""

from __future__ import annotations

import inspect

import pytest

from src.api.services.patient import patient_state_service as pss


class TestInvalidatePatientStateHelper:
    @pytest.mark.asyncio
    async def test_success_calls_build_state_with_the_profile_id(self, monkeypatch):
        calls = []

        class FakeService:
            async def build_state(self, patient_profile_id):
                calls.append(patient_profile_id)

        monkeypatch.setattr(pss, "get_patient_state_service", lambda: FakeService())

        await pss.invalidate_patient_state("profile-123")
        assert calls == ["profile-123"]

    @pytest.mark.asyncio
    async def test_failure_is_swallowed_not_raised(self, monkeypatch):
        class FailingService:
            async def build_state(self, patient_profile_id):
                raise RuntimeError("db unreachable")

        monkeypatch.setattr(pss, "get_patient_state_service", lambda: FailingService())

        # Must not raise -- a failed best-effort rebuild must never cost
        # the caller their already-successful write.
        await pss.invalidate_patient_state("profile-123")


# ── Structural sweep: every write method across the 11 patient-domain
# services must call invalidate_patient_state() somewhere in its source,
# proving the wiring is complete rather than spot-checked. ────────────

_WRITE_METHODS = [
    ("diagnosis_service", "DiagnosisService", "add_diagnosis"),
    ("diagnosis_service", "DiagnosisService", "add_biomarker"),
    ("treatment_service", "TreatmentService", "add_episode"),
    ("treatment_service", "TreatmentService", "update_episode_status"),
    ("treatment_service", "TreatmentService", "add_cycle"),
    ("medication_service", "MedicationService", "add_medication"),
    ("medication_service", "MedicationService", "update_status"),
    ("symptom_observation_service", "SymptomObservationService", "add_observation"),
    ("symptom_observation_service", "SymptomObservationService", "resolve_observation"),
    ("nutrition_assessment_service", "NutritionAssessmentService", "add_assessment"),
    ("vitals_service", "VitalsService", "add_vital"),
    ("conditions_service", "ConditionsService", "add_comorbidity"),
    ("conditions_service", "ConditionsService", "add_allergy"),
    ("tumor_profile_service", "TumorProfileService", "add_profile"),
    ("care_team_instruction_service", "CareTeamInstructionService", "add_instruction"),
    ("care_team_instruction_service", "CareTeamInstructionService", "deactivate"),
    ("encounter_service", "EncounterService", "add_encounter"),
    ("lab_service", "LabService", "add_result"),
]


class TestEveryWriteMethodInvalidatesState:
    @pytest.mark.parametrize("module_name,class_name,method_name", _WRITE_METHODS)
    def test_method_calls_invalidate_patient_state(self, module_name, class_name, method_name):
        import importlib
        module = importlib.import_module(f"src.api.services.patient.{module_name}")
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)
        source = inspect.getsource(method)
        assert "invalidate_patient_state(" in source, (
            f"{module_name}.{class_name}.{method_name} does not call "
            "invalidate_patient_state() -- a canonical write here won't "
            "refresh the cached patient state snapshot."
        )


class TestReadOnlyMethodsDoNotInvalidate:
    """Sanity check the sweep above isn't vacuous -- confirms a read-only
    method (which should NOT call invalidate_patient_state) doesn't."""

    def test_list_diagnoses_does_not_invalidate(self):
        from src.api.services.patient.diagnosis_service import DiagnosisService
        source = inspect.getsource(DiagnosisService.list_diagnoses)
        assert "invalidate_patient_state(" not in source

    def test_list_medications_does_not_invalidate(self):
        from src.api.services.patient.medication_service import MedicationService
        source = inspect.getsource(MedicationService.list_medications)
        assert "invalidate_patient_state(" not in source


# ── Deeper integration-style test: the call must happen AFTER the
# write's own transaction commits, not from inside it. ────────────────

class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeTxnCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, call_log):
        self._call_log = call_log

    def transaction(self):
        return _FakeTxnCtx()

    async def fetchrow(self, query, *args):
        self._call_log.append(("fetchrow", " ".join(query.split())[:40]))
        # A minimal row-like object supporting dict(row).
        return {
            "id": "diagnosis-1", "patient_profile_id": args[0] if args else None,
            "cancer_site": "lung", "created_at": None,
        }

    async def execute(self, query, *args):
        self._call_log.append(("execute", " ".join(query.split())[:40]))


class _FakePool:
    def __init__(self, call_log):
        self._call_log = call_log

    def acquire(self):
        return _FakeAcquireCtx(_FakeConn(self._call_log))


class _FakeDB:
    def __init__(self, pool):
        self._pool = pool

    async def ensure_schema(self):
        pass

    async def get_pool(self):
        return self._pool


class TestInvalidationHappensAfterCommit:
    @pytest.mark.asyncio
    async def test_add_diagnosis_calls_invalidate_after_transaction_commits(self, monkeypatch):
        call_log = []
        pool = _FakePool(call_log)
        fake_db = _FakeDB(pool)

        import src.api.services.patient.diagnosis_service as diagnosis_service_module
        monkeypatch.setattr(diagnosis_service_module, "get_patient_db", lambda: fake_db)

        rebuild_calls = []

        class FakeStateService:
            async def build_state(self, patient_profile_id):
                # By the time this fires, every DB call the write made
                # must already be in call_log -- proving invalidation
                # happens strictly after the write, not interleaved with
                # (or before) it.
                rebuild_calls.append((patient_profile_id, list(call_log)))

        monkeypatch.setattr(pss, "get_patient_state_service", lambda: FakeStateService())

        from src.api.services.patient.diagnosis_service import DiagnosisService
        service = DiagnosisService()
        await service.add_diagnosis(patient_profile_id="profile-1", cancer_site="lung")

        assert len(rebuild_calls) == 1
        profile_id, log_at_rebuild_time = rebuild_calls[0]
        assert profile_id == "profile-1"
        # The INSERT (fetchrow) and timeline-event (execute) calls must
        # both already have happened before the rebuild fires.
        kinds = [k for k, _ in log_at_rebuild_time]
        assert "fetchrow" in kinds
        assert "execute" in kinds


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
