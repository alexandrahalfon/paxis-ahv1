"""
Tests for physician_beta.py (2026-08-12 convergence Sprint C item 21):
the protected HTTP entrypoint into physician_rag_orchestrator.py
(item 20). Endpoint functions are called directly (matching this
codebase's existing route-test convention, see
test_patient_query_route.py) rather than through a TestClient, so
`current_user` is passed explicitly instead of resolved via
Depends(require_physician).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.routes import physician_beta as route_module
from src.api.routes.physician_beta import PhysicianQueryRequest, physician_query
from src.core.config import settings


_PHYSICIAN = {"id": "phys-1", "role": "physician"}


@pytest.fixture(autouse=True)
def _restore_flag():
    original = settings.physician_rag_beta_enabled
    yield
    settings.physician_rag_beta_enabled = original


class TestDisabledByDefault:
    @pytest.mark.asyncio
    async def test_returns_404_and_never_touches_the_orchestrator(self, monkeypatch):
        settings.physician_rag_beta_enabled = False

        async def boom(*a, **k):
            raise AssertionError("answer_physician_query must not run while disabled")
        monkeypatch.setattr(
            "src.api.services.physician.physician_rag_orchestrator.answer_physician_query", boom,
        )

        with pytest.raises(HTTPException) as exc_info:
            await physician_query(
                PhysicianQueryRequest(question="What are the options?"), current_user=_PHYSICIAN,
            )
        assert exc_info.value.status_code == 404


class TestEnabled:
    @pytest.mark.asyncio
    async def test_delegates_to_the_orchestrator_and_returns_its_dict(self, monkeypatch):
        settings.physician_rag_beta_enabled = True
        captured = {}

        class _FakeResult:
            def to_dict(self):
                return {"answer": "Adagrasib showed a PFS benefit [1].", "sources_valid": True}

        async def fake_answer(**kwargs):
            captured.update(kwargs)
            return _FakeResult()
        monkeypatch.setattr(
            "src.api.services.physician.physician_rag_orchestrator.answer_physician_query", fake_answer,
        )

        resp = await physician_query(
            PhysicianQueryRequest(
                question="What are the options?", patient_profile_id="patient-1",
                intent="therapy_selection",
            ),
            current_user=_PHYSICIAN,
        )

        assert resp == {"answer": "Adagrasib showed a PFS benefit [1].", "sources_valid": True}
        assert captured["physician_user_id"] == "phys-1"
        assert captured["question"] == "What are the options?"
        assert captured["patient_profile_id"] == "patient-1"
        assert captured["intent"] == "therapy_selection"

    @pytest.mark.asyncio
    async def test_optional_fields_default_to_none(self, monkeypatch):
        settings.physician_rag_beta_enabled = True
        captured = {}

        class _FakeResult:
            def to_dict(self):
                return {"answer": "ok"}

        async def fake_answer(**kwargs):
            captured.update(kwargs)
            return _FakeResult()
        monkeypatch.setattr(
            "src.api.services.physician.physician_rag_orchestrator.answer_physician_query", fake_answer,
        )

        await physician_query(
            PhysicianQueryRequest(question="General question."), current_user=_PHYSICIAN,
        )
        assert captured["patient_profile_id"] is None
        assert captured["intent"] is None

    @pytest.mark.asyncio
    async def test_unexpected_orchestrator_exception_becomes_a_generic_503(self, monkeypatch):
        settings.physician_rag_beta_enabled = True

        async def boom(**kwargs):
            raise RuntimeError("something internal broke")
        monkeypatch.setattr(
            "src.api.services.physician.physician_rag_orchestrator.answer_physician_query", boom,
        )

        with pytest.raises(HTTPException) as exc_info:
            await physician_query(
                PhysicianQueryRequest(question="What are the options?"), current_user=_PHYSICIAN,
            )
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == route_module._SERVER_ERROR


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
