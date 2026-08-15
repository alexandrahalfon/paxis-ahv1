"""
Tests for the patient live-PubMed fallback gate (2026-08-12 convergence
Sprint B item 9): "generic PubMed live search should be reserved for
research/study questions, or disabled for patient mode at first" --
implemented as settings.patient_pubmed_fallback_enabled, defaulting to
False. PubMed stays the absolute last resort even when enabled -- these
tests confirm both that it's skipped by default and that flipping the
setting restores the previous behavior unchanged.
"""

from __future__ import annotations

import pytest

from src.core.config import settings
from src.api.services.patient_portal.patient_chat_service import PatientChatService


@pytest.fixture
def service(monkeypatch):
    svc = PatientChatService()

    async def no_known_facts(self, patient_user_id, conversation_facts=None):
        return dict(conversation_facts or {})
    monkeypatch.setattr(PatientChatService, "known_facts_for", no_known_facts)

    async def no_profile(patient_user_id):
        return None
    monkeypatch.setattr(
        "src.api.services.patient.patient_profile_service.get_patient_profile_service",
        lambda: type("S", (), {"get_by_user": staticmethod(no_profile)})(),
    )

    async def empty_context(self, patient_user_id):
        return {}
    monkeypatch.setattr(
        "src.api.services.evidence.patient_context_service.PatientContextService.get_context",
        empty_context,
    )

    from src.api.services.evidence import multi_corpus_retriever
    async def empty_search(query_text, plan, top_k_per_collection=6, **kwargs):
        return []
    monkeypatch.setattr(multi_corpus_retriever, "search", empty_search)

    async def empty_retrieve(self, expanded_query, top_k=6):
        return []
    monkeypatch.setattr(PatientChatService, "_retrieve", empty_retrieve)

    return svc


def _pubmed_hit():
    return [{
        "title": "Nausea management in oncology", "text": "PubMed abstract text.",
        "citation": "Smith et al., 2024", "year": 2024, "source_type": "pubmed",
    }]


def _generation_client(answer="I'm glad you're reaching out."):
    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": answer})()})]

    class _Fake:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return _Resp()
    return _Fake()


class TestPubmedFallbackDefaultsToOff:
    @pytest.mark.asyncio
    async def test_retrieve_web_is_never_called_when_flag_is_off(self, service, monkeypatch):
        monkeypatch.setattr(settings, "patient_pubmed_fallback_enabled", False)

        called = []
        async def spy_web(self, query, limit=4):
            called.append(query)
            return _pubmed_hit()
        monkeypatch.setattr(PatientChatService, "_retrieve_web", spy_web)
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _generation_client())

        result = await service.answer(
            message="Thank you so much, that helps.",
            patient_user_id="user-1",
            persist=False,
        )

        assert called == []
        assert result.used_web_search is False


class TestPubmedFallbackWhenExplicitlyEnabled:
    @pytest.mark.asyncio
    async def test_retrieve_web_is_called_when_flag_is_on_and_no_other_evidence(self, service, monkeypatch):
        monkeypatch.setattr(settings, "patient_pubmed_fallback_enabled", True)

        called = []
        async def spy_web(self, query, limit=4):
            called.append(query)
            return _pubmed_hit()
        monkeypatch.setattr(PatientChatService, "_retrieve_web", spy_web)
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _generation_client("Fatigue is a common side effect [1]."),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )

        assert len(called) == 1
        assert result.used_web_search is True
        assert result.sources


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
