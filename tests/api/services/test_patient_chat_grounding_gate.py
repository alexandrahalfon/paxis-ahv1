"""
Tests for the hard grounding gate in patient_chat_service.answer()
(2026-08-12 beta audit, "make the evidence packet a true hard boundary"):
a factual medication/symptom/nutrition/treatment/diagnosis question with
zero usable evidence after every retrieval fallback must not be answered
from the model's own memory, and must not even call the model. A
conversational (general-intent) message must keep working without
evidence, per the same audit finding.
"""

from __future__ import annotations

import pytest

from src.api.services.patient_portal.patient_chat_service import (
    PatientChatService,
    get_patient_chat_service,
)
from src.api.services.evidence.patient_context_service import NO_EVIDENCE_RESPONSE


@pytest.fixture
def service(monkeypatch):
    svc = PatientChatService()

    # Skip real DB/network lookups this test doesn't care about.
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
    async def empty_search(query_text, plan, top_k_per_collection=6):
        return []
    monkeypatch.setattr(multi_corpus_retriever, "search", empty_search)

    async def empty_retrieve(self, expanded_query, top_k=6):
        return []
    monkeypatch.setattr(PatientChatService, "_retrieve", empty_retrieve)

    async def empty_web(self, query, limit=4):
        return []
    monkeypatch.setattr(PatientChatService, "_retrieve_web", empty_web)

    return svc


def _forbidden_client():
    class _Forbidden:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    raise AssertionError("generation must not run when the gate fires")
    return _Forbidden()


class TestHardGroundingGate:
    @pytest.mark.asyncio
    async def test_factual_intent_zero_evidence_is_gated_without_calling_the_model(self, service, monkeypatch):
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _forbidden_client())

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == NO_EVIDENCE_RESPONSE
        assert result.sources == []
        assert result.retrieval_used is False

    @pytest.mark.asyncio
    async def test_conversational_general_intent_still_generates_without_evidence(self, service, monkeypatch):
        class _Resp:
            class choices0:
                class message:
                    content = "I'm glad you're reaching out."
            choices = [choices0]

        class _Fake:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        return _Resp()
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _Fake())

        result = await service.answer(
            message="Thank you so much, that helps.",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == "I'm glad you're reaching out."

    @pytest.mark.asyncio
    async def test_factual_intent_with_evidence_is_not_gated(self, service, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        async def fake_search(query_text, plan, top_k_per_collection=6):
            return [{
                "doc_id": "doc-1", "title": "About pembrolizumab side effects",
                "text": "Common side effects include fatigue.", "citation": "FDA", "year": 2024,
                "collection": "oncology_medication_knowledge", "semantic_score": 0.9,
                "applicability_meta": {}, "source_key": "fda", "authority_class": "A",
            }]
        monkeypatch.setattr(multi_corpus_retriever, "search", fake_search)

        class _Resp:
            class choices0:
                class message:
                    content = "Real grounded answer [1]."
            choices = [choices0]

        class _Fake:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        return _Resp()
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _Fake())

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == "Real grounded answer [1]."
        assert result.sources


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
