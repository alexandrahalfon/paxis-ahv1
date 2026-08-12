"""
Tests that patient_chat_service.answer() passes the new shared-contract
fields (2026-08-12 convergence Sprint A item 2) through to
evidence_packet_builder.build_packet(): audience="patient", the actual
RetrievalPlan used for this turn, and a patient_snapshot_id stand-in.
"""

from __future__ import annotations

import pytest

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

    async def context_with_profile(self, patient_user_id):
        return {"patient_profile_id": "profile-42", "state": {}, "retrieval_features": {}}
    monkeypatch.setattr(
        "src.api.services.evidence.patient_context_service.PatientContextService.get_context",
        context_with_profile,
    )

    from src.api.services.evidence import multi_corpus_retriever

    async def evidence_search(query_text, plan, top_k_per_collection=6):
        return [{
            "doc_id": "doc-1", "title": "About pembrolizumab side effects",
            "text": "Common side effects include fatigue.", "citation": "FDA", "year": 2024,
            "collection": "oncology_medication_knowledge", "semantic_score": 0.9,
            "applicability_meta": {}, "source_key": "fda", "authority_class": "A",
        }]
    monkeypatch.setattr(multi_corpus_retriever, "search", evidence_search)

    async def empty_retrieve(self, expanded_query, top_k=6):
        return []
    monkeypatch.setattr(PatientChatService, "_retrieve", empty_retrieve)

    async def empty_web(self, query, limit=4):
        return []
    monkeypatch.setattr(PatientChatService, "_retrieve_web", empty_web)

    def fake_client(self):
        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "Fatigue is common [1]."})()})]

        class _Fake:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        return _Resp()
        return _Fake()
    monkeypatch.setattr(PatientChatService, "_client", fake_client)

    return svc


class TestBuildPacketReceivesSharedContractFields:
    @pytest.mark.asyncio
    async def test_audience_plan_and_snapshot_id_are_passed(self, service, monkeypatch):
        import src.api.services.evidence.evidence_packet_builder as epb

        captured = {}
        real_build_packet = epb.build_packet

        def spy_build_packet(*args, **kwargs):
            captured.update(kwargs)
            return real_build_packet(*args, **kwargs)

        # patient_chat_service imports build_packet by name inside its
        # try block (`from ...evidence_packet_builder import build_packet`),
        # so patch it at the source module -- the inline import picks up
        # the patched name at call time.
        monkeypatch.setattr(epb, "build_packet", spy_build_packet)

        await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )

        assert captured["audience"] == "patient"
        assert captured["patient_snapshot_id"] == "profile-42"
        assert captured["retrieval_plan"] is not None
        assert captured["retrieval_plan"].intent  # a real RetrievalPlan, not a placeholder


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
