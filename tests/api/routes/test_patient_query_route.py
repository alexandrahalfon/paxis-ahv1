"""
Tests for the legacy /rag/patient-query route's 2026-08-12 rewire onto
the shared evidence pipeline (multi_corpus_retriever -> applicability_
scorer -> evidence_packet_builder), per the beta audit's "do not maintain
two independent patient RAG behaviors" finding. See patient_query.py's
module docstring for why this endpoint stays unauthenticated and cannot
do patient-state personalization -- these tests only cover the retrieval
routing/fallback behavior, not personalization (there is none here).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.routes import patient_query as patient_query_module
from src.api.routes.patient_query import PatientQueryRequest, patient_query


def _fake_openai(content: str):
    def create(*args, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _forbidden_openai():
    def create(*args, **kwargs):
        raise AssertionError("generation must not run when the hard grounding gate fires")
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def _empty_search(query_text, plan, top_k_per_collection=6):
    return []


class _EmptyRagService:
    async def query(self, **kwargs):
        return {"evidence": []}


@pytest.fixture(autouse=True)
def _fake_shared_openai_client(monkeypatch):
    monkeypatch.setattr(patient_query_module, "_openai_client", _fake_openai("A plain-language answer [1]."))
    yield
    monkeypatch.setattr(patient_query_module, "_openai_client", None)


class TestMultiCorpusPreferredOverLegacyBackbone:
    @pytest.mark.asyncio
    async def test_multi_corpus_candidates_are_used_and_legacy_backbone_is_never_called(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        async def fake_search(query_text, plan, top_k_per_collection=6):
            return [{
                "doc_id": "doc-1", "title": "Managing taste changes",
                "text": "Some patient-education text about taste changes.",
                "citation": "NCI", "year": 2024, "collection": "oncology_patient_education",
                "semantic_score": 0.8, "applicability_meta": {}, "source_key": "nci",
                "authority_class": "A",
            }]

        monkeypatch.setattr(multi_corpus_retriever, "search", fake_search)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("legacy enhanced_rag_service backbone should not be called")

        monkeypatch.setattr(
            "src.api.services.enhanced_rag_service.get_enhanced_rag_service", _fail_if_called
        )

        resp = await patient_query(PatientQueryRequest(question="What can I do about metallic taste?"))
        assert resp.sources
        assert resp.sources[0]["title"] == "Managing taste changes"
        assert "Fake" not in resp.answer  # sanity: got our fake generation content through
        assert resp.answer.startswith("A plain-language answer")

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_backbone_when_multi_corpus_returns_nothing(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        async def empty_search(query_text, plan, top_k_per_collection=6):
            return []

        monkeypatch.setattr(multi_corpus_retriever, "search", empty_search)

        class FakeRagService:
            async def query(self, **kwargs):
                return {"evidence": [{
                    "title": "A literature study", "text": "Findings from a trial.",
                    "citation": "Smith et al. 2023", "doi": None, "pmid": None, "year": 2023,
                }]}

        monkeypatch.setattr(
            "src.api.services.enhanced_rag_service.get_enhanced_rag_service",
            lambda: FakeRagService(),
        )

        resp = await patient_query(PatientQueryRequest(question="What is pembrolizumab?"))
        assert resp.sources
        assert resp.sources[0]["title"] == "A literature study"


class TestHardGroundingGate:
    @pytest.mark.asyncio
    async def test_factual_intent_with_zero_evidence_is_gated_without_calling_the_model(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        monkeypatch.setattr(multi_corpus_retriever, "search", _empty_search)
        monkeypatch.setattr(
            "src.api.services.enhanced_rag_service.get_enhanced_rag_service",
            lambda: _EmptyRagService(),
        )
        monkeypatch.setattr(patient_query_module, "_openai_client", _forbidden_openai())

        resp = await patient_query(PatientQueryRequest(question="What is pembrolizumab?"))
        assert resp.sources == []
        from src.api.services.evidence.patient_context_service import NO_EVIDENCE_RESPONSE
        assert resp.answer == NO_EVIDENCE_RESPONSE

    @pytest.mark.asyncio
    async def test_conversational_general_intent_still_generates_without_evidence(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        monkeypatch.setattr(multi_corpus_retriever, "search", _empty_search)
        monkeypatch.setattr(
            "src.api.services.enhanced_rag_service.get_enhanced_rag_service",
            lambda: _EmptyRagService(),
        )
        monkeypatch.setattr(patient_query_module, "_openai_client", _fake_openai("Thanks for saying that."))

        # "Thank you" matches no factual-intent keyword pattern -> general
        # intent -> conversational messages must keep working without evidence.
        resp = await patient_query(PatientQueryRequest(question="Thank you so much for your help today."))
        assert resp.answer == "Thanks for saying that."

    @pytest.mark.asyncio
    async def test_factual_intent_with_some_evidence_is_not_gated(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        async def fake_search(query_text, plan, top_k_per_collection=6):
            return [{
                "doc_id": "doc-1", "title": "About pembrolizumab", "text": "It is a checkpoint inhibitor.",
                "citation": "FDA", "year": 2024, "collection": "oncology_medication_knowledge",
                "semantic_score": 0.9, "applicability_meta": {}, "source_key": "fda",
                "authority_class": "A",
            }]

        monkeypatch.setattr(multi_corpus_retriever, "search", fake_search)
        monkeypatch.setattr(patient_query_module, "_openai_client", _fake_openai("Real answer [1]."))

        resp = await patient_query(PatientQueryRequest(question="What is pembrolizumab?"))
        assert resp.answer == "Real answer [1]."
        assert resp.sources


class TestEmergencyTriageStillBlocks:
    @pytest.mark.asyncio
    async def test_emergency_message_short_circuits_before_any_retrieval(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("retrieval should never run for an emergency message")

        monkeypatch.setattr(multi_corpus_retriever, "search", _fail_if_called)

        resp = await patient_query(PatientQueryRequest(question="I am having thoughts of suicide"))
        assert resp.sources == []
        assert resp.answer  # a fixed safety response, not a generated one


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
