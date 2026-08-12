"""
Tests for the post-generation grounding retry/fail gate in
patient_chat_service.answer() (2026-08-12 beta audit item 8): the code
used to only log a failed grounding_validator check and return the
ungrounded answer anyway. It now retries once with a stricter prompt,
and falls back to a safe, non-fabricated response if the retry also
fails -- but only when there was actually evidence to ground in;
conversational answers with no evidence packet must keep working
unchanged (that's the pregen gate's job, covered in
test_patient_chat_grounding_gate.py, not this one).
"""

from __future__ import annotations

import pytest

from src.api.services.patient_portal.patient_chat_service import PatientChatService
from src.api.services.evidence.grounding_validator import SAFE_FALLBACK_RESPONSE


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

    return svc


def _queued_client(answers):
    """Returns a fake OpenAI client yielding one answer per call, in
    order, and records how many times it was called."""
    calls = {"count": 0, "messages": []}

    class _Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]

    class _Fake:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    i = calls["count"]
                    calls["count"] += 1
                    calls["messages"].append(kwargs.get("messages"))
                    return _Resp(answers[min(i, len(answers) - 1)])
    return _Fake(), calls


class TestFirstAttemptAlreadyGrounded:
    @pytest.mark.asyncio
    async def test_no_retry_when_first_answer_cites_evidence(self, service, monkeypatch):
        fake_client, calls = _queued_client(["Fatigue is a common side effect [1]."])
        monkeypatch.setattr(PatientChatService, "_client", lambda self: fake_client)

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )

        assert calls["count"] == 1
        assert result.answer == "Fatigue is a common side effect [1]."
        assert result.sources


class TestRetrySucceeds:
    @pytest.mark.asyncio
    async def test_retry_replaces_answer_when_second_attempt_is_grounded(self, service, monkeypatch):
        fake_client, calls = _queued_client([
            "Fatigue is a common side effect.",               # first: no citation
            "Fatigue is a common side effect [1].",            # retry: grounded
        ])
        monkeypatch.setattr(PatientChatService, "_client", lambda self: fake_client)

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )

        assert calls["count"] == 2
        assert result.answer == "Fatigue is a common side effect [1]."
        assert result.sources  # retry succeeded -- sources stay attached

        # The retry call must have included the failed first answer and
        # the retry instruction as extra turns, not just repeated the
        # original prompt verbatim.
        retry_messages = calls["messages"][1]
        assert retry_messages[-2]["role"] == "assistant"
        assert retry_messages[-2]["content"] == "Fatigue is a common side effect."
        assert retry_messages[-1]["role"] == "user"
        assert "cite" in retry_messages[-1]["content"].lower()


class TestBothAttemptsFailSafeFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_safe_response_and_clears_sources(self, service, monkeypatch):
        fake_client, calls = _queued_client([
            "Fatigue is a common side effect.",                       # first: no citation
            "This is based on general oncology guidance.",            # retry: still ungrounded
        ])
        monkeypatch.setattr(PatientChatService, "_client", lambda self: fake_client)

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )

        assert calls["count"] == 2
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources == []
        assert result.retrieval_used is False

    @pytest.mark.asyncio
    async def test_falls_back_when_retry_generation_itself_raises(self, service, monkeypatch):
        call_count = {"n": 0}

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "Fatigue is a common side effect."})()})]

        class _Fake:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        call_count["n"] += 1
                        if call_count["n"] == 1:
                            return _Resp()
                        raise RuntimeError("upstream timeout")
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _Fake())

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )

        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources == []


class TestNoEvidenceConversationalAnswersAreNeverGated:
    """Sources are legitimately empty for a conversational message (see
    test_patient_chat_grounding_gate.py); the retry/fail gate must not
    fire just because validate() would call an empty packet 'invalid'."""

    @pytest.mark.asyncio
    async def test_no_citation_no_evidence_answer_passes_through_unchanged(self, service, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever

        async def no_evidence_search(query_text, plan, top_k_per_collection=6):
            return []
        monkeypatch.setattr(multi_corpus_retriever, "search", no_evidence_search)

        fake_client, calls = _queued_client(["I'm glad you're reaching out."])
        monkeypatch.setattr(PatientChatService, "_client", lambda self: fake_client)

        result = await service.answer(
            message="Thank you so much, that helps.",
            patient_user_id="user-1",
            persist=False,
        )

        assert calls["count"] == 1
        assert result.answer == "I'm glad you're reaching out."


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
