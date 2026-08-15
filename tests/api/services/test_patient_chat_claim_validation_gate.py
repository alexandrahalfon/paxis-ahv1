"""
Tests for claim-level grounding on high-risk patient intents (2026-08-12
convergence Sprint B item 11): patient_chat_service.answer() step 6c
wires claim_grounding_validator.py (A4) on top of the mechanical
grounding retry/fail gate (item 8), but only for intents mapped onto
STRICT_VALIDATION_INTENTS via _PATIENT_STRICT_INTENT_MAP, and only when
a real EvidencePacket with sources exists.
"""

from __future__ import annotations

import pytest

from src.api.services.patient_portal.patient_chat_service import PatientChatService
from src.api.services.evidence.claim_grounding_validator import (
    ClaimAssessment,
    ClaimValidationResult,
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    UNSUPPORTED,
)
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

    async def evidence_search(query_text, plan, top_k_per_collection=6, **kwargs):
        return [{
            "doc_id": "doc-1", "title": "Pembrolizumab side effects",
            "text": "Common side effects include fatigue and rash.", "citation": "FDA", "year": 2024,
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


def _client(answer):
    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": answer})()})]

    class _Fake:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return _Resp()
    return _Fake()


class TestNonStrictIntentSkipsClaimValidation:
    @pytest.mark.asyncio
    async def test_claim_validator_never_called_for_nutrition_question(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        called = []
        async def spy_validate(answer, packet, **kwargs):
            called.append(answer)
            return ClaimValidationResult(claims=[], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", spy_validate)
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _client("Eat small meals [1]."))

        await service.answer(
            message="What foods should I eat during chemo?",
            patient_user_id="user-1",
            persist=False,
        )
        assert called == []


class TestStrictIntentTriggersClaimValidation:
    @pytest.mark.asyncio
    async def test_supported_claim_leaves_answer_unchanged(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(claim="Fatigue is a common side effect [1].", support_level=SUPPORTED),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _client("Fatigue is a common side effect [1]."),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == "Fatigue is a common side effect [1]."

    @pytest.mark.asyncio
    async def test_unsupported_claim_is_mechanically_repaired(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(
                    claim="Pembrolizumab is more effective than chemotherapy [1].",
                    support_level=UNSUPPORTED,
                    reason="Passage never makes this comparison.",
                ),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _client(
                "Pembrolizumab is more effective than chemotherapy [1]. Talk to your care team."
            ),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert "more effective than chemotherapy" not in result.answer
        assert "Talk to your care team." in result.answer

    @pytest.mark.asyncio
    async def test_partially_supported_claim_uses_the_rewrite(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(
                    claim="This drug cures the cancer [1].",
                    support_level=PARTIALLY_SUPPORTED,
                    rewrite="This drug is used to treat the cancer [1].",
                ),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _client("This drug cures the cancer [1]."),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == "This drug is used to treat the cancer [1]."

    @pytest.mark.asyncio
    async def test_unrepairable_unsupported_claim_falls_back_to_safe_response(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def fake_validate(answer, packet, **kwargs):
            # Claim text doesn't match the answer verbatim -> repair_answer
            # can't find anything to remove.
            return ClaimValidationResult(claims=[
                ClaimAssessment(claim="A sentence that isn't in the answer.", support_level=UNSUPPORTED),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _client("Fatigue is a common side effect [1]."),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_claim_validator_exception_leaves_mechanically_grounded_answer(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def failing_validate(answer, packet, **kwargs):
            raise RuntimeError("upstream timeout")
        monkeypatch.setattr(cgv, "validate_claims", failing_validate)
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _client("Fatigue is a common side effect [1]."),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == "Fatigue is a common side effect [1]."


class TestClaimValidationSkippedAfterMechanicalFallback:
    @pytest.mark.asyncio
    async def test_never_called_when_mechanical_gate_already_used_safe_fallback(self, service, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        called = []
        async def spy_validate(answer, packet, **kwargs):
            called.append(answer)
            return ClaimValidationResult(claims=[], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", spy_validate)

        # Every generation attempt returns an uncited answer -> the
        # mechanical retry/fail gate (step 6b) exhausts its retry and
        # falls back to SAFE_FALLBACK_RESPONSE before step 6c ever runs.
        monkeypatch.setattr(
            PatientChatService, "_client",
            lambda self: _client("Fatigue is a common side effect with no citation."),
        )

        result = await service.answer(
            message="What are the side effects of pembrolizumab?",
            patient_user_id="user-1",
            persist=False,
        )
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert called == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
