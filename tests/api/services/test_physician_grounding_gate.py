"""
Tests for physician_grounding_gate.py (2026-08-12 convergence Sprint C
item 19): wires mechanical grounding (evidence/grounding_validator.py)
and claim-level entailment (evidence/claim_grounding_validator.py)
around physician_answer_generator.generate(), mirroring the patient
path's retry/repair/fallback sequence (Sprint A item 8 / Sprint B item
11) as a standalone, reusable function.
"""

from __future__ import annotations

import pytest

from src.api.services.physician.physician_grounding_gate import (
    generate_grounded_physician_answer,
)
from src.api.services.evidence.evidence_packet_builder import build_packet
from src.api.services.evidence.grounding_validator import SAFE_FALLBACK_RESPONSE
from src.api.services.evidence.claim_grounding_validator import (
    ClaimAssessment,
    ClaimValidationResult,
    UNSUPPORTED,
)


def _packet(evidence=True):
    candidates = [{
        "title": "Adagrasib in KRAS G12C NSCLC", "text": "PFS benefit observed.",
        "source_key": "nejm", "citation": "Smith et al., 2024", "year": 2024,
    }] if evidence else []
    return build_packet("What are the options?", None, candidates)


def _queued_client(answers):
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


class TestFirstDraftAlreadyGrounded:
    @pytest.mark.asyncio
    async def test_no_retry_when_draft_cites_evidence(self):
        client, calls = _queued_client(["Adagrasib showed a PFS benefit [1]."])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="general", client=client,
        )
        assert calls["count"] == 1
        assert result.answer == "Adagrasib showed a PFS benefit [1]."
        assert result.sources_valid is True
        assert result.retried_mechanical is False


class TestMechanicalRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_and_replaces_the_answer(self):
        client, calls = _queued_client([
            "Adagrasib showed a PFS benefit.",       # no citation
            "Adagrasib showed a PFS benefit [1].",    # grounded
        ])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="general", client=client,
        )
        assert calls["count"] == 2
        assert result.answer == "Adagrasib showed a PFS benefit [1]."
        assert result.retried_mechanical is True
        assert result.sources_valid is True

    @pytest.mark.asyncio
    async def test_both_attempts_fail_falls_back_to_safe_response(self):
        client, calls = _queued_client([
            "Adagrasib showed a PFS benefit.",
            "This is based on general oncology guidance.",
        ])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="general", client=client,
        )
        assert calls["count"] == 2
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources_valid is False

    @pytest.mark.asyncio
    async def test_retry_generation_exception_falls_back_to_safe_response(self):
        call_count = {"n": 0}

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "No citation here."})()})]

        class _Fake:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        call_count["n"] += 1
                        if call_count["n"] == 1:
                            return _Resp()
                        raise RuntimeError("upstream timeout")

        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="general", client=_Fake(),
        )
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources_valid is False


class TestNoEvidenceIsNeverGated:
    @pytest.mark.asyncio
    async def test_uncited_answer_with_no_evidence_passes_through_unchanged(self):
        client, calls = _queued_client(["I don't have specific evidence on that."])
        result = await generate_grounded_physician_answer(
            "General question.", _packet(evidence=False), intent="general", client=client,
        )
        assert calls["count"] == 1
        assert result.answer == "I don't have specific evidence on that."
        assert result.sources_valid is True


class TestClaimLevelValidationScoping:
    @pytest.mark.asyncio
    async def test_claim_validator_not_called_for_a_non_strict_intent(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        called = []
        async def spy(answer, packet, **kwargs):
            called.append(answer)
            return ClaimValidationResult(claims=[], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", spy)

        client, _ = _queued_client(["Adagrasib showed a PFS benefit [1]."])
        await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="general", client=client,
        )
        assert called == []

    @pytest.mark.asyncio
    async def test_claim_validator_called_for_a_strict_intent(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        called = []
        async def spy(answer, packet, **kwargs):
            called.append(answer)
            return ClaimValidationResult(claims=[], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", spy)

        client, _ = _queued_client(["Adagrasib showed a PFS benefit [1]."])
        await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="therapy_selection", client=client,
        )
        assert len(called) == 1


class TestClaimLevelRepairAndFallback:
    @pytest.mark.asyncio
    async def test_unsupported_claim_is_mechanically_repaired(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(
                    claim="Adagrasib is superior to chemotherapy [1].",
                    support_level=UNSUPPORTED, reason="No such comparison in the passage.",
                ),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)

        client, _ = _queued_client([
            "Adagrasib is superior to chemotherapy [1]. Discuss with the patient.",
        ])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="therapy_selection", client=client,
        )
        assert "superior to chemotherapy" not in result.answer
        assert "Discuss with the patient." in result.answer
        assert result.sources_valid is True

    @pytest.mark.asyncio
    async def test_unrepairable_claim_falls_back_to_safe_response(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(claim="A sentence not in the answer.", support_level=UNSUPPORTED),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)

        client, _ = _queued_client(["Adagrasib showed a PFS benefit [1]."])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="therapy_selection", client=client,
        )
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources_valid is False

    @pytest.mark.asyncio
    async def test_claim_validator_exception_preserves_mechanically_grounded_answer(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        async def failing_validate(answer, packet, **kwargs):
            raise RuntimeError("upstream timeout")
        monkeypatch.setattr(cgv, "validate_claims", failing_validate)

        client, _ = _queued_client(["Adagrasib showed a PFS benefit [1]."])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="therapy_selection", client=client,
        )
        assert result.answer == "Adagrasib showed a PFS benefit [1]."
        assert result.sources_valid is True


class TestClaimValidationSkippedAfterMechanicalFallback:
    @pytest.mark.asyncio
    async def test_claim_validator_never_called_when_mechanical_gate_already_fell_back(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv

        called = []
        async def spy(answer, packet, **kwargs):
            called.append(answer)
            return ClaimValidationResult(claims=[], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", spy)

        client, _ = _queued_client([
            "No citation here.", "Still no citation, general oncology guidance.",
        ])
        result = await generate_grounded_physician_answer(
            "What are the options?", _packet(), intent="therapy_selection", client=client,
        )
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert called == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
