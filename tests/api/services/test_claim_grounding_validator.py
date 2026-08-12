"""
Tests for claim_grounding_validator.py (2026-08-12 convergence Sprint A
item 4) -- the shared, audience-agnostic claim-level entailment check
that grounding_validator.py's mechanical citation check explicitly does
not attempt (see that module's docstring).
"""

from __future__ import annotations

import json

import pytest

from src.api.services.evidence.claim_grounding_validator import (
    NOT_A_CLAIM,
    PARTIALLY_SUPPORTED,
    STRICT_VALIDATION_INTENTS,
    SUPPORTED,
    UNSUPPORTED,
    ClaimAssessment,
    ClaimValidationResult,
    repair_answer,
    requires_strict_validation,
    validate_claims,
)


_SENTINEL = object()


def _packet(evidence=_SENTINEL):
    if evidence is _SENTINEL:
        evidence = [{"title": "Study A", "text": "PFS was measured at 6 months."}]
    return {"evidence": evidence}


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]


def _client_returning(payload: dict):
    content = json.dumps(payload)

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return _FakeResp(content)
    return _Client()


def _client_raising(exc: Exception):
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    raise exc
    return _Client()


def _client_returning_raw(raw: str):
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return _FakeResp(raw)
    return _Client()


class TestRequiresStrictValidation:
    def test_strict_intents_return_true(self):
        for intent in STRICT_VALIDATION_INTENTS:
            assert requires_strict_validation(intent) is True

    def test_non_strict_intents_return_false(self):
        assert requires_strict_validation("definition") is False
        assert requires_strict_validation("general") is False

    def test_none_or_empty_returns_false(self):
        assert requires_strict_validation(None) is False
        assert requires_strict_validation("") is False


class TestValidateClaimsEmptyInputs:
    @pytest.mark.asyncio
    async def test_empty_evidence_short_circuits_without_calling_the_model(self):
        client = _client_raising(AssertionError("must not be called"))
        result = await validate_claims("Some answer.", _packet(evidence=[]), client=client)
        assert result.ran is True
        assert result.claims == []
        assert result.overall_valid is True

    @pytest.mark.asyncio
    async def test_empty_answer_short_circuits(self):
        client = _client_raising(AssertionError("must not be called"))
        result = await validate_claims("", _packet(), client=client)
        assert result.ran is True
        assert result.overall_valid is True


class TestValidateClaimsParsesModelOutput:
    @pytest.mark.asyncio
    async def test_supported_claim(self):
        client = _client_returning({
            "claims": [{
                "claim": "PFS was measured at 6 months.",
                "citations": [1], "support_level": "supported",
                "reason": "Directly stated.", "rewrite": None,
            }],
        })
        result = await validate_claims("PFS was measured at 6 months. [1]", _packet(), client=client)
        assert result.ran is True
        assert len(result.claims) == 1
        assert result.claims[0].support_level == SUPPORTED
        assert result.overall_valid is True
        assert result.needs_repair == []

    @pytest.mark.asyncio
    async def test_unsupported_claim_makes_overall_invalid(self):
        client = _client_returning({
            "claims": [{
                "claim": "Adagrasib improved progression-free survival.",
                "citations": [1], "support_level": "unsupported",
                "reason": "Passage only reports PFS was measured, not that it improved.",
                "rewrite": None,
            }],
        })
        result = await validate_claims(
            "Adagrasib improved progression-free survival. [1]", _packet(), client=client,
        )
        assert result.overall_valid is False
        assert len(result.needs_repair) == 1

    @pytest.mark.asyncio
    async def test_partially_supported_claim_carries_a_rewrite(self):
        client = _client_returning({
            "claims": [{
                "claim": "Adagrasib improved progression-free survival.",
                "citations": [1], "support_level": "partially_supported",
                "reason": "Passage reports PFS outcomes but not comparative improvement.",
                "rewrite": "The study reported progression-free survival outcomes.",
            }],
        })
        result = await validate_claims(
            "Adagrasib improved progression-free survival. [1]", _packet(), client=client,
        )
        assert result.claims[0].support_level == PARTIALLY_SUPPORTED
        assert result.claims[0].rewrite == "The study reported progression-free survival outcomes."
        # partially_supported doesn't make the answer flatly invalid --
        # only unsupported does; the caller narrows rather than blocks.
        assert result.overall_valid is True
        assert len(result.needs_repair) == 1

    @pytest.mark.asyncio
    async def test_not_a_claim_is_not_a_repair_target(self):
        client = _client_returning({
            "claims": [{
                "claim": "I'm glad you asked.",
                "citations": [], "support_level": "not_a_claim",
                "reason": "Conversational, not a factual assertion.", "rewrite": None,
            }],
        })
        result = await validate_claims("I'm glad you asked. [1]", _packet(), client=client)
        assert result.claims[0].support_level == NOT_A_CLAIM
        assert result.overall_valid is True
        assert result.needs_repair == []

    @pytest.mark.asyncio
    async def test_invalid_support_level_defaults_to_unsupported(self):
        client = _client_returning({
            "claims": [{"claim": "X", "citations": [1], "support_level": "nonsense_value", "reason": "r"}],
        })
        result = await validate_claims("X [1]", _packet(), client=client)
        assert result.claims[0].support_level == UNSUPPORTED

    @pytest.mark.asyncio
    async def test_rewrite_dropped_unless_partially_supported(self):
        client = _client_returning({
            "claims": [{
                "claim": "X", "citations": [1], "support_level": "supported",
                "reason": "r", "rewrite": "should be ignored",
            }],
        })
        result = await validate_claims("X [1]", _packet(), client=client)
        assert result.claims[0].rewrite is None

    @pytest.mark.asyncio
    async def test_claims_missing_claim_text_are_skipped(self):
        client = _client_returning({"claims": [{"citations": [1], "support_level": "supported"}]})
        result = await validate_claims("X [1]", _packet(), client=client)
        assert result.claims == []


class TestValidateClaimsFailsSoft:
    @pytest.mark.asyncio
    async def test_model_exception_returns_unran_result_not_a_raise(self):
        client = _client_raising(RuntimeError("upstream timeout"))
        result = await validate_claims("X [1]", _packet(), client=client)
        assert result.ran is False
        assert result.overall_valid is None
        assert "upstream timeout" in result.error

    @pytest.mark.asyncio
    async def test_malformed_json_returns_unran_result_not_a_raise(self):
        client = _client_returning_raw("not valid json{{{")
        result = await validate_claims("X [1]", _packet(), client=client)
        assert result.ran is False
        assert result.overall_valid is None


class TestRepairAnswer:
    def test_removes_unsupported_claim(self):
        answer = "Adagrasib improved progression-free survival. Talk to your doctor."
        result = ClaimValidationResult(claims=[
            ClaimAssessment(
                claim="Adagrasib improved progression-free survival.",
                support_level=UNSUPPORTED,
            ),
        ], ran=True)
        repaired = repair_answer(answer, result)
        assert "Adagrasib improved" not in repaired
        assert "Talk to your doctor." in repaired

    def test_substitutes_partially_supported_rewrite(self):
        answer = "Adagrasib improved progression-free survival."
        result = ClaimValidationResult(claims=[
            ClaimAssessment(
                claim="Adagrasib improved progression-free survival.",
                support_level=PARTIALLY_SUPPORTED,
                rewrite="The study reported progression-free survival outcomes.",
            ),
        ], ran=True)
        repaired = repair_answer(answer, result)
        assert repaired == "The study reported progression-free survival outcomes."

    def test_leaves_supported_and_not_a_claim_untouched(self):
        answer = "PFS was measured. Thanks for asking."
        result = ClaimValidationResult(claims=[
            ClaimAssessment(claim="PFS was measured.", support_level=SUPPORTED),
            ClaimAssessment(claim="Thanks for asking.", support_level=NOT_A_CLAIM),
        ], ran=True)
        assert repair_answer(answer, result) == answer

    def test_claim_not_found_verbatim_is_skipped_silently(self):
        answer = "The study reported outcomes."
        result = ClaimValidationResult(claims=[
            ClaimAssessment(claim="A completely different sentence.", support_level=UNSUPPORTED),
        ], ran=True)
        assert repair_answer(answer, result) == answer

    def test_returns_original_answer_unchanged_when_validator_did_not_run(self):
        answer = "Adagrasib improved progression-free survival."
        result = ClaimValidationResult(claims=[], ran=False, error="timeout")
        assert repair_answer(answer, result) == answer


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
