"""
Baseline tests for evidence/grounding_validator.py's validate() plus the
RETRY_INSTRUCTION/SAFE_FALLBACK_RESPONSE constants added for the
2026-08-12 beta audit item 8 retry/fail gate (see
test_patient_chat_grounding_retry_gate.py for the gate itself, wired into
patient_chat_service.answer()). No prior test file exercised validate()
directly before this change.
"""

from __future__ import annotations

from src.api.services.evidence.grounding_validator import (
    RETRY_INSTRUCTION,
    SAFE_FALLBACK_RESPONSE,
    validate,
)


def _packet(n_evidence: int = 2):
    return {"evidence": [{"title": f"Source {i}"} for i in range(1, n_evidence + 1)]}


class TestValidate:
    def test_valid_when_citation_present_and_in_range(self):
        result = validate("Fatigue is common [1].", _packet())
        assert result.valid is True
        assert result.reasons == []
        assert result.citations_used == [1]

    def test_invalid_when_no_citations_but_evidence_present(self):
        result = validate("Fatigue is common.", _packet())
        assert result.valid is False
        assert any("no [n] evidence citations" in r for r in result.reasons)

    def test_invalid_when_citation_out_of_range(self):
        result = validate("See [5].", _packet(n_evidence=2))
        assert result.valid is False
        assert result.invalid_citations == [5]

    def test_invalid_when_forbidden_phrase_present(self):
        result = validate(
            "This is based on general oncology guidance rather than a specific source [1].",
            _packet(),
        )
        assert result.valid is False
        assert "general oncology guidance" in result.forbidden_phrases_found

    def test_invalid_when_evidence_packet_empty(self):
        result = validate("Some answer.", _packet(n_evidence=0))
        assert result.valid is False
        assert any("evidence packet is empty" in r for r in result.reasons)

    def test_multiple_valid_citations(self):
        result = validate("Point one [1]. Point two [2].", _packet())
        assert result.valid is True
        assert result.citations_used == [1, 2]


class TestRetryFallbackConstants:
    def test_retry_instruction_is_nonempty_and_distinct_from_fallback(self):
        assert isinstance(RETRY_INSTRUCTION, str) and RETRY_INSTRUCTION.strip()
        assert isinstance(SAFE_FALLBACK_RESPONSE, str) and SAFE_FALLBACK_RESPONSE.strip()
        assert RETRY_INSTRUCTION != SAFE_FALLBACK_RESPONSE

    def test_safe_fallback_does_not_fabricate_a_citation(self):
        # The fallback must itself pass as "no evidence claimed" -- it
        # should never contain a bracketed citation number, since none of
        # the underlying content was actually verified as grounded.
        result = validate(SAFE_FALLBACK_RESPONSE, _packet())
        assert result.citations_used == []


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
