"""
Tests for physician_answer_generator.py (2026-08-12 convergence Sprint C
item 18): a physician-register generator sharing the EvidencePacket
contract (and its [n] citation numbering) with the patient path, but
with its own prompt -- plus PRECEDENCE_ORDER, the explicit care-team-
instruction-precedence mechanism (architecture review section 14).
"""

from __future__ import annotations

import pytest

from src.api.services.physician.physician_answer_generator import (
    PRECEDENCE_ORDER,
    build_precedence_directive,
    build_system_prompt,
    build_user_message,
    detect_policy_conflicts,
    generate,
)
from src.api.services.evidence.evidence_packet_builder import build_packet


def _evidence_candidate(**overrides):
    base = {
        "title": "Adagrasib in KRAS G12C NSCLC", "text": "PFS benefit observed.",
        "source_key": "nejm", "citation": "Smith et al., 2024", "year": 2024,
    }
    base.update(overrides)
    return base


class TestBuildPrecedenceDirective:
    def test_always_includes_the_full_ranking(self):
        directive = build_precedence_directive({})
        for i, name in enumerate(PRECEDENCE_ORDER, 1):
            assert f"{i}. {name.replace('_', ' ')}" in directive

    def test_includes_care_team_instructions_when_present_in_selected_context(self):
        packet = {
            "selected_patient_context": {
                "care_team_instructions": [{"text": "No NSAIDs on this regimen.", "type": "medication"}],
            },
        }
        directive = build_precedence_directive(packet)
        assert "No NSAIDs on this regimen." in directive
        assert "VALIDATED CARE-TEAM INSTRUCTIONS" in directive

    def test_falls_back_to_patient_context_when_selected_context_absent(self):
        packet = {
            "patient_context": {
                "care_team_instructions": [{"text": "Call if fever > 100.4F.", "type": "monitoring"}],
            },
        }
        directive = build_precedence_directive(packet)
        assert "Call if fever > 100.4F." in directive

    def test_no_instructions_section_when_none_present(self):
        directive = build_precedence_directive({"selected_patient_context": {}})
        assert "VALIDATED CARE-TEAM INSTRUCTIONS" not in directive

    def test_includes_safety_policy_when_present(self):
        directive = build_precedence_directive({"safety_policy": {"tier": "strict"}})
        assert "DETERMINISTIC SAFETY POLICY" in directive
        assert "strict" in directive

    def test_no_safety_policy_section_when_empty(self):
        directive = build_precedence_directive({"safety_policy": {}})
        assert "DETERMINISTIC SAFETY POLICY" not in directive

    def test_includes_interpretation_policies_when_present(self):
        directive = build_precedence_directive({
            "interpretation_policies": {"anc": "exact_value_and_trend_only"},
        })
        assert "LAB INTERPRETATION LIMITS" in directive
        assert "anc: exact_value_and_trend_only" in directive

    def test_no_interpretation_section_when_empty(self):
        directive = build_precedence_directive({"interpretation_policies": {}})
        assert "LAB INTERPRETATION LIMITS" not in directive


class TestDetectPolicyConflicts:
    def test_returns_empty_list_unconditionally(self):
        assert detect_policy_conflicts({"anything": "at all"}) == []
        assert detect_policy_conflicts({}) == []


class TestBuildSystemPrompt:
    def test_includes_both_the_base_prompt_and_the_precedence_directive(self):
        system = build_system_prompt({"safety_policy": {"tier": "strict"}})
        assert "clinical decision-support assistant" in system
        assert "PRECEDENCE" in system
        assert "strict" in system


class TestBuildUserMessage:
    def test_includes_the_question(self):
        message = build_user_message("What is the standard of care for KRAS G12C NSCLC?", {"evidence": []})
        assert "What is the standard of care for KRAS G12C NSCLC?" in message

    def test_no_evidence_block_when_packet_has_no_evidence(self):
        message = build_user_message("Q", {"evidence": []})
        assert "[1]" not in message

    def test_matches_to_prompt_block_numbering_exactly(self):
        packet = build_packet("Q", None, [_evidence_candidate()])
        message = build_user_message("What are the treatment options?", packet)
        assert "[1] Adagrasib in KRAS G12C NSCLC" in message
        assert "PFS benefit observed." in message
        assert "cite the number in brackets" in message.lower()


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]


def _client_returning(content: str):
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return _FakeResp(content)
    return _Client()


class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_the_models_content(self):
        packet = build_packet("Q", None, [_evidence_candidate()])
        client = _client_returning("Adagrasib showed a PFS benefit [1].")
        answer = await generate("What are the options?", packet, client=client)
        assert answer == "Adagrasib showed a PFS benefit [1]."

    @pytest.mark.asyncio
    async def test_passes_system_and_user_messages_to_the_client(self):
        captured = {}

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        captured.update(kwargs)
                        return _FakeResp("Answer.")

        packet = build_packet("Q", None, [_evidence_candidate()])
        await generate("What are the options?", packet, client=_Client())

        messages = captured["messages"]
        assert messages[0]["role"] == "system"
        assert "clinical decision-support assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "What are the options?" in messages[1]["content"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
