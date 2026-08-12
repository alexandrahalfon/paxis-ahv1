"""
Tests for care-team instruction injection into patient chat generation
(2026-08-12 beta audit item 5): "care-team-specific instruction > generic
education" was one of the architecture's central rules but had no
downstream consumer -- patient_state_service already captured the
instructions and evidence_packet_builder.summarize_context() now surfaces
them (see test_evidence_packet_builder.py), but nothing put the actual
instruction TEXT in front of the model. These tests cover
PatientChatService._care_team_instructions_block() directly, and that
answer() includes it in the system prompt regardless of whether the
patient also has a legacy physician-linked record.
"""

from __future__ import annotations

import pytest

from src.api.services.patient_portal.patient_chat_service import PatientChatService


class TestCareTeamInstructionsBlock:
    def test_empty_when_no_context(self):
        assert PatientChatService._care_team_instructions_block(None) == ""
        assert PatientChatService._care_team_instructions_block({}) == ""

    def test_empty_when_no_instructions(self):
        context = {"state": {"care_team_instructions": []}}
        assert PatientChatService._care_team_instructions_block(context) == ""

    def test_renders_instruction_text_and_type(self):
        context = {
            "state": {
                "care_team_instructions": [
                    {"text": "Avoid grapefruit while on this regimen.", "type": "dietary"},
                    {"text": "Call us if your temperature is above 100.4F.", "type": "monitoring"},
                ],
            },
        }
        block = PatientChatService._care_team_instructions_block(context)
        assert "Avoid grapefruit while on this regimen." in block
        assert "[dietary]" in block
        assert "Call us if your temperature is above 100.4F." in block
        assert "[monitoring]" in block

    def test_states_precedence_over_generic_education(self):
        context = {"state": {"care_team_instructions": [{"text": "No NSAIDs.", "type": "medication"}]}}
        block = PatientChatService._care_team_instructions_block(context)
        assert "precedence" in block.lower()

    def test_entries_without_text_are_skipped(self):
        context = {"state": {"care_team_instructions": [{"type": "other"}, {"text": "", "type": "other"}]}}
        assert PatientChatService._care_team_instructions_block(context) == ""


@pytest.fixture
def service(monkeypatch):
    svc = PatientChatService()

    async def no_profile(patient_user_id):
        return None
    monkeypatch.setattr(
        "src.api.services.patient.patient_profile_service.get_patient_profile_service",
        lambda: type("S", (), {"get_by_user": staticmethod(no_profile)})(),
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


def _capturing_client(captured):
    class _Resp:
        class choices0:
            class message:
                content = "Here's an answer."
        choices = [choices0]

    class _Fake:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    captured["messages"] = kwargs.get("messages")
                    return _Resp()
    return _Fake()


class TestInstructionsReachTheSystemPromptThroughAnswer:
    """context is unconditionally fetched by answer() and the
    instructions block is appended after the linked/unlinked branching --
    these confirm it actually lands in what gets sent to the model."""

    @pytest.mark.asyncio
    async def test_unlinked_patient_with_profile_gets_instructions_in_prompt(self, service, monkeypatch):
        async def no_known_facts(self, patient_user_id, conversation_facts=None):
            return dict(conversation_facts or {})  # not linked
        monkeypatch.setattr(PatientChatService, "known_facts_for", no_known_facts)

        async def context_with_instructions(self, patient_user_id):
            return {
                "state": {
                    "care_team_instructions": [
                        {"text": "No NSAIDs while on this regimen.", "type": "medication"},
                    ],
                },
                "retrieval_features": {},
            }
        monkeypatch.setattr(
            "src.api.services.evidence.patient_context_service.PatientContextService.get_context",
            context_with_instructions,
        )

        captured = {}
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _capturing_client(captured))

        await service.answer(
            message="Thanks for the update.",
            patient_user_id="user-1",
            persist=False,
        )

        system_msg = captured["messages"][0]["content"]
        assert "No NSAIDs while on this regimen." in system_msg
        assert "precedence" in system_msg.lower()

    @pytest.mark.asyncio
    async def test_linked_patient_also_gets_instructions_from_new_profile_model(self, service, monkeypatch):
        """A patient can have BOTH a legacy physician-linked record
        (facts["linked"] True, which drives the old summary branch) AND
        their own patient_profile with care_team_instructions recorded
        against it (the new model). The old branch never looks at
        `context`, so this is the case that would have silently dropped
        instructions if the fix were nested inside `elif context:`
        instead of being unconditional."""
        async def linked_facts(self, patient_user_id, conversation_facts=None):
            facts = dict(conversation_facts or {})
            facts["linked"] = True
            facts["cancer_type"] = "oral tongue SCC"
            return facts
        monkeypatch.setattr(PatientChatService, "known_facts_for", linked_facts)

        async def context_with_instructions(self, patient_user_id):
            return {
                "state": {
                    "care_team_instructions": [
                        {"text": "Call us if fever above 100.4F.", "type": "monitoring"},
                    ],
                },
                "retrieval_features": {},
            }
        monkeypatch.setattr(
            "src.api.services.evidence.patient_context_service.PatientContextService.get_context",
            context_with_instructions,
        )

        captured = {}
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _capturing_client(captured))

        await service.answer(
            message="Thanks for the update.",
            patient_user_id="user-1",
            persist=False,
        )

        system_msg = captured["messages"][0]["content"]
        assert "Call us if fever above 100.4F." in system_msg
        # The old-branch summary should still be present too -- this is
        # additive, not a replacement of the existing personalization.
        assert "oral tongue SCC" in system_msg

    @pytest.mark.asyncio
    async def test_no_instructions_adds_no_block(self, service, monkeypatch):
        async def no_known_facts(self, patient_user_id, conversation_facts=None):
            return dict(conversation_facts or {})
        monkeypatch.setattr(PatientChatService, "known_facts_for", no_known_facts)

        async def empty_context(self, patient_user_id):
            return {}
        monkeypatch.setattr(
            "src.api.services.evidence.patient_context_service.PatientContextService.get_context",
            empty_context,
        )

        captured = {}
        monkeypatch.setattr(PatientChatService, "_client", lambda self: _capturing_client(captured))

        await service.answer(
            message="Thanks for the update.",
            patient_user_id="user-1",
            persist=False,
        )

        system_msg = captured["messages"][0]["content"]
        assert "CARE TEAM INSTRUCTIONS" not in system_msg


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
