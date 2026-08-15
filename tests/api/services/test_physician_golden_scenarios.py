"""
Physician golden eval scenarios + state-sensitivity scenarios
(2026-08-12 convergence Sprint D items 23-24).

Same spirit as test_patient_golden_scenarios.py (Sprint D item 22): a
curated, named checklist run end to end through the REAL
answer_physician_query() orchestrator, with fakes only at the true I/O
boundary (authorization, patient state lookup, the legacy retriever,
extra-corpora search, and the OpenAI client). Not a duplicate of
test_physician_rag_orchestrator.py's wiring-level tests or test_
physician_grounding_gate.py's gate-level tests -- this is one place
that reads as "here is what the physician pipeline must do for these
representative real situations."

Item 24, specifically: three scenarios proving patient STATE actually
changes pipeline behavior, not just that passing it doesn't crash --
same question and same evidence, different patient_profile_id/state,
different packet/score/prompt content. This is the entire point of
building physician_context_service.py, _patient_values_for_scoring(),
and the incompatibility taxonomy (C13/C16/C17) rather than a single
audience-blind retrieval path.
"""

from __future__ import annotations

import pytest

from src.api.services.patient import patient_care_team_service as ctm
from src.api.services.physician import physician_rag_orchestrator as orch
from src.api.services.physician.physician_rag_orchestrator import (
    ACCESS_DENIED_RESPONSE,
    answer_physician_query,
)


# ── Shared fakes (deliberately duplicated, not imported, from the
# sibling test files -- matching this codebase's existing per-test-file
# independence convention, see e.g. physician_grounding_gate's own
# _queued_client duplicated identically in test_physician_rag_
# orchestrator.py). ──────────────────────────────────────────────────────

class _FakeStudy:
    def __init__(self, doc_id, title, chunks, citation=None, year=2024):
        self.doc_id = doc_id
        self.title = title
        self.chunks = chunks
        self.citation = citation
        self.year = year


class _FakeResult:
    def __init__(self, studies):
        self.studies = studies


class _FakeRetriever:
    def __init__(self, studies=None):
        self._studies = studies or []

    async def retrieve_comprehensive(self, **kwargs):
        return _FakeResult(self._studies)


def _chunk(text, **overrides):
    base = {
        "text": text, "doc_id": "doc-1", "section": "results", "chunk_id": 0,
        "point_id": "pt-1", "doc_meta": {"title": "Osimertinib in EGFR-mutant NSCLC"},
        "score_dense": 0.8, "score_lexical": 0.5, "score_crossencoder_gate": 0.9,
    }
    base.update(overrides)
    return base


def _study(text="Osimertinib showed a PFS benefit in EGFR-mutant NSCLC patients."):
    return _FakeStudy(doc_id="doc-1", title="Osimertinib in EGFR-mutant NSCLC", chunks=[_chunk(text)])


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


_GROUNDED_ANSWER = "Osimertinib showed a PFS benefit in this population [1]."


# ── D23: golden scenarios ────────────────────────────────────────────────

class TestGeneralQuestionNoPatientInView:
    @pytest.mark.asyncio
    async def test_answers_normally_with_no_patient_context(self):
        client, _ = _queued_client([_GROUNDED_ANSWER])
        result = await answer_physician_query(
            "phys-1", "What are the treatment options for EGFR-mutant NSCLC?",
            client=client, retriever=_FakeRetriever(studies=[_study()]),
        )
        assert result.authorized is True
        assert result.sources_valid is True
        assert result.answer == _GROUNDED_ANSWER
        assert result.packet["selected_patient_context"] in (None, {})


class TestAuthorizationDeniedIsAGoldenSafetyScenario:
    @pytest.mark.asyncio
    async def test_denied_access_returns_the_generic_response(self, monkeypatch):
        async def deny(*a, **k):
            return False
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", deny)

        result = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="not-my-patient",
            retriever=_FakeRetriever(studies=[_study()]),
        )
        assert result.answer == ACCESS_DENIED_RESPONSE
        assert result.authorized is False
        assert result.sources_valid is False


class TestTherapySelectionWithEvidenceIsGrounded:
    @pytest.mark.asyncio
    async def test_grounded_first_try_no_retry(self):
        # therapy_selection is a STRICT_VALIDATION_INTENT (claim_grounding_
        # validator.py) -- a second, JSON-shaped response is queued for
        # that claim-level check ("no unsupported claims found"), so this
        # scenario demonstrates the real claim-validation pass rather than
        # its unrelated malformed-JSON fail-open branch.
        client, calls = _queued_client([_GROUNDED_ANSWER, '{"claims": []}'])
        result = await answer_physician_query(
            "phys-1", "What are the options for EGFR-mutant NSCLC?", intent="therapy_selection",
            client=client, retriever=_FakeRetriever(studies=[_study()]),
        )
        assert result.answer == _GROUNDED_ANSWER
        assert result.sources_valid is True
        assert calls["count"] == 2  # mechanical grounding pass (no retry) + claim validation pass


class TestZeroEvidenceNeverGatesAPhysicianAnswer:
    @pytest.mark.asyncio
    async def test_empty_retrieval_result_passes_through_ungated(self):
        client, _ = _queued_client(["I don't have specific evidence on that."])
        result = await answer_physician_query(
            "phys-1", "General clinical question.", client=client, retriever=_FakeRetriever(studies=[]),
        )
        assert result.answer == "I don't have specific evidence on that."
        assert result.sources_valid is True


# ── D23: claim-level golden scenarios ────────────────────────────────────

class TestToxicityManagementUnsupportedClaimIsRepaired:
    @pytest.mark.asyncio
    async def test_unsupported_dosing_claim_is_mechanically_narrowed(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv
        from src.api.services.evidence.claim_grounding_validator import (
            ClaimAssessment, ClaimValidationResult, UNSUPPORTED,
        )

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(
                    claim="Reduce the dose by 50% for grade 3 diarrhea [1].",
                    support_level=UNSUPPORTED, reason="Dose-reduction percentage not stated in the passage.",
                ),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)

        client, _ = _queued_client([
            "Reduce the dose by 50% for grade 3 diarrhea [1]. Monitor closely and follow up in one week.",
        ])
        result = await answer_physician_query(
            "phys-1", "How should I manage grade 3 diarrhea on this regimen?",
            intent="toxicity_management", client=client, retriever=_FakeRetriever(studies=[_study()]),
        )
        assert "Reduce the dose by 50%" not in result.answer
        assert "Monitor closely and follow up in one week." in result.answer
        assert result.sources_valid is True


class TestTrialEligibilityUnrepairableClaimFallsBack:
    @pytest.mark.asyncio
    async def test_unrepairable_claim_falls_back_to_the_safe_response(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv
        from src.api.services.evidence.claim_grounding_validator import (
            ClaimAssessment, ClaimValidationResult, UNSUPPORTED,
        )
        from src.api.services.evidence.grounding_validator import SAFE_FALLBACK_RESPONSE

        async def fake_validate(answer, packet, **kwargs):
            return ClaimValidationResult(claims=[
                ClaimAssessment(claim="A sentence not present in the answer.", support_level=UNSUPPORTED),
            ], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", fake_validate)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        result = await answer_physician_query(
            "phys-1", "Is this patient eligible for the trial?",
            intent="trial_eligibility", client=client, retriever=_FakeRetriever(studies=[_study()]),
        )
        assert result.answer == SAFE_FALLBACK_RESPONSE
        assert result.sources_valid is False


# ── D24: state-sensitivity scenarios ─────────────────────────────────────

class TestPatientInViewChangesThePacket:
    @pytest.mark.asyncio
    async def test_packet_carries_patient_context_only_when_a_patient_is_linked(self, monkeypatch):
        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        async def fake_state(patient_profile_id):
            return {"active_diagnoses": [{"cancer_site": "lung", "histology": "adenocarcinoma"}]}
        monkeypatch.setattr(orch, "_get_patient_state", fake_state)

        client_no_patient, _ = _queued_client([_GROUNDED_ANSWER])
        no_patient = await answer_physician_query(
            "phys-1", "What are the options?", client=client_no_patient,
            retriever=_FakeRetriever(studies=[_study()]),
        )

        client_with_patient, _ = _queued_client([_GROUNDED_ANSWER])
        with_patient = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-1",
            client=client_with_patient, retriever=_FakeRetriever(studies=[_study()]),
        )

        assert no_patient.packet["patient_snapshot_id"] is None
        assert with_patient.packet["patient_snapshot_id"] == "patient-1"
        assert with_patient.packet["selected_patient_context"].get("active_diagnoses")
        assert not (no_patient.packet["selected_patient_context"] or {}).get("active_diagnoses")


class TestBiomarkerStateChangesTheApplicabilityScore:
    @pytest.mark.asyncio
    async def test_matching_biomarker_in_state_scores_higher_than_a_non_matching_one(self, monkeypatch):
        """Legacy-retrieved candidates carry no structured applicability_
        meta tags yet (clinical_retrieval_adapter.py's documented gap) --
        _set_match() falls back to literal text containment. A patient
        whose recorded biomarker term appears in the chunk text scores
        that axis 1.0; one whose term does not appear scores a neutral
        0.5, not a hard mismatch (which needs a real tag -- see the next
        scenario for that path)."""
        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        text = "Osimertinib showed a PFS benefit in EGFR-mutant NSCLC patients."

        async def egfr_state(patient_profile_id):
            return {"biomarkers": [{"biomarker_name": "EGFR"}]}

        async def kras_state(patient_profile_id):
            return {"biomarkers": [{"biomarker_name": "KRAS G12C"}]}

        monkeypatch.setattr(orch, "_get_patient_state", egfr_state)
        # therapy_selection triggers claim-level validation (a strict
        # intent) -- a second, JSON-shaped response for that pass so this
        # scenario exercises the real check rather than its unrelated
        # malformed-JSON fail-open branch.
        client_a, _ = _queued_client([_GROUNDED_ANSWER, '{"claims": []}'])
        matching = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-egfr",
            intent="therapy_selection", client=client_a,
            retriever=_FakeRetriever(studies=[_study(text)]),
        )

        monkeypatch.setattr(orch, "_get_patient_state", kras_state)
        client_b, _ = _queued_client([_GROUNDED_ANSWER, '{"claims": []}'])
        mismatched = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-kras",
            intent="therapy_selection", client=client_b,
            retriever=_FakeRetriever(studies=[_study(text)]),
        )

        matching_score = matching.packet["evidence"][0]["applicability_score"]
        mismatched_score = mismatched.packet["evidence"][0]["applicability_score"]
        assert matching_score > mismatched_score
        matching_components = matching.packet["evidence"][0]["score_components"]
        mismatched_components = mismatched.packet["evidence"][0]["score_components"]
        assert matching_components["biomarker"] == 1.0
        assert mismatched_components["biomarker"] == 0.5  # neutral, not a hard mismatch -- see docstring

    @pytest.mark.asyncio
    async def test_a_structurally_tagged_mismatch_from_extra_corpora_triggers_the_hard_penalty(self, monkeypatch):
        """The hard incompatibility path DOES fire end to end once a
        candidate actually carries structured applicability_meta tags --
        today that only comes from the newer, classified corpora via
        multi_corpus_retriever (include_extra_corpora=True), not the
        legacy retriever. This is the reachable path for it."""
        from src.api.services.evidence import multi_corpus_retriever as mcr

        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        async def kras_state(patient_profile_id):
            return {"biomarkers": [{"biomarker_name": "KRAS G12C"}]}
        monkeypatch.setattr(orch, "_get_patient_state", kras_state)

        tagged_candidate = {
            "doc_id": "guideline-1", "title": "EGFR-targeted therapy guideline",
            "text": "Guideline recommendation for EGFR-mutant disease.",
            "collection": "oncology_clinical_guidelines", "citation": "NCCN, 2025", "year": 2025,
            "metadata": {"applicability_meta": {"biomarkers": ["EGFR"]}},
        }

        async def fake_search(query_text, plan, *, audience="patient"):
            return [tagged_candidate]
        monkeypatch.setattr(mcr, "search", fake_search)

        client, _ = _queued_client([_GROUNDED_ANSWER, '{"claims": []}'])
        result = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-kras",
            intent="therapy_selection", client=client,
            retriever=_FakeRetriever(studies=[]),
            include_extra_corpora=True,
        )

        entry = next(e for e in result.packet["evidence"] if e["title"] == "EGFR-targeted therapy guideline")
        assert entry["score_components"]["hard_incompatibility"] is True
        assert entry["incompatibility_reasons"]


class TestCareTeamInstructionPrecedenceReachesTheSystemPrompt:
    @pytest.mark.asyncio
    async def test_instruction_appears_in_the_prompt_only_when_present_in_state(self, monkeypatch):
        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        async def with_instruction(patient_profile_id):
            return {"care_team_instructions": [{"text": "Avoid NSAIDs on this regimen.", "type": "medication"}]}
        monkeypatch.setattr(orch, "_get_patient_state", with_instruction)

        # therapy_selection triggers claim-level validation (a strict
        # intent) -- a second, JSON-shaped response queued for that pass
        # in both calls below, purely for a clean test signal (the
        # assertions only read the FIRST call's system prompt either way).
        client, calls = _queued_client([_GROUNDED_ANSWER, '{"claims": []}'])
        await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-1",
            # care_team_instructions is only in PHYSICIAN_CONTEXT_POLICY's
            # therapy_selection list (physician_context_service.py) --
            # required here so select_physician_context() actually
            # surfaces it for this scenario.
            intent="therapy_selection",
            client=client, retriever=_FakeRetriever(studies=[_study()]),
        )
        system_prompt = calls["messages"][0][0]["content"]
        assert "VALIDATED CARE-TEAM INSTRUCTIONS" in system_prompt
        assert "Avoid NSAIDs on this regimen." in system_prompt

        async def without_instruction(patient_profile_id):
            return {"active_diagnoses": [{"cancer_site": "lung"}]}
        monkeypatch.setattr(orch, "_get_patient_state", without_instruction)

        client2, calls2 = _queued_client([_GROUNDED_ANSWER, '{"claims": []}'])
        await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-2",
            intent="therapy_selection",
            client=client2, retriever=_FakeRetriever(studies=[_study()]),
        )
        system_prompt2 = calls2["messages"][0][0]["content"]
        assert "VALIDATED CARE-TEAM INSTRUCTIONS" not in system_prompt2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
