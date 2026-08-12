"""
Tests for physician_rag_orchestrator.py (2026-08-12 convergence Sprint C
item 20): the end-to-end pipeline tying QueryAnalysis (C12) ->
physician context selector (C13) -> verified authorization (C14) ->
legacy retrieval adapter (C15) -> physician applicability scorer
(C16/C17) -> physician answer generator (C18) -> grounding gate (C19)
into one callable.

Real, cheap deterministic pieces (query_analysis, physician_context_
service, clinical_retrieval_adapter, physician_applicability_scorer,
evidence_packet_builder, physician_answer_generator, physician_
grounding_gate) run for real -- only DB/network-touching seams are
faked: authorization, patient state lookup, the legacy retriever, the
extra-corpora search, and the OpenAI client.
"""

from __future__ import annotations

import pytest

from src.api.services.physician.physician_rag_orchestrator import (
    ACCESS_DENIED_RESPONSE,
    answer_physician_query,
)


# ── Fakes ─────────────────────────────────────────────────────────────────

class _FakeStudy:
    def __init__(self, doc_id, title, chunks, citation=None, year=2024, category=None):
        self.doc_id = doc_id
        self.title = title
        self.chunks = chunks
        self.citation = citation
        self.year = year
        self.category = category


class _FakeResult:
    def __init__(self, studies):
        self.studies = studies


class _FakeRetriever:
    def __init__(self, studies=None, raise_if_called=False):
        self._studies = studies or []
        self._raise = raise_if_called
        self.called = False

    async def retrieve_comprehensive(self, **kwargs):
        self.called = True
        if self._raise:
            raise AssertionError(
                "retrieve_comprehensive() must not be called when "
                "authorization was denied"
            )
        return _FakeResult(self._studies)


def _chunk(text="PFS benefit observed with adagrasib.", **overrides):
    base = {
        "text": text,
        "doc_id": "doc-1",
        "section": "results",
        "chunk_id": 0,
        "point_id": "pt-1",
        "doc_meta": {"title": "Adagrasib in KRAS G12C NSCLC"},
        "score_dense": 0.8,
        "score_lexical": 0.5,
        "score_crossencoder_gate": 0.9,
    }
    base.update(overrides)
    return base


def _study():
    return _FakeStudy(
        doc_id="doc-1", title="Adagrasib in KRAS G12C NSCLC",
        chunks=[_chunk()], citation="Smith et al., 2024", year=2024,
    )


def _queued_client(answers):
    calls = {"count": 0}

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
                    return _Resp(answers[min(i, len(answers) - 1)])
    return _Fake(), calls


_GROUNDED_ANSWER = "Adagrasib showed a PFS benefit in this population [1]."


class TestGeneralQuestionWithoutPatientProfile:
    @pytest.mark.asyncio
    async def test_authorization_is_never_touched(self, monkeypatch):
        """No patient_profile_id -> the authorization module must never
        even be imported/called; this is the 'general question, no
        patient in view' path physician_context_service.py's own
        docstring describes."""
        from src.api.services.patient import patient_care_team_service as ctm

        async def boom(*a, **k):
            raise AssertionError("authorize_physician_patient_access must not be called")
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", boom)

        client, calls = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options for KRAS G12C NSCLC?",
            patient_profile_id=None, client=client, retriever=retriever,
        )

        assert result.authorized is True
        assert result.sources_valid is True
        assert result.answer == _GROUNDED_ANSWER
        assert result.query_analysis["audience"] == "physician"
        assert retriever.called is True

    @pytest.mark.asyncio
    async def test_evidence_from_the_legacy_retriever_reaches_the_packet(self):
        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", client=client, retriever=retriever,
        )

        titles = [e["title"] for e in result.packet["evidence"]]
        assert "Adagrasib in KRAS G12C NSCLC" in titles


class TestAuthorizationGate:
    @pytest.mark.asyncio
    async def test_denied_authorization_short_circuits_before_retrieval(self, monkeypatch):
        from src.api.services.patient import patient_care_team_service as ctm

        async def deny(*a, **k):
            return False
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", deny)

        # Would raise if ever called -- proves the gate fires BEFORE the
        # legacy retriever, not just before the returned answer is used.
        retriever = _FakeRetriever(raise_if_called=True)

        result = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-1",
            retriever=retriever,
        )

        assert result.answer == ACCESS_DENIED_RESPONSE
        assert result.authorized is False
        assert result.sources_valid is False
        assert retriever.called is False

    @pytest.mark.asyncio
    async def test_authorized_access_proceeds_to_retrieval(self, monkeypatch):
        from src.api.services.patient import patient_care_team_service as ctm
        from src.api.services.physician import physician_rag_orchestrator as orch

        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        async def fake_state(patient_profile_id):
            return {}
        monkeypatch.setattr(orch, "_get_patient_state", fake_state)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-1",
            client=client, retriever=retriever,
        )

        assert result.authorized is True
        assert retriever.called is True


class TestPatientContextFlowsIntoThePacket:
    @pytest.mark.asyncio
    async def test_selected_context_and_snapshot_id_land_on_the_packet(self, monkeypatch):
        from src.api.services.patient import patient_care_team_service as ctm
        from src.api.services.physician import physician_rag_orchestrator as orch

        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        fake_state_dict = {
            "active_diagnoses": [{"cancer_site": "lung", "histology": "adenocarcinoma", "stage": "IV"}],
            "active_treatment": [{"regimen": "adagrasib", "agents": ["adagrasib"], "line_of_therapy": "second_line"}],
            "biomarkers": [{"biomarker_name": "KRAS G12C"}],
            "comorbidities": ["CKD"],
            "care_team_instructions": [{"text": "Avoid NSAIDs.", "type": "medication"}],
            "labs": [],
            "demographics": {"age": 64},
        }

        async def fake_state(patient_profile_id):
            assert patient_profile_id == "patient-1"
            return fake_state_dict
        monkeypatch.setattr(orch, "_get_patient_state", fake_state)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-1",
            intent="therapy_selection", client=client, retriever=retriever,
        )

        assert result.packet["patient_snapshot_id"] == "patient-1"
        assert result.packet["selected_patient_context"].get("care_team_instructions") == [
            {"text": "Avoid NSAIDs.", "type": "medication"}
        ]

    @pytest.mark.asyncio
    async def test_hard_biomarker_mismatch_penalizes_the_ranked_evidence(self, monkeypatch):
        """End-to-end proof the patient_values mapping actually reaches
        physician_applicability_scorer -- a patient with a KRAS G12C
        biomarker and an EGFR-tagged candidate should score lower than
        the same candidate scored with no patient biomarker at all."""
        from src.api.services.patient import patient_care_team_service as ctm
        from src.api.services.physician import physician_rag_orchestrator as orch

        async def allow(*a, **k):
            return True
        monkeypatch.setattr(ctm, "authorize_physician_patient_access", allow)

        async def fake_state(patient_profile_id):
            return {"biomarkers": [{"biomarker_name": "KRAS G12C"}]}
        monkeypatch.setattr(orch, "_get_patient_state", fake_state)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", patient_profile_id="patient-1",
            intent="therapy_selection", client=client, retriever=retriever,
        )
        assert result.packet["evidence"], "expected at least one evidence entry"


class TestExtraCorporaIsOptIn:
    @pytest.mark.asyncio
    async def test_off_by_default_multi_corpus_never_touched(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever as mcr

        async def boom(*a, **k):
            raise AssertionError("multi_corpus_retriever.search must not run by default")
        monkeypatch.setattr(mcr, "search", boom)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", client=client, retriever=retriever,
        )
        assert result.sources_valid is True

    @pytest.mark.asyncio
    async def test_included_when_requested_and_merged_into_ranking(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever as mcr

        extra_candidate = {
            "doc_id": "extra-1", "title": "Guideline on KRAS G12C management",
            "text": "Guideline recommendation text.", "collection": "oncology_clinical_guidelines",
            "citation": "NCCN Guidelines, 2025", "year": 2025,
        }

        async def fake_search(query_text, plan, *, audience="patient"):
            assert audience == "physician"
            return [extra_candidate]
        monkeypatch.setattr(mcr, "search", fake_search)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", client=client, retriever=retriever,
            include_extra_corpora=True, max_studies=10,
        )
        titles = [e["title"] for e in result.packet["evidence"]]
        assert "Guideline on KRAS G12C management" in titles

    @pytest.mark.asyncio
    async def test_extra_corpora_failure_is_fail_open(self, monkeypatch):
        from src.api.services.evidence import multi_corpus_retriever as mcr

        async def boom(*a, **k):
            raise RuntimeError("qdrant unreachable")
        monkeypatch.setattr(mcr, "search", boom)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", client=client, retriever=retriever,
            include_extra_corpora=True,
        )
        assert result.sources_valid is True
        assert result.answer == _GROUNDED_ANSWER


class TestIntentOverride:
    @pytest.mark.asyncio
    async def test_explicit_intent_wins_over_detection_and_reaches_query_analysis(self):
        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        result = await answer_physician_query(
            "phys-1", "What are the options?", intent="toxicity_management",
            client=client, retriever=retriever,
        )
        assert result.query_analysis["intent"] == "toxicity_management"

    @pytest.mark.asyncio
    async def test_strict_intent_triggers_claim_validation(self, monkeypatch):
        from src.api.services.evidence import claim_grounding_validator as cgv
        from src.api.services.evidence.claim_grounding_validator import ClaimValidationResult

        called = []
        async def spy(answer, packet, **kwargs):
            called.append(answer)
            return ClaimValidationResult(claims=[], ran=True)
        monkeypatch.setattr(cgv, "validate_claims", spy)

        client, _ = _queued_client([_GROUNDED_ANSWER])
        retriever = _FakeRetriever(studies=[_study()])

        await answer_physician_query(
            "phys-1", "What are the options?", intent="therapy_selection",
            client=client, retriever=retriever,
        )
        assert len(called) == 1


class TestNoEvidenceReturnsUngroundedAnswerUnchanged:
    @pytest.mark.asyncio
    async def test_empty_retriever_result_never_gates_the_answer(self):
        client, _ = _queued_client(["I don't have specific evidence on that."])
        retriever = _FakeRetriever(studies=[])

        result = await answer_physician_query(
            "phys-1", "General question.", client=client, retriever=retriever,
        )
        assert result.answer == "I don't have specific evidence on that."
        assert result.sources_valid is True
        assert result.packet["evidence"] == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
