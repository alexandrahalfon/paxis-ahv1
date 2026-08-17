"""
Tests for source_governance.py (2026-08-12 convergence Sprint B item 8):
enforcing the evidence_sources registry's active/patient_facing/
allowed_intents fields at RETRIEVAL time, not just at ingestion time
(enforce_domain(), covered separately in
test_source_domain_enforcement.py).
"""

from __future__ import annotations

import pytest

from src.api.services.evidence.source_governance import (
    enforce_source_governance,
    filter_by_source_governance,
)


def _candidate(source_key, **overrides):
    base = {"source_key": source_key, "title": "Some passage", "text": "..."}
    base.update(overrides)
    return base


def _source(**overrides):
    base = {"source_key": "nci", "active": True, "patient_facing": True, "allowed_intents": []}
    base.update(overrides)
    return base


class TestUnregisteredSourcesPassThrough:
    def test_candidate_with_no_source_key_passes_through(self):
        candidates = [{"title": "x", "text": "y"}]
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key={})
        assert out == candidates

    def test_candidate_with_unregistered_source_key_passes_through(self):
        """The existing literature corpus predates source_registry
        adoption entirely -- its candidates must not be silently
        dropped just because they were never registered."""
        candidates = [_candidate("exueed_legacy_literature")]
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key={})
        assert out == candidates


class TestActiveEnforcement:
    def test_inactive_source_is_excluded(self):
        candidates = [_candidate("nci")]
        sources = {"nci": _source(active=False)}
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert out == []

    def test_active_source_passes(self):
        candidates = [_candidate("nci")]
        sources = {"nci": _source(active=True)}
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert out == candidates

    def test_missing_active_field_defaults_to_included(self):
        candidates = [_candidate("nci")]
        sources = {"nci": {"source_key": "nci"}}  # no "active" key at all
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert out == candidates


class TestPatientFacingEnforcement:
    def test_non_patient_facing_source_excluded_for_patient_audience(self):
        candidates = [_candidate("asco_professional")]
        sources = {"asco_professional": _source(source_key="asco_professional", patient_facing=False)}
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert out == []

    def test_non_patient_facing_source_included_for_physician_audience(self):
        candidates = [_candidate("asco_professional")]
        sources = {"asco_professional": _source(source_key="asco_professional", patient_facing=False)}
        out = filter_by_source_governance(candidates, audience="physician", intent=None, sources_by_key=sources)
        assert out == candidates

    def test_missing_patient_facing_field_defaults_to_included(self):
        candidates = [_candidate("nci")]
        sources = {"nci": {"source_key": "nci", "active": True}}
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert out == candidates


class TestAllowedIntentsEnforcement:
    def test_source_scoped_to_other_intents_is_excluded(self):
        candidates = [_candidate("dailymed")]
        sources = {"dailymed": _source(source_key="dailymed", allowed_intents=["medication_explainer"])}
        out = filter_by_source_governance(
            candidates, audience="patient", intent="nutrition", sources_by_key=sources,
        )
        assert out == []

    def test_source_scoped_to_this_intent_passes(self):
        candidates = [_candidate("dailymed")]
        sources = {"dailymed": _source(source_key="dailymed", allowed_intents=["medication_explainer"])}
        out = filter_by_source_governance(
            candidates, audience="patient", intent="medication_explainer", sources_by_key=sources,
        )
        assert out == candidates

    def test_empty_allowed_intents_means_unrestricted(self):
        candidates = [_candidate("nci")]
        sources = {"nci": _source(allowed_intents=[])}
        out = filter_by_source_governance(
            candidates, audience="patient", intent="anything_at_all", sources_by_key=sources,
        )
        assert out == candidates

    def test_no_intent_given_does_not_filter_on_allowed_intents(self):
        candidates = [_candidate("dailymed")]
        sources = {"dailymed": _source(source_key="dailymed", allowed_intents=["medication_explainer"])}
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert out == candidates


class TestMixedCandidates:
    def test_only_the_violating_candidate_is_dropped(self):
        candidates = [
            _candidate("nci", title="allowed"),
            _candidate("deactivated_source", title="dropped"),
        ]
        sources = {
            "nci": _source(active=True),
            "deactivated_source": _source(source_key="deactivated_source", active=False),
        }
        out = filter_by_source_governance(candidates, audience="patient", intent=None, sources_by_key=sources)
        assert [c["title"] for c in out] == ["allowed"]


class TestEnforceSourceGovernanceAsyncWrapper:
    @pytest.mark.asyncio
    async def test_loads_all_sources_including_inactive(self, monkeypatch):
        """active_only=False is load-bearing -- an inactive source must
        be VISIBLE to the filter so it can be excluded, not silently
        absent from what list_sources() returns."""
        list_sources_calls = []

        class FakeRegistry:
            async def list_sources(self, active_only=True):
                list_sources_calls.append(active_only)
                return [{"source_key": "nci", "active": False, "patient_facing": True, "allowed_intents": []}]

        monkeypatch.setattr(
            "src.api.services.evidence.source_registry.get_source_registry",
            lambda: FakeRegistry(),
        )

        candidates = [_candidate("nci")]
        out = await enforce_source_governance(candidates, audience="patient", intent=None)

        assert list_sources_calls == [False]
        assert out == []

    @pytest.mark.asyncio
    async def test_unregistered_candidate_passes_through_the_full_async_path(self, monkeypatch):
        class FakeRegistry:
            async def list_sources(self, active_only=True):
                return []

        monkeypatch.setattr(
            "src.api.services.evidence.source_registry.get_source_registry",
            lambda: FakeRegistry(),
        )

        candidates = [_candidate("some_unregistered_source")]
        out = await enforce_source_governance(candidates)
        assert out == candidates


class TestDefaultSourcesCoverRealPatientQuestions:
    """Regression test for the bug where DEFAULT_SOURCES itself (not the
    generic filter logic the rest of this file exercises against
    synthetic sources) was misconfigured: "nci"'s allowed_intents omitted
    treatment_explainer and medication_explainer, even though
    scripts/ingest_nci_cancer_types.py ingests NCI PDQ "Treatment"
    summaries (e.g. "Breast Cancer Treatment") under source_key="nci" --
    the only source those ingestion scripts populate today. Every
    candidate from "nci" was silently dropped by
    filter_by_source_governance() for any patient question classified as
    treatment_explainer or medication_explainer (by far the most common
    shape of patient question -- "what are the side effects of my
    treatment", "tell me about chemotherapy"), so the multi-corpus
    retriever came back empty and patient_chat_service.answer() /
    patient_query.py both fell back to the clinician literature corpus
    (exueed_kb_latest) instead of ever reaching the patient-education KB.
    This exercises the actual configured DEFAULT_SOURCES data against the
    real classify_intent(), so that regression can't come back silently.
    """

    @pytest.mark.parametrize("question", [
        "What are the side effects of my treatment?",
        "Tell me about chemotherapy",
        "What does immunotherapy do?",
        "What medication will I be on?",
        "What should I eat during chemo?",
        "Is this fatigue normal?",
        "What does my biomarker report mean?",
    ])
    def test_nci_survives_governance_for_common_patient_questions(self, question):
        from src.api.services.evidence.patient_context_service import classify_intent
        from src.api.services.evidence.source_registry import DEFAULT_SOURCES

        nci = next(s for s in DEFAULT_SOURCES if s["source_key"] == "nci")
        sources = {"nci": _source(allowed_intents=nci["allowed_intents"])}
        candidates = [_candidate("nci")]

        intent = classify_intent(question)
        out = filter_by_source_governance(
            candidates, audience="patient", intent=intent, sources_by_key=sources,
        )
        assert out == candidates, (
            f"{question!r} classified as intent={intent!r}, but the 'nci' source "
            f"(the only patient-education source actually populated by "
            f"scripts/ingest_nci_cancer_types.py / ingest_nci_supportive_care.py) "
            f"was excluded -- DEFAULT_SOURCES['nci']['allowed_intents'] "
            f"({nci['allowed_intents']}) doesn't cover it, so this question would "
            f"silently fall back to the clinician literature corpus."
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
