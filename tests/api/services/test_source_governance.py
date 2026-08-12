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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
