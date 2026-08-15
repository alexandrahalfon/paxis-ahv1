"""
Staging integration / wiring checks (2026-08-12 convergence Sprint D
item 25).

Not a behavioral test of any one endpoint -- those are covered
extensively elsewhere (tests/api/routes, tests/api/services). This file
answers a different, narrower question: "would the assembled FastAPI
app this session's changes produced actually come up in staging?" --
the whole `src.api.main` module imports cleanly (catches a circular
import or a syntax error anywhere in the dependency graph the many new
convergence modules now sit in), the new physician-beta route is
actually mounted where main.py claims it is, no two routers accidentally
claim the same (method, path) pair, and the new opt-in feature flags
this session added default to OFF -- an upgrade must not silently
change existing behavior (the standing "additive only" rule this whole
program has followed).

Route introspection goes through app.openapi()["paths"] rather than
walking app.routes directly: newer FastAPI/Starlette versions wrap each
include_router() call in an internal `_IncludedRouter` object instead of
flattening its routes into app.routes, so a raw app.routes walk silently
undercounts depending on which FastAPI version is installed.
app.openapi() is the stable, version-independent way to ask "what does
this app actually serve," which is exactly the question this file asks.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def app():
    """Imports src.api.main once for this module. A failure here means
    the app itself would fail to start -- the single most important
    staging-readiness signal this file checks."""
    import src.api.main as main_module
    return main_module.app


@pytest.fixture(scope="module")
def openapi_paths(app):
    return app.openapi()["paths"]


class TestAppAssemblesCleanly:
    def test_main_module_imports_without_error(self, app):
        assert app is not None

    def test_app_serves_a_substantial_number_of_endpoints(self, openapi_paths):
        # A sanity floor, not an exact count -- this just catches the
        # catastrophic case (include_router calls silently no-op'ing, an
        # empty router list) rather than pinning an exact number that
        # would need updating on every unrelated route addition.
        assert len(openapi_paths) > 50


class TestPhysicianBetaRouteIsMounted:
    def test_query_endpoint_is_registered_under_api_prefix(self, openapi_paths):
        assert "/api/physician-beta/query" in openapi_paths

    def test_query_endpoint_accepts_post(self, openapi_paths):
        assert "post" in openapi_paths["/api/physician-beta/query"]


class TestNoRouteCollisions:
    def test_every_method_path_pair_maps_to_exactly_one_operation(self, app):
        """Two routers accidentally defining the same (method, path) is a
        real regression class in an app this size with this many router
        files -- FastAPI doesn't error on it, it just silently shadows
        one handler with another. app.routes (unlike the OpenAPI schema,
        which only ever has one operation per method+path by
        construction) is where a real collision would actually be
        visible: two distinct route objects both matching the same
        incoming request. Recurses into the newer `_IncludedRouter`
        wrapper (see module docstring) so this check covers every
        router regardless of FastAPI version."""
        def _flatten(routes):
            for r in routes:
                nested = getattr(r, "original_router", None)
                if nested is not None:
                    yield from _flatten(nested.routes)
                else:
                    yield r

        seen: dict[tuple[str, str], str] = {}
        collisions = []
        for r in _flatten(app.routes):
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None)
            if not path or not methods:
                continue
            name = getattr(r, "name", None) or str(r)
            for m in methods:
                key = (m, path)
                if key in seen and seen[key] != name:
                    collisions.append((key, seen[key], name))
                else:
                    seen[key] = name
        assert collisions == [], f"duplicate (method, path) route registrations: {collisions}"


class TestNewFeatureFlagsDefaultToOff:
    """An upgrade must never silently change existing behavior -- every
    opt-in capability this convergence program added stays off until a
    deployment explicitly turns it on."""

    def test_physician_rag_beta_defaults_off(self):
        from src.core.config import settings
        assert settings.physician_rag_beta_enabled is False

    def test_patient_pubmed_fallback_defaults_off(self):
        from src.core.config import settings
        assert settings.patient_pubmed_fallback_enabled is False

    def test_gcs_requirement_defaults_off(self):
        from src.core.config import settings
        assert settings.require_gcs_for_patient_documents is False


class TestDoNotChangeDataclassShapesAreIntact:
    """CLAUDE.md section 10 ('Do not change'): StudyEvidence and
    ComprehensiveRetrievalResult may only gain fields, never lose or
    rename one -- clinical_retrieval_adapter.py (Sprint C item 15) reads
    every one of these by name via duck-typed getattr(), so a silent
    rename there would fail open (a missing field just defaults to
    None) rather than loudly, which is exactly the failure mode this
    test exists to catch instead."""

    def test_study_evidence_still_has_every_documented_field(self):
        import dataclasses
        from src.api.services.comprehensive_retrieval import StudyEvidence

        field_names = {f.name for f in dataclasses.fields(StudyEvidence)}
        required = {
            "doc_id", "title", "citation", "year", "category",
            "initial_score", "rerank_score", "chunks", "sections_covered",
            "source", "match_score", "match_breakdown", "axis_mismatches",
            "soft_score_normalized", "patient_match_score",
            "patient_match_breakdown", "evidence_type",
        }
        missing = required - field_names
        assert missing == set(), f"StudyEvidence lost documented field(s): {missing}"

    def test_comprehensive_retrieval_result_still_has_every_documented_field(self):
        import dataclasses
        from src.api.services.comprehensive_retrieval import ComprehensiveRetrievalResult

        field_names = {f.name for f in dataclasses.fields(ComprehensiveRetrievalResult)}
        required = {
            "studies", "total_chunks", "retrieval_time_ms",
            "phase1_qdrant_docs", "phase1_postgres_docs", "phase2_docs_searched",
            "query_structure", "expanded_query", "reconciled_structure",
        }
        missing = required - field_names
        assert missing == set(), f"ComprehensiveRetrievalResult lost documented field(s): {missing}"

    def test_retrieve_comprehensive_signature_is_unchanged(self):
        """A literal characterization of the method's public signature --
        catches a future edit that renames, reorders, or removes a
        parameter (which would be a breaking change for every existing
        caller: patient_query.py, enhanced_rag_service.py, and now
        physician_rag_orchestrator.py) before it ships, rather than
        relying on every caller's own tests to happen to notice."""
        import inspect
        from src.api.services.comprehensive_retrieval import ComprehensiveRetriever

        sig = inspect.signature(ComprehensiveRetriever.retrieve_comprehensive)
        params = sig.parameters
        assert list(params.keys())[:2] == ["self", "query_text"]
        assert params["max_studies"].default == 12
        assert params["chunks_per_study"].default == 8
        assert params["category"].default is None
        assert params["accumulated_context"].default is None
        assert params["conversation_context"].default is None
        assert params["clinical_profile"].default is None
        assert params["max_guidelines"].default == 5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
