"""
Tests for the nested-event-loop bug fix in study profile persistence
(caught in review, 2026-08-12): CompleteDocumentProcessor.process_complete()
used to persist an extracted study profile to Postgres via
`asyncio.get_event_loop().run_until_complete(storage.store_study_profile(...))`
from inside its own synchronous Phase 10 block. Every known caller of
process_complete() is itself async and runs inside an already-running
event loop (document_processing_service.process_document() is awaited
from a FastAPI route/background task), so that call raised "This event
loop is already running" on every invocation -- caught by a bare
`except Exception` and logged as a generic "PostgreSQL storage failed",
silently dropping every study profile the flag was meant to store.

Storage now happens via the standalone async function
persist_study_profile_if_present(), called with a normal `await` by each
async caller after process_complete() returns. These tests exercise that
function directly -- run from inside a pytest-asyncio test, i.e. from
inside a real running event loop, which is exactly the condition that
broke the old inline call and is the direct regression proof here.
"""

from __future__ import annotations

import pytest

from src.processing.document_processor import persist_study_profile_if_present


class TestNoStudyProfilePresent:
    @pytest.mark.asyncio
    async def test_returns_none_and_does_not_touch_storage_when_absent(self, monkeypatch):
        def _forbidden():
            raise AssertionError("storage must not be touched when no study profile was extracted")
        monkeypatch.setattr(
            "src.api.services.study_profile_storage_service.get_study_profile_storage_service",
            _forbidden,
        )

        result = await persist_study_profile_if_present({}, doc_name="some-doc")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_study_profile_key_is_falsy(self):
        result = await persist_study_profile_if_present({"study_profile": None}, doc_name="some-doc")
        assert result is None
        result = await persist_study_profile_if_present({"study_profile": {}}, doc_name="some-doc")
        assert result is None


class TestPersistsWhenPresent:
    @pytest.mark.asyncio
    async def test_calls_store_study_profile_from_inside_a_running_event_loop(self, monkeypatch):
        """The actual regression proof: this test function IS an async
        coroutine running inside pytest-asyncio's event loop -- exactly
        the condition (a caller with an already-running loop) that made
        the old `run_until_complete()`-based implementation raise. A
        clean await here, with no RuntimeError, is the fix working."""
        calls = {}

        class FakeStorage:
            async def store_study_profile(self, doc_id, document_name, extracted_data, processing_duration=None):
                calls["doc_id"] = doc_id
                calls["document_name"] = document_name
                calls["extracted_data"] = extracted_data
                calls["processing_duration"] = processing_duration
                return 42

        monkeypatch.setattr(
            "src.api.services.study_profile_storage_service.get_study_profile_storage_service",
            lambda: FakeStorage(),
        )

        generated_files = {
            "study_profile": {"title": "A trial", "phase": "III"},
            "study_profile_processing_duration": 12.5,
        }
        result = await persist_study_profile_if_present(generated_files, doc_name="doc-123")

        assert result == 42
        assert generated_files["study_profile_id"] == 42
        assert calls["doc_id"] == "doc-123"
        assert calls["document_name"] == "doc-123"
        assert calls["extracted_data"] == {"title": "A trial", "phase": "III"}
        assert calls["processing_duration"] == 12.5

    @pytest.mark.asyncio
    async def test_storage_failure_is_caught_and_returns_none(self, monkeypatch):
        class FailingStorage:
            async def store_study_profile(self, **kwargs):
                raise RuntimeError("db unreachable")

        monkeypatch.setattr(
            "src.api.services.study_profile_storage_service.get_study_profile_storage_service",
            lambda: FailingStorage(),
        )

        generated_files = {"study_profile": {"title": "A trial"}}
        result = await persist_study_profile_if_present(generated_files, doc_name="doc-123")

        assert result is None
        assert "study_profile_id" not in generated_files


class TestProcessCompleteNeverCallsRunUntilComplete:
    def test_source_no_longer_contains_run_until_complete(self):
        """Static guard against the bug reappearing: process_complete()'s
        Phase 10 block must never call run_until_complete()/get_event_loop()
        again -- that combination is exactly what broke every async caller.
        Comment lines are stripped first since the method's own docstring/
        comments now explain the old bug using that exact call pattern as
        an example of what NOT to do."""
        import inspect
        from src.processing.document_processor import CompleteDocumentProcessor

        source = inspect.getsource(CompleteDocumentProcessor.process_complete)
        code_lines = [
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "run_until_complete" not in code_only
        assert "get_event_loop" not in code_only


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
