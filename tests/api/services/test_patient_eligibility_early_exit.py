"""
Tests for the early-exit behavior in run_patient_eligibility_check()
when has_patient_context is False.

Validates: Requirements 1.5
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from src.api.services.patient_eligibility_boost_service import (
    run_patient_eligibility_check,
)


@pytest.fixture
def dummy_chunks():
    return [
        {"doc_id": "study_001", "title": "Study A", "text": "Some text", "score": 0.9},
        {"doc_id": "study_002", "title": "Study B", "text": "Other text", "score": 0.8},
        {"doc_id": "study_003", "title": "Study C", "text": "More text", "score": 0.7},
    ]


@pytest.fixture
def mock_openai_client():
    return MagicMock()


@pytest.mark.asyncio
async def test_early_exit_when_has_patient_context_false(dummy_chunks, mock_openai_client):
    """When has_patient_context=False, studies are returned unmodified."""
    original_chunks = list(dummy_chunks)  # shallow copy for comparison
    result_chunks, metadata = await run_patient_eligibility_check(
        query="What is SBRT?",
        chunks=dummy_chunks,
        openai_client=mock_openai_client,
        has_patient_context=False,
    )
    assert result_chunks is dummy_chunks, "Chunks should be returned as-is (same object)"
    assert len(result_chunks) == len(original_chunks)
    assert metadata["patient_context_detected"] is False


@pytest.mark.asyncio
async def test_no_early_exit_when_has_patient_context_none(mock_openai_client):
    """When has_patient_context is not passed (None), existing behavior is preserved."""
    # With an empty query and no chunks, the function should still proceed
    # to extract_patient_context_from_query and return no-context metadata.
    result_chunks, metadata = await run_patient_eligibility_check(
        query="",
        chunks=[],
        openai_client=mock_openai_client,
        has_patient_context=None,
    )
    assert metadata["patient_context_detected"] is False


@pytest.mark.asyncio
async def test_early_exit_preserves_all_chunk_scores(dummy_chunks, mock_openai_client):
    """Early exit must not modify any chunk scores."""
    original_scores = [c["score"] for c in dummy_chunks]
    result_chunks, _ = await run_patient_eligibility_check(
        query="Define IMRT",
        chunks=dummy_chunks,
        openai_client=mock_openai_client,
        has_patient_context=False,
    )
    result_scores = [c["score"] for c in result_chunks]
    assert result_scores == original_scores
