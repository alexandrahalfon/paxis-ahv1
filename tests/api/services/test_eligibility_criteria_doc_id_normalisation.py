"""
Tests for get_eligibility_criteria_by_doc_ids doc_id normalisation.

Qdrant doc_ids carry an `_<hex-hash>` suffix that
`display-study-details.studies.doc_id` does not. The batch method
must strip the suffix before matching, otherwise every lookup
silently returns empty (the failure mode observed in the live HNSCC
stress test on the feat/eligibility-axis-expansion branch).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.services.postgres_study_details_service import (
    PostgresStudyDetailsService,
)


class _FakeRow(dict):
    """asyncpg.Record-like adapter for unit tests."""

    def __getitem__(self, key):
        return super().__getitem__(key)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.last_query = None
        self.last_args = None

    async def fetch(self, query, *args):
        self.last_query = query
        self.last_args = args
        return self._rows


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


def _service_with_rows(rows):
    """Build a PostgresStudyDetailsService with _get_pool monkey-patched
    to return a fake pool that responds with `rows`."""
    svc = PostgresStudyDetailsService()
    conn = _FakeConn([_FakeRow(r) for r in rows])
    pool = _FakePool(conn)
    svc._get_pool = AsyncMock(return_value=pool)  # type: ignore[assignment]
    return svc, conn


@pytest.mark.asyncio
async def test_strips_8_hex_hash_suffix_from_doi_style_doc_id():
    """`doi_10.1200_jco.2007.15.0102_e07e6c6e` must be queried as
    `doi_10.1200_jco.2007.15.0102` against studies.doc_id."""
    svc, conn = _service_with_rows([
        {
            "doc_id": "doi_10.1200_jco.2007.15.0102",
            "inclusion": ["recurrent HNSCC"],
            "exclusion": ["unresectable"],
        }
    ])
    result = await svc.get_eligibility_criteria_by_doc_ids(
        ["doi_10.1200_jco.2007.15.0102_e07e6c6e"]
    )
    # Original (un-normalised) doc_id is the key in the result.
    assert "doi_10.1200_jco.2007.15.0102_e07e6c6e" in result
    assert result["doi_10.1200_jco.2007.15.0102_e07e6c6e"]["inclusion"] == ["recurrent HNSCC"]
    assert result["doi_10.1200_jco.2007.15.0102_e07e6c6e"]["exclusion"] == ["unresectable"]
    # The normalised doc_id was sent to the DB query.
    assert conn.last_args[0] == ["doi_10.1200_jco.2007.15.0102"]


@pytest.mark.asyncio
async def test_strips_hash_suffix_from_author_year_doc_id():
    """Non-DOI form (`maghami-et-al-2020-...-of-squ_08609f6a`) is
    normalised the same way."""
    svc, conn = _service_with_rows([
        {
            "doc_id": "maghami-et-al-2020-diagnosis-and-management-of-squ",
            "inclusion": ["squamous cell carcinoma of unknown primary"],
            "exclusion": [],
        }
    ])
    result = await svc.get_eligibility_criteria_by_doc_ids(
        ["maghami-et-al-2020-diagnosis-and-management-of-squ_08609f6a"]
    )
    assert "maghami-et-al-2020-diagnosis-and-management-of-squ_08609f6a" in result
    assert conn.last_args[0] == ["maghami-et-al-2020-diagnosis-and-management-of-squ"]


@pytest.mark.asyncio
async def test_unresolvable_doc_ids_are_simply_omitted():
    """If the studies table doesn't have a matching row for a given
    doc_id (after normalisation), it should be omitted from the
    result. Callers treat absence as 'no structured criteria
    available'."""
    svc, _ = _service_with_rows([])  # no rows match
    result = await svc.get_eligibility_criteria_by_doc_ids(
        ["doi_10.9999_nonexistent_abcdef12"]
    )
    assert result == {}


@pytest.mark.asyncio
async def test_handles_mix_of_present_and_missing_doc_ids():
    svc, _ = _service_with_rows([
        {
            "doc_id": "doi_10.1200_jco.2007.15.0102",
            "inclusion": ["recurrent HNSCC"],
            "exclusion": [],
        }
    ])
    result = await svc.get_eligibility_criteria_by_doc_ids([
        "doi_10.1200_jco.2007.15.0102_e07e6c6e",  # present
        "doi_10.9999_missing_abcdef12",            # missing
    ])
    assert len(result) == 1
    assert "doi_10.1200_jco.2007.15.0102_e07e6c6e" in result
    assert "doi_10.9999_missing_abcdef12" not in result


@pytest.mark.asyncio
async def test_doc_id_without_hash_suffix_is_unchanged():
    """A doc_id that doesn't carry a hash suffix should be queried as-is."""
    svc, conn = _service_with_rows([
        {
            "doc_id": "JCO-Neoadjuvant_combined_modality_program_-_Q96",
            "inclusion": ["invasive bladder cancer"],
            "exclusion": [],
        }
    ])
    await svc.get_eligibility_criteria_by_doc_ids(
        ["JCO-Neoadjuvant_combined_modality_program_-_Q96"]
    )
    assert conn.last_args[0] == ["JCO-Neoadjuvant_combined_modality_program_-_Q96"]


@pytest.mark.asyncio
async def test_empty_doc_ids_short_circuits():
    """Empty input must not hit the DB at all."""
    svc = PostgresStudyDetailsService()
    svc._get_pool = AsyncMock()  # type: ignore[assignment]
    result = await svc.get_eligibility_criteria_by_doc_ids([])
    assert result == {}
    svc._get_pool.assert_not_awaited()
