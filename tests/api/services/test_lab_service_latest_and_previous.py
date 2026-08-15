"""
Tests for LabService.latest_and_previous_by_test() (2026-08-12
convergence Sprint A item 3). The real query uses a Postgres window
function (ROW_NUMBER() OVER PARTITION BY canonical_test_name ...) with
no live Postgres in this sandbox, so these tests fake conn.fetch() to
return rows already carrying the `rn` column exactly as that window
function would produce it -- this exercises the Python-side grouping
(latest = rn 1, previous = rn 2, per canonical_test_name) which is the
part actually written in this codebase.
"""

from __future__ import annotations

import pytest

from src.api.services.patient.lab_service import LabService


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, query, *args):
        return self._rows


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


class _FakeDB:
    def __init__(self, pool):
        self._pool = pool

    async def ensure_schema(self):
        pass

    async def get_pool(self):
        return self._pool


def _row(test, rn, value_numeric=None, value_text=None, unit=None, collected_at=None):
    return {
        "canonical_test_name": test, "rn": rn,
        "value_numeric": value_numeric, "value_text": value_text,
        "unit": unit, "collected_at": collected_at,
    }


@pytest.mark.asyncio
async def test_groups_latest_and_previous_by_canonical_test_name(monkeypatch):
    rows = [
        _row("anc", 1, value_numeric=1.4, unit="10^9/L", collected_at="2026-08-10"),
        _row("anc", 2, value_numeric=2.4, unit="10^9/L", collected_at="2026-07-10"),
        _row("creatinine", 1, value_numeric=1.5, unit="mg/dL", collected_at="2026-08-10"),
    ]
    conn = _FakeConn(rows)
    monkeypatch.setattr(
        "src.api.services.patient.lab_service.get_patient_db",
        lambda: _FakeDB(_FakePool(conn)),
    )

    result = await LabService().latest_and_previous_by_test("profile-1")

    assert set(result.keys()) == {"anc", "creatinine"}
    assert result["anc"]["latest"]["value_numeric"] == 1.4
    assert result["anc"]["previous"]["value_numeric"] == 2.4
    assert "previous" not in result["creatinine"]


@pytest.mark.asyncio
async def test_empty_when_no_labs_recorded(monkeypatch):
    conn = _FakeConn([])
    monkeypatch.setattr(
        "src.api.services.patient.lab_service.get_patient_db",
        lambda: _FakeDB(_FakePool(conn)),
    )
    result = await LabService().latest_and_previous_by_test("profile-1")
    assert result == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
