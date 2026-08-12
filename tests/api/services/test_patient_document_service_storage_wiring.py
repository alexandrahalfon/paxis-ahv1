"""
Tests that patient_document_service.PatientDocumentService.create_document
persists documents through patient_document_storage.store() (2026-08-12
beta audit item 3) rather than writing to local disk directly.
"""

from __future__ import annotations

import pytest

from src.api.services.patient.patient_document_service import PatientDocumentService
import src.api.services.patient.patient_document_service as pds_module


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self):
        self.inserted = None

    async def fetchrow(self, query, *args):
        (document_id, patient_profile_id, filename, content_type,
         storage_uri, document_date) = args
        self.inserted = {
            "id": document_id, "patient_profile_id": patient_profile_id,
            "filename": filename, "content_type": content_type,
            "object_storage_uri": storage_uri, "document_date": document_date,
            "extraction_status": "pending",
        }
        return self.inserted


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


class TestCreateDocumentUsesStorageModule:
    @pytest.mark.asyncio
    async def test_stores_via_patient_document_storage_and_persists_returned_uri(self, monkeypatch):
        conn = _FakeConn()
        monkeypatch.setattr(pds_module, "get_patient_db", lambda: _FakeDB(_FakePool(conn)))

        store_calls = []

        async def fake_store(patient_profile_id, document_id, filename, content):
            store_calls.append((patient_profile_id, filename, content))
            return f"gs://patient-phi-bucket/patient_documents/{patient_profile_id}/{document_id}_{filename}"

        monkeypatch.setattr(pds_module.patient_document_storage, "store", fake_store)

        service = PatientDocumentService()
        result = await service.create_document(
            patient_profile_id="profile-1", filename="labs.pdf",
            content=b"pdf bytes", content_type="application/pdf",
        )

        assert len(store_calls) == 1
        assert store_calls[0][0] == "profile-1"
        assert store_calls[0][1] == "labs.pdf"
        assert store_calls[0][2] == b"pdf bytes"

        assert result["object_storage_uri"].startswith("gs://patient-phi-bucket/patient_documents/profile-1/")
        assert conn.inserted["object_storage_uri"] == result["object_storage_uri"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
