"""
Tests for the evidence-ingestion version/Qdrant replacement ordering fix
(2026-08-12, P0 beta-audit item): new Qdrant points must be upserted
BEFORE the Postgres transaction switches current_version_id, and the OLD
version's points must be deleted afterwards by their exact
qdrant_point_id list (from evidence_chunk_registry) rather than a
payload-field Filter. If the Postgres transaction itself fails, the
just-upserted new points must be deleted as compensation.

Runs against a real in-memory QdrantClient(":memory:") (so the
PointIdsList delete calls are exercised against the real qdrant-client
API, not a mock) and a lightweight in-memory fake of the three Postgres
tables this service touches (evidence_documents, evidence_document_
versions, evidence_chunk_registry), since this sandbox has no live
Postgres — see migrations/patients_db/README.md's "Not done here" note
for the same limitation elsewhere in this codebase.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from src.api.services.evidence.evidence_ingestion_service import (
    EvidenceIngestionService,
    get_evidence_ingestion_service,
)
from src.api.services.evidence import source_registry as source_registry_module
from src.api.services import patient_db as patient_db_module


# ── In-memory fakes for the three Postgres tables this service touches ──

class FakeStore:
    def __init__(self):
        self.sources: dict = {}
        self.documents: dict = {}
        self.versions: dict = {}
        self.chunk_registry: list = []


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeTxnCtx:
    """Mirrors real asyncpg transaction semantics closely enough to matter
    for this test: mutations issued inside `async with conn.transaction():`
    are staged, not applied, and only committed to the store on a clean
    exit. An exception discards every staged mutation, including ones
    that already ran before the failing statement — real ROLLBACK, not
    just "stop on error"."""

    def __init__(self, conn: "FakeConn"):
        self.conn = conn

    async def __aenter__(self):
        self.conn._staged_ops = []
        return self

    async def __aexit__(self, exc_type, exc, tb):
        staged = self.conn._staged_ops
        self.conn._staged_ops = None
        if exc_type is None:
            for q, args in staged:
                self.conn._apply_execute(q, args)
        return False  # never suppress -- exceptions must propagate


class FakeConn:
    def __init__(self, store: FakeStore, fail_on: str = None):
        self.store = store
        self.fail_on = fail_on
        self._staged_ops = None  # None = not in a transaction

    def transaction(self):
        return _FakeTxnCtx(self)

    async def fetchrow(self, query, *args):
        q = " ".join(query.split())
        if "FROM evidence_sources" in q:
            return dict(self.store.sources[args[0]]) if args[0] in self.store.sources else None
        if "FROM evidence_document_versions WHERE id" in q:
            return dict(self.store.versions[args[0]]) if args[0] in self.store.versions else None
        if "FROM evidence_documents WHERE id" in q:
            return dict(self.store.documents[args[0]]) if args[0] in self.store.documents else None
        raise AssertionError(f"unexpected fetchrow: {q}")

    async def fetch(self, query, *args):
        q = " ".join(query.split())
        if "FROM evidence_chunk_registry WHERE evidence_document_version_id" in q:
            return [
                dict(r) for r in self.store.chunk_registry
                if r["evidence_document_version_id"] == args[0]
            ]
        raise AssertionError(f"unexpected fetch: {q}")

    async def execute(self, query, *args):
        q = " ".join(query.split())
        if self.fail_on and self.fail_on in q:
            raise RuntimeError(f"simulated Postgres failure on: {self.fail_on}")
        if self._staged_ops is not None:
            self._staged_ops.append((q, args))
        else:
            self._apply_execute(q, args)

    def _apply_execute(self, q: str, args) -> None:
        if "INSERT INTO evidence_documents" in q:
            (doc_id, source_id, doc_key, title, url, collection,
             applicability, constraints, chash) = args
            self.store.documents[doc_id] = {
                "id": doc_id, "source_id": source_id, "doc_id": doc_key, "title": title,
                "url": url, "qdrant_collection": collection, "applicability": applicability,
                "constraints": constraints, "current_version_id": None,
                "latest_content_hash": chash,
            }
        elif "UPDATE evidence_documents" in q and "SET title" in q:
            doc_id, title, applicability, chash = args
            self.store.documents[doc_id].update(
                title=title, applicability=applicability, latest_content_hash=chash,
            )
        elif "UPDATE evidence_document_versions" in q and "is_current = false" in q:
            prior_version_id, new_version_id = args
            self.store.versions[prior_version_id].update(
                is_current=False, superseded_by=new_version_id,
            )
        elif "INSERT INTO evidence_document_versions" in q:
            version_id, document_id, chash, excerpt = args
            self.store.versions[version_id] = {
                "id": version_id, "evidence_document_id": document_id, "content_hash": chash,
                "raw_text_excerpt": excerpt, "is_current": True, "superseded_by": None,
            }
        elif "UPDATE evidence_documents SET current_version_id" in q:
            doc_id, version_id = args
            self.store.documents[doc_id]["current_version_id"] = version_id
        elif "INSERT INTO evidence_chunk_registry" in q:
            row_id, doc_id, version_id, point_id, chunk_index, section_title = args
            self.store.chunk_registry.append({
                "id": row_id, "evidence_document_id": doc_id,
                "evidence_document_version_id": version_id, "qdrant_point_id": point_id,
                "chunk_index": chunk_index, "section_title": section_title,
            })
        else:
            raise AssertionError(f"unexpected execute: {q}")


class FakePool:
    def __init__(self, store: FakeStore, fail_on: str = None):
        self.store = store
        self.fail_on = fail_on

    def acquire(self):
        return _FakeAcquireCtx(FakeConn(self.store, self.fail_on))


class FakeDB:
    def __init__(self, pool: FakePool):
        self._pool = pool

    async def ensure_schema(self):
        pass

    async def get_pool(self):
        return self._pool


class FakeRetriever:
    """Real in-memory Qdrant (so PointIdsList delete/upsert calls hit the
    actual qdrant-client API), fake embedding (no OpenAI call needed --
    a fixed-length deterministic vector is all _ingest_extracted needs)."""

    def __init__(self, embed_dim: int):
        self.qdrant = QdrantClient(":memory:")
        self._embed_dim = embed_dim

    async def _embed_async(self, text: str):
        # Overwritten per-instance by _make_retriever() below; present
        # here only so the class has the attribute/signature documented.
        return [0.001] * self._embed_dim


def _make_retriever(embed_dim: int) -> FakeRetriever:
    r = FakeRetriever(embed_dim)

    async def embed(text):
        return [0.001] * embed_dim

    r._embed_async = embed
    return r


@pytest.fixture
def store():
    s = FakeStore()
    s.sources["test_source"] = {
        "id": str(uuid.uuid4()), "source_key": "test_source", "name": "Test Source",
        "domain": "example.org", "authority_class": "A", "authority_score": 1.0,
        "source_type": "patient_education", "allowed_intents": [], "patient_facing": True,
        "ingestion_method": "manual", "license_status": "unknown", "active": True,
    }
    return s


def _wire(monkeypatch, store, embed_dim=None, fail_on=None):
    if embed_dim is None:
        from src.core.config import settings
        embed_dim = settings.embed_dim
    pool = FakePool(store, fail_on=fail_on)
    fake_db = FakeDB(pool)
    monkeypatch.setattr(patient_db_module, "get_patient_db", lambda: fake_db)
    monkeypatch.setattr(source_registry_module, "get_patient_db", lambda: fake_db)
    retriever = _make_retriever(embed_dim)
    monkeypatch.setattr(EvidenceIngestionService, "_retriever", lambda self: retriever)
    return retriever


APPLICABILITY = {
    "content_type": "patient_education", "intents": ["general"], "topics": [],
    "cancer_types": ["all"], "treatment_modalities": [], "regimens": [], "drugs": [],
    "symptoms": [], "treatment_phases": [], "age_groups": [],
    "can_support_self_care": False, "can_support_triage": False, "can_support_dose_change": False,
}


class TestUpsertBeforeSwitchOrdering:
    @pytest.mark.asyncio
    async def test_reingesting_changed_content_deletes_exactly_the_old_points(self, monkeypatch, store):
        retriever = _wire(monkeypatch, store)
        service = EvidenceIngestionService()

        first = await service.ingest_document(
            source_key="test_source", doc_id="doc-1", title="Taste changes",
            raw_text="Taste changes during chemotherapy. " * 5,
            applicability=APPLICABILITY,
        )
        assert first["skipped"] is False
        collection = first["collection"]
        old_point_ids = [r["qdrant_point_id"] for r in store.chunk_registry]
        assert old_point_ids  # sanity: something was actually ingested

        count_after_first = retriever.qdrant.count(collection).count
        assert count_after_first == len(old_point_ids)

        second = await service.ingest_document(
            source_key="test_source", doc_id="doc-1", title="Taste changes",
            raw_text="COMPLETELY DIFFERENT CONTENT about nutrition and eating. " * 5,
            applicability=APPLICABILITY,
        )
        assert second["skipped"] is False
        assert second["version_id"] != first["version_id"]

        # Old points must be gone...
        retrieved_old = retriever.qdrant.retrieve(collection, ids=old_point_ids)
        assert retrieved_old == [], "superseded version's points were not deleted"

        # ...and exactly the new version's points must be present, no more.
        new_point_ids = [
            r["qdrant_point_id"] for r in store.chunk_registry
            if r["evidence_document_version_id"] == second["version_id"]
        ]
        assert new_point_ids
        count_after_second = retriever.qdrant.count(collection).count
        assert count_after_second == len(new_point_ids)

        # And the registry/version bookkeeping reflects the switch.
        assert store.versions[first["version_id"]]["is_current"] is False
        assert store.versions[first["version_id"]]["superseded_by"] == second["version_id"]
        assert store.versions[second["version_id"]]["is_current"] is True
        assert first["document_id"] == second["document_id"]
        assert store.documents[second["document_id"]]["current_version_id"] == second["version_id"]

    @pytest.mark.asyncio
    async def test_unchanged_content_is_skipped_and_touches_nothing_new(self, monkeypatch, store):
        _wire(monkeypatch, store)
        service = EvidenceIngestionService()
        text = "Stable content that never changes. " * 5

        first = await service.ingest_document(
            source_key="test_source", doc_id="doc-2", title="Stable doc",
            raw_text=text, applicability=APPLICABILITY,
        )
        registry_count_after_first = len(store.chunk_registry)

        second = await service.ingest_document(
            source_key="test_source", doc_id="doc-2", title="Stable doc",
            raw_text=text, applicability=APPLICABILITY,
        )
        assert second["skipped"] is True
        assert second["version_id"] == first["version_id"]
        assert len(store.chunk_registry) == registry_count_after_first


class TestPostgresFailureCompensation:
    @pytest.mark.asyncio
    async def test_postgres_transaction_failure_deletes_the_new_points_it_orphaned(self, monkeypatch, store):
        # Fail specifically on the chunk-registry insert -- deep enough
        # into the transaction to prove the whole transaction (not just
        # the first statement) is covered by the compensation.
        retriever = _wire(monkeypatch, store, fail_on="INSERT INTO evidence_chunk_registry")
        service = EvidenceIngestionService()

        from src.api.services.evidence.evidence_ingestion_service import stable_id
        expected_document_id = stable_id("document", "test_source", "doc-3")

        with pytest.raises(RuntimeError, match="simulated Postgres failure"):
            await service.ingest_document(
                source_key="test_source", doc_id="doc-3", title="Will fail",
                raw_text="Content that will fail to persist in Postgres. " * 5,
                applicability=APPLICABILITY,
            )

        # Nothing should have been left in Postgres -- real ROLLBACK
        # semantics (see _FakeTxnCtx): every statement staged before the
        # failing one must be discarded too, not just the one that failed.
        assert expected_document_id not in store.documents
        assert store.chunk_registry == []
        assert store.versions == {}

        # ...and the points upserted to Qdrant before the failure must
        # have been removed as compensation, not left orphaned.
        from src.core.config import settings
        collection = settings.qdrant_patient_education_collection
        assert retriever.qdrant.collection_exists(collection)
        assert retriever.qdrant.count(collection).count == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
