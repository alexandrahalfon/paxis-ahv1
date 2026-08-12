"""
Evidence Ingestion Service (Phase 3)

Chunks, embeds, and upserts ONE document's already-fetched text into the
Qdrant collection its source maps to (via source_registry.collection_for),
and records the document + its chunk point ids in evidence_documents /
evidence_chunk_registry so re-ingestion can be tracked and reversed.

Deliberately does not fetch anything from the web itself. Fetching
external pages is an ingestion-job concern (a scheduled task, an admin
action, a one-off script an operator runs with review) with licensing and
rate-limit implications outside this module's scope. Handing it raw text
keeps this service testable and keeps "what got ingested and from where"
auditable through evidence_documents.url rather than implicit in a crawl.

Reuses the same OpenAI embedding model and the same Qdrant client
construction as comprehensive_retrieval.py (via get_comprehensive_retriever)
rather than standing up a second embedding/Qdrant configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client.models import PointStruct, VectorParams, Distance

from src.core.config import settings
from src.api.services.evidence.source_registry import get_source_registry

logger = logging.getLogger(__name__)

_CHUNK_CHARS = 1800
_CHUNK_OVERLAP = 200


def _chunk_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + _CHUNK_CHARS, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - _CHUNK_OVERLAP
    return chunks


class EvidenceIngestionService:
    def _retriever(self):
        from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
        return get_comprehensive_retriever()

    async def _ensure_collection(self, collection_name: str) -> None:
        retriever = self._retriever()
        try:
            existing = await asyncio.to_thread(retriever.qdrant.collection_exists, collection_name)
        except Exception:
            existing = False
        if existing:
            return
        await asyncio.to_thread(
            retriever.qdrant.create_collection,
            collection_name=collection_name,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )
        logger.info("[EvidenceIngestion] created collection %s", collection_name)

    async def ingest_document(
        self,
        source_key: str,
        doc_id: str,
        title: str,
        raw_text: str,
        url: Optional[str] = None,
        applicability: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        registry = get_source_registry()
        source = await registry.get_source(source_key)
        if not source:
            raise ValueError(
                f"Unknown evidence source '{source_key}'. Call "
                "SourceRegistry.register_source() (or seed_default_sources()) first."
            )
        collection = registry.collection_for(source)
        await self._ensure_collection(collection)

        chunks = _chunk_text(raw_text)
        if not chunks:
            raise ValueError("No text to ingest")

        retriever = self._retriever()
        point_ids: List[str] = []
        points: List[PointStruct] = []
        for i, chunk in enumerate(chunks):
            vector = await retriever._embed_async(chunk)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}"))
            point_ids.append(point_id)
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "text": chunk,
                    "chunk_index": i,
                    "title": title,
                    "doc_meta": {
                        "title": title, "source_key": source_key,
                        "source_name": source["name"], "url": url,
                        "authority_class": source["authority_class"],
                    },
                    "applicability": applicability or {},
                },
            ))

        await asyncio.to_thread(
            retriever.qdrant.upsert, collection_name=collection, points=points
        )

        evidence_doc = await registry.register_document(
            source_key=source_key, doc_id=doc_id, title=title, url=url,
            qdrant_collection=collection, applicability=applicability, constraints=constraints,
        )

        from src.api.services.patient_db import get_patient_db
        db = get_patient_db()
        pool = await db.get_pool()
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evidence_document_versions
                    (id, evidence_document_id, content_hash, raw_text_excerpt)
                VALUES ($1, $2, $3, $4)
                """,
                str(uuid.uuid4()), evidence_doc["id"], content_hash, raw_text[:2000],
            )
            for i, pid in enumerate(point_ids):
                await conn.execute(
                    """
                    INSERT INTO evidence_chunk_registry
                        (id, evidence_document_id, qdrant_point_id, chunk_index)
                    VALUES ($1, $2, $3, $4)
                    """,
                    str(uuid.uuid4()), evidence_doc["id"], pid, i,
                )

        return {
            "doc_id": doc_id, "collection": collection,
            "chunks_ingested": len(points), "evidence_document": evidence_doc,
        }


_service: Optional[EvidenceIngestionService] = None


def get_evidence_ingestion_service() -> EvidenceIngestionService:
    global _service
    if _service is None:
        _service = EvidenceIngestionService()
    return _service
