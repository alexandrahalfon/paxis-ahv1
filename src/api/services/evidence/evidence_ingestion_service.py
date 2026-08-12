"""
Evidence Ingestion Service (Phase 3, front-half added 2026-08-12)

Two entry points:

    ingest_url(source_key, url)
        The real pipeline: fetch -> extract -> classify -> chunk ->
        embed -> upsert -> register. Use this for anything reachable by
        HTTP (source_fetcher.py + content_extractor.py handle HTML/PDF).

    ingest_document(source_key, doc_id, title, raw_text, url=None, ...)
        Manual path for content acquired another way (an operator pasted
        text, a page fetched by an agent tool in an environment where
        this service's own httpx-based fetch is blocked — see
        source_fetcher.py's docstring). Same classify/chunk/embed/upsert/
        register pipeline underneath; only how the text was obtained
        differs.

Both are idempotent and deterministic:

    document_id = stable_id("document", source_key, url_or_doc_key)
    content_hash = sha256(cleaned_text)
    version_id  = stable_id("version", document_id, content_hash)
    point_id    = stable_id("point", version_id, section_title, chunk_index)

Fetching the same URL twice with unchanged content produces the same
version_id, which is recognized as already-current and skipped —
running an ingestion job twice does not duplicate chunks. Changed
content produces a new version_id: the NEW Qdrant points are upserted
FIRST, then the Postgres transaction switches current_version_id to the
new version, and only after that transaction commits are the OLD
version's Qdrant points deleted — by their exact qdrant_point_id list
from evidence_chunk_registry, not a payload-field filter. That ordering
is deliberate (fixed 2026-08-12): the previous order deleted old points
before upserting new ones, so a failed upsert after a successful delete
left Postgres believing a version was current while zero of its points
existed in Qdrant. Upsert-then-delete means the worst case on a mid-
ingestion failure is a small number of temporary duplicate/orphaned
points, not zero retrievable evidence — and if the Postgres transaction
itself fails, the just-upserted new points are deleted as compensation
so Qdrant doesn't accumulate points nothing ever points to.

Reuses the same OpenAI embedding model and the same Qdrant client
construction as comprehensive_retrieval.py (via get_comprehensive_retriever)
rather than standing up a second embedding/Qdrant configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client.models import PointStruct, VectorParams, Distance, PointIdsList

from src.core.config import settings
from src.api.services.evidence.source_registry import get_source_registry
from src.api.services.evidence.content_extractor import ExtractedDocument, extract
from src.api.services.evidence.section_chunker import Chunk, chunk_document
from src.api.services.evidence.metadata_classifier import classify as classify_content, ClassificationResult

logger = logging.getLogger(__name__)


def stable_id(*parts: str) -> str:
    """Deterministic UUID from arbitrary string parts — same inputs
    always produce the same id, which is what makes re-ingestion
    idempotent instead of accumulating duplicate rows/points."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def content_hash_of(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def unique_section_texts(chunks: List[Chunk]) -> Dict[str, str]:
    """section_title -> text to classify, one entry per unique heading in
    document order, first-seen wins. Pulled out as a pure function (no
    OpenAI call, no I/O) so the section-grouping/dedup logic — the part
    of chunk-level classification actually worth a regression test — is
    directly testable against a list of Chunk objects. Prefers
    parent_text (the section's full text) over an individual child
    chunk's text so a long section classifies once against its complete
    content rather than once per overlapping window. Headingless chunks
    (section_title is None) are skipped: they fall back to the
    document-level classification at the call site instead."""
    out: Dict[str, str] = {}
    for chunk in chunks:
        title = chunk.section_title
        if not title or title in out:
            continue
        out[title] = chunk.parent_text or chunk.text
    return out


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

    async def _classify_sections(self, chunks: List[Chunk], doc_title: str) -> Dict[str, Dict[str, Any]]:
        """Runs classify_content() once per unique section heading (see
        unique_section_texts) and returns section_title -> applicability
        dict. Sequential, not gathered in parallel: ingestion is a
        background/CLI job, not a live user-facing request path, and
        this keeps classification calls easy to rate-limit/log one at a
        time rather than bursting one per section at once."""
        out: Dict[str, Dict[str, Any]] = {}
        for title, text in unique_section_texts(chunks).items():
            result: ClassificationResult = await asyncio.to_thread(
                classify_content, text, f"{doc_title} — {title}"
            )
            out[title] = result.to_dict()
        return out

    # ── Public entry points ─────────────────────────────────────────

    async def ingest_url(
        self,
        source_key: str,
        url: str,
        applicability_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch a URL and run it through the full pipeline. Raises
        source_fetcher.FetchError on an unreachable/oversized URL —
        callers (the CLI runner) are expected to catch that per-URL and
        continue the batch rather than aborting a whole run."""
        from src.api.services.evidence.source_fetcher import fetch_url

        fetched = await asyncio.to_thread(fetch_url, url)
        doc = extract(fetched.content, fetched.content_type, source_url=fetched.final_url)
        if not doc.is_usable():
            raise ValueError(
                f"{url} did not yield usable content after cleaning "
                f"({len(doc.plain_text.strip())} chars) — page may be JS-rendered, "
                "paywalled, or not an article page. Skipping."
            )
        return await self._ingest_extracted(
            source_key=source_key, doc_key=fetched.final_url, url=fetched.final_url,
            doc=doc, applicability_override=applicability_override,
        )

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
        """Manual path — see module docstring. doc_id is the caller's own
        stable key (e.g. a filename) when there's no URL to derive one
        from; when url is given, url is used instead so the same
        document ingested once via ingest_url() and once by hand (e.g.
        text pulled by an agent's web-fetch tool) resolves to the same
        document_id and correctly versions against each other rather
        than becoming two unrelated documents."""
        if not (raw_text or "").strip():
            raise ValueError("No text to ingest")
        doc = ExtractedDocument(title=title, sections=[], plain_text=raw_text)
        return await self._ingest_extracted(
            source_key=source_key, doc_key=url or doc_id, url=url, doc=doc,
            applicability_override=applicability, constraints=constraints,
        )

    # ── Shared pipeline ──────────────────────────────────────────────

    async def _ingest_extracted(
        self,
        source_key: str,
        doc_key: str,
        url: Optional[str],
        doc: ExtractedDocument,
        applicability_override: Optional[Dict[str, Any]] = None,
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

        document_id = stable_id("document", source_key, doc_key)
        chash = content_hash_of(doc.plain_text)
        version_id = stable_id("version", document_id, chash)

        from src.api.services.patient_db import get_patient_db
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        # ── Idempotency check ────────────────────────────────────────
        # Same document_id + same content_hash => same version_id =>
        # already ingested. Skip re-embedding entirely; this is what
        # makes running the same source list twice safe.
        async with pool.acquire() as conn:
            existing_version = await conn.fetchrow(
                "SELECT * FROM evidence_document_versions WHERE id = $1", version_id,
            )
        if existing_version and existing_version["is_current"]:
            logger.info(
                "[EvidenceIngestion] %s unchanged (version %s already current), skipping",
                doc_key, version_id,
            )
            return {
                "document_id": document_id, "version_id": version_id,
                "collection": collection, "chunks_ingested": 0, "skipped": True,
                "reason": "content unchanged since last ingestion",
            }

        await self._ensure_collection(collection)

        # ── Classification ───────────────────────────────────────────
        # Document-level: broad-context classification stored on
        # evidence_documents.applicability, for the document record and
        # any UI that shows "what is this source about" as a whole. It
        # is NOT what gets attached to every chunk below — a page that
        # covers diet AND medication AND emotional support would make
        # every one of its chunks look applicable to all three axes if
        # it were. Chunk-level classification (next) is what
        # applicability_scorer.py actually reads at retrieval time.
        applicability = applicability_override
        if applicability is None:
            result: ClassificationResult = await asyncio.to_thread(
                classify_content, doc.plain_text, doc.title
            )
            applicability = result.to_dict()

        # ── Chunk + embed + upsert ───────────────────────────────────
        chunks: List[Chunk] = chunk_document(doc)
        if not chunks:
            raise ValueError(f"{doc_key} produced no chunks after chunking")

        # Section-level: one classify() call per unique section heading,
        # not per chunk — a long section split into overlapping child
        # windows (section_chunker.CHILD_MAX_CHARS) shares one
        # classification of the section's full parent_text, since those
        # children are all still "about" the same section. Headingless
        # chunks (the PDF/no-sections fallback in section_chunker.py has
        # no section_title) have no narrower unit to classify against
        # and fall back to the document-level result above — the same
        # degradation that already applied to every chunk before this
        # change, so nothing regresses for PDFs.
        section_applicability = (
            {} if applicability_override is not None
            else await self._classify_sections(chunks, doc.title)
        )

        retriever = self._retriever()
        points: List[PointStruct] = []
        chunk_rows: List[Dict[str, Any]] = []
        for chunk in chunks:
            vector = await retriever._embed_async(chunk.text)
            point_id = stable_id("point", version_id, chunk.section_title or "", str(chunk.chunk_index))
            chunk_applicability = (
                section_applicability.get(chunk.section_title, applicability)
                if chunk.section_title else applicability
            )
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": document_id,
                    "text": chunk.text,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "parent_text": chunk.parent_text,
                    "title": doc.title,
                    "doc_meta": {
                        "title": doc.title, "source_key": source_key,
                        "source_name": source["name"], "url": url,
                        "authority_class": source["authority_class"],
                    },
                    "applicability": chunk_applicability,
                    "version_id": version_id,
                },
            ))
            chunk_rows.append({"point_id": point_id, "chunk_index": chunk.chunk_index,
                                "section_title": chunk.section_title})

        # ── Who's being superseded, and what exactly to delete once we're
        # done — looked up BEFORE any writes, from evidence_chunk_registry
        # (the source of truth for "which Qdrant points belong to this
        # version"), so deletion later never depends on a Qdrant payload
        # field matching what Postgres thinks is true. See point (18) in
        # the 2026-08-12 beta audit: exact point-ID deletion is more
        # auditable than a payload Filter and doesn't assume the payload
        # schema stayed consistent across ingestion-code versions.
        async with pool.acquire() as conn:
            evidence_doc = await conn.fetchrow(
                "SELECT * FROM evidence_documents WHERE id = $1", document_id,
            )
            prior_version_id = evidence_doc["current_version_id"] if evidence_doc else None
            prior_point_ids: List[str] = []
            if prior_version_id and prior_version_id != version_id:
                prior_rows = await conn.fetch(
                    "SELECT qdrant_point_id FROM evidence_chunk_registry "
                    "WHERE evidence_document_version_id = $1",
                    prior_version_id,
                )
                prior_point_ids = [r["qdrant_point_id"] for r in prior_rows]

        # ── Upsert the NEW points FIRST ────────────────────────────────
        # Nothing in Postgres has been written yet at this point, so a
        # failure here is a clean no-op ingestion (safe to just retry) —
        # the ordering bug this replaces (delete-old-then-upsert-new) could
        # leave Postgres believing a version was current with zero of its
        # points actually in Qdrant if the upsert failed after the delete.
        await asyncio.to_thread(retriever.qdrant.upsert, collection_name=collection, points=points)

        # ── Then the Postgres transaction that makes the new version
        # current ─────────────────────────────────────────────────────
        # Ordering matters here too: evidence_documents.current_version_id
        # references evidence_document_versions.id, which in turn
        # references evidence_documents.id — a genuine circular FK
        # dependency. Resolved the standard way: insert/update the
        # document row with current_version_id left NULL, insert the
        # version row (now able to reference an existing document), then
        # point the document at it. Doing this in the other order raises
        # a foreign-key violation on the very first ingestion.
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    if evidence_doc is None:
                        await conn.execute(
                            """
                            INSERT INTO evidence_documents
                                (id, source_id, doc_id, title, url, qdrant_collection,
                                 applicability, constraints, last_ingested_at, latest_content_hash)
                            VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb, now(), $9)
                            """,
                            document_id, source["id"], doc_key, doc.title, url, collection,
                            json.dumps(applicability), json.dumps(constraints or {}), chash,
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE evidence_documents
                               SET title = $2, applicability = $3::jsonb, last_ingested_at = now(),
                                   latest_content_hash = $4
                             WHERE id = $1
                            """,
                            document_id, doc.title, json.dumps(applicability), chash,
                        )

                    if prior_version_id:
                        await conn.execute(
                            """
                            UPDATE evidence_document_versions
                               SET is_current = false, superseded_by = $2
                             WHERE id = $1
                            """,
                            prior_version_id, version_id,
                        )

                    await conn.execute(
                        """
                        INSERT INTO evidence_document_versions
                            (id, evidence_document_id, content_hash, raw_text_excerpt, is_current)
                        VALUES ($1, $2, $3, $4, true)
                        ON CONFLICT (id) DO UPDATE SET is_current = true
                        """,
                        version_id, document_id, chash, doc.plain_text[:2000],
                    )

                    await conn.execute(
                        "UPDATE evidence_documents SET current_version_id = $2 WHERE id = $1",
                        document_id, version_id,
                    )

                    for row in chunk_rows:
                        await conn.execute(
                            """
                            INSERT INTO evidence_chunk_registry
                                (id, evidence_document_id, evidence_document_version_id,
                                 qdrant_point_id, chunk_index, section_title)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            """,
                            str(uuid.uuid4()), document_id, version_id,
                            row["point_id"], row["chunk_index"], row["section_title"],
                        )
        except Exception:
            # Compensate: the new points are now orphaned in Qdrant (no
            # Postgres row will ever reference them), so remove them
            # rather than leaving unreachable vectors behind. A small
            # number of temporary duplicates surviving briefly (the
            # window between the upsert above and this transaction) is
            # the accepted tradeoff from upserting before writing —
            # leaving them permanently orphaned on a genuine failure is
            # not.
            logger.error(
                "[EvidenceIngestion] Postgres transaction failed for %s (version %s) — "
                "removing the %d just-upserted Qdrant points as compensation",
                doc_key, version_id, len(points), exc_info=True,
            )
            try:
                await asyncio.to_thread(
                    retriever.qdrant.delete,
                    collection_name=collection,
                    points_selector=PointIdsList(points=[p.id for p in points]),
                )
            except Exception:
                logger.error(
                    "[EvidenceIngestion] compensating delete also failed for %s (version %s) — "
                    "%d Qdrant points are now orphaned and need manual cleanup",
                    doc_key, version_id, len(points), exc_info=True,
                )
            raise

        # ── Only now delete the superseded version's OLD Qdrant points,
        # by the exact point IDs looked up before any writes above.
        # Failure here is non-fatal — the new version is already fully
        # correct and current; a lingering stale point is a nuisance
        # (extra, superseded search result) not a correctness bug the way
        # zero retrievable points would be.
        if prior_point_ids:
            try:
                await asyncio.to_thread(
                    retriever.qdrant.delete,
                    collection_name=collection,
                    points_selector=PointIdsList(points=prior_point_ids),
                )
            except Exception:
                logger.warning(
                    "[EvidenceIngestion] failed to delete %d superseded points for version %s "
                    "(new version's points are still correct; stale points may linger)",
                    len(prior_point_ids), prior_version_id, exc_info=True,
                )

        logger.info(
            "[EvidenceIngestion] ingested %s -> %s (%d chunks, version %s)",
            doc_key, collection, len(points), version_id,
        )
        return {
            "document_id": document_id, "version_id": version_id, "collection": collection,
            "chunks_ingested": len(points), "skipped": False, "applicability": applicability,
        }


_service: Optional[EvidenceIngestionService] = None


def get_evidence_ingestion_service() -> EvidenceIngestionService:
    global _service
    if _service is None:
        _service = EvidenceIngestionService()
    return _service
