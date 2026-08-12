"""
Multi-Corpus Retriever (Phase 4)

Searches every collection in a RetrievalPlan in parallel and merges the
results, tagging each candidate with which corpus it came from. Reuses
the existing comprehensive_retriever's Qdrant client and embedding call
(get_comprehensive_retriever()._qdrant_query / ._embed_async) rather than
a second client — same pattern patient_chat_service._retrieve already
used for the single literature collection.

Degrades gracefully by design: qdrant_patient_education_collection,
qdrant_medication_collection, and qdrant_guideline_collection are all
newly *configured* (Phase 3) but not yet *populated* in most deployments
of this change — see evidence_ingestion_service.py. A collection that
doesn't exist yet, or exists with zero points, simply contributes nothing
and is logged at info level, never raises. This is what keeps rewiring
patient_chat_service onto this module a non-regression: with the new
collections empty, results are exactly the literature-collection search
it already did, just now correctly deduped and scored.

Dedup identity (fixed 2026-08-12, beta audit item 6): candidates used to
be deduped by doc_id alone, which collapsed every section of one
document into a single candidate — an NCI nutrition page's "Taste
changes" and "Appetite loss" sections are two independently useful
passages, not duplicates of each other, and only one of them survived.
Identity is now the exact Qdrant point id (unique per chunk by
construction — see evidence_ingestion_service.stable_id("point", ...)),
so distinct sections/chunks of the same document are preserved as
distinct candidates. Near-duplicate CONTENT (the same passage surfacing
twice, e.g. from two collections) is a different concern and is handled
later, at evidence_packet_builder.build_packet() time — see that
module's docstring.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from src.api.services.evidence.retrieval_planner import RetrievalPlan

logger = logging.getLogger(__name__)


def _retriever():
    from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
    return get_comprehensive_retriever()


async def _search_collection(vector: List[float], collection: str, limit: int) -> List[Dict[str, Any]]:
    try:
        retriever = _retriever()
        resp = await retriever._qdrant_query(
            collection_name=collection, query=vector, limit=limit,
            with_payload=True, with_vectors=False,
        )
        points = getattr(resp, "points", resp) or []
        out = []
        for p in points:
            payload = getattr(p, "payload", None) or {}
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            doc_meta = payload.get("doc_meta") or {}
            point_id = getattr(p, "id", None)
            out.append({
                # Provenance — see evidence_packet_builder.py and
                # retrieval_debug_trace.py, which carry these through to
                # the generation packet and the debug trace respectively.
                # Everything here already exists on the Qdrant point/
                # payload (evidence_ingestion_service.py's upsert); this
                # is capturing it, not fetching anything new.
                "qdrant_point_id": str(point_id) if point_id is not None else None,
                "doc_id": payload.get("doc_id"),
                "version_id": payload.get("version_id"),
                "section_title": payload.get("section_title"),
                "chunk_index": payload.get("chunk_index"),
                "url": doc_meta.get("url"),
                "source_name": doc_meta.get("source_name"),
                "title": doc_meta.get("title") or payload.get("title") or "Source",
                "text": text,
                "citation": doc_meta.get("citation") or doc_meta.get("citation_string"),
                "year": doc_meta.get("year"),
                "collection": collection,
                "semantic_score": float(getattr(p, "score", 0.0) or 0.0),
                "applicability_meta": payload.get("applicability") or {},
                "source_key": doc_meta.get("source_key"),
                "authority_class": doc_meta.get("authority_class"),
            })
        return out
    except Exception as e:
        # Missing/empty collection, transient Qdrant error, etc. — a gap
        # in one corpus must never fail the whole answer.
        logger.info("[MultiCorpusRetriever] %s unavailable (%s)", collection, e)
        return []


def _candidate_identity(item: Dict[str, Any]) -> str:
    """Dedup identity for merge() below — see the module docstring for
    why this changed from doc_id alone. Prefers the exact Qdrant point
    id (unique per chunk); falls back to a (doc_id, section_title,
    chunk_index) composite for a candidate somehow missing its point id,
    then to a text prefix as a last resort for anything from a corpus
    that predates this payload shape entirely."""
    point_id = item.get("qdrant_point_id")
    if point_id:
        return f"point:{point_id}"
    doc_id = item.get("doc_id")
    if doc_id:
        return f"doc:{doc_id}|section:{item.get('section_title')}|chunk:{item.get('chunk_index')}"
    return f"text:{(item.get('text') or '')[:80]}"


async def search(
    query_text: str, plan: RetrievalPlan, top_k_per_collection: int = 6,
    *, audience: str = "patient",
) -> List[Dict[str, Any]]:
    if not plan.collections:
        return []
    boosted_query = query_text
    if plan.boost_terms:
        boosted_query = f"{query_text} {' '.join(plan.boost_terms)}"

    retriever = _retriever()
    vector = await retriever._embed_async(boosted_query)

    buckets = await asyncio.gather(
        *(_search_collection(vector, c, top_k_per_collection) for c in plan.collections)
    )

    merged: List[Dict[str, Any]] = []
    seen = set()
    for bucket in buckets:
        for item in bucket:
            key = _candidate_identity(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

    # Source-governance enforcement (2026-08-12 convergence Sprint B item
    # 8) — re-checks each candidate's registered source against the
    # CURRENT registry state (active/patient_facing/allowed_intents), not
    # just whatever tags were baked in at ingestion time. See
    # source_governance.py's module docstring for exactly what's
    # enforced and why this fails open on any error rather than dropping
    # every result a registry lookup failure would otherwise cost.
    try:
        from src.api.services.evidence.source_governance import enforce_source_governance
        merged = await enforce_source_governance(merged, audience=audience, intent=plan.intent)
    except Exception as e:
        logger.info("[MultiCorpusRetriever] source governance enforcement skipped (%s)", e)

    return merged
