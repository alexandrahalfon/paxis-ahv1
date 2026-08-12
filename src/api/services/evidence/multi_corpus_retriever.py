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
            out.append({
                "doc_id": payload.get("doc_id"),
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


async def search(
    query_text: str, plan: RetrievalPlan, top_k_per_collection: int = 6
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
            key = item.get("doc_id") or item["text"][:80]
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged
