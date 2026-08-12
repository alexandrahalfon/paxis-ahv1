"""
Lightweight direct-Qdrant retrieval for tumor-board specialty agents.

WHY THIS EXISTS
---------------
`ComprehensiveRetriever.retrieve_comprehensive()` is a heavy end-to-end
pipeline (LLM 8-axis re-extraction, PTO search, Postgres join, Phase 3
per-doc re-search, cross-encoder gate, reranking) designed to be called
ONCE per user query with a rich clinical narrative.

When the tumor board dispatches 6 agents × 3–4 specialty sub-queries,
that adds up to ~24 concurrent calls. Each one re-runs the whole
pipeline, which:

  1. Serializes on the sync Qdrant client inside the default thread pool
  2. Spends 20–240 s in PTO search per call
  3. Rejects almost every candidate study because the cross-encoder
     (ms-marco-MiniLM-L-6-v2) hates short keyword-stuffed specialty
     sub-queries.

Latency goes to several minutes and every agent ends with
`insufficient_evidence`.

This helper bypasses all of that. It does exactly:

  1. Embed the query (1 OpenAI call)
  2. Qdrant `query_points` (1 call)
  3. Group hits by `doc_id`

That's it. No cross-encoder, no PTO, no Postgres, no Phase 3. Each
agent does its own 2–3 lightweight searches in parallel, and the agent's
LLM synthesis step is what applies specialty judgement — the cross-encoder
is not the right tool for that job.

We still reuse the `ComprehensiveRetriever` singleton for its Qdrant +
OpenAI clients, so we don't double the connection count.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import qdrant_client.models as qm
from qdrant_client import QdrantClient


@dataclass
class LightweightStudy:
    """Minimal study shape the SpecialtyAgent synthesis path expects.

    Matches the duck-typed interface used by `base_agent._build_user_message`
    and `_merge_studies` so agents don't care whether studies came from
    this helper or from the full `ComprehensiveRetriever` pipeline.

    `source` and `sections_covered` are populated by the multi-specialty
    retrieval entry point (`multi_specialty_retrieval.py`) so the same
    object can be passed straight to converters that expect the
    `StudyEvidence` shape (Trial Match, Patient Matching). Tumor board
    agents do not read these fields.
    """

    doc_id: str
    title: str = "Unknown"
    citation: Optional[str] = None
    year: Optional[int] = None
    initial_score: float = 0.0
    rerank_score: float = 0.0
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    # Populated by multi_specialty_retrieval — duck-types StudyEvidence
    source: str = "tumor_board"
    sections_covered: set = field(default_factory=set)
    # Which specialty agents retrieved this study (e.g. ["medical_oncology",
    # "radiation_oncology"]). Empty when produced by a direct
    # `lightweight_search` call.
    specialties: List[str] = field(default_factory=list)


# ─── Dedicated Qdrant client + concurrency cap ────────────────────────────
#
# `get_comprehensive_retriever()` creates its QdrantClient without a
# `timeout` argument, which defaults to ~5 s. Against the managed Qdrant
# cloud instance a single vanilla query comfortably exceeds that (6+ s in
# the live logs), so every tumor board call times out. We own our own
# client with a 120 s timeout instead of mutating the /rag singleton.
#
# We also cap concurrent Qdrant calls with an asyncio.Semaphore so the
# 6 agents × ~3 sub-queries = ~18 parallel reads don't stampede the
# managed instance.

_QDRANT_CLIENT: Optional[QdrantClient] = None
_QDRANT_SEMAPHORE: Optional[asyncio.Semaphore] = None

#: Max concurrent Qdrant `query_points` calls. Empirically, a limit=25
#: search takes ~1–2 s; with 6 workers, ~18 parallel sub-queries drain in
#: about 3 batches ≈ 6 s.
QDRANT_CONCURRENCY = 6

#: Hard cap for one read; must comfortably exceed the observed p99 of a
#: single Qdrant call under load (~10 s).
QDRANT_TIMEOUT_S = 120


def _get_qdrant_client() -> QdrantClient:
    """Lazily build a dedicated QdrantClient with a generous timeout.

    Reuses the same URL + API key as /rag via `src.core.config.settings`
    but does NOT share the misconfigured singleton in
    `comprehensive_retrieval.get_comprehensive_retriever()`.
    """
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        from src.core.config import settings
        _QDRANT_CLIENT = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=QDRANT_TIMEOUT_S,
        )
        print(
            f"[tumor_board.retrieval] Dedicated Qdrant client built "
            f"(timeout={QDRANT_TIMEOUT_S}s)"
        )
    return _QDRANT_CLIENT


def _get_semaphore() -> asyncio.Semaphore:
    """Module-level asyncio.Semaphore capped at QDRANT_CONCURRENCY.

    Lazily constructed on first use so it binds to the currently-running
    event loop (Semaphore is loop-safe from 3.10 onwards).
    """
    global _QDRANT_SEMAPHORE
    if _QDRANT_SEMAPHORE is None:
        _QDRANT_SEMAPHORE = asyncio.Semaphore(QDRANT_CONCURRENCY)
    return _QDRANT_SEMAPHORE


async def lightweight_search(
    query_text: str,
    limit_points: int = 25,
    max_chunks_per_study: int = 4,
    min_chunks_per_study: int = 1,
    category: Optional[str] = None,
) -> List[LightweightStudy]:
    """Do a single direct-Qdrant vector search and return studies grouped
    by `doc_id`.

    Args:
        query_text: Specialty sub-query string.
        limit_points: How many Qdrant points to fetch. 25 typically yields
            8–15 unique studies after grouping.
        max_chunks_per_study: Upper bound on chunks kept per study (for
            the synthesis prompt size budget).
        min_chunks_per_study: Studies with fewer than this many chunks are
            discarded (default 1 — keep all).
        category: Optional exueed category filter (e.g. "head_neck").

    Returns:
        List of LightweightStudy sorted by best chunk score, descending.
        May be empty if Qdrant has nothing for this query — that's normal,
        the caller's specialty agent will fall back to merging across its
        other sub-queries.
    """
    # Lazy import — keeps this module importable during unit tests that
    # monkey-patch `agent._retrieve` and never touch Qdrant at all.
    from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
    from src.api.services.enhanced_rag_service import build_category_match_variants
    from src.core.config import settings

    # Reuse the singleton only for its OpenAI client + embedding method
    # and the collection name. We do NOT reuse its Qdrant client; ours
    # has the correct timeout.
    retriever = get_comprehensive_retriever()
    qdrant = _get_qdrant_client()
    sema = _get_semaphore()
    collection = retriever.collection or settings.qdrant_collection

    # Build the Qdrant filter. We ALWAYS exclude PTO frames from this
    # search — they live in the same collection but are separate
    # indexable objects handled by the PTO retriever (which the main
    # /rag pipeline owns).
    must_not = [
        qm.FieldCondition(
            key="node_type", match=qm.MatchValue(value="pto_frame")
        )
    ]
    # Use a multi-variant `should` clause for the category filter so a
    # single literal value (e.g. "h&n_processed_documents") doesn't
    # silently miss studies stored under any of the other accepted
    # spellings ("H&N", "h&n", "head_neck", etc.). Without this, an
    # H&N query that hits the wrong spelling returns ZERO points and
    # the caller falls back to no filter — which is how anal SCC,
    # glioma, and melanoma studies were leaking into head-and-neck
    # comparisons.
    should_clauses: List[Any] = []
    if category:
        variants = build_category_match_variants(category)
        if variants:
            should_clauses = [
                qm.FieldCondition(
                    key="category",
                    match=qm.MatchValue(value=v),
                )
                for v in variants
            ]
            print(
                f"[tumor_board.lightweight_search] category='{category}' "
                f"→ {len(variants)} match variants"
            )
        else:
            should_clauses = [
                qm.FieldCondition(
                    key="category",
                    match=qm.MatchValue(value=category),
                )
            ]
    qfilter = qm.Filter(
        must_not=must_not,
        should=should_clauses or None,
    )

    # Gate the whole embed+search under the semaphore so we cap concurrent
    # pressure on both OpenAI and Qdrant at QDRANT_CONCURRENCY.
    async with sema:
        # 0. Expand the query with all synonyms / brand names / staging
        #    variants / clinical context BEFORE embedding, so the vector
        #    lands in the right region regardless of vocabulary.
        try:
            from src.api.services.enhanced_rag_service import expand_query
            from src.api.services.query_expansion import expand_query_comprehensive
            expanded = expand_query(query_text)
            expanded = expand_query_comprehensive(expanded)
        except Exception:
            expanded = query_text

        # 1. Embed the EXPANDED query — sync OpenAI call in a worker thread
        try:
            query_embedding = await asyncio.to_thread(
                retriever.embed_query, expanded
            )
        except Exception as e:
            print(f"[tumor_board.lightweight_search] embed failed: {e}")
            return []

        # 2. Qdrant vector search — dedicated client, 120 s timeout
        try:
            results = await asyncio.to_thread(
                qdrant.query_points,
                collection_name=collection,
                query=query_embedding,
                limit=limit_points,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"[tumor_board.lightweight_search] qdrant query failed: {e}")
            return []

    points = getattr(results, "points", None) or []

    # 3. Group points by doc_id
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    doc_meta_cache: Dict[str, Dict[str, Any]] = {}

    for point in points:
        payload = dict(point.payload or {})
        # Defensive: some points may slip through that aren't regular chunks
        if payload.get("node_type") == "pto_frame":
            continue
        doc_id = payload.get("doc_id")
        if not doc_id:
            continue
        chunk = {
            "text": payload.get("text", "") or "",
            "section": payload.get("section"),
            "score": float(getattr(point, "score", 0.0) or 0.0),
            "chunk_id": getattr(point, "id", None),
            "doc_meta": payload.get("doc_meta", {}) or {},
            # Carry the full `metadata` dict (which holds the
            # doc_level_* fields the patient_match_scorer reads —
            # doc_level_cancer_types, doc_level_sites, etc.) so
            # downstream consumers (Patient Matching, Trial Match)
            # can score per-study without a second Qdrant round-trip.
            "metadata": payload.get("metadata", {}) or {},
        }
        by_doc[doc_id].append(chunk)
        if doc_id not in doc_meta_cache and chunk["doc_meta"]:
            doc_meta_cache[doc_id] = chunk["doc_meta"]

    # 4. Build LightweightStudy objects
    studies: List[LightweightStudy] = []
    for doc_id, chunks in by_doc.items():
        if len(chunks) < min_chunks_per_study:
            continue
        chunks.sort(key=lambda c: c["score"], reverse=True)
        top = chunks[:max_chunks_per_study]
        best_score = top[0]["score"] if top else 0.0
        meta = doc_meta_cache.get(doc_id, {}) or {}
        studies.append(
            LightweightStudy(
                doc_id=doc_id,
                title=meta.get("title", "Unknown") or "Unknown",
                citation=meta.get("citation"),
                year=meta.get("year"),
                initial_score=best_score,
                rerank_score=best_score,
                chunks=top,
            )
        )

    studies.sort(key=lambda s: s.rerank_score, reverse=True)
    return studies
