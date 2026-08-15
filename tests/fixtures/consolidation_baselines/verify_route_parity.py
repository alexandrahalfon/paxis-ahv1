#!/usr/bin/env python3
"""
Verify route parity and PTO index for RAG Pipeline Consolidation (Phase 0).

Validates Requirements 1.2 and 1.3:
  1.2 — /rag/query and /rag/query/enhanced route to the same retrieval backbone
  1.3 — Query Qdrant for node_type=pto_frame count and log result

Can be run as a standalone script or imported for use in tests.

Usage:
    python tests/fixtures/consolidation_baselines/verify_route_parity.py
"""

import ast
import inspect
import os
import sys
import textwrap
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Route parity verification (Requirement 1.2)
# ---------------------------------------------------------------------------

def _get_function_source(func) -> str:
    """Return the source code of a function."""
    return inspect.getsource(func)


def verify_route_parity() -> Dict[str, object]:
    """Assert that /rag/query and /rag/query/enhanced share the same retrieval backbone.

    Inspects the actual route handler source code to confirm both endpoints
    ultimately call ``rag_service.query()`` from ``EnhancedRAGService``, which
    in turn delegates to ``ComprehensiveRetrieval.retrieve_comprehensive()``
    for study-focused retrieval.

    Returns a dict with:
        routes_use_same_service (bool): True if both handlers import and call
            the same ``get_enhanced_rag_service`` factory.
        standard_calls_rag_service_query (bool): True if /rag/query calls
            ``rag_service.query()``.
        enhanced_calls_rag_service_query (bool): True if /rag/query/enhanced
            calls ``rag_service.query()``.
        rag_service_delegates_to_comprehensive (bool): True if
            ``EnhancedRAGService.query()`` delegates study-focused retrieval
            to ``ComprehensiveRetrieval.retrieve_comprehensive()``.
        parity_confirmed (bool): True when all checks pass.
        details (str): Human-readable summary.
    """
    from src.api.routes.query import query_knowledge_base, enhanced_query

    std_src = _get_function_source(query_knowledge_base)
    enh_src = _get_function_source(enhanced_query)

    # Both handlers should import the same RAG service factory
    routes_use_same_service = (
        "get_enhanced_rag_service" in std_src
        and "get_enhanced_rag_service" in enh_src
    )

    # Both should call rag_service.query(...)
    standard_calls = "rag_service.query(" in std_src or "rag_service.query(" in std_src.replace("\n", "")
    enhanced_calls = "rag_service.query(" in enh_src or "rag_result = await rag_service.query(" in enh_src

    # EnhancedRAGService.query() should delegate to ComprehensiveRetrieval
    from src.api.services.enhanced_rag_service import EnhancedRAGService
    query_src = _get_function_source(EnhancedRAGService.query)
    delegates = "retrieve_comprehensive" in query_src or "query_study_focused" in query_src

    # query_study_focused should call retrieve_comprehensive
    qsf_src = _get_function_source(EnhancedRAGService.query_study_focused)
    qsf_delegates = "retrieve_comprehensive" in qsf_src

    parity = all([routes_use_same_service, standard_calls, enhanced_calls, delegates, qsf_delegates])

    details_lines = [
        f"routes_use_same_service: {routes_use_same_service}",
        f"standard_calls_rag_service_query: {standard_calls}",
        f"enhanced_calls_rag_service_query: {enhanced_calls}",
        f"rag_service_delegates_to_comprehensive: {delegates}",
        f"query_study_focused_calls_retrieve_comprehensive: {qsf_delegates}",
        f"parity_confirmed: {parity}",
    ]

    result = {
        "routes_use_same_service": routes_use_same_service,
        "standard_calls_rag_service_query": standard_calls,
        "enhanced_calls_rag_service_query": enhanced_calls,
        "rag_service_delegates_to_comprehensive": delegates,
        "query_study_focused_calls_retrieve_comprehensive": qsf_delegates,
        "parity_confirmed": parity,
        "details": "\n".join(details_lines),
    }
    return result


# ---------------------------------------------------------------------------
# 2. PTO index count (Requirement 1.3)
# ---------------------------------------------------------------------------

def query_pto_frame_count(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, object]:
    """Query Qdrant for the number of points with node_type=pto_frame.

    Falls back to environment variables / settings when arguments are None.

    Returns a dict with:
        pto_frame_count (int | None): Number of PTO frame points, or None on error.
        collection (str): Collection queried.
        connected (bool): Whether the Qdrant connection succeeded.
        error (str | None): Error message if connection failed.
    """
    from dotenv import load_dotenv
    load_dotenv()

    url = qdrant_url or os.getenv("QDRANT_URL", "")
    api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
    collection = collection_name or os.getenv("QDRANT_COLLECTION", "exueed_kb_latest")

    if not url:
        return {
            "pto_frame_count": None,
            "collection": collection,
            "connected": False,
            "error": "QDRANT_URL not set — cannot connect to Qdrant",
        }

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = QdrantClient(url=url, api_key=api_key, timeout=15)

        # Use count API with a filter for node_type=pto_frame
        count_result = client.count(
            collection_name=collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="node_type",
                        match=MatchValue(value="pto_frame"),
                    )
                ]
            ),
            exact=True,
        )

        pto_count = count_result.count

        print(f"[PTO Index] Collection: {collection}")
        print(f"[PTO Index] node_type=pto_frame count: {pto_count}")

        return {
            "pto_frame_count": pto_count,
            "collection": collection,
            "connected": True,
            "error": None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "pto_frame_count": None,
            "collection": collection,
            "connected": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 3. Combined verification entry point
# ---------------------------------------------------------------------------

def run_all_verifications() -> Dict[str, object]:
    """Run route parity check and PTO index query, returning combined results."""
    print("=" * 70)
    print("  Phase 0 — Route Parity & PTO Index Verification")
    print("=" * 70)

    # --- Route parity ---
    print("\n[Step 1] Verifying route parity (/rag/query vs /rag/query/enhanced)...")
    parity = verify_route_parity()
    for line in parity["details"].split("\n"):
        print(f"  {line}")
    if parity["parity_confirmed"]:
        print("  RESULT: PASS — both routes share the same retrieval backbone")
    else:
        print("  RESULT: FAIL — routes diverge (see details above)")

    # --- PTO index ---
    print("\n[Step 2] Querying Qdrant for node_type=pto_frame count...")
    pto = query_pto_frame_count()
    if pto["connected"]:
        print(f"  RESULT: {pto['pto_frame_count']} PTO frames in {pto['collection']}")
    else:
        print(f"  RESULT: Could not connect — {pto['error']}")

    print("\n" + "=" * 70)
    return {"route_parity": parity, "pto_index": pto}


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all_verifications()

    # Exit with non-zero if parity check failed
    if not results["route_parity"]["parity_confirmed"]:
        sys.exit(1)
