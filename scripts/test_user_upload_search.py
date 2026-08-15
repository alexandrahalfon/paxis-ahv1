#!/usr/bin/env python3
"""Test user upload search integration."""

import asyncio
import sys
sys.path.insert(0, '.')

async def test_query():
    from src.api.services.enhanced_rag_service import get_enhanced_rag_service
    from src.api.services.user_document_search import get_user_document_search_service
    
    # Your user ID from the logs
    user_id = "57e952f5-c624-4906-a67a-366e3668cce9"
    question = "what is the estimated 24-month disease-free survival with cemiplimab in high-risk cutaneous squamous-cell carcinoma"
    
    print(f"\n{'='*60}")
    print(f"Testing query: {question[:60]}...")
    print(f"User ID: {user_id}")
    print(f"{'='*60}\n")
    
    # Get RAG service
    rag_service = get_enhanced_rag_service()
    
    # Get query embedding
    print("Generating query embedding...")
    query_embedding = rag_service.retriever.embed_query(question)
    print(f"Embedding dimension: {len(query_embedding)}")
    
    # Search user documents
    print("\nSearching user uploads...")
    user_search = get_user_document_search_service()
    user_results = await user_search.search_user_documents(
        user_id=user_id,
        query_embedding=query_embedding,
        query_text=question,
        top_k=10,
        alpha=0.7
    )
    
    print(f"\nFound {len(user_results)} user upload results:")
    for i, r in enumerate(user_results[:5]):
        print(f"\n  [{i+1}] Score: {r.get('similarity_score', 0):.4f}")
        print(f"      Doc: {r.get('doc_id', 'N/A')[:50]}")
        print(f"      Title: {r.get('title', 'N/A')[:50]}")
        print(f"      Text: {r.get('text', '')[:150]}...")
    
    # Run Qdrant query
    print("\n\nRunning Qdrant query...")
    rag_result = rag_service.query(
        question=question,
        query_mode="hybrid",
        top_k=10
    )
    
    qdrant_evidence = rag_result.get("evidence", [])
    print(f"\nFound {len(qdrant_evidence)} Qdrant results:")
    for i, e in enumerate(qdrant_evidence[:5]):
        print(f"\n  [{i+1}] Score: {e.get('score', 0):.4f}")
        print(f"      Doc: {e.get('doc_id', 'N/A')[:50]}")
        print(f"      Title: {e.get('title', 'N/A')[:50]}")
        print(f"      Text: {e.get('text', '')[:150]}...")
    
    # Compare scores
    print("\n\n" + "="*60)
    print("SCORE COMPARISON:")
    print("="*60)
    
    if user_results:
        user_top = user_results[0].get('similarity_score', 0)
        print(f"Top user upload score: {user_top:.4f}")
    
    if qdrant_evidence:
        qdrant_top = qdrant_evidence[0].get('score', 0)
        print(f"Top Qdrant score: {qdrant_top:.4f}")
    
    # Test the score normalization
    if user_results and qdrant_evidence:
        max_qdrant = max(e.get("score", 0) for e in qdrant_evidence)
        max_user = max(r.get("similarity_score", 0) for r in user_results)
        scale_factor = max_qdrant / max_user if max_user > 0 else 1
        print(f"\nScale factor for user scores: {scale_factor:.2f}x")
        print(f"Scaled top user score: {user_top * scale_factor:.4f}")
    
    print("\nRAG Answer (first 500 chars):")
    print("-" * 40)
    print(rag_result.get("answer", "No answer")[:500])

if __name__ == "__main__":
    asyncio.run(test_query())
