#!/usr/bin/env python
"""
Test patient matching using the EnhancedRAGService pipeline.

Compare results with the SimplePatientMatchingService.
"""

import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

print("[Test] Starting RAG patient matching test...", flush=True)

from src.api.services.enhanced_rag_service import get_enhanced_rag_service
from src.core.config import settings


async def test_rag_patient_matching():
    """Test patient matching using the full RAG pipeline."""
    
    print("=" * 80, flush=True)
    print("PATIENT MATCHING TEST - Using EnhancedRAGService", flush=True)
    print("=" * 80, flush=True)
    
    # Initialize RAG service
    print("\n[Setup] Initializing EnhancedRAGService...", flush=True)
    rag_service = get_enhanced_rag_service()
    
    # Build a query from the patient profile
    patient_query = """Find clinical studies for a patient with the following profile:
    
72-year-old man with localized prostate cancer:
- PSA: 5.6 ng/mL (elevated on screening)
- Normal prostate on digital rectal exam
- Biopsy: Gleason 3+4 in 1 core, Gleason 3+3 in 3 cores
- 4/12 cores positive
- No perineural invasion (PNI negative)
- Intermediate risk prostate cancer

What clinical trials or studies are relevant for this patient's treatment options?"""

    print(f"\n[Query] {patient_query[:200]}...", flush=True)
    
    try:
        # Run the RAG query
        print("\n[RAG] Running query...", flush=True)
        result = await rag_service.query(
            question=patient_query,
            category="prostate",
            top_k=10
        )
        
        print(f"\n[RAG] Query completed", flush=True)
        
        # Extract results - RAG returns 'evidence' not 'chunks'
        evidence = result.get("evidence", [])
        sources = result.get("sources", [])
        answer = result.get("answer", "")
        
        print(f"[RAG] Retrieved {len(evidence)} evidence chunks", flush=True)
        print(f"[RAG] Found {len(sources)} unique source doc_ids", flush=True)
        
        # Display the synthesized answer
        print("\n" + "=" * 60, flush=True)
        print("RAG SYNTHESIZED ANSWER", flush=True)
        print("=" * 60, flush=True)
        print(answer[:2000] if len(answer) > 2000 else answer, flush=True)
        
        # Display results
        print("\n" + "-" * 60, flush=True)
        print("TOP MATCHES FROM RAG PIPELINE", flush=True)
        print("-" * 60, flush=True)
        
        seen_docs = set()
        match_count = 0
        
        for i, chunk in enumerate(evidence[:15], 1):
            doc_id = chunk.get("doc_id", "Unknown")
            
            # Skip duplicates
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            match_count += 1
            
            title = chunk.get("title", "Unknown")
            year = chunk.get("year", "N/A")
            doi = chunk.get("doi", "N/A")
            score = chunk.get("score", 0)
            rerank_score = chunk.get("score_rerank", chunk.get("score_dense", score))
            
            print(f"\n  {match_count}. [{rerank_score:.3f}] {title}", flush=True)
            print(f"     Year: {year}", flush=True)
            print(f"     Doc ID: {doc_id}", flush=True)
            print(f"     DOI: {doi}", flush=True)
            
            # Show text excerpt
            text = chunk.get("text", "")[:300]
            if text:
                print(f"     Text: {text}...", flush=True)
        
        print(f"\n[RAG] Total unique studies: {match_count}", flush=True)
        
        # Save results to test_output.txt
        with open("test_output.txt", "a") as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("RAG PIPELINE PATIENT MATCHING TEST\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query: {patient_query[:200]}...\n\n")
            f.write(f"Total evidence chunks: {len(evidence)}\n")
            f.write(f"Unique source doc_ids: {len(sources)}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("SYNTHESIZED ANSWER:\n")
            f.write("-" * 60 + "\n")
            f.write(answer + "\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("TOP MATCHES:\n")
            f.write("-" * 60 + "\n")
            
            seen_docs = set()
            for i, chunk in enumerate(evidence[:15], 1):
                doc_id = chunk.get("doc_id", "Unknown")
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                
                title = chunk.get("title", "Unknown")
                year = chunk.get("year", "N/A")
                doi = chunk.get("doi", "N/A")
                score = chunk.get("score_rerank", chunk.get("score_dense", chunk.get("score", 0)))
                text = chunk.get("text", "")[:200]
                
                f.write(f"\n{len(seen_docs)}. [{score:.3f}] {title}\n")
                f.write(f"   Year: {year}\n")
                f.write(f"   Doc ID: {doc_id}\n")
                f.write(f"   DOI: {doi}\n")
                f.write(f"   Matched on: {text}...\n")
        
        print(f"\n[RAG] Results saved to test_output.txt", flush=True)
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[RAG] ERROR: {e}", flush=True)
        return False


if __name__ == "__main__":
    print("[Main] Script starting...", flush=True)
    success = asyncio.run(test_rag_patient_matching())
    print(f"\n[Main] Test completed with success={success}", flush=True)
    sys.exit(0 if success else 1)
