#!/usr/bin/env python3
"""
Compare different retrieval methods to find the best approach.

Tests:
1. Standard RAG retrieval (EnhancedHybridRetriever)
2. Study-focused two-phase retrieval (StudyFocusedRetriever)

Metrics compared:
- Number of unique studies retrieved
- Sections covered per study
- Retrieval time
- Evidence quality (diversity, completeness)
"""

import asyncio
import sys
import os
import time
from typing import Dict, List, Any
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# Test queries - mix of general and patient-specific
TEST_QUERIES = [
    # General knowledge query
    "What is the standard dose for adjuvant radiation in breast cancer?",
    
    # Patient-specific query
    "65 year old female with T2N1 ER+ breast cancer s/p lumpectomy, what radiation dose?",
    
    # Trial-specific query
    "What were the outcomes of the RTOG 0617 trial for lung cancer?",
    
    # Comparison query
    "Compare hypofractionated vs conventional fractionation for prostate cancer",
]


def print_separator(title: str):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_evidence(evidence: List[Dict[str, Any]], method_name: str) -> Dict[str, Any]:
    """Analyze evidence quality metrics."""
    if not evidence:
        return {"error": "No evidence retrieved"}
    
    # Count unique studies
    doc_ids = set()
    sections_by_doc = defaultdict(set)
    chunks_by_doc = defaultdict(int)
    
    for e in evidence:
        doc_id = e.get("doc_id")
        if doc_id:
            doc_ids.add(doc_id)
            section = e.get("section")
            if section:
                sections_by_doc[doc_id].add(section)
            chunks_by_doc[doc_id] += 1
    
    # Calculate metrics
    total_chunks = len(evidence)
    unique_studies = len(doc_ids)
    avg_chunks_per_study = total_chunks / unique_studies if unique_studies > 0 else 0
    avg_sections_per_study = sum(len(s) for s in sections_by_doc.values()) / unique_studies if unique_studies > 0 else 0
    
    # Get top studies info
    top_studies = []
    for doc_id in list(doc_ids)[:5]:
        study_evidence = [e for e in evidence if e.get("doc_id") == doc_id]
        if study_evidence:
            top_studies.append({
                "doc_id": doc_id[:30] + "..." if len(doc_id) > 30 else doc_id,
                "title": (study_evidence[0].get("title") or "Unknown")[:50],
                "chunks": chunks_by_doc[doc_id],
                "sections": list(sections_by_doc[doc_id])[:5],
            })
    
    return {
        "method": method_name,
        "total_chunks": total_chunks,
        "unique_studies": unique_studies,
        "avg_chunks_per_study": round(avg_chunks_per_study, 2),
        "avg_sections_per_study": round(avg_sections_per_study, 2),
        "top_studies": top_studies,
    }


async def test_standard_retrieval(query: str) -> Dict[str, Any]:
    """Test standard RAG retrieval."""
    from src.api.services.enhanced_rag_service import get_enhanced_rag_service
    
    print(f"\n[Standard RAG] Testing...")
    t_start = time.perf_counter()
    
    try:
        rag_service = get_enhanced_rag_service()
        result = await rag_service.query(
            question=query,
            top_k=10,
            use_study_focused=False,
        )
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        evidence = result.get("evidence", [])
        
        analysis = analyze_evidence(evidence, "Standard RAG")
        analysis["retrieval_time_ms"] = round(elapsed_ms, 1)
        analysis["query_type"] = result.get("query_type", "unknown")
        
        # Add metadata info
        metadata = result.get("metadata", {})
        analysis["postgres_matches"] = metadata.get("structured_match", {}).get("doc_ids_matched", 0)
        analysis["clinical_profile"] = bool(metadata.get("clinical_profile"))
        
        return analysis
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": "Standard RAG"}


async def test_study_focused_retrieval(query: str) -> Dict[str, Any]:
    """Test study-focused two-phase retrieval."""
    from src.api.services.study_focused_retrieval import (
        get_study_focused_retriever,
        format_study_grouped_evidence,
    )
    
    print(f"\n[Study-Focused] Testing...")
    t_start = time.perf_counter()
    
    try:
        retriever = get_study_focused_retriever()
        result = await retriever.retrieve_study_focused(
            query_text=query,
            max_studies=5,
            chunks_per_study=8,
        )
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        evidence = format_study_grouped_evidence(result)
        
        analysis = analyze_evidence(evidence, "Study-Focused")
        analysis["retrieval_time_ms"] = round(elapsed_ms, 1)
        analysis["phase1_studies"] = result.phase1_doc_count
        analysis["postgres_matches"] = result.structured_match_count
        analysis["clinical_profile"] = bool(result.clinical_profile)
        
        # Add study-specific info
        analysis["studies_detail"] = [
            {
                "doc_id": s.doc_id[:30] + "..." if len(s.doc_id) > 30 else s.doc_id,
                "title": s.title[:50] if s.title else "Unknown",
                "relevance": round(s.relevance_score, 3),
                "chunks": len(s.chunks),
                "sections": list(s.sections_covered)[:5],
            }
            for s in result.studies
        ]
        
        return analysis
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": "Study-Focused"}


def print_analysis(analysis: Dict[str, Any]):
    """Pretty print analysis results."""
    if "error" in analysis:
        print(f"  ERROR: {analysis['error']}")
        return
    
    print(f"\n  Method: {analysis['method']}")
    print(f"  Retrieval Time: {analysis.get('retrieval_time_ms', 'N/A')} ms")
    print(f"  Total Chunks: {analysis['total_chunks']}")
    print(f"  Unique Studies: {analysis['unique_studies']}")
    print(f"  Avg Chunks/Study: {analysis['avg_chunks_per_study']}")
    print(f"  Avg Sections/Study: {analysis['avg_sections_per_study']}")
    print(f"  PostgreSQL Matches: {analysis.get('postgres_matches', 'N/A')}")
    print(f"  Clinical Profile: {analysis.get('clinical_profile', 'N/A')}")
    
    if analysis.get('query_type'):
        print(f"  Query Type: {analysis['query_type']}")
    
    print(f"\n  Top Studies:")
    studies = analysis.get('studies_detail') or analysis.get('top_studies', [])
    for i, study in enumerate(studies[:5], 1):
        print(f"    {i}. {study.get('title', 'Unknown')}")
        print(f"       Chunks: {study.get('chunks', 'N/A')}, Sections: {study.get('sections', [])}")
        if 'relevance' in study:
            print(f"       Relevance: {study['relevance']}")


async def compare_methods(query: str):
    """Compare all retrieval methods for a single query."""
    print_separator(f"Query: {query[:70]}...")
    
    # Run both methods
    standard_result = await test_standard_retrieval(query)
    study_focused_result = await test_study_focused_retrieval(query)
    
    # Print results
    print("\n" + "-" * 40)
    print("STANDARD RAG RETRIEVAL")
    print("-" * 40)
    print_analysis(standard_result)
    
    print("\n" + "-" * 40)
    print("STUDY-FOCUSED RETRIEVAL")
    print("-" * 40)
    print_analysis(study_focused_result)
    
    # Compare
    print("\n" + "-" * 40)
    print("COMPARISON SUMMARY")
    print("-" * 40)
    
    if "error" not in standard_result and "error" not in study_focused_result:
        time_diff = study_focused_result.get('retrieval_time_ms', 0) - standard_result.get('retrieval_time_ms', 0)
        section_diff = study_focused_result['avg_sections_per_study'] - standard_result['avg_sections_per_study']
        
        print(f"  Time difference: {time_diff:+.1f} ms (+ = study-focused slower)")
        print(f"  Section coverage diff: {section_diff:+.2f} sections/study (+ = study-focused better)")
        print(f"  Study-focused retrieves more sections per study: {section_diff > 0}")
    
    return {
        "query": query,
        "standard": standard_result,
        "study_focused": study_focused_result,
    }


async def main():
    """Run comparison tests."""
    print_separator("RETRIEVAL METHOD COMPARISON TEST")
    print(f"Testing {len(TEST_QUERIES)} queries...")
    
    all_results = []
    
    for query in TEST_QUERIES:
        result = await compare_methods(query)
        all_results.append(result)
    
    # Final summary
    print_separator("FINAL SUMMARY")
    
    standard_times = []
    study_focused_times = []
    standard_sections = []
    study_focused_sections = []
    
    for r in all_results:
        if "error" not in r["standard"]:
            standard_times.append(r["standard"].get("retrieval_time_ms", 0))
            standard_sections.append(r["standard"]["avg_sections_per_study"])
        if "error" not in r["study_focused"]:
            study_focused_times.append(r["study_focused"].get("retrieval_time_ms", 0))
            study_focused_sections.append(r["study_focused"]["avg_sections_per_study"])
    
    if standard_times and study_focused_times:
        print(f"\n  Average Retrieval Time:")
        print(f"    Standard RAG: {sum(standard_times)/len(standard_times):.1f} ms")
        print(f"    Study-Focused: {sum(study_focused_times)/len(study_focused_times):.1f} ms")
        
        print(f"\n  Average Sections per Study:")
        print(f"    Standard RAG: {sum(standard_sections)/len(standard_sections):.2f}")
        print(f"    Study-Focused: {sum(study_focused_sections)/len(study_focused_sections):.2f}")
        
        section_improvement = (sum(study_focused_sections)/len(study_focused_sections)) - (sum(standard_sections)/len(standard_sections))
        print(f"\n  Section Coverage Improvement: {section_improvement:+.2f} sections/study")
    
    print("\n" + "=" * 80)
    print("  Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
