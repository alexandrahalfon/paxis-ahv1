#!/usr/bin/env python3
"""
RAG Accuracy Diagnostic Tool
============================

Diagnoses why questions are failing by checking:
1. Document coverage - Is the answer in the knowledge base?
2. Retrieval precision - Are the right chunks being retrieved?
3. LLM interpretation - Is the LLM extracting the right answer?

Usage:
    python scripts/diagnose_rag_accuracy.py
    python scripts/diagnose_rag_accuracy.py --question "What is the dose for..."
    python scripts/diagnose_rag_accuracy.py --limit 10
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

load_dotenv('.env')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.api.services.enhanced_rag_service import get_enhanced_rag_service
from src.api.services.clinical_entity_extractor import get_clinical_entity_extractor

# Sample questions for diagnosis
DIAGNOSTIC_QUESTIONS = [
    {
        "question": "For DCIS, which feature is associated with an elevated risk of in-breast recurrence?",
        "expected_answer": "Clinical detection",
        "key_terms": ["clinical detection", "mammographic", "EORTC 10853", "Donker"],
        "category": "breast"
    },
    {
        "question": "A patient with metastatic NSCLC with four sites of bony metastasis, which describes an appropriate management strategy?",
        "expected_answer": "Platinum-based 2-drug combination therapy with pembrolizumab if the PD-L1 status is 30%",
        "key_terms": ["pembrolizumab", "platinum", "PD-L1", "KEYNOTE-189", "chemotherapy"],
        "category": "lung"
    },
    {
        "question": "A female with a cT3N1M0 ER/PR- Her2+ breast cancer receives neoadjuvant TCHP followed by mastectomy with sentinel lymph node biopsy and achieves a pCR. What adjuvant therapy is recommended?",
        "expected_answer": "RT to the chest wall and regional lymph nodes with concurrent trastuzumab +/- pertuzumab",
        "key_terms": ["chest wall", "regional nodes", "trastuzumab", "PMRT", "pCR still needs RT"],
        "category": "breast"
    },
    {
        "question": "What is the recommended RT technique for stage I testicular seminoma?",
        "expected_answer": "Para-aortic strip irradiation with 30 Gy in 15 fractions",
        "key_terms": ["para-aortic", "PA strip", "30 Gy", "20 Gy", "seminoma"],
        "category": "GU"
    },
    {
        "question": "What is the BEST treatment for a 55 year-old female who underwent breast-conserving surgery for a pT1cN1mi cM0 ER+ HER2- breast cancer and 21 gene recurrence score of 22?",
        "expected_answer": "RT followed by endocrine therapy",
        "key_terms": ["endocrine therapy", "no chemotherapy", "TAILORx", "recurrence score", "midrange"],
        "category": "breast"
    },
]


def check_document_coverage(rag_service, question: str, key_terms: List[str]) -> Dict[str, Any]:
    """Check if key terms exist in retrieved documents."""
    evidence, metadata = rag_service.retriever.retrieve(question, k_final=15)
    
    # Combine all retrieved text
    all_text = " ".join([e.get("text", "") for e in evidence]).lower()
    
    # Check which key terms are found
    found_terms = []
    missing_terms = []
    for term in key_terms:
        if term.lower() in all_text:
            found_terms.append(term)
        else:
            missing_terms.append(term)
    
    coverage = len(found_terms) / len(key_terms) if key_terms else 0
    
    return {
        "coverage_score": coverage,
        "found_terms": found_terms,
        "missing_terms": missing_terms,
        "num_chunks_retrieved": len(evidence),
        "top_citations": [e.get("citation", "Unknown")[:50] for e in evidence[:5]],
    }


def check_retrieval_precision(evidence: List[Dict], expected_answer: str, key_terms: List[str]) -> Dict[str, Any]:
    """Check if retrieved chunks contain the expected answer."""
    expected_lower = expected_answer.lower()
    
    # Check each chunk for answer presence
    chunks_with_answer = []
    chunks_with_key_terms = []
    
    for i, e in enumerate(evidence):
        text = (e.get("text", "") or "").lower()
        
        # Check for expected answer
        if any(word in text for word in expected_lower.split()[:3]):
            chunks_with_answer.append(i)
        
        # Check for key terms
        terms_in_chunk = [t for t in key_terms if t.lower() in text]
        if terms_in_chunk:
            chunks_with_key_terms.append({
                "chunk_index": i,
                "terms_found": terms_in_chunk,
                "citation": e.get("citation", "Unknown")[:50]
            })
    
    return {
        "chunks_with_answer": chunks_with_answer,
        "chunks_with_key_terms": chunks_with_key_terms,
        "answer_in_top_3": any(i < 3 for i in chunks_with_answer),
        "key_terms_in_top_3": any(c["chunk_index"] < 3 for c in chunks_with_key_terms),
    }


def diagnose_question(rag_service, question_data: Dict) -> Dict[str, Any]:
    """Run full diagnosis on a single question."""
    question = question_data["question"]
    expected = question_data["expected_answer"]
    key_terms = question_data["key_terms"]
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {question[:60]}...")
    print(f"EXPECTED: {expected}")
    print(f"{'='*70}")
    
    # 1. Extract clinical profile
    extractor = get_clinical_entity_extractor()
    profile = extractor.extract(question)
    print(f"\n[1] Clinical Profile:")
    print(f"    Cancer Type: {profile.cancer_type}")
    print(f"    Biomarkers: {profile.biomarkers}")
    print(f"    Treatments: {profile.treatments}")
    print(f"    Concepts: {profile.clinical_concepts}")
    
    # 2. Check document coverage
    coverage = check_document_coverage(rag_service, question, key_terms)
    print(f"\n[2] Document Coverage: {coverage['coverage_score']*100:.0f}%")
    print(f"    Found: {coverage['found_terms']}")
    print(f"    Missing: {coverage['missing_terms']}")
    print(f"    Top Citations: {coverage['top_citations'][:3]}")
    
    # 3. Get full RAG response
    result = rag_service.query(question, top_k=10)
    evidence = result.get("evidence", [])
    answer = result.get("answer", "")
    
    # 4. Check retrieval precision
    precision = check_retrieval_precision(evidence, expected, key_terms)
    print(f"\n[3] Retrieval Precision:")
    print(f"    Answer in top 3 chunks: {precision['answer_in_top_3']}")
    print(f"    Key terms in top 3: {precision['key_terms_in_top_3']}")
    
    # 5. Check LLM answer
    answer_correct = any(word.lower() in answer.lower() for word in expected.split()[:3])
    print(f"\n[4] LLM Answer Check:")
    print(f"    Generated: {answer[:150]}...")
    print(f"    Contains expected keywords: {answer_correct}")
    
    # 6. Determine failure mode
    if coverage['coverage_score'] < 0.5:
        failure_mode = "DOCUMENT_COVERAGE"
        recommendation = "The knowledge base is missing key documents. Ingest relevant papers."
    elif not precision['key_terms_in_top_3']:
        failure_mode = "RETRIEVAL_PRECISION"
        recommendation = "Documents exist but wrong chunks retrieved. Improve query expansion or reranking."
    elif not answer_correct:
        failure_mode = "LLM_INTERPRETATION"
        recommendation = "Right chunks retrieved but LLM extracted wrong answer. Improve prompts."
    else:
        failure_mode = "NONE"
        recommendation = "Answer appears correct!"
    
    print(f"\n[5] DIAGNOSIS: {failure_mode}")
    print(f"    Recommendation: {recommendation}")
    
    return {
        "question": question,
        "expected": expected,
        "generated": answer[:200],
        "profile": profile.to_dict(),
        "coverage": coverage,
        "precision": precision,
        "failure_mode": failure_mode,
        "recommendation": recommendation,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Specific question to diagnose")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to diagnose")
    args = parser.parse_args()
    
    print("Initializing RAG service...")
    rag_service = get_enhanced_rag_service()
    
    results = []
    
    if args.question:
        # Diagnose specific question
        question_data = {
            "question": args.question,
            "expected_answer": "Unknown",
            "key_terms": args.question.lower().split()[:5],
        }
        result = diagnose_question(rag_service, question_data)
        results.append(result)
    else:
        # Diagnose sample questions
        for q in DIAGNOSTIC_QUESTIONS[:args.limit]:
            result = diagnose_question(rag_service, q)
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    failure_counts = {}
    for r in results:
        mode = r["failure_mode"]
        failure_counts[mode] = failure_counts.get(mode, 0) + 1
    
    for mode, count in sorted(failure_counts.items()):
        print(f"  {mode}: {count}")
    
    # Save results
    output_file = "rag_diagnosis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
