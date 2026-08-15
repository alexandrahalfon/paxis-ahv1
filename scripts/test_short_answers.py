#!/usr/bin/env python3
"""
Short Answer ACR Board Exam Test for Paxis RAG Pipeline
=========================================================

Tests the SHORT ANSWER generation against all 50 ACR radiation oncology 
board questions with ground truth answers and citations.

Usage:
python test_short_answers.py              # Run all 50 questions
python test_short_answers.py --limit 10   # Run first 10
python test_short_answers.py --category breast  # Run specific category

Output:
- acr_short_answer_report_TIMESTAMP.json (detailed results)
- acr_short_answer_report_TIMESTAMP.html (beautiful side-by-side comparison)
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from difflib import SequenceMatcher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

required_vars = ['OPENAI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

try:
    from src.api.services.enhanced_rag_service import get_enhanced_rag_service
    from src.api.routes.query import _generate_short_answer_with_llm
    from src.core.config import settings
    from openai import OpenAI
except ImportError as e:
    print(f"ERROR: Cannot import services: {e}")
    sys.exit(1)

from scripts.test_acr_complete import ACR_QUESTIONS

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def check_key_concepts(answer: str, key_concepts: List[str]) -> Tuple[List[str], float]:
    answer_lower = answer.lower()
    matched = [c for c in key_concepts if c.lower() in answer_lower]
    coverage = len(matched) / len(key_concepts) if key_concepts else 0
    return matched, coverage

def extract_citations(text: str) -> List[str]:
    pattern = r"\([^)]*\d{4}[^)]*\)"
    citations = re.findall(pattern, text)
    return [c.strip('()') for c in citations]

def check_citations(answer: str, expected_cits: List[str]) -> Tuple[List[str], float]:
    answer_lower = answer.lower()
    matched = [cit for cit in expected_cits if cit.lower() in answer_lower]
    coverage = len(matched) / len(expected_cits) if expected_cits else 0
    return matched, coverage

def score_short_answer(short_answer: str, justification: str, question: Dict[str, Any]) -> Dict[str, Any]:
    short_similarity = calculate_semantic_similarity(short_answer, question['answer'])
    short_matched_concepts, short_concept_coverage = check_key_concepts(short_answer, question['key_concepts'])
    just_matched_concepts, just_concept_coverage = check_key_concepts(justification, question['key_concepts'])
    matched_citations, citation_coverage = check_citations(justification, question['citations'])
    
    if short_similarity > 0.7 and short_concept_coverage > 0.5:
        overall = "EXCELLENT"
    elif short_similarity > 0.5 and short_concept_coverage > 0.3:
        overall = "PASS"
    elif short_similarity > 0.3 or short_concept_coverage > 0.3:
        overall = "PARTIAL"
    else:
        overall = "FAIL"
    
    return {
        "short_similarity": short_similarity,
        "short_concept_coverage": short_concept_coverage,
        "short_matched_concepts": short_matched_concepts,
        "justification_concept_coverage": just_concept_coverage,
        "justification_matched_concepts": just_matched_concepts,
        "citation_coverage": citation_coverage,
        "matched_citations": matched_citations,
        "overall": overall,
        "justification_citations": extract_citations(justification)
    }

def run_short_answer_test(limit: int = None, category: str = None):
    print("="*80)
    print("SHORT ANSWER ACR BOARD EXAM TEST")
    print("="*80)
    
    questions = ACR_QUESTIONS
    if category:
        questions = [q for q in questions if q['category'] == category]
    if limit:
        questions = questions[:limit]
    
    print(f"Total Questions: {len(questions)}")
    
    rag_service = get_enhanced_rag_service()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[Test {i}/{len(questions)}] {question['id']}")
        
        try:
            rag_result = rag_service.query(question=question['question'], top_k=10)
            evidence = rag_result.get("evidence", [])
            justification = rag_result.get("answer", "")
            pto_frames = rag_result.get("pto_frames")
            
            short_answer = _generate_short_answer_with_llm(
                question=question['question'],
                evidence=evidence,
                openai_client=openai_client,
                justification=justification
            )
            
            if not short_answer:
                short_answer = justification[:150] + "..."
            
            score = score_short_answer(short_answer, justification, question)
            
            print(f"Short: {short_answer[:80]}...")
            print(f"Truth: {question['answer']}")
            print(f"Score: {score['overall']}")
            
            results.append({
                **question,
                "short_answer": short_answer,
                "justification": justification,
                "score": score,
                "pto_frames_used": len(pto_frames) if pto_frames else 0
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({**question, "error": str(e), "score": {"overall": "ERROR"}})
    
    generate_reports(results)

def generate_reports(results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total = len(results)
    valid = [r for r in results if 'error' not in r]
    passed = sum(1 for r in valid if r['score']['overall'] in ['EXCELLENT', 'PASS'])
    
    print(f"\nPassed: {passed}/{total}")
    
    json_file = f"acr_short_answer_report_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"JSON: {json_file}")
    
    html_file = f"acr_short_answer_report_{timestamp}.html"
    with open(html_file, 'w') as f:
        f.write(f"<html><head><title>Short Answer Test</title></head><body>")
        f.write(f"<h1>Short Answer ACR Test - {passed}/{total} passed</h1>")
        for r in results:
            if 'error' in r:
                f.write(f"<div style='background:#fee'><h3>{r['id']} - ERROR</h3></div>")
            else:
                bg = '#dfd' if r['score']['overall'] in ['EXCELLENT','PASS'] else '#ffd' if r['score']['overall']=='PARTIAL' else '#fdd'
                f.write(f"<div style='background:{bg};margin:10px;padding:10px'>")
                f.write(f"<h3>{r['id']} - {r['score']['overall']}</h3>")
                f.write(f"<p><b>Q:</b> {r['question'][:100]}...</p>")
                f.write(f"<p><b>Short:</b> {r['short_answer']}</p>")
                f.write(f"<p><b>Truth:</b> {r['answer']}</p>")
                f.write(f"<p>Similarity: {r['score']['short_similarity']:.1%}</p>")
                f.write("</div>")
        f.write("</body></html>")
    print(f"HTML: {html_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--category', type=str)
    args = parser.parse_args()
    run_short_answer_test(limit=args.limit, category=args.category)
