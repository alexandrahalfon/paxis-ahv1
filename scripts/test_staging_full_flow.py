#!/usr/bin/env python3
"""
Test the full staging flow for the oral tongue SCC case.
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
from dotenv import load_dotenv
load_dotenv()

TEST_QUERY = """A patient presents with a right lateral oral tongue SCC, 1.5 cm in size, 8mm depth of invasion, and ipsilateral adenopathy in level Ib (2 cm) and IIa (2.5cm). There is overt extranodal extension of the 1b node. Metastatic work-up is negative. What is the clinical stage?"""


def test_staging_detection():
    """Test if the staging detection correctly identifies this needs RAG fallback."""
    from src.api.services.staging_rag_fallback import detect_staging_keywords
    
    print("\n" + "="*80)
    print("STEP 1: STAGING KEYWORD DETECTION")
    print("="*80)
    
    result = detect_staging_keywords(TEST_QUERY)
    
    print(f"Has explicit staging: {result.has_explicit_staging}")
    print(f"Has clinical indicators: {result.has_clinical_indicators}")
    print(f"Needs RAG fallback: {result.needs_rag_fallback}")
    print(f"Detected staging keywords: {result.detected_staging_keywords}")
    print(f"Detected clinical indicators: {result.detected_clinical_indicators[:5]}")
    
    return result.needs_rag_fallback


def test_query_classifier():
    """Test the query classifier extraction."""
    from src.api.services.query_classifier_service import get_query_classifier_service
    
    print("\n" + "="*80)
    print("STEP 2: QUERY CLASSIFIER EXTRACTION")
    print("="*80)
    
    service = get_query_classifier_service()
    result = service.classify_query(TEST_QUERY)
    
    print(f"Cancer type: {result.cancer_type}")
    print(f"Cancer location: {result.cancer_location}")
    print(f"TNM T: {result.tnm_t}")
    print(f"TNM N: {result.tnm_n}")
    print(f"TNM M: {result.tnm_m}")
    print(f"Overall stage: {result.overall_stage}")
    print(f"Stage inferred: {result.stage_inferred}")
    print(f"Stage confidence: {result.stage_confidence}")
    print(f"Stage inference notes: {result.stage_inference_notes}")
    
    return result


async def test_rag_fallback():
    """Test the RAG fallback directly."""
    from src.api.services.staging_rag_fallback import staging_rag_fallback
    
    print("\n" + "="*80)
    print("STEP 3: RAG FALLBACK (DIRECT)")
    print("="*80)
    
    result = await staging_rag_fallback(
        query=TEST_QUERY,
        cancer_type="Squamous Cell Carcinoma",
        cancer_location="oral tongue"
    )
    
    if result:
        print(f"TNM T: {result.get('tnm_t')}")
        print(f"TNM N: {result.get('tnm_n')}")
        print(f"TNM M: {result.get('tnm_m')}")
        print(f"Overall stage: {result.get('overall_stage')}")
        print(f"Reasoning: {result.get('reasoning')}")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print("RAG fallback returned None")
    
    return result


def test_stage_inference_service():
    """Test the stage inference service with correct TNM values."""
    from src.api.services.stage_inference_service import infer_stage_for_query
    
    print("\n" + "="*80)
    print("STEP 4: STAGE INFERENCE SERVICE (with correct T2 N3b M0)")
    print("="*80)
    
    # Test with the CORRECT values
    result = infer_stage_for_query(
        cancer_type="Squamous Cell Carcinoma",
        cancer_location="oral tongue",
        tnm_t="T2",
        tnm_n="N3b",
        tnm_m="M0",
    )
    
    print(f"Stage group: {result.stage_group}")
    print(f"Confidence: {result.confidence}")
    print(f"Source: {result.source}")
    print(f"Cancer type key: {result.cancer_type_key}")
    print(f"TNM used: {result.tnm_used}")
    print(f"Notes: {result.notes}")
    
    return result


async def main():
    print("="*80)
    print("TESTING ORAL TONGUE SCC STAGING FLOW")
    print("="*80)
    print(f"\nQuery: {TEST_QUERY[:100]}...")
    print(f"\nExpected answer: T2N3bM0, Stage IVB")
    
    # Step 1: Detection
    needs_fallback = test_staging_detection()
    
    # Step 2: Query classifier
    classified = test_query_classifier()
    
    # Step 3: RAG fallback (if needed)
    if needs_fallback or not classified.overall_stage:
        rag_result = await test_rag_fallback()
    
    # Step 4: Stage inference with correct values
    inference = test_stage_inference_service()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Query classifier extracted: T={classified.tnm_t}, N={classified.tnm_n}, M={classified.tnm_m}")
    print(f"Query classifier stage: {classified.overall_stage}")
    print(f"Stage inference (T2N3bM0): {inference.stage_group}")
    
    if classified.overall_stage == "IVB" and classified.tnm_n == "N3B":
        print("\n[SUCCESS] Staging is correct!")
    else:
        print("\n[ISSUE] Staging may be incorrect - check the flow")


if __name__ == "__main__":
    asyncio.run(main())
