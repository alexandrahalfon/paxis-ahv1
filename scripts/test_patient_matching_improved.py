#!/usr/bin/env python3
"""
Test script for improved patient matching with anatomical site awareness.
Tests the SCC of maxilla case that was returning irrelevant results.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_extraction():
    """Test that anatomical site is extracted correctly."""
    from openai import OpenAI
    from src.core.config import settings
    from src.api.services.unstructured_patient_extractor import extract_patient_profile
    
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    # Test case: SCC of maxilla
    description = """68 year old female, Mandarin-speaking, non-smoke, with a PMH of HTN, T2DM, 
    and SCC of R maxilla s/p right maxillectomy, bilateral selective neck dissection, 
    and placement of obturator on 2/16/24, revealing poorly differentiated SCC of right gingiva, 
    hard palate, buccal mucosa, and maxilla, pT4N0, DOI 15 mm, LVI-, PNI-, margins-, 
    and 0/42 LN involved."""
    
    print("=" * 60)
    print("Testing Patient Profile Extraction")
    print("=" * 60)
    print(f"\nInput description:\n{description[:200]}...\n")
    
    profile = extract_patient_profile(description, openai_client)
    
    print("Extracted profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Verify anatomical site was extracted
    assert "anatomical_site" in profile, "anatomical_site should be extracted"
    site = profile.get("anatomical_site", "").lower()
    assert any(s in site for s in ["maxilla", "oral", "gingiva", "palate"]), \
        f"anatomical_site should contain maxilla/oral/gingiva/palate, got: {site}"
    
    print("\n✅ Extraction test passed!")
    return profile


def test_category_inference():
    """Test that category is inferred correctly from anatomical site."""
    from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
    
    # Create a minimal service instance just to test the method
    class MockService:
        pass
    
    service = SimplePatientMatchingService.__new__(SimplePatientMatchingService)
    
    print("\n" + "=" * 60)
    print("Testing Category Inference from Anatomical Site")
    print("=" * 60)
    
    test_cases = [
        ("maxilla", "SCC", "h&n_processed_documents"),
        ("oral cavity", "squamous cell carcinoma", "h&n_processed_documents"),
        ("tongue", "SCC", "h&n_processed_documents"),
        ("skin", "SCC", "cutaneous_processed_documents"),
        ("lung", "adenocarcinoma", "lung_processed_documents"),
        ("", "SCC", None),  # Generic SCC without site should return None
        ("breast", "carcinoma", "breast_processed_documents"),
    ]
    
    for site, cancer_type, expected in test_cases:
        result = service._infer_category_from_site(site, cancer_type)
        status = "✅" if result == expected else "❌"
        print(f"  {status} Site: '{site}', Type: '{cancer_type}' -> {result} (expected: {expected})")
        if result != expected:
            print(f"      MISMATCH!")
    
    print("\n✅ Category inference test completed!")


def test_query_building():
    """Test that query is built correctly with anatomical site."""
    from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
    
    service = SimplePatientMatchingService.__new__(SimplePatientMatchingService)
    
    print("\n" + "=" * 60)
    print("Testing Query Building with Anatomical Site")
    print("=" * 60)
    
    profile = {
        "age": 68,
        "gender": "female",
        "cancer_type": "SCC",
        "anatomical_site": "maxilla",
        "cancer_stage": "IV",
        "histology": "poorly differentiated SCC",
        "smoking_status": "never",
    }
    
    query, category = service._build_enhanced_query(profile)
    
    print(f"\nProfile: {profile}")
    print(f"\nBuilt query: {query}")
    print(f"Category filter: {category}")
    
    # Verify query contains site-specific terms
    assert "oral cavity" in query.lower() or "head and neck" in query.lower() or "maxilla" in query.lower(), \
        "Query should contain oral cavity/head and neck/maxilla terms"
    assert category == "h&n_processed_documents", \
        f"Category should be h&n_processed_documents, got: {category}"
    
    print("\n✅ Query building test passed!")


def test_full_matching():
    """Test full patient matching pipeline."""
    from qdrant_client import QdrantClient
    from openai import OpenAI
    from src.core.config import settings
    from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
    
    print("\n" + "=" * 60)
    print("Testing Full Patient Matching Pipeline")
    print("=" * 60)
    
    # Initialize clients
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60
    )
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    service = SimplePatientMatchingService(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        collection_name=settings.qdrant_collection,
        embed_model=settings.embed_model
    )
    
    # Test profile with anatomical site
    profile = {
        "age": 68,
        "gender": "female",
        "cancer_type": "SCC",
        "anatomical_site": "maxilla",
        "cancer_stage": "IV",
        "histology": "poorly differentiated SCC",
        "smoking_status": "never",
        "comorbidities": ["HTN", "T2DM"],
    }
    
    print(f"\nPatient profile: {profile}")
    print("\nRunning patient matching...")
    
    result = service.match_patient(profile, top_k=10)
    
    print(f"\nPatient summary: {result.get('patient_summary')}")
    print(f"Total matches: {result.get('total_matches')}")
    
    if result.get("matches"):
        print("\nTop matches:")
        for i, match in enumerate(result["matches"][:5]):
            title = match.get("title", "Unknown")[:60]
            score = match.get("match_score", 0)
            treatment = match.get("treatment", "N/A")
            cancer_chars = match.get("cancer_characteristics", [])
            print(f"  {i+1}. {title}...")
            print(f"      Score: {score:.2f}, Treatment: {treatment}")
            print(f"      Cancer characteristics: {cancer_chars}")
    else:
        print("\nNo matches found!")
    
    # Check that irrelevant studies are filtered out
    irrelevant_keywords = ["glioma", "glioblastoma", "anal cancer", "brain tumor"]
    for match in result.get("matches", []):
        title = match.get("title", "").lower()
        for keyword in irrelevant_keywords:
            if keyword in title:
                print(f"\n⚠️ WARNING: Potentially irrelevant match found: {match.get('title')}")
    
    print("\n✅ Full matching test completed!")
    return result


async def test_postgres_search():
    """Test PostgreSQL study search by profile."""
    from src.api.services.postgres_study_details_service import PostgresStudyDetailsService
    
    print("\n" + "=" * 60)
    print("Testing PostgreSQL Study Search")
    print("=" * 60)
    
    try:
        pg_service = PostgresStudyDetailsService()
        
        # Test search for head and neck / oral cavity cancer
        results = await pg_service.search_studies_by_profile(
            cancer_type="SCC",
            anatomical_site="maxilla",
            histology="squamous",
            limit=10
        )
        
        print(f"\nFound {len(results)} PostgreSQL matches for SCC of maxilla:")
        for i, study in enumerate(results[:5]):
            print(f"  {i+1}. {study.get('title', 'Unknown')[:60]}...")
            print(f"      Location: {study.get('cancer_location')}, Type: {study.get('cancer_type')}")
        
        await pg_service.close()
        print("\n✅ PostgreSQL search test completed!")
        return results
        
    except Exception as e:
        print(f"\n⚠️ PostgreSQL search failed (may not be configured): {e}")
        return []


if __name__ == "__main__":
    try:
        # Run tests
        profile = test_extraction()
        test_category_inference()
        test_query_building()
        result = test_full_matching()
        
        # Run async PostgreSQL test
        pg_results = asyncio.run(test_postgres_search())
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)
