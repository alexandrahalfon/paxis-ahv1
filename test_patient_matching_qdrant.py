#!/usr/bin/env python
"""
Test patient matching using Qdrant database directly.

This script tests the SimplePatientMatchingService with real Qdrant data.
"""

import sys
import os

# Ensure output is visible
print("[Test] Starting patient matching test...", flush=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[Test] Environment loaded", flush=True)
except Exception as e:
    print(f"[Test] Error loading dotenv: {e}", flush=True)
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from openai import OpenAI
    print("[Test] Clients imported", flush=True)
except Exception as e:
    print(f"[Test] Error importing clients: {e}", flush=True)
    sys.exit(1)

try:
    from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
    from src.core.config import settings
    print("[Test] Services imported", flush=True)
except Exception as e:
    print(f"[Test] Error importing services: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_patient_matching():
    """Test patient matching with various profiles against Qdrant."""
    
    print("=" * 80, flush=True)
    print("PATIENT MATCHING TEST - Using Qdrant Database", flush=True)
    print("=" * 80, flush=True)
    
    # Initialize clients
    print(f"\n[Setup] Connecting to Qdrant at {settings.qdrant_url}...", flush=True)
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )
    
    print(f"[Setup] Collection: {settings.qdrant_collection}", flush=True)
    
    # Verify collection exists
    try:
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)
        print(f"[Setup] Collection has {collection_info.points_count} points", flush=True)
    except Exception as e:
        print(f"[Error] Could not access collection: {e}", flush=True)
        return False
    
    print("[Setup] Connecting to OpenAI...", flush=True)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    # Create service
    print("[Setup] Initializing PatientMatchingService...", flush=True)
    service = SimplePatientMatchingService(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        collection_name=settings.qdrant_collection
    )
    
    # Test profiles
    test_profiles = [
        {
            "name": "Prostate Cancer - Gleason 3+4, Low Volume",
            "profile": {
                "cancer_type": "Prostate Cancer",
                "anatomical_site": "prostate",
                "cancer_stage": "localized",
                "age": 72,
                "gender": "male",
                "molecular_markers": ["Gleason 3+4", "Gleason 7", "PSA 5.6"],
                "histology": "adenocarcinoma",
                "disease_characteristics": "4/12 cores positive, no PNI, normal DRE",
                "risk_group": "intermediate risk"
            }
        }
    ]
    
    results_summary = []
    
    for test_case in test_profiles:
        print("\n" + "-" * 60, flush=True)
        print(f"Testing: {test_case['name']}", flush=True)
        print("-" * 60, flush=True)
        
        profile = test_case["profile"]
        print(f"Profile: {profile}", flush=True)
        
        try:
            result = service.match_patient(profile, top_k=10)
            
            matches = result.get("matches", [])
            total = result.get("total_matches", 0)
            warnings = result.get("warnings", [])
            
            print(f"\nResults:", flush=True)
            print(f"  Total matches: {total}", flush=True)
            print(f"  Patient summary: {result.get('patient_summary', 'N/A')}", flush=True)
            
            if warnings:
                print(f"  Warnings: {warnings}", flush=True)
            
            if matches:
                print(f"\n  Top {len(matches)} matches:", flush=True)
                for i, match in enumerate(matches, 1):
                    title = match.get("title", "Unknown")
                    score = match.get("match_score", 0)
                    year = match.get("year", "N/A")
                    doc_id = match.get("doc_id", "Unknown")
                    doi = match.get("doi", "N/A")
                    pmid = match.get("pmid", "N/A")
                    citation = match.get("citation", "N/A")
                    validation = match.get("validation_status", "N/A")
                    
                    print(f"\n    {i}. [{score:.0%}] {title} ({year})", flush=True)
                    print(f"       Doc ID: {doc_id}", flush=True)
                    print(f"       DOI: {doi}", flush=True)
                    print(f"       PMID: {pmid}", flush=True)
                    print(f"       Citation: {citation}", flush=True)
                    print(f"       Validation Status: {validation}", flush=True)
                    
                    # Show what was matched on
                    print(f"\n       MATCHED ON:", flush=True)
                    
                    demographics = match.get("demographics", "")
                    if demographics:
                        print(f"         Demographics: {demographics}", flush=True)
                    
                    cancer_chars = match.get("cancer_characteristics", "")
                    if cancer_chars:
                        print(f"         Cancer Characteristics: {cancer_chars}", flush=True)
                    
                    key_matches = match.get("key_matches", [])
                    if key_matches:
                        print(f"         Key Matches: {key_matches}", flush=True)
                    
                    treatment = match.get("treatment", "")
                    if treatment:
                        print(f"         Treatment: {treatment}", flush=True)
                    
                    # Show key info excerpt
                    key_info = match.get("key_info", "")
                    if key_info:
                        print(f"\n       Key Info: {key_info[:500]}...", flush=True)
                    
                    # Show relevant text excerpt
                    relevant_text = match.get("relevant_text", "")
                    if relevant_text:
                        print(f"\n       Relevant Text: {relevant_text[:500]}...", flush=True)
            else:
                print("  No matches found", flush=True)
            
            results_summary.append({
                "name": test_case["name"],
                "success": total > 0,
                "matches": total
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}", flush=True)
            results_summary.append({
                "name": test_case["name"],
                "success": False,
                "matches": 0,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("TEST SUMMARY", flush=True)
    print("=" * 80, flush=True)
    
    passed = sum(1 for r in results_summary if r["success"])
    total_tests = len(results_summary)
    
    for r in results_summary:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['matches']} matches", flush=True)
    
    print(f"\nTotal: {passed}/{total_tests} tests passed", flush=True)
    
    return passed == total_tests


if __name__ == "__main__":
    print("[Main] Script starting...", flush=True)
    try:
        success = test_patient_matching()
        print(f"[Main] Test completed with success={success}", flush=True)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[Main] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
