"""
Live Patient Matching Tests

Tests patient matching with specific patient profiles against the actual
Qdrant database to evaluate retrieval accuracy.

Run with: python tests/test_patient_matching_live.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from openai import OpenAI
from src.core.config import settings
from src.api.services.patient_matching_service_simple import SimplePatientMatchingService


def create_service():
    """Create the patient matching service with real connections."""
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60  # 60 second timeout for cloud connections
    )
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    return SimplePatientMatchingService(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        collection_name=settings.qdrant_collection,
        embed_model=settings.embed_model
    )


def print_match_results(result, profile_name):
    """Pretty print match results."""
    print(f"\n{'='*80}")
    print(f"PATIENT PROFILE: {profile_name}")
    print(f"{'='*80}")
    print(f"Patient Summary: {result.get('patient_summary', 'N/A')}")
    print(f"Total Matches: {result.get('total_matches', 0)}")
    
    if result.get('warnings'):
        print(f"Warnings: {result['warnings']}")
    
    if result.get('error'):
        print(f"ERROR: {result['error']}")
        return
    
    matches = result.get('matches', [])
    if not matches:
        print("No matches found.")
        return
    
    print(f"\n{'─'*80}")
    for i, match in enumerate(matches[:5], 1):  # Show top 5
        print(f"\n[{i}] {match.get('title', 'Unknown Title')}")
        print(f"    Author: {match.get('author', 'Unknown')} | Year: {match.get('year', 'N/A')}")
        print(f"    Match Score: {match.get('match_score', 0):.1%}")
        print(f"    Treatment: {match.get('treatment', 'N/A')}")
        
        demographics = match.get('demographics', [])
        cancer_chars = match.get('cancer_characteristics', [])
        key_matches = match.get('key_matches', [])
        
        if demographics:
            print(f"    Demographics: {', '.join(demographics)}")
        if cancer_chars:
            print(f"    Cancer Characteristics: {', '.join(cancer_chars)}")
        if key_matches:
            print(f"    Key Matches: {', '.join(key_matches)}")
        
        if match.get('key_info'):
            print(f"    Key Finding: {match['key_info'][:150]}...")
        
        print(f"    DOI: {match.get('doi', 'N/A')}")


def test_breast_cancer_her2_positive():
    """Test: 55-year-old female with HER2+ breast cancer, stage II."""
    service = create_service()
    
    profile = {
        "cancer_type": "Breast",
        "age": 55,
        "gender": "female",
        "cancer_stage": "II",
        "molecular_markers": ["HER2+"],
        "performance_status": "0"
    }
    
    result = service.match_patient(profile, top_k=10)
    print_match_results(result, "55yo Female, HER2+ Breast Cancer Stage II")
    
    return result


def test_lung_cancer_elderly():
    """Test: 72-year-old male with NSCLC, stage III."""
    service = create_service()
    
    profile = {
        "cancer_type": "Lung",
        "age": 72,
        "gender": "male",
        "cancer_stage": "III",
        "histology": "adenocarcinoma",
        "molecular_markers": ["EGFR+"],
        "performance_status": "1",
        "smoking_status": "former"
    }
    
    result = service.match_patient(profile, top_k=10)
    print_match_results(result, "72yo Male, EGFR+ Lung Adenocarcinoma Stage III")
    
    return result


def test_colorectal_cancer():
    """Test: 60-year-old male with colorectal cancer, stage III."""
    service = create_service()
    
    profile = {
        "cancer_type": "Colorectal",
        "age": 60,
        "gender": "male",
        "cancer_stage": "III",
        "performance_status": "0"
    }
    
    result = service.match_patient(profile, top_k=10)
    print_match_results(result, "60yo Male, Colorectal Cancer Stage III")
    
    return result


def test_triple_negative_breast_cancer():
    """Test: 45-year-old female with triple-negative breast cancer."""
    service = create_service()
    
    profile = {
        "cancer_type": "Breast",
        "age": 45,
        "gender": "female",
        "cancer_stage": "II",
        "molecular_markers": ["ER-", "PR-", "HER2-"],
        "histology": "invasive ductal carcinoma"
    }
    
    result = service.match_patient(profile, top_k=10)
    print_match_results(result, "45yo Female, Triple-Negative Breast Cancer Stage II")
    
    return result


def test_prostate_cancer():
    """Test: 68-year-old male with prostate cancer."""
    service = create_service()
    
    profile = {
        "cancer_type": "Prostate",
        "age": 68,
        "gender": "male",
        "cancer_stage": "II",
        "performance_status": "0"
    }
    
    result = service.match_patient(profile, top_k=10)
    print_match_results(result, "68yo Male, Prostate Cancer Stage II")
    
    return result


def test_young_breast_cancer():
    """Test: 35-year-old female with breast cancer (young patient)."""
    service = create_service()
    
    profile = {
        "cancer_type": "Breast",
        "age": 35,
        "gender": "female",
        "cancer_stage": "I",
        "molecular_markers": ["ER+", "PR+", "HER2-"]
    }
    
    result = service.match_patient(profile, top_k=10)
    print_match_results(result, "35yo Female, ER+/PR+/HER2- Breast Cancer Stage I (Young)")
    
    return result


def evaluate_accuracy(results):
    """Evaluate overall accuracy of matching results."""
    print(f"\n{'='*80}")
    print("ACCURACY EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    total_profiles = len(results)
    profiles_with_matches = sum(1 for r in results if r.get('total_matches', 0) > 0)
    
    print(f"Profiles tested: {total_profiles}")
    print(f"Profiles with matches: {profiles_with_matches}")
    print(f"Match rate: {profiles_with_matches/total_profiles:.1%}")
    
    # Calculate average match scores
    all_scores = []
    for result in results:
        for match in result.get('matches', []):
            all_scores.append(match.get('match_score', 0))
    
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        max_score = max(all_scores)
        min_score = min(all_scores)
        print(f"\nMatch Score Statistics:")
        print(f"  Average: {avg_score:.1%}")
        print(f"  Max: {max_score:.1%}")
        print(f"  Min: {min_score:.1%}")
        print(f"  Total matches across all profiles: {len(all_scores)}")
    
    # Count characteristic matches
    total_demographics = 0
    total_cancer_chars = 0
    total_key_matches = 0
    
    for result in results:
        for match in result.get('matches', []):
            total_demographics += len(match.get('demographics', []))
            total_cancer_chars += len(match.get('cancer_characteristics', []))
            total_key_matches += len(match.get('key_matches', []))
    
    print(f"\nCharacteristic Matching:")
    print(f"  Demographics matched: {total_demographics}")
    print(f"  Cancer characteristics matched: {total_cancer_chars}")
    print(f"  Key markers matched: {total_key_matches}")


def main():
    """Run all patient matching tests."""
    print("\n" + "="*80)
    print("PATIENT MATCHING LIVE TESTS")
    print("Testing retrieval accuracy with specific patient profiles")
    print("="*80)
    
    results = []
    
    try:
        # Test 1: HER2+ Breast Cancer
        print("\n[Test 1/6] HER2+ Breast Cancer...")
        results.append(test_breast_cancer_her2_positive())
        
        # Test 2: Lung Cancer (Elderly)
        print("\n[Test 2/6] Lung Cancer (Elderly)...")
        results.append(test_lung_cancer_elderly())
        
        # Test 3: Colorectal Cancer
        print("\n[Test 3/6] Colorectal Cancer...")
        results.append(test_colorectal_cancer())
        
        # Test 4: Triple-Negative Breast Cancer
        print("\n[Test 4/6] Triple-Negative Breast Cancer...")
        results.append(test_triple_negative_breast_cancer())
        
        # Test 5: Prostate Cancer
        print("\n[Test 5/6] Prostate Cancer...")
        results.append(test_prostate_cancer())
        
        # Test 6: Young Breast Cancer Patient
        print("\n[Test 6/6] Young Breast Cancer Patient...")
        results.append(test_young_breast_cancer())
        
        # Evaluate overall accuracy
        evaluate_accuracy(results)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
