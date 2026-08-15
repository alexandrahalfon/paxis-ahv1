#!/usr/bin/env python3
"""
Test script for the Query Classifier Service

Tests the extraction of structured clinical data from free-text queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.services.query_classifier_service import QueryClassifierService, StructuredQuery


def test_complex_query():
    """Test with the complex clinical query from the user"""
    
    query = """68 year old female, Mandarin-speaking, non-smoke, with a PMH of HTN, T2DM, 
    and SCC of R maxilla s/p right maxillectomy, bilateral selective neck dissection, 
    and placement of obturator on 2/16/24, revealing poorly differentiated SCC of right 
    gingiva, hard palate, buccal mucosa, and maxilla, pT4N0, DOI 15 mm, LVI-, PNI-, 
    margins-, and 0/42 LN involved. Recent CT imaging revealed a 1.4 cm x 0.9 cm right 
    mucosal pharyngeal enhancement and a 1.1 cm soft tissue density lesion at the right 
    sternocleidomastoid muscle, along with cervical lymphadenopathy at right level 1 and 
    level 4 lymph nodes, s/p b/l radical neck dissection (11/14/25) with nodal recurrence 
    at right neck, levels 1,2 and level4, no carcinoma at left neck."""
    
    print("=" * 80)
    print("QUERY CLASSIFIER TEST")
    print("=" * 80)
    print("\nInput Query:")
    print("-" * 40)
    print(query)
    print("-" * 40)
    
    service = QueryClassifierService()
    result = service.classify_query(query)
    
    print("\n\nEXTRACTED STRUCTURED DATA:")
    print("=" * 80)
    
    print("\n📋 DEMOGRAPHICS:")
    if result.age:
        print(f"  • Age: {result.age} years")
    if result.gender:
        print(f"  • Gender: {result.gender}")
    if result.race_ethnicity:
        print(f"  • Race/Ethnicity: {result.race_ethnicity}")
    if result.performance_status:
        print(f"  • Performance Status: ECOG {result.performance_status}")
    
    print("\n🔬 DIAGNOSIS:")
    if result.cancer_type:
        print(f"  • Cancer Type: {result.cancer_type}")
    if result.cancer_location:
        print(f"  • Cancer Location: {result.cancer_location}")
    if result.histopathologic_type:
        print(f"  • Histopathology: {result.histopathologic_type}")
    if result.tumor_grade:
        print(f"  • Tumor Grade: {result.tumor_grade}")
    if result.molecular_subtype:
        print(f"  • Molecular Subtype: {result.molecular_subtype}")
    
    print("\n📊 STAGING:")
    if result.tnm_t:
        print(f"  • T Stage: {result.tnm_t}")
    if result.tnm_n:
        print(f"  • N Stage: {result.tnm_n}")
    if result.tnm_m:
        print(f"  • M Stage: {result.tnm_m}")
    if result.overall_stage:
        print(f"  • Overall Stage: {result.overall_stage}")
    if result.risk_stratification:
        print(f"  • Risk: {result.risk_stratification}")
    if result.metastatic_status:
        print(f"  • Metastatic Status: {result.metastatic_status}")
    
    print("\n🔍 PATHOLOGY:")
    if result.depth_of_invasion:
        print(f"  • Depth of Invasion: {result.depth_of_invasion}")
    if result.lymphovascular_invasion:
        print(f"  • LVI: {result.lymphovascular_invasion}")
    if result.perineural_invasion:
        print(f"  • PNI: {result.perineural_invasion}")
    if result.margin_status:
        print(f"  • Margins: {result.margin_status}")
    if result.lymph_nodes_examined is not None:
        print(f"  • LN Examined: {result.lymph_nodes_examined}")
    if result.lymph_nodes_positive is not None:
        print(f"  • LN Positive: {result.lymph_nodes_positive}")
    
    print("\n💊 TREATMENT HISTORY:")
    if result.prior_surgery:
        print(f"  • Prior Surgery: {result.prior_surgery}")
    if result.prior_radiation is not None:
        print(f"  • Prior Radiation: {'Yes' if result.prior_radiation else 'No'}")
    if result.prior_chemotherapy is not None:
        print(f"  • Prior Chemotherapy: {'Yes' if result.prior_chemotherapy else 'No'}")
    if result.recurrence_status:
        print(f"  • Recurrence: {result.recurrence_status}")
    
    print("\n⚠️ RISK FACTORS:")
    if result.smoking_status:
        print(f"  • Smoking: {result.smoking_status}")
    if result.comorbidities:
        print(f"  • Comorbidities: {', '.join(result.comorbidities)}")
    
    print("\n" + "=" * 80)
    print("SEARCH SUMMARY:")
    print(result.get_search_summary())
    print("=" * 80)
    
    print("\n\nALL EXTRACTED FIELDS (for PostgreSQL):")
    print("-" * 40)
    for key, value in result.to_dict().items():
        print(f"  {key}: {value}")
    
    # Build PostgreSQL filters
    print("\n\nPOSTGRESQL FILTER CONDITIONS:")
    print("-" * 40)
    filters = service.build_postgres_filters(result)
    for i, condition in enumerate(filters["conditions"]):
        print(f"  {i+1}. {condition.strip()}")
        print(f"     Param: {filters['params'][i]}")
    
    return result


def test_simple_queries():
    """Test with simpler queries"""
    
    queries = [
        "55 year old male with stage III NSCLC, EGFR mutation positive",
        "Breast cancer patient, HER2+, ER+/PR+, stage IIA",
        "Recurrent prostate cancer after radical prostatectomy, PSA rising",
        "Pediatric medulloblastoma, high risk, post-surgical resection",
    ]
    
    service = QueryClassifierService()
    
    print("\n\n" + "=" * 80)
    print("ADDITIONAL TEST QUERIES")
    print("=" * 80)
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        result = service.classify_query(query)
        print(f"   Summary: {result.get_search_summary()}")
        fields = result.to_dict()
        if fields:
            print(f"   Fields: {fields}")
        print()


if __name__ == "__main__":
    print("\n🧪 Testing Query Classifier Service\n")
    
    # Test complex query
    test_complex_query()
    
    # Test simple queries
    test_simple_queries()
    
    print("\n✅ Tests completed!")
