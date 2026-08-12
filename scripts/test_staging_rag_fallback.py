#!/usr/bin/env python3
"""
Test script for the staging RAG fallback functionality.

Tests the detection of staging keywords and the RAG fallback mechanism.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.services.staging_rag_fallback import (
    detect_staging_keywords,
    StagingDetectionResult,
)


def test_staging_detection():
    """Test the staging keyword detection logic."""
    
    test_cases = [
        # (query, expected_has_explicit, expected_has_clinical, expected_needs_fallback)
        
        # Should trigger fallback (clinical details, no explicit staging)
        (
            "A patient presents with a right lateral oral tongue SCC, 1.5 cm in size, 8mm depth of invasion, and ipsilateral adenopathy in level Ib (2 cm) and IIa (2.5cm). There is overt extranodal extension of the 1b node. Metastatic work-up is negative. What is the clinical stage?",
            False,  # No explicit TNM/stage
            True,   # Has clinical indicators (cm, depth, adenopathy, ENE)
            True,   # Should trigger fallback
        ),
        
        # Should NOT trigger fallback (has explicit TNM)
        (
            "What is the treatment for T2N1M0 breast cancer?",
            True,   # Has explicit TNM
            False,  # No additional clinical indicators needed
            False,  # Should NOT trigger fallback
        ),
        
        # Should NOT trigger fallback (has stage group)
        (
            "What is the prognosis for Stage IIB lung cancer?",
            True,   # Has explicit stage
            False,  # No additional clinical indicators
            False,  # Should NOT trigger fallback
        ),
        
        # Should trigger fallback (clinical details only)
        (
            "Patient with 4cm breast tumor and 3 positive axillary nodes, what stage?",
            False,  # No explicit staging
            True,   # Has clinical indicators (cm, nodes)
            True,   # Should trigger fallback
        ),
        
        # Should NOT trigger fallback (general question, no clinical details)
        (
            "What is the standard of care for breast cancer?",
            False,  # No explicit staging
            False,  # No clinical indicators
            False,  # Should NOT trigger fallback
        ),
        
        # Should trigger fallback (DOI mentioned)
        (
            "Oral cavity SCC with 12mm depth of invasion, what T stage?",
            False,  # No explicit staging
            True,   # Has DOI
            True,   # Should trigger fallback
        ),
        
        # Should NOT trigger fallback (has pT staging)
        (
            "pT3N0M0 colon cancer, what adjuvant treatment?",
            True,   # Has explicit pTNM
            False,  # No additional indicators needed
            False,  # Should NOT trigger fallback
        ),
        
        # Should trigger fallback (ENE mentioned without N stage)
        (
            "Head and neck cancer with extranodal extension, what is the N stage?",
            False,  # No explicit staging
            True,   # Has ENE indicator
            True,   # Should trigger fallback
        ),
    ]
    
    print("=" * 80)
    print("STAGING KEYWORD DETECTION TESTS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, (query, exp_explicit, exp_clinical, exp_fallback) in enumerate(test_cases, 1):
        result = detect_staging_keywords(query)
        
        test_passed = (
            result.has_explicit_staging == exp_explicit and
            result.has_clinical_indicators == exp_clinical and
            result.needs_rag_fallback == exp_fallback
        )
        
        status = "PASS" if test_passed else "FAIL"
        if test_passed:
            passed += 1
        else:
            failed += 1
        
        print(f"\nTest {i}: [{status}]")
        print(f"  Query: {query[:80]}...")
        print(f"  Expected: explicit={exp_explicit}, clinical={exp_clinical}, fallback={exp_fallback}")
        print(f"  Got:      explicit={result.has_explicit_staging}, clinical={result.has_clinical_indicators}, fallback={result.needs_rag_fallback}")
        
        if result.detected_staging_keywords:
            print(f"  Staging keywords: {result.detected_staging_keywords[:3]}")
        if result.detected_clinical_indicators:
            print(f"  Clinical indicators: {result.detected_clinical_indicators[:5]}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = test_staging_detection()
    sys.exit(0 if success else 1)
