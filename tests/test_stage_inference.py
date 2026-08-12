#!/usr/bin/env python3
"""
Test script for the Stage Inference Service.

Validates TNM-to-stage mapping against known AJCC 8th edition values
for multiple cancer types.

Usage:
    python tests/test_stage_inference.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.services.stage_inference_service import StageInferenceService


def test_stage_inference():
    """Run all staging inference tests."""
    
    # Load tables explicitly
    tables_path = str(Path(__file__).parent.parent / "data" / "ajcc_staging_tables.json")
    service = StageInferenceService(tables_path=tables_path)
    
    passed = 0
    failed = 0
    total = 0
    
    def check(description, result, expected_stage, expected_confidence=None, expected_ambiguous=None):
        nonlocal passed, failed, total
        total += 1
        
        ok = True
        issues = []
        
        if result.stage_group != expected_stage:
            ok = False
            issues.append(f"stage={result.stage_group}, expected={expected_stage}")
        
        if expected_confidence and result.confidence != expected_confidence:
            ok = False
            issues.append(f"confidence={result.confidence}, expected={expected_confidence}")
        
        if expected_ambiguous is not None and result.is_ambiguous != expected_ambiguous:
            ok = False
            issues.append(f"ambiguous={result.is_ambiguous}, expected={expected_ambiguous}")
        
        if ok:
            passed += 1
            print(f"  ✅ {description} -> {result.stage_group} ({result.source})")
        else:
            failed += 1
            print(f"  ❌ {description} -> {'; '.join(issues)}")
            if result.notes:
                for note in result.notes:
                    print(f"     Note: {note}")
    
    # =================================================================
    # UNIVERSAL RULES
    # =================================================================
    print("\n" + "=" * 60)
    print("UNIVERSAL RULES")
    print("=" * 60)
    
    check(
        "M1 = Stage IV (any cancer)",
        service.infer_stage(cancer_type="breast", tnm_t="T1", tnm_n="N0", tnm_m="M1"),
        "IV", "high"
    )
    
    check(
        "Tis N0 M0 = Stage 0",
        service.infer_stage(cancer_type="breast", tnm_t="Tis", tnm_n="N0", tnm_m="M0"),
        "0", "high"
    )
    
    check(
        "Metastatic text descriptor = Stage IV",
        service.infer_stage(cancer_type="lung", metastatic_status="metastatic"),
        "IV", "high"
    )
    
    # =================================================================
    # BREAST CANCER
    # =================================================================
    print("\n" + "=" * 60)
    print("BREAST CANCER")
    print("=" * 60)
    
    check(
        "Breast T1 N0 M0 = Stage IA",
        service.infer_stage(cancer_type="breast cancer", tnm_t="T1", tnm_n="N0", tnm_m="M0"),
        "IA", "high"
    )
    
    check(
        "Breast T2 N0 M0 = Stage IIA",
        service.infer_stage(cancer_type="breast cancer", tnm_t="T2", tnm_n="N0", tnm_m="M0"),
        "IIA", "high"
    )
    
    check(
        "Breast T2 N1 M0 = Stage IIB",
        service.infer_stage(cancer_type="breast", tnm_t="T2", tnm_n="N1", tnm_m="M0"),
        "IIB", "high"
    )
    
    check(
        "Breast T3 N0 M0 = Stage IIB",
        service.infer_stage(cancer_type="breast", tnm_t="T3", tnm_n="N0", tnm_m="M0"),
        "IIB", "high"
    )
    
    check(
        "Breast T4 N1 M0 = Stage IIIB",
        service.infer_stage(cancer_type="breast", tnm_t="T4", tnm_n="N1", tnm_m="M0"),
        "IIIB", "high"
    )
    
    check(
        "Breast T2 N1 M0 is ambiguous (prognostic stage varies)",
        service.infer_stage(cancer_type="breast", tnm_t="T1", tnm_n="N1", tnm_m="M0"),
        "IIA",  # anatomic stage
        expected_ambiguous=True
    )
    
    # =================================================================
    # LUNG CANCER
    # =================================================================
    print("\n" + "=" * 60)
    print("LUNG CANCER (NSCLC)")
    print("=" * 60)
    
    check(
        "Lung T1a N0 M0 = Stage IA1",
        service.infer_stage(cancer_type="lung cancer", tnm_t="T1a", tnm_n="N0", tnm_m="M0"),
        "IA1", "high"
    )
    
    check(
        "Lung T2a N0 M0 = Stage IB",
        service.infer_stage(cancer_type="NSCLC", tnm_t="T2a", tnm_n="N0", tnm_m="M0"),
        "IB", "high"
    )
    
    check(
        "Lung T3 N0 M0 = Stage IIB",
        service.infer_stage(cancer_type="lung", tnm_t="T3", tnm_n="N0", tnm_m="M0"),
        "IIB", "high"
    )
    
    check(
        "Lung T4 N3 M0 = Stage IIIC",
        service.infer_stage(cancer_type="lung", tnm_t="T4", tnm_n="N3", tnm_m="M0"),
        "IIIC", "high"
    )
    
    check(
        "Lung any T any N M1a = Stage IVA",
        service.infer_stage(cancer_type="lung", tnm_t="T2", tnm_n="N1", tnm_m="M1a"),
        "IVA", "high"
    )
    
    # =================================================================
    # HEAD & NECK — ORAL CAVITY
    # =================================================================
    print("\n" + "=" * 60)
    print("HEAD & NECK — ORAL CAVITY")
    print("=" * 60)
    
    check(
        "Oral cavity T1 N0 M0 = Stage I",
        service.infer_stage(cancer_location="oral cavity", tnm_t="T1", tnm_n="N0", tnm_m="M0"),
        "I", "high"
    )
    
    check(
        "Tongue T2 N0 M0 = Stage II",
        service.infer_stage(cancer_location="tongue", tnm_t="T2", tnm_n="N0", tnm_m="M0"),
        "II", "high"
    )
    
    check(
        "Maxilla SCC T4a N1 M0 = Stage IVA",
        service.infer_stage(
            cancer_type="Squamous Cell Carcinoma", 
            cancer_location="maxilla",
            tnm_t="T4a", tnm_n="N1", tnm_m="M0"
        ),
        "IVA", "high"
    )
    
    check(
        "Oral cavity T3 N1 M0 = Stage III",
        service.infer_stage(cancer_location="oral cavity", tnm_t="T3", tnm_n="N1", tnm_m="M0"),
        "III", "high"
    )
    
    # =================================================================
    # HEAD & NECK — OROPHARYNX (HPV+ vs HPV-)
    # =================================================================
    print("\n" + "=" * 60)
    print("OROPHARYNX HPV+ vs HPV-")
    print("=" * 60)
    
    check(
        "Oropharynx HPV+ T2 N1 M0 = Stage I (favorable staging)",
        service.infer_stage(
            cancer_location="oropharynx", 
            hpv_status="positive",
            tnm_t="T2", tnm_n="N1", tnm_m="M0"
        ),
        "I", "high"
    )
    
    check(
        "Oropharynx HPV- T2 N1 M0 = Stage III (standard H&N staging)",
        service.infer_stage(
            cancer_location="oropharynx",
            hpv_status="negative",
            tnm_t="T2", tnm_n="N1", tnm_m="M0"
        ),
        "III", "high"
    )
    
    # =================================================================
    # THYROID (AGE-DEPENDENT)
    # =================================================================
    print("\n" + "=" * 60)
    print("THYROID (AGE-DEPENDENT)")
    print("=" * 60)
    
    check(
        "Thyroid T3 N1 M0, age 40 = Stage I (under 55)",
        service.infer_stage(
            cancer_type="papillary thyroid",
            tnm_t="T3", tnm_n="N1", tnm_m="M0",
            age=40
        ),
        "I", "high"
    )
    
    check(
        "Thyroid any T any N M1, age 40 = Stage II (under 55, M1 max is II)",
        service.infer_stage(
            cancer_type="thyroid",
            tnm_t="T3", tnm_n="N1", tnm_m="M1",
            age=40
        ),
        "II", "high"
    )
    
    check(
        "Thyroid T3 N1 M0, age 60 = Stage II (over 55)",
        service.infer_stage(
            cancer_type="thyroid",
            tnm_t="T3", tnm_n="N1", tnm_m="M0",
            age=60
        ),
        "II", "high"
    )
    
    check(
        "Thyroid T3 N1 M0, age unknown = ambiguous",
        service.infer_stage(
            cancer_type="thyroid",
            tnm_t="T3", tnm_n="N1", tnm_m="M0",
        ),
        None,
        expected_ambiguous=True
    )
    
    # =================================================================
    # COLON CANCER
    # =================================================================
    print("\n" + "=" * 60)
    print("COLON CANCER")
    print("=" * 60)
    
    check(
        "Colon T1 N0 M0 = Stage I",
        service.infer_stage(cancer_type="colon cancer", tnm_t="T1", tnm_n="N0", tnm_m="M0"),
        "I", "high"
    )
    
    check(
        "Colon T3 N0 M0 = Stage IIA",
        service.infer_stage(cancer_type="colorectal", tnm_t="T3", tnm_n="N0", tnm_m="M0"),
        "IIA", "high"
    )
    
    # =================================================================
    # PROSTATE CANCER
    # =================================================================
    print("\n" + "=" * 60)
    print("PROSTATE CANCER")
    print("=" * 60)
    
    check(
        "Prostate T2 N0 M0 = Stage II (anatomic)",
        service.infer_stage(cancer_type="prostate", tnm_t="T2", tnm_n="N0", tnm_m="M0"),
        "II",
        expected_ambiguous=True  # PSA/Gleason needed for prognostic
    )
    
    check(
        "Prostate T3a N0 M0 = Stage IIIA",
        service.infer_stage(cancer_type="prostate", tnm_t="T3a", tnm_n="N0", tnm_m="M0"),
        "IIIA", "high"
    )
    
    # =================================================================
    # PREFIX HANDLING (cT, pT, etc.)
    # =================================================================
    print("\n" + "=" * 60)
    print("PREFIX HANDLING")
    print("=" * 60)
    
    check(
        "Pathologic prefix: pT4 N0 M0 breast = Stage IIIB",
        service.infer_stage(cancer_type="breast", tnm_t="pT4", tnm_n="pN0", tnm_m="cM0"),
        "IIIB", "high"
    )
    
    check(
        "Clinical prefix: cT2a N0 M0 lung = Stage IB",
        service.infer_stage(cancer_type="lung", tnm_t="cT2a", tnm_n="cN0", tnm_m="cM0"),
        "IB", "high"
    )
    
    # =================================================================
    # FALLBACK (UNKNOWN CANCER TYPE)
    # =================================================================
    print("\n" + "=" * 60)
    print("FALLBACK (UNKNOWN CANCER TYPE)")
    print("=" * 60)
    
    check(
        "Unknown cancer T1 N0 M0 = fallback Stage I",
        service.infer_stage(cancer_type="rare_sarcoma", tnm_t="T1", tnm_n="N0", tnm_m="M0"),
        "I", "medium"
    )
    
    check(
        "Unknown cancer any T any N M1 = Stage IV",
        service.infer_stage(cancer_type="rare_sarcoma", tnm_t="T2", tnm_n="N1", tnm_m="M1"),
        "IV", "high"
    )
    
    # =================================================================
    # TEXT DESCRIPTOR INFERENCE
    # =================================================================
    print("\n" + "=" * 60)
    print("TEXT DESCRIPTOR INFERENCE")
    print("=" * 60)
    
    r = service.infer_stage_from_text("Patient with metastatic disease")
    check("'metastatic disease' -> Stage IV", r, "IV", "high")
    
    r = service.infer_stage_from_text("Carcinoma in situ found on biopsy")
    check("'carcinoma in situ' -> Stage 0", r, "0", "high")
    
    r = service.infer_stage_from_text("locally advanced tumor invading adjacent structures")
    check("'locally advanced' -> range III-IVA", r, None)
    # This returns a range, not a single stage
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  Some tests failed. This may be due to:")
        print("   - Sub-stage normalization edge cases")
        print("   - Wildcard matching order")
        print("   Review the failures and adjust tables/logic as needed.")
    else:
        print("\n✅ All tests passed!")
    
    return failed == 0


if __name__ == "__main__":
    success = test_stage_inference()
    sys.exit(0 if success else 1)
