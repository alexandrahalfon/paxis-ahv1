"""
Staging Inference Validator

Provides post-extraction validation and correction of TNM staging using
AJCC 8th Edition rules. This is a defense-in-depth layer that catches
staging errors from LLM extraction.

Key Features:
- Cancer-type-specific T staging rules (oral cavity DOI, etc.)
- N staging validation for ENE (extranodal extension)
- Stage group inference from TNM
- Integration with staging_search_expander for comprehensive tables

Usage:
    from staging_inference_validator import validate_and_correct_staging
    
    # After LLM extraction
    profile = extract_patient_profile(text, client)
    profile = validate_and_correct_staging(profile, text)

Author: AI Assistant
Version: 1.0.0
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StagingValidationResult:
    """Result of staging validation."""
    original_t: Optional[str] = None
    corrected_t: Optional[str] = None
    t_correction_reason: Optional[str] = None
    
    original_n: Optional[str] = None
    corrected_n: Optional[str] = None
    n_correction_reason: Optional[str] = None
    
    original_m: Optional[str] = None
    corrected_m: Optional[str] = None
    
    inferred_stage_group: Optional[str] = None
    stage_group_confidence: float = 0.0
    
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    @property
    def had_corrections(self) -> bool:
        return (self.original_t != self.corrected_t or 
                self.original_n != self.corrected_n or
                self.original_m != self.corrected_m)


# =============================================================================
# CANCER TYPE DETECTION
# =============================================================================

# Keywords that indicate oral cavity primary
ORAL_CAVITY_KEYWORDS = [
    "oral cavity", "oral tongue", "tongue", "floor of mouth", "fom",
    "buccal mucosa", "buccal", "gingiva", "gingival", "alveolar ridge",
    "hard palate", "palate", "maxilla", "maxillary", "mandible", "mandibular",
    "retromolar trigone", "retromolar", "lip", "oral"
]

# Keywords that indicate oropharynx
OROPHARYNX_KEYWORDS = [
    "oropharynx", "oropharyngeal", "tonsil", "tonsillar", "base of tongue",
    "bot", "soft palate", "posterior pharyngeal wall", "vallecula"
]

# Keywords that indicate larynx
LARYNX_KEYWORDS = [
    "larynx", "laryngeal", "glottis", "glottic", "supraglottis", "supraglottic",
    "subglottis", "subglottic", "vocal cord", "vocal fold"
]

# Keywords that indicate head & neck in general
HEAD_NECK_KEYWORDS = [
    "head and neck", "head & neck", "h&n", "hnscc", "scc of", "squamous cell"
]


def detect_cancer_site(text: str, anatomical_site: Optional[str] = None) -> Optional[str]:
    """
    Detect cancer site from text and/or anatomical site field.
    
    Returns:
        One of: "oral_cavity", "oropharynx", "larynx", "head_neck", or None
    """
    # Combine sources
    combined = f"{text} {anatomical_site or ''}".lower()
    
    # Check specific sites first (more specific wins)
    for kw in ORAL_CAVITY_KEYWORDS:
        if kw in combined:
            return "oral_cavity"
    
    for kw in OROPHARYNX_KEYWORDS:
        if kw in combined:
            return "oropharynx"
    
    for kw in LARYNX_KEYWORDS:
        if kw in combined:
            return "larynx"
    
    for kw in HEAD_NECK_KEYWORDS:
        if kw in combined:
            return "head_neck"
    
    return None


# =============================================================================
# MEASUREMENT EXTRACTION
# =============================================================================

def extract_tumor_size_cm(text: str, profile: Optional[Dict] = None) -> Optional[float]:
    """
    Extract tumor size in cm from text or profile.
    """
    # Try profile first
    if profile and profile.get("tumor_size"):
        size_str = profile["tumor_size"]
        match = re.search(r'(\d+(?:\.\d+)?)\s*(cm|mm)', size_str, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value if unit == "cm" else value / 10
    
    # Extract from text
    patterns = [
        r'(\d+(?:\.\d+)?)\s*cm\s*(?:in size|tumor|mass|lesion|primary)?',
        r'tumor[^\d]*(\d+(?:\.\d+)?)\s*cm',
        r'primary[^\d]*(\d+(?:\.\d+)?)\s*cm',
        r'(\d+(?:\.\d+)?)\s*cm\s*(?:x|\×)',  # Size with dimensions
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Try mm
    mm_patterns = [
        r'(\d+(?:\.\d+)?)\s*mm\s*(?:in size|tumor|mass|lesion|primary)?',
    ]
    for pattern in mm_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Only convert if it makes sense (>10mm is likely intentional mm measurement)
            if value > 10:
                return value / 10
    
    return None


def extract_doi_mm(text: str, profile: Optional[Dict] = None) -> Optional[float]:
    """
    Extract depth of invasion in mm from text or profile.
    """
    # Try profile first
    if profile and profile.get("doi"):
        doi_str = profile["doi"]
        match = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm)', doi_str, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value if unit == "mm" else value * 10
    
    # Extract from text
    patterns = [
        r'(?:DOI|depth of invasion)[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm)',
        r'(\d+(?:\.\d+)?)\s*(mm|cm)\s*(?:DOI|depth of invasion)',
        r'(\d+(?:\.\d+)?)\s*(mm|cm)\s*depth',
        r'depth[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm)',
        r'invasion[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value if unit == "mm" else value * 10
    
    return None


def detect_ene(text: str, profile: Optional[Dict] = None) -> Optional[bool]:
    """
    Detect extranodal extension (ENE) status.
    
    Returns:
        True = ENE present, False = ENE absent, None = not mentioned
    """
    text_lower = text.lower()
    
    # Check profile
    if profile:
        ene_status = (profile.get("ene") or "").lower()
        if any(pos in ene_status for pos in ['positive', 'present', 'overt', 'gross', '+']):
            return True
        if any(neg in ene_status for neg in ['negative', 'absent', '-', 'no ']):
            return False
    
    # Negative patterns (check first)
    negative_patterns = [
        r'ene\s*[-−]',
        r'ene\s*negative',
        r'no\s*(?:extranodal|ene|ece)',
        r'without\s*(?:extranodal|ene|ece)',
        r'ece\s*[-−]',
        r'ece\s*negative',
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, text_lower):
            return False
    
    # Positive patterns
    positive_patterns = [
        r'ene\s*[+\+]',
        r'ene\s*positive',
        r'overt\s*(?:extranodal|ene)',
        r'gross\s*(?:extranodal|ene)',
        r'(?:with|has)\s*(?:extranodal|ene)',
        r'extranodal\s*extension\s*(?:present|positive|identified|seen)',
        r'(?:clinically)?\s*overt\s*ene',
        r'ece\s*[+\+]',
        r'ece\s*positive',
        r'extracapsular\s*extension',
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return None


# =============================================================================
# T STAGE CALCULATION
# =============================================================================

def calculate_oral_cavity_t_stage(size_cm: float, doi_mm: float) -> Tuple[str, str]:
    """
    Calculate T stage for oral cavity using AJCC 8th edition.
    
    Uses BOTH size AND DOI - whichever gives higher T stage.
    
    Returns:
        Tuple of (T_stage, explanation)
    """
    # T stage based on size alone
    if size_cm <= 2:
        t_by_size = 1
        size_reason = f"≤2cm"
    elif size_cm <= 4:
        t_by_size = 2
        size_reason = f">2cm but ≤4cm"
    else:
        t_by_size = 3
        size_reason = f">4cm"
    
    # T stage based on DOI alone
    if doi_mm <= 5:
        t_by_doi = 1
        doi_reason = f"DOI ≤5mm"
    elif doi_mm <= 10:
        t_by_doi = 2
        doi_reason = f"DOI >5mm but ≤10mm"
    else:
        t_by_doi = 3
        doi_reason = f"DOI >10mm"
    
    # Use higher of the two
    if t_by_doi > t_by_size:
        t_stage = t_by_doi
        reason = f"Size ({size_cm}cm) suggests T{t_by_size}, but DOI ({doi_mm}mm) upgrades to T{t_by_doi}"
    elif t_by_size > t_by_doi:
        t_stage = t_by_size
        reason = f"DOI ({doi_mm}mm) suggests T{t_by_doi}, but size ({size_cm}cm) is T{t_by_size}"
    else:
        t_stage = t_by_size
        reason = f"Both size ({size_cm}cm) and DOI ({doi_mm}mm) indicate T{t_stage}"
    
    return f"T{t_stage}", reason


def calculate_oropharynx_t_stage(size_cm: float, hpv_positive: bool = False) -> Tuple[str, str]:
    """
    Calculate T stage for oropharynx (AJCC 8th edition).
    
    Note: HPV+ oropharynx uses different staging system.
    """
    if size_cm <= 2:
        return "T1", f"≤2cm"
    elif size_cm <= 4:
        return "T2", f">2cm but ≤4cm"
    else:
        return "T3", f">4cm"


# =============================================================================
# N STAGE VALIDATION
# =============================================================================

def validate_n_stage_for_ene(
    current_n: Optional[str], 
    has_ene: bool,
    prefix: str = ""
) -> Tuple[str, Optional[str]]:
    """
    Validate N stage for ENE.
    
    Per AJCC 8th Edition Head & Neck:
    - Any node with clinically overt ENE = N3b, regardless of size or number
    
    Returns:
        Tuple of (corrected_n, correction_reason)
    """
    if not has_ene:
        return current_n or "", None
    
    # If ENE is present, N stage should be N3b
    current_n_upper = (current_n or "").upper()
    
    if "N3B" in current_n_upper:
        return current_n, None  # Already correct
    
    # Correct to N3b
    corrected = f"{prefix}N3b"
    reason = "ENE present upgrades any nodal disease to N3b (AJCC 8th)"
    
    return corrected, reason


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_and_correct_staging(
    profile: Dict[str, Any], 
    original_text: str,
    use_ajcc_tables: bool = True
) -> Dict[str, Any]:
    """
    Validate and correct TNM staging in a profile using AJCC rules.
    
    This is the main entry point for staging validation.
    
    Args:
        profile: Patient profile dict with tnm_t, tnm_n, tnm_m, etc.
        original_text: Original clinical text for additional extraction
        use_ajcc_tables: Whether to use AJCCStagingTables for stage group inference
        
    Returns:
        Updated profile dict with corrected staging
    """
    result = StagingValidationResult()
    
    # Store originals
    result.original_t = profile.get("tnm_t")
    result.original_n = profile.get("tnm_n")
    result.original_m = profile.get("tnm_m")
    
    # Detect cancer site
    site = detect_cancer_site(
        original_text, 
        profile.get("anatomical_site")
    )
    
    # Determine staging prefix (c/p)
    prefix = ""
    current_t = profile.get("tnm_t", "")
    if current_t and current_t[0].lower() in ['c', 'p', 'y']:
        prefix = current_t[0].lower()
        if len(current_t) > 1 and current_t[1].lower() == 'p':
            prefix = "yp"
    elif 'pathologic' in original_text.lower() or re.search(r'\bp[TNM]', original_text):
        prefix = "p"
    elif 'clinical' in original_text.lower() or re.search(r'\bc[TNM]', original_text):
        prefix = "c"
    
    # =========================================================================
    # T Stage Validation
    # =========================================================================
    
    if site == "oral_cavity":
        size_cm = extract_tumor_size_cm(original_text, profile)
        doi_mm = extract_doi_mm(original_text, profile)
        
        if size_cm is not None and doi_mm is not None:
            calculated_t, reason = calculate_oral_cavity_t_stage(size_cm, doi_mm)
            
            # Extract T number from current
            current_t_match = re.search(r'T(\d+)', current_t or "", re.IGNORECASE)
            current_t_num = int(current_t_match.group(1)) if current_t_match else None
            
            calculated_t_num = int(re.search(r'T(\d+)', calculated_t).group(1))
            
            # Correct if calculated is higher
            if current_t_num is None or calculated_t_num > current_t_num:
                result.corrected_t = f"{prefix}{calculated_t}"
                result.t_correction_reason = reason
                profile["tnm_t"] = result.corrected_t
                
                logger.info(f"[StagingValidator] T stage corrected: {result.original_t} → {result.corrected_t}")
            else:
                result.corrected_t = result.original_t
        else:
            result.corrected_t = result.original_t
            if size_cm is None:
                result.warnings.append("Could not extract tumor size for T stage validation")
            if doi_mm is None:
                result.warnings.append("Could not extract DOI for T stage validation")
    else:
        result.corrected_t = result.original_t
    
    # =========================================================================
    # N Stage Validation (ENE)
    # =========================================================================
    
    if site in ["oral_cavity", "oropharynx", "larynx", "head_neck"]:
        has_ene = detect_ene(original_text, profile)
        
        if has_ene is True:
            corrected_n, reason = validate_n_stage_for_ene(
                result.original_n, 
                has_ene=True,
                prefix=prefix
            )
            
            if corrected_n != result.original_n:
                result.corrected_n = corrected_n
                result.n_correction_reason = reason
                profile["tnm_n"] = result.corrected_n
                
                logger.info(f"[StagingValidator] N stage corrected for ENE: {result.original_n} → {result.corrected_n}")
            else:
                result.corrected_n = result.original_n
        else:
            result.corrected_n = result.original_n
    else:
        result.corrected_n = result.original_n
    
    # M stage - no special rules, just pass through
    result.corrected_m = result.original_m
    
    # =========================================================================
    # Stage Group Inference
    # =========================================================================
    
    if use_ajcc_tables:
        try:
            from .staging_search_expander import AJCCStagingTables
            
            tables = AJCCStagingTables()
            tables.load()
            
            # Infer cancer type for table lookup
            cancer_type = tables.resolve_cancer_type(original_text)
            
            if cancer_type:
                t = re.sub(r'^[cyp]+', '', (result.corrected_t or "").lower(), flags=re.IGNORECASE)
                t = t.replace("t", "")
                n = re.sub(r'^[cyp]+', '', (result.corrected_n or "").lower(), flags=re.IGNORECASE)
                n = n.replace("n", "")
                m = re.sub(r'^[cyp]+', '', (result.corrected_m or "").lower(), flags=re.IGNORECASE)
                m = m.replace("m", "")
                
                if t and n and m:
                    stage, confidence, _ = tables.lookup_stage_group(
                        cancer_type, t, n, m
                    )
                    
                    if stage:
                        result.inferred_stage_group = stage
                        result.stage_group_confidence = confidence
                        
                        # Update profile if no stage present
                        if not profile.get("cancer_stage"):
                            profile["cancer_stage"] = stage
                            logger.info(f"[StagingValidator] Inferred stage group: {stage}")
        except ImportError:
            result.warnings.append("staging_search_expander not available for stage group inference")
        except Exception as e:
            result.warnings.append(f"Stage group inference failed: {e}")
    
    # Log if corrections were made
    if result.had_corrections:
        logger.info(f"[StagingValidator] Corrections made: T={result.original_t}→{result.corrected_t}, "
                   f"N={result.original_n}→{result.corrected_n}")
    
    return profile


def get_staging_validation_summary(
    profile: Dict[str, Any], 
    original_text: str
) -> StagingValidationResult:
    """
    Get a detailed validation result without modifying the profile.
    
    Useful for debugging or displaying validation details.
    """
    # Make a copy to avoid modifying original
    profile_copy = dict(profile)
    
    result = StagingValidationResult()
    result.original_t = profile.get("tnm_t")
    result.original_n = profile.get("tnm_n")
    result.original_m = profile.get("tnm_m")
    
    # Run validation on copy
    validate_and_correct_staging(profile_copy, original_text)
    
    result.corrected_t = profile_copy.get("tnm_t")
    result.corrected_n = profile_copy.get("tnm_n")
    result.corrected_m = profile_copy.get("tnm_m")
    result.inferred_stage_group = profile_copy.get("cancer_stage")
    
    return result


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("STAGING INFERENCE VALIDATOR TESTS")
    print("=" * 70)
    
    # Test case 1: Oral tongue with DOI upgrade
    print("\n--- Test 1: Oral Tongue DOI Upgrade ---")
    test_text = """A patient presents with a right lateral oral tongue SCC, 1.5 cm in size, 
    8mm depth of invasion, and ipsilateral adenopathy in level Ib (2 cm) and IIa (2.5cm). 
    There is overt extranodal extension of the 1b node. Metastatic work-up is negative."""
    
    profile = {
        "cancer_type": "SCC",
        "anatomical_site": "oral tongue",
        "tnm_t": "T1",  # Wrong - should be T2
        "tnm_n": "N2b",  # Wrong - should be N3b due to ENE
        "tnm_m": "M0",
    }
    
    print(f"Original: T={profile['tnm_t']}, N={profile['tnm_n']}, M={profile['tnm_m']}")
    
    corrected = validate_and_correct_staging(profile, test_text, use_ajcc_tables=False)
    
    print(f"Corrected: T={corrected['tnm_t']}, N={corrected['tnm_n']}, M={corrected['tnm_m']}")
    
    assert "T2" in corrected["tnm_t"], f"Expected T2, got {corrected['tnm_t']}"
    assert "N3b" in corrected["tnm_n"].lower(), f"Expected N3b, got {corrected['tnm_n']}"
    print("✓ Test 1 passed")
    
    # Test case 2: No DOI - should not change T stage
    print("\n--- Test 2: No DOI Mentioned ---")
    test_text2 = "Patient with 3cm oral tongue SCC, N0M0"
    profile2 = {
        "anatomical_site": "oral tongue",
        "tnm_t": "T2",
        "tnm_n": "N0",
        "tnm_m": "M0",
    }
    
    corrected2 = validate_and_correct_staging(profile2, test_text2, use_ajcc_tables=False)
    print(f"Result: T={corrected2['tnm_t']} (unchanged as expected)")
    assert corrected2["tnm_t"] == "T2", "Should not change T stage without DOI info"
    print("✓ Test 2 passed")
    
    # Test case 3: ENE negative - should not upgrade N
    print("\n--- Test 3: ENE Negative ---")
    test_text3 = "Oral cavity SCC with level II node, ENE-"
    profile3 = {
        "anatomical_site": "oral cavity",
        "tnm_n": "N1",
    }
    
    corrected3 = validate_and_correct_staging(profile3, test_text3, use_ajcc_tables=False)
    print(f"Result: N={corrected3['tnm_n']} (unchanged as expected)")
    assert corrected3["tnm_n"] == "N1", "Should not upgrade N when ENE is negative"
    print("✓ Test 3 passed")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)