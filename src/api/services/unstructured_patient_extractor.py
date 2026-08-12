"""
Extract structured PatientProfile from unstructured free-text patient descriptions.
Uses LLM for robust parsing of natural language (e.g. "65 yo male", "elderly woman", "ECOG 1").

Updated to extract TNM staging components with cancer-type-specific rules.
"""

import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI

from src.core.config import settings

# Schema for LLM output - must match PatientProfile fields
# Now includes TNM components with cancer-specific guidance
EXTRACTION_SCHEMA = {
    "age": "integer or null (single number; for 'elderly' use 70, 'young adult' use 25)",
    "gender": "string or null: 'male' | 'female' only",
    "cancer_type": "string or null: e.g. Breast, Lung, NSCLC, Colorectal, Prostate, SCC, Squamous Cell Carcinoma",
    "anatomical_site": "string or null: the body location/organ where cancer is located, e.g. maxilla, oral cavity, tongue, lung, breast, skin, cervix, anus",
    "cancer_stage": "string or null: I | II | III | IV (overall stage group)",
    "tnm_t": "string or null: T stage component (e.g., T1, T2, pT4, cT3). See AJCC rules below.",
    "tnm_n": "string or null: N stage component (e.g., N0, N1, pN2, N3b). N3b if ANY node has ENE.",
    "tnm_m": "string or null: M stage component (e.g., M0, M1)",
    "tumor_size": "string or null: tumor size if mentioned (e.g., 1.5 cm, 3 cm)",
    "doi": "string or null: depth of invasion if mentioned (e.g., 8 mm, 15 mm)",
    "histology": "string or null: e.g. adenocarcinoma, squamous, small cell, poorly differentiated",
    "molecular_markers": "array of strings or null: e.g. ['EGFR+', 'HER2-', 'PD-L1 positive']",
    "performance_status": "string or null: ECOG 0-4 as single digit, e.g. '0', '1'",
    "smoking_status": "string or null: 'never' | 'former' | 'current' only",
    "comorbidities": "array of strings or null: e.g. ['diabetes', 'hypertension']",
    "lvi": "string or null: lymphovascular invasion status (positive, negative, LVI+, LVI-)",
    "pni": "string or null: perineural invasion status (positive, negative, PNI+, PNI-)",
    "margins": "string or null: margin status (positive, negative, close)",
    "lymph_nodes": "string or null: lymph node status description (e.g., '0/42 LN involved')",
    "ene": "string or null: extranodal extension status (present, absent, overt ENE, etc.)",
    "prior_treatments": "array of strings or null: prior surgeries, chemo, radiation",
}

EXTRACTION_SYSTEM = """You are a medical data extractor specializing in oncology staging. Extract patient characteristics from clinical free text into structured JSON.

CRITICAL TNM STAGING RULES (AJCC 8th Edition):

**ORAL CAVITY (tongue, gingiva, floor of mouth, buccal mucosa, hard palate, maxilla, mandible):**
- T staging uses BOTH tumor size AND depth of invasion (DOI) - use whichever gives HIGHER T stage:
  - T1: ≤2 cm AND DOI ≤5 mm
  - T2: ≤2 cm with DOI >5 mm but ≤10 mm, OR >2 cm but ≤4 cm with DOI ≤10 mm
  - T3: >4 cm OR DOI >10 mm
  - T4a: Invasion through cortical bone, maxillary sinus, or skin of face
  - T4b: Invasion of masticator space, pterygoid plates, skull base, or encases carotid
- EXAMPLE: 1.5 cm tumor with 8 mm DOI = T2 (size says T1, but DOI >5mm upgrades to T2)

**NODAL STAGING (Head & Neck):**
- N3b: ANY lymph node with clinically overt extranodal extension (ENE), regardless of size or number
- If ENE is mentioned (overt, gross, extracapsular extension) → N3b

**BREAST:**
- T1: ≤2 cm, T2: >2-5 cm, T3: >5 cm, T4: chest wall/skin involvement

**LUNG:**
- T1: ≤3 cm, T2: >3-5 cm or involving main bronchus/visceral pleura, T3: >5-7 cm or chest wall, T4: >7 cm or mediastinum

General Rules:
- Output ONLY valid JSON with the specified keys
- Use null for any field not mentioned or unclear
- Preserve c/p prefix if stated (cT2, pT4) - pathologic (p) if post-surgery, clinical (c) if pre-surgery
- If DOI is mentioned for oral cavity, APPLY the DOI rule to determine T stage
- If ENE is mentioned for any node, N stage is N3b

Do not include any text outside the JSON object."""

EXTRACTION_USER_TEMPLATE = """Extract patient characteristics from this description:

"{text}"

Return exactly one JSON object. Apply AJCC 8th edition staging rules, especially:
- For oral cavity: consider BOTH size AND depth of invasion for T stage
- For nodes with ENE: stage as N3b regardless of size

IMPORTANT:
- Output MUST be a FLAT JSON object whose keys are exactly the field names listed
  in the schema (age, gender, cancer_type, anatomical_site, cancer_stage, tnm_t,
  tnm_n, tnm_m, tumor_size, doi, histology, molecular_markers, performance_status,
  smoking_status, comorbidities, lvi, pni, margins, lymph_nodes, ene,
  prior_treatments). Do NOT wrap the object inside another key like
  "patient_characteristics", and do NOT nest sub-objects like "tumor_characteristics"
  or "tnm_staging" — put each field at the top level.
- For anatomical_site, extract the specific body location (e.g., maxilla, oral cavity, tongue)
- For tnm_t/tnm_n/tnm_m, extract the staging components (e.g., pT4, N0, M0)
- If tumor size AND DOI are both given for oral cavity, calculate T stage using AJCC rules"""


def extract_patient_profile(unstructured_description: str, openai_client: OpenAI) -> Dict[str, Any]:
    """
    Parse free-text patient description into a structured profile dict suitable for PatientProfile.
    
    Now includes TNM extraction with cancer-type-specific AJCC rules.
    """
    text = (unstructured_description or "").strip()
    if len(text) < 10:
        print(f"[UnstructuredPatientExtractor] Text too short ({len(text)} chars), returning empty profile")
        return _empty_profile()

    print(f"[UnstructuredPatientExtractor] Extracting from text ({len(text)} chars): {text[:300]}...")

    try:
        # Use OpenAI JSON mode so the response is guaranteed-valid JSON. This
        # eliminates the prior regex-based extraction path, which only handled
        # one level of nesting and silently matched the wrong inner object on
        # deeper schemas (e.g. patient_characteristics → tumor_characteristics →
        # tumor_size produced an empty profile).
        response = openai_client.chat.completions.create(
            model=settings.openai_mini_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(text=text)},
            ],
            temperature=0.1,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()

        print(f"[UnstructuredPatientExtractor] LLM response length: {len(content)} chars")
        print(f"[UnstructuredPatientExtractor] LLM response: {content}")

        data = json.loads(content)

        # Defensive: even with the prompt asking for a flat object, mini will
        # occasionally wrap everything in "patient_characteristics" or split
        # staging into a "tnm_staging" / "tumor_characteristics" sub-object.
        # Hoist any known sub-objects up to the root before normalization.
        data = _flatten_extracted(data)

        print(f"[UnstructuredPatientExtractor] Parsed JSON keys: {list(data.keys())}")
        print(f"[UnstructuredPatientExtractor] Parsed JSON values: {data}")

        profile = _normalize_extracted(data)
        
        print(f"[UnstructuredPatientExtractor] Normalized profile keys: {list(profile.keys())}")
        print(f"[UnstructuredPatientExtractor] Normalized profile: {profile}")
        
        # Post-extraction validation: Apply AJCC rules if needed
        profile = _validate_and_correct_staging(profile, text)
        
        final_profile = {k: v for k, v in profile.items() if v is not None}
        print(f"[UnstructuredPatientExtractor] Final profile: {final_profile}")
        
        return final_profile
    except json.JSONDecodeError as e:
        print(f"[UnstructuredPatientExtractor] JSON parse error: {e}")
        print(f"[UnstructuredPatientExtractor] Raw content was: {content[:500] if 'content' in dir() else 'N/A'}")
        import traceback
        traceback.print_exc()
        return _empty_profile()
    except Exception as e:
        print(f"[UnstructuredPatientExtractor] LLM extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return _empty_profile()


_KNOWN_WRAPPERS = ("patient_characteristics", "patient", "profile", "result", "data")
_KNOWN_SUBOBJECTS = ("tumor_characteristics", "tnm_staging", "staging", "tumor")


def _flatten_extracted(data: Dict[str, Any]) -> Dict[str, Any]:
    """Hoist common LLM wrapper / sub-object shapes up to the root.

    gpt-4o-mini sometimes returns the schema wrapped in a single envelope key
    (e.g. ``{"patient_characteristics": {...}}``) or splits staging into a
    ``tnm_staging``/``tumor_characteristics`` sub-object. We tolerate both
    without re-prompting.
    """
    if not isinstance(data, dict):
        return {}

    # Unwrap a single known envelope key if present.
    if len(data) == 1:
        only_key = next(iter(data))
        if only_key in _KNOWN_WRAPPERS and isinstance(data[only_key], dict):
            data = data[only_key]

    out: Dict[str, Any] = {}
    for k, v in data.items():
        # Hoist known sub-objects (tnm_staging, tumor_characteristics, ...)
        # so their inner keys land at the top level. Lowercase all keys so
        # uppercase variants from the LLM (DOI, LVI, PNI) match the lowercase
        # schema in `_normalize_extracted`.
        if k.lower() in _KNOWN_SUBOBJECTS and isinstance(v, dict):
            for sub_k, sub_v in v.items():
                # First-write wins so explicit top-level keys aren't clobbered.
                out.setdefault(sub_k.lower(), sub_v)
        else:
            out.setdefault(k.lower(), v)
    return out


def _normalize_extracted(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize types and values to match PatientProfile."""
    out: Dict[str, Any] = {}
    
    # Age
    if data.get("age") is not None:
        try:
            out["age"] = int(data["age"])
        except (TypeError, ValueError):
            pass
    
    # Gender
    if data.get("gender") is not None:
        g = str(data["gender"]).lower().strip()
        if g in ("male", "female"):
            out["gender"] = g
    
    # Cancer type
    if data.get("cancer_type") is not None and str(data["cancer_type"]).strip():
        out["cancer_type"] = str(data["cancer_type"]).strip()
    
    # Anatomical site
    if data.get("anatomical_site") is not None and str(data["anatomical_site"]).strip():
        out["anatomical_site"] = str(data["anatomical_site"]).strip()
    
    # Cancer stage (overall)
    if data.get("cancer_stage") is not None and str(data["cancer_stage"]).strip():
        out["cancer_stage"] = str(data["cancer_stage"]).strip()
    
    # TNM components
    if data.get("tnm_t") is not None and str(data["tnm_t"]).strip():
        out["tnm_t"] = str(data["tnm_t"]).strip().upper()
    if data.get("tnm_n") is not None and str(data["tnm_n"]).strip():
        out["tnm_n"] = str(data["tnm_n"]).strip().upper()
    if data.get("tnm_m") is not None and str(data["tnm_m"]).strip():
        out["tnm_m"] = str(data["tnm_m"]).strip().upper()
    
    # Tumor size and DOI
    if data.get("tumor_size") is not None and str(data["tumor_size"]).strip():
        out["tumor_size"] = str(data["tumor_size"]).strip()
    if data.get("doi") is not None and str(data["doi"]).strip():
        out["doi"] = str(data["doi"]).strip()
    
    # Histology
    if data.get("histology") is not None and str(data["histology"]).strip():
        out["histology"] = str(data["histology"]).strip()
    
    # Molecular markers
    if data.get("molecular_markers") is not None:
        markers = data["molecular_markers"]
        if isinstance(markers, list):
            out["molecular_markers"] = [str(m).strip() for m in markers if str(m).strip()]
        elif isinstance(markers, str) and markers.strip():
            out["molecular_markers"] = [markers.strip()]
    
    # Performance status
    if data.get("performance_status") is not None:
        ps = str(data["performance_status"]).strip()
        # Extract just the digit
        ps_match = re.search(r'[0-4]', ps)
        if ps_match:
            out["performance_status"] = ps_match.group(0)
    
    # Smoking status
    if data.get("smoking_status") is not None:
        s = str(data["smoking_status"]).lower().strip()
        if s in ("never", "former", "current"):
            out["smoking_status"] = s
    
    # Comorbidities
    if data.get("comorbidities") is not None:
        comorbidities = data["comorbidities"]
        if isinstance(comorbidities, list):
            out["comorbidities"] = [str(c).strip() for c in comorbidities if str(c).strip()]
        elif isinstance(comorbidities, str) and comorbidities.strip():
            out["comorbidities"] = [comorbidities.strip()]
    
    # Pathology details
    if data.get("lvi") is not None and str(data["lvi"]).strip():
        out["lvi"] = str(data["lvi"]).strip()
    if data.get("pni") is not None and str(data["pni"]).strip():
        out["pni"] = str(data["pni"]).strip()
    if data.get("margins") is not None and str(data["margins"]).strip():
        out["margins"] = str(data["margins"]).strip()
    if data.get("lymph_nodes") is not None and str(data["lymph_nodes"]).strip():
        out["lymph_nodes"] = str(data["lymph_nodes"]).strip()
    if data.get("ene") is not None and str(data["ene"]).strip():
        out["ene"] = str(data["ene"]).strip()
    
    # Prior treatments
    if data.get("prior_treatments") is not None:
        treatments = data["prior_treatments"]
        if isinstance(treatments, list):
            out["prior_treatments"] = [str(t).strip() for t in treatments if str(t).strip()]
        elif isinstance(treatments, str) and treatments.strip():
            out["prior_treatments"] = [treatments.strip()]
    
    return out


def _validate_and_correct_staging(profile: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Validate and correct staging using AJCC rules.
    
    This catches cases where the LLM didn't apply staging rules correctly.
    """
    # Check if this is oral cavity cancer
    site = (profile.get("anatomical_site") or "").lower()
    cancer_type = (profile.get("cancer_type") or "").lower()
    
    is_oral_cavity = any(term in site or term in cancer_type for term in [
        "oral", "tongue", "gingiva", "buccal", "palate", "maxilla", "mandible",
        "floor of mouth", "retromolar", "alveolar"
    ])
    
    if is_oral_cavity:
        profile = _validate_oral_cavity_t_stage(profile, original_text)
    
    # Check N staging for ENE
    profile = _validate_n_stage_for_ene(profile, original_text)
    
    return profile


def _validate_oral_cavity_t_stage(profile: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Validate/correct T stage for oral cavity using size + DOI rules.
    
    AJCC 8th Edition Oral Cavity T staging:
    - T1: ≤2 cm AND DOI ≤5 mm
    - T2: ≤2 cm with DOI >5 mm but ≤10 mm, OR >2 cm but ≤4 cm with DOI ≤10 mm
    - T3: >4 cm OR DOI >10 mm
    - T4: Invasion of adjacent structures
    """
    # Extract tumor size in cm
    tumor_size_cm = None
    size_str = profile.get("tumor_size", "")
    if size_str:
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*(cm|mm)', size_str, re.IGNORECASE)
        if size_match:
            value = float(size_match.group(1))
            unit = size_match.group(2).lower()
            tumor_size_cm = value if unit == "cm" else value / 10
    
    # Also try to extract from original text
    if tumor_size_cm is None:
        size_patterns = [
            r'(\d+(?:\.\d+)?)\s*cm\s*(?:in size|tumor|mass|lesion)?',
            r'tumor[^\d]*(\d+(?:\.\d+)?)\s*cm',
            r'(\d+(?:\.\d+)?)\s*mm\s*(?:in size|tumor|mass|lesion)?',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, original_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if 'mm' in pattern:
                    tumor_size_cm = value / 10
                else:
                    tumor_size_cm = value
                break
    
    # Extract DOI in mm
    doi_mm = None
    doi_str = profile.get("doi", "")
    if doi_str:
        doi_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm)', doi_str, re.IGNORECASE)
        if doi_match:
            value = float(doi_match.group(1))
            unit = doi_match.group(2).lower()
            doi_mm = value if unit == "mm" else value * 10
    
    # Also try to extract DOI from original text
    if doi_mm is None:
        doi_patterns = [
            r'(?:DOI|depth of invasion)[^\d]*(\d+(?:\.\d+)?)\s*(mm|cm)',
            r'(\d+(?:\.\d+)?)\s*(mm|cm)\s*(?:DOI|depth of invasion)',
            r'(\d+(?:\.\d+)?)\s*(mm|cm)\s*depth',
        ]
        for pattern in doi_patterns:
            match = re.search(pattern, original_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                doi_mm = value if unit == "mm" else value * 10
                profile["doi"] = f"{doi_mm} mm"
                break
    
    # If we have both size and DOI, calculate correct T stage
    if tumor_size_cm is not None and doi_mm is not None:
        calculated_t = _calculate_oral_cavity_t_stage(tumor_size_cm, doi_mm)
        current_t = profile.get("tnm_t", "")
        
        # Extract just the T number from current
        current_t_num = re.search(r'T(\d+)', current_t, re.IGNORECASE)
        current_t_num = int(current_t_num.group(1)) if current_t_num else None
        
        calculated_t_num = int(re.search(r'T(\d+)', calculated_t).group(1))
        
        # If calculated T is higher, correct it
        if current_t_num is None or calculated_t_num > current_t_num:
            # Preserve prefix (c/p) if present
            prefix = ""
            if current_t and current_t[0].lower() in ['c', 'p']:
                prefix = current_t[0].lower()
            elif 'pathologic' in original_text.lower() or 'pT' in original_text:
                prefix = "p"
            elif 'clinical' in original_text.lower() or 'cT' in original_text:
                prefix = "c"
            
            corrected_t = f"{prefix}{calculated_t}"
            
            if profile.get("tnm_t") != corrected_t:
                print(f"[StagingValidation] Corrected T stage: {profile.get('tnm_t')} → {corrected_t} "
                      f"(size={tumor_size_cm}cm, DOI={doi_mm}mm)")
            
            profile["tnm_t"] = corrected_t
    
    return profile


def _calculate_oral_cavity_t_stage(size_cm: float, doi_mm: float) -> str:
    """
    Calculate T stage for oral cavity based on AJCC 8th edition.
    Uses both size AND DOI - whichever gives higher T stage.
    """
    # T stage based on size alone
    if size_cm <= 2:
        t_by_size = 1
    elif size_cm <= 4:
        t_by_size = 2
    else:
        t_by_size = 3
    
    # T stage based on DOI alone
    if doi_mm <= 5:
        t_by_doi = 1
    elif doi_mm <= 10:
        t_by_doi = 2
    else:
        t_by_doi = 3
    
    # Use higher of the two
    t_stage = max(t_by_size, t_by_doi)
    
    return f"T{t_stage}"


def _validate_n_stage_for_ene(profile: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Validate N stage for extranodal extension (ENE).
    
    Per AJCC 8th Edition:
    - Any node with clinically overt ENE = N3b, regardless of size or number
    """
    text_lower = original_text.lower()
    
    # Check for ENE indicators
    ene_patterns = [
        r'extranodal extension',
        r'extracapsular extension',
        r'ene\s*[+\-]',
        r'ece\s*[+\-]',
        r'overt\s*(?:extranodal|ene)',
        r'gross\s*(?:extranodal|ene)',
        r'clinical(?:ly)?\s*(?:overt)?\s*(?:extranodal|ene)',
    ]
    
    has_ene = False
    for pattern in ene_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Check if it's positive (not ENE- or ENE negative)
            context_start = max(0, match.start() - 10)
            context_end = min(len(text_lower), match.end() + 20)
            context = text_lower[context_start:context_end]
            
            if 'ene-' in context or 'ene -' in context or 'ene negative' in context:
                continue
            if 'no ene' in context or 'without ene' in context:
                continue
            if 'ece-' in context or 'ece -' in context or 'ece negative' in context:
                continue
            
            # Check for positive indicators
            if any(pos in context for pos in ['ene+', 'ene +', 'ene positive', 'overt', 'gross', 
                                               'ece+', 'ece +', 'with ene', 'with extranodal']):
                has_ene = True
                break
            
            # Also check profile
            ene_status = (profile.get("ene") or "").lower()
            if any(pos in ene_status for pos in ['positive', 'present', 'overt', 'gross', '+']):
                has_ene = True
                break
    
    if has_ene:
        current_n = profile.get("tnm_n", "")
        
        # If not already N3b, upgrade
        if "N3B" not in current_n.upper():
            # Preserve prefix
            prefix = ""
            if current_n and current_n[0].lower() in ['c', 'p']:
                prefix = current_n[0].lower()
            elif 'pathologic' in original_text.lower() or 'pN' in original_text:
                prefix = "p"
            elif 'clinical' in original_text.lower() or 'cN' in original_text:
                prefix = "c"
            
            corrected_n = f"{prefix}N3b"
            
            if profile.get("tnm_n") != corrected_n:
                print(f"[StagingValidation] Corrected N stage for ENE: {profile.get('tnm_n')} → {corrected_n}")
            
            profile["tnm_n"] = corrected_n
    
    return profile


def _empty_profile() -> Dict[str, Any]:
    """Return empty profile dict."""
    return {}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the staging validation logic
    print("=" * 70)
    print("TESTING ORAL CAVITY T STAGING VALIDATION")
    print("=" * 70)
    
    test_cases = [
        # (size_cm, doi_mm, expected_t)
        (1.5, 8, "T2"),   # Size says T1, DOI says T2 → T2
        (1.5, 4, "T1"),   # Both say T1 → T1
        (3.0, 4, "T2"),   # Size says T2, DOI says T1 → T2
        (3.0, 12, "T3"),  # Size says T2, DOI says T3 → T3
        (5.0, 8, "T3"),   # Size says T3, DOI says T2 → T3
        (2.0, 5, "T1"),   # Edge case: exactly T1 thresholds
        (2.0, 6, "T2"),   # Edge case: DOI just over 5mm
    ]
    
    for size, doi, expected in test_cases:
        result = _calculate_oral_cavity_t_stage(size, doi)
        status = "✓" if result == expected else "✗"
        print(f"{status} Size={size}cm, DOI={doi}mm → {result} (expected {expected})")
    
    print("\n" + "=" * 70)
    print("TESTING FULL EXTRACTION (requires OpenAI API key)")
    print("=" * 70)
    
    try:
        from src.core.config import settings
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        test_text = """A patient presents with a right lateral oral tongue SCC, 1.5 cm in size, 
        8mm depth of invasion, and ipsilateral adenopathy in level Ib (2 cm) and IIa (2.5cm). 
        There is overt extranodal extension of the 1b node. Metastatic work-up is negative."""
        
        print(f"\nInput: {test_text}\n")
        
        profile = extract_patient_profile(test_text, openai_client)
        
        print("Extracted profile:")
        for key, value in sorted(profile.items()):
            print(f"  {key}: {value}")
        
        # Verify
        expected_t = "T2"  # 1.5cm with 8mm DOI
        expected_n = "N3b"  # ENE present
        
        t_correct = expected_t.upper() in (profile.get("tnm_t") or "").upper()
        n_correct = expected_n.upper() in (profile.get("tnm_n") or "").upper()
        
        print(f"\nValidation:")
        print(f"  T stage correct ({expected_t}): {'✓' if t_correct else '✗'}")
        print(f"  N stage correct ({expected_n}): {'✓' if n_correct else '✗'}")
        
    except Exception as e:
        print(f"Skipping API test: {e}")