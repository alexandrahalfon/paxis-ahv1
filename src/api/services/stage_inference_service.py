"""
Stage Inference Service

Infers clinical stage from TNM values and clinical context using
AJCC 8th Edition staging tables loaded from JSON.

This service is called AFTER query classification (when TNM values are
extracted but overall_stage is missing) and also during context normalization.

Integration points:
- query_classifier_service.py: After LLM extraction, infer stage if missing
- staging_normalizer.py: Replace inline AJCC_STAGING_TABLES with this service
- structured_study_matcher.py: Use inferred stage for matching scores

Usage:
    from src.api.services.stage_inference_service import get_stage_inference_service

    service = get_stage_inference_service()
    result = service.infer_stage(
        cancer_type="breast",
        cancer_location="breast",
        tnm_t="T2",
        tnm_n="N1",
        tnm_m="M0",
    )
    # result.stage_group = "IIB"
    # result.is_ambiguous = True (for breast, may vary by grade/receptor)
    # result.possible_stages = ["IB", "IIA", "IIB", "IIIA"]
    # result.required_factors = ["grade", "er_status", "pr_status", "her2_status"]
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StageInferenceResult:
    """Result of a stage inference attempt."""
    
    # The resolved stage group (best guess based on anatomic staging)
    stage_group: Optional[str] = None
    
    # Whether the stage is definitive or could vary by non-anatomic factors
    is_ambiguous: bool = False
    
    # If ambiguous, all possible stages
    possible_stages: List[str] = field(default_factory=list)
    
    # Factors needed to resolve ambiguity (e.g., ["grade", "er_status"])
    required_factors: List[str] = field(default_factory=list)
    
    # Which rule or table produced this result
    source: str = ""  # "exact_match", "wildcard_match", "universal_rule", "text_descriptor", "fallback"
    
    # Confidence: high (exact match), medium (wildcard/fallback), low (text descriptor)
    confidence: str = "low"
    
    # The cancer type table that was used
    cancer_type_key: Optional[str] = None
    
    # The TNM values that were used (normalized)
    tnm_used: Optional[str] = None
    
    # Human-readable notes about the inference
    notes: List[str] = field(default_factory=list)
    
    # Metastatic status inferred
    metastatic_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if v is not None and v != [] and v != "" and v is not False:
                result[k] = v
        # Always include is_ambiguous
        result["is_ambiguous"] = self.is_ambiguous
        return result


# =============================================================================
# MAIN SERVICE
# =============================================================================

class StageInferenceService:
    """
    Service that infers clinical stage from TNM and clinical context.
    
    Loads AJCC staging tables from JSON at init time (sub-ms lookups).
    """
    
    def __init__(self, tables_path: Optional[str] = None):
        """
        Initialize with path to AJCC staging tables JSON.
        
        Args:
            tables_path: Path to ajcc_staging_tables.json. 
                         Defaults to src/data/ajcc_staging_tables.json
        """
        if tables_path is None:
            # Try multiple paths for flexibility
            # Service lives at: src/api/services/stage_inference_service.py
            # JSON lives at:    data/ajcc_staging_tables.json (project root)
            service_dir = Path(__file__).parent          # src/api/services/
            project_root = service_dir.parent.parent.parent  # project root
            candidates = [
                project_root / "data" / "ajcc_staging_tables.json",
                Path("data/ajcc_staging_tables.json"),                # CWD = project root
                Path("data") / "ajcc_staging_tables.json",
            ]
            for p in candidates:
                if p.exists():
                    tables_path = str(p)
                    break
        
        if tables_path and Path(tables_path).exists():
            with open(tables_path, "r") as f:
                self._data = json.load(f)
            logger.info(f"[StageInference] Loaded AJCC tables from {tables_path}")
        else:
            logger.warning("[StageInference] No staging tables JSON found, using empty tables")
            self._data = {"cancer_types": {}, "universal_rules": {"rules": [], "exceptions": []}, 
                          "text_descriptor_mapping": {"descriptors": {}}, "site_resolution": {"rules": []}}
        
        self._cancer_types = self._data.get("cancer_types", {})
        self._universal_rules = self._data.get("universal_rules", {})
        self._text_descriptors = self._data.get("text_descriptor_mapping", {}).get("descriptors", {})
        self._site_rules = self._data.get("site_resolution", {}).get("rules", [])
        
        # Build a flat alias -> table_key lookup for fast resolution
        self._alias_map: Dict[str, str] = {}
        for key, config in self._cancer_types.items():
            for alias in config.get("aliases", []):
                self._alias_map[alias.lower()] = key
    
    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    
    def infer_stage(
        self,
        cancer_type: Optional[str] = None,
        cancer_location: Optional[str] = None,
        tnm_t: Optional[str] = None,
        tnm_n: Optional[str] = None,
        tnm_m: Optional[str] = None,
        metastatic_status: Optional[str] = None,
        age: Optional[int] = None,
        hpv_status: Optional[str] = None,
        histopathologic_type: Optional[str] = None,
        molecular_subtype: Optional[str] = None,
        tumor_grade: Optional[str] = None,
    ) -> StageInferenceResult:
        """
        Infer clinical stage from available data.
        
        Args:
            cancer_type: Cancer type string (e.g., "breast cancer", "SCC")
            cancer_location: Anatomical location (e.g., "oral cavity", "lung")
            tnm_t: T stage (e.g., "T2", "T4a", "pT3") 
            tnm_n: N stage (e.g., "N0", "N1", "N2b")
            tnm_m: M stage (e.g., "M0", "M1")
            metastatic_status: Text descriptor (e.g., "metastatic", "localized")
            age: Patient age (critical for thyroid staging)
            hpv_status: HPV/p16 status (critical for oropharynx)
            histopathologic_type: Histology (to distinguish e.g. esophageal adeno vs squamous)
            molecular_subtype: Molecular markers (e.g., "HER2+", "ER+")
            tumor_grade: Grade (e.g., "G1", "poorly differentiated")
        
        Returns:
            StageInferenceResult with inferred stage and metadata
        """
        result = StageInferenceResult()
        
        # Normalize TNM values
        t = self._normalize_tnm_component(tnm_t, "t") if tnm_t else None
        n = self._normalize_tnm_component(tnm_n, "n") if tnm_n else None
        m = self._normalize_tnm_component(tnm_m, "m") if tnm_m else None
        
        if t or n or m:
            result.tnm_used = f"T{t or 'x'}N{n or 'x'}M{m or 'x'}"
        
        # Step 1: Apply universal rules first
        universal = self._check_universal_rules(t, n, m, metastatic_status, cancer_type, cancer_location)
        if universal:
            result.stage_group = universal["stage"]
            result.source = "universal_rule"
            result.confidence = "high"
            result.notes.append(universal["note"])
            if m and m.startswith("1"):
                result.metastatic_status = "metastatic"
            if metastatic_status and metastatic_status.lower() == "metastatic":
                result.metastatic_status = "metastatic"
            return result
        
        # Step 2: Resolve cancer type to staging table
        table_key = self._resolve_cancer_type(
            cancer_type=cancer_type,
            cancer_location=cancer_location,
            hpv_status=hpv_status,
            histopathologic_type=histopathologic_type,
        )
        result.cancer_type_key = table_key
        
        if not table_key:
            # No matching table — try text descriptors as last resort
            if metastatic_status:
                text_result = self._check_text_descriptors(metastatic_status)
                if text_result:
                    result.stage_group = text_result.get("stage")
                    result.possible_stages = text_result.get("stage_range", [])
                    result.source = "text_descriptor"
                    result.confidence = text_result.get("confidence", "low")
                    result.notes.append(f"Inferred from text descriptor: '{metastatic_status}'")
                    return result
            
            # Fallback: use general rules based on T/N/M values
            if t and n and m:
                result = self._apply_general_fallback(t, n, m)
            else:
                result.notes.append("Insufficient data: need cancer type + complete TNM for stage inference")
            return result
        
        # Step 3: Handle age-dependent staging (thyroid)
        cancer_config = self._cancer_types.get(table_key, {})
        if table_key == "thyroid_differentiated":
            return self._infer_thyroid_stage(t, n, m, age, cancer_config, result)
        
        # Step 4: Look up in the anatomic_stage table
        stage_table = cancer_config.get("anatomic_stage", {})
        
        if not (t and n and m):
            # Incomplete TNM — try partial inference
            result.notes.append("Incomplete TNM staging data")
            if m and m.startswith("1"):
                # M1 with incomplete T/N — still Stage IV for most cancers
                m1_stage = self._lookup_in_table(stage_table, "any", "any", m)
                if m1_stage:
                    result.stage_group = m1_stage
                    result.source = "partial_match_m1"
                    result.confidence = "high"
                    result.metastatic_status = "metastatic"
                    return result
            # Can't determine without complete TNM
            result.confidence = "low"
            return result
        
        # Try exact match
        stage = self._lookup_in_table(stage_table, t, n, m)
        if stage:
            result.stage_group = stage
            result.source = "exact_match"
            result.confidence = "high"
        else:
            # Try stripping sub-stage suffixes progressively
            stage = self._fuzzy_lookup(stage_table, t, n, m)
            if stage:
                result.stage_group = stage
                result.source = "fuzzy_match"
                result.confidence = "medium"
                result.notes.append(f"Matched via sub-stage normalization")
        
        # Step 5: Check for ambiguity (prognostic staging)
        if cancer_config.get("has_prognostic_stage"):
            ambiguous_cases = cancer_config.get("ambiguous_cases", [])
            for case in ambiguous_cases:
                if self._tnm_matches_case(t, n, m, case.get("tnm", "")):
                    result.is_ambiguous = True
                    result.possible_stages = case.get("prognostic_range", [])
                    result.required_factors = case.get("determining_factors", [])
                    
                    # Try to resolve with available data
                    resolved = self._try_resolve_ambiguity(
                        case, tumor_grade=tumor_grade, 
                        molecular_subtype=molecular_subtype, age=age
                    )
                    if resolved:
                        result.stage_group = resolved
                        result.is_ambiguous = False
                        result.source = "prognostic_resolved"
                        result.notes.append("Resolved prognostic stage with available factors")
                    else:
                        result.notes.append(
                            f"Anatomic stage is {result.stage_group}; "
                            f"prognostic stage requires: {', '.join(result.required_factors)}"
                        )
                    break
        
        # Set metastatic status
        if m and m.startswith("1"):
            result.metastatic_status = "metastatic"
        elif result.stage_group:
            if result.stage_group.startswith("IV"):
                result.metastatic_status = "metastatic"
            elif result.stage_group.startswith("III"):
                result.metastatic_status = "locally advanced"
            else:
                result.metastatic_status = "localized"
        
        return result
    
    def infer_stage_from_text(
        self,
        text: str,
        cancer_type: Optional[str] = None,
        cancer_location: Optional[str] = None,
    ) -> StageInferenceResult:
        """
        Infer stage from free text by checking text descriptors.
        Useful when no structured TNM is available.
        """
        text_lower = text.lower()
        
        best_result = StageInferenceResult()
        best_confidence_rank = -1
        confidence_ranks = {"high": 2, "medium": 1, "low": 0}
        
        for descriptor, mapping in self._text_descriptors.items():
            if descriptor.lower() in text_lower:
                # Check if descriptor is cancer-type-specific
                applies_to = mapping.get("applies_to")
                if applies_to:
                    table_key = self._resolve_cancer_type(cancer_type, cancer_location)
                    if table_key not in applies_to:
                        continue
                
                conf_rank = confidence_ranks.get(mapping.get("confidence", "low"), 0)
                if conf_rank > best_confidence_rank:
                    best_confidence_rank = conf_rank
                    best_result = StageInferenceResult(
                        stage_group=mapping.get("stage"),
                        possible_stages=mapping.get("stage_range", []),
                        source="text_descriptor",
                        confidence=mapping.get("confidence", "low"),
                        notes=[f"Matched text descriptor: '{descriptor}'"],
                    )
                    if mapping.get("stage") and mapping["stage"].startswith("IV"):
                        best_result.metastatic_status = "metastatic"
        
        return best_result
    
    def get_supported_cancer_types(self) -> List[Dict[str, str]]:
        """Return list of supported cancer types with their display names."""
        return [
            {"key": key, "display_name": config.get("display_name", key)}
            for key, config in self._cancer_types.items()
        ]
    
    def get_staging_table(self, cancer_type_key: str) -> Optional[Dict]:
        """Get the raw staging table for a cancer type (for debugging/display)."""
        return self._cancer_types.get(cancer_type_key)
    
    # -------------------------------------------------------------------------
    # INTERNAL: TNM normalization
    # -------------------------------------------------------------------------
    
    def _normalize_tnm_component(self, value: str, component: str) -> str:
        """
        Normalize a TNM component value.
        
        Strips prefix (c/p/y), the component letter (T/N/M), and lowercases.
        Examples: "cT2a" -> "2a", "pN1" -> "1", "M0" -> "0"
        """
        v = value.strip()
        # Strip clinical/pathologic/post-therapy prefix
        v = re.sub(r'^[cCpPyY]+', '', v)
        # Strip the T/N/M letter
        letter = component.upper()
        if v.upper().startswith(letter):
            v = v[len(letter):]
        return v.lower()
    
    # -------------------------------------------------------------------------
    # INTERNAL: Universal rules
    # -------------------------------------------------------------------------
    
    def _check_universal_rules(
        self, t: Optional[str], n: Optional[str], m: Optional[str],
        metastatic_status: Optional[str],
        cancer_type: Optional[str] = None,
        cancer_location: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Check universal rules that apply across all cancer types."""
        
        # Resolve to check exceptions
        table_key = self._resolve_cancer_type(cancer_type, cancer_location)
        exceptions = self._universal_rules.get("exceptions", [])
        
        # M1 rule (strongest) — but defer to cancer-type table if M has sub-stages (1a, 1b, 1c)
        if m and m.startswith("1"):
            if table_key not in exceptions:
                # If M1 has a sub-stage suffix AND we have a matching cancer-type table,
                # let the type-specific table handle it (e.g., lung IVA vs IVB)
                has_substage = len(m) > 1 and m != "1"
                has_type_table = table_key is not None and table_key in self._cancer_types
                if has_substage and has_type_table:
                    pass  # Skip universal rule, let type-specific table resolve
                else:
                    return {"stage": "IV", "note": "M1 (distant metastasis) = Stage IV (universal rule)"}
        
        # Tis + N0 + M0 rule
        if t == "is" and n == "0" and m == "0":
            return {"stage": "0", "note": "Tis N0 M0 = Stage 0 (carcinoma in situ, universal rule)"}
        
        # Metastatic text descriptor (no TNM needed)
        if metastatic_status and metastatic_status.lower() in ("metastatic", "distant metastasis", "disseminated"):
            if table_key not in exceptions:
                return {"stage": "IV", "note": f"Text descriptor '{metastatic_status}' = Stage IV (universal rule)"}
        
        return None
    
    # -------------------------------------------------------------------------
    # INTERNAL: Cancer type resolution
    # -------------------------------------------------------------------------
    
    def _resolve_cancer_type(
        self,
        cancer_type: Optional[str] = None,
        cancer_location: Optional[str] = None,
        hpv_status: Optional[str] = None,
        histopathologic_type: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve cancer type/location to a staging table key."""
        
        # First try site_resolution rules (location-based, more specific)
        search_terms = []
        if cancer_location:
            search_terms.append(cancer_location.lower())
        if cancer_type:
            search_terms.append(cancer_type.lower())
        
        for rule in self._site_rules:
            match_any = [m.lower() for m in rule.get("match_any", [])]
            for term in search_terms:
                for match_term in match_any:
                    if match_term in term or term in match_term:
                        # Check HPV-specific routing for oropharynx
                        if hpv_status and hpv_status.lower() in ("positive", "hpv+", "p16+", "p16 positive"):
                            if "table_key_if_hpv_positive" in rule:
                                return rule["table_key_if_hpv_positive"]
                        return rule.get("table_key")
        
        # Then try direct alias lookup
        for term in search_terms:
            if term in self._alias_map:
                return self._alias_map[term]
            # Partial match
            for alias, key in self._alias_map.items():
                if alias in term or term in alias:
                    return key
        
        # Check histology for esophageal subtype
        if histopathologic_type:
            histo_lower = histopathologic_type.lower()
            if "squamous" in histo_lower:
                for term in search_terms:
                    if "esophag" in term:
                        return "esophageal_squamous"
            if "small cell" in histo_lower:
                for term in search_terms:
                    if "lung" in term:
                        return "sclc"
        
        return None
    
    # -------------------------------------------------------------------------
    # INTERNAL: Table lookup
    # -------------------------------------------------------------------------
    
    def _lookup_in_table(
        self, table: Dict[str, str], t: str, n: str, m: str
    ) -> Optional[str]:
        """
        Look up stage in a staging table.
        Keys are formatted as "t__n__m" (e.g., "2__1__0").
        Supports 'any' wildcards.
        """
        # Exact match
        key = f"{t}__{n}__{m}"
        if key in table:
            return table[key]
        
        # Wildcard matches (check all combinations)
        wildcard_patterns = [
            f"any__{n}__{m}",
            f"{t}__any__{m}",
            f"{t}__{n}__any",
            f"any__any__{m}",
            f"any__{n}__any",
            f"{t}__any__any",
            f"any__any__any",
        ]
        for pattern in wildcard_patterns:
            if pattern in table:
                return table[pattern]
        
        return None
    
    def _fuzzy_lookup(
        self, table: Dict[str, str], t: str, n: str, m: str
    ) -> Optional[str]:
        """
        Try progressively broader matches by stripping sub-stage suffixes.
        
        For example: t="2a" -> try "2a", then "2"
                     n="2b" -> try "2b", then "2"
        """
        t_variants = self._get_tnm_variants(t)
        n_variants = self._get_tnm_variants(n)
        m_variants = self._get_tnm_variants(m)
        
        for tv in t_variants:
            for nv in n_variants:
                for mv in m_variants:
                    result = self._lookup_in_table(table, tv, nv, mv)
                    if result:
                        return result
        
        return None
    
    def _get_tnm_variants(self, value: str) -> List[str]:
        """
        Generate progressively broader variants of a TNM component.
        "2b" -> ["2b", "2"]
        "1mi" -> ["1mi", "1"]
        "4a" -> ["4a", "4"]
        """
        variants = [value]
        # Strip trailing letters
        base = re.sub(r'[a-z]+$', '', value)
        if base != value and base:
            variants.append(base)
        return variants
    
    # -------------------------------------------------------------------------
    # INTERNAL: Text descriptors
    # -------------------------------------------------------------------------
    
    def _check_text_descriptors(self, text: str) -> Optional[Dict[str, Any]]:
        """Check text against known descriptors."""
        text_lower = text.lower().strip()
        if text_lower in self._text_descriptors:
            return self._text_descriptors[text_lower]
        # Partial match
        for descriptor, mapping in self._text_descriptors.items():
            if descriptor in text_lower:
                return mapping
        return None
    
    # -------------------------------------------------------------------------
    # INTERNAL: General fallback
    # -------------------------------------------------------------------------
    
    def _apply_general_fallback(self, t: str, n: str, m: str) -> StageInferenceResult:
        """
        Apply general staging rules when no cancer-type-specific table is available.
        These are conservative estimates based on the universal logic.
        """
        result = StageInferenceResult(source="fallback", confidence="low")
        result.tnm_used = f"T{t}N{n}M{m}"
        result.notes.append("No cancer-type-specific table available; using general rules")
        
        # M1 -> Stage IV
        if m.startswith("1"):
            result.stage_group = "IV"
            result.confidence = "high"
            result.metastatic_status = "metastatic"
            return result
        
        # Tis N0 M0 -> Stage 0
        if t == "is" and n == "0" and m == "0":
            result.stage_group = "0"
            result.confidence = "high"
            return result
        
        # N0 M0: Stage depends on T
        if n == "0" and m == "0":
            t_num = re.match(r'(\d)', t)
            if t_num:
                t_val = int(t_num.group(1))
                if t_val <= 2:
                    result.stage_group = "I"
                    result.possible_stages = ["I", "II"]
                    result.confidence = "medium"
                elif t_val == 3:
                    result.stage_group = "II"
                    result.possible_stages = ["II", "III"]
                    result.confidence = "low"
                elif t_val == 4:
                    result.stage_group = "III"
                    result.possible_stages = ["III", "IVA"]
                    result.confidence = "low"
            return result
        
        # N+ M0: at least Stage II, usually III
        if not n.startswith("0") and m == "0":
            n_num = re.match(r'(\d)', n)
            if n_num:
                n_val = int(n_num.group(1))
                if n_val == 1:
                    result.stage_group = "II"
                    result.possible_stages = ["II", "III"]
                    result.confidence = "low"
                elif n_val >= 2:
                    result.stage_group = "III"
                    result.possible_stages = ["III", "IVA"]
                    result.confidence = "low"
            else:
                result.stage_group = "III"
                result.possible_stages = ["II", "III"]
                result.confidence = "low"
            return result
        
        return result
    
    # -------------------------------------------------------------------------
    # INTERNAL: Thyroid (age-dependent)
    # -------------------------------------------------------------------------
    
    def _infer_thyroid_stage(
        self, t: Optional[str], n: Optional[str], m: Optional[str],
        age: Optional[int], cancer_config: Dict, result: StageInferenceResult
    ) -> StageInferenceResult:
        """Handle thyroid cancer's age-dependent staging."""
        result.cancer_type_key = "thyroid_differentiated"
        
        if age is None:
            # Age unknown — return both possibilities
            result.is_ambiguous = True
            result.required_factors = ["age"]
            result.notes.append("Thyroid staging requires patient age (<55 vs >=55)")
            
            # If M1, it's either Stage II (young) or IVB (old)
            if m and m.startswith("1"):
                result.possible_stages = ["II", "IVB"]
                result.notes.append("M1: Stage II if age<55, Stage IVB if age>=55")
            else:
                result.possible_stages = ["I", "II", "III", "IVA"]
                result.notes.append("M0: Stage I if age<55; look up age>=55 table for specific stage")
            return result
        
        if age < 55:
            table = cancer_config.get("anatomic_stage_age_under_55", {})
            result.notes.append(f"Using thyroid age<55 staging (patient age: {age})")
        else:
            table = cancer_config.get("anatomic_stage_age_55_plus", {})
            result.notes.append(f"Using thyroid age>=55 staging (patient age: {age})")
        
        if t and n and m:
            stage = self._lookup_in_table(table, t, n, m)
            if not stage:
                stage = self._fuzzy_lookup(table, t, n, m)
            if stage:
                result.stage_group = stage
                result.source = "exact_match"
                result.confidence = "high"
            else:
                result.notes.append("No exact match found in thyroid staging table")
        else:
            result.notes.append("Incomplete TNM for thyroid staging")
        
        return result
    
    # -------------------------------------------------------------------------
    # INTERNAL: Ambiguity resolution
    # -------------------------------------------------------------------------
    
    def _tnm_matches_case(self, t: str, n: str, m: str, case_tnm: str) -> bool:
        """Check if T/N/M values match an ambiguous case description."""
        case_lower = case_tnm.lower()
        
        # Handle range patterns like "T1-T2 N0 M0"
        t_match = f"t{t}" in case_lower or "any t" in case_lower
        n_match = f"n{n}" in case_lower or "any n" in case_lower
        m_match = f"m{m}" in case_lower or "any m" in case_lower
        
        # Handle range like "T1-T2"
        t_range_match = re.search(r't(\d)-t(\d)', case_lower)
        if t_range_match:
            t_num = re.match(r'(\d)', t)
            if t_num:
                t_val = int(t_num.group(1))
                t_match = int(t_range_match.group(1)) <= t_val <= int(t_range_match.group(2))
        
        # Also match specific TNM strings
        tnm_str = f"t{t}n{n}m{m}"
        if tnm_str in case_lower.replace(" ", ""):
            return True
        
        return t_match and n_match and m_match
    
    def _try_resolve_ambiguity(
        self, case: Dict, tumor_grade: Optional[str] = None,
        molecular_subtype: Optional[str] = None, age: Optional[int] = None,
    ) -> Optional[str]:
        """
        Try to resolve an ambiguous case using available non-anatomic factors.
        Returns resolved stage or None if insufficient data.
        """
        examples = case.get("examples", {})
        if not examples:
            return None
        
        # This is a simplified matcher — for production, you'd want
        # more sophisticated logic per cancer type
        # For now, return None to indicate ambiguity (user must provide more data)
        # This keeps the system honest rather than guessing
        
        return None


# =============================================================================
# SINGLETON
# =============================================================================

_service: Optional[StageInferenceService] = None


def get_stage_inference_service(tables_path: Optional[str] = None) -> StageInferenceService:
    """Get or create the singleton StageInferenceService."""
    global _service
    if _service is None:
        _service = StageInferenceService(tables_path)
    return _service


# =============================================================================
# CONVENIENCE FUNCTION (for use in query_classifier_service)
# =============================================================================

def infer_stage_for_query(
    cancer_type: Optional[str] = None,
    cancer_location: Optional[str] = None,
    tnm_t: Optional[str] = None,
    tnm_n: Optional[str] = None,
    tnm_m: Optional[str] = None,
    metastatic_status: Optional[str] = None,
    age: Optional[int] = None,
    hpv_status: Optional[str] = None,
    histopathologic_type: Optional[str] = None,
    molecular_subtype: Optional[str] = None,
    tumor_grade: Optional[str] = None,
) -> StageInferenceResult:
    """
    Convenience function for inferring stage from query classifier output.
    
    Call this after LLM extraction when overall_stage is None but TNM values exist.
    
    Example:
        result = infer_stage_for_query(
            cancer_type="breast cancer",
            tnm_t="T2", tnm_n="N1", tnm_m="M0"
        )
        if result.stage_group:
            query.overall_stage = result.stage_group
        if result.is_ambiguous:
            query.stage_inference_notes = result.notes
    """
    service = get_stage_inference_service()
    return service.infer_stage(
        cancer_type=cancer_type,
        cancer_location=cancer_location,
        tnm_t=tnm_t,
        tnm_n=tnm_n,
        tnm_m=tnm_m,
        metastatic_status=metastatic_status,
        age=age,
        hpv_status=hpv_status,
        histopathologic_type=histopathologic_type,
        molecular_subtype=molecular_subtype,
        tumor_grade=tumor_grade,
    )