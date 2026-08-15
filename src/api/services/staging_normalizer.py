"""
Staging Normalizer for RAG Query Pipeline

This module normalizes staging information at query/generation time to improve
answer accuracy without requiring document reprocessing.

Key Features:
- Extracts TNM staging from retrieved chunks
- Validates staging against AJCC 8th edition tables (loaded from JSON)
- Resolves conflicts between clinical (c) and pathologic (p) staging
- Maps TNM to stage groups for consistency

This addresses the issue where the LLM may misinterpret staging information
in the retrieved context, leading to incorrect answers.

Updated to use shared AJCCStagingTables from staging_search_expander.py
for comprehensive coverage (35 cancer types, 600+ mappings).
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import shared staging tables
from .staging_search_expander import AJCCStagingTables, get_staging_expander


@dataclass
class StagingInfo:
    """Extracted staging information from a chunk."""
    t_stage: Optional[str] = None
    n_stage: Optional[str] = None
    m_stage: Optional[str] = None
    stage_group: Optional[str] = None
    staging_type: Optional[str] = None  # "clinical", "pathologic", "yp", or "unknown"
    confidence: float = 0.0
    source_text: str = ""
    
    def to_tnm_string(self) -> str:
        """Convert to TNM string format."""
        if self.staging_type == "clinical":
            prefix = "c"
        elif self.staging_type == "pathologic":
            prefix = "p"
        elif self.staging_type == "yp":
            prefix = "yp"
        else:
            prefix = ""
        
        parts = []
        if self.t_stage:
            parts.append(f"{prefix}T{self.t_stage}")
        if self.n_stage:
            n_prefix = prefix if prefix else ""
            parts.append(f"{n_prefix}N{self.n_stage}")
        if self.m_stage:
            m_prefix = prefix if prefix else ""
            parts.append(f"{m_prefix}M{self.m_stage}")
        return "".join(parts)
    
    def is_complete(self) -> bool:
        """Check if TNM staging is complete."""
        return all([self.t_stage, self.n_stage, self.m_stage])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "t_stage": self.t_stage,
            "n_stage": self.n_stage,
            "m_stage": self.m_stage,
            "stage_group": self.stage_group,
            "staging_type": self.staging_type,
            "confidence": self.confidence,
            "tnm_string": self.to_tnm_string() if self.is_complete() else None,
        }


class StagingNormalizer:
    """
    Normalizes staging information in retrieved context to improve answer accuracy.
    
    Usage:
        normalizer = StagingNormalizer(openai_client)
        normalized_context = normalizer.normalize_staging_in_context(
            chunks=retrieved_chunks,
            query=user_query,
            cancer_type="breast"
        )
    """
    
    # TNM extraction patterns
    TNM_PATTERNS = {
        "full_tnm": re.compile(
            r'\b([cyp]{0,2})T(is|a|[0-4][a-d]?(?:mi)?|x)\s*'
            r'([cyp]?)N([0-3][a-c]?(?:mi)?|x)\s*'
            r'([cyp]?)M([01][a-c]?|x)\b',
            re.IGNORECASE
        ),
        "t_stage": re.compile(r'\b([cyp]{0,2})T(is|a|[0-4][a-d]?(?:mi)?|x)\b', re.IGNORECASE),
        "n_stage": re.compile(r'\b([cyp]?)N([0-3][a-c]?(?:mi)?|x)\b', re.IGNORECASE),
        "m_stage": re.compile(r'\b([cyp]?)M([01][a-c]?|x)\b', re.IGNORECASE),
        "stage_group": re.compile(
            r'\bstage\s*([0IV]{1,3}|[1-4])\s*([ABC]?[123]?)\b',
            re.IGNORECASE
        ),
    }
    
    def __init__(self, openai_client: Optional[Any] = None, staging_tables: Optional[AJCCStagingTables] = None):
        """
        Initialize the staging normalizer.
        
        Args:
            openai_client: Optional OpenAI client for LLM-based extraction
            staging_tables: Optional AJCCStagingTables instance. If None, uses shared instance.
        """
        self.openai_client = openai_client
        
        # Use shared staging tables from staging_search_expander
        if staging_tables:
            self._staging_tables = staging_tables
        else:
            self._staging_tables = AJCCStagingTables()
        
        self._staging_tables.load()
    
    def extract_staging_from_text(self, text: str) -> List[StagingInfo]:
        """
        Extract staging information from text using regex patterns.
        
        Args:
            text: Text to extract staging from
            
        Returns:
            List of StagingInfo objects found in text
        """
        staging_list = []
        
        # Try to find complete TNM staging first
        for match in self.TNM_PATTERNS["full_tnm"].finditer(text):
            prefix = match.group(1) or ""
            staging_type = self._parse_prefix(prefix)
            
            staging = StagingInfo(
                t_stage=match.group(2).lower(),
                n_stage=match.group(4).lower(),
                m_stage=match.group(6).lower(),
                staging_type=staging_type,
                confidence=0.9,
                source_text=match.group(0)
            )
            staging_list.append(staging)
        
        # If no complete TNM found, try to extract individual components
        if not staging_list:
            staging = StagingInfo()
            
            t_match = self.TNM_PATTERNS["t_stage"].search(text)
            if t_match:
                prefix = t_match.group(1) or ""
                staging.t_stage = t_match.group(2).lower()
                staging.staging_type = self._parse_prefix(prefix)
                staging.confidence += 0.3
            
            n_match = self.TNM_PATTERNS["n_stage"].search(text)
            if n_match:
                staging.n_stage = n_match.group(2).lower()
                staging.confidence += 0.3
            
            m_match = self.TNM_PATTERNS["m_stage"].search(text)
            if m_match:
                staging.m_stage = m_match.group(2).lower()
                staging.confidence += 0.3
            
            if staging.t_stage or staging.n_stage or staging.m_stage:
                staging_list.append(staging)
        
        # Also extract stage group mentions
        for match in self.TNM_PATTERNS["stage_group"].finditer(text):
            stage_num = match.group(1).upper()
            stage_suffix = (match.group(2) or "").upper()
            
            # Convert Arabic to Roman
            roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV', '0': '0'}
            stage_num = roman_map.get(stage_num, stage_num)
            
            stage_group = f"{stage_num}{stage_suffix}"
            
            if staging_list:
                staging_list[0].stage_group = stage_group
            else:
                staging = StagingInfo(
                    stage_group=stage_group,
                    confidence=0.5,
                    source_text=match.group(0)
                )
                staging_list.append(staging)
        
        return staging_list
    
    def _parse_prefix(self, prefix: str) -> str:
        """Parse staging prefix to determine type."""
        prefix = prefix.lower()
        if 'yp' in prefix:
            return 'yp'
        elif 'p' in prefix:
            return 'pathologic'
        elif 'c' in prefix:
            return 'clinical'
        return 'unknown'
    
    def lookup_stage_group(
        self, 
        cancer_type: str, 
        t_stage: str, 
        n_stage: str, 
        m_stage: str
    ) -> Optional[str]:
        """
        Look up stage group from TNM using AJCC 8th edition tables.
        
        Args:
            cancer_type: Cancer type (e.g., "breast", "lung")
            t_stage: T stage value (e.g., "2", "2a")
            n_stage: N stage value (e.g., "1", "1mi")
            m_stage: M stage value (e.g., "0", "1a")
            
        Returns:
            Stage group string (e.g., "IIB") or None if not found
        """
        stage_group, confidence, alternatives = self._staging_tables.lookup_stage_group(
            cancer_type, t_stage, n_stage, m_stage
        )
        return stage_group
    
    def lookup_stage_group_with_confidence(
        self,
        cancer_type: str,
        t_stage: str,
        n_stage: str,
        m_stage: str
    ) -> Tuple[Optional[str], float, List[str]]:
        """
        Look up stage group with confidence and alternatives.
        
        Args:
            cancer_type: Cancer type
            t_stage: T stage value
            n_stage: N stage value
            m_stage: M stage value
            
        Returns:
            Tuple of (stage_group, confidence, alternative_stages)
        """
        return self._staging_tables.lookup_stage_group(
            cancer_type, t_stage, n_stage, m_stage
        )
    
    def infer_cancer_type(self, text: str) -> Optional[str]:
        """
        Infer cancer type from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Canonical cancer type name or None
        """
        return self._staging_tables.resolve_cancer_type(text)
    
    def normalize_staging_in_context(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        cancer_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normalize staging information across retrieved chunks.
        
        This method:
        1. Extracts staging from query and chunks
        2. Resolves conflicts between different staging notations
        3. Enriches staging with inferred stage groups
        
        Args:
            chunks: List of retrieved document chunks
            query: User's query
            cancer_type: Optional cancer type hint
            
        Returns:
            Dict with normalized staging context
        """
        # Extract staging from query
        query_staging = self.extract_staging_from_text(query)
        
        # Infer cancer type if not provided
        if not cancer_type:
            cancer_type = self.infer_cancer_type(query)
            for chunk in chunks:
                if not cancer_type:
                    chunk_text = chunk.get("text", "") or chunk.get("content", "")
                    cancer_type = self.infer_cancer_type(chunk_text)
                    if cancer_type:
                        break
        
        # Extract staging from chunks
        chunk_staging = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "") or chunk.get("content", "")
            staging_list = self.extract_staging_from_text(chunk_text)
            for s in staging_list:
                s.source_text = chunk.get("doc_id", "unknown")
            chunk_staging.extend(staging_list)
        
        # Enrich with stage group lookups
        all_staging = query_staging + chunk_staging
        for staging in all_staging:
            if staging.is_complete() and not staging.stage_group and cancer_type:
                staging.stage_group = self.lookup_stage_group(
                    cancer_type,
                    staging.t_stage,
                    staging.n_stage,
                    staging.m_stage
                )
        
        # Build normalized context
        result = {
            "cancer_type": cancer_type,
            "query_staging": [s.to_dict() for s in query_staging],
            "chunk_staging": [s.to_dict() for s in chunk_staging],
            "primary_staging": None,
            "staging_conflicts": [],
        }
        
        # Determine primary staging (prefer query staging, then highest confidence)
        if query_staging:
            result["primary_staging"] = query_staging[0].to_dict()
        elif chunk_staging:
            best = max(chunk_staging, key=lambda s: s.confidence)
            result["primary_staging"] = best.to_dict()
        
        # Detect conflicts
        stage_groups = set()
        for s in all_staging:
            if s.stage_group:
                stage_groups.add(s.stage_group)
        
        if len(stage_groups) > 1:
            result["staging_conflicts"] = list(stage_groups)
        
        return result
    
    def format_staging_context(self, staging: StagingInfo) -> str:
        """
        Format staging information for inclusion in prompt context.
        
        Args:
            staging: StagingInfo object
            
        Returns:
            Formatted string for prompt
        """
        if not staging.t_stage and not staging.n_stage and not staging.m_stage and not staging.stage_group:
            return ""
        
        parts = []
        
        if staging.is_complete():
            tnm = staging.to_tnm_string()
            parts.append(f"TNM Staging: {tnm}")
        else:
            if staging.t_stage:
                parts.append(f"T stage: T{staging.t_stage}")
            if staging.n_stage:
                parts.append(f"N stage: N{staging.n_stage}")
            if staging.m_stage:
                parts.append(f"M stage: M{staging.m_stage}")
        
        if staging.stage_group:
            parts.append(f"Stage Group: {staging.stage_group}")
        
        if staging.staging_type and staging.staging_type != "unknown":
            parts.append(f"Staging Type: {staging.staging_type.capitalize()}")
        
        return " | ".join(parts)
    
    def get_staging_aware_prompt_addition(self, staging_context: str) -> str:
        """
        Generate prompt addition for staging-aware answer generation.
        
        Args:
            staging_context: Formatted staging context string
            
        Returns:
            Prompt addition string
        """
        if not staging_context:
            return ""
        
        return f"""
IMPORTANT STAGING CONTEXT:
{staging_context}

When answering, ensure your response is consistent with this staging information.
If the evidence discusses different stages, prioritize information relevant to the staging above.
"""
    
    def get_supported_cancer_types(self) -> List[str]:
        """Get list of cancer types with staging tables."""
        return self._staging_tables.get_cancer_types()
    
    def get_t_definition(self, cancer_type: str, t_stage: str) -> Optional[str]:
        """
        Get the clinical definition for a T stage.
        
        Args:
            cancer_type: Cancer type
            t_stage: T stage value
            
        Returns:
            Definition string or None
        """
        return self._staging_tables.get_t_definition(cancer_type, t_stage)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_normalizer: Optional[StagingNormalizer] = None


def get_staging_normalizer(openai_client: Optional[Any] = None) -> StagingNormalizer:
    """
    Get singleton instance of the staging normalizer.
    
    Args:
        openai_client: Optional OpenAI client
        
    Returns:
        StagingNormalizer instance
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = StagingNormalizer(openai_client)
    return _normalizer


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    normalizer = StagingNormalizer()
    
    print("=" * 70)
    print("STAGING NORMALIZER DEMO")
    print(f"Supported cancer types: {len(normalizer.get_supported_cancer_types())}")
    print("=" * 70)
    
    test_texts = [
        "Patient with cT3N1M0 breast cancer",
        "Stage IIIA disease with T2N2M0",
        "pT1cN1mi cM0 ER+ HER2- breast cancer",
        "Stage I testicular seminoma",
        "ypT0N0M0 rectal cancer after neoadjuvant chemoradiation",
        "T4N2M0 laryngeal squamous cell carcinoma",
        "Stage IVB lung adenocarcinoma with brain metastases",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        staging_list = normalizer.extract_staging_from_text(text)
        
        # Infer cancer type
        cancer_type = normalizer.infer_cancer_type(text)
        print(f"  Cancer type: {cancer_type or 'Unknown'}")
        
        for s in staging_list:
            print(f"  TNM: {s.to_tnm_string()}")
            print(f"  Stage Group: {s.stage_group}")
            print(f"  Type: {s.staging_type}")
            print(f"  Confidence: {s.confidence:.2f}")
            
            # Try to lookup stage group if not present
            if s.is_complete() and not s.stage_group and cancer_type:
                inferred_stage, conf, alts = normalizer.lookup_stage_group_with_confidence(
                    cancer_type, s.t_stage, s.n_stage, s.m_stage
                )
                if inferred_stage:
                    print(f"  Inferred Stage: {inferred_stage} (confidence: {conf:.0%})")
                    if alts:
                        print(f"  Alternatives: {', '.join(alts)}")
    
    print("\n" + "=" * 70)
    print("END DEMO")
    print("=" * 70)