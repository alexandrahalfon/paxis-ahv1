"""
RAG Pre-Filter Service

Builds Qdrant filter conditions from clinical profile for Q&A mode.
Uses category payload field for filtering since search_terms is not yet populated in Qdrant.

PostgreSQL is only used for user preference filtering (post-retrieval).
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from qdrant_client import models as qm

logger = logging.getLogger(__name__)


@dataclass
class PreFilterResult:
    """Result from the clinical prefilter."""
    search_terms: List[str] = field(default_factory=list)
    category: Optional[str] = None
    qdrant_filter: Optional[qm.Filter] = None
    filter_applied: bool = False
    filter_reason: str = ""
    timing_ms: float = 0.0
    clinical_context_used: Dict[str, Any] = field(default_factory=dict)
    
    def has_filter(self) -> bool:
        """Check if filter was applied."""
        return self.filter_applied and self.qdrant_filter is not None


def is_prefilter_enabled() -> bool:
    """Check if prefiltering is enabled via environment variable."""
    return os.getenv("ENABLE_RAG_PREFILTER", "false").lower() in ("true", "1", "yes")


# Map cancer types to Qdrant category values
# NOTE: Category names must match exactly what's in Qdrant (case-sensitive)
# Qdrant categories: GI_processed_documents, GU_processed_documents, h&n_processed_documents, etc.
CANCER_TYPE_TO_CATEGORY = {
    'breast': 'breast_processed_documents',
    'lung': 'lung_processed_documents',
    'nsclc': 'lung_processed_documents',
    'sclc': 'lung_processed_documents',
    'prostate': 'prostate_processed_documents',
    'gi': 'GI_processed_documents',
    'colorectal': 'GI_processed_documents',
    'rectal': 'GI_processed_documents',
    'colon': 'GI_processed_documents',
    'esophageal': 'GI_processed_documents',
    'gastric': 'GI_processed_documents',
    'pancreatic': 'GI_processed_documents',
    'liver': 'GI_processed_documents',
    'hepatocellular': 'GI_processed_documents',
    'brain': 'cns_processed_documents',
    'cns': 'cns_processed_documents',
    'glioma': 'cns_processed_documents',
    'glioblastoma': 'cns_processed_documents',
    'gbm': 'cns_processed_documents',
    'head and neck': 'h&n_processed_documents',
    'head_and_neck': 'h&n_processed_documents',
    'h&n': 'h&n_processed_documents',
    'oropharyngeal': 'h&n_processed_documents',
    'laryngeal': 'h&n_processed_documents',
    'nasopharyngeal': 'h&n_processed_documents',
    'oral': 'h&n_processed_documents',
    'gyn': 'gyn_processed_documents',
    'cervical': 'gyn_processed_documents',
    'endometrial': 'gyn_processed_documents',
    'ovarian': 'gyn_processed_documents',
    'uterine': 'gyn_processed_documents',
    'vulvar': 'gyn_processed_documents',
    'gu': 'GU_processed_documents',
    'bladder': 'GU_processed_documents',
    'renal': 'GU_processed_documents',
    'kidney': 'GU_processed_documents',
    'testicular': 'GU_processed_documents',
    'seminoma': 'GU_processed_documents',
    'lymphoma': 'lymphoma_processed_documents',
    'hodgkin': 'lymphoma_processed_documents',
    'non-hodgkin': 'lymphoma_processed_documents',
    'pediatric': 'peds_processed_documents',
    'peds': 'peds_processed_documents',
    'melanoma': 'cutaneous_processed_documents',
    'skin': 'cutaneous_processed_documents',
    'cutaneous': 'cutaneous_processed_documents',
    'sarcoma': 'sarcoma_processed_documents',
    'bone': 'sarcoma_processed_documents',
    'soft tissue': 'sarcoma_processed_documents',
}


def get_category_from_cancer_type(cancer_type: str) -> Optional[str]:
    """Map cancer type to Qdrant category."""
    if not cancer_type:
        return None
    
    ctype = cancer_type.lower().strip()
    return CANCER_TYPE_TO_CATEGORY.get(ctype)


def build_qdrant_filter_from_clinical_profile(
    clinical_profile: Dict[str, Any],
    category: Optional[str] = None,
) -> PreFilterResult:
    """
    Build a Qdrant filter from clinical profile for Q&A mode.
    
    Uses category filter since search_terms is not yet populated in Qdrant.
    
    Args:
        clinical_profile: The clinical_profile dict from clinical_entity_extractor
        category: Optional category filter override
        
    Returns:
        PreFilterResult with qdrant_filter for vector search
    """
    start_time = time.perf_counter()
    result = PreFilterResult()
    
    if not clinical_profile:
        result.filter_reason = "no_clinical_profile"
        result.timing_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    # Extract fields from clinical_profile
    raw_profile = clinical_profile.get("raw_profile", {})
    
    cancer_type = raw_profile.get("cancer_type")
    cancer_subtype = raw_profile.get("cancer_subtype")
    anatomic_sites = raw_profile.get("anatomic_sites", [])
    
    # If no cancer_type but have subtype, use subtype
    if not cancer_type and cancer_subtype:
        cancer_type = cancer_subtype
    
    # Store what we're using
    result.clinical_context_used = {
        "cancer_type": cancer_type,
        "cancer_subtype": cancer_subtype,
        "anatomic_sites": anatomic_sites,
    }
    
    # Determine category to filter on
    filter_category = category  # Use provided category first
    
    if not filter_category and cancer_type:
        # Try to map cancer type to category
        filter_category = get_category_from_cancer_type(cancer_type)
    
    if not filter_category and cancer_subtype:
        # Try subtype
        filter_category = get_category_from_cancer_type(cancer_subtype)
    
    result.category = filter_category
    result.clinical_context_used["mapped_category"] = filter_category
    
    # Build Qdrant filter
    if filter_category:
        result.qdrant_filter = qm.Filter(must=[
            qm.FieldCondition(
                key="category",
                match=qm.MatchValue(value=filter_category)
            )
        ])
        result.filter_applied = True
        result.filter_reason = "category_filter"
        print(f"[RAGPrefilter] Category filter: {filter_category}")
    else:
        result.filter_reason = "no_category_match"
        print(f"[RAGPrefilter] No category match for cancer_type={cancer_type}")
    
    result.timing_ms = (time.perf_counter() - start_time) * 1000
    
    return result


def get_category_filter(category: str) -> Optional[qm.Filter]:
    """Get a simple category filter for Qdrant."""
    if not category:
        return None
    
    return qm.Filter(must=[
        qm.FieldCondition(
            key="category",
            match=qm.MatchValue(value=category)
        )
    ])
