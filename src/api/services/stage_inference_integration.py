"""
INTEGRATION GUIDE: Stage Inference in Query Classifier
=======================================================

This file shows the exact changes needed in query_classifier_service.py
to add implicit stage inference after LLM extraction.

FILE TO MODIFY: src/api/services/query_classifier_service.py
LOCATION: Inside the classify_query() method, AFTER the LLM extracts
          structured data and populates the StructuredQuery object.

Find the section where tnm_t, tnm_n, tnm_m, and overall_stage are set,
then add the stage inference call right after.
"""

# =============================================================================
# STEP 1: Add import at the top of query_classifier_service.py
# =============================================================================

# Add this import near the other imports:
# 
#   from src.api.services.stage_inference_service import infer_stage_for_query
#


# =============================================================================
# STEP 2: Add these fields to the StructuredQuery dataclass
# =============================================================================

# Add after the existing staging fields (overall_stage, risk_stratification, etc.):
#
#   # Stage inference metadata (populated by stage_inference_service)
#   stage_inferred: bool = False                    # True if stage was inferred (not extracted)
#   stage_ambiguous: bool = False                   # True if multiple stages are possible
#   stage_possible_stages: List[str] = field(default_factory=list)  # All possible stages if ambiguous
#   stage_required_factors: List[str] = field(default_factory=list) # What's needed to resolve
#   stage_inference_notes: List[str] = field(default_factory=list)  # Human-readable notes
#   stage_confidence: str = ""                      # "high", "medium", "low"
#


# =============================================================================
# STEP 3: Add stage inference call in classify_query() method
# =============================================================================

# Find the section in classify_query() that ends with populating staging fields.
# It looks approximately like this:
#
#   if data.get("metastatic_status"):
#       query.metastatic_status = str(data["metastatic_status"]).lower().strip()
#
# RIGHT AFTER that block, add the following:


def _example_integration_in_classify_query(query, data):
    """
    This function shows the code to ADD inside classify_query().
    Copy the contents (not the function itself) into classify_query().
    """
    
    # =====================================================
    # STAGE INFERENCE: Infer stage if not explicitly provided
    # =====================================================
    if not query.overall_stage and (query.tnm_t or query.tnm_n or query.tnm_m or query.metastatic_status):
        try:
            from src.api.services.stage_inference_service import infer_stage_for_query
            
            inference = infer_stage_for_query(
                cancer_type=query.cancer_type,
                cancer_location=query.cancer_location,
                tnm_t=query.tnm_t,
                tnm_n=query.tnm_n,
                tnm_m=query.tnm_m,
                metastatic_status=query.metastatic_status,
                age=query.age,
                hpv_status=None,  # Not in StructuredQuery yet; add if needed
                histopathologic_type=query.histopathologic_type,
                molecular_subtype=query.molecular_subtype,
                tumor_grade=query.tumor_grade,
            )
            
            if inference.stage_group and not inference.is_ambiguous:
                query.overall_stage = inference.stage_group
                query.stage_inferred = True
                query.stage_confidence = inference.confidence
                logger.info(
                    f"[StageInference] Inferred stage {inference.stage_group} "
                    f"from {inference.tnm_used} ({inference.source}, {inference.confidence})"
                )
            elif inference.is_ambiguous:
                # Set the anatomic stage as the best guess, but flag ambiguity
                if inference.stage_group:
                    query.overall_stage = inference.stage_group
                query.stage_inferred = True
                query.stage_ambiguous = True
                query.stage_possible_stages = inference.possible_stages
                query.stage_required_factors = inference.required_factors
                query.stage_inference_notes = inference.notes
                query.stage_confidence = inference.confidence
                logger.info(
                    f"[StageInference] Ambiguous stage for {inference.tnm_used}: "
                    f"possible={inference.possible_stages}, needs={inference.required_factors}"
                )
            
            # Also infer metastatic_status if not set
            if not query.metastatic_status and inference.metastatic_status:
                query.metastatic_status = inference.metastatic_status
                
        except Exception as e:
            logger.warning(f"[StageInference] Failed to infer stage: {e}")
    
    return query  # Only needed for this example function


# =============================================================================
# STEP 4: Update get_search_summary() to show inferred stage
# =============================================================================

# In the get_search_summary() method, the existing code already handles
# overall_stage, so inferred stages will automatically appear in summaries.
# Optionally, you can add a marker:
#
#   if self.overall_stage:
#       stage_str = f"stage {self.overall_stage}"
#       if self.stage_inferred:
#           stage_str += " (inferred)"
#       parts.append(stage_str)


# =============================================================================
# STEP 5: Update to_dict() to include inference metadata
# =============================================================================

# The existing to_dict() already includes all non-None fields,
# so the new fields will automatically be included when set.
# No changes needed.


# =============================================================================
# STEP 6 (OPTIONAL): Update staging_normalizer.py to use shared tables
# =============================================================================

# In staging_normalizer.py, replace the inline AJCC_STAGING_TABLES with:
#
#   from src.api.services.stage_inference_service import get_stage_inference_service
#
#   class StagingNormalizer:
#       def __init__(self, openai_client=None):
#           self._stage_service = get_stage_inference_service()
#           ...
#
#       def lookup_stage_group(self, cancer_type, t_stage, n_stage, m_stage):
#           result = self._stage_service.infer_stage(
#               cancer_type=cancer_type,
#               tnm_t=f"T{t_stage}",
#               tnm_n=f"N{n_stage}",
#               tnm_m=f"M{m_stage}",
#           )
#           return result.stage_group
#
# This eliminates the duplicate staging tables and ensures consistency.
