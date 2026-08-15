"""
Smart Search API Routes

Intelligent search combining user preferences (persistent filters) 
with case context (relevance matching).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.api.services.auth_dependencies import get_current_user, get_current_user_optional
from src.api.services.smart_search_service import (
    get_smart_search_service, 
    CaseContext
)
from src.api.services.query_classifier_service import get_query_classifier_service
from src.api.routes.user_preferences import get_preferences

router = APIRouter(prefix="/smart-search", tags=["smart-search"])


class SmartSearchRequest(BaseModel):
    """Request for smart search"""
    query: str
    top_k: int = 20
    use_preferences: bool = True  # Apply user preferences as filters
    use_case_context: bool = True  # Extract and use case context for relevance


class CaseContextResponse(BaseModel):
    """Case context extracted from query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    cancer_type: Optional[str] = None
    cancer_location: Optional[str] = None
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    overall_stage: Optional[str] = None
    treatment_setting: Optional[str] = None
    molecular_subtype: Optional[str] = None
    hpv_status: Optional[str] = None
    performance_status_ecog: Optional[int] = None
    prior_surgery: List[str] = []
    prior_radiation_completed: bool = False
    recurrence_status: Optional[str] = None


class SearchResult(BaseModel):
    """A single search result"""
    study_id: int
    study_name: Optional[str] = None
    cancer_type: Optional[str] = None
    study_type: Optional[str] = None
    number_of_patients: Optional[int] = None
    vector_score: float = 0.0
    relevance_score: float = 0.0
    combined_score: float = 0.0
    patient_relevance_score: float = 0.0  # Score 0-100 for patient-specific matching
    citation_count: Optional[int] = None
    publish_date: Optional[str] = None


class SmartSearchResponse(BaseModel):
    """Response from smart search"""
    results: List[SearchResult]
    total_matching_studies: int
    applied_preferences: Dict[str, Any]
    case_context: CaseContextResponse
    sort_by: str
    explanation: str


def _structured_to_case_context(structured) -> CaseContext:
    """Convert StructuredQuery to CaseContext"""
    return CaseContext(
        age=structured.age,
        gender=structured.gender,
        race_ethnicity=structured.race_ethnicity,
        cancer_type=structured.cancer_type,
        cancer_location=structured.cancer_location,
        histopathologic_type=structured.histopathologic_type,
        tumor_grade=structured.tumor_grade,
        tnm_t=structured.tnm_t,
        tnm_n=structured.tnm_n,
        tnm_m=structured.tnm_m,
        overall_stage=structured.overall_stage,
        metastatic_status=structured.metastatic_status,
        molecular_subtype=structured.molecular_subtype,
        prior_surgery=structured.prior_surgery if structured.prior_surgery else [],
        prior_radiation_completed=structured.prior_radiation or False,
        performance_status_ecog=structured.performance_status,
        smoking_status=structured.smoking_status,
        comorbidities=structured.comorbidities if structured.comorbidities else [],
        treatment_setting=structured.recurrence_status,  # Map recurrence to treatment setting
    )


def _case_context_to_response(ctx: CaseContext) -> CaseContextResponse:
    """Convert CaseContext to response model"""
    return CaseContextResponse(
        age=ctx.age,
        gender=ctx.gender,
        cancer_type=ctx.cancer_type,
        cancer_location=ctx.cancer_location,
        tnm_t=ctx.tnm_t,
        tnm_n=ctx.tnm_n,
        tnm_m=ctx.tnm_m,
        overall_stage=ctx.overall_stage,
        treatment_setting=ctx.treatment_setting,
        molecular_subtype=ctx.molecular_subtype,
        hpv_status=ctx.hpv_status,
        performance_status_ecog=ctx.performance_status_ecog,
        prior_surgery=ctx.prior_surgery,
        prior_radiation_completed=ctx.prior_radiation_completed,
    )


def _generate_explanation(
    preferences: Dict[str, Any], 
    case_context: CaseContext,
    total_matching: int
) -> str:
    """Generate human-readable explanation of search logic"""
    parts = []
    
    # Preference filters applied
    filters = []
    if preferences.get("study_types"):
        filters.append(f"study types: {', '.join(preferences['study_types'])}")
    if preferences.get("study_phases"):
        filters.append(f"phases: {', '.join(preferences['study_phases'])}")
    if preferences.get("min_patients"):
        filters.append(f"≥{preferences['min_patients']} patients")
    if preferences.get("countries"):
        filters.append(f"countries: {', '.join(preferences['countries'][:3])}")
    if preferences.get("min_publication_year"):
        filters.append(f"published ≥{preferences['min_publication_year']}")
    
    if filters:
        parts.append(f"Filtered to {total_matching} studies matching: {'; '.join(filters)}")
    else:
        parts.append(f"Searching across {total_matching} studies (no preference filters)")
    
    # Case context used for relevance
    context_parts = []
    if case_context.age:
        context_parts.append(f"{case_context.age}yo")
    if case_context.gender:
        context_parts.append(case_context.gender)
    if case_context.cancer_type:
        context_parts.append(case_context.cancer_type)
    if case_context.tnm_t or case_context.tnm_n:
        stage = f"{case_context.tnm_t or ''}{case_context.tnm_n or ''}"
        context_parts.append(stage)
    if case_context.treatment_setting:
        context_parts.append(f"{case_context.treatment_setting} setting")
    
    if context_parts:
        parts.append(f"Ranked by relevance to: {', '.join(context_parts)}")
    
    return ". ".join(parts)


@router.post("", response_model=SmartSearchResponse)
async def smart_search(
    request: SmartSearchRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Intelligent search combining:
    1. User's standing preferences (persistent filters)
    2. Extracted case context (relevance matching)
    
    Flow:
    1. Load user preferences → filter PostgreSQL studies
    2. Extract case context from query → build relevance scoring
    3. Search Qdrant within filtered studies
    4. Re-rank by case similarity
    5. Sort by user preference (relevance/population/date/citations/outcomes)
    """
    service = get_smart_search_service()
    
    # Step 1: Get user preferences
    preferences = {}
    if request.use_preferences and current_user:
        from src.api.routes.user_preferences import get_preferences as get_prefs_route
        prefs = await get_prefs_route(current_user)
        preferences = prefs.dict() if hasattr(prefs, 'dict') else dict(prefs)
    
    # Step 2: Extract case context from query
    case_context = CaseContext()
    if request.use_case_context:
        try:
            classifier = get_query_classifier_service()
            structured = classifier.classify_query(request.query)
            case_context = _structured_to_case_context(structured)
        except Exception as e:
            # If classification fails, continue with empty context
            pass
    
    # Step 3: Get matching study IDs from PostgreSQL (preference filtering)
    matching_study_ids = await service.get_matching_study_ids(preferences)
    
    # Step 4: For now, get studies directly from PostgreSQL
    # In full implementation, this would search Qdrant with allowed_study_ids filter
    from src.api.services.smart_search_service import _get_study_pool
    pool = await _get_study_pool()
    
    async with pool.acquire() as conn:
        if matching_study_ids:
            # Get studies matching preferences
            placeholders = ", ".join(f"${i+1}" for i in range(len(matching_study_ids[:100])))
            rows = await conn.fetch(f"""
                SELECT 
                    study_id, study_name, study_type, study_phase,
                    cancer_type, cancer_location, number_of_patients,
                    median_age, gender_distribution, performance_status,
                    staging_system_used, molecular_subtype, publish_date,
                    overall_survival, progression_free_survival,
                    citation_count
                FROM studies
                WHERE study_id IN ({placeholders})
                LIMIT {request.top_k * 2}
            """, *matching_study_ids[:100])
        else:
            rows = []
    
    # Convert to results format
    results = [
        {
            "study_id": row["study_id"],
            "score": 0.5,  # Default vector score
            "study_metadata": dict(row)
        }
        for row in rows
    ]
    
    # Step 5: Re-rank by case similarity
    if request.use_case_context:
        results = await service.rerank_results(results, case_context)
    
    # Step 6: Sort by user preference
    sort_by = preferences.get("sort_by", "relevance")
    results = await service.sort_by_preference(results, sort_by, case_context)
    
    # Limit to top_k
    results = results[:request.top_k]
    
    # Build response
    search_results = []
    for r in results:
        meta = r.get("study_metadata", {})
        search_results.append(SearchResult(
            study_id=r["study_id"],
            study_name=meta.get("study_name"),
            cancer_type=meta.get("cancer_type"),
            study_type=meta.get("study_type"),
            number_of_patients=meta.get("number_of_patients"),
            vector_score=r.get("vector_score", 0),
            relevance_score=r.get("relevance_score", 0),
            combined_score=r.get("combined_score", 0),
            patient_relevance_score=r.get("patient_relevance_score", 0),
            citation_count=meta.get("citation_count"),
            publish_date=str(meta["publish_date"]) if meta.get("publish_date") else None
        ))
    
    explanation = _generate_explanation(preferences, case_context, len(matching_study_ids))
    
    return SmartSearchResponse(
        results=search_results,
        total_matching_studies=len(matching_study_ids),
        applied_preferences=preferences,
        case_context=_case_context_to_response(case_context),
        sort_by=sort_by,
        explanation=explanation
    )


@router.post("/extract-context", response_model=CaseContextResponse)
async def extract_case_context(query: str):
    """
    Extract case context from a clinical query without performing search.
    Useful for previewing what the system understood from the query.
    """
    try:
        classifier = get_query_classifier_service()
        structured = classifier.classify_query(query)
        case_context = _structured_to_case_context(structured)
        return _case_context_to_response(case_context)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract case context: {str(e)}"
        )


@router.get("/preview-filters")
async def preview_filters(
    current_user: dict = Depends(get_current_user)
):
    """
    Preview how many studies match current user preferences.
    Useful for showing filter impact before searching.
    """
    from src.api.routes.user_preferences import get_preferences as get_prefs_route
    
    prefs = await get_prefs_route(current_user)
    preferences = prefs.dict() if hasattr(prefs, 'dict') else dict(prefs)
    
    service = get_smart_search_service()
    matching_ids = await service.get_matching_study_ids(preferences)
    
    # Get total studies for comparison
    from src.api.services.smart_search_service import _get_study_pool
    pool = await _get_study_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM studies")
    
    return {
        "total_studies": total,
        "matching_studies": len(matching_ids),
        "filter_impact": f"{len(matching_ids)}/{total} studies match your preferences",
        "preferences_applied": {
            k: v for k, v in preferences.items() 
            if v and k not in ["filters_active", "sort_by", "sort_order", "results_per_page"]
        }
    }
