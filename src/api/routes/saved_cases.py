"""
Saved Cases API Routes

Provides endpoints for saving, retrieving, and managing patient cases.
Users can save classified clinical queries and reuse them for future searches.
Includes support for saving full responses with sources and enabling alerts.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.api.services.auth_dependencies import require_physician
from src.api.services.saved_cases_service import get_saved_cases_service
from src.api.services.query_classifier_service import get_query_classifier_service

router = APIRouter(prefix="/saved-cases", tags=["saved-cases"])


class SaveCaseRequest(BaseModel):
    """Request to save a new case"""
    query: str
    case_name: Optional[str] = None
    # Optional full response data
    response_answer: Optional[str] = None
    response_sources: Optional[List[Dict[str, Any]]] = None
    response_metadata: Optional[Dict[str, Any]] = None


class SaveCaseResponse(BaseModel):
    """Response after saving a case"""
    success: bool
    case_id: int
    case_name: str
    query_summary: str
    message: str


class SavedCase(BaseModel):
    """A saved patient case"""
    id: int
    case_name: str
    original_query: str
    query_summary: str
    structured_data: Dict[str, Any]
    demographics: Dict[str, Any]
    diagnosis: Dict[str, Any]
    staging: Dict[str, Any]
    pathology: Dict[str, Any]
    treatment_history: Dict[str, Any]
    risk_factors: Dict[str, Any]
    response_answer: Optional[str] = None
    response_sources: Optional[List[Dict[str, Any]]] = None
    response_metadata: Optional[Dict[str, Any]] = None
    use_count: int
    last_used_at: Optional[str]
    created_at: str
    is_archived: bool


class SavedCasesListResponse(BaseModel):
    """Response with list of saved cases"""
    success: bool
    total: int
    cases: List[SavedCase]


class UpdateCaseRequest(BaseModel):
    """Request to update a case"""
    case_name: Optional[str] = None
    is_archived: Optional[bool] = None


class UseCaseResponse(BaseModel):
    """Response when using a case for a query"""
    success: bool
    case: SavedCase
    context_prompt: str  # Formatted context to pass to the chat


# Alert-related models
class EnableAlertRequest(BaseModel):
    """Request to enable alerts for a case"""
    email_notifications: bool = True


class AlertSettings(BaseModel):
    """Alert settings for a case"""
    alert_id: int
    case_id: int
    case_name: Optional[str] = None
    query_summary: Optional[str] = None
    alerts_enabled: bool
    email_notifications: bool
    new_matches_count: int = 0
    last_checked_at: Optional[str] = None
    last_alert_sent_at: Optional[str] = None
    created_at: Optional[str] = None


class AlertMatch(BaseModel):
    """A matching trial found by an alert"""
    doc_id: str
    title: str
    doi: Optional[str] = None
    match_score: float
    match_reasons: List[str]
    found_at: str
    viewed: bool


@router.post("", response_model=SaveCaseResponse)
async def save_case(
    request: SaveCaseRequest,
    current_user: dict = Depends(require_physician)
):
    """
    Save a classified patient case with optional full response.
    
    Takes a free-text clinical description, classifies it, and saves
    the structured data for future use. Can also save the full AI response
    with sources for reference.
    
    Example:
    ```json
    {
        "query": "68 year old female with SCC of maxilla, pT4N0...",
        "case_name": "Mrs. Smith - Oral SCC",
        "response_answer": "Based on the clinical presentation...",
        "response_sources": [{"title": "Study 1", "doi": "10.1234/..."}]
    }
    ```
    """
    print(f"[SaveCase] Received request:")
    print(f"[SaveCase]   query length: {len(request.query)}")
    print(f"[SaveCase]   case_name: {request.case_name}")
    print(f"[SaveCase]   response_answer: {len(request.response_answer) if request.response_answer else 'None'}")
    print(f"[SaveCase]   response_sources: {len(request.response_sources) if request.response_sources else 'None'}")
    print(f"[SaveCase]   response_metadata: {request.response_metadata}")
    
    if not request.query or len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 10 characters"
        )
    
    try:
        # Classify the query
        classifier = get_query_classifier_service()
        structured = classifier.classify_query(request.query)
        
        # Organize fields by category (same as classify endpoint)
        demographics = {}
        if structured.age:
            demographics["age"] = structured.age
        if structured.gender:
            demographics["gender"] = structured.gender
        if structured.race_ethnicity:
            demographics["race_ethnicity"] = structured.race_ethnicity
        if structured.performance_status:
            demographics["performance_status"] = f"ECOG {structured.performance_status}"
        
        diagnosis = {}
        if structured.cancer_type:
            diagnosis["cancer_type"] = structured.cancer_type
        if structured.cancer_location:
            diagnosis["cancer_location"] = structured.cancer_location
        if structured.histopathologic_type:
            diagnosis["histopathologic_type"] = structured.histopathologic_type
        if structured.tumor_grade:
            diagnosis["tumor_grade"] = structured.tumor_grade
        if structured.molecular_subtype:
            diagnosis["molecular_subtype"] = structured.molecular_subtype
        
        staging = {}
        if structured.tnm_t:
            staging["T_stage"] = structured.tnm_t
        if structured.tnm_n:
            staging["N_stage"] = structured.tnm_n
        if structured.tnm_m:
            staging["M_stage"] = structured.tnm_m
        if structured.overall_stage:
            staging["overall_stage"] = structured.overall_stage
        if structured.risk_stratification:
            staging["risk_stratification"] = structured.risk_stratification
        if structured.metastatic_status:
            staging["metastatic_status"] = structured.metastatic_status
        
        pathology = {}
        if structured.depth_of_invasion:
            pathology["depth_of_invasion"] = structured.depth_of_invasion
        if structured.lymphovascular_invasion:
            pathology["lymphovascular_invasion"] = structured.lymphovascular_invasion
        if structured.perineural_invasion:
            pathology["perineural_invasion"] = structured.perineural_invasion
        if structured.margin_status:
            pathology["margin_status"] = structured.margin_status
        if structured.lymph_nodes_examined is not None:
            pathology["lymph_nodes_examined"] = structured.lymph_nodes_examined
        if structured.lymph_nodes_positive is not None:
            pathology["lymph_nodes_positive"] = structured.lymph_nodes_positive
        
        treatment_history = {}
        if structured.prior_surgery:
            treatment_history["prior_surgery"] = structured.prior_surgery
        if structured.prior_radiation is not None:
            treatment_history["prior_radiation"] = structured.prior_radiation
        if structured.prior_chemotherapy is not None:
            treatment_history["prior_chemotherapy"] = structured.prior_chemotherapy
        if structured.recurrence_status:
            treatment_history["recurrence_status"] = structured.recurrence_status
        
        risk_factors = {}
        if structured.smoking_status:
            risk_factors["smoking_status"] = structured.smoking_status
        if structured.comorbidities:
            risk_factors["comorbidities"] = structured.comorbidities
        
        # Save the case
        service = get_saved_cases_service()
        result = await service.save_case(
            user_id=current_user["id"],
            original_query=request.query,
            structured_query=structured,
            case_name=request.case_name,
            demographics=demographics,
            diagnosis=diagnosis,
            staging=staging,
            pathology=pathology,
            treatment_history=treatment_history,
            risk_factors=risk_factors,
            response_answer=request.response_answer,
            response_sources=request.response_sources,
            response_metadata=request.response_metadata,
        )
        
        return SaveCaseResponse(
            success=True,
            case_id=result["id"],
            case_name=result["case_name"],
            query_summary=result["query_summary"],
            message="Case saved successfully"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error saving case: {str(e)}"
        )


@router.get("", response_model=SavedCasesListResponse)
async def get_saved_cases(
    limit: int = 20,
    include_archived: bool = False,
    current_user: dict = Depends(require_physician)
):
    """
    Get all saved cases for the current user.
    
    Returns cases sorted by most recently used/created.
    """
    try:
        service = get_saved_cases_service()
        cases = await service.get_user_cases(
            user_id=current_user["id"],
            limit=limit,
            include_archived=include_archived
        )
        
        return SavedCasesListResponse(
            success=True,
            total=len(cases),
            cases=[SavedCase(**case) for case in cases]
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cases: {str(e)}"
        )


@router.get("/{case_id}", response_model=SavedCase)
async def get_case(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Get a specific saved case by ID.
    """
    try:
        service = get_saved_cases_service()
        case = await service.get_case(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        if not case:
            raise HTTPException(
                status_code=404,
                detail="Case not found"
            )
        
        return SavedCase(**case)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving case: {str(e)}"
        )


@router.post("/{case_id}/use", response_model=UseCaseResponse)
async def use_case(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Use a saved case for a new query.
    
    This endpoint:
    1. Retrieves the case
    2. Increments the use count
    3. Returns the case with a formatted context prompt
    
    The context_prompt can be passed to the chat as system context
    to inform the AI about the patient's details.
    """
    try:
        service = get_saved_cases_service()
        case = await service.use_case(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        if not case:
            raise HTTPException(
                status_code=404,
                detail="Case not found"
            )
        
        # Build context prompt for the chat
        context_prompt = _build_context_prompt(case)
        
        return UseCaseResponse(
            success=True,
            case=SavedCase(**case),
            context_prompt=context_prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error using case: {str(e)}"
        )


@router.patch("/{case_id}")
async def update_case(
    case_id: int,
    request: UpdateCaseRequest,
    current_user: dict = Depends(require_physician)
):
    """
    Update a saved case (rename or archive).
    """
    try:
        service = get_saved_cases_service()
        case = await service.update_case(
            user_id=current_user["id"],
            case_id=case_id,
            case_name=request.case_name,
            is_archived=request.is_archived
        )
        
        if not case:
            raise HTTPException(
                status_code=404,
                detail="Case not found"
            )
        
        return {"success": True, "case": SavedCase(**case)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating case: {str(e)}"
        )


@router.delete("/{case_id}")
async def delete_case(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Permanently delete a saved case.
    """
    try:
        service = get_saved_cases_service()
        deleted = await service.delete_case(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Case not found"
            )
        
        return {"success": True, "message": "Case deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting case: {str(e)}"
        )


def _build_context_prompt(case: Dict[str, Any]) -> str:
    """
    Build a context prompt from a saved case.
    This prompt can be passed to the chat to provide patient context.
    """
    parts = ["PATIENT CASE CONTEXT:"]
    parts.append(f"Summary: {case.get('query_summary', 'N/A')}")
    parts.append("")
    
    # Demographics
    demographics = case.get("demographics", {})
    if demographics:
        parts.append("Demographics:")
        for key, value in demographics.items():
            parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        parts.append("")
    
    # Diagnosis
    diagnosis = case.get("diagnosis", {})
    if diagnosis:
        parts.append("Diagnosis:")
        for key, value in diagnosis.items():
            parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        parts.append("")
    
    # Staging
    staging = case.get("staging", {})
    if staging:
        parts.append("Staging:")
        for key, value in staging.items():
            parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        parts.append("")
    
    # Pathology
    pathology = case.get("pathology", {})
    if pathology:
        parts.append("Pathology:")
        for key, value in pathology.items():
            parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        parts.append("")
    
    # Treatment History
    treatment = case.get("treatment_history", {})
    if treatment:
        parts.append("Treatment History:")
        for key, value in treatment.items():
            if isinstance(value, bool):
                value = "Yes" if value else "No"
            parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        parts.append("")
    
    # Risk Factors
    risk = case.get("risk_factors", {})
    if risk:
        parts.append("Risk Factors:")
        for key, value in risk.items():
            if isinstance(value, list):
                value = ", ".join(value)
            parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        parts.append("")
    
    parts.append("Use this patient context when answering questions. Filter and prioritize studies relevant to this patient's characteristics.")
    
    return "\n".join(parts)


# ============================================
# Alert Management Endpoints
# ============================================

@router.post("/{case_id}/alerts", response_model=AlertSettings)
async def enable_case_alerts(
    case_id: int,
    request: EnableAlertRequest,
    current_user: dict = Depends(require_physician)
):
    """
    Enable alerts for a saved case.
    
    When enabled, the system will monitor for new trials that match
    the case criteria and notify the user via email (if enabled).
    """
    try:
        service = get_saved_cases_service()
        result = await service.enable_alerts(
            user_id=current_user["id"],
            case_id=case_id,
            email_notifications=request.email_notifications
        )
        
        return AlertSettings(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error enabling alerts: {str(e)}"
        )


@router.delete("/{case_id}/alerts")
async def disable_case_alerts(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Disable alerts for a saved case.
    """
    try:
        service = get_saved_cases_service()
        disabled = await service.disable_alerts(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        if not disabled:
            raise HTTPException(
                status_code=404,
                detail="Alert not found for this case"
            )
        
        return {"success": True, "message": "Alerts disabled"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error disabling alerts: {str(e)}"
        )


@router.get("/{case_id}/alerts", response_model=AlertSettings)
async def get_case_alert_settings(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Get alert settings for a specific case.
    """
    try:
        service = get_saved_cases_service()
        settings = await service.get_alert_settings(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        if not settings:
            raise HTTPException(
                status_code=404,
                detail="No alerts configured for this case"
            )
        
        return AlertSettings(**settings)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting alert settings: {str(e)}"
        )


@router.get("/{case_id}/alerts/matches", response_model=List[AlertMatch])
async def get_case_alert_matches(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Get new trial matches found by the alert for a case.
    """
    try:
        service = get_saved_cases_service()
        matches = await service.get_new_matches(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        return [AlertMatch(**m) for m in matches]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting matches: {str(e)}"
        )


@router.post("/{case_id}/alerts/matches/viewed")
async def mark_matches_as_viewed(
    case_id: int,
    current_user: dict = Depends(require_physician)
):
    """
    Mark all new matches as viewed for a case.
    """
    try:
        service = get_saved_cases_service()
        success = await service.mark_matches_viewed(
            user_id=current_user["id"],
            case_id=case_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Alert not found for this case"
            )
        
        return {"success": True, "message": "Matches marked as viewed"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error marking matches as viewed: {str(e)}"
        )


# Global alerts endpoint
@router.get("/alerts/all", response_model=List[AlertSettings])
async def get_all_user_alerts(
    current_user: dict = Depends(require_physician)
):
    """
    Get all alert settings for the current user across all cases.
    """
    try:
        service = get_saved_cases_service()
        alerts = await service.get_user_alerts(user_id=current_user["id"])
        
        return [AlertSettings(**a) for a in alerts]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting alerts: {str(e)}"
        )
