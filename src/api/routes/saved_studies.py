"""
Saved Studies API Routes

Provides endpoints for saving, retrieving, and managing saved studies/sources.
Users can bookmark studies from search results for later reference.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.api.services.auth_dependencies import require_physician
from src.api.services.saved_studies_service import get_saved_studies_service

router = APIRouter(prefix="/saved-studies", tags=["saved-studies"])


class SaveStudyRequest(BaseModel):
    """Request to save a study"""
    study_id: str
    title: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None


class SavedStudy(BaseModel):
    """A saved study"""
    id: int
    study_id: str
    title: Optional[str]
    doi: Optional[str]
    pmid: Optional[str]
    created_at: Optional[str]


class SavedStudiesListResponse(BaseModel):
    """Response with list of saved studies"""
    success: bool
    total: int
    studies: List[SavedStudy]


@router.post("", response_model=SavedStudy)
async def save_study(
    request: SaveStudyRequest,
    current_user: dict = Depends(require_physician)
):
    """
    Save a study to the user's collection.
    
    If the study is already saved, updates the metadata.
    """
    if not request.study_id:
        raise HTTPException(status_code=400, detail="study_id is required")
    
    try:
        service = get_saved_studies_service()
        result = await service.save_study(
            user_id=current_user["id"],
            study_id=request.study_id,
            title=request.title,
            doi=request.doi,
            pmid=request.pmid,
        )
        return SavedStudy(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving study: {str(e)}")


@router.get("", response_model=SavedStudiesListResponse)
async def get_saved_studies(
    limit: int = 50,
    current_user: dict = Depends(require_physician)
):
    """
    Get all saved studies for the current user.
    
    Returns studies sorted by most recently saved.
    """
    try:
        service = get_saved_studies_service()
        studies = await service.get_user_studies(
            user_id=current_user["id"],
            limit=limit
        )
        return SavedStudiesListResponse(
            success=True,
            total=len(studies),
            studies=[SavedStudy(**s) for s in studies]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving studies: {str(e)}")


@router.get("/check/{study_id}")
async def check_study_saved(
    study_id: str,
    current_user: dict = Depends(require_physician)
):
    """
    Check if a specific study is saved by the user.
    """
    try:
        service = get_saved_studies_service()
        is_saved = await service.is_study_saved(
            user_id=current_user["id"],
            study_id=study_id
        )
        return {"saved": is_saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking study: {str(e)}")


@router.delete("/{study_id}")
async def delete_saved_study(
    study_id: str,
    current_user: dict = Depends(require_physician)
):
    """
    Remove a study from the user's saved collection.
    """
    try:
        service = get_saved_studies_service()
        deleted = await service.delete_study(
            user_id=current_user["id"],
            study_id=study_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Study not found")
        return {"success": True, "message": "Study removed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting study: {str(e)}")
