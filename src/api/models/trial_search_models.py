"""
Pydantic models for external clinical trials search.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .query_models import PatientProfile


class TrialSearchRequest(BaseModel):
    """Request payload for ClinicalTrials.gov search."""

    query: Optional[str] = Field(None, description="Free-text patient/case description")
    patient_profile: Optional[PatientProfile] = Field(
        None, description="Structured patient profile overrides"
    )
    max_results: int = Field(25, ge=1, le=100, description="Max trials to return")
    recruiting_status: Optional[List[str]] = Field(
        None, description="Filter by recruiting status labels"
    )
    phase: Optional[List[str]] = Field(None, description="Filter by trial phases")
    location: Optional[str] = Field(None, description="Location filter (city/state/country)")


class TrialResult(BaseModel):
    """Normalized ClinicalTrials.gov result."""

    title: str
    nct_id: str
    status: Optional[str] = None
    phase: Optional[List[str]] = None
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    enrollment: Optional[int] = None
    url: Optional[str] = None
    match_score: float = Field(0.0, ge=0.0, le=1.0)
    match_reasons: List[str] = Field(default_factory=list)


class TrialSearchResponse(BaseModel):
    """Response for ClinicalTrials.gov search."""

    trials: List[TrialResult] = Field(default_factory=list)
    total_results: int = 0
    extracted_profile: Optional[Dict[str, Any]] = None
    query_terms: List[str] = Field(default_factory=list)
