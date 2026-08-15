"""
Request/Response models for Treatment Evaluation Flow.

These models support the treatment evaluation workflow including:
- Treatment discovery from medical literature
- Treatment comparison with clinical context

Validates: Requirements 2.4, 3.1, 3.3
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class TreatmentDiscoveryRequest(BaseModel):
    """
    Request for discovering relevant treatments from the medical literature.
    
    Used when the user selects "Explore all relevant treatment options" to
    automatically identify treatments based on their clinical context.
    
    Validates: Requirements 3.1
    """
    query_context: Dict[str, Any] = Field(
        ...,
        description="Patient profile, diagnosis, cancer type, stage, biomarkers, and other clinical context"
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of evidence chunks to retrieve for treatment discovery"
    )


class TreatmentOption(BaseModel):
    """
    A treatment option extracted from the medical literature.
    
    Represents a distinct treatment modality, drug name, or therapeutic approach
    identified from retrieved evidence.
    
    Validates: Requirements 3.3
    """
    name: str = Field(
        ...,
        description="Treatment name (drug name, regimen, or therapeutic approach)"
    )
    category: str = Field(
        ...,
        description="Treatment category (e.g., 'chemotherapy', 'immunotherapy', 'radiation', 'surgery', 'targeted therapy')"
    )
    frequency: int = Field(
        default=1,
        ge=1,
        description="How often this treatment is mentioned in the retrieved evidence"
    )
    supporting_studies: List[str] = Field(
        default_factory=list,
        description="List of study citations or doc_ids that mention this treatment"
    )


class TreatmentDiscoveryResponse(BaseModel):
    """
    Response from treatment discovery containing identified treatment options.
    
    Returns the treatments extracted from the medical literature along with
    an evidence summary and the query used for retrieval.
    
    Validates: Requirements 3.1, 3.3
    """
    treatments: List[TreatmentOption] = Field(
        default_factory=list,
        description="List of treatment options discovered from the literature"
    )
    evidence_summary: str = Field(
        default="",
        description="Summary of the evidence used to identify treatments"
    )
    query_used: str = Field(
        default="",
        description="The search query constructed from the clinical context"
    )


class TreatmentComparisonWithContextRequest(BaseModel):
    """
    Request for comparing treatments using the provided clinical context.
    
    Used when the user specifies treatments to compare (either manually entered
    or selected from discovered options) along with their clinical context.
    
    Validates: Requirements 2.4
    """
    treatments: List[str] = Field(
        ...,
        min_length=2,
        description="List of treatment names to compare (minimum 2 required)"
    )
    query_context: Dict[str, Any] = Field(
        ...,
        description="Patient profile, diagnosis, cancer type, stage, biomarkers, and other clinical context"
    )
    top_k: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Number of evidence chunks to retrieve per treatment arm"
    )
