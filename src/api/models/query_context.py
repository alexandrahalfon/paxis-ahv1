"""
QueryContext model for preserving clinical context throughout the treatment evaluation workflow.

Validates: Requirements 4.1, 4.2
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.models.query_models import PatientProfile


@dataclass
class QueryContext:
    """
    Data model for preserving clinical context throughout the treatment evaluation workflow.
    
    This context captures patient profile information, diagnosis details, staging, biomarkers,
    and any prior query information to ensure treatment comparisons are relevant to the
    specific clinical case.
    
    Validates: Requirements 4.1, 4.2
    """
    patient_profile: Optional["PatientProfile"] = None
    diagnosis: Optional[str] = None
    cancer_type: Optional[str] = None
    stage: Optional[str] = None
    biomarkers: List[str] = field(default_factory=list)
    prior_treatments: List[str] = field(default_factory=list)
    prior_query: Optional[str] = None
    session_id: Optional[str] = None

    def to_search_query(self) -> str:
        """
        Convert context to a search query string for RAG retrieval.
        
        Combines available clinical context fields into a coherent search query
        that can be used to retrieve relevant medical literature.
        
        Returns:
            str: A search query string combining available context fields.
        """
        query_parts = []
        
        # Add cancer type or diagnosis
        if self.cancer_type:
            query_parts.append(self.cancer_type)
        elif self.diagnosis:
            query_parts.append(self.diagnosis)
        
        # Add stage information
        if self.stage:
            query_parts.append(f"stage {self.stage}")
        
        # Add biomarkers
        if self.biomarkers:
            query_parts.append(" ".join(self.biomarkers))
        
        # Add prior treatments for context
        if self.prior_treatments:
            treatments_str = ", ".join(self.prior_treatments)
            query_parts.append(f"prior treatment: {treatments_str}")
        
        # Include patient profile details if available
        if self.patient_profile:
            profile_parts = []
            if hasattr(self.patient_profile, 'histology') and self.patient_profile.histology:
                profile_parts.append(self.patient_profile.histology)
            if hasattr(self.patient_profile, 'anatomical_site') and self.patient_profile.anatomical_site:
                profile_parts.append(self.patient_profile.anatomical_site)
            if hasattr(self.patient_profile, 'molecular_markers') and self.patient_profile.molecular_markers:
                profile_parts.extend(self.patient_profile.molecular_markers)
            if profile_parts:
                query_parts.extend(profile_parts)
        
        # Fall back to prior query if no structured context
        if not query_parts and self.prior_query:
            return self.prior_query
        
        return " ".join(query_parts) if query_parts else ""

    def is_complete(self) -> bool:
        """
        Check if minimum context is available for treatment evaluation.
        
        A context is considered complete if it has at least one of:
        - cancer_type
        - diagnosis
        - prior_query
        
        Returns:
            bool: True if minimum context is available, False otherwise.
        """
        return bool(self.cancer_type or self.diagnosis or self.prior_query)

    def to_dict(self) -> dict:
        """
        Convert QueryContext to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the context.
        """
        result = {
            "diagnosis": self.diagnosis,
            "cancer_type": self.cancer_type,
            "stage": self.stage,
            "biomarkers": self.biomarkers,
            "prior_treatments": self.prior_treatments,
            "prior_query": self.prior_query,
            "session_id": self.session_id,
        }
        
        # Convert patient_profile to dict if present
        if self.patient_profile:
            if hasattr(self.patient_profile, 'model_dump'):
                result["patient_profile"] = self.patient_profile.model_dump()
            elif hasattr(self.patient_profile, 'dict'):
                result["patient_profile"] = self.patient_profile.dict()
            else:
                result["patient_profile"] = None
        else:
            result["patient_profile"] = None
        
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "QueryContext":
        """
        Create a QueryContext from a dictionary.
        
        Args:
            data: Dictionary containing context fields.
            
        Returns:
            QueryContext: New instance populated from the dictionary.
        """
        from src.api.models.query_models import PatientProfile
        
        patient_profile = None
        if data.get("patient_profile"):
            patient_profile = PatientProfile(**data["patient_profile"])
        
        return cls(
            patient_profile=patient_profile,
            diagnosis=data.get("diagnosis"),
            cancer_type=data.get("cancer_type"),
            stage=data.get("stage"),
            biomarkers=data.get("biomarkers", []),
            prior_treatments=data.get("prior_treatments", []),
            prior_query=data.get("prior_query"),
            session_id=data.get("session_id"),
        )
