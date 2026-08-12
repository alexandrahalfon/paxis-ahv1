"""Pydantic models for the /tumor-board endpoint."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TumorBoardRequest(BaseModel):
    """Request model for a tumor-board run."""

    case_text: str = Field(
        ...,
        description="Raw clinical narrative describing the patient case.",
        min_length=20,
    )
    query_type: str = Field(
        default="treatment_recommendation",
        description="Query type hint for regex-based structuring.",
    )


class StudyCitationModel(BaseModel):
    doc_id: str
    title: str
    citation: Optional[str] = None
    year: Optional[int] = None
    relevance_score: float = 0.0
    snippet: Optional[str] = None


class ExpertAssessmentModel(BaseModel):
    specialty: str
    display_name: str
    recommendation: str = Field(
        ...,
        description='One of "favor", "against", "conditional", "insufficient_evidence".',
    )
    recommendation_text: str
    confidence: float
    key_questions: List[str] = Field(default_factory=list)
    supporting_studies: List[StudyCitationModel] = Field(default_factory=list)
    conflicting_studies: List[StudyCitationModel] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


class TumorBoardResponse(BaseModel):
    case_summary: List[str]
    expert_assessments: List[ExpertAssessmentModel]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_report(cls, report) -> "TumorBoardResponse":
        """Build a TumorBoardResponse from a TumorBoardReport dataclass."""
        return cls(
            case_summary=report.case_summary,
            expert_assessments=[
                ExpertAssessmentModel(**a.to_dict())
                for a in report.expert_assessments
            ],
            metadata=report.metadata,
        )
