"""
FastAPI route for the multi-agent tumor board.

POST /api/tumor-board
    Body: { "case_text": "<raw clinical narrative>", "query_type": "..." }

Returns a TumorBoardResponse with one ExpertAssessment per specialty agent.

Runs alongside the existing /api/rag endpoint — independent retrieval
and LLM calls, same underlying ComprehensiveRetriever singleton.
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.models.tumor_board_models import (
    TumorBoardRequest,
    TumorBoardResponse,
)
from src.api.services.auth_dependencies import get_current_user_optional
from src.api.services.tumor_board.orchestrator import (
    TumorBoardOrchestrator,
    get_tumor_board_orchestrator,
)


router = APIRouter(prefix="/tumor-board", tags=["Tumor Board"])


@router.post("", response_model=TumorBoardResponse)
async def present_case(
    req: TumorBoardRequest,
    current_user: dict = Depends(get_current_user_optional),
) -> TumorBoardResponse:
    """
    Present a patient case to the virtual tumor board and return each
    specialist's assessment in parallel.

    The request must include `case_text` — the raw clinical narrative.
    """
    orchestrator: TumorBoardOrchestrator = get_tumor_board_orchestrator()
    try:
        report = await orchestrator.present_case(
            case_text=req.case_text,
            query_type=req.query_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tumor board orchestration failed: {e}",
        )
    return TumorBoardResponse.from_report(report)


@router.get("/panel")
async def list_panel(
    current_user: dict = Depends(get_current_user_optional),
) -> dict:
    """Return the current list of specialty agents on the panel."""
    orchestrator = get_tumor_board_orchestrator()
    return {
        "agents": [
            {
                "specialty": a.specialty,
                "display_name": a.display_name,
                "max_sub_queries": a.max_sub_queries,
                "studies_per_query": a.studies_per_query,
            }
            for a in orchestrator.agents
        ]
    }
