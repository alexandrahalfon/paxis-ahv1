"""
Physician RAG beta routes (2026-08-12 convergence Sprint C item 21).

The first HTTP surface for physician_rag_orchestrator.py (Sprint C item
20) -- everything upstream of this file (QueryAnalysis, the physician
context selector, verified authorization, the legacy retrieval adapter,
the physician applicability scorer, the answer generator, the grounding
gate) already exists; this is just the protected entrypoint into it.

Kept in its own router/prefix rather than added to query.py or
patient_cases.py, for the same reason patient_portal.py's own module
docstring gives for staying separate from patient_cases.py: the
existing physician product's endpoints are untouched, and everything
here is additive. Gated behind settings.physician_rag_beta_enabled
(default False, see src/core/config.py) -- a 404 while disabled, not a
403, so an unreleased beta route doesn't advertise its own existence to
a physician account that isn't part of the beta.

Authorization for a specific patient is NOT re-checked at the route
layer: physician_rag_orchestrator.answer_physician_query() already runs
authorize_physician_patient_access() (Sprint C item 14) as a hard gate
before touching any patient data, and a denied request comes back as a
normal 200 with ACCESS_DENIED_RESPONSE (a deliberate choice -- see that
module's own docstring for why a denial must not itself distinguish
"wrong patient id" from "not your patient" via a different HTTP status).
Duplicating the check here would just be a second, potentially
drifting copy of the same rule.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.services.auth_dependencies import require_physician
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/physician-beta", tags=["Physician RAG Beta"])

_SERVER_ERROR = "Something went wrong on our end. Please try again in a moment."
_NOT_ENABLED_DETAIL = "Not found"


class PhysicianQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    patient_profile_id: Optional[str] = None
    intent: Optional[str] = Field(default=None, max_length=64)


def _require_beta_enabled() -> None:
    if not settings.physician_rag_beta_enabled:
        # See module docstring -- 404, not 403: this route should look
        # like it doesn't exist for a deployment that hasn't opted in.
        raise HTTPException(status_code=404, detail=_NOT_ENABLED_DETAIL)


@router.post("/query")
async def physician_query(
    body: PhysicianQueryRequest, current_user=Depends(require_physician),
) -> Dict[str, Any]:
    """Ask a clinical question, optionally scoped to a specific patient.
    See physician_rag_orchestrator.answer_physician_query() for the full
    pipeline and physician_context_service.py's four intent names
    (therapy_selection/treatment_sequencing/toxicity_management/
    trial_eligibility) -- `intent` is optional; QueryAnalysis (item 12)
    detects one from the question text when omitted."""
    _require_beta_enabled()
    from src.api.services.physician.physician_rag_orchestrator import (
        answer_physician_query,
    )
    try:
        result = await answer_physician_query(
            physician_user_id=current_user["id"],
            question=body.question,
            patient_profile_id=body.patient_profile_id,
            intent=body.intent,
        )
        return result.to_dict()
    except HTTPException:
        raise
    except Exception:
        logger.exception("[physician-beta/query] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)
