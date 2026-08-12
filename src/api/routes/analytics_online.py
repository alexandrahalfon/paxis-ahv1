"""
Online Analytics Route — PubMed & ClinicalTrials.gov chart data.

Accepts either a free-text natural-language request OR explicit structured
fields (search_query + metric + group_by). When structured fields are provided
the LLM parsing step is skipped, saving a round-trip.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional

router = APIRouter(prefix="/analytics", tags=["analytics-online"])


class OnlineAggregateRequest(BaseModel):
    # --- Unstructured mode ---
    nl_input: Optional[str] = Field(None, description="Free-text request, e.g. 'OS rates for NSCLC pembrolizumab grouped by arm'")

    # --- Structured mode ---
    search_query: Optional[str] = Field(None, description="Explicit search query for the database")
    metric: Optional[str] = Field(None, description="Explicit metric to extract, e.g. 'overall survival rate'")
    group_by: Optional[str] = Field(None, description="Dimension to group by (optional)")

    source: Literal["pubmed", "clinicaltrials"] = Field(default="pubmed")

    @model_validator(mode="after")
    def require_some_input(self):
        has_nl = bool(self.nl_input and self.nl_input.strip())
        has_structured = bool(self.search_query and self.metric)
        if not has_nl and not has_structured:
            raise ValueError("Provide either nl_input or both search_query and metric.")
        return self


@router.post("/online-aggregate")
async def online_aggregate(request: OnlineAggregateRequest):
    """
    Search PubMed or ClinicalTrials.gov and extract a quantitative metric for charting.

    Supports two input modes:
    - Unstructured: provide nl_input (LLM parses intent automatically)
    - Structured: provide search_query + metric (+ optional group_by); skips LLM parsing

    Response shape:
        { labels, values, unit, title, n_studies, caveats, sources,
          source_type, parsed_query, parsed_metric }
    """
    try:
        import asyncio

        from src.api.services.analytics_online_service import AnalyticsOnlineService

        svc = AnalyticsOnlineService()

        # Determine whether to use structured or NL path
        use_structured = bool(request.search_query and request.metric)

        # Both aggregate_* methods make several synchronous OpenAI calls
        # (query parsing + metric extraction) and a blocking HTTP fetch to
        # PubMed/ClinicalTrials.gov. Running them directly here would block
        # the event loop for the whole request — to_thread keeps this
        # endpoint from stalling every other concurrent request.
        if request.source == "clinicaltrials":
            result = await asyncio.to_thread(
                svc.aggregate_from_clinical_trials,
                nl_input=request.nl_input,
                search_query=request.search_query,
                metric=request.metric,
                group_by=request.group_by,
            )
        else:
            result = await asyncio.to_thread(
                svc.aggregate_from_pubmed,
                nl_input=request.nl_input,
                search_query=request.search_query,
                metric=request.metric,
                group_by=request.group_by,
            )
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Online analytics failed: {e}\n{traceback.format_exc()}"
        )
