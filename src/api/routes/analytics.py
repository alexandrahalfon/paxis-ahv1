"""
Analytics API Routes

Endpoints for aggregate queries over the study knowledge base:
- /overview          → quick stats
- /aggregate         → metric by group (e.g. avg OS by phase)
- /dose-distribution → histogram of radiation doses
- /technique-frequency → count by technique
- /outcomes-by-stage → outcome metric grouped by stage
- /forest-plot       → meta-analysis HR pooling
- /filters           → dropdown values for cancer type, phase, technique
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging
import traceback

from src.api.services import analytics_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


# ── Request / Response Models ─────────────────────────────────────

class AggregateRequest(BaseModel):
    metric: str = Field(..., description="Column to aggregate (e.g. os_rate_percent)")
    group_by: str = Field(..., description="Column to group by (e.g. normalized_phase)")
    agg: str = Field(default="avg", description="Aggregation: avg, median, min, max, sum")
    filters: Optional[Dict[str, str]] = Field(default=None, description="Filter columns")
    limit: int = Field(default=20, ge=1, le=50)


class DoseDistributionRequest(BaseModel):
    cancer_type: Optional[str] = None
    technique: Optional[str] = None
    bin_width: float = Field(default=5.0, ge=1.0, le=20.0)


class OutcomesByStageRequest(BaseModel):
    cancer_type: Optional[str] = None
    metric: str = Field(default="os_rate_percent")


class ForestPlotRequest(BaseModel):
    cancer_type: Optional[str] = None
    metric: str = Field(default="os", description="os or pfs")


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/overview")
async def overview():
    """Quick stats across the whole knowledge base."""
    try:
        return await analytics_service.get_overview()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/aggregate")
async def aggregate(request: AggregateRequest):
    """
    Compute an aggregate metric grouped by a column.

    Example: POST { "metric": "os_rate_percent", "group_by": "normalized_phase" }
    Returns: { labels, values, study_counts, unit, ... }
    """
    try:
        result = await analytics_service.aggregate_metric(
            metric=request.metric,
            group_by=request.group_by,
            agg=request.agg,
            filters=request.filters,
            limit=request.limit,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dose-distribution")
async def dose_distribution(request: DoseDistributionRequest):
    """
    Histogram of radiation total doses.

    Optional filters: cancer_type, technique.
    """
    try:
        return await analytics_service.dose_distribution(
            cancer_type=request.cancer_type,
            technique=request.technique,
            bin_width=request.bin_width,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/technique-frequency")
async def technique_frequency(
    cancer_type: Optional[str] = Query(default=None),
    limit: int = Query(default=15, ge=1, le=30),
):
    """Count of studies per radiation technique."""
    try:
        return await analytics_service.technique_frequency(
            cancer_type=cancer_type, limit=limit,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/outcomes-by-stage")
async def outcomes_by_stage(request: OutcomesByStageRequest):
    """Average outcome metric by cancer stage."""
    try:
        result = await analytics_service.outcomes_by_stage(
            cancer_type=request.cancer_type,
            metric=request.metric,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forest-plot")
async def forest_plot(request: ForestPlotRequest):
    """
    Meta-analysis forest plot with inverse-variance HR pooling.

    Returns study-level HR + CI plus pooled estimate.
    """
    try:
        return await analytics_service.meta_analysis_forest(
            cancer_type=request.cancer_type,
            metric=request.metric,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filters")
async def get_filter_options():
    """Get available values for filter dropdowns."""
    try:
        cancer_types = await analytics_service.list_cancer_types()
        techniques = await analytics_service.list_techniques()
        phases = await analytics_service.list_phases()
        return {
            "cancer_types": cancer_types,
            "techniques": techniques,
            "phases": phases,
            "metrics": sorted(analytics_service._METRIC_COLS),
            "group_by_options": sorted(analytics_service._GROUP_COLS.keys()),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
