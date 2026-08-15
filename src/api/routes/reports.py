"""
Report generation endpoints. Accept result payloads and return PDF reports.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.api.services.report_service import (
    generate_patient_match_report,
    generate_treatment_comparison_report,
    generate_query_report,
)

router = APIRouter(prefix="/report", tags=["Reports"])


@router.post("/patient-match", response_class=Response)
async def report_patient_match(payload: dict):
    """
    Generate a PDF report from a patient matching result.
    Body: same shape as PatientMatchResponse (matches, total_matches, patient_summary, extracted_profile).
    """
    try:
        pdf_bytes = generate_patient_match_report(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=patient-match-report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/treatment-comparison", response_class=Response)
async def report_treatment_comparison(payload: dict):
    """
    Generate a PDF report from a treatment comparison result.
    Body: same shape as TreatmentComparisonResponse (comparison, sources, metadata).
    """
    try:
        pdf_bytes = generate_treatment_comparison_report(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=treatment-comparison-report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/query", response_class=Response)
async def report_query(payload: dict):
    """
    Generate a PDF report from an enhanced query result.
    Body: { "question": str, "format": "standard"|"patient_handout"|"clinic_note", ...EnhancedQueryResponse }
    format defaults to "standard" if not provided.
    """
    try:
        format_type = payload.pop("format", "standard")
        if format_type not in ("standard", "patient_handout", "clinic_note"):
            format_type = "standard"
        pdf_bytes = generate_query_report(payload, format=format_type)
        filename = {"standard": "query-report", "patient_handout": "patient-handout", "clinic_note": "clinic-note"}[format_type] + ".pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
