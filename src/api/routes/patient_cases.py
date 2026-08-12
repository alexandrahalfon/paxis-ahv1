"""
Patient Cases API Routes

CRUD for patient records (Phase 1 of the patient-centric pivot) plus an
endpoint to trigger evidence auto-seeding (Phase 2). Mirrors the
saved_cases.py / saved_studies.py route conventions: bearer auth via
require_physician, current_user["id"] used as the owning physician_id.
Patient-role accounts are rejected: these are clinician-only endpoints.

Patients are scoped to the physician who created them — every read/write
below filters on physician_id so one physician can't see another's patients.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.services.auth_dependencies import require_physician
from src.api.services.patient_service import get_patient_service

router = APIRouter(prefix="/patients", tags=["Patient Cases"])


# ----------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------

class CreatePatientRequest(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None  # ISO date string
    sex: Optional[str] = None
    mrn: Optional[str] = None


class UpdatePatientRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    mrn: Optional[str] = None


class AddDiagnosisRequest(BaseModel):
    cancer_site: Optional[str] = None
    histology: Optional[str] = None
    stage: Optional[str] = None
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    diagnosis_date: Optional[str] = None
    raw_text: Optional[str] = None


class AddBiomarkerRequest(BaseModel):
    biomarker_name: str
    value: Optional[str] = None
    measured_date: Optional[str] = None
    raw_text: Optional[str] = None


class AddTreatmentRequest(BaseModel):
    treatment_type: str
    regimen: Optional[str] = None
    line_of_therapy: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[str] = None
    raw_text: Optional[str] = None


class SignalCheckRequest(BaseModel):
    query_text: str
    active_patient_id: Optional[str] = None


class SignalPreviewRequest(BaseModel):
    query_text: str


class SignalApplyRequest(BaseModel):
    action: str  # 'attach' | 'create_new' | 'ignore'
    extracted: Dict[str, Any] = {}
    raw_text: str = ""
    patient_id: Optional[str] = None
    new_patient_first_name: Optional[str] = None
    new_patient_last_name: Optional[str] = None


# ----------------------------------------------------------------------
# Patient CRUD
# ----------------------------------------------------------------------

@router.post("")
async def create_patient(
    request: CreatePatientRequest,
    current_user: dict = Depends(require_physician),
):
    """Create a new patient record owned by the current physician."""
    try:
        service = get_patient_service()
        patient = await service.create_patient(
            physician_id=current_user["id"],
            first_name=request.first_name,
            last_name=request.last_name,
            date_of_birth=request.date_of_birth,
            sex=request.sex,
            mrn=request.mrn,
        )
        return {"success": True, "patient": patient}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating patient: {e}")


@router.get("")
async def list_patients(
    limit: int = 100,
    offset: int = 0,
    include_diagnosis: bool = False,
    current_user: dict = Depends(require_physician),
):
    """List patients owned by the current physician. Pass
    include_diagnosis=true to inline each patient's latest cancer_site/
    stage — used by the patient picker/list UI so it can show more than
    just a name.

    Paginated via limit/offset. `total` is the physician's full patient
    count, so a caller can tell whether more pages exist rather than
    silently seeing only the first page."""
    try:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        service = get_patient_service()
        if include_diagnosis:
            patients = await service.list_patients_with_summary(
                current_user["id"], limit=limit, offset=offset
            )
        else:
            patients = await service.list_patients(
                current_user["id"], limit=limit, offset=offset
            )
        total = await service.count_patients(current_user["id"])
        return {
            "patients": patients,
            "count": len(patients),
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(patients)) < total,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing patients: {e}")


@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    current_user: dict = Depends(require_physician),
):
    """Full patient snapshot: demographics + latest diagnosis + biomarkers
    + treatment history + recent timeline."""
    service = get_patient_service()
    patient = await service.get_patient_full(patient_id, current_user["id"])
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}")
async def update_patient(
    patient_id: str,
    request: UpdatePatientRequest,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    patient = await service.update_patient(
        patient_id, current_user["id"], **request.model_dump(exclude_unset=True)
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"success": True, "patient": patient}


# ----------------------------------------------------------------------
# Diagnosis / biomarkers / treatment history
# ----------------------------------------------------------------------

@router.post("/{patient_id}/diagnosis")
async def add_diagnosis(
    patient_id: str,
    request: AddDiagnosisRequest,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    diagnosis = await service.add_diagnosis(
        patient_id, current_user["id"], **request.model_dump()
    )
    if diagnosis is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"success": True, "diagnosis": diagnosis}


@router.post("/{patient_id}/biomarkers")
async def add_biomarker(
    patient_id: str,
    request: AddBiomarkerRequest,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    biomarker = await service.add_biomarker(
        patient_id, current_user["id"], **request.model_dump()
    )
    if biomarker is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"success": True, "biomarker": biomarker}


@router.get("/{patient_id}/biomarkers")
async def list_biomarkers(
    patient_id: str,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    if not await service.get_patient(patient_id, current_user["id"]):
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"biomarkers": await service.list_biomarkers(patient_id)}


@router.post("/{patient_id}/treatment")
async def add_treatment(
    patient_id: str,
    request: AddTreatmentRequest,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    treatment = await service.add_treatment(
        patient_id, current_user["id"], **request.model_dump()
    )
    if treatment is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"success": True, "treatment": treatment}


@router.get("/{patient_id}/treatment")
async def list_treatment(
    patient_id: str,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    if not await service.get_patient(patient_id, current_user["id"]):
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"treatment_history": await service.list_treatment_history(patient_id)}


@router.get("/{patient_id}/timeline")
async def get_timeline(
    patient_id: str,
    limit: int = 100,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    if not await service.get_patient(patient_id, current_user["id"]):
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"timeline": await service.get_timeline(patient_id, limit=limit)}


# ----------------------------------------------------------------------
# Evidence auto-seeding (Phase 2)
# ----------------------------------------------------------------------

@router.post("/{patient_id}/seed")
async def seed_patient_evidence(
    patient_id: str,
    current_user: dict = Depends(require_physician),
):
    """Build a query from the patient's current profile, run it through the
    existing RAG pipeline, and save matched studies as auto-seeded evidence
    for this patient. Safe to call again after an update — re-seeding is
    also how Phase 4 continuous monitoring will trigger a refresh."""
    service = get_patient_service()
    patient = await service.get_patient_full(patient_id, current_user["id"])
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        from src.api.services.patient_collection_seeder import get_patient_collection_seeder

        seeder = get_patient_collection_seeder()
        result = await seeder.seed_patient_collection(
            patient_id=patient_id,
            physician_id=current_user["id"],
            patient_profile=patient,
        )
        return result
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Service not available: {e}")
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error seeding patient evidence: {e}\n{traceback.format_exc()}",
        )


@router.get("/{patient_id}/studies")
async def get_patient_studies(
    patient_id: str,
    current_user: dict = Depends(require_physician),
):
    service = get_patient_service()
    if not await service.get_patient(patient_id, current_user["id"]):
        raise HTTPException(status_code=404, detail="Patient not found")

    from src.api.services.saved_studies_service import get_saved_studies_service

    studies_service = get_saved_studies_service()
    studies = await studies_service.get_patient_studies(patient_id)
    return {"studies": studies, "count": len(studies)}


# ----------------------------------------------------------------------
# Passive signal capture (chat / other modes -> patient profiles)
# ----------------------------------------------------------------------

@router.post("/signal/preview")
async def preview_patient_signal(request: SignalPreviewRequest):
    """No-auth pre-check for anonymous visitors: cheap routing classification
    only (no LLM call, no DB access — see PatientSignalService.is_patient_specific
    for why this one is safe to leave unauthenticated while /signal and
    /signal/apply below stay behind login).

    Used to nudge a logged-out user to create an account when what they just
    typed looks patient-specific. The frontend queues the raw query text
    client-side; the real extraction only runs after they authenticate, by
    replaying that text through the normal /signal endpoint.
    """
    from src.api.services.patient_signal_service import get_patient_signal_service

    service = get_patient_signal_service()
    return {"is_patient_specific": service.is_patient_specific(request.query_text)}


@router.post("/signal")
async def check_patient_signal(
    request: SignalCheckRequest,
    current_user: dict = Depends(require_physician),
):
    """Called by the frontend after a query response in any mode. Returns
    one of:
      - {"action": "ignore"} — not patient-specific enough to act on
      - {"action": "silent_attach", ...} — confidently belongs to the
        active patient; frontend should write it via /signal/apply without
        prompting
      - {"action": "ask", "candidates": [...], ...} — frontend should show
        a confirm prompt before calling /signal/apply
    """
    try:
        from src.api.services.patient_signal_service import get_patient_signal_service

        service = get_patient_signal_service()
        result = await service.evaluate(
            physician_id=current_user["id"],
            query_text=request.query_text,
            active_patient_id=request.active_patient_id,
        )
        return result
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating patient signal: {e}\n{traceback.format_exc()}",
        )


@router.post("/signal/apply")
async def apply_patient_signal(
    request: SignalApplyRequest,
    current_user: dict = Depends(require_physician),
):
    """Writes a signal-detected patient fact after the frontend has either
    gotten physician confirmation (action='ask' path) or decided to
    silently attach (action='silent_attach' path)."""
    try:
        from src.api.services.patient_signal_service import get_patient_signal_service

        service = get_patient_signal_service()
        result = await service.apply(
            physician_id=current_user["id"],
            action=request.action,
            extracted=request.extracted,
            raw_text=request.raw_text,
            patient_id=request.patient_id,
            new_patient_first_name=request.new_patient_first_name,
            new_patient_last_name=request.new_patient_last_name,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Could not apply signal"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error applying patient signal: {e}\n{traceback.format_exc()}",
        )
