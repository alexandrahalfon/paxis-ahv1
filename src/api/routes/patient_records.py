"""
Patient Records API Routes (Phase 0-2 of the consumer-platform redesign)

Everything here is owned by the patient — scoped to the caller's own
patient_profile via require_patient + get_own_profile, never by a
physician relationship. This is the route family the architecture review
calls for in section 24: "patients should be able to manage their own
record without needing a clinician relationship."

Deliberately a separate router/prefix from patient_cases.py (physician-
owned charts, require_physician) and patient_portal.py (chat/tools/
linking, /portal/*) — this one is /patient/*, plural-free to avoid
colliding with patient_cases.py's /patients/* physician routes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.services.auth_dependencies import require_patient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patient", tags=["Patient Records"])

_SERVER_ERROR = "Something went wrong on our end. Please try again in a moment."


async def get_own_profile(current_user: dict = Depends(require_patient)) -> Dict[str, Any]:
    """Resolves (and auto-repairs, for accounts predating patient_profiles)
    the caller's own profile. Every route below depends on this instead of
    accepting a profile id from the client, so nobody can address another
    patient's record by guessing an id."""
    from src.api.services.patient.patient_profile_service import get_patient_profile_service
    try:
        return await get_patient_profile_service().ensure_profile(
            user_id=current_user["id"],
            first_name=current_user.get("first_name"),
            last_name=current_user.get("last_name"),
        )
    except Exception:
        logger.exception("[patient/profile] resolution failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Models ───────────────────────────────────────────────────────────────

class UpdateProfileBody(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None


class DiagnosisBody(BaseModel):
    cancer_site: Optional[str] = None
    histology: Optional[str] = None
    stage: Optional[str] = None
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    diagnosis_date: Optional[str] = None
    raw_text: Optional[str] = None


class BiomarkerBody(BaseModel):
    biomarker_name: str
    value: Optional[str] = None
    measured_date: Optional[str] = None


class TreatmentAgentBody(BaseModel):
    agent_name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    schedule: Optional[str] = None


class TreatmentEpisodeBody(BaseModel):
    regimen: Optional[str] = None
    modality: Optional[str] = None
    intent: Optional[str] = None
    line_of_therapy: Optional[int] = None
    start_date: Optional[str] = None
    status: str = "active"
    raw_text: Optional[str] = None
    agents: List[TreatmentAgentBody] = Field(default_factory=list)


class TreatmentStatusBody(BaseModel):
    status: str
    end_date: Optional[str] = None


class CycleBody(BaseModel):
    cycle_number: Optional[int] = None
    cycle_date: Optional[str] = None
    status: str = "completed"
    notes: Optional[str] = None


class MedicationBody(BaseModel):
    generic_name: str
    brand_name: Optional[str] = None
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    indication: Optional[str] = None
    start_date: Optional[str] = None


class MedicationStatusBody(BaseModel):
    status: str
    end_date: Optional[str] = None


class LabBody(BaseModel):
    test_name: str
    value_numeric: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    abnormal_flag: Optional[str] = None
    collected_at: Optional[str] = None


class ComorbidityBody(BaseModel):
    condition_name: str
    onset_date: Optional[str] = None
    raw_text: Optional[str] = None


class AllergyBody(BaseModel):
    allergen: str
    reaction: Optional[str] = None
    severity: Optional[str] = None


class EncounterBody(BaseModel):
    encounter_date: Optional[str] = None
    encounter_type: Optional[str] = None
    provider_name: Optional[str] = None
    organization: Optional[str] = None
    patient_summary: Optional[str] = None


class VitalBody(BaseModel):
    vital_type: str
    value_numeric: float
    unit: Optional[str] = None
    measured_at: Optional[str] = None


class ConfirmDocumentBody(BaseModel):
    confirmed_fields: Dict[str, Any]


class CareTeamMemberBody(BaseModel):
    physician_id: str
    role: str = "oncologist"
    is_primary: bool = False


# ── Profile ──────────────────────────────────────────────────────────────

@router.get("/profile")
async def get_profile(profile: dict = Depends(get_own_profile)):
    return {"profile": profile}


@router.put("/profile")
async def update_profile(body: UpdateProfileBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.patient_profile_service import get_patient_profile_service
    try:
        updated = await get_patient_profile_service().update_profile(
            profile["user_id"], **body.model_dump(exclude_unset=True)
        )
        return {"success": True, "profile": updated}
    except Exception:
        logger.exception("[patient/profile:update] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Diagnoses / biomarkers ──────────────────────────────────────────────

@router.get("/diagnoses")
async def list_diagnoses(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.diagnosis_service import get_diagnosis_service
    return {"diagnoses": await get_diagnosis_service().list_diagnoses(profile["id"])}


@router.post("/diagnoses")
async def add_diagnosis(body: DiagnosisBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.diagnosis_service import get_diagnosis_service
    diagnosis = await get_diagnosis_service().add_diagnosis(
        patient_profile_id=profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    return {"success": True, "diagnosis": diagnosis}


@router.get("/biomarkers")
async def list_biomarkers(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.diagnosis_service import get_diagnosis_service
    return {"biomarkers": await get_diagnosis_service().list_biomarkers(profile["id"])}


@router.post("/biomarkers")
async def add_biomarker(body: BiomarkerBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.diagnosis_service import get_diagnosis_service
    biomarker = await get_diagnosis_service().add_biomarker(
        patient_profile_id=profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    return {"success": True, "biomarker": biomarker}


# ── Treatments ───────────────────────────────────────────────────────────

@router.get("/treatments")
async def list_treatments(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.treatment_service import get_treatment_service
    return {"episodes": await get_treatment_service().list_episodes(profile["id"])}


@router.post("/treatments")
async def add_treatment(body: TreatmentEpisodeBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.treatment_service import get_treatment_service
    episode = await get_treatment_service().add_episode(
        patient_profile_id=profile["id"],
        regimen=body.regimen, modality=body.modality, intent=body.intent,
        line_of_therapy=body.line_of_therapy, start_date=body.start_date,
        status=body.status, raw_text=body.raw_text,
        agents=[a.model_dump() for a in body.agents],
        created_by=profile["user_id"],
    )
    return {"success": True, "episode": episode}


@router.put("/treatments/{episode_id}/status")
async def update_treatment_status(
    episode_id: str, body: TreatmentStatusBody, profile: dict = Depends(get_own_profile)
):
    from src.api.services.patient.treatment_service import get_treatment_service
    episode = await get_treatment_service().update_episode_status(
        episode_id, profile["id"], body.status, body.end_date, created_by=profile["user_id"]
    )
    if not episode:
        raise HTTPException(status_code=404, detail="Treatment episode not found")
    return {"success": True, "episode": episode}


@router.post("/treatments/{episode_id}/cycles")
async def add_cycle(episode_id: str, body: CycleBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.treatment_service import get_treatment_service
    cycle = await get_treatment_service().add_cycle(
        episode_id, profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    if not cycle:
        raise HTTPException(status_code=404, detail="Treatment episode not found")
    return {"success": True, "cycle": cycle}


@router.get("/treatments/{episode_id}/agents")
async def list_treatment_agents(episode_id: str, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.treatment_service import get_treatment_service
    return {"agents": await get_treatment_service().list_agents(episode_id)}


# ── Medications ──────────────────────────────────────────────────────────

@router.get("/medications")
async def list_medications(active_only: bool = False, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.medication_service import get_medication_service
    return {"medications": await get_medication_service().list_medications(profile["id"], active_only)}


@router.post("/medications")
async def add_medication(body: MedicationBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.medication_service import get_medication_service
    med = await get_medication_service().add_medication(
        patient_profile_id=profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    return {"success": True, "medication": med}


@router.put("/medications/{medication_id}/status")
async def update_medication_status(
    medication_id: str, body: MedicationStatusBody, profile: dict = Depends(get_own_profile)
):
    from src.api.services.patient.medication_service import get_medication_service
    med = await get_medication_service().update_status(
        medication_id, profile["id"], body.status, body.end_date, created_by=profile["user_id"]
    )
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")
    return {"success": True, "medication": med}


# ── Labs ─────────────────────────────────────────────────────────────────

@router.get("/labs")
async def list_labs(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.lab_service import get_lab_service
    return {"labs": await get_lab_service().list_results(profile["id"])}


@router.get("/labs/trend/{test_name}")
async def lab_trend(test_name: str, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.lab_service import get_lab_service
    return {"trend": await get_lab_service().get_trend(profile["id"], test_name)}


@router.post("/labs")
async def add_lab(body: LabBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.lab_service import get_lab_service
    lab = await get_lab_service().add_result(
        patient_profile_id=profile["id"], source_type="patient_manual",
        verification_status="patient_confirmed", created_by=profile["user_id"],
        **body.model_dump(),
    )
    return {"success": True, "lab": lab}


# ── Comorbidities / allergies ───────────────────────────────────────────

@router.get("/comorbidities")
async def list_comorbidities(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.conditions_service import get_conditions_service
    return {"comorbidities": await get_conditions_service().list_comorbidities(profile["id"])}


@router.post("/comorbidities")
async def add_comorbidity(body: ComorbidityBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.conditions_service import get_conditions_service
    row = await get_conditions_service().add_comorbidity(
        patient_profile_id=profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    return {"success": True, "comorbidity": row}


@router.get("/allergies")
async def list_allergies(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.conditions_service import get_conditions_service
    return {"allergies": await get_conditions_service().list_allergies(profile["id"])}


@router.post("/allergies")
async def add_allergy(body: AllergyBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.conditions_service import get_conditions_service
    row = await get_conditions_service().add_allergy(
        patient_profile_id=profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    return {"success": True, "allergy": row}


# ── Vitals ───────────────────────────────────────────────────────────────

@router.get("/vitals/{vital_type}")
async def vitals_trend(vital_type: str, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.vitals_service import get_vitals_service
    return {"trend": await get_vitals_service().get_trend(profile["id"], vital_type)}


@router.post("/vitals")
async def add_vital(body: VitalBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.vitals_service import get_vitals_service
    row = await get_vitals_service().add_vital(patient_profile_id=profile["id"], **body.model_dump())
    return {"success": True, "vital": row}


# ── Encounters ───────────────────────────────────────────────────────────

@router.get("/encounters")
async def list_encounters(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.encounter_service import get_encounter_service
    return {"encounters": await get_encounter_service().list_encounters(profile["id"])}


@router.post("/encounters")
async def add_encounter(body: EncounterBody, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.encounter_service import get_encounter_service
    row = await get_encounter_service().add_encounter(
        patient_profile_id=profile["id"], created_by=profile["user_id"], **body.model_dump()
    )
    return {"success": True, "encounter": row}


# ── Timeline / state ─────────────────────────────────────────────────────

@router.get("/timeline")
async def get_timeline(limit: int = 100, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.patient_timeline_service import get_patient_timeline_service
    return {"timeline": await get_patient_timeline_service().get_timeline(profile["id"], limit)}


@router.get("/state")
async def get_state(refresh: bool = False, profile: dict = Depends(get_own_profile)):
    """The cached patient_state_snapshot, rebuilt on demand with
    ?refresh=true. The Phase 4 retrieval layer reads the cached snapshot
    directly rather than calling this route — this is for the dashboard
    UI and for a patient/clinician wanting the current view."""
    from src.api.services.patient.patient_state_service import get_patient_state_service
    service = get_patient_state_service()
    if refresh:
        built = await service.build_state(profile["id"])
        return built
    snapshot = await service.get_latest_snapshot(profile["id"])
    if snapshot is None:
        built = await service.build_state(profile["id"])
        return built
    return {"state": snapshot["state"], "retrieval_features": snapshot["retrieval_features"],
            "as_of": snapshot.get("as_of")}


# ── Documents ────────────────────────────────────────────────────────────

@router.get("/documents")
async def list_documents(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.patient_document_service import get_patient_document_service
    return {"documents": await get_patient_document_service().list_documents(profile["id"])}


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    document_date: Optional[str] = None,
    profile: dict = Depends(get_own_profile),
):
    """Uploads and stores the file, then kicks off extraction. Nothing is
    written to the patient's canonical record yet — see
    /documents/{id}/confirm, which requires an explicit review step
    first (architecture review section 22: never trust OCR silently)."""
    from src.api.services.patient.patient_document_service import get_patient_document_service
    from src.api.services.patient.patient_document_extractor import get_patient_document_extractor

    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File is too large (25MB limit).")

    doc = await get_patient_document_service().create_document(
        patient_profile_id=profile["id"], filename=file.filename or "upload",
        content=content, content_type=file.content_type, document_date=document_date,
    )
    try:
        extraction = await get_patient_document_extractor().run(doc["id"], profile["id"])
        doc["extraction"] = extraction
    except Exception as e:
        logger.warning("[patient/documents:upload] extraction failed: %s", e)
        doc["extraction_error"] = str(e)
    return {"success": True, "document": doc}


@router.get("/documents/{document_id}")
async def get_document(document_id: str, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.patient_document_service import get_patient_document_service
    doc = await get_patient_document_service().get_document(document_id, profile["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from src.api.services.patient_db import get_patient_db
    db = get_patient_db()
    await db.ensure_schema()
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        extraction = await conn.fetchrow(
            "SELECT * FROM document_extractions WHERE document_id = $1 ORDER BY created_at DESC LIMIT 1",
            document_id,
        )
    if extraction:
        import json
        ext = dict(extraction)
        ext["id"] = str(ext["id"])
        ext["document_id"] = str(ext["document_id"])
        if isinstance(ext.get("extracted_fields"), str):
            try:
                ext["extracted_fields"] = json.loads(ext["extracted_fields"])
            except (TypeError, ValueError):
                pass
        doc["extraction"] = ext
    return {"document": doc}


@router.post("/documents/{document_id}/confirm")
async def confirm_document(
    document_id: str, body: ConfirmDocumentBody, profile: dict = Depends(get_own_profile),
):
    from src.api.services.patient.patient_document_validator import get_patient_document_validator
    try:
        return await get_patient_document_validator().confirm(
            document_id=document_id, patient_profile_id=profile["id"],
            confirmed_fields=body.confirmed_fields, created_by=profile["user_id"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("[patient/documents:confirm] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Care team (Phase 6) ──────────────────────────────────────────────────

@router.get("/care-team")
async def list_care_team(profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.patient_care_team_service import get_patient_care_team_service
    return {"care_team": await get_patient_care_team_service().list_care_team(profile["id"])}


@router.post("/care-team")
async def add_care_team_member(body: CareTeamMemberBody, profile: dict = Depends(get_own_profile)):
    """Adds a clinician directly from the patient's side. Distinct from
    the invite/request flow in patient_link_service.py, which binds to a
    physician-owned chart record — this is for a care-team member who has
    no chart on their end at all (e.g. a PCP or nutritionist added purely
    for the patient's own record-keeping and future escalation routing)."""
    from src.api.services.patient.patient_care_team_service import get_patient_care_team_service
    link = await get_patient_care_team_service().add_member(
        patient_profile_id=profile["id"], physician_id=body.physician_id,
        role=body.role, is_primary=body.is_primary,
    )
    return {"success": True, "link": link}


@router.delete("/care-team/{physician_id}")
async def remove_care_team_member(physician_id: str, profile: dict = Depends(get_own_profile)):
    from src.api.services.patient.patient_care_team_service import get_patient_care_team_service
    ok = await get_patient_care_team_service().revoke_member(profile["id"], physician_id)
    return {"revoked": ok}
