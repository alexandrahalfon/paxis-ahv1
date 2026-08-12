"""
Patient Document Validator (Phase 2)

The confirm step: takes the (possibly patient-edited) extracted fields from
document_extractions and writes them into the canonical Phase 1 tables —
only after this runs. Nothing in patient_document_extractor.py writes to
lab_results, medication_exposures, patient_diagnoses, or encounters
directly, for exactly the reason the architecture review calls out: OCR
"Hgb 8.7" vs "Hgb 87" is consequential, not cosmetic, so nothing becomes
part of the patient's record without an explicit "yes, that's right" step.

Every value written this way carries source_type='patient_upload',
source_document_id=<the document>, and verification_status=
'patient_confirmed' — never 'clinician_confirmed', since a patient
confirming their own upload is not the same as clinical verification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.api.services.patient_db import get_patient_db


class PatientDocumentValidator:
    async def confirm(
        self,
        document_id: str,
        patient_profile_id: str,
        confirmed_fields: Dict[str, Any],
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        from src.api.services.patient.patient_document_service import get_patient_document_service
        from src.api.services.patient.lab_service import get_lab_service
        from src.api.services.patient.medication_service import get_medication_service
        from src.api.services.patient.diagnosis_service import get_diagnosis_service
        from src.api.services.patient.encounter_service import get_encounter_service

        doc_service = get_patient_document_service()
        doc = await doc_service.get_document(document_id, patient_profile_id)
        if not doc:
            raise ValueError("Document not found")

        written: Dict[str, list] = {"labs": [], "medications": [], "diagnoses": [], "encounters": []}
        source_kwargs = dict(
            source_type="patient_upload",
            source_document_id=document_id,
            verification_status="patient_confirmed",
            created_by=created_by,
        )

        for lab in confirmed_fields.get("labs", []) or []:
            if not lab.get("test_name"):
                continue
            row = await get_lab_service().add_result(
                patient_profile_id=patient_profile_id,
                test_name=lab["test_name"],
                value_numeric=lab.get("value_numeric"),
                value_text=lab.get("value_text"),
                unit=lab.get("unit"),
                reference_low=lab.get("reference_low"),
                reference_high=lab.get("reference_high"),
                abnormal_flag=lab.get("abnormal_flag"),
                collected_at=lab.get("collected_at"),
                **source_kwargs,
            )
            written["labs"].append(row)

        for med in confirmed_fields.get("medications", []) or []:
            if not med.get("generic_name"):
                continue
            row = await get_medication_service().add_medication(
                patient_profile_id=patient_profile_id,
                generic_name=med["generic_name"],
                brand_name=med.get("brand_name"),
                dose=med.get("dose"),
                route=med.get("route"),
                frequency=med.get("frequency"),
                indication=med.get("indication"),
                **source_kwargs,
            )
            written["medications"].append(row)

        diagnosis = confirmed_fields.get("diagnosis") or {}
        if any(diagnosis.values()):
            row = await get_diagnosis_service().add_diagnosis(
                patient_profile_id=patient_profile_id,
                cancer_site=diagnosis.get("cancer_site"),
                histology=diagnosis.get("histology"),
                stage=diagnosis.get("stage"),
                tnm_t=diagnosis.get("tnm_t"),
                tnm_n=diagnosis.get("tnm_n"),
                tnm_m=diagnosis.get("tnm_m"),
                **source_kwargs,
            )
            written["diagnoses"].append(row)
            for b in diagnosis.get("biomarkers") or []:
                if b.get("biomarker_name"):
                    await get_diagnosis_service().add_biomarker(
                        patient_profile_id=patient_profile_id,
                        biomarker_name=b["biomarker_name"], value=b.get("value"),
                        **source_kwargs,
                    )

        encounter = confirmed_fields.get("encounter") or {}
        if encounter:
            row = await get_encounter_service().add_encounter(
                patient_profile_id=patient_profile_id,
                encounter_date=encounter.get("encounter_date"),
                encounter_type=encounter.get("encounter_type"),
                provider_name=encounter.get("provider_name"),
                organization=encounter.get("organization"),
                patient_summary=encounter.get("patient_summary"),
                structured_changes=encounter.get("structured_changes"),
                source_document_id=document_id,
                created_by=created_by,
            )
            written["encounters"].append(row)

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE document_extractions
                   SET confirmed = true, confirmed_at = now(), confirmed_fields = $2::jsonb
                 WHERE document_id = $1
                """,
                document_id, json.dumps(confirmed_fields, default=str),
            )
        await doc_service.update_status(document_id, "confirmed")

        # Confirming a document changed canonical facts — refresh the
        # cached state snapshot so the next question the patient asks
        # reflects it immediately rather than the previous snapshot.
        try:
            from src.api.services.patient.patient_state_service import get_patient_state_service
            await get_patient_state_service().build_state(patient_profile_id)
        except Exception:
            pass

        return {"success": True, "written": written}


_validator: Optional[PatientDocumentValidator] = None


def get_patient_document_validator() -> PatientDocumentValidator:
    global _validator
    if _validator is None:
        _validator = PatientDocumentValidator()
    return _validator
