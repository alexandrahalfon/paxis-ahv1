"""
Patient Signal Service

Passive patient-profile capture: detects when a free-text query (from any
mode — chat, patient matching, etc.) contains real patient-case details, and
decides whether to attach them to a patient record automatically, ask the
physician which patient it belongs to, or ignore it entirely.

Deliberately reuses two already-built services instead of inventing new
extraction:
- unified_router.get_unified_router() — module classification. Only
  QueryModule.PATIENT_SPECIFIC queries are considered at all;
  GENERAL_KNOWLEDGE and EVIDENCE_EXPLORATION queries are ignored outright,
  which removes most of the ambiguity before any patient-matching logic runs.
- query_classifier_service.get_query_classifier_service() — the existing
  regex/LLM structured field extraction (age, gender, cancer_type,
  cancer_location, tnm_t/n/m, overall_stage) already used by
  saved_cases_service.py for the same kind of clinical free text.

Decision policy (per Aysha, 2026-07-13):
- Only PATIENT_SPECIFIC queries above MODULE_CONFIDENCE_MIN are considered.
- If an active_patient_id is supplied and its latest diagnosis doesn't
  conflict with what was just extracted -> silent_attach (no prompt).
- If active_patient_id is supplied but conflicts -> ask.
- If no active_patient_id -> always ask (create new vs. attach to an
  existing patient) — never silently creates a patient record.

No new ML model — conflict/candidate matching below is a plain structured
field comparison, same spirit as the planned pattern_diff_service.
"""

from typing import Any, Dict, List, Optional

from .patient_service import get_patient_service

MODULE_CONFIDENCE_MIN = 0.55

# StructuredQuery fields (query_classifier_service.py) mapped to
# patient_diagnosis columns (patient_db.py). "age"/"sex" aren't diagnosis
# columns — age isn't stored at all (see note in apply()), sex maps to the
# patients table itself, not patient_diagnosis.
_DIAGNOSIS_FIELDS = ("cancer_site", "histology", "stage", "tnm_t", "tnm_n", "tnm_m")


def _extracted_fields(structured) -> Dict[str, Any]:
    """Pull the subset of StructuredQuery fields this service cares about.
    Returns {} if nothing concrete was extracted (caller should ignore)."""
    fields = {
        "cancer_site": structured.cancer_location,
        "histology": structured.cancer_type,
        "stage": structured.overall_stage,
        "tnm_t": structured.tnm_t,
        "tnm_n": structured.tnm_n,
        "tnm_m": structured.tnm_m,
        "sex": structured.gender,
        "age": structured.age,
    }
    return {k: v for k, v in fields.items() if v}


def _conflicts(diagnosis: Optional[Dict[str, Any]], extracted: Dict[str, Any]) -> bool:
    """Conservative conflict check between an existing patient's latest
    diagnosis and newly-extracted fields. Only flags a conflict when both
    sides state a value for the same axis and neither contains the other —
    false negatives (missed conflicts) are preferred over false positives
    (interrupting the physician unnecessarily), since Aysha's answer was
    "only ask when ambiguous."
    """
    if not diagnosis:
        return False
    for axis in ("cancer_site", "histology"):
        old = (diagnosis.get(axis) or "").strip().lower()
        new = (extracted.get(axis) or "").strip().lower()
        if old and new and old not in new and new not in old:
            return True
    return False


async def _find_candidate_patients(physician_id: str, extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score the physician's existing patients against extracted fields.
    Plain substring overlap, not a model — see module docstring."""
    service = get_patient_service()
    patients = await service.list_patients(physician_id, limit=100)

    scored = []
    for p in patients:
        diagnosis = await service.get_latest_diagnosis(p["id"])
        score = 0
        if diagnosis:
            for axis in ("cancer_site", "histology", "stage"):
                old = (diagnosis.get(axis) or "").strip().lower()
                new = (extracted.get(axis) or "").strip().lower()
                if old and new and (old in new or new in old):
                    score += 1
        if score > 0:
            scored.append({
                "patient_id": p["id"],
                "name": f"{p['first_name']} {p['last_name']}",
                "score": score,
            })
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:5]


class PatientSignalService:

    def is_patient_specific(self, query_text: str) -> bool:
        """Cheap, no-auth-required pre-check: routing classification only
        (unified_router is local/regex-based, no LLM call — safe to expose
        to anonymous traffic). Does NOT run query_classifier_service's LLM
        extraction, which costs money per call and must stay behind auth.

        Used for the pre-account flow: an anonymous visitor gets nudged to
        create an account when this returns True, but the actual field
        extraction only happens after they're authenticated (see evaluate()),
        via /patients/signal once the frontend replays the queued raw text.
        """
        from .unified_router import get_unified_router, QueryModule

        router = get_unified_router()
        routing = router.route_query(query_text)
        return routing.module == QueryModule.PATIENT_SPECIFIC and routing.module_confidence >= MODULE_CONFIDENCE_MIN

    async def evaluate(
        self,
        physician_id: str,
        query_text: str,
        active_patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from .unified_router import get_unified_router, QueryModule
        from .query_classifier_service import get_query_classifier_service

        router = get_unified_router()
        routing = router.route_query(query_text)

        if routing.module != QueryModule.PATIENT_SPECIFIC or routing.module_confidence < MODULE_CONFIDENCE_MIN:
            return {"action": "ignore", "reason": "not_patient_specific"}

        classifier = get_query_classifier_service()
        structured = classifier.classify_query(query_text)
        extracted = _extracted_fields(structured)

        if not extracted:
            return {"action": "ignore", "reason": "no_fields_extracted"}

        patient_service = get_patient_service()

        if active_patient_id:
            active_patient = await patient_service.get_patient(active_patient_id, physician_id)
            if active_patient:
                diagnosis = await patient_service.get_latest_diagnosis(active_patient_id)
                if not _conflicts(diagnosis, extracted):
                    return {
                        "action": "silent_attach",
                        "target_patient_id": active_patient_id,
                        "target_patient_name": f"{active_patient['first_name']} {active_patient['last_name']}",
                        "extracted": extracted,
                        "raw_text": query_text,
                    }
                return {
                    "action": "ask",
                    "reason": "conflict_with_active_patient",
                    "extracted": extracted,
                    "raw_text": query_text,
                    "candidates": [{
                        "patient_id": active_patient_id,
                        "name": f"{active_patient['first_name']} {active_patient['last_name']}",
                    }],
                }

        candidates = await _find_candidate_patients(physician_id, extracted)
        return {
            "action": "ask",
            "reason": "no_active_patient",
            "extracted": extracted,
            "raw_text": query_text,
            "candidates": candidates,
        }

    async def apply(
        self,
        physician_id: str,
        action: str,
        extracted: Dict[str, Any],
        raw_text: str,
        patient_id: Optional[str] = None,
        new_patient_first_name: Optional[str] = None,
        new_patient_last_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Physician-confirmed (or auto-confident) write. Reuses
        patient_service's existing CRUD methods — no new DB-writing code
        here, just orchestration.

        Note on age: StructuredQuery.age (e.g. "68 yo") is intentionally
        NOT converted into date_of_birth — a computed DOB from a stated age
        would look precise (an exact date) while actually being a guess,
        which is worse than just leaving it blank for a passively-created
        record. The raw mention still lives in diagnosis.raw_text.
        """
        patient_service = get_patient_service()

        if action == "ignore":
            return {"success": True, "ignored": True}

        reused_existing = False
        if action == "create_new":
            if not new_patient_first_name or not new_patient_last_name:
                return {"success": False, "error": "Name required to create a new patient."}
            # Reuse an existing patient with the same name instead of
            # silently creating a duplicate — this is what was producing
            # two "Jane Doe" records when the create-and-attach banner was
            # used more than once for the same patient.
            existing = await patient_service.find_patient_by_name(
                physician_id, new_patient_first_name, new_patient_last_name
            )
            if existing:
                patient = existing
                patient_id = patient["id"]
                reused_existing = True
            else:
                patient = await patient_service.create_patient(
                    physician_id=physician_id,
                    first_name=new_patient_first_name,
                    last_name=new_patient_last_name,
                    sex=extracted.get("sex"),
                )
                patient_id = patient["id"]
        elif action == "attach":
            if not patient_id:
                return {"success": False, "error": "patient_id required to attach."}
            if not await patient_service.get_patient(patient_id, physician_id):
                return {"success": False, "error": "Patient not found."}
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        diagnosis = None
        if any(k in extracted for k in _DIAGNOSIS_FIELDS):
            diagnosis = await patient_service.add_diagnosis(
                patient_id, physician_id,
                cancer_site=extracted.get("cancer_site"),
                histology=extracted.get("histology"),
                stage=extracted.get("stage"),
                tnm_t=extracted.get("tnm_t"),
                tnm_n=extracted.get("tnm_n"),
                tnm_m=extracted.get("tnm_m"),
                raw_text=raw_text,
            )

        return {
            "success": True,
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "reused_existing": reused_existing,
            "patient_name": f"{patient['first_name']} {patient['last_name']}" if action == "create_new" else None,
        }


_service: Optional[PatientSignalService] = None


def get_patient_signal_service() -> PatientSignalService:
    global _service
    if _service is None:
        _service = PatientSignalService()
    return _service
