"""
Patient Context Service (Phase 4)

Turns a raw patient message into (a) an intent label and (b) the patient's
current retrieval-relevant state, so retrieval_planner can decide *which*
corpora to search and *what* to boost, instead of the old approach of
concatenating known facts onto the question as extra query words (see
patient_chat_service.py's previous `seed = " ".join(...)` — replaced by
this module's get_patient_context + retrieval_planner.build_plan).

Intent classification is deliberately not an LLM call: it runs before
retrieval on every message, needs to be fast, and the keyword sets below
cover the common patient-question shapes well enough that a wrong label
only costs a slightly worse corpus mix, never a wrong answer — the
generation prompt itself still has the full patient message.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

INTENT_MEDICATION = "medication_explainer"
INTENT_SYMPTOM = "symptom_management"
INTENT_NUTRITION = "nutrition"
INTENT_TREATMENT = "treatment_explainer"
INTENT_DIAGNOSIS = "diagnosis_explainer"
INTENT_GENERAL = "general"

_INTENT_PATTERNS = [
    (INTENT_NUTRITION, re.compile(
        r"\b(eat|food|appetite|taste|metallic|nausea|nutrition|diet|weight loss|"
        r"can'?t eat|swallow)\b", re.IGNORECASE)),
    (INTENT_MEDICATION, re.compile(
        r"\b(medication|medicine|drug|pill|dose|dosage|side effects? of|pembrolizumab|"
        r"nivolumab|chemo(?:therapy)? drug|what is \w+mab\b|what is \w+nib\b)\b",
        re.IGNORECASE)),
    (INTENT_SYMPTOM, re.compile(
        r"\b(symptom|pain|fatigue|tired|numbness|neuropathy|rash|diarrhea|"
        r"constipation|fever|swelling|is this (?:normal|serious))\b", re.IGNORECASE)),
    (INTENT_TREATMENT, re.compile(
        r"\b(treatment|therapy|regimen|chemo(?:therapy)?|radiation|surgery|"
        r"immunotherapy|what happens (?:during|after))\b", re.IGNORECASE)),
    (INTENT_DIAGNOSIS, re.compile(
        r"\b(stage|diagnosis|biomarker|what does .* mean|grade \d|pathology|report)\b",
        re.IGNORECASE)),
]


def classify_intent(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return INTENT_GENERAL
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return INTENT_GENERAL


class PatientContextService:
    async def get_context(self, patient_user_id: str) -> Dict[str, Any]:
        """Best-effort patient context: {} for an account with no profile
        yet (brand-new signup) or any lookup failure — retrieval_planner
        treats an empty context as 'no personalization available', not an
        error, since most of this product works for a patient who hasn't
        built a record yet."""
        try:
            from src.api.services.patient.patient_profile_service import (
                get_patient_profile_service,
            )
            profile = await get_patient_profile_service().get_by_user(patient_user_id)
            if not profile:
                return {}

            from src.api.services.patient.patient_state_service import (
                get_patient_state_service,
            )
            state_service = get_patient_state_service()
            snapshot = await state_service.get_latest_snapshot(profile["id"])
            if snapshot is None:
                built = await state_service.build_state(profile["id"])
                state, features = built["state"], built["retrieval_features"]
            else:
                state, features = snapshot.get("state", {}), snapshot.get("retrieval_features", {})

            return {
                "patient_profile_id": profile["id"],
                "state": state,
                "retrieval_features": features,
            }
        except Exception:
            logger.warning(
                "[PatientContext] context load failed for user %s, answering "
                "without personalization", patient_user_id, exc_info=True,
            )
            return {}


_service: Optional[PatientContextService] = None


def get_patient_context_service() -> PatientContextService:
    global _service
    if _service is None:
        _service = PatientContextService()
    return _service
