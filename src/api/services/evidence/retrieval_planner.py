"""
Retrieval Planner (Phase 4)

intent + retrieval_features -> which Qdrant collections to search and what
to boost — the "retrieval plan" the architecture review's target pipeline
names explicitly (section 16). This is where patient context stops being
"more query words" and starts being structured selection: see section 17
of the review for the metallic-taste/FOLFOX example this is built to
handle correctly (search patient_education + medication_knowledge, boost
"dysgeusia", "FOLFOX", "oxaliplatin"; don't just embed a longer sentence).

Collection order in `collections` is priority order, followed at query
time by multi_corpus_retriever and reflected in evidence_packet_builder's
ordering — matching the fallback hierarchy in the architecture review,
section 27: patient education -> medication -> guideline -> literature,
with PubMed live search staying the last resort outside this module
entirely (patient_chat_service still owns that fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.config import settings
from src.api.services.evidence.patient_context_service import (
    INTENT_MEDICATION, INTENT_SYMPTOM, INTENT_NUTRITION,
    INTENT_TREATMENT, INTENT_DIAGNOSIS, INTENT_GENERAL,
)

# intent -> ordered list of collection settings to search, highest
# priority first. Literature always included last: it's the deepest and
# least patient-friendly corpus, appropriate as a base layer under
# purpose-built patient material rather than the first thing searched.
_INTENT_COLLECTIONS = {
    INTENT_MEDICATION: [
        "qdrant_medication_collection", "qdrant_patient_education_collection", "qdrant_collection",
    ],
    INTENT_SYMPTOM: [
        "qdrant_patient_education_collection", "qdrant_guideline_collection", "qdrant_collection",
    ],
    INTENT_NUTRITION: [
        "qdrant_patient_education_collection", "qdrant_collection",
    ],
    INTENT_TREATMENT: [
        "qdrant_patient_education_collection", "qdrant_guideline_collection", "qdrant_collection",
    ],
    INTENT_DIAGNOSIS: [
        "qdrant_patient_education_collection", "qdrant_collection",
    ],
    INTENT_GENERAL: [
        "qdrant_patient_education_collection", "qdrant_collection",
    ],
}


@dataclass
class RetrievalPlan:
    intent: str
    collections: List[str] = field(default_factory=list)
    boost_terms: List[str] = field(default_factory=list)
    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    # Structured patient values, one list per applicability_scorer.py
    # component (cancer_types/treatment_modalities/regimens/drugs/
    # symptoms/treatment_phase) — boost_terms is a flat bag of words for
    # the embedding query text; this is the same information kept
    # separated by axis so the scorer can match each component against
    # the specific thing it's about instead of one blended term list.
    patient_values: Dict[str, List[str]] = field(default_factory=dict)


def _derive_treatment_phase(features: Dict[str, Any]) -> List[str]:
    """Best-effort single-value 'phase' axis for applicability_scorer.py.
    A direct nutrition-assessment care_phase (active_treatment/
    survivorship/prevention — see nutrition_assessment_service.py's
    VALID_CARE_PHASES, which shares its vocabulary with metadata_
    classifier.py's VALID_PHASES) is preferred when present; otherwise
    an active regimen/modality on the record is enough to infer
    active_treatment. Anything else is left unspecified (empty list),
    which the scorer treats as neutral rather than guessing."""
    if features.get("nutrition_care_phase"):
        return [features["nutrition_care_phase"]]
    if features.get("active_chemotherapy") or features.get("treatment_modalities"):
        return ["active_treatment"]
    return []


def build_plan(intent: str, retrieval_features: Dict[str, Any]) -> RetrievalPlan:
    features = retrieval_features or {}
    setting_names = _INTENT_COLLECTIONS.get(intent, _INTENT_COLLECTIONS[INTENT_GENERAL])
    collections = []
    for name in setting_names:
        value = getattr(settings, name, None)
        if value and value not in collections:
            collections.append(value)

    # Soft boosts: the patient's actual regimen/agents/symptoms, not just
    # the question text — this is the "patient context becomes structured
    # selection, not more query words" behaviour the review asks for.
    boost_terms: List[str] = []
    boost_terms.extend(features.get("active_agents") or [])
    boost_terms.extend(features.get("regimens") or [])
    boost_terms.extend(features.get("symptoms") or [])
    boost_terms.extend(features.get("metastatic_sites") or [])
    boost_terms.extend(features.get("inferred_terms") or [])

    hard_constraints: Dict[str, Any] = {}
    if features.get("comorbidities"):
        hard_constraints["comorbidities"] = features["comorbidities"]

    patient_values: Dict[str, List[str]] = {
        "cancer_types": list(features.get("cancer_types") or []),
        "treatment_modalities": list(features.get("treatment_modalities") or []),
        "regimens": [r for r in (features.get("regimens") or []) if r],
        "drugs": list(features.get("active_agents") or []),
        "symptoms": list(features.get("symptoms") or []),
        "treatment_phase": _derive_treatment_phase(features),
    }

    return RetrievalPlan(
        intent=intent,
        collections=collections,
        boost_terms=[t for t in dict.fromkeys(boost_terms) if t][:8],  # dedupe, cap
        hard_constraints=hard_constraints,
        patient_values=patient_values,
    )
