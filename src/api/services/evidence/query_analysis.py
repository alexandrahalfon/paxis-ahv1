"""
QueryAnalysis (2026-08-12 patient/physician convergence, Sprint C item
12)

The typed, audience-neutral shape both the patient and physician paths
should analyze a query into — the first of the "freeze the shared
contracts" trio alongside EvidenceCandidate and EvidencePacket. Per the
convergence plan: "do not immediately replace all the legacy query
classifiers... build an adapter... eventually you can consolidate the
classifiers, but beta does not require that refactor."

This module is exactly that adapter, not a new extractor. It calls the
existing, already-cheap (regex-based, no LLM) classifiers and reshapes
their output into one QueryAnalysis:

  - from_physician_query() adapts query_structuring_service.
    structure_query_fast() (QueryStructure — site/histology/stage/TNM/
    biomarkers/treatment/clinical-history) and clinical_entity_extractor.
    ClinicalEntityExtractor().extract() (ClinicalProfile — a second,
    differently-shaped but overlapping extraction covering treatments/
    clinical_concepts/anatomic_sites) for audience="physician".
  - from_patient_message() adapts patient_context_service.
    classify_intent() (the "patient query parser") for
    audience="patient". Patient chat's actual structured clinical
    context comes from patient_state_service via a SEPARATE EvidencePacket
    field (selected_patient_context), not from parsing the raw chat
    message, so the clinical axis fields here are deliberately empty —
    this adapter's job for the patient side is just the audience/intent
    pairing, not a second attempt at clinical extraction.

What this deliberately does NOT adapt from, and why:
  - PTO frames (pto_frame_builder.py/pto_retriever.py) are structured
    Patient→Treatment→Outcome objects, not a flat intent classifier —
    their content belongs in EvidencePacket.pto_frame (already a field,
    added in Sprint A item 2), not folded into QueryAnalysis's flat
    lists.
  - query_intent_service.py's full IntentAnalysisResult pipeline (trial
    matching, follow-up-option generation, an LLM call) is a broader
    orchestration service, not a cheap classifier — adapting from it here
    would mean either running that whole expensive pipeline just to read
    one field off it, or reaching into its internals. structure_query_
    fast()'s question_focus (dose/survival/indication/toxicity/...) is
    the cheap proxy used for `intent` instead.
  - labs: none of the adapted sources extract lab values from free
    query text today (lab handling that DOES exist —
    lab_interpretation.py — is about a patient's canonical recorded
    labs, a different data source entirely). Left as an empty list
    rather than faked.

Nothing calls either from_*() function yet — same as every other Sprint
A/B convergence piece landing before its consumer. Sprint C's physician
orchestrator (item 20) is what will actually call from_physician_query().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryAnalysis:
    audience: str  # "patient" | "physician"
    intent: str
    cancer_types: List[str] = field(default_factory=list)
    histologies: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    biomarkers: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    drugs: List[str] = field(default_factory=list)
    regimens: List[str] = field(default_factory=list)
    labs: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    treatment_line_focus: Optional[str] = None
    prior_treatment_focus: List[str] = field(default_factory=list)
    patient_specific: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _dedup(values: Any) -> List[str]:
    """Flattens a mix of single values/lists/None into a deduped,
    order-preserving list of non-empty strings."""
    out: List[str] = []
    seen = set()
    items = values if isinstance(values, (list, tuple, set)) else [values]
    for v in items:
        if not v:
            continue
        s = str(v).strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def from_physician_query(
    query_text: str,
    intent: Optional[str] = None,
) -> QueryAnalysis:
    """Adapter for audience="physician". See module docstring for
    exactly which legacy classifiers this reshapes and why. `intent`
    lets a caller pass an already-known intent label (e.g. from Sprint
    C's physician context selector) instead of relying on
    structure_query_fast()'s question_focus as the default."""
    from src.api.services.query_structuring_service import structure_query_fast
    from src.api.services.clinical_entity_extractor import get_clinical_entity_extractor

    structure = structure_query_fast(query_text)
    profile = get_clinical_entity_extractor().extract(query_text)

    return QueryAnalysis(
        audience="physician",
        intent=intent or structure.question_focus or "general",
        cancer_types=_dedup([structure.cancer.site, structure.cancer.site_detail, profile.cancer_type]),
        histologies=_dedup([structure.cancer.histology, profile.cancer_subtype]),
        stages=_dedup([structure.cancer.stage, structure.cancer.get_tnm_string(), profile.stage, profile.tnm]),
        biomarkers=_dedup(list(structure.cancer.biomarkers) + list(profile.biomarkers)),
        symptoms=[],
        # ClinicalProfile.treatments mixes modalities (surgery/RT/chemo)
        # and specific agents -- coarser than a true drug/regimen split,
        # which none of the adapted sources provide today. Both drugs
        # and regimens read from the same field until a real split
        # exists; noted here rather than silently duplicated without
        # explanation.
        drugs=_dedup(profile.treatments),
        regimens=_dedup(profile.treatments),
        labs=[],
        outcomes=_dedup(profile.clinical_concepts),
        # No dedicated line-of-therapy extractor exists in the adapted
        # sources -- treatment.setting (adjuvant/neoadjuvant/definitive/
        # palliative) is the closest available proxy.
        treatment_line_focus=structure.treatment.setting,
        prior_treatment_focus=_dedup(structure.treatment.prior_treatments),
        patient_specific=structure.has_patient_context,
    )


def from_patient_message(message: str, intent: str) -> QueryAnalysis:
    """Adapter for audience="patient". See module docstring for why the
    clinical axis fields stay empty here -- patient_specific is always
    True since every patient-chat turn is inherently about that one
    patient, whether or not their record is linked."""
    return QueryAnalysis(
        audience="patient",
        intent=intent,
        patient_specific=True,
    )
