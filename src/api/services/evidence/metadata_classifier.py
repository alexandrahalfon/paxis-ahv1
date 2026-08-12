"""
Metadata Classifier (evidence ingestion front-half)

Structured extraction of the applicability contract attached to every
evidence chunk — content_type/intents/topics/cancer_types/
treatment_modalities/regimens/drugs/symptoms/treatment_phases/
age_groups/can_support_* — matching evidence_ingestion_service.py's
"applicability" payload field, so a chunk's metadata no longer has to be
hand-supplied by the caller.

Its job is classification, not clinical judgment: it reads a chunk of
already-vetted source text (only ever called against source_registry.py-
approved content) and reports which of a fixed set of tags the text
actually supports — never invents specificity the source doesn't state.

The "don't invent specificity" rule is enforced twice, not just by
prompt instruction: every cancer_type/drug/regimen/symptom the model
returns is passed through clinical_normalization.py's normalizers, and
anything that doesn't resolve to a known canonical term is DROPPED
rather than trusted verbatim. An empty/unmatched cancer_types list
becomes ["all"] — the same "unknown -> general" convention
source_registry.py's seed sources already use — so a page that never
names a cancer type doesn't get a fabricated one.

Not executed against a live OpenAI endpoint as part of this change; the
canonicalization/filtering logic (the part least trustworthy to leave
unverified) is covered by a local test against a synthetic model
response — see the commit this shipped with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.config import settings
from src.api.services.patient.clinical_normalization import (
    CANCER_SITES, SYMPTOM_PATTERNS, REGIMEN_EXPANSIONS,
    normalize_cancer_site, normalize_symptom, normalize_drug_name, expand_regimen,
)

logger = logging.getLogger(__name__)

VALID_CONTENT_TYPES = {"patient_education", "medication_knowledge", "clinical_guideline", "other"}
VALID_INTENTS = {
    "nutrition", "symptom_management", "medication_explainer", "diagnosis_explainer",
    "treatment_explainer", "side_effect", "missed_dose", "drug_interaction",
    "emotional_support", "practical_financial", "general",
}
VALID_MODALITIES = {
    "chemotherapy", "immunotherapy", "targeted_therapy", "hormone_therapy",
    "radiation", "surgery", "transplant_cellular_therapy",
}
VALID_PHASES = {"active_treatment", "survivorship", "prevention", "end_of_life"}
VALID_AGE_GROUPS = {"adult", "pediatric"}

_KNOWN_CANCER_TYPES = set(CANCER_SITES.keys())
_KNOWN_SYMPTOMS = {canonical for _, canonical in SYMPTOM_PATTERNS}
_KNOWN_REGIMENS = set(REGIMEN_EXPANSIONS.keys())

_CLASSIFY_PROMPT = """You classify patient-education/medical content for a retrieval system. \
Given the title and text below, output ONLY a JSON object with this exact shape:

{
  "content_type": "patient_education" | "medication_knowledge" | "clinical_guideline" | "other",
  "intents": [ zero or more of: nutrition, symptom_management, medication_explainer,
               diagnosis_explainer, treatment_explainer, side_effect, missed_dose,
               drug_interaction, emotional_support, practical_financial, general ],
  "topics": [ short free-text topic phrases actually discussed, e.g. "taste_changes" ],
  "cancer_types": [ cancer types EXPLICITLY named in the text; empty list if none named ],
  "treatment_modalities": [ zero or more of: chemotherapy, immunotherapy, targeted_therapy,
               hormone_therapy, radiation, surgery, transplant_cellular_therapy —
               only if explicitly discussed ],
  "regimens": [ regimen names EXPLICITLY named, e.g. "FOLFOX" ],
  "drugs": [ drug names EXPLICITLY named, brand or generic, as written in the text ],
  "symptoms": [ symptoms EXPLICITLY discussed, in the source's own words ],
  "treatment_phases": [ zero or more of: active_treatment, survivorship, prevention, end_of_life ],
  "age_groups": [ zero or more of: adult, pediatric — only if the text specifies ],
  "can_support_self_care": true|false,
  "can_support_triage": true|false,
  "can_support_dose_change": true|false
}

Hard rule: only include a value when the text actually supports it. If the text does not name a
specific cancer type, drug, or regimen, leave that list EMPTY — do not infer or guess based on what
is typically associated with the topic. can_support_triage and can_support_dose_change should almost
always be false; true only if the text is itself an authoritative clinical guideline or drug-label
section written specifically to guide that decision (e.g. a DailyMed "missed dose" section).

Reply with the JSON object only, no commentary."""


@dataclass
class ClassificationResult:
    content_type: str = "patient_education"
    intents: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    cancer_types: List[str] = field(default_factory=list)
    treatment_modalities: List[str] = field(default_factory=list)
    regimens: List[str] = field(default_factory=list)
    drugs: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    treatment_phases: List[str] = field(default_factory=list)
    age_groups: List[str] = field(default_factory=list)
    can_support_self_care: bool = False
    can_support_triage: bool = False
    can_support_dose_change: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type,
            "intents": self.intents,
            "topics": self.topics,
            "cancer_types": self.cancer_types,
            "treatment_modalities": self.treatment_modalities,
            "regimens": self.regimens,
            "drugs": self.drugs,
            "symptoms": self.symptoms,
            "treatment_phases": self.treatment_phases,
            "age_groups": self.age_groups,
            "can_support_self_care": self.can_support_self_care,
            "can_support_triage": self.can_support_triage,
            "can_support_dose_change": self.can_support_dose_change,
        }


def _client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


def _canonicalize_cancer_types(raw: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        term = normalize_cancer_site(str(item)).canonical
        if term and term in _KNOWN_CANCER_TYPES and term not in out:
            out.append(term)
    return out


def _canonicalize_symptoms(raw: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        term = normalize_symptom(str(item)).canonical
        if term and term in _KNOWN_SYMPTOMS and term not in out:
            out.append(term)
    return out


def _canonicalize_drugs(raw: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        norm = normalize_drug_name(str(item))
        if norm["canonical"] and norm["canonical"] not in out:
            out.append(norm["canonical"])
    return out


def _canonicalize_regimens(raw: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        key = str(item).strip().lower()
        if key in _KNOWN_REGIMENS and str(item) not in out:
            out.append(str(item))
    return out


def sanitize(raw: Dict[str, Any]) -> ClassificationResult:
    """The grounding step: takes the model's raw parsed JSON and drops
    everything that doesn't resolve against clinical_normalization.py's
    controlled vocabulary. Split out from classify() so it can be tested
    directly against a synthetic model response, without a live API call."""
    cancer_types = _canonicalize_cancer_types(raw.get("cancer_types", []))
    return ClassificationResult(
        content_type=raw.get("content_type") if raw.get("content_type") in VALID_CONTENT_TYPES
        else "patient_education",
        intents=[i for i in (raw.get("intents") or []) if i in VALID_INTENTS],
        topics=[str(t).strip().lower().replace(" ", "_")[:80] for t in (raw.get("topics") or []) if t][:10],
        # A page that names no cancer type is general-audience content —
        # ["all"] is the correct default, not an empty (i.e. "matches
        # nothing") list. See source_registry.py's same convention.
        cancer_types=cancer_types or ["all"],
        treatment_modalities=[m for m in (raw.get("treatment_modalities") or []) if m in VALID_MODALITIES],
        regimens=_canonicalize_regimens(raw.get("regimens", [])),
        drugs=_canonicalize_drugs(raw.get("drugs", [])),
        symptoms=_canonicalize_symptoms(raw.get("symptoms", [])),
        treatment_phases=[p for p in (raw.get("treatment_phases") or []) if p in VALID_PHASES],
        age_groups=[a for a in (raw.get("age_groups") or []) if a in VALID_AGE_GROUPS],
        can_support_self_care=bool(raw.get("can_support_self_care")),
        can_support_triage=bool(raw.get("can_support_triage")),
        can_support_dose_change=bool(raw.get("can_support_dose_change")),
    )


def classify(text: str, title: str = "") -> ClassificationResult:
    excerpt = f"Title: {title}\n\n{text}"[:6000]
    try:
        resp = _client().chat.completions.create(
            model=settings.openai_mini_model or "gpt-4o-mini",
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CLASSIFY_PROMPT},
                {"role": "user", "content": excerpt},
            ],
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning("[MetadataClassifier] classification failed, returning a minimal safe default: %s", e)
        return ClassificationResult(cancer_types=["all"])

    return sanitize(raw)
