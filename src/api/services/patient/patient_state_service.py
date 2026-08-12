"""
Patient State Service (Phase 1 centerpiece)

Builds the canonical "as of now" view of a patient — diagnoses (including
recurrence/progression as their own entries), tumor profile, active
treatment and its constituent agents, active medications, recent labs,
symptoms, nutrition, care-team instructions, comorbidities — plus the
derived retrieval_features the evidence-retrieval layer (Phase 4) actually
queries on. This is the patient_state_service.py the architecture review
describes: raw facts are not always the concepts the literature talks
about (ANC 0.7 -> "neutropenia"), so the two are kept as separate JSONB
fields on the same snapshot row.

Two read paths merge here when a legacy care-team link exists:
  1. Phase 1 tables, keyed by patient_profile_id (primary source; used
     whenever it has data for a given field).
  2. The physician-owned chart (patient_diagnosis / patient_biomarkers /
     patient_treatment_history, keyed by the legacy patients.id) via
     patient_service.get_patient_full, for patients whose only
     recorded data is still on their clinician's side.

Every call recomputes and inserts a new patient_state_snapshots row rather
than updating one in place — append-only, like the timeline it's built
from — so "what did we think this patient's state was on Aug 3" stays
answerable, and the diff/change-detection layer described in the review
(pattern_diff_service) has two snapshots to compare once it exists.

state["active_diagnosis"] (singular) is kept for backward compatibility
with existing readers (evidence_packet_builder.py, patient-dashboard.html)
— it is the first active/primary diagnosis. state["active_diagnoses"]
(plural) is the full multi-primary/recurrence/progression list Phase 1
calls for; prefer it in any new code.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from src.api.services.patient_db import get_patient_db
from src.api.services.patient.patient_profile_service import get_patient_profile_service
from src.api.services.patient.patient_care_team_service import get_patient_care_team_service
from src.api.services.patient.diagnosis_service import get_diagnosis_service
from src.api.services.patient.treatment_service import get_treatment_service
from src.api.services.patient.medication_service import get_medication_service
from src.api.services.patient.lab_service import get_lab_service
from src.api.services.patient.conditions_service import get_conditions_service
from src.api.services.patient.vitals_service import get_vitals_service
from src.api.services.patient.tumor_profile_service import get_tumor_profile_service
from src.api.services.patient.symptom_observation_service import get_symptom_observation_service
from src.api.services.patient.nutrition_assessment_service import get_nutrition_assessment_service
from src.api.services.patient.care_team_instruction_service import get_care_team_instruction_service
from src.api.services.patient.clinical_normalization import normalize_cancer_site


def _age_from_dob(dob: Any) -> Optional[int]:
    if not dob:
        return None
    try:
        born = date.fromisoformat(str(dob)[:10])
        today = date.today()
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return age if 0 <= age <= 130 else None
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PatientStateService:
    async def _legacy_chart(self, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        """Pull the physician-owned chart via a primary legacy care-team
        link, if one exists. Best-effort: any failure here must not break
        state building from the patient's own Phase 1 data."""
        try:
            links = await get_patient_care_team_service().list_care_team(patient_profile_id)
            legacy = next((l for l in links if l.get("legacy_patient_record_id")), None)
            if not legacy:
                return None
            from src.api.services.patient_service import get_patient_service
            return await get_patient_service().get_patient_full(
                legacy["legacy_patient_record_id"], legacy["physician_id"]
            )
        except Exception:
            logger.warning(
                "[PatientState] legacy chart lookup failed for profile %s",
                patient_profile_id, exc_info=True,
            )
            return None

    async def build_state(
        self, patient_profile_id: str, persist: bool = True
    ) -> Dict[str, Any]:
        profile = await get_patient_profile_service().get_by_id(patient_profile_id)
        if not profile:
            raise ValueError("Patient profile not found")

        legacy = await self._legacy_chart(patient_profile_id)

        # ── Diagnoses (multi-primary / recurrence / progression) ────────
        active_diagnoses = await get_diagnosis_service().get_active_diagnoses(patient_profile_id)
        biomarkers = await get_diagnosis_service().list_biomarkers(patient_profile_id)
        if not active_diagnoses and legacy and legacy.get("diagnosis"):
            ld = legacy["diagnosis"]
            active_diagnoses = [{
                "cancer_site": ld.get("cancer_site"), "histology": ld.get("histology"),
                "stage": ld.get("stage"), "tnm_t": ld.get("tnm_t"), "tnm_n": ld.get("tnm_n"),
                "tnm_m": ld.get("tnm_m"), "raw_text": ld.get("raw_text"),
                "diagnosis_type": "primary", "status": "active",
            }]
        if not biomarkers and legacy and legacy.get("biomarkers"):
            biomarkers = legacy["biomarkers"]
        primary_diagnosis = next(
            (d for d in active_diagnoses if d.get("diagnosis_type") == "primary"),
            (active_diagnoses[0] if active_diagnoses else None),
        )

        tumor_profile = await get_tumor_profile_service().get_latest(patient_profile_id)

        # ── Treatment ─────────────────────────────────────────────────
        episodes = await get_treatment_service().list_episodes(patient_profile_id)
        active_episodes = [e for e in episodes if e.get("status") == "active"]
        episode_agents: Dict[str, List[str]] = {}
        for ep in active_episodes:
            if ep.get("id"):
                agents = await get_treatment_service().list_agents(ep["id"], patient_profile_id)
                episode_agents[ep["id"]] = [a["agent_name"] for a in agents]

        if not episodes and legacy and legacy.get("treatment_history"):
            # No Phase 1 episode recorded yet — fall back to the legacy
            # regimen strings so active_treatment isn't empty just because
            # the patient hasn't re-entered what their chart already has.
            for t in legacy["treatment_history"]:
                if t.get("status") == "active":
                    active_episodes.append({
                        "id": None, "regimen": t.get("regimen") or t.get("treatment_type"),
                        "line_of_therapy": t.get("line_of_therapy"),
                        "start_date": t.get("start_date"),
                    })

        medications = await get_medication_service().list_medications(
            patient_profile_id, active_only=True
        )
        comorbidities = await get_conditions_service().list_comorbidities(patient_profile_id)
        allergies_raw = await get_conditions_service().list_allergies(patient_profile_id)
        recent_labs = await get_lab_service().most_recent_by_test(patient_profile_id)

        # ── Nutrition: weight trend (7/30/90d) + latest assessment ──────
        weight_trend = await get_vitals_service().weight_trend_summary(patient_profile_id)
        nutrition_assessment = await get_nutrition_assessment_service().get_latest(patient_profile_id)

        # ── Symptoms: prefer the Phase 1 table, fall back to legacy ─────
        active_symptoms: List[Dict[str, Any]] = []
        try:
            observations = await get_symptom_observation_service().list_observations(
                patient_profile_id, active_only=True, limit=20
            )
            if observations:
                active_symptoms = [
                    {
                        "name": o.get("canonical_symptom") or o.get("raw_text"),
                        "raw_text": o.get("raw_text"), "severity": o.get("severity"),
                        "onset": o.get("onset_date"),
                        "possibly_related_treatment_episode_id":
                            o.get("possibly_related_treatment_episode_id"),
                    }
                    for o in observations
                ]
            else:
                from src.api.services.patient_portal.symptom_service import get_symptom_service
                entries = await get_symptom_service().list_entries(profile["user_id"], limit=10)
                active_symptoms = [
                    {"name": e.get("symptom"), "severity": e.get("severity"), "onset": e.get("noted_on")}
                    for e in entries
                ]
        except Exception:
            logger.warning(
                "[PatientState] symptom lookup failed for profile %s, "
                "continuing without active_symptoms", patient_profile_id, exc_info=True,
            )

        care_instructions = await get_care_team_instruction_service().list_active(patient_profile_id)

        state: Dict[str, Any] = {
            "as_of": _now_iso(),
            "patient_profile_id": patient_profile_id,
            "demographics": {
                "age": _age_from_dob(profile.get("date_of_birth")),
                "sex": profile.get("sex"),
                "preferred_language": profile.get("preferred_language"),
                "timezone": profile.get("timezone"),
            },
            "active_diagnosis": primary_diagnosis,
            "active_diagnoses": active_diagnoses,
            "tumor_profile": tumor_profile,
            "biomarkers": biomarkers,
            "active_treatment": [
                {
                    "regimen": e.get("regimen"),
                    "modality": e.get("modality"),
                    "line_of_therapy": e.get("line_of_therapy"),
                    "start_date": e.get("start_date"),
                    "agents": episode_agents.get(e.get("id"), []),
                }
                for e in active_episodes
            ],
            "active_medications": [
                {"name": m.get("generic_name"), "canonical_name": m.get("canonical_name"),
                 "indication": m.get("indication")}
                for m in medications
            ],
            "active_symptoms": active_symptoms,
            "nutrition": {
                "weight_change_7d_pct": weight_trend.get("7d"),
                "weight_change_30d_pct": weight_trend.get("30d"),
                "weight_change_90d_pct": weight_trend.get("90d"),
                "appetite": (nutrition_assessment or {}).get("appetite"),
                "oral_intake_pct": (nutrition_assessment or {}).get("oral_intake_pct"),
                "swallowing_difficulty": (nutrition_assessment or {}).get("swallowing_difficulty"),
                "feeding_tube": (nutrition_assessment or {}).get("feeding_tube"),
                "diet_restrictions": (nutrition_assessment or {}).get("diet_restrictions"),
                "care_phase": (nutrition_assessment or {}).get("care_phase"),
                "assessed_nutrition_risk": (nutrition_assessment or {}).get("nutrition_risk"),
            },
            "recent_labs": {
                k: {"value": v.get("value_numeric") if v.get("value_numeric") is not None else v.get("value_text"),
                    "unit": v.get("unit"), "collected_at": v.get("collected_at")}
                for k, v in recent_labs.items()
            },
            "comorbidities": [c.get("condition_name") for c in comorbidities],
            "allergies": [a.get("allergen") for a in allergies_raw],
            "intolerances": [
                a.get("allergen") for a in allergies_raw if a.get("allergy_type") == "intolerance"
            ],
            "care_team_instructions": [
                {"text": c.get("instruction_text"), "type": c.get("instruction_type")}
                for c in care_instructions
            ],
            "has_care_team": legacy is not None,
        }

        retrieval_features = self._derive_retrieval_features(state)

        if persist:
            await self._persist_snapshot(patient_profile_id, state, retrieval_features)

        return {"state": state, "retrieval_features": retrieval_features}

    def _derive_retrieval_features(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Concept-level flags the evidence retrieval layer (Phase 4)
        queries on, distinct from the raw facts above (ANC 0.7 ->
        neutropenia). Reuses clinical_inference.run_inference against a
        narrative built from this structured state, so the same inference
        map that already powers clinician-side query structuring
        (unresectable, ICI-refractory, CPS thresholds, ...) powers this
        too instead of a second copy of those rules living here.
        """
        try:
            from src.api.services.clinical_inference import run_inference
        except Exception:
            logger.warning("[PatientState] clinical_inference unavailable, skipping derived flags", exc_info=True)
            run_inference = None

        narrative_parts: List[str] = []
        for dx in state.get("active_diagnoses") or ([state["active_diagnosis"]] if state.get("active_diagnosis") else []):
            narrative_parts.append(" ".join(
                str(v) for v in (
                    dx.get("cancer_site"), dx.get("histology"),
                    dx.get("stage"), dx.get("raw_text"),
                ) if v
            ))
        for tx in state.get("active_treatment") or []:
            narrative_parts.append(" ".join(
                str(v) for v in (tx.get("regimen"), *(tx.get("agents") or [])) if v
            ))
        for b in state.get("biomarkers") or []:
            name = b.get("biomarker_name") if isinstance(b, dict) else None
            if name:
                narrative_parts.append(f"{name} {b.get('value') or ''}".strip())
        narrative_parts.extend(c for c in (state.get("comorbidities") or []) if c)
        narrative = ". ".join(p for p in narrative_parts if p)

        diagnoses = state.get("active_diagnoses") or []
        features: Dict[str, Any] = {
            "active_chemotherapy": any(
                t.get("regimen") for t in (state.get("active_treatment") or [])
            ),
            "regimens": [
                t.get("regimen") for t in (state.get("active_treatment") or []) if t.get("regimen")
            ],
            "active_agents": sorted({
                a for t in (state.get("active_treatment") or []) for a in (t.get("agents") or [])
            }),
            # Feeds applicability_scorer.py's modality/cancer components —
            # canonical_cancer_type is already computed by
            # diagnosis_service.add_diagnosis at write time; the fallback
            # normalization here only covers legacy-chart diagnoses,
            # which never went through that write path.
            "treatment_modalities": sorted({
                t.get("modality") for t in (state.get("active_treatment") or []) if t.get("modality")
            }),
            "cancer_types": sorted({
                d.get("canonical_cancer_type") or normalize_cancer_site(d.get("cancer_site") or "").canonical
                for d in diagnoses if d.get("cancer_site")
            } - {None}),
            "symptoms": [s.get("name") for s in (state.get("active_symptoms") or []) if s.get("name")],
            "comorbidities": state.get("comorbidities") or [],
            "has_multiple_primaries": len([d for d in diagnoses if d.get("diagnosis_type") in ("primary", "second_primary")]) > 1,
            "has_recurrence_or_progression": any(
                d.get("diagnosis_type") in ("recurrence", "progression") for d in diagnoses
            ),
            "metastatic_sites": sorted({
                site for d in diagnoses for site in (d.get("metastatic_sites") or [])
            }),
            "has_active_care_instructions": bool(state.get("care_team_instructions")),
        }

        nutrition = state.get("nutrition") or {}
        # Assessed risk (from a clinician/patient nutrition_assessment)
        # takes precedence over the weight-trend-derived estimate below —
        # a direct assessment is more reliable than an inference from one
        # number.
        if nutrition.get("assessed_nutrition_risk"):
            features["nutrition_risk"] = nutrition["assessed_nutrition_risk"]
        else:
            weight_change = nutrition.get("weight_change_30d_pct")
            if weight_change is not None:
                if weight_change <= -10:
                    features["nutrition_risk"] = "high"
                elif weight_change <= -5:
                    features["nutrition_risk"] = "moderate"
                else:
                    features["nutrition_risk"] = "low"
        if nutrition.get("care_phase"):
            features["nutrition_care_phase"] = nutrition["care_phase"]

        labs = state.get("recent_labs") or {}
        anc = labs.get("anc") or labs.get("absolute neutrophil count")
        if isinstance(anc, dict) and isinstance(anc.get("value"), (int, float)) and anc["value"] < 1.5:
            features["neutropenia_risk"] = True
        platelets = labs.get("platelets") or labs.get("plt")
        if isinstance(platelets, dict) and isinstance(platelets.get("value"), (int, float)) and platelets["value"] < 100:
            features["thrombocytopenia_risk"] = True
        creatinine = labs.get("creatinine") or labs.get("cr")
        if isinstance(creatinine, dict) and isinstance(creatinine.get("value"), (int, float)) and creatinine["value"] > 1.3:
            features["renal_function_context"] = "elevated_creatinine"

        if run_inference and narrative:
            try:
                result = run_inference(narrative, {"primary_cancer": narrative})
                features["trajectory_flags"] = result.trajectory_flags
                # Merge inference-derived met sites with the structured
                # ones already on the diagnosis rows, deduped.
                features["metastatic_sites"] = sorted(
                    set(features["metastatic_sites"]) | set(result.metastatic_sites)
                )
                features["surgical_candidate"] = result.surgical_candidate
                features["inferred_terms"] = result.inferred_terms.get("primary_cancer", [])
            except Exception:
                logger.warning(
                    "[PatientState] run_inference failed for profile %s",
                    state.get("patient_profile_id"), exc_info=True,
                )

        return features

    async def _persist_snapshot(
        self, patient_profile_id: str, state: Dict[str, Any], retrieval_features: Dict[str, Any]
    ) -> None:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO patient_state_snapshots
                    (id, patient_profile_id, state, retrieval_features)
                VALUES ($1, $2, $3::jsonb, $4::jsonb)
                """,
                str(uuid.uuid4()), patient_profile_id,
                json.dumps(state, default=str), json.dumps(retrieval_features, default=str),
            )

    async def get_latest_snapshot(self, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM patient_state_snapshots
                 WHERE patient_profile_id = $1
                 ORDER BY created_at DESC LIMIT 1
                """,
                patient_profile_id,
            )
        if not row:
            return None
        d = dict(row)
        for k in ("state", "retrieval_features"):
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, ValueError):
                    pass
        d["id"] = str(d["id"])
        d["patient_profile_id"] = str(d["patient_profile_id"])
        d["as_of"] = d["as_of"].isoformat() if isinstance(d.get("as_of"), (date, datetime)) else d.get("as_of")
        d["created_at"] = (
            d["created_at"].isoformat() if isinstance(d.get("created_at"), (date, datetime)) else d.get("created_at")
        )
        return d


_service: Optional[PatientStateService] = None


def get_patient_state_service() -> PatientStateService:
    global _service
    if _service is None:
        _service = PatientStateService()
    return _service


async def invalidate_patient_state(patient_profile_id: str) -> None:
    """Best-effort snapshot rebuild after a canonical write.

    get_context() (patient_context_service.py) reads the LATEST
    patient_state_snapshots row and only rebuilds when none exists at
    all — it does not re-check staleness. Before this, a manual write
    (add_diagnosis/add_episode/add_medication/add_observation/
    add_assessment/add_vital/...) left the existing snapshot in place,
    so a patient could add a new treatment and immediately ask a
    question that retrieval answered from the state as it was BEFORE
    that write — see the 2026-08-12 beta audit, "patient state can
    become stale after manual edits". Confirmed document extraction
    already rebuilds via patient_document_validator.py calling
    build_state() after confirmation; every manual-entry write path
    should call this the same way, immediately after its transaction
    commits (never from inside the transaction itself — build_state()
    acquires its own connection from the pool, so calling it before the
    write commits would read pre-write data under READ COMMITTED and
    produce a rebuild that's still stale).

    Never raises: a failed rebuild must not cost the caller their
    successful write. The next successful write (or the next document
    confirmation) will catch up regardless — this narrows the staleness
    window, it doesn't need to be perfect to be a real improvement.
    """
    try:
        await get_patient_state_service().build_state(patient_profile_id)
    except Exception:
        logger.warning(
            "[PatientState] best-effort snapshot rebuild failed for profile %s "
            "(the write itself succeeded; retrieval may read stale state until "
            "the next successful rebuild)", patient_profile_id, exc_info=True,
        )
