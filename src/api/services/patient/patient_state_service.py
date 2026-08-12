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
from src.api.services.patient.lab_interpretation import allowed_interpretation_for


def _lab_value_shape(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """{value, unit, collected_at} from a raw lab_results row — shared by
    both the latest and previous slots in state["labs"] (2026-08-12
    convergence Sprint A item 3)."""
    if not row:
        return None
    return {
        "value": row.get("value_numeric") if row.get("value_numeric") is not None else row.get("value_text"),
        "unit": row.get("unit"),
        "collected_at": row.get("collected_at"),
    }


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

    async def _increment_state_revision(self, patient_profile_id: str) -> Optional[int]:
        """Atomically bumps patient_profiles.state_revision and returns
        the new value (2026-08-12 convergence Sprint B item 7). Called
        from invalidate_patient_state() before build_state() so every
        canonical write's rebuild attempt — success or failure — is
        preceded by a revision bump; the snapshot this rebuild produces
        (or, if it fails, the STALE snapshot already on file) can then be
        compared against this new revision by get_context() to know
        whether it's current. Returns None (rather than raising) if the
        profile row doesn't exist — a caller racing a profile deletion
        shouldn't crash a best-effort invalidation."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(
                """
                UPDATE patient_profiles SET state_revision = state_revision + 1
                 WHERE id = $1
             RETURNING state_revision
                """,
                patient_profile_id,
            )

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
        # latest_and_previous_by_test (not just most_recent_by_test)
        # because state["labs"]'s allowed_interpretation policy (2026-08-12
        # convergence Sprint A item 3) needs a previous reading to know
        # whether a trend can be stated at all — see lab_interpretation.py.
        labs_latest_previous = await get_lab_service().latest_and_previous_by_test(patient_profile_id)

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
            # Flat shape, kept verbatim for existing readers (see
            # _derive_retrieval_features below, which no longer reads
            # this for anything but is left unchanged as a shape).
            "recent_labs": {
                test: _lab_value_shape(entry.get("latest"))
                for test, entry in labs_latest_previous.items() if entry.get("latest")
            },
            # Interpretation-policy shape (2026-08-12 convergence Sprint A
            # item 3) — see lab_interpretation.py. This is what generation
            # and any future claim validator should read for labs, not
            # recent_labs above: it carries an explicit
            # allowed_interpretation so a value/trend can be stated
            # without generation inventing a named clinical conclusion
            # ("neutropenic", "renal impairment") this system never
            # validated.
            "labs": [
                {
                    "canonical_test": test,
                    "latest": _lab_value_shape(entry.get("latest")),
                    "previous": _lab_value_shape(entry.get("previous")),
                    "allowed_interpretation": allowed_interpretation_for(
                        _lab_value_shape(entry.get("latest")), _lab_value_shape(entry.get("previous")),
                    ),
                }
                for test, entry in labs_latest_previous.items() if entry.get("latest")
            ],
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
            # profile was fetched at the very top of this method, so its
            # state_revision already reflects any bump
            # invalidate_patient_state() made before calling here (or
            # whatever the profile's current value is, for a caller that
            # invoked build_state() directly, e.g. get_context() building
            # a first-ever snapshot). Stamping THIS value onto the
            # snapshot is what lets get_context() later tell a current
            # snapshot from a stale one — see patient_context_service.py.
            await self._persist_snapshot(
                patient_profile_id, state, retrieval_features,
                source_revision=profile.get("state_revision"),
            )

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

        # 2026-08-12 convergence Sprint A item 3: neutropenia_risk /
        # thrombocytopenia_risk / renal_function_context used to be
        # derived here from hard-coded thresholds (ANC < 1.5, platelets
        # < 100, creatinine > 1.3) and handed to generation/retrieval as
        # named clinical conclusions the system never actually
        # validated. Removed outright rather than left for something
        # downstream to lean on — see lab_interpretation.py's module
        # docstring. state["labs"] (built above) is the sanctioned
        # replacement: exact values, a trend when a previous reading
        # exists, and an explicit allowed_interpretation policy, with no
        # named risk label invented at this layer.

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
        self, patient_profile_id: str, state: Dict[str, Any], retrieval_features: Dict[str, Any],
        source_revision: Optional[int] = None,
    ) -> None:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO patient_state_snapshots
                    (id, patient_profile_id, state, retrieval_features, source_revision)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5)
                """,
                str(uuid.uuid4()), patient_profile_id,
                json.dumps(state, default=str), json.dumps(retrieval_features, default=str),
                source_revision,
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
    """Best-effort snapshot rebuild after a canonical write, now backed
    by a deterministic freshness check rather than a one-shot attempt
    (2026-08-12 convergence Sprint B item 7).

    Before Sprint B item 7: get_context() (patient_context_service.py)
    read the LATEST patient_state_snapshots row and only rebuilt when
    none existed at all — it never re-checked staleness. That meant a
    write whose rebuild attempt here failed (network blip, transient DB
    error) left get_context() trusting a stale snapshot FOREVER, since
    nothing about "a snapshot exists" ever changed just because it was
    stale — see the 2026-08-12 beta audit, "patient state can become
    stale after manual edits", which this function originally fixed for
    the "no rebuild was ever attempted" case but not the "the rebuild
    was attempted and failed" case.

    Now: patient_profiles.state_revision is bumped FIRST (its own
    statement, before build_state() even runs), and build_state()
    stamps the CURRENT revision onto whatever snapshot it persists as
    patient_state_snapshots.source_revision. get_context() compares the
    two and rebuilds whenever they don't match — not just when no
    snapshot exists at all — so a rebuild that fails here no longer
    strands get_context() on stale state indefinitely: the NEXT read
    sees the revision mismatch and retries the rebuild itself, every
    time, until one succeeds. Every manual-entry write path calls this
    the same way, immediately after its own transaction commits (never
    from inside the transaction itself — this acquires its own
    connection from the pool, so calling it before the write commits
    would read pre-write data under READ COMMITTED and produce a
    rebuild that's still stale).

    Never raises: a failed revision bump or rebuild must not cost the
    caller their successful write. Both are best-effort in the same way
    — if the revision bump itself fails, build_state() below still runs
    (with build_state() reading whatever revision the profile row
    currently holds), and if THAT fails too, the exception handler below
    still applies; get_context() will simply keep re-attempting on every
    subsequent read until something succeeds.
    """
    try:
        await get_patient_state_service()._increment_state_revision(patient_profile_id)
    except Exception:
        logger.warning(
            "[PatientState] state_revision bump failed for profile %s "
            "(continuing to rebuild anyway)", patient_profile_id, exc_info=True,
        )
    try:
        await get_patient_state_service().build_state(patient_profile_id)
    except Exception:
        logger.warning(
            "[PatientState] best-effort snapshot rebuild failed for profile %s "
            "(the write itself succeeded; retrieval will retry the rebuild on "
            "its next read, since state_revision no longer matches the "
            "existing snapshot)", patient_profile_id, exc_info=True,
        )
