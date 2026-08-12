"""Nutrition assessment service (Phase 1 finalization).

The table Phase 1 calls for that had no equivalent before this: appetite,
oral intake, swallowing difficulty, feeding route, diet restrictions,
food allergies, texture requirements, hydration constraints, and a
nutrition-risk level — distinguishing active-treatment nutrition concerns
from survivorship/prevention ones via care_phase, per architecture review
section 19 (don't answer an active-treatment nutrition question with
generic prevention-diet content).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event

VALID_CARE_PHASES = {"active_treatment", "survivorship", "prevention"}
VALID_APPETITE = {"poor", "fair", "good"}
VALID_RISK = {"low", "moderate", "high"}


class NutritionAssessmentService:
    async def add_assessment(
        self,
        patient_profile_id: str,
        assessment_date: Optional[str] = None,
        appetite: Optional[str] = None,
        oral_intake_pct: Optional[int] = None,
        swallowing_difficulty: Optional[bool] = None,
        feeding_tube: bool = False,
        feeding_tube_type: Optional[str] = None,
        diet_restrictions: Optional[List[str]] = None,
        food_allergies: Optional[List[str]] = None,
        texture_requirements: Optional[str] = None,
        hydration_constraints: Optional[str] = None,
        nutrition_risk: Optional[str] = None,
        care_phase: str = "active_treatment",
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if care_phase not in VALID_CARE_PHASES:
            care_phase = "active_treatment"
        if appetite and appetite not in VALID_APPETITE:
            appetite = None
        if nutrition_risk and nutrition_risk not in VALID_RISK:
            nutrition_risk = None

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        assessment_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO nutrition_assessments
                        (id, patient_profile_id, assessment_date, appetite, oral_intake_pct,
                         swallowing_difficulty, feeding_tube, feeding_tube_type,
                         diet_restrictions, food_allergies, texture_requirements,
                         hydration_constraints, nutrition_risk, care_phase,
                         source_type, source_document_id, verification_status)
                    VALUES ($1,$2,COALESCE($3, CURRENT_DATE),$4,$5,$6,$7,$8,
                            $9::jsonb,$10::jsonb,$11,$12,$13,$14,$15,$16,$17)
                    RETURNING *
                    """,
                    assessment_id, patient_profile_id, assessment_date, appetite,
                    oral_intake_pct, swallowing_difficulty, feeding_tube, feeding_tube_type,
                    json.dumps(diet_restrictions or []), json.dumps(food_allergies or []),
                    texture_requirements, hydration_constraints, nutrition_risk, care_phase,
                    source_type, source_document_id, verification_status,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "nutrition_assessment",
                    {"appetite": appetite, "nutrition_risk": nutrition_risk,
                     "care_phase": care_phase},
                    created_by=created_by, event_date=assessment_date, source=source_type,
                )
        from src.api.services.patient.patient_state_service import invalidate_patient_state
        await invalidate_patient_state(patient_profile_id)
        return row_to_dict(row)

    async def list_assessments(self, patient_profile_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM nutrition_assessments
                 WHERE patient_profile_id = $1
                 ORDER BY assessment_date DESC, created_at DESC
                 LIMIT $2
                """,
                patient_profile_id, limit,
            )
        return [row_to_dict(r) for r in rows]

    async def get_latest(self, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        rows = await self.list_assessments(patient_profile_id, limit=1)
        return rows[0] if rows else None


_service: Optional[NutritionAssessmentService] = None


def get_nutrition_assessment_service() -> NutritionAssessmentService:
    global _service
    if _service is None:
        _service = NutritionAssessmentService()
    return _service
