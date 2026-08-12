"""Care team instruction service (Phase 1 finalization).

The table Phase 1 calls for that had no equivalent before this: a
clinician's specific instruction to this patient ("no NSAIDs while on
this regimen", "call us if your temperature is above 100.4F"), which
should outrank generic patient education when the two would otherwise
conflict or when a patient asks something this instruction already
answers. Actually applying that precedence in retrieval is Phase 4/13
work (see evidence/retrieval_planner.py); this service is just where the
instruction is captured and read from.

instruction_type is a normalized-ish free field for now (not a closed
enum) — 'dietary' | 'activity' | 'medication' | 'monitoring' |
'follow_up' | 'other' are the values callers are expected to use, but
nothing here enforces it, matching the light-touch validation pattern
elsewhere in this package (e.g. conditions_service).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event

INSTRUCTION_TYPES = {"dietary", "activity", "medication", "monitoring", "follow_up", "other"}


class CareTeamInstructionService:
    async def add_instruction(
        self,
        patient_profile_id: str,
        instruction_text: str,
        instruction_type: str = "other",
        author_provider: Optional[str] = None,
        physician_id: Optional[str] = None,
        effective_from: Optional[str] = None,
        effective_to: Optional[str] = None,
        source_type: str = "clinician_entered",
        source_document_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if instruction_type not in INSTRUCTION_TYPES:
            instruction_type = "other"

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        instruction_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO care_team_instructions
                        (id, patient_profile_id, instruction_text, instruction_type,
                         author_provider, physician_id, source_type, source_document_id,
                         effective_from, effective_to)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    RETURNING *
                    """,
                    instruction_id, patient_profile_id, instruction_text, instruction_type,
                    author_provider, physician_id, source_type, source_document_id,
                    effective_from, effective_to,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "care_instruction",
                    {"instruction_type": instruction_type, "instruction_text": instruction_text},
                    created_by=created_by, event_date=effective_from, source=source_type,
                )
        return row_to_dict(row)

    async def list_active(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM care_team_instructions
                 WHERE patient_profile_id = $1 AND active = true
                   AND (effective_to IS NULL OR effective_to >= CURRENT_DATE)
                 ORDER BY created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def deactivate(
        self, instruction_id: str, patient_profile_id: str
    ) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE care_team_instructions SET active = false
                 WHERE id = $1 AND patient_profile_id = $2
                RETURNING *
                """,
                instruction_id, patient_profile_id,
            )
        return row_to_dict(row) if row else None


_service: Optional[CareTeamInstructionService] = None


def get_care_team_instruction_service() -> CareTeamInstructionService:
    global _service
    if _service is None:
        _service = CareTeamInstructionService()
    return _service
