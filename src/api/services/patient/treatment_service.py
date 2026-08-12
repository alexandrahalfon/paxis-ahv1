"""Treatment episode/cycle/agent service (Phase 1).

Treatments are modeled as episode -> cycle -> agent rather than a single
regimen string, so "I'm on FOLFOX" resolves to its constituent drugs
(oxaliplatin, fluorouracil, leucovorin) for medication-specific retrieval —
see the CLAUDE.md-adjacent architecture review, section 9.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event
from src.api.services.patient.clinical_normalization import normalize_drug_name, expand_regimen


class TreatmentService:
    async def add_episode(
        self,
        patient_profile_id: str,
        regimen: Optional[str] = None,
        modality: Optional[str] = None,
        intent: Optional[str] = None,
        line_of_therapy: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: str = "active",
        raw_text: Optional[str] = None,
        agents: Optional[List[Dict[str, Any]]] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """agents: optional list of {"agent_name", "dose", "route", "schedule"}
        inserted alongside the episode — the common case (FOLFOX -> its
        three drugs) in one call instead of a follow-up per agent.

        When agents is empty/omitted but regimen matches a known name in
        clinical_normalization.REGIMEN_EXPANSIONS (FOLFOX, R-CHOP, ...),
        its component agents are populated automatically — a patient
        typing just "FOLFOX" still gets medication-specific retrieval
        without having to name each drug themselves."""
        agent_list = list(agents or [])
        if not agent_list and regimen:
            agent_list = [{"agent_name": name} for name in expand_regimen(regimen)]

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        episode_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO treatment_episodes
                        (id, patient_profile_id, regimen, modality, intent,
                         line_of_therapy, start_date, end_date, status, raw_text,
                         source_type, source_document_id, verification_status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING *
                    """,
                    episode_id, patient_profile_id, regimen, modality, intent,
                    line_of_therapy, start_date, end_date, status, raw_text,
                    source_type, source_document_id, verification_status,
                )
                for agent in agent_list:
                    if not agent.get("agent_name"):
                        continue
                    drug_norm = normalize_drug_name(agent["agent_name"])
                    await conn.execute(
                        """
                        INSERT INTO treatment_agents
                            (id, treatment_episode_id, agent_name, rxnorm_code,
                             dose, route, schedule, canonical_name, aliases)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                        """,
                        str(uuid.uuid4()), episode_id, agent["agent_name"],
                        agent.get("rxnorm_code") or drug_norm["rxnorm_code"],
                        agent.get("dose"), agent.get("route"), agent.get("schedule"),
                        drug_norm["canonical"], json.dumps(drug_norm["aliases"]),
                    )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "treatment_started",
                    {
                        "regimen": regimen, "status": status,
                        "line_of_therapy": line_of_therapy,
                        "agents": [a.get("agent_name") for a in agent_list],
                    },
                    created_by=created_by, event_date=start_date, source=source_type,
                )
        return row_to_dict(row)

    async def update_episode_status(
        self, episode_id: str, patient_profile_id: str, status: str,
        end_date: Optional[str] = None, created_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE treatment_episodes
                       SET status = $3, end_date = COALESCE($4, end_date)
                     WHERE id = $1 AND patient_profile_id = $2
                    RETURNING *
                    """,
                    episode_id, patient_profile_id, status, end_date,
                )
                if row:
                    event_type = "treatment_stopped" if status in ("stopped", "held") else "treatment_change"
                    await append_profile_timeline_event(
                        conn, patient_profile_id, event_type,
                        {"episode_id": episode_id, "status": status, "regimen": row["regimen"]},
                        created_by=created_by, event_date=end_date,
                    )
        return row_to_dict(row) if row else None

    async def add_cycle(
        self, treatment_episode_id: str, patient_profile_id: str,
        cycle_number: Optional[int] = None, cycle_date: Optional[str] = None,
        status: str = "completed", notes: Optional[str] = None,
        delayed: bool = False, delay_reason: Optional[str] = None,
        held: bool = False, dose_reduction_pct: Optional[float] = None,
        created_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        cycle_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            async with conn.transaction():
                episode = await conn.fetchrow(
                    "SELECT regimen FROM treatment_episodes WHERE id = $1 AND patient_profile_id = $2",
                    treatment_episode_id, patient_profile_id,
                )
                if not episode:
                    return None
                row = await conn.fetchrow(
                    """
                    INSERT INTO treatment_cycles
                        (id, treatment_episode_id, cycle_number, cycle_date, status, notes,
                         delayed, delay_reason, held, dose_reduction_pct)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING *
                    """,
                    cycle_id, treatment_episode_id, cycle_number, cycle_date, status, notes,
                    delayed, delay_reason, held, dose_reduction_pct,
                )
                event_type = "cycle_held" if held else ("cycle_delayed" if delayed else "cycle_received")
                await append_profile_timeline_event(
                    conn, patient_profile_id, event_type,
                    {
                        "regimen": episode["regimen"], "cycle_number": cycle_number,
                        "delayed": delayed, "held": held,
                        "dose_reduction_pct": dose_reduction_pct,
                    },
                    created_by=created_by, event_date=cycle_date,
                )
        return row_to_dict(row)

    async def list_episodes(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM treatment_episodes
                 WHERE patient_profile_id = $1
                 ORDER BY start_date DESC NULLS LAST, created_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def get_active_episodes(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM treatment_episodes
                 WHERE patient_profile_id = $1 AND status = 'active'
                 ORDER BY start_date DESC NULLS LAST
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def list_agents(
        self, treatment_episode_id: str, patient_profile_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """patient_profile_id is optional only for the internal call from
        patient_state_service, which already scoped treatment_episode_id
        to the profile it iterated from list_episodes(). Every route-level
        caller MUST pass it — an unscoped query here would let one
        patient read another patient's treatment agents by guessing an
        episode_id (caught in Phase 0 authorization review; see
        list_cycles below for the same fix)."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            if patient_profile_id is not None:
                rows = await conn.fetch(
                    """
                    SELECT ta.* FROM treatment_agents ta
                      JOIN treatment_episodes te ON te.id = ta.treatment_episode_id
                     WHERE ta.treatment_episode_id = $1 AND te.patient_profile_id = $2
                    """,
                    treatment_episode_id, patient_profile_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM treatment_agents WHERE treatment_episode_id = $1",
                    treatment_episode_id,
                )
        return [row_to_dict(r) for r in rows]

    async def list_cycles(
        self, treatment_episode_id: str, patient_profile_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            if patient_profile_id is not None:
                rows = await conn.fetch(
                    """
                    SELECT tc.* FROM treatment_cycles tc
                      JOIN treatment_episodes te ON te.id = tc.treatment_episode_id
                     WHERE tc.treatment_episode_id = $1 AND te.patient_profile_id = $2
                     ORDER BY tc.cycle_number NULLS LAST, tc.cycle_date
                    """,
                    treatment_episode_id, patient_profile_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM treatment_cycles WHERE treatment_episode_id = $1
                     ORDER BY cycle_number NULLS LAST, cycle_date
                    """,
                    treatment_episode_id,
                )
        return [row_to_dict(r) for r in rows]


_service: Optional[TreatmentService] = None


def get_treatment_service() -> TreatmentService:
    global _service
    if _service is None:
        _service = TreatmentService()
    return _service
