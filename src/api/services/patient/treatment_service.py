"""Treatment episode/cycle/agent service (Phase 1).

Treatments are modeled as episode -> cycle -> agent rather than a single
regimen string, so "I'm on FOLFOX" resolves to its constituent drugs
(oxaliplatin, fluorouracil, leucovorin) for medication-specific retrieval —
see the CLAUDE.md-adjacent architecture review, section 9.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


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
        three drugs) in one call instead of a follow-up per agent."""
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
                for agent in (agents or []):
                    if not agent.get("agent_name"):
                        continue
                    await conn.execute(
                        """
                        INSERT INTO treatment_agents
                            (id, treatment_episode_id, agent_name, rxnorm_code,
                             dose, route, schedule)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        str(uuid.uuid4()), episode_id, agent["agent_name"],
                        agent.get("rxnorm_code"), agent.get("dose"),
                        agent.get("route"), agent.get("schedule"),
                    )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "treatment_started",
                    {
                        "regimen": regimen, "status": status,
                        "line_of_therapy": line_of_therapy,
                        "agents": [a.get("agent_name") for a in (agents or [])],
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
                        (id, treatment_episode_id, cycle_number, cycle_date, status, notes)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING *
                    """,
                    cycle_id, treatment_episode_id, cycle_number, cycle_date, status, notes,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "cycle_received",
                    {"regimen": episode["regimen"], "cycle_number": cycle_number},
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
