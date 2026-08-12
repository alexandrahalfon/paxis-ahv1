"""Vitals/weight service (Phase 1). Feeds nutrition-risk derivation in
patient_state_service — weight_change_pct(days) over the 7/30/90-day
windows the Phase 1 checklist calls for ("useful windows")."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict


class VitalsService:
    async def add_vital(
        self,
        patient_profile_id: str,
        vital_type: str,
        value_numeric: float,
        unit: Optional[str] = None,
        measured_at: Optional[str] = None,
        source_type: str = "patient_manual",
        source_document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        vid = str(uuid.uuid4())
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO patient_vitals
                    (id, patient_profile_id, vital_type, value_numeric, unit,
                     measured_at, source_type, source_document_id)
                VALUES ($1,$2,$3,$4,$5, COALESCE($6, now()), $7, $8)
                RETURNING *
                """,
                vid, patient_profile_id, vital_type, value_numeric, unit,
                measured_at, source_type, source_document_id,
            )
        return row_to_dict(row)

    async def get_trend(
        self, patient_profile_id: str, vital_type: str, limit: int = 30
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_vitals
                 WHERE patient_profile_id = $1 AND vital_type = $2
                 ORDER BY measured_at DESC LIMIT $3
                """,
                patient_profile_id, vital_type, limit,
            )
        return list(reversed([row_to_dict(r) for r in rows]))

    async def weight_change_pct(
        self, patient_profile_id: str, days: int = 30, trend: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[float]:
        """Percent change in weight_kg over the last `days` days. None
        when there isn't enough history to compute it — never fabricated.
        Accepts a pre-fetched trend (see weight_trend_summary) so callers
        computing 7/30/90 together don't hit the DB three times."""
        if trend is None:
            trend = await self.get_trend(patient_profile_id, "weight_kg", limit=90)
        if len(trend) < 2:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        baseline = None
        for point in trend:
            ts = point.get("measured_at")
            if ts and ts <= cutoff.isoformat():
                baseline = point
        if baseline is None:
            baseline = trend[0]
        latest = trend[-1]
        try:
            base_val = float(baseline["value_numeric"])
            latest_val = float(latest["value_numeric"])
            if base_val == 0:
                return None
            return round((latest_val - base_val) / base_val * 100, 1)
        except (TypeError, ValueError, KeyError):
            return None

    async def weight_trend_summary(self, patient_profile_id: str) -> Dict[str, Optional[float]]:
        """{"7d": ..., "30d": ..., "90d": ...} percent change, each None
        independently when there isn't enough history for that window."""
        trend = await self.get_trend(patient_profile_id, "weight_kg", limit=180)
        return {
            "7d": await self.weight_change_pct(patient_profile_id, 7, trend=trend),
            "30d": await self.weight_change_pct(patient_profile_id, 30, trend=trend),
            "90d": await self.weight_change_pct(patient_profile_id, 90, trend=trend),
        }


_service: Optional[VitalsService] = None


def get_vitals_service() -> VitalsService:
    global _service
    if _service is None:
        _service = VitalsService()
    return _service
