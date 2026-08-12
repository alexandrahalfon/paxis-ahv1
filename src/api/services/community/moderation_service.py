"""Moderation actions (Phase 7) — minimal, but enforces the report -> hide
-> logged-action flow rather than leaving reports to accumulate unread.
Gated by an admin-role check at the route layer (see routes/communities.py);
this service does not itself check permissions.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
        elif hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


class ModerationService:
    async def list_reports(self, status: str = "open") -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM community_reports WHERE status = $1 ORDER BY created_at DESC",
                status,
            )
        return [_row_to_dict(r) for r in rows]

    async def resolve_report(
        self, report_id: str, action: str, acted_by: str, reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """action: 'hide' | 'dismiss'. 'hide' sets the target's status to
        'hidden' (post or comment) and logs a moderation action; 'dismiss'
        just closes the report."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            report = await conn.fetchrow(
                "SELECT * FROM community_reports WHERE id = $1", report_id
            )
            if not report:
                raise ValueError("Report not found")

            if action == "hide":
                table = "community_posts" if report["target_type"] == "post" else "community_comments"
                await conn.execute(
                    f"UPDATE {table} SET status = 'hidden' WHERE id = $1",
                    report["target_id"],
                )
                await conn.execute(
                    """
                    INSERT INTO community_moderation_actions
                        (id, target_type, target_id, action, reason, acted_by)
                    VALUES ($1, $2, $3, 'hide', $4, $5)
                    """,
                    str(uuid.uuid4()), report["target_type"], report["target_id"],
                    reason, acted_by,
                )

            row = await conn.fetchrow(
                """
                UPDATE community_reports SET status = $2 WHERE id = $1
                RETURNING *
                """,
                report_id, "resolved" if action == "hide" else "dismissed",
            )
        return _row_to_dict(row)


_service: Optional[ModerationService] = None


def get_moderation_service() -> ModerationService:
    global _service
    if _service is None:
        _service = ModerationService()
    return _service
