"""Posts, comments, reactions, reports, blocking (Phase 7).

Every write here is addressed by community_profile_id, resolved from the
caller's user_id by the route layer via community_service.ensure_profile
— never by user_id or patient_profile_id directly, keeping the
pseudonymous-identity boundary described in this package's __init__.
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


class PostService:
    async def create_post(
        self, community_slug: str, community_profile_id: str,
        title: Optional[str], body: str,
    ) -> Dict[str, Any]:
        from src.api.services.community.community_service import get_community_service
        community = await get_community_service().get_community(community_slug)
        if not community:
            raise ValueError("Community not found")

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        post_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            # Posting auto-joins — a patient shouldn't have to join first
            # to ask one question in a community they found through search.
            await conn.execute(
                """
                INSERT INTO community_memberships (id, community_id, community_profile_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (community_id, community_profile_id) DO NOTHING
                """,
                str(uuid.uuid4()), community["id"], community_profile_id,
            )
            row = await conn.fetchrow(
                """
                INSERT INTO community_posts (id, community_id, community_profile_id, title, body)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                post_id, community["id"], community_profile_id, title, body,
            )
        return _row_to_dict(row)

    async def list_posts(
        self, community_slug: str, viewer_profile_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        from src.api.services.community.community_service import get_community_service
        community = await get_community_service().get_community(community_slug)
        if not community:
            return []

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p.*, cp.handle AS author_handle
                  FROM community_posts p
                  JOIN community_profiles cp ON cp.id = p.community_profile_id
                 WHERE p.community_id = $1 AND p.status = 'visible'
                   AND ($2::uuid IS NULL OR p.community_profile_id NOT IN (
                       SELECT blocked_profile_id FROM community_blocked_users
                        WHERE blocker_profile_id = $2
                   ))
                 ORDER BY p.created_at DESC
                 LIMIT $3
                """,
                community["id"], viewer_profile_id, limit,
            )
        return [_row_to_dict(r) for r in rows]

    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT p.*, cp.handle AS author_handle
                  FROM community_posts p
                  JOIN community_profiles cp ON cp.id = p.community_profile_id
                 WHERE p.id = $1
                """,
                post_id,
            )
        return _row_to_dict(row) if row else None

    async def add_comment(self, post_id: str, community_profile_id: str, body: str) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        exists = await self.get_post(post_id)
        if not exists:
            raise ValueError("Post not found")
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO community_comments (id, post_id, community_profile_id, body)
                VALUES ($1, $2, $3, $4)
                RETURNING *
                """,
                str(uuid.uuid4()), post_id, community_profile_id, body,
            )
        return _row_to_dict(row)

    async def list_comments(self, post_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.*, cp.handle AS author_handle
                  FROM community_comments c
                  JOIN community_profiles cp ON cp.id = c.community_profile_id
                 WHERE c.post_id = $1 AND c.status = 'visible'
                 ORDER BY c.created_at
                """,
                post_id,
            )
        return [_row_to_dict(r) for r in rows]

    async def react(
        self, target_type: str, target_id: str, community_profile_id: str, reaction: str = "support"
    ) -> Dict[str, Any]:
        if target_type not in ("post", "comment"):
            raise ValueError("target_type must be 'post' or 'comment'")
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO community_reactions
                    (id, target_type, target_id, community_profile_id, reaction)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (target_type, target_id, community_profile_id)
                DO UPDATE SET reaction = EXCLUDED.reaction
                RETURNING *
                """,
                str(uuid.uuid4()), target_type, target_id, community_profile_id, reaction,
            )
        return _row_to_dict(row)

    async def report(
        self, target_type: str, target_id: str, reported_by_profile_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if target_type not in ("post", "comment"):
            raise ValueError("target_type must be 'post' or 'comment'")
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO community_reports
                    (id, target_type, target_id, reported_by_profile_id, reason)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                str(uuid.uuid4()), target_type, target_id, reported_by_profile_id, reason,
            )
        return _row_to_dict(row)

    async def block_user(self, blocker_profile_id: str, blocked_profile_id: str) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO community_blocked_users (id, blocker_profile_id, blocked_profile_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (blocker_profile_id, blocked_profile_id) DO NOTHING
                """,
                str(uuid.uuid4()), blocker_profile_id, blocked_profile_id,
            )
        return {"blocked": True}


_service: Optional[PostService] = None


def get_post_service() -> PostService:
    global _service
    if _service is None:
        _service = PostService()
    return _service
