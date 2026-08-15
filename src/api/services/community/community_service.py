"""Community + pseudonymous profile service (Phase 7)."""

from __future__ import annotations

import random
import string
import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db

DEFAULT_COMMUNITIES = [
    ("breast-cancer", "Breast Cancer", "oncology"),
    ("stage-iv-breast-cancer", "Stage IV Breast Cancer", "oncology"),
    ("triple-negative-breast-cancer", "Triple-Negative Breast Cancer", "oncology"),
    ("lung-cancer", "Lung Cancer", "oncology"),
    ("egfr-lung-cancer", "EGFR+ Lung Cancer", "oncology"),
    ("colorectal-cancer", "Colorectal Cancer", "oncology"),
    ("head-neck-cancer", "Head & Neck Cancer", "oncology"),
    ("glioblastoma", "Glioblastoma", "oncology"),
    ("caregivers", "Caregivers", "support"),
    ("young-adults-with-cancer", "Young Adults With Cancer", "support"),
    ("survivorship", "Survivorship", "support"),
]


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
        elif hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


def _generate_handle() -> str:
    suffix = "".join(random.choices(string.digits, k=4))
    return f"Member-{suffix}"


class CommunityService:
    async def ensure_profile(self, user_id: str, handle: Optional[str] = None) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM community_profiles WHERE user_id = $1", user_id
            )
            if existing:
                return _row_to_dict(existing)

            for _ in range(5):
                candidate = handle or _generate_handle()
                taken = await conn.fetchval(
                    "SELECT 1 FROM community_profiles WHERE handle = $1", candidate
                )
                if not taken:
                    break
                handle = None  # retry with a fresh generated handle
            else:
                raise RuntimeError("Could not allocate a community handle")

            row = await conn.fetchrow(
                """
                INSERT INTO community_profiles (id, user_id, handle)
                VALUES ($1, $2, $3)
                RETURNING *
                """,
                str(uuid.uuid4()), user_id, candidate,
            )
        return _row_to_dict(row)

    async def get_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM community_profiles WHERE user_id = $1", user_id
            )
        return _row_to_dict(row) if row else None

    async def update_handle(self, user_id: str, handle: str) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            taken = await conn.fetchval(
                "SELECT 1 FROM community_profiles WHERE handle = $1 AND user_id != $2",
                handle, user_id,
            )
            if taken:
                raise ValueError("That handle is already taken.")
            row = await conn.fetchrow(
                "UPDATE community_profiles SET handle = $2 WHERE user_id = $1 RETURNING *",
                user_id, handle,
            )
        return _row_to_dict(row)

    async def seed_default_communities(self) -> List[Dict[str, Any]]:
        out = []
        for slug, name, category in DEFAULT_COMMUNITIES:
            out.append(await self.create_community(slug, name, category=category))
        return out

    async def create_community(
        self, slug: str, name: str, description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO communities (id, slug, name, description, category)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name
                RETURNING *
                """,
                str(uuid.uuid4()), slug, name, description, category,
            )
        return _row_to_dict(row)

    async def list_communities(self) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM communities ORDER BY name")
        return [_row_to_dict(r) for r in rows]

    async def get_community(self, slug: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM communities WHERE slug = $1", slug)
        return _row_to_dict(row) if row else None

    async def join(self, community_slug: str, community_profile_id: str) -> Dict[str, Any]:
        community = await self.get_community(community_slug)
        if not community:
            raise ValueError("Community not found")
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO community_memberships (id, community_id, community_profile_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (community_id, community_profile_id) DO NOTHING
                """,
                str(uuid.uuid4()), community["id"], community_profile_id,
            )
        return {"community_id": community["id"], "joined": True}

    async def leave(self, community_slug: str, community_profile_id: str) -> bool:
        community = await self.get_community(community_slug)
        if not community:
            return False
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM community_memberships WHERE community_id = $1 AND community_profile_id = $2",
                community["id"], community_profile_id,
            )
        return not result.endswith("0")

    async def list_my_communities(self, community_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.* FROM communities c
                JOIN community_memberships m ON m.community_id = c.id
                WHERE m.community_profile_id = $1
                ORDER BY c.name
                """,
                community_profile_id,
            )
        return [_row_to_dict(r) for r in rows]


_service: Optional[CommunityService] = None


def get_community_service() -> CommunityService:
    global _service
    if _service is None:
        _service = CommunityService()
    return _service
