"""
Per-user cache with hot in-memory layer + Postgres persistence.

Cache behavior:
- Cached results are stored temporarily (short TTL) until user gives thumbs up
- Thumbs up promotes cache entry to permanent (long TTL)
- Cache acts as a BOOST, not ground truth - always re-run query but use cache to inform ranking
"""

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from src.core.config import settings
from .account_db import get_account_db

# Cache TTL constants
TEMP_CACHE_TTL = 3600  # 1 hour for temporary cache (before thumbs up)
PERMANENT_CACHE_TTL = 86400 * 30  # 30 days for permanent cache (after thumbs up)


class CacheService:
    """Per-user cache with a small in-memory hot store."""

    def __init__(self):
        self._hot_cache: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        self._hot_max = settings.cache_hot_max_entries

    def _hot_get(self, user_id: str, cache_key: str) -> Optional[Dict[str, Any]]:
        key = (user_id, cache_key)
        value = self._hot_cache.get(key)
        if value is not None:
            self._hot_cache.move_to_end(key)
        return value

    def _hot_set(self, user_id: str, cache_key: str, value: Dict[str, Any]):
        key = (user_id, cache_key)
        self._hot_cache[key] = value
        self._hot_cache.move_to_end(key)
        if len(self._hot_cache) > self._hot_max:
            self._hot_cache.popitem(last=False)

    async def get(self, user_id: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached value. Returns None if not found or expired."""
        hot_value = self._hot_get(user_id, cache_key)
        if hot_value is not None:
            return hot_value

        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT cache_value, expires_at, is_permanent
                FROM user_cache
                WHERE user_id = $1 AND cache_key = $2
                """,
                user_id,
                cache_key,
            )
            if not row:
                return None

            expires_at = row["expires_at"]
            if expires_at and expires_at < datetime.now(timezone.utc):
                await conn.execute(
                    """
                    DELETE FROM user_cache
                    WHERE user_id = $1 AND cache_key = $2
                    """,
                    user_id,
                    cache_key,
                )
                return None

            await conn.execute(
                """
                UPDATE user_cache
                SET hit_count = hit_count + 1, last_accessed = now()
                WHERE user_id = $1 AND cache_key = $2
                """,
                user_id,
                cache_key,
            )

            value = row["cache_value"]
            if isinstance(value, str):
                value = json.loads(value)
            
            # Add metadata about cache status
            value["_cache_meta"] = {
                "is_permanent": row.get("is_permanent", False),
                "from_cache": True
            }
            
            self._hot_set(user_id, cache_key, value)
            return value
    
    async def get_for_boost(self, user_id: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached value for use as a BOOST (not ground truth).
        Returns the cached value but marks it as boost-only.
        The caller should still run the full query but can use this to inform ranking.
        """
        cached = await self.get(user_id, cache_key)
        if cached:
            cached["_cache_meta"] = cached.get("_cache_meta", {})
            cached["_cache_meta"]["use_as_boost"] = True
        return cached

    async def set(
        self,
        user_id: str,
        cache_key: str,
        cache_value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        is_permanent: bool = False,
    ):
        """
        Store cache entry.
        - is_permanent=False: Short TTL, will expire unless promoted
        - is_permanent=True: Long TTL, promoted by thumbs up
        """
        if ttl_seconds is None:
            ttl_seconds = PERMANENT_CACHE_TTL if is_permanent else TEMP_CACHE_TTL
        
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_cache (user_id, cache_key, cache_value, hit_count, expires_at, is_permanent)
                VALUES ($1, $2, $3::jsonb, 0, $4, $5)
                ON CONFLICT (user_id, cache_key)
                DO UPDATE SET
                    cache_value = EXCLUDED.cache_value,
                    expires_at = EXCLUDED.expires_at,
                    is_permanent = EXCLUDED.is_permanent,
                    last_accessed = now()
                """,
                user_id,
                cache_key,
                json.dumps(cache_value),
                expires_at,
                is_permanent,
            )

        self._hot_set(user_id, cache_key, cache_value)
    
    async def promote_to_permanent(self, user_id: str, cache_key: str) -> bool:
        """
        Promote a cache entry to permanent status (called on thumbs up).
        Extends TTL to 30 days and marks as permanent.
        """
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        
        new_expires = datetime.now(timezone.utc) + timedelta(seconds=PERMANENT_CACHE_TTL)
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE user_cache
                SET is_permanent = true, expires_at = $3
                WHERE user_id = $1 AND cache_key = $2
                """,
                user_id,
                cache_key,
                new_expires,
            )
            
            # Update hot cache if present
            key = (user_id, cache_key)
            if key in self._hot_cache:
                self._hot_cache[key]["_cache_meta"] = {"is_permanent": True}
            
            return "UPDATE 1" in result
    
    async def demote_from_permanent(self, user_id: str, cache_key: str) -> bool:
        """
        Demote a cache entry from permanent status (called on thumbs down).
        Sets short TTL and marks as non-permanent.
        """
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        
        new_expires = datetime.now(timezone.utc) + timedelta(seconds=TEMP_CACHE_TTL)
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE user_cache
                SET is_permanent = false, expires_at = $3
                WHERE user_id = $1 AND cache_key = $2
                """,
                user_id,
                cache_key,
                new_expires,
            )
            
            return "UPDATE 1" in result

    async def clear_user_cache(self, user_id: str, namespace: Optional[str] = None):
        """Clear all cache entries for a user, optionally filtered by namespace prefix."""
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            if namespace:
                await conn.execute(
                    """
                    DELETE FROM user_cache
                    WHERE user_id = $1 AND cache_key LIKE $2
                    """,
                    user_id,
                    f"{namespace}:%",
                )
            else:
                await conn.execute(
                    """
                    DELETE FROM user_cache
                    WHERE user_id = $1
                    """,
                    user_id,
                )

        # Also clear hot cache for this user
        keys_to_remove = [k for k in self._hot_cache.keys() if k[0] == user_id]
        if namespace:
            keys_to_remove = [k for k in keys_to_remove if k[1].startswith(f"{namespace}:")]
        for key in keys_to_remove:
            del self._hot_cache[key]

    async def get_top_hits(self, user_id: str, limit: int = 20):
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT cache_key, hit_count, last_accessed, created_at, is_permanent
                FROM user_cache
                WHERE user_id = $1
                ORDER BY hit_count DESC, last_accessed DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
        return [
            {
                "cache_key": row["cache_key"],
                "hit_count": row["hit_count"],
                "last_accessed": row["last_accessed"],
                "created_at": row["created_at"],
                "is_permanent": row.get("is_permanent", False),
            }
            for row in rows
        ]


def make_cache_key(namespace: str, payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


_cache_instance: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
