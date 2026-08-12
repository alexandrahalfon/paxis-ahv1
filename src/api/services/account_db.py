"""
Database helper for accounts and per-user cache.
"""

from typing import Optional

import asyncpg

from src.core.config import settings


class AccountDatabase:
    """Lazy asyncpg pool for the accounts/cache database."""

    def __init__(self):
        self._pool = None

    async def get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                # host/port/user/password all fall back to postgres_* since
                # all three DBs on this GCP instance share one server/user by
                # default. Previously host/port/user had no fallback and were
                # hardcoded to a stale IP in config.py — that silently broke
                # accounts (register/login, and get_current_user, which every
                # patient route depends on) every time postgres_host changed
                # without a matching CACHE_POSTGRES_HOST override, most
                # recently during the paxis-prod migration. Same fix applied
                # to patient_db.py.
                host=settings.cache_postgres_host or settings.postgres_host,
                port=settings.cache_postgres_port or settings.postgres_port,
                user=settings.cache_postgres_user or settings.postgres_user,
                password=settings.cache_postgres_password or settings.postgres_password,
                database=settings.cache_postgres_database,
                min_size=settings.cache_postgres_min_pool,
                max_size=settings.cache_postgres_max_pool,
                timeout=30,
            )
        return self._pool

    async def ensure_schema(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            # Additive: first_name/last_name/institution collected at
            # registration (added 2026-07-18). Nullable at the DB level so
            # existing rows from before this change don't break; required
            # for new registrations via RegisterRequest validation instead.
            await conn.execute(
                """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS institution TEXT;
                """
            )
            # Additive: account role (added 2026-08-08 for the patient
            # portal). Defaults to 'physician' so every pre-existing row
            # and every existing code path keeps its current behaviour
            # without a backfill. Values: 'physician' | 'patient' | 'admin'.
            await conn.execute(
                """
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'physician';
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_cache (
                    user_id UUID NOT NULL,
                    cache_key TEXT NOT NULL,
                    cache_value JSONB NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_accessed TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at TIMESTAMPTZ,
                    PRIMARY KEY (user_id, cache_key)
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS user_cache_expires_at_idx
                    ON user_cache (expires_at);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS user_cache_hit_count_idx
                    ON user_cache (hit_count DESC);
                """
            )
            # Add is_permanent column if missing (needed by cache promote feature)
            await conn.execute(
                """
                ALTER TABLE user_cache
                ADD COLUMN IF NOT EXISTS is_permanent BOOLEAN DEFAULT false;
                """
            )


_db_instance: Optional[AccountDatabase] = None


def get_account_db() -> AccountDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = AccountDatabase()
    return _db_instance