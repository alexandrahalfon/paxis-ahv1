"""
Database helper for the patient-centric platform (exueed-patients).

Mirrors account_db.py's lazy asyncpg pool pattern. This is a separate
database on the same GCP Postgres instance as display-study-details and
exueed_cache (see src/core/config.py patients_postgres_* settings) — kept
isolated for access-control/backup reasons, not merged into either existing
service.

physician_id columns below reference users.id in the exueed_cache database.
Postgres does not support foreign keys across databases, so that
relationship is enforced by application code, not a DB constraint.

Schema lives in patient_schema.py, not here (Phase 0 hardening,
2026-08-12) — see that module's docstring. ensure_schema() below just
executes that shared statement list; it carries no CREATE TABLE text of
its own so there is nothing here that can drift from
migrations/patients_db/versions/0001_baseline.py.

ensure_schema() remains the request-time safety net (every service method
calls it before touching the pool, so a fresh environment self-heals) but
is no longer the intended way schema changes ship to production — see
migrations/patients_db/README.md. New tables/columns should be added as a
new Alembic revision AND appended to SCHEMA_STATEMENTS, not the other way
around, so a fresh dev environment and a migrated production database
always converge on the same schema.
"""

from typing import Optional

import asyncpg

from src.core.config import settings
from src.api.services.patient_schema import SCHEMA_STATEMENTS


class PatientDatabase:
    """Lazy asyncpg pool for the exueed-patients database."""

    def __init__(self):
        self._pool = None

    async def get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                # host/port/user/password all fall back to postgres_* since
                # all three DBs share one server/user by default. host/port/
                # user previously had no fallback and were hardcoded to a
                # stale IP in config.py — fixed alongside the same bug in
                # account_db.py.
                host=settings.patients_postgres_host or settings.postgres_host,
                port=settings.patients_postgres_port or settings.postgres_port,
                user=settings.patients_postgres_user or settings.postgres_user,
                password=settings.patients_postgres_password or settings.postgres_password,
                database=settings.patients_postgres_database,
                min_size=settings.patients_postgres_min_pool,
                max_size=settings.patients_postgres_max_pool,
                timeout=30,
            )
        return self._pool

    async def close(self):
        """Close the connection pool gracefully."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ensure_schema(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            for statement in SCHEMA_STATEMENTS:
                await conn.execute(statement)


_db_instance: Optional[PatientDatabase] = None


def get_patient_db() -> PatientDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = PatientDatabase()
    return _db_instance
