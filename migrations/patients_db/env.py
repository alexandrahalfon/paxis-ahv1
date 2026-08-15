"""
Alembic environment for the exueed-patients database.

No SQLAlchemy ORM models exist for this schema (the app talks to it
directly via asyncpg — see src/api/services/patient_db.py), so
target_metadata stays None and every revision is written by hand with
op.execute(). That also means --autogenerate will not produce anything
useful here; it isn't the workflow this is set up for.

The connection URL is built from the exact same settings the app's
asyncpg pool uses (src/core/config.py patients_postgres_* falling back to
postgres_*), so there is one definition of "where is this database",
not two that can silently disagree. The driver differs on purpose:
psycopg2 here (Alembic's default sync execution model), asyncpg at
request time in the app — this file is the only place psycopg2 is used.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Repo root is three levels up from this file
# (migrations/patients_db/env.py -> migrations/patients_db -> migrations -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import settings  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _sync_database_url() -> str:
    host = settings.patients_postgres_host or settings.postgres_host
    port = settings.patients_postgres_port or settings.postgres_port
    user = settings.patients_postgres_user or settings.postgres_user
    password = settings.patients_postgres_password or settings.postgres_password
    database = settings.patients_postgres_database
    # URL-encode user/password defensively — Postgres passwords routinely
    # contain characters (@, :, /) that are not valid unescaped in a URL.
    from urllib.parse import quote_plus
    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade head --sql`)."""
    context.configure(
        url=_sync_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _sync_database_url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
