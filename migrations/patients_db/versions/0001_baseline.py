"""baseline: every table that predates real migrations

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-12

Runs the exact same statement list ensure_schema() runs
(src/api/services/patient_schema.py SCHEMA_STATEMENTS) — see that
module's docstring and migrations/patients_db/README.md for why there are
not two copies of this SQL to keep in sync.

Every statement is additive/idempotent (CREATE ... IF NOT EXISTS, ADD
COLUMN IF NOT EXISTS), so running this against a database that already
has some or all of these objects — true for every existing deployment,
since ensure_schema() already ran there — is a safe no-op beyond adding
the alembic_version row. On a genuinely empty database it builds the
schema from scratch.

downgrade() is intentionally a hard stop, not a DROP TABLE cascade: this
revision spans the platform's entire schema as it stood before real
migrations existed, so "downgrading" it would mean deleting every
patient's data. If you need to walk backward, restore from a database
backup taken before this revision was applied instead.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Union

from alembic import op

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.patient_schema import SCHEMA_STATEMENTS  # noqa: E402

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for statement in SCHEMA_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    raise RuntimeError(
        "0001_baseline cannot be downgraded — it spans the entire "
        "pre-migration schema for every patient. Restore from a backup "
        "taken before this revision instead of downgrading."
    )
