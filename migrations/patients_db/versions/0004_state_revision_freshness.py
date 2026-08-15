"""deterministic state freshness via revision counters

Revision ID: 0004_state_revision_freshness
Revises: 0003_evidence_versioning_and_debug_trace
Create Date: 2026-08-12

FROZEN_STATEMENTS below is a literal, historical snapshot of
src/api/services/patient_schema.py SCHEMA_STATEMENTS[117:119] — the two
statements added after 0003 shipped: state_revision on patient_profiles
and source_revision on patient_state_snapshots (convergence Sprint B item
7). See patient_state_service.py's invalidate_patient_state() and
patient_context_service.py's get_context() for how these two columns
replace "rebuild only when no snapshot exists at all" with "rebuild
whenever the snapshot's source_revision doesn't match the profile's
current state_revision" — closing the gap where a rebuild attempt that
failed once left every subsequent read trusting a stale snapshot forever.

Every change here is purely additive (two new nullable/defaulted
columns), so downgrade() does real work.
"""
from __future__ import annotations

from typing import List, Sequence, Union

from alembic import op

revision: str = "0004_state_revision_freshness"
down_revision: Union[str, None] = "0003_evidence_versioning_and_debug_trace"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FROZEN_STATEMENTS: List[str] = [
    '\n    ALTER TABLE patient_profiles\n        ADD COLUMN IF NOT EXISTS state_revision BIGINT NOT NULL DEFAULT 0;\n    ',
    '\n    ALTER TABLE patient_state_snapshots\n        ADD COLUMN IF NOT EXISTS source_revision BIGINT;\n    ',
]

def upgrade() -> None:
    for statement in FROZEN_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("ALTER TABLE patient_state_snapshots DROP COLUMN IF EXISTS source_revision;")
    op.execute("ALTER TABLE patient_profiles DROP COLUMN IF EXISTS state_revision;")
