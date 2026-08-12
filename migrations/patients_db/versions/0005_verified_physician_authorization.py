"""verified physician-patient authorization

Revision ID: 0005_verified_physician_authorization
Revises: 0004_state_revision_freshness
Create Date: 2026-08-12

FROZEN_STATEMENTS below is a literal, historical snapshot of
src/api/services/patient_schema.py SCHEMA_STATEMENTS[119:121] — the two
statements added after 0004 shipped: link_status/verified_physician_
user_id/verified_at/granted_by/permissions on patient_care_team_links,
plus a supporting index (convergence Sprint C item 14). See
patient_care_team_service.py's verify_link()/
authorize_physician_patient_access() for why a patient self-reporting a
physician_id must not, by itself, become authorization to read that
patient's canonical state.

Every change here is purely additive (five new nullable/defaulted
columns plus an index), so downgrade() does real work.
"""
from __future__ import annotations

from typing import List, Sequence, Union

from alembic import op

revision: str = "0005_verified_physician_authorization"
down_revision: Union[str, None] = "0004_state_revision_freshness"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FROZEN_STATEMENTS: List[str] = [
    '\n    ALTER TABLE patient_care_team_links\n        ADD COLUMN IF NOT EXISTS link_status TEXT NOT NULL DEFAULT \'invited\';\n    ALTER TABLE patient_care_team_links\n        ADD COLUMN IF NOT EXISTS verified_physician_user_id UUID;\n    ALTER TABLE patient_care_team_links\n        ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;\n    ALTER TABLE patient_care_team_links\n        ADD COLUMN IF NOT EXISTS granted_by TEXT;\n    ALTER TABLE patient_care_team_links\n        ADD COLUMN IF NOT EXISTS permissions JSONB NOT NULL DEFAULT \'[]\'::jsonb;\n    ',
    '\n    CREATE INDEX IF NOT EXISTS patient_care_team_links_verified_idx\n        ON patient_care_team_links (patient_profile_id, verified_physician_user_id, status, link_status);\n    ',
]

def upgrade() -> None:
    for statement in FROZEN_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS patient_care_team_links_verified_idx;")
    op.execute("""
        ALTER TABLE patient_care_team_links
            DROP COLUMN IF EXISTS link_status,
            DROP COLUMN IF EXISTS verified_physician_user_id,
            DROP COLUMN IF EXISTS verified_at,
            DROP COLUMN IF EXISTS granted_by,
            DROP COLUMN IF EXISTS permissions;
    """)
