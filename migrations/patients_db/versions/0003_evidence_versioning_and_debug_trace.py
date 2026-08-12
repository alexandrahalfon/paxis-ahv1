"""evidence versioning + query debug traces

Revision ID: 0003_evidence_versioning_and_debug_trace
Revises: 0002_phase1_finalization
Create Date: 2026-08-12

FROZEN_STATEMENTS below is a literal, historical snapshot of
src/api/services/patient_schema.py SCHEMA_STATEMENTS[110:117] — the seven
statements added after 0002 shipped: is_current/superseded_by tracking on
evidence_document_versions, current_version_id/latest_content_hash on
evidence_documents, evidence_document_version_id/section_title on
evidence_chunk_registry (all supporting evidence_ingestion_service.py's
deterministic-ID idempotent re-ingestion), plus the query_debug_traces
table (retrieval_debug_trace.py). This is the first revision written
frozen from the start — see 0001_baseline's docstring for why 0001/0002
had to be retrofitted into this shape and
tests/api/services/test_patient_schema_migrations.py for the regression
test that keeps future statements from drifting the same way again.

Every change here is purely additive (new nullable/defaulted columns, a
new table with its own CASCADE), so downgrade() does real work.
"""
from __future__ import annotations

from typing import List, Sequence, Union

from alembic import op

revision: str = "0003_evidence_versioning_and_debug_trace"
down_revision: Union[str, None] = "0002_phase1_finalization"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FROZEN_STATEMENTS: List[str] = [
    '\n    ALTER TABLE evidence_document_versions\n        ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT true;\n    ALTER TABLE evidence_document_versions\n        ADD COLUMN IF NOT EXISTS superseded_by UUID\n            REFERENCES evidence_document_versions(id) ON DELETE SET NULL;\n    ',
    '\n    CREATE INDEX IF NOT EXISTS evidence_document_versions_current_idx\n        ON evidence_document_versions (evidence_document_id, is_current);\n    ',
    '\n    ALTER TABLE evidence_documents\n        ADD COLUMN IF NOT EXISTS current_version_id UUID\n            REFERENCES evidence_document_versions(id) ON DELETE SET NULL;\n    ALTER TABLE evidence_documents\n        ADD COLUMN IF NOT EXISTS latest_content_hash TEXT;\n    ',
    '\n    ALTER TABLE evidence_chunk_registry\n        ADD COLUMN IF NOT EXISTS evidence_document_version_id UUID\n            REFERENCES evidence_document_versions(id) ON DELETE CASCADE;\n    ALTER TABLE evidence_chunk_registry ADD COLUMN IF NOT EXISTS section_title TEXT;\n    ',
    '\n    CREATE INDEX IF NOT EXISTS evidence_chunk_registry_version_idx\n        ON evidence_chunk_registry (evidence_document_version_id);\n    ',
    '\n    CREATE TABLE IF NOT EXISTS query_debug_traces (\n        id UUID PRIMARY KEY,\n        patient_profile_id UUID NOT NULL\n            REFERENCES patient_profiles(id) ON DELETE CASCADE,\n        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n        trace JSONB NOT NULL\n    );\n    ',
    '\n    CREATE INDEX IF NOT EXISTS query_debug_traces_profile_idx\n        ON query_debug_traces (patient_profile_id, created_at DESC);\n    ',
]

def upgrade() -> None:
    for statement in FROZEN_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS query_debug_traces;")

    op.execute("DROP INDEX IF EXISTS evidence_chunk_registry_version_idx;")
    op.execute("""
        ALTER TABLE evidence_chunk_registry
            DROP COLUMN IF EXISTS evidence_document_version_id,
            DROP COLUMN IF EXISTS section_title;
    """)
    op.execute("""
        ALTER TABLE evidence_documents
            DROP COLUMN IF EXISTS current_version_id,
            DROP COLUMN IF EXISTS latest_content_hash;
    """)
    op.execute("DROP INDEX IF EXISTS evidence_document_versions_current_idx;")
    op.execute("""
        ALTER TABLE evidence_document_versions
            DROP COLUMN IF EXISTS is_current,
            DROP COLUMN IF EXISTS superseded_by;
    """)
