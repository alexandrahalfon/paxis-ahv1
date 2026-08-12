"""phase 1 finalization: multi-primary diagnoses, tumor_profiles,
symptom_observations, nutrition_assessments, care_team_instructions,
normalization columns

Revision ID: 0002_phase1_finalization
Revises: 0001_baseline
Create Date: 2026-08-12

upgrade() runs SCHEMA_STATEMENTS[92:] from
src/api/services/patient_schema.py — everything appended after the
baseline's 92 statements, i.e. exactly the "Phase 1 finalization" block
that module's file documents. Same zero-duplication guarantee as
0001_baseline: this migration and ensure_schema() execute the identical
statement list, so they cannot drift apart.

Unlike 0001_baseline, every change here is purely additive (new tables,
new nullable/defaulted columns) and genuinely reversible, so downgrade()
does real work instead of refusing.
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

revision: str = "0002_phase1_finalization"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The baseline (0001) covers statements [0:92]; this revision owns
# everything appended after it. Asserted, not just assumed, so a future
# edit to patient_schema.py that changes the split point fails loudly at
# migration time instead of silently upgrading the wrong statements.
_BASELINE_STATEMENT_COUNT = 92


def upgrade() -> None:
    new_statements = SCHEMA_STATEMENTS[_BASELINE_STATEMENT_COUNT:]
    assert len(new_statements) == 18, (
        f"Expected 18 Phase 1 finalization statements after the baseline's "
        f"{_BASELINE_STATEMENT_COUNT}, found {len(new_statements)}. "
        "patient_schema.py changed — add a new revision instead of editing "
        "this one, or update this assertion if 0002 itself is being amended "
        "before it has shipped anywhere."
    )
    for statement in new_statements:
        op.execute(statement)


def downgrade() -> None:
    # Reverse creation order. Index drops before the column drops they'd
    # otherwise depend on.
    op.execute("DROP TABLE IF EXISTS care_team_instructions;")
    op.execute("DROP TABLE IF EXISTS nutrition_assessments;")
    op.execute("DROP TABLE IF EXISTS symptom_observations;")

    op.execute("""
        ALTER TABLE encounters
            DROP COLUMN IF EXISTS newly_ordered_tests,
            DROP COLUMN IF EXISTS next_steps,
            DROP COLUMN IF EXISTS questions_for_next_visit;
    """)
    op.execute("ALTER TABLE patient_allergies DROP COLUMN IF EXISTS allergy_type;")
    op.execute("""
        ALTER TABLE treatment_agents
            DROP COLUMN IF EXISTS canonical_name,
            DROP COLUMN IF EXISTS aliases;
    """)
    op.execute("""
        ALTER TABLE medication_exposures
            DROP COLUMN IF EXISTS canonical_name,
            DROP COLUMN IF EXISTS aliases;
    """)
    op.execute("""
        ALTER TABLE treatment_cycles
            DROP COLUMN IF EXISTS delayed,
            DROP COLUMN IF EXISTS delay_reason,
            DROP COLUMN IF EXISTS held,
            DROP COLUMN IF EXISTS dose_reduction_pct;
    """)
    op.execute("""
        ALTER TABLE patient_biomarker_results
            DROP COLUMN IF EXISTS specimen_date,
            DROP COLUMN IF EXISTS specimen_site,
            DROP COLUMN IF EXISTS biomarker_category,
            DROP COLUMN IF EXISTS canonical_gene;
    """)

    op.execute("DROP TABLE IF EXISTS tumor_profiles;")

    op.execute("DROP INDEX IF EXISTS patient_diagnoses_status_idx;")
    op.execute("""
        ALTER TABLE patient_diagnoses
            DROP COLUMN IF EXISTS diagnosis_type,
            DROP COLUMN IF EXISTS status,
            DROP COLUMN IF EXISTS effective_date,
            DROP COLUMN IF EXISTS canonical_cancer_type,
            DROP COLUMN IF EXISTS canonical_histology,
            DROP COLUMN IF EXISTS stage_system,
            DROP COLUMN IF EXISTS metastatic_sites,
            DROP COLUMN IF EXISTS related_diagnosis_id;
    """)

    op.execute("""
        ALTER TABLE patient_profiles
            DROP COLUMN IF EXISTS preferred_language,
            DROP COLUMN IF EXISTS timezone;
    """)
