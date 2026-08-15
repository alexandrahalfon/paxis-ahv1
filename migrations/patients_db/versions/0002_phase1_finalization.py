"""phase 1 finalization: multi-primary diagnoses, tumor_profiles,
symptom_observations, nutrition_assessments, care_team_instructions,
normalization columns

Revision ID: 0002_phase1_finalization
Revises: 0001_baseline
Create Date: 2026-08-12

FROZEN_STATEMENTS below is a literal, historical snapshot of
src/api/services/patient_schema.py SCHEMA_STATEMENTS[92:110] (exactly the
18 "Phase 1 finalization" statements) as it stood when this revision was
written — see 0001_baseline's docstring for why this is a frozen copy and
not a live import. The previous version of this file computed
`SCHEMA_STATEMENTS[92:]` (open-ended, not `[92:110]`) and asserted the
result was 18 statements long; that assertion was already failing by the
time evidence-versioning and query_debug_traces statements were appended
later in the same list (open-ended slice picked those up too, growing to
25), which would have raised on any `alembic upgrade head` run against a
fresh database. Freezing removes the possibility of that drift entirely
— see migrations/patients_db/README.md.

Unlike 0001_baseline, every change here is purely additive (new tables,
new nullable/defaulted columns) and genuinely reversible, so downgrade()
does real work instead of refusing.
"""
from __future__ import annotations

from typing import List, Sequence, Union

from alembic import op

revision: str = "0002_phase1_finalization"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FROZEN_STATEMENTS: List[str] = [
    '\n    ALTER TABLE patient_profiles ADD COLUMN IF NOT EXISTS preferred_language TEXT;\n    ALTER TABLE patient_profiles ADD COLUMN IF NOT EXISTS timezone TEXT;\n    ',
    "\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS diagnosis_type TEXT NOT NULL DEFAULT 'primary';\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS effective_date DATE;\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS canonical_cancer_type TEXT;\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS canonical_histology TEXT;\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS stage_system TEXT;\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS metastatic_sites JSONB NOT NULL DEFAULT '[]'::jsonb;\n    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS related_diagnosis_id UUID\n        REFERENCES patient_diagnoses(id) ON DELETE SET NULL;\n    ",
    '\n    CREATE INDEX IF NOT EXISTS patient_diagnoses_status_idx\n        ON patient_diagnoses (patient_profile_id, status);\n    ',
    "\n    CREATE TABLE IF NOT EXISTS tumor_profiles (\n        id UUID PRIMARY KEY,\n        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,\n        diagnosis_id UUID REFERENCES patient_diagnoses(id) ON DELETE SET NULL,\n        grade TEXT,\n        tumor_size_mm DOUBLE PRECISION,\n        molecular_subtype TEXT,\n        receptor_status JSONB NOT NULL DEFAULT '{}'::jsonb,\n        specimen_date DATE,\n        specimen_site TEXT,\n        raw_text TEXT,\n        source_type TEXT NOT NULL DEFAULT 'patient_manual',\n        source_document_id UUID,\n        verification_status TEXT NOT NULL DEFAULT 'extracted',\n        created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n    );\n    ",
    '\n    CREATE INDEX IF NOT EXISTS tumor_profiles_profile_idx\n        ON tumor_profiles (patient_profile_id, created_at DESC);\n    ',
    '\n    CREATE INDEX IF NOT EXISTS tumor_profiles_diagnosis_idx\n        ON tumor_profiles (diagnosis_id);\n    ',
    '\n    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS specimen_date DATE;\n    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS specimen_site TEXT;\n    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS biomarker_category TEXT;\n    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS canonical_gene TEXT;\n    ',
    '\n    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS delayed BOOLEAN NOT NULL DEFAULT false;\n    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS delay_reason TEXT;\n    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS held BOOLEAN NOT NULL DEFAULT false;\n    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS dose_reduction_pct REAL;\n    ',
    "\n    ALTER TABLE treatment_agents ADD COLUMN IF NOT EXISTS canonical_name TEXT;\n    ALTER TABLE treatment_agents ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb;\n    ALTER TABLE medication_exposures ADD COLUMN IF NOT EXISTS canonical_name TEXT;\n    ALTER TABLE medication_exposures ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb;\n    ",
    "\n    CREATE TABLE IF NOT EXISTS symptom_observations (\n        id UUID PRIMARY KEY,\n        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,\n        raw_text TEXT NOT NULL,\n        canonical_symptom TEXT,\n        severity SMALLINT,\n        onset_date DATE,\n        resolved_date DATE,\n        status TEXT NOT NULL DEFAULT 'active',\n        frequency TEXT,\n        possibly_related_treatment_episode_id UUID\n            REFERENCES treatment_episodes(id) ON DELETE SET NULL,\n        source_type TEXT NOT NULL DEFAULT 'patient_manual',\n        source_document_id UUID,\n        verification_status TEXT NOT NULL DEFAULT 'extracted',\n        created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n    );\n    ",
    '\n    CREATE INDEX IF NOT EXISTS symptom_observations_profile_idx\n        ON symptom_observations (patient_profile_id, status, onset_date DESC);\n    ',
    '\n    CREATE INDEX IF NOT EXISTS symptom_observations_canonical_idx\n        ON symptom_observations (patient_profile_id, canonical_symptom);\n    ',
    "\n    CREATE TABLE IF NOT EXISTS nutrition_assessments (\n        id UUID PRIMARY KEY,\n        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,\n        assessment_date DATE NOT NULL DEFAULT CURRENT_DATE,\n        appetite TEXT,\n        oral_intake_pct SMALLINT,\n        swallowing_difficulty BOOLEAN,\n        feeding_tube BOOLEAN NOT NULL DEFAULT false,\n        feeding_tube_type TEXT,\n        diet_restrictions JSONB NOT NULL DEFAULT '[]'::jsonb,\n        food_allergies JSONB NOT NULL DEFAULT '[]'::jsonb,\n        texture_requirements TEXT,\n        hydration_constraints TEXT,\n        nutrition_risk TEXT,\n        care_phase TEXT NOT NULL DEFAULT 'active_treatment',\n        source_type TEXT NOT NULL DEFAULT 'patient_manual',\n        source_document_id UUID,\n        verification_status TEXT NOT NULL DEFAULT 'extracted',\n        created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n    );\n    ",
    '\n    CREATE INDEX IF NOT EXISTS nutrition_assessments_profile_idx\n        ON nutrition_assessments (patient_profile_id, assessment_date DESC);\n    ',
    "\n    ALTER TABLE patient_allergies ADD COLUMN IF NOT EXISTS allergy_type TEXT NOT NULL DEFAULT 'allergy';\n    ",
    "\n    ALTER TABLE encounters ADD COLUMN IF NOT EXISTS newly_ordered_tests JSONB NOT NULL DEFAULT '[]'::jsonb;\n    ALTER TABLE encounters ADD COLUMN IF NOT EXISTS next_steps TEXT;\n    ALTER TABLE encounters ADD COLUMN IF NOT EXISTS questions_for_next_visit JSONB NOT NULL DEFAULT '[]'::jsonb;\n    ",
    "\n    CREATE TABLE IF NOT EXISTS care_team_instructions (\n        id UUID PRIMARY KEY,\n        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,\n        instruction_text TEXT NOT NULL,\n        instruction_type TEXT NOT NULL DEFAULT 'other',\n        author_provider TEXT,\n        physician_id UUID,\n        source_type TEXT NOT NULL DEFAULT 'clinician_entered',\n        source_document_id UUID,\n        effective_from DATE,\n        effective_to DATE,\n        active BOOLEAN NOT NULL DEFAULT true,\n        created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n    );\n    ",
    '\n    CREATE INDEX IF NOT EXISTS care_team_instructions_profile_idx\n        ON care_team_instructions (patient_profile_id, active, created_at DESC);\n    ',
]

def upgrade() -> None:
    for statement in FROZEN_STATEMENTS:
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
