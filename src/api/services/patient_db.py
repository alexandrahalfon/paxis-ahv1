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

IMPORTANT: ensure_schema() is additive-only (CREATE TABLE/INDEX IF NOT
EXISTS). It is not called anywhere yet — nothing runs against the live
exueed-patients database until this is wired up deliberately and reviewed.
"""

from typing import Optional

import asyncpg

from src.core.config import settings


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

    async def ensure_schema(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id UUID PRIMARY KEY,
                    physician_id UUID NOT NULL,
                    mrn TEXT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    date_of_birth DATE,
                    sex TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patients_physician_id_idx
                    ON patients (physician_id);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_diagnosis (
                    id UUID PRIMARY KEY,
                    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                    cancer_site TEXT,
                    histology TEXT,
                    stage TEXT,
                    tnm_t TEXT,
                    tnm_n TEXT,
                    tnm_m TEXT,
                    diagnosis_date DATE,
                    canonical_site_code TEXT,
                    raw_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_diagnosis_patient_id_idx
                    ON patient_diagnosis (patient_id);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_biomarkers (
                    id UUID PRIMARY KEY,
                    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                    biomarker_name TEXT NOT NULL,
                    value TEXT,
                    canonical_code TEXT,
                    measured_date DATE,
                    raw_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_biomarkers_patient_id_idx
                    ON patient_biomarkers (patient_id);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_biomarkers_name_idx
                    ON patient_biomarkers (patient_id, biomarker_name);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_treatment_history (
                    id UUID PRIMARY KEY,
                    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                    treatment_type TEXT NOT NULL,
                    regimen TEXT,
                    line_of_therapy INTEGER,
                    start_date DATE,
                    end_date DATE,
                    status TEXT,
                    raw_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_treatment_history_patient_id_idx
                    ON patient_treatment_history (patient_id);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_treatment_history_timeline_idx
                    ON patient_treatment_history (patient_id, start_date DESC);
                """
            )

            # Append-only: every patient update writes a new row here rather
            # than mutating existing state. pattern_diff_service (Phase 4)
            # reads the two most recent rows per patient to detect changes.
            # No UPDATE/DELETE methods should be exposed on this table by the
            # service layer built on top of this schema.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_timeline_events (
                    id UUID PRIMARY KEY,
                    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    event_date DATE,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_by UUID
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_timeline_events_patient_recorded_idx
                    ON patient_timeline_events (patient_id, recorded_at DESC);
                """
            )

            # ── Patient portal (added 2026-08-08) ──────────────────────
            # All additive. A patient record stays exactly what it is
            # today (a chart owned by a physician); these columns only
            # add the optional link to a patient's own login. Existing
            # records keep user_id NULL and behave unchanged.
            await conn.execute(
                """
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS user_id UUID;
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS invite_code TEXT;
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS invite_created_at TIMESTAMPTZ;
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS linked_at TIMESTAMPTZ;
                ALTER TABLE patients
                    ADD COLUMN IF NOT EXISTS link_status TEXT NOT NULL DEFAULT 'unlinked';
                """
            )
            # Partial unique index: one patient record per patient login,
            # but unlimited NULLs for the physician-only records.
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS patients_user_id_uidx
                    ON patients (user_id) WHERE user_id IS NOT NULL;
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS patients_invite_code_uidx
                    ON patients (invite_code) WHERE invite_code IS NOT NULL;
                """
            )

            # Connection requests from patients who signed up without an
            # invite code. A physician must approve before any link exists,
            # so picking a name from a list never by itself grants access.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_link_requests (
                    id UUID PRIMARY KEY,
                    patient_user_id UUID NOT NULL,
                    physician_id UUID NOT NULL,
                    patient_first_name TEXT,
                    patient_last_name TEXT,
                    date_of_birth DATE,
                    note TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    resolved_at TIMESTAMPTZ,
                    resolved_by UUID,
                    linked_patient_id UUID
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_link_requests_physician_idx
                    ON patient_link_requests (physician_id, status, created_at DESC);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_link_requests_user_idx
                    ON patient_link_requests (patient_user_id, status);
                """
            )

            # Patient conversations. Previously nothing was persisted: the
            # browser held the history and it vanished on refresh. Storing
            # it lets a patient come back to an answer, and lets an
            # escalated question carry its context to the physician.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_conversations (
                    id UUID PRIMARY KEY,
                    patient_user_id UUID NOT NULL,
                    patient_record_id UUID,
                    title TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_conversations_user_idx
                    ON patient_conversations (patient_user_id, updated_at DESC);
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_messages (
                    id UUID PRIMARY KEY,
                    conversation_id UUID NOT NULL
                        REFERENCES patient_conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    safety_category TEXT,
                    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_messages_conversation_idx
                    ON patient_messages (conversation_id, created_at);
                """
            )

            # Escalations: the physician inbox. ai_draft_answer is what
            # makes this save the physician time rather than add to their
            # load — they approve or edit a draft instead of composing.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_escalations (
                    id UUID PRIMARY KEY,
                    patient_user_id UUID NOT NULL,
                    patient_record_id UUID,
                    physician_id UUID NOT NULL,
                    conversation_id UUID,
                    question TEXT NOT NULL,
                    ai_draft_answer TEXT,
                    context_summary TEXT,
                    urgency TEXT NOT NULL DEFAULT 'routine',
                    status TEXT NOT NULL DEFAULT 'open',
                    physician_response TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    answered_at TIMESTAMPTZ
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_escalations_physician_idx
                    ON patient_escalations (physician_id, status, created_at DESC);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_escalations_user_idx
                    ON patient_escalations (patient_user_id, created_at DESC);
                """
            )

            # Symptom diary. Patient-reported only: this is what someone
            # noticed and typed, never a clinical assessment, and the
            # column names say so. Its value is turning "the last few
            # weeks were rough" into something specific to show the care
            # team at the next visit.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_symptom_entries (
                    id UUID PRIMARY KEY,
                    patient_user_id UUID NOT NULL,
                    patient_record_id UUID,
                    symptom TEXT NOT NULL,
                    severity SMALLINT,
                    noted_on DATE NOT NULL DEFAULT CURRENT_DATE,
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_symptom_entries_user_idx
                    ON patient_symptom_entries (patient_user_id, noted_on DESC);
                """
            )


_db_instance: Optional[PatientDatabase] = None


def get_patient_db() -> PatientDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = PatientDatabase()
    return _db_instance
