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

            # ── Phase 0: patient-owned profile (added 2026-08-12) ───────
            # The tables above model a clinical chart owned by a physician
            # (physician_id NOT NULL) with an optional user_id link bolted
            # on afterwards. That's backwards for a consumer product: a
            # patient who registers on their own, with no physician
            # involved yet, still needs somewhere to keep their own
            # profile and build a longitudinal record.
            #
            # patient_profiles is that anchor. It is owned by user_id and
            # exists independent of any physician chart. A profile can
            # optionally be associated with one or more physician-owned
            # `patients` rows (the pre-existing chart model) through
            # patient_care_team_links — that's how a patient with multiple
            # clinicians (medical oncology, radiation, surgery, PCP...)
            # is modeled, which a single physician_id column never could.
            #
            # Every Phase 1 longitudinal table below (treatment_episodes,
            # lab_results, medication_exposures, etc.) is keyed off
            # patient_profile_id, not the legacy patients.id, so a patient
            # who never links a clinician still gets the full record.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    date_of_birth DATE,
                    sex TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS patient_profiles_user_id_uidx
                    ON patient_profiles (user_id);
                """
            )

            # A patient_profile can be connected to multiple clinicians,
            # each with a role, rather than the single physician_id the
            # legacy `patients` table assumes. legacy_patient_record_id is
            # optional: it is set when the link was established through
            # the pre-existing invite/request flow against a physician's
            # chart (patient_link_service), so that chart's data
            # (diagnosis/biomarkers/treatment_history rows keyed by
            # patients.id) can still be pulled in as one input to this
            # profile's longitudinal record. A link with no legacy record
            # (status can still be 'active') represents a clinician a
            # patient added directly from their own side, with no chart on
            # the physician's end at all.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_care_team_links (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    physician_id UUID NOT NULL,
                    role TEXT NOT NULL DEFAULT 'oncologist',
                    legacy_patient_record_id UUID
                        REFERENCES patients(id) ON DELETE SET NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    is_primary BOOLEAN NOT NULL DEFAULT false,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_care_team_links_profile_idx
                    ON patient_care_team_links (patient_profile_id, status);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_care_team_links_physician_idx
                    ON patient_care_team_links (physician_id, status);
                """
            )
            # One active link per (profile, physician) pair — reconnecting
            # the same clinician updates the existing row rather than
            # creating a duplicate.
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS patient_care_team_links_active_uidx
                    ON patient_care_team_links (patient_profile_id, physician_id)
                    WHERE status = 'active';
                """
            )

            # ── Phase 1: longitudinal patient record (added 2026-08-12) ─
            # Every table here carries the provenance columns the beta
            # audit called out as missing: source_type (who/what asserted
            # this fact), verification_status (has a human confirmed it),
            # and source_document_id (which upload it was extracted from,
            # if any). None of these are enforced with a cross-table FK to
            # patient_documents.id below on purpose — application code
            # resolves it, matching the physician_id convention already
            # used throughout this file for cross-database references;
            # here it's cross-concern (a fact can predate the document
            # table existing for it) rather than cross-database.

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_comorbidities (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    condition_name TEXT NOT NULL,
                    canonical_code TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    onset_date DATE,
                    resolved_date DATE,
                    raw_text TEXT,
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_comorbidities_profile_idx
                    ON patient_comorbidities (patient_profile_id, status);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_allergies (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    allergen TEXT NOT NULL,
                    reaction TEXT,
                    severity TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_allergies_profile_idx
                    ON patient_allergies (patient_profile_id, status);
                """
            )

            # Diagnoses/tumor profiles/biomarkers, re-homed onto
            # patient_profile_id. The legacy patient_diagnosis /
            # patient_biomarkers tables above stay exactly as they are —
            # still keyed by the physician-owned patients.id — because
            # existing clinician-side code reads them directly. These new
            # tables are the patient-owned counterpart; patient_state_service
            # (Phase 1) merges both when a legacy link exists rather than
            # picking one as canonical.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_diagnoses (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    cancer_site TEXT,
                    histology TEXT,
                    stage TEXT,
                    tnm_t TEXT,
                    tnm_n TEXT,
                    tnm_m TEXT,
                    diagnosis_date DATE,
                    canonical_site_code TEXT,
                    raw_text TEXT,
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_diagnoses_profile_idx
                    ON patient_diagnoses (patient_profile_id, created_at DESC);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_biomarker_results (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    biomarker_name TEXT NOT NULL,
                    value TEXT,
                    canonical_code TEXT,
                    measured_date DATE,
                    raw_text TEXT,
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_biomarker_results_profile_idx
                    ON patient_biomarker_results (patient_profile_id, biomarker_name);
                """
            )

            # Treatments as episode -> cycle -> agent, so "I'm on FOLFOX"
            # resolves to its constituent drugs (oxaliplatin, fluorouracil,
            # leucovorin) for medication-specific retrieval, rather than
            # staying an opaque regimen string.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS treatment_episodes (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    regimen TEXT,
                    modality TEXT,
                    intent TEXT,
                    line_of_therapy INTEGER,
                    start_date DATE,
                    end_date DATE,
                    status TEXT NOT NULL DEFAULT 'active',
                    raw_text TEXT,
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS treatment_episodes_profile_idx
                    ON treatment_episodes (patient_profile_id, start_date DESC NULLS LAST);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS treatment_cycles (
                    id UUID PRIMARY KEY,
                    treatment_episode_id UUID NOT NULL
                        REFERENCES treatment_episodes(id) ON DELETE CASCADE,
                    cycle_number INTEGER,
                    cycle_date DATE,
                    status TEXT NOT NULL DEFAULT 'completed',
                    notes TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS treatment_cycles_episode_idx
                    ON treatment_cycles (treatment_episode_id, cycle_number);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS treatment_agents (
                    id UUID PRIMARY KEY,
                    treatment_episode_id UUID NOT NULL
                        REFERENCES treatment_episodes(id) ON DELETE CASCADE,
                    agent_name TEXT NOT NULL,
                    rxnorm_code TEXT,
                    dose TEXT,
                    route TEXT,
                    schedule TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS treatment_agents_episode_idx
                    ON treatment_agents (treatment_episode_id);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS treatment_agents_name_idx
                    ON treatment_agents (agent_name);
                """
            )

            # Non-cancer medications matter for retrieval (drug interaction
            # / eligibility questions) as much as the regimen itself.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS medication_exposures (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    generic_name TEXT NOT NULL,
                    brand_name TEXT,
                    rxnorm_code TEXT,
                    dose TEXT,
                    route TEXT,
                    frequency TEXT,
                    indication TEXT,
                    start_date DATE,
                    end_date DATE,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS medication_exposures_profile_idx
                    ON medication_exposures (patient_profile_id, status);
                """
            )

            # Labs: the longitudinal table the beta audit flagged as
            # entirely absent. unit/reference range/abnormal_flag travel
            # with the value so a trend (e.g. ANC 4.2 -> 2.1 -> 0.7) can be
            # read directly rather than re-derived downstream.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lab_results (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    loinc_code TEXT,
                    test_name TEXT NOT NULL,
                    canonical_test_name TEXT,
                    value_numeric DOUBLE PRECISION,
                    value_text TEXT,
                    unit TEXT,
                    reference_low DOUBLE PRECISION,
                    reference_high DOUBLE PRECISION,
                    abnormal_flag TEXT,
                    specimen_type TEXT,
                    collected_at TIMESTAMPTZ,
                    resulted_at TIMESTAMPTZ,
                    source_type TEXT NOT NULL DEFAULT 'patient_upload',
                    source_document_id UUID,
                    verification_status TEXT NOT NULL DEFAULT 'extracted',
                    extraction_confidence REAL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS lab_results_profile_idx
                    ON lab_results (patient_profile_id, collected_at DESC);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS lab_results_test_trend_idx
                    ON lab_results (patient_profile_id, canonical_test_name, collected_at DESC);
                """
            )

            # Vitals/weight, kept separate from lab_results (different
            # collection context — bedside/home, not a specimen) but same
            # trend-query shape. Nutrition risk in patient_state_service
            # reads weight_kg trend from here.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_vitals (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    vital_type TEXT NOT NULL,
                    value_numeric DOUBLE PRECISION,
                    unit TEXT,
                    measured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    source_type TEXT NOT NULL DEFAULT 'patient_manual',
                    source_document_id UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_vitals_trend_idx
                    ON patient_vitals (patient_profile_id, vital_type, measured_at DESC);
                """
            )

            # Appointments as a first-class object, so "what changed at my
            # last visit" has somewhere to live and can feed the timeline.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS encounters (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    encounter_date DATE,
                    encounter_type TEXT,
                    provider_name TEXT,
                    organization TEXT,
                    patient_summary TEXT,
                    clinician_note TEXT,
                    structured_changes JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source_document_id UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS encounters_profile_idx
                    ON encounters (patient_profile_id, encounter_date DESC NULLS LAST);
                """
            )

            # Uploaded documents: the object itself, kept and linked from
            # every fact extracted from it, plus the extraction record
            # (raw candidate values pending patient confirmation — see
            # patient_document_validator.py). Splitting the two lets a
            # document be re-parsed without losing the original.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_documents (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    document_type TEXT NOT NULL DEFAULT 'unclassified',
                    filename TEXT,
                    content_type TEXT,
                    object_storage_uri TEXT,
                    document_date DATE,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    extraction_status TEXT NOT NULL DEFAULT 'pending',
                    parser_version TEXT,
                    error_message TEXT
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_documents_profile_idx
                    ON patient_documents (patient_profile_id, uploaded_at DESC);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_extractions (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL
                        REFERENCES patient_documents(id) ON DELETE CASCADE,
                    extracted_fields JSONB NOT NULL DEFAULT '{}'::jsonb,
                    extraction_confidence REAL,
                    extraction_method TEXT,
                    confirmed BOOLEAN NOT NULL DEFAULT false,
                    confirmed_at TIMESTAMPTZ,
                    confirmed_fields JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS document_extractions_document_idx
                    ON document_extractions (document_id);
                """
            )

            # Cached, precomputed "as of now" view of a patient, so the
            # chat/retrieval path reads one row instead of re-querying
            # every table above on every question. Rebuilt by
            # patient_state_service whenever a fact changes.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_state_snapshots (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
                    as_of TIMESTAMPTZ NOT NULL DEFAULT now(),
                    state JSONB NOT NULL,
                    retrieval_features JSONB NOT NULL DEFAULT '{}'::jsonb,
                    rule_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS patient_state_snapshots_profile_idx
                    ON patient_state_snapshots (patient_profile_id, created_at DESC);
                """
            )

            # Append-only timeline for patient_profile-based facts.
            # Deliberately a separate table from patient_timeline_events
            # above rather than a reuse: that table's patient_id column is
            # a hard FK to the physician-owned patients.id, which an
            # unlinked consumer never has. Same append-only contract (no
            # update/delete method on the service layer) and the same
            # purpose — a structured history for patient_state_service and
            # a future diff/change-detection layer to read.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_profile_timeline_events (
                    id UUID PRIMARY KEY,
                    patient_profile_id UUID NOT NULL
                        REFERENCES patient_profiles(id) ON DELETE CASCADE,
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
                CREATE INDEX IF NOT EXISTS patient_profile_timeline_events_profile_idx
                    ON patient_profile_timeline_events (patient_profile_id, recorded_at DESC);
                """
            )

            # ── Evidence source registry (Phase 3, added 2026-08-12) ────
            # Non-PHI. Lives in this database because it is part of the
            # same patient-facing evidence system being built out here,
            # not because it belongs conceptually with clinical PHI — see
            # clinical_inference.py-style separation notes in
            # multi_corpus_retriever.py. Operational configuration for
            # *what* gets ingested into which Qdrant collection, replacing
            # a trusted-source list that would otherwise live only in a
            # prompt or be hardcoded across retrieval code.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_sources (
                    id UUID PRIMARY KEY,
                    source_key TEXT NOT NULL,
                    name TEXT NOT NULL,
                    domain TEXT,
                    authority_class TEXT NOT NULL DEFAULT 'B',
                    authority_score REAL NOT NULL DEFAULT 0.5,
                    source_type TEXT NOT NULL DEFAULT 'patient_education',
                    allowed_intents JSONB NOT NULL DEFAULT '[]'::jsonb,
                    patient_facing BOOLEAN NOT NULL DEFAULT true,
                    ingestion_method TEXT NOT NULL DEFAULT 'manual',
                    license_status TEXT NOT NULL DEFAULT 'unknown',
                    active BOOLEAN NOT NULL DEFAULT true,
                    last_verified_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS evidence_sources_key_uidx
                    ON evidence_sources (source_key);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_documents (
                    id UUID PRIMARY KEY,
                    source_id UUID NOT NULL REFERENCES evidence_sources(id) ON DELETE CASCADE,
                    doc_id TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    qdrant_collection TEXT NOT NULL,
                    applicability JSONB NOT NULL DEFAULT '{}'::jsonb,
                    constraints JSONB NOT NULL DEFAULT '{}'::jsonb,
                    last_ingested_at TIMESTAMPTZ,
                    active BOOLEAN NOT NULL DEFAULT true,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS evidence_documents_source_idx
                    ON evidence_documents (source_id);
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS evidence_documents_doc_id_uidx
                    ON evidence_documents (doc_id);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_document_versions (
                    id UUID PRIMARY KEY,
                    evidence_document_id UUID NOT NULL
                        REFERENCES evidence_documents(id) ON DELETE CASCADE,
                    version_number INTEGER NOT NULL DEFAULT 1,
                    content_hash TEXT,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    raw_text_excerpt TEXT
                );
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_ingestion_runs (
                    id UUID PRIMARY KEY,
                    source_id UUID REFERENCES evidence_sources(id) ON DELETE SET NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    finished_at TIMESTAMPTZ,
                    status TEXT NOT NULL DEFAULT 'running',
                    documents_ingested INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_chunk_registry (
                    id UUID PRIMARY KEY,
                    evidence_document_id UUID NOT NULL
                        REFERENCES evidence_documents(id) ON DELETE CASCADE,
                    qdrant_point_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS evidence_chunk_registry_document_idx
                    ON evidence_chunk_registry (evidence_document_id);
                """
            )

            # ── Community subsystem (Phase 7, added 2026-08-12) ─────────
            # Structurally and physically separate from every table above:
            # no foreign key here ever points at patient_profiles or vice
            # versa. A community post is addressed by community_profile
            # (a pseudonymous handle), never by patient_profile_id or
            # user_id directly in content-facing reads — see
            # community_service.py for the join-only-at-write boundary
            # this is meant to enforce.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_profiles (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    handle TEXT NOT NULL,
                    avatar_key TEXT,
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS community_profiles_user_uidx
                    ON community_profiles (user_id);
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS community_profiles_handle_uidx
                    ON community_profiles (handle);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS communities (
                    id UUID PRIMARY KEY,
                    slug TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS communities_slug_uidx
                    ON communities (slug);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_memberships (
                    id UUID PRIMARY KEY,
                    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
                    community_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    joined_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS community_memberships_uidx
                    ON community_memberships (community_id, community_profile_id);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_posts (
                    id UUID PRIMARY KEY,
                    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
                    community_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    title TEXT,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'visible',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS community_posts_community_idx
                    ON community_posts (community_id, created_at DESC);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_comments (
                    id UUID PRIMARY KEY,
                    post_id UUID NOT NULL REFERENCES community_posts(id) ON DELETE CASCADE,
                    community_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'visible',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS community_comments_post_idx
                    ON community_comments (post_id, created_at);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_reactions (
                    id UUID PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_id UUID NOT NULL,
                    community_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    reaction TEXT NOT NULL DEFAULT 'support',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS community_reactions_uidx
                    ON community_reactions (target_type, target_id, community_profile_id);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_reports (
                    id UUID PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_id UUID NOT NULL,
                    reported_by_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    reason TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS community_reports_status_idx
                    ON community_reports (status, created_at DESC);
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_moderation_actions (
                    id UUID PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_id UUID NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT,
                    acted_by UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS community_blocked_users (
                    id UUID PRIMARY KEY,
                    blocker_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    blocked_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS community_blocked_users_uidx
                    ON community_blocked_users (blocker_profile_id, blocked_profile_id);
                """
            )


_db_instance: Optional[PatientDatabase] = None


def get_patient_db() -> PatientDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = PatientDatabase()
    return _db_instance
