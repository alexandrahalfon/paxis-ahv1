"""
Schema for the exueed-patients database.

Single source of truth for every CREATE TABLE / CREATE INDEX / ALTER
TABLE statement this app issues against that database. Both the
runtime ensure_schema() path (patient_db.py) and the Alembic baseline
migration (migrations/patients_db/versions/0001_baseline.py) execute
this exact list, so there is exactly one place that can drift from the
other — nowhere.

Every statement is idempotent (IF NOT EXISTS / ADD COLUMN IF NOT
EXISTS), so running the full list against a database that already has
some or all of these objects — e.g. one that predates the Alembic
migration — is always safe and a no-op for anything already present.
"""

from typing import List

SCHEMA_STATEMENTS: List[str] = [
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patients_physician_id_idx
                    ON patients (physician_id);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_diagnosis_patient_id_idx
                    ON patient_diagnosis (patient_id);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_biomarkers_patient_id_idx
                    ON patient_biomarkers (patient_id);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_biomarkers_name_idx
                    ON patient_biomarkers (patient_id, biomarker_name);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_treatment_history_patient_id_idx
                    ON patient_treatment_history (patient_id);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_treatment_history_timeline_idx
                    ON patient_treatment_history (patient_id, start_date DESC);
                
    """,

    # Append-only: every patient update writes a new row here rather
    # than mutating existing state. pattern_diff_service (Phase 4)
    # reads the two most recent rows per patient to detect changes.
    # No UPDATE/DELETE methods should be exposed on this table by the
    # service layer built on top of this schema.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_timeline_events_patient_recorded_idx
                    ON patient_timeline_events (patient_id, recorded_at DESC);
                
    """,

    # ── Patient portal (added 2026-08-08) ──────────────────────
    # All additive. A patient record stays exactly what it is
    # today (a chart owned by a physician); these columns only
    # add the optional link to a patient's own login. Existing
    # records keep user_id NULL and behave unchanged.
    """
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS user_id UUID;
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS invite_code TEXT;
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS invite_created_at TIMESTAMPTZ;
                ALTER TABLE patients ADD COLUMN IF NOT EXISTS linked_at TIMESTAMPTZ;
                ALTER TABLE patients
                    ADD COLUMN IF NOT EXISTS link_status TEXT NOT NULL DEFAULT 'unlinked';
                
    """,
    # Partial unique index: one patient record per patient login,
    # but unlimited NULLs for the physician-only records.
    """
                CREATE UNIQUE INDEX IF NOT EXISTS patients_user_id_uidx
                    ON patients (user_id) WHERE user_id IS NOT NULL;
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS patients_invite_code_uidx
                    ON patients (invite_code) WHERE invite_code IS NOT NULL;
                
    """,

    # Connection requests from patients who signed up without an
    # invite code. A physician must approve before any link exists,
    # so picking a name from a list never by itself grants access.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_link_requests_physician_idx
                    ON patient_link_requests (physician_id, status, created_at DESC);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_link_requests_user_idx
                    ON patient_link_requests (patient_user_id, status);
                
    """,

    # Patient conversations. Previously nothing was persisted: the
    # browser held the history and it vanished on refresh. Storing
    # it lets a patient come back to an answer, and lets an
    # escalated question carry its context to the physician.
    """
                CREATE TABLE IF NOT EXISTS patient_conversations (
                    id UUID PRIMARY KEY,
                    patient_user_id UUID NOT NULL,
                    patient_record_id UUID,
                    title TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_conversations_user_idx
                    ON patient_conversations (patient_user_id, updated_at DESC);
                
    """,
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_messages_conversation_idx
                    ON patient_messages (conversation_id, created_at);
                
    """,

    # Escalations: the physician inbox. ai_draft_answer is what
    # makes this save the physician time rather than add to their
    # load — they approve or edit a draft instead of composing.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_escalations_physician_idx
                    ON patient_escalations (physician_id, status, created_at DESC);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_escalations_user_idx
                    ON patient_escalations (patient_user_id, created_at DESC);
                
    """,

    # Symptom diary. Patient-reported only: this is what someone
    # noticed and typed, never a clinical assessment, and the
    # column names say so. Its value is turning "the last few
    # weeks were rough" into something specific to show the care
    # team at the next visit.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_symptom_entries_user_idx
                    ON patient_symptom_entries (patient_user_id, noted_on DESC);
                
    """,

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
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS patient_profiles_user_id_uidx
                    ON patient_profiles (user_id);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_care_team_links_profile_idx
                    ON patient_care_team_links (patient_profile_id, status);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_care_team_links_physician_idx
                    ON patient_care_team_links (physician_id, status);
                
    """,
    # One active link per (profile, physician) pair — reconnecting
    # the same clinician updates the existing row rather than
    # creating a duplicate.
    """
                CREATE UNIQUE INDEX IF NOT EXISTS patient_care_team_links_active_uidx
                    ON patient_care_team_links (patient_profile_id, physician_id)
                    WHERE status = 'active';
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_comorbidities_profile_idx
                    ON patient_comorbidities (patient_profile_id, status);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_allergies_profile_idx
                    ON patient_allergies (patient_profile_id, status);
                
    """,

    # Diagnoses/tumor profiles/biomarkers, re-homed onto
    # patient_profile_id. The legacy patient_diagnosis /
    # patient_biomarkers tables above stay exactly as they are —
    # still keyed by the physician-owned patients.id — because
    # existing clinician-side code reads them directly. These new
    # tables are the patient-owned counterpart; patient_state_service
    # (Phase 1) merges both when a legacy link exists rather than
    # picking one as canonical.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_diagnoses_profile_idx
                    ON patient_diagnoses (patient_profile_id, created_at DESC);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_biomarker_results_profile_idx
                    ON patient_biomarker_results (patient_profile_id, biomarker_name);
                
    """,

    # Treatments as episode -> cycle -> agent, so "I'm on FOLFOX"
    # resolves to its constituent drugs (oxaliplatin, fluorouracil,
    # leucovorin) for medication-specific retrieval, rather than
    # staying an opaque regimen string.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS treatment_episodes_profile_idx
                    ON treatment_episodes (patient_profile_id, start_date DESC NULLS LAST);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS treatment_cycles_episode_idx
                    ON treatment_cycles (treatment_episode_id, cycle_number);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS treatment_agents_episode_idx
                    ON treatment_agents (treatment_episode_id);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS treatment_agents_name_idx
                    ON treatment_agents (agent_name);
                
    """,

    # Non-cancer medications matter for retrieval (drug interaction
    # / eligibility questions) as much as the regimen itself.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS medication_exposures_profile_idx
                    ON medication_exposures (patient_profile_id, status);
                
    """,

    # Labs: the longitudinal table the beta audit flagged as
    # entirely absent. unit/reference range/abnormal_flag travel
    # with the value so a trend (e.g. ANC 4.2 -> 2.1 -> 0.7) can be
    # read directly rather than re-derived downstream.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS lab_results_profile_idx
                    ON lab_results (patient_profile_id, collected_at DESC);
                
    """,
    """
                CREATE INDEX IF NOT EXISTS lab_results_test_trend_idx
                    ON lab_results (patient_profile_id, canonical_test_name, collected_at DESC);
                
    """,

    # Vitals/weight, kept separate from lab_results (different
    # collection context — bedside/home, not a specimen) but same
    # trend-query shape. Nutrition risk in patient_state_service
    # reads weight_kg trend from here.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_vitals_trend_idx
                    ON patient_vitals (patient_profile_id, vital_type, measured_at DESC);
                
    """,

    # Appointments as a first-class object, so "what changed at my
    # last visit" has somewhere to live and can feed the timeline.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS encounters_profile_idx
                    ON encounters (patient_profile_id, encounter_date DESC NULLS LAST);
                
    """,

    # Uploaded documents: the object itself, kept and linked from
    # every fact extracted from it, plus the extraction record
    # (raw candidate values pending patient confirmation — see
    # patient_document_validator.py). Splitting the two lets a
    # document be re-parsed without losing the original.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_documents_profile_idx
                    ON patient_documents (patient_profile_id, uploaded_at DESC);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS document_extractions_document_idx
                    ON document_extractions (document_id);
                
    """,

    # Cached, precomputed "as of now" view of a patient, so the
    # chat/retrieval path reads one row instead of re-querying
    # every table above on every question. Rebuilt by
    # patient_state_service whenever a fact changes.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_state_snapshots_profile_idx
                    ON patient_state_snapshots (patient_profile_id, created_at DESC);
                
    """,

    # Append-only timeline for patient_profile-based facts.
    # Deliberately a separate table from patient_timeline_events
    # above rather than a reuse: that table's patient_id column is
    # a hard FK to the physician-owned patients.id, which an
    # unlinked consumer never has. Same append-only contract (no
    # update/delete method on the service layer) and the same
    # purpose — a structured history for patient_state_service and
    # a future diff/change-detection layer to read.
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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS patient_profile_timeline_events_profile_idx
                    ON patient_profile_timeline_events (patient_profile_id, recorded_at DESC);
                
    """,

    # ── Evidence source registry (Phase 3, added 2026-08-12) ────
    # Non-PHI. Lives in this database because it is part of the
    # same patient-facing evidence system being built out here,
    # not because it belongs conceptually with clinical PHI — see
    # clinical_inference.py-style separation notes in
    # multi_corpus_retriever.py. Operational configuration for
    # *what* gets ingested into which Qdrant collection, replacing
    # a trusted-source list that would otherwise live only in a
    # prompt or be hardcoded across retrieval code.
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
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS evidence_sources_key_uidx
                    ON evidence_sources (source_key);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS evidence_documents_source_idx
                    ON evidence_documents (source_id);
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS evidence_documents_doc_id_uidx
                    ON evidence_documents (doc_id);
                
    """,

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
                
    """,

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
                
    """,

    """
                CREATE TABLE IF NOT EXISTS evidence_chunk_registry (
                    id UUID PRIMARY KEY,
                    evidence_document_id UUID NOT NULL
                        REFERENCES evidence_documents(id) ON DELETE CASCADE,
                    qdrant_point_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                
    """,
    """
                CREATE INDEX IF NOT EXISTS evidence_chunk_registry_document_idx
                    ON evidence_chunk_registry (evidence_document_id);
                
    """,

    # ── Community subsystem (Phase 7, added 2026-08-12) ─────────
    # Structurally and physically separate from every table above:
    # no foreign key here ever points at patient_profiles or vice
    # versa. A community post is addressed by community_profile
    # (a pseudonymous handle), never by patient_profile_id or
    # user_id directly in content-facing reads — see
    # community_service.py for the join-only-at-write boundary
    # this is meant to enforce.
    """
                CREATE TABLE IF NOT EXISTS community_profiles (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    handle TEXT NOT NULL,
                    avatar_key TEXT,
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS community_profiles_user_uidx
                    ON community_profiles (user_id);
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS community_profiles_handle_uidx
                    ON community_profiles (handle);
                
    """,

    """
                CREATE TABLE IF NOT EXISTS communities (
                    id UUID PRIMARY KEY,
                    slug TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS communities_slug_uidx
                    ON communities (slug);
                
    """,

    """
                CREATE TABLE IF NOT EXISTS community_memberships (
                    id UUID PRIMARY KEY,
                    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
                    community_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    joined_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS community_memberships_uidx
                    ON community_memberships (community_id, community_profile_id);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS community_posts_community_idx
                    ON community_posts (community_id, created_at DESC);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS community_comments_post_idx
                    ON community_comments (post_id, created_at);
                
    """,

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
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS community_reactions_uidx
                    ON community_reactions (target_type, target_id, community_profile_id);
                
    """,

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
                
    """,
    """
                CREATE INDEX IF NOT EXISTS community_reports_status_idx
                    ON community_reports (status, created_at DESC);
                
    """,

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
                
    """,

    """
                CREATE TABLE IF NOT EXISTS community_blocked_users (
                    id UUID PRIMARY KEY,
                    blocker_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    blocked_profile_id UUID NOT NULL
                        REFERENCES community_profiles(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                
    """,
    """
                CREATE UNIQUE INDEX IF NOT EXISTS community_blocked_users_uidx
                    ON community_blocked_users (blocker_profile_id, blocked_profile_id);

    """,

    # ── Phase 1 finalization (added 2026-08-12) ─────────────────────
    # Everything below closes out the "Phase 1 — Finalize the canonical
    # longitudinal patient data model" checklist: multi-primary/
    # recurrence/progression support on diagnoses, tumor_profiles,
    # normalization columns, symptom_observations (patient_profile-keyed,
    # succeeding the legacy patient_symptom_entries table for new writes
    # — see symptom_observation_service.py), nutrition_assessments,
    # care_team_instructions, and treatment_cycles delay/hold/dose-
    # reduction tracking. Hand-authored going forward (not extracted from
    # ensure_schema()) — see migrations/patients_db/README.md: new
    # changes are a migration revision first, appended here to keep
    # ensure_schema() in sync, not the other way around.

    # patient_profiles: only the two demographics Phase 1 explicitly
    # calls for (needed for evidence applicability / localization).
    # Deliberately not adding race, address, insurance, etc. just because
    # they're commonly collected elsewhere — see the Phase 1 checklist's
    # "avoid unnecessary demographic attributes" item.
    """
    ALTER TABLE patient_profiles ADD COLUMN IF NOT EXISTS preferred_language TEXT;
    ALTER TABLE patient_profiles ADD COLUMN IF NOT EXISTS timezone TEXT;
    """,

    # patient_diagnoses: multiple primaries / recurrence / progression /
    # remission as first-class, plus normalization columns. A recurrence
    # or progression entry can point back at the primary it followed via
    # related_diagnosis_id, so a patient's diagnosis history reads as a
    # chain rather than a flat, undifferentiated list.
    """
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS diagnosis_type TEXT NOT NULL DEFAULT 'primary';
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS effective_date DATE;
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS canonical_cancer_type TEXT;
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS canonical_histology TEXT;
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS stage_system TEXT;
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS metastatic_sites JSONB NOT NULL DEFAULT '[]'::jsonb;
    ALTER TABLE patient_diagnoses ADD COLUMN IF NOT EXISTS related_diagnosis_id UUID
        REFERENCES patient_diagnoses(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS patient_diagnoses_status_idx
        ON patient_diagnoses (patient_profile_id, status);
    """,

    # tumor_profiles: grade/size/receptor-status/molecular-subtype as its
    # own entity rather than crammed into patient_diagnoses — a tumor
    # profile can be re-assessed (new specimen, new stain) without
    # implying the diagnosis itself changed.
    """
    CREATE TABLE IF NOT EXISTS tumor_profiles (
        id UUID PRIMARY KEY,
        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,
        diagnosis_id UUID REFERENCES patient_diagnoses(id) ON DELETE SET NULL,
        grade TEXT,
        tumor_size_mm DOUBLE PRECISION,
        molecular_subtype TEXT,
        receptor_status JSONB NOT NULL DEFAULT '{}'::jsonb,
        specimen_date DATE,
        specimen_site TEXT,
        raw_text TEXT,
        source_type TEXT NOT NULL DEFAULT 'patient_manual',
        source_document_id UUID,
        verification_status TEXT NOT NULL DEFAULT 'extracted',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS tumor_profiles_profile_idx
        ON tumor_profiles (patient_profile_id, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS tumor_profiles_diagnosis_idx
        ON tumor_profiles (diagnosis_id);
    """,

    # patient_biomarker_results: specimen provenance + a coarse category
    # (receptor_status/pdl1/msi_mmr/tmb/variant/other) so retrieval can
    # ask "does this patient have an actionable variant" without string-
    # matching biomarker_name, plus a normalized gene symbol alongside
    # the always-preserved raw_text.
    """
    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS specimen_date DATE;
    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS specimen_site TEXT;
    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS biomarker_category TEXT;
    ALTER TABLE patient_biomarker_results ADD COLUMN IF NOT EXISTS canonical_gene TEXT;
    """,

    # treatment_cycles: delay/hold/dose-reduction tracking — the
    # difference between "cycle 4 given on schedule" and "cycle 4 held
    # for neutropenia" matters for both the timeline and retrieval.
    """
    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS delayed BOOLEAN NOT NULL DEFAULT false;
    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS delay_reason TEXT;
    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS held BOOLEAN NOT NULL DEFAULT false;
    ALTER TABLE treatment_cycles ADD COLUMN IF NOT EXISTS dose_reduction_pct REAL;
    """,

    # treatment_agents / medication_exposures: canonical_name + aliases
    # populated by clinical_normalization.py at write time. rxnorm_code
    # stays NULL unless the curated static table happens to have one —
    # see that module's docstring for why this isn't a live RxNorm API
    # integration.
    """
    ALTER TABLE treatment_agents ADD COLUMN IF NOT EXISTS canonical_name TEXT;
    ALTER TABLE treatment_agents ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb;
    ALTER TABLE medication_exposures ADD COLUMN IF NOT EXISTS canonical_name TEXT;
    ALTER TABLE medication_exposures ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb;
    """,

    # symptom_observations: the patient_profile-keyed symptom table Phase
    # 1 calls for. The legacy patient_symptom_entries table (keyed by
    # patient_user_id, no patient_profile_id, no normalization/status/
    # trajectory columns) stays exactly as it is — symptom_service.py and
    # the existing /portal/symptoms endpoints keep working unchanged —
    # but new symptom-diary writes should go through
    # symptom_observation_service.py going forward; patient_state_service
    # now prefers this table and falls back to the legacy one. Attribution
    # to a treatment is possibly_related_treatment_episode_id — named
    # deliberately, not "caused_by", since Paxis observes a temporal
    # association a patient or their care team noted, never a confirmed
    # causal mechanism.
    """
    CREATE TABLE IF NOT EXISTS symptom_observations (
        id UUID PRIMARY KEY,
        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,
        raw_text TEXT NOT NULL,
        canonical_symptom TEXT,
        severity SMALLINT,
        onset_date DATE,
        resolved_date DATE,
        status TEXT NOT NULL DEFAULT 'active',
        frequency TEXT,
        possibly_related_treatment_episode_id UUID
            REFERENCES treatment_episodes(id) ON DELETE SET NULL,
        source_type TEXT NOT NULL DEFAULT 'patient_manual',
        source_document_id UUID,
        verification_status TEXT NOT NULL DEFAULT 'extracted',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS symptom_observations_profile_idx
        ON symptom_observations (patient_profile_id, status, onset_date DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS symptom_observations_canonical_idx
        ON symptom_observations (patient_profile_id, canonical_symptom);
    """,

    # nutrition_assessments: the table Phase 1 calls for that had no
    # equivalent at all before this — appetite/intake/swallowing/feeding
    # route/restrictions/nutrition-risk, distinguishing active-treatment
    # nutrition concerns from survivorship/prevention ones (architecture
    # review section 19: don't downrank an active-treatment nutrition
    # question with generic prevention-diet content).
    """
    CREATE TABLE IF NOT EXISTS nutrition_assessments (
        id UUID PRIMARY KEY,
        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,
        assessment_date DATE NOT NULL DEFAULT CURRENT_DATE,
        appetite TEXT,
        oral_intake_pct SMALLINT,
        swallowing_difficulty BOOLEAN,
        feeding_tube BOOLEAN NOT NULL DEFAULT false,
        feeding_tube_type TEXT,
        diet_restrictions JSONB NOT NULL DEFAULT '[]'::jsonb,
        food_allergies JSONB NOT NULL DEFAULT '[]'::jsonb,
        texture_requirements TEXT,
        hydration_constraints TEXT,
        nutrition_risk TEXT,
        care_phase TEXT NOT NULL DEFAULT 'active_treatment',
        source_type TEXT NOT NULL DEFAULT 'patient_manual',
        source_document_id UUID,
        verification_status TEXT NOT NULL DEFAULT 'extracted',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS nutrition_assessments_profile_idx
        ON nutrition_assessments (patient_profile_id, assessment_date DESC);
    """,

    # patient_allergies: true allergy vs. intolerance, since a nutrition
    # or medication answer treats the two very differently (severity and
    # avoidance strictness differ).
    """
    ALTER TABLE patient_allergies ADD COLUMN IF NOT EXISTS allergy_type TEXT NOT NULL DEFAULT 'allergy';
    """,

    # encounters: explicit next_steps / newly_ordered_tests / questions
    # columns rather than everything bundled into the structured_changes
    # catch-all, so a visit recap and a "questions for next appointment"
    # list (architecture review Phase 24) can be built directly from a
    # column instead of parsing free-form JSON.
    """
    ALTER TABLE encounters ADD COLUMN IF NOT EXISTS newly_ordered_tests JSONB NOT NULL DEFAULT '[]'::jsonb;
    ALTER TABLE encounters ADD COLUMN IF NOT EXISTS next_steps TEXT;
    ALTER TABLE encounters ADD COLUMN IF NOT EXISTS questions_for_next_visit JSONB NOT NULL DEFAULT '[]'::jsonb;
    """,

    # care_team_instructions: the table Phase 1 calls for that had no
    # equivalent before — a clinician's specific instruction to this
    # patient ("no NSAIDs while on this regimen"), which should outrank
    # generic education when the two would otherwise conflict. Ranking
    # that precedence is Phase 4/13 retrieval work; this is just the
    # table to store it in.
    """
    CREATE TABLE IF NOT EXISTS care_team_instructions (
        id UUID PRIMARY KEY,
        patient_profile_id UUID NOT NULL REFERENCES patient_profiles(id) ON DELETE CASCADE,
        instruction_text TEXT NOT NULL,
        instruction_type TEXT NOT NULL DEFAULT 'other',
        author_provider TEXT,
        physician_id UUID,
        source_type TEXT NOT NULL DEFAULT 'clinician_entered',
        source_document_id UUID,
        effective_from DATE,
        effective_to DATE,
        active BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS care_team_instructions_profile_idx
        ON care_team_instructions (patient_profile_id, active, created_at DESC);
    """,

    # ── Evidence ingestion versioning (added 2026-08-12) ─────────────
    # The original evidence_document_versions/evidence_chunk_registry
    # (created above) had no is_current tracking and no FK linking a
    # chunk to the specific version it came from — meaning a re-ingested
    # document with changed content had no clean way to identify which
    # Qdrant points were stale. See evidence_ingestion_service.py's
    # deterministic-ID + idempotent-reingestion rewrite, which this
    # supports: version_id is content-hash-derived, so the same content
    # fetched twice always produces the same version_id and is
    # recognized as "already ingested, skip" rather than re-embedded.
    """
    ALTER TABLE evidence_document_versions
        ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT true;
    ALTER TABLE evidence_document_versions
        ADD COLUMN IF NOT EXISTS superseded_by UUID
            REFERENCES evidence_document_versions(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_document_versions_current_idx
        ON evidence_document_versions (evidence_document_id, is_current);
    """,
    """
    ALTER TABLE evidence_documents
        ADD COLUMN IF NOT EXISTS current_version_id UUID
            REFERENCES evidence_document_versions(id) ON DELETE SET NULL;
    ALTER TABLE evidence_documents
        ADD COLUMN IF NOT EXISTS latest_content_hash TEXT;
    """,
    """
    ALTER TABLE evidence_chunk_registry
        ADD COLUMN IF NOT EXISTS evidence_document_version_id UUID
            REFERENCES evidence_document_versions(id) ON DELETE CASCADE;
    ALTER TABLE evidence_chunk_registry ADD COLUMN IF NOT EXISTS section_title TEXT;
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_chunk_registry_version_idx
        ON evidence_chunk_registry (evidence_document_version_id);
    """,

    # ── Query debug traces (added 2026-08-12) ─────────────────────────
    # One row per patient chat query, recording every stage of the
    # retrieval/generation pipeline for debugging — see
    # retrieval_debug_trace.py. Deliberately its own table rather than a
    # generic application log line: the trace contains patient-derived
    # content (state, retrieved passages, the generated answer), so it
    # gets the same CASCADE-on-profile-delete privacy behavior as every
    # other patient-owned table, and structured JSONB querying, instead
    # of living in a log aggregator with a different (or no) retention/
    # access policy. Generic application logs get only the trace id — see
    # retrieval_debug_trace.py's module docstring.
    """
    CREATE TABLE IF NOT EXISTS query_debug_traces (
        id UUID PRIMARY KEY,
        patient_profile_id UUID NOT NULL
            REFERENCES patient_profiles(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        trace JSONB NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS query_debug_traces_profile_idx
        ON query_debug_traces (patient_profile_id, created_at DESC);
    """,

    # ── Deterministic state freshness (added 2026-08-12, convergence
    # Sprint B item 7) ─────────────────────────────────────────────────
    # Before this: get_context() (patient_context_service.py) rebuilt a
    # patient's state snapshot only when NONE existed yet — a rebuild
    # attempt that failed (network blip, transient DB error) after a
    # later canonical write left every subsequent read trusting a stale
    # snapshot forever, since "a snapshot exists" never became false
    # just because it was outdated. state_revision (bumped by
    # invalidate_patient_state() on every canonical write, BEFORE it
    # attempts a rebuild) and source_revision (stamped onto whichever
    # snapshot build_state() actually persists) let get_context() compare
    # the two and rebuild whenever they don't match, not just when no
    # snapshot exists — see patient_state_service.py's
    # invalidate_patient_state() and patient_context_service.py's
    # get_context() docstrings for the full mechanism.
    """
    ALTER TABLE patient_profiles
        ADD COLUMN IF NOT EXISTS state_revision BIGINT NOT NULL DEFAULT 0;
    """,
    """
    ALTER TABLE patient_state_snapshots
        ADD COLUMN IF NOT EXISTS source_revision BIGINT;
    """,
]
