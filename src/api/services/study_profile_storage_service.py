"""
Study Profile Storage Service

Stores extracted study profiles to PostgreSQL database.
Matches the comprehensive schema from the Colab migration pipeline.
"""

import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from .account_db import get_account_db


def safe_int(value: Any) -> Optional[int]:
    """Safely convert to integer."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r'\d+', value.replace(',', ''))
        if match:
            return int(match.group())
    return None


def safe_float(value: Any) -> Optional[float]:
    """Safely convert to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r'[\d.]+', value.replace(',', ''))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def extract_value_evidence(field_data: Any) -> tuple:
    """Extract value and evidence_quote from field."""
    if field_data is None:
        return None, None
    if isinstance(field_data, dict):
        return field_data.get('value'), field_data.get('evidence_quote')
    return field_data, None


def clean_json_string(s: str) -> str:
    """Remove control characters from string."""
    if not s:
        return s
    return ''.join(char for char in s if ord(char) >= 32 or char in '\t\n\r')


def normalize_biomarker_status(extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse molecular_subtype, biomarker_inclusion_criteria, and biomarkers into a
    normalized JSONB structure like:
        {"ER": "positive", "PR": "positive", "HER2": "negative", "genomic_assay": "Oncotype DX", "score_range": "11-25"}

    This enables exact structured matching instead of regex over free text.
    """
    result = {}

    # 1. Parse from biomarker_inclusion_criteria (new structured field)
    bic = extracted_data.get("biomarker_inclusion_criteria", {})
    if isinstance(bic, dict):
        for req in bic.get("required_biomarkers", []):
            if isinstance(req, dict) and req.get("name") and req.get("status"):
                name = req["name"].upper().strip()
                status = req["status"].lower().strip()
                result[name] = status

        genomic_assay_data = bic.get("genomic_assay", {})
        if isinstance(genomic_assay_data, dict) and genomic_assay_data.get("value"):
            result["genomic_assay"] = genomic_assay_data["value"]

        score_range_data = bic.get("score_range", {})
        if isinstance(score_range_data, dict) and score_range_data.get("value"):
            result["score_range"] = score_range_data["value"]

    # 2. Parse from diagnosis.molecular_subtype free text
    diagnosis = extracted_data.get("diagnosis", {})
    mol_subtype_data = diagnosis.get("molecular_subtype", {})
    mol_text = None
    if isinstance(mol_subtype_data, dict):
        mol_text = mol_subtype_data.get("value")
    elif isinstance(mol_subtype_data, str):
        mol_text = mol_subtype_data

    if mol_text:
        mol_lower = mol_text.lower()

        # Common receptor patterns
        _RECEPTOR_PATTERNS = [
            (r'\ber\s*[\+]\s*|estrogen\s*receptor\s*positive', "ER", "positive"),
            (r'\ber\s*[\-]\s*|estrogen\s*receptor\s*negative', "ER", "negative"),
            (r'\bpr\s*[\+]\s*|progesterone\s*receptor\s*positive', "PR", "positive"),
            (r'\bpr\s*[\-]\s*|progesterone\s*receptor\s*negative', "PR", "negative"),
            (r'her.?2\s*[\+]|her.?2\s*positive|her.?2\s*amplified', "HER2", "positive"),
            (r'her.?2\s*[\-]|her.?2\s*negative|her.?2\s*non.amplified', "HER2", "negative"),
            (r'triple\s*negative|tnbc', "TNBC", "positive"),
            (r'egfr\s*mutant|egfr\s*mutation|egfr\s*positive|egfr\s*\+', "EGFR", "mutant"),
            (r'egfr\s*wild.?type|egfr\s*negative|egfr\s*wt', "EGFR", "wild-type"),
            (r'alk\s*positive|alk\s*\+|alk\s*rearrang|alk\s*fusion', "ALK", "positive"),
            (r'alk\s*negative|alk\s*\-', "ALK", "negative"),
            (r'pd.?l1\s*positive|pd.?l1\s*\+|pd.?l1\s*high', "PD-L1", "positive"),
            (r'pd.?l1\s*negative|pd.?l1\s*\-|pd.?l1\s*low', "PD-L1", "negative"),
            (r'kras\s*mutant|kras\s*mutation|kras\s*\+', "KRAS", "mutant"),
            (r'kras\s*wild.?type|kras\s*wt', "KRAS", "wild-type"),
            (r'braf\s*v600|braf\s*mutant|braf\s*mutation|braf\s*\+', "BRAF", "mutant"),
            (r'braf\s*wild.?type|braf\s*wt', "BRAF", "wild-type"),
            (r'msi.?h|microsatellite\s*instability.?high', "MSI", "high"),
            (r'mss|microsatellite\s*stable|msi.?l', "MSI", "stable"),
            (r'brca\s*mutant|brca\s*mutation|brca1|brca2', "BRCA", "mutant"),
            (r'hpv\s*positive|hpv\s*\+|p16\s*positive|p16\s*\+', "HPV", "positive"),
            (r'hpv\s*negative|hpv\s*\-|p16\s*negative|p16\s*\-', "HPV", "negative"),
        ]

        for pattern, name, status in _RECEPTOR_PATTERNS:
            if re.search(pattern, mol_lower):
                if name not in result:  # Don't override structured data
                    result[name] = status

    # 3. Parse from inclusion_criteria text for additional biomarker requirements
    patient_chars = extracted_data.get("patient_characteristics", {})
    inclusion_criteria = patient_chars.get("inclusion_criteria", [])
    for criterion in inclusion_criteria:
        crit_text = None
        if isinstance(criterion, dict):
            crit_text = criterion.get("criterion") or criterion.get("value")
        elif isinstance(criterion, str):
            crit_text = criterion

        if crit_text:
            crit_lower = crit_text.lower()
            for pattern, name, status in _RECEPTOR_PATTERNS:
                if re.search(pattern, crit_lower):
                    if name not in result:
                        result[name] = status

    # 4. Check for genomic assay mentions in extraction data
    if "genomic_assay" not in result:
        full_text = json.dumps(extracted_data).lower()
        _ASSAY_PATTERNS = [
            (r'oncotype\s*dx', "Oncotype DX"),
            (r'mammaprint', "MammaPrint"),
            (r'decipher', "Decipher"),
            (r'prolaris', "Prolaris"),
            (r'endopredict', "EndoPredict"),
            (r'prosigna|pam\s*50', "Prosigna"),
        ]
        for pattern, assay_name in _ASSAY_PATTERNS:
            if re.search(pattern, full_text):
                result["genomic_assay"] = assay_name
                # Try to extract score range near the assay mention
                score_match = re.search(
                    pattern + r'[^.]{0,80}(?:score|rs)\s*(?:of\s*)?(\d+\s*[-–]\s*\d+|\d+|[<>≤≥]\s*\d+)',
                    full_text
                )
                if score_match and "score_range" not in result:
                    result["score_range"] = score_match.group(1).strip()
                break

    return result if result else None


class StudyProfileStorageService:
    """Service for storing extracted study profiles in PostgreSQL."""

    def __init__(self):
        self._schema_ensured = False
    
    async def _ensure_schema(self):
        """Ensure all study profile tables exist (matching Colab schema)."""
        if self._schema_ensured:
            return
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Main studies table
            # NOTE: Using TEXT instead of VARCHAR to avoid truncation/insertion failures
            # Matches Colab migration schema with extraction_data JSONB
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id SERIAL PRIMARY KEY,
                    document_name TEXT UNIQUE NOT NULL,
                    document_path TEXT,
                    extraction_timestamp TIMESTAMP,
                    processing_duration_seconds FLOAT,
                    num_sections_processed INTEGER,
                    
                    -- API usage metadata
                    api_total_tokens INTEGER,
                    api_total_cost_usd FLOAT,
                    api_model TEXT,
                    
                    -- Identifiers
                    doc_id TEXT UNIQUE,
                    doi TEXT,
                    pmid TEXT,
                    
                    -- Study details
                    study_name TEXT,
                    study_name_evidence TEXT,
                    protocol_name TEXT,
                    protocol_name_evidence TEXT,
                    trial_registration_number TEXT,
                    trial_registration_number_evidence TEXT,
                    publish_date TEXT,
                    publish_date_evidence TEXT,
                    study_type TEXT,
                    study_type_evidence TEXT,
                    study_phase TEXT,
                    study_phase_evidence TEXT,
                    analysis_type TEXT,
                    analysis_type_evidence TEXT,
                    number_of_patients INTEGER,
                    number_of_patients_evidence TEXT,
                    study_institution TEXT,
                    study_institution_evidence TEXT,
                    country TEXT,
                    country_evidence TEXT,
                    doi_evidence TEXT,
                    pmid_evidence TEXT,
                    citation_count INTEGER,
                    
                    -- Patient characteristics
                    age_range TEXT,
                    age_range_evidence TEXT,
                    median_age TEXT,
                    median_age_evidence TEXT,
                    gender_distribution TEXT,
                    gender_distribution_evidence TEXT,
                    race_ethnicity TEXT,
                    race_ethnicity_evidence TEXT,
                    performance_status TEXT,
                    performance_status_evidence TEXT,
                    
                    -- Diagnosis
                    cancer_location TEXT,
                    cancer_location_evidence TEXT,
                    cancer_type TEXT,
                    cancer_type_evidence TEXT,
                    histopathologic_type TEXT,
                    histopathologic_type_evidence TEXT,
                    tumor_grade TEXT,
                    tumor_grade_evidence TEXT,
                    molecular_subtype TEXT,
                    molecular_subtype_evidence TEXT,
                    biomarker_status JSONB,
                    genomic_assay TEXT,
                    genomic_score_range TEXT,
                    
                    -- Staging
                    staging_system_used TEXT,
                    staging_system_used_evidence TEXT,
                    risk_stratification TEXT,
                    risk_stratification_evidence TEXT,
                    metastatic_status TEXT,
                    metastatic_status_evidence TEXT,
                    extent_of_resection TEXT,
                    extent_of_resection_evidence TEXT,
                    
                    -- Outcomes
                    primary_endpoint TEXT,
                    primary_endpoint_evidence TEXT,
                    event_free_survival TEXT,
                    event_free_survival_evidence TEXT,
                    overall_survival TEXT,
                    overall_survival_evidence TEXT,
                    progression_free_survival TEXT,
                    progression_free_survival_evidence TEXT,
                    disease_free_survival TEXT,
                    disease_free_survival_evidence TEXT,
                    local_control TEXT,
                    local_control_evidence TEXT,
                    median_followup TEXT,
                    median_followup_evidence TEXT,
                    
                    -- Full extraction JSON (JSONB for flexibility and querying)
                    extraction_data JSONB,
                    
                    -- Abstract/Summary
                    abstract TEXT,
                    abstract_source TEXT,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Inclusion criteria table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS inclusion_criteria (
                    criterion_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    criterion TEXT NOT NULL,
                    evidence_quote TEXT,
                    criterion_order INTEGER
                )
            """)
            
            # Exclusion criteria table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS exclusion_criteria (
                    criterion_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    criterion TEXT NOT NULL,
                    evidence_quote TEXT,
                    criterion_order INTEGER
                )
            """)
            
            # Stage distribution table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stage_distribution (
                    distribution_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    stage_category TEXT,
                    number_of_patients TEXT,
                    percentage TEXT,
                    evidence_quote TEXT
                )
            """)
            
            # Staging components table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS staging_components (
                    component_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    component_name TEXT,
                    component_value TEXT,
                    evidence_quote TEXT
                )
            """)
            
            # Study arms table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS study_arms (
                    arm_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    arm_name TEXT,
                    description TEXT,
                    number_of_patients INTEGER,
                    evidence_quote TEXT,
                    arm_order INTEGER
                )
            """)
            
            # Chemotherapy regimens table (drugs as TEXT[] array)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chemotherapy_regimens (
                    regimen_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    regimen_name TEXT,
                    drugs TEXT[],
                    dosage_info TEXT,
                    schedule_info TEXT,
                    number_of_cycles INTEGER,
                    doxorubicin_dose TEXT,
                    doxorubicin_schedule TEXT,
                    cyclophosphamide_dose TEXT,
                    cyclophosphamide_schedule TEXT,
                    evidence_quote TEXT,
                    regimen_order INTEGER,
                    additional_details JSONB
                )
            """)
            
            # Radiation details table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS radiation_details (
                    detail_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    radiation_type TEXT,
                    total_dose TEXT,
                    fractionation TEXT,
                    technique TEXT,
                    target_volume TEXT,
                    evidence_quote TEXT,
                    detail_order INTEGER,
                    additional_details JSONB
                )
            """)
            
            # Surgery details table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS surgery_details (
                    detail_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    surgery_type TEXT,
                    description TEXT,
                    evidence_quote TEXT,
                    detail_order INTEGER,
                    additional_details JSONB
                )
            """)
            
            # Biomarkers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS biomarkers (
                    biomarker_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    biomarker_name TEXT,
                    biomarker_type TEXT,
                    measurement_method TEXT,
                    baseline_value TEXT,
                    change_from_baseline TEXT,
                    significance TEXT,
                    evidence_quote TEXT,
                    biomarker_order INTEGER,
                    additional_details JSONB
                )
            """)
            
            # Toxicity table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS toxicity (
                    toxicity_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    toxicity_type TEXT,
                    grade TEXT,
                    frequency TEXT,
                    number_of_patients TEXT,
                    timing TEXT,
                    evidence_quote TEXT,
                    toxicity_order INTEGER,
                    additional_details JSONB
                )
            """)
            
            # Dose constraints table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dose_constraints (
                    constraint_id SERIAL PRIMARY KEY,
                    study_id INTEGER REFERENCES studies(study_id) ON DELETE CASCADE,
                    organ_at_risk TEXT,
                    constraint_type TEXT,
                    dose_limit TEXT,
                    volume_limit TEXT,
                    evidence_quote TEXT,
                    constraint_order INTEGER,
                    additional_details JSONB
                )
            """)
            
            # Create indexes for faster lookups
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_studies_doc_id ON studies(doc_id);
                CREATE INDEX IF NOT EXISTS idx_studies_doi ON studies(doi);
                CREATE INDEX IF NOT EXISTS idx_studies_pmid ON studies(pmid);
                CREATE INDEX IF NOT EXISTS idx_studies_document_name ON studies(document_name);
                CREATE INDEX IF NOT EXISTS idx_studies_cancer_type ON studies(cancer_type);
                CREATE INDEX IF NOT EXISTS idx_studies_cancer_location ON studies(cancer_location);
                CREATE INDEX IF NOT EXISTS idx_studies_study_type ON studies(study_type);
                CREATE INDEX IF NOT EXISTS idx_studies_study_phase ON studies(study_phase);
            """)
            
            # JSONB index for querying within extraction_data
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_extraction_data_gin ON studies USING GIN (extraction_data);
                CREATE INDEX IF NOT EXISTS idx_biomarker_status_gin ON studies USING GIN (biomarker_status);
                CREATE INDEX IF NOT EXISTS idx_studies_genomic_assay ON studies(genomic_assay);
            """)

            # Migrate existing tables: add new columns if missing
            await conn.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='studies' AND column_name='biomarker_status') THEN
                        ALTER TABLE studies ADD COLUMN biomarker_status JSONB;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='studies' AND column_name='genomic_assay') THEN
                        ALTER TABLE studies ADD COLUMN genomic_assay TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='studies' AND column_name='genomic_score_range') THEN
                        ALTER TABLE studies ADD COLUMN genomic_score_range TEXT;
                    END IF;
                END $$;
            """)

        self._schema_ensured = True
        print("✓ Study profile schema ensured")

    async def store_study_profile(
        self,
        doc_id: str,
        document_name: str,
        extracted_data: Dict[str, Any],
        processing_duration: float = None,
        api_usage: Dict = None,
        abstract: str = None,
        abstract_source: str = None,
        force: bool = False,
    ) -> int:
        """
        Store extracted study profile in PostgreSQL.

        Default behavior: skip if a row with this doc_id or document_name
        already exists and return the existing study_id.

        With ``force=True``: delete the existing row (child rows cascade via
        ``ON DELETE CASCADE``) and re-insert from ``extracted_data``. This
        is what backfill / re-upsert flows use to correct stale data.
        """
        await self._ensure_schema()

        db = get_account_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchval(
                    "SELECT study_id FROM studies WHERE doc_id = $1 OR document_name = $2",
                    doc_id, document_name
                )

                if existing and not force:
                    print(f"  ℹ️  Study already exists (study_id={existing}), skipping")
                    return existing

                if existing and force:
                    await conn.execute(
                        "DELETE FROM studies WHERE study_id = $1", existing
                    )
                    print(f"  ↻  Replacing existing study_id={existing} (force=True)")
                
                # Extract study details
                study_details = extracted_data.get('study_details', {})
                patient_chars = extracted_data.get('patient_characteristics', {})
                diagnosis = extracted_data.get('diagnosis', {})
                staging = extracted_data.get('staging', {})
                treatment = extracted_data.get('treatment', {})
                outcomes = extracted_data.get('outcomes', {})
                
                # Helper to get value and evidence
                def get_ve(section, field):
                    data = section.get(field, {})
                    if isinstance(data, dict):
                        return data.get('value'), data.get('evidence_quote')
                    return data, None
                
                # Insert main study record
                study_name, study_name_ev = get_ve(study_details, 'study_name')
                protocol_name, protocol_name_ev = get_ve(study_details, 'protocol_name')
                trial_reg, trial_reg_ev = get_ve(study_details, 'trial_registration_number')
                pub_date, pub_date_ev = get_ve(study_details, 'publish_date')
                study_type, study_type_ev = get_ve(study_details, 'study_type')
                study_phase, study_phase_ev = get_ve(study_details, 'study_phase')
                analysis_type, analysis_type_ev = get_ve(study_details, 'analysis_type')
                num_patients, num_patients_ev = get_ve(study_details, 'number_of_patients')
                institution, institution_ev = get_ve(study_details, 'study_institution')
                country, country_ev = get_ve(study_details, 'country')
                doi, doi_ev = get_ve(study_details, 'doi')
                pmid, pmid_ev = get_ve(study_details, 'pmid')
                
                # Patient characteristics
                age_range, age_range_ev = get_ve(patient_chars, 'age_range')
                median_age, median_age_ev = get_ve(patient_chars, 'median_age')
                gender_dist, gender_dist_ev = get_ve(patient_chars, 'gender_distribution')
                race_eth, race_eth_ev = get_ve(patient_chars, 'race_ethnicity')
                perf_status, perf_status_ev = get_ve(patient_chars, 'performance_status')
                
                # Diagnosis
                cancer_loc, cancer_loc_ev = get_ve(diagnosis, 'cancer_location')
                cancer_type, cancer_type_ev = get_ve(diagnosis, 'cancer_type')
                histo_type, histo_type_ev = get_ve(diagnosis, 'histopathologic_type')
                tumor_grade, tumor_grade_ev = get_ve(diagnosis, 'tumor_grade')
                mol_subtype, mol_subtype_ev = get_ve(diagnosis, 'molecular_subtype')

                # Normalize biomarker status from all available sources
                biomarker_status = normalize_biomarker_status(extracted_data)
                genomic_assay = (biomarker_status or {}).get("genomic_assay")
                genomic_score_range = (biomarker_status or {}).get("score_range")

                # Staging
                staging_sys, staging_sys_ev = get_ve(staging, 'staging_system_used')
                risk_strat, risk_strat_ev = get_ve(staging, 'risk_stratification')
                meta_status, meta_status_ev = get_ve(staging, 'metastatic_status')
                extent_resec, extent_resec_ev = get_ve(staging, 'extent_of_resection')
                
                # Outcomes
                primary_ep, primary_ep_ev = get_ve(outcomes, 'primary_endpoint')
                efs, efs_ev = get_ve(outcomes, 'event_free_survival')
                os_val, os_ev = get_ve(outcomes, 'overall_survival')
                pfs, pfs_ev = get_ve(outcomes, 'progression_free_survival')
                dfs, dfs_ev = get_ve(outcomes, 'disease_free_survival')
                local_ctrl, local_ctrl_ev = get_ve(outcomes, 'local_control')
                median_fu, median_fu_ev = get_ve(outcomes, 'median_followup')
                
                study_id = await conn.fetchval("""
                    INSERT INTO studies (
                        doc_id, document_name, extraction_timestamp, processing_duration_seconds,
                        api_total_tokens, api_total_cost_usd, api_model,
                        doi, pmid, doi_evidence, pmid_evidence,
                        study_name, study_name_evidence,
                        protocol_name, protocol_name_evidence,
                        trial_registration_number, trial_registration_number_evidence,
                        publish_date, publish_date_evidence,
                        study_type, study_type_evidence,
                        study_phase, study_phase_evidence,
                        analysis_type, analysis_type_evidence,
                        number_of_patients, number_of_patients_evidence,
                        study_institution, study_institution_evidence,
                        country, country_evidence,
                        age_range, age_range_evidence,
                        median_age, median_age_evidence,
                        gender_distribution, gender_distribution_evidence,
                        race_ethnicity, race_ethnicity_evidence,
                        performance_status, performance_status_evidence,
                        cancer_location, cancer_location_evidence,
                        cancer_type, cancer_type_evidence,
                        histopathologic_type, histopathologic_type_evidence,
                        tumor_grade, tumor_grade_evidence,
                        molecular_subtype, molecular_subtype_evidence,
                        biomarker_status, genomic_assay, genomic_score_range,
                        staging_system_used, staging_system_used_evidence,
                        risk_stratification, risk_stratification_evidence,
                        metastatic_status, metastatic_status_evidence,
                        extent_of_resection, extent_of_resection_evidence,
                        primary_endpoint, primary_endpoint_evidence,
                        event_free_survival, event_free_survival_evidence,
                        overall_survival, overall_survival_evidence,
                        progression_free_survival, progression_free_survival_evidence,
                        disease_free_survival, disease_free_survival_evidence,
                        local_control, local_control_evidence,
                        median_followup, median_followup_evidence,
                        extraction_data,
                        abstract, abstract_source
                    ) VALUES (
                        $1, $2, NOW(), $3,
                        $4, $5, $6,
                        $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                        $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                        $31, $32, $33, $34, $35, $36, $37, $38, $39, $40,
                        $41, $42, $43, $44, $45, $46, $47, $48, $49, $50,
                        $51, $52, $53,
                        $54, $55, $56, $57, $58, $59, $60, $61,
                        $62, $63, $64, $65, $66, $67, $68, $69, $70, $71,
                        $72, $73, $74, $75, $76, $77,
                        $78
                    ) RETURNING study_id
                """,
                    doc_id, document_name, processing_duration,
                    api_usage.get('total_tokens') if api_usage else None,
                    api_usage.get('total_cost_usd') if api_usage else None,
                    api_usage.get('model') if api_usage else None,
                    doi, pmid, doi_ev, pmid_ev,
                    study_name, study_name_ev,
                    protocol_name, protocol_name_ev,
                    trial_reg, trial_reg_ev,
                    pub_date, pub_date_ev,
                    study_type, study_type_ev,
                    study_phase, study_phase_ev,
                    analysis_type, analysis_type_ev,
                    safe_int(num_patients), num_patients_ev,
                    institution, institution_ev,
                    country, country_ev,
                    age_range, age_range_ev,
                    median_age, median_age_ev,
                    gender_dist, gender_dist_ev,
                    race_eth, race_eth_ev,
                    perf_status, perf_status_ev,
                    cancer_loc, cancer_loc_ev,
                    cancer_type, cancer_type_ev,
                    histo_type, histo_type_ev,
                    tumor_grade, tumor_grade_ev,
                    mol_subtype, mol_subtype_ev,
                    json.dumps(biomarker_status) if biomarker_status else None,
                    genomic_assay, genomic_score_range,
                    staging_sys, staging_sys_ev,
                    risk_strat, risk_strat_ev,
                    meta_status, meta_status_ev,
                    extent_resec, extent_resec_ev,
                    primary_ep, primary_ep_ev,
                    efs, efs_ev,
                    os_val, os_ev,
                    pfs, pfs_ev,
                    dfs, dfs_ev,
                    local_ctrl, local_ctrl_ev,
                    median_fu, median_fu_ev,
                    json.dumps(extracted_data),  # Full extraction JSON
                    abstract, abstract_source
                )

                # Insert inclusion criteria
                inclusion_criteria = patient_chars.get('inclusion_criteria', [])
                for i, criterion in enumerate(inclusion_criteria):
                    if isinstance(criterion, dict):
                        crit_text = criterion.get('criterion') or criterion.get('value')
                        crit_ev = criterion.get('evidence_quote')
                    else:
                        crit_text = criterion
                        crit_ev = None
                    
                    if crit_text:
                        await conn.execute("""
                            INSERT INTO inclusion_criteria (study_id, criterion, evidence_quote, criterion_order)
                            VALUES ($1, $2, $3, $4)
                        """, study_id, crit_text, crit_ev, i)
                
                # Insert exclusion criteria
                exclusion_criteria = patient_chars.get('exclusion_criteria', [])
                for i, criterion in enumerate(exclusion_criteria):
                    if isinstance(criterion, dict):
                        crit_text = criterion.get('criterion') or criterion.get('value')
                        crit_ev = criterion.get('evidence_quote')
                    else:
                        crit_text = criterion
                        crit_ev = None
                    
                    if crit_text:
                        await conn.execute("""
                            INSERT INTO exclusion_criteria (study_id, criterion, evidence_quote, criterion_order)
                            VALUES ($1, $2, $3, $4)
                        """, study_id, crit_text, crit_ev, i)
                
                # Insert stage distribution
                stage_dist = staging.get('stage_distribution', [])
                for item in stage_dist:
                    if isinstance(item, dict):
                        await conn.execute("""
                            INSERT INTO stage_distribution (study_id, stage_category, number_of_patients, percentage, evidence_quote)
                            VALUES ($1, $2, $3, $4, $5)
                        """, study_id,
                            item.get('stage_category') or item.get('stage'),
                            str(item.get('number_of_patients') or item.get('n') or ''),
                            str(item.get('percentage') or item.get('percent') or ''),
                            item.get('evidence_quote'))
                
                # Insert staging components
                staging_comps = staging.get('staging_components', [])
                for item in staging_comps:
                    if isinstance(item, dict):
                        await conn.execute("""
                            INSERT INTO staging_components (study_id, component_name, component_value, evidence_quote)
                            VALUES ($1, $2, $3, $4)
                        """, study_id,
                            item.get('component_name') or item.get('name'),
                            item.get('component_value') or item.get('value'),
                            item.get('evidence_quote'))
                
                # Insert study arms
                study_arms = treatment.get('study_arms', [])
                for i, arm in enumerate(study_arms):
                    if isinstance(arm, dict):
                        await conn.execute("""
                            INSERT INTO study_arms (study_id, arm_name, description, number_of_patients, evidence_quote, arm_order)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, study_id,
                            arm.get('arm_name') or arm.get('name'),
                            arm.get('description'),
                            safe_int(arm.get('number_of_patients') or arm.get('n')),
                            arm.get('evidence_quote'),
                            i)
                
                # Insert chemotherapy regimens (drugs as TEXT[] array)
                chemo_regimens = treatment.get('chemotherapy_regimens', [])
                for i, regimen in enumerate(chemo_regimens):
                    if isinstance(regimen, dict):
                        # Handle drugs as array
                        drugs = regimen.get('drugs', [])
                        if isinstance(drugs, str):
                            drugs = [drugs]  # Convert single string to array
                        elif not isinstance(drugs, list):
                            drugs = []
                        
                        await conn.execute("""
                            INSERT INTO chemotherapy_regimens (
                                study_id, regimen_name, drugs, dosage_info, schedule_info,
                                number_of_cycles, doxorubicin_dose, doxorubicin_schedule,
                                cyclophosphamide_dose, cyclophosphamide_schedule, evidence_quote, regimen_order,
                                additional_details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        """, study_id,
                            regimen.get('regimen_name') or regimen.get('name'),
                            drugs,  # TEXT[] array
                            regimen.get('dosage_info') or regimen.get('dosage'),
                            regimen.get('schedule_info') or regimen.get('schedule'),
                            safe_int(regimen.get('number_of_cycles') or regimen.get('cycles')),
                            regimen.get('doxorubicin_dose'),
                            regimen.get('doxorubicin_schedule'),
                            regimen.get('cyclophosphamide_dose'),
                            regimen.get('cyclophosphamide_schedule'),
                            regimen.get('evidence_quote'),
                            i,
                            json.dumps(regimen)  # additional_details JSONB
                        )
                
                # Insert radiation details
                radiation_details = treatment.get('radiation_details', [])
                for i, detail in enumerate(radiation_details):
                    if isinstance(detail, dict):
                        await conn.execute("""
                            INSERT INTO radiation_details (
                                study_id, radiation_type, total_dose, fractionation,
                                technique, target_volume, evidence_quote, detail_order,
                                additional_details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """, study_id,
                            detail.get('radiation_type') or detail.get('type'),
                            detail.get('total_dose') or detail.get('dose'),
                            detail.get('fractionation'),
                            detail.get('technique'),
                            detail.get('target_volume') or detail.get('target'),
                            detail.get('evidence_quote'),
                            i,
                            json.dumps(detail)  # additional_details JSONB
                        )
                
                # Insert surgery details
                surgery_details = treatment.get('surgery_details', [])
                for i, detail in enumerate(surgery_details):
                    if isinstance(detail, dict):
                        await conn.execute("""
                            INSERT INTO surgery_details (study_id, surgery_type, description, evidence_quote, detail_order, additional_details)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, study_id,
                            detail.get('surgery_type') or detail.get('type'),
                            detail.get('description'),
                            detail.get('evidence_quote'),
                            i,
                            json.dumps(detail)  # additional_details JSONB
                        )
                
                # Insert biomarkers
                biomarkers = extracted_data.get('biomarkers', [])
                for i, biomarker in enumerate(biomarkers):
                    if isinstance(biomarker, dict):
                        await conn.execute("""
                            INSERT INTO biomarkers (
                                study_id, biomarker_name, biomarker_type, measurement_method,
                                baseline_value, change_from_baseline, significance, evidence_quote, biomarker_order,
                                additional_details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """, study_id,
                            biomarker.get('biomarker_name') or biomarker.get('name'),
                            biomarker.get('biomarker_type') or biomarker.get('type'),
                            biomarker.get('measurement_method') or biomarker.get('method'),
                            biomarker.get('baseline_value') or biomarker.get('baseline'),
                            biomarker.get('change_from_baseline') or biomarker.get('change'),
                            biomarker.get('significance'),
                            biomarker.get('evidence_quote'),
                            i,
                            json.dumps(biomarker)  # additional_details JSONB
                        )
                
                # Insert toxicity
                toxicity_list = extracted_data.get('toxicity', [])
                for i, tox in enumerate(toxicity_list):
                    if isinstance(tox, dict):
                        await conn.execute("""
                            INSERT INTO toxicity (
                                study_id, toxicity_type, grade, frequency,
                                number_of_patients, timing, evidence_quote, toxicity_order,
                                additional_details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """, study_id,
                            tox.get('toxicity_type') or tox.get('type'),
                            tox.get('grade'),
                            tox.get('frequency'),
                            str(tox.get('number_of_patients') or tox.get('n') or ''),  # TEXT type
                            tox.get('timing'),
                            tox.get('evidence_quote'),
                            i,
                            json.dumps(tox)  # additional_details JSONB
                        )
                
                # Insert dose constraints
                dose_constraints = extracted_data.get('dose_constraints', [])
                for i, constraint in enumerate(dose_constraints):
                    if isinstance(constraint, dict):
                        await conn.execute("""
                            INSERT INTO dose_constraints (
                                study_id, organ_at_risk, constraint_type, dose_limit,
                                volume_limit, evidence_quote, constraint_order,
                                additional_details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, study_id,
                            constraint.get('organ_at_risk') or constraint.get('organ'),
                            constraint.get('constraint_type') or constraint.get('type'),
                            constraint.get('dose_limit') or constraint.get('dose'),
                            constraint.get('volume_limit') or constraint.get('volume'),
                            constraint.get('evidence_quote'),
                            i,
                            json.dumps(constraint)  # additional_details JSONB
                        )
                
                return study_id
    
    async def get_study_by_doc_id(self, doc_id: str) -> Optional[Dict]:
        """Get study profile by document ID."""
        await self._ensure_schema()

        db = get_account_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM studies WHERE doc_id = $1",
                doc_id
            )
            return dict(row) if row else None

    async def list_studies_by_doc_ids(
        self, doc_ids: List[str]
    ) -> Dict[str, Dict]:
        """Batch lookup: returns {doc_id: row_dict} for doc_ids that exist.

        Hits the study-profiles schema (which lives in the exueed_cache
        DB via get_account_db()) — NOT display-study-details. Used by
        scripts/audit_doc_id_coverage.py to find Qdrant doc_ids that
        weren't ingested into the profiles schema.

        For batch-fetching inclusion/exclusion criteria from
        display-study-details, see
        PostgresStudyDetailsService.get_eligibility_criteria_by_doc_ids.
        """
        if not doc_ids:
            return {}
        await self._ensure_schema()

        db = get_account_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM studies WHERE doc_id = ANY($1::text[])",
                list(doc_ids),
            )
            return {row["doc_id"]: dict(row) for row in rows}


    async def delete_study(self, study_id: int) -> bool:
        """Delete a study and all related records."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM studies WHERE study_id = $1",
                study_id
            )
            return "DELETE 1" in result


# Singleton instance
_storage_service = None


def get_study_profile_storage_service() -> StudyProfileStorageService:
    """Get singleton instance of StudyProfileStorageService."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StudyProfileStorageService()
    return _storage_service
