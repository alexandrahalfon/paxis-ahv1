"""
User Study Profile Service

Stores extracted study profiles for user-uploaded documents.
Separate from admin-uploaded study profiles in the main 'studies' table.
Tied to user accounts via user_id.
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


class UserStudyProfileService:
    """Service for storing user-uploaded study profiles in PostgreSQL."""
    
    def __init__(self):
        self._schema_ensured = False
    
    async def _ensure_schema(self):
        """Ensure user study profile tables exist."""
        if self._schema_ensured:
            return
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Main user_study_profiles table (mirrors studies but with user_id)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_study_profiles (
                    profile_id SERIAL PRIMARY KEY,
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    upload_id VARCHAR(64) REFERENCES user_uploads(upload_id) ON DELETE CASCADE,
                    
                    document_name VARCHAR(500) NOT NULL,
                    doc_id VARCHAR(500),
                    extraction_timestamp TIMESTAMP DEFAULT NOW(),
                    processing_duration_seconds FLOAT,
                    
                    -- Identifiers
                    doi VARCHAR(255),
                    pmid VARCHAR(50),
                    
                    -- Study details (stored as JSONB for flexibility)
                    study_details JSONB,
                    patient_characteristics JSONB,
                    diagnosis JSONB,
                    staging JSONB,
                    treatment JSONB,
                    outcomes JSONB,
                    biomarkers JSONB,
                    toxicity JSONB,
                    dose_constraints JSONB,
                    
                    -- Timestamps
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    UNIQUE(user_id, upload_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_user_study_profiles_user_id 
                ON user_study_profiles(user_id);
                
                CREATE INDEX IF NOT EXISTS idx_user_study_profiles_doc_id 
                ON user_study_profiles(doc_id);
                
                CREATE INDEX IF NOT EXISTS idx_user_study_profiles_upload_id 
                ON user_study_profiles(upload_id);
            """)
        
        self._schema_ensured = True
        print("✓ User study profile schema ensured")

    async def store_user_study_profile(
        self,
        user_id: str,
        upload_id: str,
        doc_id: str,
        document_name: str,
        extracted_data: Dict[str, Any],
        processing_duration: float = None
    ) -> int:
        """
        Store extracted study profile for a user upload.
        
        Args:
            user_id: User's UUID
            upload_id: Upload ID from user_uploads table
            doc_id: Document identifier
            document_name: Document name
            extracted_data: Extracted study profile data from LLM
            processing_duration: Time taken to extract
        
        Returns:
            profile_id of the inserted record
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Check if profile already exists for this upload
            existing = await conn.fetchval(
                "SELECT profile_id FROM user_study_profiles WHERE upload_id = $1",
                upload_id
            )
            
            if existing:
                print(f"  ℹ️  User study profile already exists (profile_id={existing}), skipping")
                return existing
            
            # Extract sections from the data
            study_details = extracted_data.get('study_details', {})
            patient_chars = extracted_data.get('patient_characteristics', {})
            diagnosis = extracted_data.get('diagnosis', {})
            staging = extracted_data.get('staging', {})
            treatment = extracted_data.get('treatment', {})
            outcomes = extracted_data.get('outcomes', {})
            biomarkers = extracted_data.get('biomarkers', [])
            toxicity = extracted_data.get('toxicity', [])
            dose_constraints = extracted_data.get('dose_constraints', [])
            
            # Extract DOI and PMID
            doi = None
            pmid = None
            if isinstance(study_details.get('doi'), dict):
                doi = study_details['doi'].get('value')
            elif study_details.get('doi'):
                doi = study_details['doi']
            if isinstance(study_details.get('pmid'), dict):
                pmid = study_details['pmid'].get('value')
            elif study_details.get('pmid'):
                pmid = study_details['pmid']
            
            profile_id = await conn.fetchval("""
                INSERT INTO user_study_profiles (
                    user_id, upload_id, document_name, doc_id,
                    extraction_timestamp, processing_duration_seconds,
                    doi, pmid,
                    study_details, patient_characteristics, diagnosis,
                    staging, treatment, outcomes,
                    biomarkers, toxicity, dose_constraints
                ) VALUES (
                    $1, $2, $3, $4,
                    NOW(), $5,
                    $6, $7,
                    $8, $9, $10,
                    $11, $12, $13,
                    $14, $15, $16
                ) RETURNING profile_id
            """,
                user_id, upload_id, document_name, doc_id,
                processing_duration,
                doi, pmid,
                json.dumps(study_details), json.dumps(patient_chars), json.dumps(diagnosis),
                json.dumps(staging), json.dumps(treatment), json.dumps(outcomes),
                json.dumps(biomarkers), json.dumps(toxicity), json.dumps(dose_constraints)
            )
            
            return profile_id
    
    async def get_user_study_profile(
        self,
        user_id: str,
        upload_id: str = None,
        doc_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a user's study profile by upload_id or doc_id.
        
        Args:
            user_id: User's UUID
            upload_id: Upload ID
            doc_id: Document ID
            
        Returns:
            Study profile data or None
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            if upload_id:
                row = await conn.fetchrow(
                    "SELECT * FROM user_study_profiles WHERE user_id = $1 AND upload_id = $2",
                    user_id, upload_id
                )
            elif doc_id:
                row = await conn.fetchrow(
                    "SELECT * FROM user_study_profiles WHERE user_id = $1 AND doc_id = $2",
                    user_id, doc_id
                )
            else:
                return None
            
            if not row:
                return None
            
            return self._row_to_dict(row)
    
    async def get_user_study_profiles(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all study profiles for a user.
        
        Args:
            user_id: User's UUID
            limit: Maximum number of results
            
        Returns:
            List of study profile data
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM user_study_profiles 
                   WHERE user_id = $1 
                   ORDER BY created_at DESC 
                   LIMIT $2""",
                user_id, limit
            )
            
            return [self._row_to_dict(row) for row in rows]
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary with parsed JSON."""
        data = dict(row)
        
        # Parse JSONB fields
        for field in ['study_details', 'patient_characteristics', 'diagnosis', 
                      'staging', 'treatment', 'outcomes', 
                      'biomarkers', 'toxicity', 'dose_constraints']:
            if data.get(field):
                if isinstance(data[field], str):
                    data[field] = json.loads(data[field])
        
        # Build the response format expected by the frontend
        return {
            'profile_id': data.get('profile_id'),
            'user_id': str(data.get('user_id')) if data.get('user_id') else None,
            'upload_id': data.get('upload_id'),
            'doc_id': data.get('doc_id'),
            'document_name': data.get('document_name'),
            'doi': data.get('doi'),
            'pmid': data.get('pmid'),
            'title': self._get_title(data),
            'study_details': self._build_field_dict(data.get('study_details', {})),
            'patient_characteristics': self._build_patient_chars(data.get('patient_characteristics', {})),
            'diagnosis': self._build_field_dict(data.get('diagnosis', {})),
            'staging': self._build_staging(data.get('staging', {})),
            'treatment': data.get('treatment', {}),
            'outcomes': self._build_field_dict(data.get('outcomes', {})),
            'biomarkers': data.get('biomarkers', []),
            'toxicity': data.get('toxicity', []),
            'dose_constraints': data.get('dose_constraints', []),
            'extraction_timestamp': data.get('extraction_timestamp'),
            'created_at': data.get('created_at')
        }
    
    def _get_title(self, data: Dict) -> str:
        """Extract title from study details."""
        study_details = data.get('study_details', {})
        if isinstance(study_details.get('study_name'), dict):
            return study_details['study_name'].get('value', data.get('document_name', 'Unknown'))
        return study_details.get('study_name') or data.get('document_name', 'Unknown')
    
    def _build_field_dict(self, section_data: Dict) -> Dict:
        """Build field dictionary with labels."""
        result = {}
        for key, value in section_data.items():
            if isinstance(value, dict) and 'value' in value:
                if value['value'] is not None:
                    result[key] = {
                        'label': self._format_label(key),
                        'value': value['value'],
                        'evidence_quote': value.get('evidence_quote')
                    }
            elif value is not None and not isinstance(value, (list, dict)):
                result[key] = {
                    'label': self._format_label(key),
                    'value': value
                }
        return result
    
    def _build_patient_chars(self, data: Dict) -> Dict:
        """Build patient characteristics with criteria arrays."""
        result = self._build_field_dict(data)
        
        # Handle inclusion/exclusion criteria
        for criteria_type in ['inclusion_criteria', 'exclusion_criteria']:
            if criteria_type in data and isinstance(data[criteria_type], list):
                result[criteria_type] = [
                    {
                        'criterion': c.get('criterion') or c.get('value') if isinstance(c, dict) else c,
                        'evidence_quote': c.get('evidence_quote') if isinstance(c, dict) else None
                    }
                    for c in data[criteria_type]
                ]
        
        return result
    
    def _build_staging(self, data: Dict) -> Dict:
        """Build staging section with arrays."""
        result = self._build_field_dict(data)
        
        # Handle stage_distribution and staging_components
        for array_field in ['stage_distribution', 'staging_components']:
            if array_field in data and isinstance(data[array_field], list):
                result[array_field] = data[array_field]
        
        return result
    
    def _format_label(self, key: str) -> str:
        """Format field key as label."""
        return key.replace('_', ' ').title()
    
    async def delete_user_study_profile(self, user_id: str, upload_id: str) -> bool:
        """Delete a user's study profile."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_study_profiles WHERE user_id = $1 AND upload_id = $2",
                user_id, upload_id
            )
            return "DELETE 1" in result


# Singleton instance
_user_study_profile_service = None


def get_user_study_profile_service() -> UserStudyProfileService:
    """Get singleton instance of UserStudyProfileService."""
    global _user_study_profile_service
    if _user_study_profile_service is None:
        _user_study_profile_service = UserStudyProfileService()
    return _user_study_profile_service
