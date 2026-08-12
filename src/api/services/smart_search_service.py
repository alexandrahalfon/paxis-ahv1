"""
Smart Search Service

Combines user preferences (persistent filters) with case context (relevance matching)
for intelligent study search and ranking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import asyncpg
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Database pool
_study_pool = None

async def _get_study_pool():
    """Get connection pool for the studies database"""
    global _study_pool
    if _study_pool is None:
        _study_pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database,
            min_size=2,
            max_size=10,
            timeout=30
        )
    return _study_pool


@dataclass
class CaseContext:
    """
    Patient-specific details extracted from current query.
    These are NOT persistent filters - they're contextual matching criteria.
    Used for: relevance scoring, outcome prediction, eligibility checking.
    """
    # Patient Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    race_ethnicity: Optional[str] = None
    
    # Disease Characteristics
    cancer_type: Optional[str] = None
    cancer_location: Optional[str] = None
    histopathologic_type: Optional[str] = None
    tumor_grade: Optional[str] = None
    
    # Staging
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    overall_stage: Optional[str] = None
    metastatic_status: Optional[str] = None
    
    # Disease burden
    largest_lesion_size_cm: Optional[float] = None
    number_of_lesions: Optional[int] = None
    involved_subsites: List[str] = field(default_factory=list)
    lymph_node_levels_involved: List[str] = field(default_factory=list)
    
    # Molecular/Biomarkers
    molecular_subtype: Optional[str] = None
    specific_mutations: List[str] = field(default_factory=list)
    pd_l1_expression: Optional[str] = None
    hpv_status: Optional[str] = None
    
    # Treatment History
    prior_surgery: List[str] = field(default_factory=list)
    prior_chemotherapy_regimens: List[str] = field(default_factory=list)
    prior_radiation_completed: bool = False
    prior_radiation_dose_gy: Optional[float] = None
    number_of_prior_therapies: Optional[int] = None
    
    # Disease Timeline
    initial_diagnosis_date: Optional[str] = None
    recurrence_date: Optional[str] = None
    time_to_recurrence_months: Optional[int] = None
    treatment_setting: Optional[str] = None  # "adjuvant", "salvage", "definitive", "neoadjuvant"
    
    # Clinical Status
    performance_status_ecog: Optional[int] = None
    comorbidities: List[str] = field(default_factory=list)
    smoking_status: Optional[str] = None
    
    # Treatment Intent
    treatment_intent: Optional[str] = None  # "curative", "palliative"
    is_surgical_candidate: Optional[bool] = None
    is_radiation_candidate: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SmartSearchService:
    """
    Service for intelligent search combining preferences and case context.
    """
    
    async def build_preference_filter(self, preferences: Dict[str, Any]) -> str:
        """
        Build PostgreSQL WHERE clause from user preferences.
        
        Args:
            preferences: User preferences dict
            
        Returns:
            SQL WHERE clause conditions
        """
        conditions = []
        params = []
        param_idx = 1
        
        # Study types
        if preferences.get("study_types"):
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(preferences["study_types"])))
            conditions.append(f"study_type IN ({placeholders})")
            params.extend(preferences["study_types"])
            param_idx += len(preferences["study_types"])
        
        # Study phases
        if preferences.get("study_phases"):
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(preferences["study_phases"])))
            conditions.append(f"study_phase IN ({placeholders})")
            params.extend(preferences["study_phases"])
            param_idx += len(preferences["study_phases"])
        
        # Min patients
        if preferences.get("min_patients"):
            conditions.append(f"number_of_patients >= ${param_idx}")
            params.append(preferences["min_patients"])
            param_idx += 1
        
        # Max patients
        if preferences.get("max_patients"):
            conditions.append(f"number_of_patients <= ${param_idx}")
            params.append(preferences["max_patients"])
            param_idx += 1
        
        # Countries
        if preferences.get("countries"):
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(preferences["countries"])))
            conditions.append(f"country IN ({placeholders})")
            params.extend(preferences["countries"])
            param_idx += len(preferences["countries"])
        
        # Institutions
        if preferences.get("institutions"):
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(preferences["institutions"])))
            conditions.append(f"study_institution IN ({placeholders})")
            params.extend(preferences["institutions"])
            param_idx += len(preferences["institutions"])
        
        # Min publication year
        if preferences.get("min_publication_year"):
            conditions.append(f"EXTRACT(YEAR FROM publish_date) >= ${param_idx}")
            params.append(preferences["min_publication_year"])
            param_idx += 1
        
        # Max publication year
        if preferences.get("max_publication_year"):
            conditions.append(f"EXTRACT(YEAR FROM publish_date) <= ${param_idx}")
            params.append(preferences["max_publication_year"])
            param_idx += 1
        
        # Exclude treatment modalities (e.g., non-surgeons hiding surgical trials)
        if preferences.get("exclude_treatment_modalities"):
            for modality in preferences["exclude_treatment_modalities"]:
                # Exclude studies where this is the ONLY treatment
                conditions.append(f"NOT (study_type ILIKE ${param_idx})")
                params.append(f"%{modality}%")
                param_idx += 1
        
        # Race/ethnicity filter
        # Only include studies that enrolled patients of the specified races
        if preferences.get("race_ethnicities"):
            race_conditions = []
            for race in preferences["race_ethnicities"]:
                # Map standard race labels to search patterns
                race_patterns = self._get_race_search_patterns(race)
                for pattern in race_patterns:
                    race_conditions.append(f"race_ethnicity ILIKE ${param_idx}")
                    params.append(f"%{pattern}%")
                    param_idx += 1
            
            # If include_unknown_race is True, also include studies with no race data
            if preferences.get("include_unknown_race", True):
                race_conditions.append("race_ethnicity IS NULL")
                race_conditions.append("race_ethnicity = ''")
            
            if race_conditions:
                conditions.append(f"({' OR '.join(race_conditions)})")
        
        return conditions, params
    
    def _get_race_search_patterns(self, race_label: str) -> List[str]:
        """Get search patterns for a race label."""
        race_patterns = {
            "White": ["white", "caucasian"],
            "Black": ["black", "african american", "african-american"],
            "Asian": ["asian"],
            "Hispanic/Latino": ["hispanic", "latino"],
            "Native American": ["native american", "american indian"],
            "Pacific Islander": ["pacific islander", "hawaiian"],
            "Middle Eastern": ["middle eastern", "arab"],
            "Mixed/Other": ["mixed", "multiracial", "other"],
        }
        return race_patterns.get(race_label, [race_label.lower()])
    
    async def get_matching_study_ids(self, preferences: Dict[str, Any]) -> List[int]:
        """
        Get study IDs matching user preferences from PostgreSQL.
        
        Args:
            preferences: User preferences dict
            
        Returns:
            List of matching study_ids
        """
        if not preferences.get("filters_active", True):
            # If filters disabled, return all studies
            pool = await _get_study_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT study_id FROM studies")
                return [row["study_id"] for row in rows]
        
        conditions, params = await self.build_preference_filter(preferences)
        
        pool = await _get_study_pool()
        async with pool.acquire() as conn:
            if conditions:
                where_clause = " AND ".join(conditions)
                query = f"SELECT study_id FROM studies WHERE {where_clause}"
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch("SELECT study_id FROM studies")
            
            return [row["study_id"] for row in rows]
    
    def calculate_case_similarity(
        self, 
        case_context: CaseContext, 
        study: Dict[str, Any]
    ) -> float:
        """
        Calculate how well this study matches the patient case.
        Returns score 0-1.
        
        This is NOT a filter - it's a relevance booster.
        """
        score = 0.0
        total_weight = 0.0
        
        # Age matching (±10 years)
        if case_context.age and study.get('median_age'):
            try:
                median_age = float(study['median_age']) if study['median_age'] else None
                if median_age:
                    age_diff = abs(case_context.age - median_age)
                    if age_diff <= 10:
                        score += (1 - age_diff/10) * 0.1
                    total_weight += 0.1
            except (ValueError, TypeError):
                pass
        
        # Gender matching
        if case_context.gender and study.get('gender_distribution'):
            gender_dist = str(study['gender_distribution']).lower()
            if case_context.gender.lower() in gender_dist:
                score += 0.05
            total_weight += 0.05
        
        # Cancer type matching
        if case_context.cancer_type and study.get('cancer_type'):
            study_cancer = str(study['cancer_type']).lower()
            patient_cancer = case_context.cancer_type.lower()
            if patient_cancer in study_cancer or study_cancer in patient_cancer:
                score += 0.20  # Cancer type match is critical
            total_weight += 0.20
        
        # Cancer location matching
        if case_context.cancer_location and study.get('cancer_location'):
            study_loc = str(study['cancer_location']).lower()
            patient_loc = case_context.cancer_location.lower()
            if patient_loc in study_loc or study_loc in patient_loc:
                score += 0.15
            total_weight += 0.15
        
        # T-stage matching
        if case_context.tnm_t:
            # Check if study includes this T stage
            # Look in inclusion criteria or stage distribution
            stage_info = str(study.get('staging_system_used', '')).lower()
            if case_context.tnm_t.lower() in stage_info:
                score += 0.15
            total_weight += 0.15
        
        # N-stage matching
        if case_context.tnm_n:
            stage_info = str(study.get('staging_system_used', '')).lower()
            if case_context.tnm_n.lower() in stage_info:
                score += 0.15
            total_weight += 0.15
        
        # Treatment setting matching (adjuvant, salvage, definitive)
        if case_context.treatment_setting:
            study_type = str(study.get('study_type', '')).lower()
            study_name = str(study.get('study_name', '')).lower()
            setting = case_context.treatment_setting.lower()
            if setting in study_type or setting in study_name:
                score += 0.20  # Treatment setting is very important
            total_weight += 0.20
        
        # Performance status matching
        if case_context.performance_status_ecog is not None:
            ps_info = str(study.get('performance_status', '')).lower()
            if str(case_context.performance_status_ecog) in ps_info:
                score += 0.10
            total_weight += 0.10
        
        # Molecular subtype matching
        if case_context.molecular_subtype:
            mol_info = str(study.get('molecular_subtype', '')).lower()
            if case_context.molecular_subtype.lower() in mol_info:
                score += 0.15  # Biomarker match is critical
            total_weight += 0.15
        
        # HPV status matching (important for head/neck)
        if case_context.hpv_status:
            study_text = f"{study.get('study_name', '')} {study.get('cancer_type', '')}".lower()
            if case_context.hpv_status.lower() in study_text:
                score += 0.10
            total_weight += 0.10
        
        # Normalize
        return score / total_weight if total_weight > 0 else 0.0
    
    async def get_study_metadata(self, study_id: int) -> Optional[Dict[str, Any]]:
        """
        Get study metadata from PostgreSQL for relevance scoring.
        """
        pool = await _get_study_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    study_id, study_name, study_type, study_phase,
                    cancer_type, cancer_location, number_of_patients,
                    median_age, gender_distribution, performance_status,
                    staging_system_used, molecular_subtype, publish_date,
                    overall_survival, progression_free_survival,
                    citation_count
                FROM studies
                WHERE study_id = $1
            """, study_id)
            
            return dict(row) if row else None
    
    async def rerank_results(
        self,
        results: List[Dict[str, Any]],
        case_context: CaseContext,
        vector_weight: float = 0.6,
        relevance_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using case similarity scoring.
        
        Args:
            results: List of search results with 'study_id' and 'score'
            case_context: Patient case context
            vector_weight: Weight for vector similarity score (default 0.6)
            relevance_weight: Weight for case relevance score (default 0.4)
            
        Returns:
            Re-ranked results with combined scores
        """
        scored_results = []
        
        for result in results:
            study_id = result.get('study_id') or result.get('pg_study_id')
            if not study_id:
                scored_results.append(result)
                continue
            
            # Get study metadata
            study = await self.get_study_metadata(study_id)
            if not study:
                scored_results.append(result)
                continue
            
            # Calculate relevance score
            relevance_score = self.calculate_case_similarity(case_context, study)
            
            # Get vector score
            vector_score = result.get('score', 0.5)
            
            # Combined score
            combined_score = (vector_weight * vector_score) + (relevance_weight * relevance_score)
            
            result['relevance_score'] = relevance_score
            result['vector_score'] = vector_score
            result['combined_score'] = combined_score
            result['study_metadata'] = study
            
            scored_results.append(result)
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return scored_results
    
    async def sort_by_preference(
        self,
        results: List[Dict[str, Any]],
        sort_by: str,
        case_context: Optional[CaseContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Sort results according to user preference.
        
        Args:
            results: Search results
            sort_by: Sort preference (relevance, population, date, citations, outcomes, patient_relevance)
            case_context: Optional case context for patient-relevance and outcome-based sorting
            
        Returns:
            Sorted results
        """
        if sort_by == 'relevance':
            # Already sorted by combined_score from rerank
            return sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        elif sort_by == 'patient_relevance':
            # Sort by patient-specific relevance score
            # This boosts studies that match the patient's cancer type, histology, stage, etc.
            if case_context:
                # Calculate patient relevance for each result
                for result in results:
                    study_id = result.get('study_id') or result.get('pg_study_id')
                    if study_id:
                        study = result.get('study_metadata')
                        if not study:
                            study = await self.get_study_metadata(study_id)
                            result['study_metadata'] = study
                        
                        if study:
                            patient_score = self.calculate_patient_relevance_score(case_context, study)
                            result['patient_relevance_score'] = patient_score
                        else:
                            result['patient_relevance_score'] = 0.0
                    else:
                        result['patient_relevance_score'] = 0.0
                
                return sorted(results, key=lambda x: x.get('patient_relevance_score', 0), reverse=True)
            else:
                # No case context - fall back to relevance
                return sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        elif sort_by == 'population':
            return sorted(
                results, 
                key=lambda x: x.get('study_metadata', {}).get('number_of_patients', 0) or 0,
                reverse=True
            )
        
        elif sort_by == 'date':
            def get_date(r):
                pub_date = r.get('study_metadata', {}).get('publish_date')
                if pub_date:
                    return str(pub_date)
                return '1900-01-01'
            return sorted(results, key=get_date, reverse=True)
        
        elif sort_by == 'citations':
            return sorted(
                results,
                key=lambda x: x.get('study_metadata', {}).get('citation_count', 0) or 0,
                reverse=True
            )
        
        elif sort_by == 'outcomes':
            # Sort by best outcomes (OS, PFS)
            def get_outcome_score(r):
                meta = r.get('study_metadata', {})
                os_val = meta.get('overall_survival', '')
                pfs_val = meta.get('progression_free_survival', '')
                
                # Try to extract numeric values from outcome strings
                score = 0
                try:
                    # Look for percentages like "85%" or "85.3%"
                    import re
                    os_match = re.search(r'(\d+\.?\d*)%', str(os_val))
                    if os_match:
                        score += float(os_match.group(1))
                    pfs_match = re.search(r'(\d+\.?\d*)%', str(pfs_val))
                    if pfs_match:
                        score += float(pfs_match.group(1)) * 0.8  # Weight PFS slightly less
                except:
                    pass
                return score
            
            return sorted(results, key=get_outcome_score, reverse=True)
        
        return results
    
    def calculate_patient_relevance_score(
        self,
        case_context: CaseContext,
        study: Dict[str, Any]
    ) -> float:
        """
        Calculate a comprehensive patient relevance score.
        
        This is an enhanced version of calculate_case_similarity that:
        1. Gives higher weight to cancer type/histology matches
        2. Considers stage compatibility
        3. Factors in treatment setting alignment
        4. Includes molecular marker matching
        
        Returns score 0-100 for easier interpretation.
        """
        score = 0.0
        max_score = 0.0
        
        # === CRITICAL MATCHES (high weight) ===
        
        # Cancer type match (25 points max)
        if case_context.cancer_type:
            max_score += 25
            study_cancer = str(study.get('cancer_type', '')).lower()
            patient_cancer = case_context.cancer_type.lower()
            
            if patient_cancer == study_cancer:
                score += 25  # Exact match
            elif patient_cancer in study_cancer or study_cancer in patient_cancer:
                score += 20  # Partial match
            elif self._cancer_types_related(patient_cancer, study_cancer):
                score += 10  # Related cancer types
        
        # Cancer location/site match (20 points max)
        if case_context.cancer_location:
            max_score += 20
            study_loc = str(study.get('cancer_location', '')).lower()
            patient_loc = case_context.cancer_location.lower()
            
            if patient_loc == study_loc:
                score += 20
            elif patient_loc in study_loc or study_loc in patient_loc:
                score += 15
            elif self._locations_related(patient_loc, study_loc):
                score += 8
        
        # Histopathologic type match (15 points max)
        if case_context.histopathologic_type:
            max_score += 15
            study_histology = str(study.get('histology', '') or study.get('histopathologic_type', '')).lower()
            patient_histology = case_context.histopathologic_type.lower()
            
            if patient_histology in study_histology or study_histology in patient_histology:
                score += 15
            elif self._histology_compatible(patient_histology, study_histology):
                score += 8
        
        # === STAGING MATCHES (moderate weight) ===
        
        # Overall stage match (10 points max)
        if case_context.overall_stage:
            max_score += 10
            stage_info = f"{study.get('staging_system_used', '')} {study.get('study_name', '')}".lower()
            patient_stage = case_context.overall_stage.lower()
            
            # Check for stage in study info
            if f"stage {patient_stage}" in stage_info or f"stage{patient_stage}" in stage_info:
                score += 10
            elif patient_stage in stage_info:
                score += 6
        
        # T-stage match (5 points max)
        if case_context.tnm_t:
            max_score += 5
            stage_info = str(study.get('staging_system_used', '')).lower()
            if case_context.tnm_t.lower() in stage_info:
                score += 5
        
        # N-stage match (5 points max)
        if case_context.tnm_n:
            max_score += 5
            stage_info = str(study.get('staging_system_used', '')).lower()
            if case_context.tnm_n.lower() in stage_info:
                score += 5
        
        # === TREATMENT CONTEXT (moderate weight) ===
        
        # Treatment setting match (10 points max)
        if case_context.treatment_setting:
            max_score += 10
            study_text = f"{study.get('study_type', '')} {study.get('study_name', '')}".lower()
            setting = case_context.treatment_setting.lower()
            
            if setting in study_text:
                score += 10
            elif self._treatment_settings_compatible(setting, study_text):
                score += 5
        
        # === MOLECULAR/BIOMARKER MATCHES (important for targeted therapy) ===
        
        # Molecular subtype match (10 points max)
        if case_context.molecular_subtype:
            max_score += 10
            mol_info = str(study.get('molecular_subtype', '')).lower()
            if case_context.molecular_subtype.lower() in mol_info:
                score += 10
        
        # HPV status match (5 points max for H&N)
        if case_context.hpv_status:
            max_score += 5
            study_text = f"{study.get('study_name', '')} {study.get('cancer_type', '')}".lower()
            if case_context.hpv_status.lower() in study_text:
                score += 5
        
        # Specific mutations match (5 points max)
        if case_context.specific_mutations:
            max_score += 5
            study_text = f"{study.get('molecular_subtype', '')} {study.get('study_name', '')}".lower()
            matched = sum(1 for m in case_context.specific_mutations if m.lower() in study_text)
            if matched > 0:
                score += min(5, matched * 2)
        
        # === DEMOGRAPHIC MATCHES (lower weight) ===
        
        # Age match (5 points max)
        if case_context.age and study.get('median_age'):
            max_score += 5
            try:
                median_age = float(study['median_age'])
                age_diff = abs(case_context.age - median_age)
                if age_diff <= 5:
                    score += 5
                elif age_diff <= 10:
                    score += 3
                elif age_diff <= 15:
                    score += 1
            except (ValueError, TypeError):
                pass
        
        # Performance status match (5 points max)
        if case_context.performance_status_ecog is not None:
            max_score += 5
            ps_info = str(study.get('performance_status', '')).lower()
            if str(case_context.performance_status_ecog) in ps_info:
                score += 5
        
        # Normalize to 0-100 scale
        if max_score > 0:
            return (score / max_score) * 100
        return 0.0
    
    def _cancer_types_related(self, type1: str, type2: str) -> bool:
        """Check if two cancer types are related (e.g., NSCLC and lung cancer)"""
        related_groups = [
            {'lung', 'nsclc', 'sclc', 'non-small cell', 'small cell lung'},
            {'breast', 'ductal', 'lobular', 'triple negative', 'her2'},
            {'colorectal', 'colon', 'rectal', 'crc'},
            {'head and neck', 'h&n', 'oral', 'oropharyngeal', 'laryngeal', 'nasopharyngeal'},
            {'skin', 'melanoma', 'cutaneous', 'squamous cell skin'},
            {'gynecologic', 'cervical', 'ovarian', 'endometrial', 'uterine'},
        ]
        
        for group in related_groups:
            if any(t in type1 for t in group) and any(t in type2 for t in group):
                return True
        return False
    
    def _locations_related(self, loc1: str, loc2: str) -> bool:
        """Check if two anatomical locations are related"""
        related_locations = [
            {'oral cavity', 'tongue', 'gingiva', 'buccal', 'floor of mouth', 'hard palate', 'maxilla', 'mandible'},
            {'oropharynx', 'tonsil', 'base of tongue', 'soft palate'},
            {'larynx', 'hypopharynx', 'pharynx'},
            {'lung', 'bronchus', 'thoracic'},
            {'colon', 'rectum', 'colorectal'},
        ]
        
        for group in related_locations:
            if any(l in loc1 for l in group) and any(l in loc2 for l in group):
                return True
        return False
    
    def _histology_compatible(self, hist1: str, hist2: str) -> bool:
        """Check if histology types are compatible"""
        compatible_groups = [
            {'squamous', 'scc', 'squamous cell'},
            {'adenocarcinoma', 'adeno'},
            {'small cell', 'sclc'},
            {'non-small cell', 'nsclc'},
        ]
        
        for group in compatible_groups:
            if any(h in hist1 for h in group) and any(h in hist2 for h in group):
                return True
        return False
    
    def _treatment_settings_compatible(self, setting: str, study_text: str) -> bool:
        """Check if treatment settings are compatible"""
        compatible_settings = {
            'adjuvant': ['postoperative', 'post-operative', 'after surgery'],
            'neoadjuvant': ['preoperative', 'pre-operative', 'before surgery', 'induction'],
            'definitive': ['curative', 'radical', 'primary'],
            'salvage': ['recurrent', 'relapsed', 'refractory', 'second-line'],
            'palliative': ['metastatic', 'advanced', 'stage iv'],
        }
        
        if setting in compatible_settings:
            return any(term in study_text for term in compatible_settings[setting])
        return False


# Singleton instance
_smart_search_service: Optional[SmartSearchService] = None


def get_smart_search_service() -> SmartSearchService:
    """Get or create the smart search service singleton"""
    global _smart_search_service
    if _smart_search_service is None:
        _smart_search_service = SmartSearchService()
    return _smart_search_service
