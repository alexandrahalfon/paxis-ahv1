"""
Study Profiles Filtering Service

Connects to the study-profiles database (separate from display-study-details)
to perform pre-filtering using the cancer_types lookup table with synonyms.

This service is used by the RAG prefilter to narrow down Qdrant search space
based on clinical profile characteristics.

IMPORTANT: Uses normalized schema with JOINs:
- studies.cancer_type_id -> cancer_types.id
- diagnoses.study_id -> studies.id (for cancer_location, histology)

NOTE: Uses fresh connections per request to avoid pool conflicts in concurrent requests.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class StudyProfileMatch:
    """A matching study from the study-profiles database."""
    study_id: int
    doc_id: str
    cancer_type: str
    cancer_location: Optional[str] = None
    histopathologic_type: Optional[str] = None
    study_name: Optional[str] = None
    number_of_patients: Optional[int] = None


@dataclass 
class ProfileFilterResult:
    """Result of filtering studies by clinical profile."""
    doc_ids: List[str] = field(default_factory=list)
    study_ids: List[int] = field(default_factory=list)
    matches: List[StudyProfileMatch] = field(default_factory=list)
    search_terms_used: Dict[str, List[str]] = field(default_factory=dict)
    filter_applied: bool = False
    filter_reason: str = ""
    timing_ms: float = 0.0
    
    def has_filter(self) -> bool:
        """Check if filter was applied and has doc_ids."""
        return self.filter_applied and len(self.doc_ids) > 0


class StudyProfilesFilteringService:
    """
    Service for filtering studies from the study-profiles database.
    
    Uses fresh connections per request to avoid pool conflicts.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None,
    ):
        from src.core.config import settings as _settings
        self.host = host or _settings.postgres_host
        self.port = port or _settings.postgres_port
        self.user = user or _settings.postgres_user
        self.password = password or _settings.postgres_password
        self.database = database or _settings.study_profiles_database
        
        # Cache for cancer type search terms (thread-safe for reads)
        self._cancer_type_cache: Dict[str, List[str]] = {}
        
        print(f"[StudyProfilesFilter] Initialized for database: {self.database}")
    
    async def _create_connection(self) -> asyncpg.Connection:
        """Create a fresh connection for this request."""
        return await asyncpg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            timeout=30,
        )
    
    def _get_fallback_terms(self, cancer_type: str) -> List[str]:
        """Fallback hardcoded mappings for common cancer types."""
        ctype = cancer_type.lower().strip()
        
        mappings = {
            'ependymoma': ['ependymoma', 'brain', 'CNS', 'glioma', 'pediatric brain'],
            'medulloblastoma': ['medulloblastoma', 'brain', 'CNS', 'pediatric brain'],
            'glioblastoma': ['glioblastoma', 'GBM', 'brain', 'glioma', 'CNS'],
            'glioma': ['glioma', 'brain', 'CNS', 'astrocytoma'],
            'brain': ['brain', 'CNS', 'central nervous system', 'glioma'],
            'cns': ['CNS', 'brain', 'central nervous system', 'glioma'],
            'nsclc': ['non-small cell', 'NSCLC', 'non small cell', 'lung'],
            'sclc': ['small cell', 'SCLC', 'lung'],
            'lung': ['lung', 'pulmonary', 'NSCLC', 'SCLC'],
            'scc': ['squamous', 'squamous cell', 'SCC'],
            'squamous': ['squamous', 'squamous cell', 'SCC'],
            'head_and_neck': ['head and neck', 'H&N', 'HNSCC', 'oropharyngeal', 'laryngeal', 'nasopharyngeal'],
            'head and neck': ['head and neck', 'H&N', 'HNSCC', 'oropharyngeal', 'laryngeal'],
            'h&n': ['head and neck', 'H&N', 'HNSCC', 'oropharyngeal', 'laryngeal'],
            'breast': ['breast', 'mammary', 'DCIS', 'ductal'],
            'dcis': ['DCIS', 'ductal carcinoma', 'breast', 'in situ'],
            'prostate': ['prostate', 'prostatic'],
            'colorectal': ['colorectal', 'colon', 'rectal', 'CRC'],
            'cervical': ['cervix', 'cervical'],
            'melanoma': ['melanoma', 'skin'],
            'lymphoma': ['lymphoma', 'hodgkin', 'non-hodgkin'],
            'pancreatic': ['pancreatic', 'pancreas'],
            'ovarian': ['ovarian', 'ovary'],
            'esophageal': ['esophageal', 'esophagus'],
            'gastric': ['gastric', 'stomach'],
            'renal': ['renal', 'kidney'],
            'bladder': ['bladder', 'urothelial'],
        }
        
        return mappings.get(ctype, [ctype])
    
    def get_site_search_terms(self, anatomical_site: str) -> List[str]:
        """Get search terms for anatomical site."""
        if not anatomical_site:
            return []
        
        site = anatomical_site.lower().strip()
        
        site_mappings = {
            'maxilla': ['maxilla', 'oral cavity', 'head and neck'],
            'oral cavity': ['oral cavity', 'oral', 'head and neck'],
            'tongue': ['tongue', 'oral cavity', 'head and neck'],
            'oropharynx': ['oropharynx', 'pharynx', 'head and neck'],
            'nasopharynx': ['nasopharynx', 'pharynx', 'head and neck'],
            'larynx': ['larynx', 'head and neck'],
            'brain': ['brain', 'cerebral', 'intracranial', 'CNS'],
            'lung': ['lung', 'pulmonary', 'thoracic'],
            'breast': ['breast', 'mammary'],
            'cervix': ['cervix', 'cervical'],
            'rectum': ['rectal', 'rectum', 'colorectal'],
            'prostate': ['prostate'],
            'bladder': ['bladder', 'urothelial'],
        }
        
        return site_mappings.get(site, [site])
    
    async def _get_cancer_type_terms(self, conn: asyncpg.Connection, cancer_type: str) -> List[str]:
        """Get expanded search terms for a cancer type."""
        if not cancer_type:
            return []
        
        ctype = cancer_type.lower().strip()
        
        # Check cache first
        if ctype in self._cancer_type_cache:
            return self._cancer_type_cache[ctype]
        
        try:
            # Query the lookup table
            row = await conn.fetchrow("""
                SELECT code, label, synonyms, keywords, subtypes
                FROM cancer_types
                WHERE 
                    LOWER(code) = $1
                    OR LOWER(label) ILIKE $2
                    OR $1 = ANY(SELECT LOWER(s) FROM unnest(synonyms) s)
                    OR $1 = ANY(SELECT LOWER(k) FROM unnest(keywords) k)
                LIMIT 1
            """, ctype, f"%{ctype}%")
            
            if row:
                terms = set()
                if row['code']:
                    terms.add(row['code'])
                if row['label']:
                    terms.add(row['label'])
                for arr_field in ['synonyms', 'keywords', 'subtypes']:
                    if row.get(arr_field):
                        for item in row[arr_field]:
                            if item:
                                terms.add(item)
                if terms:
                    result = list(terms)
                    self._cancer_type_cache[ctype] = result
                    return result
        except Exception as e:
            print(f"[StudyProfilesFilter] Error querying cancer_types: {e}")
        
        # Fallback to hardcoded mappings
        fallback = self._get_fallback_terms(ctype)
        self._cancer_type_cache[ctype] = fallback
        return fallback
    
    async def filter_studies_by_profile(
        self,
        cancer_type: str = None,
        anatomical_site: str = None,
        histology: str = None,
        stage: str = None,
        limit: int = 200,
        min_results: int = 5,
    ) -> ProfileFilterResult:
        """
        Filter studies by clinical profile characteristics.
        
        Uses a fresh connection to avoid pool conflicts.
        """
        import time
        start_time = time.perf_counter()
        
        result = ProfileFilterResult()
        
        if not any([cancer_type, anatomical_site, histology]):
            result.filter_reason = "no_clinical_context"
            return result
        
        conn = None
        try:
            # Create fresh connection for this request
            conn = await self._create_connection()
            
            # Get search terms
            search_terms_used = {}
            type_terms = []
            site_terms = []
            
            if cancer_type:
                type_terms = await self._get_cancer_type_terms(conn, cancer_type)
                search_terms_used['cancer_type'] = type_terms
            
            if anatomical_site:
                site_terms = self.get_site_search_terms(anatomical_site)
                search_terms_used['anatomical_site'] = site_terms
            
            if histology:
                search_terms_used['histology'] = [histology]
            
            if stage:
                search_terms_used['stage'] = [stage]
            
            # Build query
            conditions = []
            params = []
            param_idx = 1
            
            # Cancer type conditions
            if cancer_type and type_terms:
                type_conditions = []
                for term in type_terms:
                    type_conditions.append(f"""(
                        ct.label ILIKE ${param_idx} 
                        OR ct.code ILIKE ${param_idx}
                        OR d.cancer_type_raw ILIKE ${param_idx}
                        OR d.cancer_type_normalized ILIKE ${param_idx}
                        OR d.histopathologic_type_raw ILIKE ${param_idx}
                    )""")
                    params.append(f"%{term}%")
                    param_idx += 1
                
                if type_conditions:
                    conditions.append(f"({' OR '.join(type_conditions)})")
            
            # Anatomical site conditions
            if anatomical_site and site_terms:
                site_conditions = []
                for term in site_terms:
                    site_conditions.append(f"""(
                        d.cancer_location_raw ILIKE ${param_idx}
                        OR d.anatomical_site ILIKE ${param_idx}
                    )""")
                    params.append(f"%{term}%")
                    param_idx += 1
                
                if site_conditions:
                    conditions.append(f"({' OR '.join(site_conditions)})")
            
            # Histology
            if histology:
                conditions.append(f"""(
                    d.histopathologic_type_raw ILIKE ${param_idx}
                    OR d.histopathologic_type_normalized ILIKE ${param_idx}
                )""")
                params.append(f"%{histology}%")
                param_idx += 1
            
            # Stage
            if stage:
                conditions.append(f"""
                    EXISTS (
                        SELECT 1 FROM stage_distributions sd 
                        WHERE sd.study_id = s.id 
                        AND (sd.stage_category ILIKE ${param_idx} OR sd.stage_normalized ILIKE ${param_idx})
                    )
                """)
                params.append(f"%{stage}%")
                param_idx += 1
            
            if not conditions:
                result.filter_reason = "no_valid_conditions"
                result.timing_ms = (time.perf_counter() - start_time) * 1000
                return result
            
            where_clause = " AND ".join(conditions)
            
            # Main query
            query = f"""
                SELECT DISTINCT
                    s.id as study_id,
                    s.doc_id,
                    s.study_name,
                    ct.label as cancer_type,
                    d.cancer_location_raw as cancer_location,
                    d.histopathologic_type_normalized as histopathologic_type,
                    s.number_of_patients
                FROM studies s
                LEFT JOIN cancer_types ct ON s.cancer_type_id = ct.id
                LEFT JOIN diagnoses d ON d.study_id = s.id
                WHERE {where_clause}
                  AND s.doc_id IS NOT NULL
                ORDER BY s.number_of_patients DESC NULLS LAST
                LIMIT ${param_idx}
            """
            params.append(limit)
            
            print(f"[StudyProfilesFilter] Search terms: {search_terms_used}")
            
            rows = await conn.fetch(query, *params)
            
            # Fallback to broader search if too few results
            if len(rows) < min_results and cancer_type and anatomical_site and type_terms:
                print(f"[StudyProfilesFilter] Only {len(rows)} results, trying broader search")
                rows = await self._search_cancer_type_only(conn, type_terms, limit)
            
            if not rows:
                result.filter_reason = "no_pg_matches"
                result.search_terms_used = search_terms_used
                result.timing_ms = (time.perf_counter() - start_time) * 1000
                return result
            
            # Extract results
            for row in rows:
                if row['doc_id']:
                    result.doc_ids.append(row['doc_id'])
                    result.study_ids.append(row['study_id'])
                    result.matches.append(StudyProfileMatch(
                        study_id=row['study_id'],
                        doc_id=row['doc_id'],
                        cancer_type=row['cancer_type'],
                        cancer_location=row['cancer_location'],
                        histopathologic_type=row['histopathologic_type'],
                        study_name=row['study_name'],
                        number_of_patients=row['number_of_patients'],
                    ))
            
            result.filter_applied = len(result.doc_ids) > 0
            result.filter_reason = "applied" if result.filter_applied else "no_doc_ids"
            result.search_terms_used = search_terms_used
            result.timing_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"[StudyProfilesFilter] Found {len(result.doc_ids)} studies in {result.timing_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[StudyProfilesFilter] Error: {e}")
            result.filter_reason = f"error: {str(e)}"
            result.timing_ms = (time.perf_counter() - start_time) * 1000
            return result
        finally:
            if conn:
                await conn.close()
    
    async def _search_cancer_type_only(
        self, 
        conn: asyncpg.Connection, 
        type_terms: List[str],
        limit: int
    ) -> List[asyncpg.Record]:
        """Fallback search using only cancer type terms."""
        if not type_terms:
            return []
        
        conditions = []
        params = []
        param_idx = 1
        
        for term in type_terms:
            conditions.append(f"""(
                ct.label ILIKE ${param_idx} 
                OR ct.code ILIKE ${param_idx}
                OR d.cancer_type_raw ILIKE ${param_idx}
            )""")
            params.append(f"%{term}%")
            param_idx += 1
        
        query = f"""
            SELECT DISTINCT
                s.id as study_id,
                s.doc_id,
                s.study_name,
                ct.label as cancer_type,
                d.cancer_location_raw as cancer_location,
                d.histopathologic_type_normalized as histopathologic_type,
                s.number_of_patients
            FROM studies s
            LEFT JOIN cancer_types ct ON s.cancer_type_id = ct.id
            LEFT JOIN diagnoses d ON d.study_id = s.id
            WHERE ({' OR '.join(conditions)})
              AND s.doc_id IS NOT NULL
            ORDER BY s.number_of_patients DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit)
        
        return await conn.fetch(query, *params)


# Singleton instance
_service_instance: Optional[StudyProfilesFilteringService] = None


def get_study_profiles_filtering_service() -> StudyProfilesFilteringService:
    """Get or create the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = StudyProfilesFilteringService()
    return _service_instance
