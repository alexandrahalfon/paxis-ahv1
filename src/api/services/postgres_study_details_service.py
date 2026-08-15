"""
PostgreSQL Study Details Service
Retrieves complete study profiles from PostgreSQL database
"""
from typing import Dict, List, Optional, Any
import asyncpg
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)


class PostgresStudyDetailsService:
    """Service for fetching complete study details from PostgreSQL"""
    
    def __init__(
        self,
        pg_host: str = None,
        pg_port: int = None,
        pg_user: str = None,
        pg_password: str = None,
        pg_database: str = None
    ):
        self.pg_host = pg_host or settings.postgres_host
        self.pg_port = pg_port or settings.postgres_port
        self.pg_user = pg_user or settings.postgres_user
        self.pg_password = pg_password or settings.postgres_password
        self.pg_database = pg_database or settings.postgres_database
        
        self._pool = None
        self._cache = {}  # Simple in-memory cache
    
    async def _get_pool(self):
        """Get or create connection pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database=self.pg_database,
                min_size=5,
                max_size=20,
                timeout=30
            )
        return self._pool
    
    async def get_study_details(
        self, 
        doc_id: str = None,
        pmid: str = None, 
        doi: str = None,
        title: str = None
    ) -> Dict[str, Any]:
        """
        Get complete study details from PostgreSQL
        
        Args:
            doc_id: Document ID (may be in format 'doi_10.1016_j.xxx_hash' or 'author-year-title_hash')
            pmid: PubMed ID
            doi: DOI
            title: Study title (for fallback search)
            
        Returns:
            Dictionary with all study fields and related data
        """
        # Extract DOI from doc_id if it starts with 'doi_'
        # Format: doi_10.1016_j.eururo.2019.02.003_e6bce509 -> 10.1016/j.eururo.2019.02.003
        extracted_doi = None
        doc_id_without_hash = None
        document_name_pattern = None
        
        if doc_id:
            if doc_id.startswith('doi_'):
                # Remove 'doi_' prefix and trailing hash (last segment after _)
                # Hash is 8 hex chars at the end
                import re
                hash_match = re.search(r'_([a-f0-9]{8})$', doc_id)
                if hash_match:
                    doc_id_without_hash = doc_id[:hash_match.start()]
                else:
                    doc_id_without_hash = doc_id
                
                logger.info(f"doc_id_without_hash: {doc_id_without_hash}")
                
                # Extract the DOI part (after 'doi_')
                raw_doi_part = doc_id_without_hash[4:] if doc_id_without_hash.startswith('doi_') else doc_id_without_hash
                
                # Convert underscores back to slashes/parentheses for DOI format
                # doi_10.1016_S1470-2045_20_30454-X -> 10.1016/S1470-2045(20)30454-X
                if raw_doi_part.startswith('10.'):
                    # Find first underscore after the prefix number (10.XXXX)
                    first_underscore = raw_doi_part.find('_')
                    if first_underscore > 0:
                        prefix = raw_doi_part[:first_underscore]
                        suffix = raw_doi_part[first_underscore+1:]
                        
                        # Try to convert patterns like _20_ to (20) for Lancet-style DOIs
                        # S1470-2045_20_30454-X -> S1470-2045(20)30454-X
                        suffix_with_parens = re.sub(r'_(\d{2})_', r'(\1)', suffix)
                        
                        extracted_doi = prefix + '/' + suffix_with_parens
                        logger.info(f"Extracted DOI from doc_id: {extracted_doi}")
            else:
                # For non-DOI doc_ids like 'horwitz-et-al-2016-ten-year-follow-up-Question_ACR_5191f1ea'
                # Extract the document name pattern (remove trailing hash)
                # Split by common suffixes like _ACR_, _Question_, etc.
                import re
                # Remove trailing hash (last _ followed by hex chars)
                clean_id = re.sub(r'_[a-f0-9]{6,}$', '', doc_id)
                # Remove _ACR, _Question suffixes
                clean_id = re.sub(r'_(ACR|Question).*$', '', clean_id)
                # Convert dashes to spaces for LIKE matching
                document_name_pattern = clean_id.replace('-', '%')
                logger.info(f"Extracted document name pattern: {document_name_pattern}")
        
        # Check cache
        cache_key = doc_id or pmid or doi or extracted_doi
        if cache_key in self._cache:
            logger.info(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            try:
                # Build WHERE clause with multiple search strategies
                conditions = []
                params = []
                param_idx = 1
                
                if doc_id:
                    conditions.append(f"doc_id = ${param_idx}")
                    params.append(doc_id)
                    param_idx += 1
                    # Also try matching doc_id with LIKE for partial matches
                    conditions.append(f"doc_id LIKE ${param_idx}")
                    params.append(f"%{doc_id}%")
                    param_idx += 1
                
                # Try doc_id without hash suffix
                if doc_id_without_hash and doc_id_without_hash != doc_id:
                    conditions.append(f"doc_id = ${param_idx}")
                    params.append(doc_id_without_hash)
                    param_idx += 1
                    conditions.append(f"doc_id LIKE ${param_idx}")
                    params.append(f"%{doc_id_without_hash}%")
                    param_idx += 1
                    
                    # Extract the DOI suffix pattern for flexible matching
                    # e.g., from doi_10.1016_S0140-6736_21_02098-5 extract "02098-5"
                    # This handles cases where DB has spaces or different formatting
                    import re
                    doi_suffix_match = re.search(r'(\d{4,5}-\d+)$', doc_id_without_hash)
                    if doi_suffix_match:
                        doi_suffix = doi_suffix_match.group(1)
                        conditions.append(f"doc_id LIKE ${param_idx}")
                        params.append(f"%{doi_suffix}%")
                        param_idx += 1
                        logger.info(f"Added DOI suffix search: {doi_suffix}")
                
                if extracted_doi:
                    conditions.append(f"doi = ${param_idx}")
                    params.append(extracted_doi)
                    param_idx += 1
                    # Also try with ILIKE for case-insensitive match
                    conditions.append(f"doi ILIKE ${param_idx}")
                    params.append(extracted_doi)
                    param_idx += 1
                
                if document_name_pattern:
                    # Search by document_name pattern
                    conditions.append(f"document_name ILIKE ${param_idx}")
                    params.append(f"%{document_name_pattern}%")
                    param_idx += 1
                    # Also try study_name
                    conditions.append(f"study_name ILIKE ${param_idx}")
                    params.append(f"%{document_name_pattern}%")
                    param_idx += 1
                
                if pmid:
                    conditions.append(f"pmid = ${param_idx}")
                    params.append(pmid)
                    param_idx += 1
                
                if doi:
                    conditions.append(f"doi = ${param_idx}")
                    params.append(doi)
                    param_idx += 1
                    conditions.append(f"doi ILIKE ${param_idx}")
                    params.append(doi)
                    param_idx += 1
                
                # Title-based search (fallback when DOI/PMID not available)
                if title:
                    # Clean title for search - remove common suffixes and normalize
                    clean_title = title.strip()
                    # Use full-text search on study_name
                    conditions.append(f"study_name ILIKE ${param_idx}")
                    params.append(f"%{clean_title[:100]}%")  # First 100 chars for matching
                    param_idx += 1
                    # Also try with key words from title
                    title_words = [w for w in clean_title.split() if len(w) > 4][:5]
                    if title_words:
                        # Build a pattern with key words
                        word_pattern = '%' + '%'.join(title_words) + '%'
                        conditions.append(f"study_name ILIKE ${param_idx}")
                        params.append(word_pattern)
                        param_idx += 1
                    logger.info(f"Added title search for: {clean_title[:50]}...")
                
                if not conditions:
                    return {'error': 'No identifier provided'}
                
                where_clause = " OR ".join(conditions)
                
                # Fetch main study record with ALL fields
                study_query = f"""
                    SELECT * FROM studies WHERE {where_clause} LIMIT 1
                """
                
                logger.info(f"Executing query with params: {params}")
                study = await conn.fetchrow(study_query, *params)
                
                if not study:
                    return {
                        'error': 'Study not found',
                        'doc_id': doc_id,
                        'pmid': pmid,
                        'doi': doi,
                        'extracted_doi': extracted_doi,
                        'search_params': params
                    }
                
                study_id = study['study_id']
                
                # Fetch all related data with DISTINCT to avoid duplicates
                biomarkers = await conn.fetch(
                    "SELECT DISTINCT ON (biomarker_name) * FROM biomarkers WHERE study_id = $1 ORDER BY biomarker_name, biomarker_order",
                    study_id
                )
                
                toxicity = await conn.fetch(
                    "SELECT DISTINCT ON (toxicity_type, grade) * FROM toxicity WHERE study_id = $1 ORDER BY toxicity_type, grade, toxicity_order",
                    study_id
                )
                
                study_arms = await conn.fetch(
                    "SELECT DISTINCT ON (arm_name) * FROM study_arms WHERE study_id = $1 ORDER BY arm_name, arm_order",
                    study_id
                )
                
                chemo_regimens = await conn.fetch(
                    "SELECT DISTINCT ON (regimen_name) * FROM chemotherapy_regimens WHERE study_id = $1 ORDER BY regimen_name, regimen_order",
                    study_id
                )
                
                radiation_details = await conn.fetch(
                    "SELECT DISTINCT ON (radiation_type, total_dose) * FROM radiation_details WHERE study_id = $1 ORDER BY radiation_type, total_dose, detail_order",
                    study_id
                )
                
                surgery_details = await conn.fetch(
                    "SELECT DISTINCT ON (surgery_type) * FROM surgery_details WHERE study_id = $1 ORDER BY surgery_type, detail_order",
                    study_id
                )
                
                stage_distribution = await conn.fetch(
                    "SELECT DISTINCT ON (stage_category) * FROM stage_distribution WHERE study_id = $1 ORDER BY stage_category",
                    study_id
                )
                
                staging_components = await conn.fetch(
                    "SELECT DISTINCT ON (component_name) * FROM staging_components WHERE study_id = $1 ORDER BY component_name",
                    study_id
                )
                
                inclusion_criteria = await conn.fetch(
                    "SELECT DISTINCT ON (criterion) * FROM inclusion_criteria WHERE study_id = $1 ORDER BY criterion, criterion_order",
                    study_id
                )
                
                exclusion_criteria = await conn.fetch(
                    "SELECT DISTINCT ON (criterion) * FROM exclusion_criteria WHERE study_id = $1 ORDER BY criterion, criterion_order",
                    study_id
                )
                
                dose_constraints = await conn.fetch(
                    "SELECT DISTINCT ON (organ_at_risk, constraint_type) * FROM dose_constraints WHERE study_id = $1 ORDER BY organ_at_risk, constraint_type, constraint_order",
                    study_id
                )
                
                # Build comprehensive response with all fields
                result = {
                    'study_id': study_id,
                    'doc_id': study.get('doc_id'),
                    'doi': study.get('doi'),
                    'pmid': study.get('pmid'),
                    'document_name': study.get('document_name'),
                    'title': study.get('study_name'),
                    
                    # Study Details (with evidence)
                    'study_details': self._build_field_dict(study, [
                        ('study_name', 'Study Name'),
                        ('protocol_name', 'Protocol Name'),
                        ('trial_registration_number', 'Trial Registration'),
                        ('publish_date', 'Publication Date'),
                        ('study_type', 'Study Type'),
                        ('study_phase', 'Study Phase'),
                        ('analysis_type', 'Analysis Type'),
                        ('number_of_patients', 'Number of Patients'),
                        ('citation_count', 'Times Cited'),
                        ('study_institution', 'Institution'),
                        ('country', 'Country'),
                    ]),
                    
                    # Patient Characteristics (with evidence)
                    'patient_characteristics': {
                        **self._build_field_dict(study, [
                            ('age_range', 'Age Range'),
                            ('median_age', 'Median Age'),
                            ('gender_distribution', 'Gender Distribution'),
                            ('race_ethnicity', 'Race/Ethnicity'),
                            ('performance_status', 'Performance Status'),
                        ]),
                        'inclusion_criteria': [
                            {
                                'criterion': row['criterion'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in inclusion_criteria
                        ] if inclusion_criteria else [],
                        'exclusion_criteria': [
                            {
                                'criterion': row['criterion'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in exclusion_criteria
                        ] if exclusion_criteria else []
                    },
                    
                    # Diagnosis (with evidence)
                    'diagnosis': self._build_field_dict(study, [
                        ('cancer_location', 'Cancer Location'),
                        ('cancer_type', 'Cancer Type'),
                        ('histopathologic_type', 'Histopathologic Type'),
                        ('tumor_grade', 'Tumor Grade'),
                        ('molecular_subtype', 'Molecular Subtype'),
                    ]),
                    
                    # Staging (with evidence)
                    'staging': {
                        **self._build_field_dict(study, [
                            ('staging_system_used', 'Staging System'),
                            ('risk_stratification', 'Risk Stratification'),
                            ('metastatic_status', 'Metastatic Status'),
                            ('extent_of_resection', 'Extent of Resection'),
                        ]),
                        'stage_distribution': [
                            {
                                'stage_category': row['stage_category'],
                                'number_of_patients': row['number_of_patients'],
                                'percentage': row['percentage'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in stage_distribution
                        ] if stage_distribution else [],
                        'staging_components': [
                            {
                                'component_name': row['component_name'],
                                'component_value': row['component_value'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in staging_components
                        ] if staging_components else []
                    },
                    
                    # Treatment
                    'treatment': {
                        'study_arms': [
                            {
                                'arm_name': row['arm_name'],
                                'description': row['description'],
                                'number_of_patients': row['number_of_patients'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in study_arms
                        ] if study_arms else [],
                        'chemotherapy_regimens': [
                            {
                                'regimen_name': row['regimen_name'],
                                'drugs': row['drugs'],
                                'dosage_info': row['dosage_info'],
                                'schedule_info': row['schedule_info'],
                                'number_of_cycles': row['number_of_cycles'],
                                'doxorubicin_dose': row['doxorubicin_dose'],
                                'doxorubicin_schedule': row['doxorubicin_schedule'],
                                'cyclophosphamide_dose': row['cyclophosphamide_dose'],
                                'cyclophosphamide_schedule': row['cyclophosphamide_schedule'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in chemo_regimens
                        ] if chemo_regimens else [],
                        'radiation_details': [
                            {
                                'radiation_type': row['radiation_type'],
                                'total_dose': row['total_dose'],
                                'fractionation': row['fractionation'],
                                'technique': row['technique'],
                                'target_volume': row['target_volume'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in radiation_details
                        ] if radiation_details else [],
                        'surgery_details': [
                            {
                                'surgery_type': row['surgery_type'],
                                'description': row['description'],
                                'evidence_quote': row['evidence_quote']
                            }
                            for row in surgery_details
                        ] if surgery_details else []
                    },
                    
                    # Outcomes (with evidence)
                    'outcomes': self._build_field_dict(study, [
                        ('primary_endpoint', 'Primary Endpoint'),
                        ('event_free_survival', 'Event-Free Survival'),
                        ('overall_survival', 'Overall Survival'),
                        ('progression_free_survival', 'Progression-Free Survival'),
                        ('disease_free_survival', 'Disease-Free Survival'),
                        ('local_control', 'Local Control'),
                        ('median_followup', 'Median Follow-up'),
                    ]),
                    
                    # Biomarkers
                    'biomarkers': [
                        {
                            'biomarker_name': row['biomarker_name'],
                            'biomarker_type': row['biomarker_type'],
                            'measurement_method': row['measurement_method'],
                            'baseline_value': row['baseline_value'],
                            'change_from_baseline': row['change_from_baseline'],
                            'significance': row['significance'],
                            'evidence_quote': row['evidence_quote']
                        }
                        for row in biomarkers
                    ] if biomarkers else [],
                    
                    # Toxicity
                    'toxicity': [
                        {
                            'toxicity_type': row['toxicity_type'],
                            'grade': row['grade'],
                            'frequency': row['frequency'],
                            'number_of_patients': row['number_of_patients'],
                            'timing': row['timing'],
                            'evidence_quote': row['evidence_quote']
                        }
                        for row in toxicity
                    ] if toxicity else [],
                    
                    # Dose Constraints
                    'dose_constraints': [
                        {
                            'organ_at_risk': row['organ_at_risk'],
                            'constraint_type': row['constraint_type'],
                            'dose_limit': row['dose_limit'],
                            'volume_limit': row['volume_limit'],
                            'evidence_quote': row['evidence_quote']
                        }
                        for row in dose_constraints
                    ] if dose_constraints else [],
                    
                    # Metadata
                    'abstract': study.get('abstract'),
                    'extraction_timestamp': study.get('extraction_timestamp'),
                    'processing_duration_seconds': study.get('processing_duration_seconds'),
                }
                
                # Cache the result
                self._cache[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.error(f"Error fetching study details: {e}", exc_info=True)
                return {
                    'error': str(e),
                    'doc_id': doc_id,
                    'pmid': pmid,
                    'doi': doi
                }
    
    def _build_field_dict(self, record, field_mapping: list) -> dict:
        """
        Build dictionary of fields with evidence quotes
        
        Args:
            record: Database record
            field_mapping: List of tuples (db_column, display_label)
            
        Returns:
            Dictionary with value and evidence for each field
        """
        result = {}
        for db_col, label in field_mapping:
            value = record.get(db_col)
            evidence = record.get(f"{db_col}_evidence")
            
            if value is not None:
                result[db_col] = {
                    'label': label,
                    'value': value,
                    'evidence_quote': evidence
                }
        
        return result
    
    async def get_eligibility_criteria_by_doc_ids(
        self,
        doc_ids: List[str],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Batch-fetch inclusion + exclusion criteria for a list of doc_ids.

        Returns: ``{original_doc_id: {"inclusion": [...], "exclusion": [...]}}``
        for every doc_id that resolves to a row in the studies table.
        Keys in the returned dict match the **input** doc_ids exactly
        (not the normalised forms), so callers can pair results with
        their Qdrant-side records without further work.

        Doc-id resolution. Qdrant doc_ids carry an `_<8-hex-hash>`
        suffix (`doi_10.1200_jco.2007.15.0102_e07e6c6e`) that
        `display-study-details.studies.doc_id` does NOT (it's stored
        as `doi_10.1200_jco.2007.15.0102`). The single-doc lookup
        `get_study_details()` strips the hash before matching; this
        batch method does the same to avoid silent zero-match returns.
        Also handles the non-DOI form (e.g. `maghami-et-al-2020-...
        _08609f6a`) under the same regex `_[a-f0-9]{6,}$`.

        Used by patient_eligibility_boost_service.check_patient_eligibility_for_studies
        to feed structured inclusion/exclusion criteria to the
        eligibility LLM prompt as a complement to the chunk-derived
        text. The LLM verdicts study_exclusions_violated against the
        exclusion list and uses the inclusion list as additional
        context for disease_status and surgical_candidacy.
        """
        if not doc_ids:
            return {}

        import re
        _HASH_SUFFIX = re.compile(r"_[a-f0-9]{6,}$")

        def _normalize(doc_id: str) -> str:
            return _HASH_SUFFIX.sub("", doc_id)

        # Map every normalised form back to the originals that produced
        # it (one normalised doc_id may collide for multiple originals
        # in pathological cases, so keep a list).
        normalised_to_originals: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            normalised_to_originals.setdefault(_normalize(doc_id), []).append(doc_id)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    s.doc_id,
                    COALESCE(
                        array_agg(DISTINCT ic.criterion)
                            FILTER (WHERE ic.criterion IS NOT NULL),
                        ARRAY[]::TEXT[]
                    ) AS inclusion,
                    COALESCE(
                        array_agg(DISTINCT ec.criterion)
                            FILTER (WHERE ec.criterion IS NOT NULL),
                        ARRAY[]::TEXT[]
                    ) AS exclusion
                FROM studies s
                LEFT JOIN inclusion_criteria ic ON ic.study_id = s.study_id
                LEFT JOIN exclusion_criteria ec ON ec.study_id = s.study_id
                WHERE s.doc_id = ANY($1::TEXT[])
                GROUP BY s.doc_id
                """,
                list(normalised_to_originals.keys()),
            )

        result: Dict[str, Dict[str, List[str]]] = {}
        for row in rows:
            payload = {
                "inclusion": list(row["inclusion"] or []),
                "exclusion": list(row["exclusion"] or []),
            }
            for original in normalised_to_originals.get(row["doc_id"], []):
                result[original] = payload
        return result

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache = {}
        logger.info("PostgresStudyDetailsService cache cleared")
    
    async def search_studies_by_profile(
        self,
        cancer_type: str = None,
        anatomical_site: str = None,
        histology: str = None,
        stage: str = None,
        limit: int = 20
    ) -> list:
        """
        Search studies by patient profile characteristics.
        
        Args:
            cancer_type: Type of cancer (e.g., SCC, adenocarcinoma)
            anatomical_site: Anatomical location (e.g., maxilla, oral cavity)
            histology: Histology type (e.g., squamous, poorly differentiated)
            stage: Cancer stage (e.g., IV, III)
            limit: Maximum number of results
            
        Returns:
            List of matching studies with basic info
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            try:
                conditions = []
                params = []
                param_idx = 1
                
                # Build search conditions with flexible matching
                if anatomical_site:
                    # Map anatomical sites to search terms
                    site_search_terms = self._get_site_search_terms(anatomical_site)
                    site_conditions = []
                    for term in site_search_terms:
                        site_conditions.append(f"cancer_location ILIKE ${param_idx}")
                        params.append(f"%{term}%")
                        param_idx += 1
                    if site_conditions:
                        conditions.append(f"({' OR '.join(site_conditions)})")
                
                if cancer_type:
                    # Map cancer types to search terms
                    type_search_terms = self._get_cancer_type_search_terms(cancer_type)
                    type_conditions = []
                    for term in type_search_terms:
                        type_conditions.append(f"(cancer_type ILIKE ${param_idx} OR histopathologic_type ILIKE ${param_idx})")
                        params.append(f"%{term}%")
                        param_idx += 1
                    if type_conditions:
                        conditions.append(f"({' OR '.join(type_conditions)})")
                
                if histology:
                    conditions.append(f"histopathologic_type ILIKE ${param_idx}")
                    params.append(f"%{histology}%")
                    param_idx += 1
                
                # Build query
                if conditions:
                    where_clause = " AND ".join(conditions)
                    query = f"""
                        SELECT 
                            study_id,
                            doc_id,
                            doi,
                            pmid,
                            study_name,
                            cancer_type,
                            cancer_location,
                            histopathologic_type,
                            study_type,
                            number_of_patients,
                            primary_endpoint,
                            overall_survival,
                            progression_free_survival,
                            disease_free_survival,
                            publish_date
                        FROM studies 
                        WHERE {where_clause}
                        ORDER BY publish_date DESC NULLS LAST
                        LIMIT ${param_idx}
                    """
                    params.append(limit)
                else:
                    # No conditions - return empty
                    return []
                
                logger.info(f"[PostgresSearch] Query: {query[:200]}...")
                logger.info(f"[PostgresSearch] Params: {params}")
                
                rows = await conn.fetch(query, *params)
                
                results = []
                for row in rows:
                    results.append({
                        'study_id': row['study_id'],
                        'doc_id': row['doc_id'],
                        'doi': row['doi'],
                        'pmid': row['pmid'],
                        'title': row['study_name'],
                        'cancer_type': row['cancer_type'],
                        'cancer_location': row['cancer_location'],
                        'histology': row['histopathologic_type'],
                        'study_type': row['study_type'],
                        'number_of_patients': row['number_of_patients'],
                        'primary_endpoint': row['primary_endpoint'],
                        'overall_survival': row['overall_survival'],
                        'progression_free_survival': row['progression_free_survival'],
                        'disease_free_survival': row['disease_free_survival'],
                        'publish_date': str(row['publish_date']) if row['publish_date'] else None,
                        'source': 'postgres'
                    })
                
                logger.info(f"[PostgresSearch] Found {len(results)} matching studies")
                return results
                
            except Exception as e:
                logger.error(f"Error searching studies: {e}", exc_info=True)
                return []
    
    def _get_site_search_terms(self, anatomical_site: str) -> list:
        """Get search terms for anatomical site with broad region fallback."""
        site = anatomical_site.lower().strip()
        
        # Map specific sites to search terms (includes parent region as fallback)
        site_mappings = {
            # Head and neck — specific sites
            'maxilla': ['maxilla', 'oral cavity', 'oral', 'head and neck', 'upper jaw'],
            'mandible': ['mandible', 'oral cavity', 'oral', 'head and neck', 'lower jaw'],
            'oral cavity': ['oral cavity', 'oral', 'head and neck', 'mouth'],
            'tongue': ['tongue', 'oral cavity', 'oral', 'head and neck'],
            'base of tongue': ['base of tongue', 'tongue', 'oropharynx', 'head and neck'],
            'gingiva': ['gingiva', 'oral cavity', 'oral', 'head and neck', 'gum'],
            'hard palate': ['palate', 'oral cavity', 'oral', 'head and neck'],
            'soft palate': ['palate', 'oral cavity', 'oropharynx', 'head and neck'],
            'buccal mucosa': ['buccal', 'oral cavity', 'oral', 'head and neck', 'cheek'],
            'floor of mouth': ['floor of mouth', 'oral cavity', 'oral', 'head and neck'],
            'oropharynx': ['oropharynx', 'pharynx', 'head and neck'],
            'nasopharynx': ['nasopharynx', 'pharynx', 'head and neck'],
            'hypopharynx': ['hypopharynx', 'pharynx', 'head and neck', 'pyriform'],
            'larynx': ['larynx', 'laryngeal', 'head and neck'],
            'glottis': ['glottis', 'glottic', 'larynx', 'head and neck'],
            'supraglottis': ['supraglottis', 'supraglottic', 'larynx', 'head and neck'],
            'tonsil': ['tonsil', 'tonsillar', 'oropharynx', 'head and neck'],
            'pharynx': ['pharynx', 'pharyngeal', 'head and neck'],
            'pyriform sinus': ['pyriform', 'hypopharynx', 'pharynx', 'head and neck'],
            'neck': ['neck', 'head and neck', 'cervical lymph'],
            'salivary gland': ['salivary', 'parotid', 'submandibular', 'head and neck'],
            'parotid': ['parotid', 'salivary', 'head and neck'],
            'nasal cavity': ['nasal', 'sinonasal', 'head and neck'],
            'sinus': ['sinus', 'sinonasal', 'paranasal', 'head and neck'],
            'thyroid': ['thyroid', 'head and neck'],
            'lip': ['lip', 'oral', 'head and neck'],
            # Skin
            'skin': ['skin', 'cutaneous', 'dermal'],
            # Lung
            'lung': ['lung', 'pulmonary', 'thoracic'],
            'bronchus': ['bronchus', 'bronchial', 'lung', 'pulmonary'],
            # Breast
            'breast': ['breast', 'mammary'],
            # GYN
            'cervix': ['cervix', 'cervical'],
            'uterus': ['uterus', 'uterine', 'endometrial'],
            'endometrium': ['endometrium', 'endometrial', 'uterine'],
            'ovary': ['ovary', 'ovarian'],
            'vulva': ['vulva', 'vulvar'],
            'vagina': ['vagina', 'vaginal'],
            'fallopian tube': ['fallopian', 'tubal', 'ovarian'],
            # GI
            'esophagus': ['esophagus', 'esophageal', 'gastroesophageal'],
            'stomach': ['stomach', 'gastric'],
            'liver': ['liver', 'hepatic', 'hepatocellular'],
            'pancreas': ['pancreas', 'pancreatic'],
            'colon': ['colon', 'colonic', 'colorectal'],
            'rectum': ['rectal', 'rectum', 'colorectal'],
            'anus': ['anal', 'anus'],
            'duodenum': ['duodenum', 'duodenal', 'small bowel'],
            'small bowel': ['small bowel', 'small intestine', 'intestinal'],
            'gallbladder': ['gallbladder', 'biliary'],
            'bile duct': ['bile duct', 'biliary', 'cholangiocarcinoma'],
            'appendix': ['appendix', 'appendiceal'],
            # GU
            'bladder': ['bladder', 'urothelial', 'vesical'],
            'kidney': ['kidney', 'renal'],
            'prostate': ['prostate', 'prostatic'],
            'ureter': ['ureter', 'ureteral', 'urothelial'],
            'testis': ['testis', 'testicular'],
            'penis': ['penis', 'penile'],
            # CNS
            'brain': ['brain', 'cerebral', 'intracranial'],
            'spinal cord': ['spinal cord', 'spinal', 'intradural'],
        }
        
        if site in site_mappings:
            return site_mappings[site]
        
        # Broad keyword fallback — match partial terms to parent region search terms
        broad_region_fallback = {
            # H&N keywords
            "oral": ['oral', 'oral cavity', 'head and neck'],
            "pharyn": ['pharynx', 'pharyngeal', 'head and neck'],
            "laryng": ['larynx', 'laryngeal', 'head and neck'],
            "glott": ['glottic', 'larynx', 'head and neck'],
            "tonsi": ['tonsil', 'oropharynx', 'head and neck'],
            "palat": ['palate', 'oral cavity', 'head and neck'],
            "mandib": ['mandible', 'oral cavity', 'head and neck'],
            "maxill": ['maxilla', 'oral cavity', 'head and neck'],
            "bucca": ['buccal', 'oral cavity', 'head and neck'],
            "gingi": ['gingiva', 'oral cavity', 'head and neck'],
            "saliv": ['salivary', 'head and neck'],
            "parot": ['parotid', 'salivary', 'head and neck'],
            "sinus": ['sinus', 'sinonasal', 'head and neck'],
            "nasal": ['nasal', 'sinonasal', 'head and neck'],
            "thyroid": ['thyroid', 'head and neck'],
            "mouth": ['oral cavity', 'oral', 'head and neck'],
            "neck": ['neck', 'head and neck'],
            "head": ['head and neck'],
            # Skin
            "cutan": ['cutaneous', 'skin'],
            "melano": ['melanoma', 'skin', 'cutaneous'],
            # Lung
            "pulmon": ['pulmonary', 'lung'],
            "bronch": ['bronchial', 'lung'],
            "thorac": ['thoracic', 'lung'],
            # Breast
            "mamm": ['mammary', 'breast'],
            # GYN
            "cervic": ['cervical', 'cervix'],
            "uterin": ['uterine', 'uterus'],
            "ovari": ['ovarian', 'ovary'],
            "endometri": ['endometrial', 'uterine'],
            "vulv": ['vulvar', 'vulva'],
            "vagin": ['vaginal', 'vagina'],
            "fallop": ['fallopian', 'ovarian'],
            # GI
            "esophag": ['esophageal', 'esophagus'],
            "gastric": ['gastric', 'stomach'],
            "hepat": ['hepatic', 'liver'],
            "pancrea": ['pancreatic', 'pancreas'],
            "colorect": ['colorectal', 'colon', 'rectum'],
            "rectal": ['rectal', 'rectum', 'colorectal'],
            "anal": ['anal', 'anus'],
            "duoden": ['duodenal', 'small bowel'],
            "bowel": ['bowel', 'intestinal'],
            "intestin": ['intestinal', 'bowel'],
            "biliary": ['biliary', 'bile duct'],
            "gallbladd": ['gallbladder', 'biliary'],
            # GU
            "renal": ['renal', 'kidney'],
            "urothel": ['urothelial', 'bladder'],
            "prostat": ['prostate', 'prostatic'],
            "testic": ['testicular', 'testis'],
            "penile": ['penile', 'penis'],
            # CNS
            "cerebr": ['cerebral', 'brain', 'intracranial'],
            "gliom": ['glioma', 'brain'],
            "glioblast": ['glioblastoma', 'brain'],
            "intracran": ['intracranial', 'brain'],
            "spinal": ['spinal', 'spinal cord'],
            "meningi": ['meningioma', 'brain', 'intracranial'],
        }
        
        for keyword, terms in broad_region_fallback.items():
            if keyword in site:
                print(f"[PostgresSearch] Broad keyword fallback: '{site}' matched '{keyword}'")
                return terms
        
        # Final fallback — use the raw input
        return [site]
    
    def _get_cancer_type_search_terms(self, cancer_type: str) -> list:
        """Get search terms for cancer type with synonym expansion."""
        ctype = cancer_type.lower().strip()
        
        # Map abbreviations and variants
        type_mappings = {
            'scc': ['squamous', 'squamous cell', 'SCC'],
            'squamous cell carcinoma': ['squamous', 'squamous cell', 'SCC'],
            'adenocarcinoma': ['adenocarcinoma', 'adeno'],
            'nsclc': ['non-small cell', 'NSCLC', 'non small cell'],
            'sclc': ['small cell', 'SCLC'],
            'melanoma': ['melanoma', 'cutaneous melanoma'],
            'glioma': ['glioma', 'astrocytoma'],
            'glioblastoma': ['glioblastoma', 'GBM', 'glioma'],
            'breast': ['breast', 'mammary'],
            'lung': ['lung', 'pulmonary', 'NSCLC', 'SCLC'],
            'prostate': ['prostate', 'prostatic'],
            'colorectal': ['colorectal', 'colon', 'rectal'],
            'rectal': ['rectal', 'rectum', 'colorectal'],
            'cervical': ['cervical', 'cervix'],
            'ovarian': ['ovarian', 'ovary'],
            'endometrial': ['endometrial', 'endometrium', 'uterine'],
            'bladder': ['bladder', 'urothelial'],
            'renal': ['renal', 'kidney', 'renal cell'],
            'renal cell carcinoma': ['renal cell', 'renal', 'kidney', 'RCC'],
            'hepatocellular': ['hepatocellular', 'HCC', 'liver'],
            'pancreatic': ['pancreatic', 'pancreas'],
            'esophageal': ['esophageal', 'esophagus'],
            'gastric': ['gastric', 'stomach'],
            'cholangiocarcinoma': ['cholangiocarcinoma', 'bile duct', 'biliary'],
            'meningioma': ['meningioma', 'meningeal'],
            'epidermoid': ['epidermoid', 'squamous', 'SCC'],
        }

        return type_mappings.get(ctype, [ctype])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
#
# The service holds a connection pool. Instantiating a fresh service per
# request creates a new pool each time and never closes the old ones,
# exhausting `display-study-details` `max_connections` after a handful of
# eligibility checks. Use this getter from every caller (eligibility
# service, query routes, etc.) so a single pool is shared.

_singleton_service: Optional["PostgresStudyDetailsService"] = None


def get_postgres_study_details_service() -> "PostgresStudyDetailsService":
    """Return the shared PostgresStudyDetailsService singleton.

    Lazily instantiates with credentials from `settings` on first call.
    All eligibility / details code paths should use this getter rather
    than constructing fresh `PostgresStudyDetailsService()` instances
    per request — the latter exhausts the Postgres connection pool.
    """
    global _singleton_service
    if _singleton_service is None:
        _singleton_service = PostgresStudyDetailsService(
            pg_host=settings.postgres_host,
            pg_port=settings.postgres_port,
            pg_user=settings.postgres_user,
            pg_password=settings.postgres_password,
            pg_database=settings.postgres_database,
        )
    return _singleton_service
