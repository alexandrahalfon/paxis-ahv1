"""
Preferences Filter Service

Applies user preferences as filters and boosts to search results.
- Pre-filters doc_ids from PostgreSQL study-profiles database
- Filters are applied post-Qdrant retrieval (before reranking)
- Sort preferences are applied as score boosts (after reranking)
"""

from typing import Dict, List, Any, Optional, Set
from src.api.services.account_db import get_account_db


async def get_valid_doc_ids_from_preferences(preferences: Dict[str, Any]) -> Optional[Set[str]]:
    """
    Query the study-profiles PostgreSQL database to get doc_ids that match
    user preference filters (min_patients, study_phase, etc.).
    
    Returns:
        Set of valid doc_ids, or None if no filters require PostgreSQL lookup
    """
    if not preferences:
        return None
    
    # Check if any filters require PostgreSQL lookup
    needs_pg_lookup = any([
        preferences.get('min_patients'),
        preferences.get('max_patients'),
        preferences.get('study_phases'),
        preferences.get('study_types'),
        preferences.get('min_publication_year'),
        preferences.get('max_publication_year'),
    ])
    
    if not needs_pg_lookup:
        return None
    
    conn = None
    try:
        import asyncpg
        from src.core.config import settings
        
        # Create a single connection (not a pool) with short timeout
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database="study-profiles",
            timeout=5,
            command_timeout=10,
        )
        
        # Build dynamic query based on preferences
        conditions = []
        params = []
        param_idx = 1
        
        # Min patients filter - strictly filter to studies with known patient count >= min
        if preferences.get('min_patients'):
            conditions.append(f"s.number_of_patients >= ${param_idx}")
            params.append(preferences['min_patients'])
            param_idx += 1
        
        # Max patients filter
        if preferences.get('max_patients'):
            conditions.append(f"s.number_of_patients <= ${param_idx}")
            params.append(preferences['max_patients'])
            param_idx += 1
        
        # Study phase filter
        if preferences.get('study_phases'):
            phase_conditions = []
            for phase in preferences['study_phases']:
                phase_conditions.append(f"s.study_phase ILIKE ${param_idx}")
                params.append(f"%{phase}%")
                param_idx += 1
            if phase_conditions:
                conditions.append(f"({' OR '.join(phase_conditions)})")
        
        # Study type filter
        if preferences.get('study_types'):
            type_conditions = []
            for stype in preferences['study_types']:
                type_conditions.append(f"s.study_type ILIKE ${param_idx}")
                params.append(f"%{stype}%")
                param_idx += 1
            if type_conditions:
                conditions.append(f"({' OR '.join(type_conditions)})")
        
        # Publication year filter
        if preferences.get('min_publication_year'):
            conditions.append(f"EXTRACT(YEAR FROM s.publish_date) >= ${param_idx}")
            params.append(preferences['min_publication_year'])
            param_idx += 1
        
        if preferences.get('max_publication_year'):
            conditions.append(f"EXTRACT(YEAR FROM s.publish_date) <= ${param_idx}")
            params.append(preferences['max_publication_year'])
            param_idx += 1
        
        if not conditions:
            return None
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT DISTINCT s.doc_id
            FROM studies s
            WHERE s.doc_id IS NOT NULL
              AND {where_clause}
        """
        
        print(f"[PreferencesFilter] PostgreSQL query with {len(conditions)} conditions")
        
        rows = await conn.fetch(query, *params)
        
        valid_doc_ids = {row['doc_id'] for row in rows if row['doc_id']}
        
        print(f"[PreferencesFilter] Found {len(valid_doc_ids)} valid doc_ids from PostgreSQL")
        
        return valid_doc_ids
            
    except Exception as e:
        print(f"[PreferencesFilter] Error querying PostgreSQL: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Always close the connection
        if conn:
            await conn.close()


def get_valid_doc_ids_sync(preferences: Dict[str, Any]) -> Optional[Set[str]]:
    """
    Synchronous wrapper for get_valid_doc_ids_from_preferences.
    """
    if not preferences:
        return None
    
    try:
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(get_valid_doc_ids_from_preferences(preferences)))
                return future.result(timeout=15)
        except RuntimeError:
            # No running loop - safe to use asyncio.run directly
            return asyncio.run(get_valid_doc_ids_from_preferences(preferences))
        
    except Exception as e:
        print(f"[PreferencesFilter] Error in sync wrapper: {e}")
        import traceback
        traceback.print_exc()
        return None


async def get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user preferences from database.
    Returns None if no preferences or filters not active.
    """
    if not user_id:
        return None
    
    try:
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    study_types, study_phases, cancer_types,
                    min_patients, max_patients, analysis_types,
                    treatment_modalities, countries, institutions,
                    race_ethnicities, include_unknown_race,
                    sort_by, sort_order, filters_active,
                    min_publication_year, max_publication_year,
                    require_peer_reviewed, min_followup_months,
                    required_outcomes
                FROM user_preferences
                WHERE user_id = $1
            """, user_id)
            
            if not row:
                return None
            
            # Return None if filters not active
            if not row.get('filters_active', True):
                return None
            
            return {
                'study_types': row['study_types'] or [],
                'study_phases': row['study_phases'] or [],
                'cancer_types': row['cancer_types'] or [],
                'min_patients': row['min_patients'],
                'max_patients': row['max_patients'],
                'analysis_types': row['analysis_types'] or [],
                'treatment_modalities': row['treatment_modalities'] or [],
                'countries': row['countries'] or [],
                'institutions': row['institutions'] or [],
                'race_ethnicities': row['race_ethnicities'] or [],
                'include_unknown_race': row.get('include_unknown_race', True),
                'sort_by': row['sort_by'] or 'relevance',
                'sort_order': row['sort_order'] or 'desc',
                'min_publication_year': row.get('min_publication_year'),
                'max_publication_year': row.get('max_publication_year'),
                'require_peer_reviewed': row.get('require_peer_reviewed', False),
                'min_followup_months': row.get('min_followup_months'),
                'required_outcomes': row.get('required_outcomes') or [],
            }
    except Exception as e:
        print(f"[PreferencesFilter] Error fetching preferences: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_user_preferences_sync(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Synchronous version of get_user_preferences.
    Creates its own connection to avoid pool conflicts when called from sync context.
    Uses nest_asyncio to handle nested event loops.
    """
    if not user_id:
        return None
    
    try:
        import asyncpg
        import asyncio
        from src.core.config import settings
        
        async def _fetch():
            # Create a single connection instead of using the pool
            conn = await asyncpg.connect(
                # host/port/user now fall back too, not just password — same
                # fix as account_db.py / patient_db.py. host previously had
                # no fallback and was hardcoded to a stale IP in config.py.
                host=settings.cache_postgres_host or settings.postgres_host,
                port=settings.cache_postgres_port or settings.postgres_port,
                user=settings.cache_postgres_user or settings.postgres_user,
                password=settings.cache_postgres_password or settings.postgres_password,
                database=settings.cache_postgres_database,
                timeout=10,
            )
            try:
                row = await conn.fetchrow("""
                    SELECT 
                        study_types, study_phases, cancer_types,
                        min_patients, max_patients, analysis_types,
                        treatment_modalities, countries, institutions,
                        race_ethnicities, include_unknown_race,
                        sort_by, sort_order, filters_active,
                        min_publication_year, max_publication_year,
                        require_peer_reviewed, min_followup_months,
                        required_outcomes
                    FROM user_preferences
                    WHERE user_id = $1
                """, user_id)
                
                if not row:
                    return None
                
                # Return None if filters not active
                if not row.get('filters_active', True):
                    return None
                
                return {
                    'study_types': row['study_types'] or [],
                    'study_phases': row['study_phases'] or [],
                    'cancer_types': row['cancer_types'] or [],
                    'min_patients': row['min_patients'],
                    'max_patients': row['max_patients'],
                    'analysis_types': row['analysis_types'] or [],
                    'treatment_modalities': row['treatment_modalities'] or [],
                    'countries': row['countries'] or [],
                    'institutions': row['institutions'] or [],
                    'race_ethnicities': row['race_ethnicities'] or [],
                    'include_unknown_race': row.get('include_unknown_race', True),
                    'sort_by': row['sort_by'] or 'relevance',
                    'sort_order': row['sort_order'] or 'desc',
                    'min_publication_year': row.get('min_publication_year'),
                    'max_publication_year': row.get('max_publication_year'),
                    'require_peer_reviewed': row.get('require_peer_reviewed', False),
                    'min_followup_months': row.get('min_followup_months'),
                    'required_outcomes': row.get('required_outcomes') or [],
                }
            finally:
                await conn.close()
        
        # Handle running in async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(_fetch()))
                return future.result(timeout=10)
        except RuntimeError:
            # No running loop - safe to use asyncio.run directly
            return asyncio.run(_fetch())
        
    except Exception as e:
        print(f"[PreferencesFilter] Error fetching preferences (sync): {e}")
        import traceback
        traceback.print_exc()
        return None


async def get_citation_counts_for_docs(doc_ids: List[str]) -> Dict[str, int]:
    """
    Fetch citation counts from PostgreSQL for a list of doc_ids.
    Returns a dict mapping doc_id -> citation_count.
    """
    if not doc_ids:
        return {}
    
    try:
        from src.core.config import settings
        import asyncpg
        
        pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database,
            min_size=1,
            max_size=5,
            timeout=10
        )
        
        async with pool.acquire() as conn:
            # Query by doc_id (which may be DOI or other identifier)
            rows = await conn.fetch("""
                SELECT doc_id, citation_count
                FROM studies
                WHERE doc_id = ANY($1) AND citation_count IS NOT NULL
            """, doc_ids)
            
            result = {row['doc_id']: row['citation_count'] for row in rows}
            
            # Also try matching by DOI if doc_id didn't match
            unmatched = [d for d in doc_ids if d not in result]
            if unmatched:
                doi_rows = await conn.fetch("""
                    SELECT doi, citation_count
                    FROM studies
                    WHERE doi = ANY($1) AND citation_count IS NOT NULL
                """, unmatched)
                for row in doi_rows:
                    result[row['doi']] = row['citation_count']
        
        await pool.close()
        return result
        
    except Exception as e:
        print(f"[PreferencesFilter] Error fetching citation counts: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_citation_counts_sync(doc_ids: List[str]) -> Dict[str, int]:
    """
    Synchronous wrapper for get_citation_counts_for_docs.
    Handles running async code in sync context.
    """
    if not doc_ids:
        return {}
    
    try:
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_citation_counts_for_docs(doc_ids))
                return future.result(timeout=10)
        except RuntimeError:
            # No running loop, we can use asyncio.run directly
            return asyncio.run(get_citation_counts_for_docs(doc_ids))
    except Exception as e:
        print(f"[PreferencesFilter] Error in sync wrapper: {e}")
        return {}


def apply_preference_filters(
    candidates: List[Dict[str, Any]],
    preferences: Dict[str, Any],
    valid_doc_ids: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Apply user preference filters to candidate chunks.
    Filters are applied BEFORE reranking to reduce the pool.
    
    Args:
        candidates: List of candidate chunks from Qdrant
        preferences: User preferences dict
        valid_doc_ids: Optional set of doc_ids that passed PostgreSQL filters
    
    Returns filtered list of candidates.
    """
    if not preferences:
        return candidates
    
    filtered = []
    filter_stats = {'total': len(candidates), 'passed': 0, 'filtered_out': 0, 'pg_filtered': 0}
    
    for cand in candidates:
        payload = cand.get('payload', {})
        doc_meta = payload.get('doc_meta', {})
        doc_id = payload.get('doc_id')
        
        # First check PostgreSQL filter (min_patients, study_phase, etc.)
        if valid_doc_ids is not None:
            if not doc_id or doc_id not in valid_doc_ids:
                filter_stats['filtered_out'] += 1
                filter_stats['pg_filtered'] += 1
                continue
        
        # Then check other filters (treatment modality, etc.)
        if not _passes_filters(payload, doc_meta, preferences):
            filter_stats['filtered_out'] += 1
            continue
        
        filtered.append(cand)
        filter_stats['passed'] += 1
    
    # Log filter stats
    if filter_stats['filtered_out'] > 0:
        pg_info = f" (PostgreSQL: {filter_stats['pg_filtered']})" if filter_stats['pg_filtered'] > 0 else ""
        print(f"[PreferencesFilter] Filtered: {filter_stats['passed']}/{filter_stats['total']} passed{pg_info}")
    
    return filtered


def _passes_filters(payload: Dict, doc_meta: Dict, prefs: Dict) -> bool:
    """
    Check if a candidate passes preference filters.
    
    Note: min_patients, max_patients, study_phases, study_types, and publication_year
    are now filtered via PostgreSQL (see get_valid_doc_ids_from_preferences).
    This function handles remaining filters that work on Qdrant payload data.
    """
    
    # Cancer type filter (still useful for Qdrant payload matching)
    if prefs.get('cancer_types'):
        cancer_type = (
            doc_meta.get('cancer_type') or 
            doc_meta.get('disease') or 
            payload.get('cancer_type') or ''
        ).lower()
        category = (payload.get('category') or '').lower()
        combined = f"{cancer_type} {category}"
        if not any(ct.lower() in combined for ct in prefs['cancer_types']):
            return False
    
    # Country filter
    if prefs.get('countries'):
        country = (doc_meta.get('country') or doc_meta.get('countries') or payload.get('country') or '').lower()
        institution = (doc_meta.get('institution') or '').lower()
        combined = f"{country} {institution}"
        if not any(c.lower() in combined for c in prefs['countries']):
            return False
    
    # Institution filter
    if prefs.get('institutions'):
        institution = (doc_meta.get('institution') or doc_meta.get('institutions') or '').lower()
        if not any(inst.lower() in institution for inst in prefs['institutions']):
            return False
    
    # Treatment modality filter
    if prefs.get('treatment_modalities'):
        # Check multiple fields for treatment info
        treatment = (
            doc_meta.get('treatment') or 
            doc_meta.get('treatment_modality') or 
            doc_meta.get('intervention') or
            payload.get('treatment') or 
            payload.get('treatment_modality') or
            ''
        ).lower()
        text = payload.get('text', '').lower()
        category = (payload.get('category') or '').lower()
        
        # Map filter values to search terms
        modality_keywords = {
            'radiation': ['radiation', 'radiotherapy', 'rt', 'imrt', 'vmat', 'proton', 'photon', 'external beam', 'ebrt'],
            'chemotherapy': ['chemotherapy', 'chemo', 'cisplatin', 'carboplatin', 'paclitaxel', 'docetaxel', 'gemcitabine', 'fluorouracil', '5-fu'],
            'surgery': ['surgery', 'surgical', 'resection', 'excision', 'mastectomy', 'lobectomy', 'prostatectomy'],
            'immunotherapy': ['immunotherapy', 'immune checkpoint', 'pd-1', 'pd-l1', 'pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab', 'ipilimumab'],
            'targeted_therapy': ['targeted therapy', 'targeted', 'tyrosine kinase', 'tki', 'egfr', 'her2', 'alk', 'braf', 'mek'],
            'hormone_therapy': ['hormone therapy', 'hormonal', 'endocrine', 'tamoxifen', 'letrozole', 'anastrozole', 'adt', 'androgen deprivation'],
            'brachytherapy': ['brachytherapy', 'brachy', 'hdr', 'ldr', 'interstitial', 'intracavitary'],
            'sbrt': ['sbrt', 'srs', 'stereotactic', 'radiosurgery', 'cyberknife', 'gamma knife'],
        }
        
        combined_text = f"{treatment} {text} {category}"
        
        # Check if any selected modality matches
        matched = False
        for modality in prefs['treatment_modalities']:
            keywords = modality_keywords.get(modality.lower(), [modality.lower()])
            if any(kw in combined_text for kw in keywords):
                matched = True
                break
        
        if not matched:
            return False
    
    return True


def apply_sort_boost(
    candidates: List[Dict[str, Any]],
    preferences: Dict[str, Any],
    citation_counts: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    """
    Apply sort preference as a score boost AFTER reranking.
    This adjusts final scores based on user's sort preference.
    
    Sort options:
    - relevance: No boost (default)
    - population: Boost studies with more patients
    - date: Boost newer studies
    - citations: Boost highly cited studies (requires citation_counts dict)
    - outcomes: Boost studies with strong outcomes
    - patient_relevance: Boost studies matching patient profile
    
    Args:
        candidates: List of candidate chunks
        preferences: User preferences dict
        citation_counts: Optional dict mapping doc_id/doi -> citation_count
    """
    if not preferences or not candidates:
        return candidates
    
    sort_by = preferences.get('sort_by', 'relevance')
    
    if sort_by == 'relevance':
        return candidates  # No boost needed
    
    print(f"[PreferencesFilter] Applying sort boost: {sort_by}")
    
    # Apply boost based on sort preference
    for cand in candidates:
        payload = cand.get('payload', {})
        doc_meta = payload.get('doc_meta', {})
        
        boost = _calculate_sort_boost(payload, doc_meta, sort_by, citation_counts)
        
        # Apply boost to rerank score (multiplicative)
        if 'score_rerank' in cand:
            original = cand['score_rerank']
            cand['score_rerank'] = original * (1 + boost)
            if boost > 0:
                cand['_sort_boost'] = boost  # Track for debugging
        elif 'score_dense' in cand:
            original = cand['score_dense']
            cand['score_dense'] = original * (1 + boost)
            if boost > 0:
                cand['_sort_boost'] = boost
    
    # Re-sort by boosted score
    if candidates:
        score_key = 'score_rerank' if 'score_rerank' in candidates[0] else 'score_dense'
        candidates.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    
    # Log boost stats
    boosted = [c for c in candidates if c.get('_sort_boost', 0) > 0]
    if boosted:
        print(f"[PreferencesFilter] Boosted {len(boosted)}/{len(candidates)} candidates")
    
    return candidates


def _calculate_sort_boost(
    payload: Dict, 
    doc_meta: Dict, 
    sort_by: str,
    citation_counts: Optional[Dict[str, int]] = None
) -> float:
    """Calculate boost factor based on sort preference (0.0 to 0.5)."""
    
    if sort_by == 'population':
        # Boost larger studies
        num_patients = doc_meta.get('num_patients') or doc_meta.get('n_patients') or payload.get('num_patients')
        if num_patients:
            try:
                n = int(str(num_patients).replace(',', '').split()[0])
                # Scale: 0-100 patients = 0 boost, 1000+ = 0.5 boost
                return min(0.5, n / 2000)
            except (ValueError, TypeError):
                pass
        return 0.0
    
    elif sort_by == 'date':
        # Boost newer studies
        year = doc_meta.get('year') or doc_meta.get('publication_year') or payload.get('year')
        if year:
            try:
                y = int(str(year)[:4])
                # Scale: 2015 = 0 boost, 2025 = 0.5 boost
                return min(0.5, max(0, (y - 2015) / 20))
            except (ValueError, TypeError):
                pass
        return 0.0
    
    elif sort_by == 'citations':
        # Boost highly cited studies - use citation_counts dict from PostgreSQL
        count = None
        
        # First try to get from citation_counts dict (fetched from PostgreSQL)
        if citation_counts:
            doc_id = payload.get('doc_id')
            doi = doc_meta.get('doi')
            
            if doc_id and doc_id in citation_counts:
                count = citation_counts[doc_id]
            elif doi and doi in citation_counts:
                count = citation_counts[doi]
        
        # Fallback to payload/doc_meta (in case it's embedded)
        if count is None:
            count = doc_meta.get('citations') or doc_meta.get('citation_count') or payload.get('citations')
        
        if count is not None:
            try:
                c = int(str(count).replace(',', ''))
                # Scale: 0 citations = 0 boost, 500+ = 0.5 boost
                boost = min(0.5, c / 1000)
                if boost > 0:
                    print(f"[PreferencesFilter] Citation boost: {c} citations -> {boost:.3f} boost")
                return boost
            except (ValueError, TypeError):
                pass
        return 0.0
    
    elif sort_by == 'outcomes':
        # Boost studies with strong outcomes (OS, PFS reported)
        text = payload.get('text', '').lower()
        boost = 0.0
        if 'overall survival' in text or ' os ' in text:
            boost += 0.15
        if 'progression-free survival' in text or ' pfs ' in text:
            boost += 0.15
        if 'hazard ratio' in text or ' hr ' in text:
            boost += 0.1
        if 'median' in text and ('survival' in text or 'months' in text):
            boost += 0.1
        return min(0.5, boost)
    
    elif sort_by == 'patient_relevance':
        # This is handled separately in the retrieval pipeline
        # based on clinical profile matching
        return 0.0
    
    return 0.0
