"""
User Preferences API
Manages user filter and sorting preferences for study search
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List

from src.api.services.auth_dependencies import get_current_user, get_current_user_optional
from src.api.services.account_db import get_account_db
from src.core.config import settings
import asyncpg

router = APIRouter(prefix="/user-preferences", tags=["user-preferences"])

# Study database pool (separate from accounts database)
_study_pool = None

async def get_study_pool():
    """Get connection pool for the study details database"""
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


class UserPreferences(BaseModel):
    """User preferences for filtering and sorting studies"""
    # Study characteristics
    study_types: Optional[List[str]] = []
    study_phases: Optional[List[str]] = []
    cancer_types: Optional[List[str]] = []
    min_patients: Optional[int] = None
    max_patients: Optional[int] = None
    analysis_types: Optional[List[str]] = []
    treatment_modalities: Optional[List[str]] = []  # e.g., ["surgery", "radiation", "chemotherapy"]
    
    # Geographic
    countries: Optional[List[str]] = []
    institutions: Optional[List[str]] = []
    
    # Demographics / Population filters
    race_ethnicities: Optional[List[str]] = []  # e.g., ["White", "Black", "Asian", "Hispanic"]
    # When set, only include studies that enrolled patients of these races
    # If a study doesn't report race data, it can optionally be included (see include_unknown_race)
    include_unknown_race: bool = True  # Include studies that don't report race demographics
    
    # Temporal preferences
    min_publication_year: Optional[int] = None  # e.g., 2015 for "modern trials only"
    max_publication_year: Optional[int] = None  # e.g., exclude very recent
    
    # Evidence quality
    require_peer_reviewed: bool = False
    min_followup_months: Optional[int] = None  # e.g., 24 for mature survival data
    
    # Outcome requirements
    required_outcomes: Optional[List[str]] = []  # e.g., ["overall_survival", "progression_free_survival"]
    
    # User uploads
    include_user_uploads: bool = True  # Include user's uploaded documents in search results
    
    # Sorting
    sort_by: str = "relevance"  # relevance, population, date, citations, outcomes, patient_relevance
    sort_order: str = "desc"
    
    # UI state
    filters_active: bool = True
    results_per_page: int = 20


class FilterOption(BaseModel):
    """A filter option with count"""
    value: str
    label: str
    count: int


class FilterOptionsResponse(BaseModel):
    """Response containing filter options"""
    options: List[FilterOption]
    total: int


class PreferencesSaveResponse(BaseModel):
    """Response after saving preferences"""
    success: bool
    message: str
    id: Optional[int] = None


async def _ensure_preferences_schema():
    """Ensure the user_preferences table exists"""
    db = get_account_db()
    pool = await db.get_pool()

    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id SERIAL PRIMARY KEY,
                user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,

                -- Filter preferences (stored as arrays for flexibility)
                study_types TEXT[],
                study_phases TEXT[],
                cancer_types TEXT[],
                min_patients INTEGER,
                max_patients INTEGER,
                analysis_types TEXT[],
                treatment_modalities TEXT[],
                countries TEXT[],
                institutions TEXT[],
                race_ethnicities TEXT[],
                include_unknown_race BOOLEAN DEFAULT true,
                
                -- Temporal preferences
                min_publication_year INTEGER,
                max_publication_year INTEGER,
                
                -- Evidence quality
                require_peer_reviewed BOOLEAN DEFAULT false,
                min_followup_months INTEGER,
                required_outcomes TEXT[],
                
                -- User uploads
                include_user_uploads BOOLEAN DEFAULT true,

                -- Sorting preference
                sort_by VARCHAR(50) DEFAULT 'relevance',
                sort_order VARCHAR(4) DEFAULT 'desc',

                -- UI state
                filters_active BOOLEAN DEFAULT true,

                -- Metadata
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Create index for fast user lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id
            ON user_preferences(user_id);
        """)

        # Add new columns if they don't exist (for existing tables)
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='user_preferences' AND column_name='countries') THEN
                    ALTER TABLE user_preferences ADD COLUMN countries TEXT[];
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='user_preferences' AND column_name='institutions') THEN
                    ALTER TABLE user_preferences ADD COLUMN institutions TEXT[];
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='user_preferences' AND column_name='race_ethnicities') THEN
                    ALTER TABLE user_preferences ADD COLUMN race_ethnicities TEXT[];
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='user_preferences' AND column_name='include_unknown_race') THEN
                    ALTER TABLE user_preferences ADD COLUMN include_unknown_race BOOLEAN DEFAULT true;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='user_preferences' AND column_name='include_user_uploads') THEN
                    ALTER TABLE user_preferences ADD COLUMN include_user_uploads BOOLEAN DEFAULT true;
                END IF;
            END $$;
        """)


@router.post("", response_model=PreferencesSaveResponse)
async def save_preferences(
    preferences: UserPreferences,
    current_user: dict = Depends(get_current_user)
):
    """
    Save user preferences to database.
    Uses UPSERT to update if exists, insert if not.
    """
    await _ensure_preferences_schema()
    
    db = get_account_db()
    pool = await db.get_pool()
    
    user_id = current_user["id"]
    
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            INSERT INTO user_preferences (
                user_id, study_types, study_phases, cancer_types,
                min_patients, max_patients, analysis_types,
                treatment_modalities, countries, institutions,
                race_ethnicities, include_unknown_race, include_user_uploads,
                sort_by, sort_order, filters_active,
                updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                study_types = EXCLUDED.study_types,
                study_phases = EXCLUDED.study_phases,
                cancer_types = EXCLUDED.cancer_types,
                min_patients = EXCLUDED.min_patients,
                max_patients = EXCLUDED.max_patients,
                analysis_types = EXCLUDED.analysis_types,
                treatment_modalities = EXCLUDED.treatment_modalities,
                countries = EXCLUDED.countries,
                institutions = EXCLUDED.institutions,
                race_ethnicities = EXCLUDED.race_ethnicities,
                include_unknown_race = EXCLUDED.include_unknown_race,
                include_user_uploads = EXCLUDED.include_user_uploads,
                sort_by = EXCLUDED.sort_by,
                sort_order = EXCLUDED.sort_order,
                filters_active = EXCLUDED.filters_active,
                updated_at = NOW()
            RETURNING id
        """,
            user_id,
            preferences.study_types or [],
            preferences.study_phases or [],
            preferences.cancer_types or [],
            preferences.min_patients,
            preferences.max_patients,
            preferences.analysis_types or [],
            preferences.treatment_modalities or [],
            preferences.countries or [],
            preferences.institutions or [],
            preferences.race_ethnicities or [],
            preferences.include_unknown_race,
            preferences.include_user_uploads,
            preferences.sort_by,
            preferences.sort_order,
            preferences.filters_active
        )
        
        return PreferencesSaveResponse(
            success=True,
            message="Preferences saved",
            id=result['id']
        )


@router.get("", response_model=UserPreferences)
async def get_preferences(
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Retrieve user preferences from database.
    Returns defaults if user not logged in or no preferences saved.
    """
    if not current_user:
        return UserPreferences()
    
    await _ensure_preferences_schema()
    
    db = get_account_db()
    pool = await db.get_pool()
    
    user_id = current_user["id"]
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT 
                study_types, study_phases, cancer_types,
                min_patients, max_patients, analysis_types,
                treatment_modalities, countries, institutions,
                race_ethnicities, include_unknown_race, include_user_uploads,
                sort_by, sort_order, filters_active
            FROM user_preferences
            WHERE user_id = $1
        """, user_id)
        
        if not row:
            return UserPreferences()
        
        return UserPreferences(
            study_types=row['study_types'] or [],
            study_phases=row['study_phases'] or [],
            cancer_types=row['cancer_types'] or [],
            min_patients=row['min_patients'],
            max_patients=row['max_patients'],
            analysis_types=row['analysis_types'] or [],
            treatment_modalities=row['treatment_modalities'] or [],
            countries=row['countries'] or [],
            institutions=row['institutions'] or [],
            race_ethnicities=row['race_ethnicities'] or [],
            include_unknown_race=row['include_unknown_race'] if row['include_unknown_race'] is not None else True,
            include_user_uploads=row['include_user_uploads'] if row['include_user_uploads'] is not None else True,
            sort_by=row['sort_by'] or 'relevance',
            sort_order=row['sort_order'] or 'desc',
            filters_active=row['filters_active'] if row['filters_active'] is not None else True
        )


@router.delete("")
async def delete_preferences(
    current_user: dict = Depends(get_current_user)
):
    """
    Delete user preferences (reset to defaults).
    """
    await _ensure_preferences_schema()
    
    db = get_account_db()
    pool = await db.get_pool()
    
    user_id = current_user["id"]
    
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM user_preferences WHERE user_id = $1",
            user_id
        )
        
        return {"success": True, "message": "Preferences deleted"}



@router.get("/countries", response_model=FilterOptionsResponse)
async def get_countries(
    search: str = Query(default="", description="Search term to filter countries")
):
    """
    Get list of countries from the studies database with study counts.
    Supports search filtering for autocomplete.
    """
    pool = await get_study_pool()
    
    async with pool.acquire() as conn:
        if search:
            # Filter by search term (case-insensitive)
            rows = await conn.fetch("""
                SELECT 
                    country,
                    COUNT(*) as study_count
                FROM studies
                WHERE country IS NOT NULL 
                  AND country != ''
                  AND country ILIKE $1
                GROUP BY country
                ORDER BY study_count DESC, country ASC
                LIMIT 50
            """, f"%{search}%")
        else:
            # Return all countries
            rows = await conn.fetch("""
                SELECT 
                    country,
                    COUNT(*) as study_count
                FROM studies
                WHERE country IS NOT NULL AND country != ''
                GROUP BY country
                ORDER BY study_count DESC, country ASC
            """)
        
        options = [
            FilterOption(
                value=row['country'],
                label=row['country'],
                count=row['study_count']
            )
            for row in rows
        ]
        
        return FilterOptionsResponse(
            options=options,
            total=len(options)
        )


@router.get("/institutions", response_model=FilterOptionsResponse)
async def get_institutions(
    search: str = Query(default="", description="Search term to filter institutions")
):
    """
    Get list of institutions from the studies database with study counts.
    Supports search filtering for autocomplete.
    """
    pool = await get_study_pool()
    
    async with pool.acquire() as conn:
        if search:
            # Filter by search term (case-insensitive)
            rows = await conn.fetch("""
                SELECT 
                    study_institution,
                    COUNT(*) as study_count
                FROM studies
                WHERE study_institution IS NOT NULL 
                  AND study_institution != ''
                  AND study_institution ILIKE $1
                GROUP BY study_institution
                ORDER BY study_count DESC, study_institution ASC
                LIMIT 50
            """, f"%{search}%")
        else:
            # Return all institutions (limited for performance)
            rows = await conn.fetch("""
                SELECT 
                    study_institution,
                    COUNT(*) as study_count
                FROM studies
                WHERE study_institution IS NOT NULL AND study_institution != ''
                GROUP BY study_institution
                ORDER BY study_count DESC, study_institution ASC
                LIMIT 100
            """)
        
        options = [
            FilterOption(
                value=row['study_institution'],
                label=row['study_institution'],
                count=row['study_count']
            )
            for row in rows
        ]
        
        return FilterOptionsResponse(
            options=options,
            total=len(options)
        )


@router.get("/race-ethnicities", response_model=FilterOptionsResponse)
async def get_race_ethnicities(
    search: str = Query(default="", description="Search term to filter race/ethnicity options")
):
    """
    Get list of race/ethnicity categories from the studies database with study counts.
    
    This extracts distinct race/ethnicity values from the race_ethnicity field.
    Studies may report multiple races (e.g., "White 70%, Black 20%, Asian 10%").
    
    Returns standardized race categories with counts of studies that include each race.
    """
    pool = await get_study_pool()
    
    # Standard race/ethnicity categories to look for
    standard_races = [
        ("White", ["white", "caucasian", "european"]),
        ("Black", ["black", "african american", "african-american", "african"]),
        ("Asian", ["asian", "chinese", "japanese", "korean", "vietnamese", "filipino", "indian"]),
        ("Hispanic/Latino", ["hispanic", "latino", "latina", "latinx", "mexican", "puerto rican"]),
        ("Native American", ["native american", "american indian", "alaska native", "indigenous"]),
        ("Pacific Islander", ["pacific islander", "hawaiian", "samoan", "guamanian"]),
        ("Middle Eastern", ["middle eastern", "arab", "persian", "turkish"]),
        ("Mixed/Other", ["mixed", "multiracial", "other", "multiple"]),
    ]
    
    async with pool.acquire() as conn:
        # Get all race_ethnicity values
        rows = await conn.fetch("""
            SELECT race_ethnicity, COUNT(*) as study_count
            FROM studies
            WHERE race_ethnicity IS NOT NULL AND race_ethnicity != ''
            GROUP BY race_ethnicity
        """)
        
        # Count studies for each standard race category
        race_counts = {}
        for label, keywords in standard_races:
            race_counts[label] = 0
        
        for row in rows:
            race_text = (row['race_ethnicity'] or '').lower()
            for label, keywords in standard_races:
                if any(kw in race_text for kw in keywords):
                    race_counts[label] += row['study_count']
        
        # Also count studies with no race data
        no_race_count = await conn.fetchval("""
            SELECT COUNT(*) FROM studies
            WHERE race_ethnicity IS NULL OR race_ethnicity = ''
        """)
        
        # Build options list
        options = []
        for label, _ in standard_races:
            if race_counts[label] > 0:
                if not search or search.lower() in label.lower():
                    options.append(FilterOption(
                        value=label,
                        label=label,
                        count=race_counts[label]
                    ))
        
        # Sort by count descending
        options.sort(key=lambda x: x.count, reverse=True)
        
        return FilterOptionsResponse(
            options=options,
            total=len(options)
        )


class CitationCountRequest(BaseModel):
    """Request for citation counts"""
    doi: Optional[str] = None
    pmid: Optional[str] = None


class CitationCountResponse(BaseModel):
    """Response with citation count"""
    doi: Optional[str] = None
    pmid: Optional[str] = None
    citation_count: Optional[int] = None
    source: str = "semantic_scholar"


class BatchCitationRequest(BaseModel):
    """Request for batch citation counts"""
    papers: List[CitationCountRequest]


class BatchCitationResponse(BaseModel):
    """Response with batch citation counts"""
    results: List[CitationCountResponse]
    total_fetched: int


@router.post("/citation-count", response_model=CitationCountResponse)
async def get_citation_count(request: CitationCountRequest):
    """
    Get citation count for a single paper from Semantic Scholar.
    
    Provide either DOI or PMID to look up the paper.
    Results are cached for 7 days.
    """
    from src.api.services.citation_service import get_citation_service
    
    if not request.doi and not request.pmid:
        raise HTTPException(
            status_code=400,
            detail="Either doi or pmid must be provided"
        )
    
    service = get_citation_service()
    count = await service.get_citation_count(doi=request.doi, pmid=request.pmid)
    
    return CitationCountResponse(
        doi=request.doi,
        pmid=request.pmid,
        citation_count=count,
        source="semantic_scholar"
    )


@router.post("/citation-counts/batch", response_model=BatchCitationResponse)
async def get_citation_counts_batch(request: BatchCitationRequest):
    """
    Get citation counts for multiple papers from Semantic Scholar.
    
    Limited to 20 papers per request to avoid rate limiting.
    Results are cached for 7 days and persisted to PostgreSQL.
    """
    from src.api.services.citation_service import get_citation_service
    
    if len(request.papers) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 papers per batch request"
        )
    
    service = get_citation_service()
    
    results = []
    for paper in request.papers:
        count = await service.get_citation_count(doi=paper.doi, pmid=paper.pmid)
        results.append(CitationCountResponse(
            doi=paper.doi,
            pmid=paper.pmid,
            citation_count=count,
            source="semantic_scholar"
        ))
    
    return BatchCitationResponse(
        results=results,
        total_fetched=sum(1 for r in results if r.citation_count is not None)
    )


class StudyWithCitations(BaseModel):
    """Study with citation count"""
    study_id: int
    doi: Optional[str] = None
    pmid: Optional[str] = None
    study_name: Optional[str] = None
    citation_count: Optional[int] = None
    cancer_type: Optional[str] = None
    number_of_patients: Optional[int] = None
    publish_date: Optional[str] = None


class StudiesByCitationsResponse(BaseModel):
    """Response with studies sorted by citations"""
    studies: List[StudyWithCitations]
    total: int


@router.get("/studies-by-citations", response_model=StudiesByCitationsResponse)
async def get_studies_by_citations(
    limit: int = Query(default=50, le=100, description="Maximum studies to return"),
    cancer_type: Optional[str] = Query(default=None, description="Filter by cancer type")
):
    """
    Get studies sorted by citation count (most cited first).
    
    Only returns studies that have citation counts in the database.
    Use the citation-count endpoints to fetch counts for new studies.
    """
    from src.api.services.citation_service import get_citation_service
    
    service = get_citation_service()
    studies = await service.get_studies_by_citations(limit=limit, cancer_type=cancer_type)
    
    return StudiesByCitationsResponse(
        studies=[
            StudyWithCitations(
                study_id=s["study_id"],
                doi=s.get("doi"),
                pmid=s.get("pmid"),
                study_name=s.get("study_name"),
                citation_count=s.get("citation_count"),
                cancer_type=s.get("cancer_type"),
                number_of_patients=s.get("number_of_patients"),
                publish_date=str(s["publish_date"]) if s.get("publish_date") else None
            )
            for s in studies
        ],
        total=len(studies)
    )
