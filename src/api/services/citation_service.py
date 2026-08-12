"""
Citation Count Service

Fetches citation counts from Semantic Scholar API for studies.
Caches results to avoid repeated API calls and persists to PostgreSQL.
"""

import asyncio
import httpx
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
import asyncpg
from src.core.config import settings

logger = logging.getLogger(__name__)

# Semantic Scholar API base URL
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# Database pool for studies
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


async def _ensure_citation_column():
    """Ensure the citation_count column exists in studies table"""
    pool = await _get_study_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='studies' AND column_name='citation_count') THEN
                    ALTER TABLE studies ADD COLUMN citation_count INTEGER;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='studies' AND column_name='citation_count_updated_at') THEN
                    ALTER TABLE studies ADD COLUMN citation_count_updated_at TIMESTAMPTZ;
                END IF;
            END $$;
        """)


class CitationService:
    """Service for fetching and caching citation counts"""
    
    def __init__(self):
        self._cache: Dict[str, dict] = {}  # doi/pmid -> {count, fetched_at}
        self._cache_ttl = timedelta(days=7)  # Cache for 7 days
        self._schema_ensured = False
    
    async def _ensure_schema(self):
        """Ensure database schema is ready"""
        if not self._schema_ensured:
            await _ensure_citation_column()
            self._schema_ensured = True
    
    async def get_citation_count(
        self, 
        doi: str = None, 
        pmid: str = None,
        study_id: int = None,
        persist: bool = True
    ) -> Optional[int]:
        """
        Get citation count for a paper from Semantic Scholar.
        
        Args:
            doi: DOI of the paper
            pmid: PubMed ID of the paper
            study_id: Optional study_id to persist the count to database
            persist: Whether to save the count to PostgreSQL (default True)
            
        Returns:
            Citation count or None if not found
        """
        if not doi and not pmid:
            return None
        
        await self._ensure_schema()
        
        # Check database first if we have study_id
        if persist and (doi or pmid):
            db_count = await self._get_from_db(doi=doi, pmid=pmid)
            if db_count is not None:
                return db_count
        
        # Check memory cache
        cache_key = doi or f"pmid:{pmid}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - cached["fetched_at"] < self._cache_ttl:
                return cached["count"]
        
        # Fetch from Semantic Scholar
        try:
            paper_id = f"DOI:{doi}" if doi else f"PMID:{pmid}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                    params={"fields": "citationCount,title"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    count = data.get("citationCount", 0)
                    
                    # Cache in memory
                    self._cache[cache_key] = {
                        "count": count,
                        "fetched_at": datetime.now()
                    }
                    
                    # Persist to database
                    if persist:
                        await self._save_to_db(
                            doi=doi, 
                            pmid=pmid, 
                            study_id=study_id,
                            count=count
                        )
                    
                    return count
                elif response.status_code == 404:
                    # Paper not found, cache as 0
                    self._cache[cache_key] = {
                        "count": 0,
                        "fetched_at": datetime.now()
                    }
                    return 0
                else:
                    logger.warning(f"Semantic Scholar API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching citation count: {e}")
            return None
    
    async def _get_from_db(self, doi: str = None, pmid: str = None) -> Optional[int]:
        """
        Get citation count from database if it exists and is fresh.
        
        Returns:
            Citation count or None if not found/stale
        """
        pool = await _get_study_pool()
        
        async with pool.acquire() as conn:
            # Build query based on available identifier
            if doi:
                row = await conn.fetchrow("""
                    SELECT citation_count, citation_count_updated_at
                    FROM studies
                    WHERE doi = $1 AND citation_count IS NOT NULL
                """, doi)
            elif pmid:
                row = await conn.fetchrow("""
                    SELECT citation_count, citation_count_updated_at
                    FROM studies
                    WHERE pmid = $1 AND citation_count IS NOT NULL
                """, pmid)
            else:
                return None
            
            if row and row['citation_count'] is not None:
                updated_at = row['citation_count_updated_at']
                # Check if data is still fresh (within cache TTL)
                if updated_at and (datetime.now(updated_at.tzinfo) - updated_at) < self._cache_ttl:
                    return row['citation_count']
        
        return None
    
    async def _save_to_db(
        self, 
        doi: str = None, 
        pmid: str = None, 
        study_id: int = None,
        count: int = 0
    ):
        """
        Save citation count to the studies table.
        """
        pool = await _get_study_pool()
        
        async with pool.acquire() as conn:
            try:
                if study_id:
                    await conn.execute("""
                        UPDATE studies 
                        SET citation_count = $1, citation_count_updated_at = NOW()
                        WHERE study_id = $2
                    """, count, study_id)
                elif doi:
                    await conn.execute("""
                        UPDATE studies 
                        SET citation_count = $1, citation_count_updated_at = NOW()
                        WHERE doi = $2
                    """, count, doi)
                elif pmid:
                    await conn.execute("""
                        UPDATE studies 
                        SET citation_count = $1, citation_count_updated_at = NOW()
                        WHERE pmid = $2
                    """, count, pmid)
                    
                logger.info(f"Saved citation count {count} for doi={doi}, pmid={pmid}, study_id={study_id}")
            except Exception as e:
                logger.error(f"Error saving citation count to DB: {e}")
    
    async def get_citation_counts_batch(
        self, 
        papers: List[Dict[str, str]],
        max_concurrent: int = 5,
        persist: bool = True
    ) -> Dict[str, int]:
        """
        Get citation counts for multiple papers.
        
        Args:
            papers: List of dicts with 'doi', 'pmid', and optionally 'study_id' keys
            max_concurrent: Maximum concurrent API requests
            persist: Whether to save counts to PostgreSQL
            
        Returns:
            Dict mapping doi/pmid to citation count
        """
        await self._ensure_schema()
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_one(paper: Dict[str, str]):
            async with semaphore:
                doi = paper.get("doi")
                pmid = paper.get("pmid")
                study_id = paper.get("study_id")
                key = doi or f"pmid:{pmid}"
                
                count = await self.get_citation_count(
                    doi=doi, 
                    pmid=pmid, 
                    study_id=study_id,
                    persist=persist
                )
                if count is not None:
                    results[key] = count
                    
                # Rate limiting - Semantic Scholar allows ~100 req/5min
                await asyncio.sleep(0.1)
        
        tasks = [fetch_one(p) for p in papers]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def get_studies_by_citations(
        self, 
        limit: int = 50,
        cancer_type: str = None
    ) -> List[Dict]:
        """
        Get studies sorted by citation count from database.
        
        Args:
            limit: Maximum number of studies to return
            cancer_type: Optional filter by cancer type
            
        Returns:
            List of studies with citation counts
        """
        await self._ensure_schema()
        pool = await _get_study_pool()
        
        async with pool.acquire() as conn:
            if cancer_type:
                rows = await conn.fetch("""
                    SELECT study_id, doi, pmid, study_name, citation_count, 
                           cancer_type, number_of_patients, publish_date
                    FROM studies
                    WHERE citation_count IS NOT NULL
                      AND cancer_type ILIKE $1
                    ORDER BY citation_count DESC NULLS LAST
                    LIMIT $2
                """, f"%{cancer_type}%", limit)
            else:
                rows = await conn.fetch("""
                    SELECT study_id, doi, pmid, study_name, citation_count,
                           cancer_type, number_of_patients, publish_date
                    FROM studies
                    WHERE citation_count IS NOT NULL
                    ORDER BY citation_count DESC NULLS LAST
                    LIMIT $1
                """, limit)
            
            return [dict(row) for row in rows]
    
    async def update_all_citation_counts(
        self, 
        batch_size: int = 20,
        delay_between_batches: float = 3.0
    ) -> Dict[str, int]:
        """
        Update citation counts for all studies in the database.
        Useful for periodic background updates.
        
        Args:
            batch_size: Number of papers to fetch per batch
            delay_between_batches: Seconds to wait between batches
            
        Returns:
            Stats about the update operation
        """
        await self._ensure_schema()
        pool = await _get_study_pool()
        
        stats = {"total": 0, "updated": 0, "failed": 0}
        
        async with pool.acquire() as conn:
            # Get all studies with DOI or PMID
            rows = await conn.fetch("""
                SELECT study_id, doi, pmid
                FROM studies
                WHERE doi IS NOT NULL OR pmid IS NOT NULL
            """)
            
            stats["total"] = len(rows)
            
            # Process in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                papers = [
                    {"doi": r["doi"], "pmid": r["pmid"], "study_id": r["study_id"]}
                    for r in batch
                ]
                
                results = await self.get_citation_counts_batch(papers, persist=True)
                stats["updated"] += len(results)
                stats["failed"] += len(batch) - len(results)
                
                # Rate limit between batches
                if i + batch_size < len(rows):
                    await asyncio.sleep(delay_between_batches)
        
        logger.info(f"Citation count update complete: {stats}")
        return stats
    
    def clear_cache(self):
        """Clear the in-memory citation cache"""
        self._cache.clear()


# Singleton instance
_citation_service: Optional[CitationService] = None


def get_citation_service() -> CitationService:
    """Get or create the citation service singleton"""
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationService()
    return _citation_service
