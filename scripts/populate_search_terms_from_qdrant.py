"""
Populate PostgreSQL search_terms from Qdrant Keywords

This script:
1. Queries Qdrant to get all unique keywords per pg_study_id
2. Aggregates keywords across all chunks for each study
3. Updates PostgreSQL studies.search_terms column

The keywords are already extracted and stored in Qdrant payload:
- metadata.keywords_flat: ['efs', 'os', 'enrollment', ...]
- metadata.keyword_matches: {category: [keywords]}
- category: 'peds_processed_documents', 'breast_processed_documents', etc.
"""

import asyncio
import json
import sys
from collections import defaultdict
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, '.')

import asyncpg
from qdrant_client import QdrantClient
from src.core.config import settings


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


async def get_postgres_pool():
    """Get PostgreSQL connection pool."""
    return await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database='study-profiles',
        min_size=2,
        max_size=10,
    )


def extract_keywords_from_payload(payload: dict) -> Set[str]:
    """Extract all keywords from a Qdrant payload."""
    keywords = set()
    
    # From keywords_flat
    if 'metadata' in payload and 'keywords_flat' in payload['metadata']:
        for kw in payload['metadata']['keywords_flat']:
            if kw and len(kw) > 1:  # Skip single chars
                keywords.add(kw.lower().strip())
    
    # From keyword_matches (categorized)
    if 'metadata' in payload and 'keyword_matches' in payload['metadata']:
        for category, kw_list in payload['metadata']['keyword_matches'].items():
            for kw in kw_list:
                if kw and len(kw) > 1:
                    keywords.add(kw.lower().strip())
    
    # From category (e.g., "peds_processed_documents" -> "peds", "pediatric")
    if 'category' in payload:
        cat = payload['category'].replace('_processed_documents', '').replace('_', ' ')
        keywords.add(cat.lower())
        
        # Map categories to clinical terms
        category_mappings = {
            'peds': ['pediatric', 'childhood', 'pediatric oncology'],
            'breast': ['breast cancer', 'breast', 'mammary'],
            'lung': ['lung cancer', 'lung', 'pulmonary', 'nsclc', 'sclc'],
            'gi': ['gastrointestinal', 'gi', 'colorectal', 'colon', 'rectal'],
            'gu': ['genitourinary', 'gu', 'prostate', 'bladder'],
            'gyn': ['gynecologic', 'gyn', 'cervical', 'ovarian', 'endometrial'],
            'h&n': ['head and neck', 'h&n', 'hnscc', 'oral cavity', 'larynx'],
            'cns': ['cns', 'brain', 'glioma', 'glioblastoma'],
            'skin': ['skin', 'melanoma'],
            'lymphoma': ['lymphoma', 'hodgkin', 'non-hodgkin'],
        }
        
        for key, terms in category_mappings.items():
            if key in cat.lower():
                keywords.update(terms)
    
    # From doc_meta if available
    if 'doc_meta' in payload:
        meta = payload['doc_meta']
        if meta.get('title'):
            # Extract key terms from title
            title_lower = meta['title'].lower()
            for term in ['breast', 'lung', 'prostate', 'colorectal', 'rectal', 
                        'glioblastoma', 'melanoma', 'lymphoma', 'leukemia',
                        'cervical', 'ovarian', 'bladder', 'renal', 'pancreatic',
                        'esophageal', 'gastric', 'head and neck', 'oral',
                        'radiation', 'chemotherapy', 'immunotherapy', 'surgery',
                        'neoadjuvant', 'adjuvant', 'concurrent', 'palliative',
                        'stage i', 'stage ii', 'stage iii', 'stage iv',
                        'pediatric', 'germ cell', 'sarcoma', 'ependymoma']:
                if term in title_lower:
                    keywords.add(term)
    
    return keywords


def aggregate_keywords_by_study(
    client: QdrantClient,
    collection: str,
    batch_size: int = 100,
) -> Dict[int, Set[str]]:
    """
    Scroll through Qdrant and aggregate keywords by pg_study_id.
    
    Returns:
        Dict mapping pg_study_id -> set of keywords
    """
    print(f"Scrolling Qdrant collection: {collection}")
    
    study_keywords: Dict[int, Set[str]] = defaultdict(set)
    doc_id_to_study: Dict[str, int] = {}
    
    offset = None
    total_points = 0
    points_with_study_id = 0
    
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        if not results:
            break
        
        for point in results:
            total_points += 1
            payload = point.payload
            
            # Get pg_study_id (direct link to PostgreSQL)
            pg_study_id = payload.get('pg_study_id')
            
            if pg_study_id:
                points_with_study_id += 1
                keywords = extract_keywords_from_payload(payload)
                study_keywords[pg_study_id].update(keywords)
            else:
                # Fallback: track by doc_id for later matching
                doc_id = payload.get('doc_id')
                if doc_id:
                    doc_id_to_study[doc_id] = None  # Will need to match later
        
        if total_points % 10000 == 0:
            print(f"  Processed {total_points} points, {len(study_keywords)} studies...")
        
        if offset is None:
            break
    
    print(f"\nTotal points processed: {total_points}")
    print(f"Points with pg_study_id: {points_with_study_id}")
    print(f"Unique studies found: {len(study_keywords)}")
    
    return study_keywords


async def update_postgres_search_terms(
    pool: asyncpg.Pool,
    study_keywords: Dict[int, Set[str]],
):
    """Update PostgreSQL studies.search_terms from aggregated keywords."""
    
    # First, ensure the column exists
    async with pool.acquire() as conn:
        await conn.execute("""
            ALTER TABLE studies 
            ADD COLUMN IF NOT EXISTS search_terms TEXT[];
        """)
        
        # Create GIN index if not exists
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_studies_search_terms_gin 
            ON studies USING GIN (search_terms);
        """)
    
    print(f"\nUpdating {len(study_keywords)} studies in PostgreSQL...")
    
    updated = 0
    errors = 0
    
    async with pool.acquire() as conn:
        for study_id, keywords in study_keywords.items():
            try:
                # Convert set to sorted list (for consistency)
                keywords_list = sorted(list(keywords))
                
                # Update the study
                result = await conn.execute("""
                    UPDATE studies 
                    SET search_terms = $2
                    WHERE id = $1
                """, study_id, keywords_list)
                
                if 'UPDATE 1' in result:
                    updated += 1
                
                if updated % 100 == 0:
                    print(f"  Updated {updated} studies...")
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error updating study {study_id}: {e}")
    
    print(f"\nUpdated: {updated} studies")
    print(f"Errors: {errors}")
    
    return updated


async def verify_search_terms(pool: asyncpg.Pool):
    """Verify the search_terms were populated correctly."""
    async with pool.acquire() as conn:
        # Count studies with search_terms
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE search_terms IS NOT NULL AND array_length(search_terms, 1) > 0) as with_terms,
                AVG(array_length(search_terms, 1)) FILTER (WHERE search_terms IS NOT NULL) as avg_terms,
                MAX(array_length(search_terms, 1)) as max_terms
            FROM studies
        """)
        
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        print(f"Total studies: {stats['total']}")
        print(f"With search_terms: {stats['with_terms']}")
        print(f"Average terms per study: {stats['avg_terms']:.1f}" if stats['avg_terms'] else "N/A")
        print(f"Max terms: {stats['max_terms']}")
        
        # Show sample
        samples = await conn.fetch("""
            SELECT id, doi, search_terms[1:10] as sample_terms
            FROM studies 
            WHERE search_terms IS NOT NULL 
              AND array_length(search_terms, 1) > 5
            LIMIT 3
        """)
        
        print("\nSample studies:")
        for row in samples:
            print(f"  Study {row['id']}: {row['sample_terms']}")
        
        # Test query performance
        import time
        start = time.perf_counter()
        
        test_result = await conn.fetch("""
            SELECT id, doi 
            FROM studies 
            WHERE search_terms && ARRAY['breast', 'radiation', 'stage']
              AND doi IS NOT NULL
            LIMIT 20
        """)
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\nTest query: {len(test_result)} results in {elapsed:.1f}ms")


async def main():
    """Main function."""
    print("=" * 60)
    print("POPULATE search_terms FROM QDRANT KEYWORDS")
    print("=" * 60)
    
    # Get Qdrant client
    client = get_qdrant_client()
    collection = settings.qdrant_collection
    
    # Aggregate keywords from Qdrant
    study_keywords = aggregate_keywords_by_study(client, collection)
    
    if not study_keywords:
        print("No keywords found!")
        return
    
    # Show sample
    print("\nSample keyword aggregation:")
    for study_id, keywords in list(study_keywords.items())[:3]:
        print(f"  Study {study_id}: {len(keywords)} keywords")
        print(f"    Sample: {sorted(list(keywords))[:10]}")
    
    # Get PostgreSQL pool
    pool = await get_postgres_pool()
    
    try:
        # Update PostgreSQL
        await update_postgres_search_terms(pool, study_keywords)
        
        # Verify
        await verify_search_terms(pool)
        
    finally:
        await pool.close()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
