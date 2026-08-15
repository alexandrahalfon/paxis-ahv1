"""Add doc_id from Qdrant to PostgreSQL studies table."""
import asyncio
import asyncpg
import sys

sys.path.insert(0, '.')
from qdrant_client import QdrantClient
from src.core.config import settings

async def add_doc_ids():
    # Get Qdrant client
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    # Get PostgreSQL connection
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database='study-profiles',
    )
    
    # Add doc_id column if not exists
    await conn.execute('ALTER TABLE studies ADD COLUMN IF NOT EXISTS doc_id TEXT')
    await conn.execute('CREATE INDEX IF NOT EXISTS idx_studies_doc_id ON studies(doc_id)')
    
    # Scroll Qdrant to get pg_study_id -> doc_id mapping
    mapping = {}  # pg_study_id -> doc_id
    offset = None
    
    print("Scrolling Qdrant to get pg_study_id -> doc_id mapping...")
    while True:
        results, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=['pg_study_id', 'doc_id'],
            with_vectors=False,
        )
        
        if not results:
            break
        
        for point in results:
            pg_id = point.payload.get('pg_study_id')
            doc_id = point.payload.get('doc_id')
            if pg_id and doc_id and pg_id not in mapping:
                mapping[pg_id] = doc_id
        
        if offset is None:
            break
    
    print(f'Found {len(mapping)} study -> doc_id mappings')
    
    # Update PostgreSQL
    updated = 0
    for pg_id, doc_id in mapping.items():
        result = await conn.execute(
            'UPDATE studies SET doc_id = $1 WHERE id = $2 AND doc_id IS NULL',
            doc_id, pg_id
        )
        if 'UPDATE 1' in result:
            updated += 1
    
    print(f'Updated {updated} studies with doc_id')
    
    # Verify
    count = await conn.fetchval('SELECT COUNT(*) FROM studies WHERE doc_id IS NOT NULL')
    print(f'Studies with doc_id: {count}')
    
    await conn.close()
    print('Done!')

asyncio.run(add_doc_ids())
