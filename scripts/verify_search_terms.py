"""Verify search_terms is working in PostgreSQL."""
import asyncio
import asyncpg
import time
import sys

sys.path.insert(0, '.')
from src.core.config import settings

async def verify():
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database='study-profiles',
    )
    
    # Performance test
    test_terms = ['breast', 'radiation', 'stage']
    start = time.perf_counter()
    result = await conn.fetch('''
        SELECT id, document_name
        FROM studies 
        WHERE search_terms && $1
        LIMIT 50
    ''', test_terms)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f'Query: {test_terms}')
    print(f'Results: {len(result)} studies in {elapsed:.1f}ms')
    print('search_terms is working!' if elapsed < 50 else 'Slower than expected')
    
    await conn.close()

asyncio.run(verify())
