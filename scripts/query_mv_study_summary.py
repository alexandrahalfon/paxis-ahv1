"""Query studies table from PostgreSQL."""
import asyncio
import asyncpg
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings

async def query_studies():
    print("Querying studies table from study-profiles database...")
    
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database='study-profiles',
    )
    
    # Get column names
    columns = await conn.fetch('''
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'studies'
        ORDER BY ordinal_position
    ''')
    print(f"\nStudies table has {len(columns)} columns:")
    for c in columns[:20]:  # Show first 20
        print(f"  {c['column_name']}: {c['data_type']}")
    if len(columns) > 20:
        print(f"  ... and {len(columns) - 20} more columns")
    
    # Get one example row
    row = await conn.fetchrow('SELECT * FROM studies LIMIT 1')
    if row:
        print('\n' + '='*60)
        print('Example row from studies:')
        print('='*60)
        for key, value in dict(row).items():
            if value is not None:
                val_str = str(value)[:80] + '...' if len(str(value)) > 80 else str(value)
                print(f'{key}: {val_str}')
    
    # Count rows
    count = await conn.fetchval('SELECT COUNT(*) FROM studies')
    print(f'\nTotal rows in studies: {count}')
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(query_studies())
