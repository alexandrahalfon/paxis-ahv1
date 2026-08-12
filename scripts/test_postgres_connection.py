#!/usr/bin/env python3
"""Test PostgreSQL connection and query"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def test_connection():
    # Connection params
    host = "34.21.60.224"
    port = 5432
    user = "postgres"
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = "display-study-details"
    
    print(f"Connecting to {host}:{port}/{database}...")
    
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        print("Connected successfully!")
        
        # Check table structure
        print("\n--- Table columns in 'studies' ---")
        columns = await conn.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'studies'
            ORDER BY ordinal_position
            LIMIT 20
        """)
        for col in columns:
            print(f"  {col['column_name']}: {col['data_type']}")
        
        # Count studies
        count = await conn.fetchval("SELECT COUNT(*) FROM studies")
        print(f"\nTotal studies in database: {count}")
        
        # Sample DOIs
        print("\n--- Sample DOIs in database ---")
        dois = await conn.fetch("SELECT doi, study_name FROM studies WHERE doi IS NOT NULL LIMIT 5")
        for row in dois:
            print(f"  DOI: {row['doi']}")
            print(f"  Name: {row['study_name'][:80] if row['study_name'] else 'N/A'}...")
            print()
        
        # Test specific DOI lookup
        test_doi = "10.1016/j.ijrobp.2015.12.380"
        print(f"\n--- Looking for DOI: {test_doi} ---")
        result = await conn.fetchrow("SELECT study_id, doi, study_name FROM studies WHERE doi = $1", test_doi)
        if result:
            print(f"  Found! study_id={result['study_id']}")
        else:
            # Try ILIKE
            result = await conn.fetchrow("SELECT study_id, doi, study_name FROM studies WHERE doi ILIKE $1", test_doi)
            if result:
                print(f"  Found with ILIKE! study_id={result['study_id']}")
            else:
                print("  NOT FOUND")
        
        await conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
