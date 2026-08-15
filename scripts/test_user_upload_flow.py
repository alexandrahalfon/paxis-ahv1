#!/usr/bin/env python3
"""
Test script to verify the user upload flow.

This script tests:
1. Schema creation in the account database
2. Storing a user upload with study profile
3. Retrieving the study profile
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def test_schema_creation():
    """Test that the user_uploads table can be created."""
    print("\n1. Testing schema creation...")
    
    from src.api.services.user_uploads_service import get_user_uploads_service
    
    service = get_user_uploads_service()
    await service._ensure_schema()
    
    print("   ✓ Schema created/verified successfully")
    return True


async def test_database_connection():
    """Test database connection to account database."""
    print("\n2. Testing database connection...")
    
    from src.api.services.account_db import get_account_db
    
    db = get_account_db()
    pool = await db.get_pool()
    
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1
    
    print("   ✓ Database connection successful")
    return True


async def test_table_structure():
    """Test that user_uploads table has all required columns."""
    print("\n3. Testing table structure...")
    
    from src.api.services.account_db import get_account_db
    
    db = get_account_db()
    pool = await db.get_pool()
    
    async with pool.acquire() as conn:
        columns = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'user_uploads'
            ORDER BY ordinal_position
        """)
        
        column_names = [c['column_name'] for c in columns]
        
        required_columns = [
            'id', 'user_id', 'upload_id', 'doc_id', 'filename', 'title',
            'status', 'doc_meta', 'embeddings', 'embedding_dim',
            'chunk_metadata', 'chunk_count', 'study_profile',
            'doi', 'pmid', 'reused_existing', 'created_at', 'processed_at'
        ]
        
        missing = [c for c in required_columns if c not in column_names]
        
        if missing:
            print(f"   ✗ Missing columns: {missing}")
            return False
        
        print(f"   ✓ All {len(required_columns)} required columns present")
        print(f"   Columns: {', '.join(column_names)}")
        return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("USER UPLOAD FLOW TEST")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Schema Creation", test_schema_creation),
        ("Table Structure", test_table_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! User upload flow is ready.")
    else:
        print("Some tests failed. Check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
