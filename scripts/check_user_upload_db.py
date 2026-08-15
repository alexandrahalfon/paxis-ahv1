#!/usr/bin/env python3
"""
Check PostgreSQL database for user upload data.

User uploads are stored in the exueed_cache database (account database),
NOT in the display-study-details database (admin studies).
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import asyncpg
import os


async def check_database():
    """Check all user upload related tables in the account database."""
    
    # Connect to the ACCOUNT database (exueed_cache), not the study-profiles database
    # User uploads are stored in the account database alongside users and user_cache
    host = os.getenv('CACHE_POSTGRES_HOST', os.getenv('POSTGRES_HOST'))
    port = int(os.getenv('CACHE_POSTGRES_PORT', os.getenv('POSTGRES_PORT', 5432)))
    user = os.getenv('CACHE_POSTGRES_USER', os.getenv('POSTGRES_USER'))
    password = os.getenv('CACHE_POSTGRES_PASSWORD', os.getenv('POSTGRES_PASSWORD'))
    database = os.getenv('CACHE_POSTGRES_DATABASE', 'exueed_cache')  # Default to exueed_cache
    
    print(f"Connecting to: {host}:{port}/{database}")
    
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    
    print("=" * 70)
    print(f"CHECKING ACCOUNT DATABASE ({database}) FOR USER UPLOADS")
    print("=" * 70)
    
    # 1. List all tables
    print("\n1. ALL TABLES IN DATABASE")
    print("-" * 50)
    
    tables = await conn.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    print("Tables:")
    for t in tables:
        print(f"  - {t['table_name']}")
    
    # 2. Check user_uploads table
    print("\n\n2. USER_UPLOADS TABLE")
    print("-" * 50)
    
    try:
        # First check if table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'user_uploads'
            )
        """)
        
        if not table_exists:
            print("  Table 'user_uploads' does not exist yet.")
            print("  It will be created when the first user uploads a document.")
        else:
            # Get column info
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'user_uploads'
                ORDER BY ordinal_position
            """)
            
            print("  Columns:")
            for col in columns:
                print(f"    - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
            
            # Get uploads
            uploads = await conn.fetch("""
                SELECT id, user_id, upload_id, doc_id, filename, title, status, 
                       chunk_count, embedding_dim,
                       CASE WHEN embeddings IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embeddings,
                       LENGTH(embeddings) as embeddings_size_bytes,
                       CASE WHEN study_profile IS NOT NULL THEN 'YES' ELSE 'NO' END as has_study_profile,
                       doi, pmid, reused_existing,
                       created_at
                FROM user_uploads
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            if uploads:
                print(f"\n  Found {len(uploads)} user uploads:")
                for u in uploads:
                    print(f"\n  Upload ID: {u['upload_id']}")
                    print(f"    User ID: {u['user_id']}")
                    print(f"    Doc ID: {u['doc_id']}")
                    print(f"    Filename: {u['filename']}")
                    print(f"    Title: {u['title']}")
                    print(f"    Status: {u['status']}")
                    print(f"    Chunks: {u['chunk_count']}")
                    print(f"    Embedding Dim: {u['embedding_dim']}")
                    print(f"    Has Embeddings: {u['has_embeddings']}")
                    if u['embeddings_size_bytes']:
                        print(f"    Embeddings Size: {u['embeddings_size_bytes']:,} bytes")
                    print(f"    Has Study Profile: {u['has_study_profile']}")
                    print(f"    DOI: {u['doi']}")
                    print(f"    PMID: {u['pmid']}")
                    print(f"    Reused Existing: {u['reused_existing']}")
                    print(f"    Created: {u['created_at']}")
            else:
                print("\n  No user uploads found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 3. Check users table
    print("\n\n3. USERS TABLE")
    print("-" * 50)
    
    try:
        users = await conn.fetch("""
            SELECT id, email, created_at
            FROM users
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        if users:
            print(f"Found {len(users)} users:")
            for u in users:
                print(f"  - {u['email']} (ID: {u['id']}, Created: {u['created_at']})")
        else:
            print("  No users found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 4. Check user_study_profiles table (if exists)
    print("\n\n4. USER_STUDY_PROFILES TABLE")
    print("-" * 50)
    
    try:
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'user_study_profiles'
            )
        """)
        
        if not table_exists:
            print("  Table 'user_study_profiles' does not exist.")
            print("  (Study profiles are now stored in user_uploads.study_profile column)")
        else:
            profiles = await conn.fetch("""
                SELECT profile_id, user_id, upload_id, document_name, doc_id,
                       doi, pmid,
                       CASE WHEN study_details IS NOT NULL THEN 'YES' ELSE 'NO' END as has_study_details,
                       created_at
                FROM user_study_profiles
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            if profiles:
                print(f"Found {len(profiles)} user study profiles:")
                for p in profiles:
                    print(f"\n  Profile ID: {p['profile_id']}")
                    print(f"    User ID: {p['user_id']}")
                    print(f"    Upload ID: {p['upload_id']}")
                    print(f"    Document: {p['document_name']}")
            else:
                print("  No user study profiles found")
    except Exception as e:
        print(f"  Error: {e}")
    
    await conn.close()
    print("\n" + "=" * 70)
    print("NOTE: Admin studies are in a SEPARATE database (display-study-details)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(check_database())
