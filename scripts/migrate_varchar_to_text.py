#!/usr/bin/env python3
"""
Migrate VARCHAR columns to TEXT in PostgreSQL studies table.

This fixes insertion failures caused by VARCHAR length limits being too small
for extracted study data.

Usage:
    python scripts/migrate_varchar_to_text.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


async def migrate_varchar_to_text():
    """Migrate VARCHAR columns to TEXT to prevent truncation/insertion failures."""
    
    print("=" * 60)
    print("MIGRATING VARCHAR COLUMNS TO TEXT")
    print("=" * 60)
    
    # Import asyncpg directly instead of using the pool
    try:
        import asyncpg
        from src.core.config import settings
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        return
    
    host = settings.cache_postgres_host
    port = settings.cache_postgres_port
    database = settings.cache_postgres_database
    user = settings.cache_postgres_user
    password = settings.cache_postgres_password
    
    print(f"\nConnecting to PostgreSQL...")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Database: {database}")
    print(f"  User: {user}")
    print()
    
    # Use a direct connection instead of pool
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
            ),
            timeout=15.0
        )
        print("✓ Connected to PostgreSQL")
    except asyncio.TimeoutError:
        print("❌ Connection timed out after 15 seconds.")
        print("   Check that the PostgreSQL server is accessible.")
        return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    try:
        # Check if table exists
        print("\nChecking if studies table exists...")
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'studies'
            )
        """)
        
        if not table_exists:
            print("❌ Studies table does not exist. Run the extraction pipeline first.")
            return
        
        print("✓ Studies table found")
        
        # Columns to migrate from VARCHAR to TEXT
        columns_to_migrate = [
            "protocol_name", "trial_registration_number", "study_type",
            "study_phase", "analysis_type", "country", "age_range",
            "median_age", "gender_distribution", "performance_status",
            "cancer_location", "cancer_type", "histopathologic_type",
            "tumor_grade", "molecular_subtype", "staging_system_used",
            "risk_stratification", "metastatic_status", "extent_of_resection",
            "event_free_survival", "overall_survival", "progression_free_survival",
            "disease_free_survival", "local_control", "median_followup",
            "document_name", "doc_id", "doi", "api_model", "pmid",
        ]
        
        print(f"\nMigrating {len(columns_to_migrate)} columns to TEXT...\n")
        
        migrated = 0
        skipped = 0
        errors = 0
        
        for column in columns_to_migrate:
            try:
                # Check if column exists and get its type
                row = await conn.fetchrow("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'studies' AND column_name = $1
                """, column)
                
                if not row:
                    print(f"  ⚠️  {column}: column does not exist, skipping")
                    skipped += 1
                    continue
                
                current_type = row['data_type']
                
                if current_type == 'text':
                    print(f"  ✓ {column}: already TEXT")
                    skipped += 1
                    continue
                
                # Alter column to TEXT
                await conn.execute(f'ALTER TABLE studies ALTER COLUMN "{column}" TYPE TEXT')
                
                print(f"  ✓ {column}: {current_type} → TEXT")
                migrated += 1
                
            except Exception as e:
                print(f"  ❌ {column}: error - {e}")
                errors += 1
        
        print(f"\n{'=' * 60}")
        print("MIGRATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Migrated: {migrated}")
        print(f"Skipped:  {skipped}")
        print(f"Errors:   {errors}")
        print(f"{'=' * 60}\n")
        
        if errors > 0:
            print("⚠️  Some columns had errors. Check the output above.")
        else:
            print("✅ All columns migrated successfully!")
            print("\nYou can now re-run the study profile extraction to insert")
            print("studies that previously failed due to VARCHAR limits.")
            
    finally:
        await conn.close()
        print("\n✓ Connection closed")


if __name__ == "__main__":
    asyncio.run(migrate_varchar_to_text())
