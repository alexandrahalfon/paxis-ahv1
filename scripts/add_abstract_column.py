"""
Migration script to add abstract column to existing studies table.
Run this once to update the database schema.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.services.account_db import get_account_db


async def migrate():
    """Add abstract column to studies table if it doesn't exist."""
    db = get_account_db()
    pool = await db.get_pool()
    
    async with pool.acquire() as conn:
        # Check if table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'studies'
            )
        """)
        
        if not table_exists:
            print("✗ studies table does not exist yet")
            print("  Run the document processing pipeline first to create the table")
            return
        
        # Check if column exists
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'studies' AND column_name = 'abstract'
            )
        """)
        
        if exists:
            print("✓ abstract column already exists")
        else:
            print("Adding abstract column...")
            await conn.execute("""
                ALTER TABLE studies 
                ADD COLUMN abstract TEXT,
                ADD COLUMN abstract_source VARCHAR(50)
            """)
            print("✓ abstract column added")
        
        # Show current count
        count = await conn.fetchval("SELECT COUNT(*) FROM studies")
        print(f"Total studies in database: {count}")
        
        # Show how many have abstracts
        with_abstract = await conn.fetchval(
            "SELECT COUNT(*) FROM studies WHERE abstract IS NOT NULL AND abstract != ''"
        )
        print(f"Studies with abstracts: {with_abstract}")


if __name__ == "__main__":
    asyncio.run(migrate())
