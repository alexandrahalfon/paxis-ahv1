#!/usr/bin/env python3
"""
Fix existing user upload record with missing doc_id and study_profile.
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def fix_upload():
    """Fix the existing upload record."""
    from src.api.services.account_db import get_account_db
    from src.processing.study_profile_extractor import StudyProfileExtractor
    
    db = get_account_db()
    pool = await db.get_pool()
    
    # Get the upload record
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT upload_id, filename, doc_id, study_profile
            FROM user_uploads
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        if not row:
            print("No uploads found")
            return
        
        upload_id = row['upload_id']
        filename = row['filename']
        current_doc_id = row['doc_id']
        current_profile = row['study_profile']
        
        print(f"Upload ID: {upload_id}")
        print(f"Filename: {filename}")
        print(f"Current doc_id: {current_doc_id}")
        print(f"Has study_profile: {current_profile is not None}")
        
        # Fix doc_id from filename
        doc_id = Path(filename).stem
        print(f"\nNew doc_id: {doc_id}")
        
        # Find the processed directory
        processed_base = Path("processed_documents")
        processed_dirs = sorted(processed_base.glob("tmp*"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not processed_dirs:
            print("No processed directories found")
            return
        
        # Use the most recent one
        processed_dir = processed_dirs[0]
        print(f"Using processed dir: {processed_dir}")
        
        # Extract study profile
        print("\nExtracting study profile...")
        extractor = StudyProfileExtractor()
        profile_result = extractor.extract_from_processed_dir(processed_dir)
        
        study_profile = None
        doi = None
        pmid = None
        
        if profile_result and profile_result.get("extracted_data"):
            study_profile = profile_result.get("extracted_data")
            print("Study profile extracted successfully")
            
            # Extract DOI and PMID
            study_details = study_profile.get("study_details", {})
            doi_data = study_details.get("doi", {})
            pmid_data = study_details.get("pmid", {})
            doi = doi_data.get("value") if isinstance(doi_data, dict) else doi_data
            pmid = pmid_data.get("value") if isinstance(pmid_data, dict) else pmid_data
            
            print(f"DOI: {doi}")
            print(f"PMID: {pmid}")
        else:
            print("No study profile extracted")
        
        # Update the record
        print("\nUpdating database record...")
        await conn.execute("""
            UPDATE user_uploads
            SET doc_id = $1,
                study_profile = $2,
                doi = $3,
                pmid = $4
            WHERE upload_id = $5
        """, doc_id, json.dumps(study_profile) if study_profile else None, doi, pmid, upload_id)
        
        print("✓ Record updated successfully")
        
        # Verify
        updated = await conn.fetchrow("""
            SELECT doc_id, 
                   CASE WHEN study_profile IS NOT NULL THEN 'YES' ELSE 'NO' END as has_profile,
                   doi, pmid
            FROM user_uploads
            WHERE upload_id = $1
        """, upload_id)
        
        print(f"\nVerification:")
        print(f"  doc_id: {updated['doc_id']}")
        print(f"  has_profile: {updated['has_profile']}")
        print(f"  doi: {updated['doi']}")
        print(f"  pmid: {updated['pmid']}")


if __name__ == "__main__":
    asyncio.run(fix_upload())
