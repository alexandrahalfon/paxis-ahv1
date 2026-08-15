"""
Load abstracts from processed documents into PostgreSQL studies table.
This script reads the structured_content.json files and extracts abstracts.
"""

import asyncio
import json
import os
import re
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.services.account_db import get_account_db


def clean_latex(text: str) -> str:
    """Remove LaTeX artifacts from text."""
    if not text:
        return ""
    
    # Remove common LaTeX patterns
    text = re.sub(r'\$\{\s*\}\^?\{?\d+\}?\$', '', text)
    text = re.sub(r'\$\^\{?\d+\}?\$', '', text)
    text = re.sub(r'\$\{[^}]*\}\$', '', text)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\text\w+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\cite\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\{\s*\}', '', text)
    
    return text.strip()


def extract_abstract_from_structured_content(data: dict) -> str:
    """Extract abstract from structured content JSON."""
    sections = data.get('sections', {})
    
    # Look for abstract section (case-insensitive)
    for key in sections.keys():
        if 'abstract' in key.lower():
            abstract = sections[key]
            if abstract:
                return clean_latex(abstract)
    
    # Try document_metadata
    doc_meta = data.get('document_metadata', {})
    if 'abstract' in doc_meta:
        return clean_latex(doc_meta['abstract'])
    
    return ""


def extract_identifiers_from_structured_content(data: dict) -> dict:
    """Extract DOI, PMID, and title from structured content."""
    doc_meta = data.get('document_metadata', {})
    doc_info = doc_meta.get('document_info', {})
    pub_info = doc_meta.get('publication_info', {})
    
    return {
        'title': doc_info.get('title', ''),
        'doi': pub_info.get('doi', ''),
        'pmid': pub_info.get('pmid', ''),
    }


async def main():
    """Load abstracts from processed documents into PostgreSQL."""
    processed_dir = 'processed_documents'
    
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} directory not found")
        return
    
    db = get_account_db()
    pool = await db.get_pool()
    
    # First, ensure abstract column exists
    async with pool.acquire() as conn:
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'studies' AND column_name = 'abstract'
            )
        """)
        
        if not exists:
            print("Adding abstract column to studies table...")
            await conn.execute("""
                ALTER TABLE studies 
                ADD COLUMN IF NOT EXISTS abstract TEXT,
                ADD COLUMN IF NOT EXISTS abstract_source VARCHAR(50)
            """)
            print("✓ Column added")
    
    # Process each document folder
    updated = 0
    not_found = 0
    no_abstract = 0
    
    folders = [f for f in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, f))]
    print(f"Found {len(folders)} processed document folders")
    
    for folder in folders:
        folder_path = os.path.join(processed_dir, folder)
        
        # Find structured content file
        structured_file = None
        for f in os.listdir(folder_path):
            if f.endswith('_structured_content.json'):
                structured_file = os.path.join(folder_path, f)
                break
        
        if not structured_file:
            print(f"  {folder}: No structured content file found")
            continue
        
        # Load and parse
        try:
            with open(structured_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  {folder}: Error loading JSON: {e}")
            continue
        
        # Extract abstract and identifiers
        abstract = extract_abstract_from_structured_content(data)
        identifiers = extract_identifiers_from_structured_content(data)
        
        title = identifiers['title']
        doi = identifiers['doi']
        pmid = identifiers['pmid']
        
        if not abstract:
            print(f"  {folder}: No abstract found ({title[:40]}...)")
            no_abstract += 1
            continue
        
        # Try to update database
        async with pool.acquire() as conn:
            # Try to find study by DOI or PMID
            study_id = None
            
            if doi:
                study_id = await conn.fetchval(
                    "SELECT study_id FROM studies WHERE doi = $1", doi
                )
            
            if not study_id and pmid:
                study_id = await conn.fetchval(
                    "SELECT study_id FROM studies WHERE pmid = $1", pmid
                )
            
            if not study_id and title:
                # Try fuzzy title match
                study_id = await conn.fetchval(
                    "SELECT study_id FROM studies WHERE study_name ILIKE $1",
                    f"%{title[:50]}%"
                )
            
            if study_id:
                # Update abstract
                await conn.execute("""
                    UPDATE studies 
                    SET abstract = $1, abstract_source = 'extracted'
                    WHERE study_id = $2 AND (abstract IS NULL OR abstract = '')
                """, abstract, study_id)
                
                print(f"  ✓ {folder}: Updated abstract ({len(abstract)} chars)")
                updated += 1
            else:
                print(f"  ✗ {folder}: Study not found in DB (DOI: {doi}, PMID: {pmid})")
                not_found += 1
    
    print(f"\n=== Summary ===")
    print(f"Updated: {updated}")
    print(f"No abstract in document: {no_abstract}")
    print(f"Study not found in DB: {not_found}")


if __name__ == "__main__":
    asyncio.run(main())
