"""
Fetch abstracts from PubMed for studies that have PMIDs but no abstracts.
"""

import asyncio
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Optional
import httpx

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.services.account_db import get_account_db


PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def fetch_abstract_from_pubmed(pmid: str, client: httpx.AsyncClient) -> Optional[str]:
    """Fetch abstract from PubMed using PMID."""
    try:
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml"
        }
        
        response = await client.get(PUBMED_EFETCH_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.text)
        
        # Find abstract text
        abstract_parts = []
        for abstract_text in root.findall(".//AbstractText"):
            label = abstract_text.get("Label", "")
            text = abstract_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        
        if abstract_parts:
            return " ".join(abstract_parts)
        
        return None
        
    except Exception as e:
        print(f"  Error fetching PMID {pmid}: {e}")
        return None


async def main():
    """Fetch abstracts for all studies with PMIDs but no abstracts."""
    db = get_account_db()
    pool = await db.get_pool()
    
    async with pool.acquire() as conn:
        # Get studies with PMIDs but no abstracts
        studies = await conn.fetch("""
            SELECT study_id, pmid, study_name 
            FROM studies 
            WHERE pmid IS NOT NULL 
              AND pmid != ''
              AND (abstract IS NULL OR abstract = '')
            ORDER BY study_id
        """)
        
        print(f"Found {len(studies)} studies with PMIDs but no abstracts")
        
        if not studies:
            print("Nothing to do!")
            return
        
        # Fetch abstracts
        updated = 0
        async with httpx.AsyncClient() as client:
            for i, study in enumerate(studies):
                pmid = study["pmid"]
                study_name = study["study_name"] or "Unknown"
                
                print(f"[{i+1}/{len(studies)}] Fetching abstract for PMID {pmid}...")
                print(f"  Study: {study_name[:60]}...")
                
                abstract = await fetch_abstract_from_pubmed(pmid, client)
                
                if abstract:
                    # Update database
                    await conn.execute("""
                        UPDATE studies 
                        SET abstract = $1, abstract_source = 'pubmed'
                        WHERE study_id = $2
                    """, abstract, study["study_id"])
                    
                    print(f"  ✓ Abstract saved ({len(abstract)} chars)")
                    updated += 1
                else:
                    print(f"  ✗ No abstract found")
                
                # Rate limit: 3 requests per second max for PubMed
                await asyncio.sleep(0.4)
        
        print(f"\nDone! Updated {updated}/{len(studies)} studies with abstracts.")


if __name__ == "__main__":
    asyncio.run(main())
