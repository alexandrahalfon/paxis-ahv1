"""
Fix Author Citations in Qdrant
==============================

This script fixes documents where author names were incorrectly extracted,
resulting in citations like "MD et al." instead of proper author names.

The script:
1. Scans Qdrant for documents with problematic author citations
2. Attempts to extract correct author names from the document metadata
3. Updates the citation and author_et_al fields

Usage:
    # Dry run - see what would be fixed
    python scripts/fix_author_citations.py --dry-run --limit 100

    # Live update
    python scripts/fix_author_citations.py --live --limit 500

    # Check specific document
    python scripts/fix_author_citations.py --doc-id "some_doc_id"
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Common medical credentials/titles to strip from author names
CREDENTIALS_PATTERN = re.compile(
    r',?\s*\b('
    r'MD|M\.D\.|DO|D\.O\.|PhD|Ph\.D\.|'
    r'MBBS|MBChB|FRCP|FRCS|FACS|FACP|'
    r'MPH|MS|MSc|MA|MBA|'
    r'RN|NP|PA|PA-C|'
    r'Jr\.?|Sr\.?|III|IV|'
    r'FRCR|FRANZCR|FASTRO|'
    r'Professor|Prof\.?|Dr\.?'
    r')\b\.?',
    re.IGNORECASE
)

# Pattern to detect problematic author citations
PROBLEMATIC_PATTERNS = [
    r'\bMD\s+et\s+al\b',
    r'\bPhD\s+et\s+al\b',
    r'\bDO\s+et\s+al\b',
    r'\bUnknown\s+author\b',
    r'\bet\s+al\.\s*$',  # Just "et al." with nothing before
]


def clean_author_name(name: str) -> str:
    """
    Clean an author name by removing credentials and titles.
    
    Examples:
        "Claus Garbe, MD" -> "Claus Garbe"
        "John Smith, PhD, FACS" -> "John Smith"
        "Dr. Jane Doe" -> "Jane Doe"
    """
    if not name:
        return ""
    
    # Remove credentials
    cleaned = CREDENTIALS_PATTERN.sub('', name)
    
    # Remove extra commas and whitespace
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip(' ,.')
    
    return cleaned


def extract_last_name(full_name: str) -> str:
    """
    Extract the last name from a full name, handling various formats.
    
    Examples:
        "John Smith" -> "Smith"
        "Smith, John" -> "Smith"
        "John van der Berg" -> "van der Berg"
        "Jean-Pierre Dupont" -> "Dupont"
    """
    if not full_name:
        return ""
    
    # Clean the name first
    name = clean_author_name(full_name)
    
    if not name:
        return ""
    
    # Check if it's "LastName, FirstName" format
    if ',' in name:
        parts = name.split(',')
        return parts[0].strip()
    
    # Split by spaces
    parts = name.split()
    
    if len(parts) == 1:
        return parts[0]
    
    # Handle prefixes like "van", "de", "von", etc.
    prefixes = {'van', 'von', 'de', 'del', 'della', 'di', 'da', 'le', 'la', 'el', 'al'}
    
    # Find where the last name starts
    last_name_parts = []
    for i, part in enumerate(parts):
        if part.lower() in prefixes and i < len(parts) - 1:
            last_name_parts = parts[i:]
            break
    
    if last_name_parts:
        return ' '.join(last_name_parts)
    
    # Default: last word is the last name
    return parts[-1]


def format_author_et_al(authors: List[str]) -> str:
    """
    Format authors as 'LastName et al.' with proper credential handling.
    """
    # Clean all author names
    cleaned_authors = [clean_author_name(a) for a in (authors or [])]
    cleaned_authors = [a for a in cleaned_authors if a]
    
    if not cleaned_authors:
        return "Unknown author"
    
    if len(cleaned_authors) == 1:
        return cleaned_authors[0]
    
    # Get last name of first author
    last_name = extract_last_name(cleaned_authors[0])
    
    if not last_name:
        return "Unknown author"
    
    return f"{last_name} et al."


def is_problematic_citation(citation: str, author_et_al: str) -> bool:
    """Check if a citation has problematic author formatting."""
    text_to_check = f"{citation or ''} {author_et_al or ''}"
    
    for pattern in PROBLEMATIC_PATTERNS:
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return True
    
    return False


class AuthorCitationFixer:
    """Fix author citations in Qdrant."""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection: str):
        from qdrant_client import QdrantClient
        
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)
        self.collection = collection
        
        print(f"✓ Connected to Qdrant: {qdrant_url}")
        print(f"✓ Collection: {collection}")
    
    def scan_and_fix(
        self,
        dry_run: bool = True,
        limit: Optional[int] = None,
        doc_id: Optional[str] = None
    ) -> Dict:
        """Scan for problematic citations and fix them."""
        
        print(f"\n{'='*60}")
        print(f"AUTHOR CITATION FIXER")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
        print(f"{'='*60}\n")
        
        stats = {
            "scanned": 0,
            "problematic": 0,
            "fixed": 0,
            "failed": 0,
            "examples": [],
        }
        
        seen_docs = set()
        offset = None
        batch_num = 0
        
        while True:
            batch_num += 1
            
            # Build filter if doc_id specified
            scroll_filter = None
            if doc_id:
                scroll_filter = {
                    "must": [{"key": "doc_id", "match": {"value": doc_id}}]
                }
            
            try:
                points, next_offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    limit=100,
                    offset=offset,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                print(f"\n⚠️ Scroll error at batch {batch_num}: {e}")
                print("Retrying...")
                continue
            
            if not points:
                break
            
            # Progress update every 50 batches
            if batch_num % 50 == 0:
                print(f"\n[Progress] Batch {batch_num}, Scanned: {stats['scanned']} docs, Fixed: {stats['fixed']}")
            
            for point in points:
                payload = point.payload or {}
                pid = payload.get("doc_id", str(point.id))
                
                # Skip if we've already processed this document
                if pid in seen_docs:
                    continue
                seen_docs.add(pid)
                
                stats["scanned"] += 1
                
                # Get current citation info
                doc_meta = payload.get("doc_meta", {}) or {}
                current_citation = doc_meta.get("citation") or ""
                current_author_et_al = doc_meta.get("author_et_al") or ""
                authors = doc_meta.get("authors") or []
                title = (doc_meta.get("title") or "")[:80]
                
                # Check if problematic
                if not is_problematic_citation(current_citation, current_author_et_al):
                    continue
                
                stats["problematic"] += 1
                
                # Try to fix
                new_author_et_al = format_author_et_al(authors)
                
                # Check if we actually improved it
                if new_author_et_al == "Unknown author" and not authors:
                    # Can't fix without author data
                    if len(stats["examples"]) < 20:
                        stats["examples"].append({
                            "doc_id": pid,
                            "title": title,
                            "current": current_author_et_al,
                            "new": new_author_et_al,
                            "authors": authors,
                            "status": "NO_AUTHORS",
                        })
                    continue
                
                # Log the fix
                print(f"\n{'─'*50}")
                print(f"Doc: {pid}")
                print(f"Title: {title}...")
                print(f"Authors: {authors[:3]}{'...' if len(authors) > 3 else ''}")
                print(f"Current: {current_author_et_al}")
                print(f"Fixed:   {new_author_et_al}")
                
                if len(stats["examples"]) < 50:
                    stats["examples"].append({
                        "doc_id": pid,
                        "title": title,
                        "current": current_author_et_al,
                        "new": new_author_et_al,
                        "authors": authors[:5],
                        "status": "FIXED" if not dry_run else "WOULD_FIX",
                    })
                
                if not dry_run:
                    try:
                        # Update all points with this doc_id
                        self._update_document(pid, new_author_et_al, doc_meta)
                        stats["fixed"] += 1
                        print(f"✓ Updated")
                    except Exception as e:
                        stats["failed"] += 1
                        print(f"✗ Failed: {e}")
                else:
                    stats["fixed"] += 1
                    print(f"[DRY RUN] Would update")
                
                if limit and stats["scanned"] >= limit:
                    break
            
            if limit and stats["scanned"] >= limit:
                break
            
            offset = next_offset
            if offset is None:
                break
        
        # Print summary
        self._print_summary(stats, dry_run)
        
        return stats
    
    def _update_document(self, doc_id: str, new_author_et_al: str, doc_meta: Dict):
        """Update all points with the given doc_id."""
        # Find all points with this doc_id
        points, _ = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter={
                "must": [{"key": "doc_id", "match": {"value": doc_id}}]
            },
            limit=1000,
            with_payload=True,  # Need full payload to update nested fields
            with_vectors=False,
        )
        
        if not points:
            return
        
        # Build new citation string
        year = doc_meta.get("year")
        title = doc_meta.get("title") or ""
        journal = doc_meta.get("journal") or ""
        doi = doc_meta.get("doi") or ""
        
        parts = []
        if new_author_et_al and new_author_et_al != "Unknown author":
            parts.append(new_author_et_al)
        if year:
            parts.append(f"({year})")
        if title:
            parts.append(title)
        if journal:
            parts.append(journal)
        if doi:
            parts.append(f"doi:{doi}")
        
        new_citation = " ".join(parts).strip()
        
        # Update each point with the full doc_meta object
        for point in points:
            point_id = point.id
            current_payload = point.payload or {}
            current_doc_meta = dict(current_payload.get("doc_meta", {}) or {})
            
            # Update the fields in doc_meta
            current_doc_meta["author_et_al"] = new_author_et_al
            current_doc_meta["citation"] = new_citation
            
            # Set the entire doc_meta object (dot notation doesn't work for nested fields)
            self.qdrant.set_payload(
                collection_name=self.collection,
                payload={"doc_meta": current_doc_meta},
                points=[point_id],
                wait=False,  # Don't wait for each one
            )
    
    def _print_summary(self, stats: Dict, dry_run: bool):
        """Print summary of fixes."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Documents scanned: {stats['scanned']}")
        print(f"Problematic found: {stats['problematic']}")
        print(f"{'Would fix' if dry_run else 'Fixed'}: {stats['fixed']}")
        if not dry_run:
            print(f"Failed: {stats['failed']}")
        
        if stats["examples"]:
            print(f"\n--- Sample Fixes ---")
            for ex in stats["examples"][:10]:
                status = ex.get("status", "")
                print(f"\n  [{status}] {ex['doc_id'][:40]}...")
                print(f"    {ex['current']} -> {ex['new']}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix author citations in Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fixed without updating")
    parser.add_argument("--live", action="store_true",
                        help="Actually update Qdrant")
    parser.add_argument("--limit", type=int,
                        help="Limit number of documents to process")
    parser.add_argument("--doc-id", type=str,
                        help="Fix specific document by ID")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.live:
        parser.print_help()
        print("\nError: Specify --dry-run or --live")
        sys.exit(1)
    
    # Get Qdrant config
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION")
    
    if not all([qdrant_url, qdrant_key, collection]):
        print("Error: Missing Qdrant configuration.")
        print("Set: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION")
        sys.exit(1)
    
    fixer = AuthorCitationFixer(qdrant_url, qdrant_key, collection)
    
    results = fixer.scan_and_fix(
        dry_run=not args.live,
        limit=args.limit,
        doc_id=args.doc_id,
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
