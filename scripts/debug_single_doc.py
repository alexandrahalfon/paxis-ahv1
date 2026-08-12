#!/usr/bin/env python3
"""Debug a single document to see why it's not being fixed."""
import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=120
)
collection = os.getenv('QDRANT_COLLECTION')

# Find a problematic document
print("Finding a problematic document...")

points, _ = client.scroll(
    collection_name=collection,
    limit=100,
    with_payload=True,
    with_vectors=False,
)

target_doc = None
for p in points:
    payload = p.payload or {}
    doc_meta = payload.get('doc_meta', {}) or {}
    author_et_al = doc_meta.get('author_et_al') or ''
    
    if 'MD et al' in author_et_al or 'PhD et al' in author_et_al:
        target_doc = {
            'point_id': p.id,
            'doc_id': payload.get('doc_id'),
            'doc_meta': doc_meta,
        }
        break

if not target_doc:
    print("No problematic document found in first 100 points")
    exit()

print(f"\nFound problematic document:")
print(f"  doc_id: {target_doc['doc_id']}")
print(f"  author_et_al: {target_doc['doc_meta'].get('author_et_al')}")
print(f"  authors: {target_doc['doc_meta'].get('authors')}")

# Now let's manually fix it and see what happens
authors = target_doc['doc_meta'].get('authors') or []
print(f"\n--- Attempting fix ---")

# Clean author name function
def clean_author_name(name):
    if not name:
        return ""
    credentials_pattern = re.compile(
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
    cleaned = credentials_pattern.sub('', name)
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip(' ,.')
    return cleaned

def extract_last_name(full_name):
    if not full_name:
        return ""
    name = clean_author_name(full_name)
    if not name:
        return ""
    if ',' in name:
        parts = name.split(',')
        return parts[0].strip()
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    return parts[-1]

if authors:
    first_author = authors[0]
    print(f"  First author raw: {first_author}")
    cleaned = clean_author_name(first_author)
    print(f"  After cleaning: {cleaned}")
    last_name = extract_last_name(first_author)
    print(f"  Extracted last name: {last_name}")
    new_author_et_al = f"{last_name} et al." if last_name else "Unknown author"
    print(f"  New author_et_al: {new_author_et_al}")
    
    # Now actually update it
    print(f"\n--- Updating in Qdrant ---")
    
    # Find all points with this doc_id
    doc_id = target_doc['doc_id']
    all_points, _ = client.scroll(
        collection_name=collection,
        scroll_filter={
            "must": [{"key": "doc_id", "match": {"value": doc_id}}]
        },
        limit=1000,
        with_payload=False,
        with_vectors=False,
    )
    
    point_ids = [p.id for p in all_points]
    print(f"  Found {len(point_ids)} points with doc_id: {doc_id}")
    
    if point_ids:
        # Build new citation
        doc_meta = target_doc['doc_meta']
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
        
        print(f"  New citation: {new_citation[:80]}...")
        
        # Update
        client.set_payload(
            collection_name=collection,
            payload={
                "doc_meta.author_et_al": new_author_et_al,
                "doc_meta.citation": new_citation,
            },
            points=point_ids,
            wait=True,
        )
        print(f"  ✓ Updated {len(point_ids)} points")
        
        # Verify
        print(f"\n--- Verifying ---")
        verify_points, _ = client.scroll(
            collection_name=collection,
            scroll_filter={
                "must": [{"key": "doc_id", "match": {"value": doc_id}}]
            },
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        
        if verify_points:
            vp = verify_points[0]
            vm = vp.payload.get('doc_meta', {})
            print(f"  author_et_al after update: {vm.get('author_et_al')}")
            print(f"  citation after update: {vm.get('citation', '')[:80]}...")
