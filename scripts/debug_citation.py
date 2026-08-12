#!/usr/bin/env python3
"""Debug script to check citation fields in Qdrant."""
import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=120
)

collection = os.getenv('QDRANT_COLLECTION')

# Count total problematic
print("Scanning ALL documents for 'PhD et al' or 'MD et al'...")

offset = None
problematic_count = 0
total_scanned = 0

while True:
    points, next_offset = client.scroll(
        collection_name=collection,
        limit=500,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    
    if not points:
        break
    
    for p in points:
        total_scanned += 1
        payload = p.payload or {}
        doc_meta = payload.get('doc_meta', {}) or {}
        author_et_al = doc_meta.get('author_et_al') or ''
        citation = doc_meta.get('citation') or ''
        
        if 'PhD et al' in author_et_al or 'MD et al' in author_et_al:
            problematic_count += 1
            if problematic_count <= 10:
                print(f"\n[{problematic_count}] {payload.get('doc_id', '')[:50]}")
                print(f"    author_et_al: {author_et_al}")
                print(f"    authors: {doc_meta.get('authors', [])[:2]}")
    
    offset = next_offset
    if offset is None:
        break

print(f"\n{'='*60}")
print(f"Total points scanned: {total_scanned}")
print(f"Problematic points: {problematic_count}")
