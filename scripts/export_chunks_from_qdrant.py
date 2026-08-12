#!/usr/bin/env python3
"""
Export chunks from Qdrant to JSONL for PTO frame building.

Usage:
    python export_chunks_from_qdrant.py
    
    # Or with explicit credentials
    QDRANT_URL=https://... QDRANT_API_KEY=... python export_chunks_from_qdrant.py
"""

import json
import os
import sys
from pathlib import Path

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("❌ qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "exueed_kb_latest")

# Validate
if not QDRANT_URL:
    print("❌ QDRANT_URL not set. Set it in .env or environment.")
    sys.exit(1)

print("=" * 60)
print("QDRANT CHUNK EXPORTER")
print("=" * 60)
print(f"URL:        {QDRANT_URL}")
print(f"Collection: {COLLECTION}")
print("=" * 60)

# Connect
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

# Check collection exists
try:
    info = client.get_collection(COLLECTION)
    total_points = info.points_count
    print(f"\n✅ Connected. Collection has {total_points:,} points.")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    sys.exit(1)

# Create output directory
Path("ingestion_output").mkdir(exist_ok=True)
output_path = Path("ingestion_output/03_tagged_chunks.jsonl")

print(f"\n📥 Exporting chunks to: {output_path}")

all_chunks = []
offset = None
batch_num = 0
skipped_pto = 0

while True:
    results, offset = client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    if not results:
        break
    
    for point in results:
        payload = point.payload or {}
        
        # Skip if it's already a PTO frame
        if payload.get("node_type") == "pto_frame":
            skipped_pto += 1
            continue
        
        all_chunks.append(payload)
    
    batch_num += 1
    print(f"  Batch {batch_num}: {len(all_chunks):,} chunks exported...", end="\r")
    
    if offset is None:
        break

print(f"\n\n✅ Export complete!")
print(f"   Chunks exported: {len(all_chunks):,}")
if skipped_pto > 0:
    print(f"   PTO frames skipped: {skipped_pto}")

# Save to JSONL
print(f"\n💾 Saving to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(all_chunks):,} chunks to {output_path}")

# Quick stats
print("\n" + "-" * 60)
print("QUICK STATS")
print("-" * 60)

# Count keyword coverage
has_keywords = sum(1 for c in all_chunks if c.get("metadata", {}).get("keyword_matches"))
print(f"Chunks with keyword metadata: {has_keywords:,} ({100*has_keywords/len(all_chunks):.1f}%)")

# Count by category
from collections import Counter
categories = Counter(c.get("category", "unknown") for c in all_chunks)
print(f"\nTop categories:")
for cat, count in categories.most_common(10):
    print(f"  {cat}: {count:,}")

print("-" * 60)
print(f"\n🚀 Next step:")
print(f"   python scripts/build_pto_frames.py {output_path} --upsert")
