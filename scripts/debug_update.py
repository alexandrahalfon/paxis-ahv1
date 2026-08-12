#!/usr/bin/env python3
"""Debug Qdrant payload update for nested fields."""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=120
)
collection = os.getenv('QDRANT_COLLECTION')

# Get a single point
points, _ = client.scroll(
    collection_name=collection,
    limit=1,
    with_payload=True,
    with_vectors=False,
)

if not points:
    print("No points found")
    exit()

point = points[0]
point_id = point.id
print(f"Point ID: {point_id}")
print(f"Current payload structure:")

payload = point.payload or {}
doc_meta = payload.get('doc_meta', {})
print(f"  doc_meta.author_et_al: {doc_meta.get('author_et_al')}")

# Try different update approaches

# Approach 1: Dot notation (what we've been using)
print("\n--- Approach 1: Dot notation ---")
try:
    client.set_payload(
        collection_name=collection,
        payload={"doc_meta.author_et_al": "TEST_DOT_NOTATION"},
        points=[point_id],
        wait=True,
    )
    print("  set_payload succeeded")
except Exception as e:
    print(f"  Error: {e}")

# Verify using retrieve
p1 = client.retrieve(collection_name=collection, ids=[point_id], with_payload=True)
if p1:
    dm = p1[0].payload.get('doc_meta', {})
    print(f"  After update: {dm.get('author_et_al')}")

# Approach 2: Nested dict
print("\n--- Approach 2: Nested dict ---")
try:
    # Get current full doc_meta
    current_doc_meta = dict(doc_meta)
    current_doc_meta['author_et_al'] = "TEST_NESTED_DICT"
    
    client.set_payload(
        collection_name=collection,
        payload={"doc_meta": current_doc_meta},
        points=[point_id],
        wait=True,
    )
    print("  set_payload succeeded")
except Exception as e:
    print(f"  Error: {e}")

# Verify
p2 = client.retrieve(collection_name=collection, ids=[point_id], with_payload=True)
if p2:
    dm = p2[0].payload.get('doc_meta', {})
    print(f"  After update: {dm.get('author_et_al')}")

# Restore original
print("\n--- Restoring original ---")
original_author = doc_meta.get('author_et_al')
current_doc_meta['author_et_al'] = original_author
client.set_payload(
    collection_name=collection,
    payload={"doc_meta": current_doc_meta},
    points=[point_id],
    wait=True,
)
print(f"  Restored to: {original_author}")
