#!/usr/bin/env python
"""Check what categories exist in Qdrant and search for prostate cancer studies."""

import sys
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from src.core.config import settings

print("[Check] Connecting to Qdrant...", flush=True)
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

collection = settings.qdrant_collection
print(f"[Check] Collection: {collection}", flush=True)

# Get collection info
info = client.get_collection(collection)
print(f"[Check] Total points: {info.points_count}", flush=True)

# Sample some points to see what categories exist
print("\n[Check] Sampling points to find categories...", flush=True)
sample = client.scroll(
    collection_name=collection,
    limit=100,
    with_payload=True,
    with_vectors=False
)

categories = set()
for point in sample[0]:
    if point.payload:
        cat = point.payload.get("category")
        if cat:
            categories.add(cat)

print(f"[Check] Categories found in sample: {sorted(categories)}", flush=True)

# Search for prostate in text
print("\n[Check] Searching for 'prostate' in document text...", flush=True)

# Try a text search for prostate
from openai import OpenAI
openai_client = OpenAI(api_key=settings.openai_api_key)

# Generate embedding for "prostate cancer"
response = openai_client.embeddings.create(
    model=settings.embed_model,
    input="prostate cancer Gleason score PSA localized"
)
query_vector = response.data[0].embedding

# Search without category filter
print("\n[Check] Searching WITHOUT category filter...", flush=True)
results = client.query_points(
    collection_name=collection,
    query=query_vector,
    limit=20,
    with_payload=True
)

print(f"[Check] Found {len(results.points)} results", flush=True)
for i, point in enumerate(results.points[:10], 1):
    title = point.payload.get("doc_meta", {}).get("title", "Unknown")[:60]
    category = point.payload.get("category", "N/A")
    text_preview = point.payload.get("text", "")[:100]
    print(f"  {i}. [{category}] {title}...", flush=True)
    if "prostate" in text_preview.lower():
        print(f"      Text contains 'prostate'", flush=True)

# Now search WITH GU filter
print("\n[Check] Searching WITH gu_processed_documents filter...", flush=True)
results_gu = client.query_points(
    collection_name=collection,
    query=query_vector,
    query_filter=Filter(
        should=[
            FieldCondition(key="category", match=MatchValue(value="gu_processed_documents")),
            FieldCondition(key="category", match=MatchValue(value="gu")),
            FieldCondition(key="category", match=MatchValue(value="GU")),
        ]
    ),
    limit=20,
    with_payload=True
)

print(f"[Check] Found {len(results_gu.points)} results with GU filter", flush=True)
for i, point in enumerate(results_gu.points[:10], 1):
    title = point.payload.get("doc_meta", {}).get("title", "Unknown")[:60]
    category = point.payload.get("category", "N/A")
    print(f"  {i}. [{category}] {title}...", flush=True)

# Check if prostate studies are in a different category
print("\n[Check] Looking for prostate-specific category...", flush=True)
# Scroll through more points looking for prostate
scroll_result = client.scroll(
    collection_name=collection,
    limit=1000,
    with_payload=["category", "doc_meta"],
    with_vectors=False
)

prostate_categories = set()
for point in scroll_result[0]:
    if point.payload:
        title = point.payload.get("doc_meta", {}).get("title", "")
        if "prostate" in title.lower():
            cat = point.payload.get("category", "unknown")
            prostate_categories.add(cat)
            print(f"  Found prostate study in category '{cat}': {title[:60]}...", flush=True)

if prostate_categories:
    print(f"\n[Check] Prostate studies found in categories: {prostate_categories}", flush=True)
else:
    print("\n[Check] No prostate studies found in first 1000 points", flush=True)
