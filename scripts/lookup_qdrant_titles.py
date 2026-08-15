#!/usr/bin/env python3
"""Look up document titles from Qdrant by doc_id"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

doc_ids = [
    "doi_10.1200_jco.2014.59.5132_47bec3e1",
    "doi_10.1056_NEJMoa022628_9b94548a",
    "doi_10.1182_blood.2019003877_ce86fc66",
    "doi_10.1016_j.ijrobp.2018.10.038_502c5ffa",
    "doi_10.1200_jco._43d5e2f3",
    "doi_10.1016_S1470-2045_17_30070-0_6fed199a",
    "doi_10.3390_cancers15041288_74a6bfe6",
    "doi_10.1001_jamaoncol.2023.0356_1_c03400b0",
    "doi_10.1016_S1470-2045_20_30607-0_13622e68",
    "doi_10.1056_nejmoa1308345_3c09dbac",
    "doi_10.1001_jamaoncol.2020.1808_da8e3300",
    "doi_10.1016_S1470-2045_17_30086-4_39cc3e5b",
    "doi_10.1056_NEJMoa1111961_b51da22a",
    "doi_10.1200_jco.2014.59.5132_1_9e8cc460",
    "doi_10.1056_NEJMoa071780_b8bc1934",
]

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection = os.getenv("QDRANT_COLLECTION", "exueed_kb_latest")

print(f"Looking up {len(doc_ids)} documents in {collection}...\n")

for doc_id in doc_ids:
    # Search for one point with this doc_id
    results = client.scroll(
        collection_name=collection,
        scroll_filter={
            "must": [
                {"key": "doc_id", "match": {"value": doc_id}}
            ]
        },
        limit=1,
        with_payload=True
    )
    
    points = results[0]
    if points:
        payload = points[0].payload
        title = payload.get("title") or payload.get("doc_meta", {}).get("title") or "No title"
        citation = payload.get("citation") or payload.get("doc_meta", {}).get("citation") or ""
        print(f"doc_id: {doc_id}")
        print(f"  Title: {title[:100]}...")
        print(f"  Citation: {citation[:80]}..." if citation else "")
        print()
    else:
        print(f"doc_id: {doc_id}")
        print(f"  NOT FOUND")
        print()
