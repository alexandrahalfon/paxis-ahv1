"""Query Qdrant for a specific study and show payload."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from src.core.config import settings
import json

def query_qdrant_study():
    # Connect to Qdrant
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    
    collection = settings.qdrant_collection
    print(f"Querying Qdrant collection: {collection}")
    
    # Get a sample point to show payload structure
    print("Getting sample point to show payload structure...")
    results = client.scroll(
        collection_name=collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    
    if points:
        print(f"\nFound {len(points)} point(s)")
        print("="*60)
        print("PAYLOAD STRUCTURE (first point):")
        print("="*60)
        
        payload = points[0].payload
        point_id = points[0].id
        
        print(f"\nPoint ID: {point_id}")
        
        # Pretty print the payload
        for key, value in payload.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    val_str = str(v)[:80] + '...' if len(str(v)) > 80 else str(v)
                    print(f"  {k}: {val_str}")
            elif isinstance(value, list):
                print(f"\n{key}: [{len(value)} items]")
                if value and len(value) > 0:
                    print(f"  First item: {str(value[0])[:80]}...")
            else:
                val_str = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                print(f"\n{key}: {val_str}")
        
        print("\n" + "="*60)
        print("RAW PAYLOAD JSON:")
        print("="*60)
        print(json.dumps(payload, indent=2, default=str)[:3000])
        if len(json.dumps(payload)) > 3000:
            print("... [truncated]")
            
        # Show what doc_id looks like
        print("\n" + "="*60)
        print("KEY FIELDS FOR FILTERING:")
        print("="*60)
        print(f"doc_id: {payload.get('doc_id')}")
        print(f"category: {payload.get('category')}")
        if 'doc_meta' in payload:
            print(f"doc_meta.title: {payload['doc_meta'].get('title')}")
            print(f"doc_meta.doi: {payload['doc_meta'].get('doi')}")
    else:
        print("No points found in collection")

if __name__ == "__main__":
    query_qdrant_study()
