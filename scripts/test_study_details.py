#!/usr/bin/env python3
"""Test script for GPT Study Extractor"""

import os
import sys
import json
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

from qdrant_client import QdrantClient
from src.api.services.study_details_service import StudyDetailsService

async def test_study_extraction():
    """Test extraction for the Packer medulloblastoma study"""
    
    # Initialize Qdrant client
    qdrant_url = os.environ.get('QDRANT_URL')
    qdrant_api_key = os.environ.get('QDRANT_API_KEY')
    
    if not qdrant_url or not qdrant_api_key:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set")
        return
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY must be set")
        return
    
    print("Connecting to Qdrant...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Initialize service with GPT extraction
    print("Initializing StudyDetailsService with GPT extraction...")
    service = StudyDetailsService(client, use_gpt=True)
    
    # Clear cache to force fresh extraction
    service._cache = {}
    
    # Search for the Packer study by title keywords
    print("\nSearching for study...")
    
    # Use a known doc_id for testing
    # You can change this to test different documents
    test_doc_id = "doi_10.1200_jco.2014.59.5132_47bec3e1"  # Phase II neoadjuvant chemo trial
    
    # Or search for medulloblastoma study
    results = client.scroll(
        collection_name="exueed_kb_latest",
        limit=500,
        with_payload=True,
        with_vectors=False
    )
    
    target_doc_id = None
    target_title = None
    
    if results and results[0]:
        seen_docs = set()
        for point in results[0]:
            doc_id = point.payload.get('doc_id', '')
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            title = point.payload.get('doc_meta', {}).get('title', '')
            authors = str(point.payload.get('doc_meta', {}).get('authors', ''))
            
            # Check if this is the Packer study
            if 'medulloblastoma' in title.lower() or 'packer' in authors.lower():
                target_doc_id = doc_id
                target_title = title
                print(f"Found Packer study: {doc_id}")
                break
        
        # If not found, use the test doc_id
        if not target_doc_id:
            target_doc_id = test_doc_id
            for point in results[0]:
                if point.payload.get('doc_id') == test_doc_id:
                    target_title = point.payload.get('doc_meta', {}).get('title', 'Unknown')
                    break
            print(f"Using test document: {test_doc_id}")
    
    if target_doc_id:
        doc_id = target_doc_id
        title = target_title
        
        print(f"\nFound study:")
        print(f"  doc_id: {doc_id}")
        print(f"  title: {title}")
        
        # Now extract study details
        print("\n" + "="*80)
        print("EXTRACTING STUDY DETAILS (using GPT-4o-mini)...")
        print("="*80 + "\n")
        
        result = await service.get_study_details(doc_id=doc_id)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        # Pretty print the results
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Doc ID: {result.get('doc_id', 'N/A')}")
        print(f"Overall Confidence: {result.get('overall_confidence', 0):.2%}")
        print("\n" + "-"*80)
        
        details = result.get('details', {})
        
        for category, fields in details.items():
            if not fields:
                continue
            
            print(f"\n### {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if isinstance(fields, dict):
                for field_name, field_data in fields.items():
                    if field_data is None:
                        continue
                    
                    if isinstance(field_data, dict):
                        value = field_data.get('value', 'N/A')
                        source = field_data.get('source', '')
                        confidence = field_data.get('confidence', '')
                        note = field_data.get('note', '')
                        
                        print(f"\n{field_name}:")
                        print(f"  Value: {value}")
                        if confidence:
                            print(f"  Confidence: {confidence}")
                        if source:
                            print(f"  Source: {source}")
                        if note:
                            print(f"  Note: {note}")
                    else:
                        print(f"\n{field_name}: {field_data}")
        
        # Also save raw JSON for inspection
        output_path = Path(__file__).parent.parent / 'test_study_details_output.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n\nFull JSON output saved to: {output_path}")
        
    else:
        print("Study not found in database")
        print("\nListing available doc_ids...")
        
        # List some doc_ids
        sample = client.scroll(
            collection_name="exueed_kb_latest",
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        if sample and sample[0]:
            for point in sample[0]:
                doc_id = point.payload.get('doc_id', 'N/A')
                title = point.payload.get('doc_meta', {}).get('title', 'N/A')[:60]
                print(f"  {doc_id}: {title}...")

if __name__ == "__main__":
    asyncio.run(test_study_extraction())
