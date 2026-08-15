#!/usr/bin/env python3
"""
Inspect PTO frames in Qdrant to understand what they contain.

This script helps you:
1. See example PTO frames
2. Understand what fields are populated
3. Determine if indications are captured
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

def inspect_pto_frames():
    """Inspect PTO frames in Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    # Connect to Qdrant
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        timeout=60
    )
    collection = os.getenv('QDRANT_COLLECTION', 'exueed_kb_latest')
    
    print(f"\n{'='*70}")
    print("PTO FRAME INSPECTION")
    print(f"Collection: {collection}")
    print(f"{'='*70}\n")
    
    # Method 1: Scroll through all points and filter by node_type
    print("Searching for PTO frames...")
    
    all_pto_frames = []
    offset = None
    
    # Scroll through collection to find PTO frames
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True
        )
        
        for point in results:
            if point.payload.get("node_type") == "pto_frame":
                all_pto_frames.append(point.payload)
        
        if offset is None or len(results) == 0:
            break
        
        if len(all_pto_frames) >= 50:  # Limit for inspection
            break
    
    print(f"\nFound {len(all_pto_frames)} PTO frames (limited to 50 for display)\n")
    
    if not all_pto_frames:
        print("❌ No PTO frames found in collection!")
        print("\nPossible reasons:")
        print("  1. PTO frames haven't been built/upserted yet")
        print("  2. Different collection name")
        print("  3. node_type field is named differently")
        
        # Show what node_types exist
        print("\nChecking what node_types exist in collection...")
        results, _ = client.scroll(collection_name=collection, limit=100, with_payload=True)
        node_types = set()
        for r in results:
            nt = r.payload.get("node_type", "NONE")
            node_types.add(nt)
        print(f"Node types found: {node_types}")
        return
    
    # Analyze field coverage
    print("="*70)
    print("FIELD COVERAGE ANALYSIS")
    print("="*70)
    
    field_counts = {
        "cancer_type": 0,
        "stage": 0,
        "tnm": 0,
        "biomarkers": 0,
        "treatment_modalities": 0,
        "dose_fractionation": 0,
        "chemo_agents": 0,
        "outcomes": 0,
        "indication": 0,  # Check if this exists
        "indication_criteria": 0,  # Check if this exists
    }
    
    for frame in all_pto_frames:
        for field in field_counts.keys():
            value = frame.get(field)
            if value and (not isinstance(value, (list, dict)) or len(value) > 0):
                field_counts[field] += 1
    
    total = len(all_pto_frames)
    print(f"\nField population rates (out of {total} frames):\n")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {field:25s} {bar} {count:3d} ({pct:5.1f}%)")
    
    # Show example frames
    print("\n" + "="*70)
    print("EXAMPLE PTO FRAMES")
    print("="*70)
    
    # Find frames related to breast cancer if possible
    breast_frames = [f for f in all_pto_frames 
                     if "breast" in (f.get("cancer_type") or "").lower() 
                     or "breast" in (f.get("category") or "").lower()]
    
    display_frames = breast_frames[:3] if breast_frames else all_pto_frames[:3]
    
    for i, frame in enumerate(display_frames, 1):
        print(f"\n--- Frame {i} ---")
        print(f"  Doc ID:     {frame.get('doc_id', 'N/A')}")
        print(f"  Category:   {frame.get('category', 'N/A')}")
        print(f"  Cancer:     {frame.get('cancer_type', 'N/A')}")
        print(f"  Stage:      {frame.get('stage') or frame.get('tnm') or 'N/A'}")
        print(f"  Biomarkers: {frame.get('biomarkers') or 'N/A'}")
        print(f"  Treatments: {frame.get('treatment_modalities') or 'N/A'}")
        print(f"  Dose:       {frame.get('dose_fractionation') or 'N/A'}")
        print(f"  Chemo:      {frame.get('chemo_agents') or 'N/A'}")
        print(f"  Outcomes:   {frame.get('outcomes') or 'N/A'}")
        print(f"  Confidence: {frame.get('confidence', 'N/A')}")
        
        # Check for any indication-related fields
        indication_fields = [k for k in frame.keys() if 'indicat' in k.lower()]
        if indication_fields:
            print(f"  ⭐ INDICATION FIELDS: {indication_fields}")
        
        # Show frame_text (the embedded text)
        frame_text = frame.get('frame_text', '')
        if frame_text:
            print(f"  Frame Text: {frame_text[:200]}...")
    
    # Check if any frames mention "indication"
    print("\n" + "="*70)
    print("SEARCHING FOR INDICATION-RELATED CONTENT")
    print("="*70)
    
    indication_keywords = ["indication", "indicated", "recommend", "appropriate for", 
                           "should receive", "candidates for", "criteria for"]
    
    frames_with_indication = []
    for frame in all_pto_frames:
        frame_text = (frame.get('frame_text') or '').lower()
        for kw in indication_keywords:
            if kw in frame_text:
                frames_with_indication.append((frame, kw))
                break
    
    if frames_with_indication:
        print(f"\n✅ Found {len(frames_with_indication)} frames with indication-related text:\n")
        for frame, kw in frames_with_indication[:5]:
            print(f"  - {frame.get('doc_id')}: matched '{kw}'")
            print(f"    Frame text: {frame.get('frame_text', '')[:150]}...")
    else:
        print("\n❌ No frames contain indication-related keywords in frame_text")
        print("   This suggests PTO frames focus on outcomes, not indications.")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    has_indication = field_counts.get("indication", 0) > 0 or field_counts.get("indication_criteria", 0) > 0
    has_outcomes = field_counts.get("outcomes", 0) > 0
    
    if has_indication:
        print("\n✅ Your PTO frames DO capture treatment indications!")
    else:
        print("\n❌ Your PTO frames do NOT capture treatment indications.")
        print("   They capture: Patient profile → Treatment → Outcomes")
        print("   Missing: WHY/WHEN to use the treatment")
    
    if has_outcomes:
        print(f"\n✅ Outcomes are captured in {field_counts['outcomes']}/{total} frames")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    inspect_pto_frames()
