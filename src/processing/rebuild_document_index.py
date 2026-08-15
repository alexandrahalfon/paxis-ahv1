#!/usr/bin/env python3
"""
Rebuild document indices for cached files with improved hierarchical numbering.
This avoids expensive OCR reprocessing by only regenerating the index.
"""

import json
import importlib.util
from pathlib import Path

# Import from hyphenated filename
spec = importlib.util.spec_from_file_location("hybrid_ocr", "hybrid-ocr-questionnaire.py")
hybrid_ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid_ocr)
HybridOCRQuestionnaire = hybrid_ocr.HybridOCRQuestionnaire

def rebuild_index_for_cache_file(cache_file_path: str):
    """Rebuild the document index for a single cached file."""
    cache_path = Path(cache_file_path)
    
    if not cache_path.exists():
        print(f"❌ File not found: {cache_file_path}")
        return False
    
    print(f"\n📄 Processing: {cache_path.name}")
    
    # Load existing cache
    with open(cache_path, 'r', encoding='utf-8') as f:
        cached_data = json.load(f)
    
    # Create questionnaire instance
    questionnaire = HybridOCRQuestionnaire(pdf_path=cached_data['source_pdf'])
    questionnaire.structured_content = cached_data['structured_content']
    questionnaire.pixtral_content = cached_data.get('pixtral_content', '')
    questionnaire.timestamp = cached_data['timestamp']
    
    # Rebuild index with new hierarchical structure
    print("   🔄 Rebuilding index with hierarchical numbering...")
    new_index = questionnaire.create_document_index()
    
    # Update cached data
    cached_data['document_index'] = new_index
    
    # Update processing metadata if it exists
    if 'processing_metadata' in cached_data:
        cached_data['processing_metadata']['total_paragraphs'] = new_index['metadata']['total_paragraphs']
        cached_data['processing_metadata']['total_sentences'] = new_index['metadata']['total_sentences']
        cached_data['processing_metadata']['index_rebuilt'] = True
    
    # Save back to file
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cached_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Index rebuilt: {new_index['metadata']['total_paragraphs']} paragraphs")
    return True

def rebuild_all_cached_indices():
    """Rebuild indices for all cached files."""
    cache_dir = Path("cache")
    
    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return
    
    cache_files = list(cache_dir.glob("*_processed_content.json"))
    
    if not cache_files:
        print("❌ No cached files found")
        return
    
    print(f"🔍 Found {len(cache_files)} cached files")
    print("="*70)
    
    success_count = 0
    for cache_file in cache_files:
        if rebuild_index_for_cache_file(str(cache_file)):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"✅ Successfully rebuilt {success_count}/{len(cache_files)} indices")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Rebuild specific file
        cache_file = sys.argv[1]
        rebuild_index_for_cache_file(cache_file)
    else:
        # Rebuild all cached files
        print("🔧 Rebuilding document indices for all cached files...")
        print("   (This only rebuilds the index, no OCR reprocessing)")
        print()
        rebuild_all_cached_indices()
