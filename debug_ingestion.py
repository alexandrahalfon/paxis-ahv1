import sys
sys.path.insert(0, 'src')
from pathlib import Path
import json

# Manually try the normalization logic
doc_dir = Path('processed_documents/doi_10.1056_nejmoa1607427')

# Load files
idx_files = list(doc_dir.glob("*_document_index.json"))
print(f"Found document_index files: {len(idx_files)}")

if idx_files:
    with open(idx_files[0]) as f:
        document_index = json.load(f)
    
    paragraphs = document_index.get("paragraphs", [])
    print(f"Paragraphs in file: {len(paragraphs)}")
    
    # Check what the normalizer would do
    chunks_created = 0
    for para in paragraphs:
        if 'text' in para and para['text'].strip():
            chunks_created += 1
    
    print(f"Chunks that would be created: {chunks_created}")
    
    # Show a sample paragraph
    if paragraphs:
        print(f"\nSample paragraph:")
        print(f"  Text length: {len(paragraphs[0]['text'])}")
        print(f"  Text preview: {paragraphs[0]['text'][:100]}...")
