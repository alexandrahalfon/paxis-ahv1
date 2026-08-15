import json
from pathlib import Path

doc_file = Path('processed_documents/doi_10.1056_nejmoa1607427/doi_10.1056_nejmoa1607427_document_index.json')
with open(doc_file) as f:
    data = json.load(f)

print('Document Index Structure:')
print(f'  Keys: {list(data.keys())}')
print(f'  Paragraphs: {len(data.get("paragraphs", []))} items')
print(f'  Sentences: {len(data.get("sentences", []))} items')
print(f'  Sections: {len(data.get("sections", []))} items')

# Show first paragraph structure
if data.get('paragraphs'):
    print(f'\nFirst paragraph structure:')
    print(f'  Keys: {list(data["paragraphs"][0].keys())}')
    print(f'  Sample: {str(data["paragraphs"][0])[:200]}...')
else:
    print('\n⚠️  No paragraphs found!')
