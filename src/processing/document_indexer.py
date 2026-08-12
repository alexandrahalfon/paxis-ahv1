#!/usr/bin/env python3
"""
Document Indexer

Creates searchable indexes from processed OCR content for RAG retrieval.
Supports hierarchical paragraph numbering and sentence-level indexing.
"""

import json
import re
from typing import Dict, List
from pathlib import Path
from datetime import datetime


class DocumentIndexer:
    """Creates and manages document indexes for RAG retrieval."""

    def __init__(self):
        """Initialize the document indexer."""
        self.document_index = {}

    def create_index(self, structured_content: Dict, pixtral_content: str = "") -> Dict:
        """
        Create a comprehensive index from structured and detailed content.
        
        Args:
            structured_content: Dictionary of section headings to content (from Mistral OCR)
            pixtral_content: Raw text content from Pixtral vision model
            
        Returns:
            Dictionary containing indexed sections, paragraphs, and sentences
        """
        print("📇 Creating document index with hierarchical paragraph numbering...")
        
        index = {
            "sections": {},
            "paragraphs": [],
            "sentences": [],
            "metadata": {
                "total_sections": len(structured_content),
                "total_paragraphs": 0,
                "total_sentences": 0,
                "created_at": datetime.now().isoformat()
            }
        }
        
        paragraph_id = 0
        sentence_id = 0
        
        # Index structured content with hierarchical numbering
        for section_name, content in structured_content.items():
            section_paragraphs = []
            
            # Clean section name for use as heading
            clean_section = section_name.strip().lstrip('#').strip()
            
            # Split content into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Number paragraphs within this section
            para_num = 1
            for paragraph in paragraphs:
                if paragraph:
                    # Create hierarchical paragraph reference
                    para_ref = f"{clean_section}.{para_num}"
                    
                    paragraph_info = {
                        "id": paragraph_id,
                        "paragraph_ref": para_ref,  # e.g., "Summary.1", "Methods.2"
                        "section": section_name,
                        "section_paragraph_num": para_num,
                        "text": paragraph,
                        "sentences": []
                    }
                    
                    # Split paragraph into sentences
                    sentences = self.split_into_sentences(paragraph)
                    for sentence in sentences:
                        if sentence.strip():
                            sentence_info = {
                                "id": sentence_id,
                                "paragraph_id": paragraph_id,
                                "paragraph_ref": para_ref,
                                "section": section_name,
                                "text": sentence.strip()
                            }
                            paragraph_info["sentences"].append(sentence_id)
                            index["sentences"].append(sentence_info)
                            sentence_id += 1
                    
                    section_paragraphs.append(paragraph_id)
                    index["paragraphs"].append(paragraph_info)
                    paragraph_id += 1
                    para_num += 1
            
            index["sections"][section_name] = {
                "paragraph_ids": section_paragraphs,
                "paragraph_count": para_num - 1,
                "content_length": len(content)
            }
        
        # Index Pixtral content with hierarchical numbering
        if pixtral_content:
            pixtral_section_paragraphs = []
            current_heading = "Pixtral_Content"
            para_num = 1
            
            # Split by lines and track headings
            lines = pixtral_content.split('\n')
            current_paragraph = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Skip page markers
                if line_stripped.startswith('--- Page'):
                    continue
                
                # Detect markdown headings
                if line_stripped.startswith('###') or line_stripped.startswith('##') or line_stripped.startswith('#'):
                    # Save previous paragraph if exists
                    if current_paragraph:
                        para_text = '\n'.join(current_paragraph).strip()
                        if para_text:
                            para_ref = f"{current_heading}.{para_num}"
                            paragraph_info = self._create_paragraph_info(
                                paragraph_id, para_ref, f"Pixtral: {current_heading}",
                                para_num, para_text, sentence_id, index
                            )
                            sentence_id += len(paragraph_info["sentences"])
                            
                            pixtral_section_paragraphs.append(paragraph_id)
                            index["paragraphs"].append(paragraph_info)
                            paragraph_id += 1
                            para_num += 1
                        current_paragraph = []
                    
                    # Update heading and reset paragraph counter
                    current_heading = line_stripped.lstrip('#').strip()
                    para_num = 1
                
                # Empty line might indicate paragraph break
                elif not line_stripped:
                    if current_paragraph:
                        para_text = '\n'.join(current_paragraph).strip()
                        if para_text:
                            para_ref = f"{current_heading}.{para_num}"
                            paragraph_info = self._create_paragraph_info(
                                paragraph_id, para_ref, f"Pixtral: {current_heading}",
                                para_num, para_text, sentence_id, index
                            )
                            sentence_id += len(paragraph_info["sentences"])
                            
                            pixtral_section_paragraphs.append(paragraph_id)
                            index["paragraphs"].append(paragraph_info)
                            paragraph_id += 1
                            para_num += 1
                        current_paragraph = []
                else:
                    current_paragraph.append(line)
            
            # Don't forget the last paragraph
            if current_paragraph:
                para_text = '\n'.join(current_paragraph).strip()
                if para_text:
                    para_ref = f"{current_heading}.{para_num}"
                    paragraph_info = self._create_paragraph_info(
                        paragraph_id, para_ref, f"Pixtral: {current_heading}",
                        para_num, para_text, sentence_id, index
                    )
                    sentence_id += len(paragraph_info["sentences"])
                    
                    pixtral_section_paragraphs.append(paragraph_id)
                    index["paragraphs"].append(paragraph_info)
                    paragraph_id += 1
            
            index["sections"]["Pixtral_Detailed_Content"] = {
                "paragraph_ids": pixtral_section_paragraphs,
                "content_length": len(pixtral_content)
            }
        
        index["metadata"]["total_paragraphs"] = paragraph_id
        index["metadata"]["total_sentences"] = sentence_id
        
        self.document_index = index
        print(f"✓ Document indexed: {paragraph_id} paragraphs, {sentence_id} sentences")
        print(f"  Paragraphs numbered hierarchically (e.g., 'Summary.1', 'Methods.2')")
        return index

    def _create_paragraph_info(self, paragraph_id: int, para_ref: str, section: str,
                               para_num: int, para_text: str, sentence_id: int,
                               index: Dict) -> Dict:
        """Helper to create paragraph info with sentences."""
        paragraph_info = {
            "id": paragraph_id,
            "paragraph_ref": para_ref,
            "section": section,
            "section_paragraph_num": para_num,
            "text": para_text,
            "sentences": []
        }
        
        sentences = self.split_into_sentences(para_text)
        current_sentence_id = sentence_id
        for sentence in sentences:
            if sentence.strip():
                sentence_info = {
                    "id": current_sentence_id,
                    "paragraph_id": paragraph_id,
                    "paragraph_ref": para_ref,
                    "section": section,
                    "text": sentence.strip()
                }
                paragraph_info["sentences"].append(current_sentence_id)
                index["sentences"].append(sentence_info)
                current_sentence_id += 1
        
        return paragraph_info

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def find_source_chunk(self, source_text: str) -> Dict:
        """
        Find the full paragraph chunk containing the source text.
        
        Args:
            source_text: Text snippet to search for
            
        Returns:
            Dictionary with chunk, paragraph_id, and section information
        """
        if not source_text or source_text in ["Not found", "Not specified"]:
            return {"chunk": "Not found", "paragraph_id": None, "section": "Not found"}
        
        # Search in indexed paragraphs
        for paragraph in self.document_index.get("paragraphs", []):
            if source_text.lower() in paragraph["text"].lower():
                return {
                    "chunk": paragraph["text"],
                    "paragraph_id": paragraph["id"],
                    "paragraph_ref": paragraph.get("paragraph_ref", "Unknown"),
                    "section": paragraph["section"]
                }
        
        # Fallback: return the source text itself
        return {
            "chunk": source_text,
            "paragraph_id": None,
            "paragraph_ref": "Unknown",
            "section": "Unknown"
        }

    def load_from_cache(self, cache_file: str) -> Dict:
        """
        Load a previously created index from cache.
        
        Args:
            cache_file: Path to cached processed content JSON
            
        Returns:
            The document index
        """
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.document_index = data.get("document_index", {})
        print(f"✓ Loaded index from cache: {cache_file}")
        print(f"  {self.document_index.get('metadata', {}).get('total_paragraphs', 0)} paragraphs, "
              f"{self.document_index.get('metadata', {}).get('total_sentences', 0)} sentences")
        
        return self.document_index

    def save_index(self, output_path: str) -> bool:
        """
        Save the index to a JSON file.
        
        Args:
            output_path: Path to save the index
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_index, f, indent=2, ensure_ascii=False)
            print(f"✓ Index saved to: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving index: {e}")
            return False


def create_index_from_cache(cache_file: str, output_file: str = None) -> Dict:
    """
    Convenience function to create an index from cached processed content.
    
    Args:
        cache_file: Path to cached processed content JSON
        output_file: Optional path to save the index separately
        
    Returns:
        The created document index
    """
    with open(cache_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    indexer = DocumentIndexer()
    index = indexer.create_index(
        structured_content=data.get("structured_content", {}),
        pixtral_content=data.get("pixtral_content", "")
    )
    
    if output_file:
        indexer.save_index(output_file)
    
    return index


if __name__ == "__main__":
    """Example usage: create index from a cached file."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_indexer.py <cache_file> [output_file]")
        print("\nExample:")
        print("  python document_indexer.py cache/document_processed_content.json")
        print("  python document_indexer.py cache/document_processed_content.json index/document_index.json")
        sys.exit(1)
    
    cache_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(cache_file).exists():
        print(f"✗ Cache file not found: {cache_file}")
        sys.exit(1)
    
    print(f"\n📇 Creating index from: {cache_file}")
    index = create_index_from_cache(cache_file, output_file)
    print(f"\n✓ Index created successfully!")
    print(f"  Sections: {index['metadata']['total_sections']}")
    print(f"  Paragraphs: {index['metadata']['total_paragraphs']}")
    print(f"  Sentences: {index['metadata']['total_sentences']}")
