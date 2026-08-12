"""
Source Finder Service - Uses GPT-4o-mini to find source paragraphs for extracted values.
Only called when position-based lookup fails or returns low-quality results.
"""

import os
import re
import logging
from typing import Optional, Dict, List
from openai import OpenAI

logger = logging.getLogger(__name__)

class SourceFinderService:
    """
    Service to find the source paragraph/chunk for an extracted value using GPT-4o-mini.
    Designed to be cost-effective by:
    1. Only being called when position-based lookup fails
    2. Sending only relevant chunks (not full document)
    3. Caching results
    """
    
    def __init__(self):
        self._client = None  # Lazy initialization
        self._client_initialized = False
        self._cache: Dict[str, Dict] = {}  # Cache: (doc_id, value) -> source
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if not self._client_initialized:
            self._init_client()
        return self._client
    
    def _init_client(self):
        """Initialize OpenAI client if API key is available."""
        self._client_initialized = True
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self._client = OpenAI(api_key=api_key)
            logger.info("SourceFinderService initialized with OpenAI client")
        else:
            logger.warning("OPENAI_API_KEY not set - GPT source finding disabled")
    
    def find_source(
        self,
        extracted_value: str,
        field_name: str,
        chunks: List[Dict],
        doc_id: str = None
    ) -> Optional[Dict]:
        """
        Find the source paragraph for an extracted value using GPT-4o-mini.
        
        Args:
            extracted_value: The value that was extracted (e.g., "7.9%", "N1")
            field_name: The field type (e.g., "mortality", "n_stage", "gender")
            chunks: List of document chunks to search
            doc_id: Document ID for caching
            
        Returns:
            Dict with 'section', 'text', 'is_table' keys, or None if not found
        """
        if not self.client:
            logger.debug("OpenAI client not available, skipping GPT source finding")
            return None
        
        if not chunks or not extracted_value:
            return None
        
        # Check cache first
        cache_key = f"{doc_id}:{field_name}:{extracted_value}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for source: {cache_key}")
            return self._cache[cache_key]
        
        # Filter to relevant chunks (reduce token usage)
        relevant_chunks = self._filter_relevant_chunks(extracted_value, field_name, chunks)
        
        if not relevant_chunks:
            logger.debug(f"No relevant chunks found for {field_name}={extracted_value}")
            return None
        
        # Build context from relevant chunks
        context = self._build_context(relevant_chunks)
        
        # Call GPT-4o-mini
        try:
            result = self._call_gpt(extracted_value, field_name, context)
            
            if result:
                # Cache the result
                self._cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error calling GPT for source finding: {e}")
            return None
    
    def _filter_relevant_chunks(
        self,
        value: str,
        field_name: str,
        chunks: List[Dict],
        max_chunks: int = 10
    ) -> List[Dict]:
        """
        Filter chunks to only those likely to contain the source.
        This reduces token usage significantly.
        """
        value_lower = value.lower()
        field_lower = field_name.lower()
        
        # Define context keywords for different field types
        field_context = {
            'mortality': ['mortality', 'death', 'died', 'survival', 'fatal'],
            'recurrence': ['recurrence', 'recur', 'relapse', 'local control'],
            'survival': ['survival', 'alive', 'os', 'pfs', 'dfs', 'year'],
            'n_stage': ['node', 'nodal', 'lymph', 'n0', 'n1', 'n2', 'n3', 'stage'],
            't_stage': ['tumor', 't1', 't2', 't3', 't4', 'stage', 'size'],
            'm_stage': ['metasta', 'm0', 'm1', 'distant', 'stage'],
            'stage_group': ['stage', 'i', 'ii', 'iii', 'iv', 'tnm'],
            'gender': ['women', 'men', 'female', 'male', 'sex', 'gender', 'patient'],
            'number_of_patients': ['patient', 'enrolled', 'randomized', 'n=', 'cohort', 'women', 'men'],
            'age': ['age', 'year', 'old', 'median', 'range'],
            'cancer_location': ['cancer', 'carcinoma', 'tumor', 'breast', 'lung', 'prostate'],
            'treatment': ['treatment', 'therapy', 'dose', 'gy', 'chemotherapy', 'radiation'],
        }
        
        # Get context keywords for this field
        context_keywords = field_context.get(field_lower, [])
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            section = chunk.get('section', '').lower()
            
            # Skip acknowledgments, references, etc.
            if any(skip in section for skip in ['acknowledg', 'reference', 'funding', 'conflict']):
                continue
            
            score = 0
            
            # Check if value appears in chunk
            if value_lower in chunk_text:
                score += 10
            
            # Check for partial value matches (for percentages, numbers)
            value_parts = re.findall(r'\d+\.?\d*', value)
            for part in value_parts:
                if part in chunk_text:
                    score += 5
            
            # Check for context keywords
            for keyword in context_keywords:
                if keyword in chunk_text:
                    score += 2
            
            # Prefer Results and Methods sections
            if 'result' in section:
                score += 3
            elif 'method' in section:
                score += 2
            elif 'abstract' in section:
                score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and take top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:max_chunks]]
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks for GPT."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            section = chunk.get('section', 'Unknown')
            text = chunk.get('text', '')
            
            # Normalize section name before sending to GPT
            section = self._normalize_section_for_gpt(section)
            
            # Clean the text
            text = self._clean_text(text)
            
            if text:
                context_parts.append(f"[Chunk {i+1} - Section: {section}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _normalize_section_for_gpt(self, raw_section: str) -> str:
        """Normalize section names before sending to GPT."""
        if not raw_section:
            return 'Document'
        
        section_lower = raw_section.lower().strip()
        
        # Remove AI prefixes
        ai_prefixes = ['pixtral:', 'gpt:', 'claude:', 'llm:', 'ai:', 'model:', 'assistant:']
        for prefix in ai_prefixes:
            if section_lower.startswith(prefix):
                section_lower = section_lower[len(prefix):].strip()
        
        # Remove markdown headers
        section_lower = re.sub(r'^#+\s*', '', section_lower)
        
        # Map to standard sections
        section_mappings = {
            'background': 'Background',
            'introduction': 'Background',
            'abstract': 'Abstract',
            'methods': 'Methods',
            'method': 'Methods',
            'materials and methods': 'Methods',
            'patients and methods': 'Methods',
            'results': 'Results',
            'result': 'Results',
            'findings': 'Results',
            'discussion': 'Discussion',
            'conclusion': 'Conclusion',
            'conclusions': 'Conclusion',
        }
        
        for key, value in section_mappings.items():
            if key in section_lower:
                return value
        
        # Skip non-useful sections
        skip_sections = ['acknowledgment', 'reference', 'funding', 'conflict', 'disclosure']
        for skip in skip_sections:
            if skip in section_lower:
                return 'Document'
        
        return 'Document'
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing AI preambles."""
        if not text:
            return ''
        
        # Remove common AI preambles
        patterns = [
            r'^Certainly!?\s*Here\s+is.*?:\s*',
            r'^Here\s+is\s+the\s+extracted.*?:\s*',
            r'^The\s+extracted\s+text.*?:\s*',
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def _call_gpt(self, value: str, field_name: str, context: str) -> Optional[Dict]:
        """Call GPT-4o-mini to find the source paragraph."""
        
        system_prompt = """You are a medical document analyst. Your task is to find the exact paragraph or sentence that contains a specific extracted value from a medical research paper.

Rules:
1. Find the paragraph that DIRECTLY states or contains the given value
2. Return the FULL paragraph, not just a snippet
3. Identify which section it's from (Background, Methods, Results, Discussion, Conclusion)
4. If the value appears in a table, indicate that
5. If you cannot find the exact source, return null

Respond in JSON format:
{
  "section": "Results",
  "text": "The full paragraph text here...",
  "is_table": false,
  "confidence": "high"
}

If not found, respond with: {"found": false}"""

        user_prompt = f"""Find the source paragraph for this extracted value:

Field: {field_name}
Value: {value}

Document chunks to search:
{context}

Return the paragraph that contains or states this value."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            result = json.loads(result_text)
            
            if result.get('found') == False or not result.get('text'):
                return None
            
            return {
                'section': result.get('section', 'Document'),
                'text': result.get('text', ''),
                'is_table': result.get('is_table', False),
                'gpt_confidence': result.get('confidence', 'medium')
            }
            
        except Exception as e:
            logger.error(f"GPT API error: {e}")
            return None
    
    def clear_cache(self, doc_id: str = None):
        """Clear the cache, optionally for a specific document."""
        if doc_id:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{doc_id}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


# Global instance
source_finder_service = SourceFinderService()
