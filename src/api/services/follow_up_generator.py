"""
Follow-Up Generator Service

Generates contextual follow-up suggestions based on response content.
Uses LLM analysis instead of hardcoded pattern matching to produce
relevant follow-up questions that help users explore related topics.

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from typing import List, Dict, Optional
from openai import OpenAI
from src.core.config import settings


class FollowUpGenerator:
    """
    Generates contextual follow-up suggestions based on response content.
    
    Uses LLM analysis instead of hardcoded pattern matching to produce
    relevant follow-up questions that help users explore related topics.
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize the FollowUpGenerator.
        
        Args:
            openai_client: Optional OpenAI client instance. If not provided,
                          a new client will be created using settings.
        """
        self.client = openai_client or OpenAI(api_key=settings.openai_api_key)
    
    async def generate_follow_ups(
        self,
        query: str,
        response: str,
        module: str,
        doc_titles: List[str],
        max_suggestions: int = 4
    ) -> List[Dict[str, str]]:
        """
        Generate follow-up suggestions based on response content.
        
        Args:
            query: Original user query
            response: Generated response text
            module: Current module (general_knowledge, patient_specific, evidence_exploration)
            doc_titles: Titles of source documents used
            max_suggestions: Maximum number of suggestions (2-4)
            
        Returns:
            List of follow-up suggestions with type and text
        """
        prompt = self._build_analysis_prompt(query, response, module, doc_titles)
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.openai_mini_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            suggestions = self._parse_suggestions(
                completion.choices[0].message.content,
                max_suggestions
            )
            return suggestions
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FollowUpGenerator] Error generating follow-ups: {e}")
            return []
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for follow-up generation.
        
        Returns:
            System prompt string with instructions for generating follow-ups
        """
        return """You are a medical literature assistant helping users explore oncology topics.
        
Based on the query and response provided, generate 2-4 follow-up questions that would help the user:
1. Dive deeper into specific aspects mentioned in the response
2. Explore related treatments or outcomes
3. Compare alternatives if applicable
4. Understand practical implications

Rules:
- Questions should be specific and actionable
- Reference specific treatments, trials, or outcomes from the response when possible
- Vary the types of follow-ups (dosing, outcomes, comparisons, alternatives)
- Keep questions concise (under 15 words each)

Output format (one per line):
[TYPE] Question text

Types: DOSE, OUTCOME, COMPARE, TRIAL, ALTERNATIVE, TOXICITY, ELIGIBILITY"""

    def _build_analysis_prompt(
        self, 
        query: str, 
        response: str, 
        module: str,
        doc_titles: List[str]
    ) -> str:
        """
        Build the analysis prompt for the LLM.
        
        Args:
            query: Original user query
            response: Generated response text
            module: Current module context
            doc_titles: Titles of source documents
            
        Returns:
            Formatted prompt string for LLM analysis
        """
        titles_str = "\n".join(f"- {t}" for t in doc_titles[:5]) if doc_titles else "None"
        
        return f"""Original Query: {query}

Module: {module}

Response Summary (first 1000 chars):
{response[:1000]}

Source Documents:
{titles_str}

Generate 2-4 follow-up questions based on this response."""

    def _parse_suggestions(
        self, 
        content: str, 
        max_suggestions: int
    ) -> List[Dict[str, str]]:
        """
        Parse LLM output into structured suggestions.
        
        Args:
            content: Raw LLM output text
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggestion dictionaries with 'type' and 'text' keys
        """
        suggestions = []
        
        if not content:
            return suggestions
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Parse [TYPE] Question format
            if line.startswith("["):
                bracket_end = line.find("]")
                if bracket_end > 0:
                    suggestion_type = line[1:bracket_end].upper()
                    text = line[bracket_end + 1:].strip()
                    if text:
                        suggestions.append({
                            "type": suggestion_type,
                            "text": text
                        })
            
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions


# Singleton instance
_generator_instance: Optional[FollowUpGenerator] = None


def get_follow_up_generator() -> FollowUpGenerator:
    """
    Get singleton FollowUpGenerator instance.
    
    Returns:
        FollowUpGenerator: The singleton instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = FollowUpGenerator()
    return _generator_instance
