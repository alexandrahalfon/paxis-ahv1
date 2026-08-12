"""
Service for matching new studies to existing ones by cancer type.
"""

from typing import List, Dict, Any, Optional
from ...ingestion.qdrant_client import QdrantIngestionClient


class StudyMatchingService:
    """Service for matching new studies to existing ones."""
    
    def __init__(self):
        """Initialize study matching service."""
        self.qdrant_client = QdrantIngestionClient()
    
    def extract_cancer_types_from_kb(self) -> List[str]:
        """
        Extract unique cancer types from existing knowledge base.
        
        Returns:
            List of cancer types found in the knowledge base
        """
        try:
            # Search for common cancer type keywords
            cancer_types = set()
            
            # Common cancer types to look for
            cancer_keywords = [
                "lung cancer", "breast cancer", "prostate cancer", "colorectal cancer",
                "pancreatic cancer", "liver cancer", "kidney cancer", "bladder cancer",
                "ovarian cancer", "cervical cancer", "endometrial cancer",
                "gastric cancer", "esophageal cancer", "head and neck cancer",
                "brain cancer", "melanoma", "lymphoma", "leukemia",
                "sarcoma", "thyroid cancer", "testicular cancer"
            ]
            
            # Search Qdrant for each cancer type
            for keyword in cancer_keywords:
                try:
                    # Use a simple embedding search to find documents mentioning this cancer type
                    try:
                        from ...ingestion.embeddings import EmbeddingGenerator
                        embed_gen = EmbeddingGenerator()
                        query_embedding = embed_gen.embed_texts([keyword])[0]
                    except:
                        continue
                    
                    results = self.qdrant_client.search_similar(
                        query_vector=query_embedding,
                        limit=5,
                        filters=None
                    )
                    
                    if results:
                        # Check if any results mention this cancer type
                        for result in results:
                            text = result.get("payload", {}).get("text", "").lower()
                            if keyword.lower() in text:
                                cancer_types.add(keyword)
                                break
                except:
                    continue
            
            return sorted(list(cancer_types))
            
        except Exception as e:
            print(f"Error extracting cancer types: {e}")
            return []
    
    def match_studies_by_cancer_type(
        self,
        new_studies: List[Dict[str, Any]],
        existing_cancer_types: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Match new studies to existing cancer types.
        
        Args:
            new_studies: List of new study dictionaries
            existing_cancer_types: List of cancer types in existing KB
            
        Returns:
            Dictionary mapping cancer types to matching studies
        """
        matches = {cancer_type: [] for cancer_type in existing_cancer_types}
        unmatched = []
        
        for study in new_studies:
            title = study.get("title", "").lower()
            abstract = study.get("abstract", "").lower()
            text = f"{title} {abstract}"
            
            matched = False
            for cancer_type in existing_cancer_types:
                if cancer_type.lower() in text:
                    matches[cancer_type].append(study)
                    matched = True
                    break
            
            if not matched:
                unmatched.append(study)
        
        # Add unmatched studies
        if unmatched:
            matches["_unmatched"] = unmatched
        
        return matches
    
    def get_study_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about studies in the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection_info()
            
            # Extract cancer types
            cancer_types = self.extract_cancer_types_from_kb()
            
            return {
                "total_documents": collection_info.get("points_count", 0),
                "cancer_types": cancer_types,
                "cancer_type_count": len(cancer_types)
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_documents": 0,
                "cancer_types": [],
                "cancer_type_count": 0
            }
