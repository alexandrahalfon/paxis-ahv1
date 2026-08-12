#!/usr/bin/env python3
"""
PTO-Aware Query Router and Retriever

This module provides:
1. Query routing to detect PTO-relevant queries
2. Hybrid retrieval (PTO frames + chunks)
3. Integration helpers for your existing RAG service

Repository location: src/api/services/pto_retriever.py

Usage:
    from src.api.services.pto_retriever import PTOQueryRouter, PTORetriever
    
    router = PTOQueryRouter()
    if router.should_use_pto_frames(query):
        results = retriever.search_pto_frames(query)

Integration with EnhancedRAGService:
    See PTO_INTEGRATION_GUIDE.md for full integration instructions.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# QUERY CLASSIFICATION
# =============================================================================

class QueryType(Enum):
    """Types of medical queries."""
    TREATMENT_SELECTION = "treatment_selection"  # "What treatment for X?"
    OUTCOME_LOOKUP = "outcome_lookup"           # "What are outcomes for X?"
    TRIAL_REFERENCE = "trial_reference"         # "What did TRIAL show?"
    DOSE_VOLUME = "dose_volume"                 # "What dose for X?"
    COMPARISON = "comparison"                    # "X vs Y for Z?"
    GENERAL = "general"                          # Default chunk retrieval


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query_type: QueryType
    should_use_pto: bool
    confidence: float  # 0-1
    detected_signals: List[str]
    extracted_profile: Dict[str, Any]  # Cancer type, stage, etc. from query


class PTOQueryRouter:
    """
    Routes queries to appropriate retrieval strategy.
    
    Detects when a query would benefit from PTO frame retrieval
    vs standard chunk retrieval.
    """
    
    # Patterns that suggest PTO frame retrieval would help
    TREATMENT_PATTERNS = [
        r'(?:what|which)\s+(?:is\s+the\s+)?(?:best|appropriate|recommended|standard)\s+treatment',
        r'how\s+(?:should|would|to)\s+treat',
        r'treatment\s+(?:for|of|in)',
        r'(?:what|which)\s+(?:therapy|regimen|approach)',
        r'standard\s+of\s+care\s+for',
    ]
    
    OUTCOME_PATTERNS = [
        r'(?:what|which)\s+(?:are\s+the\s+)?outcomes?',
        r'(?:survival|control)\s+(?:rate|outcome)',
        r'(?:OS|PFS|DFS|local\s+control)',
        r'prognosis\s+(?:for|of|in)',
        r'(?:what|how)\s+(?:is|was)\s+the\s+(?:\d+[- ]?year)',
    ]
    
    DOSE_PATTERNS = [
        r'(?:what|which)\s+dose',
        r'(?:appropriate|recommended|standard)\s+(?:dose|fractionation)',
        r'(?:how\s+much|what)\s+(?:radiation|Gy)',
        r'dose\s+(?:for|to|in)',
    ]
    
    COMPARISON_PATTERNS = [
        r'\bvs\.?\b',
        r'\bversus\b',
        r'compared?\s+(?:to|with)',
        r'difference\s+between',
        r'better\s+(?:than|outcome)',
    ]
    
    TRIAL_PATTERNS = [
        r'(?:what\s+did|according\s+to)\s+(?:the\s+)?([A-Z][A-Z0-9\-]+)\s+(?:trial|study)',
        r'(?:RTOG|EORTC|NSABP|GOG|SWOG|COG|NRG)\s*[\-]?\s*\d+',
        r'(?:results?|findings?)\s+(?:of|from)\s+(?:the\s+)?([A-Z][A-Z0-9\-]+)',
    ]
    
    # Profile extraction patterns
    CANCER_PATTERNS = [
        r'(breast|lung|prostate|head\s+and\s+neck|H&N|rectal|bladder|cervical|ovarian|pancreatic|esophageal|gastric|brain|glioma|melanoma|lymphoma|leukemia)\s*(?:cancer|carcinoma|tumor)?',
        r'(NSCLC|SCLC|HCC|RCC|CRC|AML|ALL|NHL|HL|GBM|HNSCC|NPC|SCC|adenocarcinoma)',
        r'(seminoma|ewing|rhabdomyosarcoma|medulloblastoma|ependymoma|neuroblastoma)',
    ]
    
    STAGE_PATTERNS = [
        r'stage\s*(I{1,3}V?|IV|[1-4])\s*([ABC])?',
        r'[cp]?T[0-4][a-d]?\s*N[0-3][a-c]?\s*M[01]',
        r'(early|locally\s+advanced|metastatic|advanced)\s+(?:stage)?',
    ]
    
    # Polarity suffix `[+\-]?` previously matched any trailing hyphen,
    # including connectors like the `-` in "p16-positive". The captured
    # group `(HPV|p16)` is what downstream actually consumes (the polarity
    # suffix only affects the full match span), but the lookahead below
    # makes the captured text consistent — `+` or `-` is recognised only
    # when followed by a word boundary, so "p16-positive" captures just
    # "p16", not "p16-".
    BIOMARKER_PATTERNS = [
        r'(ER|PR|HER2|triple\s+negative|TNBC)(?:[+\-](?=\s|$|[,.;:]))?',
        r'(HPV|p16)(?:[+\-](?=\s|$|[,.;:]))?',
        r'(EGFR|ALK|ROS1|KRAS|BRAF|PD-?L1)',
    ]
    
    def __init__(self):
        """Initialize the query router."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.treatment_re = [re.compile(p, re.IGNORECASE) for p in self.TREATMENT_PATTERNS]
        self.outcome_re = [re.compile(p, re.IGNORECASE) for p in self.OUTCOME_PATTERNS]
        self.dose_re = [re.compile(p, re.IGNORECASE) for p in self.DOSE_PATTERNS]
        self.comparison_re = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self.trial_re = [re.compile(p, re.IGNORECASE) for p in self.TRIAL_PATTERNS]
        self.cancer_re = [re.compile(p, re.IGNORECASE) for p in self.CANCER_PATTERNS]
        self.stage_re = [re.compile(p, re.IGNORECASE) for p in self.STAGE_PATTERNS]
        self.biomarker_re = [re.compile(p, re.IGNORECASE) for p in self.BIOMARKER_PATTERNS]
    
    def _check_patterns(self, query: str, patterns: List[re.Pattern]) -> Tuple[bool, List[str]]:
        """Check if query matches any patterns, return matches."""
        matches = []
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                matches.append(match.group(0))
        return len(matches) > 0, matches
    
    def _extract_profile(self, query: str) -> Dict[str, Any]:
        """Extract patient profile elements from query."""
        profile = {}
        
        # Extract cancer type
        for pattern in self.cancer_re:
            match = pattern.search(query)
            if match:
                profile['cancer_type'] = match.group(0).strip()
                break
        
        # Extract stage
        for pattern in self.stage_re:
            match = pattern.search(query)
            if match:
                profile['stage'] = match.group(0).strip()
                break
        
        # Extract biomarkers
        biomarkers = []
        for pattern in self.biomarker_re:
            matches = pattern.findall(query)
            biomarkers.extend(matches)
        if biomarkers:
            profile['biomarkers'] = list(set(biomarkers))
        
        return profile
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine optimal retrieval strategy.
        
        Args:
            query: User's query string
            
        Returns:
            QueryAnalysis with routing decision and extracted info
        """
        signals = []
        
        # Check each pattern category
        has_treatment, treatment_matches = self._check_patterns(query, self.treatment_re)
        has_outcome, outcome_matches = self._check_patterns(query, self.outcome_re)
        has_dose, dose_matches = self._check_patterns(query, self.dose_re)
        has_comparison, comparison_matches = self._check_patterns(query, self.comparison_re)
        has_trial, trial_matches = self._check_patterns(query, self.trial_re)
        
        signals.extend(treatment_matches)
        signals.extend(outcome_matches)
        signals.extend(dose_matches)
        signals.extend(comparison_matches)
        signals.extend(trial_matches)
        
        # Extract profile from query
        profile = self._extract_profile(query)
        
        # Determine query type and confidence
        query_type = QueryType.GENERAL
        confidence = 0.0
        
        if has_trial:
            query_type = QueryType.TRIAL_REFERENCE
            confidence = 0.9
        elif has_comparison:
            query_type = QueryType.COMPARISON
            confidence = 0.85
        elif has_treatment:
            query_type = QueryType.TREATMENT_SELECTION
            confidence = 0.8
        elif has_outcome:
            query_type = QueryType.OUTCOME_LOOKUP
            confidence = 0.8
        elif has_dose:
            query_type = QueryType.DOSE_VOLUME
            confidence = 0.75
        elif profile.get('cancer_type') and (profile.get('stage') or profile.get('biomarkers')):
            # Has profile info, might benefit from PTO
            query_type = QueryType.TREATMENT_SELECTION
            confidence = 0.6
        
        # Decide if PTO frames should be used
        should_use_pto = query_type != QueryType.GENERAL and confidence >= 0.5
        
        return QueryAnalysis(
            query_type=query_type,
            should_use_pto=should_use_pto,
            confidence=confidence,
            detected_signals=signals,
            extracted_profile=profile
        )
    
    def should_use_pto_frames(self, query: str) -> bool:
        """
        Simple boolean check for PTO frame retrieval.
        
        Args:
            query: User's query string
            
        Returns:
            True if PTO frame retrieval is recommended
        """
        analysis = self.analyze_query(query)
        return analysis.should_use_pto


# =============================================================================
# PTO RETRIEVER
# =============================================================================

class PTORetriever:
    """
    Retrieves PTO frames and chunks from Qdrant.
    
    Supports:
    - PTO frame search
    - Hybrid search (frames + chunks)
    - Evidence expansion from frames
    """
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the retriever.
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            collection_name: Collection name
            openai_api_key: OpenAI API key
            embedding_model: Embedding model name
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            import openai
        except ImportError:
            raise ImportError("Please install: pip install qdrant-client openai")
        
        # Initialize with longer timeout for cloud connections
        self.client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
            timeout=60  # 60 second timeout for cloud connections
        )
        self.collection_name = collection_name
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.router = PTOQueryRouter()
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def search_pto_frames(
        self,
        query: str,
        limit: int = 5,
        min_confidence: str = None,
        cancer_type_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for PTO frames matching the query.
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_confidence: Minimum frame confidence ("low", "medium", "high")
            cancer_type_filter: Optional cancer type to filter on
            
        Returns:
            List of matching PTO frames with scores
        
        Note: All filtering is done in post-processing to avoid Qdrant index requirements.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse
        
        # Embed query
        query_vector = self._embed_query(query)
        
        # Search WITHOUT any Qdrant filters - do all filtering in post-processing
        # This avoids "Index required" errors for unindexed fields
        results = []
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit * 10,  # Get extra since we'll filter in post-processing
                with_payload=True
            )
            results = results.points if hasattr(results, 'points') else results
        except AttributeError:
            # Fall back to older API
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit * 10,
                with_payload=True
            )
        except Exception as e:
            print(f"[PTO] Search error: {e}")
            return []
        
        # Post-process results - filter by node_type, cancer_type, confidence
        frames = []
        for result in results:
            payload = result.payload
            
            # Filter by node_type - only include PTO frames
            node_type = payload.get("node_type", "")
            if node_type != "pto_frame":
                continue
            
            # Filter by cancer_type if specified
            if cancer_type_filter:
                frame_cancer_type = (payload.get("cancer_type") or "").lower()
                filter_lower = cancer_type_filter.lower()
                # Flexible matching - check if either contains the other
                if filter_lower not in frame_cancer_type and frame_cancer_type not in filter_lower:
                    continue
            
            # Filter by confidence if specified
            if min_confidence:
                frame_confidence = payload.get("confidence", "low")
                confidence_order = {"low": 0, "medium": 1, "high": 2}
                if confidence_order.get(frame_confidence, 0) < confidence_order.get(min_confidence, 0):
                    continue
            
            frames.append({
                "score": result.score,
                "pto_id": payload.get("pto_id"),
                "doc_id": payload.get("doc_id"),
                "category": payload.get("category"),
                "cancer_type": payload.get("cancer_type"),
                "stage": payload.get("stage"),
                "tnm": payload.get("tnm"),
                "biomarkers": payload.get("biomarkers", []),
                "treatment_modalities": payload.get("treatment_modalities", []),
                "dose_fractionation": payload.get("dose_fractionation"),
                "chemo_agents": payload.get("chemo_agents", []),
                "outcomes": payload.get("outcomes", {}),
                "frame_text": payload.get("frame_text"),
                "evidence_chunk_ids": payload.get("evidence_chunk_ids", []),
                "doc_meta": payload.get("doc_meta", {}),
                "confidence": payload.get("confidence"),
            })
            
            if len(frames) >= limit:
                break
        
        return frames
        
        return frames
    
    def search_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_pto_frames: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for regular chunks (not PTO frames).
        
        Args:
            query: Search query
            limit: Maximum results to return
            exclude_pto_frames: Whether to exclude PTO frame nodes
            
        Returns:
            List of matching chunks with scores
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        query_filter = None
        if exclude_pto_frames:
            # Filter out PTO frames - get chunks where node_type is NOT pto_frame
            # Qdrant doesn't have NOT directly, so we filter in post-processing
            pass
        
        query_vector = self._embed_query(query)
        
        # Search using query_points (newer Qdrant API)
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit * 2 if exclude_pto_frames else limit,
                with_payload=True
            )
            # query_points returns QueryResponse with .points attribute
            results = results.points if hasattr(results, 'points') else results
        except AttributeError:
            # Fall back to older API
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit * 2 if exclude_pto_frames else limit,
                with_payload=True
            )
        
        chunks = []
        for result in results:
            payload = result.payload
            
            # Skip PTO frames if requested
            if exclude_pto_frames and payload.get("node_type") == "pto_frame":
                continue
            
            chunks.append({
                "score": result.score,
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "text": payload.get("text"),
                "category": payload.get("category"),
                "doc_meta": payload.get("doc_meta", {}),
                "metadata": payload.get("metadata", {}),
            })
            
            if len(chunks) >= limit:
                break
        
        return chunks
    
    def expand_evidence(
        self,
        evidence_chunk_ids: List[str],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve full chunk content for evidence chunk IDs.
        
        Args:
            evidence_chunk_ids: List of chunk IDs from PTO frame
            limit: Maximum chunks to retrieve
            
        Returns:
            List of chunk payloads
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        if not evidence_chunk_ids:
            return []
        
        # Scroll with filter on chunk_id
        # Note: This requires chunk_id to be indexed
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="chunk_id", match=MatchAny(any=evidence_chunk_ids[:limit]))
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        return [r.payload for r in results]
    
    def hybrid_search(
        self,
        query: str,
        pto_limit: int = 3,
        chunk_limit: int = 7,
        expand_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Perform hybrid search using PTO frames and chunks.
        
        If query is PTO-relevant, searches frames first, then augments with chunks.
        Otherwise, uses standard chunk retrieval.
        
        Args:
            query: Search query
            pto_limit: Maximum PTO frames to retrieve
            chunk_limit: Maximum chunks to retrieve
            expand_evidence: Whether to expand evidence from frames
            
        Returns:
            Dict with 'pto_frames', 'chunks', 'evidence_chunks', 'routing_info'
        """
        # Analyze query
        analysis = self.router.analyze_query(query)
        
        result = {
            "query": query,
            "routing_info": {
                "query_type": analysis.query_type.value,
                "should_use_pto": analysis.should_use_pto,
                "confidence": analysis.confidence,
                "detected_signals": analysis.detected_signals,
                "extracted_profile": analysis.extracted_profile,
            },
            "pto_frames": [],
            "chunks": [],
            "evidence_chunks": [],
        }
        
        if analysis.should_use_pto:
            # Search PTO frames first
            cancer_filter = analysis.extracted_profile.get("cancer_type")
            pto_frames = self.search_pto_frames(
                query=query,
                limit=pto_limit,
                cancer_type_filter=cancer_filter
            )
            result["pto_frames"] = pto_frames
            
            # Expand evidence if requested
            if expand_evidence and pto_frames:
                all_evidence_ids = []
                for frame in pto_frames:
                    all_evidence_ids.extend(frame.get("evidence_chunk_ids", []))
                
                if all_evidence_ids:
                    evidence_chunks = self.expand_evidence(all_evidence_ids, limit=5)
                    result["evidence_chunks"] = evidence_chunks
        
        # Always get some chunks as fallback/supplement
        chunks = self.search_chunks(
            query=query,
            limit=chunk_limit,
            exclude_pto_frames=True
        )
        result["chunks"] = chunks
        
        return result


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def format_pto_context(search_result: Dict[str, Any]) -> str:
    """
    Format hybrid search results into context for LLM.
    
    Args:
        search_result: Result from PTORetriever.hybrid_search()
        
    Returns:
        Formatted context string
    """
    parts = []
    
    # Format PTO frames
    if search_result.get("pto_frames"):
        parts.append("=== PATIENT-TREATMENT-OUTCOME RELATIONSHIPS ===\n")
        for i, frame in enumerate(search_result["pto_frames"], 1):
            frame_str = f"[Frame {i}]\n"
            
            # Profile
            profile_parts = []
            if frame.get("cancer_type"):
                profile_parts.append(f"Cancer: {frame['cancer_type']}")
            if frame.get("stage"):
                profile_parts.append(f"Stage: {frame['stage']}")
            if frame.get("tnm"):
                profile_parts.append(f"TNM: {frame['tnm']}")
            if frame.get("biomarkers"):
                profile_parts.append(f"Biomarkers: {', '.join(frame['biomarkers'])}")
            if profile_parts:
                frame_str += f"PATIENT: {'; '.join(profile_parts)}\n"
            
            # Treatment
            treatment_parts = []
            if frame.get("treatment_modalities"):
                treatment_parts.append(', '.join(frame['treatment_modalities']))
            if frame.get("dose_fractionation"):
                treatment_parts.append(frame['dose_fractionation'])
            if frame.get("chemo_agents"):
                treatment_parts.append(', '.join(frame['chemo_agents']))
            if treatment_parts:
                frame_str += f"TREATMENT: {'; '.join(treatment_parts)}\n"
            
            # Outcomes
            if frame.get("outcomes"):
                outcome_strs = [f"{k}: {v}" for k, v in frame["outcomes"].items()]
                frame_str += f"OUTCOMES: {'; '.join(outcome_strs)}\n"
            
            # Citation
            doc_meta = frame.get("doc_meta", {})
            if doc_meta:
                citation = doc_meta.get("citation_string") or doc_meta.get("title", "")
                if citation:
                    frame_str += f"SOURCE: {citation}\n"
            
            parts.append(frame_str)
    
    # Format regular chunks
    if search_result.get("chunks"):
        parts.append("\n=== SUPPORTING CONTEXT ===\n")
        for i, chunk in enumerate(search_result["chunks"], 1):
            chunk_str = f"[Chunk {i}]"
            doc_meta = chunk.get("doc_meta", {})
            if doc_meta:
                citation = doc_meta.get("citation_string") or doc_meta.get("title", "")
                if citation:
                    chunk_str += f" ({citation})"
            chunk_str += f"\n{chunk.get('text', '')[:500]}...\n"
            parts.append(chunk_str)
    
    return "\n".join(parts)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo the query router
    router = PTOQueryRouter()
    
    test_queries = [
        "What is the best treatment for a 55 year old with stage II ER+ HER2- breast cancer?",
        "What were the 5-year survival outcomes in the FAST-Forward trial?",
        "What dose should be used for definitive RT in locally advanced NSCLC?",
        "Compare chemoradiation vs surgery for rectal cancer",
        "What is the mechanism of action of pembrolizumab?",
        "A 65 year old male with T3N1M0 bladder cancer - treatment options?",
    ]
    
    print("=" * 70)
    print("PTO QUERY ROUTER DEMO")
    print("=" * 70)
    
    for query in test_queries:
        analysis = router.analyze_query(query)
        print(f"\nQuery: {query[:60]}...")
        print(f"  Type: {analysis.query_type.value}")
        print(f"  Use PTO: {analysis.should_use_pto} (confidence: {analysis.confidence:.2f})")
        if analysis.detected_signals:
            print(f"  Signals: {analysis.detected_signals[:3]}")
        if analysis.extracted_profile:
            print(f"  Profile: {analysis.extracted_profile}")
    
    print("\n" + "=" * 70)
