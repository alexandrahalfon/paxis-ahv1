"""
Staging RAG Fallback Service

When a user query contains clinical details (tumor size, DOI, node involvement, ENE, etc.)
but NO explicit staging keywords (TNM, stage + numeral), this service runs a parallel
RAG search to retrieve AJCC staging criteria and uses the LLM to infer the correct stage.

This handles complex staging scenarios like:
- "oral tongue SCC, 1.5cm, 8mm DOI, ipsilateral adenopathy with ENE" -> T2N3bM0, Stage IVB
- "breast cancer, 3cm tumor, 2 positive nodes" -> T2N1M0, Stage IIB

Integration:
- Called from query_classifier_service.py when staging keywords are absent
- Runs in parallel with main query processing
- Results merged into StructuredQuery before stage inference service
"""

import re
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# STAGING KEYWORD DETECTION
# =============================================================================

# Patterns that indicate explicit staging is already present
EXPLICIT_STAGING_PATTERNS = [
    # TNM patterns (with or without c/p/yp prefix)
    r'\b[cyp]{0,2}T[0-4is][a-d]?\s*[cyp]?N[0-3][a-c]?\s*[cyp]?M[01][a-c]?\b',  # Full TNM like T2N1M0
    r'\b[cyp]{1,2}T[0-4is][a-d]?\b',  # T stage with prefix like pT3, cT2
    r'\b[cyp]?N[0-3][a-c]?\b(?!\s*(?:level|node))',  # N stage (not followed by "level" or "node")
    r'\b[cyp]?M[01][a-c]?\b',          # M stage  
    # Stage group patterns
    r'\bstage\s*[0IV]{1,3}[ABC]?\b',   # Stage I, Stage IIA, etc.
    r'\bstage\s*[1-4][ABC]?\b',        # Stage 1, Stage 2A, etc.
]

# Patterns that indicate clinical details that COULD determine staging
CLINICAL_STAGING_INDICATORS = [
    # Tumor characteristics
    r'\b\d+\.?\d*\s*(?:cm|mm)\b',                    # Size measurements
    r'\bdepth\s*(?:of\s*)?invasion\b',              # DOI
    r'\bDOI\b',                                      # DOI abbreviation
    r'\b\d+\s*mm\s*(?:DOI|depth)\b',                # DOI with measurement
    # Node characteristics
    r'\blymph\s*node[s]?\b',                        # Lymph nodes
    r'\badenopathy\b',                              # Adenopathy
    r'\bipsilateral\b',                             # Ipsilateral nodes
    r'\bcontralateral\b',                           # Contralateral nodes
    r'\bextranodal\s*extension\b',                  # ENE
    r'\bENE\b',                                     # ENE abbreviation
    r'\bextracapsular\b',                           # Extracapsular extension
    r'\blevel\s*[IVab]+\b',                         # Neck levels
    # Invasion patterns
    r'\binvasion\b',                                # General invasion
    r'\bperineural\b',                              # PNI
    r'\blymphovascular\b',                          # LVI
    r'\bmargin[s]?\b',                              # Margins
    # Metastatic workup
    r'\bmetastatic\s*work[- ]?up\b',               # Metastatic workup
    r'\bM0\b',                                      # Explicit M0
    r'\bno\s*(?:distant\s*)?metast\w*\b',          # No metastasis
]


@dataclass
class StagingDetectionResult:
    """Result of staging keyword detection."""
    has_explicit_staging: bool = False
    has_clinical_indicators: bool = False
    detected_staging_keywords: List[str] = field(default_factory=list)
    detected_clinical_indicators: List[str] = field(default_factory=list)
    needs_rag_fallback: bool = False
    confidence: float = 0.0


def detect_staging_keywords(query: str) -> StagingDetectionResult:
    """
    Detect whether a query has explicit staging vs clinical indicators that need inference.
    
    Args:
        query: The user's query text
        
    Returns:
        StagingDetectionResult with detection details
    """
    result = StagingDetectionResult()
    query_lower = query.lower()
    
    # Check for explicit staging patterns
    for pattern in EXPLICIT_STAGING_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            result.has_explicit_staging = True
            result.detected_staging_keywords.extend(matches)
    
    # Check for clinical indicators
    for pattern in CLINICAL_STAGING_INDICATORS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            result.has_clinical_indicators = True
            result.detected_clinical_indicators.extend(matches)
    
    # Determine if RAG fallback is needed
    # Need fallback if: clinical indicators present BUT no explicit staging
    result.needs_rag_fallback = (
        result.has_clinical_indicators and 
        not result.has_explicit_staging
    )
    
    # Calculate confidence based on number of clinical indicators
    if result.needs_rag_fallback:
        indicator_count = len(result.detected_clinical_indicators)
        result.confidence = min(0.9, 0.3 + (indicator_count * 0.1))
    
    return result


# =============================================================================
# RAG STAGING RETRIEVAL
# =============================================================================

async def retrieve_staging_context(
    query: str,
    cancer_type: Optional[str] = None,
    cancer_location: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve AJCC staging criteria from the knowledge base.
    
    Args:
        query: Original user query
        cancer_type: Detected cancer type (e.g., "SCC")
        cancer_location: Detected location (e.g., "oral tongue")
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant staging context chunks
    """
    try:
        from src.api.services.enhanced_rag_service import get_enhanced_rag_service
        
        rag_service = get_enhanced_rag_service()
        
        # Build a staging-focused query
        staging_query_parts = ["AJCC 8th edition staging criteria"]
        
        if cancer_type:
            staging_query_parts.append(cancer_type)
        if cancer_location:
            staging_query_parts.append(cancer_location)
        
        # Add specific staging aspects from the query
        staging_aspects = []
        query_lower = query.lower()
        
        if "depth" in query_lower or "doi" in query_lower:
            staging_aspects.append("depth of invasion T staging")
        if "node" in query_lower or "adenopathy" in query_lower or "lymph" in query_lower:
            staging_aspects.append("nodal staging N stage")
        if "extranodal" in query_lower or "ene" in query_lower or "extracapsular" in query_lower:
            staging_aspects.append("extranodal extension N3b")
        if "metast" in query_lower:
            staging_aspects.append("metastatic M staging")
        
        if staging_aspects:
            staging_query_parts.extend(staging_aspects)
        
        staging_query = " ".join(staging_query_parts)
        
        print(f"[StagingRAG] Retrieving staging context with query: {staging_query[:100]}...")
        
        # Use the retriever directly for faster results
        retriever = rag_service.retriever
        
        # Get embedding
        query_embedding = retriever.embed_query(staging_query)
        
        # Search Qdrant using query_points (new API)
        from qdrant_client.http import models as qm
        
        results = retriever.qdrant_client.query_points(
            collection_name=retriever.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qm.Filter(
                should=[
                    qm.FieldCondition(
                        key="category",
                        match=qm.MatchValue(value="guidelines")
                    ),
                    qm.FieldCondition(
                        key="text",
                        match=qm.MatchText(text="AJCC")
                    ),
                    qm.FieldCondition(
                        key="text",
                        match=qm.MatchText(text="staging")
                    ),
                ]
            )
        ).points
        
        # Convert to evidence format
        evidence = []
        for hit in results:
            payload = hit.payload or {}
            evidence.append({
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "text": payload.get("text", ""),
                "score": hit.score,
                "category": payload.get("category"),
            })
        
        print(f"[StagingRAG] Retrieved {len(evidence)} staging context chunks")
        return evidence
        
    except Exception as e:
        logger.error(f"[StagingRAG] Failed to retrieve staging context: {e}")
        import traceback
        traceback.print_exc()
        return []


# =============================================================================
# LLM STAGING INFERENCE
# =============================================================================

STAGING_INFERENCE_PROMPT = """You are an expert oncologist specializing in cancer staging using AJCC 8th Edition criteria.

Given the patient's clinical details and the AJCC staging reference material, determine the correct TNM staging and overall stage group.

PATIENT CLINICAL DETAILS:
{clinical_details}

AJCC STAGING REFERENCE:
{staging_context}

INSTRUCTIONS:
1. Analyze the tumor characteristics to determine T stage
2. Analyze the nodal involvement to determine N stage (pay attention to ENE/extranodal extension which upgrades to N3b in H&N)
3. Determine M stage based on metastatic workup
4. Combine T, N, M to determine the overall stage group

CRITICAL STAGING RULES:
- For oral cavity/H&N: DOI >10mm = at least T3; ENE present = N3b
- N3b specifically indicates extranodal extension (ENE) in head and neck cancers
- Multiple ipsilateral nodes ≤6cm without ENE = N2b; with ENE = N3b
- Negative metastatic workup = M0

Respond with ONLY a JSON object:
{{
    "tnm_t": "T stage (e.g., T2)",
    "tnm_n": "N stage (e.g., N3b)",
    "tnm_m": "M stage (e.g., M0)",
    "overall_stage": "Stage group (e.g., IVB)",
    "reasoning": "Brief explanation of staging rationale",
    "confidence": "high/medium/low"
}}"""


async def infer_stage_from_context(
    query: str,
    staging_context: List[Dict[str, Any]],
    cancer_type: Optional[str] = None,
    cancer_location: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to infer staging from clinical details and retrieved AJCC context.
    
    Args:
        query: Original user query with clinical details
        staging_context: Retrieved AJCC staging chunks
        cancer_type: Detected cancer type
        cancer_location: Detected cancer location
        
    Returns:
        Dict with inferred staging or None if inference fails
    """
    if not staging_context:
        logger.warning("[StagingRAG] No staging context available for inference")
        return None
    
    try:
        from openai import OpenAI
        from src.core.config import settings
        import json
        
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Build clinical details string
        clinical_details = query
        if cancer_type:
            clinical_details = f"Cancer type: {cancer_type}\n{clinical_details}"
        if cancer_location:
            clinical_details = f"Location: {cancer_location}\n{clinical_details}"
        
        # Build staging context string
        context_text = "\n\n".join([
            f"Source: {c.get('title', 'Unknown')}\n{c.get('text', '')}"
            for c in staging_context[:3]  # Use top 3 chunks
        ])
        
        prompt = STAGING_INFERENCE_PROMPT.format(
            clinical_details=clinical_details,
            staging_context=context_text
        )
        
        print(f"[StagingRAG] Inferring stage with LLM...")
        
        from src.core.config import settings
        response = client.chat.completions.create(
            model=settings.openai_mini_model,
            messages=[
                {"role": "system", "content": "You are an expert oncologist. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        
        result = json.loads(content)
        
        print(f"[StagingRAG] Inferred staging: T={result.get('tnm_t')}, N={result.get('tnm_n')}, M={result.get('tnm_m')} -> Stage {result.get('overall_stage')}")
        
        return result
        
    except Exception as e:
        logger.error(f"[StagingRAG] LLM staging inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN FALLBACK FUNCTION
# =============================================================================

async def staging_rag_fallback(
    query: str,
    cancer_type: Optional[str] = None,
    cancer_location: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main entry point for staging RAG fallback.
    
    Checks if the query needs staging inference, retrieves AJCC context,
    and uses LLM to determine the correct stage.
    
    Args:
        query: User query with clinical details
        cancer_type: Detected cancer type (optional)
        cancer_location: Detected cancer location (optional)
        
    Returns:
        Dict with staging info or None if not applicable/failed
        {
            "tnm_t": "T2",
            "tnm_n": "N3b", 
            "tnm_m": "M0",
            "overall_stage": "IVB",
            "source": "rag_fallback",
            "confidence": "high"
        }
    """
    # Step 1: Check if fallback is needed
    detection = detect_staging_keywords(query)
    
    if not detection.needs_rag_fallback:
        if detection.has_explicit_staging:
            print(f"[StagingRAG] Explicit staging found: {detection.detected_staging_keywords}")
        else:
            print(f"[StagingRAG] No clinical staging indicators found")
        return None
    
    print(f"[StagingRAG] Fallback needed - clinical indicators: {detection.detected_clinical_indicators}")
    
    # Step 2: Retrieve staging context
    staging_context = await retrieve_staging_context(
        query=query,
        cancer_type=cancer_type,
        cancer_location=cancer_location,
        top_k=5
    )
    
    if not staging_context:
        print("[StagingRAG] No staging context retrieved, skipping inference")
        return None
    
    # Step 3: Infer staging with LLM
    result = await infer_stage_from_context(
        query=query,
        staging_context=staging_context,
        cancer_type=cancer_type,
        cancer_location=cancer_location,
    )
    
    if result:
        result["source"] = "rag_fallback"
        result["detection_confidence"] = detection.confidence
    
    return result


# =============================================================================
# SYNC WRAPPER FOR USE IN QUERY CLASSIFIER
# =============================================================================

def staging_rag_fallback_sync(
    query: str,
    cancer_type: Optional[str] = None,
    cancer_location: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for staging_rag_fallback.
    
    For use in synchronous code paths like query_classifier_service.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    staging_rag_fallback(query, cancer_type, cancer_location)
                )
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(
                staging_rag_fallback(query, cancer_type, cancer_location)
            )
    except Exception as e:
        logger.error(f"[StagingRAG] Sync wrapper failed: {e}")
        return None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    test_queries = [
        # Should trigger fallback (clinical details, no explicit staging)
        "A patient presents with a right lateral oral tongue SCC, 1.5 cm in size, 8mm depth of invasion, and ipsilateral adenopathy in level Ib (2 cm) and IIa (2.5cm). There is overt extranodal extension of the 1b node. Metastatic work-up is negative. What is the clinical stage?",
        
        # Should NOT trigger fallback (has explicit staging)
        "What is the treatment for T2N1M0 breast cancer?",
        
        # Should NOT trigger fallback (has stage group)
        "What is the prognosis for Stage IIB lung cancer?",
        
        # Should trigger fallback (clinical details only)
        "Patient with 4cm breast tumor and 3 positive axillary nodes, what stage?",
    ]
    
    async def test():
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"QUERY: {query[:100]}...")
            print(f"{'='*80}")
            
            detection = detect_staging_keywords(query)
            print(f"Has explicit staging: {detection.has_explicit_staging}")
            print(f"Has clinical indicators: {detection.has_clinical_indicators}")
            print(f"Needs RAG fallback: {detection.needs_rag_fallback}")
            
            if detection.needs_rag_fallback:
                result = await staging_rag_fallback(query)
                if result:
                    print(f"Inferred: T={result.get('tnm_t')}, N={result.get('tnm_n')}, M={result.get('tnm_m')}")
                    print(f"Stage: {result.get('overall_stage')}")
                    print(f"Reasoning: {result.get('reasoning')}")
    
    asyncio.run(test())
