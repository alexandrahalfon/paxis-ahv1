"""
Study-Focused Retrieval Service

Implements a two-phase retrieval approach:
1. Phase 1: Initial retrieval to identify relevant studies (doc_ids)
2. Phase 2: For each relevant study, run a targeted query to get comprehensive 
   chunks from that specific study, ensuring complete coverage of relevant sections.

This addresses the problem of fragmented chunks from various sections of documents
by consolidating information per-study before synthesis.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from qdrant_client import QdrantClient
import qdrant_client.models as qm
from openai import OpenAI
import time

from src.core.config import settings


@dataclass
class StudyEvidence:
    """Consolidated evidence from a single study."""
    doc_id: str
    title: str
    citation: Optional[str]
    year: Optional[int]
    category: Optional[str]
    relevance_score: float  # From initial retrieval
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    sections_covered: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "citation": self.citation,
            "year": self.year,
            "category": self.category,
            "relevance_score": self.relevance_score,
            "chunks": self.chunks,
            "sections_covered": list(self.sections_covered),
            "chunk_count": len(self.chunks),
        }


@dataclass
class StudyFocusedRetrievalResult:
    """Result of study-focused retrieval."""
    studies: List[StudyEvidence]
    total_chunks: int
    retrieval_time_ms: float
    phase1_doc_count: int
    phase2_chunks_per_study: Dict[str, int]
    query_structure: Optional[Dict[str, Any]] = None  # For context accumulation
    clinical_profile: Optional[Dict[str, Any]] = None
    structured_match_count: int = 0  # PostgreSQL matches
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks flattened, maintaining study grouping info."""
        all_chunks = []
        for study in self.studies:
            for chunk in study.chunks:
                chunk_with_study = {
                    **chunk,
                    "_study_title": study.title,
                    "_study_relevance": study.relevance_score,
                }
                all_chunks.append(chunk_with_study)
        return all_chunks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "studies": [s.to_dict() for s in self.studies],
            "total_chunks": self.total_chunks,
            "retrieval_time_ms": self.retrieval_time_ms,
            "phase1_doc_count": self.phase1_doc_count,
            "phase2_chunks_per_study": self.phase2_chunks_per_study,
            "query_structure": self.query_structure,
            "clinical_profile": self.clinical_profile,
            "structured_match_count": self.structured_match_count,
        }


class StudyFocusedRetriever:
    """
    Two-phase retriever that consolidates evidence per-study.
    
    Phase 1: Standard retrieval to identify relevant studies
    Phase 2: For each study, retrieve comprehensive chunks matching the query
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        collection: Optional[str] = None,
    ):
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.collection = collection or settings.qdrant_collection
        self._embed_model = settings.embed_model
    
    def embed_query(self, query_text: str) -> List[float]:
        """Generate embedding for query."""
        response = self.openai.embeddings.create(
            model=self._embed_model,
            input=query_text,
        )
        return response.data[0].embedding
    
    async def retrieve_study_focused(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        max_studies: int = 5,
        chunks_per_study: int = 8,
        min_relevance_score: float = 0.3,
        category: Optional[str] = None,
        initial_pool_size: int = 50,
        accumulated_context: Optional[Dict[str, Any]] = None,
    ) -> StudyFocusedRetrievalResult:
        """
        Two-phase retrieval that consolidates evidence per-study.
        
        Args:
            query_text: The user's query
            query_embedding: Pre-computed embedding (optional, will compute if not provided)
            max_studies: Maximum number of studies to include
            chunks_per_study: Maximum chunks to retrieve per study in phase 2
            min_relevance_score: Minimum score threshold for phase 1
            category: Optional category filter
            initial_pool_size: Size of initial retrieval pool
            accumulated_context: Optional accumulated structured context from previous queries
            
        Returns:
            StudyFocusedRetrievalResult with consolidated per-study evidence
        """
        t_start = time.perf_counter()
        
        # Get embedding if not provided
        if query_embedding is None:
            query_embedding = self.embed_query(query_text)
        
        # =====================================================
        # STEP 0: Extract clinical profile and query structure
        # =====================================================
        clinical_profile = None
        query_structure = None
        
        try:
            from src.api.services.clinical_entity_extractor import get_clinical_entity_extractor
            extractor = get_clinical_entity_extractor()
            profile = extractor.extract(query_text)
            clinical_profile = {
                "must_match": extractor.get_must_match_terms(profile),
                "should_match": extractor.get_should_match_terms(profile),
                "raw_profile": profile.to_dict(),
            }
            print(f"[StudyFocusedRetrieval] Clinical profile extracted: {list(clinical_profile['raw_profile'].keys())}")
        except Exception as e:
            print(f"[StudyFocusedRetrieval] Clinical extraction failed: {e}")
        
        try:
            from src.api.services.query_structuring_service import structure_query_fast, merge_query_structures
            from src.api.services.enhanced_rag_service import classify_query
            
            query_classification = classify_query(query_text)
            query_type = query_classification.get("primary_type", "general")
            query_structure = structure_query_fast(query_text, query_type)
            
            # Merge with accumulated context from conversation
            if accumulated_context:
                query_structure = merge_query_structures(accumulated_context, query_structure)
                print(f"[StudyFocusedRetrieval] Merged with accumulated context")
            
            if query_structure.has_patient_context:
                print(f"[StudyFocusedRetrieval] Patient context detected: "
                      f"site={query_structure.cancer.site}, "
                      f"tnm={query_structure.cancer.get_tnm_string()}")
        except Exception as e:
            print(f"[StudyFocusedRetrieval] Query structuring failed: {e}")
        
        # =====================================================
        # PHASE 1: Initial retrieval + PostgreSQL parallel search
        # =====================================================
        
        # Start PostgreSQL structured matching in parallel
        structured_match_task = None
        structured_result = None
        
        if query_structure:
            try:
                from src.api.services.structured_study_matcher import match_studies_by_structure
                query_structure_dict = query_structure.to_dict()
                structured_match_task = match_studies_by_structure(query_structure_dict, limit=50)
                print(f"[StudyFocusedRetrieval] Starting parallel PostgreSQL matching...")
            except Exception as e:
                print(f"[StudyFocusedRetrieval] PostgreSQL setup failed: {e}")
        
        # Run Qdrant + PostgreSQL in parallel
        import asyncio
        if structured_match_task:
            phase1_task = self._phase1_identify_studies(
                query_embedding=query_embedding,
                category=category,
                pool_size=initial_pool_size,
                min_score=min_relevance_score,
            )
            phase1_results, structured_result = await asyncio.gather(
                phase1_task,
                structured_match_task
            )
            
            if structured_result and structured_result.doc_ids:
                print(f"[StudyFocusedRetrieval] PostgreSQL matched {len(structured_result.doc_ids)} studies")
        else:
            phase1_results = await self._phase1_identify_studies(
                query_embedding=query_embedding,
                category=category,
                pool_size=initial_pool_size,
                min_score=min_relevance_score,
            )
        
        # Extract unique doc_ids with their best scores
        study_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for hit in phase1_results:
            doc_id = hit.get("doc_id")
            if not doc_id:
                continue
            score = hit.get("score", 0)
            if doc_id not in study_scores or score > study_scores[doc_id][0]:
                study_scores[doc_id] = (score, hit)
        
        # Boost studies that matched PostgreSQL structured search
        if structured_result and structured_result.doc_ids:
            pg_doc_ids = set(structured_result.doc_ids)
            for doc_id in study_scores:
                if doc_id in pg_doc_ids:
                    current_score, hit = study_scores[doc_id]
                    # Apply 30% boost for PostgreSQL matches
                    boosted_score = current_score * 1.3
                    study_scores[doc_id] = (boosted_score, hit)
                    print(f"[StudyFocusedRetrieval] Boosted {doc_id}: {current_score:.3f} -> {boosted_score:.3f}")
            
            # Also add PostgreSQL-only matches that weren't in Qdrant results
            for pg_doc_id in pg_doc_ids:
                if pg_doc_id not in study_scores:
                    # Add with a baseline score
                    study_scores[pg_doc_id] = (0.5, {"doc_id": pg_doc_id, "doc_meta": {}, "category": None})
                    print(f"[StudyFocusedRetrieval] Added PostgreSQL-only match: {pg_doc_id}")
        
        # Sort by score and take top studies
        sorted_studies = sorted(
            study_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:max_studies]
        
        phase1_doc_count = len(sorted_studies)
        print(f"[StudyFocusedRetrieval] Phase 1: Identified {phase1_doc_count} relevant studies")
        
        # =====================================================
        # PHASE 2: Retrieve comprehensive chunks per study
        # =====================================================
        studies: List[StudyEvidence] = []
        phase2_chunks_per_study: Dict[str, int] = {}
        
        for doc_id, (score, initial_hit) in sorted_studies:
            study_chunks = await self._phase2_retrieve_study_chunks(
                doc_id=doc_id,
                query_embedding=query_embedding,
                query_text=query_text,
                max_chunks=chunks_per_study,
            )
            
            # Extract metadata from initial hit
            doc_meta = initial_hit.get("doc_meta", {})
            
            # Build StudyEvidence
            sections_covered = set()
            for chunk in study_chunks:
                section = chunk.get("section")
                if section:
                    sections_covered.add(section)
            
            study = StudyEvidence(
                doc_id=doc_id,
                title=doc_meta.get("title", "Unknown"),
                citation=doc_meta.get("citation"),
                year=doc_meta.get("year"),
                category=initial_hit.get("category"),
                relevance_score=score,
                chunks=study_chunks,
                sections_covered=sections_covered,
            )
            studies.append(study)
            phase2_chunks_per_study[doc_id] = len(study_chunks)
            
            print(f"[StudyFocusedRetrieval] Phase 2: {doc_id} -> {len(study_chunks)} chunks, "
                  f"sections: {list(sections_covered)[:3]}...")
        
        total_chunks = sum(len(s.chunks) for s in studies)
        retrieval_time_ms = (time.perf_counter() - t_start) * 1000
        
        print(f"[StudyFocusedRetrieval] Complete: {len(studies)} studies, "
              f"{total_chunks} total chunks in {retrieval_time_ms:.1f}ms")
        
        return StudyFocusedRetrievalResult(
            studies=studies,
            total_chunks=total_chunks,
            retrieval_time_ms=retrieval_time_ms,
            phase1_doc_count=phase1_doc_count,
            phase2_chunks_per_study=phase2_chunks_per_study,
            query_structure=query_structure.to_dict() if query_structure else None,
            clinical_profile=clinical_profile,
            structured_match_count=len(structured_result.doc_ids) if structured_result else 0,
        )
    
    async def _phase1_identify_studies(
        self,
        query_embedding: List[float],
        category: Optional[str],
        pool_size: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """Phase 1: Initial vector search to identify relevant studies."""
        # Build filter
        flt = None
        if category:
            flt = qm.Filter(must=[
                qm.FieldCondition(key="category", match=qm.MatchValue(value=category))
            ])
        
        # Query Qdrant
        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=pool_size,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )
        
        # Convert to list of dicts with score filtering
        hits = []
        for point in results.points:
            if point.score < min_score:
                continue
            payload = dict(point.payload or {})
            hits.append({
                "point_id": point.id,
                "score": float(point.score),
                "doc_id": payload.get("doc_id"),
                "doc_meta": payload.get("doc_meta", {}),
                "category": payload.get("category"),
                "section": payload.get("section"),
                "text": payload.get("text", ""),
            })
        
        return hits
    
    async def _phase2_retrieve_study_chunks(
        self,
        doc_id: str,
        query_embedding: List[float],
        query_text: str,
        max_chunks: int,
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Retrieve comprehensive chunks from a specific study.
        
        Uses vector search within the study to find the most relevant chunks,
        ensuring we get comprehensive coverage of the query topic within this study.
        """
        # Filter to only this document
        doc_filter = qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
        ])
        
        # Query for chunks within this document
        # Request more than max_chunks to allow for deduplication
        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=max_chunks * 2,
            query_filter=doc_filter,
            with_payload=True,
            with_vectors=False,
        )
        
        # Convert and deduplicate
        chunks = []
        seen_texts = set()
        
        for point in results.points:
            payload = dict(point.payload or {})
            text = payload.get("text", "")
            
            # Skip near-duplicate text
            text_key = text[:200].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            
            chunk = {
                "point_id": point.id,
                "score": float(point.score),
                "doc_id": doc_id,
                "text": text,
                "section": payload.get("section"),
                "chunk_type": payload.get("chunk_type"),
                "chunk_id": payload.get("chunk_id"),
                "section_window_idx": payload.get("section_window_idx"),
                "doc_meta": payload.get("doc_meta", {}),
                "category": payload.get("category"),
            }
            
            # Include table metadata if present
            if payload.get("chunk_type") == "table_row":
                chunk["table"] = {
                    "number": payload.get("table_number"),
                    "title": payload.get("table_title"),
                    "row_index": payload.get("row_index"),
                    "headers": (payload.get("metadata") or {}).get("headers", []),
                    "raw_row": (payload.get("metadata") or {}).get("raw_row", []),
                }
            
            chunks.append(chunk)
            
            if len(chunks) >= max_chunks:
                break
        
        return chunks


# Singleton instance
_retriever_instance: Optional[StudyFocusedRetriever] = None


def get_study_focused_retriever() -> StudyFocusedRetriever:
    """Get singleton StudyFocusedRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        from qdrant_client import QdrantClient
        from openai import OpenAI
        
        qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        _retriever_instance = StudyFocusedRetriever(
            qdrant_client=qdrant,
            openai_client=openai_client,
        )
    return _retriever_instance


def format_study_grouped_evidence(
    result: StudyFocusedRetrievalResult,
    max_chunks_total: int = 12,
) -> List[Dict[str, Any]]:
    """
    Format study-focused retrieval results for synthesis.
    
    Groups chunks by study and ensures comprehensive coverage while
    respecting token limits.
    
    Args:
        result: StudyFocusedRetrievalResult from retrieve_study_focused
        max_chunks_total: Maximum total chunks to include
        
    Returns:
        List of evidence dicts formatted for gpt4o_summary_enhanced
    """
    evidence = []
    chunks_remaining = max_chunks_total
    
    # Distribute chunks across studies proportionally to relevance
    total_relevance = sum(s.relevance_score for s in result.studies)
    
    for study in result.studies:
        if chunks_remaining <= 0:
            break
        
        # Calculate chunks for this study (proportional to relevance, min 1)
        if total_relevance > 0:
            study_share = study.relevance_score / total_relevance
            study_chunks_limit = max(1, int(chunks_remaining * study_share * 1.5))
        else:
            study_chunks_limit = max(1, chunks_remaining // len(result.studies))
        
        # Add chunks from this study
        for chunk in study.chunks[:study_chunks_limit]:
            if chunks_remaining <= 0:
                break
            
            evidence_item = {
                "doc_id": study.doc_id,
                "title": study.title,
                "citation": study.citation,
                "year": study.year,
                "category": study.category,
                "score": chunk.get("score", study.relevance_score),
                "text": chunk.get("text", ""),
                "section": chunk.get("section"),
                "chunk_type": chunk.get("chunk_type"),
                "chunk_id": chunk.get("chunk_id"),
                "doc_meta": chunk.get("doc_meta", {}),
                # Study grouping metadata
                "_study_relevance": study.relevance_score,
                "_study_chunk_count": len(study.chunks),
                "_sections_in_study": list(study.sections_covered),
            }
            
            # Include table info if present
            if chunk.get("table"):
                evidence_item["table"] = chunk["table"]
            
            evidence.append(evidence_item)
            chunks_remaining -= 1
    
    return evidence


def convert_to_standard_evidence(
    result: StudyFocusedRetrievalResult,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert StudyFocusedRetrievalResult to standard evidence format
    compatible with existing RAG pipeline.
    
    Returns:
        Tuple of (evidence_list, metadata_dict)
    """
    evidence = format_study_grouped_evidence(result)
    
    metadata = {
        "study_focused_retrieval": True,
        "studies_retrieved": len(result.studies),
        "total_chunks": result.total_chunks,
        "retrieval_time_ms": result.retrieval_time_ms,
        "phase1_doc_count": result.phase1_doc_count,
        "chunks_per_study": result.phase2_chunks_per_study,
        "study_summaries": [
            {
                "doc_id": s.doc_id,
                "title": s.title,
                "relevance": s.relevance_score,
                "chunks": len(s.chunks),
                "sections": list(s.sections_covered),
            }
            for s in result.studies
        ],
    }
    
    return evidence, metadata
