"""
User Document Search Service

Searches user-uploaded documents stored in PostgreSQL.
Uses hybrid search combining semantic (cosine similarity) and keyword (BM25) search.
"""

import json
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from .account_db import get_account_db


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def simple_bm25_score(
    query_terms: List[str], 
    doc_terms: Counter, 
    doc_len: int, 
    avg_doc_len: float, 
    doc_freq: Counter, 
    total_docs: int, 
    k1: float = 1.5, 
    b: float = 0.75
) -> float:
    """Simple BM25 scoring without external libraries."""
    score = 0.0
    for term in query_terms:
        if term in doc_terms:
            tf = doc_terms[term]
            idf = max(0.1, math.log((total_docs - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5)))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    return score


def keyword_search(query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
    """Simple BM25-style keyword search."""
    query_terms = query.lower().split()
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Calculate document frequencies
    doc_freq = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freq[term] += 1
    
    # Calculate average document length
    if not tokenized_docs:
        return []
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / len(tokenized_docs)
    
    # Score each document
    scores = []
    for i, doc in enumerate(tokenized_docs):
        doc_terms = Counter(doc)
        score = simple_bm25_score(query_terms, doc_terms, len(doc), avg_doc_len, doc_freq, len(documents))
        scores.append((i, score))
    
    # Sort and return top results
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def calculate_keyword_density(query: str, chunk_text: str) -> float:
    """Calculate keyword density for better chunk scoring."""
    query_words = set(w.lower() for w in query.split() if len(w) > 2)
    chunk_words = chunk_text.lower().split()
    
    if not query_words or not chunk_words:
        return 0.0
    
    # Count query word occurrences
    matches = sum(1 for word in chunk_words if word in query_words)
    density = matches / (len(chunk_words) + 1)  # +1 for smoothing
    
    # Bonus for multiple different query words
    unique_matches = len(set(chunk_words) & query_words)
    diversity_bonus = unique_matches / len(query_words) if query_words else 0
    
    return density + (diversity_bonus * 0.3)


def enhanced_chunk_scoring(results: List[Dict], query: str) -> List[Dict]:
    """Apply enhanced scoring to chunks."""
    for result in results:
        # Original similarity score
        base_score = result.get('similarity_score', 0)
        
        # Keyword density boost
        keyword_density = calculate_keyword_density(query, result.get('text', ''))
        
        # Length penalty (very short/long chunks less useful)
        text_len = len(result.get('text', '').split())
        if text_len < 20:
            length_penalty = 0.8
        elif text_len > 500:
            length_penalty = 0.9
        else:
            length_penalty = 1.0
        
        # Apply enhanced scoring
        enhanced_score = base_score * length_penalty * (1 + keyword_density)
        result['similarity_score'] = enhanced_score
    
    return sorted(results, key=lambda x: x.get('similarity_score', 0), reverse=True)


def rerank_by_diversity(results: List[Dict], diversity_weight: float = 0.3) -> List[Dict]:
    """Rerank results to balance relevance and diversity."""
    if len(results) <= 1:
        return results
    
    reranked = [results[0]]  # Always include top result
    remaining = results[1:]
    
    while remaining:
        best_idx = 0
        best_score = -1
        
        for i, candidate in enumerate(remaining):
            # Calculate diversity penalty
            diversity_penalty = 0
            for selected in reranked:
                candidate_words = set(candidate.get('text', '').split())
                selected_words = set(selected.get('text', '').split())
                if candidate_words and selected_words:
                    text_overlap = len(candidate_words & selected_words) / len(candidate_words | selected_words)
                    diversity_penalty += text_overlap
            
            # Combined score
            combined_score = candidate.get('similarity_score', 0) - diversity_weight * diversity_penalty
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
        
        reranked.append(remaining.pop(best_idx))
    
    return reranked


class UserDocumentSearchService:
    """Service for searching user-uploaded documents using hybrid search."""
    
    def __init__(self):
        self._embedding_cache = {}  # Cache loaded embeddings per user
    
    async def search_user_documents(
        self,
        user_id: str,
        query_embedding: List[float],
        query_text: str = "",
        top_k: int = 10,
        alpha: float = 0.7  # Weight for semantic vs keyword search
    ) -> List[Dict[str, Any]]:
        """
        Search user's uploaded documents using hybrid search.
        
        Combines:
        - Semantic search (cosine similarity on embeddings)
        - Keyword search (BM25)
        - Enhanced scoring (keyword density, length penalty)
        - Diversity reranking
        
        Args:
            user_id: User ID
            query_embedding: Query embedding vector
            query_text: Original query text for keyword search
            top_k: Number of top results to return
            alpha: Weight for semantic search (1-alpha for keyword search)
            
        Returns:
            List of matching chunks with scores
        """
        # Load user's documents with embeddings
        user_docs = await self._load_user_embeddings(user_id)
        
        if not user_docs:
            return []
        
        # Convert query to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # Collect all chunks across all user documents
        all_chunks = []
        all_embeddings = []
        
        for doc in user_docs:
            embeddings = doc.get("embeddings")  # numpy array (N, dim)
            chunk_metadata = doc.get("chunk_metadata", [])
            
            if embeddings is None or len(chunk_metadata) == 0:
                continue
            
            for i, chunk in enumerate(chunk_metadata):
                if i < len(embeddings):
                    all_chunks.append({
                        "doc_id": doc.get("doc_id") or doc.get("upload_id"),
                        "upload_id": doc.get("upload_id"),
                        "title": doc.get("title"),
                        "filename": doc.get("filename"),
                        "chunk_id": chunk.get("id"),
                        "text": chunk.get("text", ""),
                        "section": chunk.get("section"),
                        "chunk_type": chunk.get("chunk_type"),
                        "source": "user_upload",
                        "chunk_index": len(all_embeddings)
                    })
                    all_embeddings.append(embeddings[i])
        
        if not all_chunks:
            return []
        
        # Convert to numpy array for vectorized operations
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        
        # Step 1: Semantic search (cosine similarity)
        semantic_scores = self._compute_semantic_scores(query_vec, embeddings_matrix)
        
        # Step 2: Keyword search (BM25)
        documents = [chunk['text'] for chunk in all_chunks]
        keyword_results = keyword_search(query_text, documents, len(documents)) if query_text else []
        
        # Normalize BM25 scores to [0, 1]
        keyword_scores = {}
        if keyword_results:
            max_bm25 = max(score for _, score in keyword_results) if keyword_results else 1
            if max_bm25 > 0:
                for idx, score in keyword_results:
                    keyword_scores[idx] = score / max_bm25
        
        # Step 3: Combine scores (hybrid)
        for i, chunk in enumerate(all_chunks):
            semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0
            keyword_score = keyword_scores.get(i, 0)
            
            # Hybrid score
            combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
            chunk['similarity_score'] = combined_score
        
        # Step 4: Enhanced scoring (keyword density, length penalty)
        if query_text:
            all_chunks = enhanced_chunk_scoring(all_chunks, query_text)
        else:
            all_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Step 5: Cross-encoder reranking (same as Qdrant pipeline)
        # This ensures scores are on the same scale as Qdrant results
        top_candidates = all_chunks[:min(len(all_chunks), 50)]  # Rerank top 50
        reranked = self._cross_encoder_rerank(top_candidates, query_text, top_k=top_k)
        
        # Step 6: Diversity reranking
        reranked = rerank_by_diversity(reranked, diversity_weight=0.2)
        
        return reranked[:top_k]
    
    def _cross_encoder_rerank(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking to user upload chunks."""
        if not chunks or not query:
            return chunks
        
        try:
            from src.api.services.enhanced_rag_service import get_cross_encoder
            
            cross_encoder = get_cross_encoder()
            if cross_encoder is None:
                return chunks
            
            # Prepare text pairs for cross-encoder
            texts = [c.get("text", "")[:512] for c in chunks]
            pairs = [(query, text) for text in texts]
            
            # Get cross-encoder scores
            scores = cross_encoder.predict(pairs)
            
            # Update chunks with cross-encoder scores
            for i, chunk in enumerate(chunks):
                chunk["similarity_score"] = float(scores[i])
            
            # Sort by cross-encoder score
            reranked = sorted(chunks, key=lambda x: x.get("similarity_score", 0), reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            print(f"[UserDocSearch] Cross-encoder rerank failed: {e}")
            return chunks
    
    def _compute_semantic_scores(
        self, 
        query_vec: np.ndarray, 
        embeddings_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity scores for all embeddings."""
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(embeddings_matrix))
        query_normalized = query_vec / query_norm
        
        # Normalize all embeddings
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings_matrix / norms
        
        # Compute cosine similarities (vectorized)
        similarities = np.dot(embeddings_normalized, query_normalized)
        
        return similarities
    
    async def _load_user_embeddings(self, user_id: str) -> List[Dict[str, Any]]:
        """Load all user documents with embeddings."""
        # Check cache first
        if user_id in self._embedding_cache:
            return self._embedding_cache[user_id]
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT upload_id, doc_id, filename, title,
                       embeddings, embedding_dim, chunk_metadata, chunk_count
                FROM user_uploads
                WHERE user_id = $1 AND status = 'completed'
                ORDER BY created_at DESC
            """, user_id)
            
            results = []
            for row in rows:
                embeddings_array = None
                if row["embeddings"] and row["embedding_dim"] and row["chunk_count"]:
                    # Deserialize embeddings from binary
                    embeddings_array = np.frombuffer(
                        row["embeddings"], 
                        dtype=np.float32
                    ).reshape(row["chunk_count"], row["embedding_dim"])
                
                chunk_metadata = row["chunk_metadata"]
                if isinstance(chunk_metadata, str):
                    chunk_metadata = json.loads(chunk_metadata)
                
                results.append({
                    "upload_id": row["upload_id"],
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "title": row["title"],
                    "embeddings": embeddings_array,
                    "chunk_metadata": chunk_metadata or [],
                    "chunk_count": row["chunk_count"],
                    "embedding_dim": row["embedding_dim"],
                })
            
            # Cache the results
            self._embedding_cache[user_id] = results
            return results
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear embedding cache for a user or all users."""
        if user_id:
            self._embedding_cache.pop(user_id, None)
        else:
            self._embedding_cache.clear()


# Singleton instance
_user_doc_search_service: Optional[UserDocumentSearchService] = None


def get_user_document_search_service() -> UserDocumentSearchService:
    global _user_doc_search_service
    if _user_doc_search_service is None:
        _user_doc_search_service = UserDocumentSearchService()
    return _user_doc_search_service
