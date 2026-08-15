"""
Property-based tests for context boosting in the RAG pipeline.

Feature: unified-rag-pipeline

Tests the following property:
- Property 21: Context boosting applied for follow-ups

Validates: Requirements 2.4, 2.5
"""

from hypothesis import given, strategies as st, settings, assume
import pytest
from typing import List, Dict, Any

from src.api.services.enhanced_rag_service import (
    boost_context_documents,
    CONTEXT_BOOST_ADDEND,
    CONTEXT_BOOST_ADDEND_DENSE,
)


# Strategy for generating valid doc_ids (non-empty strings)
doc_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P')),
    min_size=1,
    max_size=50
).filter(lambda x: len(x.strip()) > 0)


# Strategy for generating positive scores
score_strategy = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)


# Strategy for generating a single chunk with a doc_id and score
def chunk_strategy(doc_id: str = None, score: float = None):
    """Generate a chunk dictionary with optional fixed doc_id and score."""
    return st.fixed_dictionaries({
        "doc_id": st.just(doc_id) if doc_id else doc_id_strategy,
        "score": st.just(score) if score else score_strategy,
        "text": st.text(min_size=10, max_size=100),
    })


# Strategy for generating a list of chunks
chunks_strategy = st.lists(
    st.fixed_dictionaries({
        "doc_id": doc_id_strategy,
        "score": score_strategy,
        "text": st.text(min_size=10, max_size=100),
    }),
    min_size=1,
    max_size=20
)


# Strategy for generating context doc_ids
context_doc_ids_strategy = st.lists(
    doc_id_strategy,
    min_size=0,
    max_size=10
)


# Strategy for boost factor (must be > 1.0 to actually boost)
boost_factor_strategy = st.floats(min_value=1.01, max_value=3.0, allow_nan=False, allow_infinity=False)


# Feature: unified-rag-pipeline, Property 21: Context boosting applied for follow-ups
@settings(max_examples=100)
@given(
    base_score=score_strategy,
    context_doc_id=doc_id_strategy,
    non_context_doc_id=doc_id_strategy,
    boost_factor=boost_factor_strategy,
)
def test_context_boosting_applied_for_follow_ups(
    base_score: float,
    context_doc_id: str,
    non_context_doc_id: str,
    boost_factor: float,
):
    """
    Property 21: Context boosting applied for follow-ups.
    
    For any query with non-empty conversation_context containing previous doc_ids,
    the retrieval results SHALL show boosted scores for documents matching those
    doc_ids compared to the same query without context.
    
    **Validates: Requirements 2.4, 2.5**
    """
    # Ensure doc_ids are different
    assume(context_doc_id.strip() != non_context_doc_id.strip())
    
    # Create chunks - one matching context, one not
    chunks = [
        {"doc_id": context_doc_id, "score": base_score, "text": "context chunk"},
        {"doc_id": non_context_doc_id, "score": base_score, "text": "non-context chunk"},
    ]
    
    context_doc_ids = [context_doc_id]
    
    # Apply context boosting
    result = boost_context_documents(chunks, context_doc_ids, boost_factor)
    
    # Find the chunks in the result
    context_chunk = next(c for c in result if c["doc_id"] == context_doc_id)
    non_context_chunk = next(c for c in result if c["doc_id"] == non_context_doc_id)
    
    # Context chunk should have boosted score
    assert context_chunk["score_context_boost"] > non_context_chunk["score_context_boost"], (
        f"Context chunk score ({context_chunk['score_context_boost']}) should be greater than "
        f"non-context chunk score ({non_context_chunk['score_context_boost']})"
    )
    
    # Context chunk should be marked as from_context
    assert context_chunk.get("from_context") is True, (
        "Context chunk should have from_context=True"
    )


# Feature: unified-rag-pipeline, Property 21a: Chunks with matching doc_ids have scores boosted
@settings(max_examples=100)
@given(
    base_score=score_strategy,
    doc_id=doc_id_strategy,
    boost_factor=boost_factor_strategy,
)
def test_matching_doc_ids_have_scores_boosted(
    base_score: float,
    doc_id: str,
    boost_factor: float,
):
    """
    Property 21a: Chunks with doc_ids matching context_doc_ids have their scores boosted.
    
    The boosted score should equal original_score + CONTEXT_BOOST_ADDEND (additive boost).
    
    **Validates: Requirements 2.4, 2.5**
    """
    chunks = [{"doc_id": doc_id, "score": base_score, "text": "test chunk"}]
    context_doc_ids = [doc_id]
    
    result = boost_context_documents(chunks, context_doc_ids, boost_factor)
    
    # Implementation uses additive boost (not multiplicative) to handle negative cross-encoder scores
    expected_boosted_score = base_score + CONTEXT_BOOST_ADDEND
    actual_boosted_score = result[0]["score_context_boost"]
    
    # Allow small floating point tolerance
    assert abs(actual_boosted_score - expected_boosted_score) < 1e-9, (
        f"Boosted score {actual_boosted_score} should equal "
        f"base_score ({base_score}) + CONTEXT_BOOST_ADDEND ({CONTEXT_BOOST_ADDEND}) = {expected_boosted_score}"
    )
    
    assert result[0].get("from_context") is True, (
        "Chunk with matching doc_id should have from_context=True"
    )


# Feature: unified-rag-pipeline, Property 21b: Non-matching chunks retain original scores
@settings(max_examples=100)
@given(
    base_score=score_strategy,
    chunk_doc_id=doc_id_strategy,
    context_doc_id=doc_id_strategy,
    boost_factor=boost_factor_strategy,
)
def test_non_matching_chunks_retain_original_scores(
    base_score: float,
    chunk_doc_id: str,
    context_doc_id: str,
    boost_factor: float,
):
    """
    Property 21b: Chunks without matching doc_ids retain their original scores.
    
    The score_context_boost should equal the original score for non-matching chunks.
    
    **Validates: Requirements 2.4, 2.5**
    """
    # Ensure doc_ids are different
    assume(chunk_doc_id.strip() != context_doc_id.strip())
    
    chunks = [{"doc_id": chunk_doc_id, "score": base_score, "text": "test chunk"}]
    context_doc_ids = [context_doc_id]
    
    result = boost_context_documents(chunks, context_doc_ids, boost_factor)
    
    # Non-matching chunk should have score_context_boost equal to original score
    assert result[0]["score_context_boost"] == base_score, (
        f"Non-matching chunk score_context_boost ({result[0]['score_context_boost']}) "
        f"should equal original score ({base_score})"
    )
    
    # Non-matching chunk should NOT have from_context=True
    assert result[0].get("from_context") is not True, (
        "Non-matching chunk should not have from_context=True"
    )


# Feature: unified-rag-pipeline, Property 21c: Boost factor correctly applied
@settings(max_examples=100)
@given(
    base_score=score_strategy,
    doc_id=doc_id_strategy,
    boost_factor=boost_factor_strategy,
)
def test_boost_factor_correctly_applied(
    base_score: float,
    doc_id: str,
    boost_factor: float,
):
    """
    Property 21c: The boost is correctly applied as an additive constant.
    
    For matching chunks: boosted_score = original_score + CONTEXT_BOOST_ADDEND
    
    **Validates: Requirements 2.4, 2.5**
    """
    chunks = [{"doc_id": doc_id, "score": base_score, "text": "test chunk"}]
    context_doc_ids = [doc_id]
    
    result = boost_context_documents(chunks, context_doc_ids, boost_factor)
    
    expected = base_score + CONTEXT_BOOST_ADDEND
    actual = result[0]["score_context_boost"]
    
    # Verify the additive boost is correct
    assert abs(actual - expected) < 1e-9, (
        f"Boost not correctly applied: expected {expected}, got {actual}"
    )


# Feature: unified-rag-pipeline, Property 21d: Empty context_doc_ids returns chunks unchanged
@settings(max_examples=100)
@given(chunks=chunks_strategy)
def test_empty_context_doc_ids_returns_chunks_unchanged(chunks: List[Dict[str, Any]]):
    """
    Property 21d: Empty context_doc_ids returns chunks unchanged.
    
    When context_doc_ids is empty, the function should return the original chunks
    without modification.
    
    **Validates: Requirements 2.4, 2.5**
    """
    # Make a deep copy to compare
    original_chunks = [dict(c) for c in chunks]
    
    result = boost_context_documents(chunks, [], 1.2)
    
    # Result should be the same object (or equivalent)
    assert len(result) == len(original_chunks), (
        f"Result length ({len(result)}) should match original ({len(original_chunks)})"
    )
    
    # Chunks should be unchanged
    for i, (orig, res) in enumerate(zip(original_chunks, result)):
        assert orig["doc_id"] == res["doc_id"], (
            f"Chunk {i} doc_id changed from {orig['doc_id']} to {res['doc_id']}"
        )
        assert orig["score"] == res["score"], (
            f"Chunk {i} score changed from {orig['score']} to {res['score']}"
        )


# Feature: unified-rag-pipeline, Property 21e: Empty chunks list returns empty list
@settings(max_examples=100)
@given(context_doc_ids=context_doc_ids_strategy)
def test_empty_chunks_returns_empty_list(context_doc_ids: List[str]):
    """
    Property 21e: Empty chunks list returns empty list.
    
    When chunks is empty, the function should return an empty list.
    
    **Validates: Requirements 2.4, 2.5**
    """
    result = boost_context_documents([], context_doc_ids, 1.2)
    
    assert result == [], (
        f"Empty chunks should return empty list, got {result}"
    )


# Feature: unified-rag-pipeline, Property 21f: Payload doc_id extraction works
@settings(max_examples=100)
@given(
    base_score=score_strategy,
    doc_id=doc_id_strategy,
    boost_factor=boost_factor_strategy,
)
def test_payload_doc_id_extraction(
    base_score: float,
    doc_id: str,
    boost_factor: float,
):
    """
    Property 21f: Doc_id can be extracted from payload.doc_id.
    
    The function should correctly extract doc_id from chunk["payload"]["doc_id"]
    when doc_id is not directly on the chunk.
    
    **Validates: Requirements 2.4, 2.5**
    """
    # Chunk with doc_id in payload (Qdrant-style)
    chunks = [{
        "payload": {"doc_id": doc_id, "title": "Test"},
        "score": base_score,
        "text": "test chunk"
    }]
    context_doc_ids = [doc_id]
    
    result = boost_context_documents(chunks, context_doc_ids, boost_factor)
    
    expected_boosted_score = base_score + CONTEXT_BOOST_ADDEND
    actual_boosted_score = result[0]["score_context_boost"]
    
    assert abs(actual_boosted_score - expected_boosted_score) < 1e-9, (
        f"Payload doc_id not correctly extracted: expected boosted score {expected_boosted_score}, "
        f"got {actual_boosted_score}"
    )
    
    assert result[0].get("from_context") is True, (
        "Chunk with matching payload.doc_id should have from_context=True"
    )


# Feature: unified-rag-pipeline, Property 21g: Multiple context doc_ids work correctly
@settings(max_examples=100)
@given(
    base_score=score_strategy,
    doc_ids=st.lists(doc_id_strategy, min_size=2, max_size=5, unique=True),
    boost_factor=boost_factor_strategy,
)
def test_multiple_context_doc_ids(
    base_score: float,
    doc_ids: List[str],
    boost_factor: float,
):
    """
    Property 21g: Multiple context doc_ids all get boosted.
    
    When multiple doc_ids are in context_doc_ids, all matching chunks
    should be boosted.
    
    **Validates: Requirements 2.4, 2.5**
    """
    # Create chunks for each doc_id
    chunks = [
        {"doc_id": doc_id, "score": base_score, "text": f"chunk {i}"}
        for i, doc_id in enumerate(doc_ids)
    ]
    
    # Use all doc_ids as context
    context_doc_ids = doc_ids
    
    result = boost_context_documents(chunks, context_doc_ids, boost_factor)
    
    expected_boosted_score = base_score + CONTEXT_BOOST_ADDEND
    
    # All chunks should be boosted
    for i, chunk in enumerate(result):
        assert abs(chunk["score_context_boost"] - expected_boosted_score) < 1e-9, (
            f"Chunk {i} not boosted correctly: expected {expected_boosted_score}, "
            f"got {chunk['score_context_boost']}"
        )
        assert chunk.get("from_context") is True, (
            f"Chunk {i} should have from_context=True"
        )
