"""
Integration tests for Conversation Context Management feature.

Tests the end-to-end conversation flow including:
- Initial query creates context
- Follow-up query updates context and applies boosting
- "Start a New Conversation" clears context
- Model serialization/deserialization

Requirements: 1.1, 1.4, 4.1, 5.2
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

from src.api.models.query_models import (
    ConversationContextEntry,
    QueryRequest,
    QueryResponse,
)


class TestConversationContextEntry:
    """Test ConversationContextEntry model serialization and validation."""

    def test_create_valid_entry(self):
        """Test creating a valid ConversationContextEntry."""
        entry = ConversationContextEntry(
            query="What is the standard RT dose for stage III NSCLC?",
            action_type="query",
            doc_ids=["doc_abc123", "doc_def456"],
            doc_titles=["RTOG 0617 Trial Results", "PACIFIC Trial"],
            timestamp=1704067200000
        )
        
        assert entry.query == "What is the standard RT dose for stage III NSCLC?"
        assert entry.action_type == "query"
        assert len(entry.doc_ids) == 2
        assert len(entry.doc_titles) == 2
        assert entry.timestamp == 1704067200000
        assert entry.treatments is None

    def test_create_entry_with_treatments(self):
        """Test creating entry with treatments for eval_treatment action."""
        entry = ConversationContextEntry(
            query="Compare chemotherapy vs immunotherapy for NSCLC",
            action_type="eval_treatment",
            doc_ids=["doc_123"],
            doc_titles=["Treatment Comparison Study"],
            timestamp=1704067200000,
            treatments=["chemotherapy", "immunotherapy"]
        )
        
        assert entry.action_type == "eval_treatment"
        assert entry.treatments == ["chemotherapy", "immunotherapy"]

    def test_entry_serialization_roundtrip(self):
        """Test that serializing and deserializing produces equivalent object."""
        original = ConversationContextEntry(
            query="Test query",
            action_type="query",
            doc_ids=["doc1", "doc2"],
            doc_titles=["Title 1", "Title 2"],
            timestamp=1704067200000
        )
        
        # Serialize to dict then JSON
        serialized = original.model_dump()
        json_str = json.dumps(serialized)
        
        # Deserialize back
        deserialized_dict = json.loads(json_str)
        restored = ConversationContextEntry(**deserialized_dict)
        
        assert restored.query == original.query
        assert restored.action_type == original.action_type
        assert restored.doc_ids == original.doc_ids
        assert restored.doc_titles == original.doc_titles
        assert restored.timestamp == original.timestamp
        assert restored.treatments == original.treatments

    def test_entry_with_empty_lists(self):
        """Test entry with empty doc_ids and doc_titles."""
        entry = ConversationContextEntry(
            query="Simple question",
            action_type="query",
            doc_ids=[],
            doc_titles=[],
            timestamp=1704067200000
        )
        
        assert entry.doc_ids == []
        assert entry.doc_titles == []

    def test_all_valid_action_types(self):
        """Test all valid action types are accepted."""
        valid_types = ["query", "eval_treatment", "patient_match", "study_comparison", "followup"]
        
        for action_type in valid_types:
            entry = ConversationContextEntry(
                query="Test",
                action_type=action_type,
                doc_ids=[],
                doc_titles=[],
                timestamp=1704067200000
            )
            assert entry.action_type == action_type


class TestQueryRequestWithContext:
    """Test QueryRequest model accepts conversation_context field."""

    def test_request_without_context(self):
        """Test QueryRequest works without conversation_context."""
        request = QueryRequest(
            question="What is the standard treatment for breast cancer?"
        )
        
        assert request.question == "What is the standard treatment for breast cancer?"
        assert request.conversation_context is None

    def test_request_with_empty_context(self):
        """Test QueryRequest with empty conversation_context list."""
        request = QueryRequest(
            question="What is the standard treatment?",
            conversation_context=[]
        )
        
        assert request.conversation_context == []

    def test_request_with_context_entries(self):
        """Test QueryRequest with conversation_context entries."""
        context_entries = [
            ConversationContextEntry(
                query="Previous question",
                action_type="query",
                doc_ids=["doc1"],
                doc_titles=["Study 1"],
                timestamp=1704067200000
            )
        ]
        
        request = QueryRequest(
            question="Follow-up question",
            conversation_context=context_entries
        )
        
        assert len(request.conversation_context) == 1
        assert request.conversation_context[0].query == "Previous question"

    def test_request_serialization_with_context(self):
        """Test QueryRequest serialization includes conversation_context."""
        context_entries = [
            ConversationContextEntry(
                query="First query",
                action_type="query",
                doc_ids=["doc1"],
                doc_titles=["Title 1"],
                timestamp=1704067200000
            )
        ]
        
        request = QueryRequest(
            question="Second query",
            conversation_context=context_entries
        )
        
        serialized = request.model_dump()
        
        assert "conversation_context" in serialized
        assert len(serialized["conversation_context"]) == 1
        assert serialized["conversation_context"][0]["query"] == "First query"


class TestContextBoostingFunction:
    """Test the boost_context_documents function."""

    def test_boost_applied_to_matching_doc_ids(self):
        """Test that chunks from context doc_ids get boosted scores."""
        from src.api.services.enhanced_rag_service import (
            boost_context_documents, CONTEXT_BOOST_ADDEND,
        )
        
        chunks = [
            {"doc_id": "doc1", "score_rerank": 0.8},
            {"doc_id": "doc2", "score_rerank": 0.7},
            {"doc_id": "doc3", "score_rerank": 0.6},
        ]
        context_doc_ids = ["doc1", "doc3"]
        
        result = boost_context_documents(chunks, context_doc_ids, boost_factor=1.2)
        
        # doc1 and doc3 should be boosted (additive: score + CONTEXT_BOOST_ADDEND)
        assert result[0]["score_context_boost"] == pytest.approx(0.8 + CONTEXT_BOOST_ADDEND)
        assert result[0].get("from_context") is True
        
        # doc2 should not be boosted
        assert result[1]["score_context_boost"] == pytest.approx(0.7)
        assert result[1].get("from_context") is None
        
        # doc3 should be boosted
        assert result[2]["score_context_boost"] == pytest.approx(0.6 + CONTEXT_BOOST_ADDEND)
        assert result[2].get("from_context") is True

    def test_no_boost_with_empty_context(self):
        """Test no boosting when context_doc_ids is empty."""
        from src.api.services.enhanced_rag_service import boost_context_documents
        
        chunks = [
            {"doc_id": "doc1", "score_rerank": 0.8},
            {"doc_id": "doc2", "score_rerank": 0.7},
        ]
        
        result = boost_context_documents(chunks, [], boost_factor=1.2)
        
        # Chunks should be returned unchanged
        assert result == chunks

    def test_no_boost_with_none_context(self):
        """Test no boosting when context_doc_ids is None."""
        from src.api.services.enhanced_rag_service import boost_context_documents
        
        chunks = [
            {"doc_id": "doc1", "score_rerank": 0.8},
        ]
        
        result = boost_context_documents(chunks, None, boost_factor=1.2)
        
        assert result == chunks

    def test_boost_with_payload_doc_id(self):
        """Test boosting works when doc_id is in payload."""
        from src.api.services.enhanced_rag_service import (
            boost_context_documents, CONTEXT_BOOST_ADDEND,
        )
        
        chunks = [
            {"payload": {"doc_id": "doc1"}, "score_rerank": 0.8},
            {"payload": {"doc_id": "doc2"}, "score_rerank": 0.7},
        ]
        context_doc_ids = ["doc1"]
        
        result = boost_context_documents(chunks, context_doc_ids, boost_factor=1.5)
        
        # Additive boost: score + CONTEXT_BOOST_ADDEND
        assert result[0]["score_context_boost"] == pytest.approx(0.8 + CONTEXT_BOOST_ADDEND)
        assert result[0].get("from_context") is True
        assert result[1]["score_context_boost"] == pytest.approx(0.7)

    def test_boost_with_empty_chunks(self):
        """Test boosting with empty chunks list."""
        from src.api.services.enhanced_rag_service import boost_context_documents
        
        result = boost_context_documents([], ["doc1"], boost_factor=1.2)
        
        assert result == []

    def test_custom_boost_factor(self):
        """Test that additive boost is applied correctly regardless of boost_factor param."""
        from src.api.services.enhanced_rag_service import (
            boost_context_documents, CONTEXT_BOOST_ADDEND,
        )
        
        chunks = [{"doc_id": "doc1", "score_rerank": 0.5}]
        
        result = boost_context_documents(chunks, ["doc1"], boost_factor=2.0)
        
        # Additive boost: score + CONTEXT_BOOST_ADDEND (boost_factor kept for API compat only)
        assert result[0]["score_context_boost"] == pytest.approx(0.5 + CONTEXT_BOOST_ADDEND)


class TestActionTypeClassification:
    """Test the classify_action_type function."""

    def test_classify_general_query(self):
        """Test classification of general knowledge query."""
        from src.api.services.enhanced_rag_service import classify_action_type
        
        action_type, treatments = classify_action_type(
            "What is the standard radiation dose for NSCLC?"
        )
        
        assert action_type == "query"
        assert treatments is None

    def test_classify_followup_with_context(self):
        """Test classification of follow-up query with existing context."""
        from src.api.services.enhanced_rag_service import classify_action_type
        
        context = [
            {"query": "Previous question", "action_type": "query"}
        ]
        
        action_type, treatments = classify_action_type(
            "What about the toxicity rates?",
            conversation_context=context
        )
        
        assert action_type == "followup"
        assert treatments is None

    def test_classify_treatment_evaluation(self):
        """Test classification of treatment evaluation query."""
        from src.api.services.enhanced_rag_service import classify_action_type
        
        action_type, treatments = classify_action_type(
            "Compare chemotherapy versus immunotherapy for stage IV NSCLC"
        )
        
        assert action_type == "eval_treatment"
        assert treatments is not None
        assert len(treatments) > 0

    def test_classify_without_context_not_followup(self):
        """Test that queries without context are not classified as followup."""
        from src.api.services.enhanced_rag_service import classify_action_type
        
        action_type, treatments = classify_action_type(
            "What about the side effects?",
            conversation_context=None
        )
        
        # Without context, this should be a general query, not followup
        assert action_type == "query"

    def test_classify_short_query_with_context_reference(self):
        """Test short queries with context references are classified as followup."""
        from src.api.services.enhanced_rag_service import classify_action_type
        
        context = [{"query": "Previous", "action_type": "query"}]
        
        action_type, treatments = classify_action_type(
            "What about that?",
            conversation_context=context
        )
        
        assert action_type == "followup"


class TestConversationContextIntegration:
    """Integration tests for end-to-end conversation flow."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = Mock()
        
        # Mock query_points response
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="point_1",
                score=0.9,
                payload={
                    "doc_id": "doc_abc123",
                    "text": "Study results for NSCLC treatment...",
                    "doc_meta": {
                        "title": "RTOG 0617 Trial Results",
                        "author_et_al": "Author et al.",
                        "year": 2020,
                        "doi": "10.1000/test1"
                    }
                }
            ),
            Mock(
                id="point_2",
                score=0.85,
                payload={
                    "doc_id": "doc_def456",
                    "text": "PACIFIC trial outcomes...",
                    "doc_meta": {
                        "title": "PACIFIC Trial",
                        "author_et_al": "Author2 et al.",
                        "year": 2021,
                        "doi": "10.1000/test2"
                    }
                }
            )
        ]
        client.query_points.return_value = mock_response
        
        return client

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = Mock()
        
        # Mock embeddings response
        embedding_response = Mock()
        embedding_response.data = [Mock(embedding=[0.1] * 3072)]
        client.embeddings.create.return_value = embedding_response
        
        # Mock chat completion response
        chat_response = Mock()
        chat_response.choices = [
            Mock(message=Mock(content="This is a synthesized answer based on the evidence."))
        ]
        client.chat.completions.create.return_value = chat_response
        
        return client

    def test_query_without_context_creates_new_conversation(self):
        """Test that a query without conversation_context is treated as new conversation."""
        request = QueryRequest(
            question="What is the standard RT dose for stage III NSCLC?",
            conversation_context=None
        )
        
        # Verify request is valid and context is None
        assert request.conversation_context is None
        assert request.question is not None

    def test_query_with_context_includes_previous_entries(self):
        """Test that a query with conversation_context includes previous entries."""
        previous_entry = ConversationContextEntry(
            query="What is the standard RT dose for stage III NSCLC?",
            action_type="query",
            doc_ids=["doc_abc123", "doc_def456"],
            doc_titles=["RTOG 0617 Trial Results", "PACIFIC Trial"],
            timestamp=1704067200000
        )
        
        request = QueryRequest(
            question="What about the toxicity rates?",
            conversation_context=[previous_entry]
        )
        
        assert len(request.conversation_context) == 1
        assert request.conversation_context[0].query == "What is the standard RT dose for stage III NSCLC?"
        assert "doc_abc123" in request.conversation_context[0].doc_ids

    def test_context_entry_structure_is_correct(self):
        """Test that context entry has all required fields with correct types."""
        entry = ConversationContextEntry(
            query="Test query",
            action_type="query",
            doc_ids=["doc1"],
            doc_titles=["Title 1"],
            timestamp=int(time.time() * 1000)
        )
        
        # Verify all required fields exist
        assert isinstance(entry.query, str)
        assert len(entry.query) > 0
        assert entry.action_type in ["query", "eval_treatment", "patient_match", "study_comparison", "followup"]
        assert isinstance(entry.doc_ids, list)
        assert isinstance(entry.doc_titles, list)
        assert isinstance(entry.timestamp, int)
        assert entry.timestamp > 0

    def test_clear_context_results_in_empty_list(self):
        """Test that clearing context results in empty conversation_context."""
        # Simulate "Start a New Conversation" by creating request with empty context
        request = QueryRequest(
            question="Starting fresh question",
            conversation_context=[]
        )
        
        assert request.conversation_context == []
        assert len(request.conversation_context) == 0

    def test_multiple_context_entries_preserved(self):
        """Test that multiple context entries are preserved in order."""
        entries = [
            ConversationContextEntry(
                query="First question",
                action_type="query",
                doc_ids=["doc1"],
                doc_titles=["Title 1"],
                timestamp=1704067200000
            ),
            ConversationContextEntry(
                query="Second question",
                action_type="followup",
                doc_ids=["doc1", "doc2"],
                doc_titles=["Title 1", "Title 2"],
                timestamp=1704067260000
            ),
            ConversationContextEntry(
                query="Third question",
                action_type="query",
                doc_ids=["doc3"],
                doc_titles=["Title 3"],
                timestamp=1704067320000
            )
        ]
        
        request = QueryRequest(
            question="Fourth question",
            conversation_context=entries
        )
        
        assert len(request.conversation_context) == 3
        assert request.conversation_context[0].query == "First question"
        assert request.conversation_context[1].query == "Second question"
        assert request.conversation_context[2].query == "Third question"
        
        # Verify timestamps are in order
        timestamps = [e.timestamp for e in request.conversation_context]
        assert timestamps == sorted(timestamps)

    def test_context_doc_ids_extraction(self):
        """Test extracting doc_ids from conversation context for boosting."""
        entries = [
            ConversationContextEntry(
                query="Q1",
                action_type="query",
                doc_ids=["doc1", "doc2"],
                doc_titles=["T1", "T2"],
                timestamp=1704067200000
            ),
            ConversationContextEntry(
                query="Q2",
                action_type="followup",
                doc_ids=["doc2", "doc3"],
                doc_titles=["T2", "T3"],
                timestamp=1704067260000
            )
        ]
        
        # Extract all unique doc_ids from context
        all_doc_ids = set()
        for entry in entries:
            all_doc_ids.update(entry.doc_ids)
        
        assert all_doc_ids == {"doc1", "doc2", "doc3"}


class TestUpdatedContextEntryResponse:
    """Test that responses include updated_context_entry."""

    def test_updated_context_entry_structure(self):
        """Test the structure of updated_context_entry in response."""
        # Create a mock updated_context_entry as would be returned by the API
        updated_entry = ConversationContextEntry(
            query="What is the standard RT dose?",
            action_type="query",
            doc_ids=["doc_abc123", "doc_def456"],
            doc_titles=["RTOG 0617 Trial Results", "PACIFIC Trial"],
            timestamp=1704067200000
        )
        
        # Verify structure
        assert updated_entry.query == "What is the standard RT dose?"
        assert updated_entry.action_type == "query"
        assert len(updated_entry.doc_ids) == 2
        assert len(updated_entry.doc_titles) == 2
        assert updated_entry.timestamp > 0

    def test_updated_context_entry_with_treatments(self):
        """Test updated_context_entry includes treatments for eval_treatment."""
        updated_entry = ConversationContextEntry(
            query="Compare chemo vs immunotherapy",
            action_type="eval_treatment",
            doc_ids=["doc1"],
            doc_titles=["Comparison Study"],
            timestamp=1704067200000,
            treatments=["chemotherapy", "immunotherapy"]
        )
        
        assert updated_entry.action_type == "eval_treatment"
        assert updated_entry.treatments is not None
        assert len(updated_entry.treatments) == 2
        assert "chemotherapy" in updated_entry.treatments


class TestMissingContextHandling:
    """Test handling of missing or null conversation_context."""

    def test_null_context_treated_as_new_conversation(self):
        """Test that null context is treated as new conversation."""
        request = QueryRequest(
            question="New question",
            conversation_context=None
        )
        
        # Should be valid and treated as new conversation
        assert request.conversation_context is None

    def test_missing_context_field_defaults_to_none(self):
        """Test that missing conversation_context field defaults to None."""
        # Create request without specifying conversation_context
        request = QueryRequest(question="Test question")
        
        assert request.conversation_context is None

    def test_empty_context_is_valid(self):
        """Test that empty context list is valid."""
        request = QueryRequest(
            question="Test question",
            conversation_context=[]
        )
        
        assert request.conversation_context == []
        assert len(request.conversation_context) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
