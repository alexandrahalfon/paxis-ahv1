"""
Tests for SimplePatientMatchingService

Tests the patient matching functionality including:
- Query building
- Score normalization
- Match data extraction
- Pattern matching for demographics and cancer characteristics
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List


class TestSimplePatientMatchingService:
    """Test suite for SimplePatientMatchingService."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        return Mock()

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
        chat_response.choices = [Mock(message=Mock(content="1: MATCH, 2: MATCH, 3: NO_MATCH"))]
        client.chat.completions.create.return_value = chat_response
        
        return client

    @pytest.fixture
    def service(self, mock_qdrant_client, mock_openai_client):
        """Create a SimplePatientMatchingService instance with mocks."""
        with patch('src.api.services.patient_matching_service_simple.get_cross_encoder') as mock_ce:
            mock_ce.return_value = None  # Disable cross-encoder for unit tests
            
            from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
            return SimplePatientMatchingService(
                qdrant_client=mock_qdrant_client,
                openai_client=mock_openai_client,
                collection_name="test_collection"
            )

    # ============================================
    # Query Building Tests
    # ============================================

    def test_build_enhanced_query_basic(self, service):
        """Test query building with basic patient profile."""
        patient_profile = {
            "cancer_type": "Breast",
            "age": 55,
            "gender": "female"
        }
        
        query, category_filter = service._build_enhanced_query(patient_profile)
        
        assert "breast cancer patients" in query.lower()
        assert "female patients" in query.lower()
        assert "adult patients" in query.lower()
        assert category_filter == "breast_processed_documents"

    def test_build_enhanced_query_young_patient(self, service):
        """Test query building for young patients (<40)."""
        patient_profile = {
            "cancer_type": "Breast",
            "age": 35
        }
        
        query, _ = service._build_enhanced_query(patient_profile)
        
        assert "young adult patients under 40" in query.lower()

    def test_build_enhanced_query_elderly_patient(self, service):
        """Test query building for elderly patients (>=65)."""
        patient_profile = {
            "cancer_type": "Lung",
            "age": 72
        }
        
        query, _ = service._build_enhanced_query(patient_profile)
        
        assert "elderly patients 65 years or older" in query.lower()

    def test_build_enhanced_query_with_markers(self, service):
        """Test query building with molecular markers."""
        patient_profile = {
            "cancer_type": "Breast",
            "molecular_markers": ["HER2+", "ER-", "PR-"]
        }
        
        query, _ = service._build_enhanced_query(patient_profile)
        
        assert "molecular markers" in query.lower()
        assert "her2" in query.lower()

    def test_build_enhanced_query_with_stage(self, service):
        """Test query building with cancer stage."""
        patient_profile = {
            "cancer_type": "Colorectal",
            "cancer_stage": "III"
        }
        
        query, _ = service._build_enhanced_query(patient_profile)
        
        assert "stage iii" in query.lower()

    def test_build_enhanced_query_empty_profile(self, service):
        """Test query building with empty profile."""
        patient_profile = {}
        
        query, category_filter = service._build_enhanced_query(patient_profile)
        
        assert query == "clinical studies"
        assert category_filter is None

    # ============================================
    # Score Normalization Tests
    # ============================================

    def test_normalize_scores_basic(self, service):
        """Test score normalization with varied scores.

        The implementation uses a piecewise mapping based on cross-encoder score ranges
        and applies a relative ranking adjustment when scores are clustered (range < 3).
        Higher raw scores should still produce higher normalized values.
        """
        chunks = [
            {"score_rerank": 0.9},
            {"score_rerank": 0.5},
            {"score_rerank": 0.1}
        ]
        
        service._normalize_scores(chunks)
        
        # Higher raw scores should produce higher normalized values
        assert chunks[0]["score_normalized"] > chunks[1]["score_normalized"]
        assert chunks[1]["score_normalized"] > chunks[2]["score_normalized"]
        # All should be positive and below 0.95 (the cap)
        for chunk in chunks:
            assert 0.10 <= chunk["score_normalized"] <= 0.95

    def test_normalize_scores_identical(self, service):
        """Test score normalization when all scores are identical."""
        chunks = [
            {"score_rerank": 0.5},
            {"score_rerank": 0.5},
            {"score_rerank": 0.5}
        ]
        
        service._normalize_scores(chunks)
        
        # All identical scores should produce identical normalized values
        for chunk in chunks:
            assert chunk["score_normalized"] == chunks[0]["score_normalized"]
            assert 0.10 <= chunk["score_normalized"] <= 0.95

    def test_normalize_scores_empty(self, service):
        """Test score normalization with empty list."""
        chunks = []
        
        # Should not raise an error
        service._normalize_scores(chunks)
        
        assert chunks == []

    def test_normalize_scores_uses_dense_fallback(self, service):
        """Test that normalization falls back to dense score."""
        chunks = [
            {"score_dense": 0.8},
            {"score_dense": 0.4}
        ]
        
        service._normalize_scores(chunks)
        
        # Higher dense score should produce higher normalized score
        assert chunks[0]["score_normalized"] > chunks[1]["score_normalized"]
        # All should be positive and below the cap
        for chunk in chunks:
            assert 0.10 <= chunk["score_normalized"] <= 0.95

    # ============================================
    # Pattern Matching Tests
    # ============================================

    def test_generate_similarity_reasons_age_young(self, service):
        """Test age matching for young patients."""
        chunk = {
            "payload": {
                "text": "This study enrolled young patients under 40 years of age."
            }
        }
        patient_profile = {"age": 35}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "Young patients" in result["demographics"]

    def test_generate_similarity_reasons_age_elderly(self, service):
        """Test age matching for elderly patients."""
        chunk = {
            "payload": {
                "text": "Elderly patients aged 65 or older were included."
            }
        }
        patient_profile = {"age": 70}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "Elderly patients" in result["demographics"]

    def test_generate_similarity_reasons_gender_female(self, service):
        """Test gender matching for female patients."""
        chunk = {
            "payload": {
                "text": "The study included postmenopausal women with breast cancer."
            }
        }
        patient_profile = {"gender": "female"}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "Female patients" in result["demographics"]

    def test_generate_similarity_reasons_gender_male(self, service):
        """Test gender matching for male patients."""
        chunk = {
            "payload": {
                "text": "Male patients with prostate cancer were enrolled."
            }
        }
        patient_profile = {"gender": "male"}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "Male patients" in result["demographics"]

    def test_generate_similarity_reasons_cancer_type(self, service):
        """Test cancer type matching."""
        chunk = {
            "payload": {
                "text": "Patients with breast cancer were treated with chemotherapy."
            }
        }
        patient_profile = {"cancer_type": "Breast"}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "Breast" in result["cancer_characteristics"]

    def test_generate_similarity_reasons_stage(self, service):
        """Test cancer stage matching."""
        chunk = {
            "payload": {
                "text": "Stage II breast cancer patients received adjuvant therapy."
            }
        }
        patient_profile = {"cancer_stage": "II"}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "Stage II" in result["cancer_characteristics"]

    def test_generate_similarity_reasons_molecular_marker_positive(self, service):
        """Test molecular marker matching for positive markers."""
        chunk = {
            "payload": {
                "text": "HER2 positive patients received trastuzumab."
            }
        }
        patient_profile = {"molecular_markers": ["HER2+"]}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "HER2+" in result["key_matches"]

    def test_generate_similarity_reasons_performance_status(self, service):
        """Test performance status matching."""
        chunk = {
            "payload": {
                "text": "Patients with ECOG 0 or 1 were eligible."
            }
        }
        patient_profile = {"performance_status": "0"}
        
        result = service._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        assert "ECOG 0" in result["demographics"]

    # ============================================
    # Treatment Extraction Tests
    # ============================================

    def test_extract_treatment_info_chemotherapy(self, service):
        """Test treatment extraction for chemotherapy."""
        text = "Patients received chemotherapy with trastuzumab."
        
        result = service._extract_treatment_info(text)
        
        assert "Chemotherapy" in result or "Trastuzumab" in result

    def test_extract_treatment_info_immunotherapy(self, service):
        """Test treatment extraction for immunotherapy."""
        text = "Pembrolizumab immunotherapy was administered."
        
        result = service._extract_treatment_info(text)
        
        assert "Pembrolizumab" in result or "Immunotherapy" in result

    def test_extract_treatment_info_radiotherapy(self, service):
        """Test treatment extraction for radiotherapy."""
        text = "Radiation therapy was given to the primary tumor."
        
        result = service._extract_treatment_info(text)
        
        assert "Radiotherapy" in result

    def test_extract_treatment_info_surgery(self, service):
        """Test treatment extraction for surgery."""
        text = "Surgical resection was performed."
        
        result = service._extract_treatment_info(text)
        
        assert "Surgery" in result

    def test_extract_treatment_info_empty(self, service):
        """Test treatment extraction with empty text."""
        result = service._extract_treatment_info("")
        
        assert result == "Treatment information not available"

    # ============================================
    # Key Info Extraction Tests
    # ============================================

    def test_extract_key_info_with_results(self, service):
        """Test key info extraction with results keyword."""
        text = "The study demonstrated improved survival. Other findings were noted."
        
        result = service._extract_key_info(text, {})
        
        assert "demonstrated" in result.lower() or "improved" in result.lower()

    def test_extract_key_info_with_conclusion(self, service):
        """Test key info extraction with conclusion keyword."""
        text = "Background information. The conclusion showed significant benefit."
        
        result = service._extract_key_info(text, {})
        
        assert len(result) > 0

    def test_extract_key_info_empty(self, service):
        """Test key info extraction with empty text."""
        result = service._extract_key_info("", {})
        
        assert result == ""

    # ============================================
    # Patient Summary Tests
    # ============================================

    def test_build_patient_summary_full(self, service):
        """Test patient summary with full profile."""
        patient_profile = {
            "age": 55,
            "gender": "female",
            "cancer_type": "Breast",
            "cancer_stage": "II"
        }
        
        result = service._build_patient_summary(patient_profile)
        
        assert "55-year-old" in result
        assert "female" in result
        assert "Breast cancer" in result
        assert "stage II" in result

    def test_build_patient_summary_minimal(self, service):
        """Test patient summary with minimal profile."""
        patient_profile = {"cancer_type": "Lung"}
        
        result = service._build_patient_summary(patient_profile)
        
        assert "Lung cancer" in result

    def test_build_patient_summary_empty(self, service):
        """Test patient summary with empty profile."""
        result = service._build_patient_summary({})
        
        assert result == "Patient"

    # ============================================
    # Error Response Tests
    # ============================================

    def test_error_response(self, service):
        """Test error response generation."""
        patient_profile = {"cancer_type": "Breast", "age": 50}
        error_msg = "Test error message"
        
        result = service._error_response(patient_profile, error_msg)
        
        assert result["matches"] == []
        assert result["total_matches"] == 0
        assert result["error"] == error_msg
        assert error_msg in result["warnings"]
        assert "Breast cancer" in result["patient_summary"]

    # ============================================
    # Chunk Selection Tests
    # ============================================

    def test_select_best_chunks_per_document(self, service):
        """Test selecting best chunk per document."""
        chunks = [
            {"payload": {"doc_id": "doc1"}, "score_rerank": 0.9},
            {"payload": {"doc_id": "doc1"}, "score_rerank": 0.7},
            {"payload": {"doc_id": "doc2"}, "score_rerank": 0.8},
            {"payload": {"doc_id": "doc2"}, "score_rerank": 0.6},
        ]
        
        result = service._select_best_chunks_per_document(chunks, top_k=10)
        
        # Should have 2 chunks (one per document)
        assert len(result) == 2
        # Best chunk from doc1 should have score 0.9
        doc1_chunk = next(c for c in result if c["payload"]["doc_id"] == "doc1")
        assert doc1_chunk["score_rerank"] == 0.9

    def test_select_best_chunks_respects_top_k(self, service):
        """Test that chunk selection respects top_k limit."""
        chunks = [
            {"payload": {"doc_id": f"doc{i}"}, "score_rerank": 0.9 - i * 0.1}
            for i in range(10)
        ]
        
        result = service._select_best_chunks_per_document(chunks, top_k=3)
        
        assert len(result) == 3


class TestPatientMatchingIntegration:
    """Integration tests for patient matching (requires mocked services)."""

    @pytest.fixture
    def mock_search_results(self):
        """Create mock Qdrant search results."""
        results = []
        for i in range(5):
            point = Mock()
            point.id = f"point_{i}"
            point.score = 0.9 - i * 0.1
            point.payload = {
                "doc_id": f"doc_{i}",
                "text": f"Study {i} enrolled breast cancer patients with stage II disease. Treatment included chemotherapy.",
                "doc_meta": {
                    "title": f"Study {i} Title",
                    "author_et_al": f"Author {i} et al.",
                    "year": 2020 + i,
                    "doi": f"10.1000/test{i}",
                    "citation": f"Journal {i}"
                }
            }
            results.append(point)
        return results

    def test_match_patient_full_flow(self, mock_search_results):
        """Test full patient matching flow."""
        with patch('src.api.services.patient_matching_service_simple.get_cross_encoder') as mock_ce:
            mock_ce.return_value = None
            
            from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
            
            # Create mocks — the service calls qdrant.query_points(), not .search()
            mock_qdrant = Mock()
            mock_query_response = Mock()
            mock_query_response.points = mock_search_results
            mock_qdrant.query_points.return_value = mock_query_response
            
            mock_openai = Mock()
            embedding_response = Mock()
            embedding_response.data = [Mock(embedding=[0.1] * 3072)]
            mock_openai.embeddings.create.return_value = embedding_response
            
            chat_response = Mock()
            chat_response.choices = [Mock(message=Mock(content="1: MATCH, 2: MATCH, 3: MATCH, 4: MATCH, 5: MATCH"))]
            mock_openai.chat.completions.create.return_value = chat_response
            
            # Create service
            service = SimplePatientMatchingService(
                qdrant_client=mock_qdrant,
                openai_client=mock_openai,
                collection_name="test_collection"
            )
            
            # Test matching
            patient_profile = {
                "cancer_type": "Breast",
                "cancer_stage": "II",
                "age": 55,
                "gender": "female"
            }
            
            result = service.match_patient(patient_profile, top_k=5)
            
            # Verify results
            assert "matches" in result
            assert "total_matches" in result
            assert "patient_summary" in result
            assert result["total_matches"] > 0
            assert "Breast cancer" in result["patient_summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
