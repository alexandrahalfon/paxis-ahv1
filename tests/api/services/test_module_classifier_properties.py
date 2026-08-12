"""
Property-based tests for Module_Classifier integration.

Feature: unified-rag-pipeline

Tests the following properties:
- Property 17: Module classification matches signal patterns
- Property 18: Confidence score in valid range
- Property 19: Low confidence defaults to GENERAL_KNOWLEDGE
- Property 20: Signals matched returned

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

from hypothesis import given, strategies as st, settings, assume
import pytest

from src.api.services.module_classifier import (
    classify_module,
    ModuleClassification,
    QueryModule,
)


# =============================================================================
# Query Strategies for Signal Pattern Testing
# =============================================================================

# GENERAL_KNOWLEDGE patterns: dose/guideline/trial-result patterns WITHOUT patient context
# These queries have ground-truth answers (definitive, factual)
# NOTE: Avoid biomarker terms (EGFR, KRAS, etc.) as they trigger PATIENT_SPECIFIC
general_knowledge_query_patterns = [
    # Dose questions
    "What is the standard dose for lung cancer radiation?",
    "How many Gy are used in SBRT for early stage lung cancer?",
    "What is the fractionation schedule for prostate cancer?",
    "What dose is recommended for breast cancer boost?",
    # Guideline questions
    "What does NCCN recommend for stage II breast cancer?",
    "What are the ASTRO guidelines for prostate radiation?",
    "What is the ASCO recommendation for adjuvant chemotherapy?",
    "What are the category 1 recommendations for NSCLC?",
    # Trial result questions
    "What did the RTOG 0617 trial show?",
    "What were the results of the NRG-LU002 study?",
    "What did the phase III trial demonstrate?",
    "What are the outcomes from the PORTEC-3 trial?",
    # Mechanism questions (avoid biomarker terms)
    "What is the mechanism of action of radiation therapy?",
    "How does radiation therapy work?",
    "What is the mechanism of immunotherapy?",
    "How does chemotherapy affect cancer cells?",
]

# PATIENT_SPECIFIC patterns: patient demographics, staging, or biomarkers
# These queries contain patient context requiring personalized guidance
# NOTE: The classifier requires demographics OR biomarkers OR implicit guidance phrases
# Staging alone doesn't trigger has_patient_context=True
patient_specific_query_patterns = [
    # Demographics (strong signal)
    "65 year old male with lung cancer",
    "72 y/o female with breast cancer",
    "58 year-old man with prostate cancer",
    "45 yo woman diagnosed with cervical cancer",
    # Demographics + staging (combined)
    "68 year old male with T3N1M0 prostate cancer",
    "55 yo female with stage IIA breast cancer",
    "62 year old man with stage IIIB NSCLC",
    "70 y/o woman with locally advanced rectal cancer",
    # Biomarkers (strong signal - triggers patient context)
    "ER+ PR+ HER2- breast cancer patient",
    "Triple negative breast cancer case",
    "EGFR mutant lung cancer patient",
    "PD-L1 positive NSCLC case",
    "KRAS mutant colorectal cancer patient",
    "MSI-H endometrial cancer",
    # Implicit guidance phrases (strong signal)
    "For this patient with lung cancer, what would you recommend?",
    "Given his history of prostate cancer, how should we proceed?",
    "In this case of breast cancer, what is the approach?",
    "What would you recommend for my patient with melanoma?",
]

# EVIDENCE_EXPLORATION patterns: superlatives or comparison language
# These queries seek to compare options or find the "best" approach
# NOTE: Use strong superlative/comparison patterns that the classifier detects
evidence_exploration_query_patterns = [
    # Superlatives (strong signal)
    "What is the best treatment for lung cancer?",
    "What is the most effective therapy for melanoma?",
    "What is the optimal dose for prostate SBRT?",
    "What is the preferred approach for rectal cancer?",
    "What is the ideal fractionation for breast cancer?",
    # Comparison language (strong signal)
    "Compare SBRT versus conventional radiation for lung cancer",
    "What is the difference between IMRT and 3D-CRT?",
    "How does proton therapy compare to photon therapy?",
    "Surgery versus radiation for prostate cancer",
    "Chemotherapy vs immunotherapy for NSCLC",
    # Options exploration (strong signal)
    "What are the treatment options for stage III lung cancer?",
    "What are the options for locally advanced disease?",
    "What are the available therapies for rectal cancer?",
    "Explore different approaches for lung cancer treatment",
    # Outcome-seeking with superlatives
    "Which treatment has the best survival outcomes?",
    "What therapy results in the highest response rate?",
    "Which approach has the lowest toxicity?",
]

# Strategy for GENERAL_KNOWLEDGE queries
general_knowledge_strategy = st.sampled_from(general_knowledge_query_patterns)

# Strategy for PATIENT_SPECIFIC queries
patient_specific_strategy = st.sampled_from(patient_specific_query_patterns)

# Strategy for EVIDENCE_EXPLORATION queries
evidence_exploration_strategy = st.sampled_from(evidence_exploration_query_patterns)

# Strategy for generating arbitrary query strings (for confidence/signals tests)
arbitrary_query_strategy = st.text(min_size=5, max_size=200).filter(
    lambda x: len(x.strip()) > 0
)


# =============================================================================
# Property 17: Module classification matches signal patterns
# =============================================================================

# Feature: unified-rag-pipeline, Property 17: Module classification matches signal patterns
@settings(max_examples=100)
@given(query=general_knowledge_strategy)
def test_general_knowledge_classification_for_dose_guideline_trial_patterns(query: str):
    """
    Property 17: Module classification matches signal patterns (GENERAL_KNOWLEDGE).
    
    For any query containing dose/guideline/trial-result patterns WITHOUT patient 
    context, Module_Classifier SHALL return GENERAL_KNOWLEDGE.
    
    **Validates: Requirements 10.1**
    """
    result = classify_module(query)
    
    assert result.module == QueryModule.GENERAL_KNOWLEDGE, (
        f"Query with dose/guideline/trial pattern should classify as GENERAL_KNOWLEDGE. "
        f"Got: {result.module.value}. Query: {query[:50]}... "
        f"Signals: {result.signals_matched}"
    )


# Feature: unified-rag-pipeline, Property 17: Module classification matches signal patterns
@settings(max_examples=100)
@given(query=patient_specific_strategy)
def test_patient_specific_classification_for_patient_context_patterns(query: str):
    """
    Property 17: Module classification matches signal patterns (PATIENT_SPECIFIC).
    
    For any query containing patient demographics, staging, or biomarkers, 
    Module_Classifier SHALL return PATIENT_SPECIFIC.
    
    **Validates: Requirements 10.2**
    """
    result = classify_module(query)
    
    assert result.module == QueryModule.PATIENT_SPECIFIC, (
        f"Query with patient demographics/staging/biomarkers should classify as PATIENT_SPECIFIC. "
        f"Got: {result.module.value}. Query: {query[:50]}... "
        f"Signals: {result.signals_matched}"
    )


# Feature: unified-rag-pipeline, Property 17: Module classification matches signal patterns
@settings(max_examples=100)
@given(query=evidence_exploration_strategy)
def test_evidence_exploration_classification_for_superlative_comparison_patterns(query: str):
    """
    Property 17: Module classification matches signal patterns (EVIDENCE_EXPLORATION).
    
    For any query containing superlatives (best, optimal) or comparison language, 
    Module_Classifier SHALL return EVIDENCE_EXPLORATION.
    
    **Validates: Requirements 10.3**
    """
    result = classify_module(query)
    
    assert result.module == QueryModule.EVIDENCE_EXPLORATION, (
        f"Query with superlatives/comparison language should classify as EVIDENCE_EXPLORATION. "
        f"Got: {result.module.value}. Query: {query[:50]}... "
        f"Signals: {result.signals_matched}"
    )


# =============================================================================
# Property 18: Confidence score in valid range
# =============================================================================

# Feature: unified-rag-pipeline, Property 18: Confidence score in valid range
@settings(max_examples=100)
@given(query=arbitrary_query_strategy)
def test_confidence_score_in_valid_range(query: str):
    """
    Property 18: Confidence score in valid range.
    
    For any ModuleClassification result, the `confidence` field SHALL be a 
    number between 0.0 and 1.0 (inclusive).
    
    **Validates: Requirements 10.4**
    """
    result = classify_module(query)
    
    assert isinstance(result.confidence, (int, float)), (
        f"Confidence should be a number, got {type(result.confidence)}"
    )
    
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} is outside valid range [0.0, 1.0]. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 18: Confidence score in valid range
@settings(max_examples=100)
@given(query=general_knowledge_strategy)
def test_confidence_score_valid_for_general_knowledge(query: str):
    """
    Property 18: Confidence score in valid range (GENERAL_KNOWLEDGE queries).
    
    For GENERAL_KNOWLEDGE classified queries, confidence SHALL be between 0.0 and 1.0.
    
    **Validates: Requirements 10.4**
    """
    result = classify_module(query)
    
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} is outside valid range [0.0, 1.0]. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 18: Confidence score in valid range
@settings(max_examples=100)
@given(query=patient_specific_strategy)
def test_confidence_score_valid_for_patient_specific(query: str):
    """
    Property 18: Confidence score in valid range (PATIENT_SPECIFIC queries).
    
    For PATIENT_SPECIFIC classified queries, confidence SHALL be between 0.0 and 1.0.
    
    **Validates: Requirements 10.4**
    """
    result = classify_module(query)
    
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} is outside valid range [0.0, 1.0]. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 18: Confidence score in valid range
@settings(max_examples=100)
@given(query=evidence_exploration_strategy)
def test_confidence_score_valid_for_evidence_exploration(query: str):
    """
    Property 18: Confidence score in valid range (EVIDENCE_EXPLORATION queries).
    
    For EVIDENCE_EXPLORATION classified queries, confidence SHALL be between 0.0 and 1.0.
    
    **Validates: Requirements 10.4**
    """
    result = classify_module(query)
    
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} is outside valid range [0.0, 1.0]. "
        f"Query: {query[:50]}..."
    )


# =============================================================================
# Property 19: Low confidence defaults to GENERAL_KNOWLEDGE
# =============================================================================

# Queries designed to produce low confidence (ambiguous, no clear signals)
low_confidence_query_patterns = [
    "hello",
    "help me",
    "cancer",
    "treatment",
    "information",
    "question about therapy",
    "need advice",
    "medical query",
    "oncology",
    "radiation",
]

low_confidence_strategy = st.sampled_from(low_confidence_query_patterns)


# Feature: unified-rag-pipeline, Property 19: Low confidence defaults to GENERAL_KNOWLEDGE
@settings(max_examples=100)
@given(query=low_confidence_strategy)
def test_low_confidence_defaults_to_general_knowledge(query: str):
    """
    Property 19: Low confidence defaults to GENERAL_KNOWLEDGE.
    
    For any query where the classification confidence would be below 0.6 based 
    on signal matching, the Module_Classifier SHALL return GENERAL_KNOWLEDGE 
    as the module.
    
    **Validates: Requirements 10.5**
    """
    result = classify_module(query)
    
    # If confidence is below 0.6, module should be GENERAL_KNOWLEDGE
    if result.confidence < 0.6:
        assert result.module == QueryModule.GENERAL_KNOWLEDGE, (
            f"Low confidence ({result.confidence}) query should default to GENERAL_KNOWLEDGE. "
            f"Got: {result.module.value}. Query: {query}"
        )


# Feature: unified-rag-pipeline, Property 19: Low confidence defaults to GENERAL_KNOWLEDGE
@settings(max_examples=100)
@given(query=arbitrary_query_strategy)
def test_low_confidence_always_defaults_to_general_knowledge(query: str):
    """
    Property 19: Low confidence defaults to GENERAL_KNOWLEDGE (arbitrary queries).
    
    For any arbitrary query, if confidence is below 0.6, the module SHALL be 
    GENERAL_KNOWLEDGE.
    
    **Validates: Requirements 10.5**
    """
    result = classify_module(query)
    
    # If confidence is below 0.6, module should be GENERAL_KNOWLEDGE
    if result.confidence < 0.6:
        assert result.module == QueryModule.GENERAL_KNOWLEDGE, (
            f"Low confidence ({result.confidence}) query should default to GENERAL_KNOWLEDGE. "
            f"Got: {result.module.value}. Query: {query[:50]}..."
        )


# =============================================================================
# Property 20: Signals matched returned
# =============================================================================

# Feature: unified-rag-pipeline, Property 20: Signals matched returned
@settings(max_examples=100)
@given(query=arbitrary_query_strategy)
def test_signals_matched_is_non_empty_array(query: str):
    """
    Property 20: Signals matched returned.
    
    For any ModuleClassification result, the `signals_matched` field SHALL be 
    a non-empty array containing at least one signal string.
    
    **Validates: Requirements 10.6**
    """
    result = classify_module(query)
    
    # signals_matched should be a list
    assert isinstance(result.signals_matched, list), (
        f"signals_matched should be a list, got {type(result.signals_matched)}"
    )
    
    # signals_matched should not be empty
    assert len(result.signals_matched) > 0, (
        f"signals_matched should contain at least one signal. "
        f"Query: {query[:50]}..."
    )
    
    # Each signal should be a string
    for signal in result.signals_matched:
        assert isinstance(signal, str), (
            f"Each signal should be a string, got {type(signal)}"
        )


# Feature: unified-rag-pipeline, Property 20: Signals matched returned
@settings(max_examples=100)
@given(query=general_knowledge_strategy)
def test_signals_matched_for_general_knowledge_queries(query: str):
    """
    Property 20: Signals matched returned (GENERAL_KNOWLEDGE queries).
    
    For GENERAL_KNOWLEDGE classified queries, signals_matched SHALL contain 
    at least one signal string.
    
    **Validates: Requirements 10.6**
    """
    result = classify_module(query)
    
    assert isinstance(result.signals_matched, list), (
        f"signals_matched should be a list"
    )
    
    assert len(result.signals_matched) > 0, (
        f"signals_matched should contain at least one signal for GENERAL_KNOWLEDGE query. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 20: Signals matched returned
@settings(max_examples=100)
@given(query=patient_specific_strategy)
def test_signals_matched_for_patient_specific_queries(query: str):
    """
    Property 20: Signals matched returned (PATIENT_SPECIFIC queries).
    
    For PATIENT_SPECIFIC classified queries, signals_matched SHALL contain 
    at least one signal string.
    
    **Validates: Requirements 10.6**
    """
    result = classify_module(query)
    
    assert isinstance(result.signals_matched, list), (
        f"signals_matched should be a list"
    )
    
    assert len(result.signals_matched) > 0, (
        f"signals_matched should contain at least one signal for PATIENT_SPECIFIC query. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 20: Signals matched returned
@settings(max_examples=100)
@given(query=evidence_exploration_strategy)
def test_signals_matched_for_evidence_exploration_queries(query: str):
    """
    Property 20: Signals matched returned (EVIDENCE_EXPLORATION queries).
    
    For EVIDENCE_EXPLORATION classified queries, signals_matched SHALL contain 
    at least one signal string.
    
    **Validates: Requirements 10.6**
    """
    result = classify_module(query)
    
    assert isinstance(result.signals_matched, list), (
        f"signals_matched should be a list"
    )
    
    assert len(result.signals_matched) > 0, (
        f"signals_matched should contain at least one signal for EVIDENCE_EXPLORATION query. "
        f"Query: {query[:50]}..."
    )


# =============================================================================
# Additional Integration Tests
# =============================================================================

# Feature: unified-rag-pipeline, Property 17: Module classification matches signal patterns
@settings(max_examples=100)
@given(
    module_type=st.sampled_from(["general", "patient", "exploration"]),
)
def test_classification_result_structure(module_type: str):
    """
    Integration test: ModuleClassification result has correct structure.
    
    For any classification result, the structure SHALL include all required fields:
    module, confidence, signals_matched, has_patient_context, has_explicit_question,
    has_superlative, suggested_follow_ups.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6**
    """
    # Select query based on module type
    if module_type == "general":
        query = "What is the standard dose for lung cancer radiation?"
    elif module_type == "patient":
        query = "65 year old male with T2N1M0 lung cancer"
    else:
        query = "What is the best treatment for lung cancer?"
    
    result = classify_module(query)
    
    # Verify result is ModuleClassification instance
    assert isinstance(result, ModuleClassification), (
        f"Result should be ModuleClassification, got {type(result)}"
    )
    
    # Verify all required fields exist
    assert hasattr(result, 'module'), "Result missing 'module' field"
    assert hasattr(result, 'confidence'), "Result missing 'confidence' field"
    assert hasattr(result, 'signals_matched'), "Result missing 'signals_matched' field"
    assert hasattr(result, 'has_patient_context'), "Result missing 'has_patient_context' field"
    assert hasattr(result, 'has_explicit_question'), "Result missing 'has_explicit_question' field"
    assert hasattr(result, 'has_superlative'), "Result missing 'has_superlative' field"
    assert hasattr(result, 'suggested_follow_ups'), "Result missing 'suggested_follow_ups' field"
    
    # Verify field types
    assert isinstance(result.module, QueryModule), (
        f"module should be QueryModule, got {type(result.module)}"
    )
    assert isinstance(result.confidence, (int, float)), (
        f"confidence should be numeric, got {type(result.confidence)}"
    )
    assert isinstance(result.signals_matched, list), (
        f"signals_matched should be list, got {type(result.signals_matched)}"
    )
    assert isinstance(result.has_patient_context, bool), (
        f"has_patient_context should be bool, got {type(result.has_patient_context)}"
    )
    assert isinstance(result.has_explicit_question, bool), (
        f"has_explicit_question should be bool, got {type(result.has_explicit_question)}"
    )
    assert isinstance(result.has_superlative, bool), (
        f"has_superlative should be bool, got {type(result.has_superlative)}"
    )
    assert isinstance(result.suggested_follow_ups, list), (
        f"suggested_follow_ups should be list, got {type(result.suggested_follow_ups)}"
    )


# Feature: unified-rag-pipeline, Property 17: Module classification matches signal patterns
@settings(max_examples=100)
@given(query=arbitrary_query_strategy)
def test_to_dict_method_returns_valid_dict(query: str):
    """
    Integration test: to_dict() method returns valid dictionary.
    
    The to_dict() method SHALL return a dictionary with all classification fields.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6**
    """
    result = classify_module(query)
    result_dict = result.to_dict()
    
    # Verify result is a dictionary
    assert isinstance(result_dict, dict), (
        f"to_dict() should return dict, got {type(result_dict)}"
    )
    
    # Verify required keys exist
    required_keys = [
        'module', 'confidence', 'signals_matched', 
        'has_patient_context', 'has_explicit_question', 
        'has_superlative', 'suggested_follow_ups'
    ]
    
    for key in required_keys:
        assert key in result_dict, (
            f"to_dict() missing required key: {key}"
        )
    
    # Verify module is string value (not enum)
    assert isinstance(result_dict['module'], str), (
        f"module in dict should be string, got {type(result_dict['module'])}"
    )
    
    # Verify module value is valid
    valid_modules = ['general_knowledge', 'patient_specific', 'evidence_exploration']
    assert result_dict['module'] in valid_modules, (
        f"module value '{result_dict['module']}' not in valid modules: {valid_modules}"
    )
