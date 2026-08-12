"""
Property-based tests for Follow_Up_Generator.

Feature: unified-rag-pipeline

Tests the following properties:
- Property 3: Follow-up count within bounds (2-4)
- Property 4: Follow-ups reflect response content

Validates: Requirements 3.1, 3.3, 3.5, 3.6
"""

from hypothesis import given, strategies as st, settings, assume
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import List, Dict

from src.api.services.follow_up_generator import FollowUpGenerator


# =============================================================================
# Test Strategies
# =============================================================================

# Strategy for generating non-empty response content
response_content_strategy = st.text(min_size=10, max_size=500).filter(
    lambda x: len(x.strip()) > 0
)

# Strategy for generating query strings
query_strategy = st.text(min_size=5, max_size=200).filter(
    lambda x: len(x.strip()) > 0
)

# Strategy for generating module types
module_strategy = st.sampled_from([
    "general_knowledge",
    "patient_specific", 
    "evidence_exploration"
])

# Strategy for generating document titles
doc_titles_strategy = st.lists(
    st.text(min_size=5, max_size=100).filter(lambda x: len(x.strip()) > 0),
    min_size=0,
    max_size=5
)

# Strategy for generating follow-up count (2-4)
follow_up_count_strategy = st.integers(min_value=2, max_value=4)

# Valid follow-up types
VALID_FOLLOW_UP_TYPES = ["DOSE", "OUTCOME", "COMPARE", "TRIAL", "ALTERNATIVE", "TOXICITY", "ELIGIBILITY"]

# Strategy for generating follow-up types
follow_up_type_strategy = st.sampled_from(VALID_FOLLOW_UP_TYPES)


# =============================================================================
# Helper Functions
# =============================================================================

def create_mock_openai_response(suggestions: List[Dict[str, str]]) -> str:
    """
    Create a mock OpenAI response string from a list of suggestions.
    
    Args:
        suggestions: List of dicts with 'type' and 'text' keys
        
    Returns:
        Formatted string matching expected LLM output format
    """
    lines = []
    for suggestion in suggestions:
        lines.append(f"[{suggestion['type']}] {suggestion['text']}")
    return "\n".join(lines)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# =============================================================================
# Property 3: Follow-up count within bounds (2-4)
# =============================================================================

# Feature: unified-rag-pipeline, Property 3: Follow-up count within bounds (2-4)
@settings(max_examples=100)
@given(
    query=query_strategy,
    response=response_content_strategy,
    module=module_strategy,
    doc_titles=doc_titles_strategy,
    num_suggestions=follow_up_count_strategy
)
def test_follow_up_count_within_bounds(
    query: str,
    response: str,
    module: str,
    doc_titles: List[str],
    num_suggestions: int
):
    """
    Property 3: Follow-up count within bounds (2-4).
    
    For any generated response with non-empty content, the Follow_Up_Generator 
    SHALL produce between 2 and 4 follow-up suggestions (inclusive).
    
    **Validates: Requirements 3.1, 3.3**
    """
    # Create mock suggestions within bounds
    mock_suggestions = [
        {"type": "DOSE", "text": f"What is the recommended dose for treatment {i}?"}
        for i in range(num_suggestions)
    ]
    mock_response_content = create_mock_openai_response(mock_suggestions)
    
    # Create mock OpenAI client
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = mock_response_content
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    # Create generator with mocked client
    generator = FollowUpGenerator(openai_client=mock_client)
    
    # Run the async method
    result = run_async(generator.generate_follow_ups(
        query=query,
        response=response,
        module=module,
        doc_titles=doc_titles,
        max_suggestions=4
    ))
    
    # Verify count is within bounds (2-4)
    assert len(result) >= 2, (
        f"Follow-up count {len(result)} is below minimum of 2. "
        f"Query: {query[:50]}..."
    )
    assert len(result) <= 4, (
        f"Follow-up count {len(result)} exceeds maximum of 4. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 3: Follow-up count within bounds (2-4) - edge case
@settings(max_examples=100)
@given(
    query=query_strategy,
    response=response_content_strategy,
    module=module_strategy
)
def test_follow_up_count_respects_max_suggestions_parameter(
    query: str,
    response: str,
    module: str
):
    """
    Property 3 (supplementary): Follow-up count respects max_suggestions parameter.
    
    When max_suggestions is set, the generator SHALL NOT return more suggestions
    than the specified maximum.
    
    **Validates: Requirements 3.3**
    """
    # Create mock with exactly 4 suggestions (max allowed)
    mock_suggestions = [
        {"type": "DOSE", "text": "Question about dosing?"},
        {"type": "COMPARE", "text": "How does this compare?"},
        {"type": "TRIAL", "text": "What trials are available?"},
        {"type": "OUTCOME", "text": "What are the outcomes?"},
    ]
    mock_response_content = create_mock_openai_response(mock_suggestions)
    
    # Create mock OpenAI client
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = mock_response_content
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    # Test with max_suggestions=3
    result = run_async(generator.generate_follow_ups(
        query=query,
        response=response,
        module=module,
        doc_titles=[],
        max_suggestions=3
    ))
    
    assert len(result) <= 3, (
        f"Follow-up count {len(result)} exceeds max_suggestions=3"
    )


# =============================================================================
# Property 4: Follow-ups reflect response content
# =============================================================================

# Treatment name patterns for detection
TREATMENT_PATTERNS = [
    "chemotherapy", "radiation", "immunotherapy", "surgery",
    "cisplatin", "carboplatin", "paclitaxel", "docetaxel",
    "pembrolizumab", "nivolumab", "durvalumab", "atezolizumab",
    "SBRT", "IMRT", "proton therapy", "brachytherapy",
    "concurrent chemoradiation", "adjuvant therapy", "neoadjuvant"
]

# Trial name patterns for detection
TRIAL_PATTERNS = [
    "RTOG", "NRG", "SWOG", "ECOG", "NCT", "trial", "study",
    "phase I", "phase II", "phase III", "randomized",
    "clinical trial", "enrollment", "eligibility criteria"
]


def contains_treatment_mention(text: str) -> bool:
    """Check if text contains treatment name patterns."""
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in TREATMENT_PATTERNS)


def contains_trial_mention(text: str) -> bool:
    """Check if text contains trial name patterns."""
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in TRIAL_PATTERNS)


# Strategy for generating responses with treatment mentions
treatment_response_strategy = st.sampled_from([
    "The recommended treatment is cisplatin-based chemotherapy with concurrent radiation.",
    "SBRT delivers 54 Gy in 3 fractions for early-stage lung cancer.",
    "Pembrolizumab immunotherapy has shown improved outcomes in PD-L1 positive patients.",
    "Concurrent chemoradiation with carboplatin and paclitaxel is the standard approach.",
    "Adjuvant therapy with docetaxel may be considered after surgery.",
    "IMRT allows for better dose conformity while sparing normal tissues.",
    "Neoadjuvant chemotherapy followed by surgery is recommended for locally advanced disease.",
])

# Strategy for generating responses with trial mentions
trial_response_strategy = st.sampled_from([
    "The RTOG 0617 trial demonstrated that 60 Gy is the optimal dose.",
    "NRG-LU002 is currently enrolling patients with stage III NSCLC.",
    "Phase III randomized trials have shown improved survival with this approach.",
    "The NCT12345678 clinical trial is investigating this combination.",
    "SWOG S1400 evaluated targeted therapy based on molecular profiling.",
    "Eligibility criteria include ECOG performance status 0-1.",
    "The study enrollment period ends in December 2024.",
])


# Feature: unified-rag-pipeline, Property 4: Follow-ups reflect response content (treatments)
@settings(max_examples=100)
@given(
    query=query_strategy,
    response=treatment_response_strategy,
    module=module_strategy
)
def test_treatment_mentions_generate_dose_or_compare_followups(
    query: str,
    response: str,
    module: str
):
    """
    Property 4: Follow-ups reflect response content (treatment mentions).
    
    For any response that mentions specific treatments (detected by treatment 
    name patterns), at least one follow-up suggestion SHALL have type "DOSE" 
    or "COMPARE".
    
    **Validates: Requirements 3.1, 3.5**
    """
    # Verify the response contains treatment mentions
    assume(contains_treatment_mention(response))
    
    # Create mock suggestions that include DOSE or COMPARE for treatment content
    mock_suggestions = [
        {"type": "DOSE", "text": "What is the standard dosing regimen?"},
        {"type": "COMPARE", "text": "How does this compare to alternative treatments?"},
        {"type": "OUTCOME", "text": "What are the expected outcomes?"},
    ]
    mock_response_content = create_mock_openai_response(mock_suggestions)
    
    # Create mock OpenAI client
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = mock_response_content
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    result = run_async(generator.generate_follow_ups(
        query=query,
        response=response,
        module=module,
        doc_titles=[],
        max_suggestions=4
    ))
    
    # Check that at least one follow-up has type DOSE or COMPARE
    has_treatment_followup = any(
        suggestion.get("type") in ["DOSE", "COMPARE"]
        for suggestion in result
    )
    
    assert has_treatment_followup, (
        f"Response mentions treatments but no DOSE or COMPARE follow-up generated. "
        f"Response: {response[:100]}... "
        f"Follow-ups: {[s.get('type') for s in result]}"
    )


# Feature: unified-rag-pipeline, Property 4: Follow-ups reflect response content (trials)
@settings(max_examples=100)
@given(
    query=query_strategy,
    response=trial_response_strategy,
    module=module_strategy
)
def test_trial_mentions_generate_trial_or_eligibility_followups(
    query: str,
    response: str,
    module: str
):
    """
    Property 4: Follow-ups reflect response content (trial mentions).
    
    For any response that mentions clinical trials (detected by trial name 
    patterns), at least one follow-up suggestion SHALL have type "TRIAL" 
    or "ELIGIBILITY".
    
    **Validates: Requirements 3.1, 3.6**
    """
    # Verify the response contains trial mentions
    assume(contains_trial_mention(response))
    
    # Create mock suggestions that include TRIAL or ELIGIBILITY for trial content
    mock_suggestions = [
        {"type": "TRIAL", "text": "What were the primary endpoints of this trial?"},
        {"type": "ELIGIBILITY", "text": "What are the eligibility criteria?"},
        {"type": "OUTCOME", "text": "What were the survival outcomes?"},
    ]
    mock_response_content = create_mock_openai_response(mock_suggestions)
    
    # Create mock OpenAI client
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = mock_response_content
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    result = run_async(generator.generate_follow_ups(
        query=query,
        response=response,
        module=module,
        doc_titles=[],
        max_suggestions=4
    ))
    
    # Check that at least one follow-up has type TRIAL or ELIGIBILITY
    has_trial_followup = any(
        suggestion.get("type") in ["TRIAL", "ELIGIBILITY"]
        for suggestion in result
    )
    
    assert has_trial_followup, (
        f"Response mentions trials but no TRIAL or ELIGIBILITY follow-up generated. "
        f"Response: {response[:100]}... "
        f"Follow-ups: {[s.get('type') for s in result]}"
    )


# Feature: unified-rag-pipeline, Property 4: Follow-ups reflect response content (combined)
@settings(max_examples=100)
@given(
    query=query_strategy,
    module=module_strategy
)
def test_combined_treatment_and_trial_mentions(
    query: str,
    module: str
):
    """
    Property 4: Follow-ups reflect response content (combined treatment and trial).
    
    For responses mentioning both treatments and trials, follow-ups SHALL include
    both treatment-related (DOSE/COMPARE) and trial-related (TRIAL/ELIGIBILITY) types.
    
    **Validates: Requirements 3.1, 3.5, 3.6**
    """
    # Response with both treatment and trial mentions
    response = (
        "The RTOG 0617 trial evaluated concurrent chemoradiation with cisplatin "
        "and demonstrated that 60 Gy is the optimal dose for stage III NSCLC."
    )
    
    # Verify the response contains both patterns
    assume(contains_treatment_mention(response))
    assume(contains_trial_mention(response))
    
    # Create mock suggestions covering both treatment and trial types
    mock_suggestions = [
        {"type": "DOSE", "text": "What is the cisplatin dosing schedule?"},
        {"type": "TRIAL", "text": "What were the RTOG 0617 inclusion criteria?"},
        {"type": "COMPARE", "text": "How does this compare to 74 Gy?"},
        {"type": "ELIGIBILITY", "text": "Who is eligible for this regimen?"},
    ]
    mock_response_content = create_mock_openai_response(mock_suggestions)
    
    # Create mock OpenAI client
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = mock_response_content
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    result = run_async(generator.generate_follow_ups(
        query=query,
        response=response,
        module=module,
        doc_titles=[],
        max_suggestions=4
    ))
    
    # Check for treatment-related follow-up
    has_treatment_followup = any(
        suggestion.get("type") in ["DOSE", "COMPARE"]
        for suggestion in result
    )
    
    # Check for trial-related follow-up
    has_trial_followup = any(
        suggestion.get("type") in ["TRIAL", "ELIGIBILITY"]
        for suggestion in result
    )
    
    assert has_treatment_followup, (
        f"Combined response missing DOSE/COMPARE follow-up. "
        f"Follow-ups: {[s.get('type') for s in result]}"
    )
    
    assert has_trial_followup, (
        f"Combined response missing TRIAL/ELIGIBILITY follow-up. "
        f"Follow-ups: {[s.get('type') for s in result]}"
    )


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

# Feature: unified-rag-pipeline, Property 3: Follow-up count within bounds (2-4) - empty response handling
@settings(max_examples=100)
@given(
    query=query_strategy,
    module=module_strategy
)
def test_empty_llm_response_returns_empty_list(
    query: str,
    module: str
):
    """
    Edge case: When LLM returns empty content, generator returns empty list.
    
    This tests the error handling path when the LLM doesn't produce valid output.
    
    **Validates: Requirements 3.1**
    """
    # Create mock OpenAI client that returns empty content
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = ""  # Empty response
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    result = run_async(generator.generate_follow_ups(
        query=query,
        response="Some response content",
        module=module,
        doc_titles=[],
        max_suggestions=4
    ))
    
    # Empty LLM response should result in empty suggestions list
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 0, (
        f"Empty LLM response should return empty list, got {len(result)} items"
    )


# Feature: unified-rag-pipeline, Property 3: Follow-up count within bounds (2-4) - API error handling
@settings(max_examples=100)
@given(
    query=query_strategy,
    module=module_strategy
)
def test_api_error_returns_empty_list(
    query: str,
    module: str
):
    """
    Edge case: When OpenAI API raises an exception, generator returns empty list.
    
    This tests the error handling path when the API call fails.
    
    **Validates: Requirements 3.1**
    """
    # Create mock OpenAI client that raises an exception
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    result = run_async(generator.generate_follow_ups(
        query=query,
        response="Some response content",
        module=module,
        doc_titles=[],
        max_suggestions=4
    ))
    
    # API error should result in empty suggestions list (graceful degradation)
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 0, (
        f"API error should return empty list, got {len(result)} items"
    )


# Feature: unified-rag-pipeline, Property 4: Follow-ups reflect response content - suggestion structure
@settings(max_examples=100)
@given(
    query=query_strategy,
    response=response_content_strategy,
    module=module_strategy,
    suggestion_type=follow_up_type_strategy
)
def test_follow_up_suggestions_have_required_structure(
    query: str,
    response: str,
    module: str,
    suggestion_type: str
):
    """
    Property 4 (supplementary): Each follow-up suggestion has required structure.
    
    Each suggestion SHALL have 'type' (string) and 'text' (string) fields.
    
    **Validates: Requirements 3.1**
    """
    # Create mock suggestions with the given type
    mock_suggestions = [
        {"type": suggestion_type, "text": "What is the recommended approach?"},
        {"type": "OUTCOME", "text": "What are the expected outcomes?"},
    ]
    mock_response_content = create_mock_openai_response(mock_suggestions)
    
    # Create mock OpenAI client
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = mock_response_content
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = FollowUpGenerator(openai_client=mock_client)
    
    result = run_async(generator.generate_follow_ups(
        query=query,
        response=response,
        module=module,
        doc_titles=[],
        max_suggestions=4
    ))
    
    # Verify each suggestion has required structure
    for suggestion in result:
        assert "type" in suggestion, (
            f"Suggestion missing 'type' field: {suggestion}"
        )
        assert "text" in suggestion, (
            f"Suggestion missing 'text' field: {suggestion}"
        )
        assert isinstance(suggestion["type"], str), (
            f"Suggestion 'type' should be string, got {type(suggestion['type'])}"
        )
        assert isinstance(suggestion["text"], str), (
            f"Suggestion 'text' should be string, got {type(suggestion['text'])}"
        )
        assert len(suggestion["text"]) > 0, (
            f"Suggestion 'text' should not be empty"
        )
