"""
Property-based tests for Unified_Router.

Feature: unified-rag-pipeline

Tests the following properties:
- Property 1: Unified routing uses module classification as primary
- Property 2: Format hints match module type
- Property 16: Format templates exist for all combinations

Validates: Requirements 1.1, 1.2, 1.4, 1.5, 1.6, 9.2
"""

from hypothesis import given, strategies as st, settings, assume
import pytest

from src.api.services.unified_router import (
    get_unified_router,
    UnifiedRouter,
    QueryModule,
    QueryType,
    RoutingResult,
)
from src.api.services.module_classifier import classify_module


# Strategy for generating valid query strings
# Filters out empty strings and strings with only whitespace
query_strategy = st.text(min_size=5, max_size=200).filter(
    lambda x: len(x.strip()) > 0
)


# Feature: unified-rag-pipeline, Property 1: Unified routing uses module classification as primary
@settings(max_examples=100)
@given(query=query_strategy)
def test_routing_uses_module_classifier_as_primary(query: str):
    """
    Property 1: Unified routing uses module classification as primary.
    
    For any query string, the Unified_Router SHALL return a RoutingResult 
    where the `module` field matches the module returned by 
    Module_Classifier.classify_module() for the same query.
    
    **Validates: Requirements 1.1, 1.2**
    """
    router = get_unified_router()
    routing_result = router.route_query(query)
    classification = classify_module(query)
    
    # The router's module should match the classifier's module
    # Note: We compare by value since they may be different enum instances
    assert routing_result.module.value == classification.module.value, (
        f"Router module {routing_result.module.value} does not match "
        f"classifier module {classification.module.value} for query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 2: Format hints match module type
@settings(max_examples=100)
@given(query=query_strategy)
def test_format_hints_match_module_type(query: str):
    """
    Property 2: Format hints match module type.
    
    For any RoutingResult, the `format_hints.response_structure` SHALL be:
    - "direct_answer" when module is GENERAL_KNOWLEDGE
    - "patient_guidance" when module is PATIENT_SPECIFIC
    - "comparison" when module is EVIDENCE_EXPLORATION
    
    **Validates: Requirements 1.4, 1.5, 1.6**
    """
    router = get_unified_router()
    result = router.route_query(query)
    
    # Expected response_structure for each module
    expected_structure = {
        QueryModule.GENERAL_KNOWLEDGE: "direct_answer",
        QueryModule.PATIENT_SPECIFIC: "patient_guidance",
        QueryModule.EVIDENCE_EXPLORATION: "comparison",
    }
    
    actual_structure = result.format_hints.get("response_structure")
    expected = expected_structure[result.module]
    
    assert actual_structure == expected, (
        f"Format hints response_structure '{actual_structure}' does not match "
        f"expected '{expected}' for module {result.module.value}. "
        f"Query: {query[:50]}..."
    )


# Feature: unified-rag-pipeline, Property 16: Format templates exist for all combinations
@settings(max_examples=100)
@given(
    module=st.sampled_from(list(QueryModule)),
    query_type=st.sampled_from(list(QueryType))
)
def test_format_templates_exist_for_all_combinations(
    module: QueryModule, 
    query_type: QueryType
):
    """
    Property 16: Format templates exist for all combinations.
    
    For any valid (module, query_type) combination, the 
    Unified_Router._get_format_hints() method SHALL return a 
    non-empty format hints object.
    
    **Validates: Requirements 9.2**
    """
    router = get_unified_router()
    
    # Call the internal method to get format hints for the combination
    format_hints = router._get_format_hints(module, query_type)
    
    # Format hints should not be None
    assert format_hints is not None, (
        f"Format hints is None for module={module.value}, "
        f"query_type={query_type.value}"
    )
    
    # Format hints should be a non-empty dictionary
    assert isinstance(format_hints, dict), (
        f"Format hints should be a dict, got {type(format_hints)} "
        f"for module={module.value}, query_type={query_type.value}"
    )
    
    assert len(format_hints) > 0, (
        f"Format hints should not be empty for module={module.value}, "
        f"query_type={query_type.value}"
    )
    
    # Format hints should always contain response_structure
    assert "response_structure" in format_hints, (
        f"Format hints missing 'response_structure' for module={module.value}, "
        f"query_type={query_type.value}"
    )
