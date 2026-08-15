"""Shared safety layers (numerical validation, eligibility, citations).

Phase 1b extracted numerical validation here so every pipeline (P1/P2/P4/P5)
calls the same validator. Phase 5 adds eligibility + citations façades on
top — `eligibility.py` re-exports the canonical implementation in
`patient_eligibility_boost_service`, `citations.py` is a new post-generation
verifier.
"""

from src.api.services.safety.citations import (
    CitationVerdict,
    extract_cited_author_year,
    strip_unverified_citations,
    verify_citations,
)
from src.api.services.safety.eligibility import (
    apply_patient_eligibility_boost,
    apply_patient_eligibility_filter_and_boost,
    build_patient_summary,
    check_patient_eligibility_for_studies,
    extract_patient_context_from_query,
    run_patient_eligibility_check,
)
from src.api.services.safety.numerical import (
    STAT_PATTERNS,
    enrich_answer_with_stats,
    extract_numbers_with_stats,
    strip_unvalidated_numbers,
    validate_numbers_against_sources,
)

__all__ = [
    # numerical
    "STAT_PATTERNS",
    "enrich_answer_with_stats",
    "extract_numbers_with_stats",
    "strip_unvalidated_numbers",
    "validate_numbers_against_sources",
    # eligibility
    "apply_patient_eligibility_boost",
    "apply_patient_eligibility_filter_and_boost",
    "build_patient_summary",
    "check_patient_eligibility_for_studies",
    "extract_patient_context_from_query",
    "run_patient_eligibility_check",
    # citations
    "CitationVerdict",
    "extract_cited_author_year",
    "strip_unverified_citations",
    "verify_citations",
]
