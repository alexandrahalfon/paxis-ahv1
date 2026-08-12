"""Test patient eligibility boost service."""

from src.api.services.patient_eligibility_boost_service import (
    extract_patient_context_from_query,
    build_patient_summary,
)

# Test queries
test_queries = [
    "What is the best treatment for a 65 year old male with stage III NSCLC?",
    "Treatment options for 45yo female with HER2+ breast cancer stage IIA",
    "72 year old man with T3N1M0 prostate adenocarcinoma, ECOG 1, previously treated with ADT",
    "Best radiation dose for head and neck squamous cell carcinoma",
    "What are the side effects of pembrolizumab?",  # No patient context
    "55 year old woman with newly diagnosed stage IV colorectal cancer",
]

print("Testing patient context extraction:\n")
for query in test_queries:
    print(f"Query: {query}")
    context = extract_patient_context_from_query(query)
    if context:
        summary = build_patient_summary(context)
        print(f"  Context: {context}")
        print(f"  Summary: {summary}")
    else:
        print("  No patient context detected")
    print()
