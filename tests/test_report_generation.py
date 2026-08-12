#!/usr/bin/env python3
"""Tests for PDF report generation (service and API)."""
import pytest
import requests


def test_patient_match_report_service():
    """Report service generates valid patient-match PDF bytes."""
    from src.api.services.report_service import generate_patient_match_report
    data = {
        "patient_summary": "65-year-old male with NSCLC",
        "total_matches": 1,
        "matches": [{"title": "Study A", "author": "Smith", "year": 2022, "match_score": 0.9, "match_strength": "strong", "match_rationale": "Fits.", "treatment": "Chemo", "key_info": "Finding."}],
    }
    pdf = generate_patient_match_report(data)
    assert isinstance(pdf, bytes)
    assert len(pdf) > 500
    assert pdf[:4] == b"%PDF"


def test_treatment_comparison_report_service():
    """Report service generates valid treatment-comparison PDF bytes."""
    from src.api.services.report_service import generate_treatment_comparison_report
    data = {
        "comparison": {
            "treatment_a_name": "A",
            "treatment_b_name": "B",
            "treatment_a_evidence": {"efficacy": "Good", "safety": "OK", "dosing": "200mg", "outcomes": "OS"},
            "treatment_b_evidence": {"efficacy": "Standard", "safety": "Known", "dosing": "Platinum", "outcomes": "Historical"},
            "comparison_summary": "A vs B comparison.",
        },
        "sources": [],
    }
    pdf = generate_treatment_comparison_report(data)
    assert isinstance(pdf, bytes)
    assert len(pdf) > 500
    assert pdf[:4] == b"%PDF"


def test_query_report_service():
    """Report service generates valid query PDF bytes."""
    from src.api.services.report_service import generate_query_report
    data = {
        "question": "What is X?",
        "short_answer": "X is...",
        "justification": "Because...",
        "retrieval_results": [],
    }
    pdf = generate_query_report(data)
    assert isinstance(pdf, bytes)
    assert len(pdf) > 500
    assert pdf[:4] == b"%PDF"


@pytest.mark.skip(reason="Requires API server running")
def test_report_api_endpoints():
    """Optional: hit report API when server is up."""
    base = "http://localhost:8000/api/report"
    for name, path, body in [
        ("patient-match", f"{base}/patient-match", {"patient_summary": "Test", "total_matches": 0, "matches": []}),
        ("treatment-comparison", f"{base}/treatment-comparison", {"comparison": {"treatment_a_name": "A", "treatment_b_name": "B", "treatment_a_evidence": {}, "treatment_b_evidence": {}, "comparison_summary": ""}, "sources": []}),
        ("query", f"{base}/query", {"question": "Q?", "short_answer": "A", "justification": "", "retrieval_results": []}),
    ]:
        r = requests.post(path, json=body, timeout=10)
        assert r.status_code == 200, f"{name}: {r.status_code}"
        assert r.headers.get("content-type", "").startswith("application/pdf")
        assert r.content[:4] == b"%PDF"
