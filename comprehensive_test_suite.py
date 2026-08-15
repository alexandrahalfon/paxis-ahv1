#!/usr/bin/env python3
"""
Comprehensive Test Suite for Paxis Medical Literature Platform

Tests all features including:
- RAG queries with various question types and complexities
- Patient matching with different profiles
- Treatment comparison
- Deep dive queries
- Enhanced queries
- All query modes

Generates a detailed HTML report with questions and answers.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import traceback


@dataclass
class TestResult:
    """Result of a single test case"""
    test_name: str
    endpoint: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None
    response_time_ms: float = 0.0
    timestamp: str = ""


class ComprehensiveTestSuite:
    """Comprehensive test suite for Paxis platform"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def run_test(self, test_name: str, endpoint: str, method: str = "POST", 
                 data: Optional[Dict] = None, params: Optional[Dict] = None,
                 validate_content: bool = True) -> TestResult:
        """Run a single test and record results"""
        url = f"{self.base_url}{endpoint}"
        result = TestResult(
            test_name=test_name,
            endpoint=endpoint,
            request_data=data or params or {},
            timestamp=datetime.now().isoformat()
        )
        
        try:
            start = time.time()
            if method == "GET":
                response = requests.get(url, params=params, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=60)
            
            result.response_time_ms = (time.time() - start) * 1000
            result.status_code = response.status_code
            
            if response.status_code == 200:
                result.response_data = response.json()
                
                # Validate content if requested
                if validate_content:
                    validation_error = self._validate_response_content(endpoint, result.response_data)
                    if validation_error:
                        result.success = False
                        result.error_message = validation_error
                    else:
                        result.success = True
                else:
                    result.success = True
            else:
                result.success = False
                result.error_message = f"HTTP {response.status_code}: {response.text[:500]}"
                
        except Exception as e:
            # Retry once for transient connection resets (e.g., during startup)
            if "Connection reset by peer" in str(e):
                time.sleep(2)
                try:
                    if method == "GET":
                        response = requests.get(url, params=params, timeout=60)
                    else:
                        response = requests.post(url, json=data, timeout=60)
                    result.response_time_ms = (time.time() - start) * 1000
                    result.status_code = response.status_code
                    if response.status_code == 200:
                        result.response_data = response.json()
                        if validate_content:
                            validation_error = self._validate_response_content(endpoint, result.response_data)
                            if validation_error:
                                result.success = False
                                result.error_message = validation_error
                            else:
                                result.success = True
                        else:
                            result.success = True
                    else:
                        result.success = False
                        result.error_message = f"HTTP {response.status_code}: {response.text[:500]}"
                except Exception as retry_error:
                    result.success = False
                    result.error_message = f"Exception: {str(retry_error)}\n{traceback.format_exc()[:500]}"
            else:
                result.success = False
                result.error_message = f"Exception: {str(e)}\n{traceback.format_exc()[:500]}"
            
        self.results.append(result)
        return result
    
    def _validate_response_content(self, endpoint: str, response_data: Dict[str, Any]) -> Optional[str]:
        """Validate that response contains expected content. Returns error message if invalid, None if valid."""

        def is_placeholder_text(text: str) -> bool:
            if not text:
                return True
            lowered = text.lower()
            placeholder_phrases = [
                "no relevant chunks were retrieved",
                "limited efficacy data available",
                "limited safety data available",
                "limited dosing data available",
                "found in 0 studies",
                "no data available",
            ]
            return any(phrase in lowered for phrase in placeholder_phrases)

        # Health check and utility endpoints - no validation needed
        if "/health" in endpoint or "/modes" in endpoint or "/sites" in endpoint or endpoint == "/":
            return None

        # Patient matching endpoint - must have matches
        if "/patient/match" in endpoint:
            matches = response_data.get("matches", [])
            total_matches = response_data.get("total_matches", 0)
            if total_matches == 0 or len(matches) == 0:
                return "Patient matching returned 0 matches. Expected at least 1 match for valid patient profile."
            return None

        # Enhanced query - must have short_answer and justification (check before general /query)
        if "/query/enhanced" in endpoint:
            short_answer = response_data.get("short_answer", "")
            justification = response_data.get("justification", "")
            if not short_answer or short_answer.strip() == "":
                return "Enhanced query returned empty short_answer."
            if is_placeholder_text(short_answer):
                return "Enhanced query short_answer is placeholder/empty."
            if not justification or justification.strip() == "":
                return "Enhanced query returned empty justification."
            # Also check for retrieval results
            retrieval_results = response_data.get("retrieval_results", [])
            if len(retrieval_results) == 0:
                return "Enhanced query returned no retrieval results/evidence chunks."
            return None

        # Deep dive query - must have non-empty summary and evidence
        if "/deep-dive" in endpoint:
            summary = response_data.get("summary", "")
            if not summary or summary.strip() == "":
                return "Deep dive returned empty summary."
            if is_placeholder_text(summary):
                return "Deep dive summary is placeholder/empty."
            evidence = response_data.get("evidence", [])
            if len(evidence) == 0:
                return "Deep dive returned no evidence chunks."
            return None

        # Treatment comparison - should find studies or meaningful evidence
        if "/comparison/treatments" in endpoint:
            comparison = response_data.get("comparison", {})
            treatment_a_evidence = comparison.get("treatment_a_evidence", {})
            treatment_b_evidence = comparison.get("treatment_b_evidence", {})
            studies_a = treatment_a_evidence.get("studies", [])
            studies_b = treatment_b_evidence.get("studies", [])
            summary = comparison.get("comparison_summary", "")

            if len(studies_a) == 0 and len(studies_b) == 0:
                return "Treatment comparison found 0 studies for both treatments."

            if is_placeholder_text(summary):
                return "Treatment comparison summary is placeholder/empty."

            return None

        # RAG query endpoints - must have non-empty answer and evidence (check after enhanced)
        if "/query" in endpoint:
            answer = response_data.get("answer", "")
            if not answer or answer.strip() == "":
                return "Query returned empty answer."
            if is_placeholder_text(answer):
                return "Query answer is placeholder/empty."
            retrieval_results = response_data.get("retrieval_results", [])
            if len(retrieval_results) == 0:
                return "Query returned no retrieval results/evidence chunks."
            return None

        # Default: no validation for unknown endpoints
        return None
    
    # ============================================
    # TEST CASES: RAG QUERIES
    # ============================================
    
    def test_rag_queries(self):
        """Test various RAG query types with different complexities"""
        print("\n" + "="*80)
        print("TESTING: RAG Queries")
        print("="*80)
        
        # Simple questions
        simple_questions = [
            "What is the standard RT dose for breast cancer?",
            "What is the treatment for lung cancer?",
            "What are the outcomes for prostate cancer?",
        ]
        
        # Complex questions with abbreviations
        complex_questions = [
            "What is the recommended RT dose and fractionation for early-stage NSCLC patients with EGFR mutations?",
            "What are the OS and PFS outcomes for HER2+ breast cancer patients treated with adjuvant RT and trastuzumab?",
            "What is the standard of care for stage III rectal cancer with neoadjuvant CRT followed by TME?",
        ]
        
        # Dose-specific questions
        dose_questions = [
            "What is the standard radiation dose for breast cancer in 15 fractions?",
            "What dose of RT is recommended for prostate cancer?",
            "What is the appropriate SBRT dose for lung tumors?",
        ]
        
        # Outcome questions
        outcome_questions = [
            "What are the 5-year overall survival rates for stage II breast cancer?",
            "What is the local control rate for head and neck cancer with IMRT?",
            "What are the recurrence rates for early-stage lung cancer treated with SBRT?",
        ]
        
        # Treatment recommendation questions
        treatment_questions = [
            "What is the recommended treatment for stage III NSCLC?",
            "What is the best approach for locally advanced rectal cancer?",
            "What treatment should be used for triple-negative breast cancer?",
        ]
        
        # Comparison questions
        comparison_questions = [
            "What is the difference between IMRT and 3D-CRT for prostate cancer?",
            "How does adjuvant RT compare to observation for breast cancer?",
            "What are the outcomes of SBRT vs conventional RT for lung cancer?",
        ]
        
        # Staging questions
        staging_questions = [
            "What is the staging system for breast cancer?",
            "What are the TNM criteria for lung cancer staging?",
            "What stage is T2N0M0 in breast cancer?",
        ]
        
        # Indication questions
        indication_questions = [
            "When is RT indicated for breast cancer?",
            "What are the indications for adjuvant RT in head and neck cancer?",
            "When should SBRT be used for lung cancer?",
        ]
        
        # Very detailed/complex questions
        detailed_questions = [
            "What is the recommended radiation therapy dose, fractionation, and technique for a 65-year-old woman with stage II (T2N1M0) ER+/PR+/HER2- invasive ductal carcinoma of the left breast who underwent BCS with negative margins and 2 positive sentinel nodes, considering her ECOG 0 performance status and no comorbidities?",
            "For a 58-year-old male with stage IIIA (T2N2M0) NSCLC, EGFR wild-type, PD-L1 50%, ECOG 1, what is the optimal treatment sequence including neoadjuvant or adjuvant therapy, and what are the expected outcomes?",
            "What are the long-term outcomes, including 10-year OS, DFS, and late toxicity rates, for patients with locally advanced prostate cancer (T3N0M0) treated with dose-escalated EBRT (78 Gy in 39 fractions) combined with ADT?",
        ]
        
        all_questions = [
            ("Simple Questions", simple_questions),
            ("Complex Questions (Abbreviations)", complex_questions),
            ("Dose-Specific Questions", dose_questions),
            ("Outcome Questions", outcome_questions),
            ("Treatment Recommendation Questions", treatment_questions),
            ("Comparison Questions", comparison_questions),
            ("Staging Questions", staging_questions),
            ("Indication Questions", indication_questions),
            ("Very Detailed Questions", detailed_questions),
        ]
        
        for category, questions in all_questions:
            print(f"\n--- {category} ---")
            for i, question in enumerate(questions, 1):
                test_name = f"RAG Query - {category} - Q{i}"
                print(f"  Testing: {question[:60]}...")
                self.run_test(
                    test_name=test_name,
                    endpoint="/api/rag/query",
                    data={"question": question, "top_k": 10}
                )
                time.sleep(0.5)  # Rate limiting
    
    def test_rag_query_modes(self):
        """Test different query modes"""
        print("\n" + "="*80)
        print("TESTING: RAG Query Modes")
        print("="*80)
        
        question = "What is the standard RT dose for breast cancer?"
        modes = ["naive", "local", "global", "hybrid"]
        
        for mode in modes:
            test_name = f"RAG Query Mode - {mode}"
            print(f"  Testing mode: {mode}")
            self.run_test(
                test_name=test_name,
                endpoint="/api/rag/query",
                data={"question": question, "query_mode": mode, "top_k": 10}
            )
            time.sleep(0.5)
    
    def test_rag_with_site_inference(self):
        """Test RAG queries with site inference"""
        print("\n" + "="*80)
        print("TESTING: RAG Queries with Site Inference")
        print("="*80)
        
        questions = [
            "What is the standard treatment?",
            "What are the outcomes?",
            "What dose is recommended?",
        ]
        
        for i, question in enumerate(questions, 1):
            test_name = f"RAG Query with Site Inference - Q{i}"
            print(f"  Testing: {question}")
            self.run_test(
                test_name=test_name,
                endpoint="/api/rag/query",
                data={
                    "question": question,
                    "use_site_inference": True,
                    "top_k": 10
                }
            )
            time.sleep(0.5)
    
    def test_enhanced_queries(self):
        """Test enhanced query endpoint (with short answer)"""
        print("\n" + "="*80)
        print("TESTING: Enhanced Queries (Short Answer)")
        print("="*80)
        
        questions = [
            "What is the standard RT dose for breast cancer?",
            "What outcome is associated with completion axillary LND for breast cancer with sentinel node micrometastasis?",
            "What is the recommended treatment for stage III NSCLC?",
            "What are the 5-year survival rates for early-stage breast cancer?",
        ]
        
        for i, question in enumerate(questions, 1):
            test_name = f"Enhanced Query - Q{i}"
            print(f"  Testing: {question[:60]}...")
            self.run_test(
                test_name=test_name,
                endpoint="/api/rag/query/enhanced",
                data={"question": question, "top_k": 10}
            )
            time.sleep(0.5)

    def test_agent_modes(self):
        """Test basic vs conversation agent modes"""
        print("\n" + "="*80)
        print("TESTING: Agent Modes (Basic vs Conversation)")
        print("="*80)

        base_question = "What is the standard RT dose for breast cancer?"
        followup_question = "What about for stage II?"
        conversation_history = [
            {"role": "user", "content": base_question},
            {"role": "assistant", "content": "Standard whole breast RT is commonly 40 Gy in 15 fractions or 42.5 Gy in 16 fractions."}
        ]

        # Basic mode
        self.run_test(
            test_name="Agent Mode - Basic",
            endpoint="/api/rag/query",
            data={"question": base_question, "query_mode": "basic", "top_k": 10}
        )
        time.sleep(0.5)

        # Conversation mode
        self.run_test(
            test_name="Agent Mode - Conversation",
            endpoint="/api/rag/query",
            data={
                "question": followup_question,
                "query_mode": "conversation",
                "top_k": 10,
                "conversation_history": conversation_history
            }
        )
        time.sleep(0.5)

        # Enhanced conversation mode
        self.run_test(
            test_name="Agent Mode - Enhanced Conversation",
            endpoint="/api/rag/query/enhanced",
            data={
                "question": followup_question,
                "query_mode": "conversation",
                "top_k": 10,
                "conversation_history": conversation_history
            }
        )
        time.sleep(0.5)
    
    def test_deep_dive_queries(self):
        """Test deep dive queries with site context"""
        print("\n" + "="*80)
        print("TESTING: Deep Dive Queries")
        print("="*80)
        
        test_cases = [
            {
                "question": "What is the recommended chemotherapy regimen?",
                "site_key": "Breast",
            },
            {
                "question": "What are the treatment outcomes?",
                "site_key": "Lung",
            },
            {
                "question": "What dose is recommended?",
                "site_key": None,  # Auto-infer
            },
        ]
        
        for i, case in enumerate(test_cases, 1):
            test_name = f"Deep Dive Query - Q{i}"
            print(f"  Testing: {case['question']} (site: {case['site_key'] or 'auto'})")
            self.run_test(
                test_name=test_name,
                endpoint="/api/rag/deep-dive",
                data={
                    "question": case["question"],
                    "site_key": case["site_key"],
                    "top_k": 15
                }
            )
            time.sleep(0.5)
    
    # ============================================
    # TEST CASES: PATIENT MATCHING
    # ============================================
    
    def test_patient_matching(self):
        """Test patient matching with various profiles"""
        print("\n" + "="*80)
        print("TESTING: Patient Matching")
        print("="*80)
        
        profiles = [
            {
                "name": "Early-stage breast cancer patient",
                "profile": {
                    "age": 55,
                    "gender": "female",
                    "cancer_type": "breast cancer",
                    "cancer_stage": "II",
                    "histology": "invasive ductal carcinoma",
                    "molecular_markers": ["ER+", "PR+", "HER2-"],
                    "performance_status": "ECOG 0",
                }
            },
            {
                "name": "Advanced lung cancer patient",
                "profile": {
                    "age": 68,
                    "gender": "male",
                    "cancer_type": "non-small cell lung cancer",
                    "cancer_stage": "III",
                    "histology": "adenocarcinoma",
                    "molecular_markers": ["EGFR+", "PD-L1 50%"],
                    "performance_status": "ECOG 1",
                    "smoking_status": "former",
                }
            },
            {
                "name": "Prostate cancer patient",
                "profile": {
                    "age": 72,
                    "gender": "male",
                    "cancer_type": "prostate cancer",
                    "cancer_stage": "III",
                    "performance_status": "ECOG 0",
                }
            },
            {
                "name": "Head and neck cancer patient",
                "profile": {
                    "age": 58,
                    "gender": "male",
                    "cancer_type": "head and neck squamous cell carcinoma",
                    "cancer_stage": "IV",
                    "histology": "squamous cell carcinoma",
                    "performance_status": "ECOG 1",
                    "smoking_status": "current",
                }
            },
            {
                "name": "Minimal profile patient",
                "profile": {
                    "cancer_type": "breast cancer",
                    "cancer_stage": "I",
                }
            },
            {
                "name": "Complex profile with comorbidities",
                "profile": {
                    "age": 65,
                    "gender": "female",
                    "cancer_type": "colorectal cancer",
                    "cancer_stage": "III",
                    "histology": "adenocarcinoma",
                    "molecular_markers": ["MSI-H"],
                    "performance_status": "ECOG 1",
                    "comorbidities": ["diabetes", "hypertension"],
                    "smoking_status": "never",
                }
            },
        ]
        
        for profile_data in profiles:
            test_name = f"Patient Matching - {profile_data['name']}"
            print(f"  Testing: {profile_data['name']}")
            self.run_test(
                test_name=test_name,
                endpoint="/api/rag/patient/match",
                data=profile_data["profile"]
            )
            time.sleep(0.5)
    
    # ============================================
    # TEST CASES: TREATMENT COMPARISON
    # ============================================
    
    def test_treatment_comparison(self):
        """Test treatment comparison endpoint"""
        print("\n" + "="*80)
        print("TESTING: Treatment Comparison")
        print("="*80)
        
        comparisons = [
            {
                "name": "IMRT vs 3D-CRT for prostate",
                "treatment_a": "IMRT",
                "treatment_b": "3D-CRT",
                "cancer_type": "prostate cancer",
            },
            {
                "name": "Adjuvant RT vs observation for breast",
                "treatment_a": "adjuvant radiation therapy",
                "treatment_b": "observation",
                "cancer_type": "breast cancer",
            },
            {
                "name": "SBRT vs conventional RT for lung",
                "treatment_a": "SBRT",
                "treatment_b": "conventional radiation therapy",
                "cancer_type": "lung cancer",
            },
            {
                "name": "Chemoradiation vs surgery for rectal",
                "treatment_a": "chemoradiation",
                "treatment_b": "surgery",
                "cancer_type": "rectal cancer",
            },
            {
                "name": "IMRT vs VMAT for head and neck",
                "treatment_a": "IMRT",
                "treatment_b": "VMAT",
                "cancer_type": "head and neck cancer",
            },
            {
                "name": "General comparison without cancer type",
                "treatment_a": "radiation therapy",
                "treatment_b": "surgery",
                "cancer_type": None,
            },
        ]
        
        for comp in comparisons:
            test_name = f"Treatment Comparison - {comp['name']}"
            print(f"  Testing: {comp['name']}")
            data = {
                "treatment_a": comp["treatment_a"],
                "treatment_b": comp["treatment_b"],
                "top_k": 10,
            }
            if comp["cancer_type"]:
                data["cancer_type"] = comp["cancer_type"]
            if "stage" in comp:
                data["stage"] = comp["stage"]
                
            self.run_test(
                test_name=test_name,
                endpoint="/api/rag/comparison/treatments",
                data=data
            )
            time.sleep(0.5)
    
    # ============================================
    # TEST CASES: UTILITY ENDPOINTS
    # ============================================
    
    def test_utility_endpoints(self):
        """Test utility endpoints (health, modes, sites)"""
        print("\n" + "="*80)
        print("TESTING: Utility Endpoints")
        print("="*80)
        
        # Health check
        print("  Testing: Health Check")
        self.run_test(
            test_name="Health Check",
            endpoint="/api/rag/health",
            method="GET"
        )
        
        # Query modes
        print("  Testing: Query Modes")
        self.run_test(
            test_name="Query Modes",
            endpoint="/api/rag/modes",
            method="GET"
        )
        
        # Available sites
        print("  Testing: Available Sites")
        self.run_test(
            test_name="Available Sites",
            endpoint="/api/rag/sites",
            method="GET"
        )
        
        # Root endpoint
        print("  Testing: Root Endpoint")
        self.run_test(
            test_name="Root Endpoint",
            endpoint="/",
            method="GET"
        )
        
        # Health endpoint (non-RAG)
        print("  Testing: General Health Endpoint")
        self.run_test(
            test_name="General Health",
            endpoint="/health",
            method="GET"
        )
    
    # ============================================
    # RUN ALL TESTS
    # ============================================
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE FOR PAXIS PLATFORM")
        print("="*80)
        print(f"Base URL: {self.base_url}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        try:
            # Test utility endpoints first (quick checks)
            self.test_utility_endpoints()
            
            # Test RAG queries
            self.test_rag_queries()
            self.test_rag_query_modes()
            self.test_rag_with_site_inference()
            self.test_agent_modes()
            self.test_enhanced_queries()
            self.test_deep_dive_queries()
            
            # Test patient matching
            self.test_patient_matching()
            
            # Test treatment comparison
            self.test_treatment_comparison()
            
        except KeyboardInterrupt:
            print("\n\nTest suite interrupted by user")
        except Exception as e:
            print(f"\n\nFatal error in test suite: {e}")
            traceback.print_exc()
        
        # Generate report
        self.generate_report()
    
    # ============================================
    # REPORT GENERATION
    # ============================================
    
    def generate_report(self):
        """Generate comprehensive HTML report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(r.response_time_ms for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.test_name.split(" - ")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paxis Comprehensive Test Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .summary-card.failure {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        .summary-card h3 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .summary-card p {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .category {{
            margin-bottom: 40px;
        }}
        .category-header {{
            background: #34495e;
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .test-case {{
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .test-header {{
            background: #ecf0f1;
            padding: 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }}
        .test-header:hover {{
            background: #d5dbdb;
        }}
        .test-header.success {{
            background: #d5f4e6;
            border-left: 4px solid #27ae60;
        }}
        .test-header.failure {{
            background: #fadbd8;
            border-left: 4px solid #e74c3c;
        }}
        .test-content {{
            padding: 20px;
            display: none;
            background: #fafafa;
        }}
        .test-content.active {{
            display: block;
        }}
        .status-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .status-badge.success {{
            background: #27ae60;
            color: white;
        }}
        .status-badge.failure {{
            background: #e74c3c;
            color: white;
        }}
        .detail-section {{
            margin-bottom: 20px;
        }}
        .detail-section h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #3498db;
        }}
        .detail-section pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.85em;
            line-height: 1.4;
        }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .metadata-item {{
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .metadata-item strong {{
            color: #2c3e50;
        }}
        .answer-box {{
            background: #e8f8f5;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .error-box {{
            background: #fadbd8;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Paxis Comprehensive Test Report</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>{total_tests}</h3>
                <p>Total Tests</p>
            </div>
            <div class="summary-card success">
                <h3>{passed_tests}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card failure">
                <h3>{failed_tests}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card">
                <h3>{avg_response_time:.0f}ms</h3>
                <p>Avg Response Time</p>
            </div>
            <div class="summary-card">
                <h3>{duration:.1f}s</h3>
                <p>Total Duration</p>
            </div>
        </div>
        
        <div class="metadata">
            <div class="metadata-item">
                <strong>Start Time:</strong><br>
                {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div class="metadata-item">
                <strong>End Time:</strong><br>
                {end_time.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div class="metadata-item">
                <strong>Base URL:</strong><br>
                {self.base_url}
            </div>
            <div class="metadata-item">
                <strong>Success Rate:</strong><br>
                {(passed_tests/total_tests*100):.1f}%
            </div>
        </div>
"""
        
        # Add test results by category
        for category, results in sorted(categories.items()):
            category_passed = sum(1 for r in results if r.success)
            category_total = len(results)
            
            html += f"""
        <div class="category">
            <div class="category-header">
                {category} ({category_passed}/{category_total} passed)
            </div>
"""
            
            for result in results:
                status_class = "success" if result.success else "failure"
                status_text = "✓ PASS" if result.success else "✗ FAIL"
                
                # Extract answer from response
                answer_html = ""
                if result.success and result.response_data:
                    if "answer" in result.response_data:
                        answer_html = f"""
                <div class="answer-box">
                    <strong>Answer:</strong><br>
                    {result.response_data.get('answer', 'N/A')[:1000]}
                </div>
"""
                    elif "short_answer" in result.response_data:
                        answer_html = f"""
                <div class="answer-box">
                    <strong>Short Answer:</strong><br>
                    {result.response_data.get('short_answer', 'N/A')}<br><br>
                    <strong>Justification:</strong><br>
                    {result.response_data.get('justification', 'N/A')[:1000]}
                </div>
"""
                    elif "summary" in result.response_data:
                        answer_html = f"""
                <div class="answer-box">
                    <strong>Summary:</strong><br>
                    {result.response_data.get('summary', 'N/A')[:1000]}
                </div>
"""
                    elif "matches" in result.response_data:
                        matches = result.response_data.get('matches', [])
                        answer_html = f"""
                <div class="answer-box">
                    <strong>Matches Found:</strong> {len(matches)}<br>
                    <strong>Patient Summary:</strong> {result.response_data.get('patient_summary', 'N/A')}
                </div>
"""
                    elif "comparison" in result.response_data:
                        comp = result.response_data.get('comparison', {})
                        answer_html = f"""
                <div class="answer-box">
                    <strong>Comparison Summary:</strong><br>
                    {comp.get('comparison_summary', 'N/A')[:500]}
                </div>
"""
                
                error_html = ""
                if not result.success:
                    error_html = f"""
                <div class="error-box">
                    <strong>Error:</strong><br>
                    {result.error_message or 'Unknown error'}
                </div>
"""
                
                html += f"""
            <div class="test-case">
                <div class="test-header {status_class}" onclick="toggleTest(this)">
                    <div>
                        <strong>{result.test_name}</strong>
                        <span class="timestamp"> - {result.timestamp}</span>
                    </div>
                    <div>
                        <span class="status-badge {status_class}">{status_text}</span>
                        <span style="margin-left: 15px; color: #7f8c8d;">{result.response_time_ms:.0f}ms</span>
                    </div>
                </div>
                <div class="test-content">
                    <div class="detail-section">
                        <h4>Request</h4>
                        <pre>{json.dumps(result.request_data, indent=2)}</pre>
                    </div>
                    {answer_html}
                    {error_html}
                    {f'''
                    <div class="detail-section">
                        <h4>Response Metadata</h4>
                        <pre>{json.dumps(result.response_data, indent=2)[:2000]}</pre>
                    </div>
                    ''' if result.success and result.response_data else ''}
                </div>
            </div>
"""
            
            html += """
        </div>
"""
        
        html += """
    </div>
    
    <script>
        function toggleTest(header) {
            const content = header.nextElementSibling;
            content.classList.toggle('active');
        }
        
        // Auto-expand failed tests
        document.addEventListener('DOMContentLoaded', function() {
            const failedTests = document.querySelectorAll('.test-header.failure');
            failedTests.forEach(header => {
                header.nextElementSibling.classList.add('active');
            });
        });
    </script>
</body>
</html>
"""
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_report_{timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Also save JSON report
        json_filename = f"comprehensive_test_report_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests / total_tests * 100 if total_tests > 0 else 0,
                    "avg_response_time_ms": avg_response_time,
                    "duration_seconds": duration,
                    "start_time": self.start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "base_url": self.base_url,
                },
                "results": [asdict(r) for r in self.results]
            }, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("TEST SUITE COMPLETE")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"Average Response Time: {avg_response_time:.0f}ms")
        print(f"Total Duration: {duration:.1f}s")
        print(f"\n📄 HTML Report: {filename}")
        print(f"📄 JSON Report: {json_filename}")
        print("="*80)


if __name__ == "__main__":
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Starting comprehensive test suite...")
    print(f"Target URL: {base_url}")
    print("Make sure the API server is running!")
    print("\nPress Ctrl+C to stop early\n")
    
    time.sleep(2)  # Give user time to read
    
    suite = ComprehensiveTestSuite(base_url=base_url)
    suite.run_all_tests()
