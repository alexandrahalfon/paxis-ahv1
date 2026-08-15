#!/usr/bin/env python3
"""
Paxis Backend Route Verification Script

Tests all API routes and frontend integration to ensure proper deployment.
Run this after deployment to verify everything works.

Usage:
    python verify_deployment.py http://localhost:8080
    python verify_deployment.py https://your-service-url.run.app
"""

import sys
import requests
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single test."""
    endpoint: str
    method: str
    passed: bool
    status_code: int = None
    message: str = ""
    response_time: float = 0.0


class DeploymentVerifier:
    """Verify Paxis backend deployment."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results: List[TestResult] = []
    
    def test_endpoint(self, method: str, path: str, name: str, 
                     expected_status: int = 200, 
                     json_data: Dict = None,
                     check_response: callable = None) -> TestResult:
        """Test a single endpoint."""
        url = f"{self.base_url}{path}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=json_data, timeout=30)
            else:
                return TestResult(
                    endpoint=name,
                    method=method,
                    passed=False,
                    message=f"Unsupported method: {method}"
                )
            
            passed = response.status_code == expected_status
            message = "OK"
            
            if passed and check_response:
                try:
                    data = response.json()
                    check_result = check_response(data)
                    if not check_result[0]:
                        passed = False
                        message = check_result[1]
                except Exception as e:
                    passed = False
                    message = f"Response validation failed: {str(e)}"
            
            if not passed and message == "OK":
                message = f"Expected status {expected_status}, got {response.status_code}"
            
            return TestResult(
                endpoint=name,
                method=method,
                passed=passed,
                status_code=response.status_code,
                message=message,
                response_time=response.elapsed.total_seconds()
            )
            
        except requests.exceptions.Timeout:
            return TestResult(
                endpoint=name,
                method=method,
                passed=False,
                message="Request timeout (>30s)"
            )
        except requests.exceptions.ConnectionError:
            return TestResult(
                endpoint=name,
                method=method,
                passed=False,
                message="Connection failed - is the service running?"
            )
        except Exception as e:
            return TestResult(
                endpoint=name,
                method=method,
                passed=False,
                message=f"Error: {str(e)}"
            )
    
    def run_all_tests(self):
        """Run all verification tests."""
        print(f"\n{'='*80}")
        print(f"Paxis Backend Deployment Verification")
        print(f"Testing: {self.base_url}")
        print(f"{'='*80}\n")
        
        # Test 1: Root endpoint
        result = self.test_endpoint(
            "GET", "/", "Root endpoint",
            check_response=lambda r: (
                r.get("status") == "ok",
                "Missing 'status' field or not 'ok'"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 2: Basic health check
        result = self.test_endpoint(
            "GET", "/health", "Health check",
            check_response=lambda r: (
                r.get("status") == "healthy",
                "Status not 'healthy'"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 3: RAG health check
        result = self.test_endpoint(
            "GET", "/api/rag/health", "RAG health check",
            check_response=lambda r: (
                "status" in r,
                "Missing 'status' field"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 4: Query modes
        result = self.test_endpoint(
            "GET", "/api/rag/modes", "Query modes endpoint",
            check_response=lambda r: (
                "modes" in r and len(r["modes"]) > 0,
                "No modes returned"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 5: Available sites
        result = self.test_endpoint(
            "GET", "/api/rag/sites", "Available sites endpoint",
            check_response=lambda r: (
                "sites" in r and len(r["sites"]) > 0,
                "No sites returned"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 6: Simple query
        result = self.test_endpoint(
            "POST", "/api/rag/query", "RAG query endpoint",
            json_data={
                "question": "What is radiation therapy?",
                "top_k": 5
            },
            check_response=lambda r: (
                "answer" in r and "retrieval_results" in r,
                "Missing 'answer' or 'retrieval_results'"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 7: Deep dive query
        result = self.test_endpoint(
            "POST", "/api/rag/deep-dive", "RAG deep dive endpoint",
            json_data={
                "question": "What are the treatment options?",
                "top_k": 10
            },
            check_response=lambda r: (
                "summary" in r and "evidence" in r,
                "Missing 'summary' or 'evidence'"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 8: Patient matching
        result = self.test_endpoint(
            "POST", "/api/rag/patient/match", "Patient matching endpoint",
            json_data={
                "age": 65,
                "gender": "male",
                "cancer_type": "lung",
                "cancer_stage": "III"
            },
            check_response=lambda r: (
                "matches" in r,
                "Missing 'matches'"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 9: Treatment comparison
        result = self.test_endpoint(
            "POST", "/api/rag/comparison/treatments", "Treatment comparison endpoint",
            json_data={
                "treatment_a": "pembrolizumab",
                "treatment_b": "chemotherapy",
                "cancer_type": "lung"
            },
            check_response=lambda r: (
                "comparison" in r,
                "Missing 'comparison'"
            )
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 10: FastAPI docs
        result = self.test_endpoint(
            "GET", "/docs", "API documentation"
        )
        self.results.append(result)
        self._print_result(result)
        
        # Test 11: Frontend index page
        result = self.test_endpoint(
            "GET", "/", "Frontend index page",
            check_response=lambda r: (
                False,  # We're checking HTML, not JSON
                "N/A"
            )
        )
        # Override the check for HTML content
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                if "Paxis" in response.text or "<!DOCTYPE html>" in response.text:
                    result = TestResult(
                        endpoint="Frontend index page",
                        method="GET",
                        passed=True,
                        status_code=200,
                        message="HTML content loaded"
                    )
                else:
                    result = TestResult(
                        endpoint="Frontend index page",
                        method="GET",
                        passed=False,
                        status_code=200,
                        message="Unexpected content"
                    )
        except:
            pass
        self.results[-1] = result
        self._print_result(result)
        
        # Test 12: Frontend static files
        result = self.test_endpoint(
            "GET", "/css/styles.css", "Frontend CSS"
        )
        self.results.append(result)
        self._print_result(result)
        
        # Print summary
        self._print_summary()
    
    def _print_result(self, result: TestResult):
        """Print a single test result."""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        time_str = f"({result.response_time:.2f}s)" if result.response_time else ""
        status_code = f"[{result.status_code}]" if result.status_code else ""
        
        print(f"{status} {result.method:4s} {result.endpoint:40s} {status_code:6s} {time_str}")
        
        if not result.passed and result.message:
            print(f"      └─ {result.message}")
    
    def _print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print(f"\n{'='*80}")
        print(f"Summary: {passed}/{total} tests passed")
        
        if failed > 0:
            print(f"\n❌ {failed} test(s) failed:")
            for result in self.results:
                if not result.passed:
                    print(f"   - {result.endpoint}: {result.message}")
        else:
            print("\n✅ All tests passed! Deployment is working correctly.")
        
        print(f"{'='*80}\n")
        
        return failed == 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python verify_deployment.py <base_url>")
        print("\nExamples:")
        print("  python verify_deployment.py http://localhost:8080")
        print("  python verify_deployment.py https://your-service.run.app")
        sys.exit(1)
    
    base_url = sys.argv[1]
    verifier = DeploymentVerifier(base_url)
    
    try:
        verifier.run_all_tests()
        sys.exit(0 if all(r.passed for r in verifier.results) else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nVerification failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
