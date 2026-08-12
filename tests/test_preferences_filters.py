"""
Test suite for User Preferences and Sort/Filter functionality
Tests all filter options and sorting capabilities
"""

import asyncio
import httpx
import json
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8000/api"
TEST_USER_EMAIL = "test_prefs@example.com"
TEST_USER_PASSWORD = "testpassword123"


class PreferencesFilterTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = []
        
    async def close(self):
        await self.client.aclose()
        
    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"       {details}")
    
    async def setup_auth(self):
        """Register and login test user"""
        print("\n=== Setting up authentication ===")
        
        # Try to register (may fail if user exists)
        try:
            resp = await self.client.post(
                f"{self.base_url}/auth/register",
                json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
            )
            if resp.status_code == 200:
                print(f"Registered new user: {TEST_USER_EMAIL}")
        except Exception as e:
            print(f"Registration skipped (user may exist): {e}")
        
        # Login
        resp = await self.client.post(
            f"{self.base_url}/auth/login",
            json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            self.token = data.get("access_token")
            self.log_result("Authentication Setup", True, f"Token obtained for {TEST_USER_EMAIL}")
            return True
        else:
            self.log_result("Authentication Setup", False, f"Login failed: {resp.status_code}")
            return False
    
    # ==========================================
    # Test Filter Options Endpoints
    # ==========================================
    
    async def test_get_countries(self):
        """Test GET /user-preferences/countries"""
        print("\n--- Testing Countries Filter ---")
        
        # Test without search
        resp = await self.client.get(
            f"{self.base_url}/user-preferences/countries",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            options = data.get("options", [])
            total = data.get("total", 0)
            self.log_result(
                "Get Countries (no search)", 
                True, 
                f"Found {total} countries"
            )
            
            # Show sample countries
            if options:
                sample = [o["value"] for o in options[:5]]
                print(f"       Sample: {sample}")
        else:
            self.log_result("Get Countries (no search)", False, f"Status: {resp.status_code}")
        
        # Test with search
        resp = await self.client.get(
            f"{self.base_url}/user-preferences/countries?search=United",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            options = data.get("options", [])
            self.log_result(
                "Get Countries (search='United')", 
                True, 
                f"Found {len(options)} matching countries"
            )
        else:
            self.log_result("Get Countries (search)", False, f"Status: {resp.status_code}")
    
    async def test_get_institutions(self):
        """Test GET /user-preferences/institutions"""
        print("\n--- Testing Institutions Filter ---")
        
        # Test without search
        resp = await self.client.get(
            f"{self.base_url}/user-preferences/institutions",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            options = data.get("options", [])
            total = data.get("total", 0)
            self.log_result(
                "Get Institutions (no search)", 
                True, 
                f"Found {total} institutions"
            )
            
            # Show sample institutions
            if options:
                sample = [o["value"][:30] for o in options[:3]]
                print(f"       Sample: {sample}")
        else:
            self.log_result("Get Institutions (no search)", False, f"Status: {resp.status_code}")
        
        # Test with search
        resp = await self.client.get(
            f"{self.base_url}/user-preferences/institutions?search=University",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            options = data.get("options", [])
            self.log_result(
                "Get Institutions (search='University')", 
                True, 
                f"Found {len(options)} matching institutions"
            )
        else:
            self.log_result("Get Institutions (search)", False, f"Status: {resp.status_code}")
    
    async def test_get_race_ethnicities(self):
        """Test GET /user-preferences/race-ethnicities"""
        print("\n--- Testing Race/Ethnicity Filter Options ---")
        
        resp = await self.client.get(
            f"{self.base_url}/user-preferences/race-ethnicities",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            options = data.get("options", [])
            total = data.get("total", 0)
            self.log_result(
                "Get Race/Ethnicity Options", 
                True, 
                f"Found {total} race categories"
            )
            
            # Show available races
            if options:
                sample = [(o["value"], o["count"]) for o in options[:5]]
                print(f"       Available: {sample}")
        else:
            self.log_result("Get Race/Ethnicity Options", False, f"Status: {resp.status_code}")
    
    # ==========================================
    # Test Preferences CRUD
    # ==========================================
    
    async def test_save_preferences(self):
        """Test POST /user-preferences"""
        print("\n--- Testing Save Preferences ---")
        
        test_preferences = {
            "study_types": ["RCT", "Phase III"],
            "study_phases": ["Phase III", "Phase II"],
            "cancer_types": ["breast", "lung"],
            "min_patients": 100,
            "max_patients": 5000,
            "analysis_types": ["ITT", "Per-protocol"],
            "countries": ["United States", "Germany"],
            "institutions": [],
            "min_publication_year": 2015,
            "max_publication_year": 2025,
            "require_peer_reviewed": True,
            "min_followup_months": 24,
            "required_outcomes": ["overall_survival", "progression_free_survival"],
            "sort_by": "date",
            "sort_order": "desc",
            "filters_active": True,
            "results_per_page": 20
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=test_preferences
        )
        
        if resp.status_code == 200:
            data = resp.json()
            self.log_result(
                "Save Preferences", 
                data.get("success", False), 
                f"ID: {data.get('id')}"
            )
        else:
            self.log_result("Save Preferences", False, f"Status: {resp.status_code}, Body: {resp.text}")
    
    async def test_get_preferences(self):
        """Test GET /user-preferences"""
        print("\n--- Testing Get Preferences ---")
        
        resp = await self.client.get(
            f"{self.base_url}/user-preferences",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            self.log_result(
                "Get Preferences", 
                True, 
                f"Sort by: {data.get('sort_by')}, Active: {data.get('filters_active')}"
            )
            
            # Verify saved values
            if data.get("study_types"):
                print(f"       Study types: {data.get('study_types')}")
            if data.get("cancer_types"):
                print(f"       Cancer types: {data.get('cancer_types')}")
            if data.get("countries"):
                print(f"       Countries: {data.get('countries')}")
        else:
            self.log_result("Get Preferences", False, f"Status: {resp.status_code}")
    
    async def test_delete_preferences(self):
        """Test DELETE /user-preferences"""
        print("\n--- Testing Delete Preferences ---")
        
        resp = await self.client.delete(
            f"{self.base_url}/user-preferences",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            self.log_result(
                "Delete Preferences", 
                data.get("success", False), 
                data.get("message", "")
            )
        else:
            self.log_result("Delete Preferences", False, f"Status: {resp.status_code}")
        
        # Verify deletion - should return defaults
        resp = await self.client.get(
            f"{self.base_url}/user-preferences",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            # After deletion, should have default values
            is_default = (
                data.get("sort_by") == "relevance" and
                len(data.get("study_types", [])) == 0
            )
            self.log_result(
                "Verify Preferences Reset", 
                is_default, 
                f"Sort by: {data.get('sort_by')}, Study types: {data.get('study_types')}"
            )
    
    # ==========================================
    # Test Sort Options
    # ==========================================
    
    async def test_sort_options(self):
        """Test all sort_by options"""
        print("\n--- Testing Sort Options ---")
        
        sort_options = ["relevance", "date", "population", "citations", "outcomes", "patient_relevance"]
        
        for sort_by in sort_options:
            test_prefs = {
                "sort_by": sort_by,
                "sort_order": "desc",
                "filters_active": True
            }
            
            resp = await self.client.post(
                f"{self.base_url}/user-preferences",
                headers=self._headers(),
                json=test_prefs
            )
            
            if resp.status_code == 200:
                # Verify it was saved
                get_resp = await self.client.get(
                    f"{self.base_url}/user-preferences",
                    headers=self._headers()
                )
                if get_resp.status_code == 200:
                    saved = get_resp.json()
                    passed = saved.get("sort_by") == sort_by
                    self.log_result(f"Sort by '{sort_by}'", passed)
                else:
                    self.log_result(f"Sort by '{sort_by}'", False, "Failed to verify")
            else:
                self.log_result(f"Sort by '{sort_by}'", False, f"Status: {resp.status_code}")
        
        # Test sort order
        for order in ["asc", "desc"]:
            test_prefs = {
                "sort_by": "date",
                "sort_order": order,
                "filters_active": True
            }
            
            resp = await self.client.post(
                f"{self.base_url}/user-preferences",
                headers=self._headers(),
                json=test_prefs
            )
            
            if resp.status_code == 200:
                get_resp = await self.client.get(
                    f"{self.base_url}/user-preferences",
                    headers=self._headers()
                )
                if get_resp.status_code == 200:
                    saved = get_resp.json()
                    passed = saved.get("sort_order") == order
                    self.log_result(f"Sort order '{order}'", passed)
    
    # ==========================================
    # Test Citation Count Endpoints
    # ==========================================
    
    async def test_citation_count(self):
        """Test citation count endpoints"""
        print("\n--- Testing Citation Count ---")
        
        # Test single citation count with a known DOI
        test_doi = "10.1056/NEJMoa1606774"  # Common oncology paper
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences/citation-count",
            headers=self._headers(),
            json={"doi": test_doi}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            count = data.get("citation_count")
            self.log_result(
                "Get Citation Count (single)", 
                True, 
                f"DOI: {test_doi}, Citations: {count}"
            )
        else:
            self.log_result(
                "Get Citation Count (single)", 
                False, 
                f"Status: {resp.status_code}"
            )
        
        # Test batch citation counts
        batch_papers = [
            {"doi": "10.1056/NEJMoa1606774"},
            {"pmid": "28552987"}
        ]
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences/citation-counts/batch",
            headers=self._headers(),
            json={"papers": batch_papers}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            fetched = data.get("total_fetched", 0)
            self.log_result(
                "Get Citation Counts (batch)", 
                True, 
                f"Fetched: {fetched}/{len(batch_papers)}"
            )
        else:
            self.log_result(
                "Get Citation Counts (batch)", 
                False, 
                f"Status: {resp.status_code}"
            )
    
    async def test_studies_by_citations(self):
        """Test studies sorted by citations"""
        print("\n--- Testing Studies by Citations ---")
        
        resp = await self.client.get(
            f"{self.base_url}/user-preferences/studies-by-citations?limit=10",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            studies = data.get("studies", [])
            total = data.get("total", 0)
            self.log_result(
                "Get Studies by Citations", 
                True, 
                f"Found {total} studies with citation data"
            )
            
            if studies:
                top = studies[0]
                print(f"       Top cited: {top.get('study_name', 'N/A')[:50]}... ({top.get('citation_count')} citations)")
        else:
            self.log_result(
                "Get Studies by Citations", 
                False, 
                f"Status: {resp.status_code}"
            )
    
    async def test_patient_relevance_sort(self):
        """Test patient relevance sorting with smart search"""
        print("\n--- Testing Patient Relevance Sort ---")
        
        # Test smart search with patient_relevance sort
        test_query = "65 year old male with stage III squamous cell carcinoma of the oral cavity"
        
        # First set preferences to use patient_relevance sort
        prefs = {
            "sort_by": "patient_relevance",
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=prefs
        )
        
        if resp.status_code == 200:
            self.log_result("Set patient_relevance sort preference", True)
        else:
            self.log_result("Set patient_relevance sort preference", False, f"Status: {resp.status_code}")
            return
        
        # Test smart search endpoint
        resp = await self.client.post(
            f"{self.base_url}/smart-search",
            headers=self._headers(),
            json={
                "query": test_query,
                "top_k": 5,
                "use_preferences": True,
                "use_case_context": True
            }
        )
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            case_context = data.get("case_context", {})
            sort_by = data.get("sort_by", "")
            
            self.log_result(
                "Smart Search with Patient Relevance", 
                True, 
                f"Found {len(results)} results, sort_by={sort_by}"
            )
            
            # Check if case context was extracted
            if case_context.get("cancer_type") or case_context.get("cancer_location"):
                print(f"       Extracted: cancer_type={case_context.get('cancer_type')}, location={case_context.get('cancer_location')}")
            
            # Check if patient_relevance_score is present
            if results and results[0].get("patient_relevance_score", 0) > 0:
                print(f"       Top result patient_relevance_score: {results[0].get('patient_relevance_score'):.1f}")
        else:
            self.log_result(
                "Smart Search with Patient Relevance", 
                False, 
                f"Status: {resp.status_code}"
            )
        
        # Test extract-context endpoint
        resp = await self.client.post(
            f"{self.base_url}/smart-search/extract-context?query={test_query}",
            headers=self._headers()
        )
        
        if resp.status_code == 200:
            data = resp.json()
            self.log_result(
                "Extract Case Context", 
                True, 
                f"Age={data.get('age')}, Cancer={data.get('cancer_type')}"
            )
        else:
            self.log_result(
                "Extract Case Context", 
                False, 
                f"Status: {resp.status_code}"
            )
    
    # ==========================================
    # Test Filter Combinations
    # ==========================================
    
    async def test_filter_combinations(self):
        """Test various filter combinations"""
        print("\n--- Testing Filter Combinations ---")
        
        # Test 1: Study type + Phase filter
        combo1 = {
            "study_types": ["RCT"],
            "study_phases": ["Phase III"],
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo1
        )
        self.log_result(
            "Filter: Study Type + Phase", 
            resp.status_code == 200
        )
        
        # Test 2: Patient count range
        combo2 = {
            "min_patients": 50,
            "max_patients": 1000,
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo2
        )
        self.log_result(
            "Filter: Patient Count Range", 
            resp.status_code == 200
        )
        
        # Test 3: Year range + Peer reviewed
        combo3 = {
            "min_publication_year": 2018,
            "max_publication_year": 2024,
            "require_peer_reviewed": True,
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo3
        )
        self.log_result(
            "Filter: Year Range + Peer Reviewed", 
            resp.status_code == 200
        )
        
        # Test 4: Geographic filter
        combo4 = {
            "countries": ["United States", "United Kingdom"],
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo4
        )
        self.log_result(
            "Filter: Geographic (Countries)", 
            resp.status_code == 200
        )
        
        # Test 5: Outcome requirements
        combo5 = {
            "required_outcomes": ["overall_survival", "progression_free_survival"],
            "min_followup_months": 36,
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo5
        )
        self.log_result(
            "Filter: Outcome Requirements", 
            resp.status_code == 200
        )
        
        # Test 6: Treatment modalities filter
        combo6 = {
            "treatment_modalities": ["surgery", "radiation", "chemotherapy"],
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo6
        )
        
        if resp.status_code == 200:
            # Verify it was saved
            get_resp = await self.client.get(
                f"{self.base_url}/user-preferences",
                headers=self._headers()
            )
            if get_resp.status_code == 200:
                saved = get_resp.json()
                passed = saved.get("treatment_modalities") == ["surgery", "radiation", "chemotherapy"]
                self.log_result(
                    "Filter: Treatment Modalities", 
                    passed,
                    f"Saved: {saved.get('treatment_modalities')}"
                )
            else:
                self.log_result("Filter: Treatment Modalities", False, "Failed to verify")
        else:
            self.log_result("Filter: Treatment Modalities", False, f"Status: {resp.status_code}")
        
        # Test 7: Complex combination
        combo7 = {
            "study_types": ["RCT", "Phase III"],
            "cancer_types": ["breast", "lung"],
            "min_patients": 100,
            "countries": ["United States"],
            "min_publication_year": 2015,
            "require_peer_reviewed": True,
            "required_outcomes": ["overall_survival"],
            "treatment_modalities": ["radiation"],
            "sort_by": "citations",
            "sort_order": "desc",
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo7
        )
        self.log_result(
            "Filter: Complex Combination", 
            resp.status_code == 200
        )
        
        # Test 8: Race/ethnicity filter
        combo8 = {
            "race_ethnicities": ["White", "Black", "Asian"],
            "include_unknown_race": True,
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo8
        )
        
        if resp.status_code == 200:
            # Verify it was saved
            get_resp = await self.client.get(
                f"{self.base_url}/user-preferences",
                headers=self._headers()
            )
            if get_resp.status_code == 200:
                saved = get_resp.json()
                passed = (
                    set(saved.get("race_ethnicities", [])) == {"White", "Black", "Asian"} and
                    saved.get("include_unknown_race") == True
                )
                self.log_result(
                    "Filter: Race/Ethnicity", 
                    passed,
                    f"Saved: {saved.get('race_ethnicities')}, include_unknown={saved.get('include_unknown_race')}"
                )
            else:
                self.log_result("Filter: Race/Ethnicity", False, "Failed to verify")
        else:
            self.log_result("Filter: Race/Ethnicity", False, f"Status: {resp.status_code}")
        
        # Test 9: Race filter with include_unknown_race = False
        combo9 = {
            "race_ethnicities": ["Hispanic/Latino"],
            "include_unknown_race": False,
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=combo9
        )
        
        if resp.status_code == 200:
            get_resp = await self.client.get(
                f"{self.base_url}/user-preferences",
                headers=self._headers()
            )
            if get_resp.status_code == 200:
                saved = get_resp.json()
                passed = saved.get("include_unknown_race") == False
                self.log_result(
                    "Filter: Race with exclude unknown", 
                    passed,
                    f"include_unknown_race={saved.get('include_unknown_race')}"
                )
            else:
                self.log_result("Filter: Race with exclude unknown", False, "Failed to verify")
        else:
            self.log_result("Filter: Race with exclude unknown", False, f"Status: {resp.status_code}")
    
    # ==========================================
    # Test Filters Active Toggle
    # ==========================================
    
    async def test_filters_active_toggle(self):
        """Test the filters_active toggle"""
        print("\n--- Testing Filters Active Toggle ---")
        
        # Set filters with active=True
        prefs_active = {
            "study_types": ["RCT"],
            "filters_active": True
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=prefs_active
        )
        
        if resp.status_code == 200:
            get_resp = await self.client.get(
                f"{self.base_url}/user-preferences",
                headers=self._headers()
            )
            if get_resp.status_code == 200:
                data = get_resp.json()
                self.log_result(
                    "Filters Active = True", 
                    data.get("filters_active") == True
                )
        
        # Set filters with active=False
        prefs_inactive = {
            "study_types": ["RCT"],
            "filters_active": False
        }
        
        resp = await self.client.post(
            f"{self.base_url}/user-preferences",
            headers=self._headers(),
            json=prefs_inactive
        )
        
        if resp.status_code == 200:
            get_resp = await self.client.get(
                f"{self.base_url}/user-preferences",
                headers=self._headers()
            )
            if get_resp.status_code == 200:
                data = get_resp.json()
                self.log_result(
                    "Filters Active = False", 
                    data.get("filters_active") == False
                )
    
    async def test_intent_analysis(self):
        """Test the intent analysis endpoint"""
        print("\n--- Testing Intent Analysis ---")
        
        # Test 1: Patient description without question
        patient_desc = "68 year old female, non-smoker, with SCC of R maxilla s/p maxillectomy, pT4N0, negative margins, no LVI/PNI"
        
        resp = await self.client.post(
            f"{self.base_url}/rag/analyze-intent",
            headers=self._headers(),
            json={"query": patient_desc}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            intent = data.get("intent", {})
            profile = data.get("patient_profile", {})
            options = data.get("follow_up_options", [])
            
            # Should detect as patient_description without explicit question
            is_patient_desc = intent.get("intent_type") == "patient_description"
            has_no_question = not intent.get("has_explicit_question", True)
            should_prompt = data.get("should_prompt_user", False)
            
            self.log_result(
                "Intent: Patient Description", 
                is_patient_desc and has_no_question,
                f"Type: {intent.get('intent_type')}, Question: {intent.get('has_explicit_question')}"
            )
            
            # Should extract patient profile
            has_profile = profile is not None and profile.get("cancer_type") is not None
            self.log_result(
                "Patient Profile Extracted", 
                has_profile,
                f"Cancer: {profile.get('cancer_type') if profile else 'None'}, Age: {profile.get('age') if profile else 'None'}"
            )
            
            # Should offer follow-up options
            has_options = len(options) > 0
            self.log_result(
                "Follow-up Options Generated", 
                has_options,
                f"Options: {len(options)}"
            )
            
            if options:
                print(f"       Sample options: {[o.get('label') for o in options[:3]]}")
            
            # Should prompt user
            self.log_result(
                "Should Prompt User", 
                should_prompt,
                f"should_prompt_user={should_prompt}"
            )
        else:
            self.log_result("Intent: Patient Description", False, f"Status: {resp.status_code}")
        
        # Test 2: Explicit question
        explicit_question = "What is the 5-year survival rate for stage III breast cancer?"
        
        resp = await self.client.post(
            f"{self.base_url}/rag/analyze-intent",
            headers=self._headers(),
            json={"query": explicit_question}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            intent = data.get("intent", {})
            
            has_question = intent.get("has_explicit_question", False)
            question_type = intent.get("detected_question_type")
            
            self.log_result(
                "Intent: Explicit Question", 
                has_question,
                f"Question type: {question_type}"
            )
        else:
            self.log_result("Intent: Explicit Question", False, f"Status: {resp.status_code}")
        
        # Test 3: Treatment inquiry
        treatment_query = "What are the treatment options for locally advanced NSCLC?"
        
        resp = await self.client.post(
            f"{self.base_url}/rag/analyze-intent",
            headers=self._headers(),
            json={"query": treatment_query}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            intent = data.get("intent", {})
            
            self.log_result(
                "Intent: Treatment Inquiry", 
                intent.get("has_explicit_question", False),
                f"Type: {intent.get('intent_type')}, Question: {intent.get('detected_question_type')}"
            )
        else:
            self.log_result("Intent: Treatment Inquiry", False, f"Status: {resp.status_code}")
    
    # ==========================================
    # Run All Tests
    # ==========================================
    
    async def run_all_tests(self):
        """Run all preference and filter tests"""
        print("=" * 60)
        print("PREFERENCES AND FILTERS TEST SUITE")
        print("=" * 60)
        
        # Setup
        auth_ok = await self.setup_auth()
        if not auth_ok:
            print("\n⚠️  Authentication failed - some tests may fail")
        
        # Run tests
        await self.test_get_countries()
        await self.test_get_institutions()
        await self.test_get_race_ethnicities()
        await self.test_save_preferences()
        await self.test_get_preferences()
        await self.test_sort_options()
        await self.test_filter_combinations()
        await self.test_filters_active_toggle()
        await self.test_citation_count()
        await self.test_studies_by_citations()
        await self.test_patient_relevance_sort()
        await self.test_intent_analysis()
        await self.test_delete_preferences()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])
        total = len(self.results)
        
        print(f"\nTotal: {total} tests")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['test']}: {r['details']}")
        
        return passed, failed


async def main():
    tester = PreferencesFilterTester()
    try:
        passed, failed = await tester.run_all_tests()
        return 0 if failed == 0 else 1
    finally:
        await tester.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
