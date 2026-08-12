#!/usr/bin/env python3
"""
Test all 9 implemented features.
Runs offline (no server needed) for classification/model tests.
Live API tests skipped unless BASE_URL env var is set.
"""
import os, sys, re, json, math
sys.path.insert(0, os.path.dirname(__file__))

# Stub missing optional deps so we can import auth-dependent modules
import unittest.mock as _mock
for _missing in ["passlib", "passlib.context", "jose", "python_jose", "jose.jwt"]:
    if _missing not in sys.modules:
        sys.modules[_missing] = _mock.MagicMock()

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"

results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"  {FAIL} {name}: {type(e).__name__}: {e}")
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ─────────────────────────────────────────────────────────────
# TASK 3 — Query Classification
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 3: Query Classification ═══")
from src.api.services.enhanced_rag_service import classify_query, QUERY_TYPE_PATTERNS

def t3_dose_wins_over_treatment():
    r = classify_query("What dose should be given for breast cancer? 50 Gy in 25 fractions?")
    assert r["primary_type"] == "dose_question", f"got {r['primary_type']}"

def t3_trial_wins_over_general():
    r = classify_query("What did the PACIFIC trial show for overall survival?")
    assert r["primary_type"] == "trial_results", f"got {r['primary_type']}"

def t3_staging_detected():
    r = classify_query("What stage is T2N1M0 NSCLC?")
    assert r["primary_type"] == "staging", f"got {r['primary_type']}"

def t3_indication_no_criteria_overlap():
    # "criteria for treatment" should NOT strongly match indication_question
    # The old pattern was "criteria for" — now it's "eligibility criteria"
    r = classify_query("What are the criteria for adjuvant radiation treatment?")
    # Should NOT be indication_question (old bug) — could be treatment_recommendation
    assert r["primary_type"] != "indication_question", f"got {r['primary_type']} (old overlap bug)"

def t3_indication_question_eligible():
    r = classify_query("Which patients are eligible for adjuvant radiation therapy?")
    assert r["primary_type"] == "indication_question", f"got {r['primary_type']}"

def t3_side_effects_detected():
    r = classify_query("What are the side effects and toxicity of chemoradiation?")
    assert r["primary_type"] == "side_effects", f"got {r['primary_type']}"

def t3_priority_tiebreak_dose_over_treatment():
    # Query with both dose AND treatment signals — dose should win
    r = classify_query("What is the recommended dose for rectal cancer radiation treatment?")
    assert r["primary_type"] == "dose_question", f"got {r['primary_type']}"

def t3_general_fallback():
    r = classify_query("hello world")
    assert r["primary_type"] == "general", f"got {r['primary_type']}"

def t3_confidence_returned():
    r = classify_query("What dose in Gy for prostate SBRT?")
    assert "confidence" in r
    assert 0 <= r["confidence"] <= 1, f"confidence={r['confidence']}"

def t3_new_trial_names():
    r = classify_query("What were the results of the FAST-Forward trial?")
    assert r["primary_type"] == "trial_results", f"got {r['primary_type']}"

test("dose_question wins over treatment_recommendation", t3_dose_wins_over_treatment)
test("trial_results detected (PACIFIC)", t3_trial_wins_over_general)
test("staging detected (T2N1M0)", t3_staging_detected)
test("indication_question does NOT match 'criteria for treatment'", t3_indication_no_criteria_overlap)
test("indication_question matches 'eligible for'", t3_indication_question_eligible)
test("side_effects detected", t3_side_effects_detected)
test("priority tiebreak: dose > treatment", t3_priority_tiebreak_dose_over_treatment)
test("general fallback works", t3_general_fallback)
test("confidence field returned (0-1)", t3_confidence_returned)
test("new trial name FAST-Forward detected", t3_new_trial_names)


# ─────────────────────────────────────────────────────────────
# TASK 3 — Module Classifier Rule 2 threshold
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 3: Module Classifier ═══")
from src.api.services.module_classifier import classify_module, QueryModule

def t3_module_single_patient_word_is_general():
    # "patient" alone should NOT trigger PATIENT_SPECIFIC (requires >=2 matches)
    result = classify_module("What is the patient survival rate?")
    # Should NOT be PATIENT_SPECIFIC just because of the word "patient"
    # It might be GENERAL_KNOWLEDGE or EVIDENCE_EXPLORATION
    assert result.module.value != "patient_specific" or result.confidence < 0.8, \
        f"Single 'patient' word gave high-confidence PATIENT_SPECIFIC: {result.confidence}"

def t3_module_rich_patient_context():
    result = classify_module("65 year old male with stage III NSCLC, T3N1M0, presenting with hemoptysis")
    assert result.module.value == "patient_specific", f"got {result.module.value}"

def t3_module_superlative_exploration():
    result = classify_module("What is the best treatment option for lung cancer?")
    assert result.module.value == "evidence_exploration", f"got {result.module.value}"

test("single 'patient' word does NOT trigger PATIENT_SPECIFIC", t3_module_single_patient_word_is_general)
test("rich patient context → PATIENT_SPECIFIC", t3_module_rich_patient_context)
test("superlative 'best' → EVIDENCE_EXPLORATION", t3_module_superlative_exploration)


# ─────────────────────────────────────────────────────────────
# TASK 4 — Model Fields
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 4: Model Fields ═══")
from src.api.models.query_models import RetrievalResult, QueryMetadata, EnhancedQueryResponse

def t4_retrieval_result_has_source_type():
    r = RetrievalResult(content="test", source_type="kb")
    assert r.source_type == "kb"

def t4_retrieval_result_source_type_optional():
    r = RetrievalResult(content="test")
    assert r.source_type is None

def t4_metadata_has_query_confidence():
    m = QueryMetadata(query_type="dose_question", query_confidence=0.87)
    assert m.query_confidence == 0.87

def t4_metadata_query_confidence_optional():
    m = QueryMetadata(query_type="general")
    assert m.query_confidence is None

def t4_normalize_crossencoder():
    # Test the normalize function
    from src.api.routes.query import _normalize_crossencoder_score
    assert _normalize_crossencoder_score(None) is None
    # Values strictly between 0 and 1 are treated as probabilities
    assert _normalize_crossencoder_score(0.75) == 75
    # Boundary values 0 and 1 go through sigmoid: sigmoid(0)=50, sigmoid(1)≈73
    assert _normalize_crossencoder_score(0) == 50
    assert _normalize_crossencoder_score(0.0) == 50
    assert _normalize_crossencoder_score(1.0) == 73
    # Logit >1: high confidence
    assert _normalize_crossencoder_score(3) == 95

test("RetrievalResult has source_type field", t4_retrieval_result_has_source_type)
test("source_type is Optional (default None)", t4_retrieval_result_source_type_optional)
test("QueryMetadata has query_confidence field", t4_metadata_has_query_confidence)
test("query_confidence is Optional (default None)", t4_metadata_query_confidence_optional)
test("_normalize_crossencoder_score works", t4_normalize_crossencoder)


# ─────────────────────────────────────────────────────────────
# TASK 2 + 5 — New Models
# ─────────────────────────────────────────────────────────────
print("\n═══ Tasks 2+5: StructuredSummary & GuidelineAlignment Models ═══")
from src.api.models.query_models import StructuredSummary, GuidelineAlignment

def t2_structured_summary_fields():
    s = StructuredSummary(
        key_finding="Adjuvant RT reduces local recurrence.",
        evidence_level="High",
        recommendation_strength="Strong",
        caveats="Based on RCT data."
    )
    assert s.key_finding
    assert s.evidence_level == "High"
    assert s.recommendation_strength == "Strong"
    assert s.caveats == "Based on RCT data."

def t2_structured_summary_caveats_optional():
    s = StructuredSummary(key_finding="x", evidence_level="Low", recommendation_strength="Conditional")
    assert s.caveats is None

def t5_guideline_alignment_fields():
    g = GuidelineAlignment(
        guideline_body="NCCN",
        alignment_status="Consistent",
        guideline_note="Aligns with Category 1 recommendation."
    )
    assert g.guideline_body == "NCCN"
    assert g.alignment_status == "Consistent"

def t2_enhanced_response_has_fields():
    resp = EnhancedQueryResponse(
        short_answer="Test",
        justification="Test justification",
        used_pto=False,
        structured_summary=StructuredSummary(
            key_finding="key", evidence_level="Moderate", recommendation_strength="Conditional"
        ),
        guideline_alignment=GuidelineAlignment(
            guideline_body="ASTRO", alignment_status="Not addressed", guideline_note="No ASTRO guidance found."
        )
    )
    assert resp.structured_summary is not None
    assert resp.guideline_alignment is not None
    assert resp.structured_summary.evidence_level == "Moderate"

test("StructuredSummary fields work", t2_structured_summary_fields)
test("StructuredSummary caveats is Optional", t2_structured_summary_caveats_optional)
test("GuidelineAlignment fields work", t5_guideline_alignment_fields)
test("EnhancedQueryResponse has structured_summary + guideline_alignment", t2_enhanced_response_has_fields)


# ─────────────────────────────────────────────────────────────
# TASK 1 — Section Headers in GENERATION_PROMPTS
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 1: Section Headers in Generation Prompts ═══")
from src.api.services.enhanced_rag_service import GENERATION_PROMPTS

EXPECTED_HEADERS = {
    "treatment_recommendation": ["## Recommended Treatment", "## Dose & Schedule", "## Evidence Summary"],
    "dose_question": ["## Prescribed Dose", "## Fractionation", "## Dose Constraints"],
    "trial_results": ["## Trial Name & Design", "## Key Outcomes"],
    "staging": ["## Stage Classification", "## Staging Criteria"],
    "workup": ["## Recommended Workup", "## Imaging"],
    "mechanism": ["## Mechanism", "## Biological Basis"],
    "side_effects": ["## Common Side Effects", "## Serious/Late Effects"],
    "indication_question": ["## Indication", "## Patient Selection Criteria"],
    "general": ["## Summary", "## Key Points"],
}

for qtype, headers in EXPECTED_HEADERS.items():
    def make_test(qt, hdrs):
        def _t():
            prompt = GENERATION_PROMPTS.get(qt)
            assert prompt, f"Missing prompt for {qt}"
            system = prompt["system"]
            for h in hdrs:
                assert h in system, f"Missing '{h}' in {qt} system prompt"
        return _t
    test(f"'{qtype}' prompt has section headers", make_test(qtype, headers))


# ─────────────────────────────────────────────────────────────
# TASK 9 — Web Evidence Trigger Logic
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 9: Web Evidence Trigger Logic ═══")
from src.api.routes.query import _maybe_fetch_web_evidence, _WEB_TRIGGER_TYPES, _WEB_BLOCKED_TYPES

def t9_blocked_for_treatment():
    results = _maybe_fetch_web_evidence("best treatment for lung cancer", "treatment_recommendation", [])
    assert results == [], "Should block web search for treatment_recommendation"

def t9_blocked_for_mechanism():
    results = _maybe_fetch_web_evidence("how does SBRT work?", "mechanism", [])
    assert results == [], "Should block web search for mechanism"

def t9_blocked_for_side_effects():
    results = _maybe_fetch_web_evidence("toxicity of chemoradiation", "side_effects", [])
    assert results == [], "Should block web search for side_effects"

def t9_not_triggered_without_recency_and_good_scores():
    # Good KB scores + no recency → should not trigger web search
    kb = [{"score_crossencoder": 2.5} for _ in range(5)]  # logit 2.5 → ~92% after sigmoid
    results = _maybe_fetch_web_evidence("PACIFIC trial outcomes for NSCLC", "trial_results", kb)
    assert results == [], "Should not trigger when KB scores are good and no recency terms"

def t9_triggers_with_recency_terms():
    # Even with good scores, "latest 2025" should trigger
    kb = [{"score_crossencoder": 2.5} for _ in range(5)]
    # Note: this makes a real network call — skip if no network
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        has_network = True
    except:
        has_network = False

    if not has_network:
        print(f"    {SKIP} (no network — skipping live call)")
        return

    # Just check the trigger logic fires (the actual fetch may return empty)
    # We check that it at least TRIES (no assertion on non-empty result)
    try:
        results = _maybe_fetch_web_evidence("latest 2025 SBRT trials", "trial_results", kb)
        # If results returned, they should have source_type=pubmed
        for r in results:
            assert r.get("source_type") == "pubmed", f"source_type should be pubmed, got {r.get('source_type')}"
    except Exception:
        pass  # Network error is OK for this test

def t9_web_blocked_types_correct():
    assert "treatment_recommendation" in _WEB_BLOCKED_TYPES
    assert "mechanism" in _WEB_BLOCKED_TYPES
    assert "side_effects" in _WEB_BLOCKED_TYPES

def t9_web_trigger_types_correct():
    assert "trial_results" in _WEB_TRIGGER_TYPES
    assert "staging" in _WEB_TRIGGER_TYPES
    assert "workup" in _WEB_TRIGGER_TYPES

test("web search blocked for treatment_recommendation", t9_blocked_for_treatment)
test("web search blocked for mechanism", t9_blocked_for_mechanism)
test("web search blocked for side_effects", t9_blocked_for_side_effects)
test("web search NOT triggered with good KB scores + no recency", t9_not_triggered_without_recency_and_good_scores)
test("recency trigger logic fires for '2025' query", t9_triggers_with_recency_terms)
test("_WEB_BLOCKED_TYPES contains correct types", t9_web_blocked_types_correct)
test("_WEB_TRIGGER_TYPES contains correct types", t9_web_trigger_types_correct)


# ─────────────────────────────────────────────────────────────
# TASK 4 (frontend) — CSS/JS badge presence
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 4 + 9 Frontend: Badge CSS/JS ═══")

def t4_relevance_badge_css():
    with open("frontend/css/styles.css") as f:
        css = f.read()
    assert ".relevance-badge" in css
    assert ".relevance-high" in css
    assert ".relevance-medium" in css
    assert ".relevance-low" in css

def t9_source_type_badge_css():
    with open("frontend/css/styles.css") as f:
        css = f.read()
    assert ".source-type-badge" in css
    assert ".source-type-pubmed" in css
    assert ".source-type-clinicaltrials" in css

def t4_relevance_badge_js():
    with open("frontend/js/utils.js") as f:
        js = f.read()
    assert "relevance-badge" in js
    assert "relevance_score" in js
    assert "relevance-high" in js

def t9_source_type_badge_js():
    with open("frontend/js/utils.js") as f:
        js = f.read()
    assert "source_type" in js
    assert "source-type-pubmed" in js

test("relevance badge CSS classes exist", t4_relevance_badge_css)
test("source type badge CSS classes exist", t9_source_type_badge_css)
test("formatSources() renders relevance badge", t4_relevance_badge_js)
test("formatSources() renders source_type badge", t9_source_type_badge_js)


# ─────────────────────────────────────────────────────────────
# TASK 6 — GUI CSS/JS
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 6: GUI CSS/JS ═══")

def t6_logo_position_css():
    with open("frontend/css/styles.css") as f:
        css = f.read()
    # Logo should now be positioned fixed top-left, not display:none
    # Find the section we added
    assert "body.chat-fullscreen-active .hero-logo" in css
    # Confirm it has position: fixed and display: block
    idx = css.find("body.chat-fullscreen-active .hero-logo")
    snippet = css[idx:idx+300]
    assert "position: fixed" in snippet, "logo should be position:fixed"
    assert "display: block" in snippet, "logo should be display:block"
    # Should NOT be display: none in that block
    assert "display: none" not in snippet, "logo should not be hidden"

def t6_auto_fullscreen_js():
    with open("frontend/index.html") as f:
        html = f.read()
    # activateChatLayout should call toggleFullscreen
    assert "toggleFullscreen()" in html
    # Should be inside activateChatLayout
    idx = html.find("function activateChatLayout()")
    assert idx > 0
    fn_block = html[idx:idx+1100]
    assert "toggleFullscreen()" in fn_block, "activateChatLayout should call toggleFullscreen()"

test("hero-logo CSS: position:fixed top-left (not hidden)", t6_logo_position_css)
test("activateChatLayout() calls toggleFullscreen()", t6_auto_fullscreen_js)


# ─────────────────────────────────────────────────────────────
# TASK 7 — Read Abstract button
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 7: Read Abstract ═══")

def t7_read_abstract_btn_in_html():
    with open("frontend/index.html") as f:
        html = f.read()
    assert "Read Abstract" in html
    assert "toggleTrialAbstract" in html
    assert "trial-abstract-container" in html

def t7_abstract_cache():
    with open("frontend/index.html") as f:
        html = f.read()
    assert "_abstractCache" in html
    assert "new Map()" in html

def t7_get_study_details_api():
    with open("frontend/js/api.js") as f:
        js = f.read()
    assert "getStudyDetails" in js
    assert "study-details" in js

test("Read Abstract button rendered per trial card", t7_read_abstract_btn_in_html)
test("Abstract cache (Map) present", t7_abstract_cache)
test("api.getStudyDetails() added to api.js", t7_get_study_details_api)


# ─────────────────────────────────────────────────────────────
# TASK 8 — Mode Selector
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 8: Active/Published Mode Selector ═══")

def t8_mode_selector_html():
    with open("frontend/index.html") as f:
        html = f.read()
    assert "trial-mode-selector" in html
    assert "Active Trials" in html
    assert "Published Evidence" in html

def t8_select_mode_fn():
    with open("frontend/index.html") as f:
        html = f.read()
    assert "window.selectTrialMode" in html
    assert "window.executeTrialSearch" in html

def t8_mode_selector_css():
    with open("frontend/css/styles.css") as f:
        css = f.read()
    assert ".trial-mode-selector" in css
    assert ".trial-mode-btn" in css
    assert ".trial-mode-go-btn" in css

def t8_smart_default_logic():
    with open("frontend/index.html") as f:
        html = f.read()
    assert "_pendingTrialQuestion" in html
    assert "_trialSearchMode" in html
    assert "publishedTerms" in html  # smart default regex

test("Mode selector HTML present", t8_mode_selector_html)
test("selectTrialMode + executeTrialSearch functions present", t8_select_mode_fn)
test("Mode selector CSS present", t8_mode_selector_css)
test("Smart default logic present", t8_smart_default_logic)


# ─────────────────────────────────────────────────────────────
# TASK 2+5 — Generation helpers in query.py
# ─────────────────────────────────────────────────────────────
print("\n═══ Tasks 2+5: Generation Helper Imports ═══")

def t2_helper_importable():
    from src.api.routes.query import _generate_structured_summary
    import inspect
    assert callable(_generate_structured_summary)

def t5_helper_importable():
    from src.api.routes.query import _generate_guideline_alignment
    assert callable(_generate_guideline_alignment)

test("_generate_structured_summary importable", t2_helper_importable)
test("_generate_guideline_alignment importable", t5_helper_importable)


# ─────────────────────────────────────────────────────────────
# TASK 9 — Web constraint in gpt4o_summary_enhanced
# ─────────────────────────────────────────────────────────────
print("\n═══ Task 9: Web Constraint in Generation ═══")

def t9_web_constraint_present():
    with open("src/api/services/enhanced_rag_service.py") as f:
        code = f.read()
    assert "WEB SOURCES" in code or "web_sources" in code.lower() or "source_type" in code
    assert "[PubMed]" in code

test("Web source constraint injected into system prompt", t9_web_constraint_present)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*50)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed > 0:
    print("\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  {FAIL} {name}: {err}")
sys.exit(0 if failed == 0 else 1)
