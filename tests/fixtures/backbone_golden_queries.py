"""
Golden fixture data for the three canonical backbone-test queries.

These fixtures serve two purposes:
1. BEFORE_FIX_BASELINES: snapshot of the current (buggy) outputs from
   structure_query_fast(), classify_query(), and category normalization.
2. GOLDEN_QUERIES: the expected correct outputs after all fixes are applied.

After all 9 fixes, re-run and assert the outputs have changed in the
expected direction (correct biomarkers, correct site, correct classification,
non-zero criteria scores, bundle size >= 3).
"""

# ---------------------------------------------------------------------------
# The three canonical backbone-test queries
# ---------------------------------------------------------------------------

QUERY_HN_ICI_REFRACTORY = (
    "A patient with recurrent head and neck squamous cell carcinoma, "
    "PD-L1 CPS 100, previously treated with platinum-based chemotherapy "
    "and immunotherapy. What is the best next-line systemic therapy?"
)

QUERY_LUNG_EGFR = (
    "A patient with EGFR-mutant (exon 19 del) lung adenocarcinoma of the "
    "right lower lobe, liver metastases, leptomeningeal spread. Previously "
    "treated with osimertinib. What is the best next-line systemic therapy?"
)

QUERY_PROSTATE_BRCA2 = (
    "A patient with metastatic castration-resistant prostate cancer, "
    "BRCA2-mutated, previously treated with enzalutamide and docetaxel. "
    "What is the best next-line systemic therapy?"
)

# ---------------------------------------------------------------------------
# BEFORE_FIX_BASELINES — current (buggy) outputs captured on unfixed code
# ---------------------------------------------------------------------------
# These document the exact outputs of structure_query_fast() and
# classify_query() BEFORE any fixes are applied.  They serve as regression
# anchors: after all fixes, the outputs MUST differ from these baselines
# in the directions described in GOLDEN_QUERIES below.
# ---------------------------------------------------------------------------

BEFORE_FIX_BASELINES = {
    "hn_ici_refractory": {
        "query": QUERY_HN_ICI_REFRACTORY,
        "structure_query_fast": {
            "site": "head_neck",
            "filter_category": "head_neck",
            "biomarkers": ["PD-L1", "CPS 100", "CPS positive"],
            "histology": "scc",
            "stage": None,
            "query_type": "general",
            "question_focus": None,
        },
        "classify_query": {
            "primary_type": "staging",
            "confidence": 0.1,
            "scores": {
                "treatment_recommendation": 1,
                "indication_question": 0,
                "dose_question": 0,
                "trial_results": 0,
                "staging": 1,
                "workup": 0,
                "mechanism": 0,
                "side_effects": 0,
            },
        },
        "category_normalization": {
            # head_neck vs h&n_processed_documents => mismatch (Bug 9)
            "filter_category": "head_neck",
            "doc_category_example": "h&n_processed_documents",
            "exact_match": False,  # head_neck != h&n_processed_documents
        },
    },
    "lung_egfr": {
        "query": QUERY_LUNG_EGFR,
        "structure_query_fast": {
            # Site extraction: lung matched first due to dict insertion order
            # in CANCER_SITE_PATTERNS (lung before gi_hepatobiliary).
            # The bug (Bug 3) is that the first-match-wins break can pick
            # a metastatic site over the primary site when dict order differs
            # or when the metastatic pattern matches first.
            "site": "lung",
            "filter_category": "lung",
            # BUG 1: EGFR polarity flip — "EGFR-mutant" matches wild-type
            # pattern's bare `-` alternative before the mutant pattern fires
            "biomarkers": ["EGFR wild-type"],
            "histology": "adenocarcinoma",
            "stage": None,
            "query_type": "general",
            "question_focus": None,
        },
        "classify_query": {
            # BUG 8: classifier tie-breaking picks staging over
            # treatment_recommendation because PRIORITY list ranks staging
            # at position 3 vs treatment_recommendation at position 7
            "primary_type": "staging",
            "confidence": 0.1,
            "scores": {
                "treatment_recommendation": 1,
                "indication_question": 0,
                "dose_question": 0,
                "trial_results": 0,
                "staging": 1,
                "workup": 0,
                "mechanism": 0,
                "side_effects": 0,
            },
        },
        "category_normalization": {
            "filter_category": "lung",
            "doc_category_example": "lung_processed_documents",
            "exact_match": False,  # lung != lung_processed_documents
        },
    },
    "prostate_brca2": {
        "query": QUERY_PROSTATE_BRCA2,
        "structure_query_fast": {
            "site": "prostate",
            "filter_category": "prostate",
            # BUG 2: BRCA polarity flip — "BRCA2-mutated" matches wild-type
            # pattern's bare `-` alternative
            "biomarkers": ["BRCA wild-type"],
            "histology": None,
            "stage": "IV",
            "query_type": "general",
            "question_focus": None,
        },
        "classify_query": {
            # BUG 8: same tie-breaking issue
            "primary_type": "staging",
            "confidence": 0.1,
            "scores": {
                "treatment_recommendation": 1,
                "indication_question": 0,
                "dose_question": 0,
                "trial_results": 0,
                "staging": 1,
                "workup": 0,
                "mechanism": 0,
                "side_effects": 0,
            },
        },
        "category_normalization": {
            "filter_category": "prostate",
            "doc_category_example": "prostate_processed_documents",
            "exact_match": False,  # prostate != prostate_processed_documents
        },
    },
}

# ---------------------------------------------------------------------------
# GOLDEN_QUERIES — expected CORRECT outputs after all fixes
# ---------------------------------------------------------------------------
# Each entry describes the expected post-fix behavior for one canonical query.
# After all 9 fixes are applied, re-run structure_query_fast() and
# classify_query() and assert the outputs match these expectations.
# ---------------------------------------------------------------------------

GOLDEN_QUERIES = [
    {
        "name": "hn_ici_refractory",
        "query": QUERY_HN_ICI_REFRACTORY,
        "expected_site": "head_neck",
        "expected_biomarkers": ["PD-L1", "CPS 100", "CPS positive"],
        "expected_query_type": "treatment_recommendation",
        "expected_category": "head_neck",
        # Category normalization: head_neck should match h&n_processed_documents
        "expected_category_aliases": ["head_neck", "h&n_processed_documents"],
    },
    {
        "name": "lung_egfr",
        "query": QUERY_LUNG_EGFR,
        "expected_site": "lung",
        # Fix 1: EGFR-mutant must extract positive polarity
        "expected_biomarkers": ["EGFR mutant"],
        "expected_query_type": "treatment_recommendation",
        "expected_category": "lung",
        "expected_category_aliases": ["lung", "lung_processed_documents"],
    },
    {
        "name": "prostate_brca2",
        "query": QUERY_PROSTATE_BRCA2,
        "expected_site": "prostate",
        # Fix 2: BRCA2-mutated must extract positive polarity
        "expected_biomarkers": ["BRCA mutant"],
        "expected_query_type": "treatment_recommendation",
        "expected_category": "prostate",
        "expected_category_aliases": ["prostate", "prostate_processed_documents"],
    },
]

# ---------------------------------------------------------------------------
# Known category alias pairs for normalization testing (Bug 5 / Bug 9)
# ---------------------------------------------------------------------------

CATEGORY_ALIAS_PAIRS = [
    ("head_neck", "h&n_processed_documents"),
    ("lung", "lung_processed_documents"),
    ("prostate", "prostate_processed_documents"),
    ("breast", "breast_processed_documents"),
    ("gi", "gi_processed_documents"),
    ("gyn", "gyn_processed_documents"),
    ("gu", "gu_processed_documents"),
    ("cns", "cns_processed_documents"),
    ("lymphoma", "lymphoma_processed_documents"),
    ("sarcoma", "sarcoma_processed_documents"),
    ("skin", "skin_processed_documents"),
    ("thyroid", "thyroid_processed_documents"),
    ("pediatric", "pediatric_processed_documents"),
]
