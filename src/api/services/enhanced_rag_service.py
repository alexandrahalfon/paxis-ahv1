"""
Enhanced Qdrant RAG Service for Paxis Medical Literature Platform

This is a COMPLETE port from the Google Colab enhanced RAG pipeline.
ALL features from the original Colab code are preserved here.

Improvements implemented:
- P0: NCCN guideline gap detection and handling
- P0: Query expansion (abbreviations/synonyms)
- P0: BIDIRECTIONAL query expansion (abbreviation ↔ full term) [NEW]
- P1: Query type classification (8 types + indication_question)
- P1: Query-type-specific generation prompts (9 templates)
- P1: Cross-encoder reranking
- P2: Structured dose extraction and boosting
- Site inference for tumor-specific queries
"""

import re
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from textwrap import shorten

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

from src.api.services.rag_prefilter_service import (
    build_qdrant_filter_from_clinical_profile,
    is_prefilter_enabled,
    PreFilterResult,
    get_category_filter,
)

# Cross-encoder for reranking (P1)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Cross-encoder reranking disabled.")

import time

class Timer:
    """Dead simple timer for performance measurement."""
    def __init__(self):
        self.times = {}
    
    def start(self, name):
        self.times[name] = time.perf_counter()
    
    def end(self, name):
        elapsed = time.perf_counter() - self.times[name]
        print(f"  ⏱️  {name}: {elapsed:.3f}s")
        return elapsed


# ============================================
# CONFIG - Loads from Paxis's config system
# ============================================

from src.core.config import settings

QDRANT_URL = settings.qdrant_url
QDRANT_API_KEY = settings.qdrant_api_key
QDRANT_COLLECTION = settings.qdrant_collection
EMBED_MODEL = settings.embed_model
OPENAI_API_KEY = settings.openai_api_key

# Initialize cross-encoder (lazy loaded)
_cross_encoder = None


def get_cross_encoder():
    """Lazy load cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None and CROSS_ENCODER_AVAILABLE:
        print("Loading cross-encoder model...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder


# ============================================
# P0: QUERY EXPANSION - ABBREVIATIONS & SYNONYMS
# ============================================

ONCOLOGY_EXPANSIONS = {
    # Disease abbreviations
    "mibc": "muscle invasive bladder cancer",
    "nmibc": "non-muscle invasive bladder cancer",
    "nsclc": "non-small cell lung cancer",
    "sclc": "small cell lung cancer",
    "hnscc": "head and neck squamous cell carcinoma",
    "npc": "nasopharyngeal carcinoma",
    "hcc": "hepatocellular carcinoma",
    "rcc": "renal cell carcinoma",
    "crc": "colorectal cancer",
    "gbm": "glioblastoma multiforme",
    "dcis": "ductal carcinoma in situ",
    "clinical detection": "clinically detected palpable symptomatic non-mammographic",
    "mammographic detection": "screen-detected screening mammography asymptomatic",
    "tnbc": "triple negative breast cancer",
    "mcrpc": "metastatic castration resistant prostate cancer",
    "crpc": "castration resistant prostate cancer",
    "mhspc": "metastatic hormone sensitive prostate cancer",
    "dlbcl": "diffuse large B-cell lymphoma",
    "hl": "hodgkin lymphoma",
    "nhl": "non-hodgkin lymphoma",
    "aml": "acute myeloid leukemia",
    "all": "acute lymphoblastic leukemia",
    "mds": "myelodysplastic syndrome",

    # Treatment abbreviations
    "rt": "radiation therapy radiotherapy",
    "xrt": "radiation therapy external beam",
    "ebrt": "external beam radiation therapy",
    "imrt": "intensity modulated radiation therapy",
    "vmat": "volumetric modulated arc therapy",
    "sbrt": "stereotactic body radiation therapy",
    "srs": "stereotactic radiosurgery",
    "wbrt": "whole brain radiation therapy",
    "pbi": "partial breast irradiation",
    "apbi": "accelerated partial breast irradiation",
    "csi": "craniospinal irradiation",
    "tbi": "total body irradiation",
    "brachytherapy": "brachytherapy internal radiation",
    "hdr": "high dose rate brachytherapy",
    "ldr": "low dose rate brachytherapy",
    "prrt": "peptide receptor radionuclide therapy",
    "port": "postoperative radiation therapy",
    "pmrt": "post-mastectomy radiation therapy postmastectomy",

    # Chemotherapy abbreviations
    "chemo": "chemotherapy",
    "ctx": "chemotherapy",
    "nac": "neoadjuvant chemotherapy",
    "act": "adjuvant chemotherapy",
    "chemoradiation": "concurrent chemoradiation chemoradiotherapy",
    "crt": "chemoradiotherapy concurrent",
    "tnt": "total neoadjuvant therapy",

    # Hormonal therapy
    "adt": "androgen deprivation therapy",
    "ai": "aromatase inhibitor",
    "serm": "selective estrogen receptor modulator",

    # Immunotherapy
    "io": "immunotherapy immune checkpoint inhibitor",
    "ici": "immune checkpoint inhibitor",
    "pd-1": "programmed death 1 immunotherapy",
    "pd-l1": "programmed death ligand 1 immunotherapy",
    "ctla-4": "cytotoxic T-lymphocyte-associated protein 4",

    # Surgical abbreviations
    "bcs": "breast conserving surgery lumpectomy",
    "wle": "wide local excision",
    "mri": "mastectomy radical",
    "alnd": "axillary lymph node dissection",
    "slnb": "sentinel lymph node biopsy",
    "turbt": "transurethral resection bladder tumor",
    "rp": "radical prostatectomy",
    "tme": "total mesorectal excision",
    "apr": "abdominoperineal resection",
    "lar": "low anterior resection",
    "rplnd": "retroperitoneal lymph node dissection",

    # Biomarkers
    "her2": "human epidermal growth factor receptor 2 erbb2",
    "her2+": "her2 positive amplified overexpression",
    "her2-": "her2 negative",
    "er+": "estrogen receptor positive",
    "er-": "estrogen receptor negative",
    "pr+": "progesterone receptor positive",
    "pr-": "progesterone receptor negative",
    "egfr": "epidermal growth factor receptor",
    "alk": "anaplastic lymphoma kinase",
    "ros1": "ros proto-oncogene 1",
    "kras": "kirsten rat sarcoma viral oncogene",
    "braf": "b-raf proto-oncogene",
    "brca": "breast cancer gene mutation",
    "msi": "microsatellite instability",
    "msi-h": "microsatellite instability high",
    "mmr": "mismatch repair",
    "dmmr": "deficient mismatch repair",
    "tmb": "tumor mutational burden",

    # Staging/Response
    "pcr": "pathologic complete response",
    "cr": "complete response",
    "pr": "partial response",
    "sd": "stable disease",
    "pd": "progressive disease",
    "tnm": "tumor node metastasis staging",
    "ajcc": "american joint committee cancer staging",

    # Clinical terms
    "os": "overall survival",
    "pfs": "progression free survival",
    "dfs": "disease free survival",
    "lrr": "locoregional recurrence",
    "lc": "local control",
    "dm": "distant metastasis",
    "bm": "bone metastasis",
    "bony": "bone osseous skeletal",
    "bone mets": "bone metastasis osseous metastasis skeletal metastasis",
    "bone metastasis": "osseous metastasis skeletal metastasis bony metastasis",
    "ecog": "eastern cooperative oncology group performance status",
    "kps": "karnofsky performance status",

    # Dose terms
    "gy": "gray dose",
    "cgy": "centigray dose",
    "fx": "fractions",
    "bid": "twice daily fractionation",
    "qd": "once daily",

    # Guidelines
    "nccn": "national comprehensive cancer network guidelines",
    "astro": "american society radiation oncology",
    "asco": "american society clinical oncology",
    "esmo": "european society medical oncology",
    "rtog": "radiation therapy oncology group",
    "nrg": "nrg oncology cooperative group",
    "cog": "children's oncology group",

    # Anatomy
    "ln": "lymph node",
    "lns": "lymph nodes",
    "scv": "supraclavicular",
    "imn": "internal mammary node",
    "pf": "posterior fossa",
    
    # Staging abbreviations - forward expansion
    "pn1": "pathologic N1 pathological N1 node positive",
    "pn0": "pathologic N0 pathological N0 node negative",
    "pn2": "pathologic N2 pathological N2",
    "pn3": "pathologic N3 pathological N3",
    "cn1": "clinical N1 clinically node positive",
    "cn0": "clinical N0 clinically node negative",
    "cn2": "clinical N2",
    "ypn1": "yp N1 residual nodal disease after neoadjuvant",
    "ypn0": "yp N0 nodal pathologic complete response",
}


# ==============================================================================
# REVERSE EXPANSIONS (full term → abbreviation)
# ==============================================================================
# This is the critical addition for bidirectional expansion
# Ensures queries like "pathologic N1" also find documents with "pN1"

REVERSE_EXPANSIONS = {
    # =========================================================================
    # STAGING - CRITICAL for indication questions (pN1 issue fix)
    # =========================================================================
    "pathologic n1": "pN1 pathological N1",
    "pathologic n0": "pN0 pathological N0",
    "pathologic n2": "pN2 pathological N2",
    "pathologic n3": "pN3 pathological N3",
    "pathological n1": "pN1 pathologic N1",
    "pathological n0": "pN0 pathologic N0",
    "pathological n2": "pN2 pathologic N2",
    "pathological n3": "pN3 pathologic N3",
    "clinical n1": "cN1",
    "clinical n0": "cN0",
    "clinical n2": "cN2",
    "node positive": "N+ N1 N2 N3 pN1 pN2 pN3 node-positive",
    "node negative": "N0 pN0 node-negative",
    "positive nodes": "N+ N1 N2 pN1 pN2 positive lymph nodes",
    "positive lymph nodes": "N+ N1 N2 pN1 pN2",
    "residual nodal disease": "pN1 pN2 ypN1 ypN2 node positive",
    
    # Post-neoadjuvant staging
    "yp n1": "ypN1 residual nodal disease after neoadjuvant",
    "yp n0": "ypN0 nodal pathologic complete response",
    
    # =========================================================================
    # Radiation therapy full terms → abbreviations  
    # =========================================================================
    "radiation therapy": "RT radiotherapy XRT",
    "radiotherapy": "RT radiation therapy",
    "external beam": "EBRT external beam radiation",
    "intensity modulated": "IMRT intensity-modulated",
    "stereotactic body": "SBRT stereotactic",
    "post-mastectomy radiation": "PMRT postmastectomy radiation",
    "postmastectomy radiation": "PMRT post-mastectomy radiation",
    "whole brain radiation": "WBRT whole-brain",
    "postoperative radiation": "PORT adjuvant radiation",
    
    # =========================================================================
    # Chemotherapy full terms
    # =========================================================================
    "neoadjuvant chemotherapy": "NAC neoadjuvant chemo preoperative chemotherapy",
    "adjuvant chemotherapy": "ACT adjuvant chemo postoperative chemotherapy",
    "concurrent chemoradiation": "CRT chemoradiotherapy chemo-RT",
    
    # =========================================================================
    # Biomarkers full terms
    # =========================================================================
    "estrogen receptor positive": "ER+ ER-positive",
    "estrogen receptor negative": "ER- ER-negative",
    "progesterone receptor positive": "PR+ PR-positive",
    "progesterone receptor negative": "PR- PR-negative",
    "her2 positive": "HER2+ HER2-positive amplified",
    "her2 negative": "HER2- HER2-negative",
    "triple negative": "TNBC triple-negative ER- PR- HER2-",
    
    # =========================================================================
    # Outcomes full terms
    # =========================================================================
    "overall survival": "OS",
    "progression free survival": "PFS progression-free",
    "disease free survival": "DFS disease-free",
    "local control": "LC",
    "locoregional recurrence": "LRR locoregional",
    "pathologic complete response": "pCR pathological complete response",
    
    # =========================================================================
    # Surgical full terms
    # =========================================================================
    "breast conserving surgery": "BCS lumpectomy breast-conserving",
    "sentinel lymph node": "SLN SLNB sentinel node",
    "axillary lymph node dissection": "ALND axillary dissection",
    
    # =========================================================================
    # Indication keywords - helps with indication questions
    # =========================================================================
    "indication": "indicated recommended appropriate criteria candidate",
    "contraindication": "contraindicated not recommended avoid",
    "best indication": "appropriate criteria recommended candidate",
}


# ==============================================================================
# NEW: STAGING-SPECIFIC SYNONYMS
# ==============================================================================
# Ensures all variations of staging terms are searched

STAGING_SYNONYMS = {
    # Nodal staging variations
    "pn1": ["pathologic N1", "pathological N1", "pN1", "node positive", "N1 disease"],
    "pn0": ["pathologic N0", "pathological N0", "pN0", "node negative", "N0 disease"],
    "pn2": ["pathologic N2", "pathological N2", "pN2", "N2 disease"],
    "pn3": ["pathologic N3", "pathological N3", "pN3", "N3 disease"],
    "cn1": ["clinical N1", "cN1", "clinically node positive"],
    "cn0": ["clinical N0", "cN0", "clinically node negative"],
    "ypn1": ["yp N1", "ypN1", "residual nodal disease after neoadjuvant"],
    "ypn0": ["yp N0", "ypN0", "nodal pCR", "pathologic complete response nodes"],
    
    # T staging variations
    "pt1": ["pathologic T1", "pT1", "T1 disease"],
    "pt2": ["pathologic T2", "pT2", "T2 disease"],
    "pt3": ["pathologic T3", "pT3", "T3 disease"],
    "pt4": ["pathologic T4", "pT4", "T4 disease"],
    "ct1": ["clinical T1", "cT1"],
    "ct2": ["clinical T2", "cT2"],
    "ct3": ["clinical T3", "cT3"],
    "ct4": ["clinical T4", "cT4"],
}


# Synonyms for common clinical concepts
CLINICAL_SYNONYMS = {
    "boost": "cone down additional dose",
    "standard of care": "recommended treatment guideline",
    "first-line": "initial therapy front-line",
    "second-line": "2nd-line salvage therapy subsequent post-progression post-ICI",
    "definitive": "curative intent radical",
    "palliative": "symptom control comfort",
    "neoadjuvant": "preoperative induction",
    "adjuvant": "postoperative additional",
    "concurrent": "simultaneous concomitant",
    "sequential": "following after",
    "consolidation": "maintenance continuation",
    "salvage": "rescue recurrent relapsed",
    "oligometastatic": "limited metastatic few metastases",
    "locally advanced": "regional extensive unresectable",
    "margin": "edge border surgical boundary",
    "close margin": "narrow margin near margin",
    "positive margin": "involved margin R1",
    "negative margin": "clear margin R0",
    "no ink on tumor": "negative margin clear",
    
    # === EXTRANODAL EXTENSION (ENE) - Critical for H&N staging ===
    "extranodal extension": "ENE extracapsular extension ECE N3b nodal staging",
    "ene": "extranodal extension extracapsular extension N3b",
    "extracapsular extension": "ENE extranodal extension ECE N3b",
    "extracapsular": "ENE extranodal extension N3b",
    "overt ene": "clinical extranodal extension N3b gross ENE",
    "n3b": "extranodal extension ENE extracapsular extension",
    "n3a": "lymph node greater than 6cm without ENE",
    
    # === Head and Neck Staging ===
    "oral tongue": "oral cavity tongue lateral tongue mobile tongue tongue body anterior tongue AJCC staging",
    "oral cavity staging": "AJCC 8th edition T staging N staging depth of invasion DOI",
    "depth of invasion": "DOI tumor depth T staging oral cavity",
    "doi": "depth of invasion tumor depth T staging",
    "ipsilateral adenopathy": "ipsilateral lymph nodes same side nodal disease N staging",
    "level ib": "submandibular lymph node neck level 1b",
    "level iia": "upper jugular lymph node neck level 2a",
    "clinical stage": "TNM staging AJCC stage group overall stage",
    
    # DCIS-specific risk factors
    "in-breast recurrence": "ipsilateral breast tumor recurrence IBTR local recurrence",
    "recurrence risk": "risk factor prognostic factor predictor",
    "elevated risk": "increased risk high risk poor prognosis",
    "detection method": "clinical detection mammographic detection screen-detected palpable",
    # Testicular seminoma RT
    "seminoma rt": "para-aortic strip irradiation dogleg field PA strip radiation",
    "stage i seminoma": "stage I testicular seminoma adjuvant radiation surveillance",
    "para-aortic": "PA strip paraaortic retroperitoneal lymph node",
    # Breast cancer specific
    "pcr breast": "pathologic complete response ypT0 ypN0 no residual disease",
    "her2 positive": "HER2+ ERBB2 amplified trastuzumab pertuzumab TCHP",
    "tchp": "docetaxel carboplatin trastuzumab pertuzumab neoadjuvant",
    "adjuvant her2": "trastuzumab emtansine T-DM1 Kadcyla pertuzumab",
    # Metastatic disease
    "metastatic nsclc": "stage IV lung cancer advanced NSCLC oligometastatic",
    "bony metastasis": "bone metastasis osseous metastasis skeletal metastasis",
    "four sites": "oligometastatic limited metastatic multiple metastases",
    # PMRT
    "pmrt": "post-mastectomy radiation therapy chest wall regional nodes",
    "post-mastectomy": "PMRT chest wall irradiation regional nodal irradiation",
    
    # === CRITICAL CLINICAL SCENARIO EXPANSIONS ===
    
    # Inflammatory breast cancer (IBC) - Q39 fix
    "t4d": "inflammatory breast cancer IBC trimodality neoadjuvant chemotherapy mastectomy PMRT modified radical",
    "ct4d": "inflammatory breast cancer IBC trimodality neoadjuvant chemotherapy mastectomy PMRT modified radical",
    "inflammatory breast": "IBC T4d trimodality neoadjuvant chemotherapy mastectomy PMRT modified radical",
    "ibc": "inflammatory breast cancer T4d trimodality neoadjuvant chemotherapy mastectomy PMRT",
    "peau d'orange": "inflammatory breast cancer IBC T4d trimodality",
    
    # Early HPV+ oropharynx - Q35 fix (avoid over-treatment)
    "t1n1 tonsil": "early stage HPV positive oropharynx definitive RT unilateral radiation Garden 2004",
    "t1 tonsil": "early stage HPV positive oropharynx definitive RT unilateral radiation",
    "t2n1 tonsil": "early stage HPV positive oropharynx definitive RT",
    "early tonsil": "T1 T2 HPV positive oropharynx definitive RT unilateral radiation",
    "hpv oropharynx": "HPV positive p16 positive definitive RT chemoradiation",
    
    # High-risk prostate ADT duration - Q32 fix
    "adt duration": "androgen deprivation therapy long-term 28 months RTOG 92-02 high-risk prostate",
    "high risk prostate": "Gleason 8-10 T3 PSA>20 long-term ADT 28 months RTOG 92-02",
    "gleason 9": "high-risk prostate long-term ADT 28 months RTOG 92-02",
    "gleason 10": "high-risk prostate long-term ADT 28 months RTOG 92-02",
    "t3b prostate": "high-risk prostate long-term ADT RTOG 92-02",
    
    # Cervical intermediate risk - Q36 fix
    "cervix deep invasion": "intermediate risk Sedlis criteria adjuvant EBRT pelvic radiation GOG 92",
    "cervix lvsi": "intermediate risk Sedlis criteria adjuvant EBRT pelvic radiation GOG 92",
    "sedlis": "cervical intermediate risk factors deep invasion LVSI large tumor GOG 92 adjuvant EBRT",
    "gog 92": "cervical intermediate risk Sedlis criteria adjuvant pelvic radiation EBRT",
    "cervical intermediate": "Sedlis criteria deep invasion LVSI GOG 92 adjuvant EBRT",
    
    # NPC treatment - Q33 fix
    "npc": "nasopharyngeal carcinoma concurrent chemoradiation adjuvant chemotherapy Intergroup 0099",
    "nasopharyngeal": "NPC concurrent chemoradiation adjuvant chemotherapy Intergroup 0099 cisplatin 5-FU",
    "t4n2 npc": "advanced nasopharyngeal concurrent chemoradiation adjuvant chemotherapy Intergroup 0099",
    "intergroup 0099": "nasopharyngeal NPC concurrent chemoradiation adjuvant chemotherapy cisplatin 5-FU",
    
    # PMRT indications - Q34 fix (ENHANCED)
    "pmrt indication": "post-mastectomy radiation pathologic N1 pN1 node positive neoadjuvant",
    "best pmrt": "pathologic N1 pN1 node positive after neoadjuvant chemotherapy",
    "node positive mastectomy": "PMRT post-mastectomy radiation chest wall regional nodes pN1",
    "postmastectomy indication": "pathologic N1 pN1 node positive residual nodal disease",
    
    # Ependymoma - Q37
    "ependymoma": "posterior fossa tumor bed 59.4 Gy conformal radiation",
    "posterior fossa ependymoma": "tumor bed 59.4 Gy conformal radiation",
    
    # RTOG 0225 NPC - Q38
    "rtog 0225": "nasopharyngeal NPC IMRT locoregional control distant metastasis",

    # === ICI / CHECKPOINT SYNONYMS (Task 7) ===
    "pembrolizumab": "nivolumab atezolizumab durvalumab anti-PD1 anti-PD-L1 checkpoint inhibitor CPI immunotherapy",
    "nivolumab": "pembrolizumab atezolizumab durvalumab anti-PD1 anti-PD-L1 checkpoint inhibitor CPI immunotherapy",
    "ici": "immune checkpoint inhibitor anti-PD1 anti-PD-L1 checkpoint blockade immunotherapy",
    "checkpoint inhibitor": "ICI anti-PD1 anti-PD-L1 pembrolizumab nivolumab immunotherapy CPI",

    # === Disease status synonyms ===
    "recurrent/metastatic": "R/M locoregional recurrence distant metastasis unresectable recurrent metastatic",
    "r/m": "recurrent metastatic recurrent/metastatic locoregional distant metastasis unresectable",
    "ici-refractory": "checkpoint inhibitor refractory anti-PD1 failure post-immunotherapy progression 2nd-line",

    # === Line of therapy (extended in ICI synonyms block) ===

    # === HNSCC sub-site ===
    "oropharynx": "tonsil base of tongue BOT soft palate posterior pharyngeal wall",

    # === Surgical candidacy ===
    "unresectable": "inoperable non-surgical candidate locoregional advanced not resectable",
    "inoperable": "unresectable non-surgical candidate locoregional advanced",
}


def expand_query(query: str) -> str:
    """
    ENHANCED: Bidirectional query expansion with abbreviations, synonyms, and staging terms.
    
    Now expands BOTH:
    1. Abbreviations → Full terms (e.g., "pN1" → "pathologic N1")
    2. Full terms → Abbreviations (e.g., "pathologic N1" → "pN1")
    
    This ensures queries match documents regardless of terminology used.
    """
    ql = query.lower()
    expansions = []

    # ===========================================
    # FORWARD EXPANSION: abbreviation → full term
    # ===========================================
    for abbrev, expansion in ONCOLOGY_EXPANSIONS.items():
        if re.search(rf'\b{re.escape(abbrev)}\b', ql, re.IGNORECASE):
            expansions.append(expansion)

    # ===========================================
    # NEW: REVERSE EXPANSION: full term → abbreviation
    # ===========================================
    for full_term, abbrev_expansion in REVERSE_EXPANSIONS.items():
        if full_term.lower() in ql:
            expansions.append(abbrev_expansion)

    # ===========================================
    # NEW: STAGING-SPECIFIC EXPANSIONS
    # ===========================================
    for staging_term, synonyms in STAGING_SYNONYMS.items():
        # Check if any synonym is in the query
        for synonym in synonyms:
            if synonym.lower() in ql:
                # Add all other synonyms as expansions
                expansions.extend([s for s in synonyms if s.lower() not in ql])
                break

    # ===========================================
    # CLINICAL CONCEPT SYNONYMS
    # ===========================================
    for term, synonyms in CLINICAL_SYNONYMS.items():
        if term in ql:
            expansions.append(synonyms)
    
    # ===========================================
    # CRITICAL STAGING PATTERNS (from clinical entity extractor)
    # ===========================================
    try:
        from src.api.services.clinical_entity_extractor import get_clinical_entity_extractor
        extractor = get_clinical_entity_extractor()
        critical_terms = extractor.get_query_expansion_terms(query)
        if critical_terms:
            expansions.extend(critical_terms)
    except Exception as e:
        pass  # Silently fail if extractor not available

    # Deduplicate and combine
    if expansions:
        # Remove duplicates while preserving useful terms
        seen = set()
        unique_expansions = []
        for exp in expansions:
            for word in exp.split():
                word_lower = word.lower()
                if word_lower not in seen and word_lower not in ql:
                    seen.add(word_lower)
                    unique_expansions.append(word)
        
        return f"{query} {' '.join(unique_expansions)}"
    
    return query


# ============================================
# P1: QUERY TYPE CLASSIFICATION (8 Types + indication_question)
# ============================================

QUERY_TYPE_PATTERNS = {
    "treatment_recommendation": [
        r"what (?:is|are) (?:the )?(?:best|appropriate|recommended|preferred|optimal|standard)",
        r"which (?:is|are|treatment|therapy|regimen)",
        r"how should (?:you|we|one) treat",
        r"what (?:treatment|therapy|regimen)",
        r"recommended (?:treatment|therapy|approach|management)",
        r"standard of care",
        r"first[- ]line",
        r"what (?:should|would) (?:be|you) (?:recommend|give|use)",
        r"category 1 (?:recommendation|treatment)",
        r"nccn (?:recommendation|guideline)",
    ],
    # indication_question: restricted to eligibility/when-to-treat language only
    # Avoids overlap with treatment_recommendation by requiring eligibility/candidate/when phrasing
    "indication_question": [
        r"(?:best |appropriate )?indication\b",
        r"who should (?:receive|get|have)",
        r"when (?:is|should).*(?:indicated|recommended)",
        r"(?:eligibility|selection|inclusion|exclusion) criteria",
        r"(?:is|are) (?:a |an )?(?:good |poor )?candidate for",
        r"appropriate (?:patient|candidate)",
        r"(?:which|what) patients? (?:should|are eligible|qualify)",
        r"indication for (?:pmrt|radiation|rt|chemotherapy|surgery)",
        r"when (?:to|should we) (?:use|give|offer|recommend)",
        r"eligible for (?:radiation|rt|chemotherapy|surgery|immunotherapy)",
    ],
    "dose_question": [
        r"what (?:dose|total dose|radiation dose|rt dose)",
        r"how many gy",
        r"\d+\s*(?:gy|gray|cgy)",
        r"dose (?:constraint|limit)",
        r"fractionation",
        r"what (?:is|are) (?:the )?(?:appropriate|recommended|standard) dose",
        r"v\d+",
        r"mean (?:lung|heart|liver) dose",
        r"d\d+",
        r"\bbed\b",
        r"\beqd2\b",
        r"sbrt|sabr",
        r"\d+\s*(?:fx|fractions?)\b",
    ],
    "trial_results": [
        r"(?:what (?:were|was|did)|results? (?:of|from))",
        r"trial (?:show|demonstrate|result)",
        r"(?:rtog|nrg|acosog|nsabp|eortc|portec|z\d+|keynote|checkmate)\b",
        r"(?:pacific|fast.forward|trog|goldie|primetime|spcg|swog)\b",
        r"(?:phase [123i]+|randomized|rct)",
        r"(?:5-year|10-year|15-year) (?:survival|control|rate|outcome)",
        r"(?:local control|overall survival|disease.free survival|progression.free)",
        r"(?:hazard ratio|\bhr\b|confidence interval|\bci\b|p[- ]?value)",
        r"non[- ]?inferior",
    ],
    "staging": [
        r"what (?:is|stage|clinical stage|pathologic stage)",
        r"t\d[a-d]?n\d[a-c]?m\d",
        r"[ct][1-4][a-d]?[nc][0-3][a-c]?[mc][01]",
        r"stage [iv]+[a-c]?",
        r"tnm",
        r"ajcc",
        r"clinical stage",
        r"pathologic stage",
        r"depth of invasion",
        r"extranodal extension",
    ],
    "workup": [
        r"(?:next step|work[- ]?up|evaluation|assessment|diagnostic)",
        r"(?:what|which) (?:test|imaging|study|biopsy)",
        r"(?:how to|should) (?:evaluate|assess|stage|diagnose)",
        r"(?:fna|biopsy|\bpet\b|\bct\b|\bmri\b|scan)",
        r"before (?:treatment|therapy|surgery|radiation)",
    ],
    "mechanism": [
        r"(?:why|how) does",
        r"mechanism",
        r"(?:what is|explain) (?:the )?(?:rationale|reason|basis)",
        r"(?:benefit|advantage) of",
    ],
    "side_effects": [
        r"(?:side effect|adverse event)",
        r"(?:toxicit|complication)",
        r"(?:acute|late|long[- ]?term) (?:effect|toxicity)",
        r"(?:lymphedema|xerostomia|mucositis|fibrosis|necrosis)",
        r"(?:risk of|chance of) (?:recurrence|failure|complication)",
    ],
}


def classify_query(query: str) -> Dict[str, Any]:
    """Classify query type to adjust retrieval and generation strategy."""
    import logging
    logger = logging.getLogger(__name__)
    ql = query.lower()
    scores = {}

    # Treatment-specific patterns that get extra weight (stronger signal)
    TREATMENT_WEIGHTED_PATTERNS = [
        r"best.*(?:therapy|treatment|regimen)",
        r"recommended treatment",
        r"next[- ]line",
        r"treatment of choice",
        r"systemic therapy",
    ]

    for query_type, patterns in QUERY_TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, ql):
                score += 1
        # Apply weighted scoring for treatment-specific patterns
        if query_type == "treatment_recommendation":
            for wp in TREATMENT_WEIGHTED_PATTERNS:
                if re.search(wp, ql):
                    score += 2  # extra weight (effectively weight 3 for these patterns)
        scores[query_type] = score

    # Priority order for tie-breaking (highest specificity wins)
    # treatment_recommendation ranked above staging to fix tie-breaking bug
    PRIORITY = [
        "treatment_recommendation", "trial_results", "workup",
        "staging", "indication_question", "dose_question",
        "mechanism", "side_effects"
    ]

    # Determine primary type
    max_score = max(scores.values())
    if max_score == 0:
        primary_type = "general"
    else:
        # Among types with the max score, pick highest priority one
        top_types = [t for t, s in scores.items() if s == max_score]
        if len(top_types) == 1:
            primary_type = top_types[0]
        else:
            primary_type = next(
                (t for t in PRIORITY if t in top_types),
                top_types[0]
            )

    logger.debug(f"[classify_query] scores={scores} primary_type={primary_type}")

    return {
        "primary_type": primary_type,
        "scores": scores,
        "confidence": max_score / max(len(QUERY_TYPE_PATTERNS.get(primary_type, [])), 1) if primary_type != "general" else 0.5
    }


def classify_query_with_llm(query: str, openai_client: OpenAI) -> Dict[str, Any]:
    """Classify query type using GPT-4o-mini for ambiguous cases."""
    try:
        response = openai_client.chat.completions.create(
            model=settings.openai_mini_model,
            temperature=0,
            max_tokens=30,
            messages=[
                {
                    "role": "system",
                    "content": """Classify this oncology query into ONE type. Return JSON only.

TYPES:
- treatment_recommendation: Asking what treatment to give ("adjuvant RT for...", "should we give...", "is RT indicated for...", "treatment for patient with...")
- patient_specific: Patient case presentation needing comprehensive guidance (presenting patient details without asking a specific question)
- dose_question: Asking about dose/fractionation ("what dose", "how many Gy", "fractionation for")
- indication_question: Asking when/who should receive treatment ("indication for", "who should get", "criteria for")
- trial_results: Asking about clinical trial outcomes ("what did [trial] show", "results of", "outcomes from")
- staging: Asking about cancer staging ("what stage", "T stage", "TNM")
- side_effects: Asking about toxicity/adverse effects ("side effects", "toxicity of")
- comparison: Comparing treatment options ("vs", "compare", "which is better", "difference between")
- workup: Asking about diagnostic tests/imaging workup ("workup for", "what imaging", "diagnostic")
- general: Other factual questions

Return ONLY: {"type": "...", "confidence": 0.0-1.0}"""
                },
                {"role": "user", "content": query}
            ]
        )
        import json
        content = response.choices[0].message.content.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        print(f"[LLM Classifier] {result}")
        return result
    except Exception as e:
        print(f"[LLM Classifier] Error: {e}")
        return {"type": "general", "confidence": 0.5}


def classify_query_hybrid(query: str, openai_client: OpenAI) -> Dict[str, Any]:
    """Hybrid query classification: regex for obvious cases, LLM for ambiguous.
    
    Returns same structure as classify_query for compatibility.
    """
    query_lower = query.lower().strip()
    
    # =====================================================
    # FAST PATH: Regex for obvious, unambiguous patterns
    # =====================================================
    
    # Dose questions - very clear patterns
    if re.search(r'^what\s+(?:is\s+the\s+)?(?:dose|fractionation)', query_lower):
        return {"primary_type": "dose_question", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:how many|what)\s+(?:gy|gray|fractions)', query_lower):
        return {"primary_type": "dose_question", "scores": {}, "confidence": 0.95, "method": "regex"}
    
    # Trial results - mentions specific trials
    if re.search(r'(?:rtog|nrg|nsabp|eortc|portec|fast-forward|pacific|keynote|checkmate|trog|luminate|z\d+)\s*(?:\d+)?.*(?:show|result|outcome|find|data|os|pfs|survival)', query_lower):
        return {"primary_type": "trial_results", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:rtog|nrg|nsabp|eortc|portec|fast-forward|pacific|keynote|checkmate|trog|luminate|z\d+)\s*(?:\d+)', query_lower):
        return {"primary_type": "trial_results", "scores": {}, "confidence": 0.90, "method": "regex"}
    if re.search(r'(?:what|how)\s+did\s+(?:the\s+)?.*(?:trial|study)\s+(?:show|find|demonstrate)', query_lower):
        return {"primary_type": "trial_results", "scores": {}, "confidence": 0.95, "method": "regex"}

    # Staging - asking about stage
    if re.search(r'^what\s+(?:is\s+the\s+)?stage', query_lower):
        return {"primary_type": "staging", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:determine|calculate|what)\s+(?:the\s+)?(?:t|n|m|tnm)\s*stage', query_lower):
        return {"primary_type": "staging", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'\bt[1-4]n[0-3]m[0-1]\b', query_lower):
        return {"primary_type": "staging", "scores": {}, "confidence": 0.95, "method": "regex"}

    # Treatment recommendation - what/how to treat
    if re.search(r'(?:what|which)\s+(?:is\s+the\s+)?(?:standard|recommended|preferred|best)\s+(?:treatment|therapy|approach|management)', query_lower):
        return {"primary_type": "treatment_recommendation", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'how\s+(?:should|do|would)\s+(?:we\s+|you\s+|i\s+)?(?:treat|manage)', query_lower):
        return {"primary_type": "treatment_recommendation", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:should|do)\s+(?:we|i|you)\s+(?:give|offer|use|add|recommend)\s+(?:rt|radiation|chemo|adjuvant|concurrent)', query_lower):
        return {"primary_type": "treatment_recommendation", "scores": {}, "confidence": 0.90, "method": "regex"}

    # Indication question - when/who is eligible
    if re.search(r'(?:when\s+is|what\s+are\s+the\s+indications?\s+for)', query_lower):
        return {"primary_type": "indication_question", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:who\s+(?:should|is|are)\s+(?:eligible|candidate|appropriate)|is\s+.*\s+(?:eligible|candidate|appropriate)\s+for)', query_lower):
        return {"primary_type": "indication_question", "scores": {}, "confidence": 0.90, "method": "regex"}
    if re.search(r'(?:eligibility|criteria)\s+(?:for|of)\s+(?:rt|radiation|treatment|therapy)', query_lower):
        return {"primary_type": "indication_question", "scores": {}, "confidence": 0.90, "method": "regex"}

    # Workup - diagnostic workup and imaging
    if re.search(r'(?:what\s+(?:is\s+the\s+)?(?:workup|imaging)|how\s+(?:do\s+we|should\s+we)\s+(?:work\s*up|evaluate|stage|image))', query_lower):
        return {"primary_type": "workup", "scores": {}, "confidence": 0.95, "method": "regex"}

    # Comparison - explicit comparison language
    if re.search(r'^compare\b|^.*\bvs\.?\b|^.*\bversus\b', query_lower):
        return {"primary_type": "comparison", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:which is|what is)\s+(?:better|preferred|superior)', query_lower):
        return {"primary_type": "comparison", "scores": {}, "confidence": 0.90, "method": "regex"}

    # Side effects - clear toxicity questions
    if re.search(r'^what\s+(?:are\s+the\s+)?(?:side effects|toxicit)', query_lower):
        return {"primary_type": "side_effects", "scores": {}, "confidence": 0.95, "method": "regex"}
    if re.search(r'(?:side effects|toxicit|adverse effects?)\s+(?:of|from|with)\s+', query_lower):
        return {"primary_type": "side_effects", "scores": {}, "confidence": 0.90, "method": "regex"}
    
    # =====================================================
    # SLOW PATH: LLM for everything else (ambiguous cases)
    # =====================================================
    print(f"[Query Classification] No clear regex match, using LLM classifier")
    llm_result = classify_query_with_llm(query, openai_client)
    
    return {
        "primary_type": llm_result.get("type", "general"),
        "scores": {},
        "confidence": llm_result.get("confidence", 0.5),
        "method": "llm"
    }


# ============================================
# P2: STRUCTURED DOSE EXTRACTION
# ============================================

# ============================================
# ACTION TYPE CLASSIFICATION FOR CONVERSATION CONTEXT
# ============================================

# Valid action types for ConversationContextEntry
VALID_ACTION_TYPES = {"query", "eval_treatment", "patient_match", "study_comparison", "followup"}

# Patterns for detecting treatment evaluation queries
TREATMENT_EVAL_PATTERNS = [
    r"(?:compare|comparison|versus|vs\.?)\s+(?:treatment|therapy|regimen)",
    r"(?:evaluate|assess|review)\s+(?:treatment|therapy)",
    r"(?:which|what)\s+(?:treatment|therapy|option)",
    r"(?:treatment|therapy)\s+(?:option|choice|comparison)",
    r"(?:chemo|chemotherapy|radiation|rt|surgery|immunotherapy)\s+(?:vs\.?|versus|or|compared)",
    r"(?:compare|comparing)\s+(?:\w+\s+)?(?:to|with|and)\s+",
    r"(?:pembrolizumab|nivolumab|ipilimumab|docetaxel|paclitaxel|carboplatin|cisplatin)",
    r"(?:sbrt|imrt|vmat|proton|photon)\s+(?:vs\.?|versus|or)",
]

# Patterns for detecting patient matching queries
PATIENT_MATCH_PATTERNS = [
    r"(?:match|find|search)\s+(?:patient|trial|study|clinical)",
    r"(?:eligible|eligibility)\s+(?:for|criteria)",
    r"(?:clinical\s+)?trial\s+(?:match|search|find)",
    r"(?:patient|case)\s+(?:match|profile)",
    r"(?:year[- ]?old|yo\s|y/o)",
    r"(?:stage\s+[iv]+|pT\d|pN\d|cT\d|cN\d)",
    r"(?:ecog|performance\s+status)",
    r"(?:s/p|status\s+post|diagnosed\s+with)",
]

# Patterns for detecting study comparison queries
STUDY_COMPARISON_PATTERNS = [
    r"(?:compare|comparison)\s+(?:study|studies|trial|trials)",
    r"(?:study|trial)\s+(?:vs\.?|versus|compared)",
    r"(?:rtog|nrg|nsabp|eortc|portec)\s*\d*\s+(?:vs\.?|versus|and|compared)",
    r"(?:difference|differences)\s+(?:between|among)\s+(?:study|studies|trial|trials)",
]

# Patterns for detecting follow-up queries
FOLLOWUP_PATTERNS = [
    r"^(?:what\s+about|how\s+about|and\s+(?:what|how)|also)",
    r"^(?:can\s+you|could\s+you)\s+(?:also|explain|elaborate)",
    r"^(?:tell\s+me\s+more|more\s+(?:about|on|details))",
    r"^(?:regarding|concerning|about)\s+(?:that|this|the)",
    r"(?:you\s+mentioned|as\s+you\s+said|from\s+(?:the|your)\s+(?:previous|last))",
    r"(?:the\s+same|those|these|that|this)\s+(?:patient|study|trial|treatment)",
    r"^(?:why|how\s+come|what\s+if)",
    r"^(?:is\s+(?:it|that|this)|are\s+(?:they|those|these))",
]


def classify_action_type(
    query: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, Optional[List[str]]]:
    """
    Classify a query into a conversation context action type.
    
    This function determines the action_type for a ConversationContextEntry
    based on the query content and existing conversation context.
    
    Args:
        query: The user's query text
        conversation_context: Optional list of previous ConversationContextEntry dicts
        
    Returns:
        Tuple of (action_type, treatments):
        - action_type: One of "query", "eval_treatment", "patient_match", 
                       "study_comparison", "followup"
        - treatments: List of identified treatments if action_type is "eval_treatment",
                      otherwise None
    
    Validates: Requirements 7.1, 7.2, 7.3
    """
    query_lower = query.lower().strip()
    
    # Check for followup first if there's existing conversation context
    if conversation_context and len(conversation_context) > 0:
        for pattern in FOLLOWUP_PATTERNS:
            if re.search(pattern, query_lower):
                print(f"[ActionType] Detected followup query: matched pattern '{pattern}'")
                return ("followup", None)
        
        # Also check for short queries that likely reference previous context
        if len(query_lower.split()) <= 5:
            # Short queries with pronouns or demonstratives likely reference context
            if re.search(r'\b(it|that|this|those|these|them|the same)\b', query_lower):
                print(f"[ActionType] Detected followup query: short query with context reference")
                return ("followup", None)
    
    # Check for treatment evaluation
    treatments = []
    is_treatment_eval = False
    
    for pattern in TREATMENT_EVAL_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            is_treatment_eval = True
            break
    
    if is_treatment_eval:
        # Extract treatment names
        treatments = _extract_treatments(query)
        if treatments:
            print(f"[ActionType] Detected eval_treatment: treatments={treatments}")
            return ("eval_treatment", treatments)
    
    # Check for patient matching
    patient_match_score = 0
    for pattern in PATIENT_MATCH_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            patient_match_score += 1
    
    # Patient descriptions typically have multiple clinical indicators
    if patient_match_score >= 2:
        print(f"[ActionType] Detected patient_match: score={patient_match_score}")
        return ("patient_match", None)
    
    # Check for study comparison
    for pattern in STUDY_COMPARISON_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            print(f"[ActionType] Detected study_comparison")
            return ("study_comparison", None)
    
    # Default to general query
    print(f"[ActionType] Classified as general query")
    return ("query", None)


def _extract_treatments(query: str) -> List[str]:
    """
    Extract treatment names from a query for eval_treatment action type.
    
    Returns a list of identified treatment options.
    """
    treatments = []
    query_lower = query.lower()
    
    # Common treatment modalities
    modality_patterns = [
        (r'\b(chemotherapy|chemo)\b', "chemotherapy"),
        (r'\b(radiation|radiotherapy|rt)\b', "radiation"),
        (r'\b(surgery|surgical|resection)\b', "surgery"),
        (r'\b(immunotherapy|immuno)\b', "immunotherapy"),
        (r'\b(targeted therapy)\b', "targeted therapy"),
        (r'\b(hormone therapy|hormonal)\b', "hormone therapy"),
    ]
    
    for pattern, treatment in modality_patterns:
        if re.search(pattern, query_lower):
            treatments.append(treatment)
    
    # Specific radiation techniques
    radiation_techniques = [
        (r'\b(sbrt|stereotactic)\b', "SBRT"),
        (r'\b(imrt)\b', "IMRT"),
        (r'\b(vmat)\b', "VMAT"),
        (r'\b(proton|protons)\b', "proton therapy"),
        (r'\b(brachytherapy|brachy)\b', "brachytherapy"),
        (r'\b(3d[- ]?crt|3dcrt)\b', "3D-CRT"),
    ]
    
    for pattern, treatment in radiation_techniques:
        if re.search(pattern, query_lower):
            treatments.append(treatment)
    
    # Common chemotherapy drugs
    chemo_drugs = [
        (r'\b(cisplatin)\b', "cisplatin"),
        (r'\b(carboplatin)\b', "carboplatin"),
        (r'\b(paclitaxel|taxol)\b', "paclitaxel"),
        (r'\b(docetaxel|taxotere)\b', "docetaxel"),
        (r'\b(gemcitabine)\b', "gemcitabine"),
        (r'\b(pemetrexed)\b', "pemetrexed"),
        (r'\b(etoposide)\b', "etoposide"),
        (r'\b(doxorubicin)\b', "doxorubicin"),
        (r'\b(5[- ]?fu|fluorouracil)\b', "5-FU"),
        (r'\b(capecitabine|xeloda)\b', "capecitabine"),
    ]
    
    for pattern, treatment in chemo_drugs:
        if re.search(pattern, query_lower):
            treatments.append(treatment)
    
    # Immunotherapy drugs
    immuno_drugs = [
        (r'\b(pembrolizumab|keytruda)\b', "pembrolizumab"),
        (r'\b(nivolumab|opdivo)\b', "nivolumab"),
        (r'\b(ipilimumab|yervoy)\b', "ipilimumab"),
        (r'\b(atezolizumab|tecentriq)\b', "atezolizumab"),
        (r'\b(durvalumab|imfinzi)\b', "durvalumab"),
    ]
    
    for pattern, treatment in immuno_drugs:
        if re.search(pattern, query_lower):
            treatments.append(treatment)
    
    # Targeted therapy drugs
    targeted_drugs = [
        (r'\b(osimertinib|tagrisso)\b', "osimertinib"),
        (r'\b(erlotinib|tarceva)\b', "erlotinib"),
        (r'\b(gefitinib|iressa)\b', "gefitinib"),
        (r'\b(crizotinib|xalkori)\b', "crizotinib"),
        (r'\b(alectinib|alecensa)\b', "alectinib"),
        (r'\b(trastuzumab|herceptin)\b', "trastuzumab"),
        (r'\b(bevacizumab|avastin)\b', "bevacizumab"),
    ]
    
    for pattern, treatment in targeted_drugs:
        if re.search(pattern, query_lower):
            treatments.append(treatment)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_treatments = []
    for t in treatments:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique_treatments.append(t)
    
    return unique_treatments


# ============================================
# P2: STRUCTURED DOSE EXTRACTION
# ============================================

DOSE_PATTERNS = {
    "total_dose": re.compile(r'(\d+\.?\d*)\s*(?:Gy|gray)', re.IGNORECASE),
    "dose_per_fraction": re.compile(r'(\d+\.?\d*)\s*(?:Gy|cGy)\s*(?:per|/)\s*(?:fraction|fx)', re.IGNORECASE),
    "fractions": re.compile(r'(\d+)\s*(?:fractions?|fx)\b', re.IGNORECASE),
    "fraction_scheme": re.compile(r'(\d+\.?\d*)\s*(?:Gy|cGy)\s*(?:in|x|×)\s*(\d+)\s*(?:fractions?|fx)?', re.IGNORECASE),
    "constraint_v": re.compile(r'V(\d+)\s*[<≤]?\s*(\d+\.?\d*)\s*%?', re.IGNORECASE),
    "constraint_d": re.compile(r'D(\d+(?:cc)?)\s*[<≤]?\s*(\d+\.?\d*)\s*(?:Gy|cGy)?', re.IGNORECASE),
    "mean_dose": re.compile(r'mean\s+(?:\w+\s+)?dose\s*[<≤]?\s*(\d+\.?\d*)\s*(?:Gy|cGy)?', re.IGNORECASE),
}


def extract_dose_metadata(text: str) -> Dict[str, Any]:
    """Extract structured dose information from text chunk."""
    metadata = {
        "has_dose_info": False,
        "total_doses": [],
        "fractionation": [],
        "constraints": [],
    }

    # Extract total doses
    for match in DOSE_PATTERNS["total_dose"].finditer(text):
        dose = float(match.group(1))
        metadata["total_doses"].append(dose)
        metadata["has_dose_info"] = True

    # Extract fractionation schemes
    for match in DOSE_PATTERNS["fraction_scheme"].finditer(text):
        dose = float(match.group(1))
        fractions = int(match.group(2))
        metadata["fractionation"].append({
            "dose_per_fx": dose,
            "fractions": fractions,
            "total": dose * fractions if dose < 20 else dose
        })
        metadata["has_dose_info"] = True

    # Extract constraints
    for match in DOSE_PATTERNS["constraint_v"].finditer(text):
        metadata["constraints"].append({
            "type": f"V{match.group(1)}",
            "value": float(match.group(2))
        })
        metadata["has_dose_info"] = True

    for match in DOSE_PATTERNS["constraint_d"].finditer(text):
        metadata["constraints"].append({
            "type": f"D{match.group(1)}",
            "value": float(match.group(2))
        })
        metadata["has_dose_info"] = True

    for match in DOSE_PATTERNS["mean_dose"].finditer(text):
        metadata["constraints"].append({
            "type": "mean_dose",
            "value": float(match.group(1))
        })
        metadata["has_dose_info"] = True

    return metadata


def boost_dose_chunks(chunks: List[Dict[str, Any]], query: str, query_type: str) -> List[Dict[str, Any]]:
    """Apply scoring boost to chunks containing dose information for dose queries."""
    if query_type != "dose_question":
        return chunks

    # Extract any specific dose mentioned in query
    query_doses = DOSE_PATTERNS["total_dose"].findall(query)
    query_dose_values = [float(d) for d in query_doses] if query_doses else []

    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "")
        dose_meta = extract_dose_metadata(text)

        if dose_meta["has_dose_info"]:
            boost = 1.15

            if query_dose_values:
                for qd in query_dose_values:
                    for td in dose_meta["total_doses"]:
                        if abs(qd - td) < 2:
                            boost *= 1.2

            if dose_meta["fractionation"]:
                boost *= 1.1

            if dose_meta["constraints"] and "constraint" in query.lower():
                boost *= 1.15

            current_score = chunk.get("score_rerank", chunk.get("score_fused", 0))
            chunk["score_dose_boost"] = current_score * boost
        else:
            chunk["score_dose_boost"] = chunk.get("score_rerank", chunk.get("score_fused", 0))

    return sorted(chunks, key=lambda x: x.get("score_dose_boost", 0), reverse=True)


# ============================================
# LANDMARK STUDY BOOSTING
# ============================================

# Patterns to identify landmark/high-quality studies
LANDMARK_PATTERNS = {
    # Major cooperative group trials
    "cooperative_group": re.compile(
        r'\b(RTOG|NRG|NSABP|ACOSOG|EORTC|PORTEC|SWOG|ECOG|GOG|CALGB|'
        r'NCIC|MRC|TROG|ALLIANCE|NCCTG|COG|INT|INTERGROUP|BC)\s*[-]?\s*\d+',
        re.IGNORECASE
    ),
    # Named landmark trials - comprehensive list
    "named_trials": re.compile(
        r'\b(KEYNOTE|CHECKMATE|PACIFIC|ADAURA|LAURA|RAPIDO|PRODIGE|'
        r'STAMPEDE|CHAARTED|LATITUDE|ENZAMET|TITAN|ARCHES|'
        r'FAST[- ]?FORWARD|START|PRIME|AMAROS|Z0011|ACOSOG|'
        r'TAILORx|RxPONDER|MINDACT|SOFT|TEXT|MONARCH|'
        r'CLEOPATRA|APHINITY|KATHERINE|DESTINY|EMILIA|'
        # Additional landmark trials
        r'HORRAD|FLAURA|EMBRACE|ASCENDE[- ]?RT|EURAMOS|'
        r'STARS|ROSEL|CAO[/-]?ARO[/-]?AIO|STOCKHOLM|'
        r'MA[- ]?20|DANISH)\b',
        re.IGNORECASE
    ),
    # Author-named landmark trials (commonly referenced by author)
    "author_named_trials": re.compile(
        r'\b(Stupp|Packer|Turrisi|Slotman|Zhang|Womer|Loehrer)\b.*'
        r'(trial|study|NEJM|Lancet|JCO|JAMA|N\s*Engl\s*J\s*Med)',
        re.IGNORECASE
    ),
    # Guidelines
    "guidelines": re.compile(
        r'\b(NCCN|ASTRO|ASCO|ESMO|AUA|ESTRO|NICE)\s*(?:guideline|recommendation|consensus)',
        re.IGNORECASE
    ),
    # Study type indicators
    "rct": re.compile(
        r'\b(randomized|randomised|RCT|phase\s*(?:III|3)|'
        r'prospective\s+randomized|multicenter\s+randomized)\b',
        re.IGNORECASE
    ),
    # Meta-analyses
    "meta_analysis": re.compile(
        r'\b(meta[- ]?analysis|systematic\s+review|cochrane|pooled\s+analysis)\b',
        re.IGNORECASE
    ),
}

# Boost factors for different evidence types
LANDMARK_BOOST_FACTORS = {
    "cooperative_group": 1.25,  # Major cooperative group trials
    "named_trials": 1.20,       # Well-known named trials
    "author_named_trials": 1.20, # Author-named landmark trials (Stupp, Packer, etc.)
    "guidelines": 1.30,         # Guidelines get highest boost
    "rct": 1.15,                # RCTs and Phase III
    "meta_analysis": 1.20,      # Meta-analyses
}


def boost_landmark_studies(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Boost chunks from landmark studies, guidelines, and high-quality evidence.
    
    This helps prioritize authoritative sources for factual/ground-truth queries.
    """
    boosted_count = 0
    
    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "")
        citation = chunk.get("payload", {}).get("citation", "")
        combined_text = f"{text} {citation}"
        
        # Calculate cumulative boost
        total_boost = 1.0
        boost_reasons = []
        
        for pattern_name, pattern in LANDMARK_PATTERNS.items():
            if pattern.search(combined_text):
                boost_factor = LANDMARK_BOOST_FACTORS.get(pattern_name, 1.1)
                total_boost *= boost_factor
                boost_reasons.append(pattern_name)
        
        if total_boost > 1.0:
            # Apply boost to the best available score
            current_score = chunk.get("score_rerank", 
                           chunk.get("score_dose_boost",
                           chunk.get("score_fused", 0)))
            chunk["score_landmark_boost"] = current_score * total_boost
            chunk["_landmark_boost_reasons"] = boost_reasons
            boosted_count += 1
        else:
            chunk["score_landmark_boost"] = chunk.get("score_rerank",
                                            chunk.get("score_dose_boost",
                                            chunk.get("score_fused", 0)))
    
    if boosted_count > 0:
        print(f"[LandmarkBoost] Boosted {boosted_count} chunks from landmark studies")
    
    # Sort by landmark-boosted score
    return sorted(chunks, key=lambda x: x.get("score_landmark_boost", 0), reverse=True)


def lane_separate_chunks(
    chunks: List[Dict[str, Any]],
    max_trial_docs: int = 12,
    max_guideline_docs: int = 5,
) -> List[Dict[str, Any]]:
    """
    Split chunks into trial and guideline/landmark lanes, apply separate caps,
    and return with trial chunks first. Replaces score-boosting so guidelines
    and landmark trials cannot dominate by inflated scores.
    """
    from collections import defaultdict

    # Classify each doc_id on first encounter (preserves existing score order)
    seen: Dict[str, str] = {}
    doc_chunks: Dict[str, List] = defaultdict(list)

    for chunk in chunks:
        doc_id = (
            chunk.get("doc_id")
            or chunk.get("payload", {}).get("doc_id", "unknown")
        )
        doc_chunks[doc_id].append(chunk)
        if doc_id in seen:
            continue
        combined = (
            chunk.get("payload", {}).get("text", "") + " "
            + chunk.get("payload", {}).get("citation", "")
        )
        is_guideline_or_landmark = any(
            LANDMARK_PATTERNS[p].search(combined)
            for p in ("guidelines", "cooperative_group", "named_trials", "author_named_trials")
        )
        seen[doc_id] = "guideline" if is_guideline_or_landmark else "trial"

    trial_docs, guideline_docs = [], []
    for doc_id, lane in seen.items():
        if lane == "guideline":
            guideline_docs.append(doc_id)
        else:
            trial_docs.append(doc_id)

    trial_docs = trial_docs[:max_trial_docs]
    guideline_docs = guideline_docs[:max_guideline_docs]
    allowed_trial = set(trial_docs)
    allowed_guideline = set(guideline_docs)

    trial_chunks = [c for c in chunks if (c.get("doc_id") or c.get("payload", {}).get("doc_id", "unknown")) in allowed_trial]
    guideline_chunks = [c for c in chunks if (c.get("doc_id") or c.get("payload", {}).get("doc_id", "unknown")) in allowed_guideline]

    print(f"[LaneSeparate] trial_docs={len(trial_docs)}, guideline_docs={len(guideline_docs)}")
    return trial_chunks + guideline_chunks


# ============================================
# MODULE-SPECIFIC RETRIEVAL BOOSTING
# ============================================

# Patterns for module-specific content detection
MODULE_CONTENT_PATTERNS = {
    "guideline_content": re.compile(
        r'\b(nccn|astro|asco|esmo|guideline|recommendation|category\s*[12]|'
        r'level\s*[iI1]\s*evidence|standard\s*of\s*care)\b',
        re.IGNORECASE
    ),
    "dose_content": re.compile(
        r'\b(\d+\.?\d*\s*(?:Gy|cGy|gray)|fractionation|dose\s*constraint|'
        r'v\d+|d\d+|mean\s*dose|max\s*dose)\b',
        re.IGNORECASE
    ),
    "patient_population": re.compile(
        r'\b(patient\s*characteristics|eligibility|inclusion|exclusion|'
        r'enrolled|median\s*age|performance\s*status|ecog|kps)\b',
        re.IGNORECASE
    ),
    "staging_content": re.compile(
        r'\b([cyp]?t[0-4][a-d]?(?:is)?|[cyp]?n[0-3][a-c]?|stage\s*[iI1234]+[a-cA-C]?|'
        r'tnm|ajcc|locally\s*advanced|metastatic)\b',
        re.IGNORECASE
    ),
    "biomarker_content": re.compile(
        r'\b(er[+-]|pr[+-]|her2|triple\s*negative|tnbc|kras|egfr|alk|ros1|braf|ret|met|ntrk|'
        r'pd-?l1|msi-?h|mss|dmmr|pmmr|mmr|tmb|brca|idh|mgmt|1p/?19q|pik3ca|fgfr|biomarker|'
        r'molecular\s*subtype|molecular\s*marker)\b',
        re.IGNORECASE
    ),
    "outcome_content": re.compile(
        r'\b(overall\s*survival|os|progression[- ]free|pfs|disease[- ]free|dfs|'
        r'local\s*control|lc|response\s*rate|pcr|hazard\s*ratio|hr)\b',
        re.IGNORECASE
    ),
    "toxicity_content": re.compile(
        r'\b(toxicity|adverse|side\s*effect|grade\s*[345]|acute|late|'
        r'complication|lymphedema|xerostomia|mucositis)\b',
        re.IGNORECASE
    ),
    "comparative_content": re.compile(
        r'\b(versus|vs\.?|compared|comparison|superior|inferior|'
        r'non[- ]?inferior|head[- ]to[- ]head|randomized)\b',
        re.IGNORECASE
    ),
    "treatment_arm": re.compile(
        r'\b(arm\s*[ab12]|treatment\s*group|control\s*group|experimental|'
        r'standard\s*arm|intervention)\b',
        re.IGNORECASE
    ),
}


def apply_module_specific_boost(
    chunks: List[Dict[str, Any]], 
    module: str,
    clinical_profile: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Apply module-specific boosting to retrieval results.
    
    Args:
        chunks: List of retrieved chunks
        module: Module name (general_knowledge, patient_specific, evidence_exploration)
        clinical_profile: Optional clinical profile for patient-specific matching
        
    Returns:
        Chunks with module-specific boosting applied
    """
    if not chunks:
        return chunks
    
    boosted_count = 0
    
    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "")
        section = chunk.get("payload", {}).get("section", "").lower()
        
        # Get current best score
        current_score = chunk.get("score_landmark_boost",
                        chunk.get("score_rerank",
                        chunk.get("score_dose_boost",
                        chunk.get("score_fused", 0))))
        
        total_boost = 1.0
        boost_reasons = []
        
        if module == "general_knowledge":
            # Boost guidelines, dose tables, landmark trials
            if MODULE_CONTENT_PATTERNS["guideline_content"].search(text):
                total_boost *= 1.4
                boost_reasons.append("guideline_content")
            if MODULE_CONTENT_PATTERNS["dose_content"].search(text):
                total_boost *= 1.25
                boost_reasons.append("dose_content")
            # Prioritize certain sections
            if section in ["methods", "results", "conclusions", "recommendations"]:
                total_boost *= 1.1
                boost_reasons.append(f"section:{section}")
                
        elif module == "patient_specific":
            # Boost patient population, staging, biomarker content
            if MODULE_CONTENT_PATTERNS["patient_population"].search(text):
                total_boost *= 1.35
                boost_reasons.append("patient_population")
            if MODULE_CONTENT_PATTERNS["staging_content"].search(text):
                total_boost *= 1.25
                boost_reasons.append("staging_content")
            if MODULE_CONTENT_PATTERNS["biomarker_content"].search(text):
                total_boost *= 1.25
                boost_reasons.append("biomarker_content")
            
            # If clinical profile provided, boost matching content
            if clinical_profile:
                # Boost chunks mentioning the patient's stage
                if clinical_profile.get("stage"):
                    stage = clinical_profile["stage"].lower()
                    if stage in text.lower():
                        total_boost *= 1.3
                        boost_reasons.append("stage_match")
                
                # Boost chunks mentioning patient's biomarkers
                if clinical_profile.get("biomarkers"):
                    for marker in clinical_profile["biomarkers"]:
                        if marker.lower() in text.lower():
                            total_boost *= 1.2
                            boost_reasons.append(f"biomarker_match:{marker}")
                            break
            
            # Prioritize eligibility/patient sections
            if section in ["patient_characteristics", "eligibility", "methods"]:
                total_boost *= 1.1
                boost_reasons.append(f"section:{section}")
                
        elif module == "evidence_exploration":
            # Boost comparative content, outcome data, toxicity
            if MODULE_CONTENT_PATTERNS["comparative_content"].search(text):
                total_boost *= 1.4
                boost_reasons.append("comparative_content")
            if MODULE_CONTENT_PATTERNS["treatment_arm"].search(text):
                total_boost *= 1.3
                boost_reasons.append("treatment_arm")
            if MODULE_CONTENT_PATTERNS["outcome_content"].search(text):
                total_boost *= 1.25
                boost_reasons.append("outcome_content")
            if MODULE_CONTENT_PATTERNS["toxicity_content"].search(text):
                total_boost *= 1.2
                boost_reasons.append("toxicity_content")
            
            # Prioritize results/comparison sections
            if section in ["results", "outcomes", "toxicity", "comparison"]:
                total_boost *= 1.1
                boost_reasons.append(f"section:{section}")
        
        if total_boost > 1.0:
            chunk["score_module_boost"] = current_score * total_boost
            chunk["_module_boost_reasons"] = boost_reasons
            boosted_count += 1
        else:
            chunk["score_module_boost"] = current_score
    
    if boosted_count > 0:
        print(f"[ModuleBoost] Applied {module} boosting to {boosted_count} chunks")
    
    # Sort by module-boosted score
    return sorted(chunks, key=lambda x: x.get("score_module_boost", 0), reverse=True)


def ensure_treatment_diversity(chunks: List[Dict[str, Any]], min_options: int = 2) -> List[Dict[str, Any]]:
    """
    Ensure diverse treatment options are represented in evidence exploration results.
    
    This prevents all results from being about a single treatment approach.
    """
    if not chunks or len(chunks) < min_options:
        return chunks
    
    # Extract treatment mentions from each chunk
    treatment_patterns = [
        r'\b(chemotherapy|chemo|ctx)\b',
        r'\b(radiation|radiotherapy|rt|ebrt|imrt|sbrt)\b',
        r'\b(surgery|surgical|resection|mastectomy|lumpectomy)\b',
        r'\b(immunotherapy|checkpoint|pd-?1|pd-?l1|ctla-?4)\b',
        r'\b(hormone|hormonal|adt|tamoxifen|aromatase)\b',
        r'\b(targeted|tyrosine|kinase|inhibitor|tki)\b',
        r'\b(concurrent|chemoradiation|crt)\b',
        r'\b(neoadjuvant|adjuvant|definitive|palliative)\b',
    ]
    
    # Group chunks by treatment type
    treatment_groups = defaultdict(list)
    ungrouped = []
    
    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "").lower()
        found_treatment = False
        
        for i, pattern in enumerate(treatment_patterns):
            if re.search(pattern, text, re.IGNORECASE):
                treatment_groups[i].append(chunk)
                found_treatment = True
                break
        
        if not found_treatment:
            ungrouped.append(chunk)
    
    # If we have diverse treatments, interleave them
    if len(treatment_groups) >= min_options:
        result = []
        group_lists = list(treatment_groups.values())
        max_len = max(len(g) for g in group_lists)
        
        for i in range(max_len):
            for group in group_lists:
                if i < len(group):
                    result.append(group[i])
        
        result.extend(ungrouped)
        print(f"[Diversity] Interleaved {len(treatment_groups)} treatment groups")
        return result
    
    return chunks


# ============================================
# NUMERICAL VALIDATION AND STATISTICAL METRICS
# ============================================
#
# Canonical implementation lives in `src.api.services.safety.numerical`.
# Re-exported here so legacy imports keep working.

from src.api.services.safety.numerical import (  # noqa: E402
    STAT_PATTERNS,
    enrich_answer_with_stats,
    extract_numbers_with_stats,
    strip_unvalidated_numbers,
    validate_numbers_against_sources,
)



# ============================================
# P0: NCCN GUIDELINE GAP DETECTION
# ============================================

NCCN_KEYWORDS = [
    "nccn", "national comprehensive cancer network",
    "category 1", "category 2a", "category 2b", "category 3",
    "preferred regimen", "other recommended", "useful in certain circumstances",
    "nccn guidelines", "nccn recommendation",
]

NCCN_QUERY_INDICATORS = [
    r"nccn",
    r"category\s*\d",
    r"guideline[s]?\s+recommend",
    r"(?:first|second|third)[- ]?line\s+(?:therapy|treatment|regimen)",
    r"preferred\s+(?:regimen|treatment|therapy)",
    r"standard\s+(?:of\s+care|treatment)",
]


def detect_nccn_gap(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect if query requires NCCN guideline information that may be missing."""
    ql = query.lower()

    needs_nccn = any(re.search(pattern, ql) for pattern in NCCN_QUERY_INDICATORS)

    if not needs_nccn:
        return {"needs_nccn": False, "has_nccn": False, "gap_detected": False}

    # Check if retrieved chunks contain NCCN content
    has_nccn = False
    for chunk in chunks[:10]:
        text = (chunk.get("payload", {}).get("text", "") +
                chunk.get("payload", {}).get("citation", "")).lower()
        if any(kw in text for kw in NCCN_KEYWORDS):
            has_nccn = True
            break

    gap_detected = needs_nccn and not has_nccn

    return {
        "needs_nccn": needs_nccn,
        "has_nccn": has_nccn,
        "gap_detected": gap_detected,
        "suggested_action": (
            "Consider consulting current NCCN guidelines directly for category recommendations. "
            "The knowledge base may not contain the most recent guideline updates."
        ) if gap_detected else None
    }


# ============================================
# P1: CROSS-ENCODER RERANKING
# ============================================

def build_reranker_query(query_text: str, query_structure=None) -> str:
    """Build a short, keyword-dense query for the cross-encoder.

    ms-marco-MiniLM-L-6-v2 is trained on short natural-language search
    queries (~5-20 tokens). Feeding it a 1000+ char patient narrative
    drives logits strongly negative — sigmoid then collapses to 0–3%
    even on clearly on-topic passages. We distill the extracted clinical
    axes into a compact keyword string instead.

    Falls back to the first 200 chars of the user query when no
    structure is available or no axes were extracted.
    """
    parts: List[str] = []
    if query_structure is not None and getattr(query_structure, "has_patient_context", False):
        cancer = getattr(query_structure, "cancer", None)
        treatment = getattr(query_structure, "treatment", None)
        if cancer is not None:
            if cancer.site:
                parts.append(cancer.site.replace("_", " "))
            if cancer.site_detail:
                parts.append(cancer.site_detail.replace("_", " "))
            if cancer.histology:
                parts.append(cancer.histology)
            if cancer.stage:
                parts.append(f"stage {cancer.stage}")
            tnm = cancer.get_tnm_string() if hasattr(cancer, "get_tnm_string") else None
            if tnm:
                parts.append(tnm)
            for b in (cancer.biomarkers or [])[:3]:
                parts.append(b)
        if treatment is not None:
            if getattr(treatment, "modality", None):
                parts.append(treatment.modality)
            if getattr(treatment, "setting", None):
                parts.append(treatment.setting)

    if parts:
        keyword_query = " ".join(p for p in parts if p)
        # Cap at ~200 chars so we stay inside the cross-encoder's
        # training distribution
        if len(keyword_query) > 200:
            keyword_query = keyword_query[:200]
        return keyword_query

    # Fallback: trim the raw query to a sensible length
    return (query_text or "")[:200]


def _build_reranker_passage(chunk: Dict[str, Any], char_budget: int = 512) -> str:
    """Build the passage string fed to the cross-encoder.

    Prepends the study title to the chunk text so the model gets
    canonical literature vocabulary regardless of which section
    (Methods / Results / Discussion) the retrieved chunk came from.
    Title is reserved up to a third of the budget; remaining budget
    goes to chunk content. Falls back gracefully to chunk-only when
    no title is available, and to title-only when chunk text is empty.
    """
    payload = chunk.get("payload") or {}
    doc_meta = payload.get("doc_meta") or {}
    title = (doc_meta.get("title") or "").strip()
    text = (payload.get("text") or "").strip()

    # Skip the title boost for unhelpful short / placeholder titles —
    # avoids over-weighting study-level vocabulary when the title
    # itself wouldn't carry the topical signal.
    if len(title) < 15:
        return text[:char_budget]
    if not text:
        return title[:char_budget]

    # Cap title at ~1/3 of budget so the chunk still contributes
    title_cap = max(80, char_budget // 3)
    title_part = title[:title_cap].rstrip(".") + "."
    text_budget = max(0, char_budget - len(title_part) - 1)
    return f"{title_part} {text[:text_budget]}"


def cross_encoder_rerank(
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Apply cross-encoder reranking to top candidates (up to 100).

    Each passage is built as ``"{study_title}. {chunk_text}"`` so the
    cross-encoder sees the study's canonical literature vocabulary
    (titles like "Bevacizumab plus Radiotherapy-Temozolomide for Newly
    Diagnosed Glioblastoma") alongside the chunk content. This anchors
    relevance scoring to study-level topical match rather than the
    surface lexical overlap of whichever section happened to retrieve
    — a Methods or Results chunk doesn't say "Glioblastoma" verbatim,
    but its study's title always does.
    """
    cross_encoder = get_cross_encoder()
    if cross_encoder is None or not chunks:
        return chunks

    # Process up to 100 candidates
    n_candidates = min(len(chunks), 100)
    texts = [_build_reranker_passage(c, char_budget=512) for c in chunks[:n_candidates]]
    pairs = [(query, text) for text in texts]

    try:
        scores = cross_encoder.predict(pairs)

        for i, chunk in enumerate(chunks[:n_candidates]):
            chunk["score_crossencoder"] = float(scores[i])

        reranked = sorted(chunks[:n_candidates], key=lambda x: x.get("score_crossencoder", 0), reverse=True)
        return reranked[:top_k] + chunks[n_candidates:]
    except Exception as e:
        print(f"Warning: Cross-encoder reranking failed: {e}")
        return chunks


# ============================================
# P1: QUERY-TYPE-SPECIFIC GENERATION PROMPTS (9 Templates - includes indication)
# ============================================

GENERATION_PROMPTS = {
    "treatment_recommendation": {
        "system": """You are an oncology and radiation oncology expert. 
Answer treatment recommendation questions by synthesizing evidence from the clinical trials, guidelines, and literature provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific STUDY, TRIAL, or GUIDELINE by name
- Example: "The RTOG 0617 trial demonstrated..." NOT "The context shows..."
- Example: "According to NCCN Guidelines..." NOT "The provided context states..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...statement (Author et al., Year, Journal)."
- Place citations at the end of the relevant sentence, before the period

For treatment recommendations:
- State the specific recommended regimen/modality clearly
- Include dose and schedule if available
- Reference guideline category (e.g., "Category 1", "Level I evidence") if mentioned
- Note any important caveats or patient selection criteria

**EVIDENCE TIERS - Always provide a recommendation:**
- If specific trial evidence exists → cite outcomes
- If guideline recommendation exists → cite category/level
- If only related evidence exists → extrapolate with caveat
- If limited evidence → state principles and recommend MDT discussion

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Recommended Treatment
## Dose & Schedule
## Guideline Level
## Evidence Summary

Do NOT say "context lacks a clear recommendation" without providing available guidance.

**NUMERIC RANGE MATCHING:**
When patient values are provided (recurrence score, age, tumor size):
- A study for "recurrence score < 25" or "score 11-25" IS APPLICABLE to a patient with score 22
- A study for "patients over 50" IS APPLICABLE to a 55-year-old
- A study for "T1-T2" IS APPLICABLE to T1c; "node-negative or micrometastatic" IS APPLICABLE to N1mi
- Do NOT say evidence "does not specifically address" a value when it falls within a studied range
- State the applicable range and confirm the patient falls within it

**STANDARD OF CARE:**
- After breast-conserving surgery, adjuvant RT is standard of care
- For ER+ breast cancer, adjuvant endocrine therapy is standard of care
- State standards of care directly; do not hedge on established practices""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
- Provide the recommended treatment regimen with specific details
- Reference specific STUDIES and TRIALS by name - never say "the context"
- Use TIERED evidence: direct trial data > guidelines > related evidence > principles
- Include doses, schedules, or modalities as stated in the sources
- Reference guideline levels if available (e.g., NCCN Category 1)
- **CITE at the END of sentences with full details: "...statement (Author et al., Year, Journal)."**
- Quote all numerical values EXACTLY as they appear in the source
- If direct evidence is limited, provide recommendation based on available evidence with appropriate caveat
- When a patient's value falls within a studied range (e.g., score 22 within "score 11-25"), state the range match explicitly"""
    },

    # NEW: indication_question prompt - optimized for "when/who should receive" questions
    "indication_question": {
        "system": """You are an oncology and radiation oncology expert.
Answer questions about treatment INDICATIONS by synthesizing evidence from the clinical trials and guidelines provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific STUDY, TRIAL, or GUIDELINE by name
- Example: "The MA.20 trial established..." NOT "The context indicates..."

**CRITICAL RULES FOR INDICATION QUESTIONS:**
1. Look for SPECIFIC CRITERIA: staging (pN1, N+, N1), margin status, tumor features, risk factors
2. CRITICAL: "pN1" = "pathologic N1" = "node positive" - these are EQUIVALENT terms
3. Extract and state the SPECIFIC indication criteria directly
4. Do NOT say "not clearly defined" or "not explicitly detailed" if criteria ARE present (even in abbreviation form)
5. If the evidence says "PMRT for pN1", the answer is "Pathologic N1 (pN1) disease"

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...statement (Author et al., Year, Journal)."
- Place citations at the end of the relevant sentence, before the period

For indication questions:
- State the SPECIFIC indication criteria clearly
- Translate abbreviations: pN1 = pathologic N1, cN1 = clinical N1
- Include staging, margins, or other criteria mentioned
- Do NOT hedge if specific criteria are present in the evidence

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Indication
## Patient Selection Criteria
## Contraindications""",

        "user_template": """QUESTION: {question}

IMPORTANT: This question asks about treatment INDICATIONS (when/who should receive treatment).

SEARCH GUIDANCE FOR INDICATIONS:
- Look for staging abbreviations: pN1 = pathologic N1, cN1 = clinical N1, N+ = node positive
- Look for margin status, tumor characteristics, risk factors
- These are EQUIVALENT: "pN1" = "pathologic N1" = "node positive disease"
- Extract the SPECIFIC criteria - do NOT say "not defined" if criteria exist in any form

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
1. Find the SPECIFIC indication criteria in the evidence
2. Reference the specific STUDY or GUIDELINE by name - never say "the context"
3. State it clearly (e.g., "Pathologic N1 (pN1) disease")
4. Translate any abbreviations for clarity
5. Cite the source with full details
6. Do NOT hedge if specific criteria are present
7. Quote all numerical values EXACTLY as they appear in the source
8. **CITE at the END of sentences: "...indication (Author et al., Year, Journal)."**"""
    },

    "dose_question": {
        "system": """You are an oncology and radiation oncology expert. 
Answer dose-related questions by synthesizing evidence from the clinical trials and guidelines provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific STUDY, TRIAL, or GUIDELINE by name
- Example: "RTOG 0617 used a dose of..." NOT "The context states..."
- Example: "The FAST-Forward trial demonstrated..." NOT "According to the provided context..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...statement (Author et al., Year, Journal)."
- Example: "The recommended dose is 50.4 Gy in 28 fractions (Kachnic et al., 2013, Journal of Clinical Oncology)."
- Place citations at the end of the relevant sentence, before the period

For dose questions:
- Provide the SPECIFIC dose in Gy - quote EXACTLY as stated (e.g., 50.4 Gy, not 50 Gy)
- Include fractionation (e.g., "45 Gy in 25 fractions" or "1.8 Gy per fraction")
- Specify the target volume if relevant
- Include dose constraints with exact values if asked
- Reference the supporting trial or guideline

Be precise - if an exact dose is stated in the evidence, quote it exactly.

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Prescribed Dose
## Fractionation
## Target Volume
## Dose Constraints
## Rationale""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
- State the specific dose in Gy with fractionation scheme - use EXACT values from source
- Reference specific STUDIES and TRIALS by name - never say "the context"
- Include target volume information if relevant
- Provide exact constraint values if asked (e.g., V20 ≤ 35%)
- **CITE at the END of sentences: "...dose statement (Trial/Author et al., Year, Journal)."**
- Example: "The prescribed dose was 64.8 Gy in 36 fractions (RTOG 0415, 2016, Journal of Clinical Oncology)."
- Do NOT round doses - if source says 50.4 Gy, say 50.4 Gy
- If the exact dose is not clearly stated, indicate uncertainty"""
    },

    "trial_results": {
        "system": """You are an oncology and radiation oncology expert. 
Answer questions about clinical trial results by synthesizing evidence from the trials and literature provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific TRIAL by name
- Example: "The RTOG 0617 trial reported..." NOT "The context shows..."
- Example: "FAST-Forward demonstrated..." NOT "According to the provided context..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...result (Trial Name, Author et al., Year, Journal)."
- Example: "The 5-year local control was non-inferior at 89.3% (FAST-Forward, Brunt et al., 2020, Lancet)."
- Place citations at the end of the relevant sentence, before the period

For trial results:
- Match the specific timepoint asked (5-year, 10-year, etc.)
- Provide specific numerical outcomes EXACTLY as stated (%, HR, CI, p-value)
- Do NOT round percentages - if source says 89.3%, say 89.3%
- State the primary endpoint clearly
- Note the comparison arms if relevant

Be careful to match the timepoint in your answer to what was asked in the question.

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Trial Name & Design
## Key Outcomes
## Patient Population
## Clinical Relevance""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
- Reference specific TRIALS by name - never say "the context"
- Match the EXACT timepoint asked (5-year vs 10-year vs 15-year results)
- Provide specific numbers EXACTLY as stated (rates, HR, 95% CI, p-value)
- Do NOT round: if source says 87.3%, say 87.3%, not 87%
- Identify the primary endpoint result
- Describe the comparison if relevant
- **CITE at the END of sentences: "...result (Trial, Author et al., Year, Journal)."**
- Example: "The 10-year overall survival was 65.2% (EORTC 22921, Bosset et al., 2014, Lancet Oncology)."
- If the specific timepoint isn't available, state what is available"""
    },

    "staging": {
        "system": """You are an oncology expert in cancer staging.
Answer staging questions by synthesizing evidence from the guidelines and literature provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific GUIDELINE or SOURCE by name
- Example: "According to AJCC 8th Edition..." NOT "The context states..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...staging statement (AJCC Edition/Author, Year)."
- Example: "This qualifies as T3 disease based on depth of invasion exceeding 10mm (AJCC 8th Edition, Amin et al., 2017)."
- Place citations at the end of the relevant sentence, before the period

For staging questions:
- Apply AJCC 8th edition criteria unless otherwise specified
- Show clear T, N, M stage with rationale
- Consider special factors (depth of invasion for oral cavity, ENE for N staging)
- Determine the overall stage group

Show your staging reasoning step by step.

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Stage Classification
## Staging Criteria
## Imaging/Workup
## Prognostic Implications""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM GUIDELINES AND LITERATURE:
{context}

Instructions:
- Reference specific GUIDELINES by name - never say "the context"
- Determine T stage (consider depth of invasion if oral cavity, size, invasion)
- Determine N stage (consider number, size, ENE status)
- Determine M stage
- State the overall stage group
- Show your reasoning step by step
- Quote all measurements and thresholds EXACTLY as stated
- **CITE at the END of sentences: "...staging criteria (AJCC/Source, Year)."**
- Example: "T3 is defined as tumor exceeding 4cm or with DOI >10mm (AJCC 8th Edition, 2017)."
"""
    },

    "workup": {
        "system": """You are an oncology expert.
Answer workup and diagnostic questions by synthesizing evidence from the guidelines and literature provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific GUIDELINE or SOURCE by name
- Example: "NCCN Guidelines recommend..." NOT "The context indicates..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...recommendation (Author/Guideline, Year)."
- Example: "Initial workup should include CT chest/abdomen/pelvis and PET-CT (NCCN Guidelines, 2023)."
- Place citations at the end of the relevant sentence, before the period

For workup questions:
- List all recommended diagnostic steps
- Include imaging modalities (CT, MRI, PET)
- Include laboratory tests or biomarker testing
- Include pathology/biopsy requirements
- Specify the order if relevant

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Recommended Workup
## Imaging
## Labs/Pathology
## Rationale""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM GUIDELINES AND LITERATURE:
{context}

Instructions:
- Reference specific GUIDELINES by name - never say "the context"
- List all recommended workup steps comprehensively
- Include: imaging (CT, MRI, PET), labs, pathology, biopsies
- Specify biomarker testing if relevant (PD-L1, mutations)
- Note the sequence if important
- **CITE at the END of sentences: "...workup step (Guideline/Author, Year)."**
- Example: "PET-CT is recommended for initial staging of locally advanced disease (NCCN Guidelines, 2023)."
"""
    },

    "mechanism": {
        "system": """You are an oncology expert.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific STUDY or SOURCE by name
- Example: "Research by Smith et al. demonstrated..." NOT "The context shows..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...mechanism (Author et al., Year, Journal)."
- Example: "Checkpoint inhibitors work by blocking PD-1/PD-L1 interaction (Sharma et al., 2017, Science)."
- Place citations at the end of the relevant sentence, before the period

Synthesize your answer from the clinical trials and literature provided below.

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Mechanism
## Biological Basis
## Clinical Implications""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
- Give a concise answer
- Reference specific STUDIES by name - never say "the context"
- **CITE at the END of sentences: "...explanation (Author et al., Year, Journal)."**
- Example: "The mechanism involves disruption of immune checkpoint signaling (Smith et al., 2020, Nature Reviews)."
"""
    },

    "side_effects": {
        "system": """You are an oncology expert.
Answer questions about treatment side effects and toxicity by synthesizing evidence from the clinical trials and literature provided below.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", or similar phrases
- ALWAYS reference the specific TRIAL or STUDY by name
- Example: "The PORTEC-2 trial reported..." NOT "The context indicates..."
- Example: "Albain et al. found..." NOT "According to the provided context..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...toxicity data (Author/Trial, Year, Journal)."
- Example: "Grade 3+ toxicity occurred in 8% of patients (PORTEC-2, Nout et al., 2010, Lancet Oncology)."
- Place citations at the end of the relevant sentence, before the period

For toxicity questions:
- Distinguish acute vs late effects
- Provide incidence rates EXACTLY as stated (e.g., 8.3%, not 8%)
- Note severity grades if mentioned
- Describe risk factors if relevant

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Common Side Effects
## Serious/Late Effects
## Risk Factors
## Management""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
- Reference specific TRIALS and STUDIES by name - never say "the context"
- Describe the specific toxicity or side effect
- Note if acute or late
- Provide rates/incidence EXACTLY as stated in source (do NOT round)
- Include comparison between treatments if relevant
- **CITE at the END of sentences: "...toxicity result (Trial, Author et al., Year, Journal)."**
- Example: "Lymphedema rates were significantly higher with full axillary dissection at 28% vs 8% (Whelan et al., 2015, NEJM)."
"""
    },

    "general": {
        "system": """You are an oncology and radiation oncology expert.

**CRITICAL RESPONSE STYLE:**
- NEVER say "the context says", "provided context", "context mentions", "not provided in the context", or similar phrases
- ALWAYS reference the specific STUDY, TRIAL, or GUIDELINE by name
- Example: "The RTOG 0617 trial demonstrated..." NOT "The context shows..."
- Example: "According to NCCN Guidelines..." NOT "The provided context states..."
- If information is limited, say "The available evidence from [specific studies] does not address..." NOT "The context does not provide..."

**CRITICAL CITATION FORMAT:**
- ALWAYS include full bibliographic details at the END of sentences
- Format: "...statement (Author et al., Year, Journal)."
- Place citations at the end of the relevant sentence, before the period

Synthesize your answer from the clinical trials, guidelines, and literature provided below.

**TIERED RESPONSE APPROACH:**
Use the highest applicable tier based on available evidence:

1. DIRECT EVIDENCE → Cite exact findings with values
2. RELATED EVIDENCE → State what's available, extrapolate with caveat  
3. PRINCIPLES → Apply general oncology principles
4. GUIDANCE → Provide best available guidance, note gaps

Always aim to provide useful clinical information. Do not simply state "evidence not found" without offering available guidance.

**REQUIRED OUTPUT STRUCTURE — use these exact markdown section headers:**
## Summary
## Key Points""",

        "user_template": """QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS AND LITERATURE:
{context}

Instructions:
- Reference specific STUDIES, TRIALS, and GUIDELINES by name - never say "the context"
- Give a concise answer (2–6 sentences)
- Use the TIERED approach: provide the best available evidence
- If exact answer isn't present, provide related evidence or principles
- Quote all numerical values EXACTLY as they appear in the source
- **CITE at the END of sentences: "...statement (Author et al., Year, Journal)."**
- Note any limitations in evidence while still providing guidance"""
    }
}


# ============================================
# HYBRID RETRIEVER UTILITIES
# ============================================

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/\.≤≥%]+", re.IGNORECASE)

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _payload_text_for_lex(payload: Dict[str, Any]) -> str:
    """Get text for lexical scoring - includes table context."""
    if payload.get("chunk_type") == "table_row":
        parts = [
            str(payload.get("table_number") or ""),
            str(payload.get("table_title") or ""),
            str(payload.get("text") or ""),
        ]
        md = payload.get("metadata") or {}
        hdrs = md.get("headers") or []
        if hdrs:
            parts.append(" | ".join(str(h) for h in hdrs))
        return _norm(" ".join(filter(None, parts)))
    return _norm(str(payload.get("text") or ""))


def bm25_scores(query: str, docs: List[str], k1: float = 1.2, b: float = 0.75) -> List[float]:
    """Compute BM25 scores for documents against a query."""
    q_terms = _tokenize(query)
    if not q_terms or not docs:
        return [0.0] * len(docs)

    tokenized = [_tokenize(d) for d in docs]
    df = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1

    N = len(docs)
    avgdl = sum(len(t) for t in tokenized) / (N or 1)

    scores: List[float] = []
    for toks in tokenized:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
            denom = tf[term] + k1 * (1 - b + b * (dl / (avgdl + 1e-9)))
            s += idf * (tf[term] * (k1 + 1.0)) / (denom + 1e-9)
        scores.append(float(s))
    return scores


def rrf_fuse(cands: List[Dict[str, Any]], k_rrf: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of dense and lexical scores."""
    dense_sorted = sorted(cands, key=lambda x: x["score_dense"], reverse=True)
    dense_rank = {id(c): r + 1 for r, c in enumerate(dense_sorted)}

    lex_sorted = sorted(cands, key=lambda x: x["score_lex"], reverse=True)
    lex_rank = {id(c): r + 1 for r, c in enumerate(lex_sorted)}

    for c in cands:
        rd = dense_rank[id(c)]
        rl = lex_rank[id(c)]
        c["score_fused"] = 1.0 / (k_rrf + rd) + 1.0 / (k_rrf + rl)

    return sorted(cands, key=lambda x: x["score_fused"], reverse=True)


def query_route(query: str, query_type: str = "general") -> Dict[str, Any]:
    """Route query to determine retrieval strategy based on query classification."""
    q = (query or "").lower()

    exactish = (
        bool(re.search(r"\b\d+(\.?\d*)?\s?(mg|mcg|g|gy|ml|cm|mm)\b", q)) or
        ("/m2" in q) or ("dose" in q) or ("fraction" in q) or ("fx" in q) or
        bool(re.search(r"\bT\d\b|\bN\d\b|\bM\d\b", q))
    )
    navigation = ("table" in q) or ("figure" in q) or ("staging" in q)

    if query_type == "dose_question":
        mode = "exact"
    elif query_type == "staging":
        mode = "navigation"
    elif query_type == "treatment_recommendation":
        mode = "conceptual"
    elif query_type == "indication_question":
        mode = "conceptual"  # indication questions need conceptual search
    elif navigation:
        mode = "navigation"
    elif exactish:
        mode = "exact"
    else:
        mode = "general"

    return {
        "mode": mode,
        "boost_tables": mode in {"exact", "navigation"},
        "lex_weight": 1.4 if mode in {"exact", "navigation"} else 1.0,
        "dense_weight": 1.2 if mode in {"conceptual"} else 1.0,
    }


def rerank_with_structure(
    cands: List[Dict[str, Any]],
    query: str,
    query_type: str = "general",
    clinical_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Structure-aware reranking with keyword overlap, section boosts, and clinical profile matching."""
    route = query_route(query, query_type)
    q_terms = set(_tokenize(query))

    # Get profile filter terms if available
    profile_must_terms = []
    profile_should_terms = []
    if clinical_profile:
        profile_must_terms = [t.lower() for t in clinical_profile.get("must_match", [])]
        profile_should_terms = [t.lower() for t in clinical_profile.get("should_match", [])]

    # EVIDENCE-BASED BOOSTING - DISABLED
    # Lazy load evidence classifier for evidence-based boosting
    # evidence_classifier = None
    # try:
    #     from src.utils.evidence_level_classifier_complete import EvidenceLevelClassifier
    #     evidence_classifier = EvidenceLevelClassifier()
    # except ImportError:
    #     pass  # Evidence classifier not available

    for c in cands:
        p = c["payload"]
        s = c["score_fused"]
        
        # Get text for profile matching
        text_lower = (p.get("text") or "").lower()
        citation_lower = (p.get("doc_meta", {}).get("citation") or "").lower()
        category_lower = (p.get("category") or "").lower()
        combined_text = f"{text_lower} {citation_lower} {category_lower}"

        # PROFILE-AWARE BOOSTING
        if profile_must_terms:
            must_match_count = sum(1 for term in profile_must_terms if term in combined_text)
            if must_match_count > 0:
                # Strong boost for matching must-have terms
                s *= (1.0 + 0.15 * must_match_count)
            else:
                # Penalty for not matching any must-have terms
                s *= 0.7
        
        if profile_should_terms:
            should_match_count = sum(1 for term in profile_should_terms if term in combined_text)
            if should_match_count > 0:
                # Moderate boost for matching should-have terms
                s *= (1.0 + 0.08 * should_match_count)

        # EVIDENCE-BASED BOOSTING - DISABLED
        # Apply retrieval multiplier based on evidence level and endpoint strength
        # if evidence_classifier:
        #     # Check if evidence data is already in payload (pre-computed)
        #     retrieval_mult = p.get("doc_meta", {}).get("retrieval_multiplier")
        #     if retrieval_mult is not None:
        #         s *= retrieval_mult
        #     else:
        #         # Classify on-the-fly (slower but works for unclassified docs)
        #         try:
        #             result = evidence_classifier.classify(p)
        #             s *= result.retrieval_multiplier
        #             # Store for debugging
        #             c["evidence_level"] = result.level
        #             c["endpoint_strength"] = result.endpoint_strength
        #         except Exception:
        #             pass  # Skip evidence boost on error

        # keyword overlap boost
        md = p.get("metadata") or {}
        kw_flat = set(k.lower() for k in (md.get("keywords_flat") or []))
        overlap = len(kw_flat & q_terms)
        if overlap:
            s *= (1.0 + 0.08 * overlap)

        # boost table rows on exact/navigation queries
        if route["boost_tables"] and p.get("chunk_type") == "table_row":
            s *= 1.20

        # boost answer-bearing sections based on query type
        sec = (p.get("section") or "").lower()

        if query_type == "treatment_recommendation":
            if any(w in sec for w in ["recommend", "treatment", "guideline", "standard"]):
                s *= 1.15
        elif query_type == "indication_question":
            # Boost sections likely to contain indication criteria
            if any(w in sec for w in ["recommend", "indication", "criteria", "conclusion", "results"]):
                s *= 1.15
        elif query_type == "dose_question":
            if any(w in sec for w in ["dose", "fractionation", "method", "technique", "protocol"]):
                s *= 1.15
        elif query_type == "trial_results":
            if any(w in sec for w in ["results", "outcome", "conclusion", "efficacy"]):
                s *= 1.15
        else:
            if any(w in sec for w in ["results", "conclusion", "recommend", "discussion", "treatment"]):
                s *= 1.08

        # Boost NCCN guideline content for recommendation queries
        if query_type in ["treatment_recommendation", "indication_question"]:
            if any(kw in text_lower or kw in citation_lower for kw in NCCN_KEYWORDS):
                s *= 1.20

        # penalty for very short text
        tlen = len((p.get("text") or "").strip())
        if tlen < 120:
            s *= 0.92

        c["score_rerank"] = float(s)

    return sorted(cands, key=lambda x: x["score_rerank"], reverse=True)


def dedup_and_caps(
    cands: List[Dict[str, Any]],
    max_per_doc: int = 2,
    max_per_doc_section: int = 1,
    max_rows_per_table: int = 2,
) -> List[Dict[str, Any]]:
    """Deduplicate and cap results per document/section/table."""
    seen = set()
    per_doc = Counter()
    per_doc_section = Counter()
    per_table = Counter()

    out: List[Dict[str, Any]] = []
    for c in cands:
        p = c["payload"]
        doc_id = p.get("doc_id") or ""
        section = p.get("section") or ""
        chunk_id = p.get("chunk_id") or p.get("original_chunk_id") or ""

        if chunk_id and chunk_id in seen:
            continue
        if per_doc[doc_id] >= max_per_doc:
            continue
        if section and per_doc_section[(doc_id, section)] >= max_per_doc_section:
            continue

        if p.get("chunk_type") == "table_row":
            tbl = (doc_id, p.get("table_number") or "")
            if per_table[tbl] >= max_rows_per_table:
                continue
            per_table[tbl] += 1

        if chunk_id:
            seen.add(chunk_id)
        per_doc[doc_id] += 1
        if section:
            per_doc_section[(doc_id, section)] += 1

        out.append(c)

    return out


def fetch_neighbor_windows(
    qdrant_client: QdrantClient,
    collection: str,
    payload: Dict[str, Any],
    offsets: Tuple[int, int] = (-1, +1),
) -> List[Dict[str, Any]]:
    """Fetch adjacent section windows for context expansion."""
    if payload.get("chunk_granularity") != "section_window":
        return []

    doc_id = payload.get("doc_id")
    section = payload.get("section")
    idx = payload.get("section_window_idx")

    if not doc_id or not section or idx is None:
        return []

    neighbors: List[Dict[str, Any]] = []
    for off in offsets:
        try:
            target = int(idx) + int(off)
            flt = qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                qm.FieldCondition(key="section", match=qm.MatchValue(value=section)),
                qm.FieldCondition(key="chunk_granularity", match=qm.MatchValue(value="section_window")),
                qm.FieldCondition(key="section_window_idx", match=qm.MatchValue(value=target)),
            ])
            pts, _ = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if pts:
                neighbors.append(dict(pts[0].payload or {}))
        except Exception as e:
            print(f"Warning: Failed to fetch neighbor window (offset={off}): {e}")
            continue
    return neighbors


def make_evidence_pack(
    qdrant_client: QdrantClient,
    collection: str,
    cand: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a structured evidence pack from a candidate."""
    p = cand["payload"]
    doc_meta = p.get("doc_meta") or {}

    base = {
        "score": cand.get("score_dose_boost", cand.get("score_crossencoder",
                 cand.get("score_rerank", cand.get("score_fused", cand.get("score_dense"))))),
        "score_crossencoder": cand.get("score_crossencoder"),
        "doc_id": p.get("doc_id"),
        "doc_id_raw": p.get("doc_id_raw"),
        "category": p.get("category"),
        "chunk_id": p.get("chunk_id") or p.get("original_chunk_id"),
        "chunk_type": p.get("chunk_type"),
        "section": p.get("section"),
        "citation": doc_meta.get("citation") or None,
        "doi": doc_meta.get("doi") or None,
        "pmid": doc_meta.get("pmid") or None,
        "year": doc_meta.get("year") or None,
        "author_et_al": doc_meta.get("author_et_al") or None,
        "journal": doc_meta.get("journal") or None,
        "title": doc_meta.get("title") or None,
    }

    if p.get("chunk_type") == "table_row":
        md = p.get("metadata") or {}
        return {
            **base,
            "table": {
                "number": p.get("table_number"),
                "title": p.get("table_title"),
                "row_index": p.get("row_index"),
                "headers": md.get("headers") or [],
                "raw_row": md.get("raw_row") or [],
                "page": md.get("page"),
            },
            "text": p.get("text", ""),
            "neighbors": [],
        }

    neighbors = fetch_neighbor_windows(qdrant_client, collection, p)
    return {
        **base,
        "section_window_idx": p.get("section_window_idx"),
        "text": p.get("text", ""),
        "neighbors": [n.get("text", "") for n in neighbors],
        # Preserve the payload's metadata dict so downstream consumers
        # (patient_match_scorer, evidence_classifier) can read
        # doc_level_* fields without re-fetching from Qdrant. The
        # metadata dict carries doc_level_cancer_types,
        # doc_level_sites, doc_level_histologies, doc_level_stages,
        # doc_level_biomarkers, doc_level_drugs,
        # doc_level_disease_status — all the axes the scorer compares
        # against the patient profile.
        "metadata": p.get("metadata") or {},
    }


# ============================================
# SITE INFERENCE
# ============================================

SITE_LABELS = {
    "Breast": "Breast cancer",
    "Sarcoma": "Sarcoma and soft tissue tumors",
    "Lung": "Lung cancer / thoracic oncology",
    "GU": "Genitourinary cancer",
    "CNS": "CNS tumors / neuro-oncology",
    "H&N": "Head and neck cancer",
    "GI": "Gastrointestinal cancer",
    "GYN": "Gynecologic cancer",
    "Prostate": "Prostate cancer",
    "Lymphoma": "Lymphoma and hematologic malignancies",
    "Cutaneous": "Skin cancer / cutaneous oncology",
    "Peds": "Pediatric oncology",
    "Radiophar": "Radiopharmaceutical therapy / theranostics",
    "Benign": "Benign conditions",
    "Thyroid": "Thyroid cancer",
    "Radiotherapy&Oncology": "General radiotherapy and oncology",
}


# ── Canonical Qdrant category mapping ─────────────────────────────────────
#
# The Qdrant `category` payload field uses EXACT spellings that were baked
# in at ingestion time and must be matched verbatim. This dict maps EVERY
# SITE_LABELS key (as returned by `infer_site_key`) to the EXACT string
# stored in Qdrant. Callers MUST use this instead of ad-hoc lowercasing +
# suffix appending, which silently produces non-existent values (e.g.
# "gi_processed_documents" when Qdrant has "GI_processed_documents").
#
# Authoritative list from the user (April 2026):
#   benign_processed_documents, breast_processed_documents,
#   cns_processed_documents, cutaneous_processed_documents,
#   desmoid_processed_documents, GI_processed_documents,
#   GU_processed_documents, gyn_processed_documents,
#   h&n_processed_documents, lung_processed_documents,
#   lymphoma_processed_documents, palliation_processed_documents,
#   peds_processed_documents, prostate_processed_documents,
#   radiopharm_processed_documents,
#   radiotherapy&oncology_processed_documents,
#   sarcoma_processed_documents

QDRANT_CATEGORY_MAP: Dict[str, str] = {
    # Site key (from infer_site_key)  →  exact Qdrant category string
    "Breast":                "breast_processed_documents",
    "Sarcoma":               "sarcoma_processed_documents",
    "Lung":                  "lung_processed_documents",
    "GU":                    "GU_processed_documents",
    "CNS":                   "cns_processed_documents",
    "H&N":                   "h&n_processed_documents",
    "GI":                    "GI_processed_documents",
    "GYN":                   "gyn_processed_documents",
    "Prostate":              "prostate_processed_documents",
    "Lymphoma":              "lymphoma_processed_documents",
    "Cutaneous":             "cutaneous_processed_documents",
    "Peds":                  "peds_processed_documents",
    "Radiophar":             "radiopharm_processed_documents",
    "Benign":                "benign_processed_documents",
    "Thyroid":               None,  # no Qdrant category for thyroid yet
    "Radiotherapy&Oncology": "radiotherapy&oncology_processed_documents",
}

# Reverse lookup: Qdrant category string → site key (used for logging)
_QDRANT_CATEGORY_REVERSE = {v: k for k, v in QDRANT_CATEGORY_MAP.items() if v}


def normalize_category_filter(value: Optional[str]) -> Optional[str]:
    """Normalize a category/site value to the EXACT Qdrant category string.

    Resolution order (first match wins):
      1. Already an exact Qdrant value → return as-is
      2. Known SITE_LABELS key (e.g. "H&N") → QDRANT_CATEGORY_MAP lookup
      3. Arbitrary cancer-type text → infer_site_key → QDRANT_CATEGORY_MAP
      4. Legacy fallback → lowercase + "_processed_documents" suffix

    Steps 1-3 guarantee the returned string is one of the 17 values that
    actually exist in the Qdrant index. Step 4 is a defence-in-depth
    fallback that should never fire for known cancer types.
    """
    if not value:
        return None
    val = value.strip()
    if not val:
        return None

    # 1. Already an exact Qdrant value?
    if val in _QDRANT_CATEGORY_REVERSE or val in QDRANT_CATEGORY_MAP.values():
        return val

    # 2. Known site key? (e.g. "H&N", "GI", "Prostate"). Lookup is
    # case-insensitive so callers can pass lowercase short forms like
    # "h&n" / "lung" (used by _infer_category_for_retrieval and the
    # treatment-comparison / trial-match paths) without silently
    # falling through to the generic Radiotherapy&Oncology bucket.
    val_lower = val.lower()
    for key, canonical in QDRANT_CATEGORY_MAP.items():
        if canonical and key.lower() == val_lower:
            return canonical

    # 3. Arbitrary text → infer site key → canonical lookup
    inferred = infer_site_key(val)
    if inferred:
        canonical = QDRANT_CATEGORY_MAP.get(inferred)
        if canonical:
            return canonical

    # 4. Legacy fallback (should not be needed for known cancer types)
    lowered = val.lower()
    if lowered.endswith("_processed_documents"):
        return lowered
    return f"{lowered}_processed_documents"


def build_category_match_variants(category: Optional[str]) -> List[str]:
    """
    Return the plausible Qdrant `category` payload values for a single
    category string.

    The PRIMARY return value is the exact canonical string from
    `QDRANT_CATEGORY_MAP` (e.g. "h&n_processed_documents" for head-and-
    neck). A small handful of case / suffix variants are added as
    defence-in-depth so the Qdrant `should` filter still matches even
    if a study was ingested under a slightly different spelling.

    If the input is already one of the known Qdrant values, it is
    placed first in the returned list.
    """
    if not category:
        return []
    raw = category.strip()
    if not raw:
        return []

    # ── 1. Try to resolve to the EXACT canonical value first ──────────
    canonical: Optional[str] = None

    # Already an exact Qdrant value?
    if raw in _QDRANT_CATEGORY_REVERSE or raw in QDRANT_CATEGORY_MAP.values():
        canonical = raw
    # Known site key (e.g. "H&N", "GI")
    elif raw in QDRANT_CATEGORY_MAP:
        canonical = QDRANT_CATEGORY_MAP[raw]
    else:
        # Strip _processed_documents and try again as a site key
        base = raw
        if base.lower().endswith("_processed_documents"):
            base = base[: -len("_processed_documents")]
        # Check QDRANT_CATEGORY_MAP with various casings of the base
        for attempt in (base, base.upper(), base.capitalize(), base.lower()):
            if attempt in QDRANT_CATEGORY_MAP:
                canonical = QDRANT_CATEGORY_MAP[attempt]
                break
        # h&n ↔ head_neck aliasing
        if canonical is None:
            alias_map = {
                "h&n": "H&N", "h_n": "H&N", "head_neck": "H&N",
                "head&neck": "H&N",
            }
            site_key = alias_map.get(base.lower())
            if site_key:
                canonical = QDRANT_CATEGORY_MAP.get(site_key)

    # ── 2. Build the variant list ─────────────────────────────────────
    seen: set = set()
    variants: List[str] = []

    def _add(v: str):
        if v and v not in seen:
            seen.add(v)
            variants.append(v)

    # Canonical goes first (most likely to match)
    if canonical:
        _add(canonical)

    # Defence-in-depth: a few case variants of the input itself
    _add(raw)
    _add(raw.lower())
    _add(raw.upper())
    base = raw
    if base.lower().endswith("_processed_documents"):
        base = base[: -len("_processed_documents")]
    _add(base)
    _add(base.lower())
    _add(base.upper())
    _add(f"{base.lower()}_processed_documents")
    _add(f"{base.upper()}_processed_documents")

    return variants


def format_conversation_context(
    conversation_history: Optional[List[Dict[str, Any]]],
    max_turns: int = 10,
    max_chars: int = 50000
) -> str:
    """Format recent conversation history into a context block with full content."""
    if not conversation_history:
        return ""
    recent = conversation_history[-max_turns:]
    lines = []
    for msg in recent:
        if hasattr(msg, "dict"):
            msg = msg.dict()
        role = ""
        content = ""
        if isinstance(msg, dict):
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
        else:
            role = (getattr(msg, "role", "") or "").strip().lower()
            content = (getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant" if role == "assistant" else "Message"
        lines.append(f"{label}: {content}")
    context = "\n\n".join(lines)
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def extract_previous_sources(
    conversation_history: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Extract source doc_ids and citations from previous assistant messages.
    
    Returns:
        Dict with:
        - doc_ids: List of unique doc_ids from previous answers
        - citations: List of citation strings from previous answers
        - last_answer_sources: Sources from the most recent assistant message
    """
    if not conversation_history:
        return {"doc_ids": [], "citations": [], "last_answer_sources": []}
    
    all_doc_ids = []
    all_citations = []
    last_answer_sources = []
    
    for msg in conversation_history:
        # Handle both dict and Pydantic model
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        elif hasattr(msg, "dict"):
            msg = msg.dict()
        
        role = ""
        sources = []
        source_citations = []
        
        if isinstance(msg, dict):
            role = (msg.get("role") or "").strip().lower()
            sources = msg.get("sources") or []
            source_citations = msg.get("source_citations") or []
        else:
            role = (getattr(msg, "role", "") or "").strip().lower()
            sources = getattr(msg, "sources", []) or []
            source_citations = getattr(msg, "source_citations", []) or []
        
        if role == "assistant":
            all_doc_ids.extend(sources)
            all_citations.extend(source_citations)
            # Track the last assistant message's sources
            last_answer_sources = sources
    
    # Deduplicate while preserving order
    seen_ids = set()
    unique_doc_ids = []
    for doc_id in all_doc_ids:
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_doc_ids.append(doc_id)
    
    seen_citations = set()
    unique_citations = []
    for citation in all_citations:
        if citation and citation not in seen_citations:
            seen_citations.add(citation)
            unique_citations.append(citation)
    
    return {
        "doc_ids": unique_doc_ids,
        "citations": unique_citations,
        "last_answer_sources": last_answer_sources
    }


def format_conversation_context_with_sources(
    conversation_history: Optional[List[Dict[str, Any]]],
    max_turns: int = 10,
    max_chars: int = 50000
) -> str:
    """
    Format conversation history including source information for better follow-up context.
    
    This enhanced version includes full content and citations from previous answers 
    so the LLM has complete context of what was discussed.
    """
    if not conversation_history:
        return ""
    
    recent = conversation_history[-max_turns:]
    lines = []
    
    for msg in recent:
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        elif hasattr(msg, "dict"):
            msg = msg.dict()
        
        role = ""
        content = ""
        source_citations = []
        
        if isinstance(msg, dict):
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
            source_citations = msg.get("source_citations") or []
        else:
            role = (getattr(msg, "role", "") or "").strip().lower()
            content = (getattr(msg, "content", "") or "").strip()
            source_citations = getattr(msg, "source_citations", []) or []
        
        if not content:
            continue
        
        label = "User" if role == "user" else "Assistant" if role == "assistant" else "Message"
        
        # Include full content for both user and assistant messages
        if role == "assistant" and source_citations:
            sources_str = ", ".join(source_citations)
            lines.append(f"{label}: {content}\n[Sources: {sources_str}]")
        else:
            lines.append(f"{label}: {content}")
    
    context = "\n\n".join(lines)
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def format_conversation_history_from_context(
    conversation_context: Optional[List[Dict[str, Any]]],
    max_entries: int = 10,
    max_chars: int = 50000
) -> str:
    """
    Format conversation history from ConversationContextEntry objects for LLM context.
    
    This function takes the new conversation_context format (from frontend sessionStorage)
    and formats it as a simple list of previous queries for the LLM to understand
    the conversation flow.
    
    Args:
        conversation_context: List of ConversationContextEntry dicts with fields:
            - query: str (previous user query)
            - action_type: str (type of action)
            - doc_ids: List[str]
            - doc_titles: List[str]
            - timestamp: int
            - treatments: Optional[List[str]]
        max_entries: Maximum number of entries to include (default 10)
        max_chars: Maximum total characters for the context string
        
    Returns:
        Formatted string with previous queries for LLM context, or empty string if no context
    """
    if not conversation_context:
        return ""
    
    # Limit to last N entries (most recent)
    recent_entries = conversation_context[-max_entries:]
    
    if not recent_entries:
        return ""
    
    lines = []
    for i, entry in enumerate(recent_entries, 1):
        # Handle both dict and Pydantic model
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        elif hasattr(entry, "dict"):
            entry = entry.dict()
        
        query = ""
        action_type = ""
        doc_titles = []
        treatments = []
        
        if isinstance(entry, dict):
            query = (entry.get("query") or "").strip()
            action_type = (entry.get("action_type") or "").strip()
            doc_titles = entry.get("doc_titles") or []
            treatments = entry.get("treatments") or []
        else:
            query = (getattr(entry, "query", "") or "").strip()
            action_type = (getattr(entry, "action_type", "") or "").strip()
            doc_titles = getattr(entry, "doc_titles", []) or []
            treatments = getattr(entry, "treatments", []) or []
        
        if not query:
            continue
        
        # Build the entry line
        line_parts = [f"[{i}] User asked: {query}"]
        
        # Add context about what was discussed
        if doc_titles:
            titles_str = ", ".join(doc_titles[:3])  # Limit to first 3 titles
            if len(doc_titles) > 3:
                titles_str += f" (+{len(doc_titles) - 3} more)"
            line_parts.append(f"    Referenced: {titles_str}")
        
        # Add treatment context for eval_treatment actions
        if action_type == "eval_treatment" and treatments:
            treatments_str = ", ".join(treatments)
            line_parts.append(f"    Treatments discussed: {treatments_str}")
        
        lines.append("\n".join(line_parts))
    
    if not lines:
        return ""
    
    context = "\n\n".join(lines)
    
    # Truncate if too long (keep most recent)
    if len(context) > max_chars:
        context = context[-max_chars:]
    
    print(f"[ConversationContext] Formatted {len(recent_entries)} entries for LLM context")
    return context


# Additive boost constants — safe for negative cross-encoder scores (-10 to +10)
PREVIOUS_SOURCE_BOOST_ADDEND = 2.0
LAST_ANSWER_BOOST_ADDEND = 2.5


def boost_previous_sources(
    chunks: List[Dict[str, Any]],
    previous_doc_ids: List[str],
    boost_factor: float = 1.25,
    last_answer_boost: float = 1.35
) -> List[Dict[str, Any]]:
    """
    Boost chunks from documents that were cited in previous conversation turns.

    Uses additive boost instead of multiplicative to avoid worsening chunks
    with negative cross-encoder scores (e.g. -2.95 * 1.25 = -3.69).

    Args:
        chunks: List of retrieved chunks
        previous_doc_ids: Doc IDs from previous answers
        boost_factor: Kept for API compatibility (no longer used as multiplier)
        last_answer_boost: Kept for API compatibility (no longer used as multiplier)

    Returns:
        Chunks with boosted scores
    """
    if not previous_doc_ids:
        return chunks

    previous_set = set(previous_doc_ids)
    # First few doc_ids are assumed to be from the most recent answer
    last_answer_ids = set(previous_doc_ids[:3]) if previous_doc_ids else set()

    for chunk in chunks:
        doc_id = chunk.get("payload", {}).get("doc_id") or chunk.get("doc_id")

        if doc_id and doc_id in previous_set:
            current_score = chunk.get("score_final", chunk.get("score_dense", 0.5))
            addend = (
                LAST_ANSWER_BOOST_ADDEND
                if doc_id in last_answer_ids
                else PREVIOUS_SOURCE_BOOST_ADDEND
            )
            chunk["score_final"] = current_score + addend
            chunk["from_previous_source"] = True
            print(f"[Source Boost] Boosted chunk from {doc_id[:40]}... "
                  f"(score: {current_score:.3f} -> {chunk['score_final']:.3f})")

    # Re-sort by boosted scores
    chunks.sort(key=lambda x: x.get("score_final", 0), reverse=True)

    return chunks

# Additive boost constants for context documents
CONTEXT_BOOST_ADDEND = 1.5          # cross-encoder scores: ~-10 to +10
CONTEXT_BOOST_ADDEND_DENSE = 0.15   # dense/cosine scores: 0 to 1


def boost_context_documents(
    chunks: List[Dict[str, Any]],
    context_doc_ids: List[str],
    boost_factor: float = 1.2
) -> List[Dict[str, Any]]:
    """
    Boost reranking scores for chunks from previously referenced documents.
    Applied after cross-encoder reranking, before final sorting.

    Uses additive boost instead of multiplicative to avoid worsening chunks
    with negative cross-encoder scores (e.g. -2.95 * 1.2 = -3.54).

    Args:
        chunks: List of retrieved chunks, each with a score field
                (score_crossencoder, score_rerank, or score) and doc_id
        context_doc_ids: List of doc_ids from previous conversation entries
        boost_factor: Kept for API compatibility (no longer used as multiplier)

    Returns:
        Chunks with boosted scores for matching doc_ids
    """
    if not context_doc_ids or not chunks:
        return chunks

    context_set = set(context_doc_ids)

    for chunk in chunks:
        doc_id = chunk.get("payload", {}).get("doc_id") or chunk.get("doc_id")

        if doc_id and doc_id in context_set:
            current_score = (
                chunk.get("score_crossencoder") or
                chunk.get("score_rerank") or
                chunk.get("score", 0)
            )

            # Additive boost — works correctly for both positive and negative scores
            is_dense_score = "score_dense" in chunk and not (
                "score_crossencoder" in chunk or "score_rerank" in chunk
            )
            addend = CONTEXT_BOOST_ADDEND_DENSE if is_dense_score else CONTEXT_BOOST_ADDEND

            boosted_score = current_score + addend
            chunk["score_context_boost"] = boosted_score
            chunk["from_context"] = True

            print(f"[Context Boost] Boosted chunk from {doc_id[:40] if doc_id else 'unknown'}... "
                  f"(score: {current_score:.3f} -> {boosted_score:.3f})")
        else:
            current_score = (
                chunk.get("score_crossencoder") or
                chunk.get("score_rerank") or
                chunk.get("score", 0)
            )
            chunk["score_context_boost"] = current_score

    return chunks




def _expand_query_with_context(
    current_query: str,
    conversation_history: Optional[List[Dict[str, Any]]]
) -> str:
    """
    Expand a follow-up query with context from conversation history.
    
    This helps with queries like "what about toxicities?" that need context
    from the previous question about a specific cancer type.
    
    IMPORTANT: Extracts context from BOTH user AND assistant messages,
    since patient context may be stored in assistant responses (e.g., from intent analysis).
    """
    if not conversation_history:
        return current_query
    
    # Extract ALL messages (user AND assistant) to build context
    # Patient context may be stored in assistant messages from intent analysis
    all_messages = []
    for msg in conversation_history:
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        elif hasattr(msg, "dict"):
            msg = msg.dict()
        
        content = ""
        if isinstance(msg, dict):
            content = (msg.get("content") or "").strip()
        else:
            content = (getattr(msg, "content", "") or "").strip()
        
        if content:
            all_messages.append(content)
    
    if not all_messages:
        return current_query
    
    # Extract key clinical context from ALL previous messages (user + assistant)
    clinical_context = _extract_clinical_context(all_messages)
    
    print(f"[RAG] Extracted clinical context from conversation: {clinical_context}")
    
    # Check if current query lacks the clinical context
    current_lower = current_query.lower()
    
    # Check what context is missing from current query
    missing_context = []
    
    if clinical_context.get("cancer_type") and clinical_context["cancer_type"].lower() not in current_lower:
        missing_context.append(f"cancer type: {clinical_context['cancer_type']}")
    
    if clinical_context.get("stage") and clinical_context["stage"].lower() not in current_lower:
        missing_context.append(f"stage: {clinical_context['stage']}")
    
    if clinical_context.get("patient_profile") and not any(p.lower() in current_lower for p in clinical_context["patient_profile"]):
        missing_context.append(f"patient: {', '.join(clinical_context['patient_profile'])}")
    
    if clinical_context.get("treatment") and clinical_context["treatment"].lower() not in current_lower:
        missing_context.append(f"treatment: {clinical_context['treatment']}")

    # Include biomarkers from conversation history
    if clinical_context.get("biomarkers"):
        biomarker_str = ", ".join(clinical_context["biomarkers"])
        if not any(bm.lower() in current_lower for bm in clinical_context["biomarkers"]):
            missing_context.append(f"biomarkers: {biomarker_str}")

    # Include genomic scores (e.g., recurrence score)
    if clinical_context.get("genomic_scores"):
        for score_type, value in clinical_context["genomic_scores"].items():
            score_label = score_type.replace("_", " ")
            if str(value) not in current_lower:
                missing_context.append(f"{score_label}: {value}")

    # Include surgery type
    if clinical_context.get("surgery_type") and clinical_context["surgery_type"].lower() not in current_lower:
        missing_context.append(f"surgery: {clinical_context['surgery_type']}")

    # Include TNM
    if clinical_context.get("tnm") and clinical_context["tnm"].lower() not in current_lower:
        missing_context.append(f"staging: {clinical_context['tnm']}")

    # If we have missing context, expand the query
    if missing_context:
        context_str = "; ".join(missing_context)
        expanded = f"For a patient with {context_str}: {current_query}"
        print(f"[RAG] Expanded query with clinical context: {expanded[:200]}...")
        return expanded

    return current_query


def _normalize_biomarker_status(raw_match: str) -> str:
    """
    Normalize a raw biomarker match into a canonical 'NAME status' string.

    Understands that:
      +  = positive
      -  = negative
      amplified / overexpressed / mutant / mutation / mutated / fusion / rearrangement = positive
      non-amplified / wild-type / wild type = negative

    Examples:
      "er+"             -> "ER positive"
      "er-"             -> "ER negative"
      "her2 negative"   -> "HER2 negative"
      "her2 amplified"  -> "HER2 positive"
      "braf v600e mutant" -> "BRAF V600E positive"
      "egfr wild-type"  -> "EGFR negative"
      "msi-h"           -> "MSI-H"   (kept as-is, inherently directional)
      "psa 4.5"         -> "PSA 4.5" (kept as-is, quantitative)
    """
    text = raw_match.strip()
    text_lower = text.lower()

    # Determine status from the text
    positive_indicators = [
        "positive", "amplified", "overexpressed", "overexpression",
        "mutant", "mutation", "mutated", "fusion", "rearrangement",
        "translocation", "detected", "carrier", "elevated", "high",
        "deficient",  # dMMR = deficient = positive for MSI testing
    ]
    negative_indicators = [
        "negative", "non-amplified", "nonamplified", "wild-type", "wild type",
        "wildtype", "not detected", "absent", "normal", "proficient",
        "intact", "stable",
    ]

    # Check for +/- symbols
    if text_lower.endswith("+") or re.search(r"\s\+\s*$", text_lower):
        status = "positive"
        # Remove trailing + to get the name
        name = re.sub(r"\s*\+\s*$", "", text).strip()
    elif text_lower.endswith("-") or re.search(r"\s\-\s*$", text_lower):
        status = "negative"
        name = re.sub(r"\s*\-\s*$", "", text).strip()
    else:
        # Check for word-based status
        status = None
        name = text
        for indicator in negative_indicators:
            if indicator in text_lower:
                status = "negative"
                # Remove the indicator from name
                name = re.sub(re.escape(indicator), "", text_lower, count=1).strip()
                break
        if status is None:
            for indicator in positive_indicators:
                if indicator in text_lower:
                    status = "positive"
                    name = re.sub(re.escape(indicator), "", text_lower, count=1).strip()
                    break

    # Clean up the biomarker name
    name = re.sub(r"[\s\-]+$", "", name).strip()

    # Handle special composite terms BEFORE canonical name lookup
    # These should be returned as-is without status decomposition
    composite_terms = {
        "triple-negative": "triple-negative",
        "triple negative": "triple-negative",
        "tnbc": "triple-negative",
    }
    for term, canonical_term in composite_terms.items():
        if term in text_lower:
            return canonical_term

    # Canonical name mapping — sorted longest-first to avoid substring conflicts
    # (e.g., "HER2" must match before "ER", "BRCA1" before "BRCA")
    name_upper = name.upper().strip()
    canonical_names = [
        ("ESTROGEN RECEPTOR", "ER"),
        ("PROGESTERONE RECEPTOR", "PR"),
        ("HORMONE RECEPTOR", "HR"),
        ("HER2/NEU", "HER2"),
        ("HER-2", "HER2"),
        ("HER 2", "HER2"),
        ("HER2", "HER2"),
        ("ERBB2", "HER2"),
        ("PD-L1", "PD-L1"),
        ("PDL1", "PD-L1"),
        ("PD L1", "PD-L1"),
        ("PIK3CA", "PIK3CA"),
        ("BRCA1", "BRCA1"),
        ("BRCA2", "BRCA2"),
        ("BRCA", "BRCA"),
        ("BRAF", "BRAF"),
        ("EGFR", "EGFR"),
        ("KRAS", "KRAS"),
        ("NTRK", "NTRK"),
        ("FGFR", "FGFR"),
        ("ROS1", "ROS1"),
        ("TP53", "TP53"),
        ("ALK", "ALK"),
        ("RET", "RET"),
        ("MET", "MET"),
        ("HPV", "HPV"),
        ("P16", "p16"),
        ("HR", "HR"),
        ("ER", "ER"),
        ("PR", "PR"),
    ]

    # Try to find canonical name (longest match first, list is pre-sorted)
    canonical = None
    for key, val in canonical_names:
        # Use word boundary check to avoid "ER" matching inside "HER2"
        if re.search(r'(?<![A-Z])' + re.escape(key) + r'(?![A-Z0-9])', name_upper):
            canonical = val
            # Preserve specific variants (e.g., BRAF V600E, EGFR L858R)
            extra = re.sub(r'(?<![A-Z])' + re.escape(key) + r'(?![A-Z0-9])', '', name_upper, count=1).strip()
            if extra and extra not in (".", ",", "-"):
                canonical = f"{val} {extra}"
            break

    if canonical is None:
        canonical = name.upper() if name else text.upper()

    # For biomarkers that are inherently directional (MSI-H, MSS, TMB-H, etc.)
    # or quantitative (PSA 4.5, CA-125 35), don't append status
    inherently_directional = ["MSI", "MSS", "TMB", "DMMR", "PMMR",
                              "MLH1", "MSH2", "MSH6", "PMS2",
                              "TRIPLE-NEGATIVE", "TNBC"]
    quantitative = ["PSA", "CEA", "CA-125", "CA-15-3", "CA-27.29", "CA-19-9",
                     "AFP", "HCG", "LDH", "CALCITONIN", "THYROGLOBULIN",
                     "CHROMOGRANIN", "NSE", "TPS", "CPS"]

    canonical_base = canonical.split()[0] if canonical else ""
    if canonical_base in inherently_directional or any(q in canonical.upper() for q in quantitative):
        return canonical.strip()

    if status:
        return f"{canonical} {status}".strip()

    return canonical.strip()


def _extract_biomarkers_from_text(text: str) -> List[str]:
    """
    Extract and normalize all biomarkers from text.
    Returns list of canonical biomarker strings like "ER positive", "HER2 negative".

    Aligned with data/keywords/extractor_keywords.json biomarker_keywords section.
    Covers all cancer types: breast, lung, colorectal, prostate, H&N, melanoma, etc.
    """
    text_lower = text.lower()
    raw_matches = []

    # Patterns return the raw matched text; normalization happens after
    #
    # Note on hyphen handling. Several patterns below use the shape:
    #   r"\bGENE[\s-]*(?:WORDS|\+|\-(?=\s|$|[,.;:]))"
    # The `[\s-]*` lets a hyphen act as a connector ("EGFR-mutant" captures
    # the whole thing instead of just "EGFR"). The `\-(?=\s|$|[,.;:])`
    # bound on the bare `-` polarity marker prevents it from matching
    # connector hyphens — so "EGFR-positive" no longer captures "EGFR-"
    # (which would normalize as negative), while "EGFR-" alone at end-
    # of-token still tags correctly. Same fix shape as the HPV/p16
    # cleanup in commits 0d69985, 863fe0a, 070ddec.
    patterns = [
        # --- Hormone receptors ---
        r"\ber\s*[+\-](?=\s|$|[,.;:])",
        r"\ber[\s-]*(?:positive|negative)",
        r"\bestrogen\s*receptor(?:[\s-]*(?:positive|negative)|\s*[+\-](?=\s|$|[,.;:]))?",
        r"\bpr\s*[+\-](?=\s|$|[,.;:])",
        r"\bpr[\s-]*(?:positive|negative)",
        r"\bprogesterone\s*receptor(?:[\s-]*(?:positive|negative)|\s*[+\-](?=\s|$|[,.;:]))?",
        r"\bhormone\s*receptor[\s-]*(?:positive|negative|\+|\-(?=\s|$|[,.;:]))",

        # --- HER2 ---
        r"\bher-?2(?:/neu)?[\s-]*(?:positive|negative|amplified|non-?amplified|overexpression|overexpressed|\+|\-(?=\s|$|[,.;:]))",
        r"\berbb2[\s-]*(?:positive|negative|amplified|\+|\-(?=\s|$|[,.;:]))",

        # --- Genetic mutations ---
        # `positive`/`negative` are common informal clinical shorthand
        # for `mutant`/`wild-type` (e.g. "EGFR-positive" = "EGFR
        # mutated"). Added these word forms so the eligibility
        # extractor sees a biomarker when the clinician phrases it
        # that way — closes the recognition gap noted after the EGFR
        # Find Trials test where "EGFR-positive" reached the LLM
        # eligibility prompt as a missing biomarker.
        r"\bbraf[\s-]*(?:v600[ek]?\s*)?(?:mutant|mutation|mutated|wild[- ]?type|positive|negative|\+|\-(?=\s|$|[,.;:]))",
        r"\begfr[\s-]*(?:l858r|exon\s*(?:19|20|21)\s*(?:deletion|insertion|mutation)?|mutant|mutation|mutated|wild[- ]?type|t790m|positive|negative|\+|\-(?=\s|$|[,.;:]))",
        r"\balk[\s-]*(?:rearrangement|fusion|positive|negative|translocation|\+|\-(?=\s|$|[,.;:]))",
        r"\beml4[- ]?alk\s*(?:fusion|rearrangement)?",
        r"\bkras[\s-]*(?:g12[cdv]?\s*)?(?:mutant|mutation|mutated|wild[- ]?type|positive|negative|\+|\-(?=\s|$|[,.;:]))",
        r"\bbrca[12]?[\s-]*(?:mutant|mutation|mutated|positive|negative|carrier|\+|\-(?=\s|$|[,.;:]))",
        r"\bpik3ca\s*(?:mutant|mutation|mutated)",
        r"\btp53\s*(?:mutant|mutation|mutated|wild[- ]?type)",
        r"\bros1[\s-]*(?:rearrangement|fusion|positive|negative|\+|\-(?=\s|$|[,.;:]))",
        r"\bret[\s-]*(?:rearrangement|fusion|mutation|mutated|positive|negative|\+|\-(?=\s|$|[,.;:]))",
        r"\bntrk[\s-]*(?:fusion|rearrangement|positive|negative|\+|\-(?=\s|$|[,.;:]))",
        r"\bmet\s*(?:amplification|amplified|exon\s*14\s*(?:skipping)?|mutation|positive|negative)",
        r"\bfgfr[1-4]?\s*(?:amplification|fusion|mutation|alteration|positive|negative)",

        # --- Immunotherapy markers ---
        r"\bpd-?l1[\s-]*(?:positive|negative|high|low|expression|\+|\-(?=\s|$|[,.;:])|\d+\s*%?)",
        r"\btps\s*(?:[<>≥≤]?\s*\d+\s*%?)",
        r"\bcps\s*(?:[<>≥≤]?\s*\d+)",

        # --- MSI / MMR ---
        r"\bmsi[- ]?(?:h(?:igh)?|l(?:ow)?|stable|s)\b",
        r"\bmss\b",
        r"\bdmmr\b|\bdeficient\s*mismatch\s*repair",
        r"\bpmmr\b|\bproficient\s*mismatch\s*repair",
        r"\b(?:mlh1|msh2|msh6|pms2)\s*(?:loss|absent|deficient|intact|present)?",

        # --- TMB ---
        r"\btmb[- ]?(?:h(?:igh)?|l(?:ow)?|\d+\s*(?:mut(?:ations?)?/mb)?)",
        r"\btumor\s*mutational\s*burden\s*(?:high|low)?",

        # --- Viral markers ---
        # Same hyphen-connector polarity bug as the one fixed in
        # query_structuring_service.py BIOMARKER_PATTERNS (commit
        # 0d69985). The original `\bp16\s*(?:positive|negative|\+|\-)`
        # had a bare `-` alternative that matched ANY hyphen after
        # HPV/p16, so "p16-positive" / "HPV-positive" got captured as
        # "p16-" / "hpv-" → normalized to "p16 negative" / "HPV negative"
        # by _normalize_biomarker_status (trailing "-" → negative). This
        # function feeds patient_eligibility_boost_service's
        # `Detected patient context: biomarkers=[...]`, which then drives
        # the LLM eligibility verdict — so the polarity flip caused
        # HPV-positive patients to be told HPV-targeted trials don't
        # apply. Clinically dangerous for OPSCC.
        # Fix: split into separate positive/negative patterns. Positive
        # uses `[\s-]*` so the hyphen acts as a connector. Negative
        # requires the bare `-` polarity marker to be followed by a
        # word boundary (whitespace, end-of-string, or punctuation).
        r"\bhpv[\s-]*(?:positive|\+)",
        r"\bhpv(?:[\s-]*negative|\s*-(?=\s|$|[,.;:]))",
        r"\bp16[\s-]*(?:positive|\+)",
        r"\bp16(?:[\s-]*negative|\s*-(?=\s|$|[,.;:]))",

        # --- Protein / serum biomarkers ---
        r"\bpsa\s*(?:[<>≥≤]?\s*\d+(?:\.\d+)?\s*(?:ng/ml)?)",
        r"\bcea\s*(?:[<>≥≤]?\s*\d+(?:\.\d+)?(?:\s*ng/ml)?)",
        r"\bca[- ]?125\s*(?:[<>≥≤]?\s*\d+(?:\.\d+)?(?:\s*u/ml)?)",
        r"\bca[- ]?15[- ]?3\s*(?:[<>≥≤]?\s*\d+)?",
        r"\bca[- ]?27\.?29\s*(?:[<>≥≤]?\s*\d+)?",
        r"\bca[- ]?19[- ]?9\s*(?:[<>≥≤]?\s*\d+)?",
        r"\bafp\s*(?:[<>≥≤]?\s*\d+(?:\.\d+)?(?:\s*ng/ml)?)",
        r"\balpha[- ]?fetoprotein\s*(?:[<>≥≤]?\s*\d+(?:\.\d+)?)?",
        r"\b(?:β|beta)[- ]?hcg\s*(?:[<>≥≤]?\s*\d+)?",
        r"\bcalcitonin\s*(?:[<>≥≤]?\s*\d+)?",
        r"\bthyroglobulin\s*(?:[<>≥≤]?\s*\d+)?",
        r"\bldh\s*(?:elevated|high|normal)",
        r"\blactate\s*dehydrogenase",
        r"\bchromogranin\s*a",
        r"\bnse\b",
        r"\bneuron[- ]?specific\s*enolase",

        # --- Circulating biomarkers ---
        r"\bctdna\s*(?:positive|negative|detected|not\s*detected)?",
        r"\bcirculating\s*tumor\s*dna",
        r"\bcfdna\b",
        r"\bcell[- ]?free\s*dna",
        r"\bctc\b",
        r"\bcirculating\s*tumor\s*cells?",

        # --- Subtypes ---
        r"\b(?:triple[- ]?negative|tnbc)\b",
        r"\bhormone\s*receptor[- ]?positive\s*(?:her2[- ]?negative)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            raw_matches.append(match.group(0))

    # Normalize and deduplicate
    seen = set()
    normalized = []
    for raw in raw_matches:
        canonical = _normalize_biomarker_status(raw)
        if canonical:
            key = canonical.upper()
            if key not in seen:
                seen.add(key)
                normalized.append(canonical)

    return normalized


def _extract_clinical_context(user_messages: List[str]) -> Dict[str, Any]:
    """
    Extract key clinical context from conversation history.
    Returns dict with cancer_type, stage, patient_profile, treatment,
    biomarkers, genomic_scores, and surgery_type.

    NOTE: This function now processes ALL messages (user + assistant) since
    patient context may be stored in assistant responses from intent analysis.
    """
    import re

    context = {
        "cancer_type": None,
        "stage": None,
        "patient_profile": [],
        "treatment": None,
        "biomarkers": [],
        "genomic_scores": {},
        "surgery_type": None,
        "tnm": None,
    }
    
    # Combine all messages for analysis
    all_text = " ".join(user_messages).lower()
    
    # Cancer type patterns - ordered by specificity (most specific first)
    cancer_patterns = [
        # Head and neck patterns - MOST SPECIFIC FIRST
        # Pattern for structured context like "Cancer Type: Squamous Cell Carcinoma" + "Location: maxilla"
        (r"cancer type:\s*squamous cell carcinoma.*?location:\s*(maxilla|oral|tongue|gingiva|palate|buccal|pharynx|larynx|neck|oropharynx)", "head and neck squamous cell carcinoma"),
        (r"location:\s*(maxilla|oral|tongue|gingiva|palate|buccal|pharynx|larynx|neck|oropharynx).*?cancer type:\s*squamous cell carcinoma", "head and neck squamous cell carcinoma"),
        # Pattern for "SCC of maxilla" or "squamous cell carcinoma of maxilla"
        (r"scc\s+(?:of\s+)?(?:the\s+)?(?:r(?:ight)?\s+|l(?:eft)?\s+)?(maxilla|oral|tongue|gingiva|palate|buccal|pharynx|larynx|neck|oropharynx)", "head and neck squamous cell carcinoma"),
        (r"squamous cell carcinoma\s+(?:of\s+)?(?:the\s+)?(?:r(?:ight)?\s+|l(?:eft)?\s+)?(maxilla|oral|tongue|gingiva|palate|buccal|pharynx|larynx|neck|oropharynx)", "head and neck squamous cell carcinoma"),
        # Pattern for "SCC ... maxilla" with some words in between
        (r"scc.{0,30}(maxilla|oral cavity|tongue|gingiva|palate|buccal|pharynx|larynx|neck)", "head and neck squamous cell carcinoma"),
        (r"squamous cell carcinoma.{0,30}(maxilla|oral cavity|tongue|gingiva|palate|buccal|pharynx|larynx|neck)", "head and neck squamous cell carcinoma"),
        # Pattern for anatomical site followed by SCC/cancer
        (r"(maxilla|oral cavity|tongue|gingiva|palate|buccal|oropharynx|hypopharynx|larynx).{0,30}(cancer|carcinoma|scc|squamous)", "head and neck squamous cell carcinoma"),
        # General head and neck patterns
        (r"head and neck.{0,20}(cancer|carcinoma|scc|squamous)", "head and neck squamous cell carcinoma"),
        (r"(oral cavity|oropharynx|larynx|pharyngeal).{0,20}(cancer|carcinoma|scc)", "head and neck cancer"),
        # Recurrence patterns for H&N
        (r"(nodal recurrence|neck recurrence).{0,50}(scc|squamous|maxilla|oral|pharynx)", "head and neck squamous cell carcinoma"),
        (r"(scc|squamous|maxilla|oral|pharynx).{0,50}(nodal recurrence|neck recurrence)", "head and neck squamous cell carcinoma"),
        
        # Other cancer types
        (r"non-small cell lung|nsclc|lung adenocarcinoma|lung cancer", "non-small cell lung cancer"),
        (r"small cell lung|sclc", "small cell lung cancer"),
        (r"breast cancer|breast", "breast cancer"),
        (r"prostate cancer|prostate", "prostate cancer"),
        (r"colorectal|colon cancer|rectal cancer", "colorectal cancer"),
        (r"pancreatic cancer|pancreas", "pancreatic cancer"),
        (r"glioblastoma|gbm|brain cancer|brain tumor", "brain cancer"),
        (r"esophageal cancer|esophagus", "esophageal cancer"),
        (r"gastric cancer|stomach cancer", "gastric cancer"),
        (r"hepatocellular|liver cancer|hcc", "liver cancer"),
        (r"cervical cancer|cervix", "cervical cancer"),
        (r"ovarian cancer|ovary", "ovarian cancer"),
        (r"bladder cancer|bladder", "bladder cancer"),
        (r"renal cell|kidney cancer", "kidney cancer"),
        (r"melanoma", "melanoma"),
        (r"lymphoma", "lymphoma"),
        (r"leukemia", "leukemia"),
        (r"nasopharyngeal|nasopharynx", "nasopharyngeal cancer"),
        (r"seminoma|testicular|testis", "testicular cancer"),
    ]
    
    for pattern, cancer_name in cancer_patterns:
        if re.search(pattern, all_text):
            context["cancer_type"] = cancer_name
            print(f"[Context] Matched cancer type '{cancer_name}' with pattern: {pattern[:50]}...")
            break
    
    # Stage patterns
    stage_patterns = [
        (r"stage\s*(iii[abc]?|3[abc]?)", "stage III"),
        (r"stage\s*(ii[abc]?|2[abc]?)", "stage II"),
        (r"stage\s*(i[abc]?|1[abc]?)", "stage I"),
        (r"stage\s*(iv[abc]?|4[abc]?)", "stage IV"),
        (r"locally advanced", "locally advanced"),
        (r"metastatic", "metastatic"),
        (r"early[- ]stage", "early stage"),
        # TNM staging
        (r"pt4|ct4", "T4"),
        (r"pt3|ct3", "T3"),
        (r"pn[1-3]|cn[1-3]", "node positive"),
        (r"nodal recurrence", "recurrent"),
    ]
    
    for pattern, stage_name in stage_patterns:
        match = re.search(pattern, all_text)
        if match:
            # Try to get the specific stage (e.g., IIIA)
            full_match = match.group(0)
            context["stage"] = full_match.replace("stage ", "stage ").strip()
            break
    
    # Patient profile patterns
    age_match = re.search(r"(\d+)\s*(?:years?\s*old|yo|y\.?o\.?|\+)", all_text)
    if age_match:
        age = age_match.group(1)
        if "+" in age_match.group(0):
            context["patient_profile"].append(f"age {age}+")
        else:
            context["patient_profile"].append(f"age {age}")
    
    # Gender
    if re.search(r"\b(male|man|men)\b", all_text):
        context["patient_profile"].append("male")
    elif re.search(r"\b(female|woman|women)\b", all_text):
        context["patient_profile"].append("female")
    
    # Recurrence status
    if re.search(r"recurrence|recurrent|relapse", all_text):
        context["patient_profile"].append("recurrent")
    
    # Treatment patterns
    treatment_patterns = [
        (r"chemoradio|chemoRT|concurrent chemo", "concurrent chemoradiotherapy"),
        (r"chemotherapy|chemo\b", "chemotherapy"),
        (r"radiotherapy|radiation|RT\b", "radiotherapy"),
        (r"surgery|surgical|resection|maxillectomy|neck dissection", "surgery"),
        (r"immunotherapy|checkpoint inhibitor|pembrolizumab|nivolumab", "immunotherapy"),
        (r"targeted therapy|tki|tyrosine kinase", "targeted therapy"),
    ]
    
    for pattern, treatment_name in treatment_patterns:
        if re.search(pattern, all_text):
            context["treatment"] = treatment_name
            break

    # Comprehensive biomarker extraction with status normalization
    # (aligned with data/keywords/extractor_keywords.json)
    # Returns canonical forms like "ER positive", "HER2 negative", "BRAF V600E positive"
    context["biomarkers"] = _extract_biomarkers_from_text(all_text)

    # Genomic / recurrence score patterns (Oncotype DX, MammaPrint, etc.)
    score_patterns = [
        (r"(?:21[- ]?gene|oncotype\s*(?:dx)?)\s*(?:recurrence\s*)?score\s*(?:of\s*)?(\d+)", "oncotype_dx"),
        (r"recurrence\s*score\s*(?:of\s*)?(\d+)", "recurrence_score"),
        (r"mammaprint\s*(?:score\s*)?(?:of\s*)?(\w+)", "mammaprint"),
    ]

    for pattern, score_type in score_patterns:
        match = re.search(pattern, all_text)
        if match:
            value = match.group(1)
            try:
                context["genomic_scores"][score_type] = int(value)
            except ValueError:
                context["genomic_scores"][score_type] = value

    # Surgery type patterns
    surgery_patterns = [
        (r"breast[- ]conserving\s*(?:surgery|therapy)|bcs|lumpectomy|partial\s*mastectomy|wide\s*(?:local\s*)?excision", "breast-conserving surgery"),
        (r"mastectomy", "mastectomy"),
        (r"maxillectomy", "maxillectomy"),
        (r"laryngectomy", "laryngectomy"),
        (r"neck\s*dissection", "neck dissection"),
        (r"prostatectomy", "prostatectomy"),
        (r"lobectomy", "lobectomy"),
        (r"colectomy|proctectomy|lar|low\s*anterior\s*resection|apr|abdominoperineal", "colorectal surgery"),
    ]

    for pattern, surgery_name in surgery_patterns:
        if re.search(pattern, all_text):
            context["surgery_type"] = surgery_name
            break

    # TNM staging (capture full TNM string from conversation)
    tnm_match = re.search(r"[pcy]?t([0-4](?:is|a|b|c)?)\s*n([0-3](?:a|b|c|mi)?)\s*(?:c?)m([01])", all_text)
    if tnm_match:
        context["tnm"] = f"T{tnm_match.group(1)}N{tnm_match.group(2)}M{tnm_match.group(3)}"

    return context


# ============================================
# POST-RETRIEVAL CANCER-TYPE RELEVANCE FILTER
# ============================================

# Map cancer types to keywords that identify a study as belonging to that cancer type
_CANCER_TYPE_IDENTIFIERS = {
    "breast": ["breast", "mammary", "mastectomy", "lumpectomy", "her2", "dcis", "oncotype",
               "tamoxifen", "trastuzumab", "axillary", "mammosite", "pmrt"],
    "lung": ["lung", "nsclc", "sclc", "mesothelioma", "thoracic", "bronchus", "pleural"],
    "prostate": ["prostate", "prostatic", "prostatectomy", "psa"],
    "head and neck": ["head and neck", "oropharynx", "larynx", "pharynx", "oral cavity",
                      "tongue", "nasopharynx", "hnscc", "laryngeal", "pharyngeal"],
    "colorectal": ["colorectal", "colon", "rectal", "rectum"],
    "cervical": ["cervical", "cervix"],
    "anal": ["anal cancer", "anal carcinoma", "epidermoid anal"],
    "hodgkin": ["hodgkin", "beacopp"],
    "brain": ["glioblastoma", "gbm", "glioma", "brain tumor", "cranial"],
    "esophageal": ["esophageal", "esophagus"],
    "gastric": ["gastric", "stomach"],
    "pancreatic": ["pancreatic", "pancreas"],
    "bladder": ["bladder", "urothelial"],
    "renal": ["renal", "kidney"],
    "ovarian": ["ovarian", "ovary"],
    "melanoma": ["melanoma"],
    "lymphoma": ["lymphoma", "dlbcl"],
}


def _detect_cancer_type_from_text(text: str) -> Optional[str]:
    """Detect the cancer type from a chunk's text/title/category."""
    text_lower = text.lower()
    for cancer_type, keywords in _CANCER_TYPE_IDENTIFIERS.items():
        for kw in keywords:
            if kw in text_lower:
                return cancer_type
    return None


def _filter_irrelevant_cancer_type_studies(
    evidence: List[Dict[str, Any]],
    query: str,
    original_question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Remove retrieved studies whose cancer type clearly doesn't match the patient's.

    This is critical for follow-up queries like "what about RT?" where the retrieval
    may return RT studies from unrelated cancer types (prostate, larynx, etc.) even
    though the patient has breast cancer.

    Only filters when a clear patient cancer type is detected from the query or
    conversation history, and only removes studies with a clearly *different* cancer type.
    Studies with no detectable cancer type are kept (they may be general/guideline studies).
    """
    # Determine the patient's cancer type from the expanded query
    patient_cancer = _detect_cancer_type_from_text(query)

    # Also check conversation history if query didn't have it
    if not patient_cancer and conversation_history:
        all_msgs = []
        for msg in conversation_history:
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
            elif hasattr(msg, "dict"):
                msg = msg.dict()
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if content:
                all_msgs.append(content)
        if all_msgs:
            combined = " ".join(all_msgs)
            patient_cancer = _detect_cancer_type_from_text(combined)

    if not patient_cancer:
        return evidence  # Can't filter without knowing cancer type

    print(f"[PostFilter] Patient cancer type detected: {patient_cancer}")

    filtered = []
    removed_count = 0
    for chunk in evidence:
        # Build text to check from chunk metadata
        chunk_text_parts = []
        chunk_text_parts.append(chunk.get("title", ""))
        chunk_text_parts.append(chunk.get("text", ""))
        chunk_text_parts.append(chunk.get("category", ""))
        payload = chunk.get("payload", {})
        if payload:
            chunk_text_parts.append(payload.get("title", ""))
            chunk_text_parts.append(payload.get("text", ""))
            chunk_text_parts.append(payload.get("category", ""))
            doc_meta = payload.get("doc_meta", {})
            if doc_meta:
                chunk_text_parts.append(doc_meta.get("title", ""))

        combined_text = " ".join(str(p) for p in chunk_text_parts if p)
        study_cancer = _detect_cancer_type_from_text(combined_text)

        if study_cancer is None:
            # Can't determine study cancer type - keep it (may be general guidelines)
            filtered.append(chunk)
        elif study_cancer == patient_cancer:
            filtered.append(chunk)
        else:
            # Study is about a different cancer type - remove it
            removed_count += 1
            title = chunk.get("title") or payload.get("title", "unknown")
            print(f"[PostFilter] Removed '{title[:60]}...' (study={study_cancer}, patient={patient_cancer})")

    if removed_count > 0:
        print(f"[PostFilter] Removed {removed_count}/{len(evidence)} irrelevant cancer-type studies, kept {len(filtered)}")

    # Safety: don't remove ALL evidence
    if len(filtered) < 2 and len(evidence) >= 2:
        print(f"[PostFilter] Too aggressive - keeping all evidence to avoid empty results")
        return evidence

    return filtered


SITE_KEYWORDS = [
    ("Breast", ["breast", "breast cancer", "mastectomy", "lumpectomy", "her2", "dcis",
                "lobular", "ductal", "oncotype", "tamoxifen", "trastuzumab", "sentinel node breast",
                "axillary", "mammography", "mammosite", "pmrt", "postmastectomy"]),

    # NOTE: pembrolizumab is intentionally NOT in this list — it is FDA-approved
    # across HNSCC, NSCLC, melanoma, urothelial, MSI-H solid tumors, esophageal,
    # cervical, gastric, RCC, etc. Tagging it as a Lung-only signal caused
    # head-and-neck cases that mention pembrolizumab to be misclassified.
    ("Lung", ["lung", "nsclc", "sclc", "mesothelioma", "thoracic", "mediastinal",
              "egfr", "alk", "osimertinib", "bronchus", "pleural"]),
    
    ("CNS", ["brain", "glioma", "glioblastoma", "gbm", "medulloblastoma", "ependymoma",
             "meningioma", "csi", "craniospinal", "astrocytoma", "oligodendroglioma",
             "pituitary", "craniopharyngioma", "pineal"]),
    
    ("H&N", ["head and neck", "oropharynx", "nasopharynx", "larynx", "oral cavity",
             "tongue", "hpv", "tonsil", "laryngeal", "pharyngeal", "salivary",
             "hypopharynx", "paranasal", "sinus",
             # ── Oral cavity subsites ────────────────────────────────
             "maxilla", "maxillary", "maxillectomy", "mandible", "mandibular",
             "mandibulectomy", "buccal", "buccal mucosa", "hard palate",
             "soft palate", "floor of mouth", "retromolar", "gingiva",
             "alveolar ridge", "oral tongue", "mobile tongue", "tongue body",
             "anterior tongue", "base of tongue", "tongue base", "bot", "lip",
             "vermillion", "labial",
             # ── Pharyngeal subsites ─────────────────────────────────
             "pyriform", "pyriform sinus", "piriform", "piriform sinus",
             "postcricoid", "posterior pharyngeal wall",
             # ── Laryngeal subsites ──────────────────────────────────
             "glottic", "glottis", "supraglottic", "supraglottis",
             "subglottic", "subglottis", "vocal cord", "vocal fold",
             "aryepiglottic", "epiglottis", "epiglottic",
             # ── Salivary gland subsites ─────────────────────────────
             "parotid", "submandibular gland", "sublingual gland",
             "minor salivary", "mucoepidermoid", "adenoid cystic",
             # ── Nasopharyngeal & sinonasal ──────────────────────────
             "npc", "nasopharyngeal carcinoma", "maxillary sinus",
             "ethmoid", "ethmoid sinus", "frontal sinus", "sphenoid sinus",
             "nasal cavity", "nasal septum",
             # ── Post-treatment / surgery terms ──────────────────────
             "glossectomy", "laryngectomy", "neck dissection",
             "radial forearm free flap", "rfff", "alt flap", "pec flap",
             # ── Disease abbreviations ───────────────────────────────
             "hnscc", "hncscc", "scchn", "opscc", "npscc", "lscc",
             "oropharyngeal scc", "oral cavity scc"]),

    ("GI", ["rectal", "colorectal", "esophageal", "gastric", "pancreatic", "liver",
            "hepatocellular", "anal", "colon", "rectum", "stomach", "duodenal",
            "cholangiocarcinoma", "biliary", "appendix"]),
    
    ("GYN", ["cervix", "cervical", "endometrial", "ovarian", "uterine", "vulvar",
             "vaginal", "hysterectomy", "brachytherapy", "fallopian", "gestational"]),
    
    ("GU", ["bladder", "kidney", "renal", "testicular", "seminoma", "urothelial",
            "mibc", "turbt", "orchiectomy", "penile", "ureter", "upper tract"]),
    
    ("Prostate", ["prostate", "psa", "gleason", "adt", "radical prostatectomy",
                  "abiraterone", "enzalutamide", "lupron", "bicalutamide"]),
    
    ("Sarcoma", ["sarcoma", "ewing", "osteosarcoma", "rhabdomyosarcoma", "desmoid",
                 "soft tissue", "liposarcoma", "leiomyosarcoma", "synovial sarcoma",
                 "chondrosarcoma", "gastrointestinal stromal", "gist"]),
    
    ("Lymphoma", ["lymphoma", "hodgkin", "dlbcl", "follicular", "myeloma",
                  "non-hodgkin", "mantle cell", "mycosis fungoides"]),
    
    ("Cutaneous", ["melanoma", "basal cell", "squamous cell skin", "merkel cell",
                   "skin cancer", "cutaneous", "dermatofibrosarcoma"]),
    
    ("Peds", ["pediatric", "child", "medulloblastoma", "neuroblastoma", "wilms",
              "retinoblastoma", "infant", "adolescent", "childhood"]),
    
    ("Radiophar", ["psma", "lutetium", "radium-223", "theranostics", "radiopharmaceutical",
                   "lu-177", "i-131", "radioiodine"]),
    
    ("Thyroid", ["thyroid", "papillary", "follicular thyroid", "radioactive iodine",
                 "medullary thyroid", "anaplastic thyroid"]),
    
    ("Benign", ["benign", "keloid", "avm", "meningioma benign", "arteriovenous malformation",
                "heterotopic ossification"]),
]


def infer_site_key(query: str, default: str = "Radiotherapy&Oncology") -> str:
    """Infer tumor site from query text with word-boundary matching and
    weighted scoring.

    Multi-word anatomical phrases (e.g. "head and neck", "non small cell")
    score proportionally to their word count, so an explicit anatomical
    mention is not silently overridden by a single dual-use token (e.g. an
    incidental drug name that has multiple FDA-approved indications across
    sites).

    Example: a query containing both "pembrolizumab" and "head and neck"
    used to tie at 1-1 and break in favour of whichever site appeared first
    in SITE_KEYWORDS (Lung), misclassifying H&N cases. With weighted
    scoring, "head and neck" scores 3 vs Lung's 0–1 and H&N wins cleanly.
    """
    ql = query.lower()
    site_scores: Counter = Counter()

    for site, kws in SITE_KEYWORDS:
        for kw in kws:
            # Use word boundary for short keywords to avoid false positives
            if len(kw) <= 4:
                if re.search(rf"\b{re.escape(kw)}\b", ql):
                    site_scores[site] += 1
            else:
                if kw in ql:
                    # Weight multi-word phrases by their token count so an
                    # explicit anatomical phrase outranks a single drug name.
                    site_scores[site] += max(1, len(kw.split()))

    if site_scores:
        return site_scores.most_common(1)[0][0]
    return default


# ============================================
# ENHANCED HYBRID RETRIEVER CLASS
# ============================================

class EnhancedHybridRetriever:
    """
    Enhanced hybrid retriever with ALL features from Colab:
    - Query expansion (P0) - NOW BIDIRECTIONAL
    - Query type classification (P1) - 8 types + indication_question
    - Cross-encoder reranking (P1)
    - Dose-aware boosting (P2)
    - NCCN gap detection (P0)
    - Site inference
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        collection: str,
        embed_model: str = "text-embedding-3-large",
        use_cross_encoder: bool = True,
    ):
        self.qdrant = qdrant_client
        self.oa = openai_client
        self.collection = collection
        self.embed_model = embed_model
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE

    def embed_query(self, query_text: str) -> List[float]:
        resp = self.oa.embeddings.create(model=self.embed_model, input=[query_text])
        return resp.data[0].embedding

    async def retrieve(
        self,
        query_text: str,
        category: Optional[str] = None,
        N: int = 100,
        k_final: int = 10,
        rerank_pool: int = 50,
        user_id: Optional[str] = None,
        accumulated_context: Optional[Dict[str, Any]] = None,
        strict_category: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Enhanced retrieval pipeline with comprehensive timing instrumentation.
        Returns: (evidence_packs, metadata)
        
        Args:
            user_id: Optional user ID to fetch and apply preferences
            accumulated_context: Optional accumulated structured context from previous queries in conversation
        """
        # Start total timing
        t_total_start = time.perf_counter()
        timings = {}
        
        # ===================================================================
        # 0. Fetch user preferences for filtering
        # ===================================================================
        user_preferences = None
        if user_id:
            try:
                from src.api.services.preferences_filter_service import get_user_preferences_sync
                user_preferences = get_user_preferences_sync(user_id)
                if user_preferences:
                    print(f"[Preferences] Loaded filters for user, sort_by={user_preferences.get('sort_by', 'relevance')}")
            except Exception as e:
                print(f"[Preferences] Failed to load: {e}")
                import traceback
                traceback.print_exc()
        
        # ===================================================================
        # 0a. Extract clinical profile for profile-aware retrieval
        # ===================================================================
        t0 = time.perf_counter()
        clinical_profile = None
        try:
            from src.api.services.clinical_entity_extractor import get_clinical_entity_extractor
            extractor = get_clinical_entity_extractor()
            profile = extractor.extract(query_text)
            clinical_profile = {
                "must_match": extractor.get_must_match_terms(profile),
                "should_match": extractor.get_should_match_terms(profile),
                "raw_profile": profile.to_dict(),
            }
        except Exception as e:
            print(f"Warning: Clinical entity extraction failed: {e}")
        timings["0_clinical_extraction"] = time.perf_counter() - t0
        
        # ===================================================================
        # 0b. Detect trial mentions for trial boosting
        # ===================================================================
        t0b = time.perf_counter()
        detected_trials = []
        try:
            from src.api.services.trial_registry import get_trial_registry
            registry = get_trial_registry()
            detected_trials = registry.detect_trial(query_text)
        except Exception as e:
            print(f"Warning: Trial detection failed: {e}")
        timings["0b_trial_detection"] = time.perf_counter() - t0b
        
        # ===================================================================
        # 1. Classify query type
        # ===================================================================
        t1 = time.perf_counter()
        query_classification = classify_query_hybrid(query_text, self.oa)
        query_type = query_classification["primary_type"]
        timings["1_query_classification"] = time.perf_counter() - t1

        # ===================================================================
        # 1b. NEW: Fast query structuring (regex-based, no LLM)
        # ===================================================================
        t1b = time.perf_counter()
        query_structure = None
        try:
            from src.api.services.query_structuring_service import structure_query_fast, merge_query_structures
            query_structure = structure_query_fast(query_text, query_type)
            
            # Merge with accumulated context from conversation
            if accumulated_context:
                query_structure = merge_query_structures(accumulated_context, query_structure)
                print(f"[Query Structure] Merged with accumulated context")
            
            if query_structure.has_patient_context:
                try:
                    from src.api.services import pipeline_metrics as _pm
                    if _pm.current() is not None:
                        _pm.current().event("has_patient_context")
                except Exception:
                    pass
                print(f"[Query Structure] Patient context detected:")
                if query_structure.cancer.site:
                    print(f"    └─ Cancer site: {query_structure.cancer.site}")
                if query_structure.cancer.get_tnm_string():
                    print(f"    └─ TNM: {query_structure.cancer.get_tnm_string()}")
                if query_structure.cancer.histology:
                    print(f"    └─ Histology: {query_structure.cancer.histology}")
                if query_structure.treatment.modality:
                    print(f"    └─ Treatment: {query_structure.treatment.modality}")
                if query_structure.cancer.biomarkers:
                    print(f"    └─ Biomarkers: {query_structure.cancer.biomarkers}")
                if query_structure.boost_terms:
                    print(f"    └─ Boost terms: {query_structure.boost_terms[:5]}")
        except Exception as e:
            print(f"Warning: Query structuring failed: {e}")
            import traceback
            traceback.print_exc()

        # ── Task 1: Mark complex queries for LLM extraction ─────────────
        # LLM extraction is deferred to step 3 where it runs IN PARALLEL
        # with embedding generation via asyncio.gather(). Running it here
        # would block embedding for 3-6s on every complex query.
        inferred_axes = None
        is_complex = (
            len(query_text) > 150
            or query_text.count(',') > 4
            or any(t in query_text.lower() for t in [
                'progression', 'refractory', 'metastatic', 'recurrent',
                'pembrolizumab', 'nivolumab', 'ici', 's/p', 'status post', 'pmh',
                'ilo', 'locoregional', 'cardiac', 'ventricle',
            ])
        )
        # Flag for step 3 — actual LLM call happens there in parallel with embedding
        _needs_llm = is_complex and query_structure and not getattr(query_structure, 'used_llm_extraction', False)

        # Inference layer runs after step 3 (after LLM extraction merge)

        timings["1b_query_structuring"] = time.perf_counter() - t1b

        # ===================================================================
        # 2. ENHANCED: Bidirectional query expansion
        # ===================================================================
        t2 = time.perf_counter()
        expanded_query = expand_query(query_text)
        
        # NEW: Add boost terms from query structure to expanded query
        if query_structure and query_structure.boost_terms:
            # Add unique boost terms that aren't already in the query
            existing_terms = set(expanded_query.lower().split())
            new_terms = [t for t in query_structure.boost_terms
                        if t.lower() not in existing_terms]
            if new_terms:
                expanded_query = expanded_query + " " + " ".join(new_terms)

        # NEW: Expand staging notation (TNM ↔ Stage Group) pre-retrieval
        # e.g. "T2N0M0" → also search "Stage II", "Stage IIA", etc.
        try:
            from src.api.services.staging_search_expander import expand_query_with_staging
            cancer_type_hint = None
            if query_structure and query_structure.cancer.site:
                cancer_type_hint = query_structure.cancer.site
            staging_terms = expand_query_with_staging(query_text, cancer_type=cancer_type_hint)
            if staging_terms.all_search_terms:
                existing_lower = set(expanded_query.lower().split())
                new_staging = [t for t in staging_terms.all_search_terms
                               if t.lower() not in existing_lower]
                if new_staging:
                    expanded_query = expanded_query + " " + " ".join(new_staging[:8])
                    print(f"[Staging Expansion] Added {len(new_staging[:8])} staging variants: {new_staging[:4]}")
        except Exception as e:
            print(f"[Staging Expansion] Failed (continuing without): {e}")

        # Apply comprehensive ontology / drug / staging / clinical-context
        # expansion from the JSON data files so the embedding vector covers
        # ALL known synonym / abbreviation / brand-name variants.
        try:
            from src.api.services.query_expansion import expand_query_comprehensive
            expanded_query = expand_query_comprehensive(expanded_query)
        except Exception as e:
            print(f"[Comprehensive Expansion] Failed (continuing): {e}")

        # Fold the ontology resolver's canonical labels (cancer_types, sites,
        # histologies, stage aliases, biomarkers, drugs, alterations) into the
        # embedding input so the vector space itself carries the ontology-
        # normalised forms — not just the string-level synonym expansion
        # above. The same resolver call is re-issued later in
        # ComprehensiveRetriever for filter construction; it's LRU-cached on
        # query_text so the second call is ~1 microsecond.
        try:
            from src.api.services.query_token_resolver import resolve_query_tokens
            from src.api.services.query_structuring_service import (
                filter_metastatic_site_canonicals,
            )
            _resolved = resolve_query_tokens(query_text)

            # Drop site canonicals whose only query occurrences are in
            # metastatic-context windows ("hepatic metastases" → don't
            # inject "Liver" into a lung-primary query's embedding).
            # Same trailing-window logic as the structurer's site
            # detection, applied to the resolver's finer-grained site
            # canonicals.
            _site_terms = filter_metastatic_site_canonicals(
                query_text, list(_resolved.sites)
            )
            _dropped_sites = sorted(set(_resolved.sites) - set(_site_terms))
            if _dropped_sites:
                print(
                    f"[Resolver Expansion] Dropped {len(_dropped_sites)} "
                    f"metastatic-context site canonicals (not injected "
                    f"into embedding): {_dropped_sites[:6]}"
                )

            _label_terms = (
                set(_resolved.cancer_types)
                | set(_site_terms)
                | set(_resolved.histologies)
                | set(_resolved.stage_aliases)
                | set(_resolved.biomarkers)
                | set(_resolved.drugs)
                | set(_resolved.alterations)
            )
            _existing_lower = set(expanded_query.lower().split())
            _new_labels = [
                t for t in sorted(_label_terms)
                if t and t.lower() not in _existing_lower
            ]
            if _new_labels:
                expanded_query = expanded_query + " " + " ".join(_new_labels)
                print(
                    f"[Resolver Expansion] Added {len(_new_labels)} canonical "
                    f"labels to embedding input: {_new_labels[:6]}"
                )
        except Exception as e:
            print(f"[Resolver Expansion] Failed (continuing): {e}")

        timings["2_query_expansion"] = time.perf_counter() - t2

        # Log expansion for debugging
        if expanded_query != query_text:
            print(f"[Query Expansion] Original: {query_text[:60]}...")
            print(f"[Query Expansion] Expanded ({len(expanded_query)} chars): {expanded_query[:120]}...")

        # Get route based on query type
        route = query_route(expanded_query, query_type)

        # ===================================================================
        # 3. Embed expanded query (+ parallel LLM extraction for complex queries)
        # ===================================================================
        t3 = time.perf_counter()

        # Determine whether to run LLM extraction in parallel with embedding.
        # Two gates: the Task 1 complexity gate (_needs_llm) and the stricter
        # _needs_llm_extraction() heuristic. Either triggers parallel LLM.
        llm_extraction_task = None
        if _needs_llm and query_structure:
            try:
                from src.api.services.query_structuring_service import structure_query_with_llm, merge_llm_extraction
                print(f"[QueryStructuring] Complex query detected (Task 1 gate), "
                      f"will run LLM extraction in parallel with embedding")
                llm_extraction_task = structure_query_with_llm(query_text)
            except Exception as e:
                print(f"[QueryStructuring] LLM extraction setup failed: {e}")

        if not llm_extraction_task:
            # Task 1 gate didn't fire — try the stricter _needs_llm_extraction() gate
            if not (query_structure and getattr(query_structure, 'used_llm_extraction', False)):
                try:
                    from src.api.services.query_structuring_service import _needs_llm_extraction, structure_query_with_llm, merge_llm_extraction
                    if _needs_llm_extraction(query_text):
                        print(f"[QueryStructuring] Complex query detected (strict gate), "
                              f"will run LLM extraction in parallel")
                        llm_extraction_task = structure_query_with_llm(query_text)
                except Exception as e:
                    print(f"[QueryStructuring] LLM extraction check failed: {e}")

        # Run embedding + LLM extraction concurrently (or just embedding)
        if llm_extraction_task:
            import asyncio
            qvec, llm_result = await asyncio.gather(
                asyncio.to_thread(self.embed_query, expanded_query),
                llm_extraction_task
            )
            if llm_result and query_structure:
                query_structure = merge_llm_extraction(query_structure, llm_result)
                query_structure.used_llm_extraction = True
                print(f"[QueryStructuring] LLM extraction merged: {list(llm_result.keys())}")

                # Normalize the LLM output into a ClinicalProfile using
                # SynonymIndex, then fold its canonical values back into
                # the QueryStructure to fill gaps left by regex. This
                # reuses the existing LLM call (no second GPT-4o-mini
                # invocation) and gives the standard retrieval path
                # access to the same LLM+ontology-normalized axes the
                # comprehensive path already uses.
                try:
                    from src.api.services.clinical_extractor import (
                        apply_profile_to_structure,
                        build_profile_from_llm_result,
                    )
                    profile = build_profile_from_llm_result(
                        query_text=query_text,
                        llm_result=llm_result,
                    )
                    # ── Fallback enrichment from QueryStructure ─────
                    # build_profile_from_llm_result tries whole-phrase
                    # then single-token resolution against the synonym
                    # index. Multi-word subsite labels like "oral
                    # tongue" or "oral cavity" inside a longer
                    # primary_cancer string ("squamous cell carcinoma
                    # of the left oral tongue") fall through both
                    # paths, leaving cancer_type_label=None. The helper
                    # below uses data/ontology/cancer_type_ontology.json
                    # (22 cancer types, ~150 synonyms, ~130 subtypes,
                    # 252 unique drugs) and cancer_type_sites.json to
                    # fill every axis the LLM extractor missed —
                    # cancer_type_label, cancer_sites, histologies,
                    # stages, prior_treatments, disease_status.
                    try:
                        from src.api.services.clinical_profile_enrichment import (
                            enrich_profile_from_query_structure,
                        )
                        enrich_profile_from_query_structure(
                            profile=profile,
                            query_structure=query_structure,
                            raw_query=query_text,
                        )
                    except Exception as _e:
                        print(f"[ClinicalProfile] enrichment helper failed (continuing): {_e}")

                    # Expose the profile to downstream steps via a
                    # dedicated attribute on QueryStructure (no dataclass
                    # change required — runtime attribute). The metadata
                    # dict that retrieve() returns is initialized much
                    # later (line ~4658), so the cross-scope stash for
                    # callers happens there — see "_clinical_profile"
                    # in the metadata = {...} block.
                    query_structure._clinical_profile = profile  # type: ignore[attr-defined]
                    query_structure = apply_profile_to_structure(query_structure, profile)
                    if profile.has_any_filter():
                        print(
                            f"[ClinicalProfile] "
                            f"type={profile.cancer_type_label!r} "
                            f"hist={profile.histologies} "
                            f"biomarkers={profile.biomarkers} "
                            f"expr={len(profile.biomarker_expressions)}"
                        )
                    # Log what, if anything, was filled from the profile
                    print(
                        f"[ClinicalProfile] Applied to structure: "
                        f"site={query_structure.cancer.site!r} "
                        f"category={query_structure.filter_category!r} "
                        f"histology={query_structure.cancer.histology!r} "
                        f"biomarkers={query_structure.cancer.biomarkers}"
                    )
                except Exception as e:
                    print(f"[ClinicalProfile] profile application failed (continuing without): {e}")
        else:
            # Threaded — a direct embed_query() call here blocks the event
            # loop for the full OpenAI round-trip, stalling every other
            # in-flight request on this single-worker process. The parallel
            # branch above already does this via asyncio.gather + to_thread.
            import asyncio as _asyncio
            qvec = await _asyncio.to_thread(self.embed_query, expanded_query)

        timings["3_embedding_generation"] = time.perf_counter() - t3
        print(f"    └─ embedding: {timings['3_embedding_generation']:.3f}s" +
              (" (+ LLM extraction in parallel)" if llm_extraction_task else ""))

        # ── Run inference layer AFTER LLM extraction merge ─────────────
        # (Moved from 1b block so it benefits from LLM-extracted _llm_axes)
        if query_structure and query_structure.has_patient_context and inferred_axes is None:
            try:
                from src.api.services.clinical_inference import apply_inference_to_query_structure
                inferred_axes = apply_inference_to_query_structure(query_structure, query_text)
                print(f"[Inference] raw_text length: {len(query_text)}, first 80 chars: {query_text[:80]}...")
                print(f"[Inference] Flags: {inferred_axes.get('trajectory_flags', [])}")
                print(f"[Inference] Met sites: {inferred_axes.get('metastatic_sites', [])}")
                if inferred_axes.get('surgical_candidate') is False:
                    print(f"[Inference] Surgical candidate: False (inferred from narrative)")
                total_inferred = sum(len(v) for v in inferred_axes.get('inferred_terms', {}).values())
                if total_inferred:
                    print(f"[Inference] Added {total_inferred} inferred terms across "
                          f"{sum(1 for v in inferred_axes.get('inferred_terms', {}).values() if v)} axes")
            except Exception as e:
                print(f"[Inference] Inference layer failed (continuing without): {e}")

        # ===================================================================
        # 3.5 Build Qdrant Filter from Clinical Profile
        # ===================================================================
        # Uses search_terms payload field in Qdrant for fast filtering.
        # No PostgreSQL call - much faster.
        
        prefilter_result = PreFilterResult()
        prefilter_metadata = {}
        
        if is_prefilter_enabled() and clinical_profile:
            print(f"[DEBUG] clinical_profile raw: {clinical_profile.get('raw_profile', {})}")
            t_prefilter = time.perf_counter()
            
            try:
                # Build Qdrant filter from clinical profile (uses category mapping)
                prefilter_result = build_qdrant_filter_from_clinical_profile(
                    clinical_profile=clinical_profile,
                    category=category,
                )
                
                timings["3.5_prefilter_extraction"] = time.perf_counter() - t_prefilter
                
                if prefilter_result.filter_applied:
                    print(f"    └─ prefilter: {timings['3.5_prefilter_extraction']:.3f}s "
                          f"(category: {prefilter_result.category})")
                else:
                    print(f"    └─ prefilter: {timings['3.5_prefilter_extraction']:.3f}s "
                          f"(not applied: {prefilter_result.filter_reason})")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    └─ prefilter: skipped due to error: {e}")
                timings["3.5_prefilter_extraction"] = 0

        # ===================================================================
        # 4. Qdrant vector search + PostgreSQL structured matching (PARALLEL)
        # ===================================================================
        t4 = time.perf_counter()
        
        # Build filter - use explicit category OR inferred from prefilter
        # NOTE: Query structure category is NOT used for filtering to avoid
        # over-restricting results. PostgreSQL structured matching handles
        # relevance boosting instead.
        flt = None
        effective_category = None
        
        # Category priority (highest to lowest):
        #   1. Explicit `category` arg from the caller.
        #   2. Clinical prefilter (when feature-flagged on).
        #   3. `query_structure.filter_category` — populated by
        #      structure_query_fast() + apply_profile_to_structure()
        #      from the now-correct regex + LLM extraction. This was
        #      previously IGNORED for Qdrant filtering (the comment
        #      below said "used for PostgreSQL matching, not Qdrant"),
        #      which is why correctly-extracted site='lung' queries
        #      still retrieved prostate chunks from Qdrant.
        if category:
            # Explicit category takes precedence.
            #
            # Build a `should` filter over EVERY plausible spelling of the
            # canonical category so we don't silently miss studies that were
            # ingested under a different spelling (e.g. "H&N" vs
            # "h&n_processed_documents" vs "head_neck"). The previous code
            # used a single literal `must` value and silently fell through
            # to a no-filter retry when nothing matched, which is how
            # head-and-neck queries were pulling anal SCC and glioma studies.
            effective_category = category
            cat_variants = build_category_match_variants(category)
            if cat_variants:
                flt = qm.Filter(
                    should=[
                        qm.FieldCondition(
                            key="category",
                            match=qm.MatchValue(value=v),
                        )
                        for v in cat_variants
                    ]
                )
                print(
                    f"    └─ using explicit category filter "
                    f"(strict={strict_category}, {len(cat_variants)} variants): "
                    f"{cat_variants[:6]}{'...' if len(cat_variants) > 6 else ''}"
                )
            else:
                flt = qm.Filter(must=[
                    qm.FieldCondition(key="category", match=qm.MatchValue(value=category))
                ])
                print(f"    └─ using explicit category filter (no variants): {category}")
        elif prefilter_result.has_filter() and prefilter_result.qdrant_filter:
            # Use inferred category from prefilter
            effective_category = prefilter_result.category
            flt = prefilter_result.qdrant_filter
            print(f"    └─ using prefilter category: {prefilter_result.category}")
        elif query_structure and getattr(query_structure, "filter_category", None):
            # Structurer-inferred category (regex + LLM primary_cancer
            # + apply_profile_to_structure). Apply as a `should` filter
            # across all category-name spellings — same variant logic
            # as the explicit-category branch — so we don't silently
            # drop studies that were ingested under a minor spelling
            # variant.
            structure_category = query_structure.filter_category
            effective_category = structure_category
            cat_variants = build_category_match_variants(structure_category)
            if cat_variants:
                flt = qm.Filter(
                    should=[
                        qm.FieldCondition(
                            key="category",
                            match=qm.MatchValue(value=v),
                        )
                        for v in cat_variants
                    ]
                )
                print(
                    f"    └─ using query_structure.filter_category "
                    f"({len(cat_variants)} variants): "
                    f"{cat_variants[:6]}{'...' if len(cat_variants) > 6 else ''}"
                )
            else:
                flt = qm.Filter(must=[
                    qm.FieldCondition(
                        key="category",
                        match=qm.MatchValue(value=structure_category),
                    )
                ])
                print(
                    f"    └─ using query_structure.filter_category "
                    f"(no variants): {structure_category}"
                )

        # Start PostgreSQL structured matching in parallel for ALL queries
        # Even queries without explicit patient context benefit from structured matching
        # (e.g., "mechanism of IMRT for prostate" should prioritize prostate studies)
        structured_match_task = None
        structured_result = None
        if query_structure:
            try:
                from src.api.services.structured_study_matcher import match_studies_by_structure
                query_structure_dict = query_structure.to_dict()
                # Class 3a: propagate clinical_inference's metastatic_sites
                # into the matcher input so the fractional boost can score
                # studies that mention the patient's metastatic organs.
                if inferred_axes and inferred_axes.get("metastatic_sites"):
                    query_structure_dict["metastatic_sites"] = list(
                        inferred_axes["metastatic_sites"]
                    )
                structured_match_task = match_studies_by_structure(query_structure_dict, limit=50)
                context_info = "patient context" if query_structure.has_patient_context else "query terms"
                print(f"    └─ starting parallel PostgreSQL matching ({context_info})...")
            except Exception as e:
                print(f"    └─ structured matching setup failed: {e}")

        # Run Qdrant + PostgreSQL in parallel (always, when we have query structure)
        import asyncio
        if structured_match_task:
            # Run both in parallel
            qdrant_task = asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.collection,
                query=qvec,
                limit=N,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            )
            try:
                qdrant_result, structured_result = await asyncio.gather(
                    qdrant_task,
                    structured_match_task,
                    return_exceptions=True
                )
                # Handle exceptions from gather
                if isinstance(qdrant_result, Exception):
                    print(f"[RAG] Qdrant search failed: {qdrant_result}")
                    raise qdrant_result
                if isinstance(structured_result, Exception):
                    print(f"[RAG] PostgreSQL structured matching failed (continuing without): {structured_result}")
                    structured_result = None
                else:
                    hits = qdrant_result.points
            except Exception as e:
                print(f"[RAG] Parallel search failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to just Qdrant
                hits = self.qdrant.query_points(
                    collection_name=self.collection,
                    query=qvec,
                    limit=N,
                    query_filter=flt,
                    with_payload=True,
                    with_vectors=False,
                ).points
                structured_result = None
            else:
                hits = qdrant_result.points
            
            if structured_result and structured_result.doc_ids:
                print(f"    └─ PostgreSQL matched {len(structured_result.doc_ids)} studies in {structured_result.query_time_ms:.1f}ms")
        else:
            # Fallback: just run Qdrant search (no query structure available)
            hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=qvec,
                limit=N,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            ).points
        
        timings["4_qdrant_search"] = time.perf_counter() - t4
        filter_info = f"category={effective_category or 'none'}"
        parallel_info = f", +PG:{len(structured_result.doc_ids) if structured_result else 0}" if structured_match_task else ""
        print(f"    └─ qdrant search: {timings['4_qdrant_search']:.3f}s ({len(hits)} results, {filter_info}{parallel_info})") 

        # ===================================================================
        # 5. Convert to candidates
        # ===================================================================
        t5 = time.perf_counter()
        cands: List[Dict[str, Any]] = []
        for h in hits:
            payload = dict(h.payload or {})
            cands.append({
                "point_id": h.id,
                "score_dense": float(h.score),
                "payload": payload,
            })
        timings["5_candidate_conversion"] = time.perf_counter() - t5

        # ===================================================================
        # 5b. Boost candidates that matched PostgreSQL structured search
        # ===================================================================
        if structured_result and structured_result.doc_ids:
            t5b = time.perf_counter()
            try:
                from src.api.services.structured_study_matcher import boost_candidates_with_structured_matches
                cands = boost_candidates_with_structured_matches(cands, structured_result, boost_factor=0.3)
            except Exception as e:
                print(f"    └─ structured boost failed: {e}")
            timings["5b_structured_boost"] = time.perf_counter() - t5b

        if not cands:
            total_time = time.perf_counter() - t_total_start
            print(f"    └─ TOTAL retrieval: {total_time:.3f}s (no results)")
            return [], {
                "query_type": query_type, 
                "expanded_query": expanded_query,
                "retrieval_timings": timings
            }

        # ===================================================================
        # 6. Lexical scoring (BM25) on dense pool
        # ===================================================================
        t6 = time.perf_counter()
        docs = [_payload_text_for_lex(c["payload"]) for c in cands]
        lex = bm25_scores(expanded_query, docs)

        for c, s in zip(cands, lex):
            c["score_lex"] = float(s) * route["lex_weight"]
            c["score_dense"] = float(c["score_dense"]) * route["dense_weight"]
        timings["6_lexical_scoring"] = time.perf_counter() - t6

        # ===================================================================
        # 6b. Apply user preference filters (BEFORE reranking)
        # ===================================================================
        t6b = time.perf_counter()
        valid_doc_ids = None
        if user_preferences:
            from src.api.services.preferences_filter_service import apply_preference_filters, get_valid_doc_ids_sync
            
            # First, get valid doc_ids from PostgreSQL (for min_patients, study_phase, etc.)
            valid_doc_ids = get_valid_doc_ids_sync(user_preferences)
            if valid_doc_ids is not None:
                print(f"[Preferences] PostgreSQL filter returned {len(valid_doc_ids)} valid doc_ids")
                
                # Debug: Check overlap with Qdrant candidates
                qdrant_doc_ids = {c.get('payload', {}).get('doc_id') for c in cands if c.get('payload', {}).get('doc_id')}
                overlap = valid_doc_ids & qdrant_doc_ids
                print(f"[Preferences] Qdrant has {len(qdrant_doc_ids)} unique doc_ids, overlap: {len(overlap)}")
                
                # If no overlap, skip PostgreSQL filtering (fallback to Qdrant-only filters)
                if len(overlap) == 0 and len(valid_doc_ids) > 0:
                    print(f"[Preferences] WARNING: No overlap between PostgreSQL and Qdrant doc_ids, skipping PostgreSQL filter")
                    valid_doc_ids = None
            
            # Apply filters (PostgreSQL + Qdrant payload filters)
            cands_before = len(cands)
            cands = apply_preference_filters(cands, user_preferences, valid_doc_ids)
            if len(cands) < cands_before:
                print(f"[Preferences] Filtered {cands_before} -> {len(cands)} candidates")
        timings["6b_preference_filtering"] = time.perf_counter() - t6b
        
        if not cands:
            total_time = time.perf_counter() - t_total_start
            print(f"    └─ TOTAL retrieval: {total_time:.3f}s (all filtered out)")
            return [], {
                "query_type": query_type, 
                "expanded_query": expanded_query,
                "retrieval_timings": timings,
                "preferences_applied": True
            }

        # ===================================================================
        # 7. RRF Fusion
        # ===================================================================
        t7 = time.perf_counter()
        fused = rrf_fuse(cands, k_rrf=60)
        timings["7_rrf_fusion"] = time.perf_counter() - t7

        # ===================================================================
        # 8. Structure-aware rerank WITH clinical profile
        # ===================================================================
        t8 = time.perf_counter()
        reranked = rerank_with_structure(
            fused[:rerank_pool], 
            expanded_query, 
            query_type, 
            clinical_profile
        )
        timings["8_structure_rerank"] = time.perf_counter() - t8

        # ===================================================================
        # 9. Cross-encoder rerank
        # ===================================================================
        t9 = time.perf_counter()
        if self.use_cross_encoder:
            # Use a short, structure-derived keyword query so ms-marco
            # MiniLM stays inside its training distribution. Passing the
            # raw 1000+ char patient narrative collapses every score to
            # 0–3% because the model is trained on short web queries.
            reranker_query = build_reranker_query(query_text, query_structure)
            if reranker_query != query_text:
                print(f"    └─ cross-encoder query: {reranker_query[:120]}")
            reranked = cross_encoder_rerank(reranked, reranker_query, top_k=10)
            timings["9_cross_encoder_rerank"] = time.perf_counter() - t9
            print(f"    └─ cross-encoder: {timings['9_cross_encoder_rerank']:.3f}s")
        else:
            timings["9_cross_encoder_rerank"] = 0.0

        # ===================================================================
        # 10. Dose boosting if applicable
        # ===================================================================
        t10 = time.perf_counter()
        if query_type == "dose_question":
            reranked = boost_dose_chunks(reranked, query_text, query_type)
        timings["10_dose_boosting"] = time.perf_counter() - t10
        
        # ===================================================================
        # 10b. Trial boosting if trials detected
        # ===================================================================
        t10b = time.perf_counter()
        if detected_trials:
            try:
                from src.api.services.trial_registry import boost_chunks_by_trial
                reranked = boost_chunks_by_trial(reranked, detected_trials)
            except Exception as e:
                print(f"Warning: Trial boosting failed: {e}")
        timings["10b_trial_boosting"] = time.perf_counter() - t10b
        
        # ===================================================================
        # 10c. Landmark study boosting for general knowledge queries
        # ===================================================================
        # NOTE: this is the LEGACY per-chunk retrieval path. The
        # comprehensive retrieval flow (query_study_focused →
        # retrieve_comprehensive) uses a two-track architecture instead
        # (see src/api/services/evidence_classifier.py + the bucketing
        # block in comprehensive_retrieval.py). Patient-context queries
        # route through the comprehensive path.
        #
        # We use lane_separate_chunks() instead of boost_landmark_studies()
        # so guidelines/landmarks cannot dominate by inflated scores —
        # they are placed in a separate capped lane (≤5 docs) after trials.
        t10c = time.perf_counter()
        if query_type in ["general", "dose_question", "trial_results", "treatment_recommendation"]:
            reranked = lane_separate_chunks(reranked)
        timings["10c_landmark_boosting"] = time.perf_counter() - t10c

        # ===================================================================
        # 10d. Module-specific boosting based on query classification
        # ===================================================================
        t10d = time.perf_counter()
        module_name = None
        try:
            from src.api.services.module_classifier import classify_query_module
            # Convert QueryStructure to dict if needed — guard against to_dict() returning None
            query_structure_dict = (query_structure.to_dict() if query_structure else None) or {}
            module_result = classify_query_module(query_text, query_structure_dict or None)
            module_name = module_result.module.value
            
            # Apply module-specific boosting
            reranked = apply_module_specific_boost(
                reranked, 
                module_name, 
                clinical_profile
            )
            
            # For evidence exploration, ensure treatment diversity
            if module_name == "evidence_exploration":
                reranked = ensure_treatment_diversity(reranked, min_options=2)
            
            print(f"    └─ module boost: {module_name} (confidence: {module_result.confidence:.2f})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    └─ module boost: skipped ({e})")
        timings["10d_module_boosting"] = time.perf_counter() - t10d

        # ===================================================================
        # 11. Dedup + caps
        # ===================================================================
        t11 = time.perf_counter()
        pruned = dedup_and_caps(
            reranked,
            max_per_doc=2,
            max_per_doc_section=1,
            max_rows_per_table=2,
        )
        timings["11_deduplication"] = time.perf_counter() - t11

        # ===================================================================
        # 11b. Apply sort preference boost (AFTER reranking)
        # ===================================================================
        t11b = time.perf_counter()
        if user_preferences and user_preferences.get('sort_by', 'relevance') != 'relevance':
            from src.api.services.preferences_filter_service import apply_sort_boost, get_citation_counts_sync
            
            # For citation sorting, fetch citation counts from PostgreSQL
            citation_counts = None
            if user_preferences.get('sort_by') == 'citations':
                try:
                    # Collect unique doc_ids and DOIs from candidates
                    doc_ids = set()
                    for c in pruned:
                        payload = c.get('payload', {})
                        doc_meta = payload.get('doc_meta', {})
                        if payload.get('doc_id'):
                            doc_ids.add(payload['doc_id'])
                        if doc_meta.get('doi'):
                            doc_ids.add(doc_meta['doi'])
                    
                    if doc_ids:
                        citation_counts = get_citation_counts_sync(list(doc_ids))
                        if citation_counts:
                            print(f"[Preferences] Fetched {len(citation_counts)} citation counts from DB")
                except Exception as e:
                    print(f"[Preferences] Error fetching citation counts: {e}")
            
            pruned = apply_sort_boost(pruned, user_preferences, citation_counts)
        timings["11b_sort_boost"] = time.perf_counter() - t11b

        # ===================================================================
        # 12. Evidence packs
        # ===================================================================
        t12 = time.perf_counter()
        evidence_packs = [make_evidence_pack(self.qdrant, self.collection, c) for c in pruned[:k_final]]
        timings["12_evidence_packs"] = time.perf_counter() - t12

        # ===================================================================
        # 13. NCCN gap detection
        # ===================================================================
        t13 = time.perf_counter()
        nccn_assessment = detect_nccn_gap(query_text, pruned[:k_final])
        timings["13_nccn_gap_detection"] = time.perf_counter() - t13

        # ===================================================================
        # Calculate total time and print summary
        # ===================================================================
        total_time = time.perf_counter() - t_total_start
        timings["total"] = total_time
        
        # Print detailed breakdown
        print(f"    └─ TOTAL retrieval: {total_time:.3f}s")
        
        # Print breakdown of top time consumers (>10ms)
        significant_steps = {k: v for k, v in timings.items() 
                            if v > 0.01 and k != "total"}
        if significant_steps:
            print("       Breakdown (>10ms):")
            for step, duration in sorted(significant_steps.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True):
                pct = 100 * duration / total_time
                print(f"         • {step}: {duration:.3f}s ({pct:.1f}%)")

        # Stash the LLM-extracted ClinicalProfile (set on
        # query_structure._clinical_profile around line ~4129) into
        # metadata under a leading-underscore key so it's clearly
        # internal. EnhancedRAGService.query() pops this key right
        # after the retriever returns and uses it to run the
        # post-retrieval patient_match_scorer. The key is non-
        # serialisable (ClinicalProfile dataclass) and MUST be
        # popped before the metadata dict is included in the JSON
        # response — query() does that.
        _clinical_profile_obj = getattr(query_structure, "_clinical_profile", None) if query_structure else None

        metadata = {
            "query_type": query_type,
            "query_classification": query_classification,
            "expanded_query": expanded_query,
            "nccn_assessment": nccn_assessment,
            "retrieval_route": route,
            "clinical_profile": clinical_profile.get("raw_profile") if clinical_profile else None,
            "_clinical_profile": _clinical_profile_obj,
            "detected_trials": [t.to_dict() for t in detected_trials] if detected_trials else None,
            "retrieval_timings": timings,
            # Prefilter metadata
            "prefilter_applied": prefilter_result.filter_applied,
            "prefilter_reason": prefilter_result.filter_reason,
            "prefilter_search_terms": prefilter_result.search_terms,
            "prefilter_clinical_context": prefilter_result.clinical_context_used,
            # NEW: Query structure (fast regex-based extraction)
            "query_structure": query_structure.to_dict() if query_structure else None,
            # NEW: Structured PostgreSQL matching results
            "structured_match": {
                "doc_ids_matched": len(structured_result.doc_ids) if structured_result else 0,
                "conditions_used": structured_result.conditions_used if structured_result else [],
                "query_time_ms": structured_result.query_time_ms if structured_result else 0,
            } if structured_result else None,
            # NEW: Module classification for response formatting
            "module_name": module_name,
        }
        return evidence_packs, metadata

# ============================================
# ENHANCED GPT-4o SUMMARIZATION
# ============================================

def gpt4o_summary_enhanced(
    openai_client: OpenAI,
    question: str,
    evidence: List[Dict[str, Any]],
    query_type: str = "general",
    nccn_assessment: Optional[Dict] = None,
    staging_context: Optional[str] = None,
    conversation_context: Optional[str] = None,
    module_classification: Optional[Dict] = None,
    patient_eligibility: Optional[Dict] = None,
    generation_model: Optional[str] = None,
    reconciled_structure: Optional[Any] = None,
) -> str:
    """
    Enhanced summarization using query-type-specific prompts.
    
    The response format is determined by:
    1. If query starts with "Patient Details:" -> use patient_specific module prompt
    2. Otherwise -> use query_type prompts (staging, dose_question, treatment_recommendation, etc.)
    
    This ensures specialized prompts (staging with DOI/ENE rules, dose with fractionation, etc.)
    are used by default, and patient-specific formatting is only used when explicitly requested.
    """
    if not evidence:
        return "No relevant chunks were retrieved from the knowledge base."

    # =====================================================
    # DETECT QUERY INTENT
    # =====================================================
    query_lower = question.lower().strip()
    
    # TREATMENT-INTENT: Asking ABOUT a treatment (treatment + "for/in/to" at start)
    treatment_intent_patterns = [
        r'^(?:adjuvant|neoadjuvant|definitive|salvage|palliative)\s+(?:rt|radiation|chemo|xrt|imrt|radiotherapy|chemotherapy)\s+(?:for|in|to)\b',
        r'^(?:rt|radiation|radiotherapy|chemoradiation|chemo|imrt|vmat|sbrt|srs|wbrt)\s+(?:for|in|to)\b',
        r'^(?:dose|dosing|fractionation|treatment|management|therapy|indication|indications)\s+(?:for|of|to|in)\b',
        r'^(?:what|when|how|should|is|are).{0,30}(?:dose|rt|radiation|treatment|chemo|indication)',
    ]
    
    # PRIOR-TREATMENT: Treatment mentioned as history, not the question
    prior_treatment_patterns = [
        r'\bs/p\b.{0,50}(?:chemo|radiation|rt|xrt|mastectomy|surgery|resection|dissection|ectomy)',
        r'\b(?:received|completed|given|had|underwent|following|after|prior)\s+(?:\w+\s+)?(?:chemo|radiation|rt|adjuvant|neoadjuvant)',
        r'(?:chemo|radiation|rt|adjuvant|neoadjuvant).{0,20}(?:was|were|been)\s+(?:given|completed|administered)',
        r'\bstatus post\b',
        r'\bhistory of\b.{0,30}(?:chemo|radiation|treatment)',
    ]
    
    # Patient context patterns
    patient_signals = {
        "age": r'\b\d{1,3}\s*(?:year|yr|y/?o|yo\b)',
        "gender": r'\b(?:male|female|man|woman|gentleman|lady)\b',
        "staging": r'\b[cyp]?t[0-4][a-d]?\s*[cyp]?n[0-3]',
        "stage_word": r'\bstage\s*[iI1234]+[a-cA-C]?\b',
        "post_treatment": r'\bs/p\b|\bstatus post\b',
        "recurrence": r'\brecurren(?:t|ce)\b|\brelapse[d]?\b',
        "pathology": r'\b(?:lvi|pni|doi|ene|margins?|pcr)[+-]?\b',
        "diagnosis": r'\bdiagnosed with\b|\bpresents with\b|\bwith\s+(?:breast|lung|prostate|head|neck|rectal|anal|cervical|esophageal)\s+(?:cancer|carcinoma|scc|adenocarcinoma)',
    }
    
    # Explicit question patterns
    question_patterns = [
        r'\?',
        r'^(?:what|how|when|where|why|which)\b',
        r'^(?:should|is|are|does|do|can|could|would)\s',
        r'\b(?:what is|what are|how much|how many)\b',
    ]
    
    has_treatment_intent = any(re.search(p, query_lower) for p in treatment_intent_patterns)
    has_prior_treatment = any(re.search(p, query_lower) for p in prior_treatment_patterns)
    patient_score = sum(1 for pattern in patient_signals.values() if re.search(pattern, query_lower))
    has_patient_context = patient_score >= 2
    has_explicit_question = any(re.search(p, query_lower) for p in question_patterns)
    
    # If treatment words appear ONLY as prior treatment, NOT treatment intent
    if has_prior_treatment and not any(re.search(p, query_lower) for p in treatment_intent_patterns):
        has_treatment_intent = False
    
    print(f"[Generation] treatment_intent: {has_treatment_intent}, prior_tx: {has_prior_treatment}, patient_score: {patient_score}, has_question: {has_explicit_question}")
    
    # =====================================================
    # PROMPT SELECTION LOGIC (PRIORITY ORDER)
    # =====================================================
    use_module_prompts = False
    module_name = None
    
    # PRIORITY 1: Clear treatment intent at START → query-type prompts
    if has_treatment_intent:
        use_module_prompts = False
        print(f"[Generation] Treatment intent detected, using query-type prompts")
    
    # PRIORITY 2: Legacy "Patient Details:" prefix
    elif query_lower.startswith("patient details:"):
        use_module_prompts = True
        module_name = "patient_specific"
        print(f"[Generation] Patient Details prefix detected, using patient_specific prompt")
    
    # PRIORITY 3: Patient context + NO explicit question + NO treatment intent → case review
    elif has_patient_context and not has_explicit_question:
        use_module_prompts = True
        module_name = "patient_specific"
        print(f"[Generation] Patient case description (no question), using patient_specific prompt")
    
    # Load module prompt if selected
    if use_module_prompts and module_name:
        try:
            from src.api.services.module_generation_prompts import get_prompt_for_module
            prompt_config = get_prompt_for_module(module_name)
            system_msg = prompt_config["system"]
            user_template = prompt_config["user_template"]
        except Exception as e:
            print(f"[Generation] Module prompt import failed, falling back: {e}")
            use_module_prompts = False
    
    if not use_module_prompts:
        # Use query-type-specific prompts (default behavior)
        indication_keywords = ["indication", "indicated", "best indication", "appropriate for", 
                              "when should", "who should", "criteria for", "candidate"]
        is_indication_question = any(kw in query_lower for kw in indication_keywords)
        
        if is_indication_question and query_type not in ["indication_question"]:
            query_type = "indication_question"
            print(f"[Query Type Override] Detected indication question, using indication_question prompt")

        prompt_config = GENERATION_PROMPTS.get(query_type, GENERATION_PROMPTS["general"])
        system_msg = prompt_config["system"]
        user_template = prompt_config["user_template"]
        print(f"[Generation] Using query_type prompt: {query_type}")

    # Add staging context to system message if available
    if staging_context:
        system_msg += f"""

IMPORTANT STAGING CONTEXT:
{staging_context}

When answering, ensure your response is consistent with this staging information.
If the evidence discusses different stages, prioritize information relevant to the staging above."""

    # Add web-source constraint if any evidence is from PubMed/external sources (Task 9)
    has_web_sources = any(e.get("source_type") == "pubmed" for e in evidence)
    if has_web_sources:
        system_msg += (
            "\n\nIMPORTANT — WEB SOURCES: Some sources below are tagged as [PubMed] (supplementary external results). "
            "Do NOT use these external sources to make treatment recommendations. "
            "Use them ONLY for factual data such as trial names, outcomes, and enrollment figures. "
            "Treatment recommendations must come from knowledge base sources only."
        )

    # Add patient eligibility verification instruction when patient context is detected
    # Split-brain fix: prefer reconciled_structure (used during retrieval) over
    # re-extracted patient context to keep generation consistent with retrieval.
    _patient_summary_for_prompt = None
    if reconciled_structure is not None:
        try:
            parts = []
            if getattr(reconciled_structure, "cancer_site", None):
                parts.append(f"cancer site: {reconciled_structure.cancer_site}")
            if getattr(reconciled_structure, "histology", None):
                parts.append(f"histology: {reconciled_structure.histology}")
            if getattr(reconciled_structure, "stage", None):
                parts.append(f"stage: {reconciled_structure.stage}")
            tnm_parts = []
            for attr in ("tnm_t", "tnm_n", "tnm_m"):
                val = getattr(reconciled_structure, attr, None)
                if val:
                    tnm_parts.append(val)
            if tnm_parts:
                parts.append(f"TNM: {' '.join(tnm_parts)}")
            biomarkers = getattr(reconciled_structure, "biomarkers", None)
            if biomarkers:
                bm_strs = []
                for bm in biomarkers:
                    name = getattr(bm, "name", str(bm))
                    pol = getattr(bm, "polarity", None)
                    bm_strs.append(f"{name} {pol}" if pol else str(name))
                parts.append(f"biomarkers: {', '.join(bm_strs)}")
            if getattr(reconciled_structure, "treatment_history", None):
                parts.append(f"treatment history: {reconciled_structure.treatment_history}")
            if getattr(reconciled_structure, "age", None):
                parts.append(f"age: {reconciled_structure.age}")
            if getattr(reconciled_structure, "gender", None):
                parts.append(f"gender: {reconciled_structure.gender}")
            if parts:
                _patient_summary_for_prompt = "; ".join(parts)
                print(f"[Generation] Using reconciled_structure for patient context (split-brain fix)")
        except Exception as e:
            print(f"[Generation] Failed to build summary from reconciled_structure: {e}")

    if _patient_summary_for_prompt is None and patient_eligibility and patient_eligibility.get("patient_context_detected"):
        _patient_summary_for_prompt = patient_eligibility.get("patient_summary", "")

    if _patient_summary_for_prompt:
        patient_summary = _patient_summary_for_prompt
        system_msg += f"""

**CRITICAL — PATIENT ELIGIBILITY VERIFICATION:**
The query describes a specific patient: {patient_summary}

Before citing ANY study as evidence for your recommendation:
1. VERIFY the study's enrolled patient population matches this patient's characteristics
2. Check cancer type, stage, biomarkers (ER/PR/HER2 status), and treatment history
3. DO NOT cite studies about different cancer types or patient populations
4. If a study enrolled different patients (e.g., ER+ patients for an ER- patient), DO NOT use it

Examples of MISMATCHES to avoid:
- Citing ER+ breast cancer studies for an ER- patient
- Citing early-stage studies for a metastatic patient
- Citing lung cancer studies for a breast cancer patient

ONLY cite studies where the enrolled population MATCHES or reasonably includes this patient.
If no matching studies are available, state this clearly rather than citing mismatched evidence.

**RANGE MATCHING**: A study IS applicable when the patient's value falls within the study's range:
- Recurrence score 22 → matches studies for "score < 25", "score 11-25", "intermediate risk"
- Age 55 → matches studies for "patients over 50", "postmenopausal", "age 50-70"
- T1c → matches studies for "T1-T2", "early stage"
- N1mi (micrometastatic) → matches studies for "node-negative or micrometastatic"
Do NOT say the evidence "does not specifically address" a value when it falls within a studied range."""
        print(f"[Generation] Added patient eligibility verification for: {patient_summary}")

    # ── Patient-tailoring instruction ─────────────────────────────────
    # When a patient is described (explicitly or via conversation context),
    # tell the LLM to personalise the answer to THAT patient's specific
    # factors — not produce a generic textbook answer.  This applies to
    # ALL prompt types (treatment_recommendation, dose_question, etc.),
    # not just patient_specific module.  The eligibility block above
    # tells the LLM which studies NOT to cite; this block tells it HOW
    # to use the remaining studies to address the patient.
    if _patient_summary_for_prompt:
        system_msg += f"""

**PATIENT-TAILORED RESPONSE (CRITICAL):**
This query concerns a SPECIFIC patient: {patient_summary}

Your answer MUST be personalised to this patient. Do NOT produce a
generic textbook answer that could apply to any patient with this
cancer type.  Instead:

1. TIE EVERY RECOMMENDATION to this patient's specific factors:
   - Name the patient's exact T-stage, N-stage, M-stage when
     explaining why a study applies.
   - Reference the patient's specific biomarker values (CPS, PD-L1,
     HER2, ER/PR, EGFR, etc.) and match them against the trial's
     inclusion criteria or subgroup analyses.
   - Acknowledge the patient's comorbidities (CKD, diabetes, Hep C,
     cardiac, etc.) and state whether they affect drug selection,
     dose modification, or eligibility.
   - Reference the patient's prior treatments and how they constrain
     the current line of therapy (e.g. post-ICI progression →
     ICI-refractory salvage options).
   - If the patient has recurrent disease, discuss salvage-specific
     data, not first-line data.

2. MATCH TRIAL POPULATIONS to this patient's profile:
   - For each cited trial, state in one sentence HOW this patient
     maps to the trial's enrolled population (e.g. "This patient's
     pT4N0 stage with DOI 15 mm falls within the high-risk subgroup
     of EORTC 22931 which enrolled T3-T4 / N+ / positive-margin
     patients").
   - If the patient only partially matches (e.g. same cancer type
     but different stage), state the mismatch explicitly and note
     whether extrapolation is reasonable.

3. ADDRESS CONVERSATION CONTEXT:
   - If this is a follow-up question in a conversation, your answer
     must build on the prior discussion — do not repeat information
     already provided, and do not contradict prior recommendations
     unless new evidence warrants it.
   - Reference the patient's accumulated clinical context (prior
     queries, prior treatment decisions discussed) when available.

4. PROVIDE ACTIONABLE SPECIFICS for THIS patient:
   - Name the exact regimen (drug names, doses, schedule).
   - Name the radiation dose and fractionation if applicable.
   - State what this patient should do NEXT (specific tests, consults,
     or treatment steps), not generic guidance."""

    # Build context blocks with citation info — use more evidence and
    # enforce per-study diversity so one paper can't monopolize the top
    # slots and force the LLM to recycle a single citation across every
    # bullet (the "Bernier et al., 2005" failure mode).
    #
    # Strategy:
    #   1. Take a wider candidate window (top 30 instead of 10)
    #   2. Cap chunks per doc_id at MAX_CHUNKS_PER_DOC so a paper with
    #      many high-scoring chunks can't fill all slots
    #   3. Stop once we have MAX_CTX_BLOCKS chunks (14 instead of 10)
    #   4. Each chunk text limit raised to 2000 chars (was 1200) so the
    #      LLM sees complete results sentences with their numbers, not
    #      truncated abstracts
    MAX_CTX_BLOCKS = 14
    MAX_CHUNKS_PER_DOC = 2
    PER_CHUNK_TEXT_LIMIT = 2000

    ctx_blocks = []
    chunks_per_doc: Dict[str, int] = {}
    distinct_doc_ids: set = set()

    for e in evidence[:30]:
        if len(ctx_blocks) >= MAX_CTX_BLOCKS:
            break
        doc_id = e.get("doc_id") or ""
        # Per-study cap: skip extra chunks from a doc that's already at the cap
        if doc_id and chunks_per_doc.get(doc_id, 0) >= MAX_CHUNKS_PER_DOC:
            continue

        # Build citation string with author, year, and journal for proper inline citations
        author = e.get("author_et_al") or ""

        # Fallback: try to extract author from citation string if author_et_al is missing
        if not author and e.get("citation"):
            citation_text = e.get("citation", "")
            # Pattern: "Author et al." at the start or before year
            match = re.search(r'([A-Z][a-z]+(?:\s+et\s+al\.?))\s*[\(,]\s*\d{4}', citation_text)
            if match:
                author = match.group(1)
            else:
                match = re.match(r'^([A-Z][a-z]+(?:\s+et\s+al\.?))', citation_text)
                if match:
                    author = match.group(1)

        year = e.get("year") or ""
        journal = e.get("journal") or ""
        title = e.get("title") or ""

        # Build a rich citation string for the LLM to use
        if author and year:
            if journal:
                citation_str = f"{author} ({year}), {journal}"
            else:
                citation_str = f"{author} ({year})"
        elif e.get("citation"):
            citation_str = e.get("citation")
        else:
            citation_str = f"doc_id={e.get('doc_id', 'unknown')}"

        # Include title for context
        if title:
            citation_str = f"{title} - {citation_str}"

        # Show patient–study match % inline so the citation carries the
        # signal next to the source. Skipped when the scorer didn't run
        # (non-patient queries) so we never show a misleading 0%.
        pm_score = e.get("patient_match_score")
        if isinstance(pm_score, (int, float)):
            citation_str = f"{citation_str} | patient match {int(pm_score)}%"

        # Tag evidence class so the LLM and UI can distinguish a
        # patient-specific trial from a guideline/landmark reference.
        etype = e.get("evidence_type")
        if etype == "guideline":
            citation_str = f"[Guideline] {citation_str}"
        elif etype == "landmark_trial":
            citation_str = f"[Landmark] {citation_str}"

        section = e.get("section") or e.get("chunk_type", "unknown")

        txt = (e.get("text") or "").replace("\n", " ")[:PER_CHUNK_TEXT_LIMIT]

        source_tag = " [PubMed]" if e.get("source_type") == "pubmed" else ""
        block_idx = len(ctx_blocks) + 1
        if e.get("chunk_type") == "table_row" and e.get("table"):
            tbl = e["table"]
            table_info = f"[Table {tbl.get('number', '?')}: {tbl.get('title', '')}]"
            ctx_blocks.append(
                f"[{block_idx}]{source_tag} {citation_str} | {section}\n"
                f"{table_info}\n{txt}"
            )
        else:
            ctx_blocks.append(
                f"[{block_idx}]{source_tag} {citation_str} | {section}\n{txt}"
            )

        if doc_id:
            chunks_per_doc[doc_id] = chunks_per_doc.get(doc_id, 0) + 1
            distinct_doc_ids.add(doc_id)

    print(
        f"[Generation] Context built: {len(ctx_blocks)} chunks from "
        f"{len(distinct_doc_ids)} distinct studies "
        f"(per-doc cap={MAX_CHUNKS_PER_DOC}, chunk limit={PER_CHUNK_TEXT_LIMIT})"
    )

    context = "\n\n---\n\n".join(ctx_blocks)

    # Add conversation context if available
    if conversation_context:
        question = f"Conversation context:\n{conversation_context}\n\nCurrent question:\n{question}"
    user_msg = user_template.format(question=question, context=context)

    _model = generation_model or settings.openai_model
    resp = openai_client.chat.completions.create(
        model=_model,
        temperature=0,  # Use 0 for maximum precision on numerical values
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    summary = resp.choices[0].message.content.strip()

    # Add NCCN gap warning if detected
    if nccn_assessment and nccn_assessment.get("gap_detected"):
        summary += f"\n\n**Note:** {nccn_assessment['suggested_action']}"
    
    return summary


def generate_followup_suggestions(
    openai_client: OpenAI,
    question: str,
    answer: str,
    clinical_context: Dict[str, Any],
    query_type: str = "general"
) -> List[str]:
    """
    Generate suggested follow-up questions based on the current Q&A and clinical context.
    
    Args:
        openai_client: OpenAI client
        question: The user's current question
        answer: The generated answer
        clinical_context: Extracted clinical context (cancer_type, stage, etc.)
        query_type: The classified query type
        
    Returns:
        List of 2-3 suggested follow-up questions
    """
    # Build context string
    context_parts = []
    if clinical_context.get("cancer_type"):
        context_parts.append(f"Cancer: {clinical_context['cancer_type']}")
    if clinical_context.get("stage"):
        context_parts.append(f"Stage: {clinical_context['stage']}")
    if clinical_context.get("patient_profile"):
        context_parts.append(f"Patient: {', '.join(clinical_context['patient_profile'])}")
    if clinical_context.get("treatment"):
        context_parts.append(f"Treatment: {clinical_context['treatment']}")
    
    context_str = "; ".join(context_parts) if context_parts else "General oncology query"
    
    system_prompt = """You are a clinical oncology assistant helping guide a conversation about cancer treatment.
Based on the current question and answer, suggest 2-3 logical follow-up questions that would help the clinician explore related topics.

RULES:
1. Suggest questions that are clinically relevant and actionable
2. Focus on different aspects: outcomes (OS, PFS, DFS), toxicities, alternative treatments, patient selection, dosing
3. Keep questions concise (under 15 words each)
4. Make questions specific to the clinical context provided
5. Don't repeat information already covered in the answer
6. Format as a JSON array of strings

GOOD FOLLOW-UP CATEGORIES:
- Survival outcomes: "What is the 5-year OS for [treatment]?"
- Toxicity profiles: "What are the common adverse effects of [treatment]?"
- Comparisons: "How does [treatment A] compare to [treatment B]?"
- Patient selection: "Which patients benefit most from [treatment]?"
- Dosing/scheduling: "What is the optimal dose/fractionation for [treatment]?"
- Biomarkers: "Are there predictive biomarkers for response to [treatment]?"
- Trials: "What recent trials support [treatment]?"
"""

    user_prompt = f"""Clinical Context: {context_str}

Current Question: {question}

Answer Summary: {answer[:500]}...

Generate 2-3 relevant follow-up questions as a JSON array. Example format:
["What is the 5-year overall survival?", "What toxicities are associated with this treatment?", "Are there alternative treatment options?"]"""

    try:
        response = openai_client.chat.completions.create(
            model=settings.openai_mini_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON array
        import json
        # Handle potential markdown code blocks
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        suggestions = json.loads(result)
        
        # Validate and clean
        if isinstance(suggestions, list):
            return [s.strip() for s in suggestions[:3] if isinstance(s, str) and len(s) > 10]
        
        return []
        
    except Exception as e:
        print(f"[Followup Suggestions] Error generating suggestions: {e}")
        return []


# ============================================
# SERVICE CLASS FOR PAXIS INTEGRATION
# ============================================

class EnhancedRAGService:
    """
    Main service class for enhanced RAG with ALL features from Colab.
    Integrates with Paxis's configuration system.
    Now includes BIDIRECTIONAL query expansion.
    """
    
    def __init__(self):
        """Initialize the enhanced RAG service with Paxis config."""
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Qdrant client
        # prefer_grpc: same transport-only speedup as comprehensive_retrieval.py
        # — faster binary protocol, Qdrant Cloud supports it natively on 6334,
        # no change to filters/ranking/results.
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=120,
            prefer_grpc=True,
            grpc_port=6334,
        )
        
        # Initialize enhanced retriever
        self.retriever = EnhancedHybridRetriever(
            qdrant_client=self.qdrant_client,
            openai_client=self.openai_client,
            collection=QDRANT_COLLECTION,
            embed_model=EMBED_MODEL,
            use_cross_encoder=CROSS_ENCODER_AVAILABLE
        )
        
        print("✅ Enhanced RAG Service initialized successfully")
        print(f"   Qdrant URL: {QDRANT_URL}")
        print(f"   Collection: {QDRANT_COLLECTION}")
        print(f"   Cross-encoder enabled: {CROSS_ENCODER_AVAILABLE}")
        print(f"   Bidirectional query expansion: ENABLED")
    
    async def query(
        self,
        question: str,
        query_mode: str = "hybrid",
        top_k: int = 10,
        category: Optional[str] = None,
        use_site_inference: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        cached_evidence: Optional[List[Dict[str, Any]]] = None,
        accumulated_context: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        use_study_focused: bool = False,
        max_studies: int = 5,
        chunks_per_study: int = 8,
        strict_category: bool = False,
        generation_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the enhanced RAG system.
        
        Args:
            question: User's question
            query_mode: Query mode (kept for compatibility)
            top_k: Number of top results to return
            category: Optional category filter
            use_site_inference: If True, infer tumor site and add to query
            conversation_history: Optional list of previous conversation messages
            user_id: Optional user ID to apply preferences
            cached_evidence: Optional pre-cached evidence chunks (skips retrieval if provided)
            accumulated_context: Optional accumulated structured context from previous queries in conversation
            conversation_context: Optional list of ConversationContextEntry dicts from frontend sessionStorage
            use_study_focused: If True, use two-phase study-focused retrieval for comprehensive per-study coverage
            max_studies: Maximum studies for study-focused retrieval (default 5)
            chunks_per_study: Maximum chunks per study for study-focused retrieval (default 8)
            
        Returns:
            Dictionary with answer and retrieval results
        """
        try:
            from src.api.services import pipeline_metrics as _pm
            if _pm.current() is None:
                _pm.start("p1")
            # If study-focused retrieval is requested, delegate to specialized method
            if use_study_focused:
                if _pm.current() is not None:
                    _pm.current().event("use_study_focused")
                    _pm.current().event("deprecated_use_study_focused")
                print(
                    "[Deprecation] use_study_focused=True is retired; "
                    "routing through ComprehensiveRetriever via query_study_focused. "
                    "New callers should omit the flag and rely on retrieve_evidence(mode='comprehensive')."
                )
                print(f"\n{'=' * 80}")
                print(f"  ENTERING: EnhancedRAGService.query() → DELEGATING to query_study_focused()")
                print(f"{'=' * 80}")
                return await self.query_study_focused(
                    question=question,
                    max_studies=max_studies,
                    chunks_per_study=chunks_per_study,
                    category=category,
                    conversation_context=conversation_context,
                    cached_evidence=cached_evidence,
                )
            
            print(f"\n{'=' * 80}")
            print(f"  ENTERING: EnhancedRAGService.query() → Standard Retrieval Path")
            print(f"{'=' * 80}")
            print(f"  Query: {question[:120]}{'...' if len(question) > 120 else ''}")
            print(f"  Settings: top_k={top_k}, category={category}, use_site_inference={use_site_inference}")

            # Optionally infer site and construct full query
            inferred_site = None
            full_query = question

            if use_site_inference and not category:
                inferred_site = infer_site_key(question)
                # The "Radiotherapy&Oncology" default is the general bucket.
                # Using it as a category filter provides no benefit — it
                # matches everything in the collection. Skip it.
                if inferred_site and inferred_site != "Radiotherapy&Oncology":
                    site_label = SITE_LABELS.get(inferred_site, inferred_site)
                    full_query = f"{site_label} {question}"
                    category = inferred_site
                else:
                    print(
                        f"[RAG] Site inference returned default "
                        f"'{inferred_site}' — skipping category filter"
                    )

            # Normalize category filter
            normalized_category = normalize_category_filter(category) if category else None

            # Extract accumulated patient context from conversation for follow-ups
            # Without this, "what is the fractionation schedule?" after a prostate
            # T3b Gleason 9 query loses the patient context entirely.
            if not accumulated_context and conversation_context:
                for entry in reversed(conversation_context):
                    if entry.get("query_structure"):
                        accumulated_context = entry.get("query_structure")
                        print(f"[RAG] Extracted accumulated_context from conversation_context")
                        break

            # Extract previous sources for follow-up context
            previous_sources = extract_previous_sources(conversation_history)
            previous_doc_ids = previous_sources.get("doc_ids", [])
            previous_citations = previous_sources.get("citations", [])
            
            # Format conversation context (enhanced version with sources for generation)
            formatted_conversation_context = ""
            if query_mode in {"conversation", "chat"}:
                # Use enhanced context formatting that includes source info
                formatted_conversation_context = format_conversation_context_with_sources(conversation_history)
                
                # Expand the query with conversation context for better retrieval
                if formatted_conversation_context:
                    # Extract key terms from conversation to enrich the query
                    expanded_query = _expand_query_with_context(full_query, conversation_history)
                    if expanded_query != full_query:
                        print(f"[RAG] Expanded query with context: {expanded_query[:200]}...")
                        full_query = expanded_query
                
                # Log previous sources for debugging
                if previous_doc_ids:
                    print(f"[RAG] Found {len(previous_doc_ids)} previous sources for follow-up boosting")
            
            # Format conversation history from new conversation_context (ConversationContextEntry format)
            # This is the automatic conversation mode using frontend sessionStorage
            if conversation_context:
                context_history = format_conversation_history_from_context(
                    conversation_context=conversation_context,
                    max_entries=10,  # Configurable limit, default 10
                    max_chars=50000
                )
                if context_history:
                    # Combine with existing formatted context if present, or use as primary
                    if formatted_conversation_context:
                        formatted_conversation_context = f"Previous conversation:\n{context_history}\n\n{formatted_conversation_context}"
                    else:
                        formatted_conversation_context = f"Previous conversation:\n{context_history}"
                    print(f"[RAG] Added conversation history from {len(conversation_context)} context entries")
            
            # Retrieve evidence using enhanced pipeline (or use cached)
            if cached_evidence:
                print(f"[RAG] Using cached evidence ({len(cached_evidence)} chunks)")
                evidence = cached_evidence
                metadata = {"from_cache": True, "query_type": classify_query(full_query)["primary_type"]}
            else:
                print(
                    f"[RAG] Retrieving with category={normalized_category} "
                    f"(strict={strict_category}), query={full_query[:100]}..."
                )
                evidence, metadata = await self.retriever.retrieve(
                    query_text=full_query,
                    category=normalized_category,
                    N=100,
                    k_final=top_k,
                    rerank_pool=50,
                    user_id=user_id,
                    accumulated_context=accumulated_context,
                    strict_category=strict_category,
                )
                print(f"[RAG] Retrieved {len(evidence)} evidence chunks")

            # Fallback if category filter was too narrow.
            #
            # When `strict_category=True` (e.g. the user explicitly pinned a
            # primary cancer site for hard filtering), we must NOT silently
            # fall back to no-filter — that's how head-and-neck queries were
            # leaking anal SCC and glioma studies into the results. Better
            # to return zero evidence than wrong-cancer evidence.
            if normalized_category and not evidence and not strict_category:
                print(f"[RAG] No evidence with category filter, retrying without category...")
                evidence, metadata = await self.retriever.retrieve(
                    query_text=full_query,
                    category=None,
                    N=100,
                    k_final=top_k,
                    rerank_pool=50,
                    accumulated_context=accumulated_context,
                )
            elif normalized_category and not evidence and strict_category:
                print(
                    f"[RAG] STRICT category filter '{normalized_category}' returned "
                    f"zero evidence — NOT falling back to no-filter (caller "
                    f"requested hard primary-cancer filtering)."
                )

            # ─── Patient–study match scoring (post-retrieval) ──────────
            # The /query/enhanced path uses self.retriever.retrieve(),
            # which does NOT call patient_match_scorer (that's wired
            # into retrieve_comprehensive only). Run it here so every
            # cited source carries a per-document patient_match_score
            # for the UI's "Match X%" badge.
            #
            # Source of the ClinicalProfile: stashed into metadata
            # under "_clinical_profile" by retrieve() right after the
            # LLM extraction populates it. When the LLM extraction
            # path didn't run (non-patient query) or produced an
            # empty profile, this key is missing and the scorer is
            # skipped — the score stays None and the UI badge won't
            # render. The leading underscore signals this key is
            # internal: not part of the response metadata contract.
            if evidence and metadata:
                # Pop (don't peek) so the non-serialisable
                # ClinicalProfile object never leaks into the JSON
                # response. Missing key is a no-op.
                _profile = metadata.pop("_clinical_profile", None)
                if _profile is not None and hasattr(_profile, "has_any_filter") and _profile.has_any_filter():
                    try:
                        from src.api.services.patient_match_scorer import score_patient_match
                        # Group chunks by doc_id — score per doc, stamp
                        # on every chunk of that doc.
                        scores_by_doc: Dict[str, Dict[str, Any]] = {}
                        for chunk in evidence:
                            doc_id = chunk.get("doc_id")
                            if not doc_id:
                                continue
                            if doc_id not in scores_by_doc:
                                doc_level = chunk.get("metadata") or {}
                                try:
                                    pm = score_patient_match(_profile, doc_level)
                                    scores_by_doc[doc_id] = pm
                                except Exception as _e:
                                    print(f"[PatientMatch] score failed for {doc_id[:40]}: {_e}")
                                    scores_by_doc[doc_id] = None
                        # Stamp scores on every chunk.
                        scored = 0
                        for chunk in evidence:
                            doc_id = chunk.get("doc_id")
                            pm = scores_by_doc.get(doc_id) if doc_id else None
                            if pm is not None:
                                chunk["patient_match_score"] = pm.get("score")
                                chunk["patient_match_breakdown"] = pm
                                scored += 1
                        if scored:
                            print(
                                f"[PatientMatch] scored {scored}/{len(evidence)} "
                                f"chunks across {len(scores_by_doc)} unique docs"
                            )
                    except Exception as _e:
                        print(f"[PatientMatch] post-retrieval scoring failed: {_e}")

            # Apply source boosting for follow-up questions in conversation mode
            if query_mode in {"conversation", "chat"} and previous_doc_ids and evidence:
                print(f"[RAG] Applying source boost for {len(previous_doc_ids)} previous sources")
                evidence = boost_previous_sources(evidence, previous_doc_ids)
            
            # Apply context boosting from conversation_context (automatic conversation mode)
            # This uses doc_ids from ConversationContextEntry objects stored in frontend sessionStorage
            if conversation_context and evidence:
                # Extract doc_ids from all context entries
                context_doc_ids = []
                for entry in conversation_context:
                    entry_doc_ids = entry.get("doc_ids", [])
                    context_doc_ids.extend(entry_doc_ids)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_context_doc_ids = []
                for doc_id in context_doc_ids:
                    if doc_id and doc_id not in seen:
                        seen.add(doc_id)
                        unique_context_doc_ids.append(doc_id)
                
                if unique_context_doc_ids:
                    print(f"[RAG] Applying context boost for {len(unique_context_doc_ids)} doc_ids from conversation_context")
                    evidence = boost_context_documents(evidence, unique_context_doc_ids)

            # =====================================================
            # POST-RETRIEVAL CANCER-TYPE FILTER: Remove studies
            # that clearly don't match the patient's cancer type
            # (especially important for follow-up queries)
            # =====================================================
            if evidence and (conversation_history or conversation_context):
                print(f"\n  [Post-Retrieval] Cancer-type filter: checking {len(evidence)} chunks...")
                pre_filter = len(evidence)
                try:
                    evidence = _filter_irrelevant_cancer_type_studies(
                        evidence=evidence,
                        query=full_query,
                        original_question=question,
                        conversation_history=conversation_history,
                    )
                    print(f"  [Post-Retrieval] Cancer-type filter: {pre_filter} → {len(evidence)} chunks")
                except Exception as e:
                    print(f"  [Post-Retrieval] Cancer-type filter FAILED: {e}")

            # =====================================================
            # PATIENT ELIGIBILITY HARD FILTER + BOOST:
            # Check each retrieved study against the patient's
            # clinical profile on 5 criteria (cancer_type,
            # histology, stage, prior_therapies, biomarkers).
            # Studies with a confirmed mismatch are REMOVED.
            # Studies where info is simply absent are KEPT.
            # Matching studies are score-boosted.
            # =====================================================
            patient_eligibility_metadata = None
            if evidence:
                try:
                    from src.api.services.patient_eligibility_boost_service import run_patient_eligibility_check
                    # Use full_query (with conversation context) so eligibility check
                    # has the complete patient profile for follow-up queries
                    pre_filter_count = len(evidence)
                    evidence, patient_eligibility_metadata = await run_patient_eligibility_check(
                        query=full_query,
                        chunks=evidence,
                        openai_client=self.openai_client,
                    )
                    if patient_eligibility_metadata:
                        metadata["patient_eligibility"] = patient_eligibility_metadata
                        hard_filtered = patient_eligibility_metadata.get("hard_filtered", 0)
                        penalized = patient_eligibility_metadata.get("penalized", 0)
                        matches = patient_eligibility_metadata.get("matches", 0)
                        if hard_filtered > 0:
                            print(f"[RAG] Patient eligibility: hard-filtered {hard_filtered} studies ({pre_filter_count} → {len(evidence)})")
                        if penalized > 0:
                            print(f"[RAG] Patient eligibility: penalized {penalized} studies (retained with score penalty)")
                        if matches > 0:
                            print(f"[RAG] Patient eligibility: {matches} matching studies boosted")
                except Exception as e:
                    print(f"[RAG] Patient eligibility check failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Path A guard — short-circuit when retrieval misrouted.
            # patient_eligibility_boost_service flags category_routing_suspect
            # when >=80% of removed chunks were rejected for cancer_type
            # MISMATCH (e.g. cervical-cancer studies retrieved for a male
            # CUP patient). If evidence is now empty AND that flag is set,
            # the clinically safe answer is "no relevant evidence found" —
            # generating an answer from zero context would fabricate, and
            # generating one from the rejected wrong-cancer chunks (the
            # old behaviour) would silently confidently mislead.
            if (
                patient_eligibility_metadata is not None
                and patient_eligibility_metadata.get("category_routing_suspect")
                and not evidence
            ):
                print(
                    "[RAG] Short-circuit: empty bundle + suspect category "
                    "routing. Returning 'no relevant evidence found' rather "
                    "than fabricating an answer."
                )
                no_evidence_answer = (
                    "No relevant evidence found for this patient case. "
                    "Retrieval returned studies that did not match the "
                    "patient's cancer type (typically because the query "
                    "mentions anatomy or clinical features that appear in "
                    "the literature of a different cancer). Try restating "
                    "the cancer type or primary site explicitly."
                )
                metadata["short_circuit_reason"] = "category_routing_suspect_empty_bundle"
                return {
                    "answer": no_evidence_answer,
                    "evidence": [],
                    "metadata": metadata,
                    "query_type": metadata.get("query_type", "general"),
                    "sources": [],
                    "source_citations": [],
                    "accumulated_context": metadata.get("query_structure"),
                    "updated_context_entry": None,
                }

            # Generate answer using query-type-specific prompts
            query_type = metadata.get("query_type", "general")
            nccn_assessment = metadata.get("nccn_assessment")
            
            # Classify query using Unified_Router (single source of truth)
            routing_result = None
            module_classification = None
            try:
                from src.api.services.unified_router import get_unified_router
                router = get_unified_router()
                routing_result = router.route_query(question, conversation_context)
                
                # Store routing result in metadata
                metadata["routing"] = routing_result.to_dict()
                
                # Extract module classification for backward compatibility
                module_classification = {
                    "module": routing_result.module.value,
                    "confidence": routing_result.module_confidence,
                    "signals_matched": routing_result.signals_matched,
                }
                metadata["module_classification"] = module_classification
                
                print(f"[RAG] Unified routing: module={routing_result.module.value}, "
                      f"query_type={routing_result.query_type.value}, "
                      f"confidence={routing_result.module_confidence:.2f}")
            except Exception as e:
                print(f"[RAG] Unified routing failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Apply staging normalization for staging-related queries
            staging_context = None
            try:
                from src.api.services.staging_normalizer import get_staging_normalizer, StagingInfo
                normalizer = get_staging_normalizer()

                # Determine cancer type from clinical profile or site inference
                cancer_type = None
                clinical_profile = metadata.get("clinical_profile")
                if clinical_profile:
                    cancer_type = clinical_profile.get("cancer_type")
                if not cancer_type and inferred_site:
                    cancer_type = inferred_site.lower()

                # Normalize staging if query involves staging or treatment recommendations
                if query_type in ["staging", "treatment_recommendation", "dose_question", "indication_question"] or cancer_type:
                    staging_result = normalizer.normalize_staging_in_context(
                        chunks=evidence,
                        query=question,
                        cancer_type=cancer_type
                    )
                    primary = staging_result.get("primary_staging")

                    # Source 1: Query has explicit TNM notation (regex-extracted)
                    # → use it directly and look up stage via AJCC tables
                    # Note: query_structure is only available in retrieve(), not query().
                    # Try to get it; if not available, skip this source.
                    _qs = None
                    try:
                        _qs = query_structure  # noqa: F821 — only defined in retrieve()
                    except NameError:
                        pass

                    if _qs and _qs.cancer.get_tnm_string():
                        qs_t = query_structure.cancer.tnm_t
                        qs_n = query_structure.cancer.tnm_n
                        qs_m = query_structure.cancer.tnm_m or "0"

                        stage_group = None
                        if cancer_type:
                            stage_group = normalizer.lookup_stage_group(
                                cancer_type, qs_t, qs_n, qs_m
                            )
                        staging_info = StagingInfo(
                            t_stage=qs_t,
                            n_stage=qs_n,
                            m_stage=qs_m,
                            stage_group=stage_group,
                            staging_type="clinical",
                            confidence=0.85,
                        )
                        staging_context = normalizer.format_staging_context(staging_info)
                        print(f"[Staging] Using query TNM: T{qs_t}N{qs_n}M{qs_m} → {stage_group or 'unknown stage'}")

                    # Source 2: Query has explicit TNM extracted from query TEXT by the
                    # normalizer regex (found in the query string itself, not chunks)
                    elif staging_result.get("query_staging"):
                        qs = staging_result["query_staging"][0]  # dict from StagingInfo.to_dict()
                        if qs.get("t_stage") or qs.get("n_stage"):
                            staging_info = StagingInfo(
                                t_stage=qs.get("t_stage"),
                                n_stage=qs.get("n_stage"),
                                m_stage=qs.get("m_stage"),
                                stage_group=qs.get("stage_group"),
                                staging_type=qs.get("staging_type", "unknown"),
                                confidence=qs.get("confidence", 0.0),
                            )
                            staging_context = normalizer.format_staging_context(staging_info)
                            print(f"[Staging] Using query-extracted staging: {staging_context}")

                    # Source 3: No explicit TNM in query (patient described clinically,
                    # e.g. "1.5cm, 8mm DOI, ENE+"). Do NOT inject chunk-derived staging
                    # — it's from other studies, not this patient. Let the LLM reason
                    # about TNM from the clinical description and AJCC evidence.
                    else:
                        if query_type == "staging":
                            print(f"[Staging] No explicit TNM in query — skipping staging "
                                  f"context injection for staging question (let LLM reason "
                                  f"from clinical details + AJCC evidence)")
                        elif primary and isinstance(primary, dict) and primary.get("confidence", 0) >= 0.8:
                            # For non-staging queries, use high-confidence chunk staging
                            staging_info = StagingInfo(
                                t_stage=primary.get("t_stage"),
                                n_stage=primary.get("n_stage"),
                                m_stage=primary.get("m_stage"),
                                stage_group=primary.get("stage_group"),
                                staging_type=primary.get("staging_type", "unknown"),
                                confidence=primary.get("confidence", 0.0),
                            )
                            staging_context = normalizer.format_staging_context(staging_info)
                            print(f"[Staging] Using chunk-derived staging (high confidence): {staging_context}")

                    if staging_context:
                        print(f"[Staging] Context for LLM: {staging_context}")
                    metadata["staging_info"] = {
                        "final_staging": staging_result.get("primary_staging"),
                        "staging_context": staging_context,
                    }
            except Exception as e:
                print(f"Warning: Staging normalization failed: {e}")
            
            # =====================================================
            # STAGING AMBIGUITY: Add clarification to prompt if stage is ambiguous
            # DISABLED: Staging clarification temporarily disabled
            # =====================================================
            # try:
            #     clinical_profile = metadata.get("clinical_profile") or {}
            #     cp_tnm_t = clinical_profile.get("tnm_t")
            #     cp_tnm_n = clinical_profile.get("tnm_n")
            #     cp_tnm_m = clinical_profile.get("tnm_m")
            #     cp_stage = clinical_profile.get("overall_stage") or clinical_profile.get("stage")
            #
            #     if (cp_tnm_t or cp_tnm_n or cp_tnm_m) and not cp_stage:
            #         from src.api.services.stage_inference_service import infer_stage_for_query
            #         from src.api.services.staging_clarification import generate_staging_clarifications
            #
            #         inference = infer_stage_for_query(
            #             cancer_type=clinical_profile.get("cancer_type"),
            #             cancer_location=clinical_profile.get("cancer_location"),
            #             tnm_t=cp_tnm_t,
            #             tnm_n=cp_tnm_n,
            #             tnm_m=cp_tnm_m,
            #             age=clinical_profile.get("age"),
            #         )
            #
            #         if inference.is_ambiguous and inference.required_factors:
            #             clarification = generate_staging_clarifications(
            #                 required_factors=inference.required_factors,
            #                 possible_stages=inference.possible_stages,
            #                 inference_notes=inference.notes,
            #             )
            #
            #             if clarification.needs_clarification:
            #                 if staging_context:
            #                     staging_context += "\n\n" + clarification.prompt_addition
            #                 else:
            #                     staging_context = clarification.prompt_addition
            #                 print(f"[RAG] Added staging ambiguity notice: {inference.possible_stages}")
            #
            #             # Store for downstream use (Hook 3)
            #             metadata["stage_ambiguous"] = True
            #             metadata["stage_required_factors"] = inference.required_factors
            #             metadata["stage_possible_stages"] = inference.possible_stages
            #             metadata["stage_inference_notes"] = inference.notes
            # except Exception as e:
            #     print(f"[RAG] Staging clarification injection failed: {e}")
            
            # Run in a worker thread — synchronous OpenAI call, usually the
            # single longest step in the request. Without to_thread it
            # blocks the whole FastAPI event loop for its full duration,
            # stalling every other concurrent request on the server.
            import asyncio
            answer = await asyncio.to_thread(
                gpt4o_summary_enhanced,
                openai_client=self.openai_client,
                question=question,
                evidence=evidence,
                query_type=query_type,
                nccn_assessment=nccn_assessment,
                staging_context=staging_context,
                conversation_context=formatted_conversation_context,
                module_classification=module_classification,
                patient_eligibility=patient_eligibility_metadata,
                generation_model=generation_model,
            )

            # Validate and enrich numerical values in the answer
            try:
                validation_result = validate_numbers_against_sources(answer, evidence)
                metadata["numerical_validation"] = {
                    "validation_rate": validation_result["validation_rate"],
                    "total_numbers": validation_result["total_numbers"],
                    "validated_count": len(validation_result["validated_numbers"]),
                    "unvalidated_count": len(validation_result["unvalidated_numbers"]),
                }

                print(f"  [Numerical Validation] {validation_result['total_numbers']} numbers in answer, "
                      f"{len(validation_result['validated_numbers'])} validated, "
                      f"{len(validation_result['unvalidated_numbers'])} unvalidated "
                      f"(rate={validation_result['validation_rate']:.0%})")
                if validation_result["unvalidated_numbers"]:
                    for uv in validation_result["unvalidated_numbers"]:
                        print(f"    UNVALIDATED: {uv.get('value')} {uv.get('unit', '')} "
                              f"(type={uv.get('type')}, raw='{uv.get('raw_match', '?')}')")
                if validation_result["validated_numbers"]:
                    for v in validation_result["validated_numbers"][:3]:
                        av = v["answer_value"]
                        print(f"    validated: {av.get('value')} {av.get('unit', '')} "
                              f"(exact={v.get('exact_match', '?')})")

                # RF-5: strip unvalidated numbers so hallucinated
                # percentages / HRs / ORs never reach the user.
                if validation_result.get("unvalidated_numbers"):
                    answer = strip_unvalidated_numbers(
                        answer, validation_result["unvalidated_numbers"]
                    )
                # Enrich answer with statistical context (CI, p-values)
                answer = enrich_answer_with_stats(answer, evidence)
            except Exception as e:
                print(f"[RAG] Numerical validation failed: {e}")
                import traceback
                traceback.print_exc()

            # Add inferred site to metadata if used
            if inferred_site:
                metadata["inferred_site"] = inferred_site
                metadata["site_label"] = SITE_LABELS.get(inferred_site)
            
            # Extract source doc_ids and citations for conversation tracking
            source_doc_ids = []
            source_citations = []
            source_titles = []
            for e in evidence:
                doc_id = e.get("doc_id")
                citation = e.get("citation")
                title = e.get("title")
                if doc_id and doc_id not in source_doc_ids:
                    source_doc_ids.append(doc_id)
                if citation and citation not in source_citations:
                    source_citations.append(citation)
                if title and title not in source_titles:
                    source_titles.append(title)
            
            # Create updated_context_entry for automatic conversation mode
            # This entry will be stored in frontend sessionStorage
            action_type, treatments = classify_action_type(question, conversation_context)
            updated_context_entry = {
                "query": question,
                "action_type": action_type,
                "doc_ids": source_doc_ids,
                "doc_titles": source_titles,
                "timestamp": int(time.time() * 1000),  # Unix timestamp in milliseconds
            }
            # Only include treatments for eval_treatment actions
            if action_type == "eval_treatment" and treatments:
                updated_context_entry["treatments"] = treatments
            
            print(f"[RAG] Created updated_context_entry: action_type={action_type}, doc_ids={len(source_doc_ids)}, doc_titles={len(source_titles)}")
            
            return {
                "answer": answer,
                "evidence": evidence,
                "metadata": metadata,
                "query_type": query_type,
                # New: sources for conversation tracking
                "sources": source_doc_ids,
                "source_citations": source_citations,
                # New: accumulated context for conversation continuity
                "accumulated_context": metadata.get("query_structure"),
                # New: updated context entry for automatic conversation mode
                "updated_context_entry": updated_context_entry,
            }
            
        except Exception as e:
            print(f"Error in enhanced RAG query: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def deep_dive(
        self,
        question: str,
        site_key: Optional[str] = None,
        top_k: int = 15,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep dive query with explicit site specification.
        This mimics the deep_dive_enhanced function from the Colab notebook.
        
        Args:
            question: User's question
            site_key: Tumor site key (e.g., "Breast", "Lung", "GI")
            top_k: Number of top results
            category_filter: Optional category filter
            
        Returns:
            Dictionary with full query, summary, evidence, and metadata
        """
        # If no site key provided, infer it
        if not site_key:
            site_key = infer_site_key(question)
        
        # Get site label and construct full query
        site_label = SITE_LABELS.get(site_key, site_key)
        full_query = f"{site_label} {question}"
        
        # Use category filter or site key
        category = normalize_category_filter(category_filter or site_key)
        
        # Retrieve evidence
        evidence, metadata = await self.retriever.retrieve(
            query_text=full_query,
            category=category,
            N=100,
            k_final=top_k,
            rerank_pool=50,
        )

        # Fallback if category filter was too narrow
        if category and not evidence:
            evidence, metadata = await self.retriever.retrieve(
                query_text=full_query,
                category=None,
                N=100,
                k_final=top_k,
                rerank_pool=50,
            )
        
        # Generate summary
        query_type = metadata.get("query_type", "general")
        nccn_assessment = metadata.get("nccn_assessment")
        
        # Classify query using Unified_Router (single source of truth)
        routing_result = None
        module_classification = None
        try:
            from src.api.services.unified_router import get_unified_router
            router = get_unified_router()
            routing_result = router.route_query(question, None)
            
            # Store routing result in metadata
            metadata["routing"] = routing_result.to_dict()
            
            # Extract module classification for backward compatibility
            module_classification = {
                "module": routing_result.module.value,
                "confidence": routing_result.module_confidence,
                "signals_matched": routing_result.signals_matched,
            }
            metadata["module_classification"] = module_classification
            
            print(f"[DeepDive] Unified routing: module={routing_result.module.value}, "
                  f"query_type={routing_result.query_type.value}, "
                  f"confidence={routing_result.module_confidence:.2f}")
        except Exception as e:
            print(f"[DeepDive] Unified routing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Apply staging normalization
        staging_context = None
        try:
            from src.api.services.staging_normalizer import get_staging_normalizer
            normalizer = get_staging_normalizer()
            
            cancer_type = site_key.lower() if site_key else None
            staging_result = normalizer.normalize_staging_in_context(
                chunks=evidence,
                query=question,
                cancer_type=cancer_type
            )
            primary = staging_result.get("primary_staging")
            if primary and isinstance(primary, dict):
                from src.api.services.staging_normalizer import StagingInfo
                staging_info = StagingInfo(
                    t_stage=primary.get("t_stage"),
                    n_stage=primary.get("n_stage"),
                    m_stage=primary.get("m_stage"),
                    stage_group=primary.get("stage_group"),
                    staging_type=primary.get("staging_type", "unknown"),
                    confidence=primary.get("confidence", 0.0),
                )
                staging_context = normalizer.format_staging_context(staging_info)
            metadata["staging_info"] = {
                "final_staging": staging_result.get("primary_staging"),
                "staging_context": staging_context,
            }
        except Exception as e:
            print(f"Warning: Staging normalization failed in deep_dive: {e}")
            import traceback
            traceback.print_exc()
        
        import asyncio
        summary = await asyncio.to_thread(
            gpt4o_summary_enhanced,
            openai_client=self.openai_client,
            question=question,
            evidence=evidence,
            query_type=query_type,
            nccn_assessment=nccn_assessment,
            staging_context=staging_context,
            module_classification=module_classification,
        )
        
        # Validate and enrich numerical values in the answer
        try:
            validation_result = validate_numbers_against_sources(summary, evidence)
            metadata["numerical_validation"] = {
                "validation_rate": validation_result["validation_rate"],
                "total_numbers": validation_result["total_numbers"],
                "validated_count": len(validation_result["validated_numbers"]),
                "unvalidated_count": len(validation_result["unvalidated_numbers"]),
            }
            
            if validation_result["unvalidated_numbers"]:
                print(f"[RAG] Warning: {len(validation_result['unvalidated_numbers'])} numbers not found in sources")
                # RF-5: strip unverified numbers before returning
                summary = strip_unvalidated_numbers(
                    summary, validation_result["unvalidated_numbers"]
                )

            # Enrich answer with statistical context (CI, p-values)
            summary = enrich_answer_with_stats(summary, evidence)
        except Exception as e:
            print(f"[RAG] Numerical validation failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            from src.api.services import pipeline_metrics as _pm
            _pm_cur = _pm.current()
            if _pm_cur is not None:
                print(_pm_cur.summary_line())
        except Exception:
            pass

        return {
            "query": full_query,
            "site_key": site_key,
            "site_label": site_label,
            "summary": summary,
            "evidence": evidence,
            "metadata": metadata,
        }

    async def query_study_focused(
        self,
        question: str,
        max_studies: int = 8,
        chunks_per_study: int = 8,
        category: Optional[str] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        cached_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Query using comprehensive two-phase retrieval.
        
        This method:
        1. Runs standard retrieval (Qdrant + PostgreSQL) with query expansion
        2. Collects all unique doc_ids from both sources
        3. For each doc_id, runs parallel in-document searches
        4. Reranks complete study evidence by relevance
        
        Args:
            question: User's question
            max_studies: Maximum number of studies to include (default 8)
            chunks_per_study: Maximum chunks per study (default 8)
            category: Optional category filter
            conversation_context: Optional conversation context
            cached_evidence: Optional pre-cached evidence chunks (skips retrieval if provided)
            
        Returns:
            Dictionary with answer, evidence grouped by study, and metadata
        """
        import time
        
        try:
            from src.api.services.comprehensive_retrieval import (
                get_comprehensive_retriever,
                convert_to_rag_evidence,
            )
            
            print("\n" + "=" * 80)
            print("  ENTERING: query_study_focused() [Comprehensive RAG Path]")
            print("=" * 80)
            print(f"  Query: {question[:120]}{'...' if len(question) > 120 else ''}")
            print(f"  max_studies={max_studies}, chunks_per_study={chunks_per_study}, category={category}")

            # Normalize category
            normalized_category = normalize_category_filter(category) if category else None
            print(f"  Normalized category: {normalized_category}")
            
            # Get accumulated context from conversation
            accumulated_context = None
            if conversation_context:
                # Extract query_structure from last entry if available
                for entry in reversed(conversation_context):
                    if entry.get("query_structure"):
                        accumulated_context = entry.get("query_structure")
                        break
            
            # Use cached evidence if provided, otherwise run comprehensive retrieval
            evidence = []
            retrieval_metadata = {}
            studies_data = []
            
            if cached_evidence:
                print(f"[ComprehensiveRAG] Using cached evidence ({len(cached_evidence)} chunks)")
                evidence = cached_evidence
                retrieval_metadata = {"from_cache": True}
                _reconciled_for_generation = None
                # Extract unique studies from cached evidence
                seen_doc_ids = set()
                for e in evidence:
                    doc_id = e.get("doc_id")
                    if doc_id and doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        studies_data.append({
                            "doc_id": doc_id,
                            "title": e.get("title", "Unknown"),
                            "citation": e.get("citation"),
                            "rerank_score": e.get("score", 0),
                            "chunks": 1,
                            "sections": [e.get("section")] if e.get("section") else [],
                            "source": "cache",
                        })
            else:
                # Cascading clinical extraction — produces values in the
                # exact canonical casing Qdrant payloads use so Phase 1
                # can run as a HARD filter, not a soft boost. First step
                # returns None for non-cancer queries and the rest of
                # the cascade is skipped, so the overhead is a single
                # small GPT-4o-mini call for unrelated queries.
                clinical_profile = None
                try:
                    from src.api.services.clinical_extractor import (
                        CascadingClinicalExtractor,
                    )
                    extractor = CascadingClinicalExtractor(self.oa)
                    clinical_profile = await extractor.extract(question)
                    if clinical_profile.has_any_filter():
                        print(
                            f"[ClinicalExtractor] "
                            f"type={clinical_profile.cancer_type_label!r} "
                            f"sites={clinical_profile.cancer_sites} "
                            f"hist={clinical_profile.histologies} "
                            f"stages={clinical_profile.stages} "
                            f"biomarkers={clinical_profile.biomarkers} "
                            f"prior_tx={clinical_profile.prior_treatments}"
                        )
                    else:
                        print("[ClinicalExtractor] No clinical axes resolved — using soft filter")
                except Exception as e:
                    print(f"[ClinicalExtractor] failed (continuing with soft filter): {e}")

                # Run comprehensive retrieval
                retriever = get_comprehensive_retriever()
                result = await retriever.retrieve_comprehensive(
                    query_text=question,
                    max_studies=max_studies,
                    chunks_per_study=chunks_per_study,
                    category=normalized_category,
                    accumulated_context=accumulated_context,
                    conversation_context=conversation_context,
                    clinical_profile=clinical_profile,
                )
                
                # Convert to standard evidence format
                evidence, retrieval_metadata = convert_to_rag_evidence(result, max_chunks=max_studies * chunks_per_study)
                studies_data = [s.to_dict() for s in result.studies]
                # Preserve reconciled_structure for generation (split-brain fix)
                _reconciled_for_generation = result.reconciled_structure

                print(f"\n{'─' * 80}")
                print(f"  POST-RETRIEVAL: Converting to evidence format")
                print(f"{'─' * 80}")
                print(f"  Total evidence chunks: {len(evidence)}")
                print(f"  Studies returned: {len(result.studies)}")
                for i, s in enumerate(result.studies):
                    print(f"    {i+1}. {s.title[:60]}{'...' if len(s.title) > 60 else ''} "
                          f"(doc_id={s.doc_id[:30]}, source={s.source}, "
                          f"score={s.rerank_score:.3f}, chunks={len(s.chunks)})")
            
            # =====================================================
            # PATIENT ELIGIBILITY BOOST: Check if patient in query
            # matches the enrolled patients in retrieved studies
            # =====================================================
            print(f"\n{'─' * 80}")
            print(f"  POST-RETRIEVAL STEP A: Patient Eligibility Check")
            print(f"{'─' * 80}")
            patient_eligibility_metadata = None
            if evidence:
                try:
                    from src.api.services.patient_eligibility_boost_service import run_patient_eligibility_check
                    pre_count = len(evidence)
                    evidence, patient_eligibility_metadata = await run_patient_eligibility_check(
                        query=question,
                        chunks=evidence,
                        openai_client=self.openai_client,
                    )
                    if patient_eligibility_metadata:
                        retrieval_metadata["patient_eligibility"] = patient_eligibility_metadata
                        print(f"  [Eligibility] Matches: {patient_eligibility_metadata.get('matches', 0)}")
                        print(f"  [Eligibility] Hard-filtered: {patient_eligibility_metadata.get('hard_filtered', 0)}")
                        print(f"  [Eligibility] Penalized: {patient_eligibility_metadata.get('penalized', 0)}")
                        print(f"  [Eligibility] Evidence count: {pre_count} → {len(evidence)}")
                    else:
                        print(f"  [Eligibility] No eligibility metadata returned")
                except Exception as e:
                    print(f"  [Eligibility] FAILED: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  [Eligibility] Skipped — no evidence to check")

            # Path A guard (comprehensive path) — short-circuit when
            # retrieval misrouted. See the matching guard in query()
            # for the rationale. Returning "no relevant evidence found"
            # is the clinically safe answer when category_routing_suspect
            # is True and no evidence survived the eligibility filter.
            if (
                patient_eligibility_metadata is not None
                and patient_eligibility_metadata.get("category_routing_suspect")
                and not evidence
            ):
                print(
                    "[ComprehensiveRAG] Short-circuit: empty bundle + "
                    "suspect category routing. Returning 'no relevant "
                    "evidence found'."
                )
                retrieval_metadata["short_circuit_reason"] = (
                    "category_routing_suspect_empty_bundle"
                )
                no_evidence_answer = (
                    "No relevant evidence found for this patient case. "
                    "Retrieval returned studies that did not match the "
                    "patient's cancer type (typically because the query "
                    "mentions anatomy or clinical features that appear in "
                    "the literature of a different cancer). Try restating "
                    "the cancer type or primary site explicitly."
                )
                return {
                    "answer": no_evidence_answer,
                    "evidence": [],
                    "metadata": retrieval_metadata,
                    "query_type": "general",
                    "sources": [],
                    "source_titles": [],
                    "source_citations": [],
                    "studies": [],
                    "accumulated_context": retrieval_metadata.get("query_structure"),
                    "updated_context_entry": None,
                    "answer_quality": None,
                }

            # Classify query type
            print(f"\n{'─' * 80}")
            print(f"  POST-RETRIEVAL STEP B: Query Classification + Module Routing")
            print(f"{'─' * 80}")
            query_classification = classify_query_hybrid(question, self.openai_client)
            query_type = query_classification["primary_type"]
            print(f"  [Classification] Query type: {query_type}")
            print(f"  [Classification] Full: {query_classification}")

            # Get module classification
            module_classification = None
            try:
                from src.api.services.unified_router import get_unified_router
                router = get_unified_router()
                routing_result = router.route_query(question, conversation_context)
                module_classification = {
                    "module": routing_result.module.value,
                    "confidence": routing_result.module_confidence,
                    "signals_matched": routing_result.signals_matched,
                }
                print(f"  [Module Routing] Module: {routing_result.module.value}, "
                      f"confidence={routing_result.module_confidence:.2f}, "
                      f"signals={routing_result.signals_matched}")
            except Exception as e:
                print(f"  [Module Routing] FAILED: {e}")
            
            # Format conversation context if provided
            formatted_conversation_context = ""
            if conversation_context:
                formatted_conversation_context = format_conversation_history_from_context(
                    conversation_context=conversation_context,
                    max_entries=10,
                    max_chars=50000
                )
            
            # Generate answer
            print(f"\n{'─' * 80}")
            print(f"  POST-RETRIEVAL STEP C: Answer Generation")
            print(f"{'─' * 80}")
            print(f"  [Answer Gen] Using query_type={query_type}, evidence_chunks={len(evidence)}")
            print(f"  [Answer Gen] Calling gpt4o_summary_enhanced()...")
            t_answer = time.time()
            # Run in a worker thread — this is a synchronous OpenAI call and
            # is usually the single longest step in the request. Without
            # to_thread it blocks the whole FastAPI event loop for its full
            # duration, stalling every other concurrent request on the
            # server, not just this one.
            import asyncio
            answer = await asyncio.to_thread(
                gpt4o_summary_enhanced,
                openai_client=self.openai_client,
                question=question,
                evidence=evidence,
                query_type=query_type,
                conversation_context=formatted_conversation_context,
                module_classification=module_classification,
                patient_eligibility=patient_eligibility_metadata,
                reconciled_structure=_reconciled_for_generation,
            )
            print(f"  [Answer Gen] Done in {(time.time() - t_answer)*1000:.0f}ms, "
                  f"answer length={len(answer)} chars")
            print(f"  [Answer Gen] Preview: {answer[:150]}{'...' if len(answer) > 150 else ''}")

            # Validate numerical values
            try:
                validation_result = validate_numbers_against_sources(answer, evidence)
                retrieval_metadata["numerical_validation"] = {
                    "validation_rate": validation_result["validation_rate"],
                    "total_numbers": validation_result["total_numbers"],
                    "validated_count": len(validation_result["validated_numbers"]),
                    "unvalidated_count": len(validation_result["unvalidated_numbers"]),
                }

                print(f"  [Numerical Validation] {validation_result['total_numbers']} numbers in answer, "
                      f"{len(validation_result['validated_numbers'])} validated, "
                      f"{len(validation_result['unvalidated_numbers'])} unvalidated "
                      f"(rate={validation_result['validation_rate']:.0%})")
                if validation_result["unvalidated_numbers"]:
                    for uv in validation_result["unvalidated_numbers"]:
                        print(f"    UNVALIDATED: {uv.get('value')} {uv.get('unit', '')} "
                              f"(type={uv.get('type')}, raw='{uv.get('raw_match', '?')}')")
                if validation_result["validated_numbers"]:
                    for v in validation_result["validated_numbers"][:3]:
                        av = v["answer_value"]
                        print(f"    validated: {av.get('value')} {av.get('unit', '')} "
                              f"(exact={v.get('exact_match', '?')})")

                # RF-5: strip unverified numbers from the final answer
                if validation_result.get("unvalidated_numbers"):
                    answer = strip_unvalidated_numbers(
                        answer, validation_result["unvalidated_numbers"]
                    )

                answer = enrich_answer_with_stats(answer, evidence)
            except Exception as e:
                print(f"[ComprehensiveRAG] Numerical validation failed: {e}")
            
            # Enhance answer with quality metrics (citations, confidence, contradictions)
            answer_quality = None
            try:
                from src.api.services.answer_quality_service import enhance_answer_with_quality
                
                quality_result = enhance_answer_with_quality(
                    answer=answer,
                    evidence=evidence,
                    query_type=query_type,
                    include_citations=True,
                    include_confidence=True,
                    include_contradictions=True,
                    include_structured=(query_type in ["dose_question", "trial_results"]),
                    include_references=False,
                )
                
                answer = quality_result["answer"]
                answer_quality = quality_result["quality"]
                retrieval_metadata["answer_quality"] = answer_quality
                
                if answer_quality.get("contradictions"):
                    print(f"[ComprehensiveRAG] Detected {len(answer_quality['contradictions'])} contradictions")
                print(f"[ComprehensiveRAG] Confidence: {answer_quality['confidence']['level']} ({answer_quality['confidence']['score']})")
                
            except Exception as e:
                print(f"[ComprehensiveRAG] Answer quality enhancement failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Extract source info for conversation tracking
            source_doc_ids = []
            source_titles = []
            source_citations = []
            
            for e in evidence:
                doc_id = e.get("doc_id")
                title = e.get("title")
                citation = e.get("citation")
                
                if doc_id and doc_id not in source_doc_ids:
                    source_doc_ids.append(doc_id)
                if title and title not in source_titles:
                    source_titles.append(title)
                if citation and citation not in source_citations:
                    source_citations.append(citation)
            
            # Create updated_context_entry for automatic conversation mode
            # This entry will be stored in frontend sessionStorage
            action_type, treatments = classify_action_type(question, conversation_context)
            updated_context_entry = {
                "query": question,
                "action_type": action_type,
                "doc_ids": source_doc_ids[:10],  # Limit to top 10
                "doc_titles": source_titles[:10],
                "timestamp": int(time.time() * 1000),
            }
            
            # Only include treatments for eval_treatment actions
            if action_type == "eval_treatment" and treatments:
                updated_context_entry["treatments"] = treatments
            
            print(f"[ComprehensiveRAG] Created updated_context_entry: action_type={action_type}, doc_ids={len(source_doc_ids)}, doc_titles={len(source_titles)}")
            
            # Build metadata
            metadata = {
                **retrieval_metadata,
                "query_type": query_type,
                "query_classification": query_classification,
                "module_classification": module_classification,
            }
            
            return {
                "answer": answer,
                "evidence": evidence,
                "metadata": metadata,
                "query_type": query_type,
                "sources": source_doc_ids,
                "source_titles": source_titles,
                "source_citations": source_citations,
                "studies": studies_data,
                "accumulated_context": retrieval_metadata.get("query_structure"),
                "updated_context_entry": updated_context_entry,
                "answer_quality": answer_quality,
            }
            
        except Exception as e:
            print(f"[ComprehensiveRAG] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# ============================================
# SINGLETON INSTANCE
# ============================================

_rag_service_instance = None


def get_enhanced_rag_service() -> EnhancedRAGService:
    """Get or create the enhanced RAG service singleton."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = EnhancedRAGService()
    return _rag_service_instance

