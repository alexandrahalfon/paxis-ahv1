#!/usr/bin/env python3
"""
PTO Frame Builder - Extract Patient→Treatment→Outcome frames from existing chunks.

This module builds structured PTO frames using:
1. Existing keyword metadata from your SECTION_WINDOWS_JSONL
2. Regex extraction for structured fields (stage, dose, outcomes)
3. No LLM calls required - completely free

Usage:
    python pto_frame_builder.py input.jsonl output_frames.jsonl
    
Repository location: src/ingestion/pto_frame_builder.py

Author: Built for Paxis RAG platform
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Set
import argparse

# Note: When integrated into Paxis, you can import settings like:
# from src.core.config import get_settings


# =============================================================================
# REGEX PATTERNS FOR MEDICAL EXTRACTION
# =============================================================================

class MedicalPatterns:
    """Regex patterns for extracting structured medical information."""
    
    # Cancer staging patterns
    STAGE_PATTERNS = [
        # TNM staging: T1N1M0, pT2aN0M0, cT3bN2M1
        (r'[cp]?T[0-4][a-d]?[is]?\s*N[0-3][a-c]?(?:mi)?\s*M[01x]', 'tnm'),
        # Stage groupings: Stage I, Stage IIA, Stage 3B
        (r'stage\s*(I{1,3}V?|IV|[1-4])\s*([ABC])?(?:\s|,|\.|\))', 'stage_group'),
        # AJCC staging
        (r'AJCC\s*(?:stage\s*)?(I{1,3}V?|IV|[1-4])', 'ajcc'),
        # Clinical vs pathologic
        (r'(clinical|pathologic(?:al)?)\s+stage\s*(I{1,3}V?|IV|[1-4])\s*([ABC])?', 'clinical_path'),
    ]
    
    # Dose/fractionation patterns
    DOSE_PATTERNS = [
        # Standard: 50.4 Gy in 28 fractions, 66Gy/33fx
        (r'(\d+(?:\.\d+)?)\s*Gy\s*(?:in\s*)?(\d+)\s*(?:fx|fractions?)', 'dose_fx'),
        # Dose only: 54 Gy, 45Gy
        (r'(\d+(?:\.\d+)?)\s*Gy(?!\s*in|\s*/)', 'dose_only'),
        # BID fractionation: 1.5 Gy BID
        (r'(\d+(?:\.\d+)?)\s*Gy\s*BID', 'bid'),
        # Dose per fraction: 2 Gy/fx, 1.8Gy per fraction
        (r'(\d+(?:\.\d+)?)\s*Gy\s*(?:/|per)\s*(?:fx|fraction)', 'per_fx'),
    ]
    
    # Outcome patterns
    OUTCOME_PATTERNS = [
        # Overall survival: OS 85%, 5-year OS 72%
        (r'(\d+)[\s-]*(?:year|yr|mo(?:nth)?)\s*(?:OS|overall\s+survival)[:\s]*(\d+(?:\.\d+)?)\s*%', 'os_rate'),
        (r'(?:OS|overall\s+survival)[:\s]*(\d+(?:\.\d+)?)\s*%', 'os_simple'),
        # Median survival: median OS 24 months
        (r'median\s+(?:OS|overall\s+survival)[:\s]*(\d+(?:\.\d+)?)\s*(?:months?|mo|years?|yr)', 'median_os'),
        # PFS patterns
        (r'(\d+)[\s-]*(?:year|yr)\s*(?:PFS|progression[- ]free)[:\s]*(\d+(?:\.\d+)?)\s*%', 'pfs_rate'),
        (r'median\s+(?:PFS|progression[- ]free)[:\s]*(\d+(?:\.\d+)?)\s*(?:months?|mo)', 'median_pfs'),
        # Local/locoregional control
        (r'(\d+)[\s-]*(?:year|yr)\s*(?:local|locoregional)\s*control[:\s]*(\d+(?:\.\d+)?)\s*%', 'lc_rate'),
        (r'(?:local|locoregional)\s*control[:\s]*(\d+(?:\.\d+)?)\s*%', 'lc_simple'),
        # Disease-free survival
        (r'(\d+)[\s-]*(?:year|yr)\s*(?:DFS|disease[- ]free)[:\s]*(\d+(?:\.\d+)?)\s*%', 'dfs_rate'),
        # Hazard ratio
        (r'HR\s*[=:]\s*(\d+\.\d+)(?:\s*[,;]\s*(?:95%?\s*CI|p)[:\s=]*[\d\.\-\s]+)?', 'hazard_ratio'),
        # Recurrence rates
        (r'(\d+(?:\.\d+)?)\s*%\s*(?:recurrence|relapse)', 'recurrence'),
        # Response rates
        (r'(?:pCR|pathologic(?:al)?\s+complete\s+response)[:\s]*(\d+(?:\.\d+)?)\s*%', 'pcr_rate'),
    ]
    
    # Biomarker patterns
    BIOMARKER_PATTERNS = [
        # Require word boundary before ER/PR to avoid matching "cancer-",
        # "water-", "greater-", etc. as false biomarker hits.
        (r'\b(ER|estrogen\s+receptor)\s*[+\-](?!\w)', 'er_status'),
        (r'\b(PR|progesterone\s+receptor)\s*[+\-](?!\w)', 'pr_status'),
        (r'\bHER2\s*[+\-/]', 'her2_status'),
        (r'\btriple\s*negative\b', 'tnbc'),
        (r'\b(EGFR|ALK|ROS1|KRAS|BRAF|PD-?L1)\s*(?:mutation|positive|negative|[+\-]|\d+%)', 'molecular'),
        (r'\b(HPV|p16)\s*(?:positive|negative|[+\-])', 'hpv_status'),
        (r'\b(MSI|microsatellite)[- ]?(high|low|stable|H|L|S)\b', 'msi_status'),
    ]
    
    # Treatment modality patterns
    TREATMENT_MODALITIES = [
        'radiotherapy', 'radiation therapy', 'RT', 'IMRT', 'SBRT', 'SRS',
        'chemotherapy', 'chemoradiation', 'chemoRT', 'concurrent chemo',
        'surgery', 'resection', 'mastectomy', 'lumpectomy', 'orchiectomy',
        'immunotherapy', 'checkpoint inhibitor',
        'targeted therapy', 'hormone therapy', 'endocrine therapy',
        'brachytherapy', 'proton therapy',
    ]
    
    # Common chemotherapy agents
    CHEMO_AGENTS = [
        'cisplatin', 'carboplatin', 'paclitaxel', 'docetaxel',
        'trastuzumab', 'pertuzumab', 'pembrolizumab', 'nivolumab',
        'cetuximab', 'bevacizumab', 'osimertinib',
        'tamoxifen', 'anastrozole', 'letrozole',
        'vincristine', 'doxorubicin', 'cyclophosphamide',
        '5-FU', 'fluorouracil', 'capecitabine', 'gemcitabine',
        'FOLFOX', 'FOLFIRI', 'TCHP', 'AC-T',
    ]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PTOFrame:
    """Represents a Patient→Treatment→Outcome relationship frame."""

    # Identifiers
    pto_id: str
    doc_id: str
    category: str

    # ── Study identity ────────────────────────────────────────────────────
    study_title: Optional[str] = None
    study_phase: Optional[str] = None       # Phase I, II, III, IV, etc.
    study_design: Optional[str] = None      # RCT, single-arm, retrospective, etc.
    disease_area: Optional[str] = None      # "NSCLC", "Breast cancer", etc.
    cohort_summary: Optional[str] = None    # "Adults with Stage IV NSCLC, EGFR+, 1L"

    # ── Patient / eligibility profile ─────────────────────────────────────
    # Demographics
    cancer_type: Optional[str] = None
    cancer_subsite: Optional[str] = None
    cancer_location: Optional[str] = None   # e.g. "Breast", "Head and Neck"
    histology: Optional[str] = None         # e.g. "adenocarcinoma", "SCC"
    histologic_grade: Optional[str] = None  # G1, G2, G3, GX

    # Staging
    stage: Optional[str] = None             # Stage group: "Stage IIIA"
    stage_range: Optional[str] = None       # "Stage II-III" (eligible range)
    tnm: Optional[str] = None               # "T2N1M0"
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    clinical_vs_pathologic: Optional[str] = None  # "clinical", "pathologic"

    # Biomarkers
    biomarkers: List[str] = field(default_factory=list)
    required_biomarkers: List[str] = field(default_factory=list)    # Must-have for eligibility
    excluded_biomarkers: List[str] = field(default_factory=list)    # Exclusion criteria

    # Patient demographics/clinical
    age_range: Optional[str] = None         # "18-75", "≥18", etc.
    gender: Optional[str] = None            # "male", "female", "all"
    race_ethnicity: Optional[str] = None    # "all", "specific population", etc.
    ecog_range: Optional[str] = None        # "0-1", "0-2", etc.
    kps_range: Optional[str] = None         # "≥70", "≥60", etc.
    smoking_status: Optional[str] = None    # "smoker", "non-smoker", "any"
    patient_features: List[str] = field(default_factory=list)
    comorbidities_allowed: List[str] = field(default_factory=list)   # tolerated comorbidities
    comorbidity_exclusions: List[str] = field(default_factory=list)  # "active autoimmune", etc.

    # Disease status & treatment history
    disease_status: Optional[str] = None    # "locally advanced", "metastatic", "recurrent"
    measurable_disease_required: Optional[bool] = None  # RECIST measurable disease?
    line_of_therapy: Optional[str] = None   # "1L", "2L+", "treatment-naive"
    max_prior_lines: Optional[str] = None   # "≤ 2 prior lines", "unlimited"
    prior_therapy_requirements: Optional[str] = None  # "must have received platinum"
    prior_therapy_exclusions: Optional[str] = None    # "no prior immunotherapy"
    prior_surgery: Optional[str] = None     # "prior resection required", "unresectable"
    prior_radiation: Optional[str] = None   # "prior RT allowed", "no prior RT to target"
    response_status: Optional[str] = None   # "post-NAC with residual disease"
    recurrence_status: Optional[str] = None # "first recurrence", "multiply recurrent"
    time_from_prior_therapy: Optional[str] = None  # "≥ 6 months from last chemo"

    # Lab / organ function requirements
    lab_requirements: Dict[str, str] = field(default_factory=dict)  # {"creatinine": "< 1.5", "ANC": "≥ 1500"}
    organ_function_requirements: List[str] = field(default_factory=list)  # "adequate hepatic function"

    # ── Treatment (what the study evaluated) ──────────────────────────────
    treatment_modalities: List[str] = field(default_factory=list)
    treatment_details: Optional[str] = None
    treatment_arms: Optional[str] = None    # "Arm A: pembro + chemo vs Arm B: chemo"
    dose_fractionation: Optional[str] = None
    chemo_agents: List[str] = field(default_factory=list)
    treatment_setting: Optional[str] = None  # "neoadjuvant", "adjuvant", "definitive"
    treatment_duration: Optional[str] = None  # "6 cycles", "until progression"
    concurrent_therapies: List[str] = field(default_factory=list)  # concurrent agents

    # ── Outcomes ──────────────────────────────────────────────────────────
    outcomes: Dict[str, str] = field(default_factory=dict)
    primary_endpoint: Optional[str] = None   # "pCR rate", "OS", "PFS"
    secondary_endpoints: List[str] = field(default_factory=list)
    key_finding: Optional[str] = None        # "Significantly improved OS (HR 0.72)"
    response_criteria: Optional[str] = None  # "RECIST 1.1", "iRECIST", "pathologic"
    follow_up_duration: Optional[str] = None # "median 36 months"
    n_patients: Optional[int] = None         # number enrolled
    subgroup_analyses: List[str] = field(default_factory=list)  # notable subgroup findings

    # ── Inclusion / exclusion criteria (raw) ──────────────────────────────
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)

    # ── Excluded populations / contraindications ──────────────────────────
    excluded_medications: List[str] = field(default_factory=list)  # "concurrent steroids > 10mg"
    excluded_conditions: List[str] = field(default_factory=list)   # "active CNS metastases"

    # ── Evidence & metadata ───────────────────────────────────────────────
    evidence_chunk_ids: List[str] = field(default_factory=list)
    source_doc_dir_name: Optional[str] = None
    doc_meta: Dict[str, Any] = field(default_factory=dict)

    # Embedding text (generated)
    frame_text: str = ""

    # Per-section embedding texts (populated by _build_section_texts)
    section_text_patient: str = ""
    section_text_treatment: str = ""
    section_text_outcome: str = ""
    section_text_eligibility: str = ""

    # Quality metrics
    profile_signal_count: int = 0
    treatment_signal_count: int = 0
    outcome_signal_count: int = 0
    confidence: str = "low"  # low, medium, high
    extraction_method: str = "regex"  # "regex" or "llm"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['node_type'] = 'pto_frame'
        # Include per-section texts for payload inspection / debugging
        d['section_text_patient'] = self.section_text_patient
        d['section_text_treatment'] = self.section_text_treatment
        d['section_text_outcome'] = self.section_text_outcome
        d['section_text_eligibility'] = self.section_text_eligibility
        return d


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class PTOExtractor:
    """Extracts structured information from text using regex patterns."""
    
    def __init__(self):
        self.patterns = MedicalPatterns()
    
    def extract_stage(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract cancer stage from text.
        
        Returns:
            Tuple of (stage_group, tnm_stage)
        """
        text_lower = text.lower()
        stage_group = None
        tnm_stage = None
        
        for pattern, ptype in self.patterns.STAGE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if ptype == 'tnm':
                    tnm_stage = match.group(0).upper().replace(' ', '')
                elif ptype in ('stage_group', 'ajcc', 'clinical_path'):
                    # Normalize stage group
                    groups = match.groups()
                    stage_num = groups[0] if groups else ''
                    stage_letter = groups[1] if len(groups) > 1 and groups[1] else ''
                    stage_group = f"Stage {stage_num}{stage_letter or ''}".strip()
        
        return stage_group, tnm_stage
    
    def extract_dose(self, text: str) -> Optional[str]:
        """Extract radiation dose/fractionation from text."""
        for pattern, ptype in self.patterns.DOSE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if ptype == 'dose_fx':
                    dose, fx = match.groups()
                    return f"{dose} Gy in {fx} fx"
                elif ptype == 'dose_only':
                    return f"{match.group(1)} Gy"
                elif ptype == 'bid':
                    return f"{match.group(1)} Gy BID"
        return None
    
    def extract_outcomes(self, text: str) -> Dict[str, str]:
        """Extract clinical outcomes from text."""
        outcomes = {}
        
        for pattern, ptype in self.patterns.OUTCOME_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                if ptype == 'os_rate':
                    timepoint, rate = groups
                    outcomes[f'{timepoint}yr_OS'] = f"{rate}%"
                elif ptype == 'os_simple':
                    outcomes['OS'] = f"{groups[0]}%"
                elif ptype == 'median_os':
                    outcomes['median_OS'] = f"{groups[0]} mo"
                elif ptype == 'pfs_rate':
                    timepoint, rate = groups
                    outcomes[f'{timepoint}yr_PFS'] = f"{rate}%"
                elif ptype == 'median_pfs':
                    outcomes['median_PFS'] = f"{groups[0]} mo"
                elif ptype == 'lc_rate':
                    timepoint, rate = groups
                    outcomes[f'{timepoint}yr_LC'] = f"{rate}%"
                elif ptype == 'lc_simple':
                    outcomes['LC'] = f"{groups[0]}%"
                elif ptype == 'dfs_rate':
                    timepoint, rate = groups
                    outcomes[f'{timepoint}yr_DFS'] = f"{rate}%"
                elif ptype == 'hazard_ratio':
                    outcomes['HR'] = groups[0]
                elif ptype == 'recurrence':
                    outcomes['recurrence_rate'] = f"{groups[0]}%"
                elif ptype == 'pcr_rate':
                    outcomes['pCR'] = f"{groups[0]}%"
        
        return outcomes
    
    def extract_biomarkers(self, text: str) -> List[str]:
        """Extract biomarker information from text."""
        biomarkers = []
        
        for pattern, ptype in self.patterns.BIOMARKER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                biomarkers.append(match.group(0).strip())
        
        return list(set(biomarkers))
    
    def extract_treatment_modalities(self, text: str) -> List[str]:
        """Extract treatment modalities from text."""
        text_lower = text.lower()
        found = []
        
        for modality in self.patterns.TREATMENT_MODALITIES:
            if modality.lower() in text_lower:
                found.append(modality)
        
        return list(set(found))
    
    def extract_chemo_agents(self, text: str) -> List[str]:
        """Extract chemotherapy agents from text."""
        text_lower = text.lower()
        found = []
        
        for agent in self.patterns.CHEMO_AGENTS:
            if agent.lower() in text_lower:
                found.append(agent)
        
        return list(set(found))


# =============================================================================
# LLM-BASED EXTRACTION
# =============================================================================

LLM_EXTRACTION_PROMPT = """You are a clinical trial data extraction expert. Given text from a medical study, extract COMPREHENSIVE structured information about the study, the patient population it enrolled, the treatment evaluated, and the outcomes reported.

Be thorough. Extract ALL fields below. Use "null" for fields not mentioned. Copy exact values from the text when possible. Do NOT leave a field null if the information can be reasonably inferred from context.

CRITICAL FORMATTING RULES:
- stage_range: Use clean notation with spaces. CORRECT: "Stage II-III", "Stage I-IIA". WRONG: "Stage pathologicalIII", "StagepathologicalI". If staging is pathologic, put it in clinical_vs_pathologic field instead.
- key_finding: MUST contain the PRIMARY EFFICACY result with at least one number. The key finding is the study's main EFFICACY outcome (OS, PFS, DFS, local recurrence rate, pCR rate), NOT a safety/mortality metric. Example: "5-year local recurrence 7.5% vs 8.2% (non-inferior, HR 0.90)" NOT "30-day mortality was 2%".
- primary_endpoint: The endpoint the study was POWERED to evaluate (the efficacy endpoint in the primary analysis), NOT safety endpoints or secondary outcomes. Examples: "overall survival", "progression-free survival", "local recurrence rate", "pCR rate". NOT: "30-day postoperative mortality", "anastomotic leak rate", "adverse events".
- outcomes: Extract ALL numerical efficacy results. Include OS, PFS, DFS, local recurrence, pCR rate, ORR, hazard ratios with 95% CI and p-values, median survival times. Use clear keys like "5yr_OS", "3yr_PFS", "median_OS", "HR_OS", "pCR_rate".
- dose_fractionation: Include the FULL prescription dose, not per-fraction dose. CORRECT: "25 Gy in 5 fractions" or "50 Gy in 25 fractions". WRONG: "2 Gy" (that's per-fraction, not prescription).
- treatment_arms: Describe ALL arms completely. Do NOT truncate mid-sentence.
- line_of_therapy: For preoperative/neoadjuvant studies, use "treatment-naive" or "neoadjuvant (first-line)". Do NOT use "any line" unless the study truly enrolled all lines.

═══════════════════════════════════════════════════════════════
1. STUDY IDENTITY
═══════════════════════════════════════════════════════════════
- study_phase: Trial phase (e.g., "Phase III", "Phase II/III", "Retrospective", "Meta-analysis")
- study_design: Design (e.g., "randomized controlled trial", "single-arm", "retrospective cohort", "prospective observational")
- disease_area: Primary disease with specificity (e.g., "triple-negative breast cancer", "EGFR-mutant Stage IV NSCLC", "locally advanced rectal cancer")
- cohort_summary: One-sentence summary of the enrolled population covering cancer type, stage, biomarkers, line of therapy, and key eligibility (e.g., "Adults ≥18 with previously untreated Stage II-III TNBC, ECOG 0-1, adequate organ function, receiving neoadjuvant pembrolizumab + chemotherapy")
- n_patients: Number of patients enrolled (integer, e.g., 1174)

═══════════════════════════════════════════════════════════════
2. DIAGNOSIS — What cancer / disease was studied
═══════════════════════════════════════════════════════════════
- cancer_type: Specific cancer name (e.g., "breast cancer", "non-small cell lung cancer", "colorectal cancer")
- cancer_location: Anatomic location category — use one of: Breast, Thoracic/Lung, Head and Neck, Gastrointestinal, Hepatobiliary/Pancreatic, Genitourinary, Gynecologic, Skin/Melanoma, CNS/Brain, Hematologic/Lymphoma, Musculoskeletal/Sarcoma, Endocrine/Thyroid, Pediatric, Other
- cancer_subsite: Specific sub-site if mentioned (e.g., "oral tongue", "sigmoid colon", "right upper lobe", "pancreatic head")
- histology: Histopathologic type (e.g., "invasive ductal carcinoma", "adenocarcinoma", "squamous cell carcinoma", "clear cell", "neuroendocrine")
- histologic_grade: Tumor grade (e.g., "G1", "G2", "G3", "GX", "well differentiated", "poorly differentiated", "high grade")

═══════════════════════════════════════════════════════════════
3. CANCER STAGING — TNM and stage grouping
═══════════════════════════════════════════════════════════════
- stage_range: Stage eligibility range (e.g., "Stage II-III", "Stage IIIB-IV", "Stage I-IIA", "any stage", "Stage IV only")
- tnm_t: T stage range eligible (e.g., "T1-T3", "T4", "T2-T4", "any")
- tnm_n: N stage range eligible (e.g., "N0", "N+", "N0-N1", "N2-N3", "any")
- tnm_m: M stage eligible (e.g., "M0", "M1", "M0-M1", "any")
- clinical_vs_pathologic: Staging basis — "clinical", "pathologic", or "both"

═══════════════════════════════════════════════════════════════
4. BIOMARKERS — Molecular and immunohistochemical
═══════════════════════════════════════════════════════════════
- biomarkers: ALL biomarkers mentioned for the study population, with polarity (e.g., ["ER-", "PR-", "HER2-", "TNBC", "PD-L1 CPS ≥ 10", "EGFR exon 19 deletion", "BRCA1 mutant", "MSI-H"])
- required_biomarkers: Biomarkers REQUIRED for eligibility — must-have (e.g., ["EGFR mutant", "PD-L1 TPS ≥ 50%", "HER2-positive IHC 3+ or FISH+"])
- excluded_biomarkers: Biomarkers that EXCLUDED patients (e.g., ["ALK rearrangement", "HER2-positive", "EGFR wild-type"])

═══════════════════════════════════════════════════════════════
5. PATIENT DEMOGRAPHICS & CLINICAL STATUS
═══════════════════════════════════════════════════════════════
- age_range: Age eligibility (e.g., "≥18", "18-75", "≥65", "18-70")
- gender: Gender restriction (e.g., "female only", "male only", "all")
- race_ethnicity: Race/ethnicity if specified (e.g., "all", "Japanese population", "not specified")
- ecog_range: ECOG performance status range (e.g., "0-1", "0-2", "0 only")
- kps_range: Karnofsky performance status if used instead of ECOG (e.g., "≥70", "≥60")
- smoking_status: Smoking status restriction if any (e.g., "any", "never-smokers only", "not specified")

═══════════════════════════════════════════════════════════════
6. DISEASE STATUS & TREATMENT HISTORY
═══════════════════════════════════════════════════════════════
- disease_status: Disease status at enrollment (e.g., "locally advanced", "metastatic", "recurrent/metastatic", "oligometastatic", "unresectable", "newly diagnosed")
- measurable_disease_required: Was RECIST-measurable disease required? (true/false/null)
- line_of_therapy: Line of therapy (e.g., "first-line", "second-line", "2L+", "treatment-naive", "any line")
- max_prior_lines: Maximum prior systemic therapy lines allowed (e.g., "0 (treatment-naive)", "≤ 2 prior lines", "≤ 1 prior line", "unlimited")
- prior_therapy_requirements: Prior treatments REQUIRED (e.g., "must have received prior platinum-based chemotherapy", "prior definitive chemoradiation required")
- prior_therapy_exclusions: Prior treatments that EXCLUDED patients (e.g., "no prior anti-PD1/PD-L1", "no prior immunotherapy", "no prior taxane in metastatic setting")
- prior_surgery: Surgical history requirement (e.g., "prior complete resection required", "unresectable/inoperable", "post-mastectomy", "not specified")
- prior_radiation: Radiation history requirement (e.g., "no prior thoracic RT", "prior RT allowed", "post-lumpectomy RT eligible", "not specified")
- recurrence_status: Recurrence context (e.g., "first recurrence", "multiply recurrent", "de novo metastatic", "not specified")
- response_status: Response status at enrollment (e.g., "post-NAC with residual disease", "post-NAC pCR", "progressing on first-line", "stable disease on maintenance", "not specified")
- time_from_prior_therapy: Temporal requirement for washout/gap (e.g., "≥ 6 months from last chemotherapy", "≥ 4 weeks from prior RT", "≥ 2 weeks from surgery", "not specified")

═══════════════════════════════════════════════════════════════
7. LAB VALUES & ORGAN FUNCTION
═══════════════════════════════════════════════════════════════
- lab_requirements: Dict of lab value requirements (e.g., {{"ANC": "≥ 1500/μL", "platelets": "≥ 100,000/μL", "creatinine": "≤ 1.5x ULN", "bilirubin": "≤ 1.5x ULN", "AST/ALT": "≤ 2.5x ULN", "hemoglobin": "≥ 9 g/dL", "creatinine_clearance": "≥ 30 mL/min"}})
- organ_function_requirements: General organ function requirements (e.g., ["adequate hepatic function", "adequate renal function", "adequate bone marrow function", "LVEF ≥ 50%"])

═══════════════════════════════════════════════════════════════
8. TREATMENT EVALUATED
═══════════════════════════════════════════════════════════════
- treatment_setting: Treatment setting (e.g., "neoadjuvant", "adjuvant", "definitive", "palliative", "maintenance", "consolidation", "salvage", "perioperative")
- treatment_arms: Description of all study arms (e.g., "Arm A: pembrolizumab + paclitaxel/carboplatin → pembrolizumab + AC vs Arm B: placebo + paclitaxel/carboplatin → placebo + AC")
- treatment_modalities: List of modalities (e.g., ["immunotherapy", "chemotherapy", "radiation therapy", "surgery", "targeted therapy"])
- chemo_agents: ALL specific agents/drugs used (e.g., ["pembrolizumab", "paclitaxel", "carboplatin", "doxorubicin", "cyclophosphamide"])
- dose_fractionation: Radiation dose/fractionation if applicable (e.g., "50 Gy in 25 fractions", "60 Gy in 30 fx", "SBRT 54 Gy in 3 fx")
- treatment_duration: Duration/cycles (e.g., "6 cycles", "24 months or until progression", "35 cycles of adjuvant pembrolizumab")
- concurrent_therapies: Concurrent agents (e.g., ["concurrent cisplatin 40 mg/m2 weekly", "concurrent cetuximab"])

═══════════════════════════════════════════════════════════════
9. OUTCOMES & RESULTS
═══════════════════════════════════════════════════════════════
- primary_endpoint: Primary endpoint (e.g., "pCR rate (ypT0/Tis ypN0)", "overall survival", "progression-free survival")
- secondary_endpoints: Secondary endpoints (e.g., ["event-free survival", "overall survival", "objective response rate", "duration of response"])
- key_finding: Key result in one sentence with numbers (e.g., "pCR rate 64.8% vs 51.2% (delta 13.6%, p=0.0005); 3-year EFS 84.5% vs 76.8% (HR 0.63)")
- outcomes: Dict of all specific numerical outcomes (e.g., {{"pCR_rate_experimental": "64.8%", "pCR_rate_control": "51.2%", "3yr_EFS_experimental": "84.5%", "3yr_EFS_control": "76.8%", "EFS_HR": "0.63", "OS_HR": "0.72", "ORR": "45%", "median_PFS": "24.0 months", "median_OS": "not reached"}})
- response_criteria: Response assessment criteria (e.g., "RECIST 1.1", "iRECIST", "pathologic (Miller-Payne)", "Residual Cancer Burden")
- follow_up_duration: Median follow-up (e.g., "median 39.1 months", "minimum 5 years")
- subgroup_analyses: Notable subgroup findings (e.g., ["PD-L1 CPS ≥ 20: pCR 68.9% vs 54.9%", "Node-positive: EFS HR 0.58", "Age ≥ 65: similar benefit"])

═══════════════════════════════════════════════════════════════
10. ELIGIBILITY CRITERIA (comprehensive list)
═══════════════════════════════════════════════════════════════
- inclusion_criteria: ALL key inclusion criteria as a list. Include EVERY criterion you can identify — demographics, disease, biomarker, lab, temporal, functional. (e.g., ["Age ≥ 18 years", "Histologically confirmed TNBC", "Stage II-III (T1c N1-N2 or T2-T4 N0-N2)", "ECOG PS 0-1", "Adequate organ function", "No prior systemic therapy for breast cancer", "Known PD-L1 status"])
- exclusion_criteria: ALL key exclusion criteria. (e.g., ["Prior systemic anticancer therapy for breast cancer", "Active autoimmune disease requiring systemic treatment", "Active CNS metastases", "Prior anti-PD-1/PD-L1/PD-L2 therapy", "Active hepatitis B or C", "Received live vaccine within 30 days", "Concurrent systemic steroid therapy > 10 mg/day prednisone equivalent", "Pregnant or breastfeeding"])

═══════════════════════════════════════════════════════════════
11. EXCLUDED POPULATIONS & CONTRAINDICATIONS
═══════════════════════════════════════════════════════════════
- comorbidity_exclusions: Specific comorbidities that excluded patients (e.g., ["active autoimmune disease", "uncontrolled diabetes", "NYHA class III-IV heart failure", "interstitial lung disease", "active infection requiring IV antibiotics"])
- excluded_medications: Medications that excluded patients (e.g., ["systemic corticosteroids > 10mg prednisone/day", "concurrent immunosuppressive agents", "anticoagulation therapy"])
- excluded_conditions: Other excluding conditions (e.g., ["active CNS metastases", "leptomeningeal disease", "prior organ transplant", "second primary malignancy within 5 years", "pregnancy/breastfeeding"])

---

Study text:
{study_text}

---

Respond with ONLY valid JSON (no markdown code fences, no explanation). Use null for unknown fields, [] for empty lists, {{}} for empty dicts. Every field name must match EXACTLY as listed above."""


class LLMPTOExtractor:
    """Extract PTO frame fields using GPT-4o-mini for comprehensive extraction."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self._call_count = 0
        self._total_tokens = 0

    def extract_from_text(self, study_text: str, max_text_len: int = 10000) -> Optional[Dict[str, Any]]:
        """
        Extract structured PTO fields from study text using LLM.

        Args:
            study_text: Combined text from study chunks
            max_text_len: Maximum text length to send to LLM

        Returns:
            Dict of extracted fields, or None on failure
        """
        # Truncate if needed, preferring beginning (abstract/methods) and end (results)
        if len(study_text) > max_text_len:
            half = max_text_len // 2
            study_text = study_text[:half] + "\n...\n" + study_text[-half:]

        prompt = LLM_EXTRACTION_PROMPT.format(study_text=study_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical trial data extraction expert. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4000,
            )
            self._call_count += 1
            if response.usage:
                self._total_tokens += response.usage.total_tokens

            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            # ── Flatten nested section-header formats ────────────────────
            # gpt-4o-mini sometimes wraps the flat fields inside section
            # objects:
            #   {"study_identity": {"study_phase": "...", ...}, "diagnosis": {...}}
            # or numbered headers:
            #   {"1. STUDY IDENTITY": {"study_phase": "...", ...}}
            #
            # apply_to_frame() expects FLAT keys ("study_phase", "cancer_type",
            # etc.) at the top level. Detect and flatten here so the caller
            # doesn't need to know about the LLM's formatting quirks.
            if result and isinstance(result, dict):
                # Check if the top-level values are dicts (nested format)
                nested_vals = [v for v in result.values()
                               if isinstance(v, dict) and len(v) >= 2]
                if nested_vals and len(nested_vals) >= len(result) * 0.5:
                    # Majority of top-level values are dicts → flatten
                    flat: Dict[str, Any] = {}
                    for section_key, section_val in result.items():
                        if isinstance(section_val, dict):
                            flat.update(section_val)
                        else:
                            # Keep any non-dict top-level values as-is
                            flat[section_key] = section_val
                    print(f"         Flattened nested JSON: {len(result)} sections → {len(flat)} flat keys")
                    result = flat

            # Count non-null fields for quality check
            non_null = sum(1 for v in result.values()
                          if v is not None and v != "null" and v != [] and v != {})
            if non_null < 3:
                print(f"    ⚠ LLM returned only {non_null} non-null fields — possible extraction failure")
                print(f"      Raw response (first 500 chars): {content[:500]}")

            return result

        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON parse error: {e}")
            print(f"      Raw content (first 500 chars): {content[:500] if 'content' in dir() else 'N/A'}")
            return None
        except Exception as e:
            print(f"    ⚠ LLM extraction error: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        return {"calls": self._call_count, "total_tokens": self._total_tokens}

    def apply_to_frame(self, frame: PTOFrame, llm_result: Dict[str, Any]) -> PTOFrame:
        """Merge LLM extraction results into a PTOFrame."""
        if not llm_result:
            return frame

        def _get(key, default=None):
            val = llm_result.get(key)
            if val is None or val == "null" or val == "not specified" or val == "N/A":
                return default
            return val

        def _get_bool(key):
            val = llm_result.get(key)
            if val is True or val == "true":
                return True
            if val is False or val == "false":
                return False
            return None

        def _get_int(key):
            val = llm_result.get(key)
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                try:
                    return int(val.replace(",", ""))
                except (ValueError, AttributeError):
                    return None
            return None

        def _get_list(key):
            val = llm_result.get(key)
            if isinstance(val, list):
                return [v for v in val if v and v != "null" and v != "N/A"]
            return []

        def _get_dict(key):
            val = llm_result.get(key)
            if isinstance(val, dict):
                return {k: v for k, v in val.items() if v and v != "null"}
            return {}

        # ── 1. Study identity ────────────────────────────────────────────
        frame.study_phase = _get("study_phase") or frame.study_phase
        frame.study_design = _get("study_design") or frame.study_design
        frame.disease_area = _get("disease_area") or frame.disease_area
        frame.cohort_summary = _get("cohort_summary") or frame.cohort_summary
        frame.n_patients = _get_int("n_patients") or frame.n_patients

        # ── 2. Diagnosis ─────────────────────────────────────────────────
        frame.cancer_type = _get("cancer_type") or frame.cancer_type
        frame.cancer_location = _get("cancer_location") or frame.cancer_location
        frame.cancer_subsite = _get("cancer_subsite") or frame.cancer_subsite
        frame.histology = _get("histology") or frame.histology
        frame.histologic_grade = _get("histologic_grade") or frame.histologic_grade

        # ── 3. Staging — LLM OVERRIDES regex for stage fields ────────────
        # The regex often grabs the first stage mention in the text (from a
        # reference or comparison arm), while the LLM extracts the study's
        # actual eligibility range. LLM wins when available.
        llm_stage_range = _get("stage_range")
        if llm_stage_range:
            frame.stage_range = llm_stage_range
            frame.stage = llm_stage_range  # override regex stage
        llm_tnm_t = _get("tnm_t")
        if llm_tnm_t:
            frame.tnm_t = llm_tnm_t
        llm_tnm_n = _get("tnm_n")
        if llm_tnm_n:
            frame.tnm_n = llm_tnm_n
        llm_tnm_m = _get("tnm_m")
        if llm_tnm_m:
            frame.tnm_m = llm_tnm_m
        frame.clinical_vs_pathologic = _get("clinical_vs_pathologic") or frame.clinical_vs_pathologic

        # ── 4. Biomarkers — LLM REPLACES regex biomarkers ────────────────
        # The regex biomarker extractor has high false-positive rate (e.g.
        # matching "er-" in "cancer-free", "water-soluble"). When the LLM
        # provides biomarkers, we trust it over regex entirely.
        llm_biomarkers = _get_list("biomarkers")
        if llm_biomarkers:
            frame.biomarkers = llm_biomarkers  # replace, not merge
        frame.required_biomarkers = _get_list("required_biomarkers") or frame.required_biomarkers
        frame.excluded_biomarkers = _get_list("excluded_biomarkers") or frame.excluded_biomarkers

        # ── 5. Demographics & clinical ───────────────────────────────────
        frame.age_range = _get("age_range") or frame.age_range
        frame.gender = _get("gender") or frame.gender
        frame.race_ethnicity = _get("race_ethnicity") or frame.race_ethnicity
        frame.ecog_range = _get("ecog_range") or frame.ecog_range
        frame.kps_range = _get("kps_range") or frame.kps_range
        frame.smoking_status = _get("smoking_status") or frame.smoking_status

        # ── 6. Disease status & treatment history ────────────────────────
        frame.disease_status = _get("disease_status") or frame.disease_status
        frame.measurable_disease_required = _get_bool("measurable_disease_required")
        frame.line_of_therapy = _get("line_of_therapy") or frame.line_of_therapy
        frame.max_prior_lines = _get("max_prior_lines") or frame.max_prior_lines
        frame.prior_therapy_requirements = _get("prior_therapy_requirements") or frame.prior_therapy_requirements
        frame.prior_therapy_exclusions = _get("prior_therapy_exclusions") or frame.prior_therapy_exclusions
        frame.prior_surgery = _get("prior_surgery") or frame.prior_surgery
        frame.prior_radiation = _get("prior_radiation") or frame.prior_radiation
        frame.recurrence_status = _get("recurrence_status") or frame.recurrence_status
        frame.response_status = _get("response_status") or frame.response_status
        frame.time_from_prior_therapy = _get("time_from_prior_therapy") or frame.time_from_prior_therapy

        # ── 7. Lab / organ function ──────────────────────────────────────
        llm_labs = _get_dict("lab_requirements")
        if llm_labs:
            frame.lab_requirements.update(llm_labs)
        llm_organ = _get_list("organ_function_requirements")
        if llm_organ:
            frame.organ_function_requirements = llm_organ

        # ── 8. Treatment — LLM REPLACES regex modalities + agents ────────
        # Regex picks up every keyword mention in the text (including
        # background references to other treatments). The LLM identifies
        # the study's ACTUAL interventions.
        llm_modalities = _get_list("treatment_modalities")
        if llm_modalities:
            frame.treatment_modalities = llm_modalities  # replace, not merge
        llm_agents = _get_list("chemo_agents")
        if llm_agents:
            frame.chemo_agents = llm_agents  # replace, not merge
        frame.treatment_arms = _get("treatment_arms") or frame.treatment_arms
        frame.treatment_setting = _get("treatment_setting") or frame.treatment_setting
        frame.dose_fractionation = _get("dose_fractionation") or frame.dose_fractionation
        frame.treatment_duration = _get("treatment_duration") or frame.treatment_duration
        llm_concurrent = _get_list("concurrent_therapies")
        if llm_concurrent:
            frame.concurrent_therapies = llm_concurrent

        # ── 9. Outcomes — LLM REPLACES regex outcomes ────────────────────
        # Regex outcome extraction grabs numbers from throughout the text
        # (including referenced studies). LLM identifies the study's own
        # reported outcomes.
        llm_outcomes = _get_dict("outcomes")
        if llm_outcomes:
            frame.outcomes = llm_outcomes  # replace, not update
        frame.primary_endpoint = _get("primary_endpoint") or frame.primary_endpoint
        frame.secondary_endpoints = _get_list("secondary_endpoints") or frame.secondary_endpoints
        frame.key_finding = _get("key_finding") or frame.key_finding
        frame.response_criteria = _get("response_criteria") or frame.response_criteria
        frame.follow_up_duration = _get("follow_up_duration") or frame.follow_up_duration
        frame.subgroup_analyses = _get_list("subgroup_analyses") or frame.subgroup_analyses

        # ── 10. Eligibility criteria ─────────────────────────────────────
        llm_inclusion = _get_list("inclusion_criteria")
        if llm_inclusion:
            frame.inclusion_criteria = llm_inclusion
        llm_exclusion = _get_list("exclusion_criteria")
        if llm_exclusion:
            frame.exclusion_criteria = llm_exclusion

        # ── 11. Exclusions ───────────────────────────────────────────────
        frame.comorbidity_exclusions = _get_list("comorbidity_exclusions") or frame.comorbidity_exclusions
        frame.excluded_medications = _get_list("excluded_medications") or frame.excluded_medications
        frame.excluded_conditions = _get_list("excluded_conditions") or frame.excluded_conditions

        # ── Confidence scoring ───────────────────────────────────────────
        frame.extraction_method = "llm"
        filled_fields = sum(1 for v in [
            frame.cancer_type, frame.histology, frame.stage_range,
            frame.disease_status, frame.line_of_therapy, frame.treatment_setting,
            frame.primary_endpoint, frame.key_finding, frame.cohort_summary,
            frame.age_range, frame.ecog_range, frame.histologic_grade,
            frame.prior_therapy_requirements, frame.prior_therapy_exclusions,
            frame.response_status, frame.n_patients, frame.follow_up_duration,
        ] if v)
        filled_fields += min(len(frame.biomarkers), 3)
        filled_fields += min(len(frame.outcomes), 3)
        filled_fields += min(len(frame.inclusion_criteria), 3)
        filled_fields += min(len(frame.exclusion_criteria), 2)
        filled_fields += min(len(frame.lab_requirements), 2)
        filled_fields += min(len(frame.chemo_agents), 2)

        if filled_fields >= 14:
            frame.confidence = "high"
        elif filled_fields >= 5:
            frame.confidence = "medium"

        # ── Post-processing cleanup ──────────────────────────────────────
        _postprocess_frame(frame)

        return frame


def _postprocess_frame(frame: PTOFrame) -> None:
    """
    Clean up common LLM formatting issues in a PTOFrame.

    Called automatically at the end of apply_to_frame(). Fixes:
    - Malformed stage strings ("Stage pathologicalIII" → "pathologic Stage III")
    - Concatenated words with missing spaces
    - Per-fraction dose when full prescription is available in treatment_arms
    - "any line" when the study is neoadjuvant/preoperative
    - Truncated treatment_arms strings
    """

    # ── Stage cleanup ────────────────────────────────────────────────────
    # Fix "Stage pathologicalIII" → "pathologic Stage III"
    # Fix "Stage pathologicalI" → "pathologic Stage I"
    for attr in ("stage", "stage_range"):
        val = getattr(frame, attr, None)
        if not val:
            continue
        # Pattern: "Stage" followed by "pathological"/"pathologic"/"clinical"
        # concatenated with a Roman numeral
        cleaned = val
        cleaned = re.sub(
            r'[Ss]tage\s*(pathologic(?:al)?|clinical)\s*(I{1,3}V?|IV)',
            lambda m: f"{m.group(1)} Stage {m.group(2)}",
            cleaned,
        )
        # Also catch "pathologicalI" without "Stage" prefix
        cleaned = re.sub(
            r'(pathologic(?:al)?)(I{1,3}V?|IV)\b',
            lambda m: f"{m.group(1)} Stage {m.group(2)}",
            cleaned,
        )
        if cleaned != val:
            setattr(frame, attr, cleaned)

    # ── Dose cleanup ─────────────────────────────────────────────────────
    # If dose_fractionation is just a per-fraction dose ("2 Gy") but
    # treatment_arms contains a full prescription ("25 Gy in 5 fractions"),
    # extract the full prescription from treatment_arms.
    if frame.dose_fractionation and frame.treatment_arms:
        dose_val = frame.dose_fractionation.strip()
        # Check if it's a bare per-fraction dose (single number + Gy, no "in X fx")
        if re.match(r'^\d+(?:\.\d+)?\s*Gy$', dose_val):
            # Try to extract full prescription from treatment_arms
            full_dose = re.findall(
                r'(\d+(?:\.\d+)?\s*Gy\s+in\s+\d+\s*(?:fx|fractions?))',
                frame.treatment_arms, re.IGNORECASE
            )
            if full_dose:
                frame.dose_fractionation = "; ".join(full_dose)

    # ── Line of therapy cleanup ──────────────────────────────────────────
    # "any line" is usually wrong for neoadjuvant/preoperative studies
    if frame.line_of_therapy and frame.line_of_therapy.lower() in ("any line", "any"):
        if frame.treatment_setting and any(
            kw in frame.treatment_setting.lower()
            for kw in ("neoadjuvant", "preoperative", "pre-operative")
        ):
            frame.line_of_therapy = "treatment-naive"
        elif frame.prior_therapy_exclusions:
            frame.line_of_therapy = "treatment-naive"

    # ── Treatment arms truncation cleanup ────────────────────────────────
    # If treatment_arms ends mid-word (truncated), trim to last complete sentence
    if frame.treatment_arms:
        arms = frame.treatment_arms.strip()
        if arms and not arms[-1] in ".!?)":
            # Find last sentence boundary
            last_period = arms.rfind(". ")
            last_vs = arms.rfind(" vs ")
            last_versus = arms.rfind(" versus ")
            cut = max(last_period, last_vs, last_versus)
            if cut > len(arms) // 2:
                # Only trim if we're keeping more than half
                if last_vs == cut or last_versus == cut:
                    # Keep through the "vs" clause
                    remaining = arms[cut:]
                    next_space = remaining.find(" ", 4)
                    if next_space > 0:
                        frame.treatment_arms = arms[:cut + next_space]
                else:
                    frame.treatment_arms = arms[:cut + 1]

    # ── Primary endpoint cleanup ─────────────────────────────────────────
    # Flag safety endpoints as secondary if we have an efficacy endpoint in outcomes
    if frame.primary_endpoint:
        safety_endpoints = [
            "mortality", "postoperative mortality", "30-day mortality",
            "adverse event", "toxicity", "anastomotic leak",
            "complication rate",
        ]
        is_safety = any(se in frame.primary_endpoint.lower() for se in safety_endpoints)
        if is_safety and frame.outcomes:
            # Check if outcomes contain efficacy data
            efficacy_keys = [k for k in frame.outcomes
                            if any(ek in k.lower() for ek in
                                   ["os", "pfs", "dfs", "lrc", "local",
                                    "recurrence", "survival", "pcr", "orr",
                                    "response", "hr_"])]
            if efficacy_keys:
                # Swap: move current primary to secondary, promote efficacy
                if frame.primary_endpoint not in frame.secondary_endpoints:
                    frame.secondary_endpoints.insert(0, frame.primary_endpoint)
                # Infer primary from the most common efficacy key
                endpoint_map = {
                    "os": "overall survival",
                    "pfs": "progression-free survival",
                    "dfs": "disease-free survival",
                    "lrc": "locoregional control",
                    "local": "local recurrence",
                    "recurrence": "local recurrence",
                    "pcr": "pathologic complete response rate",
                    "orr": "objective response rate",
                }
                for ek in efficacy_keys:
                    for pattern, label in endpoint_map.items():
                        if pattern in ek.lower():
                            frame.primary_endpoint = label
                            break
                    else:
                        continue
                    break

    # ── Key finding from outcomes ────────────────────────────────────────
    # If key_finding is missing but outcomes dict has data, synthesize one
    if not frame.key_finding and frame.outcomes:
        parts = []
        for k, v in list(frame.outcomes.items())[:4]:
            parts.append(f"{k}: {v}")
        if parts:
            frame.key_finding = "; ".join(parts)


# =============================================================================
# VALIDATION — ground-truth check for LLM-extracted fields
# =============================================================================

def _numbers_in_text(value: str) -> List[str]:
    """Extract all numbers (int and float) from a string."""
    return re.findall(r'\d+(?:\.\d+)?', str(value))


def _value_grounded_in_source(value: str, source_text: str) -> bool:
    """Check if at least one number from value appears in source_text."""
    nums = _numbers_in_text(value)
    if not nums:
        return True  # no numbers to validate
    source_lower = source_text.lower()
    return any(n in source_lower for n in nums)


def validate_extracted_fields(frame: PTOFrame, source_text: str) -> PTOFrame:
    """Validate LLM-extracted fields against the source text.

    Checks quantitative fields (numbers, percentages) and named entities
    (drug names) against the actual source text.  If an extracted value
    cannot be grounded, it is nulled out and a warning is logged.  After
    validation the confidence score is recalculated.

    Args:
        frame: PTOFrame with LLM-extracted fields already applied.
        source_text: The combined chunk text that was sent to the LLM.

    Returns:
        The same PTOFrame (mutated in place) with ungrounded fields removed.
    """
    source_lower = source_text.lower()
    nulled_fields: List[str] = []

    # ── key_finding: check percentages/numbers ────────────────────────────
    if frame.key_finding:
        if not _value_grounded_in_source(frame.key_finding, source_text):
            print(f"    [Validation] NULLED key_finding: numbers not in source — "
                  f"{frame.key_finding[:80]}")
            nulled_fields.append("key_finding")
            frame.key_finding = None

    # ── outcomes dict: check each value's numbers ─────────────────────────
    if frame.outcomes:
        bad_keys = []
        for k, v in frame.outcomes.items():
            if not _value_grounded_in_source(str(v), source_text):
                bad_keys.append(k)
        for k in bad_keys:
            print(f"    [Validation] NULLED outcome '{k}': {frame.outcomes[k]}")
            del frame.outcomes[k]
            nulled_fields.append(f"outcomes.{k}")

    # ── n_patients: check number appears in source ────────────────────────
    if frame.n_patients is not None:
        n_str = str(frame.n_patients)
        if n_str not in source_text:
            print(f"    [Validation] NULLED n_patients: {frame.n_patients}")
            nulled_fields.append("n_patients")
            frame.n_patients = None

    # ── dose_fractionation: check Gy value in source ──────────────────────
    if frame.dose_fractionation:
        if not _value_grounded_in_source(frame.dose_fractionation, source_text):
            print(f"    [Validation] NULLED dose_fractionation: "
                  f"{frame.dose_fractionation}")
            nulled_fields.append("dose_fractionation")
            frame.dose_fractionation = None

    # ── chemo_agents: check each agent name appears (case-insensitive) ────
    if frame.chemo_agents:
        valid_agents = []
        for agent in frame.chemo_agents:
            if agent.lower() in source_lower:
                valid_agents.append(agent)
            else:
                print(f"    [Validation] NULLED chemo_agent: '{agent}'")
                nulled_fields.append(f"chemo_agents.{agent}")
        frame.chemo_agents = valid_agents

    # ── lab_requirements: check numeric thresholds in source ──────────────
    if frame.lab_requirements:
        bad_lab_keys = []
        for lab_name, lab_val in frame.lab_requirements.items():
            if not _value_grounded_in_source(str(lab_val), source_text):
                bad_lab_keys.append(lab_name)
        for k in bad_lab_keys:
            print(f"    [Validation] NULLED lab_requirement '{k}': "
                  f"{frame.lab_requirements[k]}")
            del frame.lab_requirements[k]
            nulled_fields.append(f"lab_requirements.{k}")

    # ── Recalculate confidence if fields were removed ─────────────────────
    if nulled_fields:
        print(f"    [Validation] Removed {len(nulled_fields)} ungrounded fields")
        # Re-score confidence using the same logic as LLMPTOExtractor
        filled_fields = sum(1 for v in [
            frame.cancer_type, frame.histology, frame.stage_range,
            frame.disease_status, frame.line_of_therapy, frame.treatment_setting,
            frame.primary_endpoint, frame.key_finding, frame.cohort_summary,
            frame.age_range, frame.ecog_range, frame.histologic_grade,
            frame.prior_therapy_requirements, frame.prior_therapy_exclusions,
            frame.response_status, frame.n_patients, frame.follow_up_duration,
        ] if v)
        filled_fields += min(len(frame.biomarkers), 3)
        filled_fields += min(len(frame.outcomes), 3)
        filled_fields += min(len(frame.inclusion_criteria), 3)
        filled_fields += min(len(frame.exclusion_criteria), 2)
        filled_fields += min(len(frame.lab_requirements), 2)
        filled_fields += min(len(frame.chemo_agents), 2)

        old_conf = frame.confidence
        if filled_fields >= 14:
            frame.confidence = "high"
        elif filled_fields >= 5:
            frame.confidence = "medium"
        else:
            frame.confidence = "low"
        if frame.confidence != old_conf:
            print(f"    [Validation] Confidence changed: {old_conf} → {frame.confidence}")

    return frame


# =============================================================================
# FRAME BUILDER
# =============================================================================

class PTOFrameBuilder:
    """Builds PTO frames from existing chunk data."""
    
    # Keyword categories that indicate each PTO component
    PROFILE_CATEGORIES = {
        'cancer_diagnosis_keywords',
        'cancer_staging_keywords',
        'patient_characteristics_keywords',
        'biomarker_keywords',
        'tumor_characteristics_keywords',
    }
    
    TREATMENT_CATEGORIES = {
        'treatment_keywords',
        'treatment_technique_keywords',
        'dose_keywords',
        'dose_constraint_keywords',
    }
    
    OUTCOME_CATEGORIES = {
        'outcome_keywords',
    }
    
    def __init__(self, min_signal_threshold: int = 2):
        """
        Initialize the frame builder.
        
        Args:
            min_signal_threshold: Minimum keyword signals to include a chunk
        """
        self.extractor = PTOExtractor()
        self.min_signal_threshold = min_signal_threshold
    
    def load_chunks(self, jsonl_path: Path) -> Dict[str, List[Dict]]:
        """
        Load chunks from JSONL and group by doc_id.
        
        Args:
            jsonl_path: Path to section windows JSONL
            
        Returns:
            Dictionary mapping doc_id to list of chunks
        """
        chunks_by_doc: Dict[str, List[Dict]] = defaultdict(list)
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    doc_id = chunk.get('doc_id', 'unknown')
                    chunks_by_doc[doc_id].append(chunk)
                except json.JSONDecodeError:
                    continue
        
        print(f"✓ Loaded {sum(len(v) for v in chunks_by_doc.values())} chunks from {len(chunks_by_doc)} documents")
        return dict(chunks_by_doc)
    
    def _get_keyword_matches(self, chunk: Dict) -> Dict[str, List[str]]:
        """Extract keyword_matches from chunk metadata."""
        metadata = chunk.get('metadata', {}) or {}
        return metadata.get('keyword_matches', {}) or {}
    
    def _count_signals(self, chunk: Dict, categories: Set[str]) -> int:
        """Count keyword signals in specified categories."""
        kw_matches = self._get_keyword_matches(chunk)
        count = 0
        for cat in categories:
            if cat in kw_matches:
                count += len(kw_matches[cat])
        return count
    
    def _has_signals(self, chunk: Dict, categories: Set[str]) -> bool:
        """Check if chunk has any signals in specified categories."""
        kw_matches = self._get_keyword_matches(chunk)
        for cat in categories:
            if cat in kw_matches and kw_matches[cat]:
                return True
        return False
    
    def _identify_pto_chunks(self, chunks: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Identify chunks with profile, treatment, and outcome signals.
        
        Returns:
            Tuple of (profile_chunks, treatment_chunks, outcome_chunks)
        """
        profile_chunks = []
        treatment_chunks = []
        outcome_chunks = []
        
        for chunk in chunks:
            if self._has_signals(chunk, self.PROFILE_CATEGORIES):
                profile_chunks.append(chunk)
            if self._has_signals(chunk, self.TREATMENT_CATEGORIES):
                treatment_chunks.append(chunk)
            if self._has_signals(chunk, self.OUTCOME_CATEGORIES):
                outcome_chunks.append(chunk)
        
        return profile_chunks, treatment_chunks, outcome_chunks
    
    def _generate_pto_id(self, doc_id: str, frame_num: int = 0) -> str:
        """Generate a stable, unique PTO frame ID."""
        base = f"{doc_id}__pto_{frame_num}"
        return hashlib.md5(base.encode()).hexdigest()[:16]
    
    def _build_frame_text(self, frame: PTOFrame) -> str:
        """Build the text that will be embedded for semantic search.

        Uses all available fields (LLM-extracted or regex-extracted) to produce
        a dense, searchable summary.  Format:
          STUDY: ...
          PATIENT: ...
          ELIGIBILITY: ...
          TREATMENT: ...
          OUTCOMES: ...
        """
        parts = []

        # Study identity
        study_parts = []
        if frame.study_phase:
            study_parts.append(frame.study_phase)
        if frame.disease_area:
            study_parts.append(frame.disease_area)
        if frame.cohort_summary:
            study_parts.append(frame.cohort_summary)
        if study_parts:
            parts.append(f"STUDY: {', '.join(study_parts)}")

        # Patient / profile
        profile_parts = []
        if frame.cancer_type:
            profile_parts.append(frame.cancer_type)
        if frame.cancer_subsite:
            profile_parts.append(frame.cancer_subsite)
        if frame.histology:
            profile_parts.append(frame.histology)
        if frame.stage_range or frame.stage:
            profile_parts.append(frame.stage_range or frame.stage)
        if frame.tnm:
            profile_parts.append(frame.tnm)
        if frame.disease_status:
            profile_parts.append(frame.disease_status)
        if frame.line_of_therapy:
            profile_parts.append(frame.line_of_therapy)
        if frame.biomarkers:
            profile_parts.extend(frame.biomarkers[:5])
        if frame.age_range:
            profile_parts.append(f"age {frame.age_range}")
        if frame.ecog_range:
            profile_parts.append(f"ECOG {frame.ecog_range}")
        if frame.patient_features:
            profile_parts.extend(frame.patient_features[:3])
        if profile_parts:
            parts.append(f"PATIENT: {', '.join(profile_parts)}")

        # Eligibility highlights
        elig_parts = []
        if frame.required_biomarkers:
            elig_parts.append(f"required biomarkers: {', '.join(frame.required_biomarkers[:4])}")
        if frame.excluded_biomarkers:
            elig_parts.append(f"excluded biomarkers: {', '.join(frame.excluded_biomarkers[:4])}")
        if frame.prior_therapy_requirements:
            elig_parts.append(f"prior tx required: {frame.prior_therapy_requirements}")
        if frame.prior_therapy_exclusions:
            elig_parts.append(f"prior tx excluded: {frame.prior_therapy_exclusions}")
        if frame.max_prior_lines:
            elig_parts.append(f"max prior lines: {frame.max_prior_lines}")
        if frame.prior_surgery:
            elig_parts.append(f"surgery: {frame.prior_surgery}")
        if frame.prior_radiation:
            elig_parts.append(f"radiation: {frame.prior_radiation}")
        if frame.recurrence_status:
            elig_parts.append(f"recurrence: {frame.recurrence_status}")
        if frame.response_status:
            elig_parts.append(f"response: {frame.response_status}")
        if frame.time_from_prior_therapy:
            elig_parts.append(f"washout: {frame.time_from_prior_therapy}")
        if frame.measurable_disease_required:
            elig_parts.append("measurable disease required")
        if frame.lab_requirements:
            lab_strs = [f"{k} {v}" for k, v in list(frame.lab_requirements.items())[:4]]
            elig_parts.append(f"labs: {', '.join(lab_strs)}")
        if frame.comorbidity_exclusions:
            elig_parts.append(f"excluded comorbidities: {', '.join(frame.comorbidity_exclusions[:3])}")
        if frame.excluded_conditions:
            elig_parts.append(f"excluded conditions: {', '.join(frame.excluded_conditions[:3])}")
        if frame.excluded_medications:
            elig_parts.append(f"excluded meds: {', '.join(frame.excluded_medications[:3])}")
        # Include raw criteria (first few, for embedding richness)
        if frame.inclusion_criteria:
            elig_parts.extend(frame.inclusion_criteria[:4])
        if elig_parts:
            parts.append(f"ELIGIBILITY: {'; '.join(elig_parts)}")

        # Treatment
        treatment_parts = []
        if frame.treatment_setting:
            treatment_parts.append(frame.treatment_setting)
        if frame.treatment_modalities:
            treatment_parts.extend(frame.treatment_modalities)
        if frame.dose_fractionation:
            treatment_parts.append(frame.dose_fractionation)
        if frame.chemo_agents:
            treatment_parts.extend(frame.chemo_agents[:4])
        if frame.treatment_arms:
            treatment_parts.append(frame.treatment_arms[:120])
        elif frame.treatment_details:
            treatment_parts.append(frame.treatment_details[:100])
        if frame.treatment_duration:
            treatment_parts.append(frame.treatment_duration)
        if frame.concurrent_therapies:
            treatment_parts.extend(frame.concurrent_therapies[:2])
        if treatment_parts:
            parts.append(f"TREATMENT: {', '.join(treatment_parts)}")

        # Outcomes
        outcome_parts = []
        if frame.key_finding:
            outcome_parts.append(frame.key_finding)
        elif frame.outcomes:
            outcome_parts.extend(f"{k}: {v}" for k, v in list(frame.outcomes.items())[:6])
        if frame.primary_endpoint and not frame.key_finding:
            outcome_parts.insert(0, f"endpoint: {frame.primary_endpoint}")
        if frame.follow_up_duration:
            outcome_parts.append(f"follow-up: {frame.follow_up_duration}")
        if frame.n_patients:
            outcome_parts.append(f"n={frame.n_patients}")
        if frame.response_criteria:
            outcome_parts.append(f"criteria: {frame.response_criteria}")
        if frame.subgroup_analyses:
            outcome_parts.extend(frame.subgroup_analyses[:2])
        if outcome_parts:
            parts.append(f"OUTCOMES: {'; '.join(outcome_parts)}")

        return " | ".join(parts) if parts else ""

    def _build_section_texts(self, frame: PTOFrame) -> None:
        """Build per-section embedding texts for multi-vector PTO search.

        Instead of one embedding for the entire frame_text (which dilutes
        specificity), we produce 4 focused texts -- one per PTO section --
        each of which gets its own Qdrant point. Phase 0 retrieval can
        then match a patient's biomarker axis against the patient section
        vector, or a treatment query against the treatment section vector,
        without cross-section dilution.
        """
        # Patient profile section
        parts = []
        for val in [frame.cancer_type, frame.cancer_subsite, frame.cancer_location,
                    frame.histology, frame.histologic_grade]:
            if val:
                parts.append(val)
        if frame.stage_range or frame.stage:
            parts.append(frame.stage_range or frame.stage)
        if frame.tnm:
            parts.append(frame.tnm)
        if frame.disease_status:
            parts.append(frame.disease_status)
        if frame.line_of_therapy:
            parts.append(frame.line_of_therapy)
        parts.extend(frame.biomarkers[:5])
        parts.extend(frame.required_biomarkers[:3])
        if frame.age_range:
            parts.append(f"age {frame.age_range}")
        if frame.ecog_range:
            parts.append(f"ECOG {frame.ecog_range}")
        if frame.gender and frame.gender != "all":
            parts.append(frame.gender)
        if frame.cohort_summary:
            parts.append(frame.cohort_summary)
        frame.section_text_patient = " | ".join(p for p in parts if p) if parts else ""

        # Treatment section
        parts = []
        if frame.treatment_setting:
            parts.append(frame.treatment_setting)
        parts.extend(frame.treatment_modalities[:5])
        if frame.dose_fractionation:
            parts.append(frame.dose_fractionation)
        parts.extend(frame.chemo_agents[:5])
        parts.extend(frame.concurrent_therapies[:3])
        if frame.treatment_arms:
            parts.append(frame.treatment_arms[:200])
        elif frame.treatment_details:
            parts.append(frame.treatment_details[:150])
        if frame.treatment_duration:
            parts.append(frame.treatment_duration)
        frame.section_text_treatment = " | ".join(p for p in parts if p) if parts else ""

        # Outcome section
        parts = []
        if frame.key_finding:
            parts.append(frame.key_finding)
        if frame.primary_endpoint:
            parts.append(f"primary: {frame.primary_endpoint}")
        for k, v in list(frame.outcomes.items())[:8]:
            parts.append(f"{k}: {v}")
        if frame.follow_up_duration:
            parts.append(f"follow-up: {frame.follow_up_duration}")
        if frame.n_patients:
            parts.append(f"n={frame.n_patients}")
        parts.extend(frame.subgroup_analyses[:3])
        frame.section_text_outcome = " | ".join(p for p in parts if p) if parts else ""

        # Eligibility section
        parts = []
        if frame.required_biomarkers:
            parts.append(f"required: {', '.join(frame.required_biomarkers[:4])}")
        if frame.excluded_biomarkers:
            parts.append(f"excluded biomarkers: {', '.join(frame.excluded_biomarkers[:4])}")
        if frame.prior_therapy_requirements:
            parts.append(f"prior tx required: {frame.prior_therapy_requirements}")
        if frame.prior_therapy_exclusions:
            parts.append(f"prior tx excluded: {frame.prior_therapy_exclusions}")
        if frame.prior_surgery:
            parts.append(f"surgery: {frame.prior_surgery}")
        if frame.prior_radiation:
            parts.append(f"radiation: {frame.prior_radiation}")
        if frame.recurrence_status:
            parts.append(f"recurrence: {frame.recurrence_status}")
        parts.extend(frame.inclusion_criteria[:5])
        parts.extend(frame.exclusion_criteria[:5])
        if frame.comorbidity_exclusions:
            parts.append(f"comorbidity exclusions: {', '.join(frame.comorbidity_exclusions[:3])}")
        if frame.excluded_conditions:
            parts.append(f"excluded conditions: {', '.join(frame.excluded_conditions[:3])}")
        frame.section_text_eligibility = " | ".join(p for p in parts if p) if parts else ""

    def _calculate_confidence(self, frame: PTOFrame) -> str:
        """Calculate confidence level based on signal density."""
        total_signals = (
            frame.profile_signal_count + 
            frame.treatment_signal_count + 
            frame.outcome_signal_count
        )
        
        has_all_three = (
            frame.profile_signal_count > 0 and
            frame.treatment_signal_count > 0 and
            frame.outcome_signal_count > 0
        )
        
        if has_all_three and total_signals >= 10:
            return "high"
        elif has_all_three or total_signals >= 5:
            return "medium"
        else:
            return "low"
    
    def build_frame_for_document(self, doc_id: str, chunks: List[Dict]) -> Optional[PTOFrame]:
        """
        Build a PTO frame for a single document.
        
        Args:
            doc_id: Document identifier
            chunks: List of chunks for this document
            
        Returns:
            PTOFrame if sufficient signals found, None otherwise
        """
        # Identify chunks with relevant signals
        profile_chunks, treatment_chunks, outcome_chunks = self._identify_pto_chunks(chunks)
        
        # Skip if missing key components
        if not (profile_chunks or treatment_chunks or outcome_chunks):
            return None
        
        # Get document metadata from first chunk
        first_chunk = chunks[0] if chunks else {}
        category = first_chunk.get('category', 'unknown')
        source_doc_dir = first_chunk.get('source_doc_dir_name', '')
        doc_meta = first_chunk.get('doc_meta', {})
        
        # Combine text from relevant chunks
        profile_text = " ".join([c.get('text', '') for c in profile_chunks])
        treatment_text = " ".join([c.get('text', '') for c in treatment_chunks])
        outcome_text = " ".join([c.get('text', '') for c in outcome_chunks])
        all_text = profile_text + " " + treatment_text + " " + outcome_text
        
        # Extract structured fields
        stage_group, tnm_stage = self.extractor.extract_stage(profile_text)
        dose = self.extractor.extract_dose(treatment_text)
        outcomes = self.extractor.extract_outcomes(outcome_text)
        biomarkers = self.extractor.extract_biomarkers(all_text)
        modalities = self.extractor.extract_treatment_modalities(treatment_text)
        chemo_agents = self.extractor.extract_chemo_agents(treatment_text)
        
        # Collect evidence chunk IDs
        evidence_ids = list(set(
            [c.get('chunk_id') for c in profile_chunks] +
            [c.get('chunk_id') for c in treatment_chunks] +
            [c.get('chunk_id') for c in outcome_chunks]
        ))
        
        # Count signals
        profile_signals = sum(self._count_signals(c, self.PROFILE_CATEGORIES) for c in profile_chunks)
        treatment_signals = sum(self._count_signals(c, self.TREATMENT_CATEGORIES) for c in treatment_chunks)
        outcome_signals = sum(self._count_signals(c, self.OUTCOME_CATEGORIES) for c in outcome_chunks)
        
        # Create frame
        frame = PTOFrame(
            pto_id=self._generate_pto_id(doc_id),
            doc_id=doc_id,
            category=category,
            cancer_type=category if category != 'unknown' else None,
            stage=stage_group,
            tnm=tnm_stage,
            biomarkers=biomarkers,
            treatment_modalities=modalities,
            dose_fractionation=dose,
            chemo_agents=chemo_agents,
            outcomes=outcomes,
            evidence_chunk_ids=evidence_ids,
            source_doc_dir_name=source_doc_dir,
            doc_meta=doc_meta,
            profile_signal_count=profile_signals,
            treatment_signal_count=treatment_signals,
            outcome_signal_count=outcome_signals,
        )
        
        # Calculate confidence and frame text
        frame.confidence = self._calculate_confidence(frame)
        frame.frame_text = self._build_frame_text(frame)
        self._build_section_texts(frame)

        return frame
    
    def build_all_frames(
        self, 
        chunks_by_doc: Dict[str, List[Dict]],
        min_confidence: str = "low"
    ) -> List[PTOFrame]:
        """
        Build PTO frames for all documents.
        
        Args:
            chunks_by_doc: Dictionary mapping doc_id to chunks
            min_confidence: Minimum confidence level to include ("low", "medium", "high")
            
        Returns:
            List of PTOFrame objects
        """
        confidence_levels = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_levels.get(min_confidence, 0)
        
        frames = []
        skipped = 0
        
        for doc_id, chunks in chunks_by_doc.items():
            frame = self.build_frame_for_document(doc_id, chunks)
            
            if frame is None:
                skipped += 1
                continue
            
            # Filter by confidence
            frame_level = confidence_levels.get(frame.confidence, 0)
            if frame_level >= min_level:
                frames.append(frame)
            else:
                skipped += 1
        
        print(f"✓ Built {len(frames)} PTO frames (skipped {skipped} documents)")
        return frames
    
    def save_frames(self, frames: List[PTOFrame], output_path: Path):
        """Save frames to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for frame in frames:
                f.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")
        
        print(f"✓ Saved {len(frames)} frames to {output_path}")
    
    def print_summary(self, frames: List[PTOFrame]):
        """Print summary statistics."""
        if not frames:
            print("No frames to summarize.")
            return

        n = len(frames)
        confidence_counts = defaultdict(int)
        method_counts = defaultdict(int)
        has = defaultdict(int)

        for frame in frames:
            confidence_counts[frame.confidence] += 1
            method_counts[frame.extraction_method] += 1
            if frame.stage or frame.stage_range or frame.tnm:
                has["stage"] += 1
            if frame.dose_fractionation:
                has["dose"] += 1
            if frame.outcomes:
                has["outcomes"] += 1
            if frame.biomarkers:
                has["biomarkers"] += 1
            if frame.histology:
                has["histology"] += 1
            if frame.line_of_therapy:
                has["line_of_therapy"] += 1
            if frame.disease_status:
                has["disease_status"] += 1
            if frame.treatment_setting:
                has["treatment_setting"] += 1
            if frame.age_range:
                has["age_range"] += 1
            if frame.ecog_range:
                has["ecog_range"] += 1
            if frame.inclusion_criteria:
                has["inclusion_criteria"] += 1
            if frame.exclusion_criteria:
                has["exclusion_criteria"] += 1
            if frame.cohort_summary:
                has["cohort_summary"] += 1
            if frame.key_finding:
                has["key_finding"] += 1

        def _pct(count):
            return f"{count:4d} ({100*count/n:.1f}%)"

        print("\n" + "=" * 60)
        print("PTO FRAME SUMMARY")
        print("=" * 60)
        print(f"Total frames:     {n}")
        print(f"\nExtraction method:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method:12s}: {_pct(count)}")
        print(f"\nConfidence distribution:")
        print(f"  High:   {_pct(confidence_counts['high'])}")
        print(f"  Medium: {_pct(confidence_counts['medium'])}")
        print(f"  Low:    {_pct(confidence_counts['low'])}")
        print(f"\nExtraction coverage:")
        print(f"  Stage/TNM:           {_pct(has['stage'])}")
        print(f"  Histology:           {_pct(has['histology'])}")
        print(f"  Biomarkers:          {_pct(has['biomarkers'])}")
        print(f"  Dose/fractionation:  {_pct(has['dose'])}")
        print(f"  Outcomes:            {_pct(has['outcomes'])}")
        print(f"  Disease status:      {_pct(has['disease_status'])}")
        print(f"  Line of therapy:     {_pct(has['line_of_therapy'])}")
        print(f"  Treatment setting:   {_pct(has['treatment_setting'])}")
        print(f"  Age range:           {_pct(has['age_range'])}")
        print(f"  ECOG range:          {_pct(has['ecog_range'])}")
        print(f"  Inclusion criteria:  {_pct(has['inclusion_criteria'])}")
        print(f"  Exclusion criteria:  {_pct(has['exclusion_criteria'])}")
        print(f"  Cohort summary:      {_pct(has['cohort_summary'])}")
        print(f"  Key finding:         {_pct(has['key_finding'])}")
        print("=" * 60)


# =============================================================================
# QDRANT UPSERTER
# =============================================================================

class QdrantFrameUpserter:
    """Upsert PTO frames to Qdrant."""
    
    def __init__(
        self, 
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the upserter.
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            collection_name: Target collection name
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, PayloadSchemaType
            import openai
        except ImportError:
            raise ImportError("Please install: pip install qdrant-client openai")
        
        self.client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
            timeout=60  # 60 second timeout for cloud connections
        )
        self.collection_name = collection_name
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
    
    def _embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def ensure_payload_index(self):
        """Create payload index for node_type if it doesn't exist."""
        from qdrant_client.models import PayloadSchemaType
        
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="node_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print("✓ Created payload index for 'node_type'")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("✓ Payload index for 'node_type' already exists")
            else:
                print(f"⚠ Could not create payload index: {e}")
    
    def upsert_frames(self, frames: List[PTOFrame], batch_size: int = 50):
        """
        Upsert PTO frames to Qdrant with per-section embeddings.

        For each frame, creates up to 5 Qdrant points:
        - 1 point with node_type="pto_frame" (full frame_text, backward compat)
        - 1 point with node_type="pto_frame_patient" (section_text_patient)
        - 1 point with node_type="pto_frame_treatment" (section_text_treatment)
        - 1 point with node_type="pto_frame_outcome" (section_text_outcome)
        - 1 point with node_type="pto_frame_eligibility" (section_text_eligibility)

        Section points are only created when their text exceeds 20 chars.
        All points for a frame share the same payload (doc_id, pto_id, etc.)
        with node_type overridden per section.

        Args:
            frames: List of PTOFrame objects
            batch_size: Batch size for embedding and upserts
        """
        from qdrant_client.models import PointStruct
        import uuid

        # Collect all (text, node_type, frame) tuples
        SECTION_MAP = [
            ("frame_text", "pto_frame"),
            ("section_text_patient", "pto_frame_patient"),
            ("section_text_treatment", "pto_frame_treatment"),
            ("section_text_outcome", "pto_frame_outcome"),
            ("section_text_eligibility", "pto_frame_eligibility"),
        ]

        embed_items: List[Tuple[str, str, PTOFrame]] = []  # (text, node_type, frame)
        for frame in frames:
            for attr, node_type in SECTION_MAP:
                text = getattr(frame, attr, "")
                if text and len(text) > 20:
                    embed_items.append((text, node_type, frame))

        if not embed_items:
            print("No valid frame texts to upsert.")
            return

        # Count breakdown
        type_counts = {}
        for _, nt, _ in embed_items:
            type_counts[nt] = type_counts.get(nt, 0) + 1
        print(f"Upserting {len(embed_items)} points from {len(frames)} frames:")
        for nt, cnt in sorted(type_counts.items()):
            print(f"  {nt}: {cnt}")

        # Batch embed all texts
        all_texts = [item[0] for item in embed_items]
        print(f"Generating embeddings for {len(all_texts)} texts...")
        all_embeddings = self._embed_texts(all_texts, batch_size=batch_size)

        # Create points
        points = []
        for (text, node_type, frame), embedding in zip(embed_items, all_embeddings):
            payload = frame.to_dict()
            payload["node_type"] = node_type  # Override per section
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch_pts = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_pts,
            )
            print(f"  Upserted batch {i // batch_size + 1}/"
                  f"{(len(points) - 1) // batch_size + 1}")

        print(f"✓ Successfully upserted {len(points)} PTO points "
              f"({type_counts.get('pto_frame', 0)} full + "
              f"{len(points) - type_counts.get('pto_frame', 0)} section)")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build PTO (Patient→Treatment→Outcome) frames from chunk data"
    )
    parser.add_argument(
        "input_jsonl",
        type=Path,
        help="Path to input JSONL file (section windows with keyword metadata)"
    )
    parser.add_argument(
        "output_jsonl",
        type=Path,
        help="Path to output JSONL file for PTO frames"
    )
    parser.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum confidence level to include (default: low)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use GPT-4o-mini for comprehensive extraction (costs ~$0.01/study). "
             "Falls back to regex-only if --openai-api-key is not provided."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for LLM extraction (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Limit number of studies to process (useful for testing)"
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Upsert frames to Qdrant after building"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        help="Qdrant server URL (required if --upsert)"
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        help="Qdrant API key (required if --upsert)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        help="Qdrant collection name (required if --upsert)"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key for embeddings (required if --upsert or --use-llm)"
    )

    args = parser.parse_args()

    # Load .env if available (so keys don't need to be passed as flags)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed — that's fine

    import os
    # Fall back to environment variables for keys
    if not args.openai_api_key:
        args.openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not args.qdrant_url:
        args.qdrant_url = os.environ.get("QDRANT_URL")
    if not args.qdrant_api_key:
        args.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not args.collection_name:
        args.collection_name = os.environ.get("QDRANT_COLLECTION")

    # Validate args
    if args.upsert:
        if not all([args.qdrant_url, args.qdrant_api_key, args.collection_name, args.openai_api_key]):
            parser.error("--upsert requires OpenAI + Qdrant keys (via flags or .env)")
    if args.use_llm and not args.openai_api_key:
        parser.error("--use-llm requires OPENAI_API_KEY (via --openai-api-key flag or .env)")

    # Build frames
    print("\n" + "=" * 60)
    print("PTO FRAME BUILDER")
    if args.use_llm:
        print(f"  Mode: LLM-enhanced extraction ({args.llm_model})")
    else:
        print("  Mode: Regex-only extraction")
    print("=" * 60)

    builder = PTOFrameBuilder()

    print(f"\n Loading chunks from: {args.input_jsonl}")
    chunks_by_doc = builder.load_chunks(args.input_jsonl)

    # Optionally limit for testing
    if args.max_studies:
        doc_ids = list(chunks_by_doc.keys())[:args.max_studies]
        chunks_by_doc = {k: chunks_by_doc[k] for k in doc_ids}
        print(f"  Limited to {args.max_studies} studies for testing")

    print(f"\n Building PTO frames (min confidence: {args.min_confidence})...")
    frames = builder.build_all_frames(chunks_by_doc, min_confidence=args.min_confidence)

    # ── LLM enhancement pass ─────────────────────────────────────────────
    if args.use_llm and frames:
        print(f"\n Running LLM extraction on {len(frames)} frames...")
        llm_extractor = LLMPTOExtractor(
            openai_api_key=args.openai_api_key,
            model=args.llm_model,
        )

        enhanced_count = 0
        failed_count = 0
        for i, frame in enumerate(frames):
            # Gather the best text for this doc — prefer eligibility/methods/results sections
            doc_chunks = chunks_by_doc.get(frame.doc_id, [])

            # Filter out table rows — they're structured data, not useful for LLM extraction
            narrative_chunks = [
                c for c in doc_chunks
                if c.get("chunk_type") != "table_row"
                and c.get("chunk_granularity") != "table_row_atomic"
                and len((c.get("text", "") or "")) > 50
            ]

            # Sort: eligibility/methods/abstract first
            def _chunk_priority(c):
                text_lower = (c.get("text", "") or "").lower()
                section_lower = (c.get("section", "") or "").lower()
                score = 0
                if any(k in text_lower for k in ["eligib", "inclusion", "exclusion", "enroll", "criteria"]):
                    score += 10
                if any(k in section_lower for k in ["abstract", "summary", "background"]):
                    score += 8
                if any(k in section_lower for k in ["method", "patient", "eligib", "design", "study_design"]):
                    score += 7
                if any(k in section_lower for k in ["result", "outcome", "efficacy", "survival"]):
                    score += 5
                if any(k in section_lower for k in ["introduction", "discussion"]):
                    score += 3
                # Prefer section windows with more text
                score += min(len(c.get("text", "")) / 500, 2)
                return -score

            sorted_chunks = sorted(narrative_chunks, key=_chunk_priority)

            # Use title from doc_meta to give LLM context
            title = frame.doc_meta.get("title", "") or ""
            title_prefix = f"Study title: {title}\n\n" if title else ""
            study_text = title_prefix + " ".join(c.get("text", "") for c in sorted_chunks[:10])[:8000]

            if len(study_text) < 100:
                print(f"    [{i+1}] SKIP: {frame.doc_id[:40]}... — insufficient text "
                      f"({len(narrative_chunks)} narrative chunks, {len(doc_chunks)} total)")
                failed_count += 1
                continue

            # Debug: show what's being sent
            if i < 2:  # First 2 studies in detail
                print(f"    [{i+1}] Sending {len(study_text)} chars to LLM "
                      f"({len(narrative_chunks)} narrative / {len(doc_chunks)} total chunks)")
                print(f"         Title: {title[:80]}...")
                print(f"         Text preview: {study_text[len(title_prefix):len(title_prefix)+200]}...")

            llm_result = llm_extractor.extract_from_text(study_text)
            if llm_result:
                # Debug: show what LLM returned for first 2 studies
                if i < 2:
                    non_null = {k: v for k, v in llm_result.items()
                                if v is not None and v != "null" and v != [] and v != {}}
                    print(f"         LLM returned {len(non_null)} non-null fields: {list(non_null.keys())}")

                llm_extractor.apply_to_frame(frame, llm_result)
                # Rebuild frame_text and section texts with enriched data
                frame.frame_text = builder._build_frame_text(frame)
                builder._build_section_texts(frame)
                enhanced_count += 1
            else:
                if i < 2:
                    print(f"         LLM returned None/empty")
                failed_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                stats = llm_extractor.get_stats()
                print(f"  Progress: {i+1}/{len(frames)} "
                      f"({enhanced_count} enhanced, {failed_count} failed, "
                      f"{stats['total_tokens']:,} tokens used)")

        stats = llm_extractor.get_stats()
        print(f"\n LLM extraction complete:")
        print(f"  Enhanced: {enhanced_count}/{len(frames)} frames")
        print(f"  Failed: {failed_count}")
        print(f"  API calls: {stats['calls']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        est_cost = stats['total_tokens'] / 1_000_000 * 0.30  # ~$0.15 input + $0.60 output avg
        print(f"  Estimated cost: ~${est_cost:.2f}")

    # Print summary
    builder.print_summary(frames)
    
    # Save frames
    print(f"\n💾 Saving frames to: {args.output_jsonl}")
    builder.save_frames(frames, args.output_jsonl)
    
    # Optionally upsert to Qdrant
    if args.upsert:
        print(f"\n☁️  Upserting to Qdrant collection: {args.collection_name}")
        upserter = QdrantFrameUpserter(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            collection_name=args.collection_name,
            openai_api_key=args.openai_api_key
        )
        upserter.ensure_payload_index()
        upserter.upsert_frames(frames)
    
    print("\n✅ Done!")
    
    # Print example frame
    if frames:
        print("\n" + "-" * 60)
        print("EXAMPLE FRAME:")
        print("-" * 60)
        example = frames[0]
        print(f"  PTO ID:     {example.pto_id}")
        print(f"  Doc ID:     {example.doc_id}")
        print(f"  Category:   {example.category}")
        print(f"  Stage:      {example.stage or example.tnm or 'N/A'}")
        print(f"  Biomarkers: {example.biomarkers or 'N/A'}")
        print(f"  Modalities: {example.treatment_modalities or 'N/A'}")
        print(f"  Dose:       {example.dose_fractionation or 'N/A'}")
        print(f"  Outcomes:   {example.outcomes or 'N/A'}")
        print(f"  Confidence: {example.confidence}")
        print(f"  Frame Text: {example.frame_text[:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
