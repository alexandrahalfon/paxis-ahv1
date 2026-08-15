"""
Populate Lookup Tables Script

This script extracts vocabulary data from existing project files and populates
the PostgreSQL lookup tables in the study-profiles database.

Tables to populate:
- cancer_types (synonyms, keywords, subtypes)
- drugs (if empty)
- toxicity_types (if empty)
- radiation_techniques (if empty)
- biomarker_definitions (if empty)

Data sources:
- oncology_clinical_trial_vocabulary_list.rtf (JSON with cancer types + synonyms)
- data/keywords/extractor_keywords.json (staging, diagnosis keywords)
- enhanced_rag_service.py (ONCOLOGY_EXPANSIONS, REVERSE_EXPANSIONS)
- chemotherapy_words (drug data)
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Any, Set
from pathlib import Path

import asyncpg

# ============================================================================
# CANCER TYPES DATA (from oncology_clinical_trial_vocabulary_list.rtf)
# ============================================================================

CANCER_TYPES_DATA = {
    # Code -> {label, synonyms, keywords, subtypes}
    
    # CNS / Brain
    "cns": {
        "label": "Central Nervous System Tumors",
        "synonyms": [
            "CNS tumor", "brain tumor", "brain cancer", "intracranial tumor",
            "ependymoma", "medulloblastoma", "astrocytoma", "pilocytic astrocytoma",
            "anaplastic ependymoma", "pediatric brain tumor", "DIPG",
            "diffuse intrinsic pontine glioma", "craniopharyngioma", "ATRT",
            "atypical teratoid rhabdoid tumor", "oligodendroglioma"
        ],
        "keywords": [
            "glioblastoma", "GBM", "glioblastoma multiforme", "glioma",
            "high grade glioma", "low grade glioma", "WHO grade", "meningioma"
        ],
        "subtypes": [
            "glioblastoma", "astrocytoma", "oligodendroglioma", "ependymoma",
            "medulloblastoma", "meningioma", "craniopharyngioma"
        ]
    },
    
    # Breast
    "breast": {
        "label": "Breast Cancer",
        "synonyms": [
            "breast carcinoma", "mammary cancer", "breast neoplasm",
            "breast tumor", "mammary carcinoma"
        ],
        "keywords": [
            "TNBC", "triple negative", "HER2+", "HER2 positive",
            "ER+", "estrogen receptor", "PR+", "progesterone receptor",
            "invasive ductal", "invasive lobular", "DCIS", "ductal carcinoma in situ",
            "LCIS", "lobular carcinoma in situ", "inflammatory breast cancer"
        ],
        "subtypes": [
            "invasive ductal carcinoma", "invasive lobular carcinoma",
            "ductal carcinoma in situ", "triple negative breast cancer",
            "HER2 positive", "hormone receptor positive"
        ]
    },
    
    # Head and Neck
    "h&n": {
        "label": "Head and Neck Cancer",
        "synonyms": [
            "H&N", "HNSCC", "head and neck squamous cell carcinoma",
            "oral cancer", "oropharyngeal cancer", "laryngeal cancer",
            "pharyngeal cancer", "oral cavity cancer"
        ],
        "keywords": [
            "HPV positive", "HPV negative", "p16 positive", "p16 negative",
            "oropharynx", "nasopharynx", "hypopharynx", "larynx",
            "oral cavity", "tongue", "tonsil", "base of tongue"
        ],
        "subtypes": [
            "oropharyngeal squamous cell carcinoma",
            "nasopharyngeal carcinoma", "NPC",
            "laryngeal squamous cell carcinoma",
            "oral cavity squamous cell carcinoma",
            "hypopharyngeal carcinoma"
        ]
    },
    
    # Lung
    "lung": {
        "label": "Lung Cancer",
        "synonyms": [
            "lung carcinoma", "pulmonary cancer", "lung neoplasm",
            "bronchogenic carcinoma"
        ],
        "keywords": [
            "NSCLC", "non-small cell lung cancer", "non small cell",
            "SCLC", "small cell lung cancer", "small cell",
            "adenocarcinoma", "squamous cell", "large cell"
        ],
        "subtypes": [
            "non-small cell lung cancer", "small cell lung cancer",
            "lung adenocarcinoma", "lung squamous cell carcinoma",
            "large cell carcinoma", "carcinoid"
        ]
    },
    
    # Prostate
    "prostate": {
        "label": "Prostate Cancer",
        "synonyms": [
            "prostate carcinoma", "prostatic adenocarcinoma",
            "prostate neoplasm", "prostatic cancer"
        ],
        "keywords": [
            "Gleason", "PSA", "prostate specific antigen",
            "mCRPC", "metastatic castration resistant",
            "mHSPC", "metastatic hormone sensitive",
            "CRPC", "castration resistant"
        ],
        "subtypes": [
            "adenocarcinoma of prostate", "castration resistant prostate cancer",
            "hormone sensitive prostate cancer"
        ]
    },
    
    # GI - Gastrointestinal
    "gi": {
        "label": "Gastrointestinal Cancers",
        "synonyms": [
            "GI tumor", "GI cancer", "gastrointestinal neoplasm",
            "digestive tract cancer"
        ],
        "keywords": [
            "colorectal", "CRC", "colon", "rectal", "anal",
            "esophageal", "gastric", "stomach", "pancreatic",
            "hepatocellular", "HCC", "liver", "cholangiocarcinoma",
            "biliary", "GIST", "gastrointestinal stromal tumor"
        ],
        "subtypes": [
            "colorectal cancer", "colon cancer", "rectal cancer",
            "anal cancer", "esophageal cancer", "gastric cancer",
            "pancreatic cancer", "hepatocellular carcinoma",
            "cholangiocarcinoma"
        ]
    },
    
    # GU - Genitourinary
    "gu": {
        "label": "Genitourinary Cancers",
        "synonyms": [
            "GU tumor", "GU cancer", "urologic cancer"
        ],
        "keywords": [
            "bladder", "urothelial", "MIBC", "muscle invasive bladder",
            "NMIBC", "non-muscle invasive bladder",
            "renal", "kidney", "RCC", "renal cell carcinoma",
            "testicular", "germ cell", "seminoma", "nonseminoma"
        ],
        "subtypes": [
            "bladder cancer", "urothelial carcinoma",
            "renal cell carcinoma", "testicular cancer"
        ]
    },
    
    # Gynecologic
    "gyn": {
        "label": "Gynecologic Cancers",
        "synonyms": [
            "gynecologic carcinoma", "gynecological cancer",
            "female reproductive cancer"
        ],
        "keywords": [
            "cervical", "cervix", "endometrial", "uterine",
            "ovarian", "ovary", "vulvar", "vaginal",
            "HPV", "human papillomavirus"
        ],
        "subtypes": [
            "cervical cancer", "endometrial cancer", "ovarian cancer",
            "vulvar cancer", "vaginal cancer", "uterine sarcoma"
        ]
    },
    
    # Lymphoma
    "lymphoma": {
        "label": "Lymphoma",
        "synonyms": [
            "malignant lymphoma", "lymphoid neoplasm"
        ],
        "keywords": [
            "Hodgkin", "HL", "non-Hodgkin", "NHL",
            "DLBCL", "diffuse large B-cell",
            "follicular", "mantle cell", "marginal zone",
            "Burkitt", "T-cell lymphoma"
        ],
        "subtypes": [
            "Hodgkin lymphoma", "non-Hodgkin lymphoma",
            "diffuse large B-cell lymphoma", "follicular lymphoma",
            "mantle cell lymphoma"
        ]
    },
    
    # Leukemia
    "leukemia": {
        "label": "Leukemia",
        "synonyms": [
            "blood cancer", "hematologic malignancy"
        ],
        "keywords": [
            "AML", "acute myeloid leukemia",
            "ALL", "acute lymphoblastic leukemia",
            "CML", "chronic myeloid leukemia",
            "CLL", "chronic lymphocytic leukemia",
            "MDS", "myelodysplastic syndrome"
        ],
        "subtypes": [
            "acute myeloid leukemia", "acute lymphoblastic leukemia",
            "chronic myeloid leukemia", "chronic lymphocytic leukemia"
        ]
    },
    
    # Melanoma / Skin
    "skin": {
        "label": "Skin Cancer",
        "synonyms": [
            "cutaneous cancer", "cutaneous malignancy"
        ],
        "keywords": [
            "melanoma", "malignant melanoma", "cutaneous melanoma",
            "BCC", "basal cell carcinoma", "SCC", "squamous cell carcinoma",
            "cSCC", "cutaneous squamous cell carcinoma",
            "Merkel cell", "MCC"
        ],
        "subtypes": [
            "melanoma", "basal cell carcinoma", "cutaneous squamous cell carcinoma",
            "Merkel cell carcinoma"
        ]
    },
    
    # Sarcoma
    "sarcoma": {
        "label": "Sarcoma",
        "synonyms": [
            "soft tissue sarcoma", "bone sarcoma"
        ],
        "keywords": [
            "osteosarcoma", "Ewing sarcoma", "rhabdomyosarcoma",
            "liposarcoma", "leiomyosarcoma", "synovial sarcoma",
            "GIST", "gastrointestinal stromal tumor",
            "chondrosarcoma", "undifferentiated pleomorphic sarcoma"
        ],
        "subtypes": [
            "osteosarcoma", "Ewing sarcoma", "rhabdomyosarcoma",
            "liposarcoma", "leiomyosarcoma"
        ]
    },
    
    # Pediatric
    "pediatric": {
        "label": "Pediatric Cancers",
        "synonyms": [
            "childhood cancer", "pediatric oncology",
            "pediatric malignancy"
        ],
        "keywords": [
            "neuroblastoma", "Wilms tumor", "nephroblastoma",
            "retinoblastoma", "hepatoblastoma",
            "pediatric ALL", "pediatric AML",
            "pediatric brain tumor", "rhabdomyosarcoma"
        ],
        "subtypes": [
            "neuroblastoma", "Wilms tumor", "rhabdomyosarcoma",
            "retinoblastoma", "pediatric leukemia"
        ]
    },
    
    # Thyroid
    "thyroid": {
        "label": "Thyroid Cancer",
        "synonyms": [
            "thyroid carcinoma", "thyroid neoplasm"
        ],
        "keywords": [
            "papillary", "follicular", "medullary", "anaplastic",
            "differentiated thyroid cancer", "DTC",
            "RAI", "radioactive iodine"
        ],
        "subtypes": [
            "papillary thyroid carcinoma", "follicular thyroid carcinoma",
            "medullary thyroid carcinoma", "anaplastic thyroid carcinoma"
        ]
    },
    
    # Myeloma
    "myeloma": {
        "label": "Multiple Myeloma",
        "synonyms": [
            "plasma cell myeloma", "MM"
        ],
        "keywords": [
            "multiple myeloma", "plasma cell neoplasm",
            "MGUS", "monoclonal gammopathy",
            "smoldering myeloma"
        ],
        "subtypes": []
    },
    
    # Mesothelioma
    "mesothelioma": {
        "label": "Mesothelioma",
        "synonyms": [
            "malignant mesothelioma"
        ],
        "keywords": [
            "pleural mesothelioma", "peritoneal mesothelioma",
            "asbestos related"
        ],
        "subtypes": [
            "pleural mesothelioma", "peritoneal mesothelioma"
        ]
    },
}


# ============================================================================
# DRUGS DATA (from chemotherapy_words and other sources)
# ============================================================================

DRUGS_DATA = [
    # Alkylating Agents
    {"name": "cyclophosphamide", "generic_name": "Cyclophosphamide", "brand_names": ["Cytoxan"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    {"name": "cisplatin", "generic_name": "Cisplatin", "brand_names": ["Platinol"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    {"name": "carboplatin", "generic_name": "Carboplatin", "brand_names": ["Paraplatin"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    {"name": "oxaliplatin", "generic_name": "Oxaliplatin", "brand_names": ["Eloxatin"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    {"name": "temozolomide", "generic_name": "Temozolomide", "brand_names": ["Temodar"], "drug_class": "Alkylating Agent", "mechanism": "DNA methylation"},
    {"name": "ifosfamide", "generic_name": "Ifosfamide", "brand_names": ["Ifex"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    {"name": "melphalan", "generic_name": "Melphalan", "brand_names": ["Alkeran"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    {"name": "busulfan", "generic_name": "Busulfan", "brand_names": ["Myleran", "Busulfex"], "drug_class": "Alkylating Agent", "mechanism": "DNA crosslinking"},
    
    # Antimetabolites
    {"name": "5-fluorouracil", "generic_name": "Fluorouracil", "brand_names": ["5-FU", "Adrucil"], "drug_class": "Antimetabolite", "mechanism": "Thymidylate synthase inhibition"},
    {"name": "capecitabine", "generic_name": "Capecitabine", "brand_names": ["Xeloda"], "drug_class": "Antimetabolite", "mechanism": "Thymidylate synthase inhibition"},
    {"name": "gemcitabine", "generic_name": "Gemcitabine", "brand_names": ["Gemzar"], "drug_class": "Antimetabolite", "mechanism": "DNA synthesis inhibition"},
    {"name": "methotrexate", "generic_name": "Methotrexate", "brand_names": ["Trexall"], "drug_class": "Antimetabolite", "mechanism": "DHFR inhibition"},
    {"name": "pemetrexed", "generic_name": "Pemetrexed", "brand_names": ["Alimta"], "drug_class": "Antimetabolite", "mechanism": "Multi-target antifolate"},
    {"name": "cytarabine", "generic_name": "Cytarabine", "brand_names": ["Ara-C", "Cytosar"], "drug_class": "Antimetabolite", "mechanism": "DNA polymerase inhibition"},
    
    # Anthracyclines
    {"name": "doxorubicin", "generic_name": "Doxorubicin", "brand_names": ["Adriamycin"], "drug_class": "Anthracycline", "mechanism": "Topoisomerase II inhibition"},
    {"name": "epirubicin", "generic_name": "Epirubicin", "brand_names": ["Ellence"], "drug_class": "Anthracycline", "mechanism": "Topoisomerase II inhibition"},
    {"name": "daunorubicin", "generic_name": "Daunorubicin", "brand_names": ["Cerubidine"], "drug_class": "Anthracycline", "mechanism": "Topoisomerase II inhibition"},
    {"name": "idarubicin", "generic_name": "Idarubicin", "brand_names": ["Idamycin"], "drug_class": "Anthracycline", "mechanism": "Topoisomerase II inhibition"},
    
    # Taxanes / Plant Alkaloids
    {"name": "paclitaxel", "generic_name": "Paclitaxel", "brand_names": ["Taxol", "Abraxane"], "drug_class": "Taxane", "mechanism": "Microtubule stabilization"},
    {"name": "docetaxel", "generic_name": "Docetaxel", "brand_names": ["Taxotere"], "drug_class": "Taxane", "mechanism": "Microtubule stabilization"},
    {"name": "vincristine", "generic_name": "Vincristine", "brand_names": ["Oncovin"], "drug_class": "Vinca Alkaloid", "mechanism": "Microtubule destabilization"},
    {"name": "vinblastine", "generic_name": "Vinblastine", "brand_names": ["Velban"], "drug_class": "Vinca Alkaloid", "mechanism": "Microtubule destabilization"},
    {"name": "vinorelbine", "generic_name": "Vinorelbine", "brand_names": ["Navelbine"], "drug_class": "Vinca Alkaloid", "mechanism": "Microtubule destabilization"},
    
    # Topoisomerase Inhibitors
    {"name": "etoposide", "generic_name": "Etoposide", "brand_names": ["VP-16", "Toposar"], "drug_class": "Topoisomerase Inhibitor", "mechanism": "Topoisomerase II inhibition"},
    {"name": "irinotecan", "generic_name": "Irinotecan", "brand_names": ["Camptosar"], "drug_class": "Topoisomerase Inhibitor", "mechanism": "Topoisomerase I inhibition"},
    {"name": "topotecan", "generic_name": "Topotecan", "brand_names": ["Hycamtin"], "drug_class": "Topoisomerase Inhibitor", "mechanism": "Topoisomerase I inhibition"},
    
    # Targeted Therapies
    {"name": "trastuzumab", "generic_name": "Trastuzumab", "brand_names": ["Herceptin"], "drug_class": "Monoclonal Antibody", "mechanism": "HER2 inhibition"},
    {"name": "pertuzumab", "generic_name": "Pertuzumab", "brand_names": ["Perjeta"], "drug_class": "Monoclonal Antibody", "mechanism": "HER2 inhibition"},
    {"name": "bevacizumab", "generic_name": "Bevacizumab", "brand_names": ["Avastin"], "drug_class": "Monoclonal Antibody", "mechanism": "VEGF inhibition"},
    {"name": "cetuximab", "generic_name": "Cetuximab", "brand_names": ["Erbitux"], "drug_class": "Monoclonal Antibody", "mechanism": "EGFR inhibition"},
    {"name": "rituximab", "generic_name": "Rituximab", "brand_names": ["Rituxan"], "drug_class": "Monoclonal Antibody", "mechanism": "CD20 inhibition"},
    
    # Immunotherapy
    {"name": "pembrolizumab", "generic_name": "Pembrolizumab", "brand_names": ["Keytruda"], "drug_class": "Checkpoint Inhibitor", "mechanism": "PD-1 inhibition"},
    {"name": "nivolumab", "generic_name": "Nivolumab", "brand_names": ["Opdivo"], "drug_class": "Checkpoint Inhibitor", "mechanism": "PD-1 inhibition"},
    {"name": "ipilimumab", "generic_name": "Ipilimumab", "brand_names": ["Yervoy"], "drug_class": "Checkpoint Inhibitor", "mechanism": "CTLA-4 inhibition"},
    {"name": "atezolizumab", "generic_name": "Atezolizumab", "brand_names": ["Tecentriq"], "drug_class": "Checkpoint Inhibitor", "mechanism": "PD-L1 inhibition"},
    {"name": "durvalumab", "generic_name": "Durvalumab", "brand_names": ["Imfinzi"], "drug_class": "Checkpoint Inhibitor", "mechanism": "PD-L1 inhibition"},
    
    # Hormone Therapy
    {"name": "tamoxifen", "generic_name": "Tamoxifen", "brand_names": ["Nolvadex"], "drug_class": "Hormone Therapy", "mechanism": "Estrogen receptor modulator"},
    {"name": "letrozole", "generic_name": "Letrozole", "brand_names": ["Femara"], "drug_class": "Aromatase Inhibitor", "mechanism": "Aromatase inhibition"},
    {"name": "anastrozole", "generic_name": "Anastrozole", "brand_names": ["Arimidex"], "drug_class": "Aromatase Inhibitor", "mechanism": "Aromatase inhibition"},
    {"name": "enzalutamide", "generic_name": "Enzalutamide", "brand_names": ["Xtandi"], "drug_class": "Antiandrogen", "mechanism": "Androgen receptor inhibition"},
    {"name": "abiraterone", "generic_name": "Abiraterone", "brand_names": ["Zytiga"], "drug_class": "CYP17 Inhibitor", "mechanism": "Androgen synthesis inhibition"},
]


# ============================================================================
# BIOMARKER DATA
# ============================================================================

BIOMARKER_DATA = [
    {"name": "HER2", "aliases": ["ERBB2", "HER2/neu"], "category": "Receptor", "cancer_types": ["breast", "gastric"],
     "positive_terms": ["positive", "amplified", "overexpressed", "+"], "negative_terms": ["negative", "non-amplified", "-"]},
    {"name": "ER", "aliases": ["Estrogen Receptor", "ESR1"], "category": "Hormone Receptor", "cancer_types": ["breast"],
     "positive_terms": ["positive", "+"], "negative_terms": ["negative", "-"]},
    {"name": "PR", "aliases": ["Progesterone Receptor", "PGR"], "category": "Hormone Receptor", "cancer_types": ["breast"],
     "positive_terms": ["positive", "+"], "negative_terms": ["negative", "-"]},
    {"name": "EGFR", "aliases": ["ErbB-1", "HER1"], "category": "Receptor", "cancer_types": ["lung", "h&n"],
     "positive_terms": ["mutant", "mutation", "mutated", "+"], "negative_terms": ["wild-type", "wt", "-"]},
    {"name": "ALK", "aliases": ["ALK fusion", "ALK rearrangement"], "category": "Fusion", "cancer_types": ["lung"],
     "positive_terms": ["positive", "fusion", "rearrangement", "+"], "negative_terms": ["negative", "-"]},
    {"name": "KRAS", "aliases": ["KRAS mutation"], "category": "Mutation", "cancer_types": ["lung", "gi"],
     "positive_terms": ["mutant", "mutation", "+"], "negative_terms": ["wild-type", "wt", "-"]},
    {"name": "BRAF", "aliases": ["BRAF V600E", "BRAF mutation"], "category": "Mutation", "cancer_types": ["melanoma", "gi"],
     "positive_terms": ["mutant", "mutation", "V600E", "+"], "negative_terms": ["wild-type", "wt", "-"]},
    {"name": "PD-L1", "aliases": ["CD274", "B7-H1"], "category": "Checkpoint", "cancer_types": ["lung", "melanoma", "bladder"],
     "positive_terms": ["positive", "high", "+"], "negative_terms": ["negative", "low", "-"]},
    {"name": "MSI", "aliases": ["Microsatellite Instability", "MSI-H", "dMMR"], "category": "Genomic", "cancer_types": ["gi", "endometrial"],
     "positive_terms": ["high", "MSI-H", "dMMR"], "negative_terms": ["stable", "MSS", "pMMR"]},
    {"name": "TMB", "aliases": ["Tumor Mutational Burden", "TMB-H"], "category": "Genomic", "cancer_types": ["melanoma", "lung"],
     "positive_terms": ["high", "TMB-H"], "negative_terms": ["low"]},
    {"name": "BRCA1", "aliases": ["BRCA1 mutation"], "category": "Mutation", "cancer_types": ["breast", "ovarian"],
     "positive_terms": ["mutant", "mutation", "+"], "negative_terms": ["wild-type", "wt", "-"]},
    {"name": "BRCA2", "aliases": ["BRCA2 mutation"], "category": "Mutation", "cancer_types": ["breast", "ovarian", "prostate"],
     "positive_terms": ["mutant", "mutation", "+"], "negative_terms": ["wild-type", "wt", "-"]},
    {"name": "PSA", "aliases": ["Prostate Specific Antigen"], "category": "Serum Marker", "cancer_types": ["prostate"],
     "positive_terms": ["elevated", "high"], "negative_terms": ["low", "undetectable"]},
    {"name": "HPV", "aliases": ["Human Papillomavirus", "p16"], "category": "Viral", "cancer_types": ["h&n", "cervical"],
     "positive_terms": ["positive", "p16+", "+"], "negative_terms": ["negative", "p16-", "-"]},
    {"name": "Ki-67", "aliases": ["MKI67", "proliferation index"], "category": "Proliferation", "cancer_types": ["breast", "neuroendocrine"],
     "positive_terms": ["high", "elevated"], "negative_terms": ["low"]},
]


# ============================================================================
# RADIATION TECHNIQUES DATA
# ============================================================================

RADIATION_TECHNIQUES_DATA = [
    {"raw_value": "IMRT", "normalized": "Intensity Modulated Radiation Therapy", "category": "External Beam", "is_stereotactic": False},
    {"raw_value": "VMAT", "normalized": "Volumetric Modulated Arc Therapy", "category": "External Beam", "is_stereotactic": False},
    {"raw_value": "3D-CRT", "normalized": "3D Conformal Radiation Therapy", "category": "External Beam", "is_stereotactic": False},
    {"raw_value": "SBRT", "normalized": "Stereotactic Body Radiation Therapy", "category": "Stereotactic", "is_stereotactic": True},
    {"raw_value": "SRS", "normalized": "Stereotactic Radiosurgery", "category": "Stereotactic", "is_stereotactic": True},
    {"raw_value": "WBRT", "normalized": "Whole Brain Radiation Therapy", "category": "External Beam", "is_stereotactic": False},
    {"raw_value": "PBI", "normalized": "Partial Breast Irradiation", "category": "Breast", "is_stereotactic": False},
    {"raw_value": "APBI", "normalized": "Accelerated Partial Breast Irradiation", "category": "Breast", "is_stereotactic": False},
    {"raw_value": "HDR", "normalized": "High Dose Rate Brachytherapy", "category": "Brachytherapy", "is_stereotactic": False},
    {"raw_value": "LDR", "normalized": "Low Dose Rate Brachytherapy", "category": "Brachytherapy", "is_stereotactic": False},
    {"raw_value": "EBRT", "normalized": "External Beam Radiation Therapy", "category": "External Beam", "is_stereotactic": False},
    {"raw_value": "Proton", "normalized": "Proton Beam Therapy", "category": "Particle", "is_stereotactic": False},
    {"raw_value": "CSI", "normalized": "Craniospinal Irradiation", "category": "CNS", "is_stereotactic": False},
    {"raw_value": "TBI", "normalized": "Total Body Irradiation", "category": "Systemic", "is_stereotactic": False},
]


# ============================================================================
# DATABASE POPULATION FUNCTIONS
# ============================================================================

async def get_pool(database: str = "study-profiles"):
    """Create connection pool to study-profiles database."""
    return await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "34.21.60.224"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=database,
        min_size=2,
        max_size=10,
    )


async def update_cancer_types(pool: asyncpg.Pool):
    """Update cancer_types table with comprehensive synonyms and keywords."""
    async with pool.acquire() as conn:
        for code, data in CANCER_TYPES_DATA.items():
            # Check if exists
            existing = await conn.fetchrow(
                "SELECT id, synonyms, keywords, subtypes FROM cancer_types WHERE code = $1",
                code
            )
            
            if existing:
                # Merge arrays (avoid duplicates)
                current_synonyms = set(existing['synonyms'] or [])
                current_keywords = set(existing['keywords'] or [])
                current_subtypes = set(existing.get('subtypes') or [])
                
                new_synonyms = list(current_synonyms | set(data['synonyms']))
                new_keywords = list(current_keywords | set(data['keywords']))
                new_subtypes = list(current_subtypes | set(data.get('subtypes', [])))
                
                await conn.execute("""
                    UPDATE cancer_types 
                    SET synonyms = $2, keywords = $3, subtypes = $4, updated_at = NOW()
                    WHERE code = $1
                """, code, new_synonyms, new_keywords, new_subtypes)
                
                print(f"  ✓ Updated {code}: {len(new_synonyms)} synonyms, {len(new_keywords)} keywords")
            else:
                # Insert new
                await conn.execute("""
                    INSERT INTO cancer_types (code, label, synonyms, keywords, subtypes, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                """, code, data['label'], data['synonyms'], data['keywords'], data.get('subtypes', []))
                
                print(f"  ✓ Inserted {code}: {data['label']}")


async def populate_drugs(pool: asyncpg.Pool):
    """Populate drugs table if empty."""
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM drugs")
        
        if count > 0:
            print(f"  ⚠ drugs table already has {count} rows, skipping")
            return
        
        for drug in DRUGS_DATA:
            await conn.execute("""
                INSERT INTO drugs (name, generic_name, brand_names, drug_class, mechanism, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (name) DO NOTHING
            """, 
                drug['name'], 
                drug['generic_name'], 
                drug['brand_names'],
                drug['drug_class'],
                drug['mechanism']
            )
        
        print(f"  ✓ Inserted {len(DRUGS_DATA)} drugs")


async def populate_biomarkers(pool: asyncpg.Pool):
    """Populate biomarker_definitions table if empty."""
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM biomarker_definitions")
        
        if count > 0:
            print(f"  ⚠ biomarker_definitions table already has {count} rows, skipping")
            return
        
        for biomarker in BIOMARKER_DATA:
            await conn.execute("""
                INSERT INTO biomarker_definitions (name, aliases, category, created_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (name) DO NOTHING
            """,
                biomarker['name'],
                biomarker['aliases'],
                biomarker['category']
            )
        
        print(f"  ✓ Inserted {len(BIOMARKER_DATA)} biomarkers")


async def populate_radiation_techniques(pool: asyncpg.Pool):
    """Populate radiation_techniques table if empty or missing entries."""
    async with pool.acquire() as conn:
        for tech in RADIATION_TECHNIQUES_DATA:
            await conn.execute("""
                INSERT INTO radiation_techniques (raw_value, normalized_technique, technique_category, is_stereotactic, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (raw_value) DO UPDATE SET
                    normalized_technique = EXCLUDED.normalized_technique,
                    technique_category = EXCLUDED.technique_category,
                    is_stereotactic = EXCLUDED.is_stereotactic
            """,
                tech['raw_value'],
                tech['normalized'],
                tech['category'],
                tech['is_stereotactic']
            )
        
        print(f"  ✓ Upserted {len(RADIATION_TECHNIQUES_DATA)} radiation techniques")


async def verify_population(pool: asyncpg.Pool):
    """Verify tables are populated."""
    async with pool.acquire() as conn:
        tables = [
            ("cancer_types", "SELECT COUNT(*), SUM(array_length(synonyms, 1)) FROM cancer_types"),
            ("drugs", "SELECT COUNT(*) FROM drugs"),
            ("biomarker_definitions", "SELECT COUNT(*) FROM biomarker_definitions"),
            ("radiation_techniques", "SELECT COUNT(*) FROM radiation_techniques"),
        ]
        
        print("\n📊 Table Population Status:")
        print("-" * 50)
        
        for table_name, query in tables:
            try:
                result = await conn.fetchrow(query)
                if "SUM" in query:
                    print(f"  {table_name}: {result[0]} rows, {result[1] or 0} total synonyms")
                else:
                    print(f"  {table_name}: {result[0]} rows")
            except Exception as e:
                print(f"  {table_name}: Error - {e}")


async def main():
    """Main function to populate all tables."""
    print("=" * 60)
    print("POPULATING LOOKUP TABLES IN study-profiles DATABASE")
    print("=" * 60)
    
    # Get password from env or prompt
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        password = input("Enter PostgreSQL password: ")
        os.environ["POSTGRES_PASSWORD"] = password
    
    pool = await get_pool()
    
    try:
        print("\n1. Updating cancer_types table...")
        await update_cancer_types(pool)
        
        print("\n2. Populating drugs table...")
        await populate_drugs(pool)
        
        print("\n3. Populating biomarker_definitions table...")
        await populate_biomarkers(pool)
        
        print("\n4. Populating radiation_techniques table...")
        await populate_radiation_techniques(pool)
        
        # Verify
        await verify_population(pool)
        
        print("\n✅ Population complete!")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
