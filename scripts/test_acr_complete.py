#!/usr/bin/env python3
"""
Comprehensive ACR Board Exam Test for Paxis RAG Pipeline
=========================================================

Tests RAG system against all 50 ACR radiation oncology board questions
with ground truth answers and citations.

Features:
- All 50 questions included
- Keyword-based scoring
- Semantic similarity scoring
- Citation matching analysis
- JSON + HTML reports
- Side-by-side comparison

Usage:
python test_acr_complete.py              # Run all 50 questions
python test_acr_complete.py --limit 10   # Run first 10
python test_acr_complete.py --category breast  # Run specific category

Output:
- acr_complete_report_TIMESTAMP.json (detailed results)
- acr_complete_report_TIMESTAMP.html (beautiful side-by-side comparison)
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Get project root (parent of scripts directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Change to project root and add to path
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Verify required environment variables
required_vars = ['OPENAI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

try:
    from src.api.services.enhanced_rag_service import get_enhanced_rag_service
except ImportError as e:
    print(f"ERROR: Cannot import enhanced_rag_service: {e}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Make sure you have the src/ directory with api/services/enhanced_rag_service.py")
    sys.exit(1)


# ============================================
# ALL 50 ACR BOARD EXAM QUESTIONS
# ============================================

ACR_QUESTIONS = [
    {
        "id": "Q01",
        "number": "01-Q25",
        "question": "A 45 year-old woman undergoes wide local excision for a 3 cm, grade 3, invasive lobular carcinoma. On final pathology, the sentinel node is negative, and a single margin is close at < 1mm. Which is the MOST appropriate next step in local management?",
        "answer": "Hypofractionated whole breast RT with a boost",
        "key_concepts": ["hypofractionated", "whole breast", "boost", "no ink on tumor", "no re-excision", "invasive lobular carcinoma"],
        "citations": ["Moran", "2014", "Journal of Clinical Oncology", "ASTRO"],
        "rationale": "Hypofractionated whole breast radiation therapy is standard of care for women undergoing radiation therapy to the breast, and boost is indicated based on her age, grade, and narrow margin. The accepted standard for margins for invasive breast cancer is no tumor on ink.",
        "category": "breast"
    },
    {
        "id": "Q02",
        "number": "02-Q28",
        "question": "A patient with a 2.5 cm cN0 breast cancer is determined to have a 2 mm deposit of tumor in a sentinel lymph node biopsy performed at breast conserving surgery. What outcome(s) is/are associated with completion axillary LND?",
        "answer": "Increases the risk of lymphedema without reducing axillary recurrence",
        "key_concepts": ["lymphedema", "no benefit", "axillary recurrence", "ACOSOG Z-11", "IBCSG 23-01"],
        "citations": ["Galimberti", "IBCSG 23-01", "2013", "Lancet Oncol"],
        "rationale": "Both the IBSCG 23-01 and ACOSOG Z-11 trials demonstrate an increased risk of lymphedema without difference in axillary recurrence for women with T1-2 clinically node negative breast cancer patients detected to have 1-2 positive nodes on sentinel node biopsy.",
        "category": "breast"
    },
    {
        "id": "Q03",
        "number": "03-Q38",
        "question": "What is the BEST treatment for a 55 year-old female who underwent breast-conserving surgery for a pT1cN1mi cM0 ER+ HER2- breast cancer and 21 gene recurrence score of 22?",
        "answer": "RT followed by endocrine therapy",
        "key_concepts": ["no chemotherapy", "21 gene score", "recurrence score <25", "radiation", "endocrine therapy", "TAILORx"],
        "citations": ["Sparano", "TAILORx", "NEJM", "2018"],
        "rationale": "The patient is over the age of 50 and had a 21 gene recurrence score of <25 so no chemotherapy is recommended. She underwent BCS so adjuvant radiation is standard of care. Her tumor was ER+ so she needs adjuvant endocrine therapy.",
        "category": "breast"
    },
    {
        "id": "Q04",
        "number": "04-Q96",
        "question": "A 65 year-old male diagnosed with MIBC decided to proceed with bladder preservation as his treatment choice. Given no other comorbidities, normal tolerance doses and concurrent chemotherapy, what total bladder radiation dose in 1.8 - 2.0Gy/fx is appropriate?",
        "answer": "64 Gy",
        "key_concepts": ["64 Gy", "65 Gy", "bladder preservation", "concurrent chemotherapy", "conventional fractionation"],
        "citations": ["Tester", "RTOG 8802", "1996"],
        "rationale": "65Gy in conventional fractionation is the appropriate standard regimen for bladder preservation in the setting of concurrent chemotherapy.",
        "category": "GU"
    },
    {
        "id": "Q05",
        "number": "05-Q159",
        "question": "A 65-year old smoker was found to have an asymptomatic mediastinal mass on screening low dose CT imaging that was biopsy proven to be an adenocarcinoma. Imaging showed numerous hepatic metastases and no brain metastasis. What is the recommended next step in management?",
        "answer": "Request PD-L1 and mutation testing",
        "key_concepts": ["PD-L1 testing", "mutation testing", "biomarker", "metastatic", "asymptomatic", "NCCN"],
        "citations": ["NCCN", "2021"],
        "rationale": "PD-L1 testing and mutation testing should be considered first in an asymptomatic good performance status patient to determine first line therapies.",
        "category": "lung"
    },
    {
        "id": "Q06",
        "number": "06-Q162",
        "question": "What dose constraint is acceptable in thoracic RT for Stage III NSCLC receiving chemoRT to 66 Gy in 33 fractions?",
        "answer": "Total lung V20 ≤ 35%",
        "key_concepts": ["V20", "35%", "40%", "lung constraint", "total lung", "chemoradiation"],
        "citations": ["NCCN", "2021"],
        "rationale": "Total lung dose constraint of V20≤35%-40% is the standard dose constraints for conventionally fractionated chemoradiation.",
        "category": "lung"
    },
    {
        "id": "Q07",
        "number": "07-Q195",
        "question": "An 8-year-old child has classic medulloblastoma status post GTR with no radiographic evidence of metastatic disease and no tumor cells in the CSF. What is the recommended RT dose and volume?",
        "answer": "23.4 Gy CSI, then 30.6 Gy boost to the resection cavity + margin",
        "key_concepts": ["23.4 Gy", "CSI", "30.6 Gy boost", "54 Gy total", "55.8 Gy", "average risk", "standard risk", "medulloblastoma"],
        "citations": ["Packer", "2013", "Neuro Oncol", "COG A9961"],
        "rationale": "The current standard treatment for average (standard) risk medulloblastoma is 23.4 Gy CSI with a focal boost to a total of 54-55.8 Gy.",
        "category": "CNS"
    },
    {
        "id": "Q08",
        "number": "08-Q199",
        "question": "A 16 year-old has localized Ewing sarcoma of the sacrum and is receiving standard chemotherapy and definitive RT for local control. What dose should be given to the gross disease?",
        "answer": "55.8 Gy",
        "key_concepts": ["55.8 Gy", "45 Gy", "10.8 Gy boost", "Ewing sarcoma", "definitive"],
        "citations": ["Womer", "2012", "JCO"],
        "rationale": "The standard definitive radiation therapy dose for non-vertebral Ewing sarcoma is 55.8 Gy (45 Gy to the initial volume, followed by a 10.8 Gy boost).",
        "category": "sarcoma"
    },
    {
        "id": "Q09",
        "number": "09-Q16",
        "question": "A 42-year-old female was incidentally found to have an infiltrative contrast enhancing asymptomatic 6 cm mass in the right rectus abdominus muscle. Biopsy reveals desmoid fibromatosis. What treatment should be initially recommended?",
        "answer": "Active surveillance",
        "key_concepts": ["active surveillance", "observation", "wait and see", "asymptomatic", "desmoid"],
        "citations": ["Desmoid Tumor Working Group", "2020"],
        "rationale": "For patients with asymptomatic, stable tumors, an active surveillance strategy is recommended.",
        "category": "sarcoma"
    },
    {
        "id": "Q10",
        "number": "10-Q29",
        "question": "A 50 year-old patient presents with breast cancer involving axillary and infraclavicular lymph nodes and exhibits a pCR at mastectomy and sentinel node biopsy following neo-adjuvant chemotherapy. What is the current standard of care for adjuvant RT?",
        "answer": "Adjuvant RT to the chest wall and regional nodes",
        "key_concepts": ["chest wall", "regional nodes", "N3b", "stage III", "pCR", "still need radiation", "PMRT"],
        "citations": ["McGuire", "2007"],
        "rationale": "With initial level 1 and 3 lymph node involvement, the clinical N stage is N3b and the anatomic stage group is III. In this group, even in the setting of pCR, there is an overall advantage to radiation therapy.",
        "category": "breast"
    },
    {
        "id": "Q11",
        "number": "11-Q33",
        "question": "What were the 5-year local control results for the 1-week regimens of the FAST-Forward phase III RCT as compared to 3-week hypofractionated breast RT?",
        "answer": "Non-inferior ipsilateral breast tumor relapse",
        "key_concepts": ["non-inferior", "IBTR", "26 Gy", "27 Gy", "5 fractions", "1 week", "FAST-Forward"],
        "citations": ["Brunt", "FAST-Forward", "2020", "Lancet"],
        "rationale": "The FAST-Forward trial demonstrated non-inferior IBTR in the 1 week courses vs 3 weeks.",
        "category": "breast"
    },
    {
        "id": "Q12",
        "number": "12-Q42",
        "question": "For a 54 year-old woman with newly diagnosed metastatic ER- PR- HER2+ breast cancer and an ECOG of 0, what is the preferred first-line systemic therapy?",
        "answer": "Trastuzumab, pertuzumab, and docetaxel",
        "key_concepts": ["trastuzumab", "pertuzumab", "docetaxel", "CLEOPATRA", "HER2 positive", "triple therapy"],
        "citations": ["Swain", "CLEOPATRA", "2020", "Lancet Oncology"],
        "rationale": "The Phase III CLEOPATRA study showed the addition of pertuzumab to trastuzumab and docetaxel increased the 8-year overall survival rate.",
        "category": "breast"
    },
    {
        "id": "Q13",
        "number": "13-Q66",
        "question": "For patients with resectable non-metastatic rectal adenocarcinoma, what is an acceptable treatment regimen?",
        "answer": "Short course RT (25 Gy in 5 Fx) followed by surgery within 1 week",
        "key_concepts": ["25 Gy", "5 fractions", "short course", "surgery within 1 week", "Stockholm III"],
        "citations": ["Erlandsson", "Stockholm III", "2017", "Lancet Oncol"],
        "rationale": "The Stockholm III trial showed short course radiation therapy (25 Gy in 5 fractions) followed by surgery within 1 week is acceptable.",
        "category": "GI"
    },
    {
        "id": "Q14",
        "number": "14-Q76",
        "question": "Which concurrent chemotherapy regimen is MOST appropriate for preoperative chemoRT to 50.4 Gy in a HER2+ T3N1 adenocarcinoma of the distal esophagus?",
        "answer": "Carboplatin and paclitaxel",
        "key_concepts": ["carboplatin", "paclitaxel", "HER2", "esophageal", "no trastuzumab benefit", "RTOG 1010"],
        "citations": ["Safran", "RTOG 1010", "2020"],
        "rationale": "The NRG/RTOG 1010 trial showed no benefit with the addition of trastuzumab to standard preoperative chemoradiation.",
        "category": "GI"
    },
    {
        "id": "Q15",
        "number": "15-Q83",
        "question": "In the Stockholm III trial, what was the benefit of delay to surgery when using short course RT?",
        "answer": "Decreased surgical complications",
        "key_concepts": ["decreased complications", "4-8 weeks delay", "36% vs 28%", "Stockholm III"],
        "citations": ["Erlandsson", "2017", "Lancet Oncol"],
        "rationale": "There was a slightly higher risk of surgical complications in short-course (36%) vs short-course-delay (28%).",
        "category": "GI"
    },
    {
        "id": "Q16",
        "number": "16-Q95",
        "question": "For stage I seminoma treated with orchiectomy alone and no adjuvant treatment, what are the 15-year relapse and salvage rates respectively?",
        "answer": "20% relapse and 100% salvage",
        "key_concepts": ["20%", "relapse", "100%", "salvage", "cure", "active surveillance"],
        "citations": ["Kollmannsberger", "2015", "J Clin Oncol"],
        "rationale": "The risk-adapted management approach is based on low rates of 15 year relapse of about 20% and a very high rate of cure for those who relapse.",
        "category": "GU"
    },
    {
        "id": "Q17",
        "number": "17-Q96",
        "question": "A 65 year-old male diagnosed with MIBC decided to proceed with bladder preservation. Given no other comorbidities, normal tolerance doses and concurrent chemotherapy, what total bladder radiation dose in 1.8-2.0 Gy/fx is appropriate?",
        "answer": "64 Gy",
        "key_concepts": ["64 Gy", "65 Gy", "bladder preservation", "concurrent chemotherapy"],
        "citations": ["Tester", "RTOG 8802", "1996"],
        "rationale": "65 Gy in conventional fractionation is the appropriate standard regimen for bladder preservation.",
        "category": "GU"
    },
    {
        "id": "Q18",
        "number": "18-Q98",
        "question": "A 73 year-old male is diagnosed with prostate cancer and his staging workup reveals 6 bone metastases. Which is considered a category 1 treatment recommendation per the NCCN?",
        "answer": "Abiraterone and prednisone with ADT",
        "key_concepts": ["abiraterone", "ADT", "high volume", "metastatic", "category 1", "NCCN"],
        "citations": ["NCCN", "2020"],
        "rationale": "This patient has high volume metastatic disease. Phase 3 studies showed abiraterone with ADT is associated with improved survival.",
        "category": "GU"
    },
    {
        "id": "Q19",
        "number": "19-Q105",
        "question": "What was the result of the MRC testicular tumor working group TE10 trial, which randomized men to paraaortic strip or paraaortic plus ipsilateral iliac LN RT?",
        "answer": "Side effects were decreased in the paraaortic strip RT arm",
        "key_concepts": ["decreased side effects", "azoospermia", "11% vs 35%", "paraaortic strip", "TE10"],
        "citations": ["Fossa", "1999", "JCO"],
        "rationale": "The short-term side effects and incidence of azoospermia were significantly decreased using paraaortic strip RT.",
        "category": "GU"
    },
    {
        "id": "Q20",
        "number": "20-Q110",
        "question": "As demonstrated in the PORTEC-1 study, what is the absolute improvement in local control at 5 years with the addition of RT to stage I intermediate risk endometrial cancer patients?",
        "answer": "10%",
        "key_concepts": ["10%", "4% vs 14%", "local control", "no OS benefit", "PORTEC-1"],
        "citations": ["Creutzberg", "PORTEC", "2000", "Lancet"],
        "rationale": "The 5-year actuarial locoregional recurrence rates were 4% in the radiotherapy group and 14% in the control group.",
        "category": "GYN"
    },
    {
        "id": "Q21",
        "number": "21-Q118",
        "question": "What is a benefit of IMRT over 3D-CRT after hysterectomy for gynecologic cancers?",
        "answer": "Less acute GI side effects",
        "key_concepts": ["less GI toxicity", "acute", "IMRT", "3D-CRT", "20.4% vs 7.8%", "RTOG 1203"],
        "citations": ["Klopp", "RTOG 1203", "2018", "J Clin Oncol"],
        "rationale": "In RTOG 1203, fewer women on the IMRT arm required antidiarrheal medications compared to standard RT.",
        "category": "GYN"
    },
    {
        "id": "Q22",
        "number": "22-Q133",
        "question": "Which treatment offers the highest chance of larynx preservation for locally advanced (T3/T4) larynx cancer?",
        "answer": "Concurrent chemoRT",
        "key_concepts": ["concurrent chemoradiation", "81.7%", "larynx preservation", "RTOG 91-11"],
        "citations": ["Forastiere", "RTOG 91-11", "2013"],
        "rationale": "RTOG 91-11 showed larynx preservation was highest in the concurrent chemoRT arm (10-yr 81.7%).",
        "category": "H&N"
    },
    {
        "id": "Q23",
        "number": "23-Q137",
        "question": "A 52 year-old non-smoker presents with a single right level II lymph node that measures 2.5 cm. His history, physical exam, and fiberoptic laryngoscopy are negative for a primary site. What are the next steps for work-up?",
        "answer": "FNA, HPV testing, contrast-enhanced CT, PET-CT, exam under anesthesia",
        "key_concepts": ["FNA", "HPV testing", "PET-CT", "exam under anesthesia", "PET before EUA", "unknown primary"],
        "citations": ["Maghami", "ASCO", "2020"],
        "rationale": "PET-CT should always be done prior to examination under anesthesia, as PET may guide biopsies.",
        "category": "H&N"
    },
    {
        "id": "Q24",
        "number": "24-Q144",
        "question": "A patient presents with a right lateral oral tongue SCC, 1.5 cm in size, 8mm depth of invasion, and ipsilateral adenopathy in level Ib (2 cm) and IIa (2.5cm). There is overt extranodal extension of the 1b node. What is the clinical stage?",
        "answer": "T2N3bM0, Stage IVb",
        "key_concepts": ["T2", "depth of invasion >5mm", "N3b", "ENE", "Stage IVb", "8th edition"],
        "citations": ["AJCC", "8th Edition", "2017"],
        "rationale": "A primary tumor ≤2cm with depth of invasion >5mm is T2. Any node(s) with clinically overt ENE is N3b.",
        "category": "H&N"
    },
    {
        "id": "Q25",
        "number": "25-Q165",
        "question": "Which is the MOST appropriate first-line systemic therapy for metastatic squamous cell carcinoma of the lung with PD-L1 positivity of 10%?",
        "answer": "Carboplatin-paclitaxel-pembrolizumab",
        "key_concepts": ["carboplatin", "paclitaxel", "pembrolizumab", "PD-L1 10%", "KEYNOTE 407", "squamous"],
        "citations": ["Paz-Ares", "2018", "NEJM"],
        "rationale": "Combining pembrolizumab plus chemo was demonstrated to be superior in Keynote 407 in squamous cell carcinoma.",
        "category": "lung"
    },
    {
        "id": "Q26",
        "number": "26-Q312",
        "question": "A 29-year-old man is found to have a 6.5-cm pure seminoma of the right testicle with rete testis invasion. His serum tumor markers are within normal range, and staging workup is otherwise negative. Which treatment is most appropriate?",
        "answer": "Radiation dose of 20 to 30 Gy to the paraaortic lymph nodes alone",
        "key_concepts": ["20 Gy", "25 Gy", "30 Gy", "paraaortic", "adjuvant", "stage I seminoma"],
        "citations": ["Warde", "2002", "JCO"],
        "rationale": "Adjuvant paraaortic lymph node irradiation is the best choice for stage I seminoma with high-risk features.",
        "category": "GU"
    },
    {
        "id": "Q27",
        "number": "27-Q249",
        "question": "An 8-year-old patient with embryonal rhabdomyosarcoma has five lung metastases measuring 0.5 to 1.5 cm at presentation that respond completely to induction chemotherapy. What is the most appropriate approach for lung metastases as part of consolidation therapy?",
        "answer": "15 Gy in 10 fx to both whole lungs",
        "key_concepts": ["15 Gy", "10 fractions", "whole lung irradiation", "WLI", "consolidation"],
        "citations": ["Rodeberg", "2005"],
        "rationale": "Patients with lung metastases should receive whole lung irradiation, 15 Gy in 10 fractions.",
        "category": "peds"
    },
    {
        "id": "Q28",
        "number": "28-Q38",
        "question": "In patients with thymoma, what is the preferred choice of chemotherapy regimen for patients with unresectable disease?",
        "answer": "Cisplatin, doxorubicin and cyclophosphamide (CAP)",
        "key_concepts": ["CAP", "cisplatin", "doxorubicin", "cyclophosphamide", "thymoma", "70% response"],
        "citations": ["Kim", "2004", "Lung Cancer"],
        "rationale": "For locally advanced thymomas, induction chemotherapy with CAP is recommended with an overall response rate of 70% or more.",
        "category": "thoracic"
    },
    {
        "id": "Q29",
        "number": "29-Q49",
        "question": "What is the optimal management for a medically inoperable patient with a peripherally located 4.2 cm adenocarcinoma of the upper lobe with no evidence of metastatic disease?",
        "answer": "SBRT using 3 fractions of 18 Gy",
        "key_concepts": ["SBRT", "18 Gy", "3 fractions", "54 Gy total", ">3 cm tumor", "peripheral"],
        "citations": ["Timmerman", "RTOG 0236"],
        "rationale": "The standard dose for peripheral tumors up to 5 cm is 18 Gy x 3 fractions based on RTOG 0236.",
        "category": "lung"
    },
    {
        "id": "Q30",
        "number": "30-Q68",
        "question": "What is the proper management for a SCLC patient with symptomatic thoracic disease but has extensive stage disease based on a solitary asymptomatic brain metastasis?",
        "answer": "Chemotherapy followed by RT to the thorax and brain, based on no evidence of progression",
        "key_concepts": ["chemotherapy first", "30 Gy", "10 fractions", "chest RT", "brain RT", "extensive stage"],
        "citations": ["Slotman", "2014", "Lancet Oncol"],
        "rationale": "Patients with extensive stage SCLC receive induction chemotherapy, then consolidation with chest and brain RT if no progression.",
        "category": "lung"
    },
    {
        "id": "Q31",
        "number": "31-Q129",
        "question": "What is the appropriate management for a patient with a history of metastatic NSCLC who presents with a 2 week history of bilateral lower extremity weakness and incontinence with imaging demonstrating spinal cord compression at T8?",
        "answer": "RT alone to 30 Gy in 2 weeks",
        "key_concepts": ["30 Gy", "palliative RT", "2 weeks symptoms", "no surgery", "cord compression"],
        "citations": ["Patchell", "2005", "Lancet"],
        "rationale": "Palliative radiation is appropriate since duration of symptoms does not justify surgical decompression per Patchell trial.",
        "category": "palliative"
    },
    {
        "id": "Q32",
        "number": "32-Q108",
        "question": "What is the MOST appropriate hormone ablation therapy duration to be given with external beam radiation therapy for a 58-year-old patient in good health with stage T3b, Gleason 9 (4+5), PSA 25.0 prostate cancer?",
        "answer": "28 months",
        "key_concepts": ["28 months", "long-term ADT", "high risk", "RTOG 92-02"],
        "citations": ["Horwitz", "RTOG 92-02", "2008", "JCO"],
        "rationale": "RTOG 92-02 showed superior outcomes for patients with high-risk prostate cancer with long-term hormone ablation therapy (28 months).",
        "category": "GU"
    },
    {
        "id": "Q33",
        "number": "33-Q157",
        "question": "What is the MOST appropriate management for a 52-year-old male with a T4N2M0 nasopharyngeal carcinoma?",
        "answer": "Concurrent chemoradiation with adjuvant chemotherapy",
        "key_concepts": ["concurrent chemoRT", "adjuvant chemotherapy", "cisplatin", "Intergroup 0099", "NPC"],
        "citations": ["Al-Sarraf", "Intergroup 0099", "1998"],
        "rationale": "The standard of care is concurrent chemoradiation with cisplatin-based chemotherapy with or without adjuvant chemotherapy.",
        "category": "H&N"
    },
    {
        "id": "Q34",
        "number": "34-Q171",
        "question": "What is the BEST indication for postmastectomy radiation in a patient with clinical stage II breast cancer treated by neoadjuvant chemotherapy?",
        "answer": "Pathologic N1",
        "key_concepts": ["pN1", "pathologic node positive", "postmastectomy RT", "neoadjuvant", "NSABP"],
        "citations": ["Buchholz", "2008", "J Clin Oncol"],
        "rationale": "Pathologic node positivity after neoadjuvant chemotherapy is most associated with subsequent risk of local-regional recurrence.",
        "category": "breast"
    },
    {
        "id": "Q35",
        "number": "35-Q261",
        "question": "For a healthy 45-year-old non-smoker with a cT1N1M0 squamous cell carcinoma of the right palatine tonsil with a level 2A node, what is the MOST appropriate treatment?",
        "answer": "Radiotherapy to the right tonsil and ipsilateral cervical lymph nodes",
        "key_concepts": ["ipsilateral RT", "unilateral", "T1N1", "excellent prognosis", "95% control", "tonsil"],
        "citations": ["Garden", "2004"],
        "rationale": "This patient has excellent prognosis with 5-year locoregional control of 95% with radiotherapy alone to ipsilateral fields.",
        "category": "H&N"
    },
    {
        "id": "Q36",
        "number": "36-Q10",
        "question": "What is recommended for a pT1b1N0M0 squamous cell carcinoma of the cervix with negative margins, invasion of the deep third of the cervical stroma, and extensive LVSI?",
        "answer": "EBRT",
        "key_concepts": ["EBRT", "external beam", "intermediate risk", "GOG 92", "deep invasion", "LVSI"],
        "citations": ["Rotman", "GOG 92", "2006"],
        "rationale": "Patients with intermediate risk factors benefit from postoperative external beam radiotherapy per GOG 92.",
        "category": "GYN"
    },
    {
        "id": "Q37",
        "number": "37-Q21",
        "question": "What is the MOST appropriate radiation dose and volume for treatment of a 6-year-old girl with a localized anaplastic supratentorial ependymoma status post gross total resection?",
        "answer": "59.4 Gy to the tumor bed",
        "key_concepts": ["59.4 Gy", "tumor bed", "no CSI", "localized", "anaplastic", "ependymoma"],
        "citations": ["Landau", "2013", "IJROBP"],
        "rationale": "For localized ependymoma, CSI is not appropriate. Treatment is 59.4 Gy to the tumor bed.",
        "category": "CNS"
    },
    {
        "id": "Q38",
        "number": "38-Q60",
        "question": "What are the expected 2-year locoregional control and distant freedom from progression rates following IMRT based chemoradiotherapy for nasopharyngeal carcinoma per RTOG 0225?",
        "answer": "Locoregional: 90%, Distant: 85%",
        "key_concepts": ["90%", "85%", "locoregional control", "distant control", "RTOG 0225", "IMRT", "NPC"],
        "citations": ["Lee", "RTOG 0225", "2009", "J Clin Oncol"],
        "rationale": "The RTOG 0225 trial demonstrated improved outcomes with IMRT for nasopharyngeal carcinoma.",
        "category": "H&N"
    },
    {
        "id": "Q39",
        "number": "39-Q106",
        "question": "What subsequent treatment would be recommended to a 43-year-old female with newly diagnosed cT4dN2M0 triple negative breast carcinoma who has an excellent clinical response to neoadjuvant chemotherapy?",
        "answer": "Modified radical mastectomy followed by PMRT",
        "key_concepts": ["mastectomy", "PMRT", "inflammatory", "trimodality", "regardless of response", "IBC"],
        "citations": ["Dawood", "2011", "Annals of Oncol"],
        "rationale": "Inflammatory breast cancer requires trimodality care: neoadjuvant chemotherapy, mastectomy, and PMRT regardless of response.",
        "category": "breast"
    },
    {
        "id": "Q40",
        "number": "40-Q139",
        "question": "A patient with metastatic NSCLC with four sites of bony metastasis, which describes an appropriate management strategy?",
        "answer": "Platinum-based 2-drug combination therapy with pembrolizumab if the PD-L1 status is 30%",
        "key_concepts": ["pembrolizumab", "chemotherapy", "immunotherapy", "PD-L1 30%", "KEYNOTE 189", "metastatic"],
        "citations": ["Gandhi", "KEYNOTE 189", "2018", "NEJM"],
        "rationale": "KEYNOTE 189 demonstrated improved OS and PFS with the combination of chemotherapy and pembrolizumab.",
        "category": "lung"
    },
    {
        "id": "Q41",
        "number": "41-Q189",
        "question": "For DCIS, which feature is associated with an elevated risk of in-breast recurrence?",
        "answer": "Clinical detection",
        "key_concepts": ["clinical detection", "non-mammographic", "risk factor", "IBTR", "EORTC 10853"],
        "citations": ["Donker", "2013", "J Clin Oncol"],
        "rationale": "In EORTC 10853, risk factors for IBTR included age<40, clinical detection, positive margins, and solid type.",
        "category": "breast"
    },
    {
        "id": "Q42",
        "number": "42-Q1",
        "question": "A 66-year old female with pT1 pN0(sn) grade 2 ER + breast cancer is treated with breast conserving surgery with negative margins and intends to take endocrine therapy. What is the risk of ipsilateral breast tumor recurrence at 5 years without RT?",
        "answer": "4%",
        "key_concepts": ["4%", "4.1%", "PRIME II", "no radiation", "elderly", ">=65 years"],
        "citations": ["Kunkler", "PRIME II", "2015", "Lancet Oncology"],
        "rationale": "In PRIME II study, rates of IBTR at 5 years were 1.3% with RT vs 4.1% without radiation.",
        "category": "breast"
    },
    {
        "id": "Q43",
        "number": "43-Q8",
        "question": "A female with a cT3N1M0 ER/PR- Her2+ breast cancer receives neoadjuvant TCHP followed by mastectomy with sentinel lymph node biopsy and achieves a pCR. What adjuvant therapy is recommended?",
        "answer": "RT to the chest wall and regional lymph nodes with concurrent trastuzumab +/- pertuzumab",
        "key_concepts": ["PMRT", "chest wall", "regional nodes", "stage III", "pCR still needs RT", "trastuzumab"],
        "citations": ["McGuire", "2007"],
        "rationale": "Clinical stage III breast cancer has high local recurrence risk even with pCR; PMRT is recommended.",
        "category": "breast"
    },
    {
        "id": "Q44",
        "number": "44-Q35",
        "question": "For locoregionally advanced NPC, which induction chemotherapy regimen given prior to concurrent chemoRT improved OS compared to standard in a randomized controlled Phase III trial?",
        "answer": "Gemcitabine and cisplatin",
        "key_concepts": ["gemcitabine", "cisplatin", "induction", "NPC", "OS benefit", "phase III"],
        "citations": ["Zhang", "2019", "NEJM"],
        "rationale": "A Phase III trial showed gemcitabine and cisplatin induction improved 3-year overall survival.",
        "category": "H&N"
    },
    {
        "id": "Q45",
        "number": "45-Q40",
        "question": "A patient undergoes radical inguinal orchiectomy for an 8cm right testicular seminoma. Pathology showed invasion of rete testes and LVSI; no nodal involvement on scans; serum markers returning to normal after surgery. What is the recommended RT technique?",
        "answer": "20 Gy with para-aortic only field",
        "key_concepts": ["20 Gy", "paraaortic", "stage I", "seminoma", "adjuvant", "10 fractions"],
        "citations": ["Jones", "TE18", "2016", "JCO"],
        "rationale": "The standard treatment for Stage I Seminoma is 20 Gy in 10 fractions to the para-aortic fields.",
        "category": "GU"
    },
    {
        "id": "Q46",
        "number": "46-Q78",
        "question": "What is the absolute four-year OS benefit of docetaxel when added to long-course ADT + RT for patients with high-risk localized prostate cancer?",
        "answer": "5%",
        "key_concepts": ["5%", "4%", "93% vs 89%", "docetaxel", "RTOG 0521", "high-risk"],
        "citations": ["Rosenthal", "RTOG 0521", "2019", "JCO"],
        "rationale": "Patients treated with docetaxel had a 4-year OS benefit of 93% vs 89% (p=0.034).",
        "category": "GU"
    },
    {
        "id": "Q47",
        "number": "47-Q171",
        "question": "What is the most appropriate radiation treatment for a 10-year-old patient with a 3 cm upper extremity embryonal rhabdomyosarcoma without nodal or distant metastases that has a partial response to induction chemotherapy?",
        "answer": "50.4 Gy to the primary site and no nodal radiation",
        "key_concepts": ["50.4 Gy", "28 fractions", "gross residual", "no nodal RT", "rhabdomyosarcoma"],
        "citations": ["Donaldson", "2001"],
        "rationale": "The established dose for gross residual disease is 50.4 Gy in 28 fractions when given with concurrent chemotherapy.",
        "category": "peds"
    },
    {
        "id": "Q48",
        "number": "48-Q14",
        "question": "For cT1N0M0 breast cancer treated with breast-conserving surgery and sentinel lymph node biopsy with 1 of 2 sentinel lymph nodes positive without ECE, which subsequent locoregional treatment option is best supported by level 1 evidence?",
        "answer": "No further axillary surgery and whole breast radiation +/- draining lymphatics",
        "key_concepts": ["no ALND", "whole breast RT", "Z-11", "AMAROS", "1-2 positive nodes", "regional nodes"],
        "citations": ["Giuliano", "Z-11", "2011", "JAMA"],
        "rationale": "ACOSOG Z-11 showed axillary dissection increased morbidity without improving oncologic endpoints.",
        "category": "breast"
    },
    {
        "id": "Q49",
        "number": "49-Q97",
        "question": "What is an appropriate RT regimen for a 6 year old with a completely resected medulloblastoma without anaplasia and with negative CSF?",
        "answer": "23.4 Gy CSI followed by a 30.6 Gy boost to resection bed",
        "key_concepts": ["23.4 Gy", "CSI", "30.6 Gy", "boost", "54 Gy total", "average risk", "medulloblastoma"],
        "citations": ["Packer", "2006", "JCO"],
        "rationale": "This patient has average risk medulloblastoma. Low dose CSI (23.4 Gy) with boost to 54 Gy is standard.",
        "category": "CNS"
    },
    {
        "id": "Q50",
        "number": "50-Q79",
        "question": "A 45-year-old patient undergoes complete resection of a 5 cm, completely encapsulated, type B2 thymoma. According to the SEER analysis, what does postoperative radiation therapy show?",
        "answer": "Not indicated",
        "key_concepts": ["not indicated", "no benefit", "stage I", "encapsulated", "completely resected", "thymoma"],
        "citations": ["Forquer", "SEER", "2010"],
        "rationale": "The SEER analysis showed clearly no benefit for postoperative RT in stage I completely resected thymoma.",
        "category": "thoracic"
    }
]


# ============================================
# SCORING FUNCTIONS
# ============================================

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def check_key_concepts(rag_answer: str, key_concepts: List[str]) -> Tuple[List[str], float]:
    """Check if key concepts are present in RAG answer."""
    rag_lower = rag_answer.lower()
    matched = [concept for concept in key_concepts if concept.lower() in rag_lower]
    coverage = len(matched) / len(key_concepts) if key_concepts else 0
    return matched, coverage


def extract_citations(text: str) -> List[str]:
    """Extract citations from text."""
    # Pattern: (Author et al., Year, Journal) or similar
    pattern = r'\([^)]*\d{4}[^)]*\)'
    citations = re.findall(pattern, text)
    return [c.strip('()') for c in citations]


def check_citations(rag_answer: str, expected_cits: List[str]) -> Tuple[List[str], float]:
    """Check if expected citations are present."""
    rag_lower = rag_answer.lower()
    matched = [cit for cit in expected_cits if cit.lower() in rag_lower]
    coverage = len(matched) / len(expected_cits) if expected_cits else 0
    return matched, coverage


def score_answer(rag_answer: str, question: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive scoring of RAG answer against ground truth."""
    # Answer similarity
    similarity = calculate_semantic_similarity(rag_answer, question['answer'])
    
    # Key concept coverage
    matched_concepts, concept_coverage = check_key_concepts(rag_answer, question['key_concepts'])
    
    # Citation coverage
    matched_citations, citation_coverage = check_citations(rag_answer, question['citations'])
    
    # Overall assessment
    if similarity > 0.8 and citation_coverage > 0.7:
        overall = "✅ EXCELLENT"
    elif similarity > 0.6 and concept_coverage > 0.6:
        overall = "✅ PASS"
    elif similarity > 0.5 or concept_coverage > 0.5:
        overall = "⚠️ PARTIAL"
    else:
        overall = "❌ FAIL"
    
    return {
        "similarity": similarity,
        "concept_coverage": concept_coverage,
        "citation_coverage": citation_coverage,
        "matched_concepts": matched_concepts,
        "matched_citations": matched_citations,
        "overall": overall,
        "rag_citations": extract_citations(rag_answer)
    }


# ============================================
# TEST EXECUTION
# ============================================

def run_comprehensive_test(limit: int = None, category: str = None):
    """Run comprehensive ACR test."""
    print("="*80)
    print("COMPREHENSIVE ACR BOARD EXAM TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Filter questions
    questions = ACR_QUESTIONS
    if category:
        questions = [q for q in questions if q['category'] == category]
        print(f"Category Filter: {category}")
    if limit:
        questions = questions[:limit]
        print(f"Limit: {limit} questions")
    
    print(f"Total Questions: {len(questions)}")
    print("="*80)
    
    # Initialize RAG service
    print("\nInitializing RAG service...")
    try:
        rag_service = get_enhanced_rag_service()
        print("✅ RAG service initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize RAG service: {e}")
        return
    
    # Run tests
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[Test {i}/{len(questions)}]")
        print("="*80)
        print(f"ID: {question['id']} ({question['number']})")
        print(f"Category: {question['category']}")
        print(f"Question: {question['question'][:100]}...")
        
        try:
            # Query RAG
            response = rag_service.query(
                question=question['question'],
                category=None,  # Let it infer
                top_k=10
            )
            
            rag_answer = response.get('answer', '')
            
            # Score
            score = score_answer(rag_answer, question)
            
            # Display
            print(f"\n{score['overall']}")
            print(f"Answer Similarity: {score['similarity']:.2%}")
            print(f"Concept Coverage: {score['concept_coverage']:.2%} ({len(score['matched_concepts'])}/{len(question['key_concepts'])})")
            if score['matched_concepts']:
                print(f"  Matched: {', '.join(score['matched_concepts'][:3])}{'...' if len(score['matched_concepts']) > 3 else ''}")
            print(f"Citation Coverage: {score['citation_coverage']:.2%} ({len(score['matched_citations'])}/{len(question['citations'])})")
            if score['matched_citations']:
                print(f"  Found: {', '.join(score['matched_citations'])}")
            
            # Store result
            result = {
                **question,
                "rag_answer": rag_answer,
                "score": score
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                **question,
                "error": str(e),
                "score": {"overall": "❌ ERROR"}
            })
    
    # Generate reports
    generate_reports(results)


def generate_reports(results: List[Dict[str, Any]]):
    """Generate JSON and HTML reports."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Calculate statistics
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    excellent = sum(1 for r in results if r.get('score', {}).get('overall') == '✅ EXCELLENT')
    passed = sum(1 for r in results if '✅' in r.get('score', {}).get('overall', ''))
    partial = sum(1 for r in results if '⚠️' in r.get('score', {}).get('overall', ''))
    failed = sum(1 for r in results if '❌' in r.get('score', {}).get('overall', ''))
    
    # Averages
    valid_results = [r for r in results if 'error' not in r]
    avg_similarity = sum(r['score']['similarity'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_concept = sum(r['score']['concept_coverage'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_citation = sum(r['score']['citation_coverage'] for r in valid_results) / len(valid_results) if valid_results else 0
    
    # Category breakdown
    category_stats = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'passed': 0, 'excellent': 0}
        category_stats[cat]['total'] += 1
        if '✅' in r.get('score', {}).get('overall', ''):
            category_stats[cat]['passed'] += 1
        if r.get('score', {}).get('overall') == '✅ EXCELLENT':
            category_stats[cat]['excellent'] += 1
    
    # Print summary
    print("\n\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(f"\nTotal Questions: {total}")
    print(f"✅ Excellent: {excellent} ({100*excellent/total:.1f}%)")
    print(f"✅ Pass: {passed} ({100*passed/total:.1f}%)")
    print(f"⚠️  Partial: {partial} ({100*partial/total:.1f}%)")
    print(f"❌ Fail: {failed} ({100*failed/total:.1f}%)")
    if errors:
        print(f"🔥 Errors: {errors}")
    
    print(f"\nAverage Scores:")
    print(f"  Similarity: {avg_similarity:.2%}")
    print(f"  Concept Coverage: {avg_concept:.2%}")
    print(f"  Citation Coverage: {avg_citation:.2%}")
    
    print("\n" + "-"*80)
    print("CATEGORY BREAKDOWN:")
    print("-"*80)
    for cat, stats in sorted(category_stats.items()):
        pass_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cat:15s}: {stats['passed']}/{stats['total']} ({100*pass_rate:.1f}%) - {stats['excellent']} excellent")
    
    # Save JSON
    json_file = f"acr_complete_report_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'excellent': excellent,
                'passed': passed,
                'partial': partial,
                'failed': failed,
                'errors': errors,
                'pass_rate': passed / total if total > 0 else 0,
                'avg_similarity': avg_similarity,
                'avg_concept_coverage': avg_concept,
                'avg_citation_coverage': avg_citation
            },
            'category_stats': category_stats,
            'results': results
        }, f, indent=2)
    print(f"\n📄 Detailed JSON report: {json_file}")
    
    # Generate HTML
    html_file = generate_html_report(results, timestamp)
    print(f"📊 HTML comparison report: {html_file}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


def generate_html_report(results: List[Dict[str, Any]], timestamp: str) -> str:
    """Generate beautiful HTML report."""
    # Calculate stats for summary
    total = len(results)
    passed = sum(1 for r in results if '✅' in r.get('score', {}).get('overall', ''))
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ACR Test Results - {timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .timestamp {{
            color: #718096;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .question {{
            margin: 30px 0;
            padding: 25px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #f7fafc;
        }}
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .question-id {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2d3748;
        }}
        .question-text {{
            font-size: 1.05em;
            color: #2d3748;
            margin: 15px 0;
            line-height: 1.6;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .ground-truth {{
            padding: 20px;
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            border-radius: 8px;
            border-left: 5px solid #48bb78;
        }}
        .rag-response {{
            padding: 20px;
            background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
            border-radius: 8px;
            border-left: 5px solid #4299e1;
        }}
        .box-title {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #2d3748;
        }}
        .answer-text {{
            margin: 10px 0;
            line-height: 1.6;
        }}
        .answer-container {{
            position: relative;
        }}
        .score-card {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 15px;
            margin-top: 20px;
            border-radius: 8px;
            border-left: 5px solid #f39c12;
        }}
        .score-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}
        .score-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
        }}
        .score-label {{
            font-size: 0.9em;
            color: #718096;
        }}
        .score-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2d3748;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .badge-excellent {{
            background: #48bb78;
            color: white;
        }}
        .badge-pass {{
            background: #4299e1;
            color: white;
        }}
        .badge-partial {{
            background: #ed8936;
            color: white;
        }}
        .badge-fail {{
            background: #f56565;
            color: white;
        }}
        .concepts {{
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .concept-tag {{
            display: inline-block;
            background: rgba(0,0,0,0.1);
            padding: 3px 10px;
            border-radius: 3px;
            margin: 2px;
            font-size: 0.85em;
        }}
        .citation {{
            font-size: 0.85em;
            color: #4a5568;
            margin-top: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.05);
            border-radius: 5px;
        }}
        .answer-full {{
            display: none;
        }}
        .answer-preview {{
            display: block;
        }}
        .expand-btn {{
            background: #4299e1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 10px;
            transition: background 0.3s;
        }}
        .expand-btn:hover {{
            background: #3182ce;
        }}
        .expanded .answer-full {{
            display: block;
        }}
        .expanded .answer-preview {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 ACR Board Exam Test Results</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Total Questions</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pass Rate</div>
                <div class="stat-value">{100*passed/total:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Questions Passed</div>
                <div class="stat-value">{passed}/{total}</div>
            </div>
        </div>
"""
    
    for result in results:
        if 'error' in result:
            html += f"""
        <div class="question">
            <div class="question-header">
                <div class="question-id">{result['id']} - {result['category']}</div>
                <span class="badge badge-fail">ERROR</span>
            </div>
            <div class="question-text">{result['question']}</div>
            <p style="color: #f56565;">Error: {result['error']}</p>
        </div>
"""
            continue
        
        score = result['score']
        badge_class = 'badge-fail'
        if 'EXCELLENT' in score['overall']:
            badge_class = 'badge-excellent'
        elif '✅' in score['overall']:
            badge_class = 'badge-pass'
        elif '⚠️' in score['overall']:
            badge_class = 'badge-partial'
        
        matched_concepts_html = ' '.join(f'<span class="concept-tag">✓ {c}</span>' for c in score['matched_concepts'][:5])
        
        html += f"""
        <div class="question">
            <div class="question-header">
                <div class="question-id">{result['id']} ({result['number']}) - {result['category']}</div>
                <span class="badge {badge_class}">{score['overall']}</span>
            </div>
            <div class="question-text">{result['question']}</div>
            
            <div class="comparison">
                <div class="ground-truth">
                    <div class="box-title">✓ Ground Truth Answer</div>
                    <div class="answer-text">{result['answer']}</div>
                    <div class="citation"><strong>Expected Citations:</strong> {', '.join(result['citations'])}</div>
                </div>
                <div class="rag-response">
                    <div class="box-title">🤖 RAG Generated Answer</div>
                    <div class="answer-container">
                        <div class="answer-preview">
                            <div class="answer-text">{result['rag_answer'][:400]}{'...' if len(result['rag_answer']) > 400 else ''}</div>
                        </div>
                        <div class="answer-full">
                            <div class="answer-text">{result['rag_answer']}</div>
                        </div>
                        {f'<button class="expand-btn" onclick="this.closest(\'.rag-response\').querySelector(\'.answer-container\').classList.toggle(\'expanded\'); this.textContent = this.closest(\'.answer-container\').classList.contains(\'expanded\') ? \'Show Less ▲\' : \'Show Full Answer ▼\';">Show Full Answer ▼</button>' if len(result['rag_answer']) > 400 else ''}
                    </div>
                    <div class="citation"><strong>Found Citations:</strong> {', '.join(score['rag_citations']) if score['rag_citations'] else 'None detected'}</div>
                </div>
            </div>
            
            <div class="score-card">
                <div class="box-title">📊 Scoring</div>
                <div class="score-grid">
                    <div class="score-item">
                        <div class="score-label">Answer Similarity</div>
                        <div class="score-value">{score['similarity']:.1%}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Concept Coverage</div>
                        <div class="score-value">{score['concept_coverage']:.1%}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Citation Coverage</div>
                        <div class="score-value">{score['citation_coverage']:.1%}</div>
                    </div>
                </div>
                {f'<div class="concepts"><strong>Matched Concepts:</strong> {matched_concepts_html}</div>' if matched_concepts_html else ''}
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    html_file = f"acr_complete_report_{timestamp}.html"
    with open(html_file, 'w') as f:
        f.write(html)
    
    return html_file


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ACR test suite')
    parser.add_argument('--limit', type=int, help='Limit number of questions')
    parser.add_argument('--category', type=str, help='Test specific category only')
    
    args = parser.parse_args()
    
    run_comprehensive_test(limit=args.limit, category=args.category)
