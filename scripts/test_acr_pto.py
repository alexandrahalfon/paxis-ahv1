#!/usr/bin/env python3
"""
ACR Board Exam Test with PTO Frame Retrieval
=============================================

Tests RAG system with the new PTO (Patient→Treatment→Outcome) frame retrieval
against all 50 ACR radiation oncology board questions.

Compares:
- PTO-aware hybrid retrieval (frames + chunks)
- Standard chunk-only retrieval

Features:
- All 50 questions included
- PTO routing analysis
- Side-by-side comparison (PTO vs Standard)
- Keyword-based scoring
- Citation matching analysis
- JSON + HTML reports

Usage:
    python test_acr_pto.py              # Run all 50 questions
    python test_acr_pto.py --limit 10   # Run first 10
    python test_acr_pto.py --category breast  # Run specific category
    python test_acr_pto.py --pto-only   # Only test PTO retrieval

Output:
    - acr_pto_report_TIMESTAMP.json (detailed results)
    - acr_pto_report_TIMESTAMP.html (comparison report)
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Load environment variables
load_dotenv('.env')

# Verify required environment variables
required_vars = ['OPENAI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Add project root to path (so 'src' module can be found)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.api.services.enhanced_rag_service import get_enhanced_rag_service
    from src.api.services.pto_retriever import PTORetriever, PTOQueryRouter, format_pto_context
except ImportError as e:
    print(f"ERROR: Cannot import services: {e}")
    print("Make sure you have:")
    print("  - src/api/services/enhanced_rag_service.py")
    print("  - src/api/services/pto_retriever.py")
    sys.exit(1)

from openai import OpenAI


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
    
    similarity = calculate_semantic_similarity(rag_answer, question['answer'])
    matched_concepts, concept_coverage = check_key_concepts(rag_answer, question['key_concepts'])
    matched_citations, citation_coverage = check_citations(rag_answer, question['citations'])
    
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
# PTO-AWARE RAG SERVICE
# ============================================

class PTOAwareRAGService:
    """RAG service that uses PTO frames when appropriate."""
    
    def __init__(self):
        # Initialize standard RAG service
        self.standard_rag = get_enhanced_rag_service()
        
        # Initialize PTO components
        self.pto_router = PTOQueryRouter()
        self.pto_retriever = PTORetriever(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=os.getenv('QDRANT_COLLECTION', 'exueed_kb_latest'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # OpenAI client for generation
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("✅ PTO-Aware RAG Service initialized")
    
    def query_with_pto(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """Query using PTO-aware hybrid retrieval."""
        
        # Analyze query for PTO routing
        analysis = self.pto_router.analyze_query(question)
        
        # Get hybrid results (PTO frames + chunks)
        hybrid_results = self.pto_retriever.hybrid_search(
            query=question,
            pto_limit=3,
            chunk_limit=top_k
        )
        
        # Build context
        context = format_pto_context(hybrid_results)
        
        # Add evidence from standard retrieval for comprehensive coverage
        standard_evidence = []
        try:
            standard_result = self.standard_rag.query(
                question=question,
                top_k=5
            )
            standard_evidence = standard_result.get('evidence', [])
        except:
            pass
        
        # Generate answer
        answer = self._generate_answer(question, context, hybrid_results, standard_evidence)
        
        return {
            "answer": answer,
            "routing": {
                "used_pto": analysis.should_use_pto,
                "query_type": analysis.query_type.value,
                "confidence": analysis.confidence,
                "extracted_profile": analysis.extracted_profile
            },
            "pto_frames": hybrid_results.get("pto_frames", []),
            "chunks": hybrid_results.get("chunks", []),
            "evidence": standard_evidence
        }
    
    def query_standard(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """Query using standard chunk-only retrieval."""
        result = self.standard_rag.query(
            question=question,
            top_k=top_k
        )
        return {
            "answer": result.get("answer", ""),
            "routing": {"used_pto": False, "query_type": "standard"},
            "pto_frames": [],
            "chunks": [],
            "evidence": result.get("evidence", [])
        }
    
    def _generate_answer(
        self, 
        question: str, 
        pto_context: str,
        hybrid_results: Dict,
        standard_evidence: List[Dict]
    ) -> str:
        """Generate answer using PTO context and evidence."""
        
        # Build combined context
        context_parts = []
        
        if pto_context:
            context_parts.append(pto_context)
        
        # Add standard evidence
        for i, ev in enumerate(standard_evidence[:5], 1):
            text = ev.get('text', '')[:500]
            citation = ev.get('doc_meta', {}).get('citation_string', '')
            context_parts.append(f"[Evidence {i}] {citation}\n{text}")
        
        full_context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are a radiation oncology clinical decision support assistant.

When answering questions:
1. If PATIENT-TREATMENT-OUTCOME relationships are provided, use them to match patient profiles to treatments and cite specific outcomes
2. Always cite sources using author names and years when available
3. Be concise but comprehensive
4. For dose questions, provide specific numbers
5. For treatment questions, recommend the most appropriate evidence-based approach"""

        user_prompt = f"""Question: {question}

Context:
{full_context}

Provide a clear, evidence-based answer with citations."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"


# ============================================
# TEST EXECUTION
# ============================================

def run_pto_test(limit: int = None, category: str = None, pto_only: bool = False):
    """Run ACR test with PTO frame retrieval."""
    
    print("="*80)
    print("ACR BOARD EXAM TEST - PTO FRAME RETRIEVAL")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Filter questions
    questions = ACR_QUESTIONS
    if category:
        questions = [q for q in questions if q['category'].lower() == category.lower()]
        print(f"Category Filter: {category}")
    if limit:
        questions = questions[:limit]
        print(f"Limit: {limit} questions")
    
    print(f"Total Questions: {len(questions)}")
    print(f"Mode: {'PTO-only' if pto_only else 'PTO vs Standard comparison'}")
    print("="*80)
    
    # Initialize services
    print("\nInitializing RAG services...")
    try:
        rag_service = PTOAwareRAGService()
        print("✅ Services initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run tests
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Test {i}/{len(questions)}]")
        print("="*80)
        print(f"ID: {question['id']} ({question['number']})")
        print(f"Category: {question['category']}")
        print(f"Question: {question['question']}")
        print(f"Expected Answer: {question['answer']}")
        
        result = {
            **question,
            "pto_result": None,
            "standard_result": None,
            "pto_score": None,
            "standard_score": None
        }
        
        try:
            # Query with PTO
            print("\n🔷 Testing PTO retrieval...")
            pto_response = rag_service.query_with_pto(question['question'])
            pto_answer = pto_response.get('answer', '')
            pto_score = score_answer(pto_answer, question)
            
            result["pto_result"] = {
                "answer": pto_answer,
                "routing": pto_response.get('routing', {}),
                "pto_frames_count": len(pto_response.get('pto_frames', [])),
                "pto_frames": pto_response.get('pto_frames', [])[:2]  # Store top 2 for report
            }
            result["pto_score"] = pto_score
            
            print(f"   Routing: {pto_response['routing']['query_type']} (used_pto={pto_response['routing']['used_pto']})")
            print(f"   PTO Frames: {len(pto_response.get('pto_frames', []))}")
            print(f"   Score: {pto_score['overall']} (concepts: {pto_score['concept_coverage']:.0%})")
            print(f"\n   📝 PTO Answer:\n   {pto_answer}")
            
            # Show citations from PTO frames
            pto_frames = pto_response.get('pto_frames', [])
            if pto_frames:
                print(f"\n   � PTO Frame Sources:")
                for j, frame in enumerate(pto_frames[:3], 1):
                    doc_meta = frame.get('doc_meta', {})
                    citation = doc_meta.get('citation_string') or doc_meta.get('title') or 'Unknown'
                    cancer_type = frame.get('cancer_type', 'N/A')
                    outcomes = frame.get('outcomes', {})
                    print(f"      [{j}] {citation}")
                    if outcomes:
                        print(f"          Outcomes: {outcomes}")
            
            # Show citations from chunks
            chunks = pto_response.get('chunks', [])
            if chunks:
                print(f"\n   📚 Chunk Sources:")
                for j, chunk in enumerate(chunks[:3], 1):
                    doc_meta = chunk.get('doc_meta', {})
                    citation = doc_meta.get('citation_string') or doc_meta.get('title') or 'Unknown'
                    print(f"      [{j}] {citation}")
            
            # Show citations from evidence
            evidence = pto_response.get('evidence', [])
            if evidence:
                print(f"\n   📚 Evidence Sources:")
                for j, ev in enumerate(evidence[:3], 1):
                    doc_meta = ev.get('doc_meta', {})
                    citation = doc_meta.get('citation_string') or doc_meta.get('title') or ev.get('title') or 'Unknown'
                    print(f"      [{j}] {citation}")
            
            # Query standard (for comparison)
            if not pto_only:
                print("\n🔶 Testing standard retrieval...")
                std_response = rag_service.query_standard(question['question'])
                std_answer = std_response.get('answer', '')
                std_score = score_answer(std_answer, question)
                
                result["standard_result"] = {
                    "answer": std_answer,
                    "evidence_count": len(std_response.get('evidence', []))
                }
                result["standard_score"] = std_score
                
                print(f"   Score: {std_score['overall']} (concepts: {std_score['concept_coverage']:.0%})")
                print(f"\n   📝 Standard Answer:\n   {std_answer}")
                
                # Show citations from standard evidence
                std_evidence = std_response.get('evidence', [])
                if std_evidence:
                    print(f"\n   📚 Standard Evidence Sources:")
                    for j, ev in enumerate(std_evidence[:3], 1):
                        doc_meta = ev.get('doc_meta', {})
                        citation = doc_meta.get('citation_string') or doc_meta.get('title') or ev.get('title') or 'Unknown'
                        print(f"      [{j}] {citation}")
                
                # Compare
                if pto_score['concept_coverage'] > std_score['concept_coverage']:
                    print("\n   ✅ PTO BETTER")
                elif pto_score['concept_coverage'] < std_score['concept_coverage']:
                    print("\n   ⚠️ Standard better")
                else:
                    print("\n   ➡️ Same")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            results.append(result)
    
    # Generate reports
    generate_pto_reports(results, pto_only)


def generate_pto_reports(results: List[Dict[str, Any]], pto_only: bool):
    """Generate JSON and HTML reports for PTO test."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Calculate statistics
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    
    # PTO stats - handle None values safely
    pto_passed = sum(1 for r in results if (r.get('pto_score') or {}).get('overall', '').startswith('✅'))
    pto_excellent = sum(1 for r in results if (r.get('pto_score') or {}).get('overall') == '✅ EXCELLENT')
    
    # Standard stats (if available)
    std_passed = sum(1 for r in results if (r.get('standard_score') or {}).get('overall', '').startswith('✅'))
    
    # Routing stats
    used_pto = sum(1 for r in results if (r.get('pto_result') or {}).get('routing', {}).get('used_pto', False))
    
    # Comparison: where PTO was better
    pto_better = 0
    std_better = 0
    same = 0
    for r in results:
        pto_score = r.get('pto_score') or {}
        std_score = r.get('standard_score') or {}
        if pto_score and std_score:
            pto_cov = pto_score.get('concept_coverage', 0)
            std_cov = std_score.get('concept_coverage', 0)
            if pto_cov > std_cov:
                pto_better += 1
            elif std_cov > pto_cov:
                std_better += 1
            else:
                same += 1
    
    # Averages
    valid_pto = [r for r in results if r.get('pto_score')]
    avg_pto_concept = sum(r['pto_score']['concept_coverage'] for r in valid_pto) / len(valid_pto) if valid_pto else 0
    avg_pto_citation = sum(r['pto_score']['citation_coverage'] for r in valid_pto) / len(valid_pto) if valid_pto else 0
    
    valid_std = [r for r in results if r.get('standard_score')]
    avg_std_concept = sum(r['standard_score']['concept_coverage'] for r in valid_std) / len(valid_std) if valid_std else 0
    
    # Print summary
    print("\n\n" + "="*80)
    print("FINAL REPORT - PTO FRAME RETRIEVAL TEST")
    print("="*80)
    
    print(f"\n📊 OVERALL RESULTS")
    print(f"{'─'*40}")
    print(f"Total Questions: {total}")
    print(f"Errors: {errors}")
    
    print(f"\n🔷 PTO RETRIEVAL")
    print(f"{'─'*40}")
    print(f"Pass Rate: {pto_passed}/{total} ({100*pto_passed/total:.1f}%)")
    print(f"Excellent: {pto_excellent}")
    print(f"Avg Concept Coverage: {avg_pto_concept:.1%}")
    print(f"Avg Citation Coverage: {avg_pto_citation:.1%}")
    print(f"Questions Routed to PTO: {used_pto}/{total}")
    
    if not pto_only:
        print(f"\n🔶 STANDARD RETRIEVAL (Comparison)")
        print(f"{'─'*40}")
        print(f"Pass Rate: {std_passed}/{total} ({100*std_passed/total:.1f}%)")
        print(f"Avg Concept Coverage: {avg_std_concept:.1%}")
        
        print(f"\n⚖️ HEAD-TO-HEAD COMPARISON")
        print(f"{'─'*40}")
        print(f"PTO Better: {pto_better} questions")
        print(f"Standard Better: {std_better} questions")
        print(f"Same: {same} questions")
        
        improvement = ((avg_pto_concept - avg_std_concept) / avg_std_concept * 100) if avg_std_concept > 0 else 0
        print(f"\nConcept Coverage Improvement: {improvement:+.1f}%")
    
    # Category breakdown
    print(f"\n📁 CATEGORY BREAKDOWN (PTO)")
    print(f"{'─'*40}")
    category_stats = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'passed': 0, 'used_pto': 0}
        category_stats[cat]['total'] += 1
        if r.get('pto_score', {}).get('overall', '').startswith('✅'):
            category_stats[cat]['passed'] += 1
        if r.get('pto_result', {}).get('routing', {}).get('used_pto'):
            category_stats[cat]['used_pto'] += 1
    
    for cat, stats in sorted(category_stats.items()):
        pass_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cat:15s}: {stats['passed']}/{stats['total']} ({100*pass_rate:.0f}%) - PTO used: {stats['used_pto']}")
    
    # Save JSON
    json_file = f"acr_pto_report_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': 'pto_only' if pto_only else 'comparison',
            'summary': {
                'total': total,
                'errors': errors,
                'pto_passed': pto_passed,
                'pto_excellent': pto_excellent,
                'std_passed': std_passed if not pto_only else None,
                'pto_better': pto_better if not pto_only else None,
                'std_better': std_better if not pto_only else None,
                'used_pto_routing': used_pto,
                'avg_pto_concept': avg_pto_concept,
                'avg_std_concept': avg_std_concept if not pto_only else None
            },
            'category_stats': category_stats,
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\n📄 JSON report: {json_file}")
    
    # Generate HTML
    html_file = generate_pto_html_report(results, timestamp, pto_only)
    print(f"📊 HTML report: {html_file}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


def generate_pto_html_report(results: List[Dict[str, Any]], timestamp: str, pto_only: bool) -> str:
    """Generate HTML report for PTO test."""
    
    total = len(results)
    pto_passed = sum(1 for r in results if r.get('pto_score', {}).get('overall', '').startswith('✅'))
    std_passed = sum(1 for r in results if r.get('standard_score', {}).get('overall', '').startswith('✅'))
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ACR PTO Test Results - {timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #1a1a2e;
            margin-bottom: 5px;
            font-size: 2.2em;
        }}
        .subtitle {{
            color: #718096;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}
        .stat-card {{
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
        }}
        .stat-card.pto {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .stat-card.standard {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .stat-card.comparison {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 8px 0;
        }}
        .stat-label {{
            font-size: 0.85em;
            opacity: 0.9;
        }}
        .question {{
            margin: 25px 0;
            padding: 25px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: #fafafa;
        }}
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .question-id {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2d3748;
        }}
        .badges {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8em;
        }}
        .badge-pto {{
            background: #667eea;
            color: white;
        }}
        .badge-std {{
            background: #f5576c;
            color: white;
        }}
        .badge-pass {{
            background: #48bb78;
            color: white;
        }}
        .badge-fail {{
            background: #fc8181;
            color: white;
        }}
        .badge-better {{
            background: #38b2ac;
            color: white;
        }}
        .question-text {{
            font-size: 1.05em;
            color: #2d3748;
            margin: 15px 0;
            line-height: 1.6;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        @media (max-width: 900px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .answer-box {{
            padding: 20px;
            border-radius: 10px;
        }}
        .answer-box.ground-truth {{
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            border-left: 5px solid #38a169;
        }}
        .answer-box.pto {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-left: 5px solid #667eea;
        }}
        .answer-box.standard {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-left: 5px solid #f5576c;
        }}
        .box-title {{
            font-weight: bold;
            font-size: 1em;
            margin-bottom: 10px;
            color: #2d3748;
        }}
        .answer-text {{
            line-height: 1.6;
            color: #2d3748;
        }}
        .pto-frames {{
            margin-top: 15px;
            padding: 12px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
            font-size: 0.9em;
        }}
        .pto-frame-item {{
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        .routing-info {{
            margin-top: 10px;
            padding: 10px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 5px;
            font-size: 0.85em;
        }}
        .score-row {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .score-item {{
            background: white;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .score-label {{
            color: #718096;
        }}
        .score-value {{
            font-weight: bold;
            color: #2d3748;
        }}
        .winner-tag {{
            display: inline-block;
            padding: 3px 10px;
            background: #38b2ac;
            color: white;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 ACR Board Exam Test - PTO Frame Retrieval</h1>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {'PTO Only' if pto_only else 'PTO vs Standard Comparison'}</div>
        
        <div class="summary-grid">
            <div class="stat-card pto">
                <div class="stat-label">PTO Pass Rate</div>
                <div class="stat-value">{100*pto_passed/total:.0f}%</div>
                <div class="stat-label">{pto_passed}/{total} questions</div>
            </div>
"""
    
    if not pto_only:
        pto_better = sum(1 for r in results if r.get('pto_score') and r.get('standard_score') and 
                        r['pto_score']['concept_coverage'] > r['standard_score']['concept_coverage'])
        html += f"""
            <div class="stat-card standard">
                <div class="stat-label">Standard Pass Rate</div>
                <div class="stat-value">{100*std_passed/total:.0f}%</div>
                <div class="stat-label">{std_passed}/{total} questions</div>
            </div>
            <div class="stat-card comparison">
                <div class="stat-label">PTO Better</div>
                <div class="stat-value">{pto_better}</div>
                <div class="stat-label">questions improved</div>
            </div>
"""
    
    used_pto = sum(1 for r in results if r.get('pto_result', {}).get('routing', {}).get('used_pto', False))
    html += f"""
            <div class="stat-card pto">
                <div class="stat-label">PTO Routing Used</div>
                <div class="stat-value">{used_pto}</div>
                <div class="stat-label">of {total} questions</div>
            </div>
        </div>
"""
    
    # Individual questions
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
        
        pto_score = result.get('pto_score', {})
        std_score = result.get('standard_score', {})
        pto_result = result.get('pto_result', {})
        routing = pto_result.get('routing', {})
        
        # Determine winner
        winner = None
        if pto_score and std_score:
            if pto_score['concept_coverage'] > std_score['concept_coverage']:
                winner = 'pto'
            elif std_score['concept_coverage'] > pto_score['concept_coverage']:
                winner = 'std'
        
        pto_badge_class = 'badge-pass' if pto_score.get('overall', '').startswith('✅') else 'badge-fail'
        std_badge_class = 'badge-pass' if std_score.get('overall', '').startswith('✅') else 'badge-fail'
        
        html += f"""
        <div class="question">
            <div class="question-header">
                <div class="question-id">{result['id']} ({result['number']}) - {result['category']}</div>
                <div class="badges">
                    <span class="badge badge-pto {pto_badge_class}">PTO: {pto_score.get('overall', 'N/A')}</span>
"""
        if not pto_only and std_score:
            html += f"""                    <span class="badge badge-std {std_badge_class}">STD: {std_score.get('overall', 'N/A')}</span>
"""
        if winner == 'pto':
            html += """                    <span class="badge badge-better">🏆 PTO Better</span>
"""
        elif winner == 'std':
            html += """                    <span class="badge" style="background:#f5576c;color:white;">Standard Better</span>
"""
        
        html += f"""
                </div>
            </div>
            
            <div class="question-text">{result['question']}</div>
            
            <div class="comparison-grid">
                <div class="answer-box ground-truth">
                    <div class="box-title">✓ Ground Truth</div>
                    <div class="answer-text"><strong>{result['answer']}</strong></div>
                    <div style="margin-top:10px;font-size:0.85em;color:#2d3748;">
                        <strong>Key concepts:</strong> {', '.join(result['key_concepts'][:5])}
                    </div>
                </div>
                
                <div class="answer-box pto">
                    <div class="box-title">🔷 PTO Answer {f'<span class="winner-tag">WINNER</span>' if winner == 'pto' else ''}</div>
                    <div class="answer-text">{pto_result.get('answer', 'N/A')[:600]}{'...' if len(pto_result.get('answer', '')) > 600 else ''}</div>
                    
                    <div class="routing-info">
                        <strong>Routing:</strong> {routing.get('query_type', 'N/A')} | 
                        Used PTO: {'Yes ✓' if routing.get('used_pto') else 'No'} |
                        Frames: {pto_result.get('pto_frames_count', 0)}
                    </div>
"""
        
        # Show PTO frames if any
        pto_frames = pto_result.get('pto_frames', [])
        if pto_frames:
            html += """
                    <div class="pto-frames">
                        <strong>Retrieved PTO Frames:</strong>
"""
            for pf in pto_frames[:2]:
                html += f"""
                        <div class="pto-frame-item">
                            <strong>{pf.get('cancer_type', 'N/A')}</strong> | 
                            Stage: {pf.get('stage') or pf.get('tnm') or 'N/A'} |
                            Treatment: {', '.join(pf.get('treatment_modalities', [])[:2]) or 'N/A'} |
                            Dose: {pf.get('dose_fractionation') or 'N/A'}
                        </div>
"""
            html += """
                    </div>
"""
        
        html += f"""
                    <div class="score-row">
                        <div class="score-item"><span class="score-label">Concepts:</span> <span class="score-value">{pto_score.get('concept_coverage', 0):.0%}</span></div>
                        <div class="score-item"><span class="score-label">Citations:</span> <span class="score-value">{pto_score.get('citation_coverage', 0):.0%}</span></div>
                    </div>
                </div>
"""
        
        if not pto_only and result.get('standard_result'):
            std_result = result['standard_result']
            html += f"""
                <div class="answer-box standard">
                    <div class="box-title">🔶 Standard Answer {f'<span class="winner-tag">WINNER</span>' if winner == 'std' else ''}</div>
                    <div class="answer-text">{std_result.get('answer', 'N/A')[:600]}{'...' if len(std_result.get('answer', '')) > 600 else ''}</div>
                    <div class="score-row">
                        <div class="score-item"><span class="score-label">Concepts:</span> <span class="score-value">{std_score.get('concept_coverage', 0):.0%}</span></div>
                        <div class="score-item"><span class="score-label">Citations:</span> <span class="score-value">{std_score.get('citation_coverage', 0):.0%}</span></div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    html_file = f"acr_pto_report_{timestamp}.html"
    with open(html_file, 'w') as f:
        f.write(html)
    
    return html_file


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ACR test with PTO frame retrieval')
    parser.add_argument('--limit', type=int, help='Limit number of questions')
    parser.add_argument('--category', type=str, help='Test specific category only')
    parser.add_argument('--pto-only', action='store_true', help='Only test PTO retrieval (skip comparison)')
    args = parser.parse_args()
    
    run_pto_test(limit=args.limit, category=args.category, pto_only=args.pto_only)
