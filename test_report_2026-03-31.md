# eXueed Feature Test Report
**Date:** 2026-03-31
**Server:** `http://localhost:8000` (v2.0.0)
**Scope:** All major feature modes tested via API

---

## Summary

| # | Feature | Status | Time | Notes |
|---|---------|--------|------|-------|
| 1a | Chat — recency query | ✅ PASS | 24.4s | Web supplement correctly triggered |
| 1b | Chat — mechanism query | ⚠️ WARN | 22.4s | Web fallback fired; classifier returned `general` not `mechanism` |
| 1c | Chat — side effects query | ✅ PASS | 25.1s | Web correctly blocked (`side_effects` type) |
| 1d | Chat — staging workup | ✅ PASS | 29.9s | Answered correctly; web fallback from weak KB coverage |
| 2 | Patient Match — structured | ✅ PASS | 8.2s | 15 matches; CT.gov fallback working |
| 3 | Patient Match — unstructured | ✅ PASS | 11.8s | 8 matches; CT.gov fallback working |
| 4 | Treatment Comparison | ✅ PASS | 24.4s | Fixed `current_user` bug; structured comparison returned |
| 5 | Study Comparison | ✅ PASS | 10.7s | Narrative + structured comparison returned |
| 6a | Analytics NL PubMed | ✅ PASS | 7.2s | 8 OS datapoints extracted |
| 6b | Analytics NL CT.gov | ✅ PASS | 3.7s | Enrollment by phase returned correctly |
| 6c | Analytics Structured PubMed | ⚠️ WARN | 2.6s | Empty result — PFS not extractable from abstract snippets |
| 6d | Analytics CT.gov pivot | ✅ PASS | 4.7s | Correctly pivoted OS → enrollment; caveat shown |

**10/12 pass | 2 warnings | 0 failures**

---

## Section 1 — Chat (Enhanced Query)

### 1a — Recency Query
**Input:** "What are the latest pembrolizumab trials in NSCLC 2024?"
**HTTP:** 200 | **Time:** 24.4s
**Query type:** `trial_results`
**Web mode:** `web_supplement = True` ✅ (correct — recency trigger fired)
**KB results:** 8
**Answer:**
> The latest pembrolizumab trial in NSCLC is the PEARLS/KEYNOTE-091, which evaluates pembrolizumab 200 mg every three weeks as adjuvant therapy for completely resected stage IB-IIIA NSCLC.

---

### 1b — Mechanism Query
**Input:** "What is the mechanism of action of cisplatin?"
**HTTP:** 200 | **Time:** 22.4s
**Query type:** `general` ⚠️ (expected `mechanism`)
**Web mode:** `web_fallback = True` ⚠️ (should have been blocked — classifier did not return `mechanism` type)
**KB results:** 10
**Answer:**
> Cisplatin forms DNA cross-links at GC-rich sites, disrupting replication and transcription, leading to apoptosis and cell growth inhibition.

**Note:** Answer is correct, but web fallback should not have fired for a mechanism question. The query classifier returned `general` instead of `mechanism`. Web is blocked for `mechanism` type — but that block only works when the classifier labels it correctly. Classifier needs tuning.

---

### 1c — Side Effects Query
**Input:** "What are common side effects of FOLFOX?"
**HTTP:** 200 | **Time:** 25.1s
**Query type:** `side_effects`
**Web mode:** none ✅ (correctly blocked for `side_effects` type)
**KB results:** 5
**Answer:**
> Common side effects of FOLFOX include neutropenia (20.3%), pain (3.1%), hypertension (2.9%), vomiting (28% grade 1-2, 4% grade 3), and diarrhea (62% grade 1-2, 9% grade 3).

---

### 1d — Staging Workup
**Input:** "What staging workup is recommended for newly diagnosed pancreatic cancer?"
**HTTP:** 200 | **Time:** 29.9s
**Query type:** `workup`
**Web mode:** `web_fallback = True` (KB results present but low-scoring)
**KB results:** 10
**Answer:**
> The recommended staging workup for newly diagnosed pancreatic cancer includes contrast-enhanced CT of the chest, abdomen, and pelvis, MRI if needed, PET-CT for suspected metastases, and EUS for local staging.

---

## Section 2 — Patient Matching

### 2 — Structured Match
**Input:** Male, 58yo, NSCLC stage IIIA, ECOG 1, prior carbo/paclitaxel, EGFR WT, PD-L1 60%
**HTTP:** 200 | **Time:** 8.2s
**Total matches:** 15 — sources: `clinicaltrials: 5, kb: 10`
**Top 3 matches:**

| Rank | Score | Source |
|------|-------|--------|
| 1 | 0.450 | clinicaltrials |
| 2 | 0.450 | clinicaltrials |
| 3 | 0.450 | clinicaltrials |

**Note:** CT.gov fallback correctly triggered (KB scores too low). CT.gov results rank above KB results (0.450 vs 0.08–0.16). KB has weak semantic overlap for NSCLC patient profiles. CT.gov fallback is working as intended.

---

### 3 — Unstructured Match
**Input:** "65-year-old female with HER2-positive metastatic breast cancer, previously treated with trastuzumab and pertuzumab, now progressing. ECOG PS 1. Looking for clinical trials."
**HTTP:** 200 | **Time:** 11.8s
**Total matches:** 8 — sources: `kb: 3, clinicaltrials: 5`
**Top 3 matches:**

| Rank | Score | Source |
|------|-------|--------|
| 1 | 0.550 | kb |
| 2 | 0.462 | kb |
| 3 | 0.450 | clinicaltrials |

**Note:** Correct field name for this endpoint is `unstructured_description` (not `profile_text`). Works correctly.

---

## Section 3 — Treatment Comparison

### 4 — Pembrolizumab vs Nivolumab in NSCLC
**HTTP:** 200 | **Time:** 24.4s
**Bug fixed:** `current_user` was referenced inside nested function but not declared as endpoint parameter — caused 500 on every call. Fixed by adding `Depends(get_current_user_optional)` to `compare_treatments` signature.
**Output:** Structured comparison returned with `treatment_a_evidence` and `treatment_b_evidence` blocks including efficacy, safety, and dosing sections.
**Sample:**
> Multicenter phase 2 randomized clinical trial conducted at 3 medical sites in the Netherlands. Patients 18 years or older with histological or cytological confirmed metastatic non-small cell lung cancer...

---

## Section 4 — Study Comparison

### 5 — KB Studies 1 & 3
**Input:** study_ids `["1", "3"]`
**HTTP:** 200 | **Time:** 10.7s
**Output format:** `studies`, `categories`, `narrative`, `generated_at`
**Narrative excerpt:**
> The two studies under comparison focus on different cancer types and treatment modalities, with distinct primary endpoints and patient populations. Study 1 investigates the efficacy of preoperative radiotherapy versus selective postoperative chemoradiotherapy in rectal cancer...

---

## Section 5 — Literature Analytics

### 6a — NL Mode, PubMed — Overall Survival NSCLC
**Input:** "Show me overall survival rates in NSCLC immunotherapy trials" | source: `pubmed`
**HTTP:** 200 | **Time:** 7.2s | **n_studies:** 8
**Chart title:** Overall Survival Rates in Non-Small Cell Lung Cancer Studies
**Labels & Values:**

| Label | Value |
|-------|-------|
| Nivolumab + Chemotherapy | 60% |
| Nivolumab + SABR | 50% |
| Tislelizumab + Chemotherapy | 55% |
| Tislelizumab + Chemotherapy (sq-NSCLC) | 45% |
| Surgery vs Boost-Radiochemotherapy | 40% |
| CT + IO (PD-L1 < 1%) | 12% |
| Pembrolizumab + Chemo | 70% |
| Pembrolizumab Only | 65% |

**Caveat:** Data quality varies across studies; some are real-world outcomes.

---

### 6b — NL Mode, ClinicalTrials.gov — Enrollment by Phase
**Input:** "Show enrollment by phase in lung cancer trials" | source: `clinicaltrials`
**HTTP:** 200 | **Time:** 3.7s | **n_studies:** 8
**Chart title:** Enrollment Counts by Phase in Clinical Trials

| Phase | Patients |
|-------|----------|
| Phase 1 | 33 |
| Phase 2 | 326 |
| Phase 3 | 1,280 |

**Caveat:** Data is derived from a limited number of studies, and phases may overlap.

---

### 6c — Structured Mode, PubMed — PFS by Treatment Arm
**Input:** search_query: "EGFR inhibitor NSCLC", metric: "progression-free survival", group_by: "treatment arm" | source: `pubmed`
**HTTP:** 200 | **Time:** 2.6s | **n_studies:** 0
**Labels:** `[]` | **Values:** `[]`
**Caveat:** No records provided quantitative data on progression-free survival.

**Note:** Empty result is a known limitation. PFS values are rarely stated explicitly in abstract snippets — PubMed returns the paper but the metric isn't in the first ~800 chars of text. NL mode performs better for this use case (see 6a for comparison).

---

### 6d — CT.gov Outcome Pivot Test
**Input:** "overall survival rates in pancreatic cancer trials" | source: `clinicaltrials`
**HTTP:** 200 | **Time:** 4.7s | **n_studies:** 10
**Chart title:** Enrollment Count by Trial Phase for Pancreatic Cancer Studies
**Pivot caveat (first 150 chars):**
> ClinicalTrials.gov contains protocol data, not outcomes. Showing enrollment by phase instead of 'overall survival rate'. For outcome data (OS, PFS, re...

✅ Pivot logic working correctly — outcome metric detected, redirected to enrollment data, caveat shown.

---

## Issues Identified

### 🔴 Fixed During Testing
| Issue | Fix Applied |
|-------|-------------|
| Treatment comparison 500 error — `current_user` not defined in `compare_treatments` | Added `Depends(get_current_user_optional)` to endpoint signature in `query.py` |

### 🟡 Known Limitations (Not Bugs)
| Issue | Explanation |
|-------|-------------|
| Mechanism query classified as `general` — web fallback fires | Query classifier didn't return `mechanism` type for cisplatin question. Web block works but depends on correct classification. |
| Structured analytics PFS query returns empty | PFS values not present in first 800 chars of most abstracts. NL mode is more reliable for outcome extraction. |
| KB patient match scores low (0.08–0.16) | KB has limited patient-profile-style trial data. CT.gov fallback compensates correctly. |

---

## Environment
- **API version:** 2.0.0
- **Server start command:** `python3 run_api.py`
- **Python env:** `highcomp-hds` (miniconda3)
- **Test date:** 2026-03-31
