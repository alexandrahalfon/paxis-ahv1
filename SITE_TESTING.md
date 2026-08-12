# Test All Features on the Site

Use this checklist to test every feature in the browser.

---

## 1. Start the app (one server)

In a terminal:

```bash
cd /Users/aysha/Desktop/exueed-updated
python run_api.py
```

Wait until you see: **Uvicorn running on http://0.0.0.0:8000**

Then open in Chrome (or any browser):

**http://localhost:8000**

The same server serves both the API and the frontend. You do **not** need a separate frontend server.

---

## 2. Feature checklist

### Home (http://localhost:8000)

- [ ] Page loads with logo and navigation.
- [ ] **AI Agent** tab: mode dropdown (Basic / Concise / Detailed), chat box, Send button.
- [ ] **Features** tab: cards for AI Query, Patient Matching, Treatment Comparison (links work).
- [ ] Nav links: Home, AI Query, Patient Matching, Treatment Comparison, About, Upload, API Docs.

---

### AI Query (http://localhost:8000/query.html)

- [ ] Query mode: **Basic** and **Conversation (with memory)**.
- [ ] Type a question, e.g. *What are the side effects of pembrolizumab?* → Click **Send**.
- [ ] Answer appears with short answer first; **More Information** expands to show details and sources.
- [ ] **Generate report (PDF)** strip appears above the chat after the first answer; click it → PDF downloads (`query-report.pdf`).

---

### Patient Matching (http://localhost:8000/patient-matching.html)

**Input mode: Free text**

- [ ] **Free text** is selected by default.
- [ ] Enter e.g. *65-year-old male with stage III NSCLC, EGFR mutation, never smoker, ECOG 1* → **Find Matching Studies**.
- [ ] Results: “We searched for” summary, optional “Show parsed profile”, list of matching studies with match %, strength, rationale, treatment, key finding.
- [ ] **Generate report (PDF)** appears in the results header; click → `patient-match-report.pdf` downloads.

**Input mode: Structured form**

- [ ] Click **Structured form**.
- [ ] Fill at least **Cancer Type** (e.g. Lung), optionally Age, Gender, Stage, etc. → **Find Matching Studies**.
- [ ] Results list and “We searched for” summary appear.
- [ ] **Generate report (PDF)** works for this result too.

---

### Treatment Comparison (http://localhost:8000/treatment-comparison.html)

- [ ] Enter **Treatment A** (e.g. *Pembrolizumab*) and **Treatment B** (e.g. *Chemotherapy*); optionally Cancer Type and Stage.
- [ ] Click **Compare Treatments**.
- [ ] Comparison results and summary appear (side-by-side or summary view).
- [ ] **Generate report (PDF)** in results header → `treatment-comparison-report.pdf` downloads.

---

### Upload (http://localhost:8000/upload.html)

- [ ] Page loads; upload form or instructions visible.
- [ ] If you have a test PDF, upload and confirm it submits (exact flow depends on your upload implementation).

---

### About (http://localhost:8000/about.html)

- [ ] About page loads and describes the platform.

---

### API Docs (http://localhost:8000/docs)

- [ ] Swagger UI loads; you can try **GET /health** and see **POST /api/rag/...** and **POST /api/report/...** endpoints.

---

## 3. Report generation (summary)

| Feature            | When it appears              | Button / action              | Downloaded file                    |
|--------------------|-----------------------------|------------------------------|------------------------------------|
| AI Query           | After first answer          | “Generate report (PDF)”      | `query-report.pdf`                 |
| Patient Matching   | After results (any mode)    | “Generate report (PDF)”      | `patient-match-report.pdf`         |
| Treatment Comparison | After comparison results  | “Generate report (PDF)”      | `treatment-comparison-report.pdf`  |

Each PDF should open in a viewer and show comparison tables and analysis as implemented.

---

## 4. If something fails

- **“Failed to … / API server”**: Ensure `python run_api.py` is running and nothing else is using port 8000.
- **No matches / no comparison**: Backend needs Qdrant and OpenAI configured (`.env` with `QDRANT_URL`, `OPENAI_API_KEY`, etc.).
- **Report download fails**: Check browser console (F12 → Console) and that `reportlab` is installed (`pip install reportlab`).

You can use this guide to test all features on the site end-to-end.
