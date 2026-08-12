# Testing Guide for RAG Modes

This guide shows you how to test all the RAG modes: Basic Query, Agent Mode, Patient Matching, and Treatment Comparison.

## Prerequisites

1. **Server must be running:**
   ```bash
   cd /Users/aysha/Desktop/Paxis-backend
   python run_api.py
   ```

2. **Server should be accessible at:** `http://localhost:8000`

---

## Method 1: Using the Test Script (Easiest)

Run the comprehensive test script:

```bash
cd /Users/aysha/Desktop/Paxis-backend
python test_all_modes.py
```

This will test all modes automatically and show you the results.

---

## Method 2: Using Interactive API Docs (Recommended)

1. **Open your browser and go to:**
   ```
   http://localhost:8000/docs
   ```

2. **You'll see all available endpoints:**
   - `POST /api/rag/query` - Basic and Agent mode queries
   - `POST /api/rag/patient/match` - Patient matching
   - `POST /api/rag/comparison/treatments` - Treatment comparison
   - `GET /api/rag/health` - Health check
   - `GET /api/rag/stats` - Collection statistics

3. **Click on any endpoint → "Try it out" → Enter your data → "Execute"**

---

## Method 3: Using curl Commands

### Test 1: Basic Query Mode

```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the side effects of pembrolizumab?",
    "mode": "basic",
    "top_k": 5
  }' | python -m json.tool
```

### Test 2: Agent Mode (Multi-step Reasoning)

```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare the efficacy and safety of pembrolizumab versus chemotherapy in lung cancer patients",
    "mode": "agent",
    "top_k": 5
  }' | python -m json.tool
```

### Test 3: Different Agent Modes

Try different agent modes:

**Naive Mode:**
```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are pembrolizumab side effects?",
    "mode": "naive",
    "top_k": 5
  }' | python -m json.tool
```

**Local Mode:**
```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are pembrolizumab side effects?",
    "mode": "local",
    "top_k": 5
  }' | python -m json.tool
```

**Global Mode:**
```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are pembrolizumab side effects?",
    "mode": "global",
    "top_k": 5
  }' | python -m json.tool
```

**Hybrid Mode:**
```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are pembrolizumab side effects?",
    "mode": "hybrid",
    "top_k": 5
  }' | python -m json.tool
```

### Test 4: Patient Matching Mode

```bash
curl -X POST "http://localhost:8000/api/rag/patient/match" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "gender": "male",
    "cancer_stage": "III",
    "histology": "adenocarcinoma",
    "molecular_markers": ["PD-L1+"],
    "performance_status": "ECOG 1",
    "smoking_status": "former"
  }' | python -m json.tool
```

**Simpler patient profile:**
```bash
curl -X POST "http://localhost:8000/api/rag/patient/match" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "female",
    "cancer_stage": "II",
    "histology": "squamous"
  }' | python -m json.tool
```

### Test 5: Treatment Comparison Mode

```bash
curl -X POST "http://localhost:8000/api/rag/comparison/treatments" \
  -H "Content-Type: application/json" \
  -d '{
    "treatment_a": "pembrolizumab",
    "treatment_b": "chemotherapy",
    "cancer_type": "lung cancer",
    "stage": "III"
  }' | python -m json.tool
```

**Simpler comparison:**
```bash
curl -X POST "http://localhost:8000/api/rag/comparison/treatments" \
  -H "Content-Type: application/json" \
  -d '{
    "treatment_a": "pembrolizumab",
    "treatment_b": "placebo"
  }' | python -m json.tool
```

### Test 6: Health Check

```bash
curl http://localhost:8000/api/rag/health | python -m json.tool
```

### Test 7: Collection Statistics

```bash
curl http://localhost:8000/api/rag/stats | python -m json.tool
```

---

## Method 4: Using Python Scripts

### Using the existing test script:

```bash
python test_rag_query.py "What are the side effects of pembrolizumab?"
```

### Create a custom test:

```python
import requests

# Test basic query
response = requests.post(
    "http://localhost:8000/api/rag/query",
    json={
        "question": "What are pembrolizumab side effects?",
        "mode": "agent",
        "top_k": 5
    }
)
print(response.json()["answer"])
print(f"Sources: {len(response.json()['sources'])}")
```

---

## What to Check in Responses

### For Query Responses:
✅ **Answer** - Should be comprehensive and well-formatted
✅ **Sources** - Should have proper citations with:
   - `author_et_al` (e.g., "O'Brien et al.")
   - `year` (e.g., 2022)
   - `title` (full title)
   - `journal` (journal name)
   - `doi` (DOI identifier)
   - `citation` (full citation string)
✅ **Chunks** - Should have similarity scores and metadata
✅ **Retrieved count** - Should be > 0

### For Patient Matching:
✅ **Matches** - Should return relevant studies
✅ **Match scores** - Should be between 0.0 and 1.0
✅ **Confidence scores** - Should be between 0.0 and 1.0
✅ **Citations** - Should be properly formatted

### For Treatment Comparison:
✅ **Comparisons** - Should show side-by-side data
✅ **Statistical significance** - Boolean flag
✅ **Treatment results** - Should have efficacy and safety data
✅ **Summary** - Should show total studies and statistics
✅ **Citations** - Should be properly formatted

---

## Quick Test Checklist

- [ ] Server is running (`python run_api.py`)
- [ ] Health check returns "healthy"
- [ ] Basic query returns answer with citations
- [ ] Agent mode works with different strategies
- [ ] Patient matching finds relevant studies
- [ ] Treatment comparison shows side-by-side data
- [ ] All citations are properly formatted (author, year, title, journal, DOI)

---

## Troubleshooting

### Server not responding:
```bash
# Check if server is running
curl http://localhost:8000/health

# Restart server
python run_api.py
```

### No results returned:
- Check that your Qdrant collection has data (115,479 points expected)
- Verify `.env` file has correct `QDRANT_COLLECTION=exueed_kb_latest`
- Check server logs for errors

### Citation formatting issues:
- Verify that ingested documents have proper metadata
- Check that `doc_meta` field in chunks contains author, year, title, journal, doi

---

## Example Expected Response Format

### Query Response:
```json
{
  "answer": "Pembrolizumab is associated with...",
  "sources": [
    {
      "doc_id": "...",
      "title": "Pembrolizumab versus placebo...",
      "author_et_al": "O'Brien et al.",
      "year": 2022,
      "journal": "Lancet Oncol",
      "doi": "10.1016/S1470-2045(22)00518-6",
      "citation": "O'Brien et al. (2022) Pembrolizumab versus placebo..."
    }
  ],
  "chunks": [...],
  "retrieved_count": 5
}
```

---

## Need Help?

1. Check server logs in the terminal where `run_api.py` is running
2. Visit `http://localhost:8000/docs` for interactive API documentation
3. Run `python test_all_modes.py` for comprehensive testing
