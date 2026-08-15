# Paxis Phase 3 - Quick Start Guide

## ✅ Integration Complete!

Phase 3 (Enhanced RAG Retrieval & Generation) has been integrated into your Paxis repository.

---

## 🚀 Next Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy and edit the environment file:

```bash
cp .env.example .env
nano .env  # Add your API keys
```

Required variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `QDRANT_API_KEY` - Your Qdrant API key
- `QDRANT_URL` - Your Qdrant instance URL
- `QDRANT_COLLECTION` - Collection name (default: exueed_kb_latest)

### 3. Run the API

**Option A: Using the launcher**
```bash
python run_api.py
```

**Option B: Using uvicorn directly**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Option C: Using Python module**
```bash
python -m src.api.main
```

### 4. Test the API

Open your browser and navigate to:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/query/health

Or use curl:
```bash
# Health check
curl http://localhost:8000/api/query/health

# Simple query
curl -X POST "http://localhost:8000/api/query/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the radiation dose for breast cancer?",
    "top_k": 5
  }'
```

---

## 📁 What Was Added

```
src/api/
├── services/
│   └── enhanced_rag_service.py  ← NEW (Complete RAG pipeline)
├── models/
│   └── query_models.py          ← NEW (Pydantic models)
├── routes/
│   └── query.py                 ← NEW (FastAPI endpoints)
└── main.py                      ← UPDATED/CREATED

run_api.py                       ← NEW (Quick launcher)
requirements.txt                 ← UPDATED (Phase 3 dependencies)
.env.example                     ← NEW (Configuration template)
```

---

## 🔍 Features Available

- ✅ Query expansion (50+ medical abbreviations)
- ✅ Query type classification (8 types)
- ✅ Cross-encoder reranking
- ✅ NCCN guideline gap detection
- ✅ Dose-aware boosting
- ✅ Tumor site inference
- ✅ Deep dive queries with site context

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query/query` | POST | Main query endpoint |
| `/api/query/deep-dive` | POST | Deep dive with site context |
| `/api/query/health` | GET | Health check |
| `/api/query/sites` | GET | Available tumor sites |
| `/api/query/modes` | GET | Query modes |

---

## 🔧 Troubleshooting

### Import Errors
Make sure you're running from the repository root:
```bash
cd /path/to/Paxis
python run_api.py
```

### Missing Dependencies
Install all Phase 3 dependencies:
```bash
pip install fastapi uvicorn sentence-transformers torch transformers
```

### Qdrant Connection Issues
Verify your environment variables:
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('QDRANT_URL:', os.getenv('QDRANT_URL'))"
```

---

## 📚 Documentation

For complete documentation, see:
- `PHASE3_INTEGRATION_GUIDE.md` - Detailed integration guide
- `EXECUTIVE_SUMMARY.md` - Feature overview
- API Docs: http://localhost:8000/docs (when running)

---

## ✨ What's Next?

Your Paxis system now has a complete pipeline:

```
PDF → Process → Ingest → Retrieve → Generate → Answer
                          ↑ Phase 3 (Complete!)
```

Start the API and try queries like:
- "What is the standard radiation dose for breast cancer?"
- "What were the results of RTOG 0617?"
- "What is the recommended treatment for stage III NSCLC?"

Happy querying! 🎉
