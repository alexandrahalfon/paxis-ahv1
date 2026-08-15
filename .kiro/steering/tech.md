# Tech Stack & Build

## Language & Runtime

- Python 3.12+ (production Docker image uses 3.11-slim)
- Async-first (FastAPI + asyncpg)

## Backend

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI |
| Production server | Gunicorn + Uvicorn workers |
| Vector database | Qdrant (cloud-hosted) |
| Relational database | PostgreSQL (GCP Cloud SQL) — 3 databases on same server |
| Async DB driver | asyncpg |
| ORM/migrations | SQLAlchemy + Alembic |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-large (3072 dimensions) |
| OCR | Mistral OCR + Pixtral |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers) |
| PDF reports | ReportLab |
| Auth | JWT via python-jose, passlib/bcrypt |

## Frontend

- Vanilla HTML/CSS/JS (no framework)
- Static files served via Nginx in production
- Single `frontend/` directory with page-per-file structure

## Databases

- `display-study-details` — knowledge base study profiles
- `study-profiles` — normalized profiles with lookup tables for filtering
- `exueed_cache` — user accounts, saved cases, uploads
- `exueed-patients` — patient records, diagnoses, biomarkers, timeline events

## Infrastructure

- GCP Cloud Run (containerized deployment)
- Docker + docker-compose for local/staging
- Nginx reverse proxy for frontend
- GCP Cloud Storage for document storage

## Package Management

- `requirements.txt` — unified/merged requirements (primary install target)
- `requirements/` — split requirements by concern (base, api, processing, ingestion, dev, prod)
- `setup.py` — installable package with extras (`pip install -e .[api,processing,dev]`)
- `pyproject.toml` — pytest configuration only

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python run_api.py

# Run tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_patient_matching_service.py -v

# Build Docker image
docker build -t paxis-platform .

# Run via docker-compose
docker-compose up

# Deploy to GCP Cloud Run
./deploy.sh

# Process a document
python process_document.py <path-to-pdf>

# Validate local setup
python scripts/validate_setup.py
```

## Environment Configuration

All configuration via `.env` file (see `.env.example` for full reference). Key variables:
- `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
- `POSTGRES_*` — main study database connection
- `CACHE_POSTGRES_*` — user/auth database connection
- `PATIENTS_POSTGRES_*` — patient records database connection
- Feature flags: `ENABLE_PIXTRAL`, `USE_RECONCILED_STRUCTURE`, etc.

## Testing

- pytest with pytest-asyncio (`asyncio_mode = "auto"`)
- Test directory: `tests/`
- Test fixtures in `tests/fixtures/`
- No watch mode — run `python -m pytest` directly
