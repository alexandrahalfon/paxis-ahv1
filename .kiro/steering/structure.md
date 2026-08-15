# Project Structure

## Top-Level Layout

```
src/                    # All application source code
frontend/               # Static HTML/CSS/JS frontend
tests/                  # Test suite
scripts/                # Standalone utility/migration scripts
data/                   # Static data files (ontology, keywords, staging tables)
config/                 # Configuration data (keyword lists)
migrations/             # Database migration files (Alembic)
pipeline/               # Ingestion pipeline modules (patient education)
notebooks/              # Jupyter notebooks for experimentation
deployment/             # Deployment configs (GCP, Heroku)
requirements/           # Split pip requirements by concern
uploads/                # Runtime upload directory (pending/approved/rejected/processing)
```

## Source Code (`src/`)

```
src/
├── api/
│   ├── main.py              # FastAPI app entrypoint
│   ├── models/              # Pydantic request/response models
│   ├── routes/              # API endpoint handlers (one file per feature)
│   └── services/            # Business logic layer (bulk of the codebase)
│       ├── safety/          # Safety guardrails for LLM outputs
│       ├── tumor_board/     # Multi-agent tumor board (orchestrator + agents)
│       ├── patient/         # Patient record services
│       ├── patient_portal/  # Patient-facing portal services
│       ├── physician/       # Physician-specific RAG services
│       ├── evidence/        # Evidence classification/grounding
│       └── community/       # Community features
├── cli/                     # CLI entrypoint
├── core/
│   └── config.py            # Pydantic Settings config (reads .env)
├── ingestion/               # Document ingestion pipeline
│   ├── colab_pipeline.py    # Main ingestion orchestrator
│   ├── chunk_processor.py   # Text chunking with section windows
│   ├── embeddings.py        # Embedding generation
│   ├── qdrant_client.py     # Qdrant operations
│   └── pto_frame_builder.py # Patient-Treatment-Outcome frame builder
├── processing/              # PDF document processing
│   ├── document_processor.py     # OCR + text extraction
│   ├── study_profile_extractor.py # Structured profile extraction
│   └── extractors/               # Specialized content extractors
└── utils/                   # Shared utilities (GCP sync, evidence classifier)
```

## Key Services (most development happens here)

| Service | Purpose |
|---------|---------|
| `enhanced_rag_service.py` | Main RAG orchestrator — query expansion, retrieval, synthesis |
| `comprehensive_retrieval.py` | 4-phase retrieval pipeline (Qdrant + PG + PTO + reranking) |
| `query_structuring_service.py` | Regex + LLM extraction of clinical axes from queries |
| `structured_study_matcher.py` | PostgreSQL structured matching with multi-field scoring |
| `clinical_inference.py` | Ontology inference — maps implicit clinical status to explicit labels |
| `patient_matching_service_simple.py` | Patient-to-trial matching |
| `smart_search_service.py` | Intelligent search combining preferences + context |
| `report_service.py` | PDF report generation |

## Frontend (`frontend/`)

- One HTML file per page/feature
- `frontend/js/` — JavaScript modules (API client, UI components, conversation management)
- `frontend/css/styles.css` — single stylesheet
- `frontend/assets/` — static images and assets

## Routes (`src/api/routes/`)

Each route file maps to a feature area and is mounted on the FastAPI app with a prefix (e.g., `/api/rag`, `/api/trials`, `/api/auth`).

## Conventions

- Services are instantiated as singletons or module-level instances
- Routes delegate to services; business logic stays in `services/`
- Pydantic models define API contracts in `models/`
- Configuration flows through `src/core/config.py` (Pydantic Settings reading `.env`)
- Async throughout — use `async def` for I/O-bound operations
- Print-based logging (no structured logging framework currently)
