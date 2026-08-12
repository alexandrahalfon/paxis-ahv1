# Paxis Repository Structure

After restructuring, the repository follows Python best practices:

```
Paxis/
│
├── src/                        # Source code
│   ├── api/                    # FastAPI application (NEW)
│   │   ├── models/             # Pydantic models
│   │   ├── routes/             # API endpoints
│   │   └── services/           # Business logic
│   │
│   ├── cli/                    # Command-line interface (NEW)
│   │   └── main.py             # Main entry point
│   │
│   ├── core/                   # Core configuration
│   │   └── config.py
│   │
│   ├── ingestion/              # Vector DB ingestion
│   │   ├── chunk_processor.py
│   │   ├── embeddings.py
│   │   ├── keyword_tagger.py
│   │   ├── pipeline.py
│   │   └── qdrant_client.py
│   │
│   ├── processing/             # PDF processing
│   │   ├── document_indexer.py
│   │   ├── document_metadata_extractor.py
│   │   ├── document_processor.py
│   │   └── extractors/
│   │
│   └── utils/                  # Utilities
│       └── gcp_sync.py
│
├── scripts/                    # Standalone scripts
│   ├── batch_process_documents.py
│   ├── migrate_from_colab.py
│   ├── run_colab_pipeline.py
│   └── validate_setup.py       # Moved from root
│
├── requirements/               # Split requirements (NEW)
│   ├── base.txt
│   ├── api.txt
│   ├── processing.txt
│   ├── ingestion.txt
│   └── dev.txt
│
├── deployment/                 # Deployment configs
│   ├── docker/
│   ├── gcp/
│   └── kubernetes/
│
├── data/                       # Data files
│   ├── keywords/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   └── logs/
│
├── docs/                       # Documentation
├── tests/                      # Tests
├── notebooks/                  # Jupyter notebooks
│
├── .env                        # Configuration
├── .env.example
├── .gitignore
├── README.md
├── setup.py                    # Package setup (NEW)
└── process_document.py         # Wrapper for backward compatibility
```

## Key Changes

1. **API Structure Added** - Ready for FastAPI development
2. **CLI Module** - Organized command-line interface
3. **Split Requirements** - Modular dependency management
4. **setup.py** - Installable package
5. **Root Level Cleaned** - Files moved to appropriate locations
6. **Backward Compatible** - Wrapper script maintains old interface

## Usage

### New way (after installing):
```bash
pip install -e .
exueed document.pdf
```

### Old way (still works):
```bash
python process_document.py document.pdf
```
