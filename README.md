# Paxis Medical Literature Platform

An AI-powered medical literature search platform for oncology, providing evidence-based answers, patient-to-trial matching, virtual tumor board consultations, and study comparison tools.

## Features

### AI-Powered Query
Ask questions in natural language and receive comprehensive answers with citations from peer-reviewed medical literature.

- **Query Expansion**: Automatically expands medical abbreviations (RT → radiation therapy)
- **Cross-Encoder Reranking**: Improves relevance using neural reranking
- **Conversation Mode**: Multi-turn conversations with context retention
- **Evidence Packing**: Groups sources by study with key excerpts
- **User Preferences**: Filter by cancer type, study phase, treatment modality, and more
- **Layered Responses**: Structured output with in-chat interactive modules
- **Follow-up Suggestions**: AI-generated follow-up questions for deeper exploration

### Virtual Tumor Board
Present a patient case to a panel of AI specialty agents who provide parallel assessments.

- Multi-agent architecture with specialty-specific retrieval
- Radiation oncology, medical oncology, and other specialist perspectives
- Evidence-backed recommendations with cited literature
- Parallel execution for fast response times

### Trial Search (ClinicalTrials.gov)
Search external clinical trials from ClinicalTrials.gov matched to a patient profile.

- Extracts patient profile from free-text clinical narratives
- AJCC staging validation and correction
- Structured search with condition and term separation
- Match scoring based on disease, demographics, and treatment history
- Filter by recruiting status, phase, and location

### Smart Search
Intelligent search combining persistent user preferences with case-specific context.

- Applies saved user preferences as database filters
- Extracts case context from queries for relevance scoring
- Re-ranks results by patient similarity
- Configurable sort (relevance, population size, date, citations, outcomes)
- Preview filter impact before searching

### Trial Match Mode
Toggle on "Trial Match" in the main query interface to find internal knowledge base studies matching a patient description.

- Extracts patient profile from free text (age, cancer type, stage, biomarkers)
- Finds matching trials with eligibility criteria
- Shows inclusion/exclusion criteria and patient fit assessment
- Provides match scores and reasoning

### Patient Matching
Dedicated page for detailed patient-to-trial matching.

- Enter comprehensive patient profiles
- Get ranked list of matching studies
- View eligibility criteria analysis

### Treatment Comparison
Compare different treatment approaches side-by-side.

- Enter treatments to compare (e.g., "SBRT vs conventional fractionation")
- Get structured comparison with outcomes data
- View supporting evidence for each treatment

### Study Comparison
Compare 2-4 studies side-by-side with visualizations.

- Search knowledge base or use your uploaded studies
- Find similar studies based on cancer type, location, histology
- Generate comparison charts and AI narrative summary

### Document Upload
Upload your own PDFs for analysis and searching.

- Automatic text extraction (Mistral OCR with PyMuPDF fallback)
- Study profile extraction
- Search across your uploaded documents
- Find similar studies in the knowledge base

### Knowledge Base Analytics
Explore aggregate statistics across the entire knowledge base.

- Overview stats (total studies, cancer types, avg outcomes)
- Custom aggregate queries by metric and grouping
- Dose distribution and technique frequency charts
- Outcomes by stage analysis
- Meta-analysis forest plots

### PDF Report Generation
Export results as downloadable PDF reports.

- Patient match reports
- Treatment comparison reports
- Query result reports in multiple formats: standard, patient handout, clinic note

### Query Classifier
Structured extraction from clinical narratives for database matching.

- Extracts demographics, diagnosis, staging, pathology, treatment history, risk factors
- Builds PostgreSQL filter queries from extracted fields
- Combined vector + structured search

### Saved Cases and Alerts
Save patient cases and get notified of new matching trials.

- Save queries with full responses and sources
- Save individual studies for later reference
- Enable alerts for new trial matches
- View saved cases and studies in My Saves

### Authentication and User Accounts
JWT-based authentication with user preferences.

- User registration and login
- Persistent user preferences (cancer types, study phases, countries, sort order)
- Per-user saved cases, studies, and uploads

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python) |
| Vector DB | Qdrant (cloud) |
| Relational DB | PostgreSQL (GCP Cloud SQL) |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-large (3072 dim) |
| OCR | Mistral OCR + Pixtral |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting | GCP Cloud Run |
| Reports | ReportLab (PDF generation) |

## Quick Start

### Prerequisites
- Python 3.12+
- API keys for OpenAI, Mistral, Qdrant

### Local Development

```bash
# Clone and setup
git clone <repository>
cd exueed-updated

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
python run_api.py
```

Open http://localhost:8000 in your browser.

### Docker

```bash
docker build -t exueed-platform .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  -e QDRANT_URL=$QDRANT_URL \
  -e QDRANT_API_KEY=$QDRANT_API_KEY \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  exueed-platform
```

## Project Structure

```
src/
├── api/
│   ├── main.py                 # FastAPI application
│   ├── routes/                 # API endpoints
│   │   ├── query.py            # RAG query endpoints
│   │   ├── tumor_board.py      # Virtual tumor board
│   │   ├── trials.py           # ClinicalTrials.gov search
│   │   ├── smart_search.py     # Smart search with preferences
│   │   ├── query_classifier.py # Query structuring
│   │   ├── user_uploads.py     # Document upload
│   │   ├── saved_cases.py      # Saved patient cases
│   │   ├── saved_studies.py    # Saved studies
│   │   ├── reports.py          # PDF report generation
│   │   ├── alerts.py           # Trial match alerts
│   │   ├── auth.py             # Authentication
│   │   ├── user_preferences.py # User preferences
│   │   ├── analytics.py        # KB analytics
│   │   ├── analytics_online.py # Online analytics
│   │   ├── study_details.py    # Study profiles
│   │   └── ...
│   ├── services/               # Business logic
│   │   ├── enhanced_rag_service.py
│   │   ├── patient_matching_service_simple.py
│   │   ├── query_intent_service.py
│   │   ├── query_structuring_service.py
│   │   ├── smart_search_service.py
│   │   ├── report_service.py
│   │   ├── literature_search_service.py
│   │   ├── alert_service.py
│   │   ├── auth_service.py
│   │   ├── safety/             # Safety guardrails
│   │   ├── tumor_board/        # Multi-agent tumor board
│   │   │   ├── orchestrator.py
│   │   │   ├── base_agent.py
│   │   │   ├── agents/
│   │   │   └── ...
│   │   └── ...
│   └── models/                 # Pydantic models
├── processing/                 # Document processing
│   ├── document_processor.py
│   └── study_profile_extractor.py
└── core/
    └── config.py               # Configuration

frontend/
├── index.html                  # Main query interface
├── upload.html                 # Document upload
├── study-comparison.html       # Study comparison
├── treatment-comparison.html   # Treatment comparison
├── patient-matching.html       # Patient matching
├── patient-qa.html             # Patient Q&A
├── trial-search.html           # External trial search
├── my-saves.html               # Saved cases & studies
├── analytics.html              # Knowledge base analytics
├── login.html                  # Authentication
├── admin.html                  # Admin panel
├── about.html                  # About page
├── js/
│   ├── api.js                  # API client
│   ├── conversationManager.js  # Multi-turn conversation
│   ├── conversationContext.js  # Context retention
│   ├── layered_response_renderer.js  # Structured output
│   ├── inChatModules.js        # Interactive in-chat modules
│   ├── preferences.js          # User preferences UI
│   ├── trialSearch.js          # Trial search UI
│   ├── studyDetails.js         # Study details panel
│   ├── studyDetailsRenderer.js # Study rendering
│   ├── splitPanel.js           # Split panel layout
│   ├── auth-ui.js              # Auth UI components
│   └── ...
└── css/
    └── styles.css
```

## API Endpoints

### RAG Query
```
POST /api/rag/query              # Main query endpoint
POST /api/rag/analyze-intent     # Trial match mode
POST /api/rag/deep-dive          # Deep dive with context
GET  /api/rag/health             # Health check
```

### Tumor Board
```
POST /api/tumor-board            # Present case to virtual tumor board
GET  /api/tumor-board/panel      # List specialty agents on the panel
```

### Trial Search (ClinicalTrials.gov)
```
POST /api/trials/search          # Search external clinical trials
```

### Smart Search
```
POST /api/smart-search           # Intelligent preference + context search
POST /api/smart-search/extract-context  # Extract case context from query
GET  /api/smart-search/preview-filters  # Preview filter impact
```

### Query Classifier
```
POST /api/query-classifier/classify  # Classify query into structured fields
POST /api/query-classifier/search    # Search studies with classified query
```

### Reports
```
POST /api/report/patient-match          # Generate patient match PDF
POST /api/report/treatment-comparison   # Generate treatment comparison PDF
POST /api/report/query                  # Generate query result PDF
```

### Authentication
```
POST /api/auth/register          # Create account
POST /api/auth/login             # Login (returns JWT)
GET  /api/auth/me                # Get current user
```

### User Preferences
```
GET  /api/user-preferences       # Get user preferences
PUT  /api/user-preferences       # Update preferences
```

### User Features
```
POST /api/user-uploads/process   # Upload document
GET  /api/user-uploads           # List uploads
POST /api/saved-cases            # Save a case
GET  /api/saved-cases            # List saved cases
POST /api/saved-studies          # Save a study
GET  /api/saved-studies          # List saved studies
```

### Alerts
```
POST /api/alerts                 # Create alert
GET  /api/alerts                 # List alerts
DELETE /api/alerts/{id}          # Delete alert
```

### Analytics
```
GET  /api/analytics/overview         # Knowledge base stats
GET  /api/analytics/filters          # Available filter options
POST /api/analytics/aggregate        # Custom aggregate queries
POST /api/analytics/dose-distribution
POST /api/analytics/technique-frequency
POST /api/analytics/outcomes-by-stage
POST /api/analytics/forest-plot
```

### Study Details
```
GET  /api/study-details/{doc_id} # Get study profile
POST /api/study-details/compare  # Compare studies
```

## Environment Variables

### Required
```bash
OPENAI_API_KEY=sk-...           # OpenAI API key
MISTRAL_API_KEY=...             # Mistral API key
QDRANT_URL=https://...          # Qdrant cloud URL
QDRANT_API_KEY=...              # Qdrant API key
POSTGRES_PASSWORD=...           # PostgreSQL password
```

### Optional
```bash
QDRANT_COLLECTION=exueed_kb_latest
POSTGRES_HOST=34.21.60.224
POSTGRES_DATABASE=display-study-details
AUTH_SECRET_KEY=...             # JWT secret
```

## Databases

### PostgreSQL (3 databases on same server)
- `display-study-details` — Knowledge base study profiles
- `study-profiles` — Normalized profiles with lookup tables for filtering
- `exueed_cache` — User accounts, saved cases, uploads

### Qdrant
- Collection: `exueed_kb_latest`
- Dimension: 3072 (text-embedding-3-large)

## Development

### Running Tests
```bash
python -m pytest tests/
```

### API Documentation
With the server running, visit http://localhost:8000/docs for Swagger UI.

## Deployment

See `GCP_DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

```bash
# Quick deploy to GCP Cloud Run
./deploy.sh
```

## License

Proprietary - All rights reserved.

## Team

- Aysha Allahverdiyeva - Co-Founder & Developer
- Alexandra Halfon - Co-Founder & Engineer
- Dr. Jinyu Xue - Co-Founder & Medical Advisor
