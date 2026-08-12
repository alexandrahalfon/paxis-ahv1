"""
FastAPI main application for Paxis medical literature API.

Integrated system with:
- Enhanced RAG query endpoints (Phase 3)
- Upload system with admin approval
- Alert system for literature monitoring
- Complete frontend serving
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import os

# Import routers
from .routes.query import router as query_router      # Enhanced RAG
from .routes.upload import router as upload_router    # Upload system
from .routes.alerts import router as alerts_router    # Alert system
from .routes.query_patient_matching import router as patient_matching_router    # Patient matching 
from .routes.reports import router as reports_router    # PDF reports
from .routes.study_details import router as study_details_router  # Study details (PostgreSQL)
from .routes.auth import router as auth_router  # Auth
from .routes.cache import router as cache_router  # Cache stats
from .routes.user_preferences import router as user_preferences_router  # User preferences
from .routes.query_classifier import router as query_classifier_router  # Query classifier
from .routes.saved_cases import router as saved_cases_router  # Saved cases
from .routes.saved_studies import router as saved_studies_router  # Saved studies
from .routes.user_uploads import router as user_uploads_router  # User uploads
from .routes.smart_search import router as smart_search_router  # Smart search
from .routes.trials import router as trials_router  # External trials search
from .routes.analytics import router as analytics_router  # Analytics
from .routes.patient_query import router as patient_query_router  # Patient QA
from .routes.analytics_online import router as analytics_online_router  # Online Analytics
from .routes.tumor_board import router as tumor_board_router  # Multi-agent Tumor Board
from .routes.patient_cases import router as patient_cases_router  # Patient Cases (patient-centric pivot)
from .routes.patient_portal import router as patient_portal_router  # Patient Portal (patient accounts + linking)
from .routes.patient_records import router as patient_records_router  # Patient-owned longitudinal record (Phase 0-2)
from .routes.communities import router as communities_router  # Community subsystem (Phase 7)
from .routes.physician_beta import router as physician_beta_router  # Physician RAG beta (convergence Sprint C item 21)


class NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    """Middleware to disable caching for HTML files during development."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Disable caching for HTML files and API responses
        if request.url.path.endswith('.html') or request.url.path == '/' or request.url.path.startswith('/api'):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app = FastAPI(
    title="Paxis Medical Literature Platform",
    description="Production RAG platform with upload, admin, and alert systems",
    version="2.0.0"
)

# Add no-cache middleware for HTML files (development)
app.add_middleware(NoCacheHTMLMiddleware)

# CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with /api prefix
app.include_router(query_router, prefix="/api", tags=["Enhanced RAG"])         # RAG
app.include_router(upload_router, prefix="/api", tags=["Upload System"])       # Upload & Admin
app.include_router(alerts_router, prefix="/api", tags=["Alert System"])        # Literature Alerts
app.include_router(patient_matching_router, prefix="/api", tags=["Patient Matching"])   # Patient Matching
app.include_router(reports_router, prefix="/api", tags=["Reports"])  # PDF Reports
app.include_router(study_details_router, prefix="/api", tags=["Study Details"])  # Study Details
app.include_router(auth_router, prefix="/api", tags=["Auth"])  # Auth
app.include_router(cache_router, prefix="/api", tags=["Cache"])  # Cache stats
app.include_router(user_preferences_router, prefix="/api", tags=["User Preferences"])  # User Preferences
app.include_router(query_classifier_router, prefix="/api", tags=["Query Classifier"])  # Query Classifier
app.include_router(saved_cases_router, prefix="/api", tags=["Saved Cases"])  # Saved Cases
app.include_router(saved_studies_router, prefix="/api", tags=["Saved Studies"])  # Saved Studies
app.include_router(user_uploads_router, prefix="/api", tags=["User Uploads"])  # User Uploads
app.include_router(smart_search_router, prefix="/api", tags=["Smart Search"])  # Smart Search
app.include_router(trials_router, prefix="/api", tags=["Trials"])  # External trials
app.include_router(analytics_router, prefix="/api", tags=["Analytics"])  # Analytics
app.include_router(patient_query_router, prefix="/api", tags=["Patient QA"])  # Patient QA
app.include_router(analytics_online_router, prefix="/api", tags=["Analytics Online"])  # Online Analytics
app.include_router(tumor_board_router, prefix="/api", tags=["Tumor Board"])  # Multi-agent Tumor Board
app.include_router(patient_cases_router, prefix="/api", tags=["Patient Cases"])  # Patient Cases
app.include_router(patient_portal_router, prefix="/api", tags=["Patient Portal"])  # Patient Portal
app.include_router(patient_records_router, prefix="/api", tags=["Patient Records"])  # Patient-owned record
app.include_router(communities_router, prefix="/api", tags=["Communities"])  # Community subsystem
app.include_router(physician_beta_router, prefix="/api", tags=["Physician RAG Beta"])  # Physician RAG beta

@app.on_event("startup")
async def warm_up():
    """Warm expensive singletons in the background when an instance boots.

    Without this, the first real request on every newly autoscaled Cloud
    Run instance pays the cross-encoder model load and the initial DB
    pool handshakes on top of its own work — exactly when concurrent
    load is what caused the scale-up.

    Deliberately fire-and-forget: warmup must never delay the container
    becoming ready (Cloud Run health checks) and must never prevent
    startup if a dependency is unreachable. Every step fails soft.
    """
    import asyncio

    async def _warm():
        # Cross-encoder: CPU-bound model load, the largest cold cost.
        try:
            from src.api.services.comprehensive_retrieval import get_comprehensive_retriever
            retriever = get_comprehensive_retriever()
            await asyncio.to_thread(retriever._get_cross_encoder)
            print("[Warmup] Cross-encoder ready")
        except Exception as e:
            print(f"[Warmup] Cross-encoder skipped: {e}")

        # DB pools: open connections now rather than on a user's request.
        try:
            from src.api.services.account_db import get_account_db
            await get_account_db().get_pool()
            print("[Warmup] Accounts DB pool ready")
        except Exception as e:
            print(f"[Warmup] Accounts DB pool skipped: {e}")

        try:
            from src.api.services.patient_db import get_patient_db
            await get_patient_db().get_pool()
            print("[Warmup] Patients DB pool ready")
        except Exception as e:
            print(f"[Warmup] Patients DB pool skipped: {e}")

    asyncio.create_task(_warm())


@app.get("/")
async def root():
    """Root now lands on the patient interface.

    Changed 2026-08-08: patients are the primary audience for the front
    door, clinicians reach their side via the "For Clinicians" switch or
    directly at /index.html. Every clinician page keeps its existing URL,
    so nothing bookmarked or internally linked breaks. Logged-in users
    are redirected to the side matching their role by interfaceSwitch.js.
    """
    return RedirectResponse(url="/patient-home.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


# Serve static frontend files
frontend_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "frontend"
)

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print(f"✅ Serving frontend from: {frontend_path}")
else:
    print(f"⚠️  Frontend directory not found: {frontend_path}")


if __name__ == "__main__":
    import uvicorn
    # Support GCP Cloud Run's PORT environment variable
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
