"""
API routes for study alerts and literature search.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..services.alert_service import AlertService
from ..services.literature_search_service import LiteratureSearchService
from ..services.study_matching_service import StudyMatchingService
from ..services.upload_service import UploadService
from ..services.pdf_download_service import PDFDownloadService
from fastapi import UploadFile
from pathlib import Path
from datetime import datetime

router = APIRouter(prefix="/alerts", tags=["alerts"])

alert_service = AlertService()
search_service = LiteratureSearchService()
upload_service = UploadService()
pdf_download_service = PDFDownloadService()


def _get_matching_service() -> StudyMatchingService:
    try:
        return StudyMatchingService()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Study matching unavailable: {e}")


class AlertRequest(BaseModel):
    """Request model for creating/updating alerts."""
    cancer_type: str
    search_terms: Optional[List[str]] = None
    frequency: str = "daily"
    enabled: bool = True


class SearchRequest(BaseModel):
    """Request model for searching literature."""
    cancer_type: Optional[str] = None
    days_back: int = 30
    max_results: int = 50


class MatchRequest(BaseModel):
    """Request model for matching studies."""
    studies: List[Dict[str, Any]]


@router.get("/cancer-types")
async def get_cancer_types():
    """Get cancer types from existing knowledge base."""
    try:
        matching_service = _get_matching_service()
        cancer_types = matching_service.extract_cancer_types_from_kb()
        stats = matching_service.get_study_statistics()
        return {
            "cancer_types": cancer_types,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_literature(request: SearchRequest):
    """Search medical literature databases for new studies."""
    try:
        results = search_service.search_radiation_oncology(
            cancer_type=request.cancer_type,
            days_back=request.days_back,
            max_results=request.max_results
        )
        
        # Match to existing cancer types
        matching_service = _get_matching_service()
        existing_types = matching_service.extract_cancer_types_from_kb()
        matches = matching_service.match_studies_by_cancer_type(results, existing_types)
        
        return {
            "total_results": len(results),
            "studies": results,
            "matches_by_cancer_type": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match")
async def match_studies(request: MatchRequest):
    """Match new studies to existing cancer types."""
    try:
        matching_service = _get_matching_service()
        existing_types = matching_service.extract_cancer_types_from_kb()
        matches = matching_service.match_studies_by_cancer_type(
            request.studies,
            existing_types
        )
        return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_alerts():
    """Get all alerts."""
    return alert_service.get_alerts()


@router.post("/")
async def create_alert(request: AlertRequest):
    """Create a new alert."""
    try:
        alert = alert_service.create_alert(
            cancer_type=request.cancer_type,
            search_terms=request.search_terms,
            frequency=request.frequency,
            enabled=request.enabled
        )
        return alert
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{alert_id}")
async def update_alert(alert_id: str, updates: Dict[str, Any]):
    """Update an alert."""
    try:
        alert = alert_service.update_alert(alert_id, updates)
        return alert
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert."""
    try:
        success = alert_service.delete_alert(alert_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
async def check_alerts():
    """Check all enabled alerts for new studies."""
    try:
        results = alert_service.check_alerts()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-study")
async def upload_study_from_search(study: Dict[str, Any]):
    """
    Upload a study found in literature search.
    This will:
    1. Attempt to download PDF from DOI/PMID/URL
    2. Create a pending upload entry
    3. The PDF will go through full processing when approved (OCR, Pixtral, ingestion)
    """
    try:
        # Step 1: Attempt to download PDF
        pdf_path = pdf_download_service.download_study_pdf(study)
        
        if not pdf_path or not pdf_path.exists():
            return {
                "success": False,
                "message": "Could not download PDF. Please upload manually.",
                "study": study,
                "note": "PDF download failed. You can manually upload the PDF using the Upload page."
            }
        
        # Step 2: Move PDF to pending directory and create upload entry
        import shutil
        import uuid
        
        # Generate filename
        title = study.get("title", "study")
        filename = "".join(c for c in title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = filename.replace(' ', '_') + '.pdf'
        
        # Create upload ID
        upload_id = str(uuid.uuid4())
        
        # Move PDF to pending directory
        pending_dir = upload_service.pending_dir
        pending_path = pending_dir / f"{upload_id}_{filename}"
        shutil.move(str(pdf_path), str(pending_path))
        
        # Create upload metadata
        upload = {
            "upload_id": upload_id,
            "filename": filename,
            "file_size": pending_path.stat().st_size,
            "uploaded_at": datetime.now().isoformat(),
            "status": "pending",
            "file_path": str(pending_path),
            "study_metadata": {
                "title": study.get("title"),
                "authors": study.get("authors", []),
                "journal": study.get("journal"),
                "year": study.get("year"),
                "doi": study.get("doi"),
                "pmid": study.get("pmid"),
                "source": study.get("source"),
                "abstract": study.get("abstract", "")[:500] if study.get("abstract") else ""
            },
            "metadata": None,
            "error": None
        }
        
        # Save metadata
        metadata = upload_service._load_metadata()
        metadata[upload_id] = upload
        upload_service._save_metadata(metadata)
        
        return {
            "success": True,
            "message": f"PDF downloaded and uploaded successfully. Upload ID: {upload_id}. Please approve in Admin panel to start processing.",
            "upload_id": upload_id,
            "filename": filename,
            "status": "pending",
            "study": study,
            "note": "The document will go through full processing (OCR, Pixtral, ingestion) when approved."
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading study: {str(e)}")
