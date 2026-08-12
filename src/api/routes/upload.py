"""
API routes for document upload and admin approval.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import threading
import asyncio

from ..models.upload import (
    UploadResponse, 
    PendingUpload, 
    ApprovalRequest, 
    ApprovalResponse
)
from ..services.upload_service import UploadService
from ..services.document_processing_service import DocumentProcessingService

router = APIRouter(prefix="/upload", tags=["upload"])

upload_service = UploadService()
processing_service = DocumentProcessingService()


@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing.
    
    The document will be stored in pending status until approved by an admin.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        upload_id = await upload_service.save_upload(file)
        upload = upload_service.get_upload(upload_id)
        
        return UploadResponse(
            upload_id=upload_id,
            filename=upload["filename"],
            status=upload["status"],
            message="File uploaded successfully. Waiting for admin approval.",
            uploaded_at=upload["uploaded_at"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/pending", response_model=List[PendingUpload])
async def get_pending_uploads():
    """Get all pending uploads (admin only)."""
    uploads = upload_service.get_pending_uploads()
    return [PendingUpload(**upload) for upload in uploads]


@router.get("/all")
async def get_all_uploads(status: str = None):
    """Get all uploads, optionally filtered by status (admin only)."""
    uploads = upload_service.get_all_uploads(status=status)
    return uploads


@router.get("/{upload_id}")
async def get_upload(upload_id: str):
    """Get upload details by ID."""
    upload = upload_service.get_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    return upload


def process_approved_document_sync(upload_id: str, file_path: str):
    """Synchronous wrapper for document processing (runs in background thread)."""
    from pathlib import Path
    
    upload_service.mark_processing(upload_id)
    
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            processing_service.process_document(Path(file_path), upload_id)
        )
        loop.close()
        
        if result["success"]:
            upload_service.mark_completed(upload_id)
        else:
            upload_service.mark_completed(upload_id, error=result.get("error"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        upload_service.mark_completed(upload_id, error=str(e))


@router.post("/admin/approve", response_model=ApprovalResponse)
async def approve_upload(
    request: ApprovalRequest,
    background_tasks: BackgroundTasks
):
    """
    Approve an upload and trigger processing (admin only).
    
    This will:
    1. Move file to approved directory
    2. Start background processing (OCR, vision, ingestion)
    """
    if request.action != "approve":
        raise HTTPException(status_code=400, detail="Invalid action. Use 'approve' or call reject endpoint.")
    
    try:
        upload = upload_service.approve_upload(request.upload_id, request.notes)
        
        # Start background processing in a separate thread
        thread = threading.Thread(
            target=process_approved_document_sync,
            args=(request.upload_id, upload["file_path"]),
            daemon=True
        )
        thread.start()
        
        return ApprovalResponse(
            success=True,
            message="Upload approved. Processing started in background.",
            upload_id=request.upload_id,
            status="approved"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")


@router.post("/admin/reject", response_model=ApprovalResponse)
async def reject_upload(request: ApprovalRequest):
    """
    Reject an upload (admin only).
    
    The file will be moved to rejected directory.
    """
    if request.action != "reject":
        raise HTTPException(status_code=400, detail="Invalid action. Use 'reject' or call approve endpoint.")
    
    try:
        upload = upload_service.reject_upload(request.upload_id, request.notes)
        
        return ApprovalResponse(
            success=True,
            message="Upload rejected.",
            upload_id=request.upload_id,
            status="rejected"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rejection failed: {str(e)}")
