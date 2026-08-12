"""
Pydantic models for document upload and admin approval.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    """Request model for file upload."""
    filename: str = Field(..., description="Name of the uploaded file")
    file_size: int = Field(..., description="File size in bytes")


class UploadResponse(BaseModel):
    """Response model for file upload."""
    upload_id: str = Field(..., description="Unique upload ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status: pending, approved, rejected, processing, completed")
    message: str = Field(..., description="Status message")
    uploaded_at: str = Field(..., description="Upload timestamp")


class PendingUpload(BaseModel):
    """Model for pending upload."""
    upload_id: str
    filename: str
    file_size: int
    uploaded_at: datetime
    status: str = "pending"
    metadata: Optional[dict] = None
    error: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Request model for admin approval."""
    upload_id: str = Field(..., description="Upload ID to approve")
    action: str = Field(..., description="Action: 'approve' or 'reject'")
    notes: Optional[str] = Field(None, description="Optional notes for approval/rejection")


class ApprovalResponse(BaseModel):
    """Response model for approval action."""
    success: bool
    message: str
    upload_id: str
    status: str
