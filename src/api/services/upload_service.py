"""
Service for handling document uploads and admin approval workflow.
"""

import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import UploadFile

from ...core.config import get_settings


class UploadService:
    """Service for managing document uploads and approvals."""
    
    def __init__(self):
        """Initialize upload service."""
        self.settings = get_settings()
        self.uploads_dir = Path("uploads")
        self.pending_dir = self.uploads_dir / "pending"
        self.approved_dir = self.uploads_dir / "approved"
        self.rejected_dir = self.uploads_dir / "rejected"
        self.processing_dir = self.uploads_dir / "processing"
        
        # Create directories
        for dir_path in [self.pending_dir, self.approved_dir, self.rejected_dir, self.processing_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Storage file for upload metadata
        self.metadata_file = self.uploads_dir / "uploads_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load upload metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self, metadata: Dict):
        """Save upload metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    async def save_upload(self, file: UploadFile) -> str:
        """
        Save uploaded file to pending directory.
        
        Args:
            file: Uploaded file object
            
        Returns:
            upload_id: Unique identifier for the upload
        """
        upload_id = str(uuid.uuid4())
        filename = file.filename or "unknown.pdf"
        
        # Save file to pending directory
        file_path = self.pending_dir / f"{upload_id}_{filename}"
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Save metadata
        metadata = self._load_metadata()
        metadata[upload_id] = {
            "upload_id": upload_id,
            "filename": filename,
            "file_size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "status": "pending",
            "file_path": str(file_path),
            "metadata": None,
            "error": None
        }
        self._save_metadata(metadata)
        
        return upload_id
    
    def get_pending_uploads(self) -> List[Dict]:
        """Get all pending uploads."""
        metadata = self._load_metadata()
        pending = [
            upload for upload in metadata.values()
            if upload.get("status") == "pending"
        ]
        return sorted(pending, key=lambda x: x.get("uploaded_at", ""), reverse=True)
    
    def get_all_uploads(self, status: Optional[str] = None) -> List[Dict]:
        """Get all uploads, optionally filtered by status."""
        metadata = self._load_metadata()
        uploads = list(metadata.values())
        
        if status:
            uploads = [u for u in uploads if u.get("status") == status]
        
        return sorted(uploads, key=lambda x: x.get("uploaded_at", ""), reverse=True)
    
    def get_upload(self, upload_id: str) -> Optional[Dict]:
        """Get upload by ID."""
        metadata = self._load_metadata()
        return metadata.get(upload_id)
    
    def approve_upload(self, upload_id: str, notes: Optional[str] = None) -> Dict:
        """
        Approve an upload and move it to approved directory.
        
        Args:
            upload_id: Upload ID to approve
            notes: Optional approval notes
            
        Returns:
            Updated upload metadata
        """
        metadata = self._load_metadata()
        
        if upload_id not in metadata:
            raise ValueError(f"Upload {upload_id} not found")
        
        upload = metadata[upload_id]
        
        if upload["status"] != "pending":
            raise ValueError(f"Upload {upload_id} is not pending (current status: {upload['status']})")
        
        # Move file to approved directory
        old_path = Path(upload["file_path"])
        new_filename = f"{upload_id}_{upload['filename']}"
        new_path = self.approved_dir / new_filename
        
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
        
        # Update metadata
        upload["status"] = "approved"
        upload["file_path"] = str(new_path)
        upload["approved_at"] = datetime.now().isoformat()
        if notes:
            upload["approval_notes"] = notes
        
        self._save_metadata(metadata)
        
        return upload
    
    def reject_upload(self, upload_id: str, notes: Optional[str] = None) -> Dict:
        """
        Reject an upload and move it to rejected directory.
        
        Args:
            upload_id: Upload ID to reject
            notes: Optional rejection notes
            
        Returns:
            Updated upload metadata
        """
        metadata = self._load_metadata()
        
        if upload_id not in metadata:
            raise ValueError(f"Upload {upload_id} not found")
        
        upload = metadata[upload_id]
        
        if upload["status"] != "pending":
            raise ValueError(f"Upload {upload_id} is not pending (current status: {upload['status']})")
        
        # Move file to rejected directory
        old_path = Path(upload["file_path"])
        new_filename = f"{upload_id}_{upload['filename']}"
        new_path = self.rejected_dir / new_filename
        
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
        
        # Update metadata
        upload["status"] = "rejected"
        upload["file_path"] = str(new_path)
        upload["rejected_at"] = datetime.now().isoformat()
        if notes:
            upload["rejection_notes"] = notes
        
        self._save_metadata(metadata)
        
        return upload
    
    def mark_processing(self, upload_id: str) -> Dict:
        """Mark upload as processing."""
        metadata = self._load_metadata()
        
        if upload_id not in metadata:
            raise ValueError(f"Upload {upload_id} not found")
        
        upload = metadata[upload_id]
        upload["status"] = "processing"
        upload["processing_started_at"] = datetime.now().isoformat()
        
        self._save_metadata(metadata)
        
        return upload
    
    def mark_completed(self, upload_id: str, error: Optional[str] = None) -> Dict:
        """Mark upload as completed or failed."""
        metadata = self._load_metadata()
        
        if upload_id not in metadata:
            raise ValueError(f"Upload {upload_id} not found")
        
        upload = metadata[upload_id]
        upload["status"] = "completed" if not error else "failed"
        upload["completed_at"] = datetime.now().isoformat()
        if error:
            upload["error"] = error
        
        self._save_metadata(metadata)
        
        return upload
