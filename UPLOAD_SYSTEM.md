# Document Upload & Admin Approval System

## Overview

The upload system allows users to drag-and-drop PDF documents for processing and ingestion into the knowledge base. Documents require admin approval before being processed.

## Workflow

1. **User Uploads Document** → File stored in `uploads/pending/`
2. **Admin Reviews** → Admin views pending uploads on admin page
3. **Admin Approves** → File moved to `uploads/approved/` and processing starts
4. **Processing** → Document goes through:
   - OCR (Mistral OCR)
   - Vision processing (Pixtral)
   - Table/Figure extraction
   - Chunking and embedding
   - Qdrant ingestion
5. **Completed** → Document is now searchable in the knowledge base

## Backend API Endpoints

### Upload Endpoints

- `POST /api/upload/` - Upload a PDF file
- `GET /api/upload/pending` - Get all pending uploads
- `GET /api/upload/all` - Get all uploads (with optional status filter)
- `GET /api/upload/{upload_id}` - Get specific upload details

### Admin Endpoints

- `POST /api/upload/admin/approve` - Approve an upload (triggers processing)
- `POST /api/upload/admin/reject` - Reject an upload

## Frontend Pages

### Upload Page (`/frontend/upload.html`)
- Drag-and-drop interface
- File upload with progress
- View uploaded files and their status
- **Visible in navigation** - anyone can upload

### Admin Page (`/frontend/admin.html`)
- View all uploads with status filtering
- Approve/reject pending uploads
- Add notes to approvals/rejections
- Auto-refreshes every 10 seconds
- **NOT in regular navigation** - access directly via URL

## File Structure

```
uploads/
├── pending/          # Files waiting for approval
├── approved/         # Approved files (being processed)
├── rejected/         # Rejected files
├── processing/       # Currently processing (not used, status only)
└── uploads_metadata.json  # Metadata for all uploads
```

## Status Flow

```
pending → approved → processing → completed
         ↓
      rejected
```

## How to Use

### For Users

1. Go to `http://localhost:8080/upload.html`
2. Drag and drop PDF files or click to browse
3. Files will show as "Pending" status
4. Wait for admin approval

### For Admins

1. Go to `http://localhost:8080/admin.html` (not in navigation)
2. View pending uploads
3. Click "Approve" to start processing
4. Or click "Reject" to reject the upload
5. Processing happens in background
6. Check status updates (auto-refreshes)

## Processing Details

When an upload is approved:

1. **Document Processing** (using `CompleteDocumentProcessor`):
   - Mistral OCR extraction
   - Pixtral vision processing
   - Table and figure extraction
   - Metadata extraction

2. **Ingestion Pipeline** (using `ColabIngestionPipeline`):
   - Document chunking
   - Section windowing
   - Keyword tagging
   - Embedding generation
   - Qdrant vector storage

3. **Result**: Document is searchable via RAG queries

## Notes

- Only PDF files are accepted
- Processing happens in background threads
- Admin page is hidden from navigation (access via direct URL)
- All uploads are tracked in `uploads_metadata.json`
- Processing can take several minutes depending on document size
