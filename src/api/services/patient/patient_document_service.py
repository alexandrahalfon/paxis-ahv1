"""
Patient Document Service (Phase 2)

Storage + lifecycle for patient-uploaded documents (lab reports, pathology,
imaging reports, visit summaries, medication lists — phone photo or PDF).
Separate from user_uploads_service.py on purpose: that pipeline ingests
studies into the literature/session-embedding path; this one ingests a
patient's own records into their canonical chart and never touches Qdrant
or the literature corpus. See patient_document_extractor.py for the
OCR/structuring step and patient_document_validator.py for the
confirm-before-canonicalizing step this service defers to.

Storage: GCS (settings.gcp_patient_documents_bucket) when configured,
local `patient_documents/` directory as the fallback otherwise — see
patient_document_storage.py for why patient documents get their own
dedicated bucket setting rather than reusing the literature corpus's,
and for how a document's object_storage_uri tells the two shapes apart
with no migration needed for documents already stored locally before
this change (2026-08-12 beta audit item 3, "not beta-safe for Cloud
Run" — Cloud Run instances are ephemeral and don't share a filesystem).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict
from src.api.services.patient import patient_document_storage

# Recognized so the classifier has a closed vocabulary to fall back to
# rather than free text; see patient_document_extractor.classify().
DOCUMENT_TYPES = (
    "lab", "pathology", "imaging", "visit_summary", "medication_list",
    "discharge_instructions", "unclassified",
)


class PatientDocumentService:
    async def create_document(
        self,
        patient_profile_id: str,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        document_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        document_id = str(uuid.uuid4())
        storage_uri = await patient_document_storage.store(
            patient_profile_id, document_id, filename, content
        )

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO patient_documents
                    (id, patient_profile_id, filename, content_type,
                     object_storage_uri, document_date, extraction_status)
                VALUES ($1, $2, $3, $4, $5, $6, 'pending')
                RETURNING *
                """,
                document_id, patient_profile_id, filename, content_type,
                storage_uri, document_date,
            )
        return row_to_dict(row)

    async def get_document(self, document_id: str, patient_profile_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM patient_documents WHERE id = $1 AND patient_profile_id = $2",
                document_id, patient_profile_id,
            )
        return row_to_dict(row) if row else None

    async def list_documents(self, patient_profile_id: str) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM patient_documents
                 WHERE patient_profile_id = $1
                 ORDER BY uploaded_at DESC
                """,
                patient_profile_id,
            )
        return [row_to_dict(r) for r in rows]

    async def update_status(
        self, document_id: str, status: str,
        document_type: Optional[str] = None,
        parser_version: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE patient_documents
                   SET extraction_status = $2,
                       document_type = COALESCE($3, document_type),
                       parser_version = COALESCE($4, parser_version),
                       error_message = $5
                 WHERE id = $1
                """,
                document_id, status, document_type, parser_version, error_message,
            )


_service: Optional[PatientDocumentService] = None


def get_patient_document_service() -> PatientDocumentService:
    global _service
    if _service is None:
        _service = PatientDocumentService()
    return _service
