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

Storage: local `patient_documents/` directory, mirroring
UserUploadsService's `user_uploads/` convention, since GCS credentials
(gcp_user_uploads_bucket) are optional/unset in most environments this
runs in. Swapping to GCS later only touches _store_file below.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict

_STORAGE_DIR = Path("patient_documents")

# Recognized so the classifier has a closed vocabulary to fall back to
# rather than free text; see patient_document_extractor.classify().
DOCUMENT_TYPES = (
    "lab", "pathology", "imaging", "visit_summary", "medication_list",
    "discharge_instructions", "unclassified",
)


class PatientDocumentService:
    def _store_file(self, patient_profile_id: str, document_id: str, filename: str, content: bytes) -> str:
        """Writes to local disk and returns the storage URI. Isolated so a
        future GCS swap is a one-function change, matching the
        object_storage_uri column's intent (not tied to 'local path')."""
        _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename or "upload").name
        dest_dir = _STORAGE_DIR / patient_profile_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{document_id}_{safe_name}"
        dest.write_bytes(content)
        return str(dest)

    async def create_document(
        self,
        patient_profile_id: str,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        document_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        document_id = str(uuid.uuid4())
        storage_uri = self._store_file(patient_profile_id, document_id, filename, content)

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
