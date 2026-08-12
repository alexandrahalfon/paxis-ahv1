"""
Patient Document Storage (2026-08-12, beta audit item 3)

Where a patient's uploaded document bytes actually live, abstracted
behind store()/read() so patient_document_service.py and
patient_document_extractor.py never need to know whether a given
object_storage_uri points at GCS or local disk.

Local disk (patient_documents/<profile>/<doc_id>_<filename>, a bare path
with no URI scheme — the exact shape this module always produced before
this change) was the only option before this change: fine for a single
long-lived instance with a persistent filesystem, but Cloud Run
instances are ephemeral and don't share a filesystem, so a document
uploaded to one instance could 404 when a later request lands on a
different one, and every document is lost on scale-to-zero. See the
2026-08-12 beta audit, "patient document storage is not beta-safe for
Cloud Run" — flagged as a beta blocker if document upload ships.

GCS (settings.gcp_patient_documents_bucket, uri shape
gs://bucket/patient_documents/<profile>/<doc_id>_<filename>) is used
whenever that setting is configured — a dedicated bucket, deliberately
separate from settings.gcp_bucket_name (the literature/study corpus
bucket used by gcp_sync.py) and gcp_user_uploads_bucket (a distinct,
still-unused setting for a different, unrelated feature): patient
documents are PHI, with different access-control and retention
requirements than published literature, and should never share a bucket
— or its IAM policy — with content that has no such requirements.

Local disk remains the fallback when no bucket is configured. This
sandbox, and most dev/CI environments, have no GCS credentials at all —
see gcp_sync.py's identical fallback pattern for the literature corpus —
so requiring GCS unconditionally would break local development entirely,
not just make it inconvenient. Read the module docstring on
read()/store() for exactly how the two URI shapes are told apart: no
schema migration is needed for documents already stored locally before
this change, since a bare local path never starts with "gs://".
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.core.config import settings

logger = logging.getLogger(__name__)

_LOCAL_STORAGE_DIR = Path("patient_documents")

_GCS_PREFIX = "gs://"


def is_gcs_configured() -> bool:
    return bool(settings.gcp_patient_documents_bucket)


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _upload_to_gcs(key: str, content: bytes) -> None:
    client = _gcs_client()
    bucket = client.bucket(settings.gcp_patient_documents_bucket)
    bucket.blob(key).upload_from_string(content)


def _download_from_gcs(gcs_uri: str) -> bytes:
    without_scheme = gcs_uri[len(_GCS_PREFIX):]
    bucket_name, _, key = without_scheme.partition("/")
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    return bucket.blob(key).download_as_bytes()


def _delete_from_gcs(gcs_uri: str) -> None:
    without_scheme = gcs_uri[len(_GCS_PREFIX):]
    bucket_name, _, key = without_scheme.partition("/")
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    bucket.blob(key).delete()


def _store_local(patient_profile_id: str, document_id: str, safe_name: str, content: bytes) -> str:
    _LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = _LOCAL_STORAGE_DIR / patient_profile_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{document_id}_{safe_name}"
    dest.write_bytes(content)
    return str(dest)


async def store(patient_profile_id: str, document_id: str, filename: str, content: bytes) -> str:
    """Persists content and returns the storage URI to save on the
    patient_documents row. Threaded: both the GCS client and Path.write_bytes
    are blocking calls."""
    safe_name = Path(filename or "upload").name
    if is_gcs_configured():
        key = f"patient_documents/{patient_profile_id}/{document_id}_{safe_name}"
        await asyncio.to_thread(_upload_to_gcs, key, content)
        return f"{_GCS_PREFIX}{settings.gcp_patient_documents_bucket}/{key}"
    return await asyncio.to_thread(_store_local, patient_profile_id, document_id, safe_name, content)


async def read(storage_uri: str) -> bytes:
    """Fetches content back given a URI store() returned — dispatches on
    the "gs://" prefix, so it transparently handles both new GCS-backed
    documents and every document stored locally before this change (a
    bare path, which never starts with "gs://")."""
    if storage_uri.startswith(_GCS_PREFIX):
        return await asyncio.to_thread(_download_from_gcs, storage_uri)
    return await asyncio.to_thread(Path(storage_uri).read_bytes)


async def delete(storage_uri: str) -> None:
    """Best-effort delete — never raises. Not currently called anywhere
    (no document-deletion flow exists yet), provided so one doesn't have
    to reintroduce this storage-dispatch logic when that flow is built."""
    try:
        if storage_uri.startswith(_GCS_PREFIX):
            await asyncio.to_thread(_delete_from_gcs, storage_uri)
        else:
            await asyncio.to_thread(Path(storage_uri).unlink, True)  # missing_ok=True
    except Exception:
        logger.warning("[PatientDocumentStorage] failed to delete %s", storage_uri, exc_info=True)
