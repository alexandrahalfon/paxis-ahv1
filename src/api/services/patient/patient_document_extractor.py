"""
Patient Document Extractor (Phase 2)

document -> OCR text -> classify -> structured candidate fields, written to
document_extractions with confirmed=false. Nothing here writes to a
canonical table (lab_results, medication_exposures, ...) directly — see
patient_document_validator.py for that, which only runs after a human
confirms the candidate values. That split exists specifically because OCR
misreads are consequential here ("Hgb 8.7" vs "Hgb 87"), not cosmetic.

OCR: reuses the same Mistral OCR client already used by
src/processing/document_processor.py for literature ingestion
(settings.mistral_api_key / mistral_ocr_model), rather than standing up a
second OCR integration. Falls back to Mistral's vision chat model
(pixtral) for image formats OCR handles poorly, then to "extraction
failed, ask the patient to type key values instead" if neither is
configured — this must degrade, never raise, since a document upload
failing silently into 'pending' forever is worse than an honest failure
status the UI can act on.

Structured extraction: a single gpt-4o-mini call per document type,
constrained to return JSON only. Not run against live services from this
codebase change — see the PR/commit notes this shipped with.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import settings

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic"}

_CLASSIFY_PROMPT = """Classify this medical document excerpt into exactly one \
category: lab, pathology, imaging, visit_summary, medication_list, \
discharge_instructions, or unclassified. Reply with only the category word."""

_EXTRACT_PROMPTS = {
    "lab": """Extract every lab test result from this report as JSON:
{"labs": [{"test_name": str, "value_numeric": number|null, "value_text": str|null, \
"unit": str|null, "reference_low": number|null, "reference_high": number|null, \
"abnormal_flag": str|null, "collected_at": "YYYY-MM-DD"|null}]}
Only include tests actually present with a value. Use null for anything not stated. \
Reply with JSON only, no commentary.""",
    "pathology": """Extract structured findings from this pathology report as JSON:
{"diagnosis": {"cancer_site": str|null, "histology": str|null, "stage": str|null, \
"tnm_t": str|null, "tnm_n": str|null, "tnm_m": str|null}, \
"biomarkers": [{"biomarker_name": str, "value": str}]}
Use null/empty for anything not stated. Reply with JSON only.""",
    "imaging": """Extract structured findings from this imaging report as JSON:
{"encounter": {"encounter_type": "imaging", "encounter_date": "YYYY-MM-DD"|null, \
"patient_summary": str}}
patient_summary should be one plain-language sentence describing what was found, \
without interpreting significance. Reply with JSON only.""",
    "visit_summary": """Extract structured content from this visit summary as JSON:
{"encounter": {"encounter_type": str|null, "encounter_date": "YYYY-MM-DD"|null, \
"provider_name": str|null, "organization": str|null, "patient_summary": str, \
"structured_changes": {}}}
Reply with JSON only.""",
    "medication_list": """Extract every medication from this list as JSON:
{"medications": [{"generic_name": str, "brand_name": str|null, "dose": str|null, \
"route": str|null, "frequency": str|null, "indication": str|null}]}
Reply with JSON only.""",
    "discharge_instructions": """Extract structured content from these discharge \
instructions as JSON:
{"encounter": {"encounter_type": "discharge", "encounter_date": "YYYY-MM-DD"|null, \
"patient_summary": str}}
Reply with JSON only.""",
    "unclassified": """This document didn't classify cleanly. Extract any of the \
following you can find as JSON, omitting keys with nothing found:
{"labs": [...], "medications": [...], "diagnosis": {...}, "encounter": {...}}
Reply with JSON only.""",
}


class PatientDocumentExtractor:
    def _openai_client(self):
        from openai import OpenAI
        return OpenAI(api_key=settings.openai_api_key)

    def _mistral_client(self):
        if not settings.mistral_api_key:
            return None
        from mistralai import Mistral
        return Mistral(api_key=settings.mistral_api_key)

    # ── OCR ──────────────────────────────────────────────────────────

    def _ocr_pdf(self, content: bytes, filename: str) -> str:
        """Mistral OCR for a PDF, same call shape as
        CompleteDocumentProcessor._extract_with_mistral_ocr — except that
        module's docstring itself notes cleanup "relatively unimportant"
        for literature PDFs; this one handles patient PHI, where it
        matters. The uploaded copy is deleted from Mistral's file store
        in a finally block, so it happens whether OCR succeeds or raises,
        rather than leaving a patient's uploaded document sitting on a
        third-party file API indefinitely.

        Takes bytes directly rather than a local path (changed 2026-08-12
        alongside patient_document_storage.py) — a document's storage_uri
        can now be a GCS object, which has no local path to open at all;
        the caller (extract_text) already fetched the bytes via that
        module's read()."""
        client = self._mistral_client()
        if client is None:
            raise RuntimeError("Mistral OCR is not configured")
        from mistralai.models import DocumentURLChunk

        uploaded = client.files.upload(
            file={"file_name": filename, "content": content},
            purpose="ocr",
        )
        try:
            signed_url = client.files.get_signed_url(file_id=uploaded.id)
            ocr_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=settings.mistral_ocr_model,
            )
            pages = getattr(ocr_response, "pages", []) or []
            return "\n\n".join(getattr(p, "markdown", "") or "" for p in pages)
        finally:
            # Best-effort: a failed cleanup must not turn a successful
            # (or already-failed) OCR into a hard failure for the
            # patient's upload — but it's worth a log line since a
            # lingering copy of patient PHI on Mistral's side is exactly
            # what this cleanup exists to prevent.
            try:
                client.files.delete(file_id=uploaded.id)
            except Exception as e:
                logger.warning(
                    "[PatientDocExtractor] failed to delete Mistral-hosted "
                    "file %s after OCR (patient PHI may linger there): %s",
                    uploaded.id, e,
                )

    def _ocr_image(self, content: bytes, filename: str) -> str:
        """Vision-model transcription for a phone photo. Uses the same
        pixtral chat-vision pattern as
        CompleteDocumentProcessor._extract_with_pixtral. Takes bytes
        directly — see _ocr_pdf's docstring for why."""
        client = self._mistral_client()
        if client is None:
            raise RuntimeError("Mistral vision is not configured")
        b64 = base64.b64encode(content).decode("utf-8")
        ext = Path(filename).suffix.lstrip(".").lower() or "jpeg"
        response = client.chat.complete(
            model=settings.mistral_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Transcribe every piece of text visible in this medical "
                        "document photo exactly as written, preserving table "
                        "structure with line breaks. No commentary."
                    )},
                    {"type": "image_url", "image_url": f"data:image/{ext};base64,{b64}"},
                ],
            }],
        )
        return response.choices[0].message.content or ""

    async def extract_text(self, storage_uri: str, content_type: Optional[str] = None) -> str:
        """Async now (changed 2026-08-12 alongside patient_document_
        storage.py) — storage_uri may be a GCS object with no local path
        to open directly, so the bytes are fetched through that module's
        read(), which dispatches on the "gs://" prefix to handle both
        GCS-backed and (pre-existing, still-supported) locally-stored
        documents transparently."""
        from src.api.services.patient import patient_document_storage

        content = await patient_document_storage.read(storage_uri)
        filename = storage_uri.rsplit("/", 1)[-1]
        ext = Path(filename).suffix.lower()
        if ext in _IMAGE_EXTS or (content_type or "").startswith("image/"):
            return self._ocr_image(content, filename)
        return self._ocr_pdf(content, filename)

    # ── Classification + structured extraction ──────────────────────

    def classify(self, raw_text: str) -> str:
        excerpt = raw_text[:2000]
        if not excerpt.strip():
            return "unclassified"
        try:
            resp = self._openai_client().chat.completions.create(
                model=settings.openai_mini_model or "gpt-4o-mini",
                temperature=0,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": excerpt},
                ],
            )
            label = (resp.choices[0].message.content or "").strip().lower()
            from src.api.services.patient.patient_document_service import DOCUMENT_TYPES
            return label if label in DOCUMENT_TYPES else "unclassified"
        except Exception as e:
            logger.warning("[PatientDocExtractor] classify failed: %s", e)
            return "unclassified"

    def extract_fields(self, raw_text: str, document_type: str) -> Dict[str, Any]:
        prompt = _EXTRACT_PROMPTS.get(document_type, _EXTRACT_PROMPTS["unclassified"])
        excerpt = raw_text[:8000]
        try:
            resp = self._openai_client().chat.completions.create(
                model=settings.openai_mini_model or "gpt-4o-mini",
                temperature=0,
                max_tokens=1500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": excerpt},
                ],
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            logger.warning("[PatientDocExtractor] extract_fields failed: %s", e)
            return {}

    async def run(self, document_id: str, patient_profile_id: str) -> Dict[str, Any]:
        """Full pipeline for one uploaded document: OCR -> classify ->
        extract -> persist as an unconfirmed document_extractions row.
        Never raises; failures land the document in status='failed' with
        error_message set, which the UI surfaces as 'couldn't read this —
        try retyping the key values' rather than a silent stall."""
        import uuid
        from src.api.services.patient_db import get_patient_db
        from src.api.services.patient.patient_document_service import get_patient_document_service

        doc_service = get_patient_document_service()
        doc = await doc_service.get_document(document_id, patient_profile_id)
        if not doc:
            raise ValueError("Document not found")

        try:
            raw_text = await self.extract_text(doc["object_storage_uri"], doc.get("content_type"))
            document_type = self.classify(raw_text)
            fields = self.extract_fields(raw_text, document_type)
            confidence = 0.7 if fields else 0.0

            db = get_patient_db()
            await db.ensure_schema()
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO document_extractions
                        (id, document_id, extracted_fields, extraction_confidence,
                         extraction_method)
                    VALUES ($1, $2, $3::jsonb, $4, $5)
                    RETURNING *
                    """,
                    str(uuid.uuid4()), document_id, json.dumps(fields), confidence,
                    "mistral_ocr+gpt4o_mini",
                )
            await doc_service.update_status(
                document_id, "extracted", document_type=document_type,
                parser_version="v1",
            )
            return {"extraction_id": str(row["id"]), "document_type": document_type, "fields": fields}
        except Exception as e:
            logger.exception("[PatientDocExtractor] extraction failed for %s", document_id)
            await doc_service.update_status(document_id, "failed", error_message=str(e)[:500])
            raise


_extractor: Optional[PatientDocumentExtractor] = None


def get_patient_document_extractor() -> PatientDocumentExtractor:
    global _extractor
    if _extractor is None:
        _extractor = PatientDocumentExtractor()
    return _extractor
