"""
Tests for Mistral-hosted file cleanup after patient document OCR (caught
in review, 2026-08-12): PatientDocumentExtractor._ocr_pdf() uploaded a
patient's PDF to Mistral's file API for OCR but never deleted it
afterward -- unlike literature PDFs (where the same pattern in
CompleteDocumentProcessor is explicitly documented as low-stakes), a
patient's uploaded medical record is PHI and leaving a permanent copy on
a third-party file API is a real retention concern. The upload is now
deleted in a finally block regardless of whether OCR succeeds.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.api.services.patient.patient_document_extractor import PatientDocumentExtractor


class _FakeDocumentURLChunk:
    def __init__(self, document_url):
        self.document_url = document_url


@pytest.fixture(autouse=True)
def _stub_mistralai_models(monkeypatch):
    """_ocr_pdf() does `from mistralai.models import DocumentURLChunk`
    inline. The mistralai package actually installed in this sandbox
    (2.9.2) doesn't expose that module path at all -- an environment/
    dependency-version mismatch unrelated to the OCR-cleanup behavior
    these tests exercise, and not something to fix by chasing SDK
    versions here. Stub the exact import target instead so these tests
    isolate the cleanup logic from that mismatch."""
    fake_models = types.ModuleType("mistralai.models")
    fake_models.DocumentURLChunk = _FakeDocumentURLChunk
    monkeypatch.setitem(sys.modules, "mistralai.models", fake_models)


class _FakeUploaded:
    id = "file-abc123"


class _FakeSignedUrl:
    url = "https://mistral.example/signed/file-abc123"


class _FakePage:
    def __init__(self, markdown):
        self.markdown = markdown


class _FakeOcrResponse:
    def __init__(self, pages):
        self.pages = pages


class _FakeFilesApi:
    def __init__(self, upload_result=_FakeUploaded(), get_signed_url_error=None, delete_error=None):
        self.upload_result = upload_result
        self.get_signed_url_error = get_signed_url_error
        self.delete_error = delete_error
        self.upload_calls = []
        self.delete_calls = []

    def upload(self, file, purpose):
        self.upload_calls.append((file, purpose))
        return self.upload_result

    def get_signed_url(self, file_id):
        if self.get_signed_url_error:
            raise self.get_signed_url_error
        return _FakeSignedUrl()

    def delete(self, file_id):
        self.delete_calls.append(file_id)
        if self.delete_error:
            raise self.delete_error


class _FakeOcrApi:
    def __init__(self, response=None, error=None):
        self.response = response or _FakeOcrResponse([_FakePage("extracted text")])
        self.error = error

    def process(self, document, model):
        if self.error:
            raise self.error
        return self.response


class _FakeMistralClient:
    def __init__(self, files_api=None, ocr_api=None):
        self.files = files_api or _FakeFilesApi()
        self.ocr = ocr_api or _FakeOcrApi()


@pytest.fixture
def extractor(monkeypatch):
    ext = PatientDocumentExtractor()
    return ext


class TestUploadedFileIsDeletedAfterSuccessfulOcr:
    def test_delete_called_with_the_uploaded_file_id(self, extractor, monkeypatch, tmp_path):
        fake_client = _FakeMistralClient()
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        pdf_path = tmp_path / "record.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_content = pdf_path.read_bytes()

        text = extractor._ocr_pdf(pdf_content, "record.pdf")

        assert text == "extracted text"
        assert fake_client.files.delete_calls == ["file-abc123"]

    def test_upload_happens_exactly_once(self, extractor, monkeypatch, tmp_path):
        fake_client = _FakeMistralClient()
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        pdf_path = tmp_path / "record.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_content = pdf_path.read_bytes()

        extractor._ocr_pdf(pdf_content, "record.pdf")
        assert len(fake_client.files.upload_calls) == 1


class TestUploadedFileIsDeletedEvenWhenOcrFails:
    def test_delete_still_called_and_original_exception_propagates(self, extractor, monkeypatch, tmp_path):
        ocr_error = RuntimeError("OCR service unavailable")
        fake_client = _FakeMistralClient(ocr_api=_FakeOcrApi(error=ocr_error))
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        pdf_path = tmp_path / "record.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_content = pdf_path.read_bytes()

        with pytest.raises(RuntimeError, match="OCR service unavailable"):
            extractor._ocr_pdf(pdf_content, "record.pdf")

        assert fake_client.files.delete_calls == ["file-abc123"]

    def test_delete_still_called_when_get_signed_url_fails(self, extractor, monkeypatch, tmp_path):
        fake_client = _FakeMistralClient(
            files_api=_FakeFilesApi(get_signed_url_error=RuntimeError("signed url failed"))
        )
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        pdf_path = tmp_path / "record.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_content = pdf_path.read_bytes()

        with pytest.raises(RuntimeError, match="signed url failed"):
            extractor._ocr_pdf(pdf_content, "record.pdf")

        assert fake_client.files.delete_calls == ["file-abc123"]


class TestDeleteFailureIsNonFatal:
    def test_delete_error_does_not_mask_a_successful_ocr_result(self, extractor, monkeypatch, tmp_path):
        fake_client = _FakeMistralClient(
            files_api=_FakeFilesApi(delete_error=RuntimeError("delete endpoint down"))
        )
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        pdf_path = tmp_path / "record.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_content = pdf_path.read_bytes()

        # Must not raise -- OCR still succeeded, cleanup failure is
        # logged (see the module) but never surfaced to the caller.
        text = extractor._ocr_pdf(pdf_content, "record.pdf")
        assert text == "extracted text"

    def test_delete_error_does_not_mask_the_original_ocr_exception(self, extractor, monkeypatch, tmp_path):
        ocr_error = RuntimeError("OCR service unavailable")
        fake_client = _FakeMistralClient(
            ocr_api=_FakeOcrApi(error=ocr_error),
            files_api=_FakeFilesApi(delete_error=RuntimeError("delete endpoint down")),
        )
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        pdf_path = tmp_path / "record.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        pdf_content = pdf_path.read_bytes()

        # The ORIGINAL OCR failure must be what propagates, not the
        # secondary cleanup failure.
        with pytest.raises(RuntimeError, match="OCR service unavailable"):
            extractor._ocr_pdf(pdf_content, "record.pdf")


class TestExtractTextReadsThroughStorageModule:
    """extract_text() (2026-08-12, alongside patient_document_storage.py)
    fetches bytes via that module's read() -- which transparently
    handles both a GCS uri and a locally-stored (pre-existing) bare
    path -- rather than opening object_storage_uri as a local path
    directly."""

    @pytest.mark.asyncio
    async def test_dispatches_to_ocr_pdf_for_a_pdf(self, extractor, monkeypatch):
        fake_client = _FakeMistralClient()
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        from src.api.services.patient import patient_document_storage
        monkeypatch.setattr(patient_document_storage, "read", _async_return(b"pdf bytes"))

        text = await extractor.extract_text("gs://bucket/patient_documents/p1/d1_labs.pdf")
        assert text == "extracted text"
        # The uploaded filename must be the last path segment, not the
        # full URI.
        uploaded_file = fake_client.files.upload_calls[0][0]
        assert uploaded_file["file_name"] == "d1_labs.pdf"
        assert uploaded_file["content"] == b"pdf bytes"

    @pytest.mark.asyncio
    async def test_dispatches_to_ocr_image_for_a_photo(self, extractor, monkeypatch):
        class _FakeVisionResponse:
            choices = [type("C", (), {"message": type("M", (), {"content": "transcribed text"})()})]

        class _FakeChat:
            def complete(self, model, messages):
                return _FakeVisionResponse()

        class _FakeVisionClient:
            files = _FakeFilesApi()
            chat = _FakeChat()

        monkeypatch.setattr(extractor, "_mistral_client", lambda: _FakeVisionClient())

        from src.api.services.patient import patient_document_storage
        monkeypatch.setattr(patient_document_storage, "read", _async_return(b"\xff\xd8\xff image bytes"))

        text = await extractor.extract_text(
            "gs://bucket/patient_documents/p1/d1_photo.jpg", content_type="image/jpeg",
        )
        assert text == "transcribed text"

    @pytest.mark.asyncio
    async def test_reads_via_the_pre_existing_bare_local_path_uri(self, extractor, monkeypatch, tmp_path):
        """A document stored before the GCS migration has a bare local
        path in object_storage_uri -- extract_text() must keep working
        for it with no data migration."""
        fake_client = _FakeMistralClient()
        monkeypatch.setattr(extractor, "_mistral_client", lambda: fake_client)

        local_file = tmp_path / "d1_old_upload.pdf"
        local_file.write_bytes(b"legacy pdf content")

        text = await extractor.extract_text(str(local_file))
        assert text == "extracted text"
        uploaded_file = fake_client.files.upload_calls[0][0]
        assert uploaded_file["content"] == b"legacy pdf content"


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
