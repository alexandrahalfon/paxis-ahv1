"""
Tests for patient_document_storage.py (2026-08-12 beta audit item 3):
patient documents move off local-disk-only storage (not viable on Cloud
Run -- ephemeral instances, no shared filesystem) onto GCS when
settings.gcp_patient_documents_bucket is configured, falling back to
local disk (the exact pre-existing bare-path behavior) when it isn't.

These tests never construct a real google.cloud.storage.Client -- that
import is broken in this sandbox (an unrelated cryptography/cffi
environment issue) and, more importantly, isn't what this module's own
logic is responsible for. Every test monkeypatches at the
_upload_to_gcs/_download_from_gcs/_delete_from_gcs seams instead, which
is exactly where store()/read()/delete() hand off to the GCS SDK.
"""

from __future__ import annotations

import pytest

from src.api.services.patient import patient_document_storage as pds


@pytest.fixture(autouse=True)
def _local_storage_dir(tmp_path, monkeypatch):
    """Redirects local-disk fallback storage into a pytest tmp dir so
    tests never write into the real repo working directory."""
    monkeypatch.setattr(pds, "_LOCAL_STORAGE_DIR", tmp_path / "patient_documents")
    yield


@pytest.fixture(autouse=True)
def _no_gcs_bucket_by_default(monkeypatch):
    """Most tests want the local fallback; GCS-path tests opt in
    explicitly via _gcs_bucket_configured below."""
    monkeypatch.setattr(pds.settings, "gcp_patient_documents_bucket", "")
    yield


@pytest.fixture(autouse=True)
def _gcs_not_required_by_default(monkeypatch):
    """Sprint B item 10's enforcement flag -- most tests want the
    pre-existing local-fallback behavior; the enforcement tests below
    opt in explicitly."""
    monkeypatch.setattr(pds.settings, "require_gcs_for_patient_documents", False)
    yield


def _gcs_bucket_configured(monkeypatch, bucket_name="patient-phi-bucket"):
    monkeypatch.setattr(pds.settings, "gcp_patient_documents_bucket", bucket_name)


class TestIsGcsConfigured:
    def test_false_when_bucket_unset(self):
        assert pds.is_gcs_configured() is False

    def test_true_when_bucket_set(self, monkeypatch):
        _gcs_bucket_configured(monkeypatch)
        assert pds.is_gcs_configured() is True


class TestStoreLocalFallback:
    @pytest.mark.asyncio
    async def test_writes_file_and_returns_bare_local_path(self):
        uri = await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")
        assert not uri.startswith("gs://")
        assert uri.endswith("doc-1_labs.pdf")

        from pathlib import Path
        assert Path(uri).read_bytes() == b"pdf bytes"

    @pytest.mark.asyncio
    async def test_sanitizes_filename_to_its_basename(self):
        uri = await pds.store("profile-1", "doc-1", "../../etc/passwd", b"x")
        from pathlib import Path
        assert Path(uri).name == "doc-1_passwd"

    @pytest.mark.asyncio
    async def test_separate_profiles_do_not_collide(self):
        uri_a = await pds.store("profile-a", "doc-1", "report.pdf", b"A")
        uri_b = await pds.store("profile-b", "doc-1", "report.pdf", b"B")
        assert uri_a != uri_b
        from pathlib import Path
        assert Path(uri_a).read_bytes() == b"A"
        assert Path(uri_b).read_bytes() == b"B"


class TestStoreGcs:
    @pytest.mark.asyncio
    async def test_uploads_via_gcs_and_returns_gs_uri(self, monkeypatch):
        _gcs_bucket_configured(monkeypatch, "patient-phi-bucket")
        calls = []

        def fake_upload(key, content):
            calls.append((key, content))

        monkeypatch.setattr(pds, "_upload_to_gcs", fake_upload)

        uri = await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")

        assert uri == "gs://patient-phi-bucket/patient_documents/profile-1/doc-1_labs.pdf"
        assert calls == [("patient_documents/profile-1/doc-1_labs.pdf", b"pdf bytes")]

    @pytest.mark.asyncio
    async def test_does_not_touch_local_disk_when_gcs_configured(self, monkeypatch, tmp_path):
        _gcs_bucket_configured(monkeypatch)
        monkeypatch.setattr(pds, "_upload_to_gcs", lambda key, content: None)

        await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")

        assert not (tmp_path / "patient_documents").exists() or not any(
            (tmp_path / "patient_documents").rglob("*")
        )


class TestReadDispatchesOnUriShape:
    @pytest.mark.asyncio
    async def test_reads_local_bare_path(self):
        uri = await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")
        content = await pds.read(uri)
        assert content == b"pdf bytes"

    @pytest.mark.asyncio
    async def test_reads_pre_existing_bare_path_document_unchanged(self, tmp_path):
        """Backward compatibility: a document stored under the OLD
        (pre-2026-08-12) local-only implementation is a bare path with no
        scheme -- read() must keep serving it with no migration."""
        old_style_path = tmp_path / "some_old_local_file.pdf"
        old_style_path.write_bytes(b"legacy content")

        content = await pds.read(str(old_style_path))
        assert content == b"legacy content"

    @pytest.mark.asyncio
    async def test_reads_gcs_uri_via_download(self, monkeypatch):
        monkeypatch.setattr(pds, "_download_from_gcs", lambda uri: b"gcs bytes")
        content = await pds.read("gs://patient-phi-bucket/patient_documents/p1/d1_labs.pdf")
        assert content == b"gcs bytes"

    @pytest.mark.asyncio
    async def test_gcs_read_passes_the_full_uri_through(self, monkeypatch):
        captured = {}

        def fake_download(uri):
            captured["uri"] = uri
            return b"x"

        monkeypatch.setattr(pds, "_download_from_gcs", fake_download)
        await pds.read("gs://patient-phi-bucket/patient_documents/p1/d1_labs.pdf")
        assert captured["uri"] == "gs://patient-phi-bucket/patient_documents/p1/d1_labs.pdf"


class TestDeleteIsBestEffort:
    @pytest.mark.asyncio
    async def test_deletes_local_file(self, tmp_path):
        f = tmp_path / "to_delete.pdf"
        f.write_bytes(b"x")
        await pds.delete(str(f))
        assert not f.exists()

    @pytest.mark.asyncio
    async def test_missing_local_file_does_not_raise(self, tmp_path):
        await pds.delete(str(tmp_path / "never_existed.pdf"))

    @pytest.mark.asyncio
    async def test_gcs_delete_dispatches_correctly(self, monkeypatch):
        calls = []
        monkeypatch.setattr(pds, "_delete_from_gcs", lambda uri: calls.append(uri))
        await pds.delete("gs://patient-phi-bucket/patient_documents/p1/d1_labs.pdf")
        assert calls == ["gs://patient-phi-bucket/patient_documents/p1/d1_labs.pdf"]

    @pytest.mark.asyncio
    async def test_gcs_delete_failure_is_swallowed(self, monkeypatch):
        def failing_delete(uri):
            raise RuntimeError("bucket unreachable")

        monkeypatch.setattr(pds, "_delete_from_gcs", failing_delete)
        # Must not raise.
        await pds.delete("gs://patient-phi-bucket/patient_documents/p1/d1_labs.pdf")


class TestProductionGcsEnforcement:
    """2026-08-12 convergence Sprint B item 10:
    settings.require_gcs_for_patient_documents refuses the silent
    local-disk fallback that's exactly the failure mode this module's
    own docstring warns about on Cloud Run."""

    @pytest.mark.asyncio
    async def test_raises_when_required_and_bucket_unset(self, monkeypatch):
        monkeypatch.setattr(pds.settings, "require_gcs_for_patient_documents", True)

        with pytest.raises(pds.PatientDocumentStorageMisconfigured):
            await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")

    @pytest.mark.asyncio
    async def test_does_not_write_to_local_disk_when_it_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pds.settings, "require_gcs_for_patient_documents", True)

        with pytest.raises(pds.PatientDocumentStorageMisconfigured):
            await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")

        assert not (tmp_path / "patient_documents").exists() or not any(
            (tmp_path / "patient_documents").rglob("*")
        )

    @pytest.mark.asyncio
    async def test_does_not_raise_when_bucket_is_configured(self, monkeypatch):
        monkeypatch.setattr(pds.settings, "require_gcs_for_patient_documents", True)
        _gcs_bucket_configured(monkeypatch)
        monkeypatch.setattr(pds, "_upload_to_gcs", lambda key, content: None)

        # Must not raise -- the bucket IS configured, so there's nothing
        # to enforce against.
        uri = await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")
        assert uri.startswith("gs://")

    @pytest.mark.asyncio
    async def test_local_fallback_still_works_when_not_required(self):
        """Default behavior (flag off) is completely unchanged --
        confirms this feature is additive, not a regression for
        dev/CI."""
        uri = await pds.store("profile-1", "doc-1", "labs.pdf", b"pdf bytes")
        assert not uri.startswith("gs://")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
