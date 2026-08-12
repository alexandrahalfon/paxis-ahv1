"""
Tests for evidence-source domain allowlist enforcement (caught in review,
2026-08-12): the source registry stores a `domain` per source_key (e.g.
nci -> cancer.gov), but nothing previously checked a URL passed to
ingest_url()/ingest_document() actually belonged to that domain — a call
like ingest_url("nci", "https://attacker.example/page") would ingest
attacker.example's content carrying NCI's authority_class, and a redirect
could silently leave the approved domain even for a legitimately-started
fetch. See source_registry.enforce_domain()/hostname_matches_domain() and
evidence_ingestion_service.ingest_url()/ingest_document().
"""

from __future__ import annotations

import uuid

import pytest

from src.api.services.evidence.source_registry import (
    SourceDomainMismatch, enforce_domain, hostname_matches_domain, hostname_of,
)


def _source(domain="cancer.gov", source_key="nci"):
    return {"id": str(uuid.uuid4()), "source_key": source_key, "domain": domain}


class TestHostnameOf:
    def test_extracts_lowercased_hostname(self):
        assert hostname_of("https://WWW.Cancer.gov/page") == "www.cancer.gov"

    def test_strips_port_and_userinfo(self):
        assert hostname_of("https://user:pass@cancer.gov:443/page") == "cancer.gov"

    def test_strips_trailing_dot(self):
        assert hostname_of("https://cancer.gov./page") == "cancer.gov"

    def test_empty_for_unparseable_url(self):
        assert hostname_of("not a url") == ""


class TestHostnameMatchesDomain:
    def test_exact_match(self):
        assert hostname_matches_domain("cancer.gov", "cancer.gov") is True

    def test_subdomain_matches(self):
        assert hostname_matches_domain("www.cancer.gov", "cancer.gov") is True
        assert hostname_matches_domain("faq.cancer.gov", "cancer.gov") is True

    def test_case_insensitive(self):
        assert hostname_matches_domain("WWW.CANCER.GOV", "cancer.gov") is True

    def test_prefix_lookalike_does_not_match(self):
        # 'notcancer.gov' must never match 'cancer.gov' -- a naive
        # `.endswith("cancer.gov")` (no leading dot) would wrongly accept this.
        assert hostname_matches_domain("notcancer.gov", "cancer.gov") is False

    def test_suffix_lookalike_does_not_match(self):
        assert hostname_matches_domain("cancer.gov.evil.example", "cancer.gov") is False

    def test_unrelated_domain_does_not_match(self):
        assert hostname_matches_domain("attacker.example", "cancer.gov") is False

    def test_empty_hostname_or_domain_never_matches(self):
        assert hostname_matches_domain("", "cancer.gov") is False
        assert hostname_matches_domain("cancer.gov", "") is False


class TestEnforceDomain:
    def test_matching_url_does_not_raise(self):
        enforce_domain(_source(domain="cancer.gov"), "https://www.cancer.gov/page")

    def test_mismatched_url_raises_source_domain_mismatch(self):
        with pytest.raises(SourceDomainMismatch):
            enforce_domain(_source(domain="cancer.gov"), "https://attacker.example/page")

    def test_mismatch_error_names_source_key_and_domain(self):
        with pytest.raises(SourceDomainMismatch, match="nci"):
            enforce_domain(_source(domain="cancer.gov", source_key="nci"), "https://attacker.example/page")

    def test_source_domain_mismatch_is_a_value_error(self):
        # So existing `except ValueError` call sites around ingestion
        # keep catching this without changes.
        assert issubclass(SourceDomainMismatch, ValueError)

    def test_source_with_no_registered_domain_is_not_enforced(self):
        # Explicit opt-out, not a silent gap -- see enforce_domain()'s docstring.
        enforce_domain({"source_key": "custom", "domain": None}, "https://anywhere.example/page")
        enforce_domain({"source_key": "custom"}, "https://anywhere.example/page")


class TestIngestUrlEnforcesDomainBeforeAndAfterFetch:
    @pytest.mark.asyncio
    async def test_off_domain_requested_url_is_rejected_before_fetching(self, monkeypatch):
        from src.api.services.evidence.evidence_ingestion_service import EvidenceIngestionService
        import src.api.services.evidence.evidence_ingestion_service as service_module

        class FakeRegistry:
            async def get_source(self, source_key):
                return _source(domain="cancer.gov", source_key="nci")
            def collection_for(self, source):
                return "oncology_patient_education"

        # evidence_ingestion_service.py imports get_source_registry via a
        # top-level `from source_registry import get_source_registry`, so
        # it holds its own name bound at import time -- patching
        # source_registry.get_source_registry would not affect it. Patch
        # the name as it exists in THIS module's namespace instead.
        monkeypatch.setattr(service_module, "get_source_registry", lambda: FakeRegistry())

        def _fail_if_called(url, timeout=30.0):
            raise AssertionError("fetch_url must not run when the requested hostname is off-domain")
        monkeypatch.setattr("src.api.services.evidence.source_fetcher.fetch_url", _fail_if_called)

        service = EvidenceIngestionService()
        with pytest.raises(SourceDomainMismatch):
            await service.ingest_url("nci", "https://attacker.example/fake-nci-page")

    @pytest.mark.asyncio
    async def test_redirect_off_domain_is_rejected_after_fetching(self, monkeypatch):
        from src.api.services.evidence.evidence_ingestion_service import EvidenceIngestionService
        import src.api.services.evidence.evidence_ingestion_service as service_module
        from src.api.services.evidence import source_fetcher as source_fetcher_module

        class FakeRegistry:
            async def get_source(self, source_key):
                return _source(domain="cancer.gov", source_key="nci")
            def collection_for(self, source):
                return "oncology_patient_education"

        monkeypatch.setattr(service_module, "get_source_registry", lambda: FakeRegistry())

        class FakeFetchResult:
            url = "https://www.cancer.gov/redirecting-page"
            final_url = "https://attacker.example/hijacked"  # left the approved domain
            status_code = 200
            content_type = "text/html"
            content = b"<html><body><p>hijacked content</p></body></html>"
            fetched_at = "2026-08-12T00:00:00Z"

        def fake_fetch_url(url, timeout=30.0):
            return FakeFetchResult()
        monkeypatch.setattr(source_fetcher_module, "fetch_url", fake_fetch_url)

        service = EvidenceIngestionService()
        with pytest.raises(SourceDomainMismatch):
            await service.ingest_url("nci", "https://www.cancer.gov/redirecting-page")


class TestIngestDocumentEnforcesAssertedUrlDomain:
    @pytest.mark.asyncio
    async def test_off_domain_asserted_url_is_rejected(self, monkeypatch):
        from src.api.services.evidence.evidence_ingestion_service import EvidenceIngestionService
        import src.api.services.evidence.evidence_ingestion_service as service_module

        class FakeRegistry:
            async def get_source(self, source_key):
                return _source(domain="cancer.gov", source_key="nci")

        monkeypatch.setattr(service_module, "get_source_registry", lambda: FakeRegistry())

        service = EvidenceIngestionService()
        with pytest.raises(SourceDomainMismatch):
            await service.ingest_document(
                source_key="nci", doc_id="doc-1", title="Fake",
                raw_text="Some manually-provided text.",
                url="https://attacker.example/fake-nci-page",
            )

    @pytest.mark.asyncio
    async def test_no_url_skips_the_check_entirely(self, monkeypatch):
        """Isolates ingest_document()'s own branching (url truthy? call
        enforce_domain) from the rest of the pipeline: enforce_domain is
        patched to raise if called at all, and _ingest_extracted is
        stubbed to a no-op so this doesn't also need a full Postgres/
        Qdrant fake just to prove this one branch."""
        from src.api.services.evidence.evidence_ingestion_service import EvidenceIngestionService
        import src.api.services.evidence.evidence_ingestion_service as service_module

        class FakeRegistry:
            async def get_source(self, source_key):
                return _source(domain="cancer.gov", source_key="nci")

        monkeypatch.setattr(service_module, "get_source_registry", lambda: FakeRegistry())

        def _forbidden_enforce_domain(*args, **kwargs):
            raise AssertionError("enforce_domain must not be called when url is None")
        monkeypatch.setattr(service_module, "enforce_domain", _forbidden_enforce_domain)

        async def _stub_ingest_extracted(self, **kwargs):
            return {"stubbed": True}
        monkeypatch.setattr(EvidenceIngestionService, "_ingest_extracted", _stub_ingest_extracted)

        service = EvidenceIngestionService()
        result = await service.ingest_document(
            source_key="nci", doc_id="doc-1", title="Fake",
            raw_text="Some manually-provided text.",
        )
        assert result == {"stubbed": True}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
