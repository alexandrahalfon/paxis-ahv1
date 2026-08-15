"""
Source Fetcher (evidence ingestion front-half)

Fetches one URL over HTTPS with a real, identifiable User-Agent, a size
cap, and a timeout. Returns raw bytes + detected content-type so
content_extractor.py can pick the right parser (HTML vs. PDF).

This is an ingestion-time-only concern — it runs from
scripts/ingest_evidence_source.py (an operator action against the
source_registry allowlist), never from a patient-facing request path, so
a slow or failed fetch here can never affect a chat response.

Note on this specific development sandbox: outbound HTTPS from Bash/
Python here is blocked by the session's egress proxy policy for
non-Anthropic domains (verified — see the commit this shipped with).
That is a property of this sandbox, not of this code or of a real
deployment; a production environment with normal internet egress runs
this exactly as written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = (
    "PaxisEvidenceIngestion/1.0 (+https://paxis.health; "
    "purpose: patient-education content ingestion from an approved source allowlist)"
)
MAX_BYTES = 15 * 1024 * 1024  # 15MB — a patient-education page or drug-label PDF is never legitimately larger
DEFAULT_TIMEOUT = 30.0


class FetchError(Exception):
    """Raised for any fetch failure — HTTP error, timeout, or size cap
    exceeded. Callers (connectors) are expected to let this propagate up
    to the CLI runner, which logs and moves to the next URL rather than
    aborting a whole ingestion run over one bad fetch."""


@dataclass
class FetchResult:
    url: str
    final_url: str  # after redirects — what was actually retrieved
    status_code: int
    content_type: str  # lowercased, parameters stripped, e.g. "text/html"
    content: bytes
    fetched_at: str  # ISO 8601 UTC


def fetch_url(url: str, timeout: float = DEFAULT_TIMEOUT) -> FetchResult:
    if not url.lower().startswith("https://") and not url.lower().startswith("http://"):
        raise FetchError(f"Refusing to fetch non-http(s) URL: {url}")

    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        raise FetchError(f"Fetch failed for {url}: {e}") from e

    if resp.status_code >= 400:
        raise FetchError(f"{url} returned HTTP {resp.status_code}")

    content = resp.content
    if len(content) > MAX_BYTES:
        raise FetchError(
            f"{url} exceeded the {MAX_BYTES}-byte cap ({len(content)} bytes) — "
            "refusing to process; this cap exists to keep one bad URL from "
            "consuming an ingestion run's memory/time budget"
        )

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()

    return FetchResult(
        url=url,
        final_url=str(resp.url),
        status_code=resp.status_code,
        content_type=content_type,
        content=content,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
