"""
Evidence Source Registry (Phase 3)

Operational configuration for what gets ingested into which patient-facing
Qdrant collection, replacing a trusted-source allowlist that would
otherwise live only in a prompt or be hardcoded across retrieval code —
see the CLAUDE.md-adjacent architecture review, section 20.

DEFAULT_SOURCES below is metadata only: names, domains, authority
classes, which collection each belongs to. It is not ingested content.
Calling seed_default_sources() registers these rows so an ingestion job
has somewhere to point; it inserts zero documents and does not touch
Qdrant. Populating evidence_documents (and the actual Qdrant points) is
evidence_ingestion_service.ingest_document(), run per-document against
text an admin/ingestion process has already fetched — this module does
not crawl the web itself.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from src.api.services.patient_db import get_patient_db
from src.core.config import settings


class SourceDomainMismatch(ValueError):
    """Raised by enforce_domain() when a URL's hostname isn't the
    registered source's domain or a subdomain of it. A ValueError
    subclass (not a bare Exception) so existing `except ValueError`
    callers around ingest_url()/ingest_document() keep working without
    needing to know a new exception type exists."""


def hostname_of(url: str) -> str:
    """Lowercased hostname, stripped of port/userinfo/trailing dot.
    Uses urlsplit rather than string slicing so this is correct for
    every URL shape (IPv6 literals, userinfo, explicit ports), not just
    the common case."""
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def hostname_matches_domain(hostname: str, domain: str) -> bool:
    """True when hostname IS domain, or is a subdomain of it —
    'www.cancer.gov' and 'faq.cancer.gov' both match domain
    'cancer.gov'. Exact-suffix-with-a-leading-dot only:
    'notcancer.gov' or 'cancer.gov.evil.example' must never match
    'cancer.gov' — a naive `.endswith(domain)` (no dot) would wrongly
    accept both."""
    hostname = (hostname or "").lower().rstrip(".")
    domain = (domain or "").lower().rstrip(".")
    if not hostname or not domain:
        return False
    return hostname == domain or hostname.endswith("." + domain)


def enforce_domain(source: Dict[str, Any], url: str, stage: str = "url") -> None:
    """Raise SourceDomainMismatch unless url's hostname is the source's
    registered domain (or a subdomain of it). This is the allowlist
    actually being an allowlist: evidence_sources.domain exists so a
    source_key like "nci" carries NCI's authority_class, but nothing
    previously stopped `ingest_url("nci", url)` from being called with a
    URL on a completely different host and having that content inherit
    NCI's trust rating. Callers are expected to call this BEFORE fetching
    a requested URL (so an off-allowlist host is never even fetched, not
    just never trusted) and AGAIN on the final, post-redirect URL after
    fetching (a redirect can leave the approved domain even when the
    requested URL didn't) — see evidence_ingestion_service.ingest_url().

    A source registered with no domain (domain is falsy) is intentionally
    NOT enforced here — every DEFAULT_SOURCES entry sets one, so an
    operator registering a custom source without a domain has explicitly
    opted out of this check, not fallen through it by accident."""
    domain = source.get("domain")
    if not domain:
        return
    host = hostname_of(url)
    if not hostname_matches_domain(host, domain):
        raise SourceDomainMismatch(
            f"Refusing to ingest {stage} URL {url!r} (hostname {host!r}) under "
            f"source_key={source.get('source_key')!r}: does not match its registered "
            f"domain {domain!r} (or a subdomain of it)."
        )


# Metadata only — see module docstring. authority_class follows the A/B/C
# convention from the architecture review (A = professional societies /
# government health agencies, B = condition-specific nonprofits, C =
# everything else patient_facing=True material might still come from).
DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {"source_key": "nci", "name": "National Cancer Institute", "domain": "cancer.gov",
     "authority_class": "A", "authority_score": 1.0,
     "source_type": "patient_education",
     "collection_setting": "qdrant_patient_education_collection",
     # Was missing treatment_explainer/medication_explainer even though
     # scripts/ingest_nci_cancer_types.py ingests exactly this ("Breast
     # Cancer Treatment (PDQ)", etc. -- literally titled "Treatment") and
     # scripts/ingest_nci_supportive_care.py ingests medication-adjacent
     # side-effect/interaction content ("Cancer Therapy Interactions With
     # Foods and Dietary Supplements") under this same source_key. NCI is
     # also the only source those two scripts populate today, so the
     # omission silently zeroed out patient-education retrieval for
     # treatment/medication questions -- the most common shape of patient
     # question -- for every single patient (see source_governance.py's
     # allowed_intents enforcement): every candidate got dropped at
     # retrieval time regardless of how well it matched, and the answer
     # fell back to the clinician literature corpus instead. Matches
     # pipeline/patient_education/sources.py's SOURCE["nci"].required_buckets,
     # which has always listed "treatment" as a bucket this source covers.
     "allowed_intents": [
         "nutrition", "symptom_management", "diagnosis_explainer",
         "treatment_explainer", "medication_explainer", "general",
     ]},
    {"source_key": "cancer_net", "name": "Cancer.Net (ASCO)", "domain": "cancer.net",
     "authority_class": "A", "authority_score": 0.95,
     "source_type": "patient_education",
     "collection_setting": "qdrant_patient_education_collection",
     "allowed_intents": ["diagnosis_explainer", "treatment_explainer", "general"]},
    {"source_key": "acs", "name": "American Cancer Society", "domain": "cancer.org",
     "authority_class": "A", "authority_score": 0.95,
     "source_type": "patient_education",
     "collection_setting": "qdrant_patient_education_collection",
     # Same fix as "nci" above, for the same reason: ACS's own
     # cancer.org/cancer/treatment-types.html is seeded explicitly in
     # pipeline/patient_education/sources.py, and "treatment" is in that
     # source's required_buckets there -- treatment_explainer was simply
     # missing here.
     "allowed_intents": [
         "diagnosis_explainer", "nutrition", "symptom_management",
         "treatment_explainer", "general",
     ]},
    {"source_key": "nccn_patients", "name": "NCCN Guidelines for Patients", "domain": "nccn.org",
     "authority_class": "A", "authority_score": 0.95,
     "source_type": "patient_education",
     "collection_setting": "qdrant_patient_education_collection",
     "allowed_intents": ["treatment_explainer", "general"]},
    {"source_key": "ons", "name": "Oncology Nursing Society", "domain": "ons.org",
     "authority_class": "A", "authority_score": 0.9,
     "source_type": "patient_education",
     "collection_setting": "qdrant_patient_education_collection",
     "allowed_intents": ["symptom_management", "nutrition", "treatment_explainer"]},
    {"source_key": "medlineplus", "name": "MedlinePlus", "domain": "medlineplus.gov",
     "authority_class": "A", "authority_score": 0.9,
     "source_type": "medication_knowledge",
     "collection_setting": "qdrant_medication_collection",
     "allowed_intents": ["medication_explainer", "general"]},
    {"source_key": "dailymed", "name": "DailyMed", "domain": "dailymed.nlm.nih.gov",
     "authority_class": "A", "authority_score": 0.95,
     "source_type": "medication_knowledge",
     "collection_setting": "qdrant_medication_collection",
     "allowed_intents": ["medication_explainer"]},
    {"source_key": "fda", "name": "FDA", "domain": "fda.gov",
     "authority_class": "A", "authority_score": 1.0,
     "source_type": "medication_knowledge",
     "collection_setting": "qdrant_medication_collection",
     "allowed_intents": ["medication_explainer"]},
    {"source_key": "chemocare", "name": "Chemocare", "domain": "chemocare.com",
     "authority_class": "B", "authority_score": 0.75,
     "source_type": "medication_knowledge",
     "collection_setting": "qdrant_medication_collection",
     "allowed_intents": ["medication_explainer", "symptom_management"]},
    {"source_key": "mascc", "name": "Multinational Association of Supportive Care in Cancer",
     "domain": "mascc.org", "authority_class": "A", "authority_score": 0.85,
     "source_type": "clinical_guideline",
     "collection_setting": "qdrant_guideline_collection",
     "allowed_intents": ["symptom_management", "treatment_explainer"]},
    {"source_key": "asco_guidelines", "name": "ASCO Clinical Practice Guidelines",
     "domain": "asco.org", "authority_class": "A", "authority_score": 0.95,
     "source_type": "clinical_guideline",
     "collection_setting": "qdrant_guideline_collection",
     "allowed_intents": ["treatment_explainer"]},
]


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k in ("allowed_intents", "applicability", "constraints"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, ValueError):
                pass
    for k, v in list(d.items()):
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
        elif hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


class SourceRegistry:
    async def register_source(
        self,
        source_key: str,
        name: str,
        domain: Optional[str] = None,
        authority_class: str = "B",
        authority_score: float = 0.5,
        source_type: str = "patient_education",
        allowed_intents: Optional[List[str]] = None,
        patient_facing: bool = True,
        ingestion_method: str = "manual",
        license_status: str = "unknown",
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO evidence_sources
                    (id, source_key, name, domain, authority_class, authority_score,
                     source_type, allowed_intents, patient_facing, ingestion_method,
                     license_status)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10,$11)
                ON CONFLICT (source_key) DO UPDATE SET
                    name = EXCLUDED.name, domain = EXCLUDED.domain,
                    authority_class = EXCLUDED.authority_class,
                    authority_score = EXCLUDED.authority_score,
                    source_type = EXCLUDED.source_type,
                    allowed_intents = EXCLUDED.allowed_intents,
                    patient_facing = EXCLUDED.patient_facing,
                    license_status = EXCLUDED.license_status
                RETURNING *
                """,
                str(uuid.uuid4()), source_key, name, domain, authority_class,
                authority_score, source_type, json.dumps(allowed_intents or []),
                patient_facing, ingestion_method, license_status,
            )
        return _row_to_dict(row)

    async def seed_default_sources(self) -> List[Dict[str, Any]]:
        out = []
        for s in DEFAULT_SOURCES:
            out.append(await self.register_source(
                source_key=s["source_key"], name=s["name"], domain=s["domain"],
                authority_class=s["authority_class"], authority_score=s["authority_score"],
                source_type=s["source_type"], allowed_intents=s["allowed_intents"],
            ))
        return out

    async def list_sources(self, active_only: bool = True) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        query = "SELECT * FROM evidence_sources"
        if active_only:
            query += " WHERE active = true"
        query += " ORDER BY authority_score DESC"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
        return [_row_to_dict(r) for r in rows]

    async def get_source(self, source_key: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM evidence_sources WHERE source_key = $1", source_key
            )
        return _row_to_dict(row) if row else None

    async def register_document(
        self,
        source_key: str,
        doc_id: str,
        title: str,
        url: Optional[str],
        qdrant_collection: str,
        applicability: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source = await self.get_source(source_key)
        if not source:
            raise ValueError(f"Unknown evidence source: {source_key}")

        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO evidence_documents
                    (id, source_id, doc_id, title, url, qdrant_collection,
                     applicability, constraints, last_ingested_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb, now())
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, url = EXCLUDED.url,
                    applicability = EXCLUDED.applicability,
                    constraints = EXCLUDED.constraints,
                    last_ingested_at = now()
                RETURNING *
                """,
                str(uuid.uuid4()), source["id"], doc_id, title, url, qdrant_collection,
                json.dumps(applicability or {}), json.dumps(constraints or {}),
            )
        return _row_to_dict(row)

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT ed.*, es.source_key, es.authority_class, es.authority_score,
                       es.name AS source_name
                  FROM evidence_documents ed
                  JOIN evidence_sources es ON es.id = ed.source_id
                 WHERE ed.doc_id = $1
                """,
                doc_id,
            )
        return _row_to_dict(row) if row else None

    def collection_for(self, source: Dict[str, Any]) -> str:
        """Resolve a source's target Qdrant collection from its
        source_type, falling back to the patient-education collection."""
        mapping = {
            "patient_education": settings.qdrant_patient_education_collection,
            "medication_knowledge": settings.qdrant_medication_collection,
            "clinical_guideline": settings.qdrant_guideline_collection,
        }
        return mapping.get(source.get("source_type"), settings.qdrant_patient_education_collection)


_registry: Optional[SourceRegistry] = None


def get_source_registry() -> SourceRegistry:
    global _registry
    if _registry is None:
        _registry = SourceRegistry()
    return _registry
