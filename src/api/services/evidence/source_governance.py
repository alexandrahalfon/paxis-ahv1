"""
Source Governance Enforcement (2026-08-12 convergence Sprint B item 8)

source_registry.py stores operational fields per source — active,
patient_facing, allowed_intents, license_status — but nothing at
RETRIEVAL time ever consulted them. enforce_domain() (added earlier this
sprint) checks a source's registered domain at INGESTION time, before a
document is ever fetched; that's necessary but not sufficient, because
once a chunk is in Qdrant it keeps being served on its ingestion-time
tags forever, even if the source's registry entry is later deactivated,
reclassified as not-patient-facing, or scoped away from an intent it was
originally allowed for. This module is the retrieval-time check that was
missing: a filter over already-retrieved candidates, re-checked against
the CURRENT registry state on every call.

What this enforces, because these are the fields the registry schema
(evidence_sources table / source_registry.py) actually stores today:
  - active           — a deactivated source's chunks are excluded outright.
  - patient_facing    — for audience="patient", a source not flagged
                        patient-facing is excluded (the review's own
                        framing: "require source.patient_facing unless
                        the corpus policy explicitly allows professional
                        evidence" — nothing in this codebase opts into
                        that exception yet, so the default here is a
                        hard exclude).
  - allowed_intents   — when a source specifies a non-empty allowed_intents
                        list, a candidate is excluded if the query's
                        intent isn't in it.
license_status is read but only informationally for now — nothing in
this codebase enforces a license/cache policy at retrieval time yet;
that's real future work this module doesn't pretend to have done.
collection_target and acquisition_mode, both mentioned in the
convergence plan's field list, aren't actual stored columns in this
schema (collection_setting on DEFAULT_SOURCES is a seed-time mapping,
not a per-row DB field) — nothing to enforce that doesn't already exist.

Fail-open by design, twice over:
  1. A candidate whose source_key isn't a REGISTERED source at all (the
     existing literature/exueed_kb_latest corpus predates source_registry
     adoption entirely — see architecture review item 13) passes through
     untouched. This is governance for what's registered, not a new
     requirement imposed on content that was never asked to register.
  2. A registry field that's simply absent/unset on a given source row
     (patient_facing not explicitly false, active not explicitly false,
     allowed_intents empty) is treated as "no restriction", matching
     register_source()'s own defaults (patient_facing=True,
     allowed_intents=[] meaning unrestricted).

Wired into multi_corpus_retriever.search() — see that module for the
call site — behind its own try/except, so a registry lookup failure
degrades to "don't filter" rather than breaking retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def filter_by_source_governance(
    candidates: List[Dict[str, Any]],
    *,
    audience: str,
    intent: Optional[str],
    sources_by_key: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pure filter, no I/O — takes the registry snapshot as a plain dict
    so this is testable without a database. See module docstring for the
    exact fields enforced and the fail-open rules."""
    out: List[Dict[str, Any]] = []
    for c in candidates:
        source_key = c.get("source_key")
        source = sources_by_key.get(source_key) if source_key else None
        if source is None:
            out.append(c)
            continue

        if source.get("active") is False:
            continue

        if audience == "patient" and source.get("patient_facing") is False:
            continue

        allowed_intents = source.get("allowed_intents") or []
        if allowed_intents and intent and intent not in allowed_intents:
            continue

        out.append(c)
    return out


async def enforce_source_governance(
    candidates: List[Dict[str, Any]],
    *,
    audience: str = "patient",
    intent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Async integration point: loads the current registry (ALL sources,
    including inactive ones — active_only=False, since an inactive
    source is exactly the case this function needs to be able to see and
    exclude) and applies filter_by_source_governance()."""
    from src.api.services.evidence.source_registry import get_source_registry
    registry = get_source_registry()
    sources = await registry.list_sources(active_only=False)
    sources_by_key = {s["source_key"]: s for s in sources if s.get("source_key")}
    return filter_by_source_governance(
        candidates, audience=audience, intent=intent, sources_by_key=sources_by_key,
    )
