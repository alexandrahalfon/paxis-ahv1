"""
Evidence Packet Builder (Phase 4)

Assembles the structured packet handed to generation, matching the shape
in the architecture review section 28 — question + patient_context +
ranked evidence + safety — instead of dropping the raw top-N chunks
straight into the prompt. Gives the chat/tools layer one reproducible,
loggable object per answer rather than an implicit string concatenation.

Provenance + dedup (2026-08-12 beta audit items 6/7): each evidence entry
now carries the full chain a citation needs to be auditable — Qdrant
point id, document/version id, source key/name, section, chunk index,
URL, itemized score components, and any incompatibility reasons
(applicability_scorer.py) — not just source/title/role/authority/text/
citation/year. multi_corpus_retriever.search() already dedupes by exact
chunk identity (point id) so distinct sections of one document survive
as distinct candidates; this module does a SEPARATE, later dedup pass by
normalized text content, catching the case that chunk-identity dedup
can't: the same passage genuinely appearing twice (e.g. surfaced by two
different collections/searches), which chunk-identity dedup — correctly
— does not consider a duplicate of itself.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def summarize_context(patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    patient_context = patient_context or {}
    features = patient_context.get("retrieval_features") or {}
    state = patient_context.get("state") or {}
    summary: Dict[str, Any] = {}

    if features.get("regimens"):
        summary["active_regimen"] = ", ".join(features["regimens"])
    if features.get("active_agents"):
        summary["active_agents"] = features["active_agents"]
    if features.get("symptoms"):
        summary["symptoms"] = features["symptoms"]
    if features.get("nutrition_risk"):
        summary["nutrition_risk"] = features["nutrition_risk"]
    if features.get("comorbidities"):
        summary["comorbidities"] = features["comorbidities"]

    dx = state.get("active_diagnosis") or {}
    if dx.get("cancer_site"):
        summary["cancer_type"] = dx["cancer_site"]
    if dx.get("stage"):
        summary["stage"] = dx["stage"]

    # Care-team instructions (2026-08-12 beta audit item 5): carried into
    # the packet's patient_context so the auditable object this module
    # exists to produce actually reflects what generation was told to
    # prioritize, not just the passive facts summarized above. See
    # patient_chat_service._care_team_instructions_block for how these
    # reach the system prompt with explicit precedence framing — this
    # function only surfaces them into the packet, it doesn't rank them.
    instructions = [
        {"text": i.get("text"), "type": i.get("type")}
        for i in (state.get("care_team_instructions") or []) if i.get("text")
    ]
    if instructions:
        summary["care_team_instructions"] = instructions

    return summary


def _content_key(text: str) -> str:
    """Hash of whitespace-normalized, lowercased text — identifies a
    genuine duplicate PASSAGE regardless of which chunk/collection it
    came from. Deliberately a different identity than
    multi_corpus_retriever._candidate_identity (which is per-chunk, by
    Qdrant point id) — see this module's docstring for why both exist."""
    normalized = " ".join((text or "").split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _dedup_by_content(ranked_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keeps the first (i.e. highest-ranked, since ranked_evidence is
    already sorted by applicability_score descending) occurrence of each
    distinct passage."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for e in ranked_evidence:
        key = _content_key(e.get("text") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def build_packet(
    question: str,
    patient_context: Optional[Dict[str, Any]],
    ranked_evidence: List[Dict[str, Any]],
    safety_category: str = "general",
) -> Dict[str, Any]:
    deduped = _dedup_by_content(ranked_evidence)
    return {
        "question": question,
        "patient_context": summarize_context(patient_context),
        "evidence": [
            {
                "source": e.get("source_key") or e.get("title"),
                # Provenance — see module docstring. All sourced from
                # what multi_corpus_retriever.py / applicability_scorer.py
                # already attached to the candidate; nothing new fetched
                # here.
                "qdrant_point_id": e.get("qdrant_point_id"),
                "document_id": e.get("doc_id"),
                "version_id": e.get("version_id"),
                "source_key": e.get("source_key"),
                "source_name": e.get("source_name"),
                "section": e.get("section_title"),
                "chunk_index": e.get("chunk_index"),
                "url": e.get("url"),
                "title": e.get("title"),
                "role": e.get("collection"),
                "authority": e.get("authority_class") or "literature",
                "semantic_score": e.get("semantic_relevance"),
                "applicability_score": e.get("applicability_score"),
                "score_components": e.get("components"),
                "incompatibility_reasons": e.get("incompatibility_reasons") or [],
                "text": e.get("text"),
                "citation": e.get("citation"),
                "year": e.get("year"),
            }
            for e in deduped
        ],
        "safety": {"category": safety_category, "red_flags": []},
    }


def to_prompt_block(packet: Dict[str, Any], limit: int = 5) -> str:
    """Same shape PatientChatService._evidence_block already produces, so
    swapping the source in doesn't change the prompt-building contract at
    the call site."""
    block = ""
    for i, e in enumerate(packet.get("evidence", [])[:limit], 1):
        block += f"\n[{i}] {e.get('title')}\n{(e.get('text') or '')[:500]}\n"
    return block


def to_sources(packet: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    for e in packet.get("evidence", [])[:limit]:
        out.append({
            "title": e.get("title"),
            "citation": e.get("citation"),
            "year": e.get("year"),
            "source_type": e.get("role"),
            "authority": e.get("authority"),
        })
    return out
