"""
Applicability Scorer (Phase 4)

Scores a retrieved candidate against the patient's actual state instead of
ranking on semantic similarity alone — see the architecture review,
section 18: an NCI generic dysgeusia page and an ONS FOLFOX-specific page
can have close semantic scores but very different applicability to a
specific patient, and the second should usually win.

Three components, weighted and combined:
  semantic_relevance    the vector similarity score from the corpus search
  clinical_applicability   overlap between the plan's boost terms
                           (patient's actual regimen/agents/symptoms) and
                           the candidate's text + registered applicability
                           metadata
  source_authority       from evidence_sources.authority_class when the
                          candidate came from a registered patient-facing
                          source, else a fixed value for the (already
                          curated) literature corpus

No cross-encoder here: Phase 4 already ran an LLM-cheap classification
(intent) and this scorer is meant to be fast enough to run over every
candidate from every corpus on every patient message. If ranking quality
turns out to need it, a cross-encoder gate can be added the same way
comprehensive_retrieval.py's Phase 3 gate was — this module's
score_candidate is the natural place to add it without touching callers.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.api.services.evidence.retrieval_planner import RetrievalPlan

_AUTHORITY_CLASS_SCORE = {"A": 1.0, "B": 0.75, "C": 0.5}
_LITERATURE_DEFAULT_AUTHORITY = 0.8  # already-curated peer-reviewed corpus


def score_candidate(candidate: Dict[str, Any], plan: RetrievalPlan) -> Dict[str, Any]:
    semantic = float(candidate.get("semantic_score") or 0.0)

    text_lower = (candidate.get("text") or "").lower()
    applicability_meta = candidate.get("applicability_meta") or {}
    meta_terms = " ".join(
        " ".join(v) if isinstance(v, list) else str(v)
        for v in applicability_meta.values()
    ).lower()

    boost_terms = [t.lower() for t in (plan.boost_terms or []) if t]
    if boost_terms:
        hits = sum(1 for t in boost_terms if t in text_lower or t in meta_terms)
        clinical_applicability = min(1.0, hits / len(boost_terms))
    else:
        # No patient-specific terms to check against (unlinked patient, or
        # a question with no personalizable signal) — neutral, not zero.
        clinical_applicability = 0.5

    authority_class = candidate.get("authority_class")
    source_authority = (
        _AUTHORITY_CLASS_SCORE.get(authority_class, 0.6)
        if authority_class else _LITERATURE_DEFAULT_AUTHORITY
    )

    combined = 0.45 * semantic + 0.35 * clinical_applicability + 0.20 * source_authority

    out = dict(candidate)
    out.update({
        "semantic_relevance": round(semantic, 4),
        "clinical_applicability": round(clinical_applicability, 4),
        "source_authority": round(source_authority, 4),
        "applicability_score": round(combined, 4),
    })
    return out


def rank(
    candidates: List[Dict[str, Any]], plan: RetrievalPlan, limit: int = 5
) -> List[Dict[str, Any]]:
    scored = [score_candidate(c, plan) for c in candidates]
    scored.sort(key=lambda c: c["applicability_score"], reverse=True)
    return scored[:limit]
