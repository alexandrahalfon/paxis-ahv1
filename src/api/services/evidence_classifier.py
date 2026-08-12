"""
Evidence Classifier
===================

Classifies a study's text into one of three buckets so the retrieval
pipeline can budget guidelines/landmarks separately from
patient-specific trial evidence.

Buckets:
  - ``"guideline"``       — NCCN / ASTRO / ASCO / ESMO / AUA / NICE etc.
  - ``"landmark_trial"``  — Named or cooperative-group trials that are
                            cited so often they function as shared
                            reference points (KEYNOTE-048, RTOG-0522).
  - ``"trial"``           — Everything else (single-center studies,
                            small RCTs, case series, retrospective
                            reviews, most literature).

Why a separate module
---------------------
The ``LANDMARK_PATTERNS`` used to live inside ``enhanced_rag_service``
where they fed a multiplicative 1.2×–1.3× score boost. That boost let
guidelines crowd out patient-specific studies in the shared top-k
budget. Moving the patterns here lets ``comprehensive_retrieval``
bucket studies by type and apply separate caps without pulling in the
whole RAG service.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Literal, Optional, Tuple


EvidenceType = Literal["guideline", "landmark_trial", "trial"]


# Patterns lifted from enhanced_rag_service.LANDMARK_PATTERNS. The
# two-track architecture only needs guideline/landmark buckets, so we
# collapse the six pattern groups into two, ordered by priority:
# guideline beats landmark_trial when both match.

GUIDELINE_PATTERN = re.compile(
    r"\b(NCCN|ASTRO|ASCO|ESMO|AUA|ESTRO|NICE|EANO|EAU|ESTRO|SIOG|SABCS|"
    r"ABS|SSO)\s*(?:guideline|recommendation|consensus|statement|position)",
    re.IGNORECASE,
)

LANDMARK_PATTERN = re.compile(
    # Cooperative-group trial names (with numeric ID)
    r"\b(RTOG|NRG|NSABP|ACOSOG|EORTC|PORTEC|SWOG|ECOG|GOG|CALGB|"
    r"NCIC|MRC|TROG|ALLIANCE|NCCTG|COG|INT|INTERGROUP|BC)\s*[-]?\s*\d+"
    # OR named landmark trials
    r"|\b(KEYNOTE|CHECKMATE|PACIFIC|ADAURA|LAURA|RAPIDO|PRODIGE|"
    r"STAMPEDE|CHAARTED|LATITUDE|ENZAMET|TITAN|ARCHES|"
    r"FAST[- ]?FORWARD|START|PRIME|AMAROS|Z0011|"
    r"TAILORx|RxPONDER|MINDACT|SOFT|TEXT|MONARCH|"
    r"CLEOPATRA|APHINITY|KATHERINE|DESTINY|EMILIA|"
    r"HORRAD|FLAURA|EMBRACE|ASCENDE[- ]?RT|EURAMOS|"
    r"STARS|ROSEL|CAO[/-]?ARO[/-]?AIO|STOCKHOLM|"
    r"MA[- ]?20|DANISH)\b"
    # OR author-named landmark trials (appearing with journal indicator)
    r"|\b(Stupp|Packer|Turrisi|Slotman|Loehrer)\b.*"
    r"(trial|study|NEJM|Lancet|JCO|JAMA|N\s*Engl\s*J\s*Med)",
    re.IGNORECASE,
)


def classify_evidence_text(text: str) -> EvidenceType:
    """Classify a single blob of text.

    ``text`` should be a concatenation of the study's title + citation +
    top-chunk text. Order of checks matters: guidelines outrank landmark
    trials (an NCCN document cross-referencing KEYNOTE-048 is still a
    guideline).
    """
    if not text:
        return "trial"
    if GUIDELINE_PATTERN.search(text):
        return "guideline"
    if LANDMARK_PATTERN.search(text):
        return "landmark_trial"
    return "trial"


def classify_study(
    title: Optional[str],
    citation: Optional[str],
    chunk_texts: Iterable[str],
) -> EvidenceType:
    """Classify a study using its title, citation, and the first few
    chunk texts. Keeps inspected text small so this stays cheap."""
    parts: List[str] = []
    if title:
        parts.append(title)
    if citation:
        parts.append(citation)
    # Take the first 1-2 chunk texts — title + abstract paragraph is
    # where "NCCN Guideline" would actually appear.
    for i, t in enumerate(chunk_texts):
        if i >= 2:
            break
        if t:
            parts.append(t[:800])
    return classify_evidence_text(" | ".join(parts))


def bucket_studies(
    studies: List[object],
) -> Dict[EvidenceType, List[object]]:
    """Given a list of ``StudyEvidence`` objects, return three lists
    keyed by evidence_type. Assumes each study already has
    ``evidence_type`` populated; if not, falls back to ``"trial"``."""
    buckets: Dict[EvidenceType, List[object]] = {
        "guideline": [], "landmark_trial": [], "trial": [],
    }
    for s in studies:
        etype = getattr(s, "evidence_type", None) or "trial"
        buckets.setdefault(etype, []).append(s)
    return buckets
