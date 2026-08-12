"""
Query-time keyword resolver (Fix E).

Runs the ingestion-side `KeywordTagger` against the user's patient-profile
query to produce canonical-label detections — cancer_types, sites, stages,
histologies, biomarkers, drugs, alterations — plus Roman/Arabic stage
alias expansion. The output is projected into typed `should`-clauses on the
Qdrant filter (`metadata.cancer_types_detected`, `metadata.sites_detected`,
`metadata.stages_detected`, `metadata.biomarkers_detected`,
`metadata.drugs_detected`, `metadata.histologies_detected`,
`metadata.genomic_alterations`) — the same fields the tagger writes on
every ingested chunk.

Efficiency:
  - `KeywordTagger` is a module-level singleton (built once; ~<200 ms cold).
  - `scan_text_detailed` is a single-pass regex/substring scan — tens of ms
    on typical patient profiles.
  - `resolve_query_tokens` is wrapped in `@lru_cache(maxsize=256)`, so
    repeat queries inside a session are ~1 microsecond.

The tagger is passed the raw query — it handles tokenization, word boundaries
for short acronyms, NegEx-style negation, and ambiguity damping internally.
Query-side stopword stripping is deliberately avoided because it would
destroy the negation context the tagger relies on to mark negated phrases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingestion.keyword_tagger import KeywordTagger


_TAGGER: Optional["KeywordTagger"] = None


def _get_tagger() -> "KeywordTagger":
    """Return the module-level singleton `KeywordTagger`, constructing it
    on first use. Lazy-imported so the API cold-path doesn't pay for the
    ingestion module until the first query arrives."""
    global _TAGGER
    if _TAGGER is None:
        from src.ingestion.keyword_tagger import KeywordTagger
        _TAGGER = KeywordTagger()
    return _TAGGER


@dataclass(frozen=True)
class ResolvedQueryTokens:
    """Structured resolver output for a single query.

    Fields preserve the tagger's emitted casing (cancer_types carries Title-
    Case labels from `cancer_type_ontology.json`; stages carry `"Stage IV"`-
    style strings). All sets are hashable-sorted at dataclass construction.
    """

    cancer_types: frozenset = field(default_factory=frozenset)
    sites: frozenset = field(default_factory=frozenset)
    histologies: frozenset = field(default_factory=frozenset)
    stages: frozenset = field(default_factory=frozenset)
    stage_aliases: frozenset = field(default_factory=frozenset)
    tnm: frozenset = field(default_factory=frozenset)
    biomarkers: frozenset = field(default_factory=frozenset)
    biomarker_status: frozenset = field(default_factory=frozenset)
    drugs: frozenset = field(default_factory=frozenset)
    alterations: frozenset = field(default_factory=frozenset)
    treatment_lines: frozenset = field(default_factory=frozenset)
    disease_status: frozenset = field(default_factory=frozenset)
    ajcc_cancer_keys: frozenset = field(default_factory=frozenset)
    keywords_flat: frozenset = field(default_factory=frozenset)
    negated: frozenset = field(default_factory=frozenset)

    def as_typed_filters(self) -> Dict[str, List[str]]:
        """Project onto the `metadata.*_detected` payload fields written at
        ingest by `KeywordTagger.tag_chunk`. Empty sets are omitted so the
        caller can skip the corresponding `should` clause entirely.
        """
        stages_combined = self.stages | self.stage_aliases
        mapping = (
            ("cancer_types_detected", self.cancer_types),
            ("sites_detected",        self.sites),
            ("histologies_detected",  self.histologies),
            ("stages_detected",       stages_combined),
            ("biomarkers_detected",   self.biomarkers),
            ("drugs_detected",        self.drugs),
            ("genomic_alterations",   self.alterations),
        )
        return {
            field_name: sorted(values)
            for field_name, values in mapping
            if values
        }


# ── Stage alias expansion ──────────────────────────────────────────────────
# The tagger emits "Stage IV", "Stage IVA" (Roman, Title case). Documents
# tagged by the same tagger will carry identical strings. But legacy docs
# or manually-authored tags may use Arabic or lowercase forms — expand so
# MatchAny catches both conventions.

_ROMAN_TO_ARABIC = {"I": "1", "II": "2", "III": "3", "IV": "4"}
_STAGE_RE = re.compile(r"^stage\s+([IViv]+|\d+)([abc]?)\b", re.IGNORECASE)


def _expand_stage_aliases(stages) -> Set[str]:
    out: Set[str] = set()
    for s in stages:
        m = _STAGE_RE.match((s or "").strip())
        if not m:
            out.add(s)
            continue
        raw = m.group(1).upper()
        sub = (m.group(2) or "").upper()
        arabic_map = _ROMAN_TO_ARABIC if raw in _ROMAN_TO_ARABIC else None
        if arabic_map is None:
            # Raw is already Arabic — flip to Roman
            inverse = {v: k for k, v in _ROMAN_TO_ARABIC.items()}
            roman = inverse.get(raw)
            arabic = raw
        else:
            roman = raw
            arabic = _ROMAN_TO_ARABIC[raw]
        if not roman:
            out.add(s)
            continue
        for form in (roman, arabic):
            out.add(f"Stage {form}{sub}")
            out.add(f"stage {form.lower()}{sub.lower()}")
            # Family prefix so a patient's "IVA" matches docs tagged "IV"
            if sub:
                out.add(f"Stage {form}")
                out.add(f"stage {form.lower()}")
    return out


# ── Resolver entry point ───────────────────────────────────────────────────


@lru_cache(maxsize=256)
def resolve_query_tokens(query_text: str) -> ResolvedQueryTokens:
    """Scan the query against every JSON ontology (cancer_type_ontology,
    clinical_trial_ontology, extractor_keywords, ajcc_staging_tables) via
    the ingestion-side `KeywordTagger`, and package the canonical-label
    detections for query-time filter construction.

    Returns an empty `ResolvedQueryTokens` when the query is empty.
    """
    if not query_text:
        return ResolvedQueryTokens()

    tagger = _get_tagger()
    detail = tagger.scan_text_detailed(query_text)

    stages = frozenset(detail.get("stages_detected") or ())
    stage_aliases = frozenset(_expand_stage_aliases(stages))

    ajcc_keys: Set[str] = set()
    for tag in (detail.get("ajcc_tags") or ()):
        parts = tag.split(":")
        if len(parts) >= 2 and parts[0] == "ajcc":
            ajcc_keys.add(parts[1])

    # biomarker_status is a dict {canonical: [statuses]}; flatten to
    # "canonical:status" strings matching `metadata.biomarker_status_flat`.
    bm_status_map = detail.get("biomarker_status") or {}
    bm_status_flat: Set[str] = {
        f"{canonical}:{status}"
        for canonical, statuses in bm_status_map.items()
        for status in (statuses or ())
    }

    return ResolvedQueryTokens(
        cancer_types=frozenset(detail.get("cancer_types_detected") or ()),
        sites=frozenset(detail.get("sites_detected") or ()),
        histologies=frozenset(detail.get("histologies_detected") or ()),
        stages=stages,
        stage_aliases=stage_aliases,
        tnm=frozenset(detail.get("tnm_detected") or ()),
        biomarkers=frozenset(detail.get("biomarkers_detected") or ()),
        biomarker_status=frozenset(bm_status_flat),
        drugs=frozenset(detail.get("drugs_detected") or ()),
        alterations=frozenset(detail.get("genomic_alterations") or ()),
        treatment_lines=frozenset(detail.get("treatment_lines_detected") or ()),
        disease_status=frozenset(detail.get("disease_status_detected") or ()),
        ajcc_cancer_keys=frozenset(ajcc_keys),
        keywords_flat=frozenset(detail.get("keywords_flat") or ()),
        negated=frozenset(detail.get("_negated_mentions") or ()),
    )
