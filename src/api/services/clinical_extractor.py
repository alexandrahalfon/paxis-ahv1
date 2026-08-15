"""
Clinical Profile Extractor
==========================

One GPT-4o-mini call pulls raw clinical axes out of the user query as a
free-text JSON blob. The blob is then normalized through ``SynonymIndex``
so every value ends up in the exact canonical form that appears in
Qdrant chunk payloads — ``"SCC"`` → ``"Squamous cell carcinoma"``,
``"keytruda"`` → ``"Pembrolizumab"``, ``"tongue"`` → ``"Oral cavity"``,
and so on. Semantic equivalents are expanded where appropriate
(``"MSI-H"`` → ``["MSI-H", "dMMR"]``).

Why the rewrite
---------------
The previous cascading extractor asked GPT-4o-mini six times with
constrained option lists. That made the prompts large (full taxonomy
per axis) and left correctness brittle — the LLM had to copy canonical
strings byte-for-byte. Synonym groups already exist in the codebase
for ingestion, so the cheap, robust path is: extract raw text once,
let Python do the canonicalization.

Biomarker expressions (``"CPS<100"``, ``"HER2+"``) are still parsed
with a small regex pass — that's precise and deterministic and doesn't
need an LLM.

The ``ClinicalProfile`` dataclass shape is unchanged, so no downstream
caller needs to change.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.api.services.synonym_index import get_synonym_index


# ── Controlled vocabulary for the one axis without a keyword_tagger group ─

# Disease status / trajectory terms. Applied as a *soft* boost against
# ``metadata.keywords_flat`` — hard-filtering on these would exclude too
# many otherwise-relevant studies. All values lowercase to match
# keywords_flat at query time.
CANONICAL_DISEASE_STATUS: List[str] = [
    "recurrent", "recurrence",
    "metastatic", "metastasis",
    "refractory",
    "progressing", "progression", "progressive disease",
    "relapsed", "relapse",
    "unresectable", "inoperable",
    "locally advanced", "advanced",
    "locoregional",
    "newly diagnosed", "treatment-naive", "treatment naive",
    "stable disease",
    "remission",
    "first-line", "second-line", "third-line", "salvage",
    "post-progression",
]


# ── Result dataclass ────────────────────────────────────────────────────

@dataclass
class ClinicalProfile:
    """Typed clinical axes extracted and normalized from a user query.

    All list fields contain the exact canonical strings indexed into
    ``metadata.*_detected`` Qdrant payload fields, so they can be
    passed directly to ``FieldCondition(MatchAny(...))`` hard filters.
    """

    cancer_type_key: Optional[str] = None
    cancer_type_label: Optional[str] = None        # matches cancer_types_detected

    cancer_sites: List[str] = field(default_factory=list)     # sites_detected
    histologies: List[str] = field(default_factory=list)      # histologies_detected
    stages: List[str] = field(default_factory=list)           # stages_detected
    biomarkers: List[str] = field(default_factory=list)       # biomarkers_detected
    prior_treatments: List[str] = field(default_factory=list) # drugs_detected

    # Soft-boost axis — lowercased disease-status terms for keywords_flat
    disease_status: List[str] = field(default_factory=list)

    # Reference fields — carried through for eligibility scoring / display,
    # NOT used as Qdrant filter values.
    anatomical_subsites: List[str] = field(default_factory=list)
    biomarker_expressions: List[Dict[str, Any]] = field(default_factory=list)
    inclusion_criteria_text: Optional[str] = None
    exclusion_criteria_text: Optional[str] = None

    # Terms the LLM surfaced that the synonym index could NOT resolve
    # (useful for auditing and for expanding the ontology later).
    unresolved: Dict[str, List[str]] = field(default_factory=dict)

    # Raw JSON the LLM produced, kept for debugging
    raw_extraction: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def has_any_filter(self) -> bool:
        return bool(
            self.cancer_type_label
            or self.cancer_sites
            or self.histologies
            or self.stages
            or self.biomarkers
            or self.prior_treatments
        )


# ── Biomarker expression regex parser (unchanged from prior version) ────

_BIOMARKER_NAME_GROUP = (
    r"(?P<name>"
    r"CPS|TPS|PD[-\s]?L1|PD[-\s]?1|TMB(?:-H)?|"
    r"HER2|ER|PR|AR|Ki[-\s]?67|"
    r"MSI(?:-H|-L)?|MSS|dMMR|pMMR|HRD|"
    r"KRAS|BRAF|EGFR|ALK|ROS1|RET|NTRK|MET|BRCA1?/?2?"
    r")"
)
_BIOMARKER_OP_GROUP = r"(?P<op><=|>=|≤|≥|<|>|=|\+|\-|positive|negative|mutant|wild[- ]?type)"
_BIOMARKER_VAL_GROUP = r"(?P<value>\d+(?:\.\d+)?%?)"

_BIOMARKER_EXPR_PATTERNS = [
    # "CPS<100", "PD-L1 ≥ 50%", "TMB > 10"
    re.compile(
        rf"\b{_BIOMARKER_NAME_GROUP}\s*{_BIOMARKER_OP_GROUP}\s*{_BIOMARKER_VAL_GROUP}\b",
        re.IGNORECASE,
    ),
    # "CPS 100", "TMB 10"
    re.compile(
        rf"\b{_BIOMARKER_NAME_GROUP}\s+{_BIOMARKER_VAL_GROUP}(?=[\s,;.]|$)",
        re.IGNORECASE,
    ),
    # "HER2+", "ER-", "PD-L1 positive"
    re.compile(
        rf"\b{_BIOMARKER_NAME_GROUP}(?:\s*(?P<op>\+|\-|positive|negative|mutant|wild[- ]?type))(?=[\s,;.]|$)",
        re.IGNORECASE,
    ),
]


def parse_biomarker_expressions(query: str) -> List[Dict[str, Any]]:
    """Extract structured biomarker expressions from the query.

    Returns a list of ``{"name": <canonical>, "op": ..., "value": ...,
    "raw": ...}`` dicts. Names are normalized through the SynonymIndex
    so the canonical matches ``metadata.biomarkers_detected``.
    """
    if not query:
        return []
    idx = get_synonym_index()
    seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    op_map = {
        "positive": "+", "negative": "-",
        "mutant": "mutant", "wild-type": "wt", "wild type": "wt", "wildtype": "wt",
        "≤": "<=", "≥": ">=",
    }
    for pattern in _BIOMARKER_EXPR_PATTERNS:
        for m in pattern.finditer(query):
            raw_name = m.group("name")
            # SynonymIndex handles casing / hyphenation / alias → canonical
            canonicals = idx.resolve("biomarker", raw_name)
            canonical = canonicals[0] if canonicals else raw_name
            op = (m.groupdict().get("op") or "").strip().lower() or "="
            value = (m.groupdict().get("value") or "").strip()
            op = op_map.get(op, op)
            key = (canonical, op, value)
            if key not in seen:
                seen[key] = {
                    "name": canonical,
                    "op": op,
                    "value": value,
                    "raw": m.group(0),
                }
    return list(seen.values())


# ── Extractor ───────────────────────────────────────────────────────────

_LLM_EXTRACTION_PROMPT = """\
Extract the clinical facts below from the patient query. Return ONLY a \
JSON object with these keys (use empty arrays / empty strings / null \
when the query doesn't mention an axis). Copy phrases directly from \
the query — do not invent, do not normalize, do not rename.

Required keys (all must be present):
  "cancer_type":        string | null   (e.g. "breast cancer", "head and neck", "HNSCC")
  "cancer_sites":       string[]        (organs / regions, e.g. ["lung"], ["oral tongue"])
  "histologies":        string[]        (e.g. ["SCC"], ["invasive ductal carcinoma"])
  "stages":             string[]        (AJCC format, e.g. ["Stage IIA"], ["IV"])
  "biomarkers":         string[]        (names only — thresholds go in the query as-is, we parse them separately)
  "prior_treatments":   string[]        (drug names or regimens; brand or generic OK)
  "disease_status":     string[]        (e.g. ["recurrent"], ["progressing"], ["unresectable"])
  "anatomical_subsites": string[]       (fine subsites mentioned, e.g. ["tongue"], ["base of tongue"], ["frontal lobe"])
  "inclusion_criteria": string          (any text describing criteria that would admit the patient)
  "exclusion_criteria": string          (any text describing criteria that would bar the patient)

Patient query:
---
{query}
---

JSON object:"""


class ClinicalExtractor:
    """One-shot clinical profile extractor.

    Usage
    -----
        extractor = ClinicalExtractor(openai_client)
        profile = await extractor.extract(user_query)
        if profile.has_any_filter():
            # Pass profile into retrieve_comprehensive() for hard filter
    """

    def __init__(self, openai_client, model: str = "gpt-4o-mini"):
        self.oa = openai_client
        self.model = model

    async def extract(self, user_query: str) -> ClinicalProfile:
        profile = ClinicalProfile()
        if not user_query or not user_query.strip():
            return profile

        raw = await self._extract_raw(user_query)
        if not raw:
            return profile

        profile.raw_extraction = raw
        idx = get_synonym_index()
        unresolved: Dict[str, List[str]] = {}

        # ── cancer_type (singular) ────────────────────────────────
        ct_raw = raw.get("cancer_type")
        if isinstance(ct_raw, str) and ct_raw.strip():
            ct_canonicals = idx.resolve("cancer_type", ct_raw)
            if ct_canonicals:
                # cancer_type is singular; take the first canonical
                profile.cancer_type_label = ct_canonicals[0]
                # Recover the ontology key for future use
                profile.cancer_type_key = _label_to_key(profile.cancer_type_label)
            else:
                unresolved.setdefault("cancer_type", []).append(ct_raw)

        # ── multi-value axes that map to *_detected payload fields ──
        axis_specs = (
            ("cancer_sites",     "site"),
            ("histologies",      "histology"),
            ("biomarkers",       "biomarker"),
            ("prior_treatments", "drug"),
        )
        for profile_attr, index_axis in axis_specs:
            raw_list = _ensure_list(raw.get(profile_attr))
            canonicals = idx.normalize_many(index_axis, raw_list)
            un = idx.unresolved_terms(index_axis, raw_list)
            setattr(profile, profile_attr, canonicals)
            if un:
                unresolved[profile_attr] = un

        # ── stages (no synonym group; normalize case to Roman form) ──
        raw_stages = _ensure_list(raw.get("stages"))
        profile.stages = _normalize_stages(raw_stages)

        # ── disease_status (soft boost, from controlled vocab) ───
        raw_status = [s.lower() for s in _ensure_list(raw.get("disease_status"))]
        status_set = set(CANONICAL_DISEASE_STATUS)
        profile.disease_status = [s for s in raw_status if s in status_set]
        # Also accept close forms (dropping suffixes)
        for extra in raw_status:
            if extra in status_set or extra in profile.disease_status:
                continue
            if extra.rstrip("s") in status_set:
                profile.disease_status.append(extra.rstrip("s"))

        # ── anatomical_subsites (reference only) ────────────────
        profile.anatomical_subsites = [
            s.strip() for s in _ensure_list(raw.get("anatomical_subsites")) if s and s.strip()
        ]

        # ── biomarker expressions (regex; merges names into filter) ─
        bio_exprs = parse_biomarker_expressions(user_query)
        profile.biomarker_expressions = bio_exprs
        if bio_exprs:
            merged = list(profile.biomarkers)
            for expr in bio_exprs:
                name = expr.get("name")
                if name and name not in merged:
                    merged.append(name)
            profile.biomarkers = merged

        # ── free-text eligibility signals ──────────────────────
        inc = raw.get("inclusion_criteria")
        exc = raw.get("exclusion_criteria")
        profile.inclusion_criteria_text = _nonempty_str(inc)
        profile.exclusion_criteria_text = _nonempty_str(exc)

        profile.unresolved = unresolved
        return profile

    # ── LLM call ──────────────────────────────────────────────

    async def _extract_raw(self, user_query: str) -> Dict[str, Any]:
        prompt = _LLM_EXTRACTION_PROMPT.format(query=user_query.strip())

        def _call() -> str:
            resp = self.oa.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()

        try:
            content = await asyncio.to_thread(_call)
        except Exception as e:
            print(f"[ClinicalExtractor] LLM call failed: {e}")
            return {}

        if not content:
            return {}
        try:
            obj = json.loads(content)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError as e:
            print(f"[ClinicalExtractor] JSON parse failed: {e} | content[:200]={content[:200]!r}")
            return {}


# Backwards-compatibility alias — ``CascadingClinicalExtractor`` is the
# name enhanced_rag_service.py imports today. Keep it pointing at the
# same class so no caller changes.
CascadingClinicalExtractor = ClinicalExtractor


# ── Helpers ─────────────────────────────────────────────────────────────

def _ensure_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    return []


def _nonempty_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"none", "n/a", "not applicable", "null", "[]"}:
        return None
    return s


_STAGE_CANONICAL = {
    "0", "i", "ia", "ib", "ii", "iia", "iib", "iic",
    "iii", "iiia", "iiib", "iiic",
    "iv", "iva", "ivb", "ivc",
}


def _normalize_stages(raw_stages: List[str]) -> List[str]:
    """Map raw stage strings (e.g. "IIA", "stage 2a", "IV") to the
    "Stage X" Title-Case form stored in ``metadata.stages_detected``.
    """
    out: List[str] = []
    for s in raw_stages:
        token = s.strip().lower()
        # Strip "stage" prefix
        token = re.sub(r"^stage\s+", "", token)
        # Arabic → Roman conversion for 1-4 with optional letter
        arabic = re.match(r"^(\d+)([a-c]?)$", token)
        if arabic:
            roman_map = {"1": "i", "2": "ii", "3": "iii", "4": "iv"}
            base = roman_map.get(arabic.group(1))
            if base:
                token = base + (arabic.group(2) or "")
        if token in _STAGE_CANONICAL:
            title = "Stage " + token.upper().replace("IVA", "IVA")
            # Normalize caps: "Stage IIA", "Stage IV"
            title = "Stage " + token.upper()
            if title not in out:
                out.append(title)
    return out


_LABEL_TO_KEY_CACHE: Dict[str, str] = {}


def _label_to_key(label: Optional[str]) -> Optional[str]:
    """Reverse lookup: cancer_type Title-Case label → ontology key."""
    if not label:
        return None
    if not _LABEL_TO_KEY_CACHE:
        # Lazy one-time build
        from pathlib import Path
        p = Path(__file__).resolve().parents[3] / "data" / "ontology" / "cancer_type_ontology.json"
        try:
            with p.open("r", encoding="utf-8") as f:
                for key, entry in json.load(f).items():
                    lbl = entry.get("label")
                    if lbl:
                        _LABEL_TO_KEY_CACHE[lbl] = key
        except Exception:
            return None
    return _LABEL_TO_KEY_CACHE.get(label)


# ═══════════════════════════════════════════════════════════════════════
# LLM-result → ClinicalProfile → QueryStructure helpers
#
# These are the "no duplicate LLM call" path: the existing pipeline
# already runs ``structure_query_with_llm`` in parallel with the
# embedding step and merges the result into QueryStructure.raw_text
# fields. We reuse that 8-axis dict to build a ClinicalProfile (via
# SynonymIndex) and fold its canonical values back into QueryStructure
# to fill gaps that the regex pass left empty.
#
# Two rules:
#   1. A populated regex field is NEVER overridden by a profile value.
#      The regex is a confident, literal match; the LLM is a softer
#      signal that belongs in the gaps.
#   2. Biomarkers are *merged*, not replaced — regex-extracted and
#      profile-resolved canonicals both end up in the final list.
# ═══════════════════════════════════════════════════════════════════════


def build_profile_from_llm_result(
    query_text: str,
    llm_result: Optional[Dict[str, Any]],
) -> ClinicalProfile:
    """Build a ClinicalProfile from an already-computed LLM 8-axis dict.

    Reuses the LLM output that ``structure_query_with_llm`` already
    produces, so this function makes NO new LLM calls. It runs each
    axis through ``SynonymIndex`` to derive Title-Case canonicals that
    match Qdrant payload conventions.

    Keys read from ``llm_result``:
        - ``primary_cancer`` — used for both cancer_type + histology + site
        - ``biomarker_profile`` — biomarker canonicals + expressions

    Unknown / unresolvable terms are silently dropped.
    """
    profile = ClinicalProfile()
    if not llm_result:
        return profile

    idx = get_synonym_index()

    # ── cancer_type from primary_cancer ─────────────────────────
    primary = (llm_result.get("primary_cancer") or "").strip()
    if primary:
        # Try whole-phrase resolution first; fall back to word-level
        # hits (e.g. "invasive ductal carcinoma of the left breast"
        # won't match as a whole phrase, but "breast" will).
        ct_hits = idx.resolve("cancer_type", primary)
        if not ct_hits:
            for token in primary.split():
                ct_hits = idx.resolve("cancer_type", token)
                if ct_hits:
                    break
        if ct_hits:
            profile.cancer_type_label = ct_hits[0]
            profile.cancer_type_key = _label_to_key(profile.cancer_type_label)

        # Histology: try whole-phrase, then scan for any known histology
        # canonical inside the primary_cancer string.
        hist_hits = idx.resolve("histology", primary)
        if not hist_hits:
            # Scan word-by-word for known histology canonicals.
            seen: List[str] = []
            for token in primary.split():
                for h in idx.resolve("histology", token):
                    if h not in seen:
                        seen.append(h)
            hist_hits = seen
        profile.histologies = hist_hits

    # ── biomarkers from biomarker_profile ──────────────────────
    bio_text = (llm_result.get("biomarker_profile") or "").strip()
    if bio_text:
        # Token-level lookup catches "CPS", "PD-L1", etc.
        # Scan word-by-word instead of trying to match the whole phrase.
        seen_bio: List[str] = []
        for token in _biomarker_candidate_tokens(bio_text):
            for b in idx.resolve("biomarker", token):
                if b not in seen_bio:
                    seen_bio.append(b)
        profile.biomarkers = seen_bio
        # Also capture structured expressions (CPS<100, HER2+ etc.)
        profile.biomarker_expressions = parse_biomarker_expressions(bio_text)
        # Names from expressions that weren't caught by token scan
        for expr in profile.biomarker_expressions:
            name = expr.get("name")
            if name and name not in profile.biomarkers:
                profile.biomarkers.append(name)

    # Also sweep the whole query_text for biomarker expressions the LLM
    # might not have surfaced (e.g. "CPS 100" in free narrative).
    for expr in parse_biomarker_expressions(query_text or ""):
        if expr not in profile.biomarker_expressions:
            profile.biomarker_expressions.append(expr)
        name = expr.get("name")
        if name and name not in profile.biomarkers:
            profile.biomarkers.append(name)

    profile.raw_extraction = dict(llm_result)
    return profile


def _biomarker_candidate_tokens(text: str) -> List[str]:
    """Yield candidate tokens from a biomarker-profile free-text string.

    Splits on whitespace and common punctuation, filters stop-shapes.
    Includes 1- and 2-gram candidates so "PD-L1" and "PD L1" both
    surface.
    """
    import re as _re
    cleaned = _re.sub(r"[,;]+", " ", text)
    parts = [p for p in _re.split(r"\s+", cleaned) if p]
    out: List[str] = []
    for p in parts:
        if p and len(p) >= 2:
            out.append(p)
    # 2-grams
    for i in range(len(parts) - 1):
        out.append(f"{parts[i]} {parts[i + 1]}")
    return out


def apply_profile_to_structure(query_structure, profile: ClinicalProfile):
    """Fold ClinicalProfile canonicals back into a QueryStructure.

    Rules:
      - cancer.site / filter_category: only set when regex left them empty.
      - cancer.histology: only set when regex left it empty.
      - cancer.biomarkers: merged (regex + profile canonicals, deduped).

    Returns the same ``query_structure`` mutated in place (convenience
    — callers can ignore the return value).
    """
    if query_structure is None or profile is None:
        return query_structure

    # Site / category gap-fill
    if query_structure.cancer.site is None and profile.cancer_type_key:
        query_structure.cancer.site = profile.cancer_type_key
        # filter_category defaults to the cancer_type_key itself when
        # no more specific category is known, mirroring what the regex
        # would do.
        if query_structure.filter_category is None:
            # Map ontology key to the category value the regex produces.
            # Most keys are identical to categories; a handful differ
            # (h_n → head_neck), so we consult CANCER_SITE_PATTERNS.
            query_structure.filter_category = _site_category_for_key(
                profile.cancer_type_key
            )

    # Histology gap-fill
    if not query_structure.cancer.histology and profile.histologies:
        query_structure.cancer.histology = profile.histologies[0]

    # Biomarker merge (dedup case-insensitive)
    if profile.biomarkers:
        existing_lower = {b.lower() for b in query_structure.cancer.biomarkers}
        for b in profile.biomarkers:
            if b.lower() not in existing_lower:
                query_structure.cancer.biomarkers.append(b)
                existing_lower.add(b.lower())

    return query_structure


def _site_category_for_key(cancer_type_key: str) -> Optional[str]:
    """Map an ontology cancer_type key (e.g. 'h_n') to the
    filter_category the regex would assign (e.g. 'head_neck')."""
    try:
        from src.api.services.query_structuring_service import (
            CANCER_SITE_PATTERNS,
        )
    except Exception:
        return cancer_type_key
    # Prefer a direct key hit
    if cancer_type_key in CANCER_SITE_PATTERNS:
        return CANCER_SITE_PATTERNS[cancer_type_key].get("category", cancer_type_key)
    # Fall back: a key that normalizes to the same thing
    alt_key = cancer_type_key.replace("_", "")
    for k, info in CANCER_SITE_PATTERNS.items():
        if k.replace("_", "") == alt_key:
            return info.get("category", cancer_type_key)
    return cancer_type_key
