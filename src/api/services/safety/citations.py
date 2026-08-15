"""
Citation safety layer.

After generation, extract cited (author, year) tuples from an answer string
and verify each one appears in the retrieved `EvidenceBundle`. Unverified
tuples are either stripped or flagged depending on the caller's choice.

Design mirrors `safety.numerical`:
 - `extract_cited_author_year(text)` returns the list of (author, year) tuples
   mentioned in the answer.
 - `verify_citations(text, studies)` returns a structured verdict dict.
 - `strip_unverified_citations(text, studies)` rewrites the answer by
   replacing unverified citations with `[unverified]` and bumps
   `pipeline_metrics.safety.citations_stripped`.

Patterns target the citation styles the generation path actually emits:
  "Smith et al., 2023"
  "Smith et al. 2023"
  "Smith 2023"
  "(Smith et al., 2023)"

The matcher tolerates optional leading parens, comma before the year, and
lower/upper-case `et al`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple


# Surname char class includes Latin-1 accented letters so names like
# "García-López" are captured as a single token rather than truncated at "Garc".
_CITATION_RE = re.compile(
    r"""
    (?P<author>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-']{2,})  # capitalised surname
    (?:\s+et\s+al\.?)?                                 # optional " et al." / " et al"
    [,\s]+                                             # separator
    (?P<year>(?:19|20)\d{2})                           # 4-digit year, 1900-2099
    """,
    re.VERBOSE,
)


@dataclass
class CitationVerdict:
    verified: List[Tuple[str, int]] = field(default_factory=list)
    unverified: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": [list(t) for t in self.verified],
            "unverified": [list(t) for t in self.unverified],
            "verified_count": len(self.verified),
            "unverified_count": len(self.unverified),
        }


def extract_cited_author_year(text: str) -> List[Tuple[str, int]]:
    """Return the list of (author_surname, year) tuples cited in `text`.

    Duplicates are preserved in the order they appear so callers can count
    occurrences if they wish.
    """
    if not text:
        return []
    out: List[Tuple[str, int]] = []
    for m in _CITATION_RE.finditer(text):
        author = m.group("author")
        try:
            year = int(m.group("year"))
        except ValueError:
            continue
        out.append((author, year))
    return out


def _study_citation_keys(studies: Iterable[Any]) -> Set[Tuple[str, int]]:
    """Build a set of (first_author_lower, year) tuples from retrieved studies.

    Accepts duck-typed objects: dicts or dataclasses exposing any of
    `citation` / `authors` / `first_author` / `year`. We extract as
    permissively as possible — the goal is a generous key set so legitimate
    citations are never dropped.
    """
    keys: Set[Tuple[str, int]] = set()
    for s in studies or []:
        get = (
            (lambda k, _s=s: _s.get(k)) if isinstance(s, dict)
            else (lambda k, _s=s: getattr(_s, k, None))
        )
        year = get("year")
        citation = get("citation") or ""
        authors = get("authors") or []
        first_author = get("first_author")

        candidates: List[str] = []
        if first_author:
            candidates.append(str(first_author))
        if isinstance(authors, (list, tuple)) and authors:
            candidates.append(str(authors[0]))
        if citation:
            # Best-effort: pull the leading capitalised token from the citation string
            m = re.match(
                r"\s*([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-']{2,})", str(citation)
            )
            if m:
                candidates.append(m.group(1))

        if not year and citation:
            m = re.search(r"(19|20)\d{2}", str(citation))
            if m:
                try:
                    year = int(m.group(0))
                except ValueError:
                    year = None

        if year is None:
            continue
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue

        for cand in candidates:
            # Strip punctuation and take the first whitespace-separated token.
            # Preserve accented Latin letters so "García-López" stays intact.
            cand = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\-']", " ", cand).strip()
            if not cand:
                continue
            surname = cand.split()[0]
            keys.add((surname.lower(), year_int))
    return keys


def verify_citations(text: str, studies: Iterable[Any]) -> CitationVerdict:
    """Split cited (author, year) tuples into verified vs. unverified."""
    cited = extract_cited_author_year(text)
    study_keys = _study_citation_keys(studies)
    verified: List[Tuple[str, int]] = []
    unverified: List[Tuple[str, int]] = []
    for author, year in cited:
        if (author.lower(), year) in study_keys:
            verified.append((author, year))
        else:
            unverified.append((author, year))
    return CitationVerdict(verified=verified, unverified=unverified)


def strip_unverified_citations(text: str, studies: Iterable[Any]) -> str:
    """Replace unverified (author, year) citations with `[unverified]`.

    Bumps `pipeline_metrics.safety.citations_stripped` by the number of
    citations rewritten when a metrics context is active.
    """
    if not text:
        return text
    study_keys = _study_citation_keys(studies)
    stripped = 0

    def _sub(m: re.Match) -> str:
        nonlocal stripped
        author = m.group("author")
        try:
            year = int(m.group("year"))
        except ValueError:
            return m.group(0)
        if (author.lower(), year) in study_keys:
            return m.group(0)
        stripped += 1
        return "[unverified]"

    rewritten = _CITATION_RE.sub(_sub, text)

    if stripped:
        try:
            from src.api.services import pipeline_metrics as _pm
            pm = _pm.current()
            if pm is not None:
                pm.incr("safety", "citations_stripped", stripped)
        except Exception:
            pass

    return rewritten


__all__ = [
    "CitationVerdict",
    "extract_cited_author_year",
    "strip_unverified_citations",
    "verify_citations",
]
