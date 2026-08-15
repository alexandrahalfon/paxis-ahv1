"""
Offline unit tests for src.api.services.safety.citations.

These exercise the citation extractor and the strip-unverified pass without
any live infrastructure, so they run in the normal pytest suite.
"""

from __future__ import annotations

from src.api.services.safety.citations import (
    extract_cited_author_year,
    strip_unverified_citations,
    verify_citations,
)


# ──────────────────────────────────────────────────────────────────────────────
# extract_cited_author_year
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_author_year_simple():
    out = extract_cited_author_year("Smith et al., 2023 reported ORR 42%.")
    assert ("Smith", 2023) in out


def test_extract_author_year_variants():
    text = (
        "Background per Smith 2018 and follow-up by Jones et al. 2022. "
        "See also (García-López, 2021)."
    )
    authors_years = extract_cited_author_year(text)
    assert ("Smith", 2018) in authors_years
    assert ("Jones", 2022) in authors_years
    assert ("García-López", 2021) in authors_years


def test_extract_ignores_lowercase_tokens_and_non_years():
    out = extract_cited_author_year("the study showed a 1 2-fold increase in 99.")
    assert out == []


# ──────────────────────────────────────────────────────────────────────────────
# verify_citations
# ──────────────────────────────────────────────────────────────────────────────


def test_verify_citations_splits_verified_vs_unverified():
    studies = [
        {"citation": "Smith JA et al.", "year": 2023},
        {"first_author": "García-López M", "year": 2021},
    ]
    verdict = verify_citations(
        "Results per Smith et al., 2023 and García-López 2021. "
        "Contradicted by Fake 2019.",
        studies,
    )
    assert ("Smith", 2023) in verdict.verified
    assert ("García-López", 2021) in verdict.verified
    assert ("Fake", 2019) in verdict.unverified


# ──────────────────────────────────────────────────────────────────────────────
# strip_unverified_citations
# ──────────────────────────────────────────────────────────────────────────────


def test_strip_unverified_replaces_only_unknowns():
    studies = [{"citation": "Smith JA et al.", "year": 2023}]
    text = "Per Smith et al., 2023 ORR was 50%. Fake 2019 disagreed."
    out = strip_unverified_citations(text, studies)
    assert "Smith et al., 2023" in out
    assert "[unverified]" in out
    assert "Fake 2019" not in out


def test_strip_unverified_empty_text_is_noop():
    assert strip_unverified_citations("", [{"citation": "x", "year": 2020}]) == ""
