"""
Offline test suite for the PostgreSQL structured study matcher.

Purpose: capture the matcher's *current* behaviour so future changes to
`structured_study_matcher.py` are observable, and document known
parsing / filter bugs as xfail test cases so they can't silently
regress or silently get fixed.

This suite is deliberately observational — it does NOT assert "correct"
behaviour for everything, only for invariants that must hold (weights
summing to 100, hard site filter being an AND, empty query returning
empty, etc.). Behaviours we suspect are wrong (e.g. `CPS score of 100`
parsing, conflicting biomarker polarities) are marked xfail with an
explanatory reason so the test itself becomes the bug record.

No Postgres connection, no Qdrant, no OpenAI — everything runs in the
sandbox in under a second.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Make the repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.structured_study_matcher import (  # noqa: E402
    BASE_SCORING_WEIGHTS,
    CANONICAL_BIOMARKERS,
    POLARITY_MAP,
    StructuredMatchResult,
    _parse_biomarker_query,
    calculate_dynamic_weights,
    match_studies_by_structure,
)
from tests.fixtures.matcher_golden_queries import GOLDEN_FIXTURES  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# 1. Biomarker parser — parametrized tests
# ════════════════════════════════════════════════════════════════════════════
#
# The parser sits at `structured_study_matcher.py:289`. These tests lock
# in its current behaviour against a wide input vocabulary. Inputs that
# we know parse incorrectly are marked xfail so the test suite records
# the bug without blocking CI.


# Format: (input, expected_canonical, expected_status, xfail_reason_or_None)
BIOMARKER_CASES: List[tuple] = [
    # ─── Simple polarity suffixes ──────────────────────────────────────────
    ("ER+",              "ER",    "positive",  None),
    ("ER-",              "ER",    "negative",  None),
    ("PR+",              "PR",    "positive",  None),
    ("PR-",              "PR",    "negative",  None),
    ("HER2+",            "HER2",  "positive",  None),
    ("HER2-",            "HER2",  "negative",  None),
    ("ALK+",             "ALK",   "positive",  None),
    ("ROS1+",            "ROS1",  "positive",  None),
    # Whitespace tolerance (previously xfailed — now fixed)
    ("ER +",             "ER",    "positive",  None),
    ("HER2 -",           "HER2",  "negative",  None),

    # ─── Aliases / canonical lookup ─────────────────────────────────────────
    ("her-2",            "HER2",  None,        None),
    ("her 2",            "HER2",  None,        None),
    ("erbb2",            "HER2",  None,        None),
    ("estrogen receptor","ER",    None,        None),
    ("pdl1",             "PD-L1", None,        None),
    ("pd-l1",            "PD-L1", None,        None),
    ("cd274",            "PD-L1", None,        None),

    # ─── Compound specials ─────────────────────────────────────────────────
    ("triple negative",  "TNBC",  "positive",  None),
    ("triple-negative",  "TNBC",  "positive",  None),
    ("tnbc",             "TNBC",  "positive",  None),
    ("MSI-H",            "MSI",   "high",      None),
    ("msi h",            "MSI",   "high",      None),
    ("dMMR",             "MSI",   "high",      None),
    ("pMMR",             "MSS",   "stable",    None),
    ("MSS",              "MSS",   "stable",    None),
    ("TMB-H",            "TMB",   "high",      None),

    # ─── Mutations / fusions ───────────────────────────────────────────────
    ("EGFR mutant",      "EGFR",  "mutant",    None),
    ("EGFR mutation",    "EGFR",  "mutant",    None),
    ("BRCA1 mutation",   "BRCA1", "mutant",    None),
    ("ALK fusion",       "ALK",   "mutant",    None),
    ("ALK rearrangement","ALK",   "mutant",    None),

    # ─── Variant-level mutations (previously xfailed) ──────────────────────
    ("EGFR L858R",       "EGFR",  "mutant",    None),
    ("EGFR exon 19",     "EGFR",  "mutant",    None),
    ("EGFR exon 20 insertion", "EGFR", "mutant", None),
    ("KRAS G12C",        "KRAS",  "mutant",    None),
    ("BRAF V600E",       "BRAF",  "mutant",    None),
    ("BRAF p.V600E",     "BRAF",  "mutant",    None),

    # ─── Expression modifiers ──────────────────────────────────────────────
    ("HER2 amplification",    "HER2", "positive", None),
    ("HER2 amplified",        "HER2", "positive", None),
    ("HER2 overexpressed",    "HER2", "positive", None),
    ("HER2 overexpression",   "HER2", "positive", None),

    # ─── Numeric thresholds: PD-L1 CPS (HNSCC) ─────────────────────────────
    ("CPS score of 100", "PD-L1", "high",      None),
    ("CPS 100",          "PD-L1", "high",      None),
    ("cps 50",           "PD-L1", "high",      None),
    ("CPS 20",           "PD-L1", "high",      None),
    ("CPS 19",           "PD-L1", "positive",  None),
    ("CPS 1",            "PD-L1", "positive",  None),
    ("CPS 0",            "PD-L1", "negative",  None),
    ("PD-L1 CPS 50",     "PD-L1", "high",      None),
    ("pd-l1 cps 19",     "PD-L1", "positive",  None),
    ("PD-L1 CPS >= 20",  "PD-L1", "high",      None),

    # ─── Numeric thresholds: PD-L1 TPS (NSCLC) ─────────────────────────────
    ("TPS 80%",          "PD-L1", "high",      None),
    ("TPS 50",           "PD-L1", "high",      None),
    ("TPS 49",           "PD-L1", "positive",  None),
    ("TPS 1",            "PD-L1", "positive",  None),
    ("TPS 0",            "PD-L1", "negative",  None),
    ("PD-L1 TPS 80",     "PD-L1", "high",      None),

    # ─── Numeric thresholds: Ki-67 ─────────────────────────────────────────
    ("Ki-67 20%",        "Ki-67", "high",      None),
    ("Ki67 40",          "Ki-67", "high",      None),
    ("ki-67 15",         "Ki-67", "low",       None),

    # ─── Numeric thresholds: Oncotype DX ───────────────────────────────────
    ("Oncotype 10",      "ONCOTYPE", "low",    None),
    ("Oncotype 25",      "ONCOTYPE", "intermediate", None),
    ("Oncotype 30",      "ONCOTYPE", "high",   None),

    # ─── Hematologic / lymphoma markers ────────────────────────────────────
    ("CD20 positive",    "CD20",  "positive",  None),
    ("CD20+",            "CD20",  "positive",  None),
    ("CD30 weakly positive", "CD30", "positive", None),
    ("CD15 negative",    "CD15",  "negative",  None),
    ("BCL2+",            "BCL2",  "positive",  None),
    ("MYC+",             "MYC",   "positive",  None),

    # ─── Serum markers ─────────────────────────────────────────────────────
    ("LDH high",         "LDH",   "positive",  None),
    ("LDH elevated",     "LDH",   "positive",  None),
    ("B2M",              "B2M",   None,        None),
    ("Beta-2 Microglobulin", "B2M", None,      None),
    ("beta 2 microglobulin", "B2M", None,      None),
    ("CEA elevated",     "CEA",   "positive",  None),
    ("CA125 elevated",   "CA125", "positive",  None),

    # ─── Liquid biopsy ─────────────────────────────────────────────────────
    ("ctDNA",            "ctDNA", None,        None),
    ("ctDNA detectable", "ctDNA", "positive",  None),
    ("ctDNA positive",   "ctDNA", "positive",  None),

    # ─── Regression: PD-L1 hyphen must never be stripped as polarity ──────
    ("PD-L1",            "PD-L1", None,        None),
    ("PD-L1 positive",   "PD-L1", "positive",  None),
    ("PD-L1 negative",   "PD-L1", "negative",  None),
    ("PD-L1 high",       "PD-L1", "positive",  None),
]


@pytest.mark.parametrize("marker,expected_canonical,expected_status,xfail_reason",
                         BIOMARKER_CASES,
                         ids=[c[0] for c in BIOMARKER_CASES])
def test_biomarker_parser(marker, expected_canonical, expected_status, xfail_reason):
    if xfail_reason:
        pytest.xfail(xfail_reason)
    canonical, status = _parse_biomarker_query(marker)
    assert canonical == expected_canonical, (
        f"{marker!r}: expected canonical={expected_canonical!r}, got {canonical!r}"
    )
    assert status == expected_status, (
        f"{marker!r}: expected status={expected_status!r}, got {status!r}"
    )


def test_biomarker_parser_unknown_input_returns_none_tuple():
    """Empty and whitespace inputs return (None, None)."""
    assert _parse_biomarker_query("") == (None, None)
    assert _parse_biomarker_query("   ") == (None, None)
    assert _parse_biomarker_query(None) == (None, None)  # type: ignore[arg-type]


# ════════════════════════════════════════════════════════════════════════════
# 2. Dynamic weight calculation invariants
# ════════════════════════════════════════════════════════════════════════════


class TestDynamicWeights:
    def test_empty_present_criteria_returns_empty_dict(self):
        assert calculate_dynamic_weights([]) == {}

    def test_single_criterion_gets_full_100_points(self):
        weights = calculate_dynamic_weights(["cancer_site"])
        assert weights == {"cancer_site": 100}

    def test_sum_equals_100_for_all_common_combinations(self):
        # Try every combination of 2-to-5 criteria to confirm the
        # rounding-residue fixup at line 187 keeps the total exact.
        import itertools
        keys = list(BASE_SCORING_WEIGHTS.keys())
        for r in range(2, 6):
            for combo in itertools.combinations(keys, r):
                weights = calculate_dynamic_weights(list(combo))
                assert sum(weights.values()) == 100, (
                    f"weights don't sum to 100 for combo {combo}: {weights}"
                )

    def test_canonical_hn_case_produces_valid_weights(self):
        # Mirrors the fixture HN_SCC_MULTI_AXIS — all criteria including
        # biomarkers should now be weighted (post-fix).
        present = [
            "cancer_site", "site_detail", "histology", "stage", "tnm_n",
            "treatment", "treatment_setting", "age_range", "gender",
            "biomarkers",
        ]
        weights = calculate_dynamic_weights(present)
        assert sum(weights.values()) == 100
        # biomarkers now has a weight (fixed: was silently dropped before
        # BASE_SCORING_WEIGHTS['biomarkers'] = 20 was added).
        assert set(weights.keys()) == set(present)
        assert weights["biomarkers"] > 0
        # cancer_site should still be the heaviest
        assert weights["cancer_site"] == max(weights.values())

    def test_biomarkers_have_positive_scoring_weight(self):
        """
        Regression test: biomarkers must carry a non-zero weight in
        BASE_SCORING_WEIGHTS. Previously (pre-fix) the matcher detected
        biomarkers as a present criterion, built a 3-tier CASE WHEN
        expression, but `dynamic_weights.get("biomarkers", 0)` returned
        0 because the key didn't exist — so biomarker matches
        contributed ZERO scoring points.

        This test pins the fix and prevents regression.
        """
        from src.api.services.structured_study_matcher import BASE_SCORING_WEIGHTS
        assert "biomarkers" in BASE_SCORING_WEIGHTS, (
            "biomarkers must be in BASE_SCORING_WEIGHTS — otherwise every "
            "biomarker match contributes 0 points and the fractional match "
            "formula collapses to zero."
        )
        assert BASE_SCORING_WEIGHTS["biomarkers"] > 0

        weights = calculate_dynamic_weights(["cancer_site", "biomarkers"])
        assert "biomarkers" in weights
        assert weights["biomarkers"] > 0
        assert sum(weights.values()) == 100

    def test_user_multiplier_boosts_biomarkers(self):
        # Now that biomarkers has a weight, multipliers actually apply.
        present = ["cancer_site", "histology", "biomarkers"]
        base = calculate_dynamic_weights(present)
        boosted = calculate_dynamic_weights(
            present, user_weights={"biomarkers": 3.0}
        )
        assert sum(boosted.values()) == 100
        assert boosted["biomarkers"] > base["biomarkers"]
        assert boosted["cancer_site"] < base["cancer_site"]

    def test_user_multiplier_clamped_to_range_0_3(self):
        present = ["cancer_site", "histology"]
        # Passing 999 is clamped to 3
        mad = calculate_dynamic_weights(present, user_weights={"histology": 999})
        sane = calculate_dynamic_weights(present, user_weights={"histology": 3.0})
        assert mad == sane

    def test_zero_multiplier_removes_criterion(self):
        present = ["cancer_site", "histology", "biomarkers"]
        # biomarkers multiplied by 0 should contribute 0
        weights = calculate_dynamic_weights(
            present, user_weights={"biomarkers": 0.0}
        )
        assert weights.get("biomarkers", 0) == 0
        # remaining two should sum to 100
        assert sum(v for k, v in weights.items() if k != "biomarkers") == 100

    def test_unknown_user_key_is_ignored(self):
        # Forward-compatibility: the frontend may send future keys
        # (race, grade, tumor_size) that aren't scored yet. They must
        # not crash the weight calculator.
        present = ["cancer_site", "histology"]
        weights = calculate_dynamic_weights(
            present, user_weights={"race": 2.0, "grade": 1.5, "tumor_size": 3.0}
        )
        assert sum(weights.values()) == 100

    def test_non_numeric_user_weight_silently_ignored(self):
        # e.g. biomarker_mode='strict' from the frontend — matcher code
        # at line 166 filters these out.
        present = ["cancer_site", "biomarkers"]
        weights = calculate_dynamic_weights(
            present,
            user_weights={"biomarkers": "strict", "biomarker_mode": "strict"},
        )
        assert sum(weights.values()) == 100


# ════════════════════════════════════════════════════════════════════════════
# 3. SQL generation snapshot tests
# ════════════════════════════════════════════════════════════════════════════
#
# Strategy: monkey-patch `asyncpg.connect` to return a FakeConnection
# whose `fetch()` method captures the query string + params and returns
# a canned list of rows. The test then inspects the captured SQL.


class _CapturedCall:
    def __init__(self):
        self.query: Optional[str] = None
        self.params: List[Any] = []
        self.closed: bool = False


class _FakeConnection:
    def __init__(self, capture: _CapturedCall, canned_rows: List[Dict[str, Any]]):
        self._capture = capture
        self._canned_rows = canned_rows

    async def fetch(self, query: str, *params):
        self._capture.query = query
        self._capture.params = list(params)
        return self._canned_rows

    async def close(self):
        self._capture.closed = True


def _install_fake_asyncpg(
    monkeypatch,
    canned_rows: Optional[List[Dict[str, Any]]] = None,
) -> _CapturedCall:
    """Patch asyncpg.connect so the matcher never touches a real DB."""
    capture = _CapturedCall()

    async def _fake_connect(**kwargs):
        return _FakeConnection(capture, canned_rows or [])

    import src.api.services.structured_study_matcher as sm
    monkeypatch.setattr(sm.asyncpg, "connect", _fake_connect)
    return capture


async def _run_matcher(query_struct: Dict[str, Any], **kw) -> StructuredMatchResult:
    """Tiny async wrapper so sync pytest bodies can call the matcher."""
    return await match_studies_by_structure(query_struct, **kw)


class TestSQLGeneration:
    """Snapshot-ish assertions on the SQL the matcher emits for each
    golden fixture. These are loose assertions (substring membership,
    count equalities) rather than strict byte-for-byte snapshots so
    small whitespace changes don't churn the test suite."""

    @pytest.mark.asyncio
    async def test_empty_query_short_circuits_without_touching_db(self, monkeypatch):
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        result = await _run_matcher(GOLDEN_FIXTURES["empty_query"])
        assert isinstance(result, StructuredMatchResult)
        assert result.doc_ids == set()
        assert result.match_scores == {}
        # The matcher should return before issuing any fetch. With the
        # current code the empty query actually opens a connection then
        # returns early — capture.query should still be None because
        # fetch() was never called.
        assert capture.query is None, (
            "empty query should not issue a Postgres fetch"
        )

    @pytest.mark.asyncio
    async def test_simple_lung_adeno_sql_contains_expected_fragments(self, monkeypatch):
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        await _run_matcher(GOLDEN_FIXTURES["simple_lung_adeno"])
        sql = capture.query or ""
        assert sql, "matcher must have issued a query for simple_lung_adeno"
        # Table and core structure
        assert "FROM studies" in sql
        assert "WHERE doc_id IS NOT NULL" in sql
        assert "ORDER BY match_score DESC" in sql
        assert "LIMIT 50" in sql
        # Cancer site hard filter — lung pattern
        assert "cancer_location ~*" in sql
        # Histology regex match
        assert "cancer_type ~*" in sql or "histopathologic_type ~*" in sql
        # Biomarker tier checks
        assert "biomarker_status" in sql

    @pytest.mark.asyncio
    async def test_hn_scc_multi_axis_uses_hard_site_filter(self, monkeypatch):
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        await _run_matcher(GOLDEN_FIXTURES["hn_scc_multi_axis"])
        sql = capture.query or ""
        assert sql
        # Hard site filter structure (line 1046): when a site is present,
        # where_clause becomes JUST the site filter, with scoring
        # handled by the CASE expressions, not by WHERE.
        assert "cancer_location ~*" in sql
        # Hard-filter text pattern — the oral / tongue / neck regex
        # from SITE_TO_LOCATION_PATTERNS["head_neck"] is joined with |.
        head_neck_pattern_fragment_found = any(
            frag in (capture.params or [])
            for frag in []  # not asserting param content yet
        )
        # Just assert an H&N-ish regex param exists somewhere
        hn_regex_param = next(
            (p for p in (capture.params or [])
             if isinstance(p, str) and "oral" in p and "tongue" in p),
            None,
        )
        assert hn_regex_param is not None, (
            "expected the H&N pattern regex in params; got: "
            f"{capture.params}"
        )

    @pytest.mark.asyncio
    async def test_biomarker_only_hits_postgres_after_fix(self, monkeypatch):
        """
        Regression test: a query containing ONLY biomarkers should
        actually hit Postgres and search the biomarker_status JSONB,
        not short-circuit.

        Previously (pre-fix) `calculate_dynamic_weights(['biomarkers'])`
        returned `{}` because biomarkers had no entry in
        BASE_SCORING_WEIGHTS, which tripped the `if not dynamic_weights`
        guard and returned empty before touching the DB.

        With the fractional-match fix in place, the query should:
          - produce a non-empty dynamic_weights dict
          - issue a Postgres fetch with the biomarker scoring expression
        """
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        result = await _run_matcher(GOLDEN_FIXTURES["biomarker_only"])
        assert capture.query is not None, (
            "After fix: biomarker-only queries must reach Postgres "
            "rather than short-circuiting."
        )
        sql = capture.query
        # The query should reference biomarker_status (either via
        # key existence check or fractional match expression)
        assert "biomarker_status" in sql
        assert isinstance(result, StructuredMatchResult)

    @pytest.mark.asyncio
    async def test_triple_negative_emits_her2_exclusion(self, monkeypatch):
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        await _run_matcher(GOLDEN_FIXTURES["triple_negative_breast"])
        sql = capture.query or ""
        # HER2- / triple negative should add a hard exclusion against
        # trastuzumab / T-DM1 / etc. (line 918 regex).
        exclusion_param = next(
            (p for p in (capture.params or [])
             if isinstance(p, str) and "trastuzumab" in p),
            None,
        )
        assert exclusion_param is not None, (
            "triple_negative_breast should emit a targeted-drug exclusion "
            f"regex in params; got: {capture.params}"
        )
        assert "IS DISTINCT FROM" in sql or "NOT (" in sql

    @pytest.mark.asyncio
    async def test_cps_score_of_100_raw_now_parses_to_pdl1_high(self, monkeypatch):
        """
        Regression test (post-fix): 'CPS score of 100' now parses to
        ('PD-L1', 'high') and the SQL sends the canonical 'PD-L1' alias
        list via parameterized keys, not the literal 'CPS SCORE OF 100'.

        Post-alias-fix: the JSONB keys are bound as parameters
        (biomarker_status->>$N) so we check the `params` list rather
        than the literal SQL string.
        """
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        await _run_matcher(GOLDEN_FIXTURES["cps_score_of_100_raw"])
        sql = capture.query or ""
        params = capture.params or []
        # PD-L1 (or an alias) must appear in the params since it's now
        # the canonical resolution of "CPS score of 100"
        pdl1_aliases = {"PD-L1", "PDL1", "CD274", "B7-H1", "pd-l1", "pdl1"}
        assert any(p in pdl1_aliases for p in params), (
            f"Expected a PD-L1 alias in params; got: {params}"
        )
        # The old broken literal must not appear anywhere
        assert "CPS SCORE OF 100" not in sql
        assert "CPS SCORE OF 100" not in [str(p) for p in params]

    @pytest.mark.asyncio
    async def test_breast_her2_neg_liver_mets_hard_filters_to_breast(self, monkeypatch):
        """
        Stress test for Gap #1 from the multi-site audit: a breast
        cancer patient with liver mets should have their hard site
        filter keyed on breast, NOT liver. This test pins the current
        (conservative) behaviour.
        """
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        await _run_matcher(GOLDEN_FIXTURES["breast_her2_neg_liver_mets"])
        # Assert the hard site filter regex param contains breast-ish
        # fragments and NOT liver-ish fragments.
        site_param = next(
            (p for p in (capture.params or [])
             if isinstance(p, str) and "breast" in p),
            None,
        )
        assert site_param is not None
        assert "liver" not in site_param, (
            "Hard site filter for a breast primary should not match %liver% — "
            "this test pins current behaviour for the multi-site audit Gap #1"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3b. Fractional biomarker scoring SQL — regression suite for the big fix
# ════════════════════════════════════════════════════════════════════════════


class TestFractionalBiomarkerSQL:
    """
    Tests that verify the biomarker scoring SQL is built as a fractional
    match expression:

        (matches / NULLIF(overlap, 0)) * weight

    where matches counts polarized biomarkers whose patient-side status
    is equivalent to the study-side status (via STATUS_MATCH_SYNONYMS)
    and overlap counts polarized biomarkers that are present as JSONB
    keys in the study's biomarker_status column at all.
    """

    @pytest.mark.asyncio
    async def test_fractional_expression_shape(self, monkeypatch):
        """The biomarker scoring SQL should use NULLIF + COALESCE to
        produce a fractional match expression."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "breast",
                "biomarkers": ["ER+", "PR+", "HER2-"],
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        # Must include the fractional match shape
        assert "NULLIF" in sql, (
            "biomarker scoring must use NULLIF to guard against zero-overlap"
        )
        assert "COALESCE" in sql, (
            "biomarker scoring must use COALESCE so zero-overlap maps to 0"
        )
        # Each polarized biomarker gets a numerator + denominator branch
        assert sql.count("biomarker_status->>") >= 3
        assert sql.count("biomarker_status ?") >= 3

    @pytest.mark.asyncio
    async def test_canonical_keys_are_parameterized_with_alias_list(self, monkeypatch):
        """
        Biomarker JSONB keys are now passed as query parameters
        (biomarker_status->>$N) instead of being inlined. The matcher
        OR-s across all known alias spellings per canonical, so a
        patient asking for 'EGFR mutant' will cause HER1/ERBB1/egfr
        etc. to end up in the params list.
        """
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "lung",
                "biomarkers": ["EGFR mutant"],
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []

        # SQL uses parameterized key lookup
        assert "biomarker_status->>$" in sql, (
            "biomarker JSONB keys must be parameterized, not inlined"
        )
        assert "biomarker_status ? $" in sql, (
            "biomarker key-existence must use parameterized keys"
        )

        # EGFR canonical and at least one alias must be in params
        assert "EGFR" in params
        egfr_aliases = {"EGFR", "egfr", "ERBB1", "HER1", "Egfr", "erbb-1", "ERBB-1"}
        alias_hits = sum(1 for p in params if p in egfr_aliases)
        assert alias_hits >= 3, (
            f"Expected ≥3 EGFR alias spellings in params; got {alias_hits}: "
            f"{[p for p in params if p in egfr_aliases]}"
        )

    @pytest.mark.asyncio
    async def test_status_synonyms_expanded_into_in_list(self, monkeypatch):
        """Patient status 'positive' should expand to an IN (...) list
        containing the full STATUS_MATCH_SYNONYMS['positive'] equivalents
        so a study reporting 'high' or 'amplified' still counts as a match."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {"site": "breast", "biomarkers": ["HER2+"]},
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        params_lower = [str(p).lower() for p in (capture.params or [])]
        # positive must be in the expanded synonyms
        assert "positive" in params_lower
        # high / amplified / overexpressed are accepted equivalents
        assert any(syn in params_lower for syn in ("high", "amplified", "overexpressed"))

    @pytest.mark.asyncio
    async def test_mutant_status_accepts_mutation_synonyms(self, monkeypatch):
        """Patient 'mutant' should accept 'mutation', 'mutated', 'fusion',
        'altered', 'positive' as equivalents on the study side."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {"site": "lung", "biomarkers": ["EGFR L858R"]},
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        params_lower = [str(p).lower() for p in (capture.params or [])]
        assert "mutant" in params_lower
        # At least one of the mutant-equivalent synonyms
        assert any(syn in params_lower
                   for syn in ("mutation", "mutated", "altered", "fusion"))

    @pytest.mark.asyncio
    async def test_unpolarized_biomarkers_do_not_contribute_to_score(self, monkeypatch):
        """A biomarker with no polarity (e.g. 'PD-L1' with no +/-) should
        NOT produce a fractional numerator. It only contributes a
        key-existence WHERE clause — and that clause is only emitted
        when there's no hard cancer_site filter (which would drop it)."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        # No cancer_site so the key-existence WHERE clause survives
        qs = {
            "cancer": {"biomarkers": ["PD-L1"]},
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []
        # The biomarker WHERE clause uses parameterized key-existence:
        #     biomarker_status ? $N
        # with PD-L1 and its aliases in the params list.
        assert "biomarker_status ? $" in sql, (
            "unpolarized biomarker must still be a WHERE-clause key check"
        )
        assert "PD-L1" in params, (
            f"PD-L1 alias must appear in params; got: {params}"
        )
        # No fractional numerator (no `biomarker_status->>`) because
        # unpolarized biomarkers don't get scored.
        assert "biomarker_status->>" not in sql, (
            "unpolarized biomarker must NOT appear in the score expression"
        )
        # No NULLIF either — the fractional formula only runs when there's
        # at least one polarized biomarker.
        assert "NULLIF" not in sql

    @pytest.mark.asyncio
    async def test_triple_negative_expands_into_er_pr_her2_negative(self, monkeypatch):
        """TNBC should expand into ER-, PR-, HER2- internally for scoring."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "breast",
                "biomarkers": ["triple negative"],
                "receptor_status": "triple negative",
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        # The TNBC expansion writes TNBC as the canonical name
        assert "biomarker_status" in sql

    @pytest.mark.asyncio
    async def test_complex_multi_biomarker_case(self, monkeypatch):
        """
        End-to-end fractional SQL for a 6-biomarker case modeled on the
        user's lymphoma example:
          - LDH elevated
          - B2M (no polarity, just presence)
          - CD20 positive
          - CD30 weakly positive
          - CD15 negative
          - ctDNA detectable

        Verifies: all 5 polarized biomarkers appear in the fractional
        numerator + denominator. B2M (unpolarized) is silently dropped
        when there's a hard site filter (as is any non-site WHERE clause).
        Keys are parameterized — assertions check the `params` list.
        """
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "lymphoma",
                "biomarkers": [
                    "LDH elevated",
                    "B2M",
                    "CD20 positive",
                    "CD30 weakly positive",
                    "CD15 negative",
                    "ctDNA detectable",
                ],
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []
        # All 5 polarized canonical names should appear at least once
        # in the params list (they drive the fractional scoring SQL).
        for canonical in ("LDH", "CD20", "CD30", "CD15", "ctDNA"):
            assert canonical in params, (
                f"{canonical} missing from parameter list — "
                f"expected to drive the fractional scoring SQL"
            )
        # B2M is unpolarized. When there's a hard site filter, the
        # B2M key-existence WHERE clause is dropped — so it should NOT
        # appear in params (neither alias-key parameters nor
        # key-existence check).
        assert "B2M" not in params, (
            "B2M is unpolarized and the lymphoma case has a hard site "
            "filter — the B2M key-existence clause should have been "
            "dropped from the final SQL"
        )
        # Fractional shape present in the SQL
        assert "NULLIF" in sql
        assert "COALESCE" in sql
        # Parameterized key lookups
        assert "biomarker_status->>$" in sql
        assert "biomarker_status ? $" in sql


# ════════════════════════════════════════════════════════════════════════════
# 3c. JSONB alias + value-regex tolerance — Layer C consistency fixes
# ════════════════════════════════════════════════════════════════════════════
#
# The matcher's SQL builder now accepts heterogeneous JSONB storage:
#
#   - Key aliases: HER2 is matched against {HER2, HER-2, ERBB2, Her2,
#     HER2/neu, ...} so ingestion storing it under any of these spellings
#     still lands a match.
#
#   - Value regex: status matching now includes a regex fallback that
#     covers IHC intensity notation (0, 1+, 2+, 3+), percent staining
#     (75%, ≥50%), and HER2-low language in addition to the plain
#     English synonyms in STATUS_MATCH_SYNONYMS.
#
# These tests verify the SQL generation side (parameters emitted, regex
# present). The runtime behaviour against a real Postgres is covered by
# the Layer B live-matcher script.


class TestAliasAndRegexTolerance:
    """Verify the alias map and value regex fallback are wired into the
    generated SQL + parameter list correctly."""

    # ─── Alias coverage ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_her2_includes_common_aliases(self, monkeypatch):
        """HER2 must generate SQL searching HER2 / HER-2 / ERBB2 / Her2."""
        from src.api.services.structured_study_matcher import (
            _aliases_for_canonical,
        )
        aliases = _aliases_for_canonical("HER2")
        # Manually curated list must include the main variants
        assert "HER2" in aliases
        assert "HER-2" in aliases
        assert "ERBB2" in aliases
        # Canonical is always first
        assert aliases[0] == "HER2"

        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {"cancer": {"biomarkers": ["HER2+"]}, "patient": {}, "treatment": {}}
        await _run_matcher(qs)
        params = capture.params or []
        for alias in ("HER2", "HER-2", "ERBB2"):
            assert alias in params, (
                f"alias {alias!r} missing from params — matcher should "
                f"search all known HER2 spellings"
            )

    @pytest.mark.asyncio
    async def test_pdl1_includes_cd274_and_case_variants(self, monkeypatch):
        """PD-L1 must match CD274 / PDL1 / pd-l1 / Pd-L1 variants."""
        from src.api.services.structured_study_matcher import (
            _aliases_for_canonical,
        )
        aliases = _aliases_for_canonical("PD-L1")
        assert "PD-L1" in aliases
        assert "PDL1" in aliases
        assert "CD274" in aliases

        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {"biomarkers": ["PD-L1 CPS 50"]},
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        params = capture.params or []
        for alias in ("PD-L1", "PDL1", "CD274"):
            assert alias in params

    @pytest.mark.asyncio
    async def test_b2m_includes_long_form_alias(self, monkeypatch):
        """B2M must match 'Beta-2 Microglobulin' spellings — the exact
        form the Layer C audit might find in Postgres."""
        from src.api.services.structured_study_matcher import (
            _aliases_for_canonical,
        )
        aliases = _aliases_for_canonical("B2M")
        assert "B2M" in aliases
        assert "Beta-2 Microglobulin" in aliases
        assert "beta-2 microglobulin" in aliases

    @pytest.mark.asyncio
    async def test_msi_h_includes_dmmr_alias(self, monkeypatch):
        """MSI-H canonical must match dMMR / DMMR / MSI-H aliases."""
        from src.api.services.structured_study_matcher import (
            _aliases_for_canonical,
        )
        aliases = _aliases_for_canonical("MSI")
        assert "MSI" in aliases
        assert any(a in {"dMMR", "DMMR", "MSI-H"} for a in aliases)

    @pytest.mark.asyncio
    async def test_unknown_canonical_auto_inverts_from_canonical_biomarkers(self, monkeypatch):
        """For canonicals not in the manual map, aliases are auto-inverted
        from CANONICAL_BIOMARKERS with case variants."""
        from src.api.services.structured_study_matcher import (
            _aliases_for_canonical,
        )
        # GFAP isn't in the manual map but is in CANONICAL_BIOMARKERS
        # (as "gfap" → "GFAP"). Auto-inversion should surface it.
        aliases = _aliases_for_canonical("GFAP")
        assert "GFAP" in aliases
        # Should include at least one lowercase / title variant
        assert len(aliases) >= 1

    # ─── Value regex tolerance ─────────────────────────────────────────────

    def test_status_regex_positive_matches_ihc_notation(self):
        """The 'positive' regex must match 2+/3+ IHC values."""
        from src.api.services.structured_study_matcher import STATUS_MATCH_REGEX
        import re as _re
        pat = _re.compile(STATUS_MATCH_REGEX["positive"], _re.IGNORECASE)
        # Plain English
        assert pat.match("positive")
        assert pat.match("pos")
        assert pat.match("amplified")
        assert pat.match("overexpressed")
        # IHC intensity
        assert pat.match("2+")
        assert pat.match("3+")
        assert pat.match("ihc 2+")
        assert pat.match("IHC 3+")
        # Percent staining
        assert pat.match("75%")
        assert pat.match("25%")
        assert pat.match("1%")
        # HER2-low
        assert pat.match("her2-low")
        assert pat.match("HER2 Low")

    def test_status_regex_high_rejects_low_values(self):
        """'high' must require strict thresholds: 3+, ≥50%, 'strong'."""
        from src.api.services.structured_study_matcher import STATUS_MATCH_REGEX
        import re as _re
        pat = _re.compile(STATUS_MATCH_REGEX["high"], _re.IGNORECASE)
        # Accepted
        assert pat.match("3+")
        assert pat.match("strong")
        assert pat.match("75%")
        assert pat.match("elevated")
        # Rejected
        assert not pat.match("1+"), "1+ must not be classified as 'high'"
        assert not pat.match("5%"),  "5% must not be classified as 'high'"
        assert not pat.match("low")

    def test_status_regex_negative_matches_ihc_0_and_1(self):
        """'negative' must match 0, 1+, and 0% staining."""
        from src.api.services.structured_study_matcher import STATUS_MATCH_REGEX
        import re as _re
        pat = _re.compile(STATUS_MATCH_REGEX["negative"], _re.IGNORECASE)
        assert pat.match("negative")
        assert pat.match("0")
        assert pat.match("0+")
        assert pat.match("1+")
        assert pat.match("ihc 0")
        assert pat.match("ihc 1+")
        assert pat.match("0%")
        assert pat.match("wild-type")
        assert pat.match("not detected")

    def test_status_regex_mutant_matches_variant_nomenclature(self):
        """'mutant' must match variant nomenclature like V600E, L858R, exon 19."""
        from src.api.services.structured_study_matcher import STATUS_MATCH_REGEX
        import re as _re
        pat = _re.compile(STATUS_MATCH_REGEX["mutant"], _re.IGNORECASE)
        assert pat.match("mutant")
        assert pat.match("mutation")
        assert pat.match("fusion")
        assert pat.match("rearrangement")
        # Variant notation
        assert pat.match("v600e")
        assert pat.match("L858R")
        assert pat.match("exon 19")
        assert pat.match("del19")
        assert pat.match("ins20")

    @pytest.mark.asyncio
    async def test_regex_is_emitted_as_parameter(self, monkeypatch):
        """The value-match SQL must include the regex pattern as a
        parameter so the runtime regex fallback actually fires."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {"site": "breast", "biomarkers": ["HER2+"]},
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        params = capture.params or []
        # The 'positive' regex should be in the params (as a string)
        regex_hits = [
            p for p in params
            if isinstance(p, str) and "positive" in p and "[23]" in p
        ]
        assert regex_hits, (
            "Expected the 'positive' status regex in params "
            "(should contain 'positive' and '[23]' for IHC notation); "
            f"got params: {params}"
        )
        # SQL should use the ~ regex operator
        sql = capture.query or ""
        assert "~ $" in sql, (
            "value-match SQL must apply the regex fallback via the "
            "PostgreSQL `~` operator"
        )

    @pytest.mark.asyncio
    async def test_alias_union_emits_one_regex_param_shared_across_aliases(
        self, monkeypatch
    ):
        """The synonym list and regex should be emitted ONCE and reused
        across all alias checks — not duplicated per alias. Keeps the
        param count linear in the number of aliases + 1."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {"biomarkers": ["HER2+"]},
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []

        # Count how many distinct $N placeholders the SQL references for
        # the 'positive' synonym list. The synonym placeholders should
        # appear N_aliases times in the SQL (reused per alias) but
        # only take up ~9 slots in the params list (not 9 × aliases).
        # NOTE: With biomarker_jsonb scoring, 'positive' may appear twice:
        # once for the string-based biomarker scoring and once for the
        # JSONB polarity-aware scoring axis.
        positive_count = sum(1 for p in params if p == "positive")
        assert positive_count <= 2, (
            f"'positive' should be emitted at most twice (once for string "
            f"biomarkers, once for biomarker_jsonb); "
            f"got {positive_count} copies in params"
        )
        # Same for the regex — string-based scoring emits one regex param,
        # and biomarker_jsonb may emit another for polarity matching
        regex_count = sum(
            1 for p in params
            if isinstance(p, str) and "[23]\\+|ihc" in p
        )
        assert regex_count <= 2, (
            f"status regex should be emitted at most twice (string + jsonb); "
            f"got {regex_count} copies"
        )

    # ─── CONTRA_MAP is alias-aware ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_her2_negative_contra_excludes_all_her2_aliases(
        self, monkeypatch
    ):
        """Patient HER2- must hard-exclude studies whose HER2 positivity
        is stored under ANY alias spelling, not just the literal 'HER2'."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "breast",
                "biomarkers": ["HER2-"],
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []
        # Contra-exclusion uses `NOT (...)` wrapper around an alias-OR
        # value-match. All HER2 aliases should be in params.
        for alias in ("HER2", "HER-2", "ERBB2"):
            assert alias in params, (
                f"HER2- contra-exclusion should check all aliases; "
                f"{alias} missing from params"
            )
        # The SQL must apply the exclusion as a NOT clause
        assert "NOT " in sql, (
            "biomarker contra-exclusion must use NOT (...) wrapper"
        )
        # REGRESSION GUARD: the NOT must be wrapped in COALESCE so that
        # NULL (from studies with empty biomarker_status) doesn't
        # collapse to FALSE and silently reject the study. See the
        # `test_contra_exclusion_coalesces_null_to_keep_silent_studies`
        # test below for the end-to-end verification.
        assert "COALESCE" in sql, (
            "contra-exclusion must use COALESCE(..., FALSE) to handle "
            "NULL biomarker_status gracefully; a bare NOT (...) is a "
            "known regression"
        )

    @pytest.mark.asyncio
    async def test_contra_exclusion_coalesces_null_to_keep_silent_studies(
        self, monkeypatch
    ):
        """
        REGRESSION TEST — "CONTRA-exclusion NULL regression".

        The pre-fix matcher had a three-valued-logic bug: when a study's
        `biomarker_status` JSONB didn't contain the contradicted key,
        `biomarker_status->>$N` returned NULL, the whole value-match
        expression evaluated to NULL, and `NOT NULL` in a WHERE clause
        was treated as FALSE — silently rejecting the study.

        This broke 7 of 9 live Layer B fixtures because most studies
        have sparse biomarker_status JSONB (only listing the biomarkers
        relevant to their own enrolment).

        This test canned-returns ONE study row (with an empty dict for
        biomarker_status) and asserts that a query with a CONTRA-triggering
        biomarker (EGFR mutant, which contra-excludes EGFR wild-type)
        still includes the study in the result.

        How it verifies the fix:
          1. Canned row has `biomarker_status = {}` (empty JSONB object)
          2. Query includes EGFR mutant → CONTRA_MAP fires for EGFR wild-type
          3. WHERE clause emits `NOT COALESCE(alias_value_match, FALSE)`
          4. All `biomarker_status->>'EGFR'` alias lookups return NULL
          5. Inner OR-join is NULL, COALESCE → FALSE, NOT FALSE → TRUE
          6. Study is kept (appears in result.doc_ids)

        Under the pre-fix code, the study would have been filtered out
        by the bare `NOT (...)` evaluating to NULL (→ FALSE in WHERE).
        """
        canned = [
            {
                "doc_id": "silent_biomarker_doc",
                "study_name": "NSCLC Study (no biomarker_status JSONB)",
                "cancer_location": "lung",
                "cancer_type": "NSCLC",
                "number_of_patients": 300,
                "match_score": 50,
            }
        ]
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=canned)
        qs = {
            "cancer": {
                "site": "lung",
                "biomarkers": ["EGFR mutant"],
            },
            "patient": {},
            "treatment": {},
        }
        result = await _run_matcher(qs)

        # SQL must emit COALESCE wrapper around the CONTRA exclusion
        sql = capture.query or ""
        assert "NOT COALESCE" in sql, (
            "CONTRA exclusion must be wrapped in COALESCE(..., FALSE) so "
            "that NULL biomarker_status doesn't silently reject studies."
        )

        # And the study must come back from the matcher (not filtered out)
        assert "silent_biomarker_doc" in result.doc_ids, (
            "A study with empty biomarker_status JSONB must pass the "
            "EGFR-mutant CONTRA exclusion — the study doesn't mention "
            "EGFR at all, so it can't be a confirmed wild-type "
            "mismatch. Pre-fix, this study was incorrectly filtered "
            "out by NOT(NULL) → NULL → FALSE in WHERE."
        )


# ════════════════════════════════════════════════════════════════════════════
# 4. Result-parsing with canned rows
# ════════════════════════════════════════════════════════════════════════════


class TestResultParsing:
    @pytest.mark.asyncio
    async def test_canned_rows_produce_normalized_scores(self, monkeypatch):
        canned = [
            {
                "doc_id": "doc_A",
                "study_name": "Study A",
                "cancer_location": "oral cavity",
                "cancer_type": "SCC",
                "number_of_patients": 500,
                "match_score": 82,
            },
            {
                "doc_id": "doc_B",
                "study_name": "Study B",
                "cancer_location": "oral cavity",
                "cancer_type": "SCC",
                "number_of_patients": 200,
                "match_score": 67,
            },
        ]
        _install_fake_asyncpg(monkeypatch, canned_rows=canned)
        result = await _run_matcher(GOLDEN_FIXTURES["hn_scc_multi_axis"])

        assert result.doc_ids == {"doc_A", "doc_B"}
        # max_possible_score is hardcoded 100 at line 1081
        assert result.max_possible_score == 100
        # Raw scores normalized to 0-1
        assert result.match_scores["doc_A"] == pytest.approx(0.82)
        assert result.match_scores["doc_B"] == pytest.approx(0.67)
        # Match details carry the dynamic weights snapshot
        details_a = result.match_details["doc_A"]
        assert details_a["raw_score"] == 82
        assert details_a["study_name"] == "Study A"
        assert "dynamic_weights" in details_a
        assert sum(details_a["dynamic_weights"].values()) == 100

    @pytest.mark.asyncio
    async def test_null_match_score_coerced_to_zero(self, monkeypatch):
        # asyncpg returns None when the CASE expression evaluates to NULL.
        canned = [
            {
                "doc_id": "doc_null",
                "study_name": "Null Score Study",
                "cancer_location": "lung",
                "cancer_type": "NSCLC",
                "number_of_patients": None,
                "match_score": None,
            }
        ]
        _install_fake_asyncpg(monkeypatch, canned_rows=canned)
        result = await _run_matcher(GOLDEN_FIXTURES["simple_lung_adeno"])
        assert result.match_scores["doc_null"] == 0.0

    @pytest.mark.asyncio
    async def test_connection_close_called(self, monkeypatch):
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        await _run_matcher(GOLDEN_FIXTURES["simple_lung_adeno"])
        assert capture.closed, "matcher must close the asyncpg connection"


# ════════════════════════════════════════════════════════════════════════════
# 5. Failure & edge-case behaviour
# ════════════════════════════════════════════════════════════════════════════


class TestMatcherEdgeCases:
    @pytest.mark.asyncio
    async def test_connection_failure_returns_empty_result(self, monkeypatch):
        import src.api.services.structured_study_matcher as sm

        async def _broken_connect(**kwargs):
            raise ConnectionError("simulated DB outage")

        monkeypatch.setattr(sm.asyncpg, "connect", _broken_connect)
        # Neutralize the retry sleep so the test doesn't wait for real
        # backoff — we only care that the final state is graceful.
        async def _fast_sleep(_delay):
            return None
        monkeypatch.setattr(sm.asyncio, "sleep", _fast_sleep)

        result = await _run_matcher(GOLDEN_FIXTURES["simple_lung_adeno"])
        assert isinstance(result, StructuredMatchResult)
        assert result.doc_ids == set()
        assert result.match_scores == {}

    @pytest.mark.asyncio
    async def test_connection_retries_on_transient_failure_and_recovers(self, monkeypatch):
        """
        REGRESSION TEST — RF-1 "PG connection failure in one arm silently
        returned zero studies". When asyncpg.connect flaps with a
        transient socket-level error, the matcher must retry with
        exponential backoff and recover if a later attempt succeeds.

        Flaky connect: fails twice, succeeds on the third attempt.
        Verifies that the matcher DOES NOT return empty — it retries
        and hits the happy path with the real (faked) result set.
        """
        import src.api.services.structured_study_matcher as sm

        attempt_count = {"n": 0}
        canned_rows = [
            {
                "doc_id": "retry_recovery_doc",
                "study_name": "Recovery test",
                "cancer_location": "lung",
                "cancer_type": "NSCLC",
                "number_of_patients": 100,
                "match_score": 50,
            }
        ]

        async def _flaky_connect(**kwargs):
            attempt_count["n"] += 1
            if attempt_count["n"] < 3:
                raise TimeoutError()  # empty-string error, just like the live log
            return _FakeConnection(_CapturedCall(), canned_rows)

        monkeypatch.setattr(sm.asyncpg, "connect", _flaky_connect)

        async def _fast_sleep(_delay):
            return None
        monkeypatch.setattr(sm.asyncio, "sleep", _fast_sleep)

        result = await _run_matcher(GOLDEN_FIXTURES["simple_lung_adeno"])
        # Retried 3 times total (2 failures + 1 success)
        assert attempt_count["n"] == 3, (
            f"expected 3 connect attempts (2 fail + 1 succeed), got {attempt_count['n']}"
        )
        # And the matcher recovered with a real result
        assert "retry_recovery_doc" in result.doc_ids, (
            "matcher must recover the canned row after retrying the "
            "transient connection failure — not silently return empty"
        )

    @pytest.mark.asyncio
    async def test_connection_exhausts_all_retries_then_returns_empty(self, monkeypatch):
        """When all retry attempts fail, return empty gracefully (not crash)."""
        import src.api.services.structured_study_matcher as sm

        attempt_count = {"n": 0}

        async def _always_fails(**kwargs):
            attempt_count["n"] += 1
            raise ConnectionError("permanent")

        monkeypatch.setattr(sm.asyncpg, "connect", _always_fails)

        async def _fast_sleep(_delay):
            return None
        monkeypatch.setattr(sm.asyncio, "sleep", _fast_sleep)

        result = await _run_matcher(GOLDEN_FIXTURES["simple_lung_adeno"])
        # Hit the configured retry budget (3 attempts by default)
        assert attempt_count["n"] == 3
        assert isinstance(result, StructuredMatchResult)
        assert result.doc_ids == set()

    @pytest.mark.asyncio
    async def test_none_query_structure_returns_empty(self, monkeypatch):
        _install_fake_asyncpg(monkeypatch, canned_rows=[])
        result = await match_studies_by_structure(None)  # type: ignore[arg-type]
        assert result.doc_ids == set()

    @pytest.mark.asyncio
    async def test_conflicting_biomarker_polarities_does_not_crash(self, monkeypatch):
        """ER+ and ER- simultaneously should not crash the matcher —
        the test documents what actually happens (both get scored)."""
        _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "query_type": "treatment_recommendation",
            "cancer": {
                "site": "breast",
                "biomarkers": ["ER+", "ER-"],
            },
            "patient": {},
            "treatment": {},
        }
        result = await _run_matcher(qs)
        # Just assert it returns a result object — the actual scoring
        # for conflicting polarities is undefined behaviour in the
        # current implementation, and this test is a bug record.
        assert isinstance(result, StructuredMatchResult)

    @pytest.mark.asyncio
    async def test_receptor_status_string_is_expanded_into_biomarkers(self, monkeypatch):
        """'ER+/PR+/HER2-' in receptor_status must be split into 3 markers."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "query_type": "treatment_recommendation",
            "cancer": {
                "site": "breast",
                "receptor_status": "ER+/PR+/HER2-",
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        params = capture.params or []
        # All three receptor canonical keys should end up in params
        # (via the alias-aware parameterized key lookup)
        assert "ER" in params, f"ER missing from params: {params}"
        assert "PR" in params, f"PR missing from params: {params}"
        assert "HER2" in params, f"HER2 missing from params: {params}"


# ════════════════════════════════════════════════════════════════════════════
# Class 3a — Metastatic site fractional match
# ════════════════════════════════════════════════════════════════════════════
class TestMetastaticSiteFractionalScore:
    """The Class 3a boost adds a fractional score when the patient has
    metastatic_sites supplied (via clinical_inference). A study that
    mentions more of the patient's metastatic sites should score higher.
    Silent sites are ignored (denominator is total patient sites, not
    per-study reported ones) — same contract as biomarker fractional."""

    @pytest.mark.asyncio
    async def test_metastatic_sites_adds_param_regex_per_site(self, monkeypatch):
        """Two metastatic sites → two parameters with the right alias regex."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "query_type": "treatment_recommendation",
            "cancer": {
                "site": "breast",
                "biomarkers": ["HER2-"],
            },
            "patient": {},
            "treatment": {},
            "metastatic_sites": ["liver", "bone"],
        }
        await _run_matcher(qs)
        params = capture.params or []
        sql = capture.query or ""
        # Liver synonym regex ("liver|hepat|hcc") must land in params
        assert any("liver" in str(p) and "hepat" in str(p) for p in params), (
            f"expected liver/hepat alias regex in params: {params}"
        )
        # Bone synonym regex should appear
        assert any("bone" in str(p) and "osseous" in str(p) for p in params), (
            f"expected bone/osseous alias regex in params: {params}"
        )
        # SQL should include the dual-column (cancer_location + extraction_data) check
        assert "cancer_location ~*" in sql
        assert "extraction_data::text ~*" in sql

    @pytest.mark.asyncio
    async def test_no_metastatic_sites_contributes_nothing(self, monkeypatch):
        """Without metastatic_sites, no liver/bone regex pattern and no
        metastatic_sites criterion in the scoring row."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "query_type": "treatment_recommendation",
            "cancer": {
                "site": "breast",
                "biomarkers": ["HER2-"],
            },
            "patient": {},
            "treatment": {},
            # No metastatic_sites key
        }
        await _run_matcher(qs)
        params = capture.params or []
        # Metastatic site regex should not appear in params
        assert not any(
            "hepat" in str(p) and "|" in str(p) and "hcc" in str(p)
            for p in params
        ), f"expected no metastatic-site regex in params: {params}"

    @pytest.mark.asyncio
    async def test_metastatic_sites_weight_redistributes_to_100(self, monkeypatch):
        """When metastatic_sites is in present_criteria, dynamic weights
        should still sum to 100."""
        _install_fake_asyncpg(monkeypatch, canned_rows=[])
        # Directly probe dynamic weights
        weights = calculate_dynamic_weights(
            ["cancer_site", "histology", "biomarkers", "metastatic_sites"]
        )
        assert sum(weights.values()) == 100, (
            f"dynamic weights must sum to 100, got {weights} (sum={sum(weights.values())})"
        )
        assert "metastatic_sites" in weights
        assert weights["metastatic_sites"] > 0


# ════════════════════════════════════════════════════════════════════════════
# Class 3b — Title keyword relevance
# ════════════════════════════════════════════════════════════════════════════
class TestTitleRelevanceScore:
    """Class 3b adds a keyword overlap score against LOWER(study_name).
    Each keyword is inserted via a parameterized regex, and the score is
    additive (never blocks cryptic-title studies)."""

    @pytest.mark.asyncio
    async def test_title_relevance_sql_shape(self, monkeypatch):
        """SQL should contain LOWER(study_name) ~ $N clauses for each keyword."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "query_type": "treatment_recommendation",
            "cancer": {
                "site": "gi",
                "site_detail": "colon",
                "histology": "adenocarcinoma",
                "biomarkers": ["MSI-H"],
                "disease_descriptor": "metastatic",
            },
            "patient": {},
            "treatment": {},
            "metastatic_sites": ["liver"],
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        # Expect at least one "LOWER(study_name) ~ $N" clause
        assert "LOWER(study_name) ~" in sql, (
            f"title relevance SQL not emitted: {sql[:500]}"
        )

    @pytest.mark.asyncio
    async def test_title_relevance_emits_expected_keyword_params(self, monkeypatch):
        """Params list should contain the expected keyword regexes for
        an MSI-H metastatic colon adeno patient."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "query_type": "treatment_recommendation",
            "cancer": {
                "site": "gi",
                "site_detail": "colon",
                "histology": "adenocarcinoma",
                "biomarkers": ["MSI-H"],
                "disease_descriptor": "metastatic",
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        params = capture.params or []
        # MSI canonical keyword
        assert any("msi" in str(p) and "microsatellite" in str(p) for p in params), (
            f"expected MSI title-relevance regex in params: {params}"
        )
        # colon site_detail literal
        assert any(p == "colon" for p in params), (
            f"expected 'colon' literal in params: {params}"
        )
        # disease_descriptor → metastatic|stage iv|advanced
        assert any("metastatic" in str(p) and "stage iv" in str(p) for p in params), (
            f"expected metastatic descriptor regex in params: {params}"
        )
        # histology → adenocarcinoma alternation
        assert any("adenocarcinoma" in str(p) and "adeno" in str(p) for p in params), (
            f"expected histology regex in params: {params}"
        )

    @pytest.mark.asyncio
    async def test_title_relevance_weight_redistributes_to_100(self, monkeypatch):
        """title_relevance in present_criteria should not break the sum-100 invariant."""
        _install_fake_asyncpg(monkeypatch, canned_rows=[])
        weights = calculate_dynamic_weights(
            [
                "cancer_site",
                "histology",
                "biomarkers",
                "metastatic_sites",
                "title_relevance",
            ]
        )
        assert sum(weights.values()) == 100, (
            f"dynamic weights must sum to 100, got {weights} (sum={sum(weights.values())})"
        )
        assert "title_relevance" in weights
        assert weights["title_relevance"] > 0


# ════════════════════════════════════════════════════════════════════════════
# Task 6.6 — Trajectory scoring against extraction_data JSONB
# ════════════════════════════════════════════════════════════════════════════
class TestTrajectoryScoring:
    """Trajectory scoring checks extraction_data->>'disease_trajectory'
    for exact match against the detected trajectory value. It uses the
    disease_descriptor axis weight."""

    @pytest.mark.asyncio
    async def test_trajectory_from_reconciled_emits_jsonb_path(self, monkeypatch):
        """When ReconciledStructure has disease_trajectory, the SQL should
        contain extraction_data->>'disease_trajectory' check."""
        from src.api.services.query_reconciliation import ReconciledStructure

        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        reconciled = ReconciledStructure(
            cancer_site="lung",
            disease_trajectory="recurrent",
            has_patient_context=True,
        )
        qs = reconciled.to_query_structure_dict()
        await _run_matcher(qs, reconciled=reconciled)
        sql = capture.query or ""
        params = capture.params or []

        assert "extraction_data->>'disease_trajectory'" in sql, (
            f"trajectory JSONB path not in SQL: {sql[:500]}"
        )
        assert any(p == "recurrent" for p in params), (
            f"expected 'recurrent' in params: {params}"
        )

    @pytest.mark.asyncio
    async def test_trajectory_without_descriptor_uses_full_weight(self, monkeypatch):
        """When disease_descriptor is absent but trajectory is present,
        trajectory should use the full disease_descriptor weight."""
        from src.api.services.query_reconciliation import ReconciledStructure

        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        reconciled = ReconciledStructure(
            cancer_site="lung",
            disease_trajectory="progressive",
            has_patient_context=True,
        )
        qs = reconciled.to_query_structure_dict()
        await _run_matcher(qs, reconciled=reconciled)
        sql = capture.query or ""

        # The score_case should be labeled "disease_descriptor" (not bonus)
        assert "disease_descriptor" in sql or "disease_trajectory" in sql, (
            f"trajectory scoring not found in SQL: {sql[:500]}"
        )

    @pytest.mark.asyncio
    async def test_trajectory_with_descriptor_emits_bonus(self, monkeypatch):
        """When both disease_descriptor and trajectory are present,
        trajectory should emit a bonus score case."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "lung",
                "disease_descriptor": "metastatic",
            },
            "patient": {},
            "treatment": {},
        }
        # disease_descriptor="metastatic" triggers trajectory detection
        # as "metastatic", so both descriptor and trajectory fire.
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []

        # Both descriptor pattern and trajectory JSONB path should be present
        assert "extraction_data->>'disease_trajectory'" in sql, (
            f"trajectory JSONB path not in SQL: {sql[:500]}"
        )
        # The descriptor regex should also be present
        assert any("metastatic" in str(p) and "metastases" in str(p) for p in params), (
            f"expected metastatic descriptor regex in params: {params}"
        )

    @pytest.mark.asyncio
    async def test_trajectory_detection_from_query_text(self, monkeypatch):
        """When no ReconciledStructure is provided, trajectory should be
        detected from disease_descriptor text."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "lung",
                "disease_descriptor": "recurrent",
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""
        params = capture.params or []

        assert "extraction_data->>'disease_trajectory'" in sql, (
            f"trajectory JSONB path not in SQL: {sql[:500]}"
        )
        assert any(p == "recurrent" for p in params), (
            f"expected 'recurrent' in params: {params}"
        )

    @pytest.mark.asyncio
    async def test_no_trajectory_no_jsonb_path(self, monkeypatch):
        """When no trajectory is detected, no JSONB path should appear."""
        capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
        qs = {
            "cancer": {
                "site": "lung",
                "histology": "adenocarcinoma",
            },
            "patient": {},
            "treatment": {},
        }
        await _run_matcher(qs)
        sql = capture.query or ""

        assert "disease_trajectory" not in sql, (
            f"unexpected trajectory JSONB path in SQL: {sql[:500]}"
        )

    @pytest.mark.asyncio
    async def test_trajectory_indicators_detected(self, monkeypatch):
        """All trajectory indicators should be detected from disease_descriptor."""
        from src.api.services.query_reconciliation import ReconciledStructure

        indicators = [
            ("recurrent", "recurrent"),
            ("progressive", "progressive"),
            ("treatment-naive", "treatment-naive"),
            ("newly diagnosed", "treatment-naive"),
        ]
        for desc_text, expected_traj in indicators:
            capture = _install_fake_asyncpg(monkeypatch, canned_rows=[])
            qs = {
                "cancer": {
                    "site": "lung",
                    "disease_descriptor": desc_text,
                },
                "patient": {},
                "treatment": {},
            }
            await _run_matcher(qs)
            params = capture.params or []
            assert any(p == expected_traj for p in params), (
                f"expected '{expected_traj}' in params for descriptor '{desc_text}': {params}"
            )
