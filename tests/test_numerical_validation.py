"""
Offline tests for `strip_unvalidated_numbers` (RF-5).

Purpose: verify that numerical values the LLM fabricates are replaced
with the literal token `[unverified]` before the answer is returned to
the user. The function consumes the `unvalidated_numbers` list produced
by `validate_numbers_against_sources`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.enhanced_rag_service import (  # noqa: E402
    strip_unvalidated_numbers,
)


class TestStripUnvalidatedNumbers:
    def test_replaces_each_raw_match_with_unverified(self):
        answer = "The 5-year OS was 80% (HR 0.65) with 20% improvement."
        unvalidated = [
            {"raw_match": "80%", "value": 80.0, "type": "percentage"},
            {"raw_match": "20%", "value": 20.0, "type": "percentage"},
        ]
        cleaned = strip_unvalidated_numbers(answer, unvalidated)
        assert "[unverified]" in cleaned
        assert "80%" not in cleaned
        assert "20%" not in cleaned
        # The validated HR must survive untouched
        assert "HR 0.65" in cleaned

    def test_empty_unvalidated_list_returns_answer_unchanged(self):
        answer = "5-year OS was 80% per the MACH-NC meta-analysis."
        assert strip_unvalidated_numbers(answer, []) == answer

    def test_empty_answer_returns_unchanged(self):
        assert strip_unvalidated_numbers("", [{"raw_match": "5%"}]) == ""

    def test_none_answer_returns_none(self):
        assert strip_unvalidated_numbers(None, [{"raw_match": "5%"}]) is None  # type: ignore[arg-type]

    def test_missing_raw_match_silently_skipped(self):
        answer = "foo 5% bar"
        # unvalidated entry missing raw_match — should be skipped
        cleaned = strip_unvalidated_numbers(answer, [{"value": 5.0}])
        assert cleaned == answer

    def test_only_first_occurrence_replaced(self):
        """Subsequent instances of the same raw substring may be different
        clinical values the LLM rendered identically — only strip once."""
        cleaned = strip_unvalidated_numbers(
            "5% and 5% again", [{"raw_match": "5%"}]
        )
        assert cleaned == "[unverified] and 5% again"

    def test_strip_preserves_surrounding_sentence(self):
        answer = "Survival: 80% at 5 years."
        cleaned = strip_unvalidated_numbers(
            answer, [{"raw_match": "80%"}]
        )
        # Punctuation and whitespace around the stripped value stay intact
        assert cleaned == "Survival: [unverified] at 5 years."
