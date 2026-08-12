"""
Relaxed numerical validation for RAG answers (Phase 7).

Replaces the strict strip-on-miss behaviour with graduated validation:
  VERIFIED        – exact match in source chunks
  LIKELY_CORRECT  – within ±5 % of a source number
  KNOWN_CONSTANT  – well-known clinical constant (e.g. "50 Gy")
  UNVERIFIED      – no source support (annotated in metadata, NOT stripped)

Gated behind ``settings.enable_relaxed_numval``.  When the flag is False
the caller should fall back to the existing strict validator in
``src/api/services/safety/numerical.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class NumberValidation:
    """Result of validating a single number in the generated response."""

    number_text: str                          # raw text span, e.g. "85%"
    status: str                               # VERIFIED | LIKELY_CORRECT | KNOWN_CONSTANT | UNVERIFIED
    source_text: Optional[str] = None         # source chunk text that matched
    tolerance: Optional[float] = None         # tolerance used for LIKELY_CORRECT


# ─── Known clinical constants ────────────────────────────────────────────────

KNOWN_CLINICAL_CONSTANTS: List[str] = [
    "50 Gy",
    "2 Gy/fraction",
    "1.8 Gy",
    "60 Gy",
    "45 Gy",
    "70 Gy",
    "54 Gy",
    "30 Gy",
    "20 Gy",
    "66 Gy",
    "74 Gy",
    "50.4 Gy",
    "1.8 Gy/fraction",
    "2 Gy per fraction",
    "100 mg/m2",
    "75 mg/m2",
    "200 mg",
    "240 mg",
    "480 mg",
    "3 mg/kg",
    "10 mg/kg",
]

# Pre-compile a set of lowered constants for fast lookup
_CONSTANTS_LOWER = {c.lower() for c in KNOWN_CLINICAL_CONSTANTS}

# Regex to extract numbers (with optional unit) from text
_NUMBER_RE = re.compile(
    r'(\d+\.?\d*)\s*'
    r'(%|Gy(?:/fraction)?|mg(?:/m2|/kg)?|months?|years?|mo|yr)?',
    re.IGNORECASE,
)

# Regex to extract number+unit spans that might be clinical constants
_CONSTANT_SPAN_RE = re.compile(
    r'\d+\.?\d*\s*(?:Gy(?:/fraction| per fraction)?|mg(?:/m2|/kg)?)',
    re.IGNORECASE,
)

DEFAULT_TOLERANCE = 0.05  # ±5 %


# ─── Validator ───────────────────────────────────────────────────────────────

class RelaxedNumericalValidator:
    """Validates numbers in generated responses against source evidence.

    Preconditions:
        - ``settings.enable_relaxed_numval`` is True (caller checks)
    Postconditions:
        - Every number in the response gets a ``NumberValidation`` result
        - UNVERIFIED numbers are annotated, never stripped
    """

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        self.tolerance = tolerance

    # ── single-number validation ─────────────────────────────────────────

    def validate_number(
        self,
        number_text: str,
        source_texts: List[str],
    ) -> NumberValidation:
        """Validate a single number span against source texts.

        Args:
            number_text: The raw number span from the response (e.g. "85%").
            source_texts: List of source chunk strings to validate against.

        Returns:
            NumberValidation with appropriate status.
        """
        # 1. Exact match in any source
        for src in source_texts:
            if number_text in src:
                print(f"[NumVal] VERIFIED number_text='{number_text}'")
                return NumberValidation(
                    number_text=number_text,
                    status="VERIFIED",
                    source_text=src,
                )

        # 2. Known clinical constant
        if self._is_known_constant(number_text):
            print(f"[NumVal] KNOWN_CONSTANT number_text='{number_text}'")
            return NumberValidation(
                number_text=number_text,
                status="KNOWN_CONSTANT",
            )

        # 3. ±tolerance match against source numbers
        response_value = self._extract_value(number_text)
        if response_value is not None:
            for src in source_texts:
                for src_match in _NUMBER_RE.finditer(src):
                    src_value = float(src_match.group(1))
                    if src_value == 0:
                        continue
                    pct_diff = abs(response_value - src_value) / abs(src_value)
                    if pct_diff <= self.tolerance:
                        print(
                            f"[NumVal] LIKELY_CORRECT number_text='{number_text}' "
                            f"source_value={src_value} diff={pct_diff:.2%}"
                        )
                        return NumberValidation(
                            number_text=number_text,
                            status="LIKELY_CORRECT",
                            source_text=src,
                            tolerance=pct_diff,
                        )

        # 4. No source support → UNVERIFIED (annotated, not stripped)
        print(f"[NumVal] UNVERIFIED number_text='{number_text}'")
        return NumberValidation(
            number_text=number_text,
            status="UNVERIFIED",
        )

    # ── full-response validation ─────────────────────────────────────────

    def validate_response(
        self,
        response_text: str,
        source_texts: List[str],
    ) -> Dict[str, Any]:
        """Validate all numbers in a generated response.

        Args:
            response_text: The full generated answer.
            source_texts: List of source chunk strings.

        Returns:
            Dict with keys:
                validations  – list of NumberValidation
                verified     – count of VERIFIED
                likely       – count of LIKELY_CORRECT
                constants    – count of KNOWN_CONSTANT
                unverified   – count of UNVERIFIED
                metadata     – dict of unverified annotations (for response metadata)
        """
        number_spans = self._extract_number_spans(response_text)
        validations: List[NumberValidation] = []

        for span in number_spans:
            result = self.validate_number(span, source_texts)
            validations.append(result)

        verified = sum(1 for v in validations if v.status == "VERIFIED")
        likely = sum(1 for v in validations if v.status == "LIKELY_CORRECT")
        constants = sum(1 for v in validations if v.status == "KNOWN_CONSTANT")
        unverified = sum(1 for v in validations if v.status == "UNVERIFIED")

        # Build metadata annotations for unverified numbers
        unverified_annotations = [
            {"number_text": v.number_text, "status": v.status}
            for v in validations
            if v.status == "UNVERIFIED"
        ]

        print(
            f"[NumVal] verified={verified} likely_correct={likely} "
            f"known_constants={constants} unverified={unverified}"
        )

        return {
            "validations": validations,
            "verified": verified,
            "likely_correct": likely,
            "known_constants": constants,
            "unverified": unverified,
            "metadata": {
                "unverified_numbers": unverified_annotations,
            },
        }

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_known_constant(text: str) -> bool:
        """Check if text matches a known clinical constant."""
        # Check full span
        if text.strip().lower() in _CONSTANTS_LOWER:
            return True
        # Check if any constant span inside text matches
        for span_match in _CONSTANT_SPAN_RE.finditer(text):
            if span_match.group(0).strip().lower() in _CONSTANTS_LOWER:
                return True
        return False

    @staticmethod
    def _extract_value(text: str) -> Optional[float]:
        """Extract the leading numeric value from a text span."""
        m = re.match(r'(\d+\.?\d*)', text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_number_spans(text: str) -> List[str]:
        """Extract all number-bearing spans from response text.

        Returns spans like "85%", "50 Gy", "HR 0.72", "3.5 months".
        """
        spans: List[str] = []
        seen_positions: set = set()

        # Clinical constant spans first (higher priority)
        for m in _CONSTANT_SPAN_RE.finditer(text):
            span = m.group(0).strip()
            if span and m.start() not in seen_positions:
                spans.append(span)
                for pos in range(m.start(), m.end()):
                    seen_positions.add(pos)

        # Then general number+unit spans
        for m in _NUMBER_RE.finditer(text):
            if m.start() in seen_positions:
                continue
            span = m.group(0).strip()
            # Skip bare single-digit numbers that are likely not clinical data
            if span and not (len(span) == 1 and span.isdigit()):
                spans.append(span)

        return spans
