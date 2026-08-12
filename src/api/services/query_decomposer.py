"""
Query decomposition for complex multi-intent oncology queries.

Breaks comparison queries ("X vs Y") and multi-timepoint queries
("neoadjuvant and adjuvant") into focused sub-queries for independent retrieval.
Gated behind settings.enable_query_decomposition.
"""

import json
import re
import traceback
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from src.core.config import settings


@dataclass
class DecomposedQuery:
    """A query broken into sub-queries for independent retrieval."""
    original: str
    sub_queries: List[str]
    is_decomposed: bool
    decomposition_reason: Optional[str] = None


class QueryDecomposer:
    """Decomposes complex multi-part queries into focused sub-queries.

    Triggers when:
    - Query contains comparison language ("X vs Y", "versus", "compared to")
    - Query asks about multiple timepoints ("neoadjuvant and adjuvant")

    Does NOT decompose:
    - Simple single-intent queries
    - Queries with a single patient presentation
    - When enable_query_decomposition flag is False
    """

    # Comparison language: "X vs Y", "versus", "compared to/with"
    _COMPARISON_RE = re.compile(
        r'\bvs\.?\b|\bversus\b|\bcompared?\s+(?:to|with)\b',
        re.IGNORECASE,
    )

    # Multiple treatment timepoints in the same query
    _MULTI_TIMEPOINT_RE = re.compile(
        r'\b(neoadjuvant|adjuvant|concurrent|sequential|maintenance)\b'
        r'.*\b(neoadjuvant|adjuvant|concurrent|sequential|maintenance)\b',
        re.IGNORECASE,
    )

    def should_decompose(self, query: str) -> bool:
        """Check if query would benefit from decomposition.

        Detects comparison language and multi-timepoint references.
        Simple single-intent queries return False.

        Preconditions:
            - query is a non-empty string
        Postconditions:
            - Returns True only if comparison or multi-timepoint patterns are found
        """
        if not query or not query.strip():
            return False

        return bool(
            self._COMPARISON_RE.search(query)
            or self._MULTI_TIMEPOINT_RE.search(query)
        )

    async def decompose(self, query: str) -> DecomposedQuery:
        """Decompose query into sub-queries using LLM.

        If the feature flag is disabled, or the query doesn't need decomposition,
        returns the original query as a single-element list.
        On any failure, falls back to the original query.

        Preconditions:
            - query is a non-empty string
        Postconditions:
            - Returns DecomposedQuery with at least one sub_query
            - On failure, sub_queries contains the original query
        """
        # Gate behind feature flag
        if not settings.enable_query_decomposition:
            return DecomposedQuery(
                original=query,
                sub_queries=[query],
                is_decomposed=False,
            )

        # Check if decomposition is warranted
        if not self.should_decompose(query):
            print(f"[Decomp] No decomposition needed for query")
            return DecomposedQuery(
                original=query,
                sub_queries=[query],
                is_decomposed=False,
            )

        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_mini_model,
                temperature=0,
                max_tokens=200,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Break this oncology query into independent sub-queries. "
                            "Each sub-query should be self-contained with full patient context. "
                            "Return JSON: {\"sub_queries\": [\"...\", \"...\"]}"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )

            raw = response.choices[0].message.content
            result = json.loads(raw)
            subs = result.get("sub_queries", [query])

            # Validate: must be a non-empty list of strings
            if not isinstance(subs, list) or not subs:
                subs = [query]
            subs = [s for s in subs if isinstance(s, str) and s.strip()]
            if not subs:
                subs = [query]

            is_decomposed = len(subs) > 1
            reason = "comparison detected" if self._COMPARISON_RE.search(query) else "multi-timepoint detected"

            print(f"[Decomp] original={query[:80]} subs={len(subs)}")

            return DecomposedQuery(
                original=query,
                sub_queries=subs,
                is_decomposed=is_decomposed,
                decomposition_reason=reason if is_decomposed else None,
            )

        except Exception as e:
            traceback.print_exc()
            print(f"[Decomp] Decomposition failed, falling back to original query: {e}")
            return DecomposedQuery(
                original=query,
                sub_queries=[query],
                is_decomposed=False,
            )
