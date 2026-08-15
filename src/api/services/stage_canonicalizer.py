"""
Stage canonicalization service.

Normalizes staging notation (TNM, stage groups, c/p/yp prefixes) into a
canonical structure that supports bidirectional lookup and handles
recurrence-on-staging correctly.

Gated behind settings.enable_canonicalization.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import re

from src.core.config import settings


@dataclass
class TNMCanonical:
    """Canonical TNM staging with prefix preservation.

    Preserves c/p/yp/yc prefixes and supports bidirectional
    TNM <-> Stage Group lookup.
    """
    t: Optional[str] = None           # "T4d", "T1c", "Tis"
    t_prefix: str = ""                # "c", "p", "yp", "yc"
    n: Optional[str] = None           # "N2", "N1mi"
    n_prefix: str = ""
    m: Optional[str] = None           # "M0", "M1"
    m_prefix: str = ""
    stage_group: Optional[str] = None  # "IIIA", "IIB"
    staging_type: str = "unknown"      # "clinical", "pathologic", "post_neoadjuvant"
    confidence: float = 0.0

    def tnm_string(self) -> str:
        """Reconstruct full TNM string with prefixes.

        Examples:
            t_prefix="c", t="T4d", n="N2", m="M0" -> "cT4dN2M0"
            t_prefix="yp", t="T0", n="N0", m="M0" -> "ypT0N0M0"
        """
        parts = []
        if self.t:
            parts.append(f"{self.t_prefix}{self.t}")
        if self.n:
            parts.append(f"{self.n_prefix}{self.n}")
        if self.m:
            parts.append(f"{self.m_prefix}{self.m}")
        return "".join(parts)


@dataclass
class StageHistory:
    """Tracks staging at different timepoints for recurrence handling.

    Prevents conflation of initial staging with recurrence staging.
    """
    initial_stage: Optional[TNMCanonical] = None
    recurrence_stage: Optional[TNMCanonical] = None
    is_recurrent: bool = False
    is_restaged: bool = False


class StageCanonicalizer:
    """Canonicalizes staging from ReconciledStructure.

    Key behaviors:
    - Preserves c/p/yp/yc prefixes (EntityLinker provides raw extraction)
    - Detects recurrence-on-staging: "recurrent stage II" != "initial stage II"
    - Bidirectional TNM <-> Stage Group via StagingSearchExpander
    - Handles staging variants: ypT0N0 (pCR), pT1cN1mi (micrometastasis)

    Gated behind settings.enable_canonicalization — when the flag is False,
    returns empty TNMCanonical and StageHistory with defaults.
    """

    # Recurrence indicators in staging context
    _RECURRENCE_RE = re.compile(
        r'\b(recurrent|recurrence|relapsed?|progression|second primary)\b',
        re.IGNORECASE,
    )

    def canonicalize(
        self, reconciled, raw_text: str
    ) -> Tuple[TNMCanonical, StageHistory]:
        """Extract and canonicalize staging from reconciled structure.

        When settings.enable_canonicalization is False, returns empty
        TNMCanonical and StageHistory with defaults.

        Preconditions:
            - reconciled has tnm_t, tnm_n, tnm_m, stage fields (or equivalent)
            - raw_text is the original query text
        Postconditions:
            - TNMCanonical has prefixes preserved
            - StageHistory.is_recurrent set if recurrence language detected
            - stage_group populated via StagingSearchExpander if TNM is complete

        Args:
            reconciled: ReconciledStructure or equivalent with staging fields
            raw_text: Original query text for TNM extraction and recurrence detection

        Returns:
            Tuple of (TNMCanonical, StageHistory)
        """
        if not settings.enable_canonicalization:
            print(f"[StageCanon] skipped (flag off)")
            return TNMCanonical(), StageHistory()

        from src.api.services.entity_linker import EntityLinker
        linker = EntityLinker()
        tnm_raw = linker.link_tnm(raw_text)

        canonical = TNMCanonical()

        if tnm_raw:
            canonical.t = tnm_raw["t"]
            canonical.t_prefix = tnm_raw["t_prefix"]
            canonical.n = tnm_raw["n"]
            canonical.n_prefix = tnm_raw["n_prefix"]
            canonical.m = tnm_raw["m"]
            canonical.m_prefix = tnm_raw["m_prefix"]

            # Determine staging type from prefix
            prefixes = [canonical.t_prefix, canonical.n_prefix, canonical.m_prefix]
            if any(p.startswith("yp") for p in prefixes):
                canonical.staging_type = "post_neoadjuvant"
            elif any(p.startswith("yc") for p in prefixes):
                canonical.staging_type = "post_neoadjuvant"
            elif any(p == "p" for p in prefixes):
                canonical.staging_type = "pathologic"
            elif any(p == "c" for p in prefixes):
                canonical.staging_type = "clinical"

            print(
                f"[StageCanon] extracted: tnm={canonical.tnm_string()} "
                f"type={canonical.staging_type}"
            )

        # Bidirectional stage group lookup when TNM is complete but no stage group
        if canonical.t and canonical.n and canonical.m and not canonical.stage_group:
            canonical.stage_group, canonical.confidence = self._lookup_stage_group(
                canonical, reconciled
            )

        # Detect recurrence language
        history = StageHistory(initial_stage=canonical)
        if self._RECURRENCE_RE.search(raw_text):
            history.is_recurrent = True
            history.recurrence_stage = canonical
            print(f"[StageCanon] recurrence detected in: {raw_text[:80]}")

        print(
            f"[StageCanon] tnm={canonical.tnm_string()} "
            f"group={canonical.stage_group} type={canonical.staging_type}"
        )

        return canonical, history

    def _lookup_stage_group(
        self, canonical: TNMCanonical, reconciled
    ) -> Tuple[Optional[str], float]:
        """Bidirectional stage group lookup via StagingSearchExpander.

        Args:
            canonical: TNMCanonical with t, n, m populated
            reconciled: ReconciledStructure for cancer_site hint

        Returns:
            Tuple of (stage_group, confidence) or (None, 0.0)
        """
        try:
            from src.api.services.staging_search_expander import StagingSearchExpander
            expander = StagingSearchExpander()
            site_hint = getattr(reconciled, "cancer_site", None)
            terms = expander.expand_staging_for_search(
                canonical.tnm_string(), cancer_type=site_hint
            )
            if terms.inferred_stage_groups:
                stage_group = terms.inferred_stage_groups[0][0]
                confidence = terms.inferred_stage_groups[0][1]
                print(
                    f"[StageCanon] stage group inferred: "
                    f"{canonical.tnm_string()} -> {stage_group} "
                    f"(confidence={confidence:.0%})"
                )
                return stage_group, confidence
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[StageCanon] stage group lookup failed: {e}")

        return None, 0.0
