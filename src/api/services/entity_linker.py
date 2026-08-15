"""
Entity linking service.

Links clinical shorthand to canonical clinical terms:
- CPS → PD-L1 metric (not a standalone biomarker)
- TNM with prefix preservation (c/p/yc/yp not dropped)
- Bidirectional stage group lookup via StagingSearchExpander

Requirements: 4.1, 4.2, 4.3
"""

import re
from typing import Dict, List, Optional

from src.api.services.biomarker_canonicalizer import CanonicalBiomarker


class EntityLinker:
    """Links clinical shorthand to canonical clinical terms.

    Fixes:
        - CPS → PD-L1 metric (not a standalone biomarker)
        - pT4/cT4 → TNM canonical with prefix preserved (no c/p drop)
        - Stage group ↔ TNM bidirectional (uses StagingSearchExpander)
    """

    # TNM prefix pattern: captures optional y prefix + c/p, then the T/N/M component.
    # Handles: cT4dN2M0, pT1cN1mi, ypT0N0M0, ycT2N1M0, T1N0M0 (no prefix)
    _TNM_PREFIX_RE = re.compile(
        r'\b(y?[cp])?(T(?:is|a|[0-4][a-d]?))\s*'
        r'(y?[cp])?(N(?:[0-3][a-c]?(?:mi)?))\s*'
        r'(y?[cp])?(M[01][a-c]?)\b',
        re.IGNORECASE,
    )

    def link_tnm(self, raw_text: str) -> Optional[Dict[str, str]]:
        """Extract TNM with prefix preserved.

        Preconditions:
            - raw_text contains TNM notation
        Postconditions:
            - Returns dict with keys: t, t_prefix, n, n_prefix, m, m_prefix
            - Prefixes c/p/yc/yp are preserved, not dropped
            - Returns None if no TNM match found

        Examples:
            "cT4dN2M0"  → t_prefix="c", t="T4d", n_prefix="", n="N2", m_prefix="", m="M0"
            "pT1cN1mi"  → t_prefix="p", t="T1c", n_prefix="", n="N1mi"
            "ypT0N0M0"  → t_prefix="yp", t="T0", staging_type=post_neoadjuvant
        """
        match = self._TNM_PREFIX_RE.search(raw_text)
        if not match:
            print(f"[EntityLinker] No TNM match in: {raw_text[:80]}")
            return None

        result = {
            "t_prefix": (match.group(1) or "").lower(),
            "t": match.group(2),
            "n_prefix": (match.group(3) or "").lower(),
            "n": match.group(4),
            "m_prefix": (match.group(5) or "").lower(),
            "m": match.group(6),
        }

        print(
            f"[EntityLinker] TNM extracted: "
            f"{result['t_prefix']}{result['t']}"
            f"{result['n_prefix']}{result['n']}"
            f"{result['m_prefix']}{result['m']}"
        )
        return result

    def link_cps_to_pdl1(self, biomarkers: List[CanonicalBiomarker]) -> List[CanonicalBiomarker]:
        """Ensure CPS scores are linked to PD-L1, not treated as standalone.

        CPS (Combined Positive Score) is a PD-L1 scoring metric. If CPS appears
        without an explicit PD-L1 biomarker, promote it to PD-L1 with metric=CPS.

        Preconditions:
            - biomarkers is a list of CanonicalBiomarker instances
        Postconditions:
            - Any biomarker with canonical_id starting with "CPS" is promoted
              to PD-L1 with metric="CPS" if no PD-L1 already exists
            - Original list order is preserved
        """
        has_pdl1 = any(bm.canonical_id == "PD-L1" for bm in biomarkers)

        result = []
        for bm in biomarkers:
            if bm.canonical_id.startswith("CPS") and not has_pdl1:
                print(
                    f"[EntityLinker] Promoting CPS to PD-L1: "
                    f"canonical_id={bm.canonical_id} → PD-L1, metric=CPS"
                )
                bm.canonical_id = "PD-L1"
                bm.metric = "CPS"
                has_pdl1 = True
            result.append(bm)

        return result

    def link_tnm_to_stage_group(
        self, tnm: Dict[str, str], cancer_site: Optional[str] = None
    ) -> Optional[str]:
        """Bidirectional stage group lookup via StagingSearchExpander.

        Given extracted TNM components, infer the AJCC stage group.

        Preconditions:
            - tnm dict has keys: t, n, m (from link_tnm output)
        Postconditions:
            - Returns inferred stage group string (e.g., "IIIA") or None
        """
        tnm_string = f"{tnm.get('t_prefix', '')}{tnm.get('t', '')}" \
                      f"{tnm.get('n_prefix', '')}{tnm.get('n', '')}" \
                      f"{tnm.get('m_prefix', '')}{tnm.get('m', '')}"

        if not tnm_string:
            return None

        try:
            from src.api.services.staging_search_expander import StagingSearchExpander
            expander = StagingSearchExpander()
            terms = expander.expand_staging_for_search(
                tnm_string, cancer_type=cancer_site
            )
            if terms.inferred_stage_groups:
                stage_group = terms.inferred_stage_groups[0][0]
                confidence = terms.inferred_stage_groups[0][1]
                print(
                    f"[EntityLinker] Stage group inferred: "
                    f"{tnm_string} → {stage_group} (confidence={confidence:.0%})"
                )
                return stage_group
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[EntityLinker] Stage group lookup failed: {e}")

        return None
