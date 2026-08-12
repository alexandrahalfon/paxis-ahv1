"""
Cancer type canonicalization service.

Resolves cancer type from site + histology as a keyed pair, ensuring
N+ nodal status is NOT conflated with receptor status and site
disambiguation is correct.

Uses normalize_category() from comprehensive_retrieval.py for category
resolution.

Gated behind settings.enable_canonicalization.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.core.config import settings


@dataclass
class CanonicalCancerType:
    """Cancer type resolved to canonical form."""
    site: str                    # "head_neck", "breast", "lung", etc.
    site_detail: Optional[str]   # "oral_tongue", "oropharynx", etc.
    histology: Optional[str]     # "squamous_cell_carcinoma", "adenocarcinoma"
    category: str                # Qdrant filter category
    raw_text: str                # original text span


# Site + histology keyed pairs for disambiguation.
# Maps (normalized_site, normalized_histology) -> keyed category string.
SITE_HISTOLOGY_MAP: Dict[Tuple[str, str], str] = {
    ("head_neck", "scc"): "head_neck_scc",
    ("head_neck", "squamous_cell_carcinoma"): "head_neck_scc",
    ("head_neck", "adenocarcinoma"): "head_neck_adeno",
    ("head_neck", "adenoid_cystic"): "head_neck_adenoid_cystic",
    ("lung", "adenocarcinoma"): "lung_adeno",
    ("lung", "scc"): "lung_scc",
    ("lung", "squamous_cell_carcinoma"): "lung_scc",
    ("lung", "small_cell"): "lung_sclc",
    ("lung", "sclc"): "lung_sclc",
    ("lung", "large_cell"): "lung_lcc",
    ("breast", "idc"): "breast_idc",
    ("breast", "invasive_ductal"): "breast_idc",
    ("breast", "ilc"): "breast_ilc",
    ("breast", "invasive_lobular"): "breast_ilc",
    ("colorectal", "adenocarcinoma"): "colorectal_adeno",
    ("pancreatic", "adenocarcinoma"): "pancreatic_adeno",
}

# Site detail extraction: maps keywords found in raw text to site_detail values.
_SITE_DETAIL_KEYWORDS: Dict[str, Dict[str, str]] = {
    "head_neck": {
        "oral tongue": "oral_tongue",
        "tongue": "oral_tongue",
        "oropharynx": "oropharynx",
        "oropharyngeal": "oropharynx",
        "nasopharynx": "nasopharynx",
        "nasopharyngeal": "nasopharynx",
        "hypopharynx": "hypopharynx",
        "larynx": "larynx",
        "laryngeal": "larynx",
        "oral cavity": "oral_cavity",
        "salivary": "salivary_gland",
        "paranasal": "paranasal_sinus",
        "sinus": "paranasal_sinus",
    },
}

# Histology normalization: maps common aliases to canonical histology strings.
_HISTOLOGY_ALIASES: Dict[str, str] = {
    "scc": "scc",
    "squamous": "scc",
    "squamous cell": "scc",
    "squamous cell carcinoma": "scc",
    "adeno": "adenocarcinoma",
    "adenocarcinoma": "adenocarcinoma",
    "small cell": "small_cell",
    "sclc": "small_cell",
    "large cell": "large_cell",
    "idc": "idc",
    "invasive ductal": "idc",
    "invasive ductal carcinoma": "idc",
    "ilc": "ilc",
    "invasive lobular": "ilc",
    "invasive lobular carcinoma": "ilc",
    "adenoid cystic": "adenoid_cystic",
}


class CancerTypeCanonicalizer:
    """Resolves cancer type from site + histology, keyed to avoid conflation.

    Preconditions:
        - Input is a ReconciledStructure or equivalent dict/object
    Postconditions:
        - N+ nodal status is NOT conflated with receptor status
        - Site + histology are keyed together for disambiguation
        - Returns CanonicalCancerType with normalized category
    """

    def canonicalize(self, reconciled) -> CanonicalCancerType:
        """Resolve cancer type from reconciled structure.

        Extracts site and histology from the reconciled structure, normalizes
        them, and keys them together for disambiguation via SITE_HISTOLOGY_MAP.

        Args:
            reconciled: A ReconciledStructure or object with cancer_site and
                        histology attributes.

        Returns:
            CanonicalCancerType with normalized site, site_detail, histology,
            category, and raw_text.
        """
        if not settings.enable_canonicalization:
            raw_site = getattr(reconciled, "cancer_site", None) or ""
            raw_histology = getattr(reconciled, "histology", None) or ""
            raw_text = f"{raw_site} {raw_histology}".strip()
            print(f"[Canon] cancer_type skipped (flag off) raw={raw_text}")
            return CanonicalCancerType(
                site=raw_site,
                site_detail=None,
                histology=raw_histology or None,
                category=raw_site,
                raw_text=raw_text,
            )

        raw_site = getattr(reconciled, "cancer_site", None) or ""
        raw_histology = getattr(reconciled, "histology", None) or ""
        raw_text = f"{raw_site} {raw_histology}".strip()

        # Normalize site via normalize_category from comprehensive_retrieval
        from src.api.services.comprehensive_retrieval import normalize_category
        site = normalize_category(raw_site)

        # Normalize histology
        histology = self._normalize_histology(raw_histology)

        # Extract site detail from raw text
        site_detail = self._extract_site_detail(site, raw_text)

        # Key site + histology together for disambiguation
        keyed = SITE_HISTOLOGY_MAP.get((site, histology)) if histology else None
        category = keyed if keyed else site

        print(
            f"[Canon] cancer_type site={site} histology={histology} "
            f"detail={site_detail} category={category}"
        )

        return CanonicalCancerType(
            site=site,
            site_detail=site_detail,
            histology=histology or None,
            category=category,
            raw_text=raw_text,
        )

    def _normalize_histology(self, raw: str) -> str:
        """Normalize a histology string to its canonical form.

        Returns empty string if input is empty/None.
        """
        if not raw or not raw.strip():
            return ""
        lowered = raw.strip().lower()
        return _HISTOLOGY_ALIASES.get(lowered, lowered)

    def _extract_site_detail(self, site: str, raw_text: str) -> Optional[str]:
        """Extract site detail from raw text based on the normalized site.

        For example, if site is 'head_neck' and raw_text contains 'oral tongue',
        returns 'oral_tongue'.
        """
        if not raw_text or not site:
            return None
        keywords = _SITE_DETAIL_KEYWORDS.get(site)
        if not keywords:
            return None
        lowered = raw_text.lower()
        for keyword, detail in keywords.items():
            if keyword in lowered:
                return detail
        return None
