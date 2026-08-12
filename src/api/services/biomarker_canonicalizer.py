"""
Biomarker canonicalization service.

Resolves raw biomarker mentions to canonical IDs with normalized polarity
and optional metric extraction (CPS/TPS for PD-L1).

Gated behind settings.enable_canonicalization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

from src.core.config import settings


@dataclass
class CanonicalBiomarker:
    """A biomarker resolved to its canonical form."""
    canonical_id: str              # e.g., "EGFR", "HER2", "PD-L1"
    polarity: Optional[str]        # "mutant", "wild-type", "positive", "negative", "high", "low"
    metric: Optional[str]          # "CPS", "TPS", "IHC" — for PD-L1/HER2
    metric_value: Optional[str]    # "100", ">=50%", "3+"
    raw_text: str                  # original text span
    source: str                    # "regex", "llm", "reconciled"


# Canonical ID -> list of known aliases (case-sensitive entries, matched case-insensitively)
BIOMARKER_SYNONYMS: Dict[str, List[str]] = {
    "EGFR": ["EGFR", "egfr", "ERBB1", "HER1", "epidermal growth factor receptor"],
    "HER2": ["HER2", "HER-2", "Her2", "ERBB2", "erbb2", "HER2/neu"],
    "ALK":  ["ALK", "alk", "anaplastic lymphoma kinase"],
    "KRAS": ["KRAS", "kras", "kirsten rat sarcoma"],
    "BRAF": ["BRAF", "braf", "b-raf"],
    "BRCA": ["BRCA", "brca"],
    "BRCA1": ["BRCA1", "brca1"],
    "BRCA2": ["BRCA2", "brca2"],
    "PD-L1": ["PD-L1", "PDL1", "pd-l1", "pdl1", "CD274", "B7-H1"],
    "MSI":  ["MSI", "msi", "MSI-H", "dMMR", "microsatellite instability"],
    "TMB":  ["TMB", "tmb", "TMB-H", "tumor mutational burden"],
    "HPV":  ["HPV", "hpv", "p16", "human papillomavirus"],
}

# Raw polarity string -> canonical polarity
POLARITY_CANONICAL: Dict[str, str] = {
    "mutant": "mutant", "mutation": "mutant", "mutated": "mutant",
    "positive": "positive", "+": "positive", "amplified": "positive",
    "overexpressed": "positive", "detected": "positive",
    "negative": "negative", "-": "negative", "absent": "negative",
    "wild-type": "wild-type", "wildtype": "wild-type", "wt": "wild-type",
    "high": "high", "elevated": "high",
    "low": "low", "decreased": "low",
    "fusion": "mutant", "rearrangement": "mutant", "translocation": "mutant",
    "stable": "stable", "mss": "stable", "pmmr": "stable",
}


class BiomarkerCanonicalizer:
    """Resolves raw biomarker mentions to canonical IDs with polarity.

    Gated behind settings.enable_canonicalization — when the flag is False,
    resolve() and resolve_list() return passthrough results with the original
    name and polarity unchanged.
    """

    def __init__(self):
        # Build reverse lookup: lowercased alias -> canonical_id
        self._alias_to_canonical: Dict[str, str] = {}
        for canonical_id, aliases in BIOMARKER_SYNONYMS.items():
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical_id

        # CPS/TPS metric extraction patterns for PD-L1
        self._cps_pattern = re.compile(
            r'\bCPS\s*(?:score\s*(?:of\s*)?)?\s*(?:[=≥>]\s*)?(\d+)', re.I
        )
        self._tps_pattern = re.compile(
            r'\bTPS\s*(?:[≥>]=?\s*)?(\d+)\s*%?', re.I
        )

    def resolve(
        self,
        name: str,
        polarity: Optional[str] = None,
        raw_text: str = "",
        source: str = "regex",
    ) -> CanonicalBiomarker:
        """Resolve a biomarker name + polarity to canonical form.

        When settings.enable_canonicalization is False, returns a
        CanonicalBiomarker with the original name/polarity unchanged.

        Preconditions:
            - name is a non-empty string
        Postconditions:
            - Returns CanonicalBiomarker with canonical_id from BIOMARKER_SYNONYMS
              or original name if not found
            - polarity is normalized via POLARITY_CANONICAL
            - CPS/TPS metrics extracted from raw_text if applicable
        """
        if not settings.enable_canonicalization:
            print(f"[Canon] skipped (flag off) name={name}")
            return CanonicalBiomarker(
                canonical_id=name,
                polarity=polarity,
                metric=None,
                metric_value=None,
                raw_text=raw_text,
                source=source,
            )

        canonical_id = self._alias_to_canonical.get(name.lower(), name)
        canonical_polarity = (
            POLARITY_CANONICAL.get(polarity.lower(), polarity)
            if polarity
            else None
        )

        # Extract CPS/TPS metric for PD-L1
        metric = None
        metric_value = None
        if canonical_id == "PD-L1" and raw_text:
            cps_match = self._cps_pattern.search(raw_text)
            if cps_match:
                metric = "CPS"
                metric_value = cps_match.group(1)
            else:
                tps_match = self._tps_pattern.search(raw_text)
                if tps_match:
                    metric = "TPS"
                    metric_value = f"{tps_match.group(1)}%"

        print(
            f"[Canon] biomarker={canonical_id} polarity={canonical_polarity} "
            f"metric={metric} metric_value={metric_value} source={source}"
        )

        return CanonicalBiomarker(
            canonical_id=canonical_id,
            polarity=canonical_polarity,
            metric=metric,
            metric_value=metric_value,
            raw_text=raw_text,
            source=source,
        )

    def resolve_list(
        self,
        biomarkers: List[Tuple[str, Optional[str]]],
        raw_text: str = "",
        source: str = "regex",
    ) -> List[CanonicalBiomarker]:
        """Resolve a list of (name, polarity) tuples to canonical form."""
        return [
            self.resolve(name, polarity, raw_text, source)
            for name, polarity in biomarkers
        ]
