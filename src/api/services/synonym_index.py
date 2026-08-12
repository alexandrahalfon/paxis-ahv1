"""
Unified Synonym / Variant Resolver
==================================

One place that knows how to map any raw clinical phrase ("scc",
"keytruda", "oral tongue", "msi high") onto the Title-Case canonical
values that are indexed into Qdrant chunk payloads.

Sources it reads (read once, cached in memory):
  - ``SITE_SYNONYM_GROUPS``, ``HISTOLOGY_SYNONYM_GROUPS``,
    ``BIOMARKER_SYNONYM_GROUPS``, ``DRUG_SYNONYM_GROUPS``,
    ``ALTERATION_SYNONYM_GROUPS`` from src/ingestion/keyword_tagger.py
  - ``data/ontology/cancer_type_ontology.json`` — per-cancer-type
    label + synonyms, used for the cancer_type axis
  - In-module ``_SUBSITE_FALLBACK`` map for fine-grained subsites
    that aren't covered by SITE_SYNONYM_GROUPS (e.g. "tongue" →
    "Oral cavity", "base of tongue" → "Oropharynx")
  - In-module ``_EQUIVALENT_CANONICALS`` — semantically equivalent
    canonicals that should all appear in MatchAny together (e.g.
    "MSI-H" ↔ "dMMR")

Public API
----------
    idx = get_synonym_index()
    idx.resolve("histology", "SCC")
    # → ["Squamous cell carcinoma"]

    idx.resolve("biomarker", "msi-high")
    # → ["MSI-H", "dMMR"]   (equivalence expansion)

    idx.resolve("cancer_type", "head and neck")
    # → ["Head and Neck Cancer"]

    idx.resolve("site", "tongue")
    # → ["Oral cavity"]

    idx.normalize_many("drug", ["pembrolizumab", "keytruda", "fake"])
    # → ["Pembrolizumab"]   (deduped; unknowns dropped)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set


# ── Manually curated fallback maps ──────────────────────────────────────

# Subsites that are NOT covered by SITE_SYNONYM_GROUPS variants but
# appear in clinical queries. Key is lowercased; value is the canonical
# site that they anatomically belong to.
_SUBSITE_FALLBACK: Dict[str, str] = {
    # Head & neck
    "tongue": "Oral cavity",
    "mobile tongue": "Oral cavity",
    "anterior tongue": "Oral cavity",
    "mandible": "Oral cavity",
    "maxilla": "Oral cavity",
    "lip": "Oral cavity",
    "tonsil": "Oropharynx",
    "base of tongue": "Oropharynx",
    "bot": "Oropharynx",
    "soft palate": "Oropharynx",
    "posterior pharyngeal wall": "Oropharynx",
    "piriform sinus": "Hypopharynx",
    "postcricoid": "Hypopharynx",
    "pyriform": "Hypopharynx",
    "vocal cord": "Larynx",
    "glottis": "Larynx",
    "supraglottis": "Larynx",
    "subglottis": "Larynx",
    "ethmoid": "Paranasal sinuses",
    "sphenoid": "Paranasal sinuses",
    "frontal sinus": "Paranasal sinuses",
    "maxillary sinus": "Paranasal sinuses",
    "parotid": "Salivary gland",
    "submandibular": "Salivary gland",
    "sublingual": "Salivary gland",
    # GI
    "cecum": "Colon",
    "sigmoid": "Colon",
    "ascending colon": "Colon",
    "descending colon": "Colon",
    "transverse colon": "Colon",
    "gej": "Gastroesophageal junction",
    "cardia": "Gastroesophageal junction",
    # GU
    "renal pelvis": "Kidney",
    "renal cortex": "Kidney",
    "urothelium": "Bladder",
    # Gyn
    "endometrium": "Uterus",
    "uterine body": "Uterus",
    "uterine corpus": "Uterus",
    "myometrium": "Uterus",
    "endocervix": "Cervix",
    "ectocervix": "Cervix",
    # Breast
    "areola": "Nipple",
    # CNS
    "frontal lobe": "Brain",
    "temporal lobe": "Brain",
    "parietal lobe": "Brain",
    "occipital lobe": "Brain",
    "cerebellum": "Brain",
    "brainstem": "Brain",
    "right ventricle": "Heart",
    # Lung
    "upper lobe": "Lung",
    "lower lobe": "Lung",
    "rul": "Lung",
    "lul": "Lung",
    "rml": "Lung",
}


# Canonicals that should be returned together because they represent
# the same clinical concept in slightly different vocabulary. Any
# canonical appearing here will be expanded to include its partners.
_EQUIVALENT_CANONICALS: Dict[str, List[str]] = {
    "MSI-H": ["dMMR"],
    "dMMR": ["MSI-H"],
    "MSS":  ["pMMR"],
    "pMMR": ["MSS"],
    # HER2 amplification is often coded as HER2+ in clinical notes
    "HER2 amplification": ["HER2"],
}


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANCER_ONTOLOGY = _REPO_ROOT / "data" / "ontology" / "cancer_type_ontology.json"


# ── Index ───────────────────────────────────────────────────────────────

class SynonymIndex:
    """Loads every synonym source once and exposes ``resolve(axis, term)``.

    Axes
    ----
    - ``"cancer_type"`` — ontology label ("Breast Cancer", ...)
    - ``"site"`` — canonical site ("Oral cavity", "Lung", ...)
    - ``"histology"`` — canonical histology ("Squamous cell carcinoma", ...)
    - ``"biomarker"`` — canonical biomarker ("HER2", "PD-L1", ...)
    - ``"drug"`` — canonical drug ("Pembrolizumab", ...)
    - ``"alteration"`` — genomic alteration ("EGFR L858R", "BRAF V600E", ...)
    """

    def __init__(self) -> None:
        # Per-axis map: lowercased variant → canonical
        self._axis_lookups: Dict[str, Dict[str, str]] = {
            "cancer_type": {},
            "site":        {},
            "histology":   {},
            "biomarker":   {},
            "drug":        {},
            "alteration":  {},
        }
        self._load_keyword_tagger_groups()
        self._load_cancer_ontology()
        self._load_subsite_fallback()

    # ── Loaders ────────────────────────────────────────────────────

    def _load_keyword_tagger_groups(self) -> None:
        """Pull every SYNONYM_GROUPS list from keyword_tagger."""
        try:
            from src.ingestion.keyword_tagger import (
                BIOMARKER_SYNONYM_GROUPS,
                DRUG_SYNONYM_GROUPS,
                HISTOLOGY_SYNONYM_GROUPS,
                SITE_SYNONYM_GROUPS,
            )
        except Exception as e:
            print(f"[SynonymIndex] Warning: could not load keyword_tagger groups: {e}")
            return

        for axis, groups in (
            ("biomarker", BIOMARKER_SYNONYM_GROUPS),
            ("drug",      DRUG_SYNONYM_GROUPS),
            ("histology", HISTOLOGY_SYNONYM_GROUPS),
            ("site",      SITE_SYNONYM_GROUPS),
        ):
            for g in groups:
                canonical = (g.get("canonical") or "").strip()
                if not canonical:
                    continue
                slot = g.get("slot")
                bucket = "alteration" if slot == "alteration" else axis
                lookup = self._axis_lookups[bucket]
                lookup.setdefault(canonical.lower(), canonical)
                for variant in g.get("variants", []) or []:
                    variant = (variant or "").strip()
                    if variant:
                        lookup.setdefault(variant.lower(), canonical)

    def _load_cancer_ontology(self) -> None:
        """Every cancer_type's label + synonyms → cancer_type lookup."""
        try:
            with _CANCER_ONTOLOGY.open("r", encoding="utf-8") as f:
                ontology = json.load(f)
        except Exception as e:
            print(f"[SynonymIndex] Warning: could not load cancer_type_ontology: {e}")
            return
        bucket = self._axis_lookups["cancer_type"]
        for key, entry in ontology.items():
            label = (entry.get("label") or "").strip()
            if not label:
                continue
            bucket.setdefault(key.lower(), label)
            bucket.setdefault(label.lower(), label)
            for syn in entry.get("synonyms", []) or []:
                syn = (syn or "").strip()
                if syn:
                    bucket.setdefault(syn.lower(), label)

    def _load_subsite_fallback(self) -> None:
        site_lookup = self._axis_lookups["site"]
        for variant, canonical in _SUBSITE_FALLBACK.items():
            site_lookup.setdefault(variant.lower(), canonical)

    # ── Public API ────────────────────────────────────────────────

    def resolve(self, axis: str, raw_term: str) -> List[str]:
        """Map a raw term onto a list of canonical values.

        Returns:
            An ordered, deduplicated list of canonical strings. Empty
            list if the term has no mapping (caller should decide how
            to handle unknowns — typically drop them from the filter).
        """
        if not raw_term:
            return []
        lookup = self._axis_lookups.get(axis)
        if lookup is None:
            return []
        key = raw_term.strip().lower()
        canonical = lookup.get(key)
        if not canonical:
            # Strip common suffixes and retry ("breast cancer" when
            # lookup only has "breast")
            for suffix in (" cancer", " carcinoma", " tumor", " tumors"):
                if key.endswith(suffix):
                    canonical = lookup.get(key[: -len(suffix)])
                    if canonical:
                        break
        if not canonical:
            # Normalize "&" ↔ " and " ("head and neck" vs "head & neck")
            alt = key.replace(" and ", " & ") if " and " in key else key.replace(" & ", " and ")
            if alt != key:
                canonical = lookup.get(alt)
        if not canonical:
            return []
        # Expand semantic equivalents if any.
        result = [canonical]
        for equiv in _EQUIVALENT_CANONICALS.get(canonical, []):
            if equiv not in result:
                result.append(equiv)
        return result

    def normalize_many(self, axis: str, raw_terms: List[str]) -> List[str]:
        """Resolve a list of raw terms → deduplicated canonicals.

        Unknowns are dropped silently. Use ``unresolved_terms`` to get
        the complement.
        """
        seen: List[str] = []
        seen_set: Set[str] = set()
        for t in raw_terms:
            for c in self.resolve(axis, t):
                if c not in seen_set:
                    seen.append(c)
                    seen_set.add(c)
        return seen

    def unresolved_terms(self, axis: str, raw_terms: List[str]) -> List[str]:
        """Raw terms that did NOT resolve to any canonical (for debugging)."""
        out: List[str] = []
        for t in raw_terms:
            if not self.resolve(axis, t):
                out.append(t)
        return out

    def canonicals_for(self, axis: str) -> List[str]:
        """The full canonical vocabulary for an axis (for UI/introspection)."""
        return sorted(set(self._axis_lookups.get(axis, {}).values()))


@lru_cache(maxsize=1)
def get_synonym_index() -> SynonymIndex:
    """Process-wide singleton."""
    return SynonymIndex()
