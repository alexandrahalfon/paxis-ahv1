"""
Clinical Profile Enrichment
============================

Fills any unpopulated axes on a ``ClinicalProfile`` from the regex-based
``QueryStructure`` and the raw patient narrative, using the canonical
ontology files in ``data/ontology/`` as the source of truth for:

  - cancer-type synonym → canonical label  (cancer_type_ontology.json)
  - cancer-type → anatomical sites         (cancer_type_sites.json)
  - cancer-type → histology subtypes       (cancer_type_ontology.subtypes)
  - cancer-type → canonical drug list      (cancer_type_ontology.drugs)

Built to replace the previous hardcoded fallback maps in
``enhanced_rag_service.py``. The ontology is loaded once at import time
and reused for every query (the files total ~50 KB).

The enrichment is idempotent and additive — it only writes axes that
are still unset on the profile, so it never overrides values the LLM
extractor already resolved.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "data" / "ontology"


@lru_cache(maxsize=1)
def _load_cancer_ontology() -> Dict[str, Any]:
    with open(_ONTOLOGY_DIR / "cancer_type_ontology.json", "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_cancer_sites() -> Dict[str, List[str]]:
    with open(_ONTOLOGY_DIR / "cancer_type_sites.json", "r") as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _build_all_drugs_lower() -> List[str]:
    """Return a deduplicated, lowercased list of every drug name in the
    cancer-type ontology — used to scan the patient narrative for
    medication mentions regardless of which cancer the drug primarily
    treats (osimertinib appears under lung, abiraterone under
    prostate, etc.).
    """
    seen = set()
    out: List[str] = []
    for entry in _load_cancer_ontology().values():
        for dg in entry.get("drugs", []) or []:
            dl = str(dg).strip().lower()
            if dl and dl not in seen:
                seen.add(dl)
                out.append(dl)
    # Sort by length descending so longer names match before substrings
    # of them ("docetaxel" before "doce", "pembrolizumab" before "pembro")
    out.sort(key=len, reverse=True)
    return out


@lru_cache(maxsize=1)
def _build_synonym_to_key() -> Dict[str, str]:
    """Lowercase synonym / label / key → cancer_type_key.

    Lets us resolve "glioblastoma" / "HNSCC" / "head and neck tumor" /
    "central nervous system cancer" → the right ontology key without
    requiring an exact label match.
    """
    out: Dict[str, str] = {}
    for key, entry in _load_cancer_ontology().items():
        label = str(entry.get("label", "")).strip().lower()
        if label:
            out[label] = key
        out[key.lower()] = key
        for syn in entry.get("synonyms", []) or []:
            sl = str(syn).strip().lower()
            if sl:
                out[sl] = key
        # subtypes can resolve to the parent cancer-type key as well —
        # so a query that says "glioblastoma" still gets ``cns``
        for sub in entry.get("subtypes", []) or []:
            sl = str(sub).strip().lower()
            if sl:
                out.setdefault(sl, key)
    return out


@lru_cache(maxsize=1)
def _build_all_subtypes() -> List[Tuple[str, str]]:
    """Return a list of (lowercased_subtype, canonical_subtype) pairs
    across every cancer type. Sorted longest-first so "Invasive ductal
    carcinoma" matches before "carcinoma".
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for entry in _load_cancer_ontology().values():
        for sub in entry.get("subtypes", []) or []:
            sl = str(sub).strip()
            if not sl:
                continue
            low = sl.lower()
            if low in seen:
                continue
            seen.add(low)
            pairs.append((low, sl))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


@lru_cache(maxsize=1)
def _build_all_sites() -> List[Tuple[str, str]]:
    """Return (lowercased_site, canonical_site) pairs from
    cancer_type_sites.json. Sorted longest-first so "Oral cavity" wins
    over "oral".
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for sites in _load_cancer_sites().values():
        for s in sites or []:
            sl = str(s).strip()
            if not sl:
                continue
            low = sl.lower()
            if low in seen:
                continue
            seen.add(low)
            pairs.append((low, sl))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


# ── QueryStructure regex-key → ontology key mapping ────────────────────
# The regex extractor uses its own short keys (``head_neck``, ``cns``,
# ``breast`` …); map them to the ontology keys (``h_n``, ``cns`` …) so
# we can pull the right entry.
_QS_SITE_TO_ONT_KEY: Dict[str, str] = {
    "head_neck":  "h_n",
    "breast":     "breast",
    "lung":       "lung",
    "prostate":   "prostate",
    "cns":        "cns",
    "skin":       "cutaneous",
    "cutaneous":  "cutaneous",
    "thyroid":    "thyroid",
    "sarcoma":    "sarcoma",
    "lymphoma":   "lymphoma",
    "leukemia":   "leukemia",
    "myeloma":    "myeloma",
    "gi":         "gi",
    "gu":         "gu",
    "gyn":        "gyn",
    "colorectal": "gi",
    "gastric":    "gi",
    "esophageal": "gi",
    "pancreatic": "gi",
    "hepatic":    "gi",
    "bladder":    "gu",
    "renal":      "gu",
    "kidney":     "gu",
    "ovarian":    "gyn",
    "cervical":   "gyn",
    "endometrial": "gyn",
    "neuroendocrine": "neuroendocrine",
    "mesothelioma":   "mesothelioma",
}


# ── Public API ─────────────────────────────────────────────────────────

def enrich_profile_from_query_structure(profile, query_structure, raw_query: str) -> None:
    """Mutate `profile` in place, filling unset axes from QueryStructure
    + raw narrative using the cancer-type ontology.

    Args:
        profile: A ``ClinicalProfile`` (or duck-typed equivalent).
        query_structure: The regex-extracted ``QueryStructure`` from
            ``query_structuring_service`` — used for short ontology
            keys and structured stage/disease_status fields.
        raw_query: The user's full patient narrative — scanned for
            ontology drug / subtype / site / synonym mentions.

    No return value — the profile is updated in place. Safe to call
    multiple times; idempotent.
    """
    raw_lower = (raw_query or "").lower()

    # 1. cancer_type_label — resolve via ontology synonyms / subtypes
    #    if the LLM extractor came up empty
    if not profile.cancer_type_label:
        ont_key = _resolve_ontology_key(query_structure, raw_lower)
        if ont_key:
            label = _load_cancer_ontology()[ont_key].get("label")
            if label:
                profile.cancer_type_label = label
                profile.cancer_type_key = ont_key
                print(f"[ClinicalProfile] Fallback cancer_type from ontology: {label!r} (key={ont_key})")

    # 2. cancer_sites — populate from QueryStructure.site_detail +
    #    scan ontology sites in raw text
    if not profile.cancer_sites:
        sites: List[str] = []

        # 2a. site_detail from QueryStructure regex (e.g. 'oral_cavity'
        #     → 'Oral cavity')
        sd = getattr(query_structure.cancer, "site_detail", None) if query_structure else None
        if sd:
            sd_pretty = sd.replace("_", " ").strip()
            for low, canonical in _build_all_sites():
                if low == sd_pretty.lower():
                    if canonical not in sites:
                        sites.append(canonical)
                    break
            else:
                # No ontology hit — keep the regex value title-cased
                title = sd_pretty.title()
                if title and title not in sites:
                    sites.append(title)

        # 2b. Scan the raw narrative for site keywords (catches
        #     'bone metastases', 'liver mets', etc.). Word-boundary
        #     match prevents 'bone' matching inside 'backbone'.
        for low, canonical in _build_all_sites():
            if re.search(rf"\b{re.escape(low)}\b", raw_lower) and canonical not in sites:
                sites.append(canonical)

        if sites:
            profile.cancer_sites = sites
            print(f"[ClinicalProfile] Fallback cancer_sites from ontology: {sites}")

    # 3. histologies — scan ontology subtypes in raw narrative
    if not profile.histologies:
        hist: List[str] = []
        # Use the ontology subtypes for the resolved cancer type
        # first (more relevant), then fall back to all subtypes
        primary_key = profile.cancer_type_key or _resolve_ontology_key(query_structure, raw_lower)
        primary_subtypes: List[Tuple[str, str]] = []
        if primary_key and primary_key in _load_cancer_ontology():
            for sub in _load_cancer_ontology()[primary_key].get("subtypes", []) or []:
                sl = str(sub).strip()
                if sl:
                    primary_subtypes.append((sl.lower(), sl))
            primary_subtypes.sort(key=lambda p: -len(p[0]))

        for low, canonical in primary_subtypes + _build_all_subtypes():
            # Word-boundary match — prevents 'sclc' inside 'nsclc',
            # 'dcis' inside 'ldcis', etc.
            if re.search(rf"\b{re.escape(low)}\b", raw_lower) and canonical not in hist:
                hist.append(canonical)
                if len(hist) >= 4:
                    break

        # Fall back to regex-extracted histology (covers values the
        # ontology's subtype list doesn't enumerate — IDC / ILC / DCIS
        # aren't in breast's subtypes (which are receptor-status based),
        # but doc-level metadata still tags them)
        if not hist and query_structure and query_structure.cancer.histology:
            from_regex = {
                "scc":            "Squamous cell carcinoma",
                "adenocarcinoma": "Adenocarcinoma",
                "small_cell":     "Small Cell Carcinoma",
                "large_cell":     "Large Cell Carcinoma",
                "transitional":   "Transitional Cell Carcinoma",
                "clear_cell":     "Clear Cell Carcinoma",
                "idc":            "Invasive Ductal Carcinoma",
                "ilc":            "Invasive Lobular Carcinoma",
                "dcis":           "Ductal Carcinoma In Situ",
                "lcis":           "Lobular Carcinoma In Situ",
                "seminoma":       "Seminoma",
                "nonseminoma":    "Non-Seminomatous Germ Cell Tumor",
            }.get(query_structure.cancer.histology)
            if from_regex:
                hist.append(from_regex)

        if hist:
            profile.histologies = hist
            print(f"[ClinicalProfile] Fallback histologies from ontology: {hist}")

    # 4. stages — title-cased from QueryStructure regex
    if not profile.stages and query_structure and query_structure.cancer.stage:
        profile.stages = [f"Stage {query_structure.cancer.stage}"]
        print(f"[ClinicalProfile] Fallback stage from QueryStructure: 'Stage {query_structure.cancer.stage}'")

    # 5. prior_treatments — scan the FULL 252-drug list against the
    #    patient narrative + LLM treatment axes
    if not profile.prior_treatments:
        treatments: List[str] = []
        # Pull existing regex-extracted prior treatments first
        for t in (getattr(query_structure.treatment, "prior_treatments", None) or []) if query_structure else []:
            if t and t not in treatments:
                treatments.append(t)
        # Scan the raw narrative against all canonical drug names
        for drug_lower in _build_all_drugs_lower():
            # word-boundary match prevents 'ara-c' matching inside
            # 'caracal' etc.
            if re.search(rf"\b{re.escape(drug_lower)}\b", raw_lower):
                # Use original ontology casing for display
                canonical = _find_drug_canonical(drug_lower)
                if canonical and canonical not in treatments:
                    treatments.append(canonical)
        if treatments:
            profile.prior_treatments = treatments
            print(f"[ClinicalProfile] Fallback prior_treatments from ontology: {treatments}")

    # 6. disease_status — from QueryStructure.clinical_history; also
    #    map 'post_progression' → 'recurrent' (most studies tag the
    #    latter rather than the former)
    if not profile.disease_status and query_structure:
        statuses: List[str] = []
        ch = getattr(query_structure, "clinical_history", None)
        ds = getattr(ch, "disease_status", None) if ch is not None else None
        if ds:
            statuses.append(ds)
        # Look in the narrative for common status tokens the regex
        # might miss
        for token in ("recurrent", "metastatic", "progressive", "refractory",
                      "relapsed", "unresectable", "advanced"):
            if token in raw_lower and token not in statuses:
                statuses.append(token)
        if "post_progression" in statuses and "recurrent" not in statuses:
            statuses.append("recurrent")
        if statuses:
            profile.disease_status = statuses
            print(f"[ClinicalProfile] Fallback disease_status: {statuses}")


# ── Helpers ────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _find_drug_canonical(drug_lower: str) -> Optional[str]:
    """Look up the canonical casing of a drug from the ontology."""
    for entry in _load_cancer_ontology().values():
        for dg in entry.get("drugs", []) or []:
            if str(dg).strip().lower() == drug_lower:
                return str(dg).strip()
    return None


def _resolve_ontology_key(query_structure, raw_lower: str) -> Optional[str]:
    """Resolve a cancer-type key from the QueryStructure's regex site
    field first; fall back to scanning ontology synonyms / subtypes /
    labels in the raw narrative.
    """
    # 1. QueryStructure regex (most precise when it fires)
    if query_structure is not None:
        site = getattr(query_structure.cancer, "site", None)
        if site and site in _QS_SITE_TO_ONT_KEY:
            return _QS_SITE_TO_ONT_KEY[site]

    # 2. Synonym / subtype scan against the narrative
    syn_map = _build_synonym_to_key()
    # Sort by length descending so longer phrases hit first
    # (e.g. 'central nervous system' before 'cns')
    keys_sorted = sorted(syn_map.keys(), key=len, reverse=True)
    for phrase in keys_sorted:
        if len(phrase) < 4:
            continue  # avoid 3-letter false positives like 'gyn' in unrelated text
        if re.search(rf"\b{re.escape(phrase)}\b", raw_lower):
            return syn_map[phrase]

    return None
