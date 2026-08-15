"""
Enhanced Patient Matching Service

Focus: Find clinical studies with similar patients using semantic search.
Includes filtering, caching, validation, and improved matching.
"""

import asyncio
import re
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

from .enhanced_rag_service import get_cross_encoder, _build_reranker_passage


# Simple in-memory cache for embeddings (can be upgraded to Redis later)
_embedding_cache = {}
_cache_max_size = 100


# Cancer-type canonicalization — shared by patient profile and synthetic
# doc_level builders so the two sides of the match score against identical
# canonical values (otherwise PG's literal "NSCLC" never matches the patient
# profile's canonical "Lung Cancer" label, score collapses to ~0).
_SITE_TO_KEY = {
    "lung": "lung", "nsclc": "lung", "sclc": "lung", "non-small cell": "lung", "small cell": "lung",
    "breast": "breast",
    "prostate": "prostate",
    "colon": "gi", "colorectal": "gi", "rectal": "gi", "rectum": "gi",
    "esophageal": "gi", "esophagus": "gi",
    "gastric": "gi", "stomach": "gi",
    "pancreatic": "gi", "pancreas": "gi",
    "liver": "gi", "hepatocellular": "gi", "hepat": "gi",
    "head and neck": "h_n", "head & neck": "h_n", "h&n": "h_n",
    "oral cavity": "h_n", "oropharynx": "h_n", "larynx": "h_n",
    "nasopharynx": "h_n", "hypopharynx": "h_n", "maxilla": "h_n",
    "cervical": "gyn", "cervix": "gyn",
    "endometrial": "gyn", "uterine": "gyn",
    "ovarian": "gyn", "ovary": "gyn",
    "bladder": "gu", "kidney": "gu", "renal": "gu",
    "glioblastoma": "cns", "glioma": "cns", "brain": "cns", "cns": "cns",
    "melanoma": "cutaneous", "skin": "cutaneous", "cutaneous": "cutaneous",
    "thyroid": "thyroid",
    "lymphoma": "lymphoma",
    "leukemia": "leukemia",
    "myeloma": "myeloma",
}

_LABEL_TO_CANONICAL = {
    "lung": "Lung Cancer", "breast": "Breast Cancer",
    "prostate": "Prostate Cancer", "gi": "Gastrointestinal Cancers",
    "h_n": "Head and Neck Cancer", "gyn": "Gynecologic Cancers",
    "gu": "Genitourinary Cancers", "cns": "Central Nervous System Tumors",
    "cutaneous": "Skin Cancer", "thyroid": "Thyroid Cancer",
    "lymphoma": "Lymphoma", "leukemia": "Leukemia",
    "myeloma": "Multiple Myeloma",
}


def _canonical_cancer_label(*free_text_inputs: str) -> Optional[str]:
    """Map any free-text cancer descriptor (raw cancer_type, anatomical site,
    PG diagnosis Cancer Type field, …) to the canonical title-case label the
    patient_match_scorer compares against. Returns None if no token matches.
    """
    for inp in free_text_inputs:
        if not inp:
            continue
        needle = str(inp).lower().strip()
        for token, key in _SITE_TO_KEY.items():
            if token in needle:
                return _LABEL_TO_CANONICAL.get(key)
    return None


def _pg_study_to_doc_level(
    pg_study: Dict[str, Any],
    match_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a synthetic doc_level_* dict from a PostgresStudyDetailsService
    result. Used as a fallback when a PG-only doc_id doesn't exist in Qdrant
    (which happens when PG returns DOI-derived or title-derived doc_ids that
    don't match Qdrant's ingestion-time doc_id format — observed live for
    studies like 'doi_10.1056_nejmoa2302983.' with a trailing period, and
    title-based ids like 'Pembrolizumab plus Chemotherapy for Squa...').

    Two sources are merged so we always get cancer_type/cancer_location:
      - match_details (from structured_study_matcher's match_details[doc_id])
        — has reliable cancer_location and cancer_type strings populated by
        the matcher's own SELECT
      - pg_study (from PostgresStudyDetailsService.get_study_details) — has
        rich biomarkers, treatments, and staging components but its
        diagnosis dict is often empty for these doc_ids (different DB shape)

    Maps the merged structured fields to the same axis keys the canonical
    patient_match_scorer reads (`doc_level_cancer_types`, `doc_level_sites`,
    etc.) so the scorer can run identically on both data sources.
    """
    diagnosis = pg_study.get("diagnosis", {}) or {}
    staging = pg_study.get("staging", {}) or {}
    treatment = pg_study.get("treatment", {}) or {}
    biomarkers = pg_study.get("biomarkers", []) or []
    md = match_details or {}

    def _diag_val(field_label: str) -> List[str]:
        v = (diagnosis.get(field_label, {}) or {}).get("value")
        return [v] if v else []

    def _md_val(key: str) -> List[str]:
        v = md.get(key)
        return [v] if v else []

    def _staging_categories() -> List[str]:
        # stage_distribution rows often contain noise (per-arm strata like
        # 'pT1a/pT1b' that aren't real stage groups). Only keep values that
        # look like an AJCC stage group: I/II/III/IV optionally with a/b/c
        # sub-stage suffix, optionally preceded by 'p' or 'c'.
        stage_re = re.compile(r'^[cp]?(IV|III|II|I)[ABCabc]?$')
        cats: List[str] = []
        for r in (staging.get("stage_distribution") or []):
            cat = r.get("stage_category")
            if not cat:
                continue
            cstr = str(cat).strip()
            # Some stage_category cells contain slash-separated groups like
            # 'pIIIA/pIV' — split and keep any element that looks like a
            # real stage group.
            for tok in re.split(r"[/,;]", cstr):
                tok = tok.strip()
                if tok and stage_re.match(tok):
                    cats.append(tok)
        if not cats:
            ms = (staging.get("Metastatic Status", {}) or {}).get("value")
            if ms:
                cats = [ms]
        enriched: List[str] = []
        for c in cats:
            enriched.append(c)
            if not c.lower().startswith("stage"):
                enriched.append(f"Stage {c}")
        return enriched

    def _drug_names() -> List[str]:
        """Flatten chemotherapy_regimens[].drugs which can be a string OR a
        list of strings depending on the DB row."""
        out: List[str] = []
        for r in (treatment.get("chemotherapy_regimens") or []):
            d = r.get("drugs")
            if not d:
                continue
            if isinstance(d, list):
                out.extend(str(x) for x in d if x)
            elif isinstance(d, str):
                out.append(d)
        return out

    raw_cancer_type = _diag_val("Cancer Type") + _md_val("cancer_type")
    raw_location = _diag_val("Cancer Location") + _md_val("cancer_location")
    raw_histology = _diag_val("Histopathologic Type")

    # cancer_type: include the raw PG value AND the canonical label
    # (e.g. PG 'NSCLC' → also emit 'Lung Cancer') so case-insensitive
    # equality against the patient profile's `cancer_type_label` succeeds.
    cancer_types: List[str] = []
    for v in raw_cancer_type:
        if v and v not in cancer_types:
            cancer_types.append(v)
    canonical_label = _canonical_cancer_label(
        *raw_cancer_type, *raw_location, *raw_histology
    )
    if canonical_label and canonical_label not in cancer_types:
        cancer_types.append(canonical_label)

    # sites: enrich raw location ('Peripheral lung') with a normalized
    # token ('lung') so it matches patient profiles where cancer_sites is a
    # single word.
    sites: List[str] = []
    for v in raw_location:
        if v and v not in sites:
            sites.append(v)
    for loc in raw_location:
        needle = (loc or "").lower()
        for token, _key in _SITE_TO_KEY.items():
            if token in needle and token not in sites and token != needle:
                sites.append(token)
                break

    return {
        "doc_level_sites": sites,
        "doc_level_cancer_types": cancer_types,
        "doc_level_histologies": raw_histology,
        "doc_level_stages": _staging_categories(),
        "doc_level_biomarkers": [
            b.get("biomarker_name") for b in biomarkers if b.get("biomarker_name")
        ],
        "doc_level_drugs": _drug_names(),
        "doc_level_disease_status": (
            [(staging.get("Metastatic Status", {}) or {}).get("value")]
            if (staging.get("Metastatic Status", {}) or {}).get("value")
            else []
        ),
    }


async def score_pg_only_match(
    qdrant_client: QdrantClient,
    collection: str,
    doc_id: str,
    clinical_profile,
    pg_study_fallback: Optional[Dict[str, Any]] = None,
    match_details: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Score a PG-only match using the canonical patient_match_scorer.

    The PG matcher returns only structured fields (study_name, cancer_location,
    num_patients, …) from PostgreSQL — it never fetches doc_level_*
    (doc_level_cancer_types, doc_level_sites, doc_level_stages, etc.) because
    those live exclusively in Qdrant payloads. So PG-only matches arrive
    without the metadata the scorer needs and end up with
    patient_match_score=None in the response.

    This helper closes that gap: one Qdrant scroll per doc_id to fetch a
    representative chunk's `metadata`, then run the same scorer call site as
    comprehensive_retrieval.py:1220 (the standard pipeline). Returns
    (score, breakdown) — both None on any failure (missing chunk, empty
    metadata, scorer error). Callers stamp the values onto the match dict.
    """
    if clinical_profile is None or not clinical_profile.has_any_filter():
        print(f"[score_pg_only_match] {doc_id[:40]} — skipped: no clinical_profile filters")
        return None, None
    try:
        # Fetch ONE chunk for this doc_id — doc_level_* fields are study-level
        # so any chunk carries the same metadata.
        scroll_result = await asyncio.to_thread(
            qdrant_client.scroll,
            collection_name=collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        from src.api.services.patient_match_scorer import score_patient_match

        points = scroll_result[0] if scroll_result else []
        metadata: Dict[str, Any] = {}
        source = "qdrant"
        if points:
            metadata = (points[0].payload or {}).get("metadata") or {}

        # Fallback: build a synthetic doc_level dict from the PG study record
        # when Qdrant either has no point for this doc_id (the common case —
        # PG/Qdrant doc_id formats drift) or the point has no metadata.
        if not metadata and (pg_study_fallback or match_details):
            metadata = _pg_study_to_doc_level(pg_study_fallback or {}, match_details)
            source = "pg_fallback"

        if not metadata or not any(metadata.values()):
            print(f"[score_pg_only_match] {doc_id[:40]} — no usable doc_level data (Qdrant miss + no PG fallback)")
            return None, None

        pm = score_patient_match(clinical_profile, metadata)
        score = pm.get("score")
        axes = pm.get("axes_used") or []

        # For pg_fallback matches the synthetic doc_level is often sparse —
        # the patient_match_scorer's strict denominator mode ("count only
        # axes the study reports") then over-rewards: a study where only
        # cancer_type + sites match gives 100% because the other axes are
        # treated as NA. That's the right behavior for Qdrant-stored data
        # with rich doc_level_*, but wrong for PG-fallback where missing
        # axes are a data-extraction artifact, not "the study didn't
        # measure this". Recompute using the legacy denominator (all
        # patient axes) and take the lower number.
        if source == "pg_fallback":
            breakdown = pm.get("breakdown") or {}
            weighted_sum = sum(
                (v.get("ratio") or 0) * (v.get("weight") or 0)
                for v in breakdown.values()
                if v.get("patient")
            )
            legacy_total = sum(
                (v.get("weight") or 0)
                for v in breakdown.values()
                if v.get("patient")
            )
            if legacy_total > 0:
                legacy_score = int(round(100.0 * weighted_sum / legacy_total))
                if legacy_score < score:
                    print(
                        f"[score_pg_only_match] {doc_id[:40]} — legacy denom "
                        f"penalty: {score} → {legacy_score} "
                        f"(strict undercounted {legacy_total - sum((v.get('weight') or 0) for v in breakdown.values() if v.get('study_has_data'))} weight of missing-axis penalty)"
                    )
                    score = legacy_score
                    pm["score"] = score

        print(f"[score_pg_only_match] {doc_id[:40]} — score={score} axes={len(axes)} ({source})")
        # Verbose diagnostic when score is suspiciously high (≥90) or
        # collapsed (0) — both are signals something is off.
        if (score == 0 and len(axes) > 0) or (score is not None and score >= 90):
            print(f"[score_pg_only_match]   doc_level: {metadata}")
            print(f"[score_pg_only_match]   patient cancer_type_label={getattr(clinical_profile, 'cancer_type_label', None)!r} cancer_sites={getattr(clinical_profile, 'cancer_sites', None)} stages={getattr(clinical_profile, 'stages', None)} biomarkers={getattr(clinical_profile, 'biomarkers', None)} prior_treatments={getattr(clinical_profile, 'prior_treatments', None)} disease_status={getattr(clinical_profile, 'disease_status', None)}")
            print(f"[score_pg_only_match]   breakdown: {pm.get('breakdown')}")
        return score, pm
    except Exception as e:
        print(f"[score_pg_only_match] Failed for {doc_id[:40]}: {e}")
        return None, None


def _build_clinical_profile_from_patient_dict(profile_dict: Dict[str, Any]):
    """Construct a ClinicalProfile from a patient_profile dict so the
    canonical patient_match_scorer can be run against multi-specialty
    results (mirrors what the chat pipeline does via
    clinical_profile_enrichment.enrich_profile_from_query_structure).

    Returns None if the dict has nothing matchable.
    """
    try:
        from src.api.services.clinical_extractor import ClinicalProfile
    except Exception:
        return None

    if not profile_dict:
        return None

    profile = ClinicalProfile()

    # cancer_type_label + key — map common forms to ontology keys
    cancer_type = (profile_dict.get("cancer_type") or "").strip()
    anatomical_site = (profile_dict.get("anatomical_site") or "").strip()
    # Reuse the module-level _SITE_TO_KEY / _LABEL_TO_CANONICAL tables so
    # patient profile and synthetic PG doc_level builder normalize to the
    # SAME canonical form — otherwise PG 'NSCLC' never matches patient
    # 'Lung Cancer' under the scorer's case-insensitive equality.
    ct_key = None
    for needle in (cancer_type.lower(), anatomical_site.lower()):
        if not needle:
            continue
        for token, key in _SITE_TO_KEY.items():
            if token in needle:
                ct_key = key
                break
        if ct_key:
            break
    if ct_key:
        profile.cancer_type_key = ct_key
        profile.cancer_type_label = _LABEL_TO_CANONICAL.get(ct_key)
    elif cancer_type:
        profile.cancer_type_label = cancer_type  # Keep raw — better than None

    # cancer_sites — from anatomical_site
    if anatomical_site:
        profile.cancer_sites = [anatomical_site]

    # histologies — pass through directly
    histology = profile_dict.get("histology")
    if histology:
        profile.histologies = [str(histology).strip()]

    # stages — from cancer_stage, formatted as "Stage X"
    stage = profile_dict.get("cancer_stage")
    if stage:
        s = str(stage).strip()
        profile.stages = [s if s.lower().startswith("stage") else f"Stage {s}"]

    # biomarkers — apply polarity-aware formatter so "EGFR-positive" stays
    # intact rather than getting corrupted at scoring time
    markers = profile_dict.get("molecular_markers") or []
    if markers:
        profile.biomarkers = [_format_marker_polarity(m) for m in markers if m]

    # prior_treatments — pass through
    pts = profile_dict.get("prior_treatments") or []
    if pts:
        profile.prior_treatments = [str(t).strip() for t in pts if t]

    # disease_status — from recurrence_status / disease_descriptor, and
    # inferred from stage IV / tnm_m=M1 since the LLM extractor doesn't
    # always populate disease_descriptor. Without this inference a stage IV
    # patient's disease_status stays empty and the scorer can't tell their
    # metastatic profile apart from localized-disease studies — we observed
    # a stage IA surgical trial scoring 100% for a stage IV NSCLC patient
    # in live testing because every axis except (missing) disease_status
    # superficially matched.
    statuses = []
    rs = profile_dict.get("recurrence_status")
    if rs:
        statuses.append(str(rs).lower().strip())
    dd = profile_dict.get("disease_descriptor")
    if dd:
        statuses.append(str(dd).lower().strip())
    # Stage IV / M1 → metastatic
    stage_raw = str(profile_dict.get("cancer_stage") or "").upper()
    tnm_m_raw = str(profile_dict.get("tnm_m") or "").upper()
    if ("IV" in stage_raw or "4" in stage_raw or tnm_m_raw.startswith("M1")):
        if "metastatic" not in statuses:
            statuses.append("metastatic")
    if statuses:
        profile.disease_status = statuses

    return profile


def _format_marker_polarity(m: str) -> str:
    """Polarity-aware formatter for a single molecular_markers entry.

    Inputs are heterogeneous because they come from multiple upstream
    extractors (Pydantic PatientProfile from /patient/match callers,
    LLM unstructured extractor at unstructured_patient_extractor.py,
    etc.). Accept all of: "EGFR+", "EGFR-", "EGFR positive",
    "EGFR-positive", "HPV-negative", "p16 positive", "BRAF V600E mutant".

    Rules (checked in order):
      1. If the string already contains a polarity word (positive /
         negative / mutant / wild-type), keep it as-is. This is the
         critical case the previous heuristic broke: "EGFR-positive"
         was wrongly tagged as negative because of the connector
         hyphen.
      2. Otherwise, if it ends with `+` or `-` at a token boundary,
         translate to the explicit polarity word.
      3. Otherwise return as-is (no detectable polarity).
    """
    if not m:
        return m
    text = str(m).strip()
    if not text:
        return m
    text_lower = text.lower()
    polarity_words = ("positive", "negative", "mutant", "mutation",
                      "mutated", "wild-type", "wild type", "wildtype",
                      "amplified", "amplification", "overexpression",
                      "overexpressed", "fusion", "rearrangement",
                      "translocation", "deficient", "proficient")
    for word in polarity_words:
        if word in text_lower:
            return text  # Already has word polarity
    # Trailing +/- shorthand
    if text.endswith("+"):
        return f"{text[:-1].rstrip()} positive"
    if text.endswith("-"):
        return f"{text[:-1].rstrip()} negative"
    return text


# Canonical site keys accepted by the PG matcher's SITE_TO_LOCATION_PATTERNS.
# Must stay in sync with src/api/services/structured_study_matcher.py:956.
_CANONICAL_SITES = {
    "head_neck", "breast", "lung", "prostate", "gi", "gyn",
    "gu", "cns", "lymphoma", "sarcoma", "skin", "thyroid",
}

# anatomical_site values → canonical site key.
# Substring match, longest patterns first. None of the values here are
# anatomical_site values we expect from the LLM extractor (maxilla, oral cavity,
# left upper lobe, …); they're not free-form cancer names.
_ANATOMICAL_SITE_TO_CANONICAL = [
    # head & neck
    ("oral cavity", "head_neck"), ("oropharyn", "head_neck"),
    ("hypopharyn", "head_neck"), ("nasopharyn", "head_neck"),
    ("maxilla", "head_neck"), ("mandible", "head_neck"),
    ("buccal", "head_neck"), ("tongue", "head_neck"),
    ("tonsil", "head_neck"), ("palate", "head_neck"),
    ("gingiva", "head_neck"), ("salivary", "head_neck"),
    ("parotid", "head_neck"), ("larynx", "head_neck"),
    ("pharynx", "head_neck"), ("glottis", "head_neck"),
    ("neck", "head_neck"), ("sinus", "head_neck"),
    # lung / thoracic
    ("upper lobe", "lung"), ("lower lobe", "lung"), ("middle lobe", "lung"),
    ("bronch", "lung"), ("pulmonary", "lung"),
    ("mediastin", "lung"), ("pleura", "lung"),
    ("lung", "lung"),
    # breast
    ("breast", "breast"), ("mammary", "breast"), ("axillary tail", "breast"),
    # prostate / GU
    ("prostate", "prostate"),
    ("bladder", "gu"), ("kidney", "gu"), ("renal", "gu"),
    ("ureter", "gu"), ("urothel", "gu"), ("testis", "gu"), ("testicul", "gu"),
    # GI
    ("esophag", "gi"), ("stomach", "gi"), ("gastric", "gi"),
    ("colon", "gi"), ("rectum", "gi"), ("rectal", "gi"),
    ("anal", "gi"), ("anus", "gi"),
    ("liver", "gi"), ("hepat", "gi"),
    ("pancrea", "gi"), ("gallbladder", "gi"), ("bile", "gi"),
    # GYN
    ("cervix", "gyn"), ("cervical", "gyn"),
    ("uterus", "gyn"), ("uterine", "gyn"), ("endometri", "gyn"),
    ("ovary", "gyn"), ("ovarian", "gyn"),
    ("vulva", "gyn"), ("vagina", "gyn"),
    # CNS
    ("brain", "cns"), ("cerebr", "cns"), ("glioma", "cns"),
    ("meningioma", "cns"), ("spinal", "cns"), ("spine", "cns"),
    # Skin
    ("skin", "skin"), ("cutaneous", "skin"), ("melanoma", "skin"),
    ("merkel", "skin"),
    # Thyroid
    ("thyroid", "thyroid"),
]

# cancer_type values → canonical site key. Used when anatomical_site is absent.
_CANCER_TYPE_TO_CANONICAL = [
    # Lung
    ("nsclc", "lung"), ("sclc", "lung"), ("lung", "lung"),
    # Breast
    ("breast", "breast"), ("dcis", "breast"),
    # Prostate / GU
    ("prostate", "prostate"), ("prostatic", "prostate"),
    ("bladder", "gu"), ("kidney", "gu"), ("renal", "gu"),
    ("urothelial", "gu"), ("rcc", "gu"),
    # Head & neck (full phrases before short tokens)
    ("head and neck", "head_neck"), ("head & neck", "head_neck"),
    ("hnscc", "head_neck"), ("h&n", "head_neck"),
    # GI
    ("colorectal", "gi"), ("colon", "gi"), ("rectal", "gi"),
    ("esophageal", "gi"), ("esophagus", "gi"),
    ("gastric", "gi"), ("stomach", "gi"),
    ("pancreatic", "gi"), ("pancreas", "gi"),
    ("hepatocellular", "gi"), ("hcc", "gi"), ("liver", "gi"),
    # GYN
    ("cervical", "gyn"), ("cervix", "gyn"),
    ("endometrial", "gyn"), ("uterine", "gyn"),
    ("ovarian", "gyn"), ("ovary", "gyn"),
    # CNS
    ("glioblastoma", "cns"), ("gbm", "cns"), ("glioma", "cns"), ("brain", "cns"),
    # Skin
    ("melanoma", "skin"),
    # Lymphoma / sarcoma / thyroid
    ("lymphoma", "lymphoma"),
    ("sarcoma", "sarcoma"),
    ("thyroid", "thyroid"),
]

# cancer_type values that are actually histologies (not anatomical sites).
# When the extractor labels these as cancer_type and profile.histology is empty,
# route them into histology instead of dropping them.
_HISTOLOGY_LIKE_CANCER_TYPES = (
    "scc", "squamous cell carcinoma", "squamous",
    "adenocarcinoma", "adeno",
    "small cell", "sclc",  # sclc resolves to "lung" via cancer-type map first
    "transitional cell", "urothelial carcinoma",
    "clear cell carcinoma",
    "ductal carcinoma", "lobular carcinoma",
    "neuroendocrine",
)


def _anatomical_site_to_canonical(anatomical_site: str) -> Optional[str]:
    """Map a free-text anatomical_site to a PG-matcher canonical key.

    Returns None if no pattern matches — caller should leave cancer.site unset
    rather than fall back to a non-canonical value.
    """
    s = anatomical_site.lower().strip()
    if not s:
        return None
    for pattern, canonical in _ANATOMICAL_SITE_TO_CANONICAL:
        if pattern in s:
            return canonical
    return None


def _cancer_type_to_canonical(cancer_type: str) -> Optional[str]:
    """Map a free-text cancer_type to a PG-matcher canonical key.

    Returns None when cancer_type doesn't look like an anatomical descriptor
    (e.g. "SCC" is a histology — caller routes it elsewhere).
    """
    s = cancer_type.lower().strip()
    if not s:
        return None
    for pattern, canonical in _CANCER_TYPE_TO_CANONICAL:
        if pattern in s:
            return canonical
    return None


def _looks_like_histology(cancer_type: str) -> bool:
    """Heuristic — true when cancer_type is a histology label, not a site.

    Uses negative lookbehind to avoid the "non-small cell" → small_cell trap:
    "small cell" is a substring of "non-small cell lung cancer" but the patient
    has NSCLC, not SCLC. The same applies to "non-squamous", "non-adeno", etc.
    """
    s = cancer_type.lower().strip()
    for h in _HISTOLOGY_LIKE_CANCER_TYPES:
        # `(?<!non[- ])` rejects "non-small cell" / "non small cell" but allows
        # plain "small cell". `(?<!non)` alone wouldn't catch "non small".
        if re.search(rf'(?<!non[- ])(?<!non)\b{re.escape(h)}', s):
            return True
    return False


def build_query_structure_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a patient profile dict to the query_structure format expected by
    match_studies_by_structure for PostgreSQL structured matching.

    Args:
        profile: Patient profile dict with keys like cancer_type, cancer_stage,
                 histology, molecular_markers, age, gender, performance_status, etc.

    Returns:
        Dict with nested cancer, patient, treatment structures
    """
    if not profile:
        return {}

    # ── Site resolution ──────────────────────────────────────────────────
    # The PG matcher validates `cancer.site` against SITE_TO_LOCATION_PATTERNS
    # in structured_study_matcher.py — only these 12 canonical keys are
    # accepted: head_neck, breast, lung, prostate, gi, gyn, gu, cns, lymphoma,
    # sarcoma, skin, thyroid. Anything else fails validation silently and the
    # hard site filter is disabled, which is how oral-SCC patients were
    # getting NSCLC and prostate trials at rank #1.
    #
    # Precedence:
    #   1. anatomical_site (e.g. "maxilla", "left upper lobe") → canonical key
    #      via ANATOMICAL_SITE_TO_CANONICAL. Raw value stays in `site_detail`.
    #   2. else cancer_type via CANCER_TYPE_TO_CANONICAL.
    #   3. else if cancer_type looks like a histology (SCC, adenocarcinoma, …)
    #      and `profile.histology` is empty, route it into histology instead.
    #   4. else cancer.site stays unset — never fall back to raw cancer_type.
    cancer: Dict[str, Any] = {}

    anatomical_site = (profile.get("anatomical_site") or "").strip()
    cancer_type = (profile.get("cancer_type") or "").strip()

    # 1. anatomical_site → canonical site
    if anatomical_site:
        canonical = _anatomical_site_to_canonical(anatomical_site)
        if canonical:
            cancer["site"] = canonical
        cancer["site_detail"] = anatomical_site  # keep specific sub-site

    # 2. cancer_type → canonical site (only if anatomical didn't resolve one)
    if not cancer.get("site") and cancer_type:
        canonical = _cancer_type_to_canonical(cancer_type)
        if canonical:
            cancer["site"] = canonical

    # 3. cancer_type that's actually a histology → histology slot (only if
    #    profile.histology is empty, so we don't clobber an explicit value)
    histology_input = (profile.get("histology") or "").strip()
    if cancer_type and not histology_input and _looks_like_histology(cancer_type):
        histology_input = cancer_type

    # ── Histology mapping ────────────────────────────────────────────────
    if histology_input:
        histology = histology_input.lower()
        histology_mapping = {
            "adenocarcinoma": "adenocarcinoma",
            "squamous": "squamous",
            "scc": "squamous",  # the canonical histology key is "squamous"
            "small cell": "small_cell",
            "large cell": "large_cell",
            "ductal": "ductal",
            "lobular": "lobular",
            "clear cell": "clear_cell",
            "papillary": "papillary",
            "serous": "serous",
            "mucinous": "mucinous",
        }
        for key, hist_type in histology_mapping.items():
            # Negative lookbehind for "non-" / "non " so "non-small cell"
            # doesn't get classified as small_cell histology (it's NSCLC,
            # a completely different disease from SCLC). Same applies to
            # "non-squamous", etc.
            if re.search(rf'(?<!non[- ])(?<!non){re.escape(key)}', histology):
                cancer["histology"] = hist_type
                break
        if not cancer.get("histology"):
            cancer["histology"] = histology
    
    # Stage
    if profile.get("cancer_stage"):
        stage = profile["cancer_stage"].upper()
        # Normalize stage to I, II, III, IV
        if "IV" in stage or "4" in stage:
            cancer["stage"] = "IV"
        elif "III" in stage or "3" in stage:
            cancer["stage"] = "III"
        elif "II" in stage or "2" in stage:
            cancer["stage"] = "II"
        elif "I" in stage or "1" in stage:
            cancer["stage"] = "I"
        else:
            cancer["stage"] = stage
        
        # Check for disease descriptors
        stage_lower = stage.lower()
        if "metastatic" in stage_lower or "m1" in stage_lower:
            cancer["disease_descriptor"] = "metastatic"
        elif "locally advanced" in stage_lower:
            cancer["disease_descriptor"] = "locally_advanced"
        elif "early" in stage_lower:
            cancer["disease_descriptor"] = "early"
    
    # Build patient context
    patient = {}
    
    if profile.get("age"):
        patient["age"] = int(profile["age"])
    
    if profile.get("gender"):
        patient["gender"] = profile["gender"].lower()
    
    if profile.get("performance_status") is not None:
        ps = profile["performance_status"]
        if isinstance(ps, str):
            # Extract number from string like "ECOG 1" or "1"
            match = re.search(r'\d', ps)
            if match:
                patient["performance_status"] = int(match.group())
        else:
            patient["performance_status"] = int(ps)
    
    # Build treatment context
    treatment = {}
    
    if profile.get("prior_treatments"):
        treatments = profile["prior_treatments"]
        if isinstance(treatments, list):
            treatment["prior"] = treatments
        elif isinstance(treatments, str):
            treatment["prior"] = [t.strip() for t in treatments.split(",")]
    
    # Construct the query structure
    query_structure = {}
    if cancer:
        query_structure["cancer"] = cancer
    if patient:
        query_structure["patient"] = patient
    if treatment:
        query_structure["treatment"] = treatment
    
    print(f"[build_query_structure_from_profile] Input: {profile}")
    print(f"[build_query_structure_from_profile] Output: {query_structure}")
    
    return query_structure


class SimplePatientMatchingService:
    """Enhanced patient matching with filtering, caching, and validation."""
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        collection_name: str,
        embed_model: str = "text-embedding-3-large"
    ):
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.collection = collection_name
        self.embed_model = embed_model
    
    def match_patient(
        self,
        patient_profile: Dict[str, Any],
        top_k: int = 15
    ) -> Dict[str, Any]:
        """
        Find clinical studies with similar patients.
        
        Uses semantic search with MANDATORY category filtering based on cancer type + anatomical site.
        """
        # Build improved query
        query, category_filter = self._build_enhanced_query(patient_profile)
        
        # STRICT: If we have anatomical site, category filter is MANDATORY
        anatomical_site = patient_profile.get("anatomical_site")
        strict_filter = bool(anatomical_site and category_filter)
        
        if strict_filter:
            print(f"[Patient Matching] STRICT MODE: Category filter '{category_filter}' is MANDATORY (site: {anatomical_site})")
        
        # Generate embedding (with caching)
        try:
            query_vector = self._get_embedding_cached(query)
        except Exception as e:
            print(f"[Patient Matching] Error generating embedding: {e}")
            return self._error_response(patient_profile, f"Embedding generation failed: {str(e)}")
        
        # Search with category filtering
        try:
            search_results = self._search_with_filters(
                query_vector=query_vector,
                category_filter=category_filter,
                limit=min(100, top_k * 5)
            )
        except Exception as e:
            print(f"[Patient Matching] Error searching Qdrant: {e}")
            # Only retry without filter if NOT in strict mode
            if not strict_filter:
                try:
                    search_results = self._search_with_filters(
                        query_vector=query_vector,
                        category_filter=None,
                        limit=min(100, top_k * 5)
                    )
                except Exception as retry_error:
                    return self._error_response(patient_profile, f"Search failed: {str(retry_error)}")
            else:
                return self._error_response(patient_profile, f"Search failed with required category filter: {str(e)}")
        
        # If no results with category filter in strict mode, return empty (don't fall back)
        if not search_results and strict_filter:
            print(f"[Patient Matching] No results with mandatory category filter '{category_filter}'")
            return {
                "matches": [],
                "total_matches": 0,
                "patient_summary": self._build_patient_summary(patient_profile),
                "warnings": [f"No studies found for {anatomical_site} cancer. The database may not have studies for this specific cancer site."]
            }
        
        # Only fall back to no filter if NOT in strict mode
        if not search_results and category_filter and not strict_filter:
            try:
                print(f"[Patient Matching] Falling back to no category filter")
                search_results = self._search_with_filters(
                    query_vector=query_vector,
                    category_filter=None,
                    limit=min(100, top_k * 5)
                )
            except Exception as retry_error:
                return self._error_response(patient_profile, f"Search failed: {str(retry_error)}")

        if not search_results:
            return {
                "matches": [],
                "total_matches": 0,
                "patient_summary": self._build_patient_summary(patient_profile),
                "warnings": ["No matching studies found. Try adjusting search criteria."]
            }
        
        print(f"[Patient Matching] Found {len(search_results)} search results")
        
        # Convert to chunks
        chunks = []
        for scored_point in search_results:
            chunks.append({
                "point_id": scored_point.id,
                "score_dense": float(scored_point.score),
                "payload": dict(scored_point.payload or {}),
            })
        
        # Rerank — distil the query into the short keyword form
        # ms-marco-MiniLM expects (the verbose `query` from
        # `_build_enhanced_query` is fine for the embedding stage but
        # out-of-distribution for the cross-encoder, which was trained
        # on short web-search queries). Also prepend the study title to
        # each chunk so the cross-encoder sees canonical literature
        # vocabulary regardless of which section retrieved. Both
        # techniques are now consistent with the main RAG pipeline
        # (commits fc39a71, e09b3ee, fd8e229).
        cross_encoder = get_cross_encoder()
        if cross_encoder and chunks:
            reranker_query = self._build_reranker_query_from_profile(patient_profile) or query[:200]
            texts = [_build_reranker_passage(c, char_budget=512) for c in chunks[:50]]
            pairs = [(reranker_query, text) for text in texts]
            scores = cross_encoder.predict(pairs)
            for i, chunk in enumerate(chunks[:50]):
                chunk["score_rerank"] = float(scores[i])
            chunks = sorted(chunks[:50], key=lambda x: x.get("score_rerank", 0), reverse=True) + chunks[50:]
        
        # Select best chunk per document (Fix #5)
        final_chunks = self._select_best_chunks_per_document(chunks, top_k)
        
        print(f"[Patient Matching] After deduplication: {len(final_chunks)} chunks")
        
        # Log raw scores before normalization
        for i, chunk in enumerate(final_chunks[:5]):
            raw_score = chunk.get("score_rerank") or chunk.get("score_dense", 0.0)
            title = chunk.get("payload", {}).get("doc_meta", {}).get("title", "Unknown")[:50]
            print(f"[Patient Matching] #{i+1} Raw score: {raw_score:.2f} - {title}...")
        
        # Normalize scores
        self._normalize_scores(final_chunks)
        
        # Validate matches with semantic validation (Fix #6)
        validated_chunks = self._validate_matches_semantically(
            final_chunks, 
            patient_profile, 
            query
        )
        
        # Generate matches with improved extraction
        matches = []
        for chunk in validated_chunks:
            match_data = self._build_match_data(chunk, patient_profile)
            if match_data:
                matches.append(match_data)
        
        return {
            "matches": matches,
            "total_matches": len(matches),
            "patient_summary": self._build_patient_summary(patient_profile),
        }
    
    def _build_reranker_query_from_profile(self, profile: Dict[str, Any]) -> str:
        """Build a short, keyword-dense query for the cross-encoder.

        Mirrors ``enhanced_rag_service.build_reranker_query``: ms-marco-
        MiniLM is trained on short natural-language search queries and
        saturates negatively on long verbose inputs. We distil the
        patient profile into a ~150-char keyword string capped at the
        most discriminating axes (cancer_type, site, histology, stage,
        and top biomarkers — polarity-formatted via
        ``_format_marker_polarity`` to keep "EGFR-positive" intact
        rather than corrupting it to "EGFR-positive negative").
        """
        parts: List[str] = []
        if profile.get("cancer_type"):
            parts.append(str(profile["cancer_type"]).strip())
        if profile.get("anatomical_site"):
            parts.append(str(profile["anatomical_site"]).strip())
        if profile.get("histology"):
            parts.append(str(profile["histology"]).strip())
        if profile.get("cancer_stage"):
            parts.append(f"stage {profile['cancer_stage']}".strip())
        for m in (profile.get("molecular_markers") or [])[:3]:
            if m:
                parts.append(_format_marker_polarity(m))
        if profile.get("performance_status"):
            parts.append(f"ECOG {profile['performance_status']}")
        query = " ".join(p for p in parts if p)
        return query[:200]

    def _build_enhanced_query(self, patient_profile: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Build improved query with anatomical site awareness for better matching."""
        query_parts = []
        category_filter = None
        
        # Get cancer type and anatomical site
        cancer_type = patient_profile.get("cancer_type", "")
        anatomical_site = patient_profile.get("anatomical_site", "")
        
        # Build combined cancer description for better semantic matching
        cancer_description = self._build_cancer_description(cancer_type, anatomical_site)
        
        # Determine category filter based on anatomical site (more reliable than cancer type alone)
        category_filter = self._infer_category_from_site(anatomical_site, cancer_type)
        
        if cancer_description:
            query_parts.append(f"{cancer_description} patients")
        
        # Demographics with better phrasing
        if patient_profile.get("age"):
            age = patient_profile["age"]
            if age < 40:
                query_parts.append("young adult patients under 40 years")
            elif age >= 65:
                query_parts.append("elderly patients 65 years or older")
            else:
                query_parts.append(f"adult patients aged {age} years")
        
        if patient_profile.get("gender"):
            gender = patient_profile["gender"].lower()
            query_parts.append(f"{gender} patients")
        
        # Cancer characteristics
        if patient_profile.get("cancer_stage"):
            stage = patient_profile["cancer_stage"]
            query_parts.append(f"stage {stage} cancer")
        
        if patient_profile.get("histology"):
            histology = patient_profile["histology"]
            query_parts.append(histology)
        
        if patient_profile.get("molecular_markers"):
            markers = patient_profile["molecular_markers"]
            # Polarity-aware marker phrasing.
            # The previous heuristic — `"+" in m` / `"-" in m` — false-
            # positived on connector hyphens. For "EGFR-positive" / "HPV-
            # positive" (string forms the LLM unstructured extractor
            # produces), `"-" in m` is True even though `m` already
            # encodes polarity in its word form. Output became
            # "EGFR-positive negative", corrupting the embedding query.
            # Same fix shape as the regex sweep across the codebase:
            # check for explicit polarity words first; only fall back to
            # +/- shorthand when the trailing character is a real
            # polarity marker (at end of token).
            marker_text = " ".join(_format_marker_polarity(m) for m in markers)
            query_parts.append(f"molecular markers {marker_text}")
        
        if patient_profile.get("performance_status"):
            ps = patient_profile["performance_status"]
            query_parts.append(f"ECOG performance status {ps}")
        
        # Build semantic query
        if query_parts:
            query = "clinical studies enrolling patients with: " + ", ".join(query_parts)
        else:
            query = "clinical studies"
        
        print(f"[Patient Matching] Built query: {query[:200]}...")
        print(f"[Patient Matching] Category filter: {category_filter}")
        
        return query, category_filter
    
    def _build_cancer_description(self, cancer_type: str, anatomical_site: str) -> str:
        """Build a specific cancer description combining type and site."""
        cancer_type = (cancer_type or "").strip().lower()
        anatomical_site = (anatomical_site or "").strip().lower()
        
        # Map common abbreviations to full names
        cancer_type_map = {
            "scc": "squamous cell carcinoma",
            "nsclc": "non-small cell lung cancer",
            "sclc": "small cell lung cancer",
            "hcc": "hepatocellular carcinoma",
            "rcc": "renal cell carcinoma",
            "crc": "colorectal cancer",
        }
        
        # Expand abbreviations
        if cancer_type in cancer_type_map:
            cancer_type = cancer_type_map[cancer_type]
        
        # Map anatomical sites to broader categories for query construction
        site_to_region = {
            # Head and neck sites
            "maxilla": "oral cavity head and neck",
            "mandible": "oral cavity head and neck",
            "oral cavity": "oral cavity head and neck",
            "tongue": "oral cavity head and neck",
            "gingiva": "oral cavity head and neck",
            "hard palate": "oral cavity head and neck",
            "soft palate": "oropharynx head and neck",
            "buccal mucosa": "oral cavity head and neck",
            "floor of mouth": "oral cavity head and neck",
            "oropharynx": "oropharynx head and neck",
            "nasopharynx": "nasopharynx head and neck",
            "hypopharynx": "hypopharynx head and neck",
            "larynx": "larynx head and neck",
            "tonsil": "oropharynx head and neck",
            "base of tongue": "oropharynx head and neck",
            # Other sites
            "skin": "cutaneous skin",
            "lung": "lung thoracic",
            "breast": "breast",
            "cervix": "cervical gynecologic",
            "anus": "anal gastrointestinal",
            "esophagus": "esophageal gastrointestinal",
        }
        
        # Build description
        if anatomical_site and cancer_type:
            region = site_to_region.get(anatomical_site, anatomical_site)
            # For SCC, be very specific about the site
            if "squamous" in cancer_type or cancer_type == "scc":
                return f"{region} squamous cell carcinoma"
            return f"{region} {cancer_type}"
        elif anatomical_site:
            region = site_to_region.get(anatomical_site, anatomical_site)
            return f"{region} cancer"
        elif cancer_type:
            return f"{cancer_type} cancer"
        return ""
    
    def _infer_category_from_site(self, anatomical_site: str, cancer_type: str) -> Optional[str]:
        """Infer Qdrant category from anatomical site (more reliable than cancer type alone)."""
        anatomical_site = (anatomical_site or "").strip().lower()
        cancer_type = (cancer_type or "").strip().lower()
        
        # Site-to-category mapping (anatomical site is most reliable)
        # Categories in Qdrant use format: {site}_processed_documents
        site_to_category = {
            # Head and neck
            "maxilla": "h&n_processed_documents",
            "mandible": "h&n_processed_documents",
            "oral cavity": "h&n_processed_documents",
            "tongue": "h&n_processed_documents",
            "gingiva": "h&n_processed_documents",
            "hard palate": "h&n_processed_documents",
            "soft palate": "h&n_processed_documents",
            "buccal mucosa": "h&n_processed_documents",
            "floor of mouth": "h&n_processed_documents",
            "oropharynx": "h&n_processed_documents",
            "nasopharynx": "h&n_processed_documents",
            "hypopharynx": "h&n_processed_documents",
            "larynx": "h&n_processed_documents",
            "tonsil": "h&n_processed_documents",
            "base of tongue": "h&n_processed_documents",
            "pharynx": "h&n_processed_documents",
            "neck": "h&n_processed_documents",
            "salivary gland": "h&n_processed_documents",
            "parotid": "h&n_processed_documents",
            # Skin
            "skin": "cutaneous_processed_documents",
            # Lung
            "lung": "lung_processed_documents",
            "bronchus": "lung_processed_documents",
            # Breast
            "breast": "breast_processed_documents",
            # GYN
            "cervix": "gyn_processed_documents",
            "uterus": "gyn_processed_documents",
            "ovary": "gyn_processed_documents",
            "endometrium": "gyn_processed_documents",
            "vulva": "gyn_processed_documents",
            "vagina": "gyn_processed_documents",
            # GI
            "anus": "gi_processed_documents",
            "rectum": "gi_processed_documents",
            "colon": "gi_processed_documents",
            "esophagus": "gi_processed_documents",
            "stomach": "gi_processed_documents",
            "liver": "gi_processed_documents",
            "pancreas": "gi_processed_documents",
            # GU
            "bladder": "gu_processed_documents",
            "kidney": "gu_processed_documents",
            # Prostate (has its own Qdrant category)
            "prostate": "prostate_processed_documents",
            # CNS
            "brain": "cns_processed_documents",
        }
        
        # First try anatomical site (most reliable)
        if anatomical_site in site_to_category:
            return site_to_category[anatomical_site]
        
        # Check if site contains any known keywords
        for site_key, category in site_to_category.items():
            if site_key in anatomical_site:
                return category
        
        # Fall back to cancer type inference (less reliable for generic types like SCC)
        # Only use cancer type if it's specific enough
        specific_cancer_types = {
            "breast": "breast_processed_documents",
            "lung": "lung_processed_documents",
            "nsclc": "lung_processed_documents",
            "sclc": "lung_processed_documents",
            "prostate": "prostate_processed_documents",
            "melanoma": "cutaneous_processed_documents",
            "glioma": "cns_processed_documents",
            "glioblastoma": "cns_processed_documents",
            "colorectal": "gi_processed_documents",
            "rectal": "gi_processed_documents",
            "cervical": "gyn_processed_documents",
            "ovarian": "gyn_processed_documents",
            "endometrial": "gyn_processed_documents",
            "bladder": "gu_processed_documents",
            "renal": "gu_processed_documents",
        }
        
        if cancer_type in specific_cancer_types:
            return specific_cancer_types[cancer_type]
        
        # For generic types like SCC, don't apply category filter (let semantic search handle it)
        # This is better than returning wrong category
        if cancer_type in ["scc", "squamous cell carcinoma", "adenocarcinoma", "carcinoma"]:
            print(f"[Patient Matching] Generic cancer type '{cancer_type}' without specific site - no category filter")
            return None
        
        # Try the original normalize_category_filter as last resort
        try:
            from .enhanced_rag_service import normalize_category_filter
            return normalize_category_filter(cancer_type)
        except Exception:
            return None
    
    def _get_embedding_cached(self, query: str) -> List[float]:
        """Get embedding with caching (Fix #2)."""
        # Create cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]
        
        # Generate embedding
        embedding_response = self.openai.embeddings.create(
            model=self.embed_model,
            input=[query]
        )
        embedding = embedding_response.data[0].embedding
        
        # Cache it (simple LRU: remove oldest if cache full)
        if len(_embedding_cache) >= _cache_max_size:
            # Remove first item (simple FIFO)
            _embedding_cache.pop(next(iter(_embedding_cache)))
        _embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _search_with_filters(
        self, 
        query_vector: List[float], 
        category_filter: Optional[str],
        limit: int
    ) -> List:
        """Search Qdrant with MANDATORY category filtering."""
        # Build filter if category provided
        query_filter = None
        if category_filter:
            # Try multiple category formats to handle different naming conventions
            # Qdrant may have: h&n_processed_documents, H&N, h&n, etc.
            base_category = category_filter.replace("_processed_documents", "")
            category_variants = [
                category_filter,  # h&n_processed_documents
                base_category,  # h&n
                base_category.upper(),  # H&N
                base_category.capitalize(),  # H&n
                f"{base_category.upper()}_processed_documents",  # H&N_processed_documents
            ]
            
            # Remove duplicates while preserving order
            category_variants = list(dict.fromkeys(category_variants))
            
            print(f"[Patient Matching] Searching with category variants: {category_variants}")
            
            # Use should - at least one variant must match
            query_filter = qm.Filter(
                should=[
                    qm.FieldCondition(key="category", match=qm.MatchValue(value=variant))
                    for variant in category_variants
                ]
            )
        
        # Search with filter
        search_results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        
        return search_results.points
    
    def _select_best_chunks_per_document(self, chunks: List[Dict], top_k: int) -> List[Dict]:
        """Select best chunk per document, not just first (Fix #5)."""
        # Group by document
        doc_chunks = {}
        for chunk in chunks:
            payload = chunk.get("payload", {})
            doc_id = payload.get("doc_id") or payload.get("doc_id_raw")
            
            if doc_id:
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk)
            else:
                # No doc_id - keep as separate entries (limit to 3)
                if len([c for c in chunks if not (c.get("payload", {}).get("doc_id") or c.get("payload", {}).get("doc_id_raw"))]) <= 3:
                    doc_chunks[f"_no_id_{len(doc_chunks)}"] = [chunk]
        
        # Select best chunk per document
        final_chunks = []
        for doc_id, doc_chunk_list in doc_chunks.items():
            # Sort by score and take best
            best_chunk = max(
                doc_chunk_list,
                key=lambda c: c.get("score_rerank") or c.get("score_dense", 0.0)
            )
            final_chunks.append(best_chunk)
            if len(final_chunks) >= top_k:
                break
        
        # Sort final chunks by score
        final_chunks.sort(
            key=lambda c: c.get("score_rerank") or c.get("score_dense", 0.0),
            reverse=True
        )
        
        return final_chunks[:top_k]
    
    def _normalize_scores(self, chunks: List[Dict]):
        """
        Convert cross-encoder scores to meaningful match percentages.
        
        Cross-encoder scores typically range from -10 to +10.
        We use a more conservative mapping to avoid inflated scores:
        - Scores >= 10: 90-95% (exceptional match)
        - Scores 7-10: 75-90% (strong match)
        - Scores 4-7: 55-75% (good match)
        - Scores 1-4: 35-55% (moderate match)
        - Scores -2 to 1: 15-35% (weak match)
        - Scores < -2: 0-15% (poor match)
        
        Also applies relative ranking adjustment to differentiate results.
        """
        if not chunks:
            return
        
        # First pass: calculate raw normalized scores
        raw_scores = []
        for chunk in chunks:
            raw_score = chunk.get("score_rerank") or chunk.get("score_dense", 0.0)
            raw_scores.append(raw_score)
            
            # More conservative absolute score mapping
            if raw_score >= 10:
                normalized = 0.90 + min((raw_score - 10) * 0.01, 0.05)  # 90-95%
            elif raw_score >= 7:
                normalized = 0.75 + (raw_score - 7) * 0.05  # 75-90%
            elif raw_score >= 4:
                normalized = 0.55 + (raw_score - 4) * 0.067  # 55-75%
            elif raw_score >= 1:
                normalized = 0.35 + (raw_score - 1) * 0.067  # 35-55%
            elif raw_score >= -2:
                normalized = 0.15 + (raw_score + 2) * 0.067  # 15-35%
            else:
                normalized = max(0.0, 0.15 + (raw_score + 2) * 0.05)  # 0-15%
            
            chunk["score_normalized"] = max(0.0, min(0.95, normalized))
            chunk["raw_score"] = raw_score
        
        # Second pass: apply relative ranking adjustment
        # This ensures differentiation between results even when raw scores are similar
        if len(chunks) > 1:
            max_score = max(raw_scores)
            min_score = min(raw_scores)
            score_range = max_score - min_score
            
            # Only apply relative adjustment if scores are clustered (range < 3)
            if score_range < 3 and score_range > 0:
                for i, chunk in enumerate(chunks):
                    raw_score = chunk["raw_score"]
                    base_normalized = chunk["score_normalized"]
                    
                    # Calculate relative position (0 = worst, 1 = best)
                    relative_pos = (raw_score - min_score) / score_range
                    
                    # Apply a small penalty based on rank position
                    # Top result keeps score, others get progressively reduced
                    rank_penalty = (1 - relative_pos) * 0.15  # Up to 15% penalty
                    
                    chunk["score_normalized"] = max(0.10, base_normalized - rank_penalty)
    
    def _validate_matches_semantically(
        self,
        chunks: List[Dict],
        patient_profile: Dict[str, Any],
        query: str
    ) -> List[Dict]:
        """
        Validate matches using hard eligibility filtering on rule-based criteria:
          1. Cancer type and histology
          2. Tumor stage or disease status
          3. Prior therapies and treatment lines
          4. Molecular biomarkers (only when the study explicitly reports them)

        A study is REMOVED only if it explicitly contradicts the patient profile
        on one of these criteria.  Studies that simply don't report a criterion
        are kept (NOT_AVAILABLE ≠ MISMATCH).
        """
        if not chunks:
            return chunks

        # Build patient summary for validation
        patient_summary = self._build_patient_summary(patient_profile)

        # Determine which hard-filter criteria are active based on profile
        cancer_type = patient_profile.get("cancer_type", "")
        anatomical_site = patient_profile.get("anatomical_site", "")
        cancer_description = self._build_cancer_description(cancer_type, anatomical_site)

        active_criteria = []
        criteria_desc_parts = []

        if cancer_type or anatomical_site:
            active_criteria.append("cancer_type")
            criteria_desc_parts.append(f"  - cancer_type: {cancer_description or cancer_type or anatomical_site}")

        histology = patient_profile.get("histology", "")
        if histology:
            active_criteria.append("histology")
            criteria_desc_parts.append(f"  - histology: {histology}")

        stage = patient_profile.get("cancer_stage", "")
        if stage:
            active_criteria.append("stage")
            criteria_desc_parts.append(f"  - stage: {stage}")

        prior_treatments = patient_profile.get("prior_treatments", "")
        if prior_treatments:
            active_criteria.append("prior_therapies")
            treatments_str = ", ".join(prior_treatments) if isinstance(prior_treatments, list) else str(prior_treatments)
            criteria_desc_parts.append(f"  - prior_therapies: {treatments_str}")

        markers = list(patient_profile.get("molecular_markers", []) or [])
        # Include recurrence score as part of biomarkers for eligibility evaluation
        recurrence_score = patient_profile.get("recurrence_score")
        if recurrence_score is not None:
            markers.append(f"21-gene recurrence score {recurrence_score}")
        if markers:
            active_criteria.append("biomarkers")
            markers_str = ", ".join(markers) if isinstance(markers, list) else str(markers)
            criteria_desc_parts.append(f"  - biomarkers: {markers_str}")

        # If no active criteria at all, skip validation
        if not active_criteria:
            print(f"[Patient Matching] Skipping hard eligibility filter (no active criteria)")
            for chunk in chunks:
                chunk["validation_status"] = "unvalidated"
            return chunks

        criteria_description = "\n".join(criteria_desc_parts)

        # Prepare validation for top 10 chunks
        validation_chunks = chunks[:10]
        validation_texts = []
        for chunk in validation_chunks:
            text = chunk.get("payload", {}).get("text", "")[:500]
            title = chunk.get("payload", {}).get("doc_meta", {}).get("title", "Unknown")
            validation_texts.append(f"Title: {title}\nText: {text}")

        try:
            import json as _json

            all_criteria = ["cancer_type", "histology", "stage", "prior_therapies", "biomarkers"]

            # Secondary axes required by the scoring layer in
            # apply_patient_eligibility_filter_and_boost.  The LLM
            # prompt only evaluates core axes; secondary axes default
            # to NOT_AVAILABLE so the scoring layer has a complete
            # verdict dict to iterate over.
            _SECONDARY_AXES = [
                "performance_status",
                "age_range",
                "gender",
                "modality",
                "metastatic_sites",
                "comorbidity_compatibility",
                "study_phase",
                "landmark_trial_status",
                "recency",
            ]

            validation_prompt = f"""You are a clinical trial eligibility expert. For each study below, evaluate whether the patient matches the study's enrolled population on EACH criterion.

Patient Profile: {patient_summary}
Cancer Type/Site: {cancer_description if cancer_description else 'Not specified'}
Query: {query}
Active criteria:
{criteria_description}

Studies to evaluate:
{chr(10).join([f"{i+1}. {text[:400]}..." for i, text in enumerate(validation_texts)])}

For EACH study, evaluate EACH criterion independently and respond with one of:
- "MATCH" — the study's enrolled population clearly includes this patient on this criterion
- "MISMATCH" — the study explicitly enrolled a DIFFERENT population on this criterion (e.g., study is for lung cancer but patient has breast cancer; study is for HER2+ but patient is HER2-; study enrolled only stage IV but patient is stage II; study required treatment-naive but patient was previously treated)
- "NOT_AVAILABLE" — the study does NOT explicitly report or restrict on this criterion (the information is simply absent from the study text)

CRITICAL RULES:
1. "NOT_AVAILABLE" means the study text does not mention or restrict on that criterion AT ALL. This is DIFFERENT from MISMATCH.
2. A study that does not mention biomarkers → biomarkers = NOT_AVAILABLE (NOT a MISMATCH).
3. A study that does not mention histology → histology = NOT_AVAILABLE.
4. A study that does not restrict by stage → stage = NOT_AVAILABLE.
5. Be STRICT about MISMATCH: only use it when the study EXPLICITLY contradicts the patient.
6. Be STRICT about biomarker polarity: ER+ ≠ ER-, HER2+ ≠ HER2-.
7. Range matching: a study for "stage II-III" MATCHES a stage III patient.
8. SAME ORGAN SYSTEM ≠ SAME CANCER: cancers in the same organ system are DIFFERENT cancers. Prostate cancer ≠ bladder cancer ≠ renal cancer (all GU). Colon cancer ≠ rectal cancer ≠ gastric cancer (all GI). Cervical cancer ≠ ovarian cancer ≠ endometrial cancer (all GYN). If the study explicitly enrolls a DIFFERENT specific cancer from the same organ system, mark cancer_type as MISMATCH.

Respond in EXACTLY this JSON format (no other text):
{{
  "1": {{"cancer_type": "MATCH/MISMATCH/NOT_AVAILABLE", "histology": "...", "stage": "...", "prior_therapies": "...", "biomarkers": "...", "reason": "brief summary"}},
  "2": {{...}},
  ...
}}"""

            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a clinical trial eligibility expert. Respond ONLY with the requested JSON. Be strict about confirmed mismatches but never confuse missing information with a mismatch."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=1200,
            )

            validation_result = response.choices[0].message.content
            print(f"[Patient Matching] Hard eligibility filter response: {validation_result[:300]}...")

            # Parse JSON response (reuse parser from eligibility service)
            from src.api.services.patient_eligibility_boost_service import _parse_eligibility_json
            parsed = _parse_eligibility_json(validation_result, len(validation_chunks))

            validated_chunks = []
            removed_count = 0
            # Track WHICH axis caused each removal so we can honour Path A
            # (refuse to restore wrong-cancer evidence) below — mirrors the
            # logic in patient_eligibility_boost_service.apply_patient_eligibility_filter_and_boost.
            cancer_type_mismatch_removals = 0

            for i, chunk in enumerate(validation_chunks):
                study_key = str(i + 1)
                if study_key in parsed:
                    verdicts = parsed[study_key]

                    # Check for hard mismatch on any active criterion
                    has_mismatch = False
                    mismatched_criteria = []
                    match_count = 0
                    for criterion in all_criteria:
                        verdict = verdicts.get(criterion, "NOT_AVAILABLE").upper()
                        if verdict not in ("MATCH", "MISMATCH", "NOT_AVAILABLE"):
                            verdict = "NOT_AVAILABLE"
                        # Biomarkers-when-declared rule: only count
                        # biomarkers as MISMATCH when the study text
                        # explicitly declares a biomarker requirement
                        # (i.e., the LLM returned MATCH or MISMATCH).
                        # If the LLM returned NOT_AVAILABLE for
                        # biomarkers, the study is silent — force
                        # NOT_AVAILABLE regardless.
                        if criterion == "biomarkers" and verdict == "MISMATCH":
                            # The LLM said MISMATCH, but we need to
                            # check if the study actually declared a
                            # biomarker requirement.  Since the LLM
                            # returned MISMATCH, it believes the study
                            # text mentions biomarkers.  We trust the
                            # LLM's MISMATCH here (it means the study
                            # declared a contradicting requirement).
                            pass  # keep as MISMATCH — study declared a requirement
                        if verdict == "MISMATCH" and criterion in active_criteria:
                            has_mismatch = True
                            mismatched_criteria.append(criterion)
                        if verdict == "MATCH":
                            match_count += 1

                    if has_mismatch:
                        title = chunk.get("payload", {}).get("doc_meta", {}).get("title", "Unknown")[:50]
                        print(f"[Patient Matching] HARD FILTER removed '{title}' — mismatch on: {mismatched_criteria}")
                        chunk["validation_status"] = "no_match"
                        chunk["criteria_verdicts"] = {
                            **{c: verdicts.get(c, "NOT_AVAILABLE").upper() for c in all_criteria},
                            **{a: "NOT_AVAILABLE" for a in _SECONDARY_AXES},
                        }
                        removed_count += 1
                        if "cancer_type" in mismatched_criteria:
                            cancer_type_mismatch_removals += 1
                        continue

                    # Determine status
                    if match_count > 0:
                        chunk["validation_status"] = "match"
                    else:
                        chunk["validation_status"] = "possible"
                    chunk["criteria_verdicts"] = {
                        **{c: verdicts.get(c, "NOT_AVAILABLE").upper() for c in all_criteria},
                        **{a: "NOT_AVAILABLE" for a in _SECONDARY_AXES},
                    }
                    validated_chunks.append(chunk)
                else:
                    chunk["validation_status"] = "unknown"
                    validated_chunks.append(chunk)

            # Add remaining chunks (beyond top 10) without validation
            for chunk in chunks[10:]:
                chunk["validation_status"] = "unvalidated"
            validated_chunks.extend(chunks[10:])

            # Safety: don't remove ALL evidence — but be surgical about WHAT gets
            # restored. A cancer_type MISMATCH means the study is for the WRONG
            # cancer entirely (NSCLC patient → SCLC trial), and restoring it
            # would silently fabricate confident answers over wrong-cancer
            # literature. Other mismatches (stage, disease_status, prior therapy)
            # mean "right cancer, wrong context" — those are reasonable to
            # restore at reduced confidence when the bundle would otherwise be
            # empty.
            #
            # This mirrors Path A in
            # patient_eligibility_boost_service.apply_patient_eligibility_filter_and_boost,
            # which the multi-specialty path already honours.
            if len(validated_chunks) < 2 and len(chunks) >= 2:
                # Bucket the removed chunks: those rejected on a hard-drop
                # axis stay gone forever; those rejected only on non-hard-drop
                # axes are eligible to come back at reduced score. Mirrors the
                # generalized Path A in patient_eligibility_boost_service.
                # Hard-drop axes (per oncologist hard-filter requirements):
                #   cancer_type, histology, stage, prior_therapies,
                #   disease_status, surgical_candidacy (all MISMATCH-drop),
                #   plus study_exclusions_violated (MATCH-drop, inverted).
                from src.api.services.patient_eligibility_boost_service import (
                    HARD_DROP_AXES, HARD_DROP_ON_MATCH_AXES,
                )

                def _is_hard_drop(c):
                    v = c.get("criteria_verdicts", {}) or {}
                    for axis in HARD_DROP_AXES:
                        if v.get(axis) == "MISMATCH":
                            return True
                    for axis in HARD_DROP_ON_MATCH_AXES:
                        if v.get(axis) == "MATCH":
                            return True
                    return False

                hard_drop_rejected = [
                    c for c in chunks
                    if c.get("validation_status") == "no_match" and _is_hard_drop(c)
                ]
                other_rejected = [
                    c for c in chunks
                    if c.get("validation_status") == "no_match" and not _is_hard_drop(c)
                ]
                if hard_drop_rejected:
                    print(
                        f"[Patient Matching] Refusing to restore {len(hard_drop_rejected)} "
                        f"hard-drop-axis-mismatched chunks (Path A — never restore studies the "
                        f"patient is structurally ineligible for, regardless of bundle size)."
                    )
                # Restore the non-hard-drop rejections so the bundle isn't empty.
                for chunk in other_rejected:
                    chunk["validation_status"] = "possible"
                    validated_chunks.append(chunk)
                if hard_drop_rejected and not other_rejected:
                    # All removals were hard-drop — return whatever survived
                    return validated_chunks
                if other_rejected:
                    print(f"[Patient Matching] Restored {len(other_rejected)} non-hard-drop-mismatched chunks at reduced confidence")
                return validated_chunks

            print(f"[Patient Matching] Hard eligibility filter: kept {len(validated_chunks)}, removed {removed_count} from top 10")

            return validated_chunks

        except Exception as e:
            print(f"[Patient Matching] Hard eligibility filter failed: {e}, using all chunks")
            for chunk in chunks:
                chunk["validation_status"] = "unvalidated"
            return chunks
    
    def _build_match_data(self, chunk: Dict, patient_profile: Dict[str, Any]) -> Optional[Dict]:
        """Build match data with improved extraction."""
        payload = chunk.get("payload", {})
        doc_meta = payload.get("doc_meta", {})
        if not doc_meta:
            doc_meta = {
                "title": payload.get("title"),
                "author_et_al": payload.get("author_et_al"),
                "citation": payload.get("citation"),
                "doi": payload.get("doi"),
                "pmid": payload.get("pmid"),
                "year": payload.get("year"),
            }
        
        # Generate match information with improved matching
        match_info = self._generate_similarity_reasons_enhanced(chunk, patient_profile)
        
        normalized_score = chunk.get("score_normalized", 0.0)
        validation_status = chunk.get("validation_status", "unknown")
        
        # Adjust score based on validation status
        # "match" = full score, "possible" = 70% of score, "unknown/unvalidated" = 50% of score
        if validation_status == "match":
            final_score = normalized_score
        elif validation_status == "possible":
            final_score = normalized_score * 0.7
        else:
            final_score = normalized_score * 0.5
        
        text_content = payload.get("text", "") or payload.get("content", "")
        
        # Extract information
        treatment_info = self._extract_treatment_info(text_content)
        key_info = self._extract_key_info(text_content, patient_profile)
        
        # Get doc_id from payload or doc_meta
        doc_id = payload.get("doc_id") or doc_meta.get("doc_id")
        pmid = doc_meta.get("pmid") or payload.get("pmid")
        
        return {
            "title": doc_meta.get("title") or "Unknown",
            "author": doc_meta.get("author_et_al"),
            "year": doc_meta.get("year"),
            "match_score": round(final_score, 3),
            "validation_status": validation_status,
            "treatment": treatment_info,
            "demographics": match_info.get("demographics", []),
            "cancer_characteristics": match_info.get("cancer_characteristics", []),
            "key_matches": match_info.get("key_matches", []),
            "key_info": key_info,
            "relevant_text": text_content,
            "doi": doc_meta.get("doi"),
            "pmid": pmid,
            "citation": doc_meta.get("citation"),
            "doc_id": doc_id,
            "criteria_verdicts": chunk.get("criteria_verdicts"),
            # Surface patient_match_scorer output if it was stamped
            # on this chunk by match_patient_comprehensive. Lets the
            # Find Trials UI render the same Strong/Moderate/Weak/
            # Limited badge the chat UI uses (axis-overlap based,
            # NA-aware), independent of the cross-encoder match_score.
            "patient_match_score": chunk.get("patient_match_score"),
            "patient_match_breakdown": chunk.get("patient_match_breakdown"),
        }
    
    def _generate_similarity_reasons_enhanced(
        self,
        chunk: Dict[str, Any],
        patient_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate match information with improved regex/pattern matching including anatomical site."""
        payload = chunk.get("payload", {})
        text = (payload.get("text", "") or payload.get("content", "")).lower()
        
        match_info = {
            "demographics": [],
            "cancer_characteristics": [],
            "key_matches": []
        }
        
        # Improved age matching with regex (Fix #3)
        if patient_profile.get("age"):
            age = patient_profile["age"]
            if age < 40:
                # Match: "young", "younger", "aged 18-40", "<40", etc.
                if re.search(r"\b(young|younger|aged\s+\d+\s*-\s*40|aged\s+<40|<40|under\s+40)\b", text):
                    match_info["demographics"].append("Young patients")
            elif age >= 65:
                # Match: "elderly", "older", "aged 65+", "≥65", etc.
                if re.search(r"\b(elderly|older|aged\s+65|≥65|>=65|65\+|65\s*or\s*older)\b", text):
                    match_info["demographics"].append("Elderly patients")
            else:
                # Match: "adult", "aged X", etc.
                if re.search(r"\b(adult|aged\s+\d+)\b", text):
                    match_info["demographics"].append("Adult patients")
        
        # Improved gender matching (Fix #3)
        if patient_profile.get("gender"):
            gender = patient_profile["gender"].lower()
            if gender == "female":
                if re.search(r"\b(women|female|postmenopausal|premenopausal)\b", text):
                    match_info["demographics"].append("Female patients")
            elif gender == "male":
                if re.search(r"\b(men|male)\b", text):
                    match_info["demographics"].append("Male patients")
        
        # Anatomical site matching (NEW - critical for SCC cases)
        if patient_profile.get("anatomical_site"):
            site = patient_profile["anatomical_site"].lower()
            site_keywords = {
                "maxilla": ["maxilla", "maxillary", "upper jaw"],
                "mandible": ["mandible", "mandibular", "lower jaw"],
                "oral cavity": ["oral cavity", "oral", "mouth"],
                "tongue": ["tongue", "lingual"],
                "gingiva": ["gingiva", "gingival", "gum"],
                "hard palate": ["hard palate", "palate"],
                "buccal mucosa": ["buccal", "cheek"],
                "oropharynx": ["oropharynx", "oropharyngeal"],
                "nasopharynx": ["nasopharynx", "nasopharyngeal"],
                "larynx": ["larynx", "laryngeal"],
                "skin": ["skin", "cutaneous"],
            }
            
            # Check for site-specific keywords
            keywords = site_keywords.get(site, [site])
            for kw in keywords:
                if kw in text:
                    match_info["cancer_characteristics"].append(f"Site: {patient_profile['anatomical_site']}")
                    break
        
        # Cancer characteristics with better matching
        if patient_profile.get("cancer_type"):
            cancer_lower = patient_profile["cancer_type"].lower()
            # Handle SCC specifically
            if cancer_lower in ["scc", "squamous cell carcinoma"]:
                if re.search(r"\b(squamous|scc|squamous\s+cell)\b", text):
                    match_info["cancer_characteristics"].append("Squamous cell carcinoma")
            else:
                # Match cancer type with word boundaries
                if re.search(rf"\b{cancer_lower}\s+cancer\b|\b{cancer_lower}\b", text):
                    match_info["cancer_characteristics"].append(patient_profile["cancer_type"])
        
        # Improved stage matching (Fix #3, #4)
        if patient_profile.get("cancer_stage"):
            stage = patient_profile["cancer_stage"].lower()
            # Match: "stage II", "stage 2", "stage IIA", etc.
            stage_pattern = rf"\bstage\s+{stage}[a-c]?\b|\bstage\s+{stage.upper()}[A-C]?\b"
            if re.search(stage_pattern, text, re.IGNORECASE):
                match_info["cancer_characteristics"].append(f"Stage {patient_profile['cancer_stage']}")
        
        # Histology matching
        if patient_profile.get("histology"):
            histology = patient_profile["histology"].lower()
            if histology in text:
                match_info["cancer_characteristics"].append(patient_profile["histology"])
        
        # Improved molecular marker matching (Fix #4)
        # CRITICAL: Correctly distinguish positive from negative biomarker status.
        # "ER+" must NOT match "ER-" and vice versa.
        if patient_profile.get("molecular_markers"):
            for marker in patient_profile["molecular_markers"]:
                marker_clean = marker.replace("+", "").replace("-", "").lower()
                marker_positive = "+" in marker or "positive" in marker.lower()
                marker_negative = "-" in marker or "negative" in marker.lower()

                # Match with CORRECT polarity only
                if marker_positive:
                    # Match ONLY positive: "HER2+", "HER2 positive", "HER2 amplified"
                    # Do NOT match "HER2-" or "HER2 negative"
                    pattern = (
                        rf"\b{re.escape(marker_clean)}\s*\+|"
                        rf"\b{re.escape(marker_clean)}\s+(?:positive|amplified|overexpress\w*|mutant|mutation|mutated|fusion|rearrange\w*)"
                    )
                    if re.search(pattern, text, re.IGNORECASE):
                        match_info["key_matches"].append(marker)
                elif marker_negative:
                    # Match ONLY negative: "HER2-", "HER2 negative", "HER2 non-amplified"
                    # Do NOT match "HER2+" or "HER2 positive"
                    pattern = (
                        rf"\b{re.escape(marker_clean)}\s*\-|"
                        rf"\b{re.escape(marker_clean)}\s+(?:negative|non.?amplified|wild.?type|wt|absent)"
                    )
                    if re.search(pattern, text, re.IGNORECASE):
                        match_info["key_matches"].append(marker)
                else:
                    # Neutral marker (no polarity, e.g. "PD-L1 expression")
                    if marker_clean in text:
                        match_info["key_matches"].append(marker)
        
        # Performance status matching
        if patient_profile.get("performance_status"):
            ps = str(patient_profile["performance_status"])
            # Match: "ECOG 0", "PS 0", "performance status 0", etc.
            if re.search(rf"\b(ecog|ps|performance\s+status)\s+{ps}\b", text, re.IGNORECASE):
                match_info["demographics"].append(f"ECOG {ps}")
        
        return match_info
    
    def _extract_key_info(self, text: str, patient_profile: Dict[str, Any]) -> str:
        """Extract key information or findings from the study text."""
        if not text:
            return ""
        
        sentences = text.split('.')
        
        key_phrases = [
            "result", "conclusion", "finding", "demonstrated", "showed", 
            "improved", "efficacy", "survival", "response", "benefit"
        ]
        
        for sentence in sentences[:5]:
            sentence_lower = sentence.lower()
            if any(phrase in sentence_lower for phrase in key_phrases):
                cleaned = sentence.strip()
                if len(cleaned) > 20 and len(cleaned) < 200:
                    return cleaned + "."
        
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) > 30 and len(cleaned) < 200:
                return cleaned + "."
        
        return text[:150].strip() + "..."
    
    def _extract_treatment_info(self, text: str) -> str:
        """Extract treatment information from text."""
        if not text:
            return "Treatment information not available"
        
        text_lower = text.lower()
        treatments = []
        
        # Expanded treatment list
        treatment_keywords = {
            "trastuzumab": "Trastuzumab",
            "herceptin": "Trastuzumab",
            "pembrolizumab": "Pembrolizumab",
            "keytruda": "Pembrolizumab",
            "atezolizumab": "Atezolizumab",
            "tecentriq": "Atezolizumab",
            "nivolumab": "Nivolumab",
            "opdivo": "Nivolumab",
            "durvalumab": "Durvalumab",
            "imfinzi": "Durvalumab",
            "chemotherapy": "Chemotherapy",
            "chemo": "Chemotherapy",
            "radiotherapy": "Radiotherapy",
            "radiation": "Radiotherapy",
            "rt": "Radiotherapy",
            "surgery": "Surgery",
            "surgical": "Surgery",
            "endocrine therapy": "Endocrine Therapy",
            "hormone therapy": "Endocrine Therapy",
            "targeted therapy": "Targeted Therapy",
            "immunotherapy": "Immunotherapy"
        }
        
        for keyword, treatment_name in treatment_keywords.items():
            if keyword in text_lower and treatment_name not in treatments:
                treatments.append(treatment_name)
        
        if treatments:
            return ", ".join(treatments[:3])
        else:
            sentences = text.split('.')
            if sentences:
                first_sent = sentences[0][:100]
                return first_sent + "..." if len(first_sent) == 100 else first_sent
            return "Treatment information not available"
    
    def _build_patient_summary(self, patient_profile: Dict[str, Any]) -> str:
        """Build patient summary string including anatomical site."""
        parts = []
        if patient_profile.get("age"):
            parts.append(f"{patient_profile['age']}-year-old")
        if patient_profile.get("gender"):
            parts.append(patient_profile["gender"])
        
        # Build cancer description with site
        cancer_type = patient_profile.get("cancer_type", "")
        anatomical_site = patient_profile.get("anatomical_site", "")
        
        if anatomical_site and cancer_type:
            parts.append(f"with {cancer_type} of {anatomical_site}")
        elif cancer_type:
            parts.append(f"with {cancer_type} cancer")
        elif anatomical_site:
            parts.append(f"with cancer of {anatomical_site}")
        
        if patient_profile.get("cancer_stage"):
            parts.append(f"(stage {patient_profile['cancer_stage']})")
        
        return " ".join(parts) if parts else "Patient"
    
    def _error_response(self, patient_profile: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Return error response with partial results if available (Fix #7)."""
        return {
            "matches": [],
            "total_matches": 0,
            "patient_summary": self._build_patient_summary(patient_profile),
            "error": error_msg,
            "warnings": [error_msg]
        }

    # ------------------------------------------------------------------
    # Comprehensive-retrieval entry point (used by /rag/patient/match
    # and /rag/patient/match/unstructured)
    # ------------------------------------------------------------------
    async def match_patient_comprehensive(
        self,
        patient_profile: Dict[str, Any],
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Find clinical studies for a patient using the SAME multi-specialty
        retrieval pipeline used by the tumor board (and now by Trial Match
        and Treatment Comparison): six specialty agents each build their
        own specialty-aware sub-queries from the patient case bundle, run
        them in parallel via the tumor board's lightweight Qdrant search,
        and the results are merged across specialties with a small
        consensus boost. The LLM expert-assessment step is intentionally
        skipped — this method's job is to return matched studies, not a
        per-specialty narrative.

        Returns the same dict shape as :meth:`match_patient` so that the
        existing route-level structured-matcher merging and rationale
        generation continue to work unchanged.
        """
        # Build query and category exactly as the legacy path does so the
        # downstream eligibility validator and pattern-based reasons stay
        # consistent.
        query, category_filter = self._build_enhanced_query(patient_profile)

        try:
            from src.api.services.multi_specialty_retrieval import (
                retrieve_evidence_multispecialty,
                studies_to_validator_chunks,
            )
        except Exception as e:
            print(
                f"[Patient Matching] multi_specialty_retrieval import failed: {e}"
            )
            # Hard fallback to the legacy Qdrant-only path
            import asyncio as _asyncio
            return await _asyncio.to_thread(self.match_patient, patient_profile, top_k)

        anatomical_site = patient_profile.get("anatomical_site")
        strict_filter = bool(anatomical_site and category_filter)
        if strict_filter:
            print(
                f"[Patient Matching] STRICT MODE: Category filter "
                f"'{category_filter}' is MANDATORY (site: {anatomical_site})"
            )

        # Run the multi-specialty (tumor-board style) retrieval. Pull a
        # wide pool of studies so the downstream eligibility validator
        # and dedup logic have something to filter; final cap is applied
        # at the end.
        try:
            ms_result = await retrieve_evidence_multispecialty(
                case_text=query,
                query_type="treatment_recommendation",
                category=category_filter,
                max_studies=max(top_k * 2, top_k + 5),
            )
        except Exception as e:
            print(f"[Patient Matching] Multi-specialty retrieval failed: {e}")
            if not strict_filter:
                try:
                    ms_result = await retrieve_evidence_multispecialty(
                        case_text=query,
                        query_type="treatment_recommendation",
                        category=None,
                        max_studies=max(top_k * 2, top_k + 5),
                    )
                except Exception as retry_error:
                    return self._error_response(
                        patient_profile,
                        f"Search failed: {str(retry_error)}",
                    )
            else:
                return self._error_response(
                    patient_profile,
                    f"Search failed with required category filter: {str(e)}",
                )

        merged_studies = ms_result.merged_studies

        # Strict mode: do NOT silently fall back to no filter.
        if not merged_studies and strict_filter:
            print(
                f"[Patient Matching] No multi-specialty results with mandatory "
                f"category filter '{category_filter}'"
            )
            return {
                "matches": [],
                "total_matches": 0,
                "patient_summary": self._build_patient_summary(patient_profile),
                "warnings": [
                    f"No studies found for {anatomical_site} cancer. The "
                    f"database may not have studies for this specific cancer site."
                ],
            }

        # Lenient mode: if filtered run came back empty, retry without filter
        if not merged_studies and category_filter and not strict_filter:
            try:
                print(
                    "[Patient Matching] Multi-specialty retrieval empty — "
                    "retrying without category filter"
                )
                ms_result = await retrieve_evidence_multispecialty(
                    case_text=query,
                    query_type="treatment_recommendation",
                    category=None,
                    max_studies=max(top_k * 2, top_k + 5),
                )
                merged_studies = ms_result.merged_studies
            except Exception as retry_error:
                return self._error_response(
                    patient_profile,
                    f"Search failed: {str(retry_error)}",
                )

        if not merged_studies:
            return {
                "matches": [],
                "total_matches": 0,
                "patient_summary": self._build_patient_summary(patient_profile),
                "warnings": [
                    "No matching studies found. Try adjusting search criteria."
                ],
            }

        print(
            f"[Patient Matching] Multi-specialty retrieval returned "
            f"{len(merged_studies)} studies "
            f"(specialties: {sorted(ms_result.per_specialty.keys())}, "
            f"skipped: {sorted(ms_result.skipped.keys())})"
        )

        # Convert merged LightweightStudy objects to the per-document chunk
        # shape that the legacy validator / scorer / match-builder expects.
        chunks = studies_to_validator_chunks(merged_studies)

        # Sort by rerank score so dedup picks the strongest evidence
        chunks.sort(
            key=lambda c: c.get("score_rerank") or c.get("score_dense") or 0.0,
            reverse=True,
        )

        # Pick the best chunk per document (defensive no-op since the
        # multi-specialty pipeline already returns one study per doc_id).
        final_chunks = self._select_best_chunks_per_document(chunks, top_k)
        print(
            f"[Patient Matching] After multi-specialty dedup: "
            f"{len(final_chunks)} chunks"
        )

        for i, chunk in enumerate(final_chunks[:5]):
            raw_score = chunk.get("score_rerank") or chunk.get("score_dense", 0.0)
            title = (
                chunk.get("payload", {}).get("doc_meta", {}).get("title", "Unknown")
            )[:50]
            specs = chunk.get("_specialties") or []
            print(
                f"[Patient Matching] #{i+1} multi-specialty raw score: "
                f"{raw_score:.2f} ({'+'.join(specs) or 'tumor_board'}) - {title}..."
            )

        # Reuse the existing normaliser + hard-eligibility filter so that
        # response shape and downstream behaviour are identical to the
        # legacy Qdrant-only path.
        self._normalize_scores(final_chunks)
        validated_chunks = self._validate_matches_semantically(
            final_chunks, patient_profile, query
        )

        # Patient_match_score — stamp the axis-overlap scorer's output on
        # each chunk so the Find Trials UI can surface the same per-study
        # "Strong/Moderate/Weak/Limited" match badge the chat UI uses.
        # The scorer is NA-aware (axes the study didn't measure don't
        # penalise the score) and caps wrong-cancer matches at 35%.
        # Independent of the cross-encoder normalized match_score — the
        # two scores answer different questions: chunk relevance vs.
        # study-population fit.
        try:
            from src.api.services.patient_match_scorer import score_patient_match
            clinical_profile = _build_clinical_profile_from_patient_dict(patient_profile)
            if clinical_profile is not None and clinical_profile.has_any_filter():
                scored_doc_ids = 0
                for chunk in validated_chunks:
                    doc_level = (chunk.get("payload") or {}).get("metadata") or {}
                    if not doc_level:
                        continue
                    try:
                        pm = score_patient_match(clinical_profile, doc_level)
                        chunk["patient_match_score"] = pm.get("score")
                        chunk["patient_match_breakdown"] = pm
                        scored_doc_ids += 1
                    except Exception as _e:
                        print(f"[Patient Matching] patient_match score failed for chunk: {_e}")
                if scored_doc_ids:
                    print(
                        f"[Patient Matching] patient_match_scorer stamped scores on "
                        f"{scored_doc_ids}/{len(validated_chunks)} chunks"
                    )
        except Exception as _e:
            print(f"[Patient Matching] patient_match_scorer wiring failed: {_e}")

        matches = []
        for chunk in validated_chunks:
            match_data = self._build_match_data(chunk, patient_profile)
            if match_data:
                # Tag matches with their multi-specialty provenance so the
                # route-level merger can surface which experts found them.
                source_tag = chunk.get("_source")
                if source_tag:
                    match_data.setdefault("source", source_tag)
                specs = chunk.get("_specialties") or []
                if specs:
                    match_data.setdefault("specialties", list(specs))
                matches.append(match_data)

        return {
            "matches": matches,
            "total_matches": len(matches),
            "patient_summary": self._build_patient_summary(patient_profile),
        }
