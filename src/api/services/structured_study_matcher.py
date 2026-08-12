"""
Structured Study Matcher Service

Queries PostgreSQL to find studies matching structured query fields.
Runs in PARALLEL with Qdrant vector search for hybrid retrieval.

Scoring system (weighted by clinical importance):
- Cancer site/location: 25 points (most important for relevance)
- Histology: 15 points (subsumption scoring handles partial matches)
- Stage/TNM: 15 points (rebalanced — stage subsumption is common)
- Treatment modality: 15 points (relevant for dose/outcome questions)
- Patient demographics: 10 points (age, gender matching)
- Study quality: 10 points (phase, patient count)

Total possible: 100 points, normalized to 0-1 score
"""

import asyncio
import asyncpg
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from src.core.config import settings
from src.api.services.query_reconciliation import ReconciledStructure


# Canonical biomarker lookup — maps common names/aliases to canonical keys.
# Used to normalize user query biomarkers before matching against biomarker_status JSONB.
CANONICAL_BIOMARKERS = {
    # Hormone receptors
    "er": "ER", "estrogen receptor": "ER", "esr1": "ER",
    "pr": "PR", "progesterone receptor": "PR", "pgr": "PR",
    "her2": "HER2", "her-2": "HER2", "her 2": "HER2", "erbb2": "HER2", "her2/neu": "HER2",
    # Classic driver mutations
    "egfr": "EGFR", "erbb-1": "EGFR", "her1": "EGFR",
    "alk": "ALK",
    # NOTE: "alk fusion" / "alk rearrangement" are intentionally NOT in this
    # map so they fall through to _parse_polarity_phrase (stage 6) which
    # correctly sets status='mutant'. Same for "braf v600e" / "kras mutation".
    "kras": "KRAS",
    "braf": "BRAF",
    "brca": "BRCA", "brca1": "BRCA1", "brca2": "BRCA2",
    "ros1": "ROS1", "ret": "RET", "met": "MET",
    "ntrk": "NTRK", "ntrk1": "NTRK", "ntrk2": "NTRK", "ntrk3": "NTRK",
    "fgfr": "FGFR", "fgfr1": "FGFR1", "fgfr2": "FGFR2", "fgfr3": "FGFR3",
    "hras": "HRAS", "nras": "NRAS",
    "idh": "IDH", "idh1": "IDH1", "idh2": "IDH2",
    "pik3ca": "PIK3CA", "pten": "PTEN", "akt1": "AKT1",
    "atm": "ATM", "palb2": "PALB2", "chek2": "CHEK2", "rad51": "RAD51",
    "mgmt": "MGMT",
    "tp53": "TP53", "p53": "TP53",
    "notch1": "NOTCH1", "notch2": "NOTCH2",
    "ctnnb1": "CTNNB1", "beta-catenin": "CTNNB1",
    "arid1a": "ARID1A", "smad4": "SMAD4",
    # Checkpoint / immunotherapy
    "pd-l1": "PD-L1", "pdl1": "PD-L1", "cd274": "PD-L1", "b7-h1": "PD-L1",
    "pd-1": "PD-1", "pd1": "PD-1", "pdcd1": "PD-1",
    "ctla-4": "CTLA-4", "ctla4": "CTLA-4",
    "lag-3": "LAG-3", "lag3": "LAG-3",
    # Genomic instability
    "msi": "MSI", "msi-h": "MSI", "dmmr": "MSI", "microsatellite instability": "MSI",
    "mss": "MSS", "pmmr": "MSS", "microsatellite stable": "MSS",
    "tmb": "TMB", "tmb-h": "TMB", "tumor mutational burden": "TMB",
    # Viral markers
    "hpv": "HPV", "human papillomavirus": "HPV", "p16": "HPV",
    "ebv": "EBV", "epstein-barr virus": "EBV",
    "hbv": "HBV", "hepatitis b": "HBV",
    "hcv": "HCV", "hepatitis c": "HCV",
    # Proliferation / IHC
    "ki-67": "Ki-67", "ki67": "Ki-67", "mki67": "Ki-67",
    # Hematologic / lymphoma surface markers
    "cd3": "CD3", "cd4": "CD4", "cd5": "CD5", "cd8": "CD8",
    "cd10": "CD10", "cd15": "CD15", "cd19": "CD19", "cd20": "CD20",
    "cd22": "CD22", "cd23": "CD23", "cd25": "CD25",
    "cd30": "CD30", "cd34": "CD34", "cd38": "CD38",
    "cd45": "CD45", "cd56": "CD56", "cd79a": "CD79a",
    "cd138": "CD138",
    "bcl-2": "BCL2", "bcl2": "BCL2",
    "bcl-6": "BCL6", "bcl6": "BCL6",
    "myc": "MYC", "c-myc": "MYC",
    "ccnd1": "CCND1", "cyclin d1": "CCND1",
    "ezh2": "EZH2", "btk": "BTK",
    # Serum / soluble biomarkers
    "psa": "PSA", "prostate specific antigen": "PSA",
    "ldh": "LDH", "lactate dehydrogenase": "LDH", "serum ldh": "LDH",
    "b2m": "B2M", "β2m": "B2M", "beta-2 microglobulin": "B2M",
    "beta 2 microglobulin": "B2M", "beta2 microglobulin": "B2M",
    "β2-microglobulin": "B2M", "beta-2-microglobulin": "B2M",
    "crp": "CRP", "c-reactive protein": "CRP",
    "cea": "CEA", "carcinoembryonic antigen": "CEA",
    "ca 15-3": "CA15-3", "ca15-3": "CA15-3", "ca-15-3": "CA15-3",
    "ca 27-29": "CA27-29", "ca27-29": "CA27-29",
    "ca 125": "CA125", "ca125": "CA125", "ca-125": "CA125",
    "ca 19-9": "CA19-9", "ca19-9": "CA19-9", "ca-19-9": "CA19-9",
    "afp": "AFP", "alpha-fetoprotein": "AFP", "alpha fetoprotein": "AFP",
    "hcg": "hCG", "beta-hcg": "hCG", "β-hcg": "hCG",
    "chromogranin a": "CGA", "chromogranin": "CGA",
    "nse": "NSE", "neuron-specific enolase": "NSE",
    "cyfra 21-1": "CYFRA21-1", "cyfra": "CYFRA21-1",
    "pro-grp": "proGRP", "pro grp": "proGRP", "progrp": "proGRP",
    # Liquid biopsy / dynamics
    "ctdna": "ctDNA", "ct-dna": "ctDNA", "circulating tumor dna": "ctDNA",
    "ctc": "CTC", "circulating tumor cells": "CTC",
    # Prostate-specific
    "psma": "PSMA",
    "ar": "AR", "androgen receptor": "AR", "ar-v7": "AR-V7",
    # Targeted stains / cell-surface targets
    "s100": "S100", "hmb-45": "HMB45", "melan-a": "MelanA",
    "gfap": "GFAP", "synaptophysin": "SYNAPTOPHYSIN",
    "vimentin": "VIMENTIN", "desmin": "DESMIN",
    "sstr2": "SSTR2", "sstr5": "SSTR5", "somatostatin receptor": "SSTR",
    "nectin-4": "NECTIN4", "nectin4": "NECTIN4",
    "claudin 18.2": "CLDN18.2", "cldn18.2": "CLDN18.2",
    "trop-2": "TROP2", "trop2": "TROP2",
    "folate receptor alpha": "FRA", "fr-alpha": "FRA",
    # Special compound
    "triple negative": "TNBC", "triple-negative": "TNBC", "tnbc": "TNBC",
}

# Maps polarity indicators from user queries to normalized status values
POLARITY_MAP = {
    # Positive indicators
    "+": "positive", "positive": "positive", "amplified": "positive",
    "overexpressed": "positive", "detected": "positive", "high": "positive",
    # Negative indicators
    "-": "negative", "negative": "negative", "non-amplified": "negative",
    "absent": "negative", "not detected": "negative", "low": "negative",
    # Mutation indicators
    "mutant": "mutant", "mutation": "mutant", "mutated": "mutant",
    "altered": "mutant", "fusion": "mutant", "rearrangement": "mutant",
    "translocation": "mutant",
    # Wild-type indicators
    "wild-type": "wild-type", "wild type": "wild-type", "wt": "wild-type",
    # MSI-specific
    "msi-h": "high", "msi h": "high", "mss": "stable", "msi-l": "stable",
}


@dataclass
class StructuredMatchResult:
    """Result from structured PostgreSQL matching."""
    doc_ids: Set[str]
    match_scores: Dict[str, float]  # doc_id -> match score (0-1)
    match_details: Dict[str, Dict[str, Any]]  # doc_id -> {matched_criteria, points, etc}
    query_time_ms: float
    conditions_used: List[str]
    max_possible_score: float = 100.0


@dataclass
class AxisContribution:
    """Per-axis score component for PG match breakdown."""
    axis_name: str           # e.g., "cancer_site", "biomarkers"
    points_earned: float     # e.g., 28.0
    max_points: float        # e.g., 30.0
    label: str               # "exact match", "partial match", "not reported"


@dataclass
class PGMatchBreakdown:
    """Container for total PG match score and per-axis contributions."""
    total_score: float                    # 0-100 normalized
    axis_contributions: List[AxisContribution]
    axis_mismatches: List[str]            # axes where score = 0 despite data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "axes": [
                {
                    "axis": ac.axis_name,
                    "earned": ac.points_earned,
                    "max": ac.max_points,
                    "label": ac.label,
                }
                for ac in self.axis_contributions
            ],
            "mismatches": self.axis_mismatches,
        }


# Scoring weights for different criteria categories (base weights)
# These are redistributed dynamically based on which criteria are present in the query
BASE_SCORING_WEIGHTS = {
    "cancer_site": 25,      # Most important - determines relevance
    "site_detail": 10,      # Bonus for specific location match
    "histology": 15,        # Reduced — subsumption scoring handles partial matches
    "stage": 15,            # Stage category (I, II, III, IV) — rebalanced
    "disease_descriptor": 8, # Locally advanced, metastatic, etc.
    "tnm_t": 5,             # T stage component
    "tnm_n": 5,             # N stage component
    "tnm_m": 5,             # M stage component
    "treatment": 15,        # Treatment modality
    "treatment_setting": 8, # Neoadjuvant, adjuvant, definitive, palliative
    "age_range": 5,         # Patient age matching
    "gender": 3,            # Patient gender matching
    "performance_status": 5, # ECOG/KPS matching
    "biomarkers": 20,       # Molecular / immunohistochemistry profile match
                            # (fractional: matches / overlap, see
                            # _build_biomarker_score_sql below)
    "metastatic_sites": 8,  # Metastatic site fractional match (Class 3a).
                            # Patient metastatic_sites list is compared
                            # against study cancer_location + extraction_data;
                            # silent sites are ignored (same semantic as
                            # biomarkers — matches / total_met_sites).
    "title_relevance": 10,  # Keyword overlap between patient profile and
                            # study_name (Class 3b). Additive tiebreaker
                            # that penalizes off-intent top hits (e.g.
                            # preop rectal studies for a metastatic colon
                            # patient) without blocking cryptic titles.
    "metastatic_status": 15, # Exact match against studies.metastatic_status column
    "risk_stratification": 8, # Exact match against studies.risk_stratification column
    "biomarker_jsonb": 20,  # JSONB containment match against studies.biomarker_status
                            # with STATUS_MATCH_SYNONYMS polarity expansion
                            # (increased from 14 — biomarker match critical for targeted therapy)
}


# Status synonym expansion: for each patient-side polarity, list the
# study-side polarities that should be treated as equivalent when
# diffing `biomarker_status->>'KEY'` against the patient's expected value.
#
# Clinical intent:
#   - "high" (e.g. PD-L1 high, TMB-H) is a STRICTER form of "positive";
#     a study reporting "positive" is a partial match for a patient
#     whose status is "high". We accept either.
#   - "mutant" / "mutation" / "altered" / "positive" (for gene-level
#     alterations) all collapse to the same class.
#   - "wild-type" / "wt" / "negative" all collapse to the opposite class.
#
# The SQL builder uses `IN (...)` with these synonym lists so a single
# biomarker match can fire on any equivalent study label.
STATUS_MATCH_SYNONYMS: Dict[str, List[str]] = {
    "positive":  ["positive", "pos", "+", "detected", "elevated", "high",
                  "amplified", "overexpressed", "overexpression"],
    "high":      ["high", "elevated", "positive", "strongly positive",
                  "amplified", "overexpressed", "overexpression"],
    "low":       ["low", "decreased", "weakly positive"],
    "negative":  ["negative", "neg", "-", "absent", "undetectable",
                  "undetected", "not detected", "wild-type", "wt"],
    "mutant":    ["mutant", "mutation", "mutated", "altered", "positive",
                  "fusion", "rearrangement", "translocation", "amplified"],
    "wild-type": ["wild-type", "wild type", "wt", "negative", "unaltered"],
    "stable":    ["stable", "mss", "negative", "pmmr"],
}


# ── Status-value regex fallback ────────────────────────────────────────────
#
# Pathology and trial reports store biomarker polarity in many formats that
# don't map cleanly to the STATUS_MATCH_SYNONYMS word list:
#
#   - IHC intensity notation: "0", "1+", "2+", "3+", "IHC 2+"
#   - Percent staining:       "75%", "≥50%", "<1%"
#   - Ambiguous English:      "weakly positive", "not amplified"
#   - Numeric-only thresholds: "50" (CPS/TPS), "450" (LDH)
#
# Each entry below is a single PostgreSQL-compatible regex (POSIX ERE,
# case-insensitive at the caller via LOWER(...)) that is OR-ed into the
# value-match check alongside the IN-list. Anchored with ^...$ so it
# matches the whole value field, tolerant of leading/trailing whitespace.
#
# Clinical conventions used:
#   HER2 IHC: 0/1+ negative, 2+ equivocal→positive, 3+ high
#   PD-L1 TPS: 0 negative, 1-49% positive, 50%+ high
#   PD-L1 CPS: 0 negative, 1-19 positive, 20+ high
#   Ki-67: ≤14% low, ≥20% high, 15-19% intermediate (approx.)
#   Allred/H-score: not handled here (too ambiguous without context)
STATUS_MATCH_REGEX: Dict[str, str] = {
    # Any unambiguous sign of expression
    "positive": (
        r"^\s*("
        r"positive|pos|\+|detected|elevated|high(?:ly)?|amplified|"
        r"overexpressed|overexpression|moderate(?:ly)? ?positive|"
        r"strong(?:ly)? ?positive|strong|moderate|"
        r"[23]\+|ihc\s*[23]\+?|"
        r"[1-9][0-9]*\s*%|"
        r"her2-?low|her2 low"
        r")\s*$"
    ),
    # Strictly high: IHC 3+, ≥50% staining, "strong", "elevated"
    "high": (
        r"^\s*("
        r"high(?:ly)?|strong(?:ly)?|elevated|amplified|overexpressed|"
        r"3\+|ihc\s*3\+?|"
        r"[5-9][0-9]\s*%|[1-9][0-9]{2,}\s*%"
        r")\s*$"
    ),
    # Explicit non-expression (IHC 0/1+ are conventionally negative)
    "negative": (
        r"^\s*("
        r"negative|neg|-|absent|undetectable|not\s*detected|"
        r"not\s*amplified|not\s*expressed|wild[-\s]?type|wt|"
        r"0\+?|ihc\s*[01]\+?|1\+|"
        r"0\s*%"
        r")\s*$"
    ),
    # Low expression: IHC 1+, <20% staining, "weak"
    "low": (
        r"^\s*("
        r"low|decreased|weak(?:ly)?(?:\s*positive)?|"
        r"1\+|ihc\s*1\+?|"
        r"[1-9]\s*%|1[0-9]\s*%"
        r")\s*$"
    ),
    # Any mutation / structural alteration descriptor
    "mutant": (
        r"^\s*("
        r"mutant|mutation(?:s)?|mutated|altered|alteration|"
        r"positive|detected|amplified|"
        r"fusion(?:\s*(?:positive|detected))?|"
        r"rearrangement|translocation|"
        r"[a-z]\d+[a-z]|exon\s*\d+|del\w*|ins\w*|dup\w*"
        r")\s*$"
    ),
    # Unmutated / reference sequence
    "wild-type": (
        r"^\s*("
        r"wild[-\s]?type|wt|negative|unaltered|"
        r"not\s*mutated|no\s*mutation|not\s*detected|absent"
        r")\s*$"
    ),
    # MSI/MMR stable
    "stable": (
        r"^\s*("
        r"stable|mss|negative|pmmr|"
        r"microsatellite\s*stable|mismatch\s*repair\s*proficient"
        r")\s*$"
    ),
}


# ── Canonical → alias JSONB key map ────────────────────────────────────────
#
# Postgres's JSONB operators (`->>`, `?`) are case-sensitive on the key, so a
# patient biomarker of "HER2" will not find a row where ingestion stored the
# key as "HER-2" or "ERBB2". This map provides the set of spellings to check
# for each canonical biomarker; the SQL builder OR-s across them so a match
# fires regardless of which spelling the ingestion pipeline chose.
#
# Explicit overrides are kept small and focused on the known-ambiguous cases.
# For canonicals not listed here, `_aliases_for_canonical()` auto-generates a
# reasonable set by inverting CANONICAL_BIOMARKERS and adding common case
# variants.
#
# RULE: every alias must match the JSONB-safe regex
# ^[A-Za-z0-9][A-Za-z0-9\-\._/ ]*$  — no quotes, no semicolons. Checked at
# runtime by _is_safe_jsonb_key().
CANONICAL_TO_ALIASES_MANUAL: Dict[str, List[str]] = {
    # ─── Breast / hormone receptors ─────────────────────────────────────
    "HER2":   ["HER2", "HER-2", "Her2", "ERBB2", "erbb2", "HER2/neu", "Her2/neu"],
    "ER":     ["ER", "er", "ESR1", "Estrogen Receptor"],
    "PR":     ["PR", "pr", "PGR", "Progesterone Receptor"],
    # ─── Checkpoint / immunotherapy ─────────────────────────────────────
    "PD-L1":  ["PD-L1", "PDL1", "pd-l1", "pdl1", "CD274", "B7-H1"],
    "PD-1":   ["PD-1", "PD1", "pd-1", "PDCD1"],
    "CTLA-4": ["CTLA-4", "CTLA4", "ctla-4"],
    "LAG-3":  ["LAG-3", "LAG3", "lag-3"],
    # ─── Classic drivers ────────────────────────────────────────────────
    "EGFR":   ["EGFR", "egfr", "ERBB1", "HER1"],
    "ALK":    ["ALK", "alk"],
    "ROS1":   ["ROS1", "ros1"],
    "KRAS":   ["KRAS", "kras"],
    "BRAF":   ["BRAF", "braf"],
    "BRCA":   ["BRCA", "brca"],
    "BRCA1":  ["BRCA1", "brca1", "BRCA 1"],
    "BRCA2":  ["BRCA2", "brca2", "BRCA 2"],
    "PIK3CA": ["PIK3CA", "pik3ca"],
    "TP53":   ["TP53", "tp53", "p53"],
    "MGMT":   ["MGMT", "mgmt"],
    # ─── Genomic instability ────────────────────────────────────────────
    "MSI":    ["MSI", "msi", "MSI-H", "msi-h", "dMMR", "DMMR",
               "Microsatellite Instability", "microsatellite instability"],
    "MSS":    ["MSS", "mss", "pMMR", "PMMR", "Microsatellite Stable"],
    "TMB":    ["TMB", "tmb", "TMB-H", "tmb-h", "Tumor Mutational Burden"],
    # ─── Viral ──────────────────────────────────────────────────────────
    "HPV":    ["HPV", "hpv", "p16", "P16", "HPV/p16"],
    "EBV":    ["EBV", "ebv", "Epstein-Barr Virus"],
    # ─── Hematologic / lymphoma markers ─────────────────────────────────
    "CD20":   ["CD20", "cd20"],
    "CD30":   ["CD30", "cd30"],
    "CD15":   ["CD15", "cd15"],
    "CD3":    ["CD3", "cd3"],
    "CD5":    ["CD5", "cd5"],
    "CD19":   ["CD19", "cd19"],
    "BCL2":   ["BCL2", "bcl2", "BCL-2", "bcl-2"],
    "BCL6":   ["BCL6", "bcl6", "BCL-6", "bcl-6"],
    "MYC":    ["MYC", "myc", "c-MYC", "c-myc"],
    # ─── Serum / soluble ────────────────────────────────────────────────
    "LDH":    ["LDH", "ldh", "Lactate Dehydrogenase", "lactate dehydrogenase"],
    "B2M":    ["B2M", "b2m", "β2M", "Beta-2 Microglobulin",
               "beta-2 microglobulin", "beta 2 microglobulin",
               "Beta2-Microglobulin", "β2-microglobulin"],
    "CRP":    ["CRP", "crp", "C-Reactive Protein"],
    "CEA":    ["CEA", "cea", "Carcinoembryonic Antigen"],
    "CA125":  ["CA125", "CA-125", "ca-125", "CA 125"],
    "CA19-9": ["CA19-9", "CA-19-9", "ca19-9", "CA 19-9"],
    "CA15-3": ["CA15-3", "CA-15-3", "ca15-3", "CA 15-3"],
    "CA27-29": ["CA27-29", "CA-27-29", "ca27-29", "CA 27-29"],
    "AFP":    ["AFP", "afp", "Alpha-Fetoprotein"],
    "PSA":    ["PSA", "psa", "Prostate-Specific Antigen"],
    # ─── Liquid biopsy ──────────────────────────────────────────────────
    "ctDNA":  ["ctDNA", "ctdna", "ct-DNA", "CT-DNA",
               "Circulating Tumor DNA", "circulating tumor dna"],
    "CTC":    ["CTC", "ctc", "Circulating Tumor Cells"],
    # ─── Prostate-specific ──────────────────────────────────────────────
    "PSMA":   ["PSMA", "psma"],
    "AR":     ["AR", "ar", "Androgen Receptor"],
    # ─── Pan-cancer targets ─────────────────────────────────────────────
    "Ki-67":  ["Ki-67", "Ki67", "KI-67", "MKI67", "mki67", "ki-67", "Ki 67"],
    "NTRK":   ["NTRK", "ntrk", "NTRK1", "NTRK2", "NTRK3"],
    "RET":    ["RET", "ret"],
    "MET":    ["MET", "met"],
    "FGFR":   ["FGFR", "fgfr", "FGFR1", "FGFR2", "FGFR3"],
    "IDH":    ["IDH", "idh", "IDH1", "IDH2"],
    "IDH1":   ["IDH1", "idh1"],
    "IDH2":   ["IDH2", "idh2"],
    "NECTIN4": ["NECTIN4", "Nectin-4", "nectin-4", "NECTIN-4"],
    "CLDN18.2": ["CLDN18.2", "Claudin 18.2", "claudin 18.2", "CLDN 18.2"],
    "TROP2":  ["TROP2", "Trop-2", "TROP-2", "trop2"],
    "FRA":    ["FRA", "FRα", "Folate Receptor Alpha", "folate receptor alpha"],
    # ─── MSI compound / triple negative ─────────────────────────────────
    "TNBC":   ["TNBC", "tnbc", "Triple Negative", "triple negative",
               "triple-negative"],
}


# JSONB-safe key regex: accepts letters/digits/hyphen/underscore/dot/slash/
# space (for multi-word alias like "Beta-2 Microglobulin"). Rejects quotes,
# semicolons, backslashes, null bytes, and other SQL-hostile characters.
# This is a defense in depth — we parameterize all alias keys in SQL so
# this check is belt-and-suspenders.
_SAFE_JSONB_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\._/ α-ωΑ-Ωβ²]*$")


def _is_safe_jsonb_key(key: str) -> bool:
    """True if `key` is safe to use as a JSONB lookup value."""
    return bool(key) and bool(_SAFE_JSONB_KEY_RE.match(key))


# Auto-generated alias list cache — populated lazily.
_ALIASES_CACHE: Dict[str, List[str]] = {}


def _aliases_for_canonical(canonical: str, max_aliases: int = 8) -> List[str]:
    """
    Return the Postgres JSONB keys to check for this canonical biomarker.

    Lookup priority:
        1. CANONICAL_TO_ALIASES_MANUAL (manually curated per-canonical list)
        2. Auto-inverted CANONICAL_BIOMARKERS (finds all query-side aliases
           that map to this canonical, adds common case variants)
        3. The canonical itself (always included)

    The returned list is deduplicated, filtered through `_is_safe_jsonb_key`,
    and capped at `max_aliases` entries so the generated SQL stays bounded.
    The canonical form is always the first entry.
    """
    if canonical in _ALIASES_CACHE:
        return _ALIASES_CACHE[canonical][:max_aliases]

    collected: List[str] = []
    seen: Set[str] = set()

    def _add(name: str) -> None:
        if not name or not _is_safe_jsonb_key(name):
            return
        if name in seen:
            return
        seen.add(name)
        collected.append(name)

    # Canonical always first
    _add(canonical)

    # Manual override
    manual = CANONICAL_TO_ALIASES_MANUAL.get(canonical, [])
    for alias in manual:
        _add(alias)

    # Auto-invert CANONICAL_BIOMARKERS: find all query-side aliases that
    # map to this canonical, add them plus common case variants
    for query_alias, c in CANONICAL_BIOMARKERS.items():
        if c != canonical:
            continue
        _add(query_alias)
        _add(query_alias.upper())
        _add(query_alias.title())

    _ALIASES_CACHE[canonical] = collected
    return collected[:max_aliases]


def _build_alias_key_exists_sql(
    canonical: str,
    params: List[Any],
    param_idx: int,
) -> Tuple[str, int]:
    """
    Build an OR'd key-existence SQL fragment across all known aliases for a
    canonical biomarker. Uses parameterized keys to avoid SQL injection.

    Returns (sql_fragment, new_param_idx). Appends aliases to `params`.

    Example output (for HER2):
        (biomarker_status ? $N OR biomarker_status ? $N+1 OR ...)
    """
    aliases = _aliases_for_canonical(canonical)
    if not aliases:
        return ("FALSE", param_idx)
    parts: List[str] = []
    for alias in aliases:
        parts.append(f"biomarker_status ? ${param_idx}")
        params.append(alias)
        param_idx += 1
    return ("(" + " OR ".join(parts) + ")", param_idx)


def _build_alias_value_match_sql(
    canonical: str,
    status: str,
    params: List[Any],
    param_idx: int,
) -> Tuple[str, int]:
    """
    Build an OR'd value-match SQL fragment that fires if ANY known alias of
    the canonical biomarker has a value matching the patient's status.

    For each alias, generates:
        (LOWER(TRIM(biomarker_status->>$N)) IN (synonym_list)
         OR LOWER(TRIM(biomarker_status->>$N)) ~ $regex_N)

    The synonym list and regex parameters are shared across aliases (same
    placeholders reused), so the parameter count stays linear in the
    number of aliases.

    Returns (sql_fragment, new_param_idx). Appends synonyms, regex, and
    alias-key parameters to `params` in that order.
    """
    synonyms = STATUS_MATCH_SYNONYMS.get(status, [status])
    regex = STATUS_MATCH_REGEX.get(status)

    # 1. Add synonym params (one placeholder per synonym, reused across aliases)
    in_placeholders: List[str] = []
    for syn in synonyms:
        in_placeholders.append(f"${param_idx}")
        params.append(syn.lower())
        param_idx += 1
    in_list = ", ".join(in_placeholders)

    # 2. Add regex param (one placeholder, reused across aliases)
    regex_placeholder: Optional[str] = None
    if regex:
        regex_placeholder = f"${param_idx}"
        params.append(regex)
        param_idx += 1

    # 3. Per-alias check
    aliases = _aliases_for_canonical(canonical)
    if not aliases:
        return ("FALSE", param_idx)

    per_alias: List[str] = []
    for alias in aliases:
        alias_placeholder = f"${param_idx}"
        params.append(alias)
        param_idx += 1

        val_expr = f"LOWER(TRIM(biomarker_status->>{alias_placeholder}))"
        in_check = f"{val_expr} IN ({in_list})"
        if regex_placeholder:
            per_alias.append(f"({in_check} OR {val_expr} ~ {regex_placeholder})")
        else:
            per_alias.append(in_check)

    return ("(" + " OR ".join(per_alias) + ")", param_idx)


# ── Class 3a: metastatic site fractional match helper ─────────────────────
#
# Given a patient's list of metastatic sites (e.g. ["liver", "bone"]), build
# a fractional-match SQL score expression that adds weight points to a study
# for each site it explicitly mentions in either `cancer_location` or the
# full `extraction_data` text. Silent sites are ignored (i.e. the
# denominator is the total patient sites, not per-study reported sites),
# mirroring how `biomarkers` fractional scoring treats unreported markers
# as neutral.
#
# Formula:
#     score = COALESCE(
#         ((Σ CASE WHEN site_hit THEN 1 ELSE 0 END)::float / N_sites) * weight
#     , 0)
#
# Reuses SITE_TO_LOCATION_PATTERNS when the provided site name is a known
# canonical key. Otherwise falls back to an escaped literal match so unknown
# site strings are still matched safely.

# Minimal metastatic-site pattern list for sites that are NOT already top-level
# keys in SITE_TO_LOCATION_PATTERNS (those would otherwise fall through to
# re.escape fallback, which is correct but narrow). This intentionally only
# maps common metastatic organ terms to their clinical aliases; canonical
# cancer sites like "lung" / "liver" are already in SITE_TO_LOCATION_PATTERNS.
_METASTATIC_SITE_SYNONYMS: Dict[str, List[str]] = {
    "liver":        ["liver", "hepat", "hcc"],
    "bone":         ["bone", "osseous", "skeletal"],
    "brain":        ["brain", "cerebr", "intracranial", "cns metast"],
    "lung":         ["lung", "pulmonary", "pulm"],
    "lymph_node":   ["lymph node", "nodal", "lymphadenopathy"],
    "adrenal":      ["adrenal"],
    "peritoneum":   ["periton", "peritoneal carcinomatosis"],
    "pleura":       ["pleura", "pleural"],
    "skin":         ["skin metast", "cutaneous metast", "dermal metast"],
    "cardiac":      ["cardiac", "heart", "right ventric", "left ventric", "pericard"],
    "right ventricle": ["right ventric", "cardiac"],
    "left ventricle":  ["left ventric", "cardiac"],
}


def _build_metastatic_site_match_sql(
    met_sites: List[str],
    weight: int,
    params: List[Any],
    param_idx: int,
) -> Tuple[Optional[str], int]:
    """
    Build a fractional-match score expression for a patient's metastatic
    sites. Returns (score_expr_or_None, new_param_idx). Returns None if
    the input list is empty or weight <= 0 (caller should skip the
    criterion).

    Each site becomes one CASE WHEN summand:
        CASE WHEN (cancer_location ~* $N OR extraction_data::text ~* $N)
             THEN 1 ELSE 0 END

    The regex per site unions every known alias for that site (e.g. liver →
    "liver|hepat|hcc"). Unknown sites fall back to a case-insensitive
    escaped literal. All site regex values are parameterized, so user-
    supplied site strings cannot break the SQL.
    """
    if not met_sites or weight <= 0:
        return None, param_idx

    case_terms: List[str] = []
    for site in met_sites:
        site_lc = (site or "").strip().lower()
        if not site_lc:
            continue

        # Prefer the metastatic synonym list; then the canonical cancer-site
        # patterns; finally a literal fallback.
        aliases: List[str] = []
        if site_lc in _METASTATIC_SITE_SYNONYMS:
            aliases = _METASTATIC_SITE_SYNONYMS[site_lc]
        elif site_lc in SITE_TO_LOCATION_PATTERNS:
            # Strip the % wildcards to get the regex fragment.
            aliases = [p.replace("%", "") for p in SITE_TO_LOCATION_PATTERNS[site_lc]]
        else:
            aliases = [re.escape(site_lc)]

        if not aliases:
            continue

        regex_combined = "|".join(aliases)
        placeholder = f"${param_idx}"
        params.append(regex_combined)
        param_idx += 1

        case_terms.append(
            f"CASE WHEN (cancer_location ~* {placeholder} "
            f"OR extraction_data::text ~* {placeholder}) THEN 1 ELSE 0 END"
        )

    if not case_terms:
        return None, param_idx

    numerator = " + ".join(case_terms)
    denom = max(1, len(case_terms))
    score_expr = (
        f"COALESCE((({numerator})::float / {denom}) * {weight}, 0)"
    )
    return score_expr, param_idx


# ── Class 3b: title keyword relevance helper ──────────────────────────────
#
# Build a fractional match score over keywords extracted from the patient
# profile against the study's `study_name`. The goal is to surface studies
# whose titles name the patient's specific clinical intent (e.g. "MSI-H
# metastatic colorectal") over generic titles that happen to share the
# cancer site (e.g. "preoperative chemoradiation for rectal cancer").
#
# Additive tiebreaker: a cryptic title (e.g. "ECOG-ACRIN EA1131") scores 0
# on this component but still retains its baseline from other criteria.
#
# Formula:
#     score = COALESCE(
#         ((Σ CASE WHEN LOWER(study_name) ~ kw_regex THEN 1 ELSE 0 END)::float
#          / N_keywords) * weight
#     , 0)

# Max keywords emitted per query. Caps the SQL size; also keeps the
# fractional denominator small enough that each keyword contributes
# meaningfully.
_TITLE_RELEVANCE_MAX_KEYWORDS = 6


def _extract_title_keywords(
    cancer: Dict[str, Any],
    treatment: Dict[str, Any],
    metastatic_sites: List[str],
) -> List[str]:
    """
    Build a small prioritized list of keyword regexes suitable for
    LOWER(study_name) ~ <regex> matching. Each entry is a single regex
    alternation (no anchors) safe for POSIX ERE.

    Priority order (up to _TITLE_RELEVANCE_MAX_KEYWORDS):
      1. Biomarker canonicals (up to 2)
      2. site_detail
      3. disease_descriptor (metastatic / advanced / locally advanced)
      4. metastatic sites union
      5. histology
      6. treatment modality
    """
    out: List[str] = []

    # 1. Up to 2 biomarker canonicals (lowercased).
    biomarkers = cancer.get("biomarkers") or []
    seen_bm: Set[str] = set()
    for raw in biomarkers[:4]:  # scan up to 4 to find 2 canonicals
        try:
            canonical, _ = _parse_biomarker_query(raw)
        except Exception:
            canonical = None
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen_bm:
            continue
        seen_bm.add(key)
        # MSI gets its common clinical synonyms; other markers use plain
        # lowercase (matcher is case-insensitive via LOWER(study_name)).
        if key == "msi":
            out.append(r"msi|microsatellite|dmmr|mismatch repair")
        elif key == "mss":
            out.append(r"mss|microsatellite stable|pmmr")
        elif key == "tmb":
            out.append(r"tmb|tumor mutational burden")
        elif key in ("pd-l1", "pdl1"):
            out.append(r"pd-?l1|cps|tps")
        elif key == "her2":
            out.append(r"her2|erbb2")
        else:
            out.append(re.escape(key))
        if len(seen_bm) >= 2:
            break

    # 2. site_detail (e.g. "oral cavity", "colon")
    site_detail = (cancer.get("site_detail") or "").strip()
    if site_detail and len(out) < _TITLE_RELEVANCE_MAX_KEYWORDS:
        # Use the phrase as-is (lowercased) — this captures compound
        # subsites like "oral cavity". Escape regex metacharacters.
        out.append(re.escape(site_detail.lower()))

    # 3. disease_descriptor → regex alternation
    dd = (cancer.get("disease_descriptor") or "").strip().lower()
    if dd and len(out) < _TITLE_RELEVANCE_MAX_KEYWORDS:
        dd_map = {
            "metastatic":         r"metastatic|stage iv|advanced",
            "locally advanced":   r"locally advanced|locoregionally advanced|stage iii",
            "advanced":           r"advanced|stage iv|metastatic",
            "early stage":        r"early stage|early-stage|stage i\b|stage ii\b",
            "oligometastatic":    r"oligometast|oligo",
            "recurrent":          r"recurrent|recurrence|relapsed",
        }
        out.append(dd_map.get(dd, re.escape(dd)))

    # 4. metastatic sites (joined into one alternation)
    if metastatic_sites and len(out) < _TITLE_RELEVANCE_MAX_KEYWORDS:
        frags: List[str] = []
        for s in metastatic_sites[:4]:
            sl = (s or "").strip().lower()
            if not sl:
                continue
            frags.extend(_METASTATIC_SITE_SYNONYMS.get(sl, [re.escape(sl)]))
        if frags:
            out.append("|".join(frags))

    # 5. histology
    histology = (cancer.get("histology") or "").strip().lower()
    if histology and len(out) < _TITLE_RELEVANCE_MAX_KEYWORDS:
        h_map = {
            "scc":             r"squamous|scc|epidermoid",
            "adenocarcinoma":  r"adenocarcinoma|adeno",
            "small_cell":      r"small cell|sclc",
            "large_cell":      r"large cell",
            "clear_cell":      r"clear cell|ccrcc",
            "transitional":    r"transitional|urothel|tcc",
            "ductal":          r"ductal|idc|dcis",
            "lobular":         r"lobular|ilc|lcis",
            "neuroendocrine":  r"neuroendocrine|net|carcinoid",
            "melanoma":        r"melanoma|melanocytic",
        }
        out.append(h_map.get(histology, re.escape(histology)))

    # 6. treatment modality
    modality = (treatment.get("modality") or "").strip().lower()
    if modality and len(out) < _TITLE_RELEVANCE_MAX_KEYWORDS:
        m_map = {
            "radiation":     r"radiation|radiotherapy|imrt|sbrt|chemoradiat",
            "chemotherapy":  r"chemotherap|cisplatin|carboplatin|5-fu",
            "surgery":       r"surgery|surgical|resection",
            "immunotherapy": r"immunotherap|pembro|nivo|checkpoint|pd-?1|pd-?l1",
            "targeted":      r"targeted|tki|trastuzumab|cetuximab",
            "hormonal":      r"endocrine|hormon|tamoxifen|letrozole|aromatase",
        }
        out.append(m_map.get(modality, re.escape(modality)))

    return out[:_TITLE_RELEVANCE_MAX_KEYWORDS]


def _build_title_relevance_sql(
    keywords: List[str],
    weight: int,
    params: List[Any],
    param_idx: int,
) -> Tuple[Optional[str], int]:
    """
    Build a fractional-match score expression for keyword overlap with
    `study_name`. Returns (score_expr_or_None, new_param_idx). Returns
    None for empty input or weight <= 0.

    Each keyword becomes one `CASE WHEN LOWER(study_name) ~ $N THEN 1
    ELSE 0 END`. The keyword itself is parameterized to prevent SQL
    injection via user-supplied strings.
    """
    if not keywords or weight <= 0:
        return None, param_idx

    case_terms: List[str] = []
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw:
            continue
        placeholder = f"${param_idx}"
        params.append(kw)
        param_idx += 1
        case_terms.append(
            f"CASE WHEN LOWER(study_name) ~ {placeholder} THEN 1 ELSE 0 END"
        )

    if not case_terms:
        return None, param_idx

    numerator = " + ".join(case_terms)
    denom = max(1, len(case_terms))
    score_expr = (
        f"COALESCE((({numerator})::float / {denom}) * {weight}, 0)"
    )
    return score_expr, param_idx


# Weight redistribution groups - when a criterion is missing, its weight
# is redistributed to related criteria in the same group
WEIGHT_GROUPS = {
    "diagnosis": ["cancer_site", "site_detail", "histology"],
    "staging": ["stage", "disease_descriptor", "tnm_t", "tnm_n", "tnm_m"],
    "clinical": ["treatment", "treatment_setting", "age_range", "gender", "performance_status"],
}


def calculate_dynamic_weights(
    present_criteria: List[str],
    user_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """
    Calculate dynamic weights based on which criteria are present in the query,
    optionally applying user-supplied multipliers from the Match Criteria panel.
    
    Args:
        present_criteria: List of criteria keys that are present in the query
        user_weights: Optional dict of user-supplied multipliers (e.g. {'biomarkers': 2.0, 'age': 0.5}).
                      Keys use frontend names mapped to internal criteria keys.
                      Values are multipliers where 1.0 = default.
        
    Returns:
        Dict mapping criterion -> adjusted weight (totaling 100)
    """
    if not present_criteria:
        return {}
    
    # Map frontend criteria names to internal scoring keys
    # Keys that don't map to any current scoring criterion are silently ignored,
    # allowing the frontend to send future keys (race, grade, tumor_size, etc.)
    # without breaking existing scoring.
    _USER_KEY_MAP = {
        "cancer_site": ["cancer_site", "site_detail"],
        "histology": ["histology"],
        "stage": ["stage", "disease_descriptor", "tnm_t", "tnm_n", "tnm_m"],
        "treatment": ["treatment"],
        "biomarkers": ["biomarkers"],  # biomarkers match against molecular_subtype column
        "age": ["age_range"],
        "performance_status": ["performance_status"],
        # New granular keys from the Match Criteria panel (stubs for future backend support)
        "race": [],
        "sex": ["gender"],
        "comorbidities": [],
        "recurrence_status": [],
        "grade": [],
        "tumor_size": [],
        "treatment_setting": ["treatment_setting"],
    }
    
    # Start with base weights for present criteria
    weights = {k: float(BASE_SCORING_WEIGHTS[k]) for k in present_criteria if k in BASE_SCORING_WEIGHTS}
    
    if not weights:
        return {}
    
    # Apply user multipliers before redistribution
    if user_weights:
        for user_key, multiplier in user_weights.items():
            # Skip non-numeric keys (e.g. biomarker_mode='strict')
            if not isinstance(multiplier, (int, float)):
                continue
            multiplier = max(0.0, min(3.0, float(multiplier)))  # Clamp to [0, 3]
            internal_keys = _USER_KEY_MAP.get(user_key, [user_key])
            for ik in internal_keys:
                if ik in weights:
                    weights[ik] = weights[ik] * multiplier
    
    # Redistribute proportionally so total = 100
    present_total = sum(weights.values())
    if present_total <= 0:
        return {}
    
    scale_factor = 100 / present_total
    
    adjusted_weights = {}
    for k, v in weights.items():
        adjusted_weights[k] = round(v * scale_factor)
    
    # Ensure total is exactly 100 (handle rounding)
    total = sum(adjusted_weights.values())
    if total != 100 and adjusted_weights:
        # Add/subtract difference from highest weight criterion
        max_key = max(adjusted_weights, key=adjusted_weights.get)
        adjusted_weights[max_key] += (100 - total)
    
    return adjusted_weights


# Mapping from query structure cancer sites to PostgreSQL cancer_location patterns
SITE_TO_LOCATION_PATTERNS = {
    "head_neck": [
        "%oral%", "%tongue%", "%pharynx%", "%larynx%", "%maxilla%", 
        "%mandible%", "%palate%", "%tonsil%", "%neck%", "%head%",
        "%nasopharyn%", "%oropharyn%", "%hypopharyn%", "%glotti%",
        "%salivary%", "%parotid%", "%buccal%", "%hnscc%"
    ],
    "breast": ["%breast%", "%mammary%", "%dcis%", "%idc%", "%ilc%"],
    "lung": ["%lung%", "%pulmonary%", "%bronch%", "%thorac%", "%nsclc%", "%sclc%"],
    "prostate": ["%prostate%", "%prostatic%"],
    "gi": [
        "%esophag%", "%gastric%", "%stomach%", "%colon%", "%rectal%", 
        "%rectum%", "%colorectal%", "%anal%", "%liver%", "%hepat%",
        "%pancrea%", "%bile%", "%gallbladder%", "%crc%", "%hcc%"
    ],
    "gyn": ["%cervix%", "%cervical%", "%uterus%", "%uterine%", "%ovary%", "%ovarian%", "%vulva%", "%vagina%", "%endometri%"],
    "gu": ["%bladder%", "%kidney%", "%renal%", "%urothel%", "%testis%", "%testicul%", "%rcc%", "%mibc%", "%nmibc%"],
    "cns": ["%brain%", "%glioma%", "%glioblastoma%", "%meningioma%", "%spine%", "%spinal%", "%cerebr%", "%gbm%"],
    "lymphoma": ["%lymphoma%", "%hodgkin%", "%dlbcl%", "%nhl%"],
    "sarcoma": ["%sarcoma%", "%soft tissue%"],
    "skin": ["%melanoma%", "%skin%", "%cutaneous%", "%merkel%"],
    "thyroid": ["%thyroid%"],
}

# Histology patterns with synonyms/abbreviations
HISTOLOGY_TO_PATTERNS = {
    "scc": ["%squamous%", "%SCC%", "%epidermoid%", "%keratinizing%"],
    "adenocarcinoma": ["%adenocarcinoma%", "%adeno%", "%glandular%"],
    "small_cell": ["%small cell%", "%SCLC%", "%oat cell%"],
    "large_cell": ["%large cell%", "%LCLC%"],
    "clear_cell": ["%clear cell%", "%ccRCC%"],
    "transitional": ["%transitional%", "%urothelial%", "%TCC%"],
    "ductal": ["%ductal%", "%IDC%", "%DCIS%"],
    "lobular": ["%lobular%", "%ILC%", "%LCIS%"],
    "neuroendocrine": ["%neuroendocrine%", "%NET%", "%carcinoid%"],
    "melanoma": ["%melanoma%", "%melanocytic%"],
}

# Treatment modality patterns with synonyms
TREATMENT_TO_PATTERNS = {
    "radiation": [
        "%radiation%", "%radiotherapy%", "%RT%", "%IMRT%", "%VMAT%", 
        "%proton%", "%photon%", "%SBRT%", "%SRS%", "%brachytherapy%",
        "%EBRT%", "%XRT%", "%chemoradiation%", "%chemoradiotherapy%", "%CRT%"
    ],
    "chemotherapy": [
        "%chemotherapy%", "%chemo%", "%cisplatin%", "%carboplatin%",
        "%docetaxel%", "%paclitaxel%", "%5-FU%", "%fluorouracil%",
        "%doxorubicin%", "%cyclophosphamide%", "%gemcitabine%"
    ],
    "surgery": [
        "%surgery%", "%surgical%", "%resection%", "%excision%",
        "%mastectomy%", "%lumpectomy%", "%prostatectomy%", "%nephrectomy%",
        "%lobectomy%", "%colectomy%", "%laryngectomy%", "%glossectomy%"
    ],
    "immunotherapy": [
        "%immunotherapy%", "%checkpoint%", "%PD-1%", "%PD-L1%", 
        "%pembrolizumab%", "%nivolumab%", "%ipilimumab%", "%atezolizumab%",
        "%durvalumab%", "%avelumab%", "%ICI%", "%IO%"
    ],
    "targeted": [
        "%targeted%", "%TKI%", "%trastuzumab%", "%pertuzumab%",
        "%lapatinib%", "%erlotinib%", "%gefitinib%", "%osimertinib%",
        "%bevacizumab%", "%cetuximab%"
    ],
    "hormonal": [
        "%hormonal%", "%hormone%", "%endocrine%", "%tamoxifen%",
        "%letrozole%", "%anastrozole%", "%ADT%", "%androgen%",
        "%enzalutamide%", "%abiraterone%"
    ],
}

# Stage normalization - maps extracted stage to search patterns
# These are converted to regex by stripping % and joining with |
# IMPORTANT: Do NOT include TNM components (T1, T2, etc.) here — those are
# separate criteria. Stage patterns must match the actual stage classification.
STAGE_TO_PATTERNS = {
    "I": ["stage I[^IV]", "stage 1[^0-9]", "stage I\\b"],
    "IA": ["stage IA", "stage 1A"],
    "IB": ["stage IB", "stage 1B"],
    "II": ["stage II[^I]", "stage 2[^0-9]", "stage II\\b"],
    "IIA": ["stage IIA", "stage 2A"],
    "IIB": ["stage IIB", "stage 2B"],
    "III": ["stage III", "stage 3", "locally advanced"],
    "IIIA": ["stage IIIA", "stage 3A"],
    "IIIB": ["stage IIIB", "stage 3B"],
    "IIIC": ["stage IIIC", "stage 3C"],
    "IV": ["stage IV", "stage 4", "metastatic"],
    "IVA": ["stage IVA", "stage 4A"],
    "IVB": ["stage IVB", "stage 4B"],
}


# ── Staged biomarker parser helpers ────────────────────────────────────────
#
# The old parser walked POLARITY_MAP with plain `if word in base` substring
# checks, which caused the hyphen in "pd-l1" to be interpreted as a negative
# polarity suffix ("PD-L1 CPS 50" → ('PDL1 CPS 50', 'negative')). It also
# had no numeric-threshold or variant-mutation handling, so inputs like
# "CPS score of 100", "EGFR L858R", and "KRAS G12C" fell through to the
# uppercase fallback unchanged.
#
# The new parser runs in stages, each stage trying to match a specific
# input class. It returns as soon as any stage fires.

_SPECIAL_COMPOUNDS: Dict[str, tuple] = {
    "triple negative": ("TNBC", "positive"),
    "triple-negative": ("TNBC", "positive"),
    "tnbc": ("TNBC", "positive"),
    "msi-h": ("MSI", "high"),
    "msi h": ("MSI", "high"),
    "msi high": ("MSI", "high"),
    "mss": ("MSS", "stable"),
    "msi-l": ("MSS", "stable"),
    "msi l": ("MSS", "stable"),
    "msi low": ("MSS", "stable"),
    "dmmr": ("MSI", "high"),
    "mmr-deficient": ("MSI", "high"),
    "pmmr": ("MSS", "stable"),
    "mmr-proficient": ("MSS", "stable"),
    "tmb-h": ("TMB", "high"),
    "tmb high": ("TMB", "high"),
}


def _parse_numeric_threshold(marker_lower: str) -> Optional[tuple]:
    """
    Recognise numeric biomarker thresholds and map them to (canonical, status).

    Handles:
        "CPS 100", "CPS score of 100", "PD-L1 CPS 50", "pd-l1 cps ≥ 20"
          → (PD-L1, high or positive based on threshold)
        "TPS 80%", "pd-l1 tps ≥ 50"
          → (PD-L1, high or positive)
        "Oncotype 25", "21-gene recurrence score of 30"
          → (ONCOTYPE, low/intermediate/high)
        "Ki-67 20%"
          → (Ki-67, high or low)

    Clinical thresholds are based on the regulatory labels most widely used
    for each assay (pembrolizumab CPS ≥ 20 / ≥ 1 for HNSCC, TPS ≥ 50 / ≥ 1
    for NSCLC, Oncotype 11/25 TailorX, Ki-67 ≥ 20% as the breast-cancer
    high-proliferation threshold).
    """
    # Normalize common separator noise so the regexes are simpler
    normalized = marker_lower
    normalized = normalized.replace("≥", ">=").replace("≤", "<=")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # ── PD-L1 CPS / TPS ────────────────────────────────────────────────────
    # Matches: "cps 100", "cps of 100", "cps score of 100", "cps = 100",
    #          "cps >= 20", "pd-l1 cps 50", "pdl1 tps 80%"
    m = re.match(
        r"^(?:pd-?l1\s+)?(cps|tps)"
        r"(?:\s*(?:score|value))?"
        r"(?:\s*(?:of|=|>=?|<=?))?"
        r"\s*(\d+)\s*%?$",
        normalized,
    )
    if m:
        kind = m.group(1).upper()
        value = int(m.group(2))
        if value == 0:
            return ("PD-L1", "negative")
        if kind == "CPS":
            # HNSCC pembrolizumab monotherapy: CPS >= 20 is "high"
            return ("PD-L1", "high" if value >= 20 else "positive")
        # kind == "TPS" — NSCLC pembrolizumab monotherapy: TPS >= 50 is "high"
        return ("PD-L1", "high" if value >= 50 else "positive")

    # ── Oncotype DX (breast) ───────────────────────────────────────────────
    m = re.match(
        r"^(?:21[-\s]?gene\s+)?"
        r"(?:oncotype(?:\s*dx)?)?"
        r"\s*(?:recurrence\s*score|rs)?"
        r"\s*(?:of|=)?"
        r"\s*(\d+)$",
        normalized,
    )
    if m and ("oncotype" in normalized or "recurrence" in normalized or "21-gene" in normalized):
        value = int(m.group(1))
        if value < 11:
            return ("ONCOTYPE", "low")
        if value < 26:
            return ("ONCOTYPE", "intermediate")
        return ("ONCOTYPE", "high")

    # ── Ki-67 proliferation index ──────────────────────────────────────────
    m = re.match(
        r"^(?:ki-?67|mki67)\s*(?:index\s*)?(?:of|=)?\s*(\d+)\s*%?$",
        normalized,
    )
    if m:
        value = int(m.group(1))
        return ("Ki-67", "high" if value >= 20 else "low")

    return None


# Canonical-marker prefixes that commonly carry protein or exon variants.
# When we see "<marker> L858R" / "<marker> exon 19" / "<marker> p.V600E", we
# map to (canonical, "mutant") — the specific variant is clinically
# meaningful but lives at a finer granularity than the matcher currently
# models.
_VARIANT_PATTERN = re.compile(
    r"^([a-z][a-z0-9\-]*)\s+"
    r"(?:"
    r"(?:p\.)[a-z]\d+[a-z]\b"          # protein change with p. prefix: p.V600E
    r"|[a-z]\d{2,}[a-z]\b"             # bare protein change: L858R, V600E, G12C
    r"|(?:exon|codon)\s*\d+(?:\s+\w+)?"  # exon 19, codon 12 mutation, etc.
    r"|(?:del|ins|dup)\s*\w*"          # del19, ins20
    r")"
)
# NOTE: amplification / amplified / overexpressed / fusion / rearrangement /
# translocation are handled in _parse_polarity_phrase (stage 6), not as
# variants, because clinically they are polarity words describing HOW the
# gene is altered (positive expression vs. mutant structure).


def _parse_variant_mutation(marker_lower: str) -> Optional[tuple]:
    """
    Recognise "<gene> <variant>" patterns like:
        "EGFR L858R", "EGFR exon 19", "EGFR exon 20 insertion",
        "KRAS G12C", "BRAF V600E", "BRAF p.V600E", "NTRK fusion",
        "HER2 amplification"

    Returns (canonical_marker, "mutant") or None.
    """
    m = _VARIANT_PATTERN.match(marker_lower)
    if not m:
        return None
    base = m.group(1)
    canonical = CANONICAL_BIOMARKERS.get(base)
    if canonical is None:
        # Not a known gene prefix — bail out so the uppercase fallback runs
        return None
    return (canonical, "mutant")


# Trailing polarity phrases, ordered LONGEST-FIRST so the parser strips
# "weakly positive" before it strips "positive".
_POLARITY_PHRASES: List[tuple] = sorted(
    [
        ("wild-type", "wild-type"), ("wild type", "wild-type"), ("wt", "wild-type"),
        ("mutant", "mutant"), ("mutation", "mutant"), ("mutated", "mutant"),
        ("fusion", "mutant"), ("rearrangement", "mutant"), ("translocation", "mutant"),
        ("amplified", "positive"), ("amplification", "positive"),
        ("overexpressed", "positive"), ("overexpression", "positive"),
        ("strongly positive", "positive"), ("weakly positive", "positive"),
        ("focally positive", "positive"),
        ("positive", "positive"), ("pos", "positive"),
        ("elevated", "positive"), ("high", "positive"), ("detected", "positive"),
        ("detectable", "positive"), ("expressed", "positive"),
        ("negative", "negative"), ("neg", "negative"),
        ("absent", "negative"), ("undetectable", "negative"),
        ("low", "negative"), ("not detected", "negative"),
    ],
    key=lambda x: -len(x[0]),
)

_MODIFIER_WORDS_RE = re.compile(
    r"\b(weakly|strongly|focally|partially|strong|weak|partial|mild|marked)\b"
)


def _parse_polarity_phrase(marker_lower: str) -> Optional[tuple]:
    """
    Parse a marker whose tail is a polarity word or phrase, e.g.
    "CD20 positive", "EGFR mutant", "HER2 amplified", "CD30 weakly positive".

    Uses longest-phrase-wins word-boundary matching on the TAIL of the
    string — so "pd-l1" is NEVER mistaken for a negative polarity
    (the hyphen is in the middle, not a trailing suffix).
    """
    for phrase, status in _POLARITY_PHRASES:
        # Require the phrase to appear as a trailing whole phrase, either
        # at end of string or preceded by a space.
        if marker_lower == phrase:
            return None  # a bare polarity word with no marker — unrecognisable
        if marker_lower.endswith(" " + phrase):
            base = marker_lower[: -(len(phrase) + 1)].strip()
            base = _MODIFIER_WORDS_RE.sub("", base).strip()
            base = re.sub(r"\s+", " ", base)
            if not base:
                return None
            canonical = CANONICAL_BIOMARKERS.get(base)
            if canonical is None:
                return (base.upper(), status)
            return (canonical, status)
    return None


_POLARITY_SUFFIX_RE = re.compile(r"^(.+?)\s*([+-])$")


def _parse_polarity_suffix(marker_clean: str) -> Optional[tuple]:
    """
    Parse markers ending in standalone +/-, tolerating an optional space
    before the sign (e.g. "ER+", "ER +", "HER2-", "HER2 -").

    Key fix vs the old parser: the regex requires the sign to be the LAST
    character with no trailing content, so the hyphen inside "pd-l1" is
    never stripped — PD-L1 never enters this path because it doesn't end
    in a bare +/-.
    """
    m = _POLARITY_SUFFIX_RE.match(marker_clean)
    if not m:
        return None
    base = m.group(1).strip().lower()
    sign = m.group(2)
    if not base:
        return None
    canonical = CANONICAL_BIOMARKERS.get(base, base.upper())
    status = "positive" if sign == "+" else "negative"
    return (canonical, status)


def _parse_biomarker_query(marker: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a user biomarker query string into (canonical_name, status).

    Staged parser — each stage tries one input class and returns if it
    matches. Stages, in order:

        0. Empty / whitespace-only
        1. Special compound cases (triple negative, MSI-H, dMMR, etc.)
        2. Direct CANONICAL_BIOMARKERS lookup (full string)
        3. Numeric threshold (CPS 100, TPS 80%, Oncotype 25, Ki-67 20%)
        4. Protein/exon variant (EGFR L858R, KRAS G12C, BRAF V600E, exon 19)
        5. Trailing +/- suffix with whitespace tolerance (ER+, HER2 -)
        6. Trailing polarity phrase (CD20 positive, EGFR mutant)
        7. Canonical lookup on the stripped base (fallback)
        8. Uppercase fallback (never returns None for non-empty input)

    Examples:
        "ER+"            -> ("ER", "positive")
        "HER2 -"         -> ("HER2", "negative")
        "EGFR mutant"    -> ("EGFR", "mutant")
        "EGFR L858R"     -> ("EGFR", "mutant")
        "EGFR exon 19"   -> ("EGFR", "mutant")
        "KRAS G12C"      -> ("KRAS", "mutant")
        "triple negative"-> ("TNBC", "positive")
        "MSI-H"          -> ("MSI", "high")
        "CPS score of 100" -> ("PD-L1", "high")
        "PD-L1 CPS 50"   -> ("PD-L1", "high")
        "TPS 80%"        -> ("PD-L1", "high")
        "CD20 positive"  -> ("CD20", "positive")
        "CD30 weakly positive" -> ("CD30", "positive")
        "PD-L1"          -> ("PD-L1", None)
    """
    if not marker or not marker.strip():
        return (None, None)

    marker_clean = marker.strip()
    marker_lower = marker_clean.lower()

    # Stage 1: special compounds
    if marker_lower in _SPECIAL_COMPOUNDS:
        return _SPECIAL_COMPOUNDS[marker_lower]

    # Stage 2: full-string canonical lookup
    if marker_lower in CANONICAL_BIOMARKERS:
        return (CANONICAL_BIOMARKERS[marker_lower], None)

    # Stage 3: numeric thresholds (CPS/TPS/Oncotype/Ki-67)
    hit = _parse_numeric_threshold(marker_lower)
    if hit:
        return hit

    # Stage 4: protein/exon variants
    hit = _parse_variant_mutation(marker_lower)
    if hit:
        return hit

    # Stage 5: trailing +/- suffix (with whitespace tolerance)
    hit = _parse_polarity_suffix(marker_clean)
    if hit:
        return hit

    # Stage 6: trailing polarity phrase
    hit = _parse_polarity_phrase(marker_lower)
    if hit:
        return hit

    # Stage 7: fallback canonical lookup on the lowercased input
    canonical = CANONICAL_BIOMARKERS.get(marker_lower)
    if canonical:
        return (canonical, None)

    # Stage 8: uppercase fallback
    return (marker_clean.upper(), None)


def _build_score_case(condition: str, weight: int, param_idx: int) -> str:
    """Build a CASE WHEN statement for scoring."""
    return f"CASE WHEN {condition} THEN {weight} ELSE 0 END"


def _biomarker_to_postgres_regex(marker: str) -> Optional[str]:
    """
    Convert a canonical biomarker name to a PostgreSQL regex pattern that
    preserves polarity (positive vs negative).

    CRITICAL: "ER+" must NOT match studies about "ER-" patients and vice versa.

    Examples:
        "ER+"           -> "er.*(positive|\\+)" (matches "ER positive" or "ER+")
        "ER-"           -> "er.*(negative|\\-)" (matches "ER negative" or "ER-")
        "HER2+"         -> "her.?2.*(positive|amplified|overexpress|\\+)"
        "HER2-"         -> "her.?2.*(negative|non.amplified|\\-)"
        "EGFR mutant"   -> "egfr.*(mutant|mutation|mutated|positive|\\+)"
        "EGFR wild-type"-> "egfr.*(wild.type|wt|negative|\\-)"
        "MSI-H"         -> "msi.?h|microsatellite.instability.high"
        "ALK+"          -> "alk.*(positive|rearrang|fusion|translocation|\\+)"
        "triple negative"-> "triple.negative|tnbc"
    """
    marker_lower = marker.lower().strip()

    # Special cases first
    if marker_lower in ("triple negative", "triple-negative", "tnbc"):
        return "triple.negative|tnbc"
    if marker_lower == "msi-h" or marker_lower == "msi h":
        return "msi.?h|microsatellite.instability.high"
    if marker_lower in ("mss", "msi-l"):
        return "mss|msi.?l|microsatellite.stable"
    if marker_lower == "dmmr":
        return "dmmr|mismatch.repair.deficient"
    if marker_lower == "pmmr":
        return "pmmr|mismatch.repair.proficient"
    if marker_lower == "tmb-h":
        return "tmb.?h|tumor.mutational.burden.high"

    # Positive status indicators
    positive_terms = r"positive|amplified|overexpress|mutant|mutation|mutated|fusion|rearrang|translocation|detected|\+"
    # Negative status indicators
    negative_terms = r"negative|non.amplified|wild.type|wt|absent|not.detected|\-"

    # Determine polarity from marker name
    is_positive = any(ind in marker_lower for ind in [
        "+", "positive", "amplified", "mutant", "mutation", "mutated",
        "fusion", "rearrangement", "translocation", "detected",
        "altered", "elevated", "high", "methylated",
    ])
    is_negative = any(ind in marker_lower for ind in [
        "-", "negative", "wild-type", "wild type", "wt",
        "non-amplified", "absent", "unmethylated", "low",
    ])

    # Extract base biomarker name (strip status words)
    status_words = [
        "positive", "negative", "mutant", "mutation", "mutated",
        "amplified", "non-amplified", "wild-type", "wild type", "wt",
        "fusion", "rearrangement", "translocation", "altered",
        "elevated", "high", "low", "detected", "methylated", "unmethylated",
    ]
    base = marker_lower
    for word in status_words:
        base = base.replace(word, "")
    base = base.replace("+", "").replace("-", "").strip()
    base = re.sub(r"\s+", "", base)  # remove internal spaces for regex

    if not base:
        return None

    # Special handling for HER2 (can be written as HER2, HER-2, HER 2)
    if base in ("her2", "her-2", "her 2", "erbb2"):
        base_pattern = "her.?2|erbb2"
    elif base in ("pdl1", "pd-l1", "pd l1"):
        base_pattern = "pd.?l1"
    elif base.startswith("brca"):
        base_pattern = base  # brca1, brca2
    elif base.startswith("fgfr"):
        base_pattern = "fgfr[1-4]?"
    else:
        base_pattern = re.escape(base)

    # Build polarity-aware regex
    if is_positive:
        return f"{base_pattern}.*({positive_terms})"
    elif is_negative:
        return f"{base_pattern}.*({negative_terms})"
    else:
        # No clear polarity - just match the base name (e.g. "PD-L1", "PSA elevated")
        return base_pattern


async def match_studies_by_structure(
    query_structure: Dict[str, Any],
    limit: int = 50,
    user_weights: Optional[Dict[str, float]] = None,
    resolver_hints: Optional[Any] = None,
    reconciled: Optional[ReconciledStructure] = None,
) -> StructuredMatchResult:
    """
    Query PostgreSQL to find studies matching the structured query fields.
    Returns studies ranked by weighted match score.

    Args:
        query_structure: Dict from QueryStructure.to_dict()
        limit: Maximum number of doc_ids to return
        user_weights: Optional user-supplied criteria weight multipliers
        resolver_hints: Optional `ResolvedQueryTokens` from
            `query_token_resolver.resolve_query_tokens(query_text)`. When
            present, canonical biomarkers the resolver detected are folded
            into `query_structure["cancer"]["biomarkers"]` if the LLM/regex
            path missed them. Existing fields are not overwritten — resolver
            is a gap-filler, not a source of truth.
        reconciled: Optional ReconciledStructure from query reconciliation.
            When provided, patient fields are read from this structure
            instead of the raw query_structure dict.

    Returns:
        StructuredMatchResult with matching doc_ids and weighted scores
    """
    import time
    start_time = time.perf_counter()

    # ── Reconciled override ───────────────────────────────────────────────
    # When a ReconciledStructure is provided, convert it to the dict format
    # expected by the rest of this function and use it in place of the raw
    # query_structure.
    if reconciled is not None:
        query_structure = reconciled.to_query_structure_dict()

    if not query_structure:
        return StructuredMatchResult(
            doc_ids=set(),
            match_scores={},
            match_details={},
            query_time_ms=0,
            conditions_used=[]
        )

    # ── Fix H: Resolver hint pre-seeding ──────────────────────────────────
    # Only fills *missing* axes; never overwrites existing structure.
    if resolver_hints is not None:
        cancer = query_structure.setdefault("cancer", {}) or {}
        query_structure["cancer"] = cancer
        existing_biomarkers = cancer.get("biomarkers") or []
        resolver_biomarkers = sorted(getattr(resolver_hints, "biomarkers", ()))
        if not existing_biomarkers and resolver_biomarkers:
            cancer["biomarkers"] = resolver_biomarkers
            print(
                f"    [PG Input] resolver_hints seeded biomarkers="
                f"{resolver_biomarkers}"
            )
    
    conn = None
    # Retry connection with exponential backoff. Five parallel arms of the
    # Visual Comparison flow can race five asyncpg.connect() calls at once
    # against the managed Postgres; under load one often flaps with a
    # bare socket-level TimeoutError whose repr is empty. Silently falling
    # back to a zero-result matcher on the first failure is a data-loss
    # risk (user sees the arm returned no evidence and never knows why).
    # Retry 3 times with 0.5s → 1.5s → 4s backoff before giving up.
    _RETRY_DELAYS = (0.5, 1.5, 4.0)
    last_error: Optional[Exception] = None
    for attempt_idx, delay in enumerate(_RETRY_DELAYS):
        try:
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_database,
                timeout=10,           # was 5 — too aggressive under 5-way parallel load
                command_timeout=15,
            )
            if attempt_idx > 0:
                print(
                    f"[StructuredStudyMatcher] PG connection succeeded on "
                    f"retry #{attempt_idx} after {delay}s backoff"
                )
            break
        except (TimeoutError, asyncio.TimeoutError, Exception) as conn_error:
            last_error = conn_error
            # Use repr() + type().__name__ so we NEVER print an empty string.
            # Bare socket TimeoutErrors often stringify to '' which is why
            # the live log showed "PostgreSQL connection failed: " with no
            # error body.
            err_repr = (
                f"{type(conn_error).__name__}: {conn_error!r}"
                if not str(conn_error)
                else f"{type(conn_error).__name__}: {conn_error} ({conn_error!r})"
            )
            is_final = attempt_idx == len(_RETRY_DELAYS) - 1
            if is_final:
                print(
                    f"[StructuredStudyMatcher] PG connection FAILED after "
                    f"{len(_RETRY_DELAYS)} attempts: {err_repr}"
                )
            else:
                print(
                    f"[StructuredStudyMatcher] PG connect attempt "
                    f"{attempt_idx + 1}/{len(_RETRY_DELAYS)} failed "
                    f"({err_repr}) — retrying in {delay}s"
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise
    if conn is None:
        # All retries exhausted — return empty result instead of crashing
        # the caller. Caller (Visual Comparison arm, /rag query, etc.)
        # can still proceed on Qdrant evidence alone.
        return StructuredMatchResult(
            doc_ids=set(),
            match_scores={},
            match_details={},
            query_time_ms=0,
            conditions_used=[],
        )
    
    try:
        print(f"\n  {'─' * 70}")
        print(f"    STRUCTURED STUDY MATCHER: PostgreSQL Matching")
        print(f"  {'─' * 70}")

        # Build scoring components
        # Each entry is (axis_name, sql_expression) so we can build per-axis columns
        score_cases: List[Tuple[str, str]] = []
        where_conditions = []
        params = []
        param_idx = 1
        condition_names = []
        present_criteria = []  # Track which criteria are present for dynamic weighting

        # Extract nested fields
        cancer = query_structure.get("cancer") or {}
        patient = query_structure.get("patient") or {}
        treatment = query_structure.get("treatment") or {}

        print(f"    [PG Input] cancer: site={cancer.get('site')}, histology={cancer.get('histology')}, "
              f"stage={cancer.get('stage')}, tnm_t={cancer.get('tnm_t')}, tnm_n={cancer.get('tnm_n')}, tnm_m={cancer.get('tnm_m')}")
        print(f"    [PG Input] cancer: biomarkers={cancer.get('biomarkers', [])}, "
              f"receptor_status={cancer.get('receptor_status')}, "
              f"disease_descriptor={cancer.get('disease_descriptor')}")
        print(f"    [PG Input] patient: age={patient.get('age')}, gender={patient.get('gender')}, perf_status={patient.get('performance_status')}")
        print(f"    [PG Input] treatment: modality={treatment.get('modality')}")
        
        # First pass: detect which criteria are present
        if cancer.get("site") and cancer.get("site") in SITE_TO_LOCATION_PATTERNS:
            present_criteria.append("cancer_site")
        if cancer.get("site_detail"):
            present_criteria.append("site_detail")
        if cancer.get("histology") and cancer.get("histology") in HISTOLOGY_TO_PATTERNS:
            present_criteria.append("histology")
        if cancer.get("stage"):
            present_criteria.append("stage")
        if cancer.get("disease_descriptor"):
            present_criteria.append("disease_descriptor")
        if cancer.get("tnm_t"):
            present_criteria.append("tnm_t")
        if cancer.get("tnm_n"):
            present_criteria.append("tnm_n")
        if cancer.get("tnm_m"):
            present_criteria.append("tnm_m")
        if treatment.get("modality") and treatment.get("modality") in TREATMENT_TO_PATTERNS:
            present_criteria.append("treatment")
        if treatment.get("setting"):
            present_criteria.append("treatment_setting")
        if patient.get("age"):
            present_criteria.append("age_range")
        if patient.get("gender"):
            present_criteria.append("gender")
        if patient.get("performance_status"):
            present_criteria.append("performance_status")
        if cancer.get("biomarkers") or cancer.get("receptor_status"):
            present_criteria.append("biomarkers")
            present_criteria.append("biomarker_jsonb")
        # Class 3a: metastatic sites (fed from clinical_inference via a new
        # top-level `metastatic_sites` key on the matcher input dict).
        _met_sites_input = query_structure.get("metastatic_sites") or []
        if _met_sites_input:
            present_criteria.append("metastatic_sites")
        # Class 3b: title keyword relevance fires whenever there is at least
        # one source field we can pull keywords from (biomarker, site_detail,
        # histology, disease_descriptor, metastatic site, or modality).
        _title_kw_probe = (
            bool(cancer.get("biomarkers"))
            or bool(cancer.get("site_detail"))
            or bool(cancer.get("histology"))
            or bool(cancer.get("disease_descriptor"))
            or bool(_met_sites_input)
            or bool(treatment.get("modality"))
        )
        if _title_kw_probe:
            present_criteria.append("title_relevance")

        # Metastatic status detection: use ReconciledStructure when available,
        # otherwise detect from query_structure dict or raw text indicators.
        _metastatic_status_value: Optional[str] = None
        if reconciled is not None:
            _metastatic_status_value = reconciled.pg_metastatic_status()
        else:
            # Detect from query_structure fields: disease_descriptor, tnm_m, stage
            _desc = cancer.get("disease_descriptor", "") or ""
            _tnm_m_val = cancer.get("tnm_m", "") or ""
            _stage_val = (cancer.get("stage") or "").upper()
            _desc_lower = _desc.lower()
            if "non-metastatic" in _desc_lower or "non metastatic" in _desc_lower:
                _metastatic_status_value = "non-metastatic"
            elif "locally advanced" in _desc_lower:
                _metastatic_status_value = "locally_advanced"
            elif "metastatic" in _desc_lower or "metastases" in _desc_lower or "distant metastases" in _desc_lower:
                _metastatic_status_value = "metastatic"
            elif str(_tnm_m_val).strip() == "1":
                _metastatic_status_value = "metastatic"
            elif _stage_val.startswith("IV"):
                _metastatic_status_value = "metastatic"
        if _metastatic_status_value:
            present_criteria.append("metastatic_status")

        # Risk stratification detection: use ReconciledStructure when available,
        # otherwise detect from query_structure dict fields.
        _risk_strat_value: Optional[str] = None
        if reconciled is not None:
            _risk_strat_value = reconciled.risk_level
        else:
            # Detect from disease_descriptor or other cancer fields
            _risk_text = " ".join(
                str(v) for v in [
                    cancer.get("disease_descriptor", ""),
                    cancer.get("risk_group", ""),
                ] if v
            ).lower()
            _risk_patterns = [
                ("high-risk", ["high-risk", "high risk"]),
                ("intermediate-risk", ["intermediate-risk", "intermediate risk"]),
                ("low-risk", ["low-risk", "low risk"]),
                ("favorable", ["favorable"]),
                ("unfavorable", ["unfavorable"]),
            ]
            for _risk_label, _risk_kws in _risk_patterns:
                if any(kw in _risk_text for kw in _risk_kws):
                    _risk_strat_value = _risk_label
                    break
        if _risk_strat_value:
            present_criteria.append("risk_stratification")

        # Trajectory detection: use ReconciledStructure when available,
        # otherwise detect from query text / disease_descriptor.
        # Trajectory scoring is folded into the disease_descriptor axis
        # weight — it doesn't need its own weight in BASE_SCORING_WEIGHTS.
        _trajectory_value: Optional[str] = None
        if reconciled is not None:
            _trajectory_value = reconciled.disease_trajectory
        else:
            # Detect trajectory indicators from disease_descriptor or LLM axes
            _traj_text = " ".join(
                str(v) for v in [
                    cancer.get("disease_descriptor", ""),
                    cancer.get("stage", ""),
                ] if v
            ).lower()
            _traj_indicators = {
                "recurrent": ["recurrent", "recurrence", "relapsed"],
                "progressive": ["progressive", "progression"],
                "treatment-naive": ["treatment-naive", "treatment naive", "newly diagnosed", "untreated"],
                "metastatic": ["metastatic"],
                "locally_advanced": ["locally advanced"],
            }
            for _traj_label, _traj_kws in _traj_indicators.items():
                if any(kw in _traj_text for kw in _traj_kws):
                    _trajectory_value = _traj_label
                    break
        # When trajectory is detected but disease_descriptor is not already
        # in present_criteria, add it so the axis weight gets allocated.
        if _trajectory_value and "disease_descriptor" not in present_criteria:
            present_criteria.append("disease_descriptor")

        # Calculate dynamic weights based on present criteria
        print(f"    [PG Criteria] Present criteria: {present_criteria}")
        dynamic_weights = calculate_dynamic_weights(present_criteria, user_weights=user_weights)

        if not dynamic_weights:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"    [PG Criteria] No criteria detected → returning empty result")
            return StructuredMatchResult(
                doc_ids=set(),
                match_scores={},
                match_details={},
                query_time_ms=elapsed_ms,
                conditions_used=[]
            )

        print(f"    [PG Weights] Dynamic weights (total=100): {dynamic_weights}")
        if user_weights:
            print(f"    [PG Weights] User weight overrides applied: {user_weights}")
        
        # 1. Cancer site/location matching
        cancer_site = cancer.get("site")
        if cancer_site and cancer_site in SITE_TO_LOCATION_PATTERNS:
            patterns = SITE_TO_LOCATION_PATTERNS[cancer_site]
            # Convert ILIKE patterns to regex: %oral% -> oral
            regex_patterns = [p.replace('%', '') for p in patterns]
            regex_combined = '|'.join(regex_patterns)
            
            condition = f"cancer_location ~* ${param_idx}"
            params.append(regex_combined)
            param_idx += 1
            
            weight = dynamic_weights.get("cancer_site", 0)
            where_conditions.append(condition)
            score_cases.append(("cancer_site", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"cancer_site={cancer_site}")
        
        # 2. Site detail - bonus points
        site_detail = cancer.get("site_detail")
        if site_detail:
            condition = f"cancer_location ILIKE ${param_idx}"
            params.append(f"%{site_detail.replace('_', ' ')}%")
            param_idx += 1
            
            weight = dynamic_weights.get("site_detail", 0)
            where_conditions.append(condition)
            score_cases.append(("site_detail", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"site_detail={site_detail}")
        
        # 3. Histology matching
        histology = cancer.get("histology")
        if histology and histology in HISTOLOGY_TO_PATTERNS:
            patterns = HISTOLOGY_TO_PATTERNS[histology]
            regex_patterns = [p.replace('%', '') for p in patterns]
            regex_combined = '|'.join(regex_patterns)
            
            condition = f"(cancer_type ~* ${param_idx} OR histopathologic_type ~* ${param_idx})"
            params.append(regex_combined)
            param_idx += 1
            
            weight = dynamic_weights.get("histology", 0)
            where_conditions.append(condition)
            score_cases.append(("histology", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"histology={histology}")
        
        # 4. Stage matching - use expanded patterns if available
        stage = cancer.get("stage")
        if stage:
            # Use expanded stage patterns if available, otherwise simple match
            if stage.upper() in STAGE_TO_PATTERNS:
                patterns = STAGE_TO_PATTERNS[stage.upper()]
                # Patterns are already regex-ready
                regex_combined = '|'.join(patterns)
                condition = f"extraction_data::text ~* ${param_idx}"
                params.append(regex_combined)
            else:
                condition = f"extraction_data::text ~* ${param_idx}"
                params.append(f"stage {stage}\\b")
            param_idx += 1
            
            weight = dynamic_weights.get("stage", 0)
            where_conditions.append(condition)
            score_cases.append(("stage", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"stage={stage}")
        
        # 4b. Disease descriptor matching (locally advanced, metastatic, etc.)
        disease_desc = cancer.get("disease_descriptor")
        if disease_desc:
            desc_patterns = {
                "locally advanced": "locally advanced|locoregionally advanced",
                "metastatic": "metastatic|metastases|distant metast",
                "advanced": "advanced|late.stage",
                "early stage": "early.stage|early",
                "oligometastatic": "oligometast",
            }
            pattern = desc_patterns.get(disease_desc, re.escape(disease_desc))
            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(pattern)
            param_idx += 1
            
            weight = dynamic_weights.get("disease_descriptor", 0)
            where_conditions.append(condition)
            score_cases.append(("disease_descriptor", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"descriptor={disease_desc}")

        # 4c. Trajectory scoring against extraction_data->>'disease_trajectory'.
        # Bonus scoring folded into the disease_descriptor axis weight.
        # When disease_descriptor is already scored above, trajectory adds
        # a secondary match path. When only trajectory is present, it
        # provides the full disease_descriptor contribution.
        if _trajectory_value:
            traj_weight = dynamic_weights.get("disease_descriptor", 0)
            if traj_weight > 0:
                traj_condition = f"LOWER(TRIM(extraction_data->>'disease_trajectory')) = ${param_idx}"
                params.append(_trajectory_value.lower().strip())
                param_idx += 1
                # If disease_descriptor already emitted a score_case, trajectory
                # is additive (bonus). Otherwise it's the primary contributor.
                if not disease_desc:
                    score_cases.append(
                        ("disease_descriptor",
                         f"CASE WHEN {traj_condition} THEN {traj_weight} ELSE 0 END")
                    )
                else:
                    # Bonus: half-weight for trajectory match on top of descriptor
                    bonus = round(traj_weight * 0.5, 1)
                    score_cases.append(
                        ("disease_trajectory_bonus",
                         f"CASE WHEN {traj_condition} THEN {bonus} ELSE 0 END")
                    )
                condition_names.append(f"trajectory={_trajectory_value}")
        
        # 5. TNM T stage — range-aware regex matching.
        #
        # A patient with T3 should match studies that say "T3", "T2-3",
        # "T2-T3", "T1-4", "T2-4", "cT3", "pT3", etc. The old literal
        # ILIKE '%T3%' missed range notations like "T2-3" (which contains
        # "2-3" but not literally "T3"). We build a regex that catches
        # the exact value AND common range notations that include it.
        tnm_t = cancer.get("tnm_t")
        if tnm_t:
            t_val = str(tnm_t).strip().lower().lstrip("t")
            t_num = re.match(r'(\d)', t_val)
            if t_num:
                tv = int(t_num.group(1))
                # Match: exact T{v}, and any range T{a}-{b} where a <= v <= b
                range_patterns = [f"T{tv}", f"pT{tv}", f"cT{tv}", f"ypT{tv}"]
                for low in range(0, tv + 1):
                    for high in range(tv, 5):
                        if low < high:
                            range_patterns.append(f"T{low}-{high}")
                            range_patterns.append(f"T{low}-T{high}")
                regex = "|".join(re.escape(p) for p in range_patterns)
            else:
                regex = f"T{re.escape(t_val)}"

            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(regex)
            param_idx += 1

            weight = dynamic_weights.get("tnm_t", 0)
            where_conditions.append(condition)
            score_cases.append(("tnm_t", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"tnm_t=T{tnm_t} (range-aware)")

        # 6. TNM N stage — range-aware
        tnm_n = cancer.get("tnm_n")
        if tnm_n:
            n_val = str(tnm_n).strip().lower().lstrip("n")
            n_num = re.match(r'(\d)', n_val)
            if n_num:
                nv = int(n_num.group(1))
                range_patterns = [f"N{nv}", f"pN{nv}", f"cN{nv}", f"ypN{nv}"]
                for low in range(0, nv + 1):
                    for high in range(nv, 4):
                        if low < high:
                            range_patterns.append(f"N{low}-{high}")
                            range_patterns.append(f"N{low}-N{high}")
                # N+ means N1 or above
                if nv >= 1:
                    range_patterns.append("N\\+")
                    range_patterns.append("node positive")
                regex = "|".join(re.escape(p).replace("\\\\\\+", "\\+") for p in range_patterns)
            else:
                regex = f"N{re.escape(n_val)}"

            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(regex)
            param_idx += 1

            weight = dynamic_weights.get("tnm_n", 0)
            where_conditions.append(condition)
            score_cases.append(("tnm_n", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"tnm_n=N{tnm_n} (range-aware)")

        # 6b. TNM M stage — simple (M0 or M1, no ranges needed)
        tnm_m = cancer.get("tnm_m")
        if tnm_m:
            m_val = str(tnm_m).strip().lower().lstrip("m")
            range_patterns = [f"M{m_val}", f"pM{m_val}", f"cM{m_val}"]
            if m_val == "1":
                range_patterns.extend(["metastatic", "distant metastasis", "M1"])
            elif m_val == "0":
                range_patterns.extend(["non-metastatic", "no distant", "M0"])
            regex = "|".join(re.escape(p) for p in range_patterns)

            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(regex)
            param_idx += 1

            weight = dynamic_weights.get("tnm_m", 0)
            where_conditions.append(condition)
            score_cases.append(("tnm_m", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"tnm_m=M{tnm_m}")

        # 7. Treatment modality
        modality = treatment.get("modality")
        if modality and modality in TREATMENT_TO_PATTERNS:
            patterns = TREATMENT_TO_PATTERNS[modality]
            regex_patterns = [p.replace('%', '') for p in patterns]
            regex_combined = '|'.join(regex_patterns)
            
            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(regex_combined)
            param_idx += 1
            
            weight = dynamic_weights.get("treatment", 0)
            where_conditions.append(condition)
            score_cases.append(("treatment", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"treatment={modality}")
        
        # 7b. Treatment setting (neoadjuvant, adjuvant, etc.)
        treatment_setting = treatment.get("setting")
        if treatment_setting:
            setting_patterns = {
                "neoadjuvant": "neoadjuvant|preoperative|pre-operative|induction|NAC",
                "adjuvant": "adjuvant|postoperative|post-operative",
                "definitive": "definitive|concurrent|chemoradiation",
                "palliative": "palliative|best supportive|comfort",
                "concurrent": "concurrent|chemoradiation|chemoradiotherapy|CRT",
            }
            setting_regex = setting_patterns.get(treatment_setting, re.escape(treatment_setting))
            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(setting_regex)
            param_idx += 1

            weight = dynamic_weights.get("treatment_setting", 0)
            where_conditions.append(condition)
            score_cases.append(("treatment_setting", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"setting={treatment_setting}")

        # 8. Age range matching — range-aware.
        #
        # A 68-year-old patient should match studies with "age ≥18",
        # "age 18-75", "age 20-76", "≥65", etc. The old decade-only
        # ILIKE '%60%' missed studies that said "18-75" or "≥18".
        # Now builds a regex that catches the exact age, the decade,
        # and common range notations.
        patient_age = patient.get("age")
        if patient_age:
            age = int(patient_age)
            age_decade = (age // 10) * 10
            # Build regex patterns for range matching
            age_patterns = [
                str(age),                    # exact age
                str(age_decade),             # decade
                f"≥\\s*{age}",              # ≥68
                f">=\\s*{age}",             # >=68
            ]
            # Ranges that include this age: "X-Y" where X ≤ age ≤ Y
            # Common lower bounds: 18, 20, 21
            # Common upper bounds: 70, 75, 80, 85, 99
            for low in [18, 20, 21, 30, 40, 50, 60, 65]:
                if low <= age:
                    age_patterns.append(f"{low}\\s*[-–]")  # "18-" or "65-"
            # Match "≥18" style (patient qualifies for any lower bound ≤ age)
            for threshold in [18, 20, 21, 40, 50, 60, 65, 70, 75]:
                if threshold <= age:
                    age_patterns.append(f"≥\\s*{threshold}")
                    age_patterns.append(f">=\\s*{threshold}")

            age_regex = "|".join(age_patterns)
            condition = f"(age_range ~* ${param_idx} OR median_age ~* ${param_idx} OR extraction_data::text ~* ${param_idx})"
            params.append(age_regex)
            param_idx += 1
            
            weight = dynamic_weights.get("age_range", 0)
            where_conditions.append(condition)
            score_cases.append(("age_range", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"age~{patient_age}")
        
        # 9. Gender matching
        patient_gender = patient.get("gender")
        if patient_gender:
            # Match gender in extraction_data (patient characteristics, eligibility, etc.)
            gender_synonyms = {
                "male": "male|men|man",
                "female": "female|women|woman",
            }
            gender_pattern = gender_synonyms.get(patient_gender.lower(), re.escape(patient_gender))
            condition = f"extraction_data::text ~* ${param_idx}"
            params.append(gender_pattern)
            param_idx += 1

            weight = dynamic_weights.get("gender", 0)
            where_conditions.append(condition)
            score_cases.append(("gender", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"gender={patient_gender}")

        # 10. Performance status — range-aware.
        #
        # A patient with ECOG 1 should match studies that say "ECOG 0-1",
        # "ECOG 0-2", "PS ≤ 2", "ECOG ≤ 1", etc. The old literal match
        # only caught exact strings.
        perf_status = patient.get("performance_status")
        if perf_status:
            # Extract the numeric value from "ECOG 1", "PS 0", "1", etc.
            ps_num_match = re.search(r'(\d)', str(perf_status))
            if ps_num_match:
                ps_val = int(ps_num_match.group(1))
                ps_patterns = [
                    f"ECOG {ps_val}",
                    f"PS {ps_val}",
                    f"ECOG PS {ps_val}",
                ]
                # Ranges that include this PS value
                for low in range(0, ps_val + 1):
                    for high in range(ps_val, 6):
                        if low < high:
                            ps_patterns.append(f"{low}\\s*[-–]\\s*{high}")
                # ≤ thresholds
                for threshold in range(ps_val, 6):
                    ps_patterns.append(f"≤\\s*{threshold}")
                    ps_patterns.append(f"<=\\s*{threshold}")
                ps_regex = "|".join(ps_patterns)
                condition = f"(performance_status ~* ${param_idx} OR extraction_data::text ~* ${param_idx})"
                params.append(ps_regex)
            else:
                condition = f"performance_status ILIKE ${param_idx}"
                params.append(f"%{perf_status.split()[0]}%")
            param_idx += 1

            weight = dynamic_weights.get("performance_status", 0)
            where_conditions.append(condition)
            score_cases.append(("performance_status", f"CASE WHEN {condition} THEN {weight} ELSE 0 END"))
            condition_names.append(f"perf_status={perf_status} (range-aware)")
        
        # 10. Biomarker matching — fractional per-marker match.
        #
        #   Given N patient biomarkers, for each study compute:
        #     matches  = # of patient biomarkers reported in the study
        #                with a status that matches (polarity-equivalent)
        #     overlap  = # of patient biomarkers reported in the study at all
        #     score    = (matches / overlap) * weight      if overlap > 0
        #                0                                 if overlap == 0
        #
        #   Rationale: a study that is SILENT on a patient biomarker is
        #   treated as NEUTRAL, not as a mismatch. Denominator excludes
        #   silent biomarkers so a study that reports 2 markers (both
        #   matching) scores the same as a study that reports 5 (all
        #   matching). This matches the clinical intent: trials don't
        #   typically enumerate every biomarker, so silence isn't exclusion.
        #
        #   Safety-net: the biomarker `CONTRA_MAP` hard exclusion still
        #   applies below — that's the mechanism for "patient is HER2-, do
        #   NOT return trastuzumab studies" etc. The fractional score is
        #   additive on top of hard exclusion.
        #
        # Include both biomarkers list AND receptor_status (ER/PR/HER2)
        biomarkers = list(cancer.get("biomarkers", []))

        # Also include receptor_status (e.g. "ER+/PR+/HER2-") as individual biomarkers
        receptor_status = cancer.get("receptor_status", "")
        if receptor_status and receptor_status != "triple negative":
            for part in receptor_status.split("/"):
                part = part.strip()
                if part and part not in biomarkers:
                    biomarkers.append(part)
        elif receptor_status == "triple negative":
            biomarkers.append("triple negative")

        if biomarkers:
            # Parse each marker into (canonical, status). Deduplicate by
            # canonical name so "ER+, ER positive" isn't double-counted.
            parsed_markers: List[tuple] = []
            seen_canonicals: Set[str] = set()
            for marker in biomarkers:
                canonical, status = _parse_biomarker_query(marker)
                if canonical and canonical not in seen_canonicals:
                    parsed_markers.append((canonical, status))
                    seen_canonicals.add(canonical)

            # Only polarized markers contribute to the fractional score.
            # Unpolarized markers still get a key-existence WHERE-clause
            # entry so studies that report the marker at all are eligible.
            polarized = [(c, s) for c, s in parsed_markers if s]
            unpolarized = [c for c, s in parsed_markers if not s]

            weight = dynamic_weights.get("biomarkers", 0)

            if polarized and weight > 0:
                match_terms: List[str] = []    # numerator CASE WHENs
                reported_terms: List[str] = [] # denominator CASE WHENs

                for canonical, status in polarized:
                    # Alias-aware value match: OR across all known JSONB
                    # key spellings (HER2 / HER-2 / ERBB2 / ...) AND
                    # across both the IN-list of status synonyms and
                    # the status regex fallback (covers IHC 2+/3+,
                    # percent staining, etc.).
                    value_sql, param_idx = _build_alias_value_match_sql(
                        canonical, status, params, param_idx
                    )
                    match_terms.append(
                        f"CASE WHEN {value_sql} THEN 1 ELSE 0 END"
                    )
                    # Denominator: alias-aware key-existence check.
                    key_sql, param_idx = _build_alias_key_exists_sql(
                        canonical, params, param_idx
                    )
                    reported_terms.append(
                        f"CASE WHEN {key_sql} THEN 1 ELSE 0 END"
                    )

                if match_terms:
                    numerator = " + ".join(match_terms)
                    denominator = " + ".join(reported_terms)
                    # Fractional match score: (matches / overlap) * weight
                    # COALESCE handles the zero-overlap case → 0 points
                    score_expr = (
                        f"COALESCE((({numerator})::float "
                        f"/ NULLIF({denominator}, 0)) * {weight}, 0)"
                    )
                    score_cases.append(("biomarkers", score_expr))
                    condition_names.append(
                        f"biomarkers=frac({len(polarized)})"
                    )

            # WHERE clause: a study is eligible if it reports ANY of the
            # patient's biomarkers (polarized or unpolarized). This keeps
            # biomarkers a "contributing" OR condition when there's no
            # hard cancer_site filter. Uses the same alias-aware
            # key-existence check as the denominator so heterogeneous
            # JSONB spellings all match.
            #
            # IMPORTANT: skip this WHERE clause entirely when cancer_site
            # is set, because the hard-filter pruning below will drop it
            # anyway and we'd be emitting dead parameters otherwise.
            has_hard_site_filter = bool(
                cancer.get("site")
                and cancer.get("site") in SITE_TO_LOCATION_PATTERNS
            )
            if not has_hard_site_filter:
                or_parts: List[str] = []
                for canonical, _ in parsed_markers:
                    key_sql, param_idx = _build_alias_key_exists_sql(
                        canonical, params, param_idx
                    )
                    if key_sql != "FALSE":
                        or_parts.append(key_sql)
                if or_parts:
                    where_conditions.append("(" + " OR ".join(or_parts) + ")")
                    if not polarized:
                        # Unpolarized-only biomarkers still count as a
                        # present criterion so the where-clause OR
                        # includes them, but they contribute 0 scoring
                        # points.
                        condition_names.append(
                            f"biomarkers=unpolarized({len(unpolarized)})"
                        )

        # 11. Class 3a — metastatic site fractional match.
        # Pure SCORING contribution (no where_conditions entry), so the
        # hard cancer_site filter still governs eligibility. A study that
        # mentions more of the patient's metastatic sites gets a
        # proportional boost.
        if _met_sites_input:
            met_weight = dynamic_weights.get("metastatic_sites", 0)
            met_score_expr, param_idx = _build_metastatic_site_match_sql(
                _met_sites_input, met_weight, params, param_idx
            )
            if met_score_expr is not None:
                score_cases.append(("metastatic_sites", met_score_expr))
                condition_names.append(
                    f"metastatic_sites=frac({len(_met_sites_input)})"
                )

        # 12. Class 3b — title keyword relevance (additive tiebreaker).
        # Also a pure SCORING contribution with no where_conditions entry,
        # so cryptic titles (e.g. "ECOG-ACRIN EA1131") are never hard-
        # excluded — they just score 0 on this component.
        if _title_kw_probe:
            title_weight = dynamic_weights.get("title_relevance", 0)
            title_kws = _extract_title_keywords(
                cancer, treatment, _met_sites_input
            )
            if title_kws and title_weight > 0:
                title_score_expr, param_idx = _build_title_relevance_sql(
                    title_kws, title_weight, params, param_idx
                )
                if title_score_expr is not None:
                    score_cases.append(("title_relevance", title_score_expr))
                    condition_names.append(
                        f"title_relevance=frac({len(title_kws)})"
                    )

        # 13. Metastatic status scoring — exact match against studies.metastatic_status.
        # Pure SCORING contribution (no where_conditions entry).
        if _metastatic_status_value:
            met_status_weight = dynamic_weights.get("metastatic_status", 0)
            if met_status_weight > 0:
                condition = f"LOWER(TRIM(metastatic_status)) = ${param_idx}"
                params.append(_metastatic_status_value.lower().strip())
                param_idx += 1
                score_cases.append(
                    ("metastatic_status", f"CASE WHEN {condition} THEN {met_status_weight} ELSE 0 END")
                )
                condition_names.append(f"metastatic_status={_metastatic_status_value}")

        # 14. Risk stratification scoring — exact match against studies.risk_stratification.
        # Pure SCORING contribution (no where_conditions entry).
        if _risk_strat_value:
            risk_strat_weight = dynamic_weights.get("risk_stratification", 0)
            if risk_strat_weight > 0:
                condition = f"LOWER(TRIM(risk_stratification)) = ${param_idx}"
                params.append(_risk_strat_value.lower().strip())
                param_idx += 1
                score_cases.append(
                    ("risk_stratification", f"CASE WHEN {condition} THEN {risk_strat_weight} ELSE 0 END")
                )
                condition_names.append(f"risk_stratification={_risk_strat_value}")

        # 15. Biomarker JSONB scoring — polarity-aware fractional match
        #     against studies.biomarker_status using JSONB containment
        #     operators. Separate from the existing "biomarkers" axis
        #     (which uses string matching on extraction_data). This axis
        #     uses _build_alias_key_exists_sql() and
        #     _build_alias_value_match_sql() for alias-aware JSONB lookups
        #     with STATUS_MATCH_SYNONYMS polarity expansion.
        #
        #   Formula:
        #     score = (Σ CASE WHEN value_match THEN 1 ELSE 0 END)
        #             / total_patient_biomarkers * weight
        #
        #   Pure SCORING contribution (no where_conditions entry).
        if biomarkers:
            # Re-parse biomarkers (same logic as existing biomarker section)
            _jsonb_parsed: List[tuple] = []
            _jsonb_seen: Set[str] = set()
            for marker in biomarkers:
                canonical, status = _parse_biomarker_query(marker)
                if canonical and canonical not in _jsonb_seen:
                    _jsonb_parsed.append((canonical, status))
                    _jsonb_seen.add(canonical)

            _jsonb_polarized = [(c, s) for c, s in _jsonb_parsed if s]

            jsonb_weight = dynamic_weights.get("biomarker_jsonb", 0)

            if _jsonb_polarized and jsonb_weight > 0:
                jsonb_match_terms: List[str] = []
                for canonical, status in _jsonb_polarized:
                    value_sql, param_idx = _build_alias_value_match_sql(
                        canonical, status, params, param_idx
                    )
                    jsonb_match_terms.append(
                        f"CASE WHEN {value_sql} THEN 1 ELSE 0 END"
                    )

                if jsonb_match_terms:
                    numerator = " + ".join(jsonb_match_terms)
                    total = len(_jsonb_polarized)
                    score_expr = (
                        f"COALESCE((({numerator})::float / {total}) * {jsonb_weight}, 0)"
                    )
                    score_cases.append(("biomarker_jsonb", score_expr))
                    condition_names.append(
                        f"biomarker_jsonb=frac({len(_jsonb_polarized)})"
                    )

        # If no conditions, return empty result
        if not where_conditions:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return StructuredMatchResult(
                doc_ids=set(),
                match_scores={},
                match_details={},
                query_time_ms=elapsed_ms,
                conditions_used=[]
            )
        
        # Build the query with weighted scoring
        # IMPORTANT: Use cancer_site as a hard AND filter when present.
        # Other conditions are OR-ed for scoring but site must match.
        # score_cases is List[Tuple[str, str]] — (axis_name, sql_expr)
        score_expression = " + ".join(expr for _, expr in score_cases)

        # ── Hard biomarker exclusion filter ──────────────────────────────
        # When the patient has an explicit biomarker status, exclude studies
        # whose biomarker_status or study text clearly indicates the
        # OPPOSITE.  This applies across ALL cancer types — not just breast.
        #
        # Each entry: patient_marker → (JSONB key, contradictory JSONB
        # status, optional text regex that catches targeted-drug study
        # names so they are hard-filtered even when biomarker_status JSONB
        # is not populated).
        biomarker_exclusions = []
        _CONTRA_MAP = {
            # ── Breast receptors ────────────────────────────────────────
            "HER2-":  ("HER2", "positive",
                       r"her.?2.*(positive|amplified|overexpress|\+)"
                       r"|trastuzumab|pertuzumab|t-dm1|t-dxd|ado-trastuzumab"
                       r"|tucatinib|neratinib|lapatinib|margetuximab"),
            "HER2+":  ("HER2", "negative", None),
            "HER2 amplified": ("HER2", "negative", None),  # same as HER2+
            "ER-":    ("ER", "positive",
                       r"\ber.?(positive|\+).*(?:tamoxifen|letrozole|anastrozole|fulvestrant|exemestane)"
                       r"|endocrine.therap|hormonal.therap|aromatase.inhibitor"),
            "ER+":    ("ER", "negative", None),
            "PR-":    ("PR", "positive", None),
            "PR+":    ("PR", "negative", None),
            "triple negative": ("TNBC", None,  # no JSONB check — use text only
                       r"her.?2.*(positive|amplified|overexpress|\+)"
                       r"|trastuzumab|pertuzumab|t-dm1|t-dxd|ado-trastuzumab"
                       r"|tucatinib|neratinib|lapatinib|margetuximab"),

            # ── EGFR (lung, pan-cancer) ─────────────────────────────────
            "EGFR mutant":    ("EGFR", "wild-type", None),
            "EGFR wild-type": ("EGFR", "mutant",
                       r"osimertinib|erlotinib|gefitinib|afatinib|dacomitinib"
                       r"|lazertinib|amivantamab.*egfr|mobocertinib"
                       r"|egfr.*(mutant|mutation|del.?19|l858r|exon)"),

            # ── ALK (lung) ──────────────────────────────────────────────
            "ALK+":   ("ALK", "negative", None),
            "ALK-":   ("ALK", "positive",
                       r"crizotinib|alectinib|ceritinib|brigatinib|lorlatinib"
                       r"|alk.*(positive|rearrang|fusion|translocation|\+)"),

            # ── ROS1 (lung) ─────────────────────────────────────────────
            "ROS1+":  ("ROS1", "negative", None),

            # ── KRAS (lung, CRC, pancreas) ──────────────────────────────
            "KRAS mutant":    ("KRAS", "wild-type", None),
            "KRAS wild-type": ("KRAS", "mutant",
                       r"sotorasib|adagrasib|kras.*g12c"
                       r"|kras.*(mutant|mutation|mutated)"),

            # ── BRAF (melanoma, CRC, lung, thyroid) ─────────────────────
            "BRAF mutant":    ("BRAF", "wild-type", None),
            "BRAF wild-type": ("BRAF", "mutant",
                       r"vemurafenib|dabrafenib|encorafenib"
                       r"|braf.*(v600|mutant|mutation|mutated)"),

            # ── RET (lung, thyroid) ─────────────────────────────────────
            "RET+":   ("RET", "negative",  None),

            # ── MET (lung) ──────────────────────────────────────────────
            "MET altered": ("MET", "negative",  None),

            # ── NTRK (pan-cancer) ───────────────────────────────────────
            "NTRK fusion": ("NTRK", "negative", None),

            # ── BRCA (breast, ovarian, prostate, pancreas) ──────────────
            "BRCA mutant":    ("BRCA", "wild-type", None),
            "BRCA wild-type": ("BRCA", "mutant",
                       r"olaparib|rucaparib|niraparib|talazoparib|veliparib"
                       r"|parp.inhibitor|brca.*(mutant|mutation|pathogenic)"),

            # ── MSI / MMR (CRC, endometrial, gastric, pan-cancer) ──────
            "MSI-H":  ("MSI", "stable",   None),
            "dMMR":   ("MSI", "stable",   None),    # dMMR ≈ MSI-H
            "MSS":    ("MSI", "high",
                       r"msi.?h|microsatellite.instability.high"
                       r"|dmmr|mismatch.repair.deficient"),
            "pMMR":   ("MSI", "high",     None),    # pMMR ≈ MSS

            # ── IDH (glioma) ────────────────────────────────────────────
            "IDH mutant":    ("IDH", "wild-type", None),
            "IDH wild-type": ("IDH", "mutant",
                       r"vorasidenib|ivosidenib|enasidenib|idh.*(mutant|mutation)"),

            # ── MGMT (glioblastoma) ─────────────────────────────────────
            "MGMT methylated":   ("MGMT", "unmethylated", None),
            "MGMT unmethylated": ("MGMT", "methylated", None),

            # ── HPV / p16 (head & neck) ─────────────────────────────────
            "HPV+":   ("HPV", "negative", None),
            "HPV-":   ("HPV", "positive", None),

            # ── PIK3CA (breast) ─────────────────────────────────────────
            "PIK3CA mutant": ("PIK3CA", "wild-type", None),

            # ── FGFR (bladder, cholangiocarcinoma) ──────────────────────
            "FGFR altered":  ("FGFR", "wild-type", None),

            # ── PD-L1 expression ────────────────────────────────────────
            # Not exclusionary — PD-L1 low patients may still receive ICI
            # in combinations, so no hard filter here.
        }

        for marker in biomarkers:
            contra = _CONTRA_MAP.get(marker.strip())
            if not contra:
                continue
            bm_key, contra_status, text_regex = contra
            # Alias-aware exclude: a study is dropped if ANY alias of the
            # contra biomarker key carries a contradictory status. Uses the
            # same value-match helper as the scoring code, so synonyms +
            # IHC regex both fire here.
            #
            # Pre-fix behaviour: this check only looked at the literal
            # canonical key (e.g. 'HER2'), missing studies where ingestion
            # stored the status under 'HER-2' or 'ERBB2'.
            if contra_status:
                value_sql, param_idx = _build_alias_value_match_sql(
                    bm_key, contra_status, params, param_idx
                )
                # COALESCE wraps the inner expression so NULL (from
                # studies that don't report this biomarker at all)
                # collapses to FALSE before NOT inverts it. Without
                # this wrapper, three-valued logic produced
                # NULL → NOT NULL → FALSE in WHERE, which
                # silently rejected every study with a sparse
                # biomarker_status JSONB. See plan file CURRENT TASK
                # "Fix CONTRA-exclusion NULL regression".
                excl = f"(NOT COALESCE({value_sql}, FALSE))"
                biomarker_exclusions.append(excl)
            # Also exclude studies whose study_name/cancer_type text matches targeted drugs
            if text_regex:
                excl_text = f"(NOT (study_name ~* ${param_idx} OR cancer_type ~* ${param_idx}))"
                params.append(text_regex)
                param_idx += 1
                biomarker_exclusions.append(excl_text)
                print(f"    [PG Exclusion] Hard-excluding: {marker} → /{text_regex[:60]}.../")

        if biomarker_exclusions:
            condition_names.append(f"biomarker_exclusions={len(biomarker_exclusions)}")
            print(f"    [PG Exclusion] {len(biomarker_exclusions)} biomarker exclusion clauses applied")

        # Separate site condition (hard filter) from scoring conditions
        hard_filter = None
        scoring_conditions = []
        for i, cond in enumerate(where_conditions):
            cond_name = condition_names[i] if i < len(condition_names) else ""
            if cond_name.startswith("cancer_site="):
                hard_filter = cond
            else:
                scoring_conditions.append(cond)

        # Build WHERE clause: site is mandatory AND, others are OR for scoring
        # When site is present, ALL results must match the site.
        # Other conditions contribute to the score but are not required.
        if hard_filter:
            where_clause = hard_filter
        elif scoring_conditions:
            where_clause = " OR ".join(scoring_conditions)
        else:
            where_clause = " OR ".join(where_conditions)

        # Append biomarker exclusion clauses as hard AND filters
        if biomarker_exclusions:
            where_clause = f"({where_clause}) AND {' AND '.join(biomarker_exclusions)}"
        
        print(f"    [PG Query] Hard site filter: {hard_filter is not None}")
        print(f"    [PG Query] Scoring conditions ({len(scoring_conditions)}): {condition_names}")
        print(f"    [PG Query] Score expression components: {len(score_cases)}")
        
        # Build per-axis SELECT columns for PGMatchBreakdown
        axis_select_columns = []
        for axis_name, expr in score_cases:
            col_alias = f"axis_{axis_name}"
            axis_select_columns.append(f"({expr}) as {col_alias}")

        axis_columns_sql = ""
        if axis_select_columns:
            axis_columns_sql = ",\n                " + ",\n                ".join(axis_select_columns)

        query = f"""
            SELECT 
                doc_id,
                study_name,
                cancer_location,
                cancer_type,
                number_of_patients,
                ({score_expression}) as match_score{axis_columns_sql}
            FROM studies
            WHERE doc_id IS NOT NULL
              AND ({where_clause})
            ORDER BY match_score DESC, number_of_patients DESC NULLS LAST
            LIMIT {limit}
        """
        
        rows = await conn.fetch(query, *params)
        print(f"    [PG Results] Rows returned: {len(rows)}")

        # Build result with detailed scoring
        # With dynamic weights, max_possible_score is always 100
        max_possible_score = 100
        doc_ids = set()
        match_scores = {}
        match_details = {}
        
        for row in rows:
            doc_id = row['doc_id']
            raw_score = row['match_score'] or 0
            
            doc_ids.add(doc_id)
            # Normalize score to 0-1 (raw_score is out of 100 with dynamic weights)
            normalized_score = raw_score / max_possible_score
            match_scores[doc_id] = normalized_score

            # Build per-axis AxisContributions from the per-axis columns
            axis_contributions = []
            axis_mismatches = []
            for axis_name, _ in score_cases:
                col_alias = f"axis_{axis_name}"
                try:
                    earned = float(row[col_alias] or 0)
                except (KeyError, TypeError):
                    earned = 0.0
                max_pts = float(dynamic_weights.get(axis_name, 0))
                if max_pts <= 0:
                    continue
                # Determine label
                if earned >= max_pts:
                    label = "exact match"
                elif earned > 0:
                    label = "partial match"
                else:
                    label = "not reported"
                axis_contributions.append(AxisContribution(
                    axis_name=axis_name,
                    points_earned=earned,
                    max_points=max_pts,
                    label=label,
                ))
                if earned == 0:
                    axis_mismatches.append(axis_name)

            breakdown = PGMatchBreakdown(
                total_score=raw_score,
                axis_contributions=axis_contributions,
                axis_mismatches=axis_mismatches,
            )

            match_details[doc_id] = {
                "raw_score": raw_score,
                "max_possible": max_possible_score,
                "normalized_score": normalized_score,
                "study_name": row['study_name'],
                "cancer_location": row['cancer_location'],
                "cancer_type": row['cancer_type'],
                "num_patients": row['number_of_patients'],
                "matched_criteria": condition_names,
                "dynamic_weights": dynamic_weights,
                "match_breakdown": breakdown.to_dict(),
            }
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Log top matches
        if doc_ids:
            top_score = max(match_scores.values()) if match_scores else 0
            print(f"    [PG Results] Found {len(doc_ids)} studies in {elapsed_ms:.1f}ms")
            print(f"    [PG Results] Top score: {top_score:.0%} (max possible: {max_possible_score})")
            # Show top 5 matches
            sorted_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for rank, (did, score) in enumerate(sorted_matches, 1):
                detail = match_details.get(did, {})
                # `.get(key, default)` returns the stored value when the key
                # exists, even if that value is None — so we can't rely on
                # the default to feed [:N] slicing. Coalesce explicitly.
                study_name = (detail.get('study_name') or did or '?')
                cancer_loc = (detail.get('cancer_location') or '?')
                num_patients = detail.get('num_patients')
                print(f"      {rank}. [{score:.0%}] {study_name[:50]}... "
                      f"(site={cancer_loc[:20]}, "
                      f"n={num_patients if num_patients is not None else '?'})")
        else:
            print(f"    [PG Results] No matching studies found in {elapsed_ms:.1f}ms")
        
        return StructuredMatchResult(
            doc_ids=doc_ids,
            match_scores=match_scores,
            match_details=match_details,
            query_time_ms=elapsed_ms,
            conditions_used=condition_names,
            max_possible_score=max_possible_score
        )
        
    except Exception as e:
        print(f"[StructuredMatcher] Error: {e}")
        import traceback
        traceback.print_exc()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return StructuredMatchResult(
            doc_ids=set(),
            match_scores={},
            match_details={},
            query_time_ms=elapsed_ms,
            conditions_used=[]
        )
    finally:
        if conn:
            await conn.close()


def boost_candidates_with_structured_matches(
    candidates: List[Dict[str, Any]],
    structured_result: StructuredMatchResult,
    boost_factor: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Boost Qdrant candidates that also matched in PostgreSQL structured search.
    
    Args:
        candidates: List of candidate chunks from Qdrant
        structured_result: Result from match_studies_by_structure()
        boost_factor: Maximum boost to apply (0.3 = 30% boost for perfect match)
        
    Returns:
        Candidates with boosted scores for structured matches
    """
    if not structured_result.doc_ids:
        return candidates
    
    boosted_count = 0
    
    for cand in candidates:
        payload = cand.get('payload', {})
        doc_id = payload.get('doc_id')
        
        if doc_id and doc_id in structured_result.doc_ids:
            # Get match score (0-1) and apply proportional boost
            match_score = structured_result.match_scores.get(doc_id, 0.5)
            boost = boost_factor * match_score
            
            # Apply boost to dense score
            if 'score_dense' in cand:
                cand['score_dense'] = cand['score_dense'] * (1 + boost)
                cand['_structured_boost'] = boost
                cand['_structured_match_score'] = match_score
                boosted_count += 1
    
    if boosted_count > 0:
        print(f"[StructuredMatcher] Boosted {boosted_count} candidates from PostgreSQL matches")
    
    return candidates


async def match_studies_by_structure_simple(
    cancer_site: Optional[str] = None,
    histology: Optional[str] = None,
    treatment_modality: Optional[str] = None,
    limit: int = 50
) -> Set[str]:
    """
    Simplified interface for structured matching.
    Returns just the set of matching doc_ids.
    """
    query_structure = {
        "cancer": {
            "site": cancer_site,
            "histology": histology,
        },
        "treatment": {
            "modality": treatment_modality,
        }
    }
    
    result = await match_studies_by_structure(query_structure, limit)
    return result.doc_ids
