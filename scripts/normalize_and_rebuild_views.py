#!/usr/bin/env python3
"""
Comprehensive Normalization & Materialized View Rebuild

Pulls canonical term→synonym mappings from the project ontology/vocabulary
to build robust SQL normalization functions, then rebuilds both MVs.

Sources:
  - oncology_clinical_trial_vocabulary_list.rtf → {term, synonyms} pairs
  - final_merged_canonical_oncology_ct_ontology.rtf → phase list

Usage:
    python scripts/normalize_and_rebuild_views.py
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════
# CANONICAL MAPPINGS (from vocabulary list)
# ══════════════════════════════════════════════════════════════════

# Phase mappings from vocabulary + ontology
# canonical_term → [synonyms]
PHASE_MAP = {
    "Phase I":     ["Phase 1", "Phase I study", "first-in-human", "phase 1 trial", "phase I trial"],
    "Phase I/II":  ["Phase 1/2", "Phase I-II", "Phase 1-2", "Phase I and II", "Phase 1 / 2"],
    "Phase II":    ["Phase 2", "Phase II study", "Phase IIA", "Phase IIB", "Phase IIR",
                    "Phase IIR/III", "Phase II and Phase III", "Phase II-III"],
    "Phase II/III":["Phase 2/3", "Phase 2-3", "Phase 2 / 3", "Phase II/III",
                    "Phase II and III"],
    "Phase III":   ["Phase 3", "Phase III study", "Phase 3 trial", "phase III trial"],
    "Phase IV":    ["Phase 4", "Phase IV study", "post-marketing study"],
}

# Cancer type mappings from vocabulary Disease Types section
CANCER_TYPE_MAP = {
    "Breast Cancer":                ["breast carcinoma", "invasive breast cancer", "breast adenocarcinoma",
                                     "ductal carcinoma", "lobular carcinoma", "DCIS", "breast neoplasm"],
    "Non-Small Cell Lung Cancer":   ["NSCLC", "non-small cell lung", "non-small-cell lung",
                                     "lung adenocarcinoma", "lung squamous cell"],
    "Small Cell Lung Cancer":       ["SCLC", "small cell lung", "small-cell lung"],
    "Glioblastoma":                 ["GBM", "glioblastoma multiforme"],
    "Melanoma":                     ["malignant melanoma", "cutaneous melanoma"],
    "Colorectal Cancer":            ["colon and rectal cancer", "CRC", "colon cancer", "rectal cancer",
                                     "rectal adenocarcinoma", "colon adenocarcinoma", "colorectal adenocarcinoma"],
    "Pancreatic Cancer":            ["pancreatic adenocarcinoma", "pancreatic carcinoma", "pancreas cancer"],
    "Hepatocellular Carcinoma":     ["HCC", "liver cancer", "hepatocellular cancer"],
    "Renal Cell Carcinoma":         ["RCC", "kidney cancer", "renal cancer"],
    "Bladder Cancer":               ["urothelial carcinoma", "bladder carcinoma", "urothelial cancer"],
    "Ovarian Cancer":               ["ovarian carcinoma", "ovarian neoplasm"],
    "Endometrial Cancer":           ["uterine cancer", "endometrial carcinoma", "uterine carcinoma"],
    "Cervical Cancer":              ["cervical carcinoma", "cervix cancer"],
    "Head and Neck Cancer":         ["head and neck squamous cell carcinoma", "HNSCC",
                                     "oropharyngeal cancer", "oropharyngeal carcinoma",
                                     "oral cavity cancer", "laryngeal cancer", "hypopharyngeal cancer"],
    "Nasopharyngeal Carcinoma":     ["NPC", "nasopharyngeal cancer", "nasopharynx cancer"],
    "Prostate Cancer":              ["prostate carcinoma", "prostate adenocarcinoma"],
    "Hodgkin Lymphoma":             ["Hodgkin disease", "Hodgkin's disease", "Hodgkin's lymphoma",
                                     "classical Hodgkin lymphoma"],
    "Non-Hodgkin Lymphoma":         ["NHL", "diffuse large B-cell lymphoma", "DLBCL",
                                     "follicular lymphoma", "mantle cell lymphoma"],
    "Brain Metastases":             ["brain mets", "cerebral metastases", "intracranial metastases"],
    "Rhabdomyosarcoma":             ["RMS", "rhabdomyosarcoma"],
    "Ewing Sarcoma":                ["Ewing's sarcoma", "Ewing tumor"],
    "Neuroblastoma":                [],
    "Wilms Tumor":                  ["nephroblastoma", "Wilm's tumor"],
    "Medulloblastoma":              [],
    "Meningioma":                   [],
    "Ependymoma":                   [],
    "Esophageal Cancer":            ["esophageal carcinoma", "esophageal adenocarcinoma",
                                     "esophageal squamous cell carcinoma", "oesophageal cancer"],
    "Gastric Cancer":               ["stomach cancer", "gastric carcinoma", "gastric adenocarcinoma"],
    "Basal Cell Carcinoma":         ["BCC"],
    "Merkel Cell Carcinoma":        [],
    "Osteosarcoma":                 ["osteogenic sarcoma"],
    "Pediatric ALL":                ["acute lymphoblastic leukemia"],
    "Multiple Myeloma":             ["myeloma"],
    "Sarcoma":                      ["soft tissue sarcoma"],
    "Cutaneous SCC":                ["cutaneous squamous cell carcinoma", "cSCC"],
    "Desmoid Tumor":                ["aggressive fibromatosis", "desmoid-type fibromatosis"],
}

# Radiation technique mappings from vocabulary Radiation Therapies section
TECHNIQUE_MAP = {
    "EBRT":                 ["external beam radiation therapy", "external beam radiotherapy",
                             "external beam radiation", "external beam RT", "external radiation"],
    "IMRT":                 ["intensity-modulated radiation therapy", "intensity modulated radiation therapy",
                             "intensity-modulated radiotherapy", "intensity modulated RT"],
    "VMAT":                 ["volumetric modulated arc therapy", "volumetric-modulated arc therapy",
                             "RapidArc"],
    "3D-CRT":               ["3D conformal radiation therapy", "3D conformal", "3-dimensional conformal",
                             "3D-conformal", "conformal radiotherapy", "conformal radiation therapy",
                             "conventional radiotherapy", "conventional radiation therapy",
                             "conventional RT"],
    "SBRT":                 ["stereotactic body radiotherapy", "stereotactic body radiation therapy",
                             "stereotactic ablative radiotherapy", "SABR",
                             "stereotactic body RT"],
    "SRS":                  ["stereotactic radiosurgery", "stereotactic radiation surgery",
                             "Gamma Knife", "gamma knife surgery", "gamma knife radiosurgery",
                             "CyberKnife", "Linac-based SRS"],
    "IGRT":                 ["image-guided radiation therapy", "image guided radiation therapy",
                             "image-guided radiotherapy", "image guided RT"],
    "Proton Therapy":       ["proton beam therapy", "proton radiation", "proton beam",
                             "proton beam radiation therapy", "PBT"],
    "Brachytherapy":        ["internal radiation", "intracavitary radiation", "interstitial radiation",
                             "intracavitary brachytherapy", "interstitial brachytherapy",
                             "HDR brachytherapy", "LDR brachytherapy",
                             "high dose rate brachytherapy", "low dose rate brachytherapy"],
    "TBI":                  ["total body irradiation"],
    "WBRT":                 ["whole brain radiotherapy", "whole brain radiation therapy",
                             "whole brain RT", "whole-brain radiation"],
    "Electron Therapy":     ["electron beam therapy", "electron beam", "electron beam radiation",
                             "electron therapy"],
    "Carbon Ion Therapy":   ["carbon ion radiotherapy", "heavy ion therapy", "particle therapy"],
}


def build_normalize_phase_sql() -> str:
    """Generate PL/pgSQL function from PHASE_MAP."""
    lines = [
        "CREATE OR REPLACE FUNCTION normalize_phase(txt TEXT)",
        "RETURNS TEXT AS $$",
        "DECLARE",
        "    low TEXT;",
        "BEGIN",
        "    IF txt IS NULL OR TRIM(txt) = '' THEN RETURN NULL; END IF;",
        "    low := LOWER(TRIM(txt));",
    ]
    # Check combined phases first (most specific), then single phases
    ordered = ["Phase I/II", "Phase II/III", "Phase IV", "Phase III", "Phase II", "Phase I"]
    for canonical in ordered:
        synonyms = PHASE_MAP.get(canonical, [])
        all_variants = [canonical.lower()] + [s.lower() for s in synonyms]
        conditions = " OR ".join([f"low = '{v.replace(chr(39), chr(39)+chr(39))}'" for v in all_variants])
        lines.append(f"    IF {conditions} THEN RETURN '{canonical}'; END IF;")
    lines += ["    RETURN INITCAP(TRIM(txt));", "END;", "$$ LANGUAGE plpgsql IMMUTABLE;"]
    return "\n".join(lines)


def build_normalize_cancer_type_sql() -> str:
    """Generate PL/pgSQL function from CANCER_TYPE_MAP using ILIKE patterns."""
    lines = [
        "CREATE OR REPLACE FUNCTION normalize_cancer_type(txt TEXT)",
        "RETURNS TEXT AS $$",
        "DECLARE",
        "    low TEXT;",
        "BEGIN",
        "    IF txt IS NULL OR TRIM(txt) = '' THEN RETURN NULL; END IF;",
        "    low := LOWER(TRIM(txt));",
    ]
    # Order by specificity: longer/more-specific patterns first
    # Sort by number of synonyms (more specific types usually have more synonyms)
    for canonical, synonyms in sorted(CANCER_TYPE_MAP.items(), key=lambda x: -len(x[0])):
        all_variants = [canonical.lower()] + [s.lower() for s in synonyms]
        # Escape single quotes in LIKE patterns AND canonical
        conditions = " OR ".join([f"low LIKE '%{v.replace(chr(39), chr(39)+chr(39))}%'" for v in all_variants])
        safe_canonical = canonical.replace("'", "''")
        lines.append(f"    IF {conditions} THEN RETURN '{safe_canonical}'; END IF;")
    lines += ["    RETURN INITCAP(TRIM(txt));", "END;", "$$ LANGUAGE plpgsql IMMUTABLE;"]
    return "\n".join(lines)


def build_normalize_technique_sql() -> str:
    """Generate PL/pgSQL function from TECHNIQUE_MAP using ILIKE patterns."""
    lines = [
        "CREATE OR REPLACE FUNCTION normalize_technique(txt TEXT)",
        "RETURNS TEXT AS $$",
        "DECLARE",
        "    low TEXT;",
        "BEGIN",
        "    IF txt IS NULL OR TRIM(txt) = '' THEN RETURN NULL; END IF;",
        "    low := LOWER(TRIM(txt));",
    ]
    # Check specific techniques before generic ones
    for canonical, synonyms in TECHNIQUE_MAP.items():
        all_variants = [canonical.lower()] + [s.lower() for s in synonyms]
        conditions = " OR ".join([f"low LIKE '%{v.replace(chr(39), chr(39)+chr(39))}%'" for v in all_variants])
        safe_canonical = canonical.replace("'", "''")
        lines.append(f"    IF {conditions} THEN RETURN '{safe_canonical}'; END IF;")
    # Catch-all: if contains "radiotherapy" or "radiation therapy" but didn't match above
    lines.append("    IF low LIKE '%radiotherapy%' OR low LIKE '%radiation therapy%' THEN RETURN 'Radiotherapy (unspecified)'; END IF;")
    lines += ["    RETURN INITCAP(TRIM(txt));", "END;", "$$ LANGUAGE plpgsql IMMUTABLE;"]
    return "\n".join(lines)


async def main():
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "34.21.60.224"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DATABASE", "display-study-details"),
    )

    # ── 1. Create/replace helper functions ────────────────────────
    print("=" * 60)
    print("CREATING NORMALIZATION FUNCTIONS")
    print("=" * 60)

    # Text parsing functions (keep from v2)
    print("  Creating extract_percent()...")
    await conn.execute(r"""
    CREATE OR REPLACE FUNCTION extract_percent(txt TEXT)
    RETURNS NUMERIC AS $$
    DECLARE m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        m := regexp_match(txt, '(\d+\.?\d*)\s*%');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    print("  Creating extract_months()...")
    await conn.execute(r"""
    CREATE OR REPLACE FUNCTION extract_months(txt TEXT)
    RETURNS NUMERIC AS $$
    DECLARE m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        m := regexp_match(txt, '(\d+\.?\d*)\s*months?');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        m := regexp_match(txt, '(\d+\.?\d*)\s*years?');
        IF m IS NOT NULL THEN RETURN (m[1]::numeric * 12.0); END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    print("  Creating extract_gy()...")
    await conn.execute(r"""
    CREATE OR REPLACE FUNCTION extract_gy(txt TEXT)
    RETURNS NUMERIC AS $$
    DECLARE m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        m := regexp_match(txt, '(\d+\.?\d*)\s*[-]?\s*[Gg][Yy]');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        m := regexp_match(txt, '(\d+\.?\d*)\s*[Gg]ray');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    print("  Creating extract_fractions()...")
    await conn.execute(r"""
    CREATE OR REPLACE FUNCTION extract_fractions(txt TEXT)
    RETURNS INTEGER AS $$
    DECLARE m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        m := regexp_match(txt, '(\d+)\s*fr');
        IF m IS NOT NULL THEN RETURN m[1]::integer; END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    # Vocabulary-driven normalization functions
    print("  Creating normalize_phase() from vocabulary...")
    sql = build_normalize_phase_sql()
    await conn.execute(sql)

    print("  Creating normalize_cancer_type() from vocabulary...")
    sql = build_normalize_cancer_type_sql()
    await conn.execute(sql)

    print("  Creating normalize_technique() from vocabulary...")
    sql = build_normalize_technique_sql()
    await conn.execute(sql)

    print("  All functions created.\n")

    # ── 2. Rebuild mv_study_summary ───────────────────────────────
    print("=" * 60)
    print("REBUILDING mv_study_summary")
    print("=" * 60)

    await conn.execute(r"""
    DROP MATERIALIZED VIEW IF EXISTS mv_study_summary CASCADE;

    CREATE MATERIALIZED VIEW mv_study_summary AS
    SELECT
        s.study_id,
        s.document_name,
        s.study_name,
        s.doi,
        s.pmid,
        normalize_cancer_type(s.cancer_type) AS cancer_type,
        s.cancer_type AS cancer_type_raw,
        normalize_phase(s.study_phase) AS normalized_phase,
        s.study_phase AS phase_raw,
        s.study_type AS study_type_normalized,
        s.number_of_patients,
        s.study_institution,
        s.country,
        CASE WHEN s.study_type ILIKE '%random%' THEN true ELSE false END AS is_randomized,
        CASE WHEN s.study_institution ILIKE '%multi%' OR s.country ILIKE '%,%'
             THEN true ELSE false END AS is_multi_center,
        (regexp_match(s.median_age, '(\d+\.?\d*)'))[1]::numeric AS median_age_numeric,
        extract_percent(s.overall_survival) AS os_rate_percent,
        extract_months(s.overall_survival) AS os_median_months,
        extract_percent(s.progression_free_survival) AS pfs_rate_percent,
        extract_months(s.progression_free_survival) AS pfs_median_months,
        extract_percent(s.disease_free_survival) AS dfs_rate_percent,
        extract_months(s.disease_free_survival) AS dfs_median_months,
        extract_percent(s.local_control) AS lc_rate_percent,
        extract_months(s.median_followup) AS median_followup_months,
        s.overall_survival AS os_raw,
        s.progression_free_survival AS pfs_raw,
        s.local_control AS lc_raw,
        s.median_followup AS followup_raw
    FROM studies s;

    CREATE UNIQUE INDEX ON mv_study_summary (study_id);
    CREATE INDEX ON mv_study_summary (cancer_type);
    CREATE INDEX ON mv_study_summary (normalized_phase);
    """)

    count = await conn.fetchval("SELECT COUNT(*) FROM mv_study_summary")
    os_count = await conn.fetchval("SELECT COUNT(*) FROM mv_study_summary WHERE os_rate_percent IS NOT NULL")
    ct_count = await conn.fetchval("SELECT COUNT(DISTINCT cancer_type) FROM mv_study_summary WHERE cancer_type IS NOT NULL")
    ph_count = await conn.fetchval("SELECT COUNT(DISTINCT normalized_phase) FROM mv_study_summary WHERE normalized_phase IS NOT NULL")

    print(f"  Total rows:            {count}")
    print(f"  OS % parsed:           {os_count}")
    print(f"  Distinct cancer types: {ct_count}")
    print(f"  Distinct phases:       {ph_count}")

    print("\n  Phases:")
    rows = await conn.fetch("""
        SELECT normalized_phase, COUNT(*) AS n
        FROM mv_study_summary WHERE normalized_phase IS NOT NULL
        GROUP BY normalized_phase ORDER BY n DESC
    """)
    for r in rows:
        print(f"    {r['n']:4d}  {r['normalized_phase']}")

    print("\n  Top 20 cancer types:")
    rows = await conn.fetch("""
        SELECT cancer_type, COUNT(*) AS n
        FROM mv_study_summary WHERE cancer_type IS NOT NULL
        GROUP BY cancer_type ORDER BY n DESC LIMIT 20
    """)
    for r in rows:
        print(f"    {r['n']:4d}  {r['cancer_type']}")

    # ── 3. Rebuild mv_radiation_summary ───────────────────────────
    print("\n" + "=" * 60)
    print("REBUILDING mv_radiation_summary")
    print("=" * 60)

    await conn.execute("""
    DROP MATERIALIZED VIEW IF EXISTS mv_radiation_summary CASCADE;

    CREATE MATERIALIZED VIEW mv_radiation_summary AS
    SELECT
        rd.id AS radiation_id,
        rd.study_id,
        s.document_name,
        normalize_cancer_type(s.cancer_type) AS cancer_type,
        normalize_phase(s.study_phase) AS normalized_phase,
        rd.technique AS technique_raw,
        normalize_technique(rd.technique) AS normalized_technique,
        rd.radiation_type,
        extract_gy(rd.total_dose) AS total_dose_gy,
        extract_gy(rd.fractionation) AS fraction_dose_gy,
        extract_fractions(rd.total_dose) AS number_of_fractions,
        COALESCE(extract_fractions(rd.fractionation), extract_fractions(rd.total_dose)) AS fractions_from_text,
        extract_gy(rd.additional_details->>'total_dose') AS dose_from_json,
        extract_gy(rd.additional_details->>'CSI_dose') AS csi_dose_gy,
        extract_gy(rd.additional_details->>'boost_dose') AS boost_dose_gy,
        COALESCE(
            extract_gy(rd.total_dose),
            extract_gy(rd.additional_details->>'total_dose'),
            extract_gy(rd.additional_details->>'CSI_dose'),
            extract_gy(rd.evidence_quote)
        ) AS best_dose_gy,
        rd.target_volume,
        s.number_of_patients
    FROM radiation_details rd
    JOIN studies s ON s.study_id = rd.study_id;

    CREATE INDEX ON mv_radiation_summary (study_id);
    CREATE INDEX ON mv_radiation_summary (cancer_type);
    CREATE INDEX ON mv_radiation_summary (normalized_technique);
    CREATE INDEX ON mv_radiation_summary (best_dose_gy) WHERE best_dose_gy IS NOT NULL;
    """)

    count = await conn.fetchval("SELECT COUNT(*) FROM mv_radiation_summary")
    dose_count = await conn.fetchval("SELECT COUNT(*) FROM mv_radiation_summary WHERE best_dose_gy IS NOT NULL")
    tech_count = await conn.fetchval("SELECT COUNT(DISTINCT normalized_technique) FROM mv_radiation_summary WHERE normalized_technique IS NOT NULL")

    print(f"  Total rows:            {count}")
    print(f"  With dose parsed:      {dose_count}")
    print(f"  Distinct techniques:   {tech_count}")

    print("\n  Top 15 techniques (normalized):")
    rows = await conn.fetch("""
        SELECT normalized_technique, COUNT(*) AS n
        FROM mv_radiation_summary WHERE normalized_technique IS NOT NULL
        GROUP BY normalized_technique ORDER BY n DESC LIMIT 15
    """)
    for r in rows:
        print(f"    {r['n']:4d}  {r['normalized_technique']}")

    # ── 4. Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Restart server and test:")
    print("=" * 60)
    print("  curl http://localhost:8000/api/analytics/overview")
    print("  open http://localhost:8000/analytics.html")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())