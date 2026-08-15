#!/usr/bin/env python3
"""
Create Materialized Views v2 — with text parsing

Your DB has raw text for outcomes ("91.3% at 3 years"), doses ("66 Gy in 33 fractions"),
phases ("Phase III" vs "Phase 3"), and cancer types with duplicates.

This script creates views that:
  1. Parse numeric values from text with regex
  2. Normalize phases → "Phase I", "Phase II", "Phase III"
  3. Normalize cancer types → group common variants
  4. Extract dose_gy from radiation_details.total_dose text

Usage:
    python scripts/create_materialized_views_v2.py
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()


async def main():
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "34.21.60.224"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DATABASE", "display-study-details"),
    )

    # ══════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS (created as SQL functions for reuse)
    # ══════════════════════════════════════════════════════════════
    print("Creating helper functions...")

    await conn.execute(r"""
    -- Extract first percentage from text: "91.3% at 3 years" → 91.3
    CREATE OR REPLACE FUNCTION extract_percent(txt TEXT)
    RETURNS NUMERIC AS $$
    DECLARE
        m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        m := regexp_match(txt, '(\d+\.?\d*)\s*%');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    await conn.execute(r"""
    -- Extract months value: "10.9 months" → 10.9, "45.2 months" → 45.2
    CREATE OR REPLACE FUNCTION extract_months(txt TEXT)
    RETURNS NUMERIC AS $$
    DECLARE
        m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        -- "X months" or "X month"
        m := regexp_match(txt, '(\d+\.?\d*)\s*months?');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        -- "X years" → convert
        m := regexp_match(txt, '(\d+\.?\d*)\s*years?');
        IF m IS NOT NULL THEN RETURN (m[1]::numeric * 12.0); END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    await conn.execute(r"""
    -- Extract first Gy value: "66 Gy in 33 fractions" → 66.0
    CREATE OR REPLACE FUNCTION extract_gy(txt TEXT)
    RETURNS NUMERIC AS $$
    DECLARE
        m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        -- "XX Gy" or "XX-Gy" or "XXGy"
        m := regexp_match(txt, '(\d+\.?\d*)\s*[-]?\s*[Gg][Yy]');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        -- "XX Gray"
        m := regexp_match(txt, '(\d+\.?\d*)\s*[Gg]ray');
        IF m IS NOT NULL THEN RETURN m[1]::numeric; END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    await conn.execute(r"""
    -- Extract fractions: "33 fractions" → 33
    CREATE OR REPLACE FUNCTION extract_fractions(txt TEXT)
    RETURNS INTEGER AS $$
    DECLARE
        m TEXT[];
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        m := regexp_match(txt, '(\d+)\s*fr');
        IF m IS NOT NULL THEN RETURN m[1]::integer; END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    await conn.execute(r"""
    -- Normalize phase: "Phase III", "Phase 3", "Phase 2/3" → clean values
    CREATE OR REPLACE FUNCTION normalize_phase(txt TEXT)
    RETURNS TEXT AS $$
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        IF txt ~* 'phase\s*(2|II)\s*[/-]\s*(3|III)' THEN RETURN 'Phase II/III'; END IF;
        IF txt ~* 'phase\s*(1|I)\s*[/-]\s*(2|II)' THEN RETURN 'Phase I/II'; END IF;
        IF txt ~* 'phase\s*(III|3)\b' AND txt !~* '(I/III|II/III|2/3|1/3)' THEN RETURN 'Phase III'; END IF;
        IF txt ~* 'phase\s*(IIR|IIB|IIA|II)\b' THEN RETURN 'Phase II'; END IF;
        IF txt ~* 'phase\s*2\b' AND txt !~* '(2/3|2-3|2 / 3)' THEN RETURN 'Phase II'; END IF;
        IF txt ~* 'phase\s*(Ib|Ia|I)\b' AND txt !~* '(I/II|I-II|II)' THEN RETURN 'Phase I'; END IF;
        IF txt ~* 'phase\s*1\b' AND txt !~* '(1/2|1-2)' THEN RETURN 'Phase I'; END IF;
        IF txt ~* 'phase\s*(IV|4)' THEN RETURN 'Phase IV'; END IF;
        RETURN txt;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    await conn.execute(r"""
    -- Normalize cancer type: group common variants
    CREATE OR REPLACE FUNCTION normalize_cancer_type(txt TEXT)
    RETURNS TEXT AS $$
    BEGIN
        IF txt IS NULL THEN RETURN NULL; END IF;
        -- NSCLC variants
        IF txt ~* 'non.?small.?cell\s*lung' THEN RETURN 'Non-Small Cell Lung Cancer'; END IF;
        IF txt ~* '\bNSCLC\b' THEN RETURN 'Non-Small Cell Lung Cancer'; END IF;
        -- SCLC
        IF txt ~* 'small.?cell\s*lung' THEN RETURN 'Small Cell Lung Cancer'; END IF;
        IF txt ~* '\bSCLC\b' THEN RETURN 'Small Cell Lung Cancer'; END IF;
        -- Breast
        IF txt ~* 'breast\s*(cancer|carcinoma|adenocarcinoma)' THEN RETURN 'Breast Cancer'; END IF;
        IF txt ~* 'invasive breast' THEN RETURN 'Breast Cancer'; END IF;
        IF txt ~* 'ductal carcinoma' THEN RETURN 'Breast Cancer'; END IF;
        -- Prostate
        IF txt ~* 'prostate' THEN RETURN 'Prostate Cancer'; END IF;
        -- Cervical
        IF txt ~* 'cervic' THEN RETURN 'Cervical Cancer'; END IF;
        -- NPC
        IF txt ~* 'nasopharyng' THEN RETURN 'Nasopharyngeal Carcinoma'; END IF;
        IF txt ~* '\bNPC\b' THEN RETURN 'Nasopharyngeal Carcinoma'; END IF;
        -- Head & Neck SCC
        IF txt ~* 'head\s*and\s*neck\s*(squamous|cancer|carcinoma)' THEN RETURN 'Head and Neck SCC'; END IF;
        IF txt ~* 'oropharyn' THEN RETURN 'Head and Neck SCC'; END IF;
        IF txt ~* '\bHNSCC\b' THEN RETURN 'Head and Neck SCC'; END IF;
        -- GBM
        IF txt ~* 'glioblastoma' THEN RETURN 'Glioblastoma'; END IF;
        IF txt ~* '\bGBM\b' THEN RETURN 'Glioblastoma'; END IF;
        -- Rectal
        IF txt ~* 'rectal' THEN RETURN 'Rectal Cancer'; END IF;
        -- Pancreatic
        IF txt ~* 'pancrea' THEN RETURN 'Pancreatic Cancer'; END IF;
        -- Esophageal
        IF txt ~* 'esophag' THEN RETURN 'Esophageal Cancer'; END IF;
        -- Endometrial
        IF txt ~* 'endometri' THEN RETURN 'Endometrial Cancer'; END IF;
        -- Bladder
        IF txt ~* 'bladder' THEN RETURN 'Bladder Cancer'; END IF;
        -- Hodgkin
        IF txt ~* 'hodgkin' THEN RETURN 'Hodgkin Lymphoma'; END IF;
        -- Non-Hodgkin
        IF txt ~* 'non.?hodgkin|diffuse large b' THEN RETURN 'Non-Hodgkin Lymphoma'; END IF;
        -- Melanoma
        IF txt ~* 'melanoma' THEN RETURN 'Melanoma'; END IF;
        -- Brain mets
        IF txt ~* 'brain\s*metast' THEN RETURN 'Brain Metastases'; END IF;
        -- Rhabdomyosarcoma
        IF txt ~* 'rhabdomyosar' THEN RETURN 'Rhabdomyosarcoma'; END IF;
        -- SCC generic (only if not already caught)
        IF txt ~* '^squamous\s*cell\s*carcinoma$' THEN RETURN 'Squamous Cell Carcinoma'; END IF;
        -- Adenocarcinoma generic
        IF txt ~* '^adenocarcinoma$' THEN RETURN 'Adenocarcinoma'; END IF;
        -- Hepatocellular
        IF txt ~* 'hepatocellular|liver cancer' THEN RETURN 'Hepatocellular Carcinoma'; END IF;
        -- Renal
        IF txt ~* 'renal cell|kidney cancer' THEN RETURN 'Renal Cell Carcinoma'; END IF;
        -- Colorectal
        IF txt ~* 'colorectal|colon cancer' THEN RETURN 'Colorectal Cancer'; END IF;
        -- Gastric
        IF txt ~* 'gastric|stomach' THEN RETURN 'Gastric Cancer'; END IF;
        -- Ovarian
        IF txt ~* 'ovari' THEN RETURN 'Ovarian Cancer'; END IF;
        -- Meningioma
        IF txt ~* 'meningioma' THEN RETURN 'Meningioma'; END IF;
        -- Medulloblastoma
        IF txt ~* 'medulloblast' THEN RETURN 'Medulloblastoma'; END IF;
        -- Ependymoma
        IF txt ~* 'ependymoma' THEN RETURN 'Ependymoma'; END IF;
        -- Neuroblastoma
        IF txt ~* 'neuroblast' THEN RETURN 'Neuroblastoma'; END IF;
        -- Wilms
        IF txt ~* 'wilms|nephroblast' THEN RETURN 'Wilms Tumor'; END IF;
        -- Ewing
        IF txt ~* 'ewing' THEN RETURN 'Ewing Sarcoma'; END IF;
        -- Return cleaned original
        RETURN INITCAP(TRIM(txt));
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    print("  Helper functions created.\n")

    # ══════════════════════════════════════════════════════════════
    # mv_study_summary
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("CREATING mv_study_summary")
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

        -- Normalized cancer type
        normalize_cancer_type(s.cancer_type) AS cancer_type,
        s.cancer_type AS cancer_type_raw,

        -- Normalized phase
        normalize_phase(s.study_phase) AS normalized_phase,
        s.study_phase AS phase_raw,

        -- Study design
        s.study_type AS study_type_normalized,
        s.number_of_patients,
        s.study_institution,
        s.country,

        -- Booleans (extract from text if needed)
        CASE WHEN s.study_type ILIKE '%random%' THEN true
             ELSE false END AS is_randomized,
        CASE WHEN s.study_institution ILIKE '%multi%' OR s.country ILIKE '%,%'
             THEN true ELSE false END AS is_multi_center,

        -- Patient characteristics (extract from text)
        extract_months(s.median_age)  AS median_age_years,  -- usually just a number
        (regexp_match(s.median_age, '(\d+\.?\d*)'))[1]::numeric AS median_age_numeric,

        -- Parsed outcomes
        extract_percent(s.overall_survival)          AS os_rate_percent,
        extract_months(s.overall_survival)            AS os_median_months,
        extract_percent(s.progression_free_survival)  AS pfs_rate_percent,
        extract_months(s.progression_free_survival)   AS pfs_median_months,
        extract_percent(s.disease_free_survival)      AS dfs_rate_percent,
        extract_months(s.disease_free_survival)       AS dfs_median_months,
        extract_percent(s.local_control)              AS lc_rate_percent,
        extract_months(s.median_followup)             AS median_followup_months,

        -- Raw text for reference
        s.overall_survival AS os_raw,
        s.progression_free_survival AS pfs_raw,
        s.local_control AS lc_raw,
        s.median_followup AS followup_raw,

        -- Biomarker status (structured JSONB)
        s.biomarker_status,
        s.genomic_assay,
        s.genomic_score_range,
        s.molecular_subtype

    FROM studies s;

    CREATE UNIQUE INDEX ON mv_study_summary (study_id);
    CREATE INDEX ON mv_study_summary (cancer_type);
    CREATE INDEX ON mv_study_summary (normalized_phase);
    CREATE INDEX ON mv_study_summary (os_rate_percent) WHERE os_rate_percent IS NOT NULL;
    CREATE INDEX ON mv_study_summary USING GIN (biomarker_status) WHERE biomarker_status IS NOT NULL;
    CREATE INDEX ON mv_study_summary (genomic_assay) WHERE genomic_assay IS NOT NULL;
    """)

    count = await conn.fetchval("SELECT COUNT(*) FROM mv_study_summary")
    os_count = await conn.fetchval("SELECT COUNT(*) FROM mv_study_summary WHERE os_rate_percent IS NOT NULL")
    pfs_count = await conn.fetchval("SELECT COUNT(*) FROM mv_study_summary WHERE pfs_rate_percent IS NOT NULL")
    ct_count = await conn.fetchval("SELECT COUNT(DISTINCT cancer_type) FROM mv_study_summary WHERE cancer_type IS NOT NULL")
    ph_count = await conn.fetchval("SELECT COUNT(DISTINCT normalized_phase) FROM mv_study_summary WHERE normalized_phase IS NOT NULL")

    print(f"  Total rows:          {count}")
    print(f"  With OS % parsed:    {os_count}")
    print(f"  With PFS % parsed:   {pfs_count}")
    print(f"  Distinct cancer types: {ct_count}")
    print(f"  Distinct phases:     {ph_count}")

    # Show sample parsed values
    print("\n  Sample parsed outcomes:")
    rows = await conn.fetch("""
        SELECT cancer_type, normalized_phase, os_rate_percent, os_median_months,
               pfs_rate_percent, median_followup_months, os_raw
        FROM mv_study_summary
        WHERE os_rate_percent IS NOT NULL
        LIMIT 5
    """)
    for r in rows:
        print(f"    OS={r['os_rate_percent']}% | PFS={r['pfs_rate_percent']} | fu={r['median_followup_months']}mo | {r['cancer_type']} | {r['normalized_phase']}")
        print(f"      raw: {(r['os_raw'] or '')[:60]}")

    print(f"\n  Top cancer types (normalized):")
    rows = await conn.fetch("""
        SELECT cancer_type, COUNT(*) AS n
        FROM mv_study_summary WHERE cancer_type IS NOT NULL
        GROUP BY cancer_type ORDER BY n DESC LIMIT 15
    """)
    for r in rows:
        print(f"    {r['n']:4d}  {r['cancer_type']}")

    # ══════════════════════════════════════════════════════════════
    # mv_radiation_summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CREATING mv_radiation_summary")
    print("=" * 60)

    await conn.execute("""
    DROP MATERIALIZED VIEW IF EXISTS mv_radiation_summary CASCADE;

    CREATE MATERIALIZED VIEW mv_radiation_summary AS
    SELECT
        rd.id AS radiation_id,
        rd.study_id,
        s.document_name,

        -- Cancer type + phase from study
        normalize_cancer_type(s.cancer_type) AS cancer_type,
        normalize_phase(s.study_phase) AS normalized_phase,

        -- Technique
        rd.technique AS technique_raw,
        UPPER(TRIM(rd.technique)) AS normalized_technique,
        rd.radiation_type,

        -- Parse dose from text
        extract_gy(rd.total_dose) AS total_dose_gy,
        extract_gy(rd.fractionation) AS fraction_dose_gy,  -- sometimes "3 Gy per fraction"
        extract_fractions(rd.total_dose) AS number_of_fractions,
        COALESCE(extract_fractions(rd.fractionation), extract_fractions(rd.total_dose)) AS fractions_from_text,

        -- Also try extracting from additional_details JSONB
        extract_gy(rd.additional_details->>'total_dose') AS dose_from_json,
        extract_gy(rd.additional_details->>'CSI_dose') AS csi_dose_gy,
        extract_gy(rd.additional_details->>'boost_dose') AS boost_dose_gy,

        -- Best dose: prefer direct column, fall back to JSON
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

    print(f"  Total rows:         {count}")
    print(f"  With dose parsed:   {dose_count}")
    print(f"  Distinct techniques: {tech_count}")

    print("\n  Sample parsed radiation:")
    rows = await conn.fetch("""
        SELECT cancer_type, normalized_technique, best_dose_gy, number_of_fractions
        FROM mv_radiation_summary
        WHERE best_dose_gy IS NOT NULL
        LIMIT 8
    """)
    for r in rows:
        print(f"    {r['best_dose_gy']} Gy | {r['number_of_fractions']} fx | {r['normalized_technique']} | {r['cancer_type']}")

    print(f"\n  Top techniques:")
    rows = await conn.fetch("""
        SELECT normalized_technique, COUNT(*) AS n
        FROM mv_radiation_summary WHERE normalized_technique IS NOT NULL
        GROUP BY normalized_technique ORDER BY n DESC LIMIT 10
    """)
    for r in rows:
        print(f"    {r['n']:4d}  {r['normalized_technique']}")

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nRestart the server and test:")
    print("  curl http://localhost:8000/api/analytics/overview")
    print("  open http://localhost:8000/analytics.html")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())