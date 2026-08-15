#!/usr/bin/env python3
"""Quick diagnostic: show actual columns + sample data for analytics."""
import asyncio, asyncpg, os
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

    # 1) Outcome-related columns on studies
    print("=" * 70)
    print("OUTCOME COLUMNS ON studies TABLE")
    print("=" * 70)
    cols = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'studies'
          AND (column_name ILIKE '%survival%' OR column_name ILIKE '%os_%'
               OR column_name ILIKE '%pfs%' OR column_name ILIKE '%dfs%'
               OR column_name ILIKE '%local_control%' OR column_name ILIKE '%lc_%'
               OR column_name ILIKE '%followup%' OR column_name ILIKE '%follow_up%'
               OR column_name ILIKE '%endpoint%' OR column_name ILIKE '%hr%'
               OR column_name ILIKE '%hazard%' OR column_name ILIKE '%median%')
        ORDER BY column_name
    """)
    for c in cols:
        print(f"  {c['column_name']:45s} {c['data_type']}")

    # Sample values for key outcome columns
    print("\nSAMPLE VALUES (first 5 non-null):")
    for col_name in ['overall_survival', 'progression_free_survival', 'disease_free_survival',
                     'local_control', 'median_followup', 'primary_endpoint']:
        try:
            rows = await conn.fetch(f"""
                SELECT {col_name} FROM studies
                WHERE {col_name} IS NOT NULL AND {col_name} != ''
                LIMIT 5
            """)
            vals = [r[col_name][:80] for r in rows]
            print(f"\n  {col_name}:")
            for v in vals:
                print(f"    → {v}")
        except:
            print(f"\n  {col_name}: COLUMN NOT FOUND")

    # 2) radiation_details columns
    print("\n" + "=" * 70)
    print("RADIATION_DETAILS COLUMNS")
    print("=" * 70)
    cols = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'radiation_details'
        ORDER BY column_name
    """)
    for c in cols:
        print(f"  {c['column_name']:35s} {c['data_type']}")

    # Sample radiation data
    print("\nSAMPLE radiation_details (3 rows):")
    rows = await conn.fetch("SELECT * FROM radiation_details LIMIT 3")
    for r in rows:
        print(f"  {dict(r)}")

    # 3) Cancer type values
    print("\n" + "=" * 70)
    print("TOP 20 CANCER TYPE VALUES (from studies.cancer_type)")
    print("=" * 70)
    try:
        rows = await conn.fetch("""
            SELECT cancer_type, COUNT(*) AS n
            FROM studies
            WHERE cancer_type IS NOT NULL
            GROUP BY cancer_type
            ORDER BY n DESC
            LIMIT 20
        """)
        for r in rows:
            print(f"  {r['n']:4d}  {r['cancer_type']}")
    except:
        print("  cancer_type column not found")

    # 4) Study phase values
    print("\n" + "=" * 70)
    print("STUDY PHASE VALUES (from studies.study_phase)")
    print("=" * 70)
    try:
        rows = await conn.fetch("""
            SELECT study_phase, COUNT(*) AS n
            FROM studies
            WHERE study_phase IS NOT NULL
            GROUP BY study_phase
            ORDER BY n DESC
            LIMIT 20
        """)
        for r in rows:
            print(f"  {r['n']:4d}  {r['study_phase']}")
    except:
        print("  study_phase column not found")

    # 5) Check stage_distribution table
    print("\n" + "=" * 70)
    print("STAGE_DISTRIBUTION CHECK")
    print("=" * 70)
    for tname in ['stage_distribution', 'stage_distributions']:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", tname)
        if exists:
            cnt = await conn.fetchval(f"SELECT COUNT(*) FROM {tname}")
            print(f"  {tname}: {cnt} rows")
            cols = await conn.fetch("""
                SELECT column_name FROM information_schema.columns WHERE table_name = $1 ORDER BY column_name
            """, tname)
            print(f"  Columns: {[c['column_name'] for c in cols]}")
        else:
            print(f"  {tname}: NOT FOUND")

    await conn.close()

asyncio.run(main())
