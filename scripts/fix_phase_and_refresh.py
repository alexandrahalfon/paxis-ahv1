#!/usr/bin/env python3
"""Fix normalize_phase: remove \\b (not supported in PostgreSQL POSIX regex)."""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

FIXED_FUNCTION = r"""
CREATE OR REPLACE FUNCTION normalize_phase(txt TEXT)
RETURNS TEXT AS $$
BEGIN
    IF txt IS NULL THEN RETURN NULL; END IF;
    -- Combined phases first (most specific)
    IF txt ~* 'phase\s*(2|II)\s*[/-]\s*(3|III)' THEN RETURN 'Phase II/III'; END IF;
    IF txt ~* 'phase\s*(1|I)\s*[/-]\s*(2|II)' THEN RETURN 'Phase I/II'; END IF;
    -- Phase III: roman first, then arabic
    IF txt ~* 'phase\s*III' AND txt !~* '(I/III|II/III)' THEN RETURN 'Phase III'; END IF;
    IF txt ~* 'phase\s*3' AND txt !~* '(2/3|2-3|1/3|2 / 3)' THEN RETURN 'Phase III'; END IF;
    -- Phase II
    IF txt ~* 'phase\s*(IIR|IIB|IIA|II)' THEN RETURN 'Phase II'; END IF;
    IF txt ~* 'phase\s*2' AND txt !~* '(2/3|2-3|2 / 3)' THEN RETURN 'Phase II'; END IF;
    -- Phase IV
    IF txt ~* 'phase\s*IV' THEN RETURN 'Phase IV'; END IF;
    IF txt ~* 'phase\s*4' THEN RETURN 'Phase IV'; END IF;
    -- Phase I (last — most permissive)
    IF txt ~* 'phase\s*(Ib|Ia|I)' AND txt !~* '(I/II|I-II|II)' THEN RETURN 'Phase I'; END IF;
    IF txt ~* 'phase\s*1' AND txt !~* '(1/2|1-2)' THEN RETURN 'Phase I'; END IF;
    RETURN txt;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""

async def main():
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "34.21.60.224"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DATABASE", "display-study-details"),
    )

    print("Fixing normalize_phase function...")
    await conn.execute(FIXED_FUNCTION)

    print("Refreshing mv_study_summary...")
    await conn.execute("REFRESH MATERIALIZED VIEW mv_study_summary;")

    print("\nPhase distribution after fix:")
    rows = await conn.fetch("""
        SELECT normalized_phase, COUNT(*) AS n
        FROM mv_study_summary WHERE normalized_phase IS NOT NULL
        GROUP BY normalized_phase ORDER BY n DESC
    """)
    for r in rows:
        print(f"  {r['n']:4d}  {r['normalized_phase']}")

    print("\nDone!")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
