#!/usr/bin/env python3
"""Fix normalize_phase function and refresh materialized view."""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def fix():
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', '34.21.60.224'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
        database=os.getenv('POSTGRES_DATABASE', 'display-study-details')
    )
    
    print("Creating normalize_phase function...")
    await conn.execute(r'''
        CREATE OR REPLACE FUNCTION normalize_phase(txt TEXT)
        RETURNS TEXT AS $$
        BEGIN
            IF txt IS NULL THEN RETURN NULL; END IF;
            IF txt ~* 'phase\s*(2|II)\s*[/-]\s*(3|III)' THEN RETURN 'Phase II/III'; END IF;
            IF txt ~* 'phase\s*(1|I)\s*[/-]\s*(2|II)' THEN RETURN 'Phase I/II'; END IF;
            IF txt ~* 'phase\s*III' AND txt !~* '(I/III|II/III)' THEN RETURN 'Phase III'; END IF;
            IF txt ~* 'phase\s*3' AND txt !~* '(2/3|2-3|1/3|2 / 3)' THEN RETURN 'Phase III'; END IF;
            IF txt ~* 'phase\s*(IIR|IIB|IIA|II)' THEN RETURN 'Phase II'; END IF;
            IF txt ~* 'phase\s*2' AND txt !~* '(2/3|2-3|2 / 3)' THEN RETURN 'Phase II'; END IF;
            IF txt ~* 'phase\s*IV' THEN RETURN 'Phase IV'; END IF;
            IF txt ~* 'phase\s*4' THEN RETURN 'Phase IV'; END IF;
            IF txt ~* 'phase\s*(Ib|Ia|I)' AND txt !~* '(I/II|I-II|II)' THEN RETURN 'Phase I'; END IF;
            IF txt ~* 'phase\s*1' AND txt !~* '(1/2|1-2)' THEN RETURN 'Phase I'; END IF;
            RETURN txt;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    ''')
    
    print("Refreshing mv_study_summary...")
    await conn.execute('REFRESH MATERIALIZED VIEW mv_study_summary;')
    
    print("\nPhase distribution:")
    rows = await conn.fetch('''
        SELECT normalized_phase, COUNT(*) AS n
        FROM mv_study_summary WHERE normalized_phase IS NOT NULL
        GROUP BY normalized_phase ORDER BY n DESC
    ''')
    for r in rows:
        print(f'  {r["n"]:4d}  {r["normalized_phase"]}')
    
    await conn.close()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(fix())
