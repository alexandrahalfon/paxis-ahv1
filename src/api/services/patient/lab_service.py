"""Lab result service (Phase 1) — the longitudinal lab schema the beta
audit flagged as entirely missing. Trend queries (ANC 4.2 -> 2.1 -> 0.7)
are the whole point, so canonical_test_name + collected_at are indexed
together (see lab_results_test_trend_idx in patient_db.py).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient._common import row_to_dict, append_profile_timeline_event


class LabService:
    async def add_result(
        self,
        patient_profile_id: str,
        test_name: str,
        canonical_test_name: Optional[str] = None,
        loinc_code: Optional[str] = None,
        value_numeric: Optional[float] = None,
        value_text: Optional[str] = None,
        unit: Optional[str] = None,
        reference_low: Optional[float] = None,
        reference_high: Optional[float] = None,
        abnormal_flag: Optional[str] = None,
        specimen_type: Optional[str] = None,
        collected_at: Optional[str] = None,
        source_type: str = "patient_upload",
        source_document_id: Optional[str] = None,
        verification_status: str = "extracted",
        extraction_confidence: Optional[float] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        result_id = str(uuid.uuid4())
        canonical = canonical_test_name or test_name.strip().lower()

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Duplicate-ingestion prevention (Phase 1 checklist item):
                # the same test/date/value already on file is treated as
                # "already recorded", not re-inserted — this is what
                # stops a patient re-uploading the identical report (or
                # a document being confirmed twice) from doubling their
                # lab history. IS NOT DISTINCT FROM (rather than =) so
                # NULL collected_at / NULL value_text compare as equal to
                # another NULL, matching normal "same missing data" intent.
                existing = await conn.fetchrow(
                    """
                    SELECT * FROM lab_results
                     WHERE patient_profile_id = $1 AND canonical_test_name = $2
                       AND collected_at IS NOT DISTINCT FROM $3
                       AND value_numeric IS NOT DISTINCT FROM $4
                       AND value_text IS NOT DISTINCT FROM $5
                     LIMIT 1
                    """,
                    patient_profile_id, canonical, collected_at, value_numeric, value_text,
                )
                if existing:
                    return row_to_dict(existing)

                row = await conn.fetchrow(
                    """
                    INSERT INTO lab_results
                        (id, patient_profile_id, loinc_code, test_name, canonical_test_name,
                         value_numeric, value_text, unit, reference_low, reference_high,
                         abnormal_flag, specimen_type, collected_at, source_type,
                         source_document_id, verification_status, extraction_confidence)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                    RETURNING *
                    """,
                    result_id, patient_profile_id, loinc_code, test_name, canonical,
                    value_numeric, value_text, unit, reference_low, reference_high,
                    abnormal_flag, specimen_type, collected_at, source_type,
                    source_document_id, verification_status, extraction_confidence,
                )
                await append_profile_timeline_event(
                    conn, patient_profile_id, "lab_result",
                    {
                        "test_name": test_name, "value_numeric": value_numeric,
                        "value_text": value_text, "unit": unit,
                        "abnormal_flag": abnormal_flag,
                    },
                    created_by=created_by, source=source_type,
                )
        from src.api.services.patient.patient_state_service import invalidate_patient_state
        await invalidate_patient_state(patient_profile_id)
        return row_to_dict(row)

    async def list_results(
        self, patient_profile_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM lab_results
                 WHERE patient_profile_id = $1
                 ORDER BY collected_at DESC NULLS LAST, created_at DESC
                 LIMIT $2
                """,
                patient_profile_id, limit,
            )
        return [row_to_dict(r) for r in rows]

    async def get_trend(
        self, patient_profile_id: str, canonical_test_name: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Chronological (oldest first) values for one test — what a
        trend chart or 'is this getting worse' check reads directly."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM lab_results
                 WHERE patient_profile_id = $1 AND canonical_test_name = $2
                 ORDER BY collected_at DESC NULLS LAST
                 LIMIT $3
                """,
                patient_profile_id, canonical_test_name.strip().lower(), limit,
            )
        return list(reversed([row_to_dict(r) for r in rows]))

    async def most_recent_by_test(self, patient_profile_id: str) -> Dict[str, Dict[str, Any]]:
        """Latest value per canonical_test_name — what patient_state_service
        wants for the 'recent_labs' snapshot section."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (canonical_test_name) *
                  FROM lab_results
                 WHERE patient_profile_id = $1
                 ORDER BY canonical_test_name, collected_at DESC NULLS LAST
                """,
                patient_profile_id,
            )
        return {r["canonical_test_name"]: row_to_dict(r) for r in rows}

    async def latest_and_previous_by_test(self, patient_profile_id: str) -> Dict[str, Dict[str, Any]]:
        """Latest AND previous value per canonical_test_name (2026-08-12
        convergence Sprint A item 3) — what the interpretation-policy lab
        shape needs: exact_value_and_trend_only requires two points, not
        one, to say a value moved. Returns
        {canonical_test_name: {"latest": row, "previous": row|None}}."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY canonical_test_name
                               ORDER BY collected_at DESC NULLS LAST, created_at DESC
                           ) AS rn
                      FROM lab_results
                     WHERE patient_profile_id = $1
                ) ranked
                 WHERE rn <= 2
                 ORDER BY canonical_test_name, rn
                """,
                patient_profile_id,
            )
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = row_to_dict(r)
            test = d.get("canonical_test_name")
            slot = "latest" if d.get("rn") == 1 else "previous"
            out.setdefault(test, {})[slot] = d
        return out


_service: Optional[LabService] = None


def get_lab_service() -> LabService:
    global _service
    if _service is None:
        _service = LabService()
    return _service
