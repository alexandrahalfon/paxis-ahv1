"""
Patient Service

CRUD for the patient-centric platform: patients, diagnosis, biomarkers,
treatment history, and the append-only timeline. Uses patient_db.py
(exueed-patients database — separate from display-study-details and
exueed_cache; see src/core/config.py).

Every mutation also appends a row to patient_timeline_events so
pattern_diff_service (Phase 4, not yet built) has a structured history to
diff against. Timeline rows are never updated or deleted by this service.

physician_id fields are UUIDs from the exueed_cache `users` table. There is
no DB-level foreign key across databases, so ownership checks below are
done in application code (WHERE physician_id = $1) rather than relying on a
constraint.
"""

import json
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from .patient_db import get_patient_db


def _iso(value):
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            d[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


class PatientService:
    """Service for managing patient records, owned per-physician."""

    async def _ensure_schema(self):
        await get_patient_db().ensure_schema()

    async def _append_timeline_event(
        self,
        conn,
        patient_id: str,
        event_type: str,
        payload: Dict[str, Any],
        created_by: Optional[str] = None,
        event_date: Optional[str] = None,
        source: str = "manual",
    ):
        """Append a timeline event using an already-acquired connection.

        Append-only: this is the only write path into
        patient_timeline_events. No update/delete method exists on purpose.
        """
        await conn.execute(
            """
            INSERT INTO patient_timeline_events
                (id, patient_id, event_type, event_date, payload, source, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            str(uuid.uuid4()),
            patient_id,
            event_type,
            event_date,
            json.dumps(payload, default=str),
            source,
            created_by,
        )

    # ------------------------------------------------------------------
    # Patients
    # ------------------------------------------------------------------

    async def create_patient(
        self,
        physician_id: str,
        first_name: str,
        last_name: str,
        date_of_birth: Optional[str] = None,
        sex: Optional[str] = None,
        mrn: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()

        patient_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patients
                        (id, physician_id, mrn, first_name, last_name, date_of_birth, sex)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id, physician_id, mrn, first_name, last_name,
                              date_of_birth, sex, created_at, updated_at
                    """,
                    patient_id, physician_id, mrn, first_name, last_name,
                    date_of_birth, sex,
                )
                await self._append_timeline_event(
                    conn, patient_id, "patient_created",
                    {"first_name": first_name, "last_name": last_name, "mrn": mrn},
                    created_by=physician_id,
                )

        return _row_to_dict(row)

    async def get_patient(self, patient_id: str, physician_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, physician_id, mrn, first_name, last_name,
                       date_of_birth, sex, created_at, updated_at,
                       link_status, invite_code
                FROM patients
                WHERE id = $1 AND physician_id = $2
                """,
                patient_id, physician_id,
            )
        return _row_to_dict(row) if row else None

    async def list_patients(
        self, physician_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, physician_id, mrn, first_name, last_name,
                       date_of_birth, sex, created_at, updated_at
                FROM patients
                WHERE physician_id = $1
                ORDER BY updated_at DESC
                LIMIT $2 OFFSET $3
                """,
                physician_id, limit, offset,
            )
        return [_row_to_dict(r) for r in rows]

    async def count_patients(self, physician_id: str) -> int:
        """Total patients for this physician, so a paginated caller can
        tell whether more pages exist."""
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM patients WHERE physician_id = $1",
                    physician_id,
                )
                or 0
            )

    async def list_patients_with_summary(
        self, physician_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Same as list_patients, but with each patient's latest diagnosis
        (cancer_site/stage) inlined so a UI list can show something
        meaningful without a follow-up call per patient. N+1 query pattern
        (one extra query per patient) is deliberately accepted here — this
        is called on-demand when a picker/list UI opens, not on every page
        load, and patient counts are small at this stage."""
        patients = await self.list_patients(physician_id, limit=limit, offset=offset)
        for p in patients:
            diagnosis = await self.get_latest_diagnosis(p["id"])
            if diagnosis:
                p["latest_diagnosis"] = {
                    "cancer_site": diagnosis.get("cancer_site"),
                    "histology": diagnosis.get("histology"),
                    "stage": diagnosis.get("stage"),
                }
            else:
                p["latest_diagnosis"] = None
        return patients

    async def find_patient_by_name(
        self, physician_id: str, first_name: str, last_name: str
    ) -> Optional[Dict[str, Any]]:
        """Case-insensitive exact name match, scoped to this physician's
        patients. Used to prevent the passive-capture flow from creating a
        duplicate patient record when a physician (or the auto-attach
        prompt) re-enters a name that already exists — returns the most
        recently updated match if more than one somehow exists."""
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, physician_id, mrn, first_name, last_name,
                       date_of_birth, sex, created_at, updated_at
                FROM patients
                WHERE physician_id = $1
                  AND lower(first_name) = lower($2)
                  AND lower(last_name) = lower($3)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                physician_id, first_name.strip(), last_name.strip(),
            )
        return _row_to_dict(row) if row else None

    async def update_patient(
        self,
        patient_id: str,
        physician_id: str,
        **fields,
    ) -> Optional[Dict[str, Any]]:
        """Update demographic fields. Only keys in `fields` that are not
        None are applied. Appends a timeline event with the changed keys."""
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()

        allowed = {"mrn", "first_name", "last_name", "date_of_birth", "sex"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return await self.get_patient(patient_id, physician_id)

        set_clauses = []
        params = []
        idx = 1
        for k, v in updates.items():
            set_clauses.append(f"{k} = ${idx}")
            params.append(v)
            idx += 1
        set_clauses.append("updated_at = now()")

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    UPDATE patients SET {", ".join(set_clauses)}
                    WHERE id = ${idx} AND physician_id = ${idx + 1}
                    RETURNING id, physician_id, mrn, first_name, last_name,
                              date_of_birth, sex, created_at, updated_at
                    """,
                    *params, patient_id, physician_id,
                )
                if row:
                    await self._append_timeline_event(
                        conn, patient_id, "patient_updated", updates,
                        created_by=physician_id,
                    )
        return _row_to_dict(row) if row else None

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    async def add_diagnosis(
        self,
        patient_id: str,
        physician_id: str,
        cancer_site: Optional[str] = None,
        histology: Optional[str] = None,
        stage: Optional[str] = None,
        tnm_t: Optional[str] = None,
        tnm_n: Optional[str] = None,
        tnm_m: Optional[str] = None,
        diagnosis_date: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_schema()
        if not await self.get_patient(patient_id, physician_id):
            return None

        db = get_patient_db()
        pool = await db.get_pool()
        diagnosis_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_diagnosis
                        (id, patient_id, cancer_site, histology, stage,
                         tnm_t, tnm_n, tnm_m, diagnosis_date, raw_text)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id, patient_id, cancer_site, histology, stage,
                              tnm_t, tnm_n, tnm_m, diagnosis_date,
                              canonical_site_code, raw_text, created_at
                    """,
                    diagnosis_id, patient_id, cancer_site, histology, stage,
                    tnm_t, tnm_n, tnm_m, diagnosis_date, raw_text,
                )
                await self._append_timeline_event(
                    conn, patient_id, "diagnosis_update",
                    {
                        "cancer_site": cancer_site, "histology": histology,
                        "stage": stage, "tnm_t": tnm_t, "tnm_n": tnm_n,
                        "tnm_m": tnm_m,
                    },
                    created_by=physician_id, event_date=diagnosis_date,
                )
        return _row_to_dict(row)

    async def get_latest_diagnosis(self, patient_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, patient_id, cancer_site, histology, stage,
                       tnm_t, tnm_n, tnm_m, diagnosis_date,
                       canonical_site_code, raw_text, created_at
                FROM patient_diagnosis
                WHERE patient_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                patient_id,
            )
        return _row_to_dict(row) if row else None

    # ------------------------------------------------------------------
    # Biomarkers
    # ------------------------------------------------------------------

    async def add_biomarker(
        self,
        patient_id: str,
        physician_id: str,
        biomarker_name: str,
        value: Optional[str] = None,
        measured_date: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_schema()
        if not await self.get_patient(patient_id, physician_id):
            return None

        db = get_patient_db()
        pool = await db.get_pool()
        biomarker_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_biomarkers
                        (id, patient_id, biomarker_name, value, measured_date, raw_text)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id, patient_id, biomarker_name, value,
                              canonical_code, measured_date, raw_text, created_at
                    """,
                    biomarker_id, patient_id, biomarker_name, value,
                    measured_date, raw_text,
                )
                await self._append_timeline_event(
                    conn, patient_id, "biomarker_update",
                    {"biomarker_name": biomarker_name, "value": value},
                    created_by=physician_id, event_date=measured_date,
                )
        return _row_to_dict(row)

    async def list_biomarkers(self, patient_id: str) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_id, biomarker_name, value, canonical_code,
                       measured_date, raw_text, created_at
                FROM patient_biomarkers
                WHERE patient_id = $1
                ORDER BY created_at DESC
                """,
                patient_id,
            )
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Treatment history
    # ------------------------------------------------------------------

    async def add_treatment(
        self,
        patient_id: str,
        physician_id: str,
        treatment_type: str,
        regimen: Optional[str] = None,
        line_of_therapy: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_schema()
        if not await self.get_patient(patient_id, physician_id):
            return None

        db = get_patient_db()
        pool = await db.get_pool()
        treatment_id = str(uuid.uuid4())

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO patient_treatment_history
                        (id, patient_id, treatment_type, regimen, line_of_therapy,
                         start_date, end_date, status, raw_text)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id, patient_id, treatment_type, regimen,
                              line_of_therapy, start_date, end_date, status,
                              raw_text, created_at
                    """,
                    treatment_id, patient_id, treatment_type, regimen,
                    line_of_therapy, start_date, end_date, status, raw_text,
                )
                await self._append_timeline_event(
                    conn, patient_id, "treatment_change",
                    {
                        "treatment_type": treatment_type, "regimen": regimen,
                        "status": status, "line_of_therapy": line_of_therapy,
                    },
                    created_by=physician_id, event_date=start_date,
                )
        return _row_to_dict(row)

    async def list_treatment_history(self, patient_id: str) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_id, treatment_type, regimen, line_of_therapy,
                       start_date, end_date, status, raw_text, created_at
                FROM patient_treatment_history
                WHERE patient_id = $1
                ORDER BY start_date DESC NULLS LAST, created_at DESC
                """,
                patient_id,
            )
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    async def get_timeline(self, patient_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        db = get_patient_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_id, event_type, event_date, recorded_at,
                       payload, source, created_by
                FROM patient_timeline_events
                WHERE patient_id = $1
                ORDER BY recorded_at DESC
                LIMIT $2
                """,
                patient_id, limit,
            )
        out = []
        for r in rows:
            d = _row_to_dict(r)
            if isinstance(d.get("payload"), str):
                try:
                    d["payload"] = json.loads(d["payload"])
                except (TypeError, ValueError):
                    pass
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    async def get_patient_full(self, patient_id: str, physician_id: str) -> Optional[Dict[str, Any]]:
        """Full patient snapshot: demographics + latest diagnosis + all
        biomarkers + treatment history + recent timeline. Used by the
        detail view and as the profile fed into patient_collection_seeder."""
        patient = await self.get_patient(patient_id, physician_id)
        if not patient:
            return None

        diagnosis = await self.get_latest_diagnosis(patient_id)
        biomarkers = await self.list_biomarkers(patient_id)
        treatment_history = await self.list_treatment_history(patient_id)
        timeline = await self.get_timeline(patient_id, limit=20)

        return {
            **patient,
            "diagnosis": diagnosis,
            "biomarkers": biomarkers,
            "treatment_history": treatment_history,
            "recent_timeline": timeline,
        }


_patient_service: Optional[PatientService] = None


def get_patient_service() -> PatientService:
    global _patient_service
    if _patient_service is None:
        _patient_service = PatientService()
    return _patient_service
