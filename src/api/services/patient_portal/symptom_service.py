"""
Symptom diary.

What it is: a patient logs what they noticed, how bad it was (1-5), and
when. Nothing more. It is deliberately not a symptom checker and gives no
assessment back, because "should I be worried about this" is triage and
triage is the highest-risk thing a patient-facing tool can do.

What it is for: turning "the last few weeks have been rough" into
something specific at the next appointment. Patients routinely
under-report in clinic because they cannot remember. A dated list fixes
that, and it is the kind of thing an oncologist actually wants.

Two safeguards:

* Every entry runs through the same triage as the chat. Someone logging
  "chest pain" gets the urgent-care response immediately rather than a
  tidy row in a diary.
* Sharing with the physician is explicit, never automatic. The summary
  goes through the existing escalation queue, so it lands in the inbox
  they already check.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

from src.api.services.patient_db import get_patient_db
from src.api.services.patient_portal import patient_safety_service as safety

logger = logging.getLogger(__name__)

SEVERITY_LABELS = {
    1: "barely noticeable",
    2: "mild",
    3: "moderate",
    4: "bad",
    5: "severe",
}


class SymptomService:
    async def add(
        self,
        patient_user_id: str,
        symptom: str,
        severity: Optional[int] = None,
        noted_on: Optional[str] = None,
        note: Optional[str] = None,
        patient_record_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log one entry. Returns the entry plus any safety response.

        The triage result is returned rather than raised so the caller can
        both save the entry and show the urgent message: someone logging
        something serious should still have it recorded.
        """
        text = (symptom or "").strip()
        if not text:
            raise ValueError("Please describe the symptom.")
        if severity is not None and not (1 <= int(severity) <= 5):
            raise ValueError("Severity should be between 1 and 5.")

        combined = f"{text} {note or ''}".strip()
        tri = safety.triage(combined)

        entry_id = str(uuid.uuid4())
        on = None
        if noted_on:
            try:
                on = date.fromisoformat(str(noted_on)[:10])
            except Exception:
                on = None

        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO patient_symptom_entries
                    (id, patient_user_id, patient_record_id, symptom,
                     severity, noted_on, note)
                VALUES ($1, $2, $3, $4, $5, COALESCE($6, CURRENT_DATE), $7)
                """,
                entry_id, patient_user_id, patient_record_id, text,
                int(severity) if severity is not None else None, on, note,
            )

        return {
            "id": entry_id,
            "symptom": text,
            "severity": severity,
            "noted_on": (on or date.today()).isoformat(),
            "note": note,
            "safety_category": tri.category,
            "safety_message": safety.emergency_response(tri) if tri.blocks_answer else None,
        }

    async def list_entries(
        self, patient_user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, symptom, severity, noted_on, note, created_at
                  FROM patient_symptom_entries
                 WHERE patient_user_id = $1
                 ORDER BY noted_on DESC, created_at DESC
                 LIMIT $2
                """,
                patient_user_id, limit,
            )
        return [
            {
                "id": str(r["id"]),
                "symptom": r["symptom"],
                "severity": r["severity"],
                "severity_label": SEVERITY_LABELS.get(r["severity"] or 0),
                "noted_on": r["noted_on"].isoformat() if r["noted_on"] else None,
                "note": r["note"],
            }
            for r in rows
        ]

    async def delete(self, entry_id: str, patient_user_id: str) -> bool:
        """Ownership-scoped delete. Patients should be able to remove
        something they logged by mistake."""
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM patient_symptom_entries
                 WHERE id = $1 AND patient_user_id = $2
                """,
                entry_id, patient_user_id,
            )
        return result.endswith("1")

    @staticmethod
    def build_summary(entries: List[Dict[str, Any]]) -> str:
        """Plain-text summary for the care team, newest first.

        Grouped by symptom so a physician sees "nausea, 6 times over 3
        weeks, worst 4/5" rather than a raw log they have to parse.
        """
        if not entries:
            return "No symptoms logged."

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            grouped.setdefault(e["symptom"].strip().lower(), []).append(e)

        lines = []
        for name, items in sorted(
            grouped.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            sevs = [i["severity"] for i in items if i.get("severity")]
            dates = sorted(i["noted_on"] for i in items if i.get("noted_on"))
            bits = [f"{name} ({len(items)}x"]
            if dates:
                bits.append(
                    f" {dates[0]} to {dates[-1]}" if dates[0] != dates[-1]
                    else f" on {dates[0]}"
                )
            if sevs:
                bits.append(f", worst {max(sevs)}/5")
            bits.append(")")
            line = "".join(bits)
            notes = [i["note"] for i in items if i.get("note")]
            if notes:
                line += f" - {notes[0][:120]}"
            lines.append(line)

        return "Patient-reported symptoms:\n" + "\n".join(f"- {l}" for l in lines)


_service: Optional[SymptomService] = None


def get_symptom_service() -> SymptomService:
    global _service
    if _service is None:
        _service = SymptomService()
    return _service
