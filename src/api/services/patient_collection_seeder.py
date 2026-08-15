"""
Patient Collection Seeder (Phase 2 of the patient-centric pivot)

Builds a query from a patient's current profile and runs it through the
existing enhanced_rag_service pipeline unchanged, then saves the top
matches as patient-scoped evidence via saved_studies_service. This is the
only new piece of retrieval-adjacent code in Phase 2 — comprehensive_retrieval.py,
enhanced_rag_service.py, soft_scorer.py, etc. are all reused as-is per the
"do not change" list.

Called from POST /api/patients/{patient_id}/seed (src/api/routes/patient_cases.py),
and will also be the re-seed trigger for Phase 4 continuous monitoring
(pattern_diff_service, not yet built) once a timeline event indicates the
profile changed materially.
"""

from typing import Any, Dict, List, Optional


def _age_from_dob(dob: Any) -> Optional[int]:
    """Compute age in years from a date/datetime/ISO-string DOB.

    Returns None for missing or unparseable values — never raises, since
    seeding must not fail on a malformed DOB.
    """
    if not dob:
        return None
    try:
        from datetime import date, datetime
        if isinstance(dob, datetime):
            born = dob.date()
        elif isinstance(dob, date):
            born = dob
        else:
            born = date.fromisoformat(str(dob)[:10])
        today = date.today()
        age = today.year - born.year - (
            (today.month, today.day) < (born.month, born.day)
        )
        return age if 0 <= age <= 130 else None
    except Exception:
        return None


def _build_query_from_profile(patient_profile: Dict[str, Any]) -> str:
    """Turn a patient_service.get_patient_full() dict into a free-text
    query string for enhanced_rag_service.query(). Mirrors the kind of
    narrative the existing query pipeline already expects (see the test
    patient profile in CLAUDE.md) rather than inventing a new input shape.
    """
    # Demographics are tracked separately from clinical content so the
    # caller can tell "demographics only" (nothing worth searching for)
    # apart from a real clinical profile. Before age was included, a
    # DOB-only patient produced an empty string and hit the early exit;
    # now it would produce "80 y.o." and trigger a full, pointless
    # retrieval. See _profile_has_clinical_content().
    parts: List[str] = []

    dob = patient_profile.get("date_of_birth")
    sex = patient_profile.get("sex")
    demo_bits = []
    # Age from DOB — previously read and dropped, so an 80-year-old and a
    # 40-year-old with identical diagnoses seeded identical collections.
    # The pipeline extracts and matches on age, and its expected narrative
    # format leads with it ("80 y.o. male ...").
    age = _age_from_dob(dob)
    if age is not None:
        demo_bits.append(f"{age} y.o.")
    if sex:
        demo_bits.append(sex)
    if demo_bits:
        parts.append(" ".join(demo_bits))

    diagnosis = patient_profile.get("diagnosis") or {}
    if diagnosis:
        dx_bits = []
        if diagnosis.get("stage"):
            dx_bits.append(f"Stage {diagnosis['stage']}")
        if diagnosis.get("histology"):
            dx_bits.append(diagnosis["histology"])
        if diagnosis.get("cancer_site"):
            dx_bits.append(f"of the {diagnosis['cancer_site']}")
        tnm = "".join(filter(None, [diagnosis.get("tnm_t"), diagnosis.get("tnm_n"), diagnosis.get("tnm_m")]))
        if tnm:
            dx_bits.append(f"({tnm})")
        if dx_bits:
            parts.append(" ".join(dx_bits))
        if diagnosis.get("raw_text"):
            parts.append(diagnosis["raw_text"])

    biomarkers = patient_profile.get("biomarkers") or []
    for b in biomarkers:
        name = b.get("biomarker_name")
        value = b.get("value")
        if name and value:
            parts.append(f"{name} {value}")
        elif name:
            parts.append(name)

    treatment_history = patient_profile.get("treatment_history") or []
    for t in treatment_history:
        bits = [t.get("treatment_type"), t.get("regimen"), t.get("status")]
        bits = [b for b in bits if b]
        if bits:
            parts.append(" ".join(bits))
        if t.get("raw_text"):
            parts.append(t["raw_text"])

    return ". ".join(p for p in parts if p).strip()


def _profile_has_clinical_content(patient_profile: Dict[str, Any]) -> bool:
    """True when the profile has something worth searching the literature for.

    Age and sex alone are not enough: seeding on "80 y.o. male" returns
    noise and costs a full retrieval. Requires at least one populated
    diagnosis field, biomarker, or treatment-history entry.
    """
    diagnosis = patient_profile.get("diagnosis") or {}
    if any(
        diagnosis.get(k)
        for k in ("stage", "histology", "cancer_site", "tnm_t", "tnm_n", "tnm_m", "raw_text")
    ):
        return True
    if patient_profile.get("biomarkers"):
        return True
    if patient_profile.get("treatment_history"):
        return True
    return False


class PatientCollectionSeeder:
    """Builds a patient-profile query and seeds matched studies."""

    async def seed_patient_collection(
        self,
        patient_id: str,
        physician_id: str,
        patient_profile: Dict[str, Any],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        query_text = _build_query_from_profile(patient_profile)
        if not query_text or not _profile_has_clinical_content(patient_profile):
            return {
                "success": False,
                "message": "Patient profile has no diagnosis, biomarkers, or "
                            "treatment history yet, nothing to seed from.",
                "query_text": "",
                "matched": [],
            }

        # Retrieval-only path. The previous implementation called the full
        # rag_service.query(), which runs GPT-4o answer generation and then
        # threw the answer away — every patient intake with auto-seed paid
        # several seconds and an LLM bill for text nobody sees. The
        # retrieval backbone returns the same matched studies without
        # generation.
        from src.api.services.retrieval_backbone import retrieve_evidence
        from src.api.services.saved_studies_service import get_saved_studies_service

        bundle = await retrieve_evidence(
            query_text,
            mode="comprehensive",
            max_studies=top_k,
        )

        studies_service = get_saved_studies_service()

        saved: List[Dict[str, Any]] = []
        for s in bundle.studies[:top_k]:
            doc_id = s.doc_id
            if not doc_id:
                continue
            # doi/pmid live in chunk-level doc_meta when present
            doc_meta = (s.chunks[0].get("doc_meta") or {}) if s.chunks else {}
            saved_study = await studies_service.save_study_for_patient(
                patient_id=patient_id,
                user_id=physician_id,
                study_id=doc_id,
                title=s.title,
                doi=doc_meta.get("doi"),
                pmid=doc_meta.get("pmid"),
                source="core",
                auto_seeded=True,
            )
            saved.append(saved_study)

        return {
            "success": True,
            "query_text": query_text,
            "matched_count": len(saved),
            "matched": saved,
        }


_seeder: Optional[PatientCollectionSeeder] = None


def get_patient_collection_seeder() -> PatientCollectionSeeder:
    global _seeder
    if _seeder is None:
        _seeder = PatientCollectionSeeder()
    return _seeder
