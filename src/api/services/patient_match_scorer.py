"""
Patient–Study Match Scorer (v2)
===============================

Given a ``ClinicalProfile`` (the user's patient) and a study's
doc-level metadata (from Qdrant chunk payloads' ``metadata.doc_level_*``
fields), compute a 0–100 score expressing how closely the patients
enrolled in the study look like the user's patient.

Scoring formula
---------------
For each clinical axis the patient specified, compute:

    overlap_ratio = |patient_values ∩ study_values| / |patient_values|

Axes are weighted (``AXIS_WEIGHTS``). The denominator only counts
axes the study has data on — an axis the patient cares about but
where ``doc_level_*`` is empty is treated as ``not_assessed`` rather
than a 0% mismatch (most pre-2018 trials simply don't tag PD-L1 /
CPS; that should not penalise them vs. a study that tagged a
different biomarker).

    score = 100 * Σ(weight_i * overlap_i) / Σ(weight_i)
                                              over axes the study reports

To prevent a study with sparse metadata from scoring 100% from a
lucky single match, ``cancer_type`` is treated as a required axis:
if the patient specified one but the study has no cancer_type tags
*at all*, we fall back to the legacy denominator (counts all
patient-populated axes) so the missing-data discount can't game the
total.

If the patient specified a cancer_type and the study's cancer_type
list contains values but none match, we cap the final score at
``HARD_CANCER_TYPE_CAP`` — a wrong-cancer study cannot score higher
than that regardless of incidental overlap on other axes.

Biomarker comparison is substring-tolerant (case-insensitive) so the
patient's bare canonical 'CPS' will match a study's 'CPS positive'
or 'CPS ≥ 1' tags.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Weights & caps ──────────────────────────────────────────────────────

AXIS_WEIGHTS: Dict[str, int] = {
    "cancer_type":      25,
    "cancer_sites":     15,
    "histologies":      15,
    "stages":           10,
    "biomarkers":       15,
    "prior_treatments": 10,
    "disease_status":   10,
}

# If the patient's cancer_type is specified and the study doesn't match
# it, cap the overall score at this value. Prevents lung-cancer studies
# from scoring 80% because of incidental biomarker overlap.
HARD_CANCER_TYPE_CAP: int = 35


# Maps each ``ClinicalProfile`` attribute to the Qdrant doc-level
# payload field that should be compared against it. All doc_level_*
# fields are lists of canonical strings (or empty).
_AXIS_TO_DOCLEVEL: Dict[str, str] = {
    "cancer_type":      "doc_level_cancer_types",
    "cancer_sites":     "doc_level_sites",
    "histologies":      "doc_level_histologies",
    "stages":           "doc_level_stages",
    "biomarkers":       "doc_level_biomarkers",
    "prior_treatments": "doc_level_drugs",
    "disease_status":   "doc_level_disease_status",
}


# ── Public API ─────────────────────────────────────────────────────────

def score_patient_match(
    clinical_profile,
    doc_level_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Score one study against one patient profile.

    Args:
        clinical_profile: ``ClinicalProfile`` instance (or any duck-typed
            object with the expected axis attributes).
        doc_level_metadata: The ``metadata`` dict from a Qdrant chunk
            payload. Only the ``doc_level_*`` keys are read. Missing keys
            are treated as empty lists.

    Returns:
        A dict with keys:
          - ``score``: int 0–100
          - ``breakdown``: dict with per-axis detail so the UI can show
            what matched and what didn't:
              ``{axis: {"patient": [...], "study": [...],
                        "matched": [...], "ratio": 0.0–1.0,
                        "weight": int}}``
          - ``capped``: bool — True if the cancer_type cap was applied
          - ``axes_used``: list of axes that contributed to scoring
    """
    breakdown: Dict[str, Any] = {}
    axes_used: List[str] = []
    weighted_sum = 0.0
    weight_total = 0
    # Legacy denominator — counts every patient-populated axis even
    # when the study has no data. Used as a fallback when the study's
    # cancer_type tags are missing entirely, so a study can't game the
    # NA discount.
    legacy_weight_total = 0

    cancer_type_specified = bool(_patient_values_for_axis(clinical_profile, "cancer_type"))
    cancer_type_has_study_data = bool(_study_values_for_axis(doc_level_metadata, "cancer_type"))
    cancer_type_match: Optional[bool] = None

    for axis, weight in AXIS_WEIGHTS.items():
        patient_values = _patient_values_for_axis(clinical_profile, axis)
        if not patient_values:
            continue  # axis not specified by the patient — no signal

        study_values = _study_values_for_axis(doc_level_metadata, axis)
        matched = _match_axis_values(axis, patient_values, study_values)
        ratio = len(matched) / len(patient_values) if patient_values else 0.0
        study_has_data = bool(study_values)

        breakdown[axis] = {
            "patient": sorted(patient_values),
            "study":   sorted(study_values),
            "matched": sorted(matched),
            "ratio":   round(ratio, 3),
            "weight":  weight,
            "study_has_data": study_has_data,
        }

        axes_used.append(axis)
        legacy_weight_total += weight
        if study_has_data:
            # Counted in the strict denominator
            weighted_sum += weight * ratio
            weight_total += weight

        if axis == "cancer_type":
            cancer_type_match = bool(matched) if study_has_data else None

    # Pick the denominator:
    #  - If the study has cancer_type data: use the strict denominator
    #    (axes where study reported something). NA axes don't pull
    #    the score down.
    #  - If the study has no cancer_type data at all OR no axes have
    #    data at all: fall back to the legacy denominator so the
    #    discount can't inflate a sparse-metadata study to 100%.
    use_strict = (
        weight_total > 0
        and (not cancer_type_specified or cancer_type_has_study_data)
    )
    denominator = weight_total if use_strict else legacy_weight_total

    if denominator == 0:
        score = 0
    else:
        score = int(round(100.0 * weighted_sum / denominator))

    capped = False
    if cancer_type_match is False and score > HARD_CANCER_TYPE_CAP:
        score = HARD_CANCER_TYPE_CAP
        capped = True

    return {
        "score":     max(0, min(100, score)),
        "breakdown": breakdown,
        "capped":    capped,
        "axes_used": axes_used,
        "denominator_mode": "strict" if use_strict else "legacy",
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _patient_values_for_axis(profile, axis: str) -> List[str]:
    """Read the patient's values for this axis. Scalar axes return a
    single-item list so every comparison is list-vs-list."""
    if axis == "cancer_type":
        lbl = getattr(profile, "cancer_type_label", None)
        return [lbl] if lbl else []
    return list(getattr(profile, axis, []) or [])


def _study_values_for_axis(doc_level: Dict[str, Any], axis: str) -> List[str]:
    key = _AXIS_TO_DOCLEVEL.get(axis)
    if not key:
        return []
    values = doc_level.get(key) if isinstance(doc_level, dict) else None
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values if v]


def _intersect_ci(a: List[str], b: List[str]) -> List[str]:
    """Case-insensitive intersection, preserving the casing from ``a``."""
    b_lower = {str(x).strip().lower() for x in b if x}
    out: List[str] = []
    for x in a:
        if not x:
            continue
        if str(x).strip().lower() in b_lower:
            out.append(x)
    return out


def _match_axis_values(axis: str, patient: List[str], study: List[str]) -> List[str]:
    """Axis-aware comparison.

    Biomarkers and prior_treatments are matched substring-tolerant
    (case-insensitive) so a patient's bare 'CPS' matches a study's
    'CPS positive' / 'CPS ≥ 1', and 'pembrolizumab' matches
    'pembrolizumab 200mg q3w'. All other axes use strict equality.
    """
    if axis in ("biomarkers", "prior_treatments"):
        return _intersect_substring_ci(patient, study)
    return _intersect_ci(patient, study)


def _intersect_substring_ci(a: List[str], b: List[str]) -> List[str]:
    """Case-insensitive substring intersection.

    A patient value counts as matched if any study value contains it
    as a substring (or vice versa — covers both 'CPS' matching 'CPS
    positive' and 'pembrolizumab 200mg' matching 'pembrolizumab').
    """
    b_lower = [str(x).strip().lower() for x in b if x]
    out: List[str] = []
    for x in a:
        if not x:
            continue
        xl = str(x).strip().lower()
        if not xl:
            continue
        for bl in b_lower:
            if xl == bl or xl in bl or bl in xl:
                out.append(x)
                break
    return out
