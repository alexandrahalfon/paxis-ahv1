"""
Eligibility safety layer.

Canonical import path for patient-eligibility filtering and boosting across
the RAG pipelines. The actual implementation lives in
`patient_eligibility_boost_service.py` (and a thinner variant in
`patient_trial_eligibility_service.py`); this module is a single façade so
new callers don't have to pick between them and existing callers can be
migrated incrementally.

The functions here bump `pipeline_metrics.eligibility.*` counters when
available (see `patient_eligibility_boost_service` call sites) so the
per-pipeline summary emitted by `pipeline_metrics.summary_line()` includes
eligibility verdict counts for every surface that calls this layer.
"""

from src.api.services.patient_eligibility_boost_service import (
    apply_patient_eligibility_boost,
    apply_patient_eligibility_filter_and_boost,
    build_patient_summary,
    check_patient_eligibility_for_studies,
    extract_patient_context_from_query,
    run_patient_eligibility_check,
)

__all__ = [
    "apply_patient_eligibility_boost",
    "apply_patient_eligibility_filter_and_boost",
    "build_patient_summary",
    "check_patient_eligibility_for_studies",
    "extract_patient_context_from_query",
    "run_patient_eligibility_check",
]
