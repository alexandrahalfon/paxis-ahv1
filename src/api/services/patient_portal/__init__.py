"""
Patient portal services (added 2026-08-08).

Everything patient-account-facing lives in this package so the physician
product is untouched. Nothing in here modifies existing services; the
patient side reuses the same retrieval pipeline and the same patient
records, only with different entry points and safeguards.
"""
