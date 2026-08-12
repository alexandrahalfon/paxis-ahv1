"""
Patient-owned data layer (Phase 0+ of the consumer-platform redesign).

Everything under this package is keyed by patient_profile_id — the
consumer-owned profile created at registration (see patient_profile_service)
— rather than by the physician-owned `patients.id` used throughout the
older src/api/services/patient_service.py and patient_db.py tables that
predate it. See patient_db.py's "Phase 0: patient-owned profile" schema
comment for the reasoning.

The older physician-owned patient_service.py is left untouched: existing
clinician-side chart CRUD keeps working exactly as it does today. This
package is additive, not a replacement.
"""
