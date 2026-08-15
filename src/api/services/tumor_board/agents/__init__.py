"""Specialty agents for the tumor board."""

from .medical_oncology import MedicalOncologyAgent
from .radiation_oncology import RadiationOncologyAgent
from .surgical_oncology import SurgicalOncologyAgent
from .pathology_molecular import PathologyMolecularAgent
from .radiology import RadiologyAgent
from .palliative_care import PalliativeCareAgent

ALL_AGENT_CLASSES = [
    MedicalOncologyAgent,
    RadiationOncologyAgent,
    SurgicalOncologyAgent,
    PathologyMolecularAgent,
    RadiologyAgent,
    PalliativeCareAgent,
]

__all__ = [
    "MedicalOncologyAgent",
    "RadiationOncologyAgent",
    "SurgicalOncologyAgent",
    "PathologyMolecularAgent",
    "RadiologyAgent",
    "PalliativeCareAgent",
    "ALL_AGENT_CLASSES",
]
