"""
Tumor Board Multi-Agent Architecture

A parallel orchestration layer that simulates a multidisciplinary tumor
board by dispatching the same structured patient case to specialty
agents (Medical Oncology, Radiation Oncology, Surgical Oncology,
Pathology/Molecular, Radiology, Palliative Care). Each agent generates
specialty-specific sub-queries against the existing RAG pipeline and
returns a structured assessment with supporting / conflicting evidence.

Runs alongside the existing /rag pipeline; exposed at /api/tumor-board.
"""

from .case_bundle import PatientCaseBundle, build_case_bundle
from .base_agent import SpecialtyAgent, ExpertAssessment, StudyCitation
from .orchestrator import TumorBoardOrchestrator, TumorBoardReport

__all__ = [
    "PatientCaseBundle",
    "build_case_bundle",
    "SpecialtyAgent",
    "ExpertAssessment",
    "StudyCitation",
    "TumorBoardOrchestrator",
    "TumorBoardReport",
]
