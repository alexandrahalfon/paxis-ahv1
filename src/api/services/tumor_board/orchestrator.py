"""
TumorBoardOrchestrator — dispatches a single patient case to N specialty
agents in parallel and assembles a multi-perspective TumorBoardReport.

The orchestrator's only jobs:
    1. Build a PatientCaseBundle ONCE from the raw narrative.
    2. asyncio.gather() over every SpecialtyAgent.evaluate().
    3. Convert any failures into skipped-or-errored ExpertAssessments so
       that the API response is always shape-stable.
    4. Package everything into a TumorBoardReport with metadata.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_agent import ExpertAssessment, SpecialtyAgent
from .case_bundle import PatientCaseBundle, build_case_bundle
from .agents import ALL_AGENT_CLASSES


@dataclass
class TumorBoardReport:
    """Final output of one tumor-board run."""

    case_summary: List[str]
    expert_assessments: List[ExpertAssessment]
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_summary": self.case_summary,
            "expert_assessments": [a.to_dict() for a in self.expert_assessments],
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }


class TumorBoardOrchestrator:
    """
    Orchestrates parallel specialty-agent evaluation over a single case.

    Usage:
        orchestrator = TumorBoardOrchestrator()
        report = await orchestrator.present_case(case_text)

    Agents do their own lightweight Qdrant retrieval via
    `tumor_board.retrieval.lightweight_search` — the orchestrator does not
    own any retriever directly. The `retriever` constructor arg is
    accepted for legacy / test compatibility and is ignored.
    """

    def __init__(
        self,
        agents: Optional[List[SpecialtyAgent]] = None,
        retriever: Optional[Any] = None,  # legacy, unused
    ):
        self.agents: List[SpecialtyAgent] = (
            agents if agents is not None
            else [cls() for cls in ALL_AGENT_CLASSES]
        )
        # Accepted for backwards compat — tests that pass a FakeRetriever
        # should now instead monkey-patch `agent._retrieve` directly.
        self._retriever = retriever

    async def present_case(
        self,
        case_text: str,
        query_type: str = "treatment_recommendation",
    ) -> TumorBoardReport:
        """Run the tumor board on a raw clinical narrative."""
        if not case_text or not case_text.strip():
            raise ValueError("case_text must not be empty")

        t_start = time.perf_counter()

        try:
            from src.api.services import pipeline_metrics as _pm
            if _pm.current() is None:
                _pm.start("p4")
        except Exception:
            pass

        print("\n" + "=" * 80)
        print("  TUMOR BOARD: present_case()")
        print("=" * 80)
        print(f"  Case ({len(case_text)} chars): "
              f"{case_text[:160]}{'...' if len(case_text) > 160 else ''}")
        print(f"  Agents on panel: {[a.display_name for a in self.agents]}")

        # 1. Build the bundle once — all agents share it
        t_bundle_start = time.perf_counter()
        bundle = await build_case_bundle(case_text, query_type=query_type)
        t_bundle_ms = (time.perf_counter() - t_bundle_start) * 1000
        print(f"  [TumorBoard] Case bundle built in {t_bundle_ms:.0f}ms "
              f"(llm_extraction={bundle.used_llm_extraction}, "
              f"category={bundle.category})")

        if bundle.has_patient_context:
            try:
                from src.api.services import pipeline_metrics as _pm
                if _pm.current() is not None:
                    _pm.current().event("has_patient_context")
            except Exception:
                pass

        # 2. Dispatch all agents in parallel
        tasks = [agent.evaluate(bundle) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Convert exceptions into error assessments so the response is
        #    always shape-stable
        assessments: List[ExpertAssessment] = []
        for agent, result in zip(self.agents, results):
            if isinstance(result, ExpertAssessment):
                assessments.append(result)
            else:
                err = f"{type(result).__name__}: {result}"
                print(f"  [TumorBoard] {agent.specialty} raised: {err}")
                assessments.append(
                    ExpertAssessment.error_assessment(
                        specialty=agent.specialty,
                        display_name=agent.display_name,
                        error=err,
                    )
                )

        total_ms = (time.perf_counter() - t_start) * 1000

        # Small logging summary
        for a in assessments:
            status = "SKIPPED" if a.skipped else ("ERROR" if a.error else a.recommendation.upper())
            print(
                f"  [TumorBoard] {a.display_name:<25} "
                f"{status:<22} "
                f"conf={a.confidence:.2f} "
                f"({a.elapsed_ms:.0f}ms)"
            )

        metadata = {
            "total_elapsed_ms": total_ms,
            "case_bundle_elapsed_ms": t_bundle_ms,
            "used_llm_extraction": bundle.used_llm_extraction,
            "agent_count": len(self.agents),
            "agents_run": [a.specialty for a in self.agents],
            "trajectory_flags": bundle.trajectory_flags,
            "metastatic_sites": bundle.metastatic_sites,
            "surgical_candidate": bundle.surgical_candidate,
        }

        try:
            from src.api.services import pipeline_metrics as _pm
            _pm_cur = _pm.current()
            if _pm_cur is not None:
                print(_pm_cur.summary_line())
        except Exception:
            pass

        return TumorBoardReport(
            case_summary=bundle.summary_lines(),
            expert_assessments=assessments,
            raw_text=case_text,
            metadata=metadata,
        )


# ─── Singleton ──────────────────────────────────────────────────────────────

_orchestrator_instance: Optional[TumorBoardOrchestrator] = None


def get_tumor_board_orchestrator() -> TumorBoardOrchestrator:
    """Return a lazily-constructed TumorBoardOrchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TumorBoardOrchestrator()
    return _orchestrator_instance
