"""
Unit tests for the multi-agent tumor board orchestrator and its specialty
agents. These tests run entirely offline — no Qdrant, OpenAI, or network
access — by:

  - Building a QueryStructure directly and handing it to a PatientCaseBundle
  - Mocking the ComprehensiveRetriever so agents see fake StudyEvidence
  - Monkey-patching the LLM synthesis hook so SpecialtyAgent._synthesize()
    does not touch the network.

The goal of these tests is to lock in:
  1. The canonical 80 y.o. oral-tongue SCC case extracts the right fields.
  2. Each specialty agent generates queries that reflect its specialty lens.
  3. The orchestrator dispatches all agents in parallel and handles failure.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Make the repository root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.services.query_structuring_service import (  # noqa: E402
    CancerContext,
    ClinicalHistory,
    PatientContext,
    QueryStructure,
    TreatmentContext,
    structure_query_fast,
)
from src.api.services.clinical_inference import (  # noqa: E402
    apply_inference_to_query_structure,
)
from src.api.services.tumor_board.agents import (  # noqa: E402
    MedicalOncologyAgent,
    PalliativeCareAgent,
    PathologyMolecularAgent,
    RadiationOncologyAgent,
    RadiologyAgent,
    SurgicalOncologyAgent,
)
from src.api.services.tumor_board.base_agent import (  # noqa: E402
    ExpertAssessment,
    RECOMMENDATION_VERDICTS,
    SpecialtyAgent,
    StudyCitation,
)
from src.api.services.tumor_board.case_bundle import PatientCaseBundle  # noqa: E402
from src.api.services.tumor_board.orchestrator import (  # noqa: E402
    TumorBoardOrchestrator,
    TumorBoardReport,
)
from src.api.services.tumor_board.retrieval import LightweightStudy  # noqa: E402
from tests.fixtures.tumor_board_cases import (  # noqa: E402
    CANONICAL_ORAL_TONGUE_SCC_CASE,
    EMPTY_CASE,
    SIMPLE_DOSE_QUERY,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _build_canonical_bundle() -> PatientCaseBundle:
    """Build a PatientCaseBundle from the canonical case using regex-only
    structuring (no LLM) and then running the clinical-inference pass.

    This mirrors what `build_case_bundle()` does without triggering the
    async LLM extractor — tests stay fast and offline.
    """
    structure = structure_query_fast(
        CANONICAL_ORAL_TONGUE_SCC_CASE,
        query_type="treatment_recommendation",
    )
    # The regex extractor doesn't detect some comorbidities directly; add
    # them so downstream filters can see them. The full production path
    # populates these through the LLM 8-axis extractor.
    if "CKD" not in structure.patient.comorbidities:
        structure.patient.comorbidities.append("CKD")
    if "Hep C" not in structure.patient.comorbidities:
        structure.patient.comorbidities.append("Hep C")

    inferred_axes = apply_inference_to_query_structure(
        structure, CANONICAL_ORAL_TONGUE_SCC_CASE
    )
    return PatientCaseBundle(
        raw_text=CANONICAL_ORAL_TONGUE_SCC_CASE,
        structure=structure,
        inferred_axes=inferred_axes,
        used_llm_extraction=False,
    )


def _fake_study(doc_id: str, title: str, score: float = 0.8) -> LightweightStudy:
    """Construct a LightweightStudy for tests."""
    return LightweightStudy(
        doc_id=doc_id,
        title=title,
        citation=f"{title} (J Fake Onc 2024)",
        year=2024,
        initial_score=score,
        rerank_score=score,
        chunks=[
            {"section": "results", "text": f"Mock result text for {title}.", "score": score},
        ],
    )


def _patch_retrieve(
    agent: SpecialtyAgent, studies_per_call: int = 3
) -> List[str]:
    """Replace the agent's `_retrieve` method with a deterministic stub
    that returns N LightweightStudy objects per sub-query. Returns a
    shared `calls` list so tests can inspect which queries were run."""
    calls: List[str] = []

    async def _fake_retrieve(self, query_text: str, category=None) -> List[LightweightStudy]:  # noqa: ARG001
        calls.append(query_text)
        return [
            _fake_study(
                doc_id=f"doc_{abs(hash((query_text, i))) % 10_000}",
                title=f"Result {i} for {query_text[:40]}",
                score=0.9 - 0.1 * i,
            )
            for i in range(studies_per_call)
        ]

    agent._retrieve = types.MethodType(_fake_retrieve, agent)  # type: ignore[method-assign]
    return calls


def _patch_synth(agent: SpecialtyAgent, recommendation: str = "conditional") -> None:
    """Replace the LLM synthesis call with an offline stub that builds a
    deterministic ExpertAssessment from the studies the base class just
    retrieved."""
    async def _fake_synth(bundle, sub_queries, studies):
        citations = [
            StudyCitation(
                doc_id=s.doc_id,
                title=s.title,
                citation=s.citation,
                year=s.year,
                relevance_score=float(s.rerank_score or 0),
            )
            for s in studies[:3]
        ]
        return ExpertAssessment(
            specialty=agent.specialty,
            display_name=agent.display_name,
            recommendation=recommendation,
            recommendation_text=f"Offline stub assessment for {agent.display_name}.",
            confidence=0.7,
            key_questions=[f"{agent.display_name} question?"],
            supporting_studies=citations,
            conflicting_studies=[],
            next_steps=[f"Offline next step for {agent.display_name}"],
        )

    agent._synthesize = _fake_synth  # type: ignore[method-assign]


# ────────────────────────────────────────────────────────────────────────────
# 1. Case-bundle extraction
# ────────────────────────────────────────────────────────────────────────────


class TestCaseBundleCanonicalCase:
    """The canonical 80 y.o. case must extract the minimum fields the
    specialty agents rely on."""

    def test_basic_demographics_and_cancer(self):
        bundle = _build_canonical_bundle()
        assert bundle.patient.age == 80
        assert bundle.patient.gender == "male"
        # head_neck is the category, oral_tongue is the site_detail
        assert (bundle.cancer.site == "head_neck") or bundle.cancer.site_detail
        assert bundle.cancer.histology == "scc"

    def test_inference_flags_ici_refractory_and_unresectable(self):
        bundle = _build_canonical_bundle()
        assert "ici_refractory" in bundle.trajectory_flags, (
            f"expected ici_refractory in {bundle.trajectory_flags}"
        )
        # surgical_candidate should be False (no longer a surgical candidate)
        assert bundle.surgical_candidate is False

    def test_inference_detects_cardiac_metastasis(self):
        bundle = _build_canonical_bundle()
        met_sites_lower = " ".join(bundle.metastatic_sites).lower()
        # The inference layer tags either "right ventricle" or "cardiac"
        assert ("right ventricle" in met_sites_lower
                or "cardiac" in met_sites_lower), (
            f"expected cardiac/right-ventricle flag in {bundle.metastatic_sites}"
        )

    def test_summary_lines_are_non_empty(self):
        bundle = _build_canonical_bundle()
        lines = bundle.summary_lines()
        assert len(lines) >= 3
        joined = " ".join(lines).lower()
        assert "scc" in joined or "carcinoma" in joined

    def test_has_patient_context(self):
        bundle = _build_canonical_bundle()
        assert bundle.has_patient_context is True


# ────────────────────────────────────────────────────────────────────────────
# 2. Per-specialty sub-query generation
# ────────────────────────────────────────────────────────────────────────────


class TestSpecialtySubQueries:
    """Each agent should produce sub-queries aligned with its specialty."""

    @pytest.fixture(scope="class")
    def bundle(self) -> PatientCaseBundle:
        return _build_canonical_bundle()

    @staticmethod
    def _joined(agent: SpecialtyAgent, bundle: PatientCaseBundle) -> str:
        queries = agent.build_sub_queries(bundle)
        assert queries, f"{agent.specialty} produced no sub-queries"
        return " || ".join(q.lower() for q in queries)

    def test_medical_oncology_hits_ici_and_ckd(self, bundle):
        agent = MedicalOncologyAgent()
        joined = self._joined(agent, bundle)
        assert "ici" in joined or "anti-pd1" in joined or "checkpoint" in joined
        assert "cisplatin" in joined or "carboplatin" in joined or "ckd" in joined

    def test_radiation_oncology_mentions_reirradiation_and_neck_constraints(self, bundle):
        agent = RadiationOncologyAgent()
        joined = self._joined(agent, bundle)
        # No prior RT is stated explicitly, but agent should still offer
        # either palliative RT (because unresectable) or head/neck OAR
        # queries (because of site).
        assert "palliative" in joined or "reirradiation" in joined or "sbrt" in joined
        assert "carotid" in joined or "mandible" in joined or "head" in joined

    def test_surgical_oncology_flags_salvage_or_unresectable(self, bundle):
        agent = SurgicalOncologyAgent()
        joined = self._joined(agent, bundle)
        assert "salvage" in joined or "unresectable" in joined
        # Reconstructed-field surgery concern
        assert "flap" in joined or "reconstruction" in joined or "radiation" in joined

    def test_pathology_requests_ngs_and_primary_resistance(self, bundle):
        agent = PathologyMolecularAgent()
        joined = self._joined(agent, bundle)
        assert "ngs" in joined or "molecular" in joined or "her2" in joined
        # Biomarkers in the canonical case include CPS 100 (detected via
        # the regex biomarker list)
        biomarkers = " ".join(bundle.biomarkers).upper()
        if "CPS" in biomarkers or "PD-L1" in biomarkers:
            assert "primary resistance" in joined or "pembrolizumab" in joined

    def test_radiology_questions_cardiac_lesion(self, bundle):
        agent = RadiologyAgent()
        joined = self._joined(agent, bundle)
        assert "cardiac" in joined or "ventricle" in joined or "thrombus" in joined

    def test_palliative_flags_hospice_and_symptom_burden(self, bundle):
        agent = PalliativeCareAgent()
        joined = self._joined(agent, bundle)
        assert "prognosis" in joined or "hospice" in joined
        assert "palliative" in joined


# ────────────────────────────────────────────────────────────────────────────
# 3. Relevance filters
# ────────────────────────────────────────────────────────────────────────────


class TestRelevanceFilters:
    """Agents should skip cases they cannot meaningfully evaluate."""

    def test_pathology_skips_when_no_histology_or_biomarker(self):
        # Construct a bundle with patient context but no histology / biomarkers
        structure = QueryStructure(
            original_query="55 year old male with a colon mass",
        )
        structure.patient = PatientContext(age=55, gender="male")
        structure.cancer = CancerContext(site="gi")
        structure.has_patient_context = True
        bundle = PatientCaseBundle(
            raw_text="55 year old male with a colon mass",
            structure=structure,
            inferred_axes={},
        )
        agent = PathologyMolecularAgent()
        skip_reason = agent.relevance_filter(bundle)
        assert skip_reason is not None
        assert "histology" in skip_reason.lower()

    def test_all_agents_skip_on_empty_case(self):
        empty = QueryStructure(original_query="")
        bundle = PatientCaseBundle(
            raw_text="",
            structure=empty,
            inferred_axes={},
        )
        for agent_cls in (MedicalOncologyAgent, RadiationOncologyAgent,
                          SurgicalOncologyAgent, PalliativeCareAgent,
                          RadiologyAgent, PathologyMolecularAgent):
            agent = agent_cls()
            assert agent.relevance_filter(bundle) is not None


# ────────────────────────────────────────────────────────────────────────────
# 4. Orchestrator behaviour
# ────────────────────────────────────────────────────────────────────────────


class TestOrchestrator:
    """End-to-end orchestrator behaviour with fake retriever + LLM."""

    @pytest.mark.asyncio
    async def test_orchestrator_dispatches_all_agents_in_parallel(self, monkeypatch):
        bundle = _build_canonical_bundle()

        # Patch build_case_bundle to return the pre-built bundle (skip LLM)
        async def _fake_build(case_text, query_type="treatment_recommendation"):
            return bundle

        monkeypatch.setattr(
            "src.api.services.tumor_board.orchestrator.build_case_bundle",
            _fake_build,
        )

        agents = [
            MedicalOncologyAgent(),
            RadiationOncologyAgent(),
            SurgicalOncologyAgent(),
            PathologyMolecularAgent(),
            RadiologyAgent(),
            PalliativeCareAgent(),
        ]
        all_calls: List[str] = []
        for a in agents:
            all_calls.extend([])  # placeholder
            retrieve_calls = _patch_retrieve(a)
            # Track the underlying list so we can assert after the run
            a._test_calls = retrieve_calls  # type: ignore[attr-defined]
            _patch_synth(a, recommendation="conditional")

        orchestrator = TumorBoardOrchestrator(agents=agents)
        report = await orchestrator.present_case(CANONICAL_ORAL_TONGUE_SCC_CASE)

        assert isinstance(report, TumorBoardReport)
        assert len(report.expert_assessments) == 6
        # Every assessment should have a legitimate verdict
        for a in report.expert_assessments:
            assert a.recommendation in RECOMMENDATION_VERDICTS
        # All specialties should have run
        specialties = {a.specialty for a in report.expert_assessments}
        assert specialties == {
            "medical_oncology",
            "radiation_oncology",
            "surgical_oncology",
            "pathology_molecular",
            "radiology",
            "palliative_care",
        }
        # Each agent should have fired at least one lightweight retrieval
        total_retrieval_calls = sum(len(a._test_calls) for a in agents)  # type: ignore[attr-defined]
        assert total_retrieval_calls >= 6

    @pytest.mark.asyncio
    async def test_orchestrator_converts_agent_exception_into_error_assessment(
        self, monkeypatch
    ):
        bundle = _build_canonical_bundle()

        async def _fake_build(case_text, query_type="treatment_recommendation"):
            return bundle

        monkeypatch.setattr(
            "src.api.services.tumor_board.orchestrator.build_case_bundle",
            _fake_build,
        )

        # Good agent: normal stub
        good = MedicalOncologyAgent()
        _patch_retrieve(good)
        _patch_synth(good)

        # Bad agent: retrieval succeeds but synthesize blows up
        bad = RadiationOncologyAgent()
        _patch_retrieve(bad)

        async def _boom(*a, **kw):
            raise RuntimeError("simulated LLM failure")

        bad._synthesize = _boom  # type: ignore[method-assign]

        orchestrator = TumorBoardOrchestrator(agents=[good, bad])
        report = await orchestrator.present_case(CANONICAL_ORAL_TONGUE_SCC_CASE)

        assert len(report.expert_assessments) == 2
        assessments = {a.specialty: a for a in report.expert_assessments}

        good_a = assessments["medical_oncology"]
        assert good_a.error is None
        assert good_a.recommendation == "conditional"

        bad_a = assessments["radiation_oncology"]
        # The base class catches the exception inside evaluate() and turns
        # it into an error_assessment (not an orchestrator-level error)
        assert bad_a.error is not None
        assert "simulated" in bad_a.error.lower()
        assert bad_a.recommendation == "insufficient_evidence"

    @pytest.mark.asyncio
    async def test_orchestrator_runs_agents_concurrently(self, monkeypatch):
        """All agents sleeping 0.5s should finish in well under 6*0.5s."""
        bundle = _build_canonical_bundle()

        async def _fake_build(case_text, query_type="treatment_recommendation"):
            return bundle

        monkeypatch.setattr(
            "src.api.services.tumor_board.orchestrator.build_case_bundle",
            _fake_build,
        )

        agents = [MedicalOncologyAgent(), RadiationOncologyAgent(),
                  SurgicalOncologyAgent(), PathologyMolecularAgent(),
                  RadiologyAgent(), PalliativeCareAgent()]

        async def _slow_evaluate(self, bundle, retriever=None):
            await asyncio.sleep(0.5)
            return ExpertAssessment(
                specialty=self.specialty,
                display_name=self.display_name,
                recommendation="conditional",
                recommendation_text="slow stub",
                confidence=0.5,
            )

        for a in agents:
            a.evaluate = types.MethodType(_slow_evaluate, a)  # type: ignore[method-assign]

        orchestrator = TumorBoardOrchestrator(agents=agents)

        t0 = time.perf_counter()
        await orchestrator.present_case(CANONICAL_ORAL_TONGUE_SCC_CASE)
        elapsed = time.perf_counter() - t0
        # Sequential would take ~3.0s (6 × 0.5); parallel should be < 1.5s
        assert elapsed < 1.5, f"agents appear sequential — elapsed={elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_orchestrator_rejects_empty_case(self, monkeypatch):
        orchestrator = TumorBoardOrchestrator(agents=[MedicalOncologyAgent()])
        with pytest.raises(ValueError):
            await orchestrator.present_case(EMPTY_CASE)


# ────────────────────────────────────────────────────────────────────────────
# 5. Base agent integration — the evaluate() flow
# ────────────────────────────────────────────────────────────────────────────


class TestBaseAgentEvaluateFlow:
    """Exercise the common evaluate() path using one concrete agent."""

    @pytest.mark.asyncio
    async def test_evaluate_happy_path_returns_assessment_with_citations(self):
        bundle = _build_canonical_bundle()
        agent = MedicalOncologyAgent()
        _patch_retrieve(agent, studies_per_call=4)
        _patch_synth(agent, recommendation="favor")

        result = await agent.evaluate(bundle)

        assert isinstance(result, ExpertAssessment)
        assert result.specialty == "medical_oncology"
        assert result.recommendation == "favor"
        assert result.confidence == 0.7
        assert result.supporting_studies, "stub should populate supporting studies"
        assert result.sub_queries, "sub_queries should be attached for audit"
        assert result.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_evaluate_with_no_retrieval_returns_insufficient_evidence(self):
        bundle = _build_canonical_bundle()
        agent = MedicalOncologyAgent()
        _patch_synth(agent)
        # Patch _retrieve to always return zero studies
        _patch_retrieve(agent, studies_per_call=0)

        result = await agent.evaluate(bundle)
        assert result.recommendation == "insufficient_evidence"
        assert result.confidence == 0.0
