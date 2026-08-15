"""
Old-vs-new pipeline comparison tests (2026-08-12 convergence Sprint D
item 26).

The whole convergence program's central promise -- stated explicitly in
CLAUDE.md's "do not change" list and repeated in clinical_retrieval_
adapter.py's own module docstring ("Your existing retrieval stack is one
of the strongest parts of Paxis... this module is the one-directional
adapter... without reprocessing the KB or rewriting a single line of the
retriever itself") -- is that the new physician pipeline is a pure
ADDITION alongside the legacy one, never a fork or a silent replacement.
This file is what actually checks that claim rather than just asserting
it in prose:

  1. clinical_retrieval_adapter.py's reshape is lossless and one-way --
     every value on the new EvidenceCandidate traces back to the exact
     legacy StudyEvidence/chunk value, nothing invented or dropped.
  2. The new orchestrator calls the legacy retriever with ONLY the
     kwargs ComprehensiveRetriever.retrieve_comprehensive() actually
     accepts (checked against the real method's signature, not a
     permissive fake) -- it cannot have silently started passing
     something the untouched legacy method doesn't understand.
  3. The SAME underlying legacy retrieval result, converted through the
     OLD format (convert_to_rag_evidence(), what enhanced_rag_service.py
     and patient_query.py already consume) and the NEW format
     (clinical_retrieval_adapter.adapt_legacy_results(), what the
     physician pipeline consumes), represent the identical set of
     chunks -- two views of one truth, not two pipelines that could
     silently disagree.
  4. Toggling settings.physician_rag_beta_enabled -- the flag that gates
     the NEW route entirely -- has zero effect on the legacy, unrelated
     /rag/patient-query route or on answer_physician_query() itself
     (the flag lives only in the route layer, per physician_beta.py's
     own docstring; the orchestrator function has no opinion on it).
"""

from __future__ import annotations

import inspect

import pytest

from src.api.services.comprehensive_retrieval import (
    ComprehensiveRetrievalResult, StudyEvidence, convert_to_rag_evidence,
)
from src.api.services.physician.clinical_retrieval_adapter import adapt_legacy_results


def _legacy_study() -> StudyEvidence:
    return StudyEvidence(
        doc_id="doc-42",
        title="Osimertinib in EGFR-mutant NSCLC",
        citation="Smith et al., 2024",
        year=2024,
        category="lung",
        chunks=[
            {
                "point_id": "pt-1", "doc_id": "doc-42", "section": "results", "chunk_id": 0,
                "text": "Osimertinib showed a significant PFS benefit.",
                "doc_meta": {"title": "Osimertinib in EGFR-mutant NSCLC", "url": "https://example.org/1"},
                "score_dense": 0.81, "score_lexical": 0.4, "score_crossencoder_gate": 0.9,
            },
            {
                "point_id": "pt-2", "doc_id": "doc-42", "section": "safety", "chunk_id": 1,
                "text": "Grade 3+ adverse events occurred in 12% of patients.",
                "doc_meta": {"title": "Osimertinib in EGFR-mutant NSCLC", "url": "https://example.org/1"},
                "score_dense": 0.62, "score_lexical": 0.3, "score_crossencoder_gate": 0.7,
            },
        ],
    )


class TestAdapterReshapeIsLosslessAndOneWay:
    def test_every_candidate_field_traces_back_to_the_source_chunk(self):
        study = _legacy_study()
        candidates = adapt_legacy_results([study])

        assert len(candidates) == len(study.chunks)  # nothing dropped, nothing invented
        for candidate, chunk in zip(candidates, study.chunks):
            assert candidate.text == chunk["text"]
            assert candidate.section == chunk["section"]
            assert candidate.chunk_index == chunk["chunk_id"]
            assert candidate.qdrant_point_id == chunk["point_id"]
            assert candidate.document_id == study.doc_id
            assert candidate.title == study.title
            assert candidate.metadata["citation"] == study.citation
            assert candidate.metadata["year"] == study.year
            assert candidate.semantic_score == chunk["score_dense"]
            assert candidate.bm25_score == chunk["score_lexical"]
            assert candidate.cross_encoder_score == chunk["score_crossencoder_gate"]

    def test_a_chunk_with_no_text_is_skipped_not_fabricated(self):
        study = _legacy_study()
        study.chunks.append({
            "point_id": "pt-3", "doc_id": "doc-42", "section": "empty", "chunk_id": 2, "text": "",
        })
        candidates = adapt_legacy_results([study])
        assert len(candidates) == 2  # the empty-text chunk contributes nothing, silently


class TestOrchestratorOnlyPassesDocumentedKwargsToTheLegacyRetriever:
    @pytest.mark.asyncio
    async def test_kwargs_bind_cleanly_against_the_real_method_signature(self):
        from src.api.services.comprehensive_retrieval import ComprehensiveRetriever
        from src.api.services.physician.physician_rag_orchestrator import answer_physician_query

        real_sig = inspect.signature(ComprehensiveRetriever.retrieve_comprehensive)
        captured = {}

        class _SpyRetriever:
            async def retrieve_comprehensive(self, **kwargs):
                captured.update(kwargs)
                return ComprehensiveRetrievalResult(
                    studies=[], total_chunks=0, retrieval_time_ms=0.0,
                    phase1_qdrant_docs=0, phase1_postgres_docs=0, phase2_docs_searched=0,
                )

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "No evidence available."})()})]

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        return _Resp()

        await answer_physician_query(
            "phys-1", "What are the options?", client=_Client(), retriever=_SpyRetriever(),
        )

        # Would raise TypeError if the orchestrator passed a kwarg the
        # real (unmodified) method doesn't accept -- binding against the
        # untouched method's own signature, not a permissive fake.
        real_sig.bind(self=object(), **captured)


class TestBothFormatsAgreeOnTheSameUnderlyingEvidence:
    @pytest.mark.asyncio
    async def test_old_format_and_new_format_represent_the_identical_chunk_set(self):
        study = _legacy_study()
        result = ComprehensiveRetrievalResult(
            studies=[study], total_chunks=2, retrieval_time_ms=120.0,
            phase1_qdrant_docs=1, phase1_postgres_docs=0, phase2_docs_searched=1,
        )

        old_evidence, _ = convert_to_rag_evidence(result)
        new_candidates = adapt_legacy_results(result.studies)

        old_texts = {e["text"] for e in old_evidence}
        new_texts = {c.text for c in new_candidates}
        assert old_texts == new_texts == {
            "Osimertinib showed a significant PFS benefit.",
            "Grade 3+ adverse events occurred in 12% of patients.",
        }

        old_doc_ids = {e["doc_id"] for e in old_evidence}
        new_doc_ids = {c.document_id for c in new_candidates}
        assert old_doc_ids == new_doc_ids == {"doc-42"}


class TestFeatureFlagIsolation:
    """physician_rag_beta_enabled gates the ROUTE (physician_beta.py)
    only -- see that module's own docstring. The orchestrator function
    and every pre-existing route are flag-agnostic."""

    @pytest.mark.asyncio
    async def test_orchestrator_works_identically_regardless_of_the_flag(self, monkeypatch):
        from src.core.config import settings
        from src.api.services.physician.physician_rag_orchestrator import answer_physician_query

        class _FakeRetriever:
            async def retrieve_comprehensive(self, **kwargs):
                return ComprehensiveRetrievalResult(
                    studies=[], total_chunks=0, retrieval_time_ms=0.0,
                    phase1_qdrant_docs=0, phase1_postgres_docs=0, phase2_docs_searched=0,
                )

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "Stable answer."})()})]

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        return _Resp()

        original = settings.physician_rag_beta_enabled
        try:
            settings.physician_rag_beta_enabled = False
            off_result = await answer_physician_query(
                "phys-1", "General question.", client=_Client(), retriever=_FakeRetriever(),
            )
            settings.physician_rag_beta_enabled = True
            on_result = await answer_physician_query(
                "phys-1", "General question.", client=_Client(), retriever=_FakeRetriever(),
            )
        finally:
            settings.physician_rag_beta_enabled = original

        assert off_result.answer == on_result.answer == "Stable answer."

    @pytest.mark.asyncio
    async def test_legacy_patient_query_route_is_unaffected_by_the_flag(self, monkeypatch):
        from src.core.config import settings
        from src.api.routes import patient_query as patient_query_module
        from src.api.routes.patient_query import PatientQueryRequest, patient_query

        async def empty_search(query_text, plan, top_k_per_collection=6):
            return []

        class _EmptyRagService:
            async def query(self, **kwargs):
                return {"evidence": []}

        def _fake_openai(content: str):
            from types import SimpleNamespace
            def create(*args, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        from src.api.services.evidence import multi_corpus_retriever
        monkeypatch.setattr(multi_corpus_retriever, "search", empty_search)
        monkeypatch.setattr(
            "src.api.services.enhanced_rag_service.get_enhanced_rag_service",
            lambda: _EmptyRagService(),
        )
        monkeypatch.setattr(patient_query_module, "_openai_client", _fake_openai("Legacy answer."))

        original = settings.physician_rag_beta_enabled
        try:
            settings.physician_rag_beta_enabled = False
            off_resp = await patient_query(PatientQueryRequest(question="What is pembrolizumab?"))
            settings.physician_rag_beta_enabled = True
            on_resp = await patient_query(PatientQueryRequest(question="What is pembrolizumab?"))
        finally:
            settings.physician_rag_beta_enabled = original

        assert off_resp.answer == on_resp.answer


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
