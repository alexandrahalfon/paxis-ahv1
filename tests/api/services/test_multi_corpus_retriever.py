"""
Tests for multi_corpus_retriever.py's dedup/provenance fix (2026-08-12
beta audit item 6): candidates used to be deduped by doc_id alone, which
collapsed every section of one document into a single candidate. Dedup
identity is now the exact Qdrant point id, and each candidate carries the
provenance fields (qdrant_point_id/version_id/section_title/chunk_index/
url/source_name) evidence_packet_builder.py and retrieval_debug_trace.py
now read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.services.evidence import multi_corpus_retriever as mcr
from src.api.services.evidence.retrieval_planner import build_plan
from src.api.services.evidence.patient_context_service import INTENT_NUTRITION


def _point(point_id, doc_id, section_title, chunk_index, text, score=0.8,
           source_key="nci", source_name="National Cancer Institute",
           url="https://www.cancer.gov/nutrition", version_id="v1",
           authority_class="A", applicability=None):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "doc_id": doc_id,
            "text": text,
            "section_title": section_title,
            "chunk_index": chunk_index,
            "version_id": version_id,
            "doc_meta": {
                "title": "Nutrition During Cancer Treatment",
                "source_key": source_key,
                "source_name": source_name,
                "url": url,
                "authority_class": authority_class,
            },
            "applicability": applicability or {},
        },
    )


class _FakeRetriever:
    def __init__(self, points_by_collection):
        self._points_by_collection = points_by_collection

    async def _embed_async(self, text):
        return [0.01] * 8

    async def _qdrant_query(self, collection_name, query, limit, with_payload, with_vectors):
        points = self._points_by_collection.get(collection_name, [])
        return SimpleNamespace(points=points[:limit])


class TestCandidateIdentity:
    def test_prefers_point_id(self):
        item = {"qdrant_point_id": "abc", "doc_id": "doc-1", "section_title": "X", "chunk_index": 0}
        assert mcr._candidate_identity(item) == "point:abc"

    def test_falls_back_to_doc_section_chunk_composite(self):
        item = {"qdrant_point_id": None, "doc_id": "doc-1", "section_title": "Taste changes", "chunk_index": 0}
        assert mcr._candidate_identity(item) == "doc:doc-1|section:Taste changes|chunk:0"

    def test_falls_back_to_text_prefix_as_last_resort(self):
        item = {"qdrant_point_id": None, "doc_id": None, "text": "some passage text here"}
        assert mcr._candidate_identity(item) == "text:some passage text here"

    def test_different_sections_of_the_same_document_get_different_identities(self):
        a = {"qdrant_point_id": "point-a", "doc_id": "doc-1", "section_title": "Taste changes", "chunk_index": 0}
        b = {"qdrant_point_id": "point-b", "doc_id": "doc-1", "section_title": "Appetite loss", "chunk_index": 0}
        assert mcr._candidate_identity(a) != mcr._candidate_identity(b)


class TestSearchPreservesDistinctSectionsOfOneDocument:
    """The actual regression proof: an NCI nutrition page with distinct
    'Taste changes' and 'Appetite loss' sections must survive as TWO
    candidates, not collapse into one the way doc_id-only dedup did."""

    @pytest.mark.asyncio
    async def test_two_sections_same_doc_id_both_survive(self, monkeypatch):
        points = [
            _point("point-taste", "doc-nci-nutrition", "Taste changes", 0, "Foods may taste metallic."),
            _point("point-appetite", "doc-nci-nutrition", "Appetite loss", 0, "Eat small frequent meals."),
        ]
        fake_retriever = _FakeRetriever({"oncology_patient_education": points})
        monkeypatch.setattr(mcr, "_retriever", lambda: fake_retriever)

        plan = build_plan(INTENT_NUTRITION, {})
        plan.collections = ["oncology_patient_education"]

        results = await mcr.search("nutrition during chemo", plan)

        assert len(results) == 2
        section_titles = {r["section_title"] for r in results}
        assert section_titles == {"Taste changes", "Appetite loss"}

    @pytest.mark.asyncio
    async def test_same_point_id_across_two_collection_buckets_is_deduped(self, monkeypatch):
        # Edge case: the exact same chunk somehow surfaced from two
        # searches (e.g. re-run against the same collection twice) --
        # this SHOULD still collapse to one candidate.
        shared_point = _point("point-shared", "doc-1", "Section A", 0, "Some text.")
        fake_retriever = _FakeRetriever({
            "oncology_patient_education": [shared_point],
            "oncology_medication_knowledge": [shared_point],
        })
        monkeypatch.setattr(mcr, "_retriever", lambda: fake_retriever)

        plan = build_plan(INTENT_NUTRITION, {})
        plan.collections = ["oncology_patient_education", "oncology_medication_knowledge"]

        results = await mcr.search("query", plan)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_candidates_carry_full_provenance(self, monkeypatch):
        points = [_point(
            "point-1", "doc-1", "Taste changes", 2, "Some passage.",
            source_key="nci", source_name="National Cancer Institute",
            url="https://www.cancer.gov/x", version_id="version-abc",
        )]
        fake_retriever = _FakeRetriever({"oncology_patient_education": points})
        monkeypatch.setattr(mcr, "_retriever", lambda: fake_retriever)

        plan = build_plan(INTENT_NUTRITION, {})
        plan.collections = ["oncology_patient_education"]

        results = await mcr.search("query", plan)
        assert len(results) == 1
        r = results[0]
        assert r["qdrant_point_id"] == "point-1"
        assert r["doc_id"] == "doc-1"
        assert r["version_id"] == "version-abc"
        assert r["section_title"] == "Taste changes"
        assert r["chunk_index"] == 2
        assert r["url"] == "https://www.cancer.gov/x"
        assert r["source_name"] == "National Cancer Institute"
        assert r["source_key"] == "nci"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
