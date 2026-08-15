"""
Tests for the subsite-hierarchy demotion guard upgrade and the chunk
scoring heuristic.

The demotion guard previously downgraded any cancer_type MISMATCH where
patient and study were sibling subsites in the same ontology family,
which let "N0 primary oral cancer" studies survive for recurrent
post-progression patients. The new guard skips demotion when
disease_status=MISMATCH or surgical_candidacy=MISMATCH is also in the
verdict.
"""

import pytest

from src.api.services.patient_eligibility_boost_service import (
    _demote_subsite_hierarchy_mismatch,
)


def _chunk_with_keyword(keyword: str) -> dict:
    """Build a chunk whose metadata.keywords_flat list includes a
    given site keyword so the demotion path can find it."""
    return {
        "doc_id": "test_doc_1",
        "title": "Test study",
        "text": f"Study population: {keyword} cancer cohort.",
        "payload": {
            "metadata": {"keywords_flat": [keyword]},
            "doc_meta": {"title": "Test study"},
        },
    }


PATIENT_HN = {
    "cancer_type": "head and neck cancer",
    "cancer_subsite": "oral tongue",
}


class TestDemotionStillFiresOnPureSubsiteArtifact:
    """Baseline: when only cancer_type=MISMATCH is in the verdicts and
    the study is in a sibling subsite, demotion should still fire (the
    legacy behavior we want to preserve)."""

    def test_sibling_subsite_demotes_cancer_type(self):
        verdicts = {"cancer_type": "MISMATCH"}
        chunk = _chunk_with_keyword("oral cavity")
        demoted = _demote_subsite_hierarchy_mismatch(chunk, verdicts, PATIENT_HN)
        assert demoted is True
        assert verdicts["cancer_type"] == "POSSIBLE"


class TestDemotionSkippedWhenDiseaseStatusMismatch:
    """When the LLM verdict ALSO contains disease_status=MISMATCH the
    cancer_type MISMATCH represents a real population mismatch (primary
    vs recurrent), not a subsite-hierarchy artifact. Demotion must NOT
    fire — the cancer_type MISMATCH must stay so the study hard-drops."""

    def test_disease_status_mismatch_blocks_demotion(self):
        verdicts = {
            "cancer_type": "MISMATCH",
            "disease_status": "MISMATCH",
        }
        chunk = _chunk_with_keyword("oral cavity")
        demoted = _demote_subsite_hierarchy_mismatch(chunk, verdicts, PATIENT_HN)
        assert demoted is False
        assert verdicts["cancer_type"] == "MISMATCH"
        assert verdicts["disease_status"] == "MISMATCH"

    def test_disease_status_match_does_not_block_demotion(self):
        """Only MISMATCH on disease_status should block. MATCH /
        COMPATIBLE / NOT_AVAILABLE leave the legacy demotion working."""
        verdicts = {
            "cancer_type": "MISMATCH",
            "disease_status": "MATCH",
        }
        chunk = _chunk_with_keyword("oral cavity")
        demoted = _demote_subsite_hierarchy_mismatch(chunk, verdicts, PATIENT_HN)
        assert demoted is True


class TestDemotionSkippedWhenSurgicalCandidacyMismatch:
    """Same logic as disease_status: a surgical-candidacy MISMATCH
    means the study population genuinely differs even if subsites
    overlap, so the cancer_type MISMATCH must not be demoted."""

    def test_surgical_candidacy_mismatch_blocks_demotion(self):
        verdicts = {
            "cancer_type": "MISMATCH",
            "surgical_candidacy": "MISMATCH",
        }
        chunk = _chunk_with_keyword("oral cavity")
        demoted = _demote_subsite_hierarchy_mismatch(chunk, verdicts, PATIENT_HN)
        assert demoted is False
        assert verdicts["cancer_type"] == "MISMATCH"

    def test_both_disease_and_surgical_mismatch_blocks_demotion(self):
        """Most realistic case for the CLAUDE.md golden profile:
        recurrent + not-candidate → both axes return MISMATCH and
        demotion must not fire."""
        verdicts = {
            "cancer_type": "MISMATCH",
            "disease_status": "MISMATCH",
            "surgical_candidacy": "MISMATCH",
        }
        chunk = _chunk_with_keyword("oral cavity")
        demoted = _demote_subsite_hierarchy_mismatch(chunk, verdicts, PATIENT_HN)
        assert demoted is False
