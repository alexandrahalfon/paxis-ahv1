"""
Unit tests for `extract_disease_status` and `extract_surgical_candidacy`.

These cover the regex-based extraction patterns added to
`query_structuring_service.py` to support the new eligibility hard-filter
axes: disease_status and surgical_candidacy.
"""

import pytest

from src.api.services.query_structuring_service import (
    extract_disease_status,
    extract_surgical_candidacy,
    structure_query,
)


class TestDiseaseStatusExtraction:

    @pytest.mark.parametrize("query,expected", [
        ("progressing on pembrolizumab", "post_progression"),
        ("refractory to ICI", "post_progression"),
        ("post-progression on systemic therapy", "post_progression"),
        ("failed on first-line chemotherapy", "post_progression"),
        ("post-ICI salvage candidate", "post_progression"),
        ("post-immunotherapy progression", "post_progression"),
    ])
    def test_post_progression_patterns(self, query, expected):
        assert extract_disease_status(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("metastatic disease to the right ventricle", "metastatic"),
        ("metastatic colorectal adenocarcinoma", "metastatic"),
        ("distant metastasis", "metastatic"),
        ("stage IV disease", "metastatic"),
        ("stage 4 cancer", "metastatic"),
        ("M1 disease", "metastatic"),
        ("cM1", "metastatic"),
        ("pM1", "metastatic"),
    ])
    def test_metastatic_patterns(self, query, expected):
        assert extract_disease_status(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("biopsy-proven recurrent SCC", "recurrent"),
        ("biopsy proven recurrent oral tongue", "recurrent"),
        ("recurrent lesion in the left neck", "recurrent"),
        ("recurrent disease following resection", "recurrent"),
        ("local recurrence after radiotherapy", "recurrent"),
        ("locoregional recurrence", "recurrent"),
        ("recurrence of melanoma", "recurrent"),
        ("salvage setting", "recurrent"),
        ("R/M HNSCC", "recurrent"),
    ])
    def test_recurrent_patterns(self, query, expected):
        assert extract_disease_status(query) == expected

    @pytest.mark.parametrize("query", [
        "65 year old male with bladder cancer",
        "patient presents for evaluation",
        "newly diagnosed adenocarcinoma",
        "stage II oral tongue squamous cell carcinoma",
    ])
    def test_no_match_returns_none(self, query):
        assert extract_disease_status(query) is None

    def test_post_progression_wins_over_recurrent(self):
        """When both signals appear, post_progression should win
        (most specific). The CLAUDE.md golden patient has both:
        recurrent SCC + progressing on ICI. The clinical reality is
        post-progression."""
        query = (
            "biopsy-proven recurrent SCC, started on pembrolizumab, "
            "now progressing on ICI"
        )
        assert extract_disease_status(query) == "post_progression"

    def test_extraction_threaded_into_clinical_history(self):
        """structure_query should populate clinical_history.disease_status
        when the regex extracts a label."""
        struct = structure_query(
            "65 year old male with metastatic prostate adenocarcinoma",
            query_type="general",
        )
        assert struct.clinical_history.disease_status == "metastatic"


class TestSurgicalCandidacyExtraction:

    @pytest.mark.parametrize("query", [
        "no longer a surgical candidate",
        "no longer surgical candidate",
        "not a surgical candidate",
        "not surgical candidate",
        "unresectable head and neck cancer",
        "inoperable lung tumor",
        "salvage surgery not feasible",
        "salvage resection not possible",
    ])
    def test_not_candidate_patterns(self, query):
        assert extract_surgical_candidacy(query) == "not_candidate"

    @pytest.mark.parametrize("query", [
        "declined surgery",
        "patient declined resection",
        "refused surgery",
        "patient refusal of surgery",
    ])
    def test_declined_patterns(self, query):
        assert extract_surgical_candidacy(query) == "declined"

    @pytest.mark.parametrize("query", [
        "will undergo cystectomy next month",
        "planned for resection",
        "will proceed with mastectomy",
        "will proceed to lobectomy",
    ])
    def test_candidate_patterns(self, query):
        assert extract_surgical_candidacy(query) == "candidate"

    @pytest.mark.parametrize("query", [
        "status post left partial glossectomy",
        "65 year old with bladder cancer",
        "stage II adenocarcinoma",
        "underwent prostatectomy in 2020",
    ])
    def test_past_surgery_does_not_imply_current_candidate(self, query):
        """Past surgery alone should not mark patient as a current
        surgical candidate. Only future-tense / planning language
        triggers candidate."""
        assert extract_surgical_candidacy(query) is None

    def test_no_longer_candidate_wins_over_incidental_candidate_mention(self):
        """A patient who is 'no longer a surgical candidate' should not
        be mistakenly flagged as candidate just because the phrase
        'surgical candidate' appears."""
        query = (
            "previously evaluated as a surgical candidate, but now "
            "no longer a surgical candidate following progression"
        )
        assert extract_surgical_candidacy(query) == "not_candidate"

    def test_extraction_threaded_into_clinical_history(self):
        struct = structure_query(
            "unresectable head and neck cancer with locoregional progression",
            query_type="general",
        )
        assert struct.clinical_history.surgical_candidacy == "not_candidate"


class TestClaudeMdGoldenProfile:
    """End-to-end check on the CLAUDE.md golden patient profile."""

    GOLDEN = (
        "80 y.o. male non-smoker with a PMH HTN, Hep C, BPH, CKD, "
        "transverse colon adenocarcinoma s/p extended right hemicolectomy, "
        "and initial Stage II (pT2pN0M0R0) squamous cell carcinoma of the "
        "left oral tongue, status post left partial glossectomy. "
        "Recurrent lesion in the left level I neck, biopsy-proven recurrent "
        "SCC with a CPS score of 100, started on pembrolizumab "
        "(declined combination with chemotherapy) and is no longer a "
        "surgical candidate following significant locoregional progression "
        "on ICI with radiographic concern for metastatic disease to the "
        "right ventricle and progressing on systemic therapy."
    )

    def test_golden_disease_status(self):
        assert extract_disease_status(self.GOLDEN) == "post_progression"

    def test_golden_surgical_candidacy(self):
        assert extract_surgical_candidacy(self.GOLDEN) == "not_candidate"

    def test_golden_structured_query(self):
        struct = structure_query(self.GOLDEN, query_type="treatment_recommendation")
        assert struct.clinical_history.disease_status == "post_progression"
        assert struct.clinical_history.surgical_candidacy == "not_candidate"
