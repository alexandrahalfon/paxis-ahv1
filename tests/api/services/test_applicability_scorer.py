"""
Tests for the itemized, intent-weighted applicability scorer
(applicability_scorer.py) and its supporting patient_values plumbing in
retrieval_planner.py, added 2026-08-12 alongside chunk-level metadata
classification.

These are the milestones specified when this rewrite was scoped:
  1. A metallic-taste/FOLFOX-style scenario scores highly and produces
     itemized components (not one blended number).
  2. A deliberately bad case — a chemotherapy patient served
     targeted-therapy-only content — scores measurably lower than
     comparable general content, via the explicit modality-conflict
     penalty, not just averaged into a single component.
  3. The same question against two different patients ranks candidates
     differently because of component scores, not LLM wording.
Plus direct coverage of the neutral/general/named match semantics that
back all of the above.
"""

import pytest

from src.api.services.evidence.retrieval_planner import RetrievalPlan, build_plan
from src.api.services.evidence.applicability_scorer import (
    WEIGHTS_BY_INTENT, score_candidate, rank, _set_match,
)
from src.api.services.evidence.patient_context_service import (
    INTENT_MEDICATION, INTENT_SYMPTOM, INTENT_NUTRITION,
    INTENT_TREATMENT, INTENT_DIAGNOSIS, INTENT_GENERAL,
)


def _candidate(**overrides):
    base = {
        "doc_id": "doc-1",
        "text": "General information about managing cancer treatment.",
        "semantic_score": 0.7,
        "applicability_meta": {},
        "authority_class": "A",
    }
    base.update(overrides)
    return base


class TestWeightsByIntent:
    @pytest.mark.parametrize("intent", list(WEIGHTS_BY_INTENT.keys()))
    def test_weights_sum_to_one(self, intent):
        assert sum(WEIGHTS_BY_INTENT[intent].values()) == pytest.approx(1.0)

    def test_covers_every_declared_intent(self):
        assert set(WEIGHTS_BY_INTENT.keys()) == {
            INTENT_MEDICATION, INTENT_SYMPTOM, INTENT_NUTRITION,
            INTENT_TREATMENT, INTENT_DIAGNOSIS, INTENT_GENERAL,
        }


class TestSetMatchSemantics:
    def test_unspecified_on_both_sides_is_neutral(self):
        assert _set_match([], [], "") == 0.5
        assert _set_match(None, None, "") == 0.5

    def test_unspecified_patient_is_neutral_even_with_candidate_tags(self):
        assert _set_match([], ["dysgeusia"], "") == 0.5

    def test_unspecified_candidate_is_neutral_even_with_patient_terms(self):
        assert _set_match(["dysgeusia"], [], "") == 0.5

    def test_general_token_scores_075(self):
        assert _set_match(["colorectal"], ["all"], "", general_token="all") == 0.75

    def test_named_overlap_scores_1(self):
        assert _set_match(["dysgeusia"], ["dysgeusia", "nausea"], "") == 1.0

    def test_named_mismatch_with_no_text_corroboration_scores_0(self):
        assert _set_match(["dysgeusia"], ["neuropathy"], "unrelated text") == 0.0

    def test_named_mismatch_but_text_corroborates_scores_1(self):
        # Chunk wasn't classified with the patient's term, but the raw
        # text literally contains it — treated as a match, not invented.
        assert _set_match(["dysgeusia"], ["neuropathy"], "watch for dysgeusia") == 1.0


class TestRetrievalPlanPatientValues:
    def test_patient_values_populated_from_retrieval_features(self):
        plan = build_plan(INTENT_SYMPTOM, {
            "symptoms": ["dysgeusia"], "regimens": ["FOLFOX"],
            "active_agents": ["oxaliplatin"], "cancer_types": ["colorectal"],
            "treatment_modalities": ["chemotherapy"],
        })
        assert plan.patient_values["symptoms"] == ["dysgeusia"]
        assert plan.patient_values["regimens"] == ["FOLFOX"]
        assert plan.patient_values["drugs"] == ["oxaliplatin"]
        assert plan.patient_values["cancer_types"] == ["colorectal"]
        assert plan.patient_values["treatment_modalities"] == ["chemotherapy"]
        assert plan.patient_values["treatment_phase"] == ["active_treatment"]

    def test_treatment_phase_prefers_explicit_care_phase(self):
        plan = build_plan(INTENT_GENERAL, {"nutrition_care_phase": "survivorship"})
        assert plan.patient_values["treatment_phase"] == ["survivorship"]

    def test_treatment_phase_empty_when_nothing_known(self):
        plan = build_plan(INTENT_GENERAL, {})
        assert plan.patient_values["treatment_phase"] == []


class TestMetallicTasteScenario:
    """Milestone 1: FOLFOX/oxaliplatin patient, dysgeusia question,
    against an ONS-style chunk explicitly tagged for that regimen and
    symptom — should score highly with every component visible."""

    def _plan(self):
        return build_plan(INTENT_SYMPTOM, {
            "symptoms": ["dysgeusia"], "regimens": ["FOLFOX"],
            "active_agents": ["oxaliplatin"], "cancer_types": ["colorectal"],
            "treatment_modalities": ["chemotherapy"],
        })

    def test_matching_chunk_scores_highly_with_itemized_components(self):
        plan = self._plan()
        candidate = _candidate(
            text="Managing taste changes during FOLFOX (oxaliplatin) chemotherapy.",
            semantic_score=0.82,
            applicability_meta={
                "symptoms": ["dysgeusia"], "regimens": ["FOLFOX"],
                "drugs": ["oxaliplatin"], "cancer_types": ["colorectal"],
                "treatment_modalities": ["chemotherapy"],
                "treatment_phases": ["active_treatment"],
            },
        )
        scored = score_candidate(candidate, plan)
        c = scored["components"]
        assert c["symptom"] == 1.0
        assert c["regimen"] == 1.0
        assert c["drug"] == 1.0
        assert c["cancer"] == 1.0
        assert c["modality"] == 1.0
        assert c["modality_conflict"] is False
        assert scored["applicability_score"] > 0.9

    def test_general_dysgeusia_page_scores_lower_than_regimen_specific_page(self):
        # The architecture review's own example (section 18): a generic
        # NCI dysgeusia page vs. an ONS FOLFOX-specific one should not
        # score identically for a FOLFOX patient.
        plan = self._plan()
        specific = score_candidate(_candidate(
            text="Taste changes on FOLFOX.", semantic_score=0.8,
            applicability_meta={
                "symptoms": ["dysgeusia"], "regimens": ["FOLFOX"],
                "drugs": ["oxaliplatin"], "cancer_types": ["all"],
                "treatment_modalities": ["chemotherapy"],
            },
        ), plan)
        general = score_candidate(_candidate(
            text="Taste changes during cancer treatment.", semantic_score=0.8,
            applicability_meta={
                "symptoms": ["dysgeusia"], "cancer_types": ["all"],
            },
        ), plan)
        assert specific["applicability_score"] > general["applicability_score"]


class TestModalityConflictPenalty:
    """Milestone 2: a chemotherapy patient served targeted-therapy-only
    content should score measurably below comparable general content —
    not just carry one zeroed-out component among eight."""

    def _plan(self):
        return build_plan(INTENT_TREATMENT, {
            "treatment_modalities": ["chemotherapy"], "regimens": ["FOLFOX"],
        })

    def test_conflicting_modality_is_flagged_and_penalized(self):
        plan = self._plan()
        scored = score_candidate(_candidate(
            text="Targeted therapy options for EGFR-mutated tumors.",
            semantic_score=0.75,
            applicability_meta={"treatment_modalities": ["targeted_therapy"], "cancer_types": ["all"]},
        ), plan)
        assert scored["components"]["modality"] == 0.0
        assert scored["components"]["modality_conflict"] is True

    def test_conflicting_modality_scores_below_general_content_same_semantic(self):
        plan = self._plan()
        conflicting = score_candidate(_candidate(
            text="Targeted therapy options for EGFR-mutated tumors.",
            semantic_score=0.75,
            applicability_meta={"treatment_modalities": ["targeted_therapy"], "cancer_types": ["all"]},
        ), plan)
        general = score_candidate(_candidate(
            text="General overview of cancer treatment options.",
            semantic_score=0.75,
            applicability_meta={"cancer_types": ["all"]},
        ), plan)
        assert conflicting["applicability_score"] < general["applicability_score"]

    def test_no_conflict_flagged_when_candidate_modality_unspecified(self):
        plan = self._plan()
        scored = score_candidate(_candidate(
            text="General information about managing side effects.",
            semantic_score=0.75,
            applicability_meta={"cancer_types": ["all"]},
        ), plan)
        assert scored["components"]["modality_conflict"] is False
        assert scored["components"]["modality"] == 0.5  # unspecified -> neutral

    def test_incompatibility_reasons_populated_on_conflict(self):
        """Carried through to evidence_packet_builder.py/retrieval_debug_
        trace.py (2026-08-12 beta audit) so a reviewer sees WHY a score
        dropped, not just that it did."""
        plan = self._plan()
        scored = score_candidate(_candidate(
            text="Targeted therapy options for EGFR-mutated tumors.",
            semantic_score=0.75,
            applicability_meta={"treatment_modalities": ["targeted_therapy"], "cancer_types": ["all"]},
        ), plan)
        assert scored["incompatibility_reasons"], "expected a non-empty reasons list on conflict"
        assert "modality_mismatch" in scored["incompatibility_reasons"][0]
        assert "chemotherapy" in scored["incompatibility_reasons"][0]
        assert "targeted_therapy" in scored["incompatibility_reasons"][0]

    def test_incompatibility_reasons_empty_when_no_conflict(self):
        plan = self._plan()
        scored = score_candidate(_candidate(
            text="General information about managing side effects.",
            semantic_score=0.75,
            applicability_meta={"cancer_types": ["all"]},
        ), plan)
        assert scored["incompatibility_reasons"] == []

    def test_no_conflict_flagged_when_patient_modality_unspecified(self):
        plan = build_plan(INTENT_TREATMENT, {})
        scored = score_candidate(_candidate(
            applicability_meta={"treatment_modalities": ["targeted_therapy"]},
        ), plan)
        assert scored["components"]["modality_conflict"] is False


class TestCrossPatientRankingDiffers:
    """Milestone 3: the same two candidates, ranked for two different
    patients, should reorder because of differing component scores."""

    def test_ranking_reorders_across_patients(self):
        folfox_candidate = _candidate(
            doc_id="folfox-doc", text="FOLFOX-specific dosing and taste-change guidance.",
            semantic_score=0.78,
            applicability_meta={
                "regimens": ["FOLFOX"], "drugs": ["oxaliplatin"],
                "symptoms": ["dysgeusia"], "cancer_types": ["colorectal"],
            },
        )
        immunotherapy_candidate = _candidate(
            doc_id="ici-doc", text="Immunotherapy-related taste-change guidance.",
            semantic_score=0.80,
            applicability_meta={
                "drugs": ["pembrolizumab"], "treatment_modalities": ["immunotherapy"],
                "symptoms": ["dysgeusia"], "cancer_types": ["all"],
            },
        )
        candidates = [folfox_candidate, immunotherapy_candidate]

        folfox_patient_plan = build_plan(INTENT_SYMPTOM, {
            "symptoms": ["dysgeusia"], "regimens": ["FOLFOX"],
            "active_agents": ["oxaliplatin"], "cancer_types": ["colorectal"],
        })
        ici_patient_plan = build_plan(INTENT_SYMPTOM, {
            "symptoms": ["dysgeusia"], "active_agents": ["pembrolizumab"],
            "treatment_modalities": ["immunotherapy"],
        })

        ranked_for_folfox_patient = rank(candidates, folfox_patient_plan)
        ranked_for_ici_patient = rank(candidates, ici_patient_plan)

        assert ranked_for_folfox_patient[0]["doc_id"] == "folfox-doc"
        assert ranked_for_ici_patient[0]["doc_id"] == "ici-doc"


class TestRank:
    def test_rank_sorts_descending_and_respects_limit(self):
        plan = build_plan(INTENT_GENERAL, {})
        candidates = [
            _candidate(doc_id="low", semantic_score=0.1),
            _candidate(doc_id="high", semantic_score=0.9),
            _candidate(doc_id="mid", semantic_score=0.5),
        ]
        ranked = rank(candidates, plan, limit=2)
        assert [c["doc_id"] for c in ranked] == ["high", "mid"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
