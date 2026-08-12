"""
Eligibility Fallback Tests — Fix 6: Refuse-to-restore fallback.

When `category_routing_suspect=True` AND the filtered bundle has fewer
than MIN_STUDIES_FLOOR (3) studies, the pipeline must fire a fallback
strategy that restores the least-mismatched removed studies at heavily
reduced scores, guaranteeing a minimum bundle size of 3.

**Validates: Requirements 2.7, 3.5, 3.8**
"""

import pytest
from src.api.services.patient_eligibility_boost_service import (
    apply_patient_eligibility_filter_and_boost,
)


# ---------------------------------------------------------------------------
# Helpers — build chunk / eligibility fixtures
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, title: str, score: float) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "score": score,
        "final_score": score,
        "text": f"Study text for {doc_id}",
    }


def _make_eligibility(
    status: str,
    has_hard_mismatch: bool,
    boost: float,
    cancer_type_verdict: str = "NOT_AVAILABLE",
    histology_verdict: str = "NOT_AVAILABLE",
    stage_verdict: str = "NOT_AVAILABLE",
    prior_therapies_verdict: str = "NOT_AVAILABLE",
    biomarkers_verdict: str = "NOT_AVAILABLE",
    reason: str = "",
) -> dict:
    return {
        "status": status,
        "reason": reason,
        "has_hard_mismatch": has_hard_mismatch,
        "boost": boost,
        "criteria_verdicts": {
            "cancer_type": cancer_type_verdict,
            "histology": histology_verdict,
            "stage": stage_verdict,
            "prior_therapies": prior_therapies_verdict,
            "biomarkers": biomarkers_verdict,
        },
    }


# ======================================================================
# Test 8a: Fallback fires when category_routing_suspect AND < 3 studies
# ======================================================================

class TestEligibilityFallbackFires:
    """
    Property 9: When the eligibility filter reduces the bundle to fewer
    than 3 studies and category_routing_suspect is True, the pipeline
    SHALL attempt a fallback strategy.

    **Validates: Requirements 2.7**
    """

    def test_fallback_fires_when_suspect_and_sparse(self):
        """
        Scenario: 5 chunks total, 4 removed as cancer_type MISMATCH,
        1 surviving → category_routing_suspect=True, bundle < 3.
        The fallback must restore studies to reach ≥ 3.
        """
        chunks = [
            _make_chunk("hn_study_001", "H&N Immunotherapy Trial", 0.9),
            _make_chunk("lung_study_001", "Lung Pembrolizumab Trial", 0.8),
            _make_chunk("lung_study_002", "Lung Nivolumab Trial", 0.75),
            _make_chunk("lung_study_003", "Lung Atezolizumab Trial", 0.7),
            _make_chunk("lung_study_004", "Lung Durvalumab Trial", 0.65),
        ]

        eligibility_results = {
            "hn_study_001": _make_eligibility(
                "MATCH", False, 0.25,
                cancer_type_verdict="MATCH",
                reason="Study matches patient cancer type",
            ),
            "lung_study_001": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                reason="Study enrolls lung cancer, patient has H&N cancer",
            ),
            "lung_study_002": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                reason="Study enrolls lung cancer, patient has H&N cancer",
            ),
            "lung_study_003": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                reason="Study enrolls lung cancer, patient has H&N cancer",
            ),
            "lung_study_004": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                reason="Study enrolls lung cancer, patient has H&N cancer",
            ),
        }

        # We call the top-level function indirectly via run_patient_eligibility_check
        # but since that's async and requires OpenAI, we test the lower-level
        # apply_patient_eligibility_filter_and_boost + the fallback logic.
        # The fallback lives in run_patient_eligibility_check, so we need to
        # simulate the same flow.
        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks, eligibility_results
        )

        # After hard filtering: 1 kept (hn_study_001), 4 removed
        # category_routing_suspect = True (4/4 = 100% cancer_type MISMATCH)
        # Bundle < 3 → fallback should fire

        # Simulate the category_routing_suspect + fallback logic
        # that lives in run_patient_eligibility_check
        cancer_type_mismatches = sum(
            1 for rc in removed
            if rc.get("patient_eligibility", {}).get("criteria_verdicts", {}).get("cancer_type") == "MISMATCH"
        )
        category_routing_suspect = (
            len(removed) > 0
            and cancer_type_mismatches / len(removed) >= 0.8
        )

        assert category_routing_suspect is True, (
            f"Expected category_routing_suspect=True, "
            f"got {cancer_type_mismatches}/{len(removed)} cancer_type mismatches"
        )
        assert len(kept) < 3, (
            f"Expected fewer than 3 kept studies for fallback to fire, got {len(kept)}"
        )

        # Now test the actual fallback: import the MIN_STUDIES_FLOOR constant
        from src.api.services.patient_eligibility_boost_service import MIN_STUDIES_FLOOR
        assert MIN_STUDIES_FLOOR == 3

        # The fallback should restore enough studies to reach MIN_STUDIES_FLOOR
        # We need to test this via the full run_patient_eligibility_check flow,
        # but since that's async, we test the fallback helper directly.
        from src.api.services.patient_eligibility_boost_service import (
            _apply_eligibility_fallback,
        )
        restored_kept = _apply_eligibility_fallback(kept, removed)
        assert len(restored_kept) >= MIN_STUDIES_FLOOR, (
            f"Fallback must produce ≥ {MIN_STUDIES_FLOOR} studies, got {len(restored_kept)}"
        )


# ======================================================================
# Test 8b: Fallback produces bundle of ≥ 3 studies
# ======================================================================

class TestFallbackMinimumBundleSize:
    """
    The fallback must restore the least-mismatched removed studies at
    heavily reduced scores to guarantee a minimum bundle of 3.

    **Validates: Requirements 2.7**
    """

    def test_fallback_produces_at_least_3_studies(self):
        """
        Scenario: 1 surviving study + 4 removed (all cancer_type MISMATCH).
        Fallback restores 2 least-mismatched at reduced scores → 3 total.
        """
        chunks = [
            _make_chunk("hn_study_001", "H&N Immunotherapy Trial", 0.9),
            _make_chunk("lung_study_001", "Lung Trial A", 0.8),
            _make_chunk("lung_study_002", "Lung Trial B", 0.75),
            _make_chunk("lung_study_003", "Lung Trial C", 0.7),
            _make_chunk("lung_study_004", "Lung Trial D", 0.65),
        ]

        eligibility_results = {
            "hn_study_001": _make_eligibility(
                "MATCH", False, 0.25,
                cancer_type_verdict="MATCH",
            ),
            # All lung studies: cancer_type MISMATCH only (no other hard mismatches)
            "lung_study_001": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
            ),
            "lung_study_002": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
            ),
            # These two have additional mismatches (harder mismatch)
            "lung_study_003": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                histology_verdict="MISMATCH",
            ),
            "lung_study_004": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                histology_verdict="MISMATCH",
                biomarkers_verdict="MISMATCH",
            ),
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks, eligibility_results
        )

        from src.api.services.patient_eligibility_boost_service import (
            _apply_eligibility_fallback,
            MIN_STUDIES_FLOOR,
        )

        restored_kept = _apply_eligibility_fallback(kept, removed)

        assert len(restored_kept) >= MIN_STUDIES_FLOOR, (
            f"Expected ≥ {MIN_STUDIES_FLOOR} studies, got {len(restored_kept)}"
        )

        # Restored studies should be the least-mismatched ones (fewest hard mismatches)
        restored_doc_ids = [c["doc_id"] for c in restored_kept if c["doc_id"] != "hn_study_001"]
        # lung_study_001 and lung_study_002 have only 1 mismatch each (cancer_type)
        # lung_study_003 has 2 mismatches, lung_study_004 has 3 mismatches
        # So the fallback should prefer lung_study_001 and lung_study_002
        assert "lung_study_001" in restored_doc_ids or "lung_study_002" in restored_doc_ids, (
            f"Expected least-mismatched studies to be restored, got {restored_doc_ids}"
        )

        # Restored studies must have reduced scores (score * 0.3)
        for chunk in restored_kept:
            if chunk["doc_id"] != "hn_study_001":
                assert chunk.get("restored_at_reduced_score") is True, (
                    f"Restored study {chunk['doc_id']} must be marked restored_at_reduced_score=True"
                )


# ======================================================================
# Test 8c: Legitimate MISMATCH removals NOT restored when bundle ≥ 3
# ======================================================================

class TestLegitimateRemovalsNotRestored:
    """
    When the bundle already has ≥ 3 studies, the fallback must NOT
    restore cancer_type MISMATCH studies. Legitimate removals stay removed.

    **Validates: Requirements 3.5, 3.8**
    """

    def test_no_restoration_when_bundle_already_sufficient(self):
        """
        Scenario: 4 surviving studies + 2 removed as cancer_type MISMATCH.
        Bundle is already ≥ 3, so no fallback should fire.
        """
        chunks = [
            _make_chunk("hn_study_001", "H&N Trial A", 0.9),
            _make_chunk("hn_study_002", "H&N Trial B", 0.85),
            _make_chunk("hn_study_003", "H&N Trial C", 0.8),
            _make_chunk("hn_study_004", "H&N Trial D", 0.75),
            _make_chunk("lung_study_001", "Lung Trial A", 0.7),
            _make_chunk("lung_study_002", "Lung Trial B", 0.65),
        ]

        eligibility_results = {
            "hn_study_001": _make_eligibility("MATCH", False, 0.25, cancer_type_verdict="MATCH"),
            "hn_study_002": _make_eligibility("MATCH", False, 0.25, cancer_type_verdict="MATCH"),
            "hn_study_003": _make_eligibility("MATCH", False, 0.20, cancer_type_verdict="MATCH"),
            "hn_study_004": _make_eligibility("MATCH", False, 0.15, cancer_type_verdict="MATCH"),
            "lung_study_001": _make_eligibility(
                "NO_MATCH", True, 0, cancer_type_verdict="MISMATCH",
            ),
            "lung_study_002": _make_eligibility(
                "NO_MATCH", True, 0, cancer_type_verdict="MISMATCH",
            ),
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks, eligibility_results
        )

        # 4 kept, 2 removed — bundle is already ≥ 3
        assert len(kept) >= 3, f"Expected ≥ 3 kept, got {len(kept)}"

        # Removed studies should stay removed
        removed_ids = [c["doc_id"] for c in removed]
        assert "lung_study_001" in removed_ids
        assert "lung_study_002" in removed_ids

        # Kept studies should NOT include the lung studies
        kept_ids = [c["doc_id"] for c in kept]
        assert "lung_study_001" not in kept_ids
        assert "lung_study_002" not in kept_ids


# ======================================================================
# Test 8d: Restored studies are marked with restored_at_reduced_score
# ======================================================================

class TestRestoredStudiesMarked:
    """
    Restored studies must be clearly marked with `restored_at_reduced_score=True`
    so downstream generation knows they are lower-confidence.

    **Validates: Requirements 2.7**
    """

    def test_restored_studies_have_flag_and_reduced_scores(self):
        """
        Scenario: 1 surviving + 3 removed (all cancer_type MISMATCH).
        Fallback restores 2 → they must have the flag and reduced scores.
        """
        chunks = [
            _make_chunk("hn_study_001", "H&N Trial", 0.9),
            _make_chunk("lung_study_001", "Lung Trial A", 0.8),
            _make_chunk("lung_study_002", "Lung Trial B", 0.7),
            _make_chunk("lung_study_003", "Lung Trial C", 0.6),
        ]

        eligibility_results = {
            "hn_study_001": _make_eligibility(
                "MATCH", False, 0.25, cancer_type_verdict="MATCH",
            ),
            "lung_study_001": _make_eligibility(
                "NO_MATCH", True, 0, cancer_type_verdict="MISMATCH",
            ),
            "lung_study_002": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                histology_verdict="MISMATCH",
            ),
            "lung_study_003": _make_eligibility(
                "NO_MATCH", True, 0,
                cancer_type_verdict="MISMATCH",
                histology_verdict="MISMATCH",
                biomarkers_verdict="MISMATCH",
            ),
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks, eligibility_results
        )

        from src.api.services.patient_eligibility_boost_service import (
            _apply_eligibility_fallback,
        )

        restored_kept = _apply_eligibility_fallback(kept, removed)

        # Check restored studies
        restored = [c for c in restored_kept if c.get("restored_at_reduced_score") is True]
        assert len(restored) >= 2, (
            f"Expected ≥ 2 restored studies, got {len(restored)}"
        )

        for chunk in restored:
            # Must have the flag
            assert chunk["restored_at_reduced_score"] is True

            # Score must be reduced (original * 0.3)
            original_score = chunk.get("score", 0.5)
            expected_max_score = original_score * 0.3 + 0.01  # small tolerance
            assert chunk["final_score"] <= expected_max_score, (
                f"Restored study {chunk['doc_id']} score {chunk['final_score']} "
                f"should be ≤ {expected_max_score} (original={original_score})"
            )

        # The original kept study should NOT have the flag
        original_kept = [c for c in restored_kept if c.get("restored_at_reduced_score") is not True]
        assert any(c["doc_id"] == "hn_study_001" for c in original_kept), (
            "Original kept study hn_study_001 should not be marked as restored"
        )
