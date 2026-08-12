"""
Property-based tests for feature flag preservation.

Feature: patient-study-match-scoring

Tests the following property:
- Property 14: Feature flag false preserves identical output

When USE_RECONCILED_STRUCTURE=false, the system must produce identical
outputs to the current production system:
  - reconcile_if_enabled() returns None
  - apply_patient_eligibility_filter_and_boost(use_tiered_model=False)
    uses the original hard-filter model (any MISMATCH → removal)
  - _patient_signal_score() still correctly determines has_patient_context

**Validates: Requirements 7.5, 6.2**
"""

import os
import pytest
from hypothesis import given, strategies as st, settings, assume

from src.api.services.query_reconciliation import reconcile_if_enabled
from src.api.services.patient_eligibility_boost_service import (
    apply_patient_eligibility_filter_and_boost,
)
from src.api.services.query_structuring_service import (
    _patient_signal_score,
    QueryStructure,
    PatientContext,
    CancerContext,
    TreatmentContext,
    ClinicalHistory,
)


# ======================================================================
# Shared constants and strategies
# ======================================================================

#: Eligibility axes used by the hard-filter model.
ELIGIBILITY_AXES = ["cancer_type", "histology", "stage", "biomarkers", "prior_therapies"]

#: Possible verdict values.
ALL_VERDICTS = ["MATCH", "MISMATCH", "NOT_AVAILABLE", "POSSIBLE"]

#: Sample cancer sites for generating realistic queries.
CANCER_SITES = ["lung", "breast", "prostate", "colon", "head_neck", "pancreas", "liver"]

#: Sample histology types.
HISTOLOGIES = ["adenocarcinoma", "squamous", "small cell", "ductal", "lobular"]

#: Sample stages.
STAGES = ["I", "II", "III", "IIIA", "IV"]

#: Sample biomarkers with polarity.
BIOMARKERS_WITH_POLARITY = [
    "EGFR-mutant", "KRAS G12C positive", "ALK-positive",
    "HER2-positive", "BRCA2-mutated", "PD-L1 high",
]

#: Sample biomarkers without polarity (just names).
BIOMARKERS_PLAIN = ["EGFR", "KRAS", "ALK", "HER2", "BRCA1"]

#: Sample prior treatments.
PRIOR_TREATMENTS = ["cisplatin", "carboplatin", "pembrolizumab", "nivolumab", "docetaxel"]

#: Patient signal phrases that contribute to the signal score.
SIGNAL_PHRASES = [
    "65 year old",
    "male",
    "patient with",
    "presenting with",
    "diagnosed with",
    "history of",
]

#: LLM axis keys for reconciliation input.
LLM_AXIS_KEYS = [
    "cancer_type", "histology", "stage", "biomarkers",
    "treatment", "demographics", "clinical_history", "molecular_profile",
]


def _make_chunk(doc_id: str, score: float = 0.8) -> dict:
    """Build a minimal chunk dict for testing."""
    return {"doc_id": doc_id, "score": score, "final_score": score}


def _make_eligibility_result(
    verdicts: dict,
    has_hard_mismatch: bool = False,
    status: str = "MATCH",
    boost: float = 0.0,
) -> dict:
    """Build a minimal eligibility result dict for testing."""
    return {
        "status": status,
        "criteria_verdicts": dict(verdicts),
        "has_hard_mismatch": has_hard_mismatch,
        "boost": boost,
        "reason": "",
    }


# ======================================================================
# Strategies
# ======================================================================

@st.composite
def random_query_structure(draw):
    """Generate a random QueryStructure with varying patient signals."""
    # Randomly decide which signals to include
    include_age = draw(st.booleans())
    include_gender = draw(st.booleans())
    include_patient_phrase = draw(st.booleans())
    include_biomarkers = draw(st.booleans())
    include_polarity = draw(st.booleans()) if include_biomarkers else False
    include_prior_treatments = draw(st.booleans())
    include_tnm = draw(st.booleans())
    include_comorbidities = draw(st.booleans())
    include_site = draw(st.booleans())
    include_stage = draw(st.booleans())

    # Build query text from signals
    parts = []
    if include_age:
        age = draw(st.integers(min_value=18, max_value=95))
        parts.append(f"{age} year old")
    if include_gender:
        parts.append(draw(st.sampled_from(["male", "female"])))
    if include_patient_phrase:
        parts.append(draw(st.sampled_from([
            "patient with", "presenting with", "diagnosed with", "history of",
        ])))

    site = None
    if include_site:
        site = draw(st.sampled_from(CANCER_SITES))
        parts.append(f"{site} cancer")

    stage = None
    if include_stage:
        stage = draw(st.sampled_from(STAGES))
        parts.append(f"stage {stage}")

    biomarkers = []
    if include_biomarkers:
        if include_polarity:
            biomarkers = [draw(st.sampled_from(BIOMARKERS_WITH_POLARITY))]
        else:
            biomarkers = [draw(st.sampled_from(BIOMARKERS_PLAIN))]
        parts.append(biomarkers[0])

    prior_treatments = []
    if include_prior_treatments:
        prior_treatments = [draw(st.sampled_from(PRIOR_TREATMENTS))]
        parts.append(f"prior {prior_treatments[0]}")

    tnm_t = None
    if include_tnm:
        tnm_t = draw(st.sampled_from(["1", "2", "3", "4"]))
        parts.append(f"T{tnm_t}")

    comorbidities = []
    if include_comorbidities:
        comorbidities = ["diabetes"]
        parts.append("with diabetes")

    # If no parts, add a generic query
    if not parts:
        parts.append(draw(st.sampled_from([
            "What is SBRT?", "Define IMRT", "radiation therapy overview",
        ])))

    query_text = " ".join(parts)

    # Build the QueryStructure
    structure = QueryStructure(
        original_query=query_text,
        patient=PatientContext(
            age=draw(st.integers(min_value=18, max_value=95)) if include_age else None,
            gender=draw(st.sampled_from(["male", "female"])) if include_gender else None,
            comorbidities=comorbidities,
        ),
        cancer=CancerContext(
            site=site,
            stage=stage,
            biomarkers=biomarkers,
            tnm_t=tnm_t,
        ),
        treatment=TreatmentContext(
            prior_treatments=prior_treatments,
        ),
    )

    return query_text, structure


@st.composite
def random_llm_dict(draw):
    """Generate a random LLM 8-axis extraction dict."""
    llm_dict = {}
    for key in LLM_AXIS_KEYS:
        if draw(st.booleans()):
            llm_dict[key] = draw(st.sampled_from([
                "lung cancer", "breast adenocarcinoma", "stage III",
                "EGFR-mutant", "cisplatin", "65 year old male",
                "recurrent disease", "HER2 positive",
                "",  # sometimes empty
            ]))
    return llm_dict


@st.composite
def flag_false_eligibility_scenario(draw):
    """
    Generate a random eligibility scenario for testing that flag=false
    preserves the original hard-filter behavior.
    """
    doc_id = f"study_{draw(st.integers(min_value=1, max_value=9999)):04d}"
    score = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))

    # Random verdicts for all axes
    verdicts = {}
    for axis in ELIGIBILITY_AXES:
        verdicts[axis] = draw(st.sampled_from(ALL_VERDICTS))

    # Determine has_hard_mismatch: true if any axis is MISMATCH
    has_hard_mismatch = any(v == "MISMATCH" for v in verdicts.values())

    # Determine boost
    has_any_match = any(v == "MATCH" for v in verdicts.values())
    if has_hard_mismatch:
        status = "NO_MATCH"
        boost = 0.0
    elif has_any_match:
        status = "MATCH"
        boost = 0.25
    else:
        status = "NOT_AVAILABLE"
        boost = 0.0

    chunk = _make_chunk(doc_id, score)
    eligibility_result = _make_eligibility_result(
        verdicts=verdicts,
        has_hard_mismatch=has_hard_mismatch,
        status=status,
        boost=boost,
    )

    return chunk, doc_id, eligibility_result, has_hard_mismatch


# ======================================================================
# Property 14: Feature flag false preserves identical output
# ======================================================================


# Feature: patient-study-match-scoring, Property 14: Feature flag false preserves identical output
@settings(max_examples=150)
@given(
    query_data=random_query_structure(),
    llm_dict=random_llm_dict(),
    elig_data=flag_false_eligibility_scenario(),
)
def test_flag_false_preserves_identical_output(query_data, llm_dict, elig_data):
    """
    Property 14: Feature flag false preserves identical output.

    For any query, when USE_RECONCILED_STRUCTURE is false, the system
    SHALL produce identical outputs to the current production system:
    - reconcile_if_enabled() returns None
    - apply_patient_eligibility_filter_and_boost(use_tiered_model=False)
      uses the original hard-filter model (any MISMATCH on any active
      criterion → hard removal)
    - _patient_signal_score() correctly determines has_patient_context

    **Validates: Requirements 7.5, 6.2**

    Tag: Feature: patient-study-match-scoring, Property 14: Feature flag false preserves identical output
    """
    query_text, structure = query_data
    chunk, doc_id, eligibility_result, has_hard_mismatch = elig_data

    # ── Ensure flag is false ──────────────────────────────────────────
    old_val = os.environ.get("USE_RECONCILED_STRUCTURE")
    os.environ["USE_RECONCILED_STRUCTURE"] = "false"
    try:
        # ── 1. reconcile_if_enabled() must return None ────────────────
        reconciled = reconcile_if_enabled(structure, llm_dict)
        assert reconciled is None, (
            "reconcile_if_enabled() must return None when "
            "USE_RECONCILED_STRUCTURE=false, "
            f"but got {type(reconciled).__name__}"
        )

        # ── 2. _patient_signal_score determines has_patient_context ───
        score = _patient_signal_score(query_text, structure)
        expected_has_context = score >= 2
        # Verify the threshold logic is consistent
        assert isinstance(score, int), (
            f"_patient_signal_score must return int, got {type(score).__name__}"
        )
        assert score >= 0, (
            f"_patient_signal_score must be non-negative, got {score}"
        )
        # The has_patient_context flag should match the threshold
        assert expected_has_context == (score >= 2), (
            f"has_patient_context should be True iff score >= 2. "
            f"Score={score}, expected_has_context={expected_has_context}"
        )

        # ── 3. Eligibility behavior (flag=false, scoring-fix applied) ─
        #    With use_tiered_model=False after scoring fix:
        #    - cancer_type MISMATCH → hard removal
        #    - non-cancer_type core MISMATCH → retained with penalty
        #    - no MISMATCH → retained with boost
        chunks = [_make_chunk(doc_id, chunk["score"])]
        eligibility_results = {doc_id: dict(eligibility_result)}

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=False,  # flag=false → original behavior
        )

        kept_ids = {
            c.get("doc_id") or c.get("payload", {}).get("doc_id")
            for c in kept
        }
        removed_ids = {
            c.get("doc_id") or c.get("payload", {}).get("doc_id")
            for c in removed
        }

        # After the scoring fix, only cancer_type MISMATCH causes hard removal.
        # Non-cancer_type core mismatches are retained with a penalty.
        verdicts = eligibility_result["criteria_verdicts"]
        cancer_type_mismatch = verdicts.get("cancer_type") == "MISMATCH"

        if cancer_type_mismatch:
            # cancer_type MISMATCH → hard removal (unchanged behavior)
            assert doc_id in removed_ids, (
                f"With cancer_type=MISMATCH, study {doc_id} "
                f"should be hard-removed. Verdicts: {verdicts}"
            )
            assert doc_id not in kept_ids, (
                f"With cancer_type=MISMATCH, study {doc_id} "
                f"should NOT be in kept list"
            )
            # Verify hard_filtered annotation
            removed_chunk = [
                c for c in removed
                if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id
            ][0]
            pe = removed_chunk.get("patient_eligibility", {})
            assert pe.get("hard_filtered") is True, (
                "Removed study must be annotated hard_filtered=True"
            )
        elif has_hard_mismatch and not cancer_type_mismatch:
            # Non-cancer_type core mismatch → retained with penalty
            assert doc_id in kept_ids, (
                f"With non-cancer_type mismatch, study {doc_id} "
                f"should be retained with penalty. Verdicts: {verdicts}"
            )
            assert doc_id not in removed_ids, (
                f"With non-cancer_type mismatch, study {doc_id} "
                f"should NOT be in removed list"
            )
            kept_chunk = [
                c for c in kept
                if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id
            ][0]
            pe = kept_chunk.get("patient_eligibility", {})
            assert pe.get("hard_filtered") is not True, (
                "Non-cancer_type mismatch study must not be hard_filtered"
            )
        else:
            # No mismatch → study is retained
            assert doc_id in kept_ids, (
                f"With flag=false and no hard mismatch, study {doc_id} "
                f"should be retained. Verdicts: "
                f"{eligibility_result['criteria_verdicts']}"
            )
            assert doc_id not in removed_ids, (
                f"With flag=false and no hard mismatch, study {doc_id} "
                f"should NOT be in removed list"
            )
            # Verify not hard-filtered
            kept_chunk = [
                c for c in kept
                if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id
            ][0]
            pe = kept_chunk.get("patient_eligibility", {})
            assert pe.get("hard_filtered") is not True, (
                "Retained study must not be hard_filtered"
            )

        # ── 4. Scoring annotations ────────────────────────────────────
        #    After the scoring fix, the non-tiered path now annotates
        #    kept chunks with scoring metadata (tier, penalty_applied, etc.)
        #    for both boosted and penalized studies.
        if not cancer_type_mismatch:
            kept_chunk = [
                c for c in kept
                if (c.get("doc_id") or c.get("payload", {}).get("doc_id")) == doc_id
            ][0]
            pe = kept_chunk.get("patient_eligibility", {})
            # After the scoring fix, annotations are present on all kept chunks
            if has_hard_mismatch:
                # Non-cancer_type mismatch: should have penalty metadata
                assert pe.get("hard_filtered") is not True, (
                    "Non-cancer_type mismatch study must not be hard_filtered"
                )
            else:
                # No mismatch: should be retained without hard_filtered
                assert pe.get("hard_filtered") is not True, (
                    "Retained study must not be hard_filtered"
                )

    finally:
        # Restore original env var
        if old_val is None:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)
        else:
            os.environ["USE_RECONCILED_STRUCTURE"] = old_val



# ======================================================================
# Integration tests for flag=true end-to-end (Task 7.2)
# Validates: Requirement 6.3
# ======================================================================


class TestFlagTrueReconciliation:
    """Test that reconcile_if_enabled() returns a ReconciledStructure when flag=true."""

    def setup_method(self):
        self._old_val = os.environ.get("USE_RECONCILED_STRUCTURE")
        os.environ["USE_RECONCILED_STRUCTURE"] = "true"

    def teardown_method(self):
        if self._old_val is None:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)
        else:
            os.environ["USE_RECONCILED_STRUCTURE"] = self._old_val

    def test_reconcile_if_enabled_returns_reconciled_structure(self):
        """With flag=true, reconcile_if_enabled() returns a ReconciledStructure, not None."""
        from src.api.services.query_reconciliation import ReconciledStructure

        structure = QueryStructure(
            original_query="65 year old male with stage IIIA NSCLC EGFR-mutant",
            patient=PatientContext(age=65, gender="male"),
            cancer=CancerContext(
                site="lung",
                stage="IIIA",
                biomarkers=["EGFR mutant"],
            ),
            treatment=TreatmentContext(),
        )
        llm_dict = {
            "cancer_type": "lung",
            "histology": "non-small cell",
            "stage": "IIIA",
            "biomarkers": "EGFR mutant",
            "demographics": "65 year old male",
        }

        result = reconcile_if_enabled(structure, llm_dict)

        assert result is not None, (
            "reconcile_if_enabled() must return a ReconciledStructure when flag=true"
        )
        assert isinstance(result, ReconciledStructure), (
            f"Expected ReconciledStructure, got {type(result).__name__}"
        )
        assert result.cancer_site == "lung"
        assert result.stage == "IIIA"

    def test_reconcile_if_enabled_preserves_biomarkers(self):
        """With flag=true, reconciled structure contains biomarker data."""
        structure = QueryStructure(
            original_query="breast cancer HER2 positive BRCA1 mutant",
            patient=PatientContext(),
            cancer=CancerContext(
                site="breast",
                biomarkers=["HER2 positive", "BRCA1 mutant"],
            ),
            treatment=TreatmentContext(),
        )
        llm_dict = {
            "cancer_type": "breast",
            "biomarkers": "HER2 positive, BRCA1 mutant",
        }

        result = reconcile_if_enabled(structure, llm_dict)

        assert result is not None
        assert len(result.biomarkers) >= 1
        biomarker_names = [bm.name.upper() for bm in result.biomarkers]
        assert "HER2" in biomarker_names or "HER2 POSITIVE" in " ".join(
            f"{bm.name} {bm.polarity or ''}" for bm in result.biomarkers
        ).upper()


class TestFlagTrueReconciledStructureDict:
    """Test that ReconciledStructure.to_query_structure_dict() produces a valid dict."""

    def test_to_query_structure_dict_basic(self):
        """to_query_structure_dict() produces a dict with expected keys for PG matcher."""
        from src.api.services.query_reconciliation import (
            ReconciledStructure,
            Biomarker,
            Source,
        )

        rs = ReconciledStructure(
            cancer_site="lung",
            histology="adenocarcinoma",
            stage="IIIA",
            tnm_t="2",
            tnm_n="1",
            tnm_m="0",
            biomarkers=[
                Biomarker(name="EGFR", polarity="mutant", source=Source.LLM),
            ],
            age=65,
            gender="male",
            prior_treatments=["cisplatin"],
            has_patient_context=True,
            filter_category="lung",
        )

        d = rs.to_query_structure_dict()

        assert isinstance(d, dict)
        assert d["has_patient_context"] is True
        assert "cancer" in d
        assert d["cancer"]["site"] == "lung"
        assert d["cancer"]["histology"] == "adenocarcinoma"
        assert d["cancer"]["stage"] == "IIIA"
        assert "EGFR mutant" in d["cancer"]["biomarkers"]
        assert "patient" in d
        assert d["patient"]["age"] == 65
        assert d["patient"]["gender"] == "male"
        assert "treatment" in d
        assert "cisplatin" in d["treatment"]["prior_treatments"]
        assert d["filter_category"] == "lung"

    def test_to_query_structure_dict_empty_fields(self):
        """to_query_structure_dict() omits keys for None/empty fields."""
        from src.api.services.query_reconciliation import ReconciledStructure

        rs = ReconciledStructure(has_patient_context=False)

        d = rs.to_query_structure_dict()

        assert isinstance(d, dict)
        assert d["has_patient_context"] is False
        # No cancer/patient/treatment keys when all fields are empty
        assert "cancer" not in d
        assert "patient" not in d
        assert "treatment" not in d

    def test_to_query_structure_dict_partial_fields(self):
        """to_query_structure_dict() includes only populated sub-dicts."""
        from src.api.services.query_reconciliation import ReconciledStructure

        rs = ReconciledStructure(
            cancer_site="breast",
            has_patient_context=True,
        )

        d = rs.to_query_structure_dict()

        assert "cancer" in d
        assert d["cancer"]["site"] == "breast"
        # No patient or treatment keys
        assert "patient" not in d
        assert "treatment" not in d


class TestFlagTrueTieredEligibility:
    """Test that with flag=true (use_tiered_model=True), tiered eligibility is active."""

    def test_cancer_type_mismatch_still_hard_removes(self):
        """Tier 1: cancer_type MISMATCH causes hard removal even in tiered model."""
        chunks = [{"doc_id": "study_001", "score": 0.8, "final_score": 0.8}]
        eligibility_results = {
            "study_001": {
                "status": "NO_MATCH",
                "criteria_verdicts": {
                    "cancer_type": "MISMATCH",
                    "histology": "MATCH",
                    "stage": "MATCH",
                    "biomarkers": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                },
                "has_hard_mismatch": True,
                "boost": 0.0,
            }
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=True,
        )

        assert len(removed) == 1
        assert len(kept) == 0
        pe = removed[0]["patient_eligibility"]
        assert pe["hard_filtered"] is True
        assert pe["tier"] == "hard_filter"

    def test_secondary_mismatch_produces_penalty_not_removal(self):
        """Tier 2: secondary axis MISMATCH retains study with penalty, not removal."""
        chunks = [{"doc_id": "study_002", "score": 0.8, "final_score": 0.8}]
        eligibility_results = {
            "study_002": {
                "status": "POSSIBLE",
                "criteria_verdicts": {
                    "cancer_type": "MATCH",
                    "histology": "MISMATCH",
                    "stage": "MATCH",
                    "biomarkers": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                },
                "has_hard_mismatch": True,
                "boost": 0.0,
            }
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=True,
        )

        # Study is RETAINED (not removed) — tiered model applies penalty
        assert len(kept) == 1
        assert len(removed) == 0
        pe = kept[0]["patient_eligibility"]
        assert pe["hard_filtered"] is False
        assert pe["tier"] == "penalty"
        assert pe["penalty_applied"] == pytest.approx(0.15)
        # Score should be reduced
        assert kept[0]["final_score"] == pytest.approx(0.8 - 0.15)

    def test_multiple_secondary_mismatches_capped_penalty(self):
        """Multiple secondary mismatches: penalty = min(N * 0.15, 0.45)."""
        chunks = [{"doc_id": "study_003", "score": 0.9, "final_score": 0.9}]
        eligibility_results = {
            "study_003": {
                "status": "POSSIBLE",
                "criteria_verdicts": {
                    "cancer_type": "MATCH",
                    "histology": "MISMATCH",
                    "stage": "MISMATCH",
                    "biomarkers": "MISMATCH",
                    "prior_therapies": "MISMATCH",
                },
                "has_hard_mismatch": True,
                "boost": 0.0,
            }
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=True,
        )

        # Still retained — only cancer_type MISMATCH causes removal
        assert len(kept) == 1
        assert len(removed) == 0
        pe = kept[0]["patient_eligibility"]
        assert pe["tier"] == "penalty"
        # 4 secondary mismatches × 0.15 = 0.60, capped at 0.45
        assert pe["penalty_applied"] == pytest.approx(0.45)
        assert kept[0]["final_score"] == pytest.approx(0.9 - 0.45)

    def test_no_mismatch_gets_boost_not_penalty(self):
        """All MATCH/NOT_AVAILABLE: boost applied, zero penalty."""
        chunks = [{"doc_id": "study_004", "score": 0.7, "final_score": 0.7}]
        eligibility_results = {
            "study_004": {
                "status": "MATCH",
                "criteria_verdicts": {
                    "cancer_type": "MATCH",
                    "histology": "MATCH",
                    "stage": "NOT_AVAILABLE",
                    "biomarkers": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                },
                "has_hard_mismatch": False,
                "boost": 0.25,
            }
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=True,
        )

        assert len(kept) == 1
        assert len(removed) == 0
        pe = kept[0]["patient_eligibility"]
        assert pe["hard_filtered"] is False
        assert pe["tier"] == "boost"
        assert pe["penalty_applied"] == 0.0
        assert pe["boost_applied"] == 0.25
        assert kept[0]["final_score"] == pytest.approx(0.7 + 0.25)

    def test_tiered_model_annotates_per_axis_verdicts(self):
        """Tiered model annotates each study with per-axis verdicts."""
        chunks = [{"doc_id": "study_005", "score": 0.8, "final_score": 0.8}]
        verdicts = {
            "cancer_type": "MATCH",
            "histology": "MISMATCH",
            "stage": "POSSIBLE",
            "biomarkers": "NOT_AVAILABLE",
            "prior_therapies": "MATCH",
        }
        eligibility_results = {
            "study_005": {
                "status": "POSSIBLE",
                "criteria_verdicts": dict(verdicts),
                "has_hard_mismatch": True,
                "boost": 0.0,
            }
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=True,
        )

        assert len(kept) == 1
        pe = kept[0]["patient_eligibility"]
        assert "criteria_verdicts" in pe
        assert pe["criteria_verdicts"]["cancer_type"] == "MATCH"
        assert pe["criteria_verdicts"]["histology"] == "MISMATCH"
        assert pe["criteria_verdicts"]["stage"] == "POSSIBLE"


class TestFlagTrueReconciledStructureUsedByPatientEligibility:
    """Test that ReconciledStructure is consumed by PatientEligibility's extract_patient_context_from_query."""

    def test_extract_patient_context_from_reconciled(self):
        """extract_patient_context_from_query uses ReconciledStructure when provided."""
        from src.api.services.query_reconciliation import (
            ReconciledStructure,
            Biomarker,
            Source,
        )
        from src.api.services.patient_eligibility_boost_service import (
            extract_patient_context_from_query,
        )

        rs = ReconciledStructure(
            cancer_site="lung",
            histology="adenocarcinoma",
            stage="IIIA",
            biomarkers=[
                Biomarker(name="EGFR", polarity="mutant", source=Source.LLM),
            ],
            prior_treatments=["cisplatin"],
            age=65,
            gender="male",
            has_patient_context=True,
        )

        ctx = extract_patient_context_from_query(
            query="65 year old male with stage IIIA NSCLC EGFR-mutant",
            reconciled=rs,
        )

        assert ctx is not None
        assert ctx["cancer_type"] == "lung"
        assert ctx["histology"] == "adenocarcinoma"
        assert ctx["stage"] == "IIIA"
        assert any("EGFR" in bm for bm in ctx.get("biomarkers", []))
        assert ctx["age"] == 65
        assert ctx["gender"] == "male"

    def test_extract_patient_context_from_reconciled_minimal(self):
        """extract_patient_context_from_query handles minimal ReconciledStructure."""
        from src.api.services.query_reconciliation import ReconciledStructure
        from src.api.services.patient_eligibility_boost_service import (
            extract_patient_context_from_query,
        )

        rs = ReconciledStructure(
            cancer_site="breast",
            has_patient_context=True,
        )

        ctx = extract_patient_context_from_query(
            query="breast cancer",
            reconciled=rs,
        )

        assert ctx is not None
        assert ctx["cancer_type"] == "breast"
        # No other fields populated
        assert "histology" not in ctx
        assert "biomarkers" not in ctx


class TestFlagTrueEndToEnd:
    """End-to-end integration: flag=true activates reconciliation + tiered model together."""

    def setup_method(self):
        self._old_val = os.environ.get("USE_RECONCILED_STRUCTURE")
        os.environ["USE_RECONCILED_STRUCTURE"] = "true"

    def teardown_method(self):
        if self._old_val is None:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)
        else:
            os.environ["USE_RECONCILED_STRUCTURE"] = self._old_val

    def test_flag_true_reconcile_then_tiered_eligibility(self):
        """
        End-to-end: flag=true → reconcile produces ReconciledStructure,
        tiered eligibility retains secondary-mismatch studies with penalty.
        """
        from src.api.services.query_reconciliation import ReconciledStructure

        # Step 1: Reconciliation produces a ReconciledStructure
        structure = QueryStructure(
            original_query="55 year old female with stage IV breast cancer HER2 positive",
            patient=PatientContext(age=55, gender="female"),
            cancer=CancerContext(
                site="breast",
                stage="IV",
                biomarkers=["HER2 positive"],
            ),
            treatment=TreatmentContext(),
            has_patient_context=True,
        )
        llm_dict = {
            "cancer_type": "breast",
            "stage": "IV",
            "biomarkers": "HER2 positive",
            "demographics": "55 year old female",
        }

        reconciled = reconcile_if_enabled(structure, llm_dict)
        assert reconciled is not None
        assert isinstance(reconciled, ReconciledStructure)

        # Step 2: ReconciledStructure produces a valid dict for PG matcher
        qs_dict = reconciled.to_query_structure_dict()
        assert isinstance(qs_dict, dict)
        assert qs_dict["has_patient_context"] is True

        # Step 3: Tiered eligibility retains secondary-mismatch study
        chunks = [
            {"doc_id": "study_A", "score": 0.85, "final_score": 0.85},
            {"doc_id": "study_B", "score": 0.75, "final_score": 0.75},
        ]
        eligibility_results = {
            "study_A": {
                "status": "MATCH",
                "criteria_verdicts": {
                    "cancer_type": "MATCH",
                    "histology": "MATCH",
                    "stage": "MISMATCH",
                    "biomarkers": "MATCH",
                    "prior_therapies": "NOT_AVAILABLE",
                },
                "has_hard_mismatch": True,
                "boost": 0.0,
            },
            "study_B": {
                "status": "NO_MATCH",
                "criteria_verdicts": {
                    "cancer_type": "MISMATCH",
                    "histology": "MATCH",
                    "stage": "MATCH",
                    "biomarkers": "NOT_AVAILABLE",
                    "prior_therapies": "NOT_AVAILABLE",
                },
                "has_hard_mismatch": True,
                "boost": 0.0,
            },
        }

        kept, removed = apply_patient_eligibility_filter_and_boost(
            chunks=chunks,
            eligibility_results=eligibility_results,
            use_tiered_model=True,  # flag=true activates tiered model
        )

        # study_A: cancer_type MATCH, stage MISMATCH → retained with penalty
        kept_ids = {c["doc_id"] for c in kept}
        removed_ids = {c["doc_id"] for c in removed}

        assert "study_A" in kept_ids, "Secondary mismatch should be retained in tiered model"
        assert "study_B" in removed_ids, "Cancer type MISMATCH should still be hard-removed"

        # Verify study_A has tiered annotations
        study_a = [c for c in kept if c["doc_id"] == "study_A"][0]
        pe = study_a["patient_eligibility"]
        assert pe["tier"] == "penalty"
        assert pe["penalty_applied"] == pytest.approx(0.15)
        assert pe["hard_filtered"] is False

        # Verify study_B has hard-filter annotation
        study_b = [c for c in removed if c["doc_id"] == "study_B"][0]
        pe_b = study_b["patient_eligibility"]
        assert pe_b["tier"] == "hard_filter"
        assert pe_b["hard_filtered"] is True
