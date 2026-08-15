"""
Property-based tests for query reconciliation.

Feature: patient-study-match-scoring

Tests the following properties:
- Property 2: Reconciliation produces a valid ReconciledStructure
- Property 3: Reconciliation biomarker priority — LLM wins on polarity
- Property 4: Reconciliation site priority — regex wins
- Property 5: Reconciliation agreement passthrough

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.api.services.query_structuring_service import (
    QueryStructure,
    PatientContext,
    CancerContext,
    TreatmentContext,
    ClinicalHistory,
)
from src.api.services.query_reconciliation import (
    reconcile,
    ReconciledStructure,
    Source,
    Biomarker,
)


# ======================================================================
# Shared strategies and helpers
# ======================================================================

# Canonical cancer sites for generating realistic test data
CANCER_SITES = [
    "lung", "breast", "prostate", "colorectal", "head_neck",
    "pancreatic", "ovarian", "melanoma", "bladder", "renal",
    "liver", "gastric", "esophageal", "thyroid", "cervical",
]

HISTOLOGY_TYPES = [
    "adenocarcinoma", "squamous_cell_carcinoma", "small_cell",
    "large_cell", "ductal", "lobular", "transitional_cell",
    "clear_cell", "papillary", "serous",
]

STAGES = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV", "IVA", "IVB"]

BIOMARKER_NAMES = [
    "EGFR", "ALK", "KRAS", "BRAF", "HER2", "PD-L1",
    "BRCA1", "BRCA2", "ROS1", "MET", "RET", "NTRK",
    "ER", "PR", "MSI",
]

POLARITIES = ["mutant", "wild-type", "positive", "negative", "amplified", "overexpressed"]

GENDERS = ["male", "female"]

TREATMENT_MODALITIES = ["radiation", "chemotherapy", "immunotherapy", "surgery", "targeted_therapy"]

TREATMENT_SETTINGS = ["adjuvant", "neoadjuvant", "definitive", "palliative", "concurrent"]

PRIOR_TREATMENTS = [
    "cisplatin", "carboplatin", "pembrolizumab", "nivolumab",
    "docetaxel", "paclitaxel", "doxorubicin", "gemcitabine",
]

# The 8 LLM axes as defined in the design
LLM_AXIS_KEYS = [
    "cancer_type", "histology", "stage", "biomarkers",
    "prior_treatments", "treatment_setting", "demographics", "comorbidities",
]

# Strategies for generating random field values
site_strategy = st.sampled_from(CANCER_SITES)
histology_strategy = st.sampled_from(HISTOLOGY_TYPES)
stage_strategy = st.sampled_from(STAGES)
biomarker_name_strategy = st.sampled_from(BIOMARKER_NAMES)
polarity_strategy = st.sampled_from(POLARITIES)
gender_strategy = st.sampled_from(GENDERS)
age_strategy = st.integers(min_value=18, max_value=99)
treatment_strategy = st.sampled_from(PRIOR_TREATMENTS)
modality_strategy = st.sampled_from(TREATMENT_MODALITIES)
setting_strategy = st.sampled_from(TREATMENT_SETTINGS)


def _build_query_structure(
    site=None,
    histology=None,
    stage=None,
    tnm_t=None,
    tnm_n=None,
    tnm_m=None,
    biomarkers=None,
    receptor_status=None,
    age=None,
    gender=None,
    performance_status=None,
    prior_treatments=None,
    modality=None,
    setting=None,
):
    """Build a QueryStructure with the given fields populated."""
    patient = PatientContext(
        age=age,
        gender=gender,
        performance_status=performance_status,
    )
    cancer = CancerContext(
        site=site,
        histology=histology,
        stage=stage,
        tnm_t=tnm_t,
        tnm_n=tnm_n,
        tnm_m=tnm_m,
        biomarkers=biomarkers or [],
        receptor_status=receptor_status,
    )
    treatment = TreatmentContext(
        modality=modality,
        setting=setting,
        prior_treatments=prior_treatments or [],
    )
    return QueryStructure(
        original_query="test query",
        patient=patient,
        cancer=cancer,
        treatment=treatment,
        clinical_history=ClinicalHistory(),
    )


def _build_llm_dict(**kwargs):
    """Build an LLM 8-axis dict with the given key-value pairs."""
    llm = {}
    for key in LLM_AXIS_KEYS:
        llm[key] = kwargs.get(key, "")
    return llm



# ======================================================================
# Property 2: Reconciliation produces a valid ReconciledStructure
# ======================================================================

# Composite strategy for generating random QueryStructure + LLM dict pairs
@st.composite
def query_structure_and_llm_dict(draw):
    """Generate a random QueryStructure and a random LLM dict with 8-axis keys."""
    # Randomly decide which fields to populate in the regex structure
    site = draw(st.one_of(st.none(), site_strategy))
    histology = draw(st.one_of(st.none(), histology_strategy))
    stage = draw(st.one_of(st.none(), stage_strategy))
    tnm_t = draw(st.one_of(st.none(), st.sampled_from(["1", "2", "3", "4"])))
    tnm_n = draw(st.one_of(st.none(), st.sampled_from(["0", "1", "2", "3"])))
    tnm_m = draw(st.one_of(st.none(), st.sampled_from(["0", "1"])))
    biomarker_list = draw(st.lists(
        st.builds(
            lambda name, pol: f"{name} {pol}",
            biomarker_name_strategy,
            polarity_strategy,
        ),
        min_size=0,
        max_size=3,
    ))
    receptor_status = draw(st.one_of(st.none(), st.sampled_from(["ER+", "PR+", "HER2+", "triple-negative"])))
    age = draw(st.one_of(st.none(), age_strategy))
    gender = draw(st.one_of(st.none(), gender_strategy))
    perf_status = draw(st.one_of(st.none(), st.sampled_from(["ECOG 0", "ECOG 1", "ECOG 2"])))
    prior_tx = draw(st.lists(treatment_strategy, min_size=0, max_size=3))
    modality = draw(st.one_of(st.none(), modality_strategy))
    setting = draw(st.one_of(st.none(), setting_strategy))

    regex_struct = _build_query_structure(
        site=site,
        histology=histology,
        stage=stage,
        tnm_t=tnm_t,
        tnm_n=tnm_n,
        tnm_m=tnm_m,
        biomarkers=biomarker_list,
        receptor_status=receptor_status,
        age=age,
        gender=gender,
        performance_status=perf_status,
        prior_treatments=prior_tx,
        modality=modality,
        setting=setting,
    )

    # Build a random LLM dict with 8-axis keys
    llm_dict = {}
    for key in LLM_AXIS_KEYS:
        # Randomly populate or leave empty
        if draw(st.booleans()):
            if key == "cancer_type":
                llm_dict[key] = draw(site_strategy)
            elif key == "histology":
                llm_dict[key] = draw(histology_strategy)
            elif key == "stage":
                llm_dict[key] = draw(stage_strategy)
            elif key == "biomarkers":
                name = draw(biomarker_name_strategy)
                pol = draw(polarity_strategy)
                llm_dict[key] = f"{name} {pol}"
            elif key == "prior_treatments":
                llm_dict[key] = draw(treatment_strategy)
            elif key == "treatment_setting":
                llm_dict[key] = draw(setting_strategy)
            elif key == "demographics":
                a = draw(age_strategy)
                g = draw(gender_strategy)
                llm_dict[key] = f"{a} year old {g}"
            elif key == "comorbidities":
                llm_dict[key] = draw(st.sampled_from(["diabetes", "hypertension", "COPD", ""]))
            else:
                llm_dict[key] = ""
        else:
            llm_dict[key] = ""

    return regex_struct, llm_dict


# Feature: patient-study-match-scoring, Property 2: Reconciliation produces a valid ReconciledStructure
@settings(max_examples=150)
@given(data=query_structure_and_llm_dict())
def test_reconciliation_produces_valid_reconciled_structure(data):
    """
    Property 2: Reconciliation produces a valid ReconciledStructure.

    For any pair of regex-extracted QueryStructure and LLM-extracted 8-axis dict,
    reconcile() SHALL return a ReconciledStructure that contains all fields present
    in the input QueryStructure plus the LLM-extracted 8-axis text spans.

    **Validates: Requirements 2.1, 2.6**

    Tag: Feature: patient-study-match-scoring, Property 2: Reconciliation produces a valid ReconciledStructure
    """
    regex_struct, llm_dict = data

    result = reconcile(regex_struct, llm_dict)

    # Must return a ReconciledStructure
    assert isinstance(result, ReconciledStructure), (
        f"reconcile() must return a ReconciledStructure, got {type(result)}"
    )

    # All required fields must exist (not raise AttributeError)
    # Cancer context fields
    assert hasattr(result, "cancer_site")
    assert hasattr(result, "cancer_site_source")
    assert hasattr(result, "histology")
    assert hasattr(result, "histology_source")
    assert hasattr(result, "stage")
    assert hasattr(result, "tnm_t")
    assert hasattr(result, "tnm_n")
    assert hasattr(result, "tnm_m")
    assert hasattr(result, "biomarkers")
    assert hasattr(result, "receptor_status")

    # Patient context fields
    assert hasattr(result, "age")
    assert hasattr(result, "gender")
    assert hasattr(result, "performance_status")

    # Treatment context fields
    assert hasattr(result, "prior_treatments")
    assert hasattr(result, "treatment_setting")
    assert hasattr(result, "treatment_modality")

    # LLM axes preserved
    assert hasattr(result, "llm_axes")
    assert isinstance(result.llm_axes, dict)

    # Metadata fields
    assert hasattr(result, "filter_category")
    assert hasattr(result, "has_patient_context")
    assert hasattr(result, "disagreements")
    assert isinstance(result.disagreements, list)

    # Trajectory fields
    assert hasattr(result, "disease_trajectory")
    assert hasattr(result, "metastatic_status")
    assert hasattr(result, "risk_level")

    # Biomarkers must be a list of Biomarker instances
    assert isinstance(result.biomarkers, list)
    for bm in result.biomarkers:
        assert isinstance(bm, Biomarker), (
            f"Each biomarker must be a Biomarker instance, got {type(bm)}"
        )
        assert bm.name is not None and len(bm.name) > 0, (
            "Biomarker name must be non-empty"
        )

    # When regex provides a site, it must appear in the output
    if regex_struct.cancer.site:
        assert result.cancer_site is not None, (
            f"Regex provided site '{regex_struct.cancer.site}' but reconciled site is None"
        )

    # When regex provides age, it must appear in the output
    if regex_struct.patient.age is not None:
        assert result.age is not None, (
            f"Regex provided age {regex_struct.patient.age} but reconciled age is None"
        )

    # When regex provides gender, it must appear in the output
    if regex_struct.patient.gender is not None:
        assert result.gender is not None, (
            f"Regex provided gender '{regex_struct.patient.gender}' but reconciled gender is None"
        )

    # LLM axes dict should contain the provided LLM keys
    for key, value in llm_dict.items():
        if value:  # non-empty LLM values should be preserved
            assert key in result.llm_axes, (
                f"LLM axis '{key}' with value '{value}' not found in llm_axes"
            )


# ======================================================================
# Property 3: Reconciliation biomarker priority — LLM wins on polarity
# ======================================================================

@st.composite
def biomarker_polarity_disagreement(draw):
    """
    Generate a random biomarker name and two different polarity values
    (one for regex, one for LLM) that disagree.
    """
    name = draw(biomarker_name_strategy)
    regex_polarity = draw(polarity_strategy)
    llm_polarity = draw(polarity_strategy)
    # Ensure they disagree
    assume(regex_polarity != llm_polarity)
    return name, regex_polarity, llm_polarity


# Feature: patient-study-match-scoring, Property 3: Reconciliation biomarker priority — LLM wins on polarity
@settings(max_examples=150)
@given(data=biomarker_polarity_disagreement())
def test_reconciliation_biomarker_llm_wins_on_polarity(data):
    """
    Property 3: Reconciliation biomarker priority — LLM wins on polarity.

    For any biomarker where the regex extraction and LLM extraction disagree
    on polarity, the reconciled output SHALL use the LLM-extracted polarity,
    and the disagreement SHALL be recorded in ReconciledStructure.disagreements.

    **Validates: Requirements 2.2, 2.5**

    Tag: Feature: patient-study-match-scoring, Property 3: Reconciliation biomarker priority — LLM wins on polarity
    """
    biomarker_name, regex_polarity, llm_polarity = data

    # Build regex structure with the biomarker at regex polarity
    regex_struct = _build_query_structure(
        site="lung",
        biomarkers=[f"{biomarker_name} {regex_polarity}"],
    )

    # Build LLM dict with the biomarker at LLM polarity
    llm_dict = _build_llm_dict(
        biomarkers=f"{biomarker_name} {llm_polarity}",
        cancer_type="lung",
    )

    result = reconcile(regex_struct, llm_dict)

    assert isinstance(result, ReconciledStructure)

    # Find the reconciled biomarker matching our name
    matching_biomarkers = [
        bm for bm in result.biomarkers
        if bm.name.upper() == biomarker_name.upper()
    ]
    assert len(matching_biomarkers) >= 1, (
        f"Expected biomarker '{biomarker_name}' in reconciled output, "
        f"got biomarkers: {[bm.name for bm in result.biomarkers]}"
    )

    reconciled_bm = matching_biomarkers[0]

    # LLM polarity must win
    assert reconciled_bm.polarity == llm_polarity, (
        f"LLM polarity should win: expected '{llm_polarity}', "
        f"got '{reconciled_bm.polarity}' for biomarker '{biomarker_name}'. "
        f"Regex had '{regex_polarity}'."
    )

    # Disagreement must be recorded
    assert len(result.disagreements) >= 1, (
        f"Expected at least 1 disagreement logged for biomarker polarity conflict "
        f"(regex='{regex_polarity}', llm='{llm_polarity}'), "
        f"got {len(result.disagreements)} disagreements"
    )

    # Find the specific biomarker disagreement
    biomarker_disagreements = [
        d for d in result.disagreements
        if "biomarker" in str(d).lower() or biomarker_name.lower() in str(d).lower()
    ]
    assert len(biomarker_disagreements) >= 1, (
        f"Expected a disagreement entry mentioning biomarker '{biomarker_name}', "
        f"got disagreements: {result.disagreements}"
    )


# ======================================================================
# Property 4: Reconciliation site priority — regex wins
# ======================================================================

@st.composite
def site_disagreement(draw):
    """Generate two different cancer site names for regex vs LLM disagreement."""
    regex_site = draw(site_strategy)
    llm_site = draw(site_strategy)
    assume(regex_site != llm_site)
    return regex_site, llm_site


# Feature: patient-study-match-scoring, Property 4: Reconciliation site priority — regex wins
@settings(max_examples=150)
@given(data=site_disagreement())
def test_reconciliation_site_regex_wins(data):
    """
    Property 4: Reconciliation site priority — regex wins.

    For any cancer site where the regex extraction and LLM extraction disagree,
    the reconciled output SHALL use the regex-extracted site value.

    **Validates: Requirements 2.3**

    Tag: Feature: patient-study-match-scoring, Property 4: Reconciliation site priority — regex wins
    """
    regex_site, llm_site = data

    regex_struct = _build_query_structure(site=regex_site)

    llm_dict = _build_llm_dict(cancer_type=llm_site)

    result = reconcile(regex_struct, llm_dict)

    assert isinstance(result, ReconciledStructure)

    # Regex site must win
    assert result.cancer_site == regex_site, (
        f"Regex site should win: expected '{regex_site}', "
        f"got '{result.cancer_site}'. LLM had '{llm_site}'."
    )

    # Source should indicate regex
    assert result.cancer_site_source == Source.REGEX, (
        f"cancer_site_source should be REGEX when regex wins, "
        f"got {result.cancer_site_source}"
    )


# ======================================================================
# Property 5: Reconciliation agreement passthrough
# ======================================================================

# Fields where we test agreement passthrough
AGREEMENT_FIELDS = ["site", "histology", "stage", "gender"]


@st.composite
def agreed_field_values(draw):
    """
    Generate a random field name and a value that both regex and LLM agree on.
    Returns (field_name, agreed_value).
    """
    field_name = draw(st.sampled_from(AGREEMENT_FIELDS))

    if field_name == "site":
        value = draw(site_strategy)
    elif field_name == "histology":
        value = draw(histology_strategy)
    elif field_name == "stage":
        value = draw(stage_strategy)
    elif field_name == "gender":
        value = draw(gender_strategy)
    else:
        value = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))))

    return field_name, value


# Feature: patient-study-match-scoring, Property 5: Reconciliation agreement passthrough
@settings(max_examples=150)
@given(data=agreed_field_values())
def test_reconciliation_agreement_passthrough(data):
    """
    Property 5: Reconciliation agreement passthrough.

    For any field where the regex extraction and LLM extraction agree on the value,
    the reconciled output SHALL contain that exact value without modification.

    **Validates: Requirements 2.4**

    Tag: Feature: patient-study-match-scoring, Property 5: Reconciliation agreement passthrough
    """
    field_name, agreed_value = data

    # Build regex structure and LLM dict with the same value for the field
    if field_name == "site":
        regex_struct = _build_query_structure(site=agreed_value)
        llm_dict = _build_llm_dict(cancer_type=agreed_value)
        result = reconcile(regex_struct, llm_dict)
        assert result.cancer_site == agreed_value, (
            f"Agreed site should pass through: expected '{agreed_value}', "
            f"got '{result.cancer_site}'"
        )

    elif field_name == "histology":
        regex_struct = _build_query_structure(histology=agreed_value)
        llm_dict = _build_llm_dict(histology=agreed_value)
        result = reconcile(regex_struct, llm_dict)
        assert result.histology == agreed_value, (
            f"Agreed histology should pass through: expected '{agreed_value}', "
            f"got '{result.histology}'"
        )

    elif field_name == "stage":
        regex_struct = _build_query_structure(stage=agreed_value)
        llm_dict = _build_llm_dict(stage=agreed_value)
        result = reconcile(regex_struct, llm_dict)
        assert result.stage == agreed_value, (
            f"Agreed stage should pass through: expected '{agreed_value}', "
            f"got '{result.stage}'"
        )

    elif field_name == "gender":
        regex_struct = _build_query_structure(gender=agreed_value)
        llm_dict = _build_llm_dict(demographics=f"55 year old {agreed_value}")
        result = reconcile(regex_struct, llm_dict)
        assert result.gender == agreed_value, (
            f"Agreed gender should pass through: expected '{agreed_value}', "
            f"got '{result.gender}'"
        )

    # When fields agree, the source should be AGREED
    assert isinstance(result, ReconciledStructure)

    # No disagreements should be logged for the agreed field
    field_disagreements = [
        d for d in result.disagreements
        if field_name.lower() in str(d).lower()
    ]
    assert len(field_disagreements) == 0, (
        f"No disagreement should be logged for agreed field '{field_name}' "
        f"with value '{agreed_value}', but got: {field_disagreements}"
    )


# ======================================================================
# Task 3.4: Feature flag gating — reconcile_if_enabled()
# ======================================================================

import os
from unittest.mock import patch
from src.api.services.query_reconciliation import reconcile_if_enabled


class TestReconcileIfEnabled:
    """Tests for reconcile_if_enabled() feature flag gating.

    **Validates: Requirements 2.7**
    """

    def _make_inputs(self):
        """Build a minimal regex structure + LLM dict for testing."""
        regex_struct = _build_query_structure(site="lung", stage="IIIA")
        llm_dict = _build_llm_dict(cancer_type="lung", stage="IIIA")
        return regex_struct, llm_dict

    def test_returns_none_when_flag_is_false(self):
        """When USE_RECONCILED_STRUCTURE is false, reconcile_if_enabled() returns None."""
        regex_struct, llm_dict = self._make_inputs()
        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "false"}):
            result = reconcile_if_enabled(regex_struct, llm_dict)
        assert result is None, (
            f"Expected None when flag is false, got {type(result)}"
        )

    def test_returns_none_when_flag_is_not_set(self):
        """When USE_RECONCILED_STRUCTURE is not set, reconcile_if_enabled() returns None (default false)."""
        regex_struct, llm_dict = self._make_inputs()
        env = os.environ.copy()
        env.pop("USE_RECONCILED_STRUCTURE", None)
        with patch.dict(os.environ, env, clear=True):
            result = reconcile_if_enabled(regex_struct, llm_dict)
        assert result is None, (
            f"Expected None when flag is not set, got {type(result)}"
        )

    def test_returns_reconciled_structure_when_flag_is_true(self):
        """When USE_RECONCILED_STRUCTURE is true, reconcile_if_enabled() returns a ReconciledStructure."""
        regex_struct, llm_dict = self._make_inputs()
        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "true"}):
            result = reconcile_if_enabled(regex_struct, llm_dict)
        assert isinstance(result, ReconciledStructure), (
            f"Expected ReconciledStructure when flag is true, got {type(result)}"
        )
        assert result.cancer_site == "lung"
        assert result.stage == "IIIA"

    def test_returns_reconciled_structure_when_flag_is_1(self):
        """When USE_RECONCILED_STRUCTURE is '1', reconcile_if_enabled() returns a ReconciledStructure."""
        regex_struct, llm_dict = self._make_inputs()
        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "1"}):
            result = reconcile_if_enabled(regex_struct, llm_dict)
        assert isinstance(result, ReconciledStructure), (
            f"Expected ReconciledStructure when flag is '1', got {type(result)}"
        )

    def test_returns_reconciled_structure_when_flag_is_yes(self):
        """When USE_RECONCILED_STRUCTURE is 'yes', reconcile_if_enabled() returns a ReconciledStructure."""
        regex_struct, llm_dict = self._make_inputs()
        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "yes"}):
            result = reconcile_if_enabled(regex_struct, llm_dict)
        assert isinstance(result, ReconciledStructure), (
            f"Expected ReconciledStructure when flag is 'yes', got {type(result)}"
        )

    def test_returns_none_for_unrecognized_flag_value(self):
        """When USE_RECONCILED_STRUCTURE has an unrecognized value, reconcile_if_enabled() returns None."""
        regex_struct, llm_dict = self._make_inputs()
        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "maybe"}):
            result = reconcile_if_enabled(regex_struct, llm_dict)
        assert result is None, (
            f"Expected None for unrecognized flag value 'maybe', got {type(result)}"
        )

    def test_downstream_uses_original_when_flag_false(self):
        """When flag is false, None result signals downstream to use original QueryStructure."""
        regex_struct, llm_dict = self._make_inputs()
        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "false"}):
            reconciled = reconcile_if_enabled(regex_struct, llm_dict)

        # Simulate downstream consumer logic
        if reconciled is not None:
            site = reconciled.cancer_site
        else:
            site = regex_struct.cancer.site

        assert site == "lung"
        assert reconciled is None

    def test_downstream_uses_reconciled_when_flag_true(self):
        """When flag is true, downstream consumers use the ReconciledStructure."""
        regex_struct = _build_query_structure(site="lung", histology="adenocarcinoma")
        llm_dict = _build_llm_dict(cancer_type="lung", histology="squamous_cell_carcinoma")

        with patch.dict(os.environ, {"USE_RECONCILED_STRUCTURE": "true"}):
            reconciled = reconcile_if_enabled(regex_struct, llm_dict)

        assert reconciled is not None
        # LLM wins on histology per priority rules
        assert reconciled.histology == "squamous_cell_carcinoma"
        # Regex wins on site
        assert reconciled.cancer_site == "lung"
