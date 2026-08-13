from pipeline.patient_education.discovery import bucket_for, normalize_url, path_allowed
from pipeline.patient_education.runner import assert_repo_compatibility
from pipeline.patient_education.sources import SOURCES


def test_pipeline_source_registry_contract():
    assert_repo_compatibility()


def test_tracking_params_are_removed():
    assert normalize_url("https://www.cancer.gov/types/breast?utm_source=x&foo=1") == \
        "https://www.cancer.gov/types/breast?foo=1"


def test_nci_scope_accepts_patient_content_and_rejects_research():
    scope = SOURCES["nci"]
    assert path_allowed("https://www.cancer.gov/about-cancer/treatment/types/chemotherapy", scope)
    assert not path_allowed("https://www.cancer.gov/research/some-study", scope)


def test_bucket_examples():
    assert bucket_for("https://www.cancer.net/navigating-cancer-care/how-cancer-treated/immunotherapy") == "treatment"
    assert bucket_for("https://www.cancer.org/cancer/supportive-care/nutrition-activity-with-cancer.html") == "nutrition_lifestyle"
