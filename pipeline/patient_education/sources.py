from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SourceScope:
    """Discovery policy for one already-registered Paxis evidence source.

    source_key MUST match src.api.services.evidence.source_registry.DEFAULT_SOURCES.
    This module controls discovery only; authority, Qdrant collection routing,
    and document/version persistence remain owned by the Paxis source registry
    and EvidenceIngestionService.
    """

    source_key: str
    domain: str
    seed_urls: List[str]
    include_prefixes: List[str]
    exclude_prefixes: List[str] = field(default_factory=list)
    exclude_regexes: List[str] = field(default_factory=list)
    required_buckets: List[str] = field(default_factory=list)


COMMON_EXCLUDES = [
    r"/search(?:/|$)",
    r"/user(?:/|$)",
    r"/login(?:/|$)",
    r"/register(?:/|$)",
    r"/donat(?:e|ion)",
    r"/give(?:/|$)",
    r"/volunteer(?:/|$)",
    r"/fundrais",
    r"/event(?:s)?(?:/|$)",
    r"/news(?:/|$)",
    r"/press(?:/|$)",
    r"/media(?:/|$)",
    r"/careers?(?:/|$)",
    r"/jobs?(?:/|$)",
    r"/privacy",
    r"/terms",
    r"/contact-us",
    r"[?&](?:utm_|gclid|fbclid)",
]


SOURCES: Dict[str, SourceScope] = {
    "nci": SourceScope(
        source_key="nci",
        domain="cancer.gov",
        seed_urls=[
            "https://www.cancer.gov/about-cancer",
            "https://www.cancer.gov/types",
            "https://www.cancer.gov/about-cancer/treatment/types",
            "https://www.cancer.gov/about-cancer/treatment/side-effects",
            "https://www.cancer.gov/about-cancer/treatment/side-effects/nutrition",
        ],
        include_prefixes=["/about-cancer", "/types"],
        exclude_prefixes=["/news-events", "/research", "/grants-training"],
        exclude_regexes=COMMON_EXCLUDES,
        required_buckets=[
            "cancer_type",
            "diagnosis_testing",
            "treatment",
            "side_effects",
            "nutrition_lifestyle",
            "supportive_care",
        ],
    ),
    "cancer_net": SourceScope(
        source_key="cancer_net",
        domain="cancer.net",
        seed_urls=[
            "https://www.cancer.net/cancer-types",
            "https://www.cancer.net/navigating-cancer-care",
            "https://www.cancer.net/coping-with-cancer",
            "https://www.cancer.net/survivorship",
            "https://www.cancer.net/about-us/asco-answers-patient-education-materials",
        ],
        include_prefixes=[
            "/cancer-types",
            "/navigating-cancer-care",
            "/coping-with-cancer",
            "/survivorship",
            "/about-us/asco-answers-patient-education-materials",
        ],
        exclude_regexes=COMMON_EXCLUDES + [
            r"/blog(?:/|$)",
            r"/podcast",
        ],
        required_buckets=[
            "cancer_type",
            "diagnosis_testing",
            "treatment",
            "side_effects",
            "nutrition_lifestyle",
            "supportive_care",
            "financial_practical",
            "caregiver",
            "survivorship",
        ],
    ),
    "acs": SourceScope(
        source_key="acs",
        domain="cancer.org",
        seed_urls=[
            "https://www.cancer.org/cancer.html",
            "https://www.cancer.org/cancer/types.html",
            "https://www.cancer.org/cancer/treatment-types.html",
            "https://www.cancer.org/cancer/managing-cancer/side-effects.html",
            "https://www.cancer.org/cancer/supportive-care.html",
            "https://www.cancer.org/cancer/supportive-care/nutrition-activity-with-cancer.html",
            "https://www.cancer.org/cancer/survivorship.html",
            "https://www.cancer.org/cancer/caregivers.html",
        ],
        include_prefixes=["/cancer"],
        exclude_prefixes=["/cancer/research"],
        exclude_regexes=COMMON_EXCLUDES,
        required_buckets=[
            "cancer_type",
            "diagnosis_testing",
            "treatment",
            "side_effects",
            "nutrition_lifestyle",
            "supportive_care",
            "financial_practical",
            "caregiver",
            "survivorship",
        ],
    ),
}
