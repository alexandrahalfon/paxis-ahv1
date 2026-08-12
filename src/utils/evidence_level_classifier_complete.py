"""
Evidence Level Classifier for Oncology Literature - COMPLETE VERSION
=====================================================================

Comprehensive classification based on ALL terms from authoritative sources:
1. PubMed MeSH Publication Types (https://www.nlm.nih.gov/mesh/pubtypes.html)
2. NCI PDQ Levels of Evidence (https://www.cancer.gov/publications/pdq/levels-evidence/treatment)
3. Oxford CEBM Levels of Evidence (https://www.cebm.ox.ac.uk/resources/levels-of-evidence)
"""

import re
import json
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum


class EvidenceLevel(Enum):
    """Evidence levels following NCI PDQ hierarchy."""
    LEVEL_1 = 1  # Systematic Review/Meta-analysis/Guideline
    LEVEL_2 = 2  # Randomized Controlled Trial
    LEVEL_3 = 3  # Prospective Non-randomized Study
    LEVEL_4 = 4  # Retrospective/Observational Study
    LEVEL_5 = 5  # Case Report/Case Series
    LEVEL_6 = 6  # Expert Opinion/Editorial
    LEVEL_7 = 7  # Unclassified


EVIDENCE_LEVEL_NAMES = {
    1: "Level I - Systematic Review/Meta-analysis/Guideline",
    2: "Level II - Randomized Controlled Trial",
    3: "Level III - Prospective Non-randomized Study",
    4: "Level IV - Retrospective/Observational Study",
    5: "Level V - Case Report/Case Series",
    6: "Level VI - Expert Opinion/Editorial",
    7: "Level VII - Unclassified",
}

NCI_PDQ_EVIDENCE_CODES = {
    "A1": "RCT with overall survival/total mortality endpoint",
    "A2": "Meta-analysis of RCTs with OS/mortality endpoint",
    "A3": "RCT with well-assessed quality of life endpoint",
    "B1": "RCT with EFS/RFS/DFS/PFS endpoint",
    "B2": "Meta-analysis with EFS/DFS/PFS/QoL endpoint",
    "B3": "RCT with tumor response rate endpoint",
    "B4": "Non-randomized multicenter prospective controlled trial",
    "C1": "Case series/observational with OS/mortality/QoL endpoint",
    "C2": "Case series/observational with EFS/RFS/DFS/PFS endpoint",
    "C3": "Case series/observational with tumor response endpoint",
    "D": "Expert opinion/anecdotal evidence",
}


@dataclass
class ClassificationResult:
    """Result of evidence level classification."""
    level: int
    level_name: str
    evidence_type: str
    confidence: float
    method: str
    nci_pdq_code: Optional[str] = None
    mesh_publication_type: Optional[str] = None
    matched_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class ComprehensiveEvidenceKeywords:
    """Complete keyword definitions from all authoritative sources."""

    # LEVEL 1: Systematic Reviews, Meta-analyses, Guidelines
    LEVEL_1_RULES = [
        {
            "patterns": [
                r"\bnccn\b",
                r"nccn\s+(?:clinical\s+practice\s+)?guideline",
                r"nccn\s+(?:categories?|recommendation)",
                r"\besmo\s+(?:clinical\s+practice\s+)?guideline",
                r"\basco\s+(?:clinical\s+practice\s+)?guideline",
                r"\bastro\s+(?:clinical\s+practice\s+)?guideline",
                r"\bestro\s+guideline",
                r"\babs\s+(?:consensus|guideline)",
                r"\bgec[- ]?estro\b",
                r"clinical\s+practice\s+guideline",
                r"practice\s+guideline",
                r"treatment\s+guideline",
                r"management\s+guideline",
                r"evidence[- ]based\s+guideline",
            ],
            "evidence_type": "Practice Guideline",
            "mesh_type": "Practice Guideline",
            "confidence": 0.95,
            "nci_code": "A2",
        },
        {
            "patterns": [
                r"consensus\s+guideline",
                r"expert\s+guideline",
                r"international\s+guideline",
                r"national\s+guideline",
            ],
            "evidence_type": "Guideline",
            "mesh_type": "Guideline",
            "confidence": 0.88,
            "nci_code": "A2",
        },
        {
            "patterns": [
                r"consensus\s+development\s+conference",
                r"consensus\s+conference",
                r"consensus\s+statement(?!\s+alone)",
                r"nih\s+consensus",
                r"consensus\s+panel",
            ],
            "evidence_type": "Consensus Development Conference",
            "mesh_type": "Consensus Development Conference",
            "confidence": 0.85,
            "nci_code": "A2",
        },
        {
            "patterns": [
                r"meta[- ]?analysis",
                r"meta[- ]?analyses",
                r"pooled\s+analysis",
                r"pooled\s+data\s+(?:analysis|from)",
                r"network\s+meta[- ]?analysis",
                r"individual\s+patient\s+data\s+meta",
                r"ipd\s+meta[- ]?analysis",
                r"bayesian\s+meta[- ]?analysis",
                r"quantitative\s+synthesis",
                r"combined\s+analysis\s+of\s+(?:randomized|rct)",
            ],
            "evidence_type": "Meta-Analysis",
            "mesh_type": "Meta-Analysis",
            "confidence": 0.92,
            "nci_code": "A2",
        },
        {
            "patterns": [
                r"systematic\s+review",
                r"systematic\s+literature\s+review",
                r"cochrane\s+(?:review|database|collaboration)",
                r"umbrella\s+review",
                r"overview\s+of\s+(?:systematic\s+)?reviews",
                r"prisma(?:[- ]compliant)?",
                r"preferred\s+reporting\s+items\s+for\s+systematic",
            ],
            "evidence_type": "Systematic Review",
            "mesh_type": "Systematic Review",
            "confidence": 0.92,
            "nci_code": "A2",
        },
    ]


    # LEVEL 2: Randomized Controlled Trials
    LEVEL_2_RULES = [
        {
            "patterns": [
                r"randomized\s+controlled\s+trial",
                r"randomised\s+controlled\s+trial",
                r"randomized\s+controlled\s+study",
                r"randomised\s+controlled\s+study",
                r"randomized\s+clinical\s+trial",
                r"randomised\s+clinical\s+trial",
                r"\brct\b",
                r"\brcts\b",
            ],
            "evidence_type": "Randomized Controlled Trial",
            "mesh_type": "Randomized Controlled Trial",
            "confidence": 0.95,
            "nci_code": "A1",
        },
        {
            "patterns": [
                r"phase\s*(?:3|iii)\s*(?:trial|study)?",
                r"phase\s*iii/iv",
                r"phase\s*3/4",
            ],
            "evidence_type": "Phase III Clinical Trial",
            "mesh_type": "Clinical Trial, Phase III",
            "confidence": 0.90,
            "nci_code": "A1",
        },
        {
            "patterns": [
                r"phase\s*(?:4|iv)\s*(?:trial|study)?",
                r"post[- ]?marketing\s+(?:study|trial|surveillance)",
            ],
            "evidence_type": "Phase IV Clinical Trial",
            "mesh_type": "Clinical Trial, Phase IV",
            "confidence": 0.88,
            "nci_code": "A1",
        },
        {
            "patterns": [
                r"pragmatic\s+(?:clinical\s+)?trial",
                r"pragmatic\s+(?:clinical\s+)?study",
                r"practical\s+clinical\s+trial",
                r"effectiveness\s+trial",
            ],
            "evidence_type": "Pragmatic Clinical Trial",
            "mesh_type": "Pragmatic Clinical Trial",
            "confidence": 0.88,
            "nci_code": "B1",
        },
        {
            "patterns": [
                r"equivalence\s+trial",
                r"equivalence\s+study",
                r"non[- ]?inferiority\s+(?:trial|study)",
                r"non[- ]?inferior(?:ity)?\s+design",
                r"superiority\s+(?:trial|study)",
            ],
            "evidence_type": "Equivalence/Non-inferiority Trial",
            "mesh_type": "Equivalence Trial",
            "confidence": 0.88,
            "nci_code": "B1",
        },
        {
            "patterns": [
                r"adaptive\s+(?:clinical\s+)?trial",
                r"adaptive\s+(?:clinical\s+)?design",
                r"adaptive\s+randomization",
                r"response[- ]?adaptive",
                r"bayesian\s+adaptive",
                r"platform\s+trial",
                r"basket\s+trial",
                r"umbrella\s+trial",
            ],
            "evidence_type": "Adaptive Clinical Trial",
            "mesh_type": "Adaptive Clinical Trial",
            "confidence": 0.85,
            "nci_code": "B1",
        },
        {
            "patterns": [
                r"\brandomized\b(?!.*review)(?!.*meta)",
                r"\brandomised\b(?!.*review)(?!.*meta)",
                r"randomly\s+(?:assigned|allocated|divided)",
                r"random\s+(?:assignment|allocation)",
                r"stratified\s+randomization",
                r"block\s+randomization",
            ],
            "evidence_type": "Randomized Trial",
            "mesh_type": "Randomized Controlled Trial",
            "confidence": 0.85,
            "nci_code": "B1",
        },
        {
            "patterns": [
                r"double[- ]?blind(?:ed)?",
                r"triple[- ]?blind(?:ed)?",
                r"single[- ]?blind(?:ed)?",
                r"placebo[- ]?controlled",
                r"sham[- ]?controlled",
                r"active[- ]?controlled",
            ],
            "evidence_type": "Blinded Controlled Trial",
            "mesh_type": "Randomized Controlled Trial",
            "confidence": 0.88,
            "nci_code": "A1",
        },
        {
            "patterns": [
                r"\brtog\s*\d+",
                r"\bnrg[- ]?(?:oncology\s+)?\w*\d+",
                r"\bartc\b",
                r"\bnsabp\s*[a-z]?[- ]?\d+",
                r"\bacosog\s*z?\d+",
                r"\balliance\s+[a-z]*\d+",
                r"\beortc\s*\d+",
                r"\bestro\s*\d+",
                r"\bgerman\s+breast\s+group",
                r"\bswog\s*\d+",
                r"\becog[- ]?\d+",
                r"\becog[- ]?acrin",
                r"\bcalgb\s*\d+",
                r"\bncctg\s*\d+",
                r"\bgog\s*\d+",
                r"\bcog\s+[a-z]*\d+",
                r"\brtog/ncctg\b",
                r"\btrog\s*\d+",
                r"\bportec[- ]?\d*\b",
                r"\bfast[- ]?forward\b",
                r"\bprime\s*(?:ii|2)\b",
                r"\bscorrad\b",
                r"\brapid\b.*trial",
                r"\bimport\s+(?:low|high)\b",
                r"\bconvert\b.*trial",
                r"\bpacific\b.*trial",
                r"\bkeynote[- ]?\d+",
                r"\bcheckmate[- ]?\d+",
                r"\bimpower\d+",
                r"\bcaspian\b",
                r"\bchariot\b",
                r"\btapur\b",
                r"\bnci[- ]?match\b",
                r"\blunar\b",
                r"\badaura\b",
                r"\blaura\b",
            ],
            "evidence_type": "Named Clinical Trial",
            "mesh_type": "Randomized Controlled Trial",
            "confidence": 0.92,
            "nci_code": "A1",
        },
        {
            "patterns": [
                r"multicenter\s+randomized",
                r"multicentre\s+randomised",
                r"multi[- ]?center\s+randomized",
                r"multi[- ]?centre\s+randomised",
                r"international\s+randomized",
            ],
            "evidence_type": "Multicenter Randomized Trial",
            "mesh_type": "Multicenter Study",
            "confidence": 0.90,
            "nci_code": "A1",
        },
    ]


    # LEVEL 3: Prospective Non-randomized Studies
    LEVEL_3_RULES = [
        {
            "patterns": [
                r"phase\s*(?:1|i)\s*(?:trial|study)(?!\s*[-/]\s*(?:2|ii|3|iii))",
                r"phase\s*1a\b",
                r"phase\s*1b\b",
                r"first[- ]?in[- ]?human",
                r"dose[- ]?escalation\s+(?:trial|study)",
                r"dose[- ]?finding\s+(?:trial|study)",
            ],
            "evidence_type": "Phase I Clinical Trial",
            "mesh_type": "Clinical Trial, Phase I",
            "confidence": 0.85,
            "nci_code": "B4",
        },
        {
            "patterns": [
                r"phase\s*(?:2|ii)\s*(?:trial|study)?",
                r"phase\s*2a\b",
                r"phase\s*2b\b",
            ],
            "evidence_type": "Phase II Clinical Trial",
            "mesh_type": "Clinical Trial, Phase II",
            "confidence": 0.85,
            "nci_code": "B4",
        },
        {
            "patterns": [
                r"phase\s*(?:1|i)\s*[-/]\s*(?:2|ii)",
                r"phase\s*(?:2|ii)\s*[-/]\s*(?:3|iii)",
            ],
            "evidence_type": "Phase I/II Clinical Trial",
            "mesh_type": "Clinical Trial, Phase II",
            "confidence": 0.83,
            "nci_code": "B4",
        },
        {
            "patterns": [
                r"controlled\s+clinical\s+trial(?!.*random)",
                r"controlled\s+study(?!.*random)",
                r"non[- ]?randomized\s+controlled",
                r"non[- ]?randomised\s+controlled",
            ],
            "evidence_type": "Controlled Clinical Trial (Non-randomized)",
            "mesh_type": "Controlled Clinical Trial",
            "confidence": 0.80,
            "nci_code": "B4",
        },
        {
            "patterns": [
                r"clinical\s+trial(?!.*phase)(?!.*random)",
                r"interventional\s+(?:trial|study)",
            ],
            "evidence_type": "Clinical Trial",
            "mesh_type": "Clinical Trial",
            "confidence": 0.75,
            "nci_code": "B4",
        },
        {
            "patterns": [
                r"clinical\s+study(?!.*random)(?!.*retrospect)",
                r"clinical\s+investigation",
            ],
            "evidence_type": "Clinical Study",
            "mesh_type": "Clinical Study",
            "confidence": 0.70,
            "nci_code": "B4",
        },
        {
            "patterns": [
                r"prospective\s+(?:study|trial|cohort|analysis|evaluation)",
                r"prospectively\s+(?:enrolled|collected|evaluated|followed)",
                r"single[- ]?arm\s+(?:study|trial)",
                r"non[- ]?randomized\s+(?:study|trial)",
                r"pilot\s+(?:study|trial)",
                r"feasibility\s+(?:study|trial)",
                r"proof[- ]?of[- ]?concept",
            ],
            "evidence_type": "Prospective Study",
            "mesh_type": "Clinical Study",
            "confidence": 0.80,
            "nci_code": "B4",
        },
    ]


    # LEVEL 4: Retrospective/Observational Studies
    LEVEL_4_RULES = [
        {
            "patterns": [
                r"observational\s+(?:study|analysis|cohort)",
                r"non[- ]?interventional\s+(?:study|analysis)",
            ],
            "evidence_type": "Observational Study",
            "mesh_type": "Observational Study",
            "confidence": 0.85,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"\bretrospective\b",
                r"retrospectively\s+(?:reviewed|analyzed|identified|collected|evaluated)",
                r"medical\s+records?\s+(?:review|analysis)",
                r"chart\s+review",
                r"claims\s+(?:data|database)\s+analysis",
            ],
            "evidence_type": "Retrospective Study",
            "mesh_type": "Observational Study",
            "confidence": 0.88,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"cohort\s+(?:study|analysis)",
                r"longitudinal\s+(?:study|analysis|cohort)",
                r"follow[- ]?up\s+study",
                r"inception\s+cohort",
            ],
            "evidence_type": "Cohort Study",
            "mesh_type": "Observational Study",
            "confidence": 0.85,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"case[- ]?control\s+(?:study|analysis)",
                r"matched[- ]?control",
                r"nested\s+case[- ]?control",
            ],
            "evidence_type": "Case-Control Study",
            "mesh_type": "Observational Study",
            "confidence": 0.85,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"comparative\s+(?:study|analysis|effectiveness)",
                r"comparison\s+of\s+(?:outcomes|treatments|techniques)",
                r"compared\s+(?:outcomes|survival|results)",
            ],
            "evidence_type": "Comparative Study",
            "mesh_type": "Comparative Study",
            "confidence": 0.80,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"evaluation\s+(?:study|of)",
                r"assessment\s+(?:study|of)",
                r"evaluating\s+(?:the\s+)?(?:efficacy|effectiveness|safety)",
            ],
            "evidence_type": "Evaluation Study",
            "mesh_type": "Evaluation Study",
            "confidence": 0.75,
            "nci_code": "C2",
        },
        {
            "patterns": [
                r"validation\s+(?:study|of)",
                r"validating\s+(?:a\s+)?(?:model|score|nomogram)",
                r"external\s+validation",
                r"internal\s+validation",
            ],
            "evidence_type": "Validation Study",
            "mesh_type": "Validation Study",
            "confidence": 0.78,
            "nci_code": "C2",
        },
        {
            "patterns": [
                r"twin\s+study",
                r"twin\s+registry",
                r"monozygotic.*dizygotic",
            ],
            "evidence_type": "Twin Study",
            "mesh_type": "Twin Study",
            "confidence": 0.82,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"population[- ]?based\s+(?:study|analysis|cohort)",
                r"cross[- ]?sectional\s+(?:study|analysis)",
                r"ecological\s+study",
                r"registry[- ]?based\s+(?:study|analysis)",
            ],
            "evidence_type": "Population-Based Study",
            "mesh_type": "Observational Study",
            "confidence": 0.82,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"\bseer\s+(?:analysis|database|data|study|medicare)",
                r"\bseer[- ]?medicare\b",
                r"\bncdb\b",
                r"national\s+cancer\s+database",
                r"surveillance.*epidemiology.*end\s+results",
                r"database\s+(?:analysis|study|review)",
            ],
            "evidence_type": "Database Analysis",
            "mesh_type": "Observational Study",
            "confidence": 0.82,
            "nci_code": "C1",
        },
        {
            "patterns": [
                r"institutional\s+(?:experience|review|series|analysis)",
                r"single[- ]?institution(?:al)?\s+(?:study|experience|review|analysis|series)",
                r"single[- ]?center\s+(?:study|experience|review|analysis)",
                r"single[- ]?centre\s+(?:study|experience|review|analysis)",
                r"multi[- ]?institution(?:al)?\s+(?:review|analysis|experience)(?!.*random)",
                r"multi[- ]?center\s+(?:review|analysis)(?!.*random)",
                r"multi[- ]?centre\s+(?:review|analysis)(?!.*random)",
            ],
            "evidence_type": "Institutional Experience",
            "mesh_type": "Observational Study",
            "confidence": 0.78,
            "nci_code": "C2",
        },
        {
            "patterns": [
                r"technical\s+report",
                r"scientific\s+report",
            ],
            "evidence_type": "Technical Report",
            "mesh_type": "Technical Report",
            "confidence": 0.70,
            "nci_code": "C2",
        },
    ]


    # LEVEL 5: Case Reports/Case Series
    LEVEL_5_RULES = [
        {
            "patterns": [
                r"\bcase\s+report\b",
                r"\bcase\s+reports\b",
                r"report\s+of\s+a\s+case",
                r"report\s+of\s+(?:two|three|four|five|six|\d+)\s+cases",
                r"we\s+(?:report|present)\s+(?:a\s+)?(?:case|patient)",
                r"here\s+(?:we\s+)?report",
                r"a\s+(?:rare|unusual|unique)\s+case",
                r"an?\s+unusual\s+(?:case|presentation)",
                r"first\s+(?:reported\s+)?case",
            ],
            "evidence_type": "Case Report",
            "mesh_type": "Case Reports",
            "confidence": 0.92,
            "nci_code": "C3",
        },
        {
            "patterns": [
                r"\bcase\s+series\b",
                r"series\s+of\s+(?:\d+\s+)?(?:cases|patients)",
                r"consecutive\s+(?:cases|patients)",
                r"\b\d+\s+cases\s+of\b",
                r"small\s+series",
            ],
            "evidence_type": "Case Series",
            "mesh_type": "Case Reports",
            "confidence": 0.90,
            "nci_code": "C2",
        },
        {
            "patterns": [
                r"\d{2}[- ]?year[- ]?old\s+(?:man|woman|male|female|patient|boy|girl)",
                r"chief\s+complaint",
                r"(?:was\s+)?presented\s+(?:to|with)\s+(?:our|the)",
                r"was\s+admitted\s+(?:to|with)",
                r"clinical\s+presentation",
            ],
            "evidence_type": "Case Report",
            "mesh_type": "Case Reports",
            "confidence": 0.75,
            "nci_code": "C3",
        },
    ]

    # LEVEL 6: Expert Opinion/Editorial
    LEVEL_6_RULES = [
        {
            "patterns": [
                r"\beditorial\b",
                r"editor[']?s?\s+(?:note|comment|perspective)",
                r"invited\s+editorial",
            ],
            "evidence_type": "Editorial",
            "mesh_type": "Editorial",
            "confidence": 0.92,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"\bcommentary\b",
                r"\bcomment\s+on\b",
                r"editorial\s+comment",
                r"invited\s+commentary",
            ],
            "evidence_type": "Commentary",
            "mesh_type": "Comment",
            "confidence": 0.90,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"letter\s+to\s+(?:the\s+)?editor",
                r"correspondence",
                r"reply\s+to",
                r"response\s+to\s+(?:letter|comment)",
            ],
            "evidence_type": "Letter",
            "mesh_type": "Letter",
            "confidence": 0.90,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"narrative\s+review",
                r"literature\s+review(?!.*systematic)",
                r"review\s+article(?!.*systematic)",
                r"review\s+of\s+(?:the\s+)?literature(?!.*systematic)",
                r"overview\s+of",
                r"current\s+concepts",
                r"state[- ]of[- ]the[- ]art\s+review",
                r"comprehensive\s+review(?!.*systematic)",
            ],
            "evidence_type": "Narrative Review",
            "mesh_type": "Review",
            "confidence": 0.82,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"expert\s+opinion",
                r"expert\s+consensus(?!\s+guideline)",
                r"clinical\s+experience",
                r"in\s+(?:our|the\s+author[']?s?)\s+(?:experience|opinion|view)",
                r"from\s+our\s+experience",
            ],
            "evidence_type": "Expert Opinion",
            "mesh_type": "Comment",
            "confidence": 0.88,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"personal\s+narrative",
                r"personal\s+experience",
                r"personal\s+account",
                r"reflections\s+on",
            ],
            "evidence_type": "Personal Narrative",
            "mesh_type": "Personal Narrative",
            "confidence": 0.85,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"\bperspective\b",
                r"\bviewpoint\b",
                r"personal\s+view",
                r"point\s+of\s+view",
                r"opinion\s+piece",
            ],
            "evidence_type": "Perspective",
            "mesh_type": "Comment",
            "confidence": 0.75,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"\bnews\s+(?:article|report|item)\b",
                r"press\s+release",
            ],
            "evidence_type": "News",
            "mesh_type": "News",
            "confidence": 0.80,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"\binterview\s+with\b",
                r"interview\s+of\b",
            ],
            "evidence_type": "Interview",
            "mesh_type": "Interview",
            "confidence": 0.80,
            "nci_code": "D",
        },
        {
            "patterns": [
                r"\bpreprint\b",
                r"not\s+peer[- ]?reviewed",
                r"medrxiv",
                r"biorxiv",
                r"arxiv",
            ],
            "evidence_type": "Preprint",
            "mesh_type": "Preprint",
            "confidence": 0.70,
            "nci_code": "D",
        },
    ]

    @classmethod
    def get_all_rules(cls) -> Dict[int, List[Dict]]:
        """Return all rules organized by evidence level."""
        return {
            1: cls.LEVEL_1_RULES,
            2: cls.LEVEL_2_RULES,
            3: cls.LEVEL_3_RULES,
            4: cls.LEVEL_4_RULES,
            5: cls.LEVEL_5_RULES,
            6: cls.LEVEL_6_RULES,
        }


class EvidenceLevelClassifier:
    """Comprehensive evidence level classifier using all authoritative medical sources."""

    def __init__(self):
        self.compiled_rules = self._compile_rules()

    def _compile_rules(self) -> Dict[int, List[Dict]]:
        """Compile all regex patterns."""
        compiled = {}
        all_rules = ComprehensiveEvidenceKeywords.get_all_rules()

        for level, rules in all_rules.items():
            compiled[level] = []
            for rule in rules:
                compiled_patterns = [re.compile(p, re.IGNORECASE) for p in rule["patterns"]]
                compiled[level].append({
                    "patterns": compiled_patterns,
                    "evidence_type": rule["evidence_type"],
                    "mesh_type": rule.get("mesh_type"),
                    "confidence": rule["confidence"],
                    "nci_code": rule.get("nci_code"),
                })

        return compiled

    def classify(self, payload: Dict) -> ClassificationResult:
        """Classify a document/chunk based on its payload."""
        doc_meta = payload.get("doc_meta", {})
        title = (doc_meta.get("title") or "").strip()
        citation = (doc_meta.get("citation") or "").strip()
        category = (payload.get("category") or "").strip()
        abstract = (doc_meta.get("abstract") or "").strip()

        # Primary: title + citation (most reliable)
        primary_text = f"{title} {title} {citation}"
        result = self._check_text(primary_text, confidence_modifier=1.0)
        if result.level < 7:
            return result

        # Secondary: category + abstract
        secondary_text = f"{category} {abstract}"
        result = self._check_text(secondary_text, confidence_modifier=0.9)
        if result.level < 7:
            return result

        # Tertiary: chunk text (lowest confidence)
        text_sample = (payload.get("text") or "")[:1000].strip()
        result = self._check_text(text_sample, confidence_modifier=0.7)
        if result.level < 7:
            return result

        return ClassificationResult(
            level=7,
            level_name=EVIDENCE_LEVEL_NAMES[7],
            evidence_type="Unclassified",
            confidence=0.0,
            method="default",
            matched_patterns=[],
        )

    def _check_text(self, text: str, confidence_modifier: float = 1.0) -> ClassificationResult:
        """Check text against all rules."""
        if not text or len(text) < 10:
            return ClassificationResult(
                level=7,
                level_name=EVIDENCE_LEVEL_NAMES[7],
                evidence_type="Unclassified",
                confidence=0.0,
                method="no_text",
                matched_patterns=[],
            )

        matches = []
        for level in [1, 2, 3, 4, 5, 6]:
            for rule in self.compiled_rules[level]:
                for pattern in rule["patterns"]:
                    match = pattern.search(text)
                    if match:
                        matches.append({
                            "level": level,
                            "evidence_type": rule["evidence_type"],
                            "mesh_type": rule.get("mesh_type"),
                            "confidence": rule["confidence"] * confidence_modifier,
                            "nci_code": rule.get("nci_code"),
                            "matched": match.group(0),
                        })

        if not matches:
            return ClassificationResult(
                level=7,
                level_name=EVIDENCE_LEVEL_NAMES[7],
                evidence_type="Unclassified",
                confidence=0.0,
                method="no_match",
                matched_patterns=[],
            )

        # Best match: prefer higher evidence, then higher confidence
        best = min(matches, key=lambda m: (m["level"], -m["confidence"]))

        return ClassificationResult(
            level=best["level"],
            level_name=EVIDENCE_LEVEL_NAMES[best["level"]],
            evidence_type=best["evidence_type"],
            confidence=round(best["confidence"], 3),
            method="rule_based",
            nci_pdq_code=best.get("nci_code"),
            mesh_publication_type=best.get("mesh_type"),
            matched_patterns=[m["matched"] for m in matches if m["level"] == best["level"]][:5],
        )

    def get_statistics(self) -> Dict:
        """Return statistics about the classifier rules."""
        all_rules = ComprehensiveEvidenceKeywords.get_all_rules()
        stats = {
            "total_rules": 0,
            "total_patterns": 0,
            "rules_by_level": {},
            "patterns_by_level": {},
        }

        for level, rules in all_rules.items():
            pattern_count = sum(len(r["patterns"]) for r in rules)
            stats["rules_by_level"][level] = len(rules)
            stats["patterns_by_level"][level] = pattern_count
            stats["total_rules"] += len(rules)
            stats["total_patterns"] += pattern_count

        return stats


class EvidenceLevelBatchUpdater:
    """Batch update Qdrant collection with evidence level classifications."""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        from qdrant_client import QdrantClient
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)
        self.collection = collection_name
        self.classifier = EvidenceLevelClassifier()
        self._doc_cache: Dict[str, ClassificationResult] = {}

    def run(
        self,
        batch_size: int = 100,
        dry_run: bool = True,
        limit: Optional[int] = None,
        save_report: bool = True,
        report_path: str = "evidence_classification_report.json"
    ) -> Dict:
        """Run batch classification."""
        print(f"\n{'='*60}")
        print(f"Evidence Level Classification - COMPLETE VERSION")
        print(f"Collection: {self.collection}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
        print(f"{'='*60}")

        stats_info = self.classifier.get_statistics()
        print(f"\nClassifier: {stats_info['total_rules']} rules, {stats_info['total_patterns']} patterns")

        stats = {
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "classifier_stats": stats_info,
            "total_points": 0,
            "already_classified": 0,
            "newly_classified": 0,
            "unique_documents": 0,
            "by_level": defaultdict(int),
            "by_type": defaultdict(int),
            "by_mesh_type": defaultdict(int),
            "by_nci_code": defaultdict(int),
            "low_confidence": [],
            "sample_classifications": [],
        }

        offset = None
        points_processed = 0
        update_batch = []

        while True:
            points, next_offset = self.qdrant.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id", str(point.id))
                existing_level = payload.get("doc_meta", {}).get("evidence_level")

                if existing_level is not None:
                    stats["already_classified"] += 1
                    stats["by_level"][existing_level] += 1
                    continue

                if doc_id in self._doc_cache:
                    result = self._doc_cache[doc_id]
                else:
                    result = self.classifier.classify(payload)
                    self._doc_cache[doc_id] = result
                    stats["unique_documents"] += 1

                stats["newly_classified"] += 1
                stats["by_level"][result.level] += 1
                stats["by_type"][result.evidence_type] += 1

                if result.mesh_publication_type:
                    stats["by_mesh_type"][result.mesh_publication_type] += 1
                if result.nci_pdq_code:
                    stats["by_nci_code"][result.nci_pdq_code] += 1

                if result.confidence < 0.7 and result.level < 7:
                    stats["low_confidence"].append({
                        "point_id": str(point.id),
                        "doc_id": doc_id,
                        "title": payload.get("doc_meta", {}).get("title", "")[:100],
                        "classification": result.to_dict(),
                    })

                if len(stats["sample_classifications"]) < 100:
                    stats["sample_classifications"].append({
                        "point_id": str(point.id),
                        "doc_id": doc_id,
                        "title": payload.get("doc_meta", {}).get("title", "")[:100],
                        "classification": result.to_dict(),
                    })

                update_batch.append({"point_id": point.id, "result": result})

                if len(update_batch) >= 50 and not dry_run:
                    self._apply_updates(update_batch)
                    update_batch = []

            stats["total_points"] += len(points)
            points_processed += len(points)

            if points_processed % 500 == 0:
                print(f"  Processed {points_processed} points...")

            if limit and points_processed >= limit:
                break

            offset = next_offset
            if offset is None:
                break

        if update_batch and not dry_run:
            self._apply_updates(update_batch)

        stats["finished_at"] = datetime.now().isoformat()
        stats["by_level"] = dict(stats["by_level"])
        stats["by_type"] = dict(stats["by_type"])
        stats["by_mesh_type"] = dict(stats["by_mesh_type"])
        stats["by_nci_code"] = dict(stats["by_nci_code"])

        self._print_summary(stats)

        if save_report:
            with open(report_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nReport saved to: {report_path}")

        return stats

    def _apply_updates(self, updates: List[Dict]):
        """Apply updates to Qdrant."""
        for update in updates:
            point_id = update["point_id"]
            result = update["result"]

            try:
                self.qdrant.set_payload(
                    collection_name=self.collection,
                    payload={
                        "doc_meta.evidence_level": result.level,
                        "doc_meta.evidence_level_name": result.level_name,
                        "doc_meta.evidence_type": result.evidence_type,
                        "doc_meta.evidence_confidence": result.confidence,
                        "doc_meta.evidence_nci_code": result.nci_pdq_code,
                        "doc_meta.evidence_mesh_type": result.mesh_publication_type,
                        "doc_meta.evidence_method": result.method,
                    },
                    points=[point_id],
                    wait=False,
                )
            except Exception as e:
                print(f"  Warning: Failed to update {point_id}: {e}")

    def _print_summary(self, stats: Dict):
        """Print summary."""
        print(f"\n{'='*60}")
        print("CLASSIFICATION SUMMARY")
        print(f"{'='*60}")

        total = stats["newly_classified"] + stats["already_classified"]
        print(f"\nTotal points: {stats['total_points']}")
        print(f"Already classified: {stats['already_classified']}")
        print(f"Newly classified: {stats['newly_classified']}")
        print(f"Unique documents: {stats['unique_documents']}")

        print(f"\n--- Distribution by Level ---")
        for level in sorted(stats["by_level"].keys()):
            count = stats["by_level"][level]
            pct = count / max(total, 1) * 100
            bar = "█" * int(pct / 2)
            print(f"  L{level}: {count:>6} ({pct:>5.1f}%) {bar}")

        print(f"\n--- Top Evidence Types ---")
        for etype, count in sorted(stats["by_type"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {etype}: {count}")

        if stats.get("by_nci_code"):
            print(f"\n--- NCI PDQ Codes ---")
            for code, count in sorted(stats["by_nci_code"].items()):
                print(f"  {code}: {count}")

        if stats["low_confidence"]:
            print(f"\n⚠️  Low confidence: {len(stats['low_confidence'])} items")


class ClassificationTester:
    """Test classifier against known examples."""

    def __init__(self):
        self.classifier = EvidenceLevelClassifier()

    def run_tests(self) -> Dict:
        """Run comprehensive tests covering all MeSH types."""
        test_cases = [
            ({"doc_meta": {"title": "NCCN Clinical Practice Guidelines in Oncology: Breast Cancer"}}, 1, "Practice Guideline"),
            ({"doc_meta": {"title": "Meta-analysis of adjuvant radiation therapy"}}, 1, "Meta-Analysis"),
            ({"doc_meta": {"title": "Systematic review of hypofractionated radiotherapy"}}, 1, "Systematic Review"),
            ({"doc_meta": {"title": "Randomized controlled trial of hypofractionation"}}, 2, "Randomized Controlled Trial"),
            ({"doc_meta": {"title": "Phase III trial comparing IMRT vs 3D-CRT"}}, 2, "Phase III Clinical Trial"),
            ({"doc_meta": {"title": "RTOG 0617: A randomized phase III comparison"}}, 2, "Randomized Controlled Trial"),
            ({"doc_meta": {"title": "Phase II study of concurrent chemoradiation"}}, 3, "Phase II Clinical Trial"),
            ({"doc_meta": {"title": "Prospective evaluation of toxicity outcomes"}}, 3, "Prospective Study"),
            ({"doc_meta": {"title": "Retrospective analysis of treatment outcomes"}}, 4, "Retrospective Study"),
            ({"doc_meta": {"title": "SEER database analysis of survival trends"}}, 4, "Database Analysis"),
            ({"doc_meta": {"title": "Case report: Radiation recall dermatitis"}}, 5, "Case Report"),
            ({"doc_meta": {"title": "Editorial: The future of precision radiation oncology"}}, 6, "Editorial"),
            ({"doc_meta": {"title": "Letter to the Editor: Response to recent findings"}}, 6, "Letter"),
            ({"doc_meta": {"title": "Treatment outcomes in breast cancer patients"}}, 7, "Should be unclassified"),
        ]

        results = {"total": len(test_cases), "passed": 0, "failed": 0, "failures": []}

        print("\n" + "="*70)
        print("RUNNING CLASSIFICATION TESTS")
        print("="*70 + "\n")

        for payload, expected, description in test_cases:
            result = self.classifier.classify(payload)
            title = payload.get("doc_meta", {}).get("title", "")[:55]

            if result.level == expected:
                print(f"  ✓ L{expected} | {title}...")
                results["passed"] += 1
            else:
                print(f"  ✗ Expected L{expected}, Got L{result.level} | {title}...")
                results["failed"] += 1
                results["failures"].append({
                    "title": title,
                    "expected": expected,
                    "got": result.level,
                    "description": description,
                })

        print(f"\n{'='*70}")
        print(f"RESULTS: {results['passed']}/{results['total']} passed ({results['passed']/results['total']*100:.1f}%)")
        print(f"{'='*70}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evidence Level Classifier for Oncology Literature")
    parser.add_argument("--test", action="store_true", help="Run tests on known examples")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without updating")
    parser.add_argument("--live", action="store_true", help="Update collection")
    parser.add_argument("--limit", type=int, help="Limit points to process")
    parser.add_argument("--qdrant-url", type=str, help="Qdrant URL")
    parser.add_argument("--qdrant-key", type=str, help="Qdrant API key")
    parser.add_argument("--collection", type=str, help="Collection name")

    args = parser.parse_args()

    if args.test:
        tester = ClassificationTester()
        tester.run_tests()
        return

    if args.dry_run or args.live:
        qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL")
        qdrant_key = args.qdrant_key or os.getenv("QDRANT_API_KEY")
        collection = args.collection or os.getenv("QDRANT_COLLECTION")

        if not all([qdrant_url, qdrant_key, collection]):
            print("Error: Missing Qdrant configuration.")
            print("Set via arguments or environment variables:")
            print("  QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION")
            return

        updater = EvidenceLevelBatchUpdater(qdrant_url, qdrant_key, collection)
        updater.run(dry_run=args.dry_run, limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
