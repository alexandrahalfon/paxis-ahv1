"""
Clinical Inference Layer

Maps implicit clinical facts in patient narratives to explicit clinical labels
used in the literature. Called after LLM axis extraction and before embedding.

Example:
    "no longer a surgical candidate following locoregional progression"
    → adds: unresectable, inoperable, salvage not feasible, locoregional failure

    "progressing on pembrolizumab"
    → adds: ICI-refractory, anti-PD1 failure, checkpoint refractory, 2nd-line
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ── Inference map ──────────────────────────────────────────────────────────────
# Each entry: trigger pattern → list of terms to add to that axis
# Patterns are matched case-insensitively against the raw narrative text.

INFERENCE_MAP: Dict[str, List[str]] = {

    # ═══════════════════════════════════════════════════════════════════════
    # SURGICAL CANDIDACY
    # ═══════════════════════════════════════════════════════════════════════
    r"no longer (?:a )?surgical candidate": [
        "unresectable", "inoperable", "salvage surgery not feasible",
        "non-surgical management",
    ],
    r"not (?:a )?surgical candidate": [
        "unresectable", "inoperable", "non-surgical",
    ],
    r"declined surgery|refused surgery|patient declines? (?:surgical|resection|operation)": [
        "surgery declined", "non-surgical", "systemic therapy only",
    ],
    r"unresectable|non-?resectable": [
        "inoperable", "non-surgical candidate", "locoregional advanced",
    ],
    r"inoperable|medically inoperable": [
        "unresectable", "non-surgical candidate", "definitive non-surgical",
    ],
    r"borderline resectable": [
        "borderline resectable", "neoadjuvant consideration",
        "preoperative therapy", "marginal resectability",
    ],
    r"locally advanced.{0,30}(?:unresectable|inoperable)": [
        "locally advanced unresectable", "definitive chemoradiation",
        "non-surgical definitive treatment",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # ICI / CHECKPOINT INHIBITOR STATUS
    # ═══════════════════════════════════════════════════════════════════════
    r"progress(?:ing|ion) on (?:pembrolizumab|nivolumab|atezolizumab|durvalumab|avelumab|cemiplimab|ipilimumab|ICI|immunotherapy|checkpoint|anti.PD.?[1L])": [
        "ICI-refractory", "anti-PD1 failure", "anti-PD-L1 failure",
        "checkpoint inhibitor refractory", "post-immunotherapy",
        "second-line", "2nd-line systemic", "salvage",
    ],
    r"refractory to (?:pembrolizumab|nivolumab|atezolizumab|durvalumab|ICI|immunotherapy|checkpoint|anti.PD)": [
        "ICI-refractory", "checkpoint inhibitor refractory",
        "anti-PD1 failure", "post-ICI progression",
    ],
    r"ICI.{0,20}(?:progress|fail|refract|resist)": [
        "ICI-refractory", "checkpoint inhibitor failure",
        "2nd-line", "post-checkpoint progression",
    ],
    r"locoregional progression on (?:ICI|immunotherapy|checkpoint|pembrolizumab|nivolumab)": [
        "ICI-refractory", "anti-PD1 failure", "locoregional failure",
        "2nd-line systemic therapy",
    ],
    r"(?:primary|innate|de novo)\s+(?:resistance|refractory).{0,20}(?:ICI|immunotherapy|checkpoint)": [
        "primary ICI resistance", "innate resistance",
        "non-responder", "cold tumor",
    ],
    r"acquired resistance.{0,20}(?:ICI|immunotherapy|checkpoint)": [
        "acquired ICI resistance", "secondary resistance",
        "initial responder", "late progression",
    ],
    r"hyperprogression|hyper.?progression": [
        "hyperprogressive disease", "HPD", "rapid progression on ICI",
        "accelerated tumor growth",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # TARGETED THERAPY RESISTANCE
    # ═══════════════════════════════════════════════════════════════════════
    r"(?:progress|fail|refract|resist).{0,30}(?:erlotinib|gefitinib|afatinib|osimertinib|EGFR.?TKI)": [
        "EGFR TKI resistant", "EGFR TKI failure", "post-EGFR-TKI progression",
        "T790M testing", "osimertinib candidate", "third-generation TKI",
    ],
    r"(?:progress|fail|refract|resist).{0,30}(?:crizotinib|alectinib|ceritinib|brigatinib|lorlatinib|ALK.?(?:TKI|inhibitor))": [
        "ALK TKI resistant", "ALK inhibitor failure",
        "next-generation ALK inhibitor", "ALK resistance mutation",
    ],
    r"(?:progress|fail|refract|resist).{0,30}(?:imatinib|dasatinib|nilotinib|bosutinib|ponatinib|TKI)": [
        "TKI resistant", "tyrosine kinase inhibitor failure",
        "next-line TKI", "BCR-ABL resistance",
    ],
    r"(?:progress|fail|refract|resist).{0,30}(?:vemurafenib|dabrafenib|encorafenib|BRAF.?inhibitor)": [
        "BRAF inhibitor resistant", "BRAF/MEK failure",
        "post-targeted-therapy", "MAPK pathway resistance",
    ],
    r"(?:progress|fail|refract|resist).{0,30}(?:trastuzumab|pertuzumab|T-?DM1|T-?DXd|HER2.?(?:therapy|targeted|directed))": [
        "HER2-targeted therapy resistant", "anti-HER2 failure",
        "post-trastuzumab progression", "HER2 resistance",
    ],
    r"(?:progress|fail|refract|resist).{0,30}(?:olaparib|niraparib|rucaparib|talazoparib|PARP.?inhibitor)": [
        "PARP inhibitor resistant", "PARPi failure",
        "post-PARPi progression", "homologous recombination",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # HORMONE / ENDOCRINE THERAPY
    # ═══════════════════════════════════════════════════════════════════════
    r"(?:progress|fail|refract|resist).{0,30}(?:tamoxifen|letrozole|anastrozole|exemestane|aromatase inhibitor|AI|endocrine|hormonal)": [
        "endocrine resistant", "hormone-refractory",
        "AI-resistant", "endocrine therapy failure",
        "CDK4/6 inhibitor candidate", "fulvestrant candidate",
    ],
    r"(?:ER|estrogen receptor)\s*(?:\+|positive)": [
        "ER-positive disease", "endocrine therapy eligible",
        "hormone receptor positive", "hormone-sensitive",
    ],
    r"(?:PR|progesterone receptor)\s*(?:\+|positive)": [
        "PR-positive", "hormone receptor positive",
    ],
    r"(?:ER|estrogen receptor)\s*(?:-|negative)": [
        "ER-negative", "endocrine therapy not indicated",
    ],
    r"triple.negative|TNBC": [
        "triple-negative breast cancer", "TNBC",
        "ER-negative PR-negative HER2-negative",
        "chemotherapy primary", "immunotherapy candidate",
    ],
    r"hormone.?refractory|castration.?resist": [
        "castration-resistant", "CRPC", "hormone-refractory",
        "abiraterone candidate", "enzalutamide candidate",
        "next-line hormonal", "mCRPC",
    ],
    r"(?:progress|fail).{0,20}(?:ADT|androgen deprivation|LHRH|leuprolide|goserelin)": [
        "castration-resistant prostate cancer", "CRPC", "mCRPC",
        "post-ADT progression", "next-line systemic",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # RADIATION TREATMENT STATUS
    # ═══════════════════════════════════════════════════════════════════════
    r"(?:prior|previous|s/p|status\s*post|history of)\s+(?:radiation|radiotherapy|RT|XRT|EBRT|IMRT|chemoradiation|CRT)": [
        "prior radiation", "post-radiation", "previously irradiated",
        "re-irradiation candidate", "radiation recall risk",
    ],
    r"(?:progress|recur|fail).{0,30}(?:after|following|post).{0,15}(?:radiation|RT|CRT|chemoradiation)": [
        "post-radiation recurrence", "radiation failure",
        "salvage therapy", "re-irradiation consideration",
    ],
    r"re.?irradiat": [
        "re-irradiation", "cumulative dose concern",
        "prior radiation field overlap", "toxicity risk increased",
    ],
    r"radiation.?na[ïi]ve|no prior (?:radiation|RT)": [
        "radiation naive", "no prior RT", "definitive RT candidate",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # LINE OF THERAPY
    # ═══════════════════════════════════════════════════════════════════════
    r"started on (?:pembrolizumab|nivolumab|atezolizumab|durvalumab|ipilimumab)": [
        "first-line immunotherapy", "1L checkpoint inhibitor",
    ],
    r"first.line|1st.line|1L\b|frontline|initial (?:therapy|treatment)": [
        "first-line", "1L", "frontline", "treatment-naive",
    ],
    r"second.line|2nd.line|2L\b|salvage therapy": [
        "second-line systemic", "2L", "salvage", "post-progression treatment",
    ],
    r"third.line|3rd.line|3L\b|heavily pretreated": [
        "third-line", "3L", "heavily pretreated",
        "multiply refractory", "late-line",
    ],
    r"treatment.?na[ïi]ve|chemo.?na[ïi]ve|untreated": [
        "treatment-naive", "previously untreated",
        "first-line candidate", "de novo presentation",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # CPS / PD-L1 THRESHOLDS
    # ═══════════════════════════════════════════════════════════════════════
    r"CPS\s*(?:score\s*(?:of\s*)?)?(?:=\s*)?100": [
        "CPS ≥ 1", "CPS ≥ 20", "CPS ≥ 50", "CPS ≥ 80", "CPS 100",
        "PD-L1 high expression", "immunotherapy eligible",
        "pembrolizumab indicated",
    ],
    r"CPS\s*(?:score\s*(?:of\s*)?)?(?:=\s*)?\d{2,}": [
        "CPS ≥ 1", "CPS ≥ 20", "PD-L1 positive", "immunotherapy eligible",
    ],
    r"CPS\s*(?:score\s*(?:of\s*)?)?(?:=\s*)?\d+": [
        "CPS ≥ 1", "PD-L1 expression",
    ],
    r"TPS\s*(?:≥|>=?)\s*50|TPS\s*(?:score\s*)?(?:of\s*)?\d{2,}": [
        "TPS ≥ 50%", "PD-L1 high", "first-line pembrolizumab monotherapy",
    ],
    r"PD-?L1\s*(?:high|positive|\+|≥\s*\d)": [
        "PD-L1 positive", "immunotherapy eligible",
        "checkpoint inhibitor candidate",
    ],
    r"PD-?L1\s*(?:negative|low|-|<\s*1)": [
        "PD-L1 negative", "PD-L1 low expression",
        "combination therapy preferred",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # BIOMARKERS — GENERAL
    # ═══════════════════════════════════════════════════════════════════════
    r"MSI.?H|microsatellite instability.?high|dMMR|deficient mismatch repair": [
        "MSI-H", "dMMR", "mismatch repair deficient",
        "pembrolizumab eligible", "immunotherapy candidate",
    ],
    r"MSS|microsatellite stable|pMMR|proficient mismatch repair": [
        "MSS", "pMMR", "microsatellite stable",
        "immunotherapy less likely", "chemotherapy backbone",
    ],
    r"TMB.?(?:high|H|≥\s*10)": [
        "TMB-high", "high tumor mutational burden",
        "immunotherapy candidate", "pembrolizumab eligible",
    ],
    r"EGFR\s*(?:mut|positive|\+|exon\s*(?:19|21)|del19|L858R)": [
        "EGFR mutant", "EGFR-positive", "TKI eligible",
        "osimertinib candidate", "targeted therapy",
    ],
    r"ALK\s*(?:positive|\+|rearranged|fusion|translocation)": [
        "ALK-positive", "ALK fusion", "ALK rearrangement",
        "crizotinib candidate", "alectinib candidate",
    ],
    r"ROS1\s*(?:positive|\+|rearranged|fusion)": [
        "ROS1-positive", "ROS1 fusion",
        "crizotinib candidate", "entrectinib candidate",
    ],
    r"KRAS\s*(?:G12C|mut|positive|\+)": [
        "KRAS mutant", "KRAS G12C", "sotorasib candidate",
        "adagrasib candidate", "targeted therapy eligible",
    ],
    r"BRAF\s*(?:V600E?|mut|positive|\+)": [
        "BRAF mutant", "BRAF V600E", "vemurafenib candidate",
        "dabrafenib/trametinib candidate",
    ],
    r"BRCA\s*(?:1|2)?\s*(?:mut|positive|\+|pathogenic|germline)": [
        "BRCA mutant", "BRCA-positive", "PARP inhibitor eligible",
        "olaparib candidate", "HRD-positive",
    ],
    r"HER2\s*(?:positive|\+|amplified|overexpress|3\+|2\+/FISH\+)": [
        "HER2-positive", "HER2 amplified",
        "trastuzumab eligible", "HER2-directed therapy",
    ],
    r"HER2\s*(?:negative|-|low|0|1\+)": [
        "HER2-negative", "HER2-low",
        "T-DXd candidate" if "low" else "non-HER2-directed",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # HPV / p16
    # ═══════════════════════════════════════════════════════════════════════
    # NOTE: The negative pattern's bare `\bp16\s*-` alternative was matching
    # connector hyphens in "p16-positive" (same root-cause as the polarity
    # bugs fixed in commits 0d69985 and 863fe0a). The result: BOTH the
    # positive and negative patterns fired for a "p16-positive" query,
    # adding contradictory terms to the biomarker_profile axis. Doesn't
    # drive the eligibility verdict (a different extractor handles that),
    # but broadens retrieval expansion in both directions, hurting precision.
    # Fix: require the bare `-` polarity marker to be followed by whitespace,
    # end-of-string, or punctuation — so "p16-" tags negative but
    # "p16-positive" does not.
    r"\bp16\s*\+|HPV.?positive|HPV.?associated|p16.?positive": [
        "HPV-positive HNSCC", "p16-positive", "HPV-related oropharyngeal",
        "favorable biology", "de-escalation candidate",
    ],
    r"\bp16\s*-(?=\s|$|[,.;:])|HPV.?negative|HPV.?unrelated|p16.?negative": [
        "HPV-negative HNSCC", "p16-negative",
        "non-HPV-associated", "unfavorable biology",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # PERFORMANCE STATUS
    # ═══════════════════════════════════════════════════════════════════════
    r"ECOG\s*(?:PS\s*)?(?:=\s*)?0": [
        "ECOG 0", "fully active", "excellent performance status",
    ],
    r"ECOG\s*(?:PS\s*)?(?:=\s*)?1": [
        "ECOG 1", "ambulatory", "good performance status",
    ],
    r"ECOG\s*(?:PS\s*)?(?:=\s*)?2": [
        "ECOG 2", "ambulatory >50%", "moderate performance status",
        "treatment-eligible borderline",
    ],
    r"ECOG\s*(?:PS\s*)?(?:=\s*)?[34]": [
        "ECOG 3-4", "poor performance status", "limited activity",
        "best supportive care", "palliative focus",
    ],
    r"KPS\s*(?:=\s*)?\d{2}|Karnofsky\s*(?:=\s*)?\d{2}": [
        "Karnofsky performance status", "KPS documented",
    ],
    r"bedridden|bed.?bound|confined to bed|wheelchair.?bound": [
        "poor performance status", "ECOG 3-4", "KPS < 50",
        "palliative intent only", "best supportive care",
    ],
    r"frail|frailty|(?:elderly|geriatric).{0,15}(?:unfit|vulnerable)": [
        "frail patient", "geriatric assessment needed",
        "dose reduction", "modified regimen", "fitness-adjusted treatment",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # METASTATIC SITES — COMPREHENSIVE
    # ═══════════════════════════════════════════════════════════════════════
    r"right ventricl": [
        "cardiac metastasis", "right ventricular metastasis",
        "intracardiac metastasis", "right heart involvement",
        "cardiac involvement", "distant metastasis",
    ],
    r"(?:concern for|suspected|radiographic)\s+metastatic disease": [
        "suspected distant metastasis", "metastatic workup",
        "systemic disease", "M1 disease",
    ],
    r"metastatic disease to the (?:right ventricle|heart|cardiac)": [
        "cardiac metastasis", "intracardiac metastasis", "M1 disease",
        "distant metastasis", "right ventricular involvement",
    ],
    r"brain met|cerebral met|intracranial met|CNS met": [
        "brain metastasis", "cerebral metastasis", "intracranial disease",
        "CNS involvement", "whole brain radiation candidate",
        "SRS candidate", "dexamethasone",
    ],
    r"bone met|osseous met|skeletal met|bony met": [
        "bone metastasis", "osseous metastasis", "skeletal involvement",
        "denosumab", "zoledronic acid", "pathologic fracture risk",
    ],
    r"liver met|hepatic met": [
        "liver metastasis", "hepatic metastasis",
        "hepatic involvement", "visceral metastasis",
    ],
    r"lung met|pulmonary met|pulmonary nodul": [
        "lung metastasis", "pulmonary metastasis",
        "pulmonary nodule", "thoracic disease",
    ],
    r"peritoneal (?:met|carcinoma|disease|implant)|carcinomatosis": [
        "peritoneal metastasis", "peritoneal carcinomatosis",
        "diffuse peritoneal disease", "HIPEC candidate",
    ],
    r"adrenal (?:met|involvement|mass|lesion)": [
        "adrenal metastasis", "adrenal involvement",
        "distant metastasis",
    ],
    r"pleural (?:met|effusion|disease|involvement)|malignant.?(?:pleural|effusion)": [
        "pleural metastasis", "malignant pleural effusion",
        "pleural disease", "thoracentesis",
    ],
    r"leptomeningeal|carcinomatous meningitis|CSF.?positive": [
        "leptomeningeal disease", "carcinomatous meningitis",
        "CSF-positive", "intrathecal therapy candidate",
    ],
    r"skin met|cutaneous met|subcutaneous met|in-?transit met": [
        "skin metastasis", "cutaneous metastasis",
        "subcutaneous metastasis", "in-transit metastasis",
    ],
    r"lymph.?node met|distant (?:nodal|lymph.?node)|supraclavicular|mediastinal (?:lymph|node)": [
        "distant lymph node metastasis", "non-regional nodal disease",
        "M1 nodal", "systemic disease",
    ],
    r"oligometastatic|oligoprogress": [
        "oligometastatic disease", "limited metastatic disease",
        "SBRT candidate", "metastasis-directed therapy",
        "consolidative local therapy",
    ],
    r"widely metastatic|polymetastatic|diffuse metasta": [
        "widely metastatic", "polymetastatic disease",
        "palliative systemic therapy", "best supportive care consideration",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # COMORBIDITY → TREATMENT IMPLICATIONS
    # ═══════════════════════════════════════════════════════════════════════
    r"\bCKD\b|chronic kidney disease|renal impairment|renal insufficiency|GFR\s*<\s*\d": [
        "cisplatin ineligible", "carboplatin preferred",
        "renal insufficiency", "dose modification required",
    ],
    r"\bHep(?:atitis)?\s*[BC]\b|HBV|HCV|hepatitis": [
        "hepatic comorbidity", "liver disease",
        "immunosuppression risk", "hepatic function impaired",
        "viral reactivation risk",
    ],
    r"\bCHF\b|heart failure|(?:left|right) ventricular dysfunction|ejection fraction.{0,10}(?:reduced|low|\d{1,2}%)": [
        "cardiac comorbidity", "anthracycline contraindicated",
        "cardiotoxicity risk", "cardiac monitoring required",
    ],
    r"\bDM\b|diabetes|diabetic|A1C|hemoglobin A1C": [
        "diabetes mellitus", "steroid caution",
        "metabolic comorbidity", "wound healing concern",
    ],
    r"\bCOPD\b|chronic obstructive|emphysema|pulmonary fibrosis|ILD|interstitial lung": [
        "pulmonary comorbidity", "pneumonitis risk increased",
        "ICI-related pneumonitis concern", "pulmonary function impaired",
    ],
    r"autoimmune|(?:lupus|rheumatoid|crohn|colitis|psoriasis).{0,20}(?:history|active|known)": [
        "autoimmune comorbidity", "ICI autoimmune flare risk",
        "immunosuppression history", "checkpoint inhibitor caution",
    ],
    r"prior organ transplant|transplant recipient|immunosuppressed": [
        "transplant recipient", "ICI contraindicated relative",
        "immunosuppressed", "rejection risk with immunotherapy",
    ],
    r"declined (?:combination with )?chemotherapy|refused chemo": [
        "chemotherapy refused", "immunotherapy monotherapy",
        "chemo-free regimen", "single-agent systemic",
    ],
    r"declined (?:all )?treatment|refused treatment|comfort measures|hospice": [
        "treatment declined", "best supportive care",
        "palliative intent", "hospice referral",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # RECURRENCE PATTERNS
    # ═══════════════════════════════════════════════════════════════════════
    r"biopsy.proven recurrent": [
        "biopsy-confirmed recurrence", "pathologically confirmed recurrence",
        "recurrent/metastatic", "R/M disease",
    ],
    r"recurrent (?:SCC|squamous|lesion|disease|carcinoma|adenocarcinoma|tumor)": [
        "locoregional recurrence", "recurrent disease",
        "recurrent/metastatic", "R/M disease", "salvage setting",
    ],
    r"local recurrence|locoregional recurrence|in.?field recurrence": [
        "locoregional recurrence", "local failure",
        "salvage therapy", "re-irradiation candidate",
    ],
    r"distant recurrence|distant metasta|systemic recurrence": [
        "distant recurrence", "systemic disease",
        "M1 disease", "palliative systemic therapy",
    ],
    r"multiloculated.*collection|sublingual.*collection": [
        "abscess formation", "locoregional complication",
        "post-operative collection",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # CHEMOTHERAPY-SPECIFIC STATUS
    # ═══════════════════════════════════════════════════════════════════════
    r"platinum.?(?:refractory|resistant|failure|ineligible)": [
        "platinum-refractory", "platinum-resistant",
        "non-platinum regimen", "post-platinum",
    ],
    r"cisplatin.?(?:ineligible|unfit|contraindicated)": [
        "cisplatin-ineligible", "carboplatin substitute",
        "cetuximab consideration", "modified regimen",
    ],
    r"(?:taxane|paclitaxel|docetaxel).?(?:refractory|resistant|failure)": [
        "taxane-refractory", "post-taxane progression",
        "non-taxane salvage",
    ],
    r"(?:anthracycline|doxorubicin|adriamycin).?(?:refractory|resistant|failure|exposed)": [
        "anthracycline-exposed", "anthracycline-refractory",
        "cardiac monitoring needed", "lifetime dose limit",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # TREATMENT RESPONSE STATUS
    # ═══════════════════════════════════════════════════════════════════════

    # ── Excellent / complete response ────────────────────────────────────
    r"(?:excellent|good|major|significant|dramatic|marked|near.?complete)\s*(?:clinical\s*)?(?:response|regression|remission)": [
        "excellent clinical response", "major response",
        "pCR candidate", "pathologic complete response candidate",
        "downstaged", "de-escalation candidate",
        "response-adapted therapy", "favorable treatment response",
    ],
    r"(?:pathologic|pathological)\s*complete\s*response|pCR|\bypCR\b|ypT0\s*N0|ypT0|RCB[\s-]*(?:0|I)\b": [
        "pathologic complete response", "pCR", "ypT0N0",
        "excellent response", "de-escalation candidate",
        "adjuvant de-escalation", "treatment de-intensification",
        "residual cancer burden 0",
    ],
    r"complete\s*(?:clinical|radiographic|metabolic)\s*response|(?:imaging|PET|CT)\s*(?:shows?|demonstrat|reveal).{0,20}(?:complete|no).{0,10}(?:response|resolution|remission)": [
        "complete clinical response", "cCR", "metabolic complete response",
        "favorable imaging response", "de-escalation consideration",
    ],

    # ── Partial / good response ──────────────────────────────────────────
    r"partial\s*(?:clinical\s*)?response|(?:tumor|disease)\s*(?:shrinkage|regression|reduction)": [
        "partial response", "tumor regression",
        "treatment-responsive disease", "continuing current therapy",
    ],
    r"downstagd?e?|down.?staged": [
        "downstaged", "treatment response", "resectability improved",
        "conversion to resectable", "favorable response",
    ],

    # ── Poor / no response ───────────────────────────────────────────────
    r"(?:poor|minimal|no|absent|lack of)\s*(?:clinical\s*)?(?:response|regression|benefit)": [
        "poor response", "treatment-refractory",
        "non-responder", "escalation candidate",
        "alternative regimen", "clinical trial candidate",
    ],
    r"(?:stable|no change|unchanged)\s*(?:disease|tumor|lesion)": [
        "stable disease", "SD", "no response",
        "alternative therapy consideration",
    ],
    r"progressive disease|(?:tumor|disease)\s*(?:growth|enlargement|progression)\s*(?:on|during|despite)": [
        "progressive disease", "PD", "treatment failure",
        "switch therapy", "clinical trial candidate",
    ],
    r"(?:residual|persistent)\s*(?:disease|tumor|mass)": [
        "residual disease", "incomplete response",
        "adjuvant therapy indicated", "post-treatment residual",
    ],

    # ── Neoadjuvant-specific ─────────────────────────────────────────────
    r"(?:neoadjuvant|pre.?operative|induction|pre.?surgical)\s*(?:chemo|chemotherapy|systemic|treatment)": [
        "neoadjuvant chemotherapy", "NAC", "preoperative systemic therapy",
        "response assessment", "post-NAC surgery",
    ],
    r"(?:post|after|following|s/p|status.?post)\s*(?:neoadjuvant|NAC|pre.?operative)\s*(?:chemo|chemotherapy|systemic|treatment)?": [
        "post-neoadjuvant", "post-NAC", "NAC-treated",
        "post-preoperative chemotherapy",
        "response-guided adjuvant", "adjuvant decision",
    ],
    r"(?:response|responded)\s*(?:to|after|following)\s*(?:neoadjuvant|NAC|pre.?operative|induction)": [
        "neoadjuvant responder", "NAC response",
        "response-adapted adjuvant therapy",
        "de-escalation candidate",
    ],

    # ── Adjuvant escalation / de-escalation ──────────────────────────────
    r"de.?escalat|de.?intensif|treatment.?reduction|less.?intensive": [
        "de-escalation", "treatment de-intensification",
        "reduced adjuvant", "response-adapted",
    ],
    r"escalat|intensif|augment": [
        "escalation", "treatment intensification",
        "high-risk adjuvant", "aggressive adjuvant",
    ],
}


@dataclass
class InferenceResult:
    """Result of inference layer processing."""
    original_axes: Dict[str, str]
    expanded_axes: Dict[str, str]
    inferred_terms: Dict[str, List[str]]   # axis_name → list of added terms
    trajectory_flags: List[str] = field(default_factory=list)
    metastatic_sites: List[str] = field(default_factory=list)
    surgical_candidate: Optional[bool] = None


def run_inference(
    raw_text: str,
    axes: Dict[str, str],
) -> InferenceResult:
    """
    Apply inference rules to raw patient narrative and expand axes.

    Args:
        raw_text: Full raw patient narrative (used for pattern matching)
        axes: Dict of axis_name → axis_string (from LLM extraction)

    Returns:
        InferenceResult with expanded axes and extracted flags
    """
    print(f"    [Inference Engine] Running inference on {len(raw_text)} chars of raw text")
    print(f"    [Inference Engine] Input axes: {list(axes.keys())}")
    added: Dict[str, List[str]] = {k: [] for k in axes}
    trajectory_flags: List[str] = []
    metastatic_sites: List[str] = []
    surgical_candidate: Optional[bool] = None

    # Apply each inference rule against the full narrative
    matched_patterns = 0
    for pattern, terms in INFERENCE_MAP.items():
        if re.search(pattern, raw_text, re.IGNORECASE):
            matched_patterns += 1
            # Determine which axis this inference best belongs to
            axis = _assign_to_axis(pattern, terms, axes)
            added[axis] = list(set(added[axis] + terms))
            # Show first 3 pattern matches in detail
            if matched_patterns <= 8:
                print(f"    [Inference Engine] MATCH: /{pattern[:50]}/ → axis={axis}, "
                      f"+{len(terms)} terms: {terms[:3]}{'...' if len(terms) > 3 else ''}")
    print(f"    [Inference Engine] Total pattern matches: {matched_patterns} out of {len(INFERENCE_MAP)} rules")

    # Extract trajectory flags
    print(f"    [Inference Engine] Checking trajectory flags...")
    if re.search(r"progress(?:ing|ion) on .{0,40}(?:pembrolizumab|nivolumab|atezolizumab|durvalumab|ICI|checkpoint|immunotherapy|anti.PD)", raw_text, re.IGNORECASE):
        trajectory_flags.append("ici_refractory")
        trajectory_flags.append("progressing_on_ici")
    if re.search(r"(?:progress|fail|refract|resist).{0,30}(?:erlotinib|gefitinib|osimertinib|EGFR.?TKI)", raw_text, re.IGNORECASE):
        trajectory_flags.append("egfr_tki_resistant")
    if re.search(r"(?:progress|fail|refract|resist).{0,30}(?:crizotinib|alectinib|ALK)", raw_text, re.IGNORECASE):
        trajectory_flags.append("alk_tki_resistant")
    if re.search(r"(?:progress|fail|refract|resist).{0,30}(?:tamoxifen|letrozole|anastrozole|aromatase|endocrine|hormonal)", raw_text, re.IGNORECASE):
        trajectory_flags.append("endocrine_resistant")
    if re.search(r"castration.?resist|hormone.?refract", raw_text, re.IGNORECASE):
        trajectory_flags.append("castration_resistant")
    if re.search(r"platinum.?(?:refract|resist|fail)", raw_text, re.IGNORECASE):
        trajectory_flags.append("platinum_refractory")
    if re.search(r"no longer (?:a )?surgical|not (?:a )?surgical candidate|unresectable|inoperable|medically inoperable", raw_text, re.IGNORECASE):
        surgical_candidate = False

    # Treatment response trajectory flags
    if re.search(r"(?:excellent|good|major|significant|dramatic|marked|near.?complete)\s*(?:clinical\s*)?(?:response|regression|remission)", raw_text, re.IGNORECASE):
        trajectory_flags.append("excellent_response")
    if re.search(r"(?:pathologic|pathological)\s*complete\s*response|pCR|\bypCR\b|ypT0", raw_text, re.IGNORECASE):
        trajectory_flags.append("pcr")
        trajectory_flags.append("excellent_response")
    if re.search(r"(?:poor|minimal|no|absent|lack of)\s*(?:clinical\s*)?(?:response|regression|benefit)", raw_text, re.IGNORECASE):
        trajectory_flags.append("poor_response")
    if re.search(r"(?:neoadjuvant|pre.?operative|induction|NAC)\s*(?:chemo|chemotherapy|systemic|treatment)", raw_text, re.IGNORECASE):
        trajectory_flags.append("post_neoadjuvant")
    if re.search(r"(?:post|after|following|s/p|status.?post)\s*(?:neoadjuvant|NAC|pre.?operative)", raw_text, re.IGNORECASE):
        trajectory_flags.append("post_neoadjuvant")
    if re.search(r"de.?escalat|de.?intensif", raw_text, re.IGNORECASE):
        trajectory_flags.append("de_escalation")
    if re.search(r"(?:residual|persistent)\s*(?:disease|tumor|mass)", raw_text, re.IGNORECASE):
        trajectory_flags.append("residual_disease")

    if trajectory_flags:
        print(f"    [Inference Engine] Trajectory flags: {trajectory_flags}")
    if surgical_candidate is not None:
        print(f"    [Inference Engine] Surgical candidate: {surgical_candidate}")

    # Extract metastatic sites
    met_patterns = [
        (r"right ventricl", "right ventricle"),
        (r"cardiac met|intracardiac met|heart met", "cardiac"),
        (r"lung met|pulmonary met|pulmonary nodul", "lung"),
        (r"liver met|hepatic met", "liver"),
        (r"bone met|osseous met|skeletal met|bony met", "bone"),
        (r"brain met|cerebral met|intracranial met|CNS met", "brain"),
        (r"peritoneal (?:met|carcinoma|disease)|carcinomatosis", "peritoneal"),
        (r"adrenal (?:met|involvement|mass)", "adrenal"),
        (r"pleural (?:met|effusion|disease)|malignant.?effusion", "pleural"),
        (r"leptomeningeal|carcinomatous meningitis", "leptomeningeal"),
        (r"skin met|cutaneous met|subcutaneous met|in.?transit met", "skin"),
        (r"distant (?:nodal|lymph.?node)|supraclavicular met|mediastinal met", "distant lymph node"),
    ]
    for pat, label in met_patterns:
        if re.search(pat, raw_text, re.IGNORECASE):
            metastatic_sites.append(label)

    if metastatic_sites:
        print(f"    [Inference Engine] Metastatic sites detected: {metastatic_sites}")

    # Build expanded axes
    expanded_axes = {}
    for axis_name, axis_str in axes.items():
        extra = added.get(axis_name, [])
        if extra:
            expanded_axes[axis_name] = axis_str + " " + " ".join(extra)
        else:
            expanded_axes[axis_name] = axis_str

    # ── Per-axis bidirectional expansion ──────────────────────────────────
    # Run the same ONCOLOGY_EXPANSIONS / REVERSE_EXPANSIONS / STAGING_SYNONYMS
    # / CLINICAL_SYNONYMS expansion independently on each axis string so that
    # axis-specific sub-queries carry their own expanded vocabulary.
    for axis_name in expanded_axes:
        expanded_axes[axis_name] = _expand_axis_text(expanded_axes[axis_name])

    # Enrich primary_cancer axis with AJCC cancer-type synonyms
    try:
        from src.api.services.ontology_loader import expand_cancer_site_synonyms
        primary = expanded_axes.get("primary_cancer", "")
        if primary:
            site_synonyms = expand_cancer_site_synonyms(primary)
            if site_synonyms:
                # Add synonyms not already present
                existing_lower = primary.lower()
                new_terms = [s for s in site_synonyms if s.lower() not in existing_lower]
                if new_terms:
                    expanded_axes["primary_cancer"] = primary + " " + " ".join(new_terms)
    except Exception:
        pass  # ontology not available — continue with hardcoded terms

    # Enrich biomarker_profile axis with ontology biomarker terms
    try:
        from src.api.services.ontology_loader import get_biomarker_keywords
        bio_axis = expanded_axes.get("biomarker_profile", "")
        if bio_axis:
            bio_lower = bio_axis.lower()
            matched_categories: List[str] = []
            for cat_name, cat_terms in get_biomarker_keywords().items():
                for t in cat_terms:
                    if t.lower() in bio_lower:
                        matched_categories.extend(
                            ct for ct in cat_terms
                            if ct.lower() not in bio_lower
                        )
                        break
            if matched_categories:
                expanded_axes["biomarker_profile"] = (
                    bio_axis + " " + " ".join(matched_categories[:15])
                )
    except Exception:
        pass

    # Enrich disease_trajectory axis with ICI resistance terms from trial ontology
    try:
        from src.api.services.ontology_loader import get_ici_resistance_terms
        traj_axis = expanded_axes.get("disease_trajectory", "")
        if traj_axis and trajectory_flags:
            ici_terms = get_ici_resistance_terms()
            traj_lower = traj_axis.lower()
            new_ici = [t for t in ici_terms if t.lower() not in traj_lower]
            if new_ici:
                expanded_axes["disease_trajectory"] = (
                    traj_axis + " " + " ".join(new_ici[:10])
                )
    except Exception:
        pass

    # Enrich metastatic_concern axis with metastatic pattern terms from trial ontology
    try:
        from src.api.services.ontology_loader import get_metastatic_pattern_terms
        met_axis = expanded_axes.get("metastatic_concern", "")
        if met_axis and metastatic_sites:
            met_terms = get_metastatic_pattern_terms()
            met_lower = met_axis.lower()
            new_met = [t for t in met_terms if t.lower() not in met_lower]
            if new_met:
                expanded_axes["metastatic_concern"] = (
                    met_axis + " " + " ".join(new_met[:10])
                )
    except Exception:
        pass

    return InferenceResult(
        original_axes=axes,
        expanded_axes=expanded_axes,
        inferred_terms=added,
        trajectory_flags=trajectory_flags,
        metastatic_sites=metastatic_sites,
        surgical_candidate=surgical_candidate,
    )


def _expand_axis_text(axis_text: str) -> str:
    """
    Apply bidirectional query expansion (ONCOLOGY_EXPANSIONS, REVERSE_EXPANSIONS,
    STAGING_SYNONYMS, CLINICAL_SYNONYMS) to a single axis string.

    This is the per-axis equivalent of expand_query() in enhanced_rag_service.py.
    Each axis gets its own expansion so that axis-specific sub-queries carry
    vocabulary relevant to that axis only.
    """
    if not axis_text or len(axis_text) < 3:
        return axis_text

    try:
        from src.api.services.ontology_loader import get_expansion_tables
        tables = get_expansion_tables()
    except Exception:
        return axis_text

    al = axis_text.lower()
    expansions: List[str] = []

    # Forward expansion: abbreviation → full term
    for abbrev, expansion in tables["oncology"].items():
        if re.search(rf'\b{re.escape(abbrev)}\b', al, re.IGNORECASE):
            expansions.append(expansion)

    # Reverse expansion: full term → abbreviation
    for full_term, abbrev_expansion in tables["reverse"].items():
        if full_term.lower() in al:
            expansions.append(abbrev_expansion)

    # Staging-specific expansions
    for staging_term, synonyms in tables["staging"].items():
        for synonym in synonyms:
            if synonym.lower() in al:
                expansions.extend(s for s in synonyms if s.lower() not in al)
                break

    # Clinical concept synonyms
    for term, syns in tables["clinical"].items():
        if term in al:
            expansions.append(syns)

    if not expansions:
        return axis_text

    # Deduplicate: keep terms not already in the axis text
    seen: Set[str] = set()
    unique: List[str] = []
    for exp in expansions:
        for word in exp.split():
            wl = word.lower()
            if wl not in seen and wl not in al and len(wl) > 1:
                seen.add(wl)
                unique.append(word)

    # Cap expansion to avoid bloating any single axis
    return axis_text + " " + " ".join(unique[:30]) if unique else axis_text


def _assign_to_axis(pattern: str, terms: List[str], axes: Dict[str, str]) -> str:
    """Heuristically assign inferred terms to the most relevant axis."""
    # Pattern-to-axis hints
    axis_hints = {
        # Metastatic concern
        "ventricl|cardiac met|intracardiac": "metastatic_concern",
        "metastatic|M1|distant|oligomet|brain met|bone met|liver met|lung met|peritoneal|adrenal|pleural|leptomeningeal|skin met|subcutaneous|polymetastatic": "metastatic_concern",
        # Patient factors
        "surgical|inoperable|borderline resect": "patient_factors",
        "CKD|Hep|renal|hepatic|CHF|heart failure|COPD|diabetes|autoimmune|transplant|frail|bedridden|declined.*treatment|hospice|comfort": "patient_factors",
        "ECOG|KPS|Karnofsky|performance status|bed.?bound": "patient_factors",
        # Disease trajectory
        "ICI|checkpoint|pembrolizumab|nivolumab|atezolizumab|durvalumab|immunotherapy.{0,5}(?:refract|fail|resist)": "disease_trajectory",
        "recurrent|recurrence": "disease_trajectory",
        "progress|refract|resist": "disease_trajectory",
        "platinum.?refract|taxane.?refract|anthracycline": "disease_trajectory",
        "hyperprogress|acquired resistance|primary resistance": "disease_trajectory",
        # Current treatment
        "line|salvage|1L|2L|3L|frontline": "current_treatment",
        "started on|na[ïi]ve|untreated": "current_treatment",
        "declined.*chemo|refused chemo|chemo.?free": "current_treatment",
        # Biomarker profile
        "CPS|PD-?L1|TPS": "biomarker_profile",
        "HPV|p16": "biomarker_profile",
        "MSI|dMMR|pMMR|TMB": "biomarker_profile",
        "EGFR|ALK|ROS1|KRAS|BRAF|BRCA|HER2": "biomarker_profile",
        "ER.?positive|ER.?negative|PR.?positive|triple.negative|TNBC": "biomarker_profile",
        "hormone.?receptor|estrogen receptor|progesterone receptor": "biomarker_profile",
        "castration.?resist|hormone.?refract|endocrine.?resist": "disease_trajectory",
        # Prior treatment
        "prior.*(?:radiation|RT|XRT)|post.?radiation|re.?irradiat|radiation.?na": "prior_definitive_treatment",
        # Targeted therapy resistance
        "EGFR.?TKI|ALK.?TKI|erlotinib|gefitinib|osimertinib|crizotinib|alectinib|imatinib|vemurafenib|dabrafenib|trastuzumab|PARP|olaparib": "disease_trajectory",
        # Treatment response → disease_trajectory axis
        "excellent|complete.*response|pCR|ypT0|partial.*response|poor.*response|residual.*disease|stable.*disease|progressive disease": "disease_trajectory",
        "neoadjuvant|pre.?operative|induction|post.?NAC|post.?neoadjuvant": "disease_trajectory",
        "de.?escalat|de.?intensif|escalat|intensif": "disease_trajectory",
        "downstagd?e?|down.?staged|conversion|resectab": "disease_trajectory",
    }
    for hint_pat, axis_name in axis_hints.items():
        if re.search(hint_pat, pattern, re.IGNORECASE):
            if axis_name in axes:
                return axis_name
    # Fallback: first available axis
    return next(iter(axes), "primary_cancer")


def _build_axes_from_structure(query_structure) -> Dict[str, str]:
    """Build 8-axis dict from QueryStructure regex-extracted fields as fallback.

    Used when LLM axes are not available (e.g. simple queries that skip LLM
    extraction, or when the LLM call fails).

    Args:
        query_structure: QueryStructure dataclass from query_structuring_service

    Returns:
        Dict mapping axis names to string values built from structure fields.
    """
    cancer = query_structure.cancer
    treatment = getattr(query_structure, "treatment", None)
    patient = getattr(query_structure, "patient", None)
    clinical_history = getattr(query_structure, "clinical_history", None)

    tnm_parts = []
    tnm_str = cancer.get_tnm_string() if hasattr(cancer, "get_tnm_string") else None
    if tnm_str:
        tnm_parts.append(tnm_str)
    for attr in ("doi", "pni", "lvi", "margins", "grade", "lymph_nodes"):
        val = getattr(cancer, attr, None)
        if val:
            tnm_parts.append(f"{attr}: {val}")

    return {
        "primary_cancer": f"{cancer.site or ''} {cancer.histology or ''} {cancer.site_detail or ''}".strip(),
        "tnm_pathology": (f"{cancer.stage or ''} {' '.join(tnm_parts)}").strip(),
        "prior_definitive_treatment": getattr(treatment, "raw_text", "") or "",
        "current_treatment": "",  # no separate field in regex extraction
        "biomarker_profile": " ".join(cancer.biomarkers or []),
        "disease_trajectory": (
            getattr(clinical_history, "recurrence_info", "") or ""
        ),
        "metastatic_concern": (
            getattr(clinical_history, "imaging_findings", "") or ""
        ),
        "patient_factors": " ".join(getattr(patient, "comorbidities", []) or []),
    }


def apply_inference_to_query_structure(query_structure, raw_text: str) -> dict:
    """
    Run inference rules against the raw clinical narrative text,
    then merge inferred terms into structured axes derived from query_structure.

    Args:
        query_structure: QueryStructure dataclass from query_structuring_service
        raw_text: The original raw query string — the full clinical narrative.
                  This is what the regex patterns are written to match against.
                  Do NOT pass query_structure.to_dict() or any structured repr here.

    Returns:
        Dict with expanded_axes, inferred_terms, trajectory_flags,
        metastatic_sites, and surgical_candidate.
    """
    _empty = {
        "expanded_axes": {},
        "trajectory_flags": [],
        "metastatic_sites": [],
        "surgical_candidate": None,
        "inferred_terms": {},
    }

    # Guard: if raw_text is accidentally a dict or dataclass, recover the string
    if not isinstance(raw_text, str):
        raw_text = getattr(query_structure, "original_query", "") or ""
        if not raw_text:
            print("[Inference] WARNING: raw_text is not a string and no fallback "
                  "available — inference will return empty")
            return _empty

    if not raw_text or len(raw_text) < 5:
        return _empty

    # Build axes dict from query_structure's LLM axes or fallback to structure fields.
    # Check public llm_axes first (ReconciledStructure), then private _llm_axes
    # (QueryStructure), then fall back to building from regex-extracted fields.
    axes: Dict[str, str] = {}

    # 1. Try public llm_axes (e.g. from ReconciledStructure)
    llm_axes = getattr(query_structure, "llm_axes", None)
    if llm_axes and isinstance(llm_axes, dict) and any(llm_axes.values()):
        print(f"    [Inference Wrapper] Using LLM-extracted axes (public llm_axes)")
        axes = dict(llm_axes)
    else:
        # 2. Try private _llm_axes (set dynamically on QueryStructure by merge_llm_extraction)
        llm_axes = getattr(query_structure, "_llm_axes", None)
        if llm_axes and isinstance(llm_axes, dict) and any(llm_axes.values()):
            print(f"    [Inference Wrapper] Using LLM-extracted axes (8-axis template)")
            axes = dict(llm_axes)

    if axes:
        # Ensure all 8 axis keys are present (fill missing with empty string)
        for key in ("primary_cancer", "tnm_pathology", "prior_definitive_treatment",
                     "current_treatment", "biomarker_profile", "disease_trajectory",
                     "metastatic_concern", "patient_factors"):
            axes.setdefault(key, "")
        for k, v in axes.items():
            if v:
                print(f"      {k}: {v[:60]}{'...' if len(v) > 60 else ''}")
    elif hasattr(query_structure, "used_llm_extraction") and query_structure.used_llm_extraction:
        # LLM extraction ran but axes dict is missing — build from structure fields
        print(f"    [Inference Wrapper] LLM extraction flagged but axes unavailable, "
              f"building from structure fields")
        axes = _build_axes_from_structure(query_structure)
    else:
        print(f"    [Inference Wrapper] Using regex-extracted axes (fallback)")
        axes = _build_axes_from_structure(query_structure)

    result = run_inference(raw_text, axes)

    return {
        "expanded_axes": result.expanded_axes,
        "inferred_terms": result.inferred_terms,
        "trajectory_flags": result.trajectory_flags,
        "metastatic_sites": result.metastatic_sites,
        "surgical_candidate": result.surgical_candidate,
    }
