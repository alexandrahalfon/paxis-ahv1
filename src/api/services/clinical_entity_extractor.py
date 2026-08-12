"""
Clinical Entity Extractor for Profile-Aware Retrieval

Extracts medically relevant entities from queries to use as filters/boosters
in the retrieval system.

Entities extracted:
- Cancer type (breast, lung, prostate, etc.)
- Stage (I, II, III, IV, TNM)
- Biomarkers (ER, PR, HER2, EGFR, etc.)
- Treatment modalities (RT, chemo, surgery, etc.)
- Clinical concepts (pCR, recurrence, survival, etc.)
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ClinicalProfile:
    """Extracted clinical profile from a query."""
    cancer_type: Optional[str] = None
    cancer_subtype: Optional[str] = None
    stage: Optional[str] = None
    tnm: Optional[str] = None
    biomarkers: List[str] = field(default_factory=list)
    treatments: List[str] = field(default_factory=list)
    clinical_concepts: List[str] = field(default_factory=list)
    anatomic_sites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cancer_type": self.cancer_type,
            "cancer_subtype": self.cancer_subtype,
            "stage": self.stage,
            "tnm": self.tnm,
            "biomarkers": self.biomarkers,
            "treatments": self.treatments,
            "clinical_concepts": self.clinical_concepts,
            "anatomic_sites": self.anatomic_sites,
        }
    
    def get_filter_terms(self) -> List[str]:
        """Get all terms that should be used for filtering/boosting."""
        terms = []
        if self.cancer_type:
            terms.append(self.cancer_type)
        if self.cancer_subtype:
            terms.append(self.cancer_subtype)
        if self.stage:
            terms.append(self.stage)
        terms.extend(self.biomarkers)
        terms.extend(self.treatments)
        terms.extend(self.clinical_concepts)
        terms.extend(self.anatomic_sites)
        return terms
    
    def is_empty(self) -> bool:
        return not any([
            self.cancer_type, self.cancer_subtype, self.stage, self.tnm,
            self.biomarkers, self.treatments, self.clinical_concepts, self.anatomic_sites
        ])


class ClinicalEntityExtractor:
    """
    Extracts clinical entities from medical queries.
    
    Usage:
        extractor = ClinicalEntityExtractor()
        profile = extractor.extract("A 55yo female with cT3N1M0 ER+ HER2- breast cancer...")
        print(profile.cancer_type)  # "breast"
        print(profile.biomarkers)   # ["ER+", "HER2-"]
    """
    
    # Cancer type patterns
    CANCER_TYPES = {
        "breast": r"\b(breast|mammary)\s*(cancer|carcinoma|tumor|ca)?\b",
        "lung": r"\b(lung|pulmonary|nsclc|sclc|non-small cell|small cell)\s*(cancer|carcinoma|tumor|ca)?\b",
        "prostate": r"\b(prostate|prostatic)\s*(cancer|carcinoma|adenocarcinoma|ca)?\b",
        "colorectal": r"\b(colorectal|colon|rectal|rectum|crc)\s*(cancer|carcinoma|adenocarcinoma|ca)?\b",
        "head_and_neck": r"\b(head and neck|h&n|hnscc|oropharynx|oropharyngeal|larynx|laryngeal|oral cavity|nasopharynx|npc|hypopharynx)\s*(cancer|carcinoma|scc)?\b",
        "cervical": r"\b(cervical|cervix)\s*(cancer|carcinoma|ca)?\b",
        "endometrial": r"\b(endometrial|uterine|uterus)\s*(cancer|carcinoma|ca)?\b",
        "ovarian": r"\b(ovarian|ovary)\s*(cancer|carcinoma|ca)?\b",
        "bladder": r"\b(bladder|urothelial|mibc|nmibc)\s*(cancer|carcinoma|ca)?\b",
        "kidney": r"\b(kidney|renal|rcc)\s*(cancer|carcinoma|cell carcinoma|ca)?\b",
        "pancreatic": r"\b(pancreatic|pancreas)\s*(cancer|carcinoma|adenocarcinoma|ca)?\b",
        "esophageal": r"\b(esophageal|esophagus|gastroesophageal|gej)\s*(cancer|carcinoma|adenocarcinoma|ca)?\b",
        "gastric": r"\b(gastric|stomach)\s*(cancer|carcinoma|adenocarcinoma|ca)?\b",
        "liver": r"\b(liver|hepatocellular|hcc|hepatic)\s*(cancer|carcinoma|ca)?\b",
        "brain": r"\b(brain|glioma|glioblastoma|gbm|astrocytoma|meningioma|cns)\s*(cancer|tumor)?\b",
        "melanoma": r"\b(melanoma|skin cancer)\b",
        "lymphoma": r"\b(lymphoma|hodgkin|non-hodgkin|dlbcl|nhl|hl)\b",
        "leukemia": r"\b(leukemia|aml|all|cml|cll)\b",
        "testicular": r"\b(testicular|testis|seminoma|non-seminoma|germ cell)\s*(cancer|tumor|ca)?\b",
        "thyroid": r"\b(thyroid|papillary|follicular|medullary|anaplastic)\s*(cancer|carcinoma|ca)?\b",
        "sarcoma": r"\b(sarcoma|soft tissue|bone tumor|osteosarcoma|ewing)\b",
    }
    
    # Cancer subtypes
    CANCER_SUBTYPES = {
        # Breast subtypes
        "dcis": r"\b(dcis|ductal carcinoma in situ)\b",
        "lcis": r"\b(lcis|lobular carcinoma in situ)\b",
        "idc": r"\b(idc|invasive ductal|infiltrating ductal)\b",
        "ilc": r"\b(ilc|invasive lobular|infiltrating lobular)\b",
        "inflammatory_breast_cancer": r"\b(inflammatory breast|ibc|t4d|ct4d|pt4d)\b",
        "triple_negative": r"\b(triple negative|tnbc|triple-negative)\b",
        "paget": r"\b(paget|paget's disease)\b",
        
        # Lung subtypes
        "nsclc": r"\b(nsclc|non-small cell)\b",
        "sclc": r"\b(sclc|small cell lung)\b",
        "adenocarcinoma": r"\b(adenocarcinoma)\b",
        "squamous": r"\b(squamous cell|scc)\b",
        
        # Head and neck subtypes
        # Pattern accepts hyphenated forms like "p16-positive oropharynx"
        # via [\s-]? — previously required a literal space, missing the
        # common hyphenated medical-record phrasing.
        "hpv_positive_oropharynx": r"\b(?:hpv[\s-]?\+|hpv[\s-]?positive|p16[\s-]?\+|p16[\s-]?positive).*(oropharynx|tonsil|base of tongue)\b",
        "nasopharyngeal": r"\b(nasopharyngeal|npc|nasopharynx)\b",
        
        # GYN subtypes
        "cervical_squamous": r"\b(cervical|cervix).*(squamous)\b",
        "endometrial_serous": r"\b(serous|papillary serous).*(endometrial|uterine)\b",
        
        # Other
        "seminoma": r"\b(seminoma)\b",
        "non_seminoma": r"\b(non-seminoma|nonseminoma)\b",
        "ependymoma": r"\b(ependymoma)\b",
        "medulloblastoma": r"\b(medulloblastoma)\b",
    }
    
    # Critical staging patterns that imply specific treatment protocols
    CRITICAL_STAGING_PATTERNS = {
        # Inflammatory breast cancer (T4d) - requires trimodality
        "inflammatory_breast_cancer": [
            r"[cp]?t4d",  # Match T4d anywhere (cT4dN2M0, pT4d, etc.)
            r"\binflammatory\s+breast\b",
            r"\bibc\b",
            r"\berythema.*(breast|skin)\b",
            r"\bpeau\s+d'orange\b",
        ],
        # Locally advanced breast requiring neoadjuvant
        "locally_advanced_breast": [
            r"[cp]?t4[abc]",  # Match T4a/b/c anywhere
            r"[cp]?t3.*n[12]",  # T3N1 or T3N2
        ],
        # Early stage HPV+ oropharynx - may omit surgery
        "early_hpv_oropharynx": [
            r"\b(t1|t2).*n[01].*(tonsil|oropharynx|base of tongue).*(hpv|p16)\b",
            r"\b(hpv|p16).*(t1|t2).*n[01].*(tonsil|oropharynx)\b",
            r"\bt[12]n[01].*tonsil\b",
            r"\bearly.*(tonsil|oropharynx)\b",
        ],
        # High-risk prostate requiring long-term ADT
        "high_risk_prostate": [
            r"\bgleason\s*(9|10|[45]\+[45])\b",
            r"[cp]?t3[ab]?.*prostate",
            r"\bpsa\s*[>≥]\s*20\b",
            r"\bhigh[- ]?risk\s+prostate\b",
        ],
        # Cervical with intermediate risk factors (GOG 92 criteria)
        "cervical_intermediate_risk": [
            r"\bcervix.*deep.*invasion\b",
            r"\bcervix.*lvsi\b",
            r"\bcervix.*large.*tumor\b",
            r"\bsedlis\s+criteria\b",
            r"\bdeep\s+invasion.*lvsi\b",
            r"\blvsi.*deep\s+invasion\b",
        ],
    }
    
    # Stage patterns
    STAGE_PATTERNS = [
        # Roman numeral stages
        (r"\bstage\s*(0|I{1,3}V?|IV)\s*([ABC])?\b", "stage"),
        # Numeric stages
        (r"\bstage\s*([0-4])\s*([ABC])?\b", "stage"),
        # TNM staging
        (r"\b[cp]?T([0-4]|is|a|x)[a-d]?\s*N([0-3]|x)[a-c]?\s*M([01]|x)\b", "tnm"),
        # Clinical vs pathologic
        (r"\b(clinical|pathologic)\s+stage\b", "stage_type"),
        # Descriptive stages
        (r"\b(early stage|locally advanced|metastatic|advanced|oligometastatic)\b", "stage_desc"),
    ]
    
    # Biomarker patterns
    BIOMARKERS = {
        # Breast biomarkers
        "ER+": r"\b(er\+|er positive|estrogen receptor positive|er-positive)\b",
        "ER-": r"\b(er-|er negative|estrogen receptor negative|er-negative)\b",
        "PR+": r"\b(pr\+|pr positive|progesterone receptor positive|pr-positive)\b",
        "PR-": r"\b(pr-|pr negative|progesterone receptor negative|pr-negative)\b",
        "HER2+": r"\b(her2\+|her2 positive|her2-positive|erbb2\+|her2 amplified)\b",
        "HER2-": r"\b(her2-|her2 negative|her2-negative)\b",
        "HER2 mutant": r"\b(her2 mutant|her2 mutation|her2 mutated)\b",
        
        # Lung biomarkers
        "EGFR mutant": r"\b(egfr\+|egfr positive|egfr mutant|egfr mutation|egfr mutated|egfr exon \d+|egfr l858r|egfr t790m|egfr del ?19)\b",
        "EGFR wild-type": r"\b(egfr wild[\s-]?type|egfr wt|egfr negative|egfr-)\b",
        "ALK+": r"\b(alk\+|alk positive|alk rearrangement|alk fusion|alk translocation)\b",
        "ALK-": r"\b(alk-|alk negative)\b",
        "ROS1+": r"\b(ros1\+|ros1 positive|ros1 rearrangement|ros1 fusion)\b",
        "KRAS mutant": r"\b(kras\+|kras mutant|kras mutation|kras mutated|kras g12c|kras g12d|kras g12v)\b",
        "KRAS wild-type": r"\b(kras wild[\s-]?type|kras wt|kras negative|kras-)\b",
        "BRAF mutant": r"\b(braf\+|braf mutant|braf mutation|braf mutated|braf v600e|braf v600)\b",
        "BRAF wild-type": r"\b(braf wild[\s-]?type|braf wt|braf negative|braf-)\b",
        "RET+": r"\b(ret fusion|ret rearrangement|ret positive|ret\+|ret mutant|ret mutation)\b",
        "MET altered": r"\b(met exon ?14|met amplification|met amplified|met overexpression)\b",
        "NTRK fusion": r"\b(ntrk fusion|ntrk positive|ntrk\+|ntrk rearrangement)\b",
        
        # PD-L1 expression
        "PD-L1 high": r"\b(pd-?l1)\s*(positive|\+|high|>?\s*50%?|tps\s*[>≥]\s*\d+)\b",
        "PD-L1 low": r"\b(pd-?l1)\s*(negative|-|low|<?\s*1%?|tps\s*[<≤]\s*\d+)\b",
        "PD-L1": r"\b(pd-?l1)\s*(expression|status|score|tps|cps)\b",
        
        # MSI / MMR
        "MSI-H": r"\b(msi-?h|msi high|microsatellite instability[\s-]*high)\b",
        "MSS": r"\b(mss|msi-?l|microsatellite stable)\b",
        "dMMR": r"\b(dmmr|mismatch repair deficient)\b",
        "pMMR": r"\b(pmmr|mismatch repair proficient)\b",
        
        # TMB
        "TMB-H": r"\b(tmb[\s-]*high|tmb[\s-]*h|tumor mutational burden[\s-]*high)\b",
        
        # BRCA
        "BRCA mutant": r"\b(brca\+|brca1|brca2|brca mutation|brca positive|brca mutant|brca pathogenic)\b",
        "BRCA wild-type": r"\b(brca wild[\s-]?type|brca wt|brca negative|brca-)\b",
        
        # HPV / p16
        # The previous HPV- pattern had bare `hpv-` and `p16-` alternatives
        # that matched connector hyphens in "p16-positive" / "HPV-positive",
        # so the wrong polarity was tagged (same root-cause as commits
        # 0d69985, 863fe0a, and the clinical_inference.py sweep). Also
        # the positive pattern previously required a space between the
        # marker and polarity word, missing common hyphenated forms.
        # Fixes:
        #   - Positive: [\s-]? lets hyphen act as connector
        #     ("p16-positive" matches; was previously a miss)
        #   - Negative: bare `-` polarity marker requires word boundary
        #     after it ("p16-" matches; "p16-positive" no longer false-
        #     positives as negative)
        "HPV+": r"\b(?:hpv[\s-]?\+|hpv[\s-]?positive|p16[\s-]?\+|p16[\s-]?positive)",
        "HPV-": r"\b(?:hpv-(?=\s|$|[,.;:])|hpv[\s-]?negative|p16-(?=\s|$|[,.;:])|p16[\s-]?negative)",
        
        # CNS biomarkers
        "IDH mutant": r"\b(idh[12]? mutant|idh mutation|idh mutated|idh\+)\b",
        "IDH wild-type": r"\b(idh wild[\s-]?type|idh wt)\b",
        "MGMT methylated": r"\b(mgmt methylated|mgmt methylation|mgmt promoter methylat)\b",
        "MGMT unmethylated": r"\b(mgmt unmethylated|mgmt un-?methylated)\b",
        "1p/19q co-deleted": r"\b(1p/?19q co-?deleted|1p/?19q co-?deletion|1p/?19q loss)\b",
        
        # Other actionable
        "PIK3CA mutant": r"\b(pik3ca mutant|pik3ca mutation|pik3ca mutated|pik3ca\+)\b",
        "FGFR altered": r"\b(fgfr[1-4]? alteration|fgfr altered|fgfr fusion|fgfr mutation|fgfr amplification)\b",
        
        # Prostate
        "PSA elevated": r"\b(psa\s*(?:>|≥|>=)\s*[\d.]+)\b",
    }
    
    # Treatment patterns
    TREATMENTS = {
        # Radiation
        "radiation": r"\b(radiation|radiotherapy|rt|xrt|ebrt|imrt|vmat|sbrt|srs|wbrt)\b",
        "brachytherapy": r"\b(brachytherapy|hdr|ldr|internal radiation)\b",
        "proton": r"\b(proton|proton therapy|proton beam)\b",
        
        # Surgery
        "surgery": r"\b(surgery|surgical|resection|excision)\b",
        "mastectomy": r"\b(mastectomy|mrm|simple mastectomy|radical mastectomy)\b",
        "lumpectomy": r"\b(lumpectomy|bcs|breast conserving|wide local excision|wle)\b",
        "lymph_node_surgery": r"\b(alnd|slnb|sentinel node|axillary dissection|lymph node dissection)\b",
        
        # Systemic therapy
        "chemotherapy": r"\b(chemotherapy|chemo|ctx|neoadjuvant chemo|adjuvant chemo)\b",
        "immunotherapy": r"\b(immunotherapy|checkpoint inhibitor|ici|pd-1|pd-l1|pembrolizumab|nivolumab|atezolizumab|durvalumab|ipilimumab)\b",
        "targeted_therapy": r"\b(targeted therapy|tki|tyrosine kinase)\b",
        "hormone_therapy": r"\b(hormone therapy|endocrine therapy|hormonal|tamoxifen|aromatase inhibitor|ai|adt|androgen deprivation)\b",
        "her2_therapy": r"\b(trastuzumab|herceptin|pertuzumab|perjeta|tdm1|t-dm1|kadcyla|tchp)\b",
        
        # Combined modality
        "chemoradiation": r"\b(chemoradiation|chemoradiotherapy|crt|concurrent chemo)\b",
        "neoadjuvant": r"\b(neoadjuvant|preoperative|induction)\b",
        "adjuvant": r"\b(adjuvant|postoperative)\b",
    }
    
    # Clinical concepts
    CLINICAL_CONCEPTS = {
        "pcr": r"\b(pcr|pathologic complete response|ypt0|ypn0|complete response)\b",
        "recurrence": r"\b(recurrence|recurrent|relapse|relapsed|ibtr|local recurrence|locoregional recurrence)\b",
        "survival": r"\b(survival|os|overall survival|pfs|progression free|dfs|disease free)\b",
        "metastasis": r"\b(metastasis|metastatic|mets|distant metastasis|oligometastatic)\b",
        "margin": r"\b(margin|margins|positive margin|negative margin|close margin|surgical margin)\b",
        "risk_factor": r"\b(risk factor|risk of|elevated risk|increased risk|high risk|low risk)\b",
        "dose": r"\b(dose|gy|gray|fractionation|fraction|cgy)\b",
        "toxicity": r"\b(toxicity|side effect|adverse|complication)\b",
    }
    
    # Anatomic sites for radiation
    ANATOMIC_SITES = {
        "chest_wall": r"\b(chest wall|cw)\b",
        "regional_nodes": r"\b(regional nodes|regional lymph|nodal|axilla|axillary|supraclavicular|scv|internal mammary|imn)\b",
        "whole_breast": r"\b(whole breast|wbi|breast irradiation)\b",
        "partial_breast": r"\b(partial breast|pbi|apbi)\b",
        "brain": r"\b(brain|whole brain|wbrt|cranial|intracranial)\b",
        "spine": r"\b(spine|spinal|vertebral)\b",
        "pelvis": r"\b(pelvis|pelvic)\b",
        "para_aortic": r"\b(para-aortic|paraaortic|pa nodes|pa strip)\b",
    }
    
    # Treatment protocol hints for critical staging patterns
    TREATMENT_PROTOCOL_HINTS = {
        "inflammatory_breast_cancer": {
            "protocol": "trimodality",
            "components": ["neoadjuvant chemotherapy", "mastectomy", "PMRT"],
            "note": "IBC (T4d) requires trimodality: neoadjuvant chemo → modified radical mastectomy → PMRT. BCS is contraindicated.",
        },
        "locally_advanced_breast": {
            "protocol": "neoadjuvant",
            "components": ["neoadjuvant chemotherapy", "surgery", "adjuvant RT"],
            "note": "Locally advanced breast cancer typically requires neoadjuvant systemic therapy.",
        },
        "early_hpv_oropharynx": {
            "protocol": "definitive_rt",
            "components": ["definitive RT", "unilateral radiation"],
            "note": "Early-stage HPV+ oropharynx (T1-2N0-1) can be treated with definitive RT alone with excellent outcomes.",
        },
        "high_risk_prostate": {
            "protocol": "long_term_adt",
            "components": ["EBRT", "long-term ADT 28 months", "RTOG 92-02"],
            "note": "High-risk prostate requires long-term ADT (28 months per RTOG 92-02).",
        },
        "cervical_intermediate_risk": {
            "protocol": "adjuvant_ebrt",
            "components": ["EBRT", "pelvic radiation", "GOG 92"],
            "note": "Cervical cancer with intermediate risk factors (Sedlis criteria) benefits from adjuvant pelvic RT per GOG 92.",
        },
    }
    
    def __init__(self):
        """Initialize the extractor with compiled patterns."""
        self._compile_patterns()
        self._compile_critical_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self.cancer_type_re = {k: re.compile(v, re.IGNORECASE) for k, v in self.CANCER_TYPES.items()}
        self.cancer_subtype_re = {k: re.compile(v, re.IGNORECASE) for k, v in self.CANCER_SUBTYPES.items()}
        self.stage_re = [(re.compile(p, re.IGNORECASE), t) for p, t in self.STAGE_PATTERNS]
        self.biomarker_re = {k: re.compile(v, re.IGNORECASE) for k, v in self.BIOMARKERS.items()}
        self.treatment_re = {k: re.compile(v, re.IGNORECASE) for k, v in self.TREATMENTS.items()}
        self.concept_re = {k: re.compile(v, re.IGNORECASE) for k, v in self.CLINICAL_CONCEPTS.items()}
        self.site_re = {k: re.compile(v, re.IGNORECASE) for k, v in self.ANATOMIC_SITES.items()}
    
    def _compile_critical_patterns(self):
        """Compile critical staging patterns for protocol detection."""
        self.critical_staging_re = {}
        for pattern_name, patterns in self.CRITICAL_STAGING_PATTERNS.items():
            self.critical_staging_re[pattern_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def extract(self, query: str) -> ClinicalProfile:
        """
        Extract clinical entities from a query.
        
        Args:
            query: The medical query string
            
        Returns:
            ClinicalProfile with extracted entities
        """
        profile = ClinicalProfile()
        
        # Extract cancer type
        for cancer_type, pattern in self.cancer_type_re.items():
            if pattern.search(query):
                profile.cancer_type = cancer_type
                break
        
        # Extract cancer subtype
        for subtype, pattern in self.cancer_subtype_re.items():
            if pattern.search(query):
                profile.cancer_subtype = subtype
                break
        
        # Extract stage
        for pattern, stage_type in self.stage_re:
            match = pattern.search(query)
            if match:
                if stage_type == "tnm":
                    profile.tnm = match.group(0).upper()
                elif stage_type == "stage":
                    profile.stage = match.group(0)
                elif stage_type == "stage_desc":
                    profile.stage = match.group(1)
                break
        
        # Extract biomarkers
        for biomarker, pattern in self.biomarker_re.items():
            if pattern.search(query):
                profile.biomarkers.append(biomarker)
        
        # Extract treatments
        for treatment, pattern in self.treatment_re.items():
            if pattern.search(query):
                profile.treatments.append(treatment)
        
        # Extract clinical concepts
        for concept, pattern in self.concept_re.items():
            if pattern.search(query):
                profile.clinical_concepts.append(concept)
        
        # Extract anatomic sites
        for site, pattern in self.site_re.items():
            if pattern.search(query):
                profile.anatomic_sites.append(site)
        
        return profile
    
    def get_must_match_terms(self, profile: ClinicalProfile) -> List[str]:
        """
        Get terms that MUST be present in retrieved documents.
        
        These are high-specificity terms that define the clinical scenario.
        """
        must_match = []
        
        # Cancer type is usually required
        if profile.cancer_type:
            must_match.append(profile.cancer_type)
        
        # Specific biomarkers are important
        if profile.biomarkers:
            must_match.extend(profile.biomarkers)
        
        # Specific subtype
        if profile.cancer_subtype:
            must_match.append(profile.cancer_subtype)
        
        return must_match
    
    def get_should_match_terms(self, profile: ClinicalProfile) -> List[str]:
        """
        Get terms that SHOULD be present (boost if present).
        
        These are contextual terms that improve relevance.
        """
        should_match = []
        
        # Stage info
        if profile.stage:
            should_match.append(profile.stage)
        if profile.tnm:
            should_match.append(profile.tnm)
        
        # Treatments
        should_match.extend(profile.treatments)
        
        # Clinical concepts
        should_match.extend(profile.clinical_concepts)
        
        # Anatomic sites
        should_match.extend(profile.anatomic_sites)
        
        return should_match
    
    def detect_critical_staging(self, query: str) -> List[str]:
        """
        Detect critical staging patterns that imply specific treatment protocols.
        
        Args:
            query: The medical query string
            
        Returns:
            List of detected critical staging pattern names
        """
        detected = []
        for pattern_name, compiled_patterns in self.critical_staging_re.items():
            for pattern in compiled_patterns:
                if pattern.search(query):
                    detected.append(pattern_name)
                    break  # Only add each pattern type once
        return detected
    
    def get_treatment_protocol_hint(self, critical_staging: str) -> Optional[Dict[str, Any]]:
        """
        Get treatment protocol hint for a critical staging pattern.
        
        Args:
            critical_staging: Name of the critical staging pattern
            
        Returns:
            Dictionary with protocol info or None if not found
        """
        return self.TREATMENT_PROTOCOL_HINTS.get(critical_staging)
    
    def get_query_expansion_terms(self, query: str) -> List[str]:
        """
        Get additional query expansion terms based on detected critical staging.
        
        Args:
            query: The medical query string
            
        Returns:
            List of expansion terms to add to the query
        """
        expansion_terms = []
        critical_patterns = self.detect_critical_staging(query)
        
        for pattern in critical_patterns:
            hint = self.get_treatment_protocol_hint(pattern)
            if hint:
                expansion_terms.extend(hint.get("components", []))
        
        return expansion_terms


# Singleton instance
_extractor = None

def get_clinical_entity_extractor() -> ClinicalEntityExtractor:
    """Get singleton instance of the extractor."""
    global _extractor
    if _extractor is None:
        _extractor = ClinicalEntityExtractor()
    return _extractor


# Example usage
if __name__ == "__main__":
    extractor = ClinicalEntityExtractor()
    
    test_queries = [
        "A female with a cT3N1M0 ER/PR- Her2+ breast cancer receives neoadjuvant TCHP followed by mastectomy with sentinel lymph node biopsy and achieves a pCR. What adjuvant therapy is recommended?",
        "For DCIS, which feature is associated with an elevated risk of in-breast recurrence?",
        "A patient with metastatic NSCLC with four sites of bony metastasis, which describes an appropriate management strategy?",
        "What is the recommended RT technique for stage I testicular seminoma?",
        "A 55-year-old female with pT1cN1mi cM0 ER+ HER2- breast cancer and 21-gene recurrence score of 22",
    ]
    
    print("=" * 70)
    print("CLINICAL ENTITY EXTRACTION DEMO")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query[:80]}...")
        profile = extractor.extract(query)
        print(f"  Cancer Type: {profile.cancer_type}")
        print(f"  Subtype: {profile.cancer_subtype}")
        print(f"  Stage: {profile.stage}")
        print(f"  TNM: {profile.tnm}")
        print(f"  Biomarkers: {profile.biomarkers}")
        print(f"  Treatments: {profile.treatments}")
        print(f"  Concepts: {profile.clinical_concepts}")
        print(f"  Sites: {profile.anatomic_sites}")
        print(f"  Must Match: {extractor.get_must_match_terms(profile)}")
        print(f"  Should Match: {extractor.get_should_match_terms(profile)}")
