"""
Trial Registry for Landmark Radiation Oncology Trials

This module provides trial name detection and boosting for the RAG pipeline.
It handles:
- Exact trial name matching
- Alias matching (RTOG 9202 vs RTOG 92-02)
- Implied trial detection based on key terms
- Trial-specific boosting for retrieval

Key Features:
- Comprehensive registry of landmark trials
- Fuzzy matching for trial name variations
- Integration with retrieval pipeline for boosting
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TrialMatch:
    """Represents a detected trial match."""
    canonical_name: str
    match_type: str  # "exact", "alias", "implied"
    confidence: float
    cancer_type: str
    key_terms: List[str]
    matched_alias: Optional[str] = None
    pmid: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "cancer_type": self.cancer_type,
            "key_terms": self.key_terms,
            "matched_alias": self.matched_alias,
            "pmid": self.pmid,
        }


class TrialRegistry:
    """
    Registry of landmark radiation oncology trials with aliases.
    
    Usage:
        registry = TrialRegistry()
        trials = registry.detect_trial("What was the dose in FAST-Forward?")
        # Returns: [TrialMatch(canonical_name="FAST-Forward", confidence=1.0, ...)]
    """
    
    LANDMARK_TRIALS = {
        # ============================================
        # PROSTATE CANCER
        # ============================================
        "RTOG 92-02": {
            "aliases": ["RTOG 9202", "9202", "RTOG92-02"],
            "cancer_type": "Prostate",
            "key_terms": ["long-term androgen deprivation", "prostate", "ADT duration"],
            "pmid": "18636019"
        },
        "RTOG 94-13": {
            "aliases": ["RTOG 9413", "9413"],
            "cancer_type": "Prostate",
            "key_terms": ["whole pelvic", "prostate-only", "pelvic RT"],
            "pmid": "14581419"
        },
        "RTOG 05-34": {
            "aliases": ["RTOG 0534", "0534", "SPPORT"],
            "cancer_type": "Prostate",
            "key_terms": ["short-term ADT", "salvage", "post-prostatectomy"],
            "pmid": "26304882"
        },
        "RTOG 01-26": {
            "aliases": ["RTOG 0126", "0126"],
            "cancer_type": "Prostate",
            "key_terms": ["dose escalation", "79.2 Gy", "70.2 Gy", "intermediate risk"],
            "pmid": "27084661"
        },
        "STAMPEDE": {
            "aliases": ["STAMPEDE trial", "Systemic Therapy in Advancing"],
            "cancer_type": "Prostate",
            "key_terms": ["abiraterone", "docetaxel", "metastatic", "hormone-sensitive"],
            "pmid": "26215850"
        },
        "ASCENDE-RT": {
            "aliases": ["ASCENDE RT", "LDR-boost", "ASCENDE"],
            "cancer_type": "Prostate",
            "key_terms": ["brachytherapy boost", "dose escalation", "LDR"],
            "pmid": "27626365"
        },
        "FLAME": {
            "aliases": ["FLAME trial"],
            "cancer_type": "Prostate",
            "key_terms": ["focal boost", "95 Gy", "intraprostatic lesion"],
            "pmid": "33139241"
        },
        "CHHiP": {
            "aliases": ["CHHiP trial", "Conventional or Hypofractionated"],
            "cancer_type": "Prostate",
            "key_terms": ["hypofractionation", "60 Gy", "57 Gy", "prostate"],
            "pmid": "27084661"
        },
        "HYPO-RT-PC": {
            "aliases": ["HYPO-RT", "HYPO RT PC"],
            "cancer_type": "Prostate",
            "key_terms": ["ultra-hypofractionation", "42.7 Gy", "7 fractions"],
            "pmid": "27084661"
        },
        
        # ============================================
        # BREAST CANCER
        # ============================================
        "FAST-Forward": {
            "aliases": ["FAST Forward", "Fast-Forward", "UK FAST-Forward", "FAST FORWARD"],
            "cancer_type": "Breast",
            "key_terms": ["hypofractionation", "26 Gy", "5 fractions", "1 week"],
            "pmid": "32580883"
        },
        "START A": {
            "aliases": ["START-A", "Standardisation of Breast A", "STARTA"],
            "cancer_type": "Breast",
            "key_terms": ["39 Gy", "41.6 Gy", "13 fractions", "hypofractionation"],
            "pmid": "18639001"
        },
        "START B": {
            "aliases": ["START-B", "Standardisation of Breast B", "STARTB"],
            "cancer_type": "Breast",
            "key_terms": ["40 Gy", "15 fractions", "hypofractionation", "3 weeks"],
            "pmid": "18639001"
        },
        "IMPORT HIGH": {
            "aliases": ["IMPORT-HIGH", "Import High", "IMPORTHIGH"],
            "cancer_type": "Breast",
            "key_terms": ["simultaneous integrated boost", "breast conservation", "SIB"],
            "pmid": "32740262"
        },
        "IMPORT LOW": {
            "aliases": ["IMPORT-LOW", "Import Low", "IMPORTLOW"],
            "cancer_type": "Breast",
            "key_terms": ["partial breast", "reduced dose", "low risk"],
            "pmid": "28864552"
        },
        "NSABP B-17": {
            "aliases": ["NSABP-B17", "B-17", "B17"],
            "cancer_type": "Breast",
            "key_terms": ["DCIS", "lumpectomy", "radiation", "local recurrence"],
            "pmid": "8598792"
        },
        "NSABP B-24": {
            "aliases": ["NSABP-B24", "B-24", "B24"],
            "cancer_type": "Breast",
            "key_terms": ["DCIS", "tamoxifen", "radiation"],
            "pmid": "10561349"
        },
        "EORTC 10853": {
            "aliases": ["EORTC-10853", "10853"],
            "cancer_type": "Breast",
            "key_terms": ["DCIS", "breast conserving", "radiation", "recurrence"],
            "pmid": "16818906"
        },
        "Z0011": {
            "aliases": ["ACOSOG Z0011", "Z-0011", "ACOSOG-Z0011"],
            "cancer_type": "Breast",
            "key_terms": ["sentinel node", "axillary dissection", "micrometastasis", "SLNB"],
            "pmid": "21304082"
        },
        "MA.20": {
            "aliases": ["MA20", "MA-20", "NCIC MA.20"],
            "cancer_type": "Breast",
            "key_terms": ["regional nodal irradiation", "RNI", "node-positive"],
            "pmid": "25832824"
        },
        "EORTC 22922": {
            "aliases": ["EORTC-22922", "22922"],
            "cancer_type": "Breast",
            "key_terms": ["internal mammary", "medial supraclavicular", "nodal irradiation"],
            "pmid": "25832825"
        },
        "AMAROS": {
            "aliases": ["AMAROS trial", "EORTC AMAROS"],
            "cancer_type": "Breast",
            "key_terms": ["axillary RT", "axillary dissection", "sentinel node positive"],
            "pmid": "25304656"
        },
        "PRIME II": {
            "aliases": ["PRIME-II", "PRIME 2"],
            "cancer_type": "Breast",
            "key_terms": ["elderly", "omission of RT", "low risk", "65 years"],
            "pmid": "25637716"
        },
        "CALGB 9343": {
            "aliases": ["CALGB-9343", "9343"],
            "cancer_type": "Breast",
            "key_terms": ["elderly", "tamoxifen alone", "omission of RT", "70 years"],
            "pmid": "15337807"
        },
        
        # ============================================
        # LUNG CANCER
        # ============================================
        "RTOG 06-17": {
            "aliases": ["RTOG 0617", "0617"],
            "cancer_type": "Lung",
            "key_terms": ["74 Gy", "60 Gy", "locally advanced NSCLC", "dose escalation"],
            "pmid": "21555689"
        },
        "PACIFIC": {
            "aliases": ["PACIFIC trial", "durvalumab consolidation"],
            "cancer_type": "Lung",
            "key_terms": ["durvalumab", "immunotherapy", "stage III NSCLC", "consolidation"],
            "pmid": "29241097"
        },
        "RTOG 91-05": {
            "aliases": ["RTOG 9105", "9105"],
            "cancer_type": "Lung",
            "key_terms": ["CHART", "hyperfractionation", "NSCLC"],
            "pmid": "10561351"
        },
        "RTOG 73-01": {
            "aliases": ["RTOG 7301", "7301"],
            "cancer_type": "Lung",
            "key_terms": ["split course", "continuous", "NSCLC", "dose"],
            "pmid": "6997309"
        },
        "CONVERT": {
            "aliases": ["CONVERT trial"],
            "cancer_type": "Lung",
            "key_terms": ["SCLC", "twice daily", "once daily", "limited stage"],
            "pmid": "28262584"
        },
        
        # ============================================
        # HEAD & NECK CANCER
        # ============================================
        "RTOG 95-01": {
            "aliases": ["RTOG 9501", "9501"],
            "cancer_type": "Head and Neck",
            "key_terms": ["postoperative", "concurrent cisplatin", "high risk"],
            "pmid": "15184404"
        },
        "EORTC 22931": {
            "aliases": ["EORTC-22931", "22931"],
            "cancer_type": "Head and Neck",
            "key_terms": ["postoperative radiotherapy", "cisplatin", "adjuvant"],
            "pmid": "15184404"
        },
        "RTOG 01-29": {
            "aliases": ["RTOG 0129", "0129"],
            "cancer_type": "Head and Neck",
            "key_terms": ["accelerated fractionation", "standard fractionation", "concomitant boost"],
            "pmid": "16757720"
        },
        "RTOG 05-22": {
            "aliases": ["RTOG 0522", "0522"],
            "cancer_type": "Head and Neck",
            "key_terms": ["cetuximab", "cisplatin", "concurrent"],
            "pmid": "24569458"
        },
        "PARSPORT": {
            "aliases": ["PARSPORT trial"],
            "cancer_type": "Head and Neck",
            "key_terms": ["parotid sparing", "IMRT", "xerostomia"],
            "pmid": "20970214"
        },
        "De-ESCALaTE": {
            "aliases": ["De-ESCALaTE HPV", "DEESCALATE"],
            "cancer_type": "Head and Neck",
            "key_terms": ["HPV positive", "cetuximab", "cisplatin", "oropharynx"],
            "pmid": "30449625"
        },
        "Intergroup 0099": {
            "aliases": ["INT 0099", "INT-0099", "Intergroup-0099", "Al-Sarraf"],
            "cancer_type": "Nasopharyngeal",
            "key_terms": ["nasopharyngeal", "NPC", "concurrent chemoradiation", "adjuvant chemotherapy", "cisplatin", "5-FU"],
            "pmid": "9840525"
        },
        "RTOG 0225": {
            "aliases": ["RTOG-0225", "0225"],
            "cancer_type": "Nasopharyngeal",
            "key_terms": ["nasopharyngeal", "NPC", "IMRT", "locoregional control", "distant metastasis"],
            "pmid": "19720927"
        },
        "Garden 2004": {
            "aliases": ["Garden et al", "Garden study", "MD Anderson oropharynx"],
            "cancer_type": "Head and Neck",
            "key_terms": ["oropharynx", "tonsil", "T1", "T2", "unilateral", "ipsilateral", "definitive RT"],
            "pmid": "15337807"
        },
        
        # ============================================
        # GYN CANCER
        # ============================================
        "GOG 92": {
            "aliases": ["GOG-92", "Gynecologic Oncology Group 92"],
            "cancer_type": "Cervical",
            "key_terms": ["adjuvant radiation", "early-stage", "cervical", "intermediate risk"],
            "pmid": "10078488"
        },
        "GOG 120": {
            "aliases": ["GOG-120"],
            "cancer_type": "Cervical",
            "key_terms": ["concurrent cisplatin", "chemoradiation", "locally advanced"],
            "pmid": "10561337"
        },
        "PORTEC-1": {
            "aliases": ["PORTEC 1", "PORTEC I", "PORTEC1"],
            "cancer_type": "Endometrial",
            "key_terms": ["pelvic radiation", "intermediate risk", "endometrial"],
            "pmid": "10793176"
        },
        "PORTEC-2": {
            "aliases": ["PORTEC 2", "PORTEC II", "PORTEC2"],
            "cancer_type": "Endometrial",
            "key_terms": ["vaginal brachytherapy", "pelvic RT", "high-intermediate risk"],
            "pmid": "20813631"
        },
        "PORTEC-3": {
            "aliases": ["PORTEC 3", "PORTEC III", "PORTEC3"],
            "cancer_type": "Endometrial",
            "key_terms": ["chemoradiation", "high risk", "chemotherapy"],
            "pmid": "29449189"
        },
        "GOG 99": {
            "aliases": ["GOG-99"],
            "cancer_type": "Endometrial",
            "key_terms": ["adjuvant radiation", "intermediate risk", "endometrial"],
            "pmid": "15337807"
        },
        
        # ============================================
        # GI CANCER
        # ============================================
        "RTOG 98-11": {
            "aliases": ["RTOG 9811", "9811"],
            "cancer_type": "Anal",
            "key_terms": ["anal cancer", "5-FU", "mitomycin", "cisplatin"],
            "pmid": "18838666"
        },
        "ACT II": {
            "aliases": ["ACT-II", "ACT 2"],
            "cancer_type": "Anal",
            "key_terms": ["anal cancer", "maintenance", "cisplatin", "mitomycin"],
            "pmid": "23541725"
        },
        "German Rectal": {
            "aliases": ["CAO/ARO/AIO-94", "German Rectal Trial"],
            "cancer_type": "Rectal",
            "key_terms": ["preoperative", "postoperative", "chemoradiation", "rectal"],
            "pmid": "15496622"
        },
        "CROSS": {
            "aliases": ["CROSS trial"],
            "cancer_type": "Esophageal",
            "key_terms": ["neoadjuvant chemoradiation", "esophageal", "41.4 Gy"],
            "pmid": "22646630"
        },
        
        # ============================================
        # CNS
        # ============================================
        "RTOG 02-93": {
            "aliases": ["RTOG 0293", "0293"],
            "cancer_type": "CNS",
            "key_terms": ["glioblastoma", "temozolomide", "radiation"],
            "pmid": "15758009"
        },
        "EORTC 26981": {
            "aliases": ["EORTC-26981", "Stupp trial", "Stupp"],
            "cancer_type": "CNS",
            "key_terms": ["glioblastoma", "temozolomide", "stupp", "concurrent"],
            "pmid": "15758009"
        },
        "RTOG 98-02": {
            "aliases": ["RTOG 9802", "9802"],
            "cancer_type": "CNS",
            "key_terms": ["low-grade glioma", "PCV", "chemotherapy"],
            "pmid": "27050206"
        },
        
        # ============================================
        # TESTICULAR
        # ============================================
        "MRC TE10": {
            "aliases": ["TE10", "MRC-TE10"],
            "cancer_type": "Testicular",
            "key_terms": ["seminoma", "para-aortic", "dogleg", "stage I"],
            "pmid": "10561349"
        },
        "MRC TE18": {
            "aliases": ["TE18", "MRC-TE18"],
            "cancer_type": "Testicular",
            "key_terms": ["seminoma", "20 Gy", "30 Gy", "dose reduction"],
            "pmid": "15784156"
        },
    }
    
    def __init__(self):
        """Initialize the trial registry."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._trial_patterns = {}
        for trial_name, info in self.LANDMARK_TRIALS.items():
            # Create pattern for trial name and aliases
            all_names = [trial_name] + info.get("aliases", [])
            # Escape special regex characters and create pattern
            escaped = [re.escape(name) for name in all_names]
            pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
            self._trial_patterns[trial_name] = pattern
    
    def detect_trial(self, query: str) -> List[TrialMatch]:
        """
        Detect trial mentions in query with fuzzy matching.
        
        Args:
            query: The user's query string
            
        Returns:
            List of TrialMatch objects for detected trials
        """
        query_lower = query.lower()
        detected_trials = []
        
        for trial_name, trial_info in self.LANDMARK_TRIALS.items():
            # Check using compiled pattern
            pattern = self._trial_patterns[trial_name]
            match = pattern.search(query)
            
            if match:
                matched_text = match.group(1)
                # Determine if exact or alias match
                if matched_text.lower() == trial_name.lower():
                    match_type = "exact"
                    confidence = 1.0
                    matched_alias = None
                else:
                    match_type = "alias"
                    confidence = 0.95
                    matched_alias = matched_text
                
                detected_trials.append(TrialMatch(
                    canonical_name=trial_name,
                    match_type=match_type,
                    confidence=confidence,
                    cancer_type=trial_info["cancer_type"],
                    key_terms=trial_info["key_terms"],
                    matched_alias=matched_alias,
                    pmid=trial_info.get("pmid")
                ))
                continue
            
            # Check if query implies this trial (key terms match)
            if self._implies_trial(query_lower, trial_info):
                detected_trials.append(TrialMatch(
                    canonical_name=trial_name,
                    match_type="implied",
                    confidence=0.7,
                    cancer_type=trial_info["cancer_type"],
                    key_terms=trial_info["key_terms"],
                    pmid=trial_info.get("pmid")
                ))
        
        # Remove duplicates, keep highest confidence
        unique_trials = {}
        for trial in detected_trials:
            name = trial.canonical_name
            if name not in unique_trials or trial.confidence > unique_trials[name].confidence:
                unique_trials[name] = trial
        
        return list(unique_trials.values())
    
    def _implies_trial(self, query: str, trial_info: Dict) -> bool:
        """
        Check if query implies a specific trial based on key terms.
        Requires at least 2 key terms to match for implied detection.
        """
        matches = sum(1 for term in trial_info["key_terms"] if term.lower() in query)
        return matches >= 2
    
    def get_trial_boost_factor(self, detected_trials: List[TrialMatch]) -> float:
        """Get the boost factor based on detected trials."""
        if not detected_trials:
            return 1.0
        
        # Get highest confidence match
        max_confidence = max(t.confidence for t in detected_trials)
        
        if max_confidence >= 0.9:
            return 2.0  # Strong boost for exact/alias matches
        elif max_confidence >= 0.7:
            return 1.5  # Moderate boost for implied matches
        else:
            return 1.2  # Small boost
    
    def get_trial_names_for_filter(self, detected_trials: List[TrialMatch]) -> List[str]:
        """Get canonical trial names for filtering."""
        return [t.canonical_name for t in detected_trials if t.confidence >= 0.9]


# Singleton instance
_registry = None

def get_trial_registry() -> TrialRegistry:
    """Get singleton instance of the trial registry."""
    global _registry
    if _registry is None:
        _registry = TrialRegistry()
    return _registry



def boost_chunks_by_trial(
    chunks: List[Dict[str, Any]],
    detected_trials: List[TrialMatch]
) -> List[Dict[str, Any]]:
    """
    Boost chunks from detected trials.
    
    Args:
        chunks: List of retrieved chunks
        detected_trials: List of detected trial matches
        
    Returns:
        Chunks with boosted scores, sorted by score
    """
    if not detected_trials:
        return chunks
    
    trial_names = [t.canonical_name for t in detected_trials]
    trial_aliases = []
    trial_confidence = {}
    
    for trial in detected_trials:
        trial_confidence[trial.canonical_name] = trial.confidence
        aliases = TrialRegistry.LANDMARK_TRIALS.get(trial.canonical_name, {}).get("aliases", [])
        trial_aliases.extend(aliases)
    
    boosted_chunks = []
    for chunk in chunks:
        # Get text fields to search
        payload = chunk.get("payload", chunk)
        text = (payload.get("text") or "").lower()
        citation = (payload.get("doc_meta", {}).get("citation") or 
                   payload.get("citation") or "").lower()
        title = (payload.get("doc_meta", {}).get("title") or 
                payload.get("title") or "").lower()
        
        combined_text = f"{text} {citation} {title}"
        
        # Check for trial matches
        boost_factor = 1.0
        matched_trial = None
        
        for trial_name in trial_names:
            if trial_name.lower() in combined_text:
                confidence = trial_confidence.get(trial_name, 0.9)
                boost_factor = max(boost_factor, confidence * 2.0)
                matched_trial = trial_name
                break
            
            # Check aliases
            aliases = TrialRegistry.LANDMARK_TRIALS.get(trial_name, {}).get("aliases", [])
            for alias in aliases:
                if alias.lower() in combined_text:
                    confidence = trial_confidence.get(trial_name, 0.9)
                    boost_factor = max(boost_factor, confidence * 1.8)
                    matched_trial = trial_name
                    break
            
            if matched_trial:
                break
        
        # Apply boost
        original_score = chunk.get("score", chunk.get("score_rerank", chunk.get("score_fused", 0)))
        chunk["original_score"] = original_score
        chunk["score_trial_boost"] = original_score * boost_factor
        chunk["trial_boost_applied"] = boost_factor > 1.0
        chunk["matched_trial"] = matched_trial
        
        boosted_chunks.append(chunk)
    
    # Sort by boosted score
    boosted_chunks.sort(key=lambda x: x.get("score_trial_boost", 0), reverse=True)
    
    return boosted_chunks


# Example usage
if __name__ == "__main__":
    registry = TrialRegistry()
    
    test_queries = [
        "What was the dose used in FAST-Forward trial?",
        "What were the results of RTOG 9202?",
        "Is 26 Gy in 5 fractions safe for breast hypofractionation?",
        "Compare RTOG 0617 and PACIFIC for stage III NSCLC",
        "What is the role of vaginal brachytherapy vs pelvic RT in endometrial cancer?",
        "What is the recommended RT technique for stage I testicular seminoma?",
    ]
    
    print("=" * 70)
    print("TRIAL REGISTRY DETECTION DEMO")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        trials = registry.detect_trial(query)
        if trials:
            for t in trials:
                print(f"  - {t.canonical_name} ({t.match_type}, conf={t.confidence:.2f})")
                if t.matched_alias:
                    print(f"    Matched alias: {t.matched_alias}")
        else:
            print("  No trials detected")
