"""
Module Classifier Service

Classifies user queries into one of three modules:
1. GENERAL_KNOWLEDGE - Factual Q&A with ground truth answers
2. PATIENT_SPECIFIC - Personalized guidance for a patient case
3. EVIDENCE_EXPLORATION - Comparative analysis and option exploration

This classification drives:
- Retrieval strategy (what to prioritize)
- Generation format (how to structure the response)
- Follow-up suggestions (module transfer prompts)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class QueryModule(Enum):
    """The three core query modules."""
    GENERAL_KNOWLEDGE = "general_knowledge"
    PATIENT_SPECIFIC = "patient_specific"
    EVIDENCE_EXPLORATION = "evidence_exploration"


@dataclass
class ModuleClassification:
    """Result of module classification."""
    module: QueryModule
    confidence: float  # 0-1
    signals_matched: List[str]
    has_patient_context: bool
    has_explicit_question: bool
    has_superlative: bool
    suggested_follow_ups: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module.value,
            "confidence": self.confidence,
            "signals_matched": self.signals_matched,
            "has_patient_context": self.has_patient_context,
            "has_explicit_question": self.has_explicit_question,
            "has_superlative": self.has_superlative,
            "suggested_follow_ups": self.suggested_follow_ups,
        }


# ============================================
# DETECTION PATTERNS
# ============================================

# General Knowledge: Direct questions with ground truth answers
# Key insight: These are questions that have a DEFINITIVE answer (dose, trial results, guidelines)
GENERAL_KNOWLEDGE_PATTERNS = {
    "direct_question": [
        r"^what is the (?:dose|standard|guideline|recommendation)",
        r"^what (?:dose|treatment|therapy|regimen) (?:is|for)",
        r"^how many (?:gy|gray|fractions|fx)",
        r"^what did .* (?:trial|study) show",
        r"^what (?:are|were) the results",
        r"^what is the mechanism",
        r"^how does .* work",
        r"^what are the side effects",
        r"^what is the (?:5|10|15)[- ]?year",
        r"^what is the standard of care",
        r"^what is the role of",
        r"^when is .* indicated",  # Simple indication questions without patient context
        r"^is .* (?:indicated|recommended|appropriate)",
    ],
    # NEW: Ground truth question patterns - questions with definitive answers
    "ground_truth_question": [
        r"^what is the (?:gold standard|first[- ]line|standard)",
        r"^what is the (?:recommended|appropriate|typical) (?:dose|fractionation|schedule)",
        r"^how (?:much|many|long)",
        r"^what (?:percentage|proportion|rate)",
        r"^what is the (?:survival|response|control) rate",
        r"^what is the (?:median|mean|average)",
        r"^what are the (?:criteria|requirements|indications) for",
        r"^define\b",
        r"^explain\b",
    ],
    "guideline_reference": [
        r"\bnccn\b",
        r"\bastro\b",
        r"\basco\b",
        r"\bguideline[s]?\b",
        r"\bcategory [12]\b",
        r"\blevel [iI1] evidence\b",
    ],
    "trial_reference": [
        r"\b(?:rtog|nrg|nsabp|acosog|eortc|portec|z\d+|keynote|checkmate)\b",
        r"\bphase [123iI]+\b",
        r"\brandomized\b",
    ],
}

# Patient-Specific: Patient case with demographics/diagnosis
PATIENT_CONTEXT_PATTERNS = {
    "demographics": [
        r"\d{1,3}\s*(?:year|yr|y/?o)[s\s-]*old",
        r"\b(?:male|female|man|woman|gentleman|lady)\b",
        r"\becog\s*(?:ps\s*)?\d\b",
        r"\bkps\s*\d{2,3}\b",
    ],
    "diagnosis": [
        r"\bdiagnosed with\b",
        r"\bpresents with\b",
        r"\bhistory of\b",
        r"\bs/p\b",
        r"\bstatus post\b",
        r"\bpmh\b",
    ],
    "staging": [
        r"\b[cyp]?t[0-4][a-d]?(?:is)?\s*[cyp]?n[0-3][a-c]?",
        r"\bstage\s*[iI1234]+[a-cA-C]?\b",
        r"\bmetastatic\b",
        r"\blocally advanced\b",
    ],
    "biomarkers": [
        r"\ber[+-]\b",
        r"\bpr[+-]\b",
        r"\bher2[+-]?\b",
        r"\btriple negative\b",
        r"\btnbc\b",
        r"\bkras\b",
        r"\begfr\b",
        r"\balk\b",
        r"\bpd-?l1\b",
        r"\bmsi[- ]?h\b",
    ],
    "pathology": [
        r"\bdoi\b",
        r"\blvi[+-]?\b",
        r"\bpni[+-]?\b",
        r"\bmargin[s]?\b",
        r"\bgrade\s*[123iI]+\b",
        r"\bpoorly differentiated\b",
        r"\bwell differentiated\b",
    ],
    "implicit_guidance": [
        r"for (?:this|my|the) patient",
        r"given (?:his|her|their) (?:history|stage|diagnosis|case)",
        r"in this case",
        r"what would you recommend",
        r"how should (?:we|i) (?:treat|manage|proceed)",
    ],
}

# Evidence Exploration: Comparative/evaluative questions
# Key insight: These questions seek to COMPARE options or find the "best" - no single answer
EVIDENCE_EXPLORATION_PATTERNS = {
    "superlatives": [
        r"\b(?:the )?best\b",
        r"\bmost (?:appropriate|effective|optimal|suitable)\b",
        r"\bpreferred\b",
        r"\boptimal\b",
        r"\bideal\b",
    ],
    "comparison": [
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
        r"\bwhich is (?:better|superior|preferred)\b",
        r"\bhow do .* compare\b",
    ],
    "exploration": [
        r"\bwhat are the (?:options|alternatives|choices)\b",
        r"\bwhat (?:are|is) (?:the )?(?:treatment )?options?\b",  # NEW: catch "what are the options"
        r"\bexplore\b",
        r"\bwhat if\b",
        r"\bpros and cons\b",
        r"\bdifferent (?:approaches|strategies|treatments|options)\b",
        r"\bavailable (?:treatments|options|therapies)\b",
        r"\boptions for\b",  # NEW: catch "options for X"
    ],
    # NEW: Outcome-seeking patterns - questions referencing comparative outcomes
    "outcome_seeking": [
        r"\bbest (?:outcome|survival|control|response)\b",
        r"\bhighest (?:survival|response|control)\b",
        r"\blowest (?:toxicity|recurrence|mortality)\b",
        r"\bwhich .* had (?:the )?(?:best|better|highest|lowest)\b",
        # NEW: Added outcome comparison patterns
        r"\bwhich (?:treatment|therapy|regimen|approach) (?:has|had|shows|showed)",
        r"\bwhat (?:treatment|therapy|regimen) (?:results|resulted) in",
        r"\bsuperior (?:outcome|survival|response|control)\b",
        r"\binferior (?:outcome|survival|response|control)\b",
    ],
    # NEW: Evaluative language patterns
    "evaluative": [
        r"\bweigh(?:ing)? (?:the )?(?:options|risks|benefits)\b",
        r"\btrade[- ]?off\b",
        r"\brisk[- ]?benefit\b",
        r"\badvantage[s]? (?:of|and|or) disadvantage\b",
        r"\bstrength[s]? (?:of|and|or) weakness\b",
    ],
}

# Negative signals - patterns that indicate NOT a certain module
NEGATIVE_SIGNALS = {
    "general_knowledge": [
        # If patient context is present, not general knowledge
        r"\d{1,3}\s*(?:year|yr|y/?o)",
        r"\b(?:male|female|man|woman)\b.*\b(?:with|diagnosed|presents)\b",
    ],
    "patient_specific": [
        # If asking for comparison/options, not patient-specific
        r"\bcompare\b",
        r"\bwhat are the options\b",
        r"\balternatives\b",
    ],
}

# Question markers - detect if explicit question is present
QUESTION_MARKERS = re.compile(
    r'(\?|'
    r'^what\s|^how\s|^why\s|^when\s|^where\s|^which\s|^who\s|'
    r'^is\s|^are\s|^can\s|^could\s|^should\s|^would\s|^will\s|'
    r'^do\s|^does\s|^did\s|^has\s|^have\s)',
    re.IGNORECASE | re.MULTILINE
)


def _count_pattern_matches(text: str, patterns: Dict[str, List[str]]) -> Tuple[int, List[str]]:
    """Count how many patterns match and return matched signal names."""
    text_lower = text.lower()
    total_matches = 0
    matched_signals = []
    
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text_lower, re.IGNORECASE):
                total_matches += 1
                matched_signals.append(f"{category}:{pattern[:30]}")
                break  # Only count one match per category
    
    return total_matches, matched_signals


def _has_superlative(text: str) -> bool:
    """Check if query contains superlative/comparative language."""
    text_lower = text.lower()
    for pattern in EVIDENCE_EXPLORATION_PATTERNS["superlatives"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    for pattern in EVIDENCE_EXPLORATION_PATTERNS["outcome_seeking"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    # NEW: Also check evaluative patterns
    for pattern in EVIDENCE_EXPLORATION_PATTERNS.get("evaluative", []):
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _has_ground_truth_question(text: str) -> bool:
    """
    Check if query is asking for a ground truth / definitive answer.
    
    Ground truth questions have a single correct answer (dose, trial result, guideline).
    This is distinct from exploration questions that seek to compare options.
    """
    text_lower = text.lower()
    
    # Check direct question patterns
    for pattern in GENERAL_KNOWLEDGE_PATTERNS.get("direct_question", []):
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check ground truth question patterns
    for pattern in GENERAL_KNOWLEDGE_PATTERNS.get("ground_truth_question", []):
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check guideline/trial references (these typically have definitive answers)
    for pattern in GENERAL_KNOWLEDGE_PATTERNS.get("guideline_reference", []):
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    for pattern in GENERAL_KNOWLEDGE_PATTERNS.get("trial_reference", []):
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def _has_explicit_question(text: str) -> bool:
    """Check if query contains an explicit question."""
    return bool(QUESTION_MARKERS.search(text))


def _has_patient_context(text: str) -> bool:
    """Check if query contains patient-specific information."""
    text_lower = text.lower()
    
    # Check for demographics
    for pattern in PATIENT_CONTEXT_PATTERNS["demographics"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check for diagnosis + staging combination
    has_diagnosis = any(
        re.search(p, text_lower, re.IGNORECASE) 
        for p in PATIENT_CONTEXT_PATTERNS["diagnosis"]
    )
    has_staging = any(
        re.search(p, text_lower, re.IGNORECASE) 
        for p in PATIENT_CONTEXT_PATTERNS["staging"]
    )
    
    if has_diagnosis and has_staging:
        return True
    
    # Check for biomarkers (strong signal)
    for pattern in PATIENT_CONTEXT_PATTERNS["biomarkers"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check for implicit guidance phrases
    for pattern in PATIENT_CONTEXT_PATTERNS["implicit_guidance"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def classify_module(query: str, query_structure: Optional[Dict] = None) -> ModuleClassification:
    """
    Classify a query into one of the three modules.
    
    Classification logic (ENHANCED):
    1. If superlative/comparative/outcome-seeking language -> EVIDENCE_EXPLORATION
    2. If patient context + no explicit question -> PATIENT_SPECIFIC (implicit "what should we do?")
    3. If patient context + explicit question asking for comparison -> EVIDENCE_EXPLORATION
    4. If patient context + explicit question (not comparison) -> PATIENT_SPECIFIC
    5. If direct question with ground truth answer + no patient context -> GENERAL_KNOWLEDGE
    6. Default based on strongest signal match
    
    Key insight from user:
    - GENERAL_KNOWLEDGE: Questions with definitive/ground truth answers
    - PATIENT_SPECIFIC: Patient info provided without specific question = implicit guidance request
    - EVIDENCE_EXPLORATION: Superlatives, comparisons, "best outcome" type questions
    
    Args:
        query: The user's query string
        query_structure: Optional pre-computed query structure dict
        
    Returns:
        ModuleClassification with module, confidence, and signals
    """
    query_lower = query.lower()
    
    # Detect key characteristics
    has_patient = _has_patient_context(query)
    has_question = _has_explicit_question(query)
    has_superlative = _has_superlative(query)
    
    # NEW: Detect if question has a ground truth answer (definitive, not comparative)
    has_ground_truth_question = _has_ground_truth_question(query)
    
    # Count pattern matches for each module
    general_matches, general_signals = _count_pattern_matches(query, GENERAL_KNOWLEDGE_PATTERNS)
    patient_matches, patient_signals = _count_pattern_matches(query, PATIENT_CONTEXT_PATTERNS)
    exploration_matches, exploration_signals = _count_pattern_matches(query, EVIDENCE_EXPLORATION_PATTERNS)
    
    # Use query_structure if provided for additional context
    if query_structure and isinstance(query_structure, dict):
        if query_structure.get("has_patient_context"):
            has_patient = True
            patient_matches += 2
        cancer = query_structure.get("cancer")
        if isinstance(cancer, dict) and cancer.get("site"):
            patient_matches += 1
    
    # Classification decision tree (ENHANCED)
    module = QueryModule.GENERAL_KNOWLEDGE
    confidence = 0.5
    signals = []
    
    # Rule 1: Superlative/comparative/outcome-seeking language -> EVIDENCE_EXPLORATION
    # This takes priority because "best treatment for X" is exploration even with patient context
    if has_superlative or exploration_matches >= 2:
        module = QueryModule.EVIDENCE_EXPLORATION
        confidence = 0.7 + (exploration_matches * 0.1)
        signals = exploration_signals
        if has_superlative:
            signals.append("superlative_detected")
    
    # Rule 1b: Exploration patterns (options, alternatives, compare) -> EVIDENCE_EXPLORATION
    # Even a single strong exploration signal should trigger this module
    elif exploration_matches >= 1 and not has_patient:
        # Check if it's a strong exploration signal (options, compare, alternatives)
        strong_exploration = any(
            sig.startswith(("exploration:", "comparison:")) 
            for sig in exploration_signals
        )
        if strong_exploration:
            module = QueryModule.EVIDENCE_EXPLORATION
            confidence = 0.7 + (exploration_matches * 0.1)
            signals = exploration_signals
            signals.append("exploration_pattern_detected")
    
    # Rule 2: Patient context + NO explicit question -> PATIENT_SPECIFIC
    # Require ≥2 patient context pattern matches to avoid triggering on single "patient" word
    if module == QueryModule.GENERAL_KNOWLEDGE and has_patient and not has_question and patient_matches >= 2:
        module = QueryModule.PATIENT_SPECIFIC
        confidence = 0.85 + (patient_matches * 0.03)  # High confidence for implicit guidance
        signals = patient_signals
        signals.append("patient_context_no_question_implicit_guidance")
    
    # Rule 3: Patient context + explicit question
    elif module == QueryModule.GENERAL_KNOWLEDGE and has_patient and has_question:
        # If asking about options/comparison -> EVIDENCE_EXPLORATION
        if exploration_matches >= 1:
            module = QueryModule.EVIDENCE_EXPLORATION
            confidence = 0.7
            signals = exploration_signals + patient_signals[:2]
            signals.append("patient_with_comparison_question")
        # Otherwise -> PATIENT_SPECIFIC (asking about their specific case)
        else:
            module = QueryModule.PATIENT_SPECIFIC
            confidence = 0.75
            signals = patient_signals
            signals.append("patient_context_with_question")
    
    # Rule 4: Direct question with ground truth answer + no patient context -> GENERAL_KNOWLEDGE
    elif module == QueryModule.GENERAL_KNOWLEDGE and has_question and not has_patient:
        # Check if it's a ground truth question (definitive answer exists)
        if has_ground_truth_question or general_matches >= 1:
            module = QueryModule.GENERAL_KNOWLEDGE
            confidence = 0.8 + (general_matches * 0.05)
            signals = general_signals
            signals.append("ground_truth_question")
        else:
            # Generic question without clear ground truth - still general knowledge
            module = QueryModule.GENERAL_KNOWLEDGE
            confidence = 0.65
            signals = general_signals if general_signals else ["generic_question"]
    
    # Rule 5: Default based on strongest match
    elif module == QueryModule.GENERAL_KNOWLEDGE:
        if patient_matches > general_matches and patient_matches > exploration_matches:
            module = QueryModule.PATIENT_SPECIFIC
            confidence = 0.6
            signals = patient_signals
        elif exploration_matches > general_matches:
            module = QueryModule.EVIDENCE_EXPLORATION
            confidence = 0.6
            signals = exploration_signals
        else:
            module = QueryModule.GENERAL_KNOWLEDGE
            confidence = 0.6
            signals = general_signals if general_signals else ["default"]
    
    # Cap confidence at 0.95
    confidence = min(confidence, 0.95)

    import logging
    _logger = logging.getLogger(__name__)
    _logger.debug(
        f"[classify_module] module={module.value} confidence={confidence:.2f} "
        f"has_patient={has_patient} has_question={has_question} "
        f"patient_matches={patient_matches} signals={signals[:3]}"
    )

    # Generate follow-up suggestions based on module
    follow_ups = _generate_follow_ups(module, has_patient, query)

    return ModuleClassification(
        module=module,
        confidence=confidence,
        signals_matched=signals[:5],  # Limit to top 5 signals
        has_patient_context=has_patient,
        has_explicit_question=has_question,
        has_superlative=has_superlative,
        suggested_follow_ups=follow_ups,
    )


def _generate_follow_ups(module: QueryModule, has_patient: bool, query: str = "") -> List[str]:
    """Generate suggested follow-up questions based on module.
    
    Follow-ups prefixed with [NAV:page] are cross-module navigation targets.
    The frontend strips the prefix and navigates to the target page.
    Follow-ups prefixed with [REPORT:format] trigger PDF report generation.
    Follow-ups without a prefix are sent as in-chat conversation queries.
    """
    query_lower = query.lower()
    
    # Detect if patient handout or clinic note would be relevant
    _has_treatment_options = bool(re.search(
        r'\b(option|treatment|compare|versus|vs\.?|choice|alternative)\b', query_lower
    ))
    _has_outcome_language = bool(re.search(
        r'\b(outcome|survival|side effect|toxicity|efficacy|response|which|what|recommend)\b', query_lower
    ))
    _has_workup_language = bool(re.search(
        r'\b(next step|work[\s-]?up|evaluation|assessment|diagnostic|mpmri|psad|biopsy|imaging)\b', query_lower
    ))
    
    offer_handout = _has_treatment_options and _has_outcome_language
    offer_clinic_note = _has_workup_language and has_patient
    
    if module == QueryModule.GENERAL_KNOWLEDGE:
        follow_ups = []
        if offer_handout:
            follow_ups.append("[REPORT:patient_handout]Generate Patient Handout (PDF)")
        return follow_ups
    
    elif module == QueryModule.PATIENT_SPECIFIC:
        follow_ups = []
        if offer_clinic_note:
            follow_ups.append("[REPORT:clinic_note]Generate Clinic Note (PDF)")
        elif offer_handout:
            follow_ups.append("[REPORT:patient_handout]Generate Patient Handout (PDF)")
        return follow_ups
    
    elif module == QueryModule.EVIDENCE_EXPLORATION:
        follow_ups = []
        if offer_handout:
            follow_ups.append("[REPORT:patient_handout]Generate Patient Handout (PDF)")
        return follow_ups
    
    return []


# ============================================
# MODULE-SPECIFIC GENERATION PROMPTS
# ============================================
# NOTE: Actual prompts are defined in module_generation_prompts.py
# This section is kept for backward compatibility - use get_module_prompt()
# which delegates to module_generation_prompts.py


def get_module_prompt(module: QueryModule) -> Dict[str, str]:
    """Get the generation prompt configuration for a module.
    
    Delegates to module_generation_prompts.py for the actual prompts.
    """
    from src.api.services.module_generation_prompts import get_prompt_for_module
    return get_prompt_for_module(module.value)


# ============================================
# RETRIEVAL STRATEGY BY MODULE
# ============================================

MODULE_RETRIEVAL_STRATEGIES = {
    QueryModule.GENERAL_KNOWLEDGE: {
        "description": "Prioritize guidelines, landmark trials, and dose tables for factual Q&A",
        "boost_patterns": {
            "guideline": 1.5,       # NCCN, ASTRO, ASCO guidelines - highest priority
            "landmark_trial": 1.4,  # Practice-changing trials (RTOG, NRG, etc.)
            "dose_table": 1.3,      # Structured dose information
            "meta_analysis": 1.25,  # Systematic reviews
            "phase_3": 1.2,         # Phase III trials
        },
        "chunk_selection": "top_k",  # Select top K most relevant
        "k": 8,
        "prioritize_sections": ["methods", "results", "conclusions", "guidelines", "recommendations"],
    },
    
    QueryModule.PATIENT_SPECIFIC: {
        "description": "Match similar patient populations in studies for personalized guidance",
        "boost_patterns": {
            "population_match": 1.5,  # Studies with similar patient characteristics
            "staging_match": 1.4,     # Studies with matching stage
            "biomarker_match": 1.4,   # Studies with matching biomarkers
            "eligibility": 1.3,       # Eligibility criteria sections
            "patient_characteristics": 1.3,  # Patient demographics sections
        },
        "chunk_selection": "diverse",  # Select diverse chunks covering different aspects
        "k": 10,
        "prioritize_sections": ["patient_characteristics", "eligibility", "methods", "results"],
        # NEW: Use whatever patient info is provided - flexible matching
        "flexible_matching": True,
    },
    
    QueryModule.EVIDENCE_EXPLORATION: {
        "description": "Retrieve multiple treatment arms and comparative data for option comparison",
        "boost_patterns": {
            "comparative_trial": 1.5,  # Head-to-head comparisons
            "treatment_arm": 1.4,      # Different treatment approaches
            "outcome_data": 1.3,       # Survival, response data
            "toxicity_data": 1.25,     # Adverse effects for trade-off analysis
            "meta_analysis": 1.3,      # Systematic reviews comparing options
        },
        "chunk_selection": "grouped",  # Group by treatment approach
        "k": 15,  # More chunks to cover multiple options
        "prioritize_sections": ["results", "outcomes", "toxicity", "conclusions", "comparison"],
        # NEW: Ensure diverse treatment options are represented
        "ensure_diversity": True,
        "min_treatment_options": 2,
    },
}


def get_retrieval_strategy(module: QueryModule) -> Dict[str, Any]:
    """Get the retrieval strategy configuration for a module."""
    return MODULE_RETRIEVAL_STRATEGIES.get(module, MODULE_RETRIEVAL_STRATEGIES[QueryModule.GENERAL_KNOWLEDGE])


# ============================================
# SINGLETON / CONVENIENCE FUNCTIONS
# ============================================

def classify_query_module(query: str, query_structure: Optional[Dict] = None) -> ModuleClassification:
    """
    Main entry point for module classification.
    
    Args:
        query: User's query string
        query_structure: Optional pre-computed query structure
        
    Returns:
        ModuleClassification result
    """
    return classify_module(query, query_structure)


def get_module_config(module: QueryModule) -> Dict[str, Any]:
    """Get complete configuration for a module (prompt + retrieval strategy)."""
    return {
        "prompt": get_module_prompt(module),
        "retrieval": get_retrieval_strategy(module),
    }
