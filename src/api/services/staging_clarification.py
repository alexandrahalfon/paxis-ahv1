"""
Staging Clarification Generator

Generates human-readable clarification questions when stage inference
is ambiguous and additional factors are needed.

This module is GENERIC — it works for any cancer type by mapping
required_factors (from StageInferenceResult) to question templates.

Integration points:
- query_intent_service.py: Inject clarification FollowUpOptions
- enhanced_rag_service.py: Add staging ambiguity to RAG prompt
- Frontend: Display as clickable follow-up suggestions

Usage:
    from src.api.services.staging_clarification import generate_staging_clarifications

    clarifications = generate_staging_clarifications(
        stage_result=inference_result,
        patient_context={"cancer_type": "breast", "tnm_t": "T2", ...}
    )
    # Returns list of ClarificationQuestion objects
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClarificationQuestion:
    """A single clarification question to resolve staging ambiguity."""
    
    # The missing factor this resolves (e.g., "hpv_status", "grade", "age")
    factor: str
    
    # Human-readable question shown to the user
    question: str
    
    # Possible answer options (if finite/enumerable)
    options: List[str] = field(default_factory=list)
    
    # Query template: when user picks an option, this template is filled
    # and appended to their original query for re-processing
    query_template: str = ""
    
    # Clinical context for why this matters
    rationale: str = ""
    
    # Priority (higher = ask first). Based on clinical impact.
    priority: int = 0


@dataclass
class StagingClarification:
    """Complete clarification result for a staging ambiguity."""
    
    # The questions to ask
    questions: List[ClarificationQuestion] = field(default_factory=list)
    
    # Human-readable summary of the ambiguity (for RAG prompt injection)
    ambiguity_summary: str = ""
    
    # Formatted prompt addition for the LLM
    prompt_addition: str = ""
    
    # Whether any clarifications are needed
    needs_clarification: bool = False


# =============================================================================
# FACTOR -> QUESTION MAPPING
# =============================================================================
# This is the core mapping. Each factor that can appear in 
# StageInferenceResult.required_factors maps to a question definition.
# 
# To add support for a new factor, simply add an entry here.
# No code changes needed elsewhere.

FACTOR_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    
    # === HPV / p16 (Oropharynx) ===
    "hpv_status": {
        "question": "What is the HPV/p16 status?",
        "options": ["HPV+/p16+", "HPV-/p16-", "Unknown/not tested"],
        "query_template": "The patient is {answer}",
        "rationale": "HPV status is critical for oropharyngeal staging — HPV+ cancers use a different, more favorable staging system (AJCC 8th Ed).",
        "priority": 10,
    },
    "p16_status": {
        # Alias for hpv_status — deduplicated in generation
        "alias_of": "hpv_status",
    },
    
    # === Age (Thyroid) ===
    "age": {
        "question": "What is the patient's age?",
        "options": ["Under 55", "55 or older"],
        "query_template": "The patient is {answer} years old",
        "rationale": "For differentiated thyroid cancer, patients under 55 can only be Stage I (M0) or Stage II (M1). The full staging system applies only to patients 55 and older.",
        "priority": 10,
    },
    
    # === Grade (Breast, Esophageal, others) ===
    "grade": {
        "question": "What is the tumor grade?",
        "options": ["Grade 1 (well differentiated)", "Grade 2 (moderately differentiated)", "Grade 3 (poorly differentiated)"],
        "query_template": "The tumor is {answer}",
        "rationale": "Tumor grade affects prognostic staging and may shift the stage group up or down from the anatomic stage.",
        "priority": 8,
    },
    
    # === Hormone Receptors (Breast) ===
    "er_status": {
        "question": "What is the ER (estrogen receptor) status?",
        "options": ["ER positive", "ER negative"],
        "query_template": "The tumor is {answer}",
        "rationale": "ER status is required for breast cancer prognostic staging (AJCC 8th Ed). ER+ tumors may be down-staged.",
        "priority": 9,
    },
    "pr_status": {
        "question": "What is the PR (progesterone receptor) status?",
        "options": ["PR positive", "PR negative"],
        "query_template": "The tumor is {answer}",
        "rationale": "PR status contributes to breast cancer prognostic staging alongside ER and HER2.",
        "priority": 7,
    },
    "her2_status": {
        "question": "What is the HER2 status?",
        "options": ["HER2 positive", "HER2 negative", "HER2 equivocal"],
        "query_template": "The tumor is {answer}",
        "rationale": "HER2 status is required for breast cancer prognostic staging. HER2+ tumors have different prognostic staging than HER2- tumors.",
        "priority": 9,
    },
    
    # === PSA / Gleason (Prostate) ===
    "psa": {
        "question": "What is the PSA level?",
        "options": ["PSA < 10 ng/mL", "PSA 10-20 ng/mL", "PSA > 20 ng/mL"],
        "query_template": "PSA is {answer}",
        "rationale": "PSA level is required for prostate cancer prognostic staging. PSA ≥20 can shift T1-T2 tumors from Stage II to Stage IIIA.",
        "priority": 9,
    },
    "grade_group": {
        "question": "What is the Grade Group (Gleason)?",
        "options": ["Grade Group 1 (Gleason ≤6)", "Grade Group 2 (Gleason 3+4=7)", "Grade Group 3 (Gleason 4+3=7)", "Grade Group 4 (Gleason 8)", "Grade Group 5 (Gleason 9-10)"],
        "query_template": "Gleason score is {answer}",
        "rationale": "Grade Group is required for prostate cancer prognostic staging. Grade Group 5 shifts any T-stage to Stage IIIC.",
        "priority": 9,
    },
    "gleason_score": {
        # Alias for grade_group
        "alias_of": "grade_group",
    },
    
    # === Serum Markers (Testicular) ===
    "serum_markers": {
        "question": "What are the serum tumor marker levels (AFP, hCG, LDH)?",
        "options": ["S0 (normal)", "S1 (mildly elevated)", "S2 (moderately elevated)", "S3 (highly elevated)"],
        "query_template": "Serum markers are {answer}",
        "rationale": "Serum tumor markers determine sub-staging for Stage III testicular cancer (IIIA vs IIIB vs IIIC).",
        "priority": 8,
    },
    
    # === Oncotype (Breast) ===
    "oncotype_score": {
        "question": "What is the Oncotype DX recurrence score?",
        "options": ["Low risk (score < 18)", "Intermediate risk (score 18-30)", "High risk (score ≥ 31)"],
        "query_template": "Oncotype DX score is {answer}",
        "rationale": "Oncotype DX score can influence prognostic staging and treatment decisions for ER+ breast cancer.",
        "priority": 6,
    },
    
    # === Location (Esophageal) ===
    "location": {
        "question": "Where in the esophagus is the tumor located?",
        "options": ["Upper esophagus", "Middle esophagus", "Lower esophagus / GEJ"],
        "query_template": "The tumor is located in the {answer}",
        "rationale": "Tumor location affects prognostic staging for esophageal cancer.",
        "priority": 5,
    },
}


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_staging_clarifications(
    required_factors: List[str],
    possible_stages: List[str],
    inference_notes: List[str],
    patient_context: Optional[Dict[str, Any]] = None,
) -> StagingClarification:
    """
    Generate clarification questions for ambiguous staging.
    
    Args:
        required_factors: List of factor keys from StageInferenceResult
                         (e.g., ["hpv_status", "p16_status"])
        possible_stages: List of possible stage strings
                        (e.g., ["Stage I (HPV+)", "Stage III (HPV-)"])
        inference_notes: Notes from the inference engine
        patient_context: Dict with known patient data (to avoid asking
                        about things we already know)
    
    Returns:
        StagingClarification with questions and prompt additions
    """
    if not required_factors:
        return StagingClarification(needs_clarification=False)
    
    patient_context = patient_context or {}
    questions = []
    seen_factors = set()
    
    for factor in required_factors:
        # Resolve aliases
        defn = FACTOR_DEFINITIONS.get(factor, {})
        if "alias_of" in defn:
            factor = defn["alias_of"]
            defn = FACTOR_DEFINITIONS.get(factor, {})
        
        # Skip if we already have this factor or already added a question for it
        if factor in seen_factors:
            continue
        if factor in patient_context and patient_context[factor] is not None:
            continue
        
        seen_factors.add(factor)
        
        if not defn:
            # Unknown factor — generate a generic question
            pretty_name = factor.replace("_", " ").title()
            questions.append(ClarificationQuestion(
                factor=factor,
                question=f"What is the patient's {pretty_name}?",
                query_template=f"The patient's {pretty_name} is {{answer}}",
                rationale=f"{pretty_name} is needed to determine the exact stage.",
                priority=5,
            ))
        else:
            questions.append(ClarificationQuestion(
                factor=factor,
                question=defn["question"],
                options=defn.get("options", []),
                query_template=defn.get("query_template", ""),
                rationale=defn.get("rationale", ""),
                priority=defn.get("priority", 5),
            ))
    
    # Sort by priority (highest first)
    questions.sort(key=lambda q: q.priority, reverse=True)
    
    # Build ambiguity summary
    stages_str = " or ".join(possible_stages) if possible_stages else "multiple stages"
    factors_str = ", ".join(q.question.rstrip("?") for q in questions)
    
    ambiguity_summary = (
        f"Staging is ambiguous — this patient could be {stages_str}. "
        f"To determine the exact stage, the following information is needed: {factors_str}."
    )
    
    # Build prompt addition for RAG
    prompt_lines = [
        "STAGING AMBIGUITY NOTICE:",
        f"The patient's TNM staging maps to {stages_str}.",
        "The following additional information is needed to determine the exact stage:",
    ]
    for q in questions:
        prompt_lines.append(f"  - {q.question} ({q.rationale})")
    
    if inference_notes:
        prompt_lines.append("")
        prompt_lines.append("Staging notes:")
        for note in inference_notes:
            prompt_lines.append(f"  - {note}")
    
    prompt_lines.extend([
        "",
        "INSTRUCTIONS:",
        "1. Acknowledge the staging ambiguity in your answer.",
        "2. Present findings relevant to ALL possible stages.",
        "3. Clearly state what additional information would resolve the staging.",
        "4. If treatment differs between possible stages, explain the differences.",
    ])
    
    prompt_addition = "\n".join(prompt_lines)
    
    return StagingClarification(
        questions=questions,
        ambiguity_summary=ambiguity_summary,
        prompt_addition=prompt_addition,
        needs_clarification=True,
    )


# =============================================================================
# CONVERSION TO FOLLOW-UP OPTIONS
# =============================================================================

def clarifications_to_follow_up_options(
    clarification: StagingClarification,
    original_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Convert StagingClarification into FollowUpOption-compatible dicts.
    
    These can be used directly by query_intent_service._generate_follow_up_options()
    or injected into the follow_up_options list.
    
    Returns dicts with keys matching FollowUpOption dataclass:
        action_type, label, description, query_template, priority
    """
    options = []
    
    for q in clarification.questions:
        if q.options:
            # Create one follow-up option per answer choice
            for opt in q.options:
                filled_template = q.query_template.replace("{answer}", opt)
                
                # Build a query that re-states the original with the new info
                if original_query:
                    full_query = f"{original_query}. {filled_template}"
                else:
                    full_query = filled_template
                
                options.append({
                    "action_type": "staging_clarification",
                    "label": f"{opt}",
                    "description": f"{q.question} → {opt}",
                    "query_template": full_query,
                    "priority": q.priority,
                })
        else:
            # Open-ended question — single follow-up that prompts user input
            options.append({
                "action_type": "staging_clarification",
                "label": q.question,
                "description": q.rationale,
                "query_template": q.query_template,
                "priority": q.priority,
            })
    
    return options


# =============================================================================
# CONVERSION TO SUGGESTED FOLLOWUPS (string list for enhanced_rag_service)
# =============================================================================

def clarifications_to_followup_strings(
    clarification: StagingClarification,
) -> List[str]:
    """
    Convert StagingClarification into simple follow-up suggestion strings.
    
    These can be prepended to the suggested_followups list in
    enhanced_rag_service.generate_followup_suggestions().
    
    Returns list of strings like:
        ["What is the HPV/p16 status? (needed for staging)",
         "What is the tumor grade? (needed for staging)"]
    """
    suggestions = []
    for q in clarification.questions:
        suggestion = f"{q.question}"
        if q.options:
            suggestion += f" ({' / '.join(q.options[:3])})"
        suggestions.append(suggestion)
    return suggestions
