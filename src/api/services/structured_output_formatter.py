"""
Structured Output Formatter

LIGHTWEIGHT ADD-ON - Does NOT change your RAG pipeline.

This module:
1. Detects query type from the query text (simple pattern matching)
2. Provides format instructions to append to your existing prompt
3. Parses LLM output into the layered structure

Usage:
    from structured_output_formatter import get_format_instructions, parse_structured_response
    
    # In your existing RAG service:
    query_type = detect_query_type(query)
    format_instructions = get_format_instructions(query_type)
    
    # Append to your existing prompt
    full_prompt = your_existing_prompt + "\n\n" + format_instructions
    
    # After LLM response:
    structured = parse_structured_response(llm_output, query_type)
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


# ============================================
# QUERY TYPE DETECTION (Simple pattern matching)
# ============================================

QUERY_TYPE_PATTERNS = {
    "dose_question": [
        r"\bdose\b", r"\bGy\b", r"\bfraction", r"\bfx\b", r"\bcGy\b",
        r"how much radiation", r"what dose", r"dosing", r"fractionation"
    ],
    "staging": [
        r"\bstage\b", r"\bT\d", r"\bN\d", r"\bM[01]", r"\bTNM\b", r"\bAJCC\b",
        r"what stage", r"staging"
    ],
    "trial_results": [
        r"\bresults\b", r"\boutcomes?\b", r"\bOS\b", r"\bPFS\b", r"\bDFS\b",
        r"overall survival", r"progression.free", r"hazard ratio", r"\bHR\b"
    ],
    "side_effects": [
        r"\btoxicit", r"\bside effect", r"\badverse", r"\bcomplication",
        r"what are the risks", r"late effects"
    ],
    "indication_question": [
        r"\bindication", r"when should", r"who should", r"criteria for",
        r"is .+ indicated", r"when is .+ used"
    ],
    "treatment_recommendation": [
        r"\brecommend", r"should I use", r"what treatment", r"best option",
        r"first.line", r"standard of care"
    ],
    "mechanism": [
        r"\bmechanism\b", r"how does .+ work", r"\bpathway\b", r"mode of action"
    ],
    "workup": [
        r"\bworkup\b", r"\bdiagnostic\b", r"\bimaging\b", r"what tests",
        r"how to evaluate", r"work.up"
    ],
    "comparison": [
        r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
        r"better than", r"difference between", r"which is better"
    ],
}


def detect_query_type(query: str) -> str:
    """
    Detect query type from the query text using pattern matching.
    Returns the query type string.
    """
    query_lower = query.lower()
    
    # Check each type's patterns
    scores = {}
    for query_type, patterns in QUERY_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, query_lower, re.IGNORECASE))
        if score > 0:
            scores[query_type] = score
    
    # Return highest scoring type, or 'general' if no matches
    if scores:
        return max(scores, key=scores.get)
    return "general"


# ============================================
# FORMAT INSTRUCTIONS (Append to your prompt)
# ============================================

FORMAT_INSTRUCTIONS = {
    "dose_question": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with the key dose recommendation)

PRESCRIPTION DETAILS:
- Dose: (exact dose with fractionation, e.g., "50.4 Gy in 28 fractions (1.8 Gy/fx)")
- Target: (target volume)
- Technique: (IMRT, 3D-CRT, etc.)
- Concurrent therapy: (if applicable)
- Key constraints: (critical organ limits)

EXPLANATION: (2-3 sentences with rationale and citation)
""",

    "staging": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1 sentence with the final stage)

STAGING DERIVATION:
- T stage: (value) - (reasoning)
- N stage: (value) - (reasoning)  
- M stage: (value) - (reasoning)
- Overall stage: (value per AJCC edition)

EXPLANATION: (2-3 sentences with any special considerations)
""",

    "trial_results": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with the key finding)

TRIAL RESULTS:
- Trial: (name, author, year)
- Design: (phase, N, arms)
- Primary endpoint: (what was measured)
- Results: (outcomes for each arm with HR, CI, p-value)
- Conclusion: (clinical implication)

EXPLANATION: (2-3 sentences with context)
""",

    "side_effects": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences summarizing main toxicities)

ACUTE TOXICITIES:
- (Effect): (incidence %), Grade 3+: (%), Management: (brief)

LATE TOXICITIES:
- (Effect): (incidence %), Risk factors: (if known), Management: (brief)

EXPLANATION: (2-3 sentences with monitoring recommendations)
""",

    "indication_question": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with the key indication criteria)

INDICATIONS:
- (criterion 1)
- (criterion 2)
...

CONSIDER IF:
- (relative indication with note)
...

NOT INDICATED:
- (exclusion criterion)
...

EXPLANATION: (2-3 sentences with guideline source)
""",

    "treatment_recommendation": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with the recommendation)

RECOMMENDATION:
- Preferred regimen: (specific treatment)
- Alternative: (if applicable)
- Guideline category: (1/2A/2B if applicable)
- Source: (NCCN, ASTRO, etc.)

EXPLANATION: (2-3 sentences with rationale and evidence level)
""",

    "comparison": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with the key difference)

COMPARISON:
| Outcome | Option A | Option B | Difference |
|---------|----------|----------|------------|
| (metric)| (value)  | (value)  | (delta)    |

OPTION A FAVORED WHEN: (patient selection criteria)
OPTION B FAVORED WHEN: (patient selection criteria)

EXPLANATION: (2-3 sentences with synthesis)
""",

    "mechanism": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with the core mechanism)

EXPLANATION: (3-4 sentences explaining the biological pathway, target, and clinical relevance)
""",

    "workup": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences with essential workup)

DIAGNOSTIC WORKUP:
- (test 1) - (purpose)
- (test 2) - (purpose)
- (test 3) - (purpose)
...

EXPLANATION: (2-3 sentences with timing/sequencing notes)
""",

    "general": """
RESPONSE FORMAT: Structure your answer as follows:

BRIEF ANSWER: (1-2 sentences directly answering the question)

EXPLANATION: (2-4 sentences with supporting detail and citations)
""",
}


def get_format_instructions(query_type: str) -> str:
    """Get format instructions to append to your prompt."""
    return FORMAT_INSTRUCTIONS.get(query_type, FORMAT_INSTRUCTIONS["general"])


# ============================================
# RESPONSE PARSING
# ============================================

@dataclass
class StructuredResponse:
    """The parsed structured response."""
    brief_answer: str = ""
    structured_details: Dict[str, Any] = field(default_factory=dict)
    structured_format: str = "simple_prose"
    explanation: str = ""
    query_type: str = "general"
    raw_response: str = ""  # Keep original for fallback
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Mapping of query types to structured format names
QUERY_TYPE_TO_FORMAT = {
    "dose_question": "prescription",
    "staging": "derivation_steps",
    "trial_results": "results_table",
    "side_effects": "toxicity_timeline",
    "indication_question": "criteria_checklist",
    "treatment_recommendation": "recommendation_card",
    "comparison": "comparison_table",
    "mechanism": "simple_prose",
    "workup": "ordered_checklist",
    "general": "simple_prose",
}


def parse_structured_response(llm_output: str, query_type: str) -> StructuredResponse:
    """
    Parse LLM output into structured response format.
    Works with the format instructions above.
    """
    result = StructuredResponse(
        query_type=query_type,
        structured_format=QUERY_TYPE_TO_FORMAT.get(query_type, "simple_prose"),
        raw_response=llm_output
    )
    
    # Try to extract BRIEF ANSWER
    brief_match = re.search(
        r"BRIEF ANSWER:\s*(.+?)(?=\n\n|\n[A-Z]|\nPRESCRIPTION|\nSTAGING|\nTRIAL|\nACUTE|\nINDICATIONS|\nRECOMMENDATION|\nCOMPARISON|\nEXPLANATION|\nDIAGNOSTIC|$)",
        llm_output,
        re.IGNORECASE | re.DOTALL
    )
    if brief_match:
        result.brief_answer = brief_match.group(1).strip()
    else:
        # Fallback: use first sentence or two
        sentences = re.split(r'(?<=[.!?])\s+', llm_output.strip())
        result.brief_answer = ' '.join(sentences[:2]) if sentences else llm_output[:200]
    
    # Try to extract EXPLANATION
    explanation_match = re.search(
        r"EXPLANATION:\s*(.+?)(?=\n\n[A-Z]|$)",
        llm_output,
        re.IGNORECASE | re.DOTALL
    )
    if explanation_match:
        result.explanation = explanation_match.group(1).strip()
    
    # Extract structured details based on query type
    result.structured_details = _extract_structured_details(llm_output, query_type)
    
    return result


def _extract_structured_details(text: str, query_type: str) -> Dict[str, Any]:
    """Extract structured details based on query type."""
    if query_type == "dose_question":
        return _extract_prescription(text)
    elif query_type == "staging":
        return _extract_staging(text)
    elif query_type == "trial_results":
        return _extract_trial_results(text)
    elif query_type == "side_effects":
        return _extract_toxicities(text)
    elif query_type == "indication_question":
        return _extract_indications(text)
    elif query_type == "comparison":
        return _extract_comparison(text)
    elif query_type == "workup":
        return _extract_workup(text)
    elif query_type == "treatment_recommendation":
        return _extract_recommendation(text)
    else:
        return {}


def _extract_prescription(text: str) -> Dict:
    """Extract prescription details."""
    details = {}
    patterns = {
        "dose": r"(?:Dose|Dosing):\s*(.+?)(?=\n|$)",
        "target": r"Target:\s*(.+?)(?=\n|$)",
        "technique": r"Technique:\s*(.+?)(?=\n|$)",
        "concurrent": r"Concurrent[^:]*:\s*(.+?)(?=\n|$)",
        "constraints": r"(?:Key )?[Cc]onstraints?:\s*(.+?)(?=\n\n|\nEXPLANATION|$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            details[key] = match.group(1).strip()
    return details


def _extract_staging(text: str) -> Dict:
    """Extract staging derivation."""
    details = {}
    
    # T stage
    t_match = re.search(r"T stage:\s*(T\d[abc]?)\s*[-–]\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
    if t_match:
        details["t_stage"] = {"stage": t_match.group(1), "reasoning": t_match.group(2).strip()}
    
    # N stage
    n_match = re.search(r"N stage:\s*(N\d[abc]?)\s*[-–]\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
    if n_match:
        details["n_stage"] = {"stage": n_match.group(1), "reasoning": n_match.group(2).strip()}
    
    # M stage
    m_match = re.search(r"M stage:\s*(M[01])\s*[-–]\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
    if m_match:
        details["m_stage"] = {"stage": m_match.group(1), "reasoning": m_match.group(2).strip()}
    
    # Overall
    overall_match = re.search(r"Overall stage:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
    if overall_match:
        details["overall_stage"] = overall_match.group(1).strip()
    
    return details


def _extract_trial_results(text: str) -> Dict:
    """Extract trial results."""
    details = {}
    patterns = {
        "trial_name": r"Trial:\s*(.+?)(?=\n|$)",
        "design": r"Design:\s*(.+?)(?=\n|$)",
        "primary_endpoint": r"Primary endpoint:\s*(.+?)(?=\n|$)",
        "results": r"Results:\s*(.+?)(?=\n\n|\nConclusion|$)",
        "conclusion": r"Conclusion:\s*(.+?)(?=\n\n|\nEXPLANATION|$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            details[key] = match.group(1).strip()
    return details


def _extract_toxicities(text: str) -> Dict:
    """Extract toxicity information."""
    details = {"acute": [], "late": []}
    
    # Extract acute section
    acute_section = re.search(
        r"ACUTE TOXICITIES?:(.+?)(?=LATE TOXICITIES?:|EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if acute_section:
        items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|$)", acute_section.group(1), re.DOTALL)
        for item in items:
            details["acute"].append(item.strip())
    
    # Extract late section
    late_section = re.search(
        r"LATE TOXICITIES?:(.+?)(?=EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if late_section:
        items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|$)", late_section.group(1), re.DOTALL)
        for item in items:
            details["late"].append(item.strip())
    
    return details


def _extract_indications(text: str) -> Dict:
    """Extract indication criteria."""
    details = {"indications": [], "consider_if": [], "not_indicated": []}
    
    # Extract indications section
    ind_section = re.search(
        r"INDICATIONS?:(.+?)(?=CONSIDER IF:|NOT INDICATED:|EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if ind_section:
        items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|$)", ind_section.group(1), re.DOTALL)
        details["indications"] = [i.strip() for i in items if i.strip()]
    
    # Extract consider if section
    consider_section = re.search(
        r"CONSIDER IF:(.+?)(?=NOT INDICATED:|EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if consider_section:
        items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|$)", consider_section.group(1), re.DOTALL)
        details["consider_if"] = [c.strip() for c in items if c.strip()]
    
    # Extract not indicated section
    not_ind_section = re.search(
        r"NOT INDICATED:(.+?)(?=EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not_ind_section:
        items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|$)", not_ind_section.group(1), re.DOTALL)
        details["not_indicated"] = [n.strip() for n in items if n.strip()]
    
    return details


def _extract_comparison(text: str) -> Dict:
    """Extract comparison table."""
    details = {"rows": [], "option_a_favored": "", "option_b_favored": ""}
    
    # Try to parse markdown table
    table_match = re.search(r"\|.+\|.+\|.+\|.+\|", text, re.DOTALL)
    if table_match:
        lines = table_match.group(0).strip().split('\n')
        for line in lines:
            if '|' in line and '---' not in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) >= 3:
                    details["rows"].append(cells)
    
    # Extract favored sections
    a_match = re.search(
        r"OPTION A FAVORED WHEN:\s*(.+?)(?=\nOPTION B|\nEXPLANATION|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if a_match:
        details["option_a_favored"] = a_match.group(1).strip()
    
    b_match = re.search(
        r"OPTION B FAVORED WHEN:\s*(.+?)(?=\nEXPLANATION|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if b_match:
        details["option_b_favored"] = b_match.group(1).strip()
    
    return details


def _extract_workup(text: str) -> Dict:
    """Extract diagnostic workup checklist."""
    details = {"steps": []}
    
    # Extract workup section
    workup_section = re.search(
        r"DIAGNOSTIC WORKUP:(.+?)(?=EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if workup_section:
        items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|$)", workup_section.group(1), re.DOTALL)
        for item in items:
            # Try to split on " - " for test and purpose
            parts = item.split(' - ', 1)
            if len(parts) == 2:
                details["steps"].append({"test": parts[0].strip(), "purpose": parts[1].strip()})
            else:
                details["steps"].append({"test": item.strip(), "purpose": ""})
    
    return details


def _extract_recommendation(text: str) -> Dict:
    """Extract treatment recommendation."""
    details = {}
    
    # Extract recommendation section
    rec_section = re.search(
        r"RECOMMENDATION:(.+?)(?=EXPLANATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if rec_section:
        rec_text = rec_section.group(1)
        
        # Preferred regimen
        pref_match = re.search(r"Preferred regimen:\s*(.+?)(?=\n|$)", rec_text, re.IGNORECASE)
        if pref_match:
            details["preferred_regimen"] = pref_match.group(1).strip()
        
        # Alternative
        alt_match = re.search(r"Alternative:\s*(.+?)(?=\n|$)", rec_text, re.IGNORECASE)
        if alt_match:
            details["alternative"] = alt_match.group(1).strip()
        
        # Guideline category
        cat_match = re.search(r"Guideline category:\s*(.+?)(?=\n|$)", rec_text, re.IGNORECASE)
        if cat_match:
            details["guideline_category"] = cat_match.group(1).strip()
        
        # Source
        src_match = re.search(r"Source:\s*(.+?)(?=\n|$)", rec_text, re.IGNORECASE)
        if src_match:
            details["source"] = src_match.group(1).strip()
    
    return details


# ============================================
# CONVENIENCE FUNCTION
# ============================================

def format_response(query: str, llm_output: str) -> StructuredResponse:
    """
    One-liner: detect query type and parse response.
    
    Usage:
        structured = format_response(query, llm_output)
        return structured.to_dict()
    """
    query_type = detect_query_type(query)
    return parse_structured_response(llm_output, query_type)
