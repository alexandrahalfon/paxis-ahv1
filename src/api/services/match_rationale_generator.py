"""
Generate plain-language match rationale and strength (strong/partial/possible) for patient-study matches.
Uses a single batch LLM call for efficiency.
"""

import re
from typing import List, Dict, Any
from openai import OpenAI

SYSTEM = """You are a clinical research assistant. For each study, write one short sentence (under 25 words) explaining why this study matches the patient, for a clinician. Then assign match strength:
- strong: study eligibility clearly aligns with the patient (cancer type, stage, biomarkers, demographics).
- partial: some criteria align but not all, or population is broader.
- possible: tangentially relevant, different population or setting.
Output format for each study, one per line: RATIONALE|STRENGTH
Example: "This trial enrolled EGFR-positive NSCLC patients with ECOG 0-1, matching the patient profile."|strong"""


def generate_rationales(
    openai_client: OpenAI,
    patient_summary: str,
    matches: List[Dict[str, Any]],
    max_matches: int = 10,
) -> List[Dict[str, Any]]:
    """
    For each match, produce match_rationale (str) and match_strength (strong|partial|possible).
    Returns list of dicts with keys rationale, strength (one per match, in same order).
    """
    if not matches or not patient_summary:
        return [{"rationale": None, "strength": None} for _ in matches]

    to_process = matches[:max_matches]
    # Build compact study summaries for the prompt
    study_lines = []
    for i, m in enumerate(to_process):
        title = (m.get("title") or "Unknown")[:80]
        treatment = (m.get("treatment") or "")[:60]
        demographics = m.get("demographics") or []
        cancer_chars = m.get("cancer_characteristics") or []
        key_matches = m.get("key_matches") or []
        excerpt = (m.get("relevant_text") or "")[:200]
        study_lines.append(
            f"{i+1}. {title} | Treatment: {treatment} | Demographics: {demographics} | Cancer: {cancer_chars} | Markers: {key_matches} | Excerpt: {excerpt}"
        )

    user_content = f"""Patient: {patient_summary}

Studies:
{chr(10).join(study_lines)}

For each study 1-{len(to_process)}, output one line: your one-sentence rationale|strength (strong, partial, or possible)."""

    result = [{"rationale": None, "strength": None} for _ in matches]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        text = (response.choices[0].message.content or "").strip()
        for i, line in enumerate(text.split("\n")):
            if i >= len(to_process):
                break
            line = line.strip()
            # Remove leading "1." etc
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if "|" in line:
                rationale, strength = line.split("|", 1)
                rationale = rationale.strip().strip('"')
                strength = strength.strip().lower()
                if strength not in ("strong", "partial", "possible"):
                    strength = "partial"
                result[i] = {"rationale": rationale or None, "strength": strength}
            else:
                result[i] = {"rationale": line[:200] if line else None, "strength": "partial"}
    except Exception as e:
        print(f"[MatchRationaleGenerator] Failed: {e}")
    # Pad for any matches beyond what we processed
    while len(result) < len(matches):
        result.append({"rationale": None, "strength": None})
    return result[:len(matches)]
