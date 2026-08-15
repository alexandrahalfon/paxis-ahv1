"""
Numerical validation for RAG answers.

Extracts statistical values (percentages, hazard / odds ratios, CIs,
p-values, doses) from generated answers, validates each against the
retrieved evidence, and replaces unverified numbers with an explicit
`[unverified]` token so end-users never see a fabricated statistic.

Canonical home — called from every pipeline:
    P1: EnhancedRAGService.query
    P2: ComprehensiveRetriever (via EnhancedRAGService.query_study_focused)
    P4: TumorBoard specialist summaries
    P5: QueryIntentService formatted_response

`enhanced_rag_service.py` keeps thin back-compat wrappers that re-export
these symbols so legacy call sites keep working without signature churn.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ─── Regex patterns ──────────────────────────────────────────────────────────

STAT_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "percentage": re.compile(
        r'(\d+\.?\d*)\s*%'
        r'(?:\s*\(?\s*(?:'
        r'(?:95%?\s*)?CI[:\s]*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*%?|'
        r'p\s*[=<>]\s*(\d*\.?\d+)|'
        r'HR[:\s]*(\d+\.?\d*)'
        r')\s*\)?)?',
        re.IGNORECASE,
    ),
    "hazard_ratio": re.compile(
        r'(?:HR|hazard\s+ratio)[:\s]*(\d+\.?\d*)'
        r'(?:\s*\(?\s*(?:95%?\s*)?CI[:\s]*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*\)?)?'
        r'(?:\s*[,;]?\s*p\s*[=<>]\s*(\d*\.?\d+))?',
        re.IGNORECASE,
    ),
    "odds_ratio": re.compile(
        r'(?:OR|odds\s+ratio)[:\s]*(\d+\.?\d*)'
        r'(?:\s*\(?\s*(?:95%?\s*)?CI[:\s]*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*\)?)?'
        r'(?:\s*[,;]?\s*p\s*[=<>]\s*(\d*\.?\d+))?',
        re.IGNORECASE,
    ),
    "p_value": re.compile(r'p\s*[=<>]\s*(\d*\.?\d+)', re.IGNORECASE),
    "ci": re.compile(
        r'(?:95%?\s*)?CI[:\s]*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)',
        re.IGNORECASE,
    ),
    "dose_gy": re.compile(r'(\d+\.?\d*)\s*(?:Gy|gray)', re.IGNORECASE),
    "survival_time": re.compile(
        r'(\d+\.?\d*)\s*(?:months?|years?|mo|yr)', re.IGNORECASE,
    ),
    "generic_number": re.compile(r'(\d+\.?\d*)'),
}


# ─── Extraction ──────────────────────────────────────────────────────────────

def extract_numbers_with_stats(text: str) -> List[Dict[str, Any]]:
    """Extract numerical values (%, HR, OR, dose) with their surrounding stats."""
    results: List[Dict[str, Any]] = []

    for match in STAT_PATTERNS["percentage"].finditer(text):
        result = {
            "value": float(match.group(1)),
            "type": "percentage",
            "unit": "%",
            "raw_match": match.group(0),
            "position": match.start(),
        }
        if match.group(2) and match.group(3):
            result["ci_low"] = float(match.group(2))
            result["ci_high"] = float(match.group(3))
        if match.group(4):
            result["p_value"] = match.group(4)
        if match.group(5):
            result["hr"] = float(match.group(5))
        results.append(result)

    for match in STAT_PATTERNS["hazard_ratio"].finditer(text):
        result = {
            "value": float(match.group(1)),
            "type": "hazard_ratio",
            "unit": "HR",
            "raw_match": match.group(0),
            "position": match.start(),
        }
        if match.group(2) and match.group(3):
            result["ci_low"] = float(match.group(2))
            result["ci_high"] = float(match.group(3))
        if match.group(4):
            result["p_value"] = match.group(4)
        results.append(result)

    for match in STAT_PATTERNS["odds_ratio"].finditer(text):
        result = {
            "value": float(match.group(1)),
            "type": "odds_ratio",
            "unit": "OR",
            "raw_match": match.group(0),
            "position": match.start(),
        }
        if match.group(2) and match.group(3):
            result["ci_low"] = float(match.group(2))
            result["ci_high"] = float(match.group(3))
        if match.group(4):
            result["p_value"] = match.group(4)
        results.append(result)

    for match in STAT_PATTERNS["dose_gy"].finditer(text):
        results.append({
            "value": float(match.group(1)),
            "type": "dose",
            "unit": "Gy",
            "raw_match": match.group(0),
            "position": match.start(),
        })

    return results


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_numbers_against_sources(
    answer: str,
    evidence: List[Dict[str, Any]],
    tolerance: float = 1.0,
) -> Dict[str, Any]:
    """Validate numbers in `answer` against `evidence` source text."""
    answer_numbers = extract_numbers_with_stats(answer)
    source_text = " ".join([e.get("text", "") for e in evidence])
    source_numbers = extract_numbers_with_stats(source_text)

    source_lookup: Dict[Any, List[Dict[str, Any]]] = {}
    for sn in source_numbers:
        key = (sn["type"], round(sn["value"], 1))
        source_lookup.setdefault(key, []).append(sn)

    validated: List[Dict[str, Any]] = []
    unvalidated: List[Dict[str, Any]] = []

    for an in answer_numbers:
        # "95% CI" is boilerplate — skip validation
        if an["type"] == "percentage" and an["value"] == 95.0:
            raw = an.get("raw_match", "")
            if "CI" in raw or "confidence" in raw.lower():
                continue

        key = (an["type"], round(an["value"], 1))
        if key in source_lookup:
            source_match = max(source_lookup[key], key=lambda x: len(x))
            validated.append({
                "answer_value": an,
                "source_value": source_match,
                "exact_match": True,
            })
        else:
            found = False
            for sn in source_numbers:
                if sn["type"] == an["type"]:
                    if abs(sn["value"] - an["value"]) <= tolerance:
                        validated.append({
                            "answer_value": an,
                            "source_value": sn,
                            "exact_match": False,
                            "difference": abs(sn["value"] - an["value"]),
                        })
                        found = True
                        break
            if not found:
                unvalidated.append(an)

    total = len(answer_numbers)
    validation_rate = len(validated) / total if total > 0 else 1.0

    return {
        "validated_numbers": validated,
        "unvalidated_numbers": unvalidated,
        "validation_rate": validation_rate,
        "total_numbers": total,
    }


# ─── Stripping unverified values ─────────────────────────────────────────────

def strip_unvalidated_numbers(
    answer: str,
    unvalidated: List[Dict[str, Any]],
) -> str:
    """Replace each unvalidated number's `raw_match` with "[unverified]".

    Bumps `pipeline_metrics.safety.numerical_stripped` once per replacement
    so the regression-test harness can assert on this safety trigger.
    """
    if not answer or not unvalidated:
        return answer
    cleaned = answer
    replaced = 0
    for uv in unvalidated:
        raw = uv.get("raw_match")
        if not raw:
            continue
        idx = cleaned.find(raw)
        if idx >= 0:
            cleaned = cleaned[:idx] + "[unverified]" + cleaned[idx + len(raw):]
            replaced += 1
    if replaced:
        print(
            f"  [Numerical Validation] Stripped {replaced} unverified "
            f"numbers from answer"
        )
        try:
            from src.api.services import pipeline_metrics
            pipeline_metrics.incr("safety", "numerical_stripped", replaced)
        except Exception:
            pass
    return cleaned


# ─── Enrichment ──────────────────────────────────────────────────────────────

def enrich_answer_with_stats(
    answer: str,
    evidence: List[Dict[str, Any]],
) -> str:
    """Append CI / p-value / HR context to validated percentages in `answer`."""
    source_text = " ".join([e.get("text", "") for e in evidence])
    source_numbers = extract_numbers_with_stats(source_text)

    source_lookup: Dict[Any, Dict[str, Any]] = {}
    for sn in source_numbers:
        key = (sn["type"], round(sn["value"], 1))
        if key not in source_lookup or len(sn) > len(source_lookup[key]):
            source_lookup[key] = sn

    enrichments: List[Dict[str, Any]] = []
    for match in STAT_PATTERNS["percentage"].finditer(answer):
        value = float(match.group(1))
        key = ("percentage", round(value, 1))
        if key in source_lookup:
            source = source_lookup[key]
            match_text = match.group(0)
            additions: List[str] = []
            if "ci_low" in source and "CI" not in match_text:
                additions.append(f"95% CI: {source['ci_low']}-{source['ci_high']}%")
            if "p_value" in source and "p" not in match_text.lower():
                additions.append(f"p={source['p_value']}")
            if "hr" in source and "HR" not in match_text:
                additions.append(f"HR={source['hr']}")
            if additions:
                enrichments.append({
                    "position": match.end(),
                    "addition": f" ({', '.join(additions)})",
                })

    enriched_answer = answer
    for e in sorted(enrichments, key=lambda x: x["position"], reverse=True):
        pos = e["position"]
        enriched_answer = enriched_answer[:pos] + e["addition"] + enriched_answer[pos:]

    return enriched_answer
