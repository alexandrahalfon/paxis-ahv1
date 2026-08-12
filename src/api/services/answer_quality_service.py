"""
Answer Quality Service

Provides answer quality enhancements:
1. Citation linking - inline references to specific studies with [1], [2] format
2. Confidence scoring - based on evidence agreement across studies
3. Contradiction detection - flags conflicting findings between studies
4. Structured output - dose tables, comparison matrices for specific query types
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CitationInfo:
    """Information about a citation."""
    index: int  # [1], [2], etc.
    doc_id: str
    title: str
    citation: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None


@dataclass
class ContradictionFinding:
    """A detected contradiction between studies."""
    topic: str  # What the contradiction is about (e.g., "dose", "survival rate")
    finding_a: str  # First finding
    source_a: str  # Source of first finding (study title/citation)
    doc_id_a: str
    finding_b: str  # Contradicting finding
    source_b: str  # Source of contradicting finding
    doc_id_b: str
    severity: str  # "major" or "minor"
    context: Optional[str] = None  # Additional context explaining the difference


@dataclass
class ConfidenceAssessment:
    """Assessment of answer confidence based on evidence."""
    overall_score: float  # 0.0 to 1.0
    level: str  # "high", "moderate", "low", "uncertain"
    factors: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class AnswerQualityResult:
    """Complete answer quality analysis."""
    citations: List[CitationInfo]
    citation_map: Dict[str, int]  # doc_id -> citation index
    confidence: ConfidenceAssessment
    contradictions: List[ContradictionFinding]
    consensus_findings: List[Dict[str, Any]]
    structured_data: Optional[Dict[str, Any]] = None


class AnswerQualityService:
    """
    Service for enhancing answer quality with citations, confidence, and contradiction detection.
    """
    
    def __init__(self):
        # Patterns for extracting key findings
        self._dose_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(?:Gy|cGy)(?:\s*(?:in|/)\s*(\d+)\s*(?:fractions?|fx))?',
            re.IGNORECASE
        )
        self._survival_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*%\s*(?:(\d+)[- ]?year\s+)?(?:OS|DFS|PFS|survival|control|recurrence)',
            re.IGNORECASE
        )
        self._hr_pattern = re.compile(
            r'HR\s*[=:]?\s*(\d+\.\d+)(?:\s*[\(\[]?\s*(?:95%?\s*)?CI[:\s]*(\d+\.\d+)\s*[-–]\s*(\d+\.\d+))?',
            re.IGNORECASE
        )
        self._pvalue_pattern = re.compile(
            r'p\s*[=<>]\s*(\d+\.?\d*)',
            re.IGNORECASE
        )
    
    def analyze_evidence(
        self,
        evidence: List[Dict[str, Any]],
        query_type: str = "general",
    ) -> AnswerQualityResult:
        """
        Analyze evidence for quality metrics.
        
        Args:
            evidence: List of evidence chunks
            query_type: Type of query for specialized analysis
            
        Returns:
            AnswerQualityResult with citations, confidence, and contradictions
        """
        # Build citation map
        citations, citation_map = self._build_citation_map(evidence)
        
        # Extract key findings from each study
        findings_by_study = self._extract_findings(evidence)
        
        # Detect contradictions
        contradictions = self._detect_contradictions(findings_by_study, citation_map)
        
        # Find consensus
        consensus = self._find_consensus(findings_by_study)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            evidence=evidence,
            contradictions=contradictions,
            consensus=consensus,
            query_type=query_type,
        )
        
        # Generate structured data for specific query types
        structured_data = None
        if query_type == "dose_question":
            structured_data = self._build_dose_table(findings_by_study, citation_map)
        elif query_type == "trial_results":
            structured_data = self._build_results_comparison(findings_by_study, citation_map)
        
        return AnswerQualityResult(
            citations=citations,
            citation_map=citation_map,
            confidence=confidence,
            contradictions=contradictions,
            consensus_findings=consensus,
            structured_data=structured_data,
        )
    
    def _build_citation_map(
        self,
        evidence: List[Dict[str, Any]],
    ) -> Tuple[List[CitationInfo], Dict[str, int]]:
        """Build citation list and mapping from doc_id to citation index."""
        citations = []
        citation_map = {}
        seen_doc_ids = set()
        
        for e in evidence:
            doc_id = e.get("doc_id")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            
            seen_doc_ids.add(doc_id)
            index = len(citations) + 1
            
            citation = CitationInfo(
                index=index,
                doc_id=doc_id,
                title=e.get("title", "Unknown"),
                citation=e.get("citation"),
                author=e.get("author"),
                year=e.get("year"),
                doi=e.get("doi"),
                pmid=e.get("pmid"),
            )
            citations.append(citation)
            citation_map[doc_id] = index
        
        return citations, citation_map
    
    def _extract_findings(
        self,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract key findings from each study."""
        findings_by_study = defaultdict(lambda: {
            "doc_id": None,
            "title": None,
            "citation": None,
            "doses": [],
            "survival_rates": [],
            "hazard_ratios": [],
            "p_values": [],
            "recommendations": [],
            "raw_texts": [],
        })
        
        for e in evidence:
            doc_id = e.get("doc_id")
            if not doc_id:
                continue
            
            study = findings_by_study[doc_id]
            study["doc_id"] = doc_id
            study["title"] = e.get("title", "Unknown")
            study["citation"] = e.get("citation")
            
            text = e.get("text", "")
            study["raw_texts"].append(text)
            
            # Extract doses
            for match in self._dose_pattern.finditer(text):
                dose_val = float(match.group(1))
                fractions = int(match.group(2)) if match.group(2) else None
                study["doses"].append({
                    "value": dose_val,
                    "fractions": fractions,
                    "text": match.group(0),
                })
            
            # Extract survival rates
            for match in self._survival_pattern.finditer(text):
                rate = float(match.group(1))
                timepoint = int(match.group(2)) if match.group(2) else None
                study["survival_rates"].append({
                    "rate": rate,
                    "timepoint_years": timepoint,
                    "text": match.group(0),
                })
            
            # Extract hazard ratios
            for match in self._hr_pattern.finditer(text):
                hr = float(match.group(1))
                ci_low = float(match.group(2)) if match.group(2) else None
                ci_high = float(match.group(3)) if match.group(3) else None
                study["hazard_ratios"].append({
                    "hr": hr,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "text": match.group(0),
                })
            
            # Extract p-values
            for match in self._pvalue_pattern.finditer(text):
                p_val = float(match.group(1))
                study["p_values"].append(p_val)
        
        return dict(findings_by_study)
    
    def _detect_contradictions(
        self,
        findings_by_study: Dict[str, Dict[str, Any]],
        citation_map: Dict[str, int],
    ) -> List[ContradictionFinding]:
        """Detect contradictions between studies."""
        contradictions = []
        studies = list(findings_by_study.values())
        
        for i, study_a in enumerate(studies):
            for study_b in studies[i+1:]:
                # Check dose contradictions
                dose_contradictions = self._check_dose_contradictions(study_a, study_b)
                contradictions.extend(dose_contradictions)
                
                # Check survival rate contradictions
                survival_contradictions = self._check_survival_contradictions(study_a, study_b)
                contradictions.extend(survival_contradictions)
                
                # Check HR contradictions (one shows benefit, other shows harm)
                hr_contradictions = self._check_hr_contradictions(study_a, study_b)
                contradictions.extend(hr_contradictions)
        
        return contradictions
    
    def _check_dose_contradictions(
        self,
        study_a: Dict[str, Any],
        study_b: Dict[str, Any],
    ) -> List[ContradictionFinding]:
        """Check for contradictory dose recommendations."""
        contradictions = []
        
        doses_a = study_a.get("doses", [])
        doses_b = study_b.get("doses", [])
        
        if not doses_a or not doses_b:
            return contradictions
        
        # Compare primary doses (first mentioned in each study)
        for dose_a in doses_a[:2]:  # Check first 2 doses
            for dose_b in doses_b[:2]:
                val_a = dose_a["value"]
                val_b = dose_b["value"]
                
                # Check if doses differ significantly (>10% difference)
                if abs(val_a - val_b) / max(val_a, val_b) > 0.10:
                    # Check if fractionation also differs
                    fx_a = dose_a.get("fractions")
                    fx_b = dose_b.get("fractions")
                    
                    severity = "minor"
                    context = None
                    
                    # Major if both total dose AND fractionation differ significantly
                    if fx_a and fx_b and abs(fx_a - fx_b) > 3:
                        severity = "major"
                        context = f"Different fractionation schemes: {fx_a} vs {fx_b} fractions"
                    
                    contradictions.append(ContradictionFinding(
                        topic="dose",
                        finding_a=dose_a["text"],
                        source_a=study_a.get("title", "Unknown"),
                        doc_id_a=study_a["doc_id"],
                        finding_b=dose_b["text"],
                        source_b=study_b.get("title", "Unknown"),
                        doc_id_b=study_b["doc_id"],
                        severity=severity,
                        context=context,
                    ))
                    break  # Only report one dose contradiction per study pair
            if contradictions:
                break
        
        return contradictions
    
    def _check_survival_contradictions(
        self,
        study_a: Dict[str, Any],
        study_b: Dict[str, Any],
    ) -> List[ContradictionFinding]:
        """Check for contradictory survival outcomes."""
        contradictions = []
        
        rates_a = study_a.get("survival_rates", [])
        rates_b = study_b.get("survival_rates", [])
        
        if not rates_a or not rates_b:
            return contradictions
        
        # Compare rates at same timepoints
        for rate_a in rates_a:
            for rate_b in rates_b:
                tp_a = rate_a.get("timepoint_years")
                tp_b = rate_b.get("timepoint_years")
                
                # Only compare if same timepoint (or both unspecified)
                if tp_a != tp_b and (tp_a is not None and tp_b is not None):
                    continue
                
                val_a = rate_a["rate"]
                val_b = rate_b["rate"]
                
                # Check if rates differ significantly (>15% absolute difference)
                if abs(val_a - val_b) > 15:
                    timepoint_str = f"{tp_a}-year " if tp_a else ""
                    
                    contradictions.append(ContradictionFinding(
                        topic=f"{timepoint_str}survival/control rate",
                        finding_a=rate_a["text"],
                        source_a=study_a.get("title", "Unknown"),
                        doc_id_a=study_a["doc_id"],
                        finding_b=rate_b["text"],
                        source_b=study_b.get("title", "Unknown"),
                        doc_id_b=study_b["doc_id"],
                        severity="major" if abs(val_a - val_b) > 25 else "minor",
                        context=f"Difference of {abs(val_a - val_b):.1f}% may reflect different patient populations or treatment eras",
                    ))
                    break
            if contradictions:
                break
        
        return contradictions
    
    def _check_hr_contradictions(
        self,
        study_a: Dict[str, Any],
        study_b: Dict[str, Any],
    ) -> List[ContradictionFinding]:
        """Check for contradictory hazard ratios (one favors treatment, other doesn't)."""
        contradictions = []
        
        hrs_a = study_a.get("hazard_ratios", [])
        hrs_b = study_b.get("hazard_ratios", [])
        
        if not hrs_a or not hrs_b:
            return contradictions
        
        for hr_a in hrs_a[:1]:  # Primary HR
            for hr_b in hrs_b[:1]:
                val_a = hr_a["hr"]
                val_b = hr_b["hr"]
                
                # Check if one shows benefit (HR < 1) and other shows harm (HR > 1)
                if (val_a < 0.9 and val_b > 1.1) or (val_a > 1.1 and val_b < 0.9):
                    contradictions.append(ContradictionFinding(
                        topic="treatment effect (hazard ratio)",
                        finding_a=hr_a["text"],
                        source_a=study_a.get("title", "Unknown"),
                        doc_id_a=study_a["doc_id"],
                        finding_b=hr_b["text"],
                        source_b=study_b.get("title", "Unknown"),
                        doc_id_b=study_b["doc_id"],
                        severity="major",
                        context="Studies show opposite treatment effects - may reflect different patient populations, treatments, or endpoints",
                    ))
        
        return contradictions
    
    def _find_consensus(
        self,
        findings_by_study: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find consensus findings across studies."""
        consensus = []
        
        # Group doses by value (within 5% tolerance)
        dose_groups = defaultdict(list)
        for doc_id, study in findings_by_study.items():
            for dose in study.get("doses", []):
                # Round to nearest 5 Gy for grouping
                rounded = round(dose["value"] / 5) * 5
                dose_groups[rounded].append({
                    "value": dose["value"],
                    "fractions": dose.get("fractions"),
                    "study": study["title"],
                    "doc_id": doc_id,
                })
        
        # Report doses mentioned in 2+ studies
        for dose_val, studies in dose_groups.items():
            if len(studies) >= 2:
                unique_studies = list({s["doc_id"]: s for s in studies}.values())
                if len(unique_studies) >= 2:
                    consensus.append({
                        "type": "dose",
                        "value": f"{dose_val} Gy",
                        "study_count": len(unique_studies),
                        "studies": [s["study"] for s in unique_studies],
                        "doc_ids": [s["doc_id"] for s in unique_studies],
                    })
        
        # Group survival rates by value (within 5% tolerance)
        survival_groups = defaultdict(list)
        for doc_id, study in findings_by_study.items():
            for rate in study.get("survival_rates", []):
                # Round to nearest 5%
                rounded = round(rate["rate"] / 5) * 5
                key = (rounded, rate.get("timepoint_years"))
                survival_groups[key].append({
                    "rate": rate["rate"],
                    "timepoint": rate.get("timepoint_years"),
                    "study": study["title"],
                    "doc_id": doc_id,
                })
        
        for (rate_val, timepoint), studies in survival_groups.items():
            if len(studies) >= 2:
                unique_studies = list({s["doc_id"]: s for s in studies}.values())
                if len(unique_studies) >= 2:
                    tp_str = f"{timepoint}-year " if timepoint else ""
                    consensus.append({
                        "type": "survival_rate",
                        "value": f"{tp_str}{rate_val}%",
                        "study_count": len(unique_studies),
                        "studies": [s["study"] for s in unique_studies],
                        "doc_ids": [s["doc_id"] for s in unique_studies],
                    })
        
        # Sort by study count
        consensus.sort(key=lambda x: x["study_count"], reverse=True)
        
        return consensus
    
    def _calculate_confidence(
        self,
        evidence: List[Dict[str, Any]],
        contradictions: List[ContradictionFinding],
        consensus: List[Dict[str, Any]],
        query_type: str,
    ) -> ConfidenceAssessment:
        """Calculate confidence score based on evidence quality."""
        factors = {
            "evidence_count": len(evidence),
            "unique_studies": len(set(e.get("doc_id") for e in evidence if e.get("doc_id"))),
            "major_contradictions": sum(1 for c in contradictions if c.severity == "major"),
            "minor_contradictions": sum(1 for c in contradictions if c.severity == "minor"),
            "consensus_findings": len(consensus),
            "high_consensus_count": sum(1 for c in consensus if c["study_count"] >= 3),
        }
        
        # Base score from evidence quantity
        if factors["unique_studies"] >= 5:
            base_score = 0.8
        elif factors["unique_studies"] >= 3:
            base_score = 0.7
        elif factors["unique_studies"] >= 2:
            base_score = 0.6
        else:
            base_score = 0.4
        
        # Boost for consensus
        consensus_boost = min(0.15, factors["high_consensus_count"] * 0.05)
        
        # Penalty for contradictions
        contradiction_penalty = (
            factors["major_contradictions"] * 0.15 +
            factors["minor_contradictions"] * 0.05
        )
        
        # Calculate final score
        score = max(0.1, min(1.0, base_score + consensus_boost - contradiction_penalty))
        
        # Determine level
        if score >= 0.75:
            level = "high"
            explanation = "Multiple studies with consistent findings support this answer."
        elif score >= 0.55:
            level = "moderate"
            if factors["major_contradictions"] > 0:
                explanation = "Evidence exists but studies show some conflicting findings."
            else:
                explanation = "Limited but consistent evidence supports this answer."
        elif score >= 0.35:
            level = "low"
            explanation = "Limited evidence available; interpret with caution."
        else:
            level = "uncertain"
            explanation = "Insufficient or highly conflicting evidence."
        
        factors["score_breakdown"] = {
            "base": base_score,
            "consensus_boost": consensus_boost,
            "contradiction_penalty": contradiction_penalty,
        }
        
        return ConfidenceAssessment(
            overall_score=round(score, 2),
            level=level,
            factors=factors,
            explanation=explanation,
        )
    
    def _build_dose_table(
        self,
        findings_by_study: Dict[str, Dict[str, Any]],
        citation_map: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Build structured dose table for dose questions."""
        rows = []
        
        for doc_id, study in findings_by_study.items():
            doses = study.get("doses", [])
            if not doses:
                continue
            
            citation_idx = citation_map.get(doc_id, 0)
            
            for dose in doses[:2]:  # Max 2 doses per study
                rows.append({
                    "study": study["title"],
                    "citation_index": citation_idx,
                    "dose_gy": dose["value"],
                    "fractions": dose.get("fractions"),
                    "dose_per_fx": round(dose["value"] / dose["fractions"], 2) if dose.get("fractions") else None,
                })
        
        if not rows:
            return None
        
        return {
            "type": "dose_table",
            "columns": ["Study", "Total Dose (Gy)", "Fractions", "Dose/Fx (Gy)"],
            "rows": rows,
        }
    
    def _build_results_comparison(
        self,
        findings_by_study: Dict[str, Dict[str, Any]],
        citation_map: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Build structured comparison for trial results."""
        rows = []
        
        for doc_id, study in findings_by_study.items():
            citation_idx = citation_map.get(doc_id, 0)
            
            # Get primary survival rate
            rates = study.get("survival_rates", [])
            primary_rate = rates[0] if rates else None
            
            # Get primary HR
            hrs = study.get("hazard_ratios", [])
            primary_hr = hrs[0] if hrs else None
            
            if primary_rate or primary_hr:
                rows.append({
                    "study": study["title"],
                    "citation_index": citation_idx,
                    "survival_rate": f"{primary_rate['rate']}%" if primary_rate else None,
                    "timepoint": f"{primary_rate['timepoint_years']}-year" if primary_rate and primary_rate.get("timepoint_years") else None,
                    "hazard_ratio": primary_hr["hr"] if primary_hr else None,
                    "hr_ci": f"{primary_hr['ci_low']}-{primary_hr['ci_high']}" if primary_hr and primary_hr.get("ci_low") else None,
                })
        
        if not rows:
            return None
        
        return {
            "type": "results_comparison",
            "columns": ["Study", "Survival Rate", "Timepoint", "HR", "95% CI"],
            "rows": rows,
        }



def add_inline_citations(
    answer: str,
    evidence: List[Dict[str, Any]],
    citation_map: Dict[str, int],
) -> str:
    """
    Add inline citation references [1], [2] to answer text.
    
    Matches study names, author names, and trial names in the answer
    to their corresponding citation indices.
    
    Args:
        answer: The generated answer text
        evidence: List of evidence chunks
        citation_map: Mapping from doc_id to citation index
        
    Returns:
        Answer with inline citations added
    """
    if not citation_map:
        return answer
    
    # Build patterns for each study
    study_patterns = []
    for e in evidence:
        doc_id = e.get("doc_id")
        if not doc_id or doc_id not in citation_map:
            continue
        
        idx = citation_map[doc_id]
        title = e.get("title", "")
        citation = e.get("citation", "")
        author = e.get("author", "")
        
        # Extract trial name patterns (e.g., "RTOG 0617", "MA.20", "FAST-Forward")
        trial_patterns = []
        
        # Common trial name patterns
        trial_match = re.search(
            r'\b((?:RTOG|NRG|NSABP|ACOSOG|EORTC|PORTEC|GOG|SWOG|CALGB|ECOG|Z|MA|B|C|N|E)\s*[-.]?\s*\d+[A-Z]?)\b',
            title,
            re.IGNORECASE
        )
        if trial_match:
            trial_patterns.append(trial_match.group(1))
        
        # Named trials (e.g., "FAST-Forward", "TAILORx", "KEYNOTE")
        named_trial = re.search(
            r'\b(FAST[- ]?Forward|TAILORx|KEYNOTE[- ]?\d*|CheckMate[- ]?\d*|PACIFIC|LAURA|ADAURA)\b',
            title,
            re.IGNORECASE
        )
        if named_trial:
            trial_patterns.append(named_trial.group(1))
        
        # Author patterns (e.g., "Sparano et al.", "Whelan")
        author_patterns = []
        if author:
            # First author last name
            first_author = author.split(",")[0].split(" ")[0].strip()
            if len(first_author) > 2:
                author_patterns.append(first_author)
        
        # Also check citation for author
        if citation:
            citation_author = re.match(r'^([A-Z][a-z]+)', citation)
            if citation_author:
                author_patterns.append(citation_author.group(1))
        
        study_patterns.append({
            "doc_id": doc_id,
            "idx": idx,
            "trial_patterns": trial_patterns,
            "author_patterns": author_patterns,
            "title": title,
        })
    
    # Apply citations to answer
    modified_answer = answer
    citations_added = set()
    
    for study in study_patterns:
        idx = study["idx"]
        citation_ref = f"[{idx}]"
        
        # Skip if already cited
        if idx in citations_added:
            continue
        
        # Try trial name patterns first (most specific)
        for pattern in study["trial_patterns"]:
            # Escape special regex chars but keep the pattern flexible
            escaped = re.escape(pattern).replace(r"\ ", r"\s*").replace(r"\-", r"[-\s]?")
            regex = re.compile(rf'\b({escaped})\b(?!\s*\[\d+\])', re.IGNORECASE)
            
            if regex.search(modified_answer):
                # Add citation after first occurrence
                modified_answer = regex.sub(rf'\1 {citation_ref}', modified_answer, count=1)
                citations_added.add(idx)
                break
        
        if idx in citations_added:
            continue
        
        # Try author patterns
        for pattern in study["author_patterns"]:
            # Match "Author et al." or just "Author" followed by year or verb
            regex = re.compile(
                rf'\b({re.escape(pattern)})\s*(?:et\s+al\.?)?(?:\s*\(\d{{4}}\))?(?!\s*\[\d+\])',
                re.IGNORECASE
            )
            
            if regex.search(modified_answer):
                modified_answer = regex.sub(rf'\1 {citation_ref}', modified_answer, count=1)
                citations_added.add(idx)
                break
    
    return modified_answer


def format_contradiction_warning(
    contradictions: List[ContradictionFinding],
    citation_map: Dict[str, int],
) -> Optional[str]:
    """
    Format contradiction findings as a warning section.
    
    Args:
        contradictions: List of detected contradictions
        citation_map: Mapping from doc_id to citation index
        
    Returns:
        Formatted warning string or None if no contradictions
    """
    if not contradictions:
        return None
    
    # Group by severity
    major = [c for c in contradictions if c.severity == "major"]
    minor = [c for c in contradictions if c.severity == "minor"]
    
    lines = []
    
    if major:
        lines.append("**Note: Conflicting Evidence Detected**")
        lines.append("")
        for c in major[:3]:  # Max 3 major contradictions
            idx_a = citation_map.get(c.doc_id_a, "?")
            idx_b = citation_map.get(c.doc_id_b, "?")
            lines.append(f"- **{c.topic.title()}**: {c.finding_a} [{idx_a}] vs {c.finding_b} [{idx_b}]")
            if c.context:
                lines.append(f"  *{c.context}*")
    
    if minor and not major:
        lines.append("**Note: Some variation in reported values across studies**")
        lines.append("")
        for c in minor[:2]:  # Max 2 minor contradictions
            idx_a = citation_map.get(c.doc_id_a, "?")
            idx_b = citation_map.get(c.doc_id_b, "?")
            lines.append(f"- {c.topic.title()}: {c.finding_a} [{idx_a}] vs {c.finding_b} [{idx_b}]")
    
    return "\n".join(lines) if lines else None


def format_confidence_indicator(confidence: ConfidenceAssessment) -> str:
    """
    Format confidence assessment as a brief indicator.
    
    Args:
        confidence: The confidence assessment
        
    Returns:
        Formatted confidence string
    """
    level_icons = {
        "high": "Strong",
        "moderate": "Moderate", 
        "low": "Limited",
        "uncertain": "Uncertain",
    }
    
    icon = level_icons.get(confidence.level, "")
    factors = confidence.factors
    
    parts = [f"**Evidence Confidence: {icon}**"]
    
    # Add brief explanation
    if confidence.level == "high":
        parts.append(f"({factors['unique_studies']} studies with consistent findings)")
    elif confidence.level == "moderate":
        if factors.get("major_contradictions", 0) > 0:
            parts.append(f"({factors['unique_studies']} studies, some conflicting findings)")
        else:
            parts.append(f"({factors['unique_studies']} studies)")
    elif confidence.level == "low":
        parts.append(f"(limited evidence from {factors['unique_studies']} study/studies)")
    else:
        parts.append("(insufficient evidence)")
    
    return " ".join(parts)


def format_citation_list(citations: List[CitationInfo]) -> str:
    """
    Format citation list as references section.
    
    Args:
        citations: List of citation info objects
        
    Returns:
        Formatted references string
    """
    if not citations:
        return ""
    
    lines = ["", "**References:**"]
    
    for c in citations:
        if c.citation:
            lines.append(f"[{c.index}] {c.citation}")
        elif c.title:
            year_str = f" ({c.year})" if c.year else ""
            lines.append(f"[{c.index}] {c.title}{year_str}")
    
    return "\n".join(lines)


def format_structured_data(structured_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Format structured data (dose table, comparison matrix) as markdown.
    
    Args:
        structured_data: Structured data dict from analysis
        
    Returns:
        Formatted markdown table or None
    """
    if not structured_data:
        return None
    
    data_type = structured_data.get("type")
    rows = structured_data.get("rows", [])
    
    if not rows:
        return None
    
    if data_type == "dose_table":
        lines = ["", "**Dose Summary:**", ""]
        lines.append("| Study | Total Dose | Fractions | Dose/Fx |")
        lines.append("|-------|------------|-----------|---------|")
        
        for row in rows[:6]:  # Max 6 rows
            study = row["study"][:30] + "..." if len(row["study"]) > 30 else row["study"]
            dose = f"{row['dose_gy']} Gy"
            fx = str(row["fractions"]) if row.get("fractions") else "-"
            dose_fx = f"{row['dose_per_fx']} Gy" if row.get("dose_per_fx") else "-"
            citation = f"[{row['citation_index']}]" if row.get("citation_index") else ""
            lines.append(f"| {study} {citation} | {dose} | {fx} | {dose_fx} |")
        
        return "\n".join(lines)
    
    elif data_type == "results_comparison":
        lines = ["", "**Results Summary:**", ""]
        lines.append("| Study | Survival | Timepoint | HR | 95% CI |")
        lines.append("|-------|----------|-----------|-----|--------|")
        
        for row in rows[:6]:
            study = row["study"][:25] + "..." if len(row["study"]) > 25 else row["study"]
            survival = row.get("survival_rate") or "-"
            timepoint = row.get("timepoint") or "-"
            hr = str(row["hazard_ratio"]) if row.get("hazard_ratio") else "-"
            ci = row.get("hr_ci") or "-"
            citation = f"[{row['citation_index']}]" if row.get("citation_index") else ""
            lines.append(f"| {study} {citation} | {survival} | {timepoint} | {hr} | {ci} |")
        
        return "\n".join(lines)
    
    return None


# Singleton instance
_answer_quality_service: Optional[AnswerQualityService] = None


def get_answer_quality_service() -> AnswerQualityService:
    """Get singleton AnswerQualityService instance."""
    global _answer_quality_service
    if _answer_quality_service is None:
        _answer_quality_service = AnswerQualityService()
    return _answer_quality_service


def enhance_answer_with_quality(
    answer: str,
    evidence: List[Dict[str, Any]],
    query_type: str = "general",
    include_citations: bool = True,
    include_confidence: bool = True,
    include_contradictions: bool = True,
    include_structured: bool = True,
    include_references: bool = False,
) -> Dict[str, Any]:
    """
    Enhance answer with quality metrics and formatting.
    
    This is the main entry point for answer quality enhancement.
    
    Args:
        answer: The generated answer text
        evidence: List of evidence chunks
        query_type: Type of query for specialized analysis
        include_citations: Add inline [1], [2] citations
        include_confidence: Add confidence indicator
        include_contradictions: Add contradiction warnings
        include_structured: Add structured tables for dose/results
        include_references: Add references section at end
        
    Returns:
        Dict with enhanced answer and quality metadata
    """
    service = get_answer_quality_service()
    
    # Analyze evidence
    quality = service.analyze_evidence(evidence, query_type)
    
    # Start with original answer
    enhanced_answer = answer
    
    # Add inline citations
    if include_citations:
        enhanced_answer = add_inline_citations(
            enhanced_answer,
            evidence,
            quality.citation_map,
        )
    
    # Build additional sections
    sections = []
    
    # Add structured data (dose table, results comparison)
    if include_structured and quality.structured_data:
        structured_text = format_structured_data(quality.structured_data)
        if structured_text:
            sections.append(structured_text)
    
    # Add contradiction warning
    if include_contradictions and quality.contradictions:
        warning = format_contradiction_warning(
            quality.contradictions,
            quality.citation_map,
        )
        if warning:
            sections.append("")
            sections.append(warning)
    
    # Add confidence indicator
    if include_confidence:
        confidence_text = format_confidence_indicator(quality.confidence)
        sections.append("")
        sections.append(confidence_text)
    
    # Add references
    if include_references:
        refs = format_citation_list(quality.citations)
        if refs:
            sections.append(refs)
    
    # Combine answer with sections
    if sections:
        enhanced_answer = enhanced_answer + "\n" + "\n".join(sections)
    
    return {
        "answer": enhanced_answer,
        "original_answer": answer,
        "quality": {
            "confidence": {
                "score": quality.confidence.overall_score,
                "level": quality.confidence.level,
                "explanation": quality.confidence.explanation,
                "factors": quality.confidence.factors,
            },
            "contradictions": [
                {
                    "topic": c.topic,
                    "finding_a": c.finding_a,
                    "source_a": c.source_a,
                    "finding_b": c.finding_b,
                    "source_b": c.source_b,
                    "severity": c.severity,
                    "context": c.context,
                }
                for c in quality.contradictions
            ],
            "consensus": quality.consensus_findings,
            "citation_count": len(quality.citations),
        },
        "citations": [
            {
                "index": c.index,
                "doc_id": c.doc_id,
                "title": c.title,
                "citation": c.citation,
                "year": c.year,
            }
            for c in quality.citations
        ],
        "structured_data": quality.structured_data,
    }
