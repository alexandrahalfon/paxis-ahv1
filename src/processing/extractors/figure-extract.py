#!/usr/bin/env python3
"""
Table, Chart, and Figure Extraction Strategy for Clinical Trial PDFs

This module provides comprehensive strategies for ensuring accurate extraction
of structured data (tables), visualizations (charts), and important figures
from clinical trial documents.

Key insight: Tables and figures are where the actual trial data lives.
Text extraction is just context. Get these right, and your RAG system works.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TableType(Enum):
    """Classification of table types found in clinical trials"""
    TREATMENT_DOSAGE = "treatment_dosage"        # Drug name, dose, schedule
    PATIENT_CHARACTERISTICS = "patient_characteristics"  # Age, gender, ECOG
    OUTCOMES = "outcomes"                         # Survival, response, efficacy
    ADVERSE_EVENTS = "adverse_events"            # Safety data
    BASELINE = "baseline"                        # Study baseline characteristics
    DOSAGE_MODIFICATION = "dosage_modification"  # Dose adjustments
    PHARMACOKINETICS = "pharmacokinetics"        # PK parameters
    INCLUSION_EXCLUSION = "inclusion_exclusion"  # I/E criteria
    OTHER = "other"


class ChartType(Enum):
    """Classification of chart/figure types"""
    SURVIVAL_CURVE = "survival_curve"            # Kaplan-Meier most important
    RESPONSE_RATE = "response_rate"              # CR/PR/SD/PD breakdown
    EFFICACY_COMPARISON = "efficacy_comparison"  # Drug A vs Drug B
    TOXICITY_PROFILE = "toxicity_profile"        # Adverse event rates
    TIMELINE = "timeline"                        # Study timeline
    OTHER = "other"


@dataclass
class ExtractedTable:
    """Structured representation of an extracted table"""
    table_id: str
    table_type: TableType
    page_number: int
    
    # Raw data
    raw_text: str                      # Full text extraction
    
    # Structured data
    headers: List[str]                 # Column headers
    rows: List[List[str]]              # Table data as list of rows
    
    # Metadata
    caption: str                       # Table caption/title
    footnotes: List[str]               # Footnotes and explanations
    units: Dict[str, str]              # Column units (e.g., "Dose" → "mg/m²")
    
    # Quality metrics
    extraction_confidence: float       # 0.0-1.0
    validation_status: str             # "valid", "needs_review", "invalid"
    errors: List[str]                  # Extraction issues found
    
    # Source tracking
    source_method: str                 # "mistral_ocr", "pixtral", or "both"
    source_text_snippets: List[str]   # Original text for verification


@dataclass
class ExtractedChart:
    """Structured representation of extracted chart/figure"""
    figure_id: str
    chart_type: ChartType
    page_number: int
    
    # Raw data
    raw_image_base64: Optional[str]   # Original image if captured
    
    # Extracted data
    title: str                        # Figure title
    caption: str                      # Figure caption
    axis_labels: Dict[str, str]       # x_label, y_label, etc.
    extracted_values: Dict            # Extracted numeric data
    curves_data: List[Dict]           # For curves: x, y, confidence_interval
    
    # Metadata
    scale: str                        # Linear, log, etc.
    legend_items: List[str]           # Legend labels
    
    # Quality metrics
    extraction_confidence: float
    validation_status: str
    errors: List[str]
    
    # Source tracking
    source_method: str                # "pixtral", "image_analysis", etc.


# =========== STRATEGY 1: ENHANCED PIXTRAL PROMPTING ===========

class EnhancedTableExtractionPrompts:
    """
    Specific prompts for Pixtral to extract tables with high accuracy.
    
    Key insight: Pixtral is vision-based and can see table structure,
    but needs specific guidance to extract properly.
    """
    
    @staticmethod
    def table_extraction_prompt() -> str:
        """Comprehensive prompt for table extraction from medical documents"""
        
        return """You are extracting tables from a clinical trial document.
        
CRITICAL INSTRUCTIONS for TABLE EXTRACTION:

1. TABLE IDENTIFICATION
   - Identify ALL tables/data matrices visible on this page
   - Include both formal tables and informal data grids
   - Note if table spans multiple pages

2. FOR EACH TABLE, EXTRACT:
   
   A. STRUCTURE
      - Column headers (exact text, preserve abbreviations)
      - All row headers
      - All data cells (preserve exact values, units, percentages)
      - Merged cells (indicate merge span)
      
   B. NUMERIC DATA (CRITICAL FOR MEDICAL ACCURACY)
      - All numbers must be EXACT (not rounded/summarized)
      - Preserve decimal places (60.5 mg NOT ~60 mg)
      - Preserve ranges (60-80 mg, NOT "around 60")
      - Preserve confidence intervals (24.1-28.5, NOT ~24-28)
      - Preserve statistical notations (p<0.001, p=0.05)
      
   C. UNITS AND SCALES
      - Extract units for each column (mg, mg/m², %, months, years, etc.)
      - Identify which values use which units
      - Note any unit conversions or clarifications
      
   D. FOOTNOTES AND EXPLANATIONS
      - Extract ALL footnotes (marked with *, †, ‡, numbers, letters)
      - Explain what each notation means
      - Include any table notes at bottom
      
   E. CONTEXT
      - Table title/caption
      - Which row contains the data being measured
      - Any special groupings or subheadings within table
      - Total n/number of patients if shown

3. DATA VALIDATION
   - If you see a column with percentages, verify they add to ~100%
   - If you see means with n, verify sample sizes make sense
   - If you see statistical tests, note the p-value and significance level
   - Flag any data that looks inconsistent

4. OUTPUT FORMAT
   Provide ONLY valid JSON (no markdown, no code blocks):
   {
     "tables": [
       {
         "table_number": 1,
         "title": "exact title from document",
         "headers": ["Header 1", "Header 2", "Header 3"],
         "units": {"Header 1": "mg/m²", "Header 2": "%", "Header 3": "months"},
         "rows": [
           ["Row Label", "Value 1", "Value 2"],
           ["Row Label 2", "Value 1b", "Value 2b"]
         ],
         "footnotes": [
           "* Definition for asterisk notation",
           "† Definition for dagger notation"
         ],
         "extraction_confidence": 0.95,
         "issues": []  // Any concerns about extraction accuracy
       }
     ]
   }

CRITICAL: 
- Do NOT summarize table values
- Do NOT round numbers
- Do NOT skip columns
- Do NOT infer missing data
- Preserve EXACT formatting and values from the original table"""
    
    @staticmethod
    def dosage_table_prompt() -> str:
        """Specialized prompt for dosage/treatment tables (most critical)"""
        
        return """Extract treatment/dosage information from this table with EXTREME precision.

Clinical trial dosage tables are CRITICAL - errors here affect patient safety.

FOR DOSAGE/TREATMENT TABLES:

1. EXTRACT EXACTLY:
   - Drug name (generic AND brand name if both shown)
   - Dose amount (e.g., "60 mg/m²", "300 mg/day")
   - Route of administration (IV, oral, subcutaneous, intramuscular)
   - Schedule (e.g., "Day 1, 8, 15 of 28-day cycle")
   - Frequency (daily, weekly, every X days)
   - Duration (how long treatment given)
   - Cycle length if applicable
   - Retreatment schedule
   - Maximum cumulative dose if specified

2. PRESERVE EXACT FORMATTING:
   - "60 mg/m²" NOT "60mg/m2"
   - "every 21 days" NOT "21 day intervals"
   - "IV over 1 hour" NOT "1 hour IV"

3. HANDLE COMPLEX SCHEDULES:
   - Drug A: 60 mg/m² on days 1,8,15 + Drug B: 300 mg/day continuously
   - Extract both as separate rows if combined
   - Note which drugs are in combination

4. INCLUDE MODIFICATIONS:
   - Dose escalation/de-escalation criteria
   - Modification based on toxicity
   - Hold/skip day rules
   - Retreatment eligibility

Example correct output:
{
  "drug_name": "Doxorubicin (Adriamycin)",
  "dose": "60 mg/m²",
  "route": "IV",
  "schedule_days": "Day 1",
  "schedule_frequency": "Every 21 days",
  "cycle_length_days": 21,
  "number_of_cycles": 8,
  "administration_time": "1 hour"
}

Provide ONLY valid JSON."""
    
    @staticmethod
    def figure_extraction_prompt() -> str:
        """Specialized prompt for extracting data from figures/charts"""
        
        return """Extract quantitative data from figures/charts in clinical trials.

CRITICAL: Charts contain key trial results. Extract accurately.

FOR EACH FIGURE/CHART:

1. IDENTIFY CHART TYPE:
   - Kaplan-Meier survival curve (MOST IMPORTANT)
   - Bar chart (response rates, adverse events)
   - Line plot (continuous outcomes)
   - Forest plot (hazard ratios)
   - Scatter plot
   - Other

2. FOR KAPLAN-MEIER CURVES (highest priority):
   - Treatment groups being compared
   - Median survival time for each group (exact numbers)
   - Survival rates at key timepoints (6 months, 1 year, 2 year, etc.)
   - Number at risk (n) at each timepoint
   - P-value for log-rank test
   - Confidence intervals if shown
   - Censoring marks on curve
   
3. FOR BAR CHARTS:
   - Category labels
   - Exact percentages or counts for each bar
   - Error bars/confidence intervals
   - Statistical significance indicators
   - Legend groups
   
4. FOR LINE PLOTS:
   - All data points visible on the plot
   - Timepoints on x-axis
   - Values on y-axis
   - Confidence intervals if present
   
5. EXTRACT AXIS INFORMATION:
   - X-axis: label, scale (linear/log), units, range
   - Y-axis: label, scale, units, range
   - Any secondary axes
   
6. EXTRACT TEXT FROM FIGURE:
   - Title
   - Caption
   - Legend
   - Notes
   - Statistical annotations

Example output for Kaplan-Meier:
{
  "figure_type": "survival_curve",
  "title": "Overall Survival by Treatment Group",
  "groups": [
    {
      "name": "Doxorubicin",
      "median_survival_months": 24.1,
      "median_survival_confidence_interval": "20.5-28.7",
      "survival_rates": {
        "6_months": "0.95",
        "12_months": "0.78",
        "24_months": "0.42"
      }
    }
  ],
  "p_value": "0.023",
  "log_rank_test": "p<0.05"
}

Provide ONLY valid JSON."""


# =========== STRATEGY 2: TABLE DETECTION & VALIDATION ===========

class TableDetectionValidator:
    """
    Validate extracted tables for accuracy and completeness.
    
    This layer checks that what we extracted actually makes sense medically.
    """
    
    def validate_table_structure(self, table: ExtractedTable) -> Tuple[bool, List[str]]:
        """
        Validate table has proper structure.
        
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check headers exist
        if not table.headers:
            issues.append("No headers found")
            return False, issues
        
        # Check all rows have same number of columns
        if table.rows:
            expected_cols = len(table.headers)
            for i, row in enumerate(table.rows):
                if len(row) != expected_cols:
                    issues.append(
                        f"Row {i} has {len(row)} columns, expected {expected_cols}"
                    )
        
        # Check for empty cells (might indicate extraction failure)
        empty_count = sum(1 for row in table.rows for cell in row if not cell.strip())
        if empty_count > len(table.rows):  # More empties than rows = suspicious
            issues.append(f"High number of empty cells: {empty_count}")
        
        return len(issues) == 0, issues
    
    def validate_dosage_table(self, table: ExtractedTable) -> Tuple[bool, List[str]]:
        """
        Specialized validation for dosage tables.
        Medical accuracy is critical here.
        """
        issues = []
        
        if table.table_type != TableType.TREATMENT_DOSAGE:
            return True, []  # Not a dosage table
        
        # Check for required fields in dosage tables
        required_keywords = ["drug", "dose", "schedule", "route"]
        found_keywords = [kw for kw in required_keywords 
                         if any(kw in h.lower() for h in table.headers)]
        
        if len(found_keywords) < 2:
            issues.append(f"Missing dosage fields. Found: {found_keywords}")
        
        # Validate numeric dosages
        dosage_issues = self._validate_dosage_numbers(table)
        issues.extend(dosage_issues)
        
        # Check for medical abbreviations/terminology
        medical_terms = ["IV", "mg", "day", "cycle", "hour", "hr", "mg/m²"]
        has_medical = any(any(term in cell for cell in row for row in table.rows)
                         for term in medical_terms)
        
        if not has_medical:
            issues.append("Table doesn't contain expected medical terminology")
        
        return len(issues) == 0, issues
    
    def _validate_dosage_numbers(self, table: ExtractedTable) -> List[str]:
        """Validate that dosage numbers are realistic"""
        issues = []
        
        # Typical ranges for chemotherapy (mg, mg/m²)
        dose_patterns = [
            (r"(\d+(?:\.\d+)?)\s*mg/m²", (0, 1000)),   # 0-1000 mg/m²
            (r"(\d+(?:\.\d+)?)\s*mg/kg", (0, 100)),    # 0-100 mg/kg
            (r"(\d+(?:\.\d+)?)\s*mg(?!\w)", (0, 5000)), # 0-5000 mg
        ]
        
        for row in table.rows:
            for cell in row:
                for pattern, (min_val, max_val) in dose_patterns:
                    matches = re.findall(pattern, cell, re.IGNORECASE)
                    for match in matches:
                        try:
                            val = float(match)
                            if not (min_val <= val <= max_val):
                                issues.append(
                                    f"Unusual dosage value: {val} (expected {min_val}-{max_val})"
                                )
                        except ValueError:
                            pass
        
        return issues
    
    def validate_outcomes_table(self, table: ExtractedTable) -> Tuple[bool, List[str]]:
        """Validate outcomes table (survival, response rates, etc.)"""
        issues = []
        
        if table.table_type != TableType.OUTCOMES:
            return True, []
        
        # Validate percentages add up
        for row in table.rows:
            percentages = []
            for cell in row:
                # Extract percentages
                pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", cell)
                percentages.extend([float(m) for m in pct_matches])
            
            # Check if they're response rates (CR+PR+SD+PD should sum to ~100)
            if len(percentages) >= 4:
                total = sum(percentages[:4])  # Assuming CR, PR, SD, PD order
                if total > 0 and not (95 <= total <= 105):  # Allow small rounding
                    issues.append(f"Response rates don't sum to 100%: {total}%")
        
        # Validate survival times are realistic
        survival_matches = []
        for row in table.rows:
            for cell in row:
                if "months" in cell.lower() or "years" in cell.lower():
                    nums = re.findall(r"(\d+(?:\.\d+)?)\s*(?:month|year)", cell)
                    survival_matches.extend(nums)
        
        for survival_time in survival_matches:
            val = float(survival_time)
            if val < 0.1 or val > 600:  # 0.1 months to 50 years
                issues.append(f"Unrealistic survival time: {val}")
        
        return len(issues) == 0, issues
    
    def cross_validate_extraction_methods(self, 
                                         mistral_table: str,
                                         pixtral_table: str) -> Dict:
        """
        Compare tables extracted by different methods.
        If both methods agree, confidence is high.
        If they disagree, flag for review.
        """
        
        # Simple similarity check
        mistral_clean = re.sub(r"\s+", "", mistral_table.lower())
        pixtral_clean = re.sub(r"\s+", "", pixtral_table.lower())
        
        # Calculate character-level similarity
        common_chars = sum(1 for a, b in zip(mistral_clean, pixtral_clean) if a == b)
        max_len = max(len(mistral_clean), len(pixtral_clean))
        
        similarity = common_chars / max_len if max_len > 0 else 0.0
        
        return {
            "similarity_score": similarity,
            "methods_agree": similarity > 0.85,
            "requires_review": similarity < 0.85,
            "confidence": "HIGH" if similarity > 0.9 else "MEDIUM" if similarity > 0.75 else "LOW"
        }


# =========== STRATEGY 3: DUAL EXTRACTION APPROACH ===========

class DualTableExtractionStrategy:
    """
    Extract tables using BOTH Mistral OCR and Pixtral,
    then intelligently merge them.
    
    Key insight: Each method has strengths:
    - Mistral OCR: Good at detecting table structure/hierarchy
    - Pixtral: Good at seeing visual elements and text layout
    
    Using both provides validation and redundancy.
    """
    
    def __init__(self):
        self.validator = TableDetectionValidator()
    
    def extract_tables_from_page(self, 
                                page_image_base64: str,
                                page_text_ocr: str,
                                page_number: int) -> List[ExtractedTable]:
        """
        Extract tables from a single page using dual approach.
        """
        
        extracted_tables = []
        
        # Approach 1: Extract with Pixtral (vision-based)
        pixtral_tables = self._extract_with_pixtral(page_image_base64, page_number)
        
        # Approach 2: Extract with text patterns (Mistral OCR text)
        mistral_tables = self._extract_with_text_patterns(page_text_ocr, page_number)
        
        # Approach 3: Merge and deduplicate
        merged_tables = self._merge_table_extractions(
            pixtral_tables, mistral_tables, page_number
        )
        
        # Validate each merged table
        for table in merged_tables:
            is_valid, issues = self.validator.validate_table_structure(table)
            
            if table.table_type == TableType.TREATMENT_DOSAGE:
                is_valid, dosage_issues = self.validator.validate_dosage_table(table)
                issues.extend(dosage_issues)
            elif table.table_type == TableType.OUTCOMES:
                is_valid, outcome_issues = self.validator.validate_outcomes_table(table)
                issues.extend(outcome_issues)
            
            table.errors = issues
            table.validation_status = "valid" if is_valid else "needs_review"
            
            extracted_tables.append(table)
        
        return extracted_tables
    
    def _extract_with_pixtral(self, page_image_base64: str, 
                              page_number: int) -> List[ExtractedTable]:
        """Extract tables using Pixtral vision model"""
        # Implementation would call Pixtral with table extraction prompt
        # and parse the JSON response into ExtractedTable objects
        pass
    
    def _extract_with_text_patterns(self, page_text_ocr: str,
                                    page_number: int) -> List[ExtractedTable]:
        """Extract tables from OCR text using pattern matching"""
        # Implementation would use regex/heuristics to detect tables
        # from text layout (multiple columns, aligned numbers, etc.)
        pass
    
    def _merge_table_extractions(self, pixtral_tables: List[ExtractedTable],
                                mistral_tables: List[ExtractedTable],
                                page_number: int) -> List[ExtractedTable]:
        """
        Merge tables from both extraction methods.
        
        Strategy:
        1. If both extracted same table → use most complete version
        2. If only one extracted → include it
        3. If conflict → flag for review
        """
        
        merged = []
        used_mistral_indices = set()
        
        # For each Pixtral table, find matching Mistral table
        for pix_table in pixtral_tables:
            best_match = None
            best_similarity = 0.0
            best_match_idx = -1
            
            for i, mis_table in enumerate(mistral_tables):
                if i in used_mistral_indices:
                    continue
                
                # Compare tables
                similarity = self.validator.cross_validate_extraction_methods(
                    "\n".join(mis_table.rows),
                    "\n".join(pix_table.rows)
                )
                
                if similarity["similarity_score"] > best_similarity:
                    best_similarity = similarity["similarity_score"]
                    best_match = mis_table
                    best_match_idx = i
            
            if best_match and best_similarity > 0.85:
                # Merge: use most complete data from both
                merged_table = self._merge_two_tables(pix_table, best_match)
                merged_table.source_method = "both"
                merged.append(merged_table)
                used_mistral_indices.add(best_match_idx)
            else:
                # No match, include Pixtral table as-is
                pix_table.source_method = "pixtral"
                merged.append(pix_table)
        
        # Add any Mistral tables that weren't matched
        for i, mis_table in enumerate(mistral_tables):
            if i not in used_mistral_indices:
                mis_table.source_method = "mistral_ocr"
                merged.append(mis_table)
        
        return merged
    
    def _merge_two_tables(self, table1: ExtractedTable, 
                         table2: ExtractedTable) -> ExtractedTable:
        """
        Merge two versions of the same table.
        Prefer data from the extraction with higher confidence.
        """
        
        # Use table with higher extraction confidence as primary
        primary = table1 if table1.extraction_confidence >= table2.extraction_confidence else table2
        secondary = table2 if primary is table1 else table1
        
        # Merge headers (use primary, fill gaps from secondary)
        merged_headers = primary.headers.copy()
        for i, h in enumerate(secondary.headers):
            if i >= len(merged_headers) or not merged_headers[i]:
                merged_headers[i] = h
        
        # Merge rows (use primary, compare with secondary for validation)
        merged_rows = primary.rows.copy()
        
        # Merge footnotes and units
        merged_footnotes = list(set(primary.footnotes + secondary.footnotes))
        merged_units = {**primary.units, **secondary.units}
        
        # Create merged table
        merged = ExtractedTable(
            table_id=primary.table_id,
            table_type=primary.table_type,
            page_number=primary.page_number,
            raw_text=primary.raw_text,
            headers=merged_headers,
            rows=merged_rows,
            caption=primary.caption or secondary.caption,
            footnotes=merged_footnotes,
            units=merged_units,
            extraction_confidence=(primary.extraction_confidence + secondary.extraction_confidence) / 2,
            validation_status="valid",  # Will be re-validated
            errors=[],
            source_method="both",
            source_text_snippets=primary.source_text_snippets + secondary.source_text_snippets
        )
        
        return merged


# =========== STRATEGY 4: FIGURE/CHART DATA EXTRACTION ===========

class ChartDataExtractor:
    """
    Extract numeric data from figures and charts.
    
    Kaplan-Meier curves are most important - they contain trial outcomes.
    """
    
    def extract_kaplan_meier_data(self, chart_image_base64: str) -> ExtractedChart:
        """
        Extract specific data from Kaplan-Meier survival curve.
        
        Most important data:
        - Median survival times for each group
        - Survival percentages at key timepoints
        - P-value
        - Log-rank test result
        """
        
        # Use Pixtral with specialized KM prompt
        km_prompt = """This is a Kaplan-Meier survival curve from a clinical trial.

Extract with EXTREME precision:

1. Treatment groups shown
2. For each group:
   - Median survival time (exact number in months)
   - Survival rates at 6, 12, 24 months (exact percentages)
   - Number at risk (n) at each timepoint
   - Confidence intervals if shown

3. Statistical results:
   - P-value from log-rank test
   - Hazard ratio if shown
   - Significance level

4. Curve characteristics:
   - Which group has better survival (visual inspection)
   - Crossover points if any
   - Events vs censoring ratio if shown

Provide ONLY valid JSON with exact numbers extracted from the curve."""
        
        # Call Pixtral with image and prompt
        # Parse response to ExtractedChart
        pass
    
    def extract_bar_chart_data(self, chart_image_base64: str) -> ExtractedChart:
        """Extract data from response rate or adverse event bar charts"""
        pass
    
    def extract_toxicity_profile(self, chart_image_base64: str) -> ExtractedChart:
        """Extract adverse event grades and frequencies"""
        pass


# =========== STRATEGY 5: QUALITY METRICS ===========

class TableExtractionQualityMetrics:
    """
    Track quality metrics for table extraction.
    These help identify which tables need clinical review.
    """
    
    @staticmethod
    def calculate_extraction_confidence(table: ExtractedTable) -> float:
        """
        Calculate confidence score for extracted table (0.0-1.0).
        
        Factors:
        - Source method (both > pixtral > mistral)
        - Validation issues found
        - Medical accuracy checks
        - Structural completeness
        """
        
        confidence = 1.0
        
        # Factor 1: Source method
        if table.source_method == "both":
            confidence *= 0.95  # Both sources agree - very reliable
        elif table.source_method == "pixtral":
            confidence *= 0.85  # Vision-based alone
        elif table.source_method == "mistral_ocr":
            confidence *= 0.75  # Text-based alone
        
        # Factor 2: Validation issues
        if table.errors:
            confidence *= (1.0 - len(table.errors) * 0.05)  # 5% per issue
        
        # Factor 3: Completeness
        if not table.headers:
            confidence *= 0.5
        elif not table.rows:
            confidence *= 0.3
        elif not table.caption:
            confidence *= 0.9
        
        # Factor 4: Unit information
        if not table.units:
            confidence *= 0.85  # Missing units is concerning
        
        return max(0.0, min(1.0, confidence))
    
    @staticmethod
    def generate_extraction_report(tables: List[ExtractedTable]) -> Dict:
        """Generate comprehensive quality report for extraction"""
        
        return {
            "total_tables": len(tables),
            "by_status": {
                "valid": sum(1 for t in tables if t.validation_status == "valid"),
                "needs_review": sum(1 for t in tables if t.validation_status == "needs_review"),
                "invalid": sum(1 for t in tables if t.validation_status == "invalid"),
            },
            "by_type": {
                table_type.value: sum(1 for t in tables if t.table_type == table_type)
                for table_type in TableType
            },
            "by_source": {
                "mistral_ocr": sum(1 for t in tables if t.source_method == "mistral_ocr"),
                "pixtral": sum(1 for t in tables if t.source_method == "pixtral"),
                "both": sum(1 for t in tables if t.source_method == "both"),
            },
            "average_confidence": sum(t.extraction_confidence for t in tables) / len(tables) if tables else 0.0,
            "tables_needing_review": [
                {
                    "table_id": t.table_id,
                    "page": t.page_number,
                    "reason": t.errors,
                    "confidence": t.extraction_confidence
                }
                for t in tables if t.validation_status == "needs_review"
            ]
        }


# =========== INTEGRATION WITH EXTRACTION PIPELINE ===========

class EnhancedDocumentProcessor:
    """
    Extended DocumentProcessor with table/chart extraction.
    
    Integration points:
    - Phase 1: Extract tables as structured data
    - Phase 2: Index tables separately (with special handling)
    - Export: Tables in both text and structured formats
    """
    
    def extract_tables_and_figures(self, pdf_path: str):
        """
        Main method to extract all tables and figures from PDF.
        
        Returns tables with validation and confidence scores.
        """
        
        strategy = DualTableExtractionStrategy()
        chart_extractor = ChartDataExtractor()
        quality_metrics = TableExtractionQualityMetrics()
        
        all_tables = []
        all_charts = []
        
        # Process each page
        for page_num, page_image in enumerate(self.pages):
            # Extract tables
            page_tables = strategy.extract_tables_from_page(
                page_image_base64=self._convert_to_base64(page_image),
                page_text_ocr=self.page_text[page_num],
                page_number=page_num + 1
            )
            all_tables.extend(page_tables)
            
            # Extract figures/charts
            # (implementation would detect chart regions and extract)
        
        # Generate report
        quality_report = quality_metrics.generate_extraction_report(all_tables)
        
        # Export structured table data
        export_data = {
            "tables": [
                {
                    "table_id": t.table_id,
                    "type": t.table_type.value,
                    "page": t.page_number,
                    "headers": t.headers,
                    "rows": t.rows,
                    "units": t.units,
                    "footnotes": t.footnotes,
                    "confidence": t.extraction_confidence,
                    "status": t.validation_status,
                    "errors": t.errors,
                    "source": t.source_method
                }
                for t in all_tables
            ],
            "quality_report": quality_report,
            "charts": [
                {
                    "chart_id": c.figure_id,
                    "type": c.chart_type.value,
                    "page": c.page_number,
                    "title": c.title,
                    "extracted_values": c.extracted_values,
                    "confidence": c.extraction_confidence
                }
                for c in all_charts
            ]
        }
        
        return export_data, all_tables, all_charts


def main():
    """Main function to run figure/table extraction."""
    import os
    from pathlib import Path
    from datetime import datetime
    from dotenv import load_dotenv
    from pdf2image import convert_from_path
    from mistralai import Mistral
    import base64
    from io import BytesIO
    
    load_dotenv()
    
    # Get PDF path from environment or use provided path
    pdf_path = os.getenv('DOCUMENT_PATH', 
        "/Users/ahalfon/Downloads/References for RAG Questions/Trastuzumab with trimodality treatment for esophageal adenocarcinoma with HER2 overexpression - Q76.pdf")
    
    if pdf_path.startswith('"') and pdf_path.endswith('"'):
        pdf_path = pdf_path[1:-1]
    
    if not os.path.exists(pdf_path):
        print(f"✗ PDF file not found: {pdf_path}")
        return
    
    print("\n" + "="*70)
    print("FIGURE AND TABLE EXTRACTION")
    print("="*70)
    print(f"Document: {os.path.basename(pdf_path)}\n")
    
    # Initialize Mistral client
    mistral_api_key = os.getenv('MISTRAL_API_KEY')
    if not mistral_api_key:
        print("✗ Mistral API key not found. Please set MISTRAL_API_KEY in .env file.")
        return
    
    mistral_client = Mistral(api_key=mistral_api_key)
    mistral_model = os.getenv('MISTRAL_MODEL', 'pixtral-large-latest')
    
    # Convert PDF to images
    print("📄 Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=200)
    print(f"✓ Converted to {len(images)} pages\n")
    
    all_tables = []
    all_figures = []
    
    # Process each page
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"📊 Processing page {page_num}/{len(images)}...")
        
        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Extract tables using enhanced prompt
        table_prompt = EnhancedTableExtractionPrompts.table_extraction_prompt()
        
        try:
            response = mistral_client.chat.complete(
                model=mistral_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": table_prompt},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                        ]
                    }
                ]
            )
            
            result_text = response.choices[0].message.content
            
            # Clean up markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            tables = result.get("tables", [])
            
            for table in tables:
                table["page"] = page_num
                all_tables.append(table)
            
            if tables:
                print(f"  ✓ Found {len(tables)} table(s)")
                for table in tables:
                    print(f"    - {table.get('title', 'Untitled table')}")
        
        except Exception as e:
            print(f"  ⚠ Error extracting tables: {e}")
        
        # Extract figures using enhanced prompt
        figure_prompt = EnhancedTableExtractionPrompts.figure_extraction_prompt()
        
        try:
            response = mistral_client.chat.complete(
                model=mistral_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": figure_prompt},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                        ]
                    }
                ]
            )
            
            result_text = response.choices[0].message.content
            
            # Clean up markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # Handle both dict and list responses
            if isinstance(result, dict):
                figures = result.get("figures", [])
            elif isinstance(result, list):
                figures = result
            else:
                figures = []
            
            for figure in figures:
                if isinstance(figure, dict):
                    figure["page"] = page_num
                    all_figures.append(figure)
            
            if figures:
                print(f"  ✓ Found {len(figures)} figure(s)")
                for figure in figures:
                    print(f"    - {figure.get('title', 'Untitled')} ({figure.get('figure_type', 'unknown')})")
        
        except Exception as e:
            print(f"  ⚠ Error extracting figures: {e}")
    
    # Save results
    output_dir = Path("extracted_figures_tables")
    output_dir.mkdir(exist_ok=True)
    
    doc_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{doc_name}_figures_tables_{timestamp}.json"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "source_pdf": pdf_path,
        "extraction_summary": {
            "total_tables": len(all_tables),
            "total_figures": len(all_figures)
        },
        "tables": all_tables,
        "figures": all_figures
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Tables extracted: {len(all_tables)}")
    print(f"Figures extracted: {len(all_figures)}")
    print(f"\n✓ Results saved to: {output_file}")
    
    if all_tables:
        print("\nTables:")
        for table in all_tables:
            print(f"  - Page {table['page']}: {table.get('title', 'Untitled')}")
    
    if all_figures:
        print("\nFigures:")
        for figure in all_figures:
            print(f"  - Page {figure['page']}: {figure.get('title', 'Untitled')}")


if __name__ == "__main__":
    main()