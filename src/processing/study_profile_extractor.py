"""
Study Profile Extractor

Extracts structured study details from processed documents using LLM.
Integrates with the document processing pipeline to populate PostgreSQL
study_profiles table for the study details display.

Based on the comprehensive extraction pipeline from Colab.
"""

import json
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings


# =============================================================================
# EXTRACTION SCHEMA
# =============================================================================

EXTRACTION_SCHEMA = {
    "study_details": {
        "study_name": {"value": None, "evidence_quote": None},
        "protocol_name": {"value": None, "evidence_quote": None},
        "trial_registration_number": {"value": None, "evidence_quote": None},
        "publish_date": {"value": None, "evidence_quote": None},
        "study_type": {"value": None, "evidence_quote": None},
        "study_phase": {"value": None, "evidence_quote": None},
        "analysis_type": {"value": None, "evidence_quote": None},
        "number_of_patients": {"value": None, "evidence_quote": None},
        "study_institution": {"value": None, "evidence_quote": None},
        "country": {"value": None, "evidence_quote": None},
        "pmid": {"value": None, "evidence_quote": None},
        "doi": {"value": None, "evidence_quote": None}
    },
    "patient_characteristics": {
        "age_range": {"value": None, "evidence_quote": None},
        "median_age": {"value": None, "evidence_quote": None},
        "gender_distribution": {"value": None, "evidence_quote": None},
        "race_ethnicity": {"value": None, "evidence_quote": None},
        "performance_status": {"value": None, "evidence_quote": None},
        "inclusion_criteria": [],
        "exclusion_criteria": []
    },
    "diagnosis": {
        "cancer_location": {"value": None, "evidence_quote": None},
        "cancer_type": {"value": None, "evidence_quote": None},
        "histopathologic_type": {"value": None, "evidence_quote": None},
        "tumor_grade": {"value": None, "evidence_quote": None},
        "molecular_subtype": {"value": None, "evidence_quote": None}
    },
    "staging": {
        "staging_system_used": {"value": None, "evidence_quote": None},
        "stage_distribution": [],
        "risk_stratification": {"value": None, "evidence_quote": None},
        "metastatic_status": {"value": None, "evidence_quote": None},
        "extent_of_resection": {"value": None, "evidence_quote": None},
        "staging_components": []
    },
    "treatment": {
        "study_arms": [],
        "chemotherapy_regimens": [],
        "radiation_details": [],
        "surgery_details": []
    },
    "outcomes": {
        "primary_endpoint": {"value": None, "evidence_quote": None},
        "event_free_survival": {"value": None, "evidence_quote": None},
        "overall_survival": {"value": None, "evidence_quote": None},
        "progression_free_survival": {"value": None, "evidence_quote": None},
        "disease_free_survival": {"value": None, "evidence_quote": None},
        "local_control": {"value": None, "evidence_quote": None},
        "median_followup": {"value": None, "evidence_quote": None}
    },
    "biomarkers": [],
    "biomarker_inclusion_criteria": {
        "required_biomarkers": [],
        "genomic_assay": {"value": None, "evidence_quote": None},
        "score_range": {"value": None, "evidence_quote": None}
    },
    "toxicity": [],
    "dose_constraints": []
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical oncology data extraction engine with deep knowledge of medical literature, clinical trials, and treatment protocols.

Your task is to extract ALL relevant structured data from a medical document. You MUST thoroughly search the ENTIRE document including:
- Main text (abstract, introduction, methods, results, discussion)
- ALL tables (patient characteristics, treatment arms, outcomes, toxicity)
- Supplementary materials
- Figure captions and legends
- Footnotes and appendices

CRITICAL EXTRACTION RULES:
1. Extract ONLY information explicitly present in the document - DO NOT infer or assume
2. If a field is not explicitly stated anywhere, return null for both value and evidence_quote
3. Every extracted value MUST include a short evidence_quote copied EXACTLY from the document
4. DO NOT summarize evidence quotes - copy them verbatim (exact text from document)
5. Look in ALL sections and ALL tables - information can appear anywhere
6. For clinical trials, pay special attention to:
   - Tables showing patient characteristics, treatment arms, outcomes, toxicity
   - Methods section for inclusion/exclusion criteria
   - Protocol schemas and study design sections
7. Extract specific numbers, percentages, doses, and schedules whenever mentioned
8. For staging: Pediatric cancers use MANY different systems (COG groups, Chang M, INSS, INRG, PRETEXT, SIOP, risk groups) - NOT just TNM
9. For inclusion/exclusion criteria: Look in Methods/Patients/Eligibility sections - extract EVERY criterion listed
10. If multiple studies are present, extract data for the PRIMARY study
11. DO NOT hallucinate numbers or make up information
12. For biomarker_inclusion_criteria:
    - required_biomarkers: list ALL biomarker requirements for study enrollment, each as {"name": "ER", "status": "positive"} or {"name": "HER2", "status": "negative"}. Include receptor status (ER, PR, HER2), mutation status (EGFR, ALK, KRAS, BRAF), and any other molecular markers required for eligibility.
    - genomic_assay: name of any genomic assay used (e.g. "Oncotype DX", "MammaPrint", "Decipher", "Prolaris", "EndoPredict")
    - score_range: the score range or threshold used for eligibility (e.g. "11-25", "≤25", "high risk")

OUTPUT FORMAT:
- Return valid JSON only
- Use null (not empty strings) for missing fields
- For evidence_quote: copy exact text, preserving numbers, units, and context
- For arrays: include ALL relevant items found in the document"""


def create_extraction_prompt(document_content: str) -> str:
    """Create the extraction prompt with document content."""
    schema_json = json.dumps(EXTRACTION_SCHEMA, indent=2)
    
    return f"""Extract ALL relevant structured fields from the medical document below.

**IMPORTANT:** Search the ENTIRE document including all text, tables, and supplementary materials.

Return JSON matching this exact structure:
{schema_json}

DOCUMENT TO EXTRACT FROM:
═══════════════════════════════════════════════════════════════
{document_content}
═══════════════════════════════════════════════════════════════

EXTRACTION INSTRUCTIONS:
1. Read the ENTIRE document including all tables
2. For each field in the schema, search the entire document
3. Extract specific values with exact evidence quotes
4. If a field is not found anywhere, use null
5. Pay special attention to tables - they often contain critical data
6. Return valid JSON only - no markdown formatting

Begin extraction now:"""


# =============================================================================
# DOCUMENT CONTENT LOADER
# =============================================================================

class ProcessedDocumentLoader:
    """Load and combine content from processed document files."""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4
    
    def find_document_files(self, doc_dir: Path) -> Dict[str, Optional[Path]]:
        """Find all processed document files in a directory."""
        files = {
            'pixtral_content': None,
            'structured_content': None,
            'tables': None,
            'table_dictionaries': None,
            'references': None,
            'document_index': None,
            'tables_csv_dir': None
        }
        
        for pattern, key in [
            ('*_pixtral_content.txt', 'pixtral_content'),
            ('*_structured_content.json', 'structured_content'),
            ('*_tables.json', 'tables'),
            ('*_table_dictionaries.json', 'table_dictionaries'),
            ('*_references.json', 'references'),
            ('*_document_index.json', 'document_index'),
        ]:
            matches = list(doc_dir.glob(pattern))
            if matches:
                files[key] = matches[0]
        
        tables_csv_dir = doc_dir / 'tables_csv'
        if tables_csv_dir.exists() and tables_csv_dir.is_dir():
            files['tables_csv_dir'] = tables_csv_dir
        
        return files
    
    def load_json_file(self, file_path: Path) -> Dict:
        """Load a JSON file safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    ⚠️  Error loading {file_path.name}: {e}")
            return {}
    
    def load_text_file(self, file_path: Path) -> str:
        """Load a text file safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"    ⚠️  Error loading {file_path.name}: {e}")
            return ""
    
    def load_csv_table(self, csv_path: Path) -> str:
        """Load CSV table as markdown with proper handling."""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                title = ""
                if first_line.startswith('#'):
                    title = first_line.strip('# \n\r')
                    f.seek(0)
                    next(f)
                else:
                    f.seek(0)
                
                reader = csv.reader(f)
                all_rows = list(reader)
                all_rows = [row for row in all_rows if row and any(cell for cell in row)]
                
                if not all_rows:
                    return ""
                
                # Process headers
                header_row = all_rows[0]
                processed_headers = []
                last_non_empty = None
                
                for i, header in enumerate(header_row):
                    if header and header.strip():
                        processed_headers.append(header)
                        last_non_empty = header
                    else:
                        if last_non_empty:
                            processed_headers.append(f"{last_non_empty}_col{i}")
                        else:
                            processed_headers.append(f"Column_{i}")
                
                if not processed_headers:
                    processed_headers = ["Column_0"]
                
                # Build markdown table
                markdown = ""
                if title:
                    markdown += f"\n### {title}\n\n"
                
                markdown += "| " + " | ".join(str(h) for h in processed_headers) + " |\n"
                markdown += "| " + " | ".join(["---"] * len(processed_headers)) + " |\n"
                
                for row in all_rows[1:]:
                    if not row or not any(cell for cell in row):
                        continue
                    
                    values = []
                    for i in range(len(processed_headers)):
                        if i < len(row):
                            cell = row[i]
                            values.append(str(cell) if cell else "")
                        else:
                            values.append("")
                    
                    markdown += "| " + " | ".join(values) + " |\n"
                
                return markdown
                
        except Exception as e:
            print(f"    ⚠️  Error loading CSV {csv_path.name}: {e}")
            return ""
    
    def combine_document_content(self, doc_dir: Path) -> str:
        """Combine all processed document content into a single string."""
        print(f"  📄 Loading content from {doc_dir.name}...")
        files = self.find_document_files(doc_dir)
        combined_parts = []
        
        # Structured content (main source)
        if files['structured_content']:
            print("    Loading structured content...")
            structured = self.load_json_file(files['structured_content'])
            
            if 'document_metadata' in structured:
                metadata = structured['document_metadata']
                doc_info = metadata.get('document_info', {})
                pub_info = metadata.get('publication_info', {})
                
                combined_parts.append("# DOCUMENT METADATA")
                combined_parts.append(f"Title: {doc_info.get('title', 'Unknown')}")
                combined_parts.append(f"Journal: {pub_info.get('journal', 'Unknown')}")
                combined_parts.append(f"Publication Date: {pub_info.get('publication_date', 'Unknown')}")
                combined_parts.append(f"DOI: {pub_info.get('doi', 'Unknown')}")
                combined_parts.append("")
            
            if 'sections' in structured:
                combined_parts.append("# DOCUMENT CONTENT")
                for section_name, section_content in structured['sections'].items():
                    if section_content and section_name not in ['document_metadata']:
                        combined_parts.append(f"\n## {section_name}")
                        combined_parts.append(section_content)
        
        # Tables from CSV
        if files['tables_csv_dir']:
            print("    Loading tables from CSV...")
            csv_files = sorted(files['tables_csv_dir'].glob('*.csv'))
            if csv_files:
                combined_parts.append("\n# TABLES AND STRUCTURED DATA\n")
                for csv_file in csv_files:
                    table_md = self.load_csv_table(csv_file)
                    if table_md:
                        combined_parts.append(table_md)
                        combined_parts.append("")
        
        # Pixtral content as fallback
        if len('\n'.join(combined_parts)) < 1000 and files['pixtral_content']:
            print("    Using pixtral OCR content as fallback...")
            pixtral = self.load_text_file(files['pixtral_content'])
            pixtral = re.sub(r'--- Page \d+ \(Pixtral\) ---', '', pixtral)
            combined_parts.append("\n# FULL DOCUMENT TEXT (OCR)\n")
            combined_parts.append(pixtral)
        
        combined_text = '\n'.join(combined_parts)
        combined_text = re.sub(r'\n{3,}', '\n\n', combined_text)
        
        tokens = self.estimate_tokens(combined_text)
        print(f"    Combined {len(combined_text)} chars, ~{tokens:,} tokens")
        
        return combined_text


# =============================================================================
# STUDY PROFILE EXTRACTOR
# =============================================================================

class StudyProfileExtractor:
    """Extract structured study profiles from processed documents."""
    
    def __init__(self):
        settings = get_settings()
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.study_profile_model or "gpt-4o-mini"
        self.loader = ProcessedDocumentLoader()
        
        self.total_tokens = 0
        self.total_cost = 0.0
        
        print(f"✓ Initialized StudyProfileExtractor")
        print(f"  Model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        """Call LLM with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Track usage
            usage = response.usage
            self.total_tokens += usage.total_tokens
            
            # Cost estimation (gpt-4o-mini pricing)
            input_cost = (usage.prompt_tokens / 1_000_000) * 0.150
            output_cost = (usage.completion_tokens / 1_000_000) * 0.600
            self.total_cost += (input_cost + output_cost)
            
            content = response.choices[0].message.content
            
            # Clean up JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON decode error: {e}")
            raise
        except Exception as e:
            print(f"    ⚠️  LLM call error: {e}")
            raise
    
    def extract_from_processed_dir(self, doc_dir: Path) -> Dict[str, Any]:
        """
        Extract study profile from a processed document directory.
        
        Args:
            doc_dir: Path to processed document directory containing
                     *_structured_content.json, *_tables.json, etc.
        
        Returns:
            Extracted study profile dictionary
        """
        print(f"\n{'='*60}")
        print(f"Extracting: {doc_dir.name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Load and combine document content
            combined_content = self.loader.combine_document_content(doc_dir)
            
            if not combined_content or len(combined_content) < 100:
                raise ValueError("No usable content found")
            
            # Truncate if too long (gpt-4o-mini has 128K context, but keep reasonable)
            # ~100K chars = ~25K tokens, leaving room for prompt and response
            max_chars = 100000
            if len(combined_content) > max_chars:
                print(f"    Truncating content from {len(combined_content)} to {max_chars} chars")
                combined_content = combined_content[:max_chars]
            
            # Create extraction prompt
            prompt = create_extraction_prompt(combined_content)
            
            # Call LLM for extraction
            print(f"  🔍 Extracting study profile...")
            extracted = self._call_llm(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt
            )
            
            duration = time.time() - start_time
            
            result = {
                "document_name": doc_dir.name,
                "document_path": str(doc_dir),
                "extraction_timestamp": datetime.now().isoformat(),
                "processing_duration_seconds": round(duration, 2),
                "extracted_data": extracted,
                "api_usage": {
                    "total_tokens": self.total_tokens,
                    "total_cost_usd": round(self.total_cost, 4)
                }
            }
            
            print(f"  ✅ Completed in {duration:.1f}s")
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                "document_name": doc_dir.name,
                "document_path": str(doc_dir),
                "extraction_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "extracted_data": None
            }
    
    def extract_from_content(self, content: str, doc_name: str = "document") -> Dict[str, Any]:
        """
        Extract study profile from raw document content string.
        
        Args:
            content: Combined document content string
            doc_name: Name for logging
        
        Returns:
            Extracted study profile dictionary
        """
        print(f"  🔍 Extracting study profile from content...")
        start_time = time.time()
        
        try:
            # Truncate if too long (gpt-4o-mini has 128K context)
            max_chars = 100000
            if len(content) > max_chars:
                content = content[:max_chars]
            
            prompt = create_extraction_prompt(content)
            
            extracted = self._call_llm(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt
            )
            
            duration = time.time() - start_time
            
            return {
                "document_name": doc_name,
                "extraction_timestamp": datetime.now().isoformat(),
                "processing_duration_seconds": round(duration, 2),
                "extracted_data": extracted
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                "document_name": doc_name,
                "extraction_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "extracted_data": None
            }
    
    def get_usage_summary(self) -> Dict:
        """Get API usage summary."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "model": self.model
        }
