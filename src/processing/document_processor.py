#!/usr/bin/env python3
"""
Complete Document Processor

Processes a clinical trial PDF document and generates all outputs:
1. Mistral OCR structured content
2. Pixtral detailed content
3. Document index
4. Extracted tables (JSON)
5. Extracted tables (CSV)
6. Extracted figures/charts
7. Table dictionaries
8. Processed content cache

All outputs saved in one organized folder per document.
"""

import json
import os
import csv
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO
from dotenv import load_dotenv

# Import from our organized modules
from ..core.config import get_settings
from .document_metadata_extractor import DocumentMetadataExtractor
from .document_indexer import DocumentIndexer

load_dotenv()


class CompleteDocumentProcessor:
    """Process entire document and generate all outputs."""

    def __init__(self, pdf_path: str, output_dir: Optional[str] = None):
        """Initialize with PDF path."""
        import openai
        from mistralai import Mistral
        
        settings = get_settings()
        
        # Initialize OpenAI client
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.openai_model = settings.openai_model
        
        # Initialize Mistral client
        if not settings.mistral_api_key:
            raise ValueError("Mistral API key not found. Please set MISTRAL_API_KEY in .env file.")
        
        self.mistral_client = Mistral(api_key=settings.mistral_api_key)
        self.mistral_model = settings.mistral_model
        self.mistral_ocr_model = settings.mistral_ocr_model
        
        self.pdf_path = pdf_path
        self.doc_name = Path(pdf_path).stem
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory for this document
        if output_dir:
            self.output_dir = Path(output_dir) / self.doc_name
        else:
            self.output_dir = Path(settings.output_dir) / self.doc_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for extracted data
        self.structured_content = {}
        self.references_content = {}
        self.pixtral_content = ""
        self.document_index = {}
        self.extracted_tables = []
        self.extracted_figures = []
        
        print(f"✓ Initialized Complete Document Processor")
        print(f"  Document: {self.doc_name}")
        print(f"  Output directory: {self.output_dir}")

    def process_complete(self) -> Dict:
        """
        Process entire document and generate all outputs.
        
        Returns summary of all generated files.
        """
        print("\n" + "="*70)
        print("COMPLETE DOCUMENT PROCESSING")
        print("="*70)
        print(f"Document: {self.doc_name}")
        print(f"Timestamp: {self.timestamp}")
        print("="*70)
        
        generated_files = {
            "document": self.doc_name,
            "timestamp": self.timestamp,
            "output_directory": str(self.output_dir),
            "files": {}
        }
        
        # Phase 1: Mistral OCR - Structured Content
        print("\n📄 PHASE 1: Extracting structured content (Mistral OCR)...")
        self.structured_content = self._extract_with_mistral_ocr()
        ocr_file = self._save_structured_content()
        generated_files["files"]["structured_content"] = str(ocr_file)
        
        # Save references separately if found
        if self.references_content:
            references_file = self._save_references()
            generated_files["files"]["references"] = str(references_file)
        
        # Phase 2: Pixtral - Detailed Content
        print("\n🔍 PHASE 2: Extracting detailed content (Pixtral)...")
        self.pixtral_content = self._extract_with_pixtral()
        pixtral_file = self._save_pixtral_content()
        generated_files["files"]["pixtral_content"] = str(pixtral_file)
        
        # Phase 3: Document Indexing
        print("\n📇 PHASE 3: Creating document index...")
        self.document_index = self._create_document_index()
        index_file = self._save_document_index()
        generated_files["files"]["document_index"] = str(index_file)
        
        # Phase 4: Extract Tables and Figures
        print("\n📊 PHASE 4: Extracting tables and figures...")
        self._extract_tables_and_figures()
        tables_json_file = self._save_tables_json()
        generated_files["files"]["tables_json"] = str(tables_json_file)
        
        # Phase 5: Save Tables as CSV
        print("\n📋 PHASE 5: Saving tables as CSV...")
        csv_files = self._save_tables_as_csv()
        generated_files["files"]["tables_csv"] = csv_files
        
        # Phase 6: Create Table Dictionaries
        print("\n🗂️  PHASE 6: Creating table dictionaries...")
        dict_file = self._create_table_dictionaries()
        generated_files["files"]["table_dictionaries"] = str(dict_file)
        
        # Phase 7: Save Complete Processed Content (for cache)
        print("\n💾 PHASE 7: Saving complete processed content to cache...")
        cache_file = self._save_processed_content_to_cache()
        generated_files["files"]["processed_content_cache"] = str(cache_file)
        
        # Phase 8: Generate Summary Report
        print("\n📝 PHASE 8: Generating summary report...")
        summary_file = self._generate_summary_report(generated_files)
        generated_files["files"]["summary_report"] = str(summary_file)
        
        # Phase 9: Auto-sync to GCP (if enabled)
        settings = get_settings()
        if settings.auto_sync_gcp:
            print("\n☁️  PHASE 9: Auto-syncing to GCP bucket...")
            try:
                from ..utils.gcp_sync import GCPBucketSync
                
                sync = GCPBucketSync()
                sync_stats = sync.sync_document(self.doc_name)
                
                print(f"  ✓ Synced to GCP: {sync_stats['uploaded']} files uploaded")
                generated_files["gcp_sync"] = sync_stats
                
            except Exception as e:
                print(f"  ⚠️  GCP sync failed: {e}")
        
        # Phase 10: Extract Study Profile (if enabled)
        if getattr(settings, 'extract_study_profiles', False):
            print("\n📋 PHASE 10: Extracting study profile...")
            try:
                profile_result = self.extract_study_profile()
                if profile_result and profile_result.get("extracted_data"):
                    profile_file = self._save_study_profile(profile_result)
                    generated_files["files"]["study_profile"] = str(profile_file)
                    generated_files["study_profile"] = profile_result.get("extracted_data")
                    print(f"  ✓ Study profile extracted")
                    
                    # Store to PostgreSQL
                    try:
                        import asyncio
                        from ..api.services.study_profile_storage_service import get_study_profile_storage_service
                        
                        storage = get_study_profile_storage_service()
                        study_id = asyncio.get_event_loop().run_until_complete(
                            storage.store_study_profile(
                                doc_id=self.doc_name,
                                document_name=self.doc_name,
                                extracted_data=profile_result.get("extracted_data"),
                                processing_duration=profile_result.get("processing_duration_seconds")
                            )
                        )
                        print(f"  ✓ Study profile stored in PostgreSQL: study_id={study_id}")
                        generated_files["study_profile_id"] = study_id
                    except Exception as e:
                        print(f"  ⚠️  PostgreSQL storage failed: {e}")
            except Exception as e:
                print(f"  ⚠️  Study profile extraction failed: {e}")
        
        print("\n" + "="*70)
        print("✓ PROCESSING COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir}")
        
        if settings.auto_sync_gcp:
            print(f"✓ Synced to: gs://{settings.gcp_bucket_name}/processed_documents/{self.doc_name}")
        
        return generated_files

    def _extract_with_mistral_ocr(self) -> Dict:
        """Extract structured content using Mistral OCR, with PyMuPDF fallback."""
        try:
            from mistralai.models import DocumentURLChunk
            
            with open(self.pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            uploaded = self.mistral_client.files.upload(
                file={"file_name": os.path.basename(self.pdf_path), "content": pdf_bytes},
                purpose="ocr"
            )

            signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded.id)

            ocr_response = self.mistral_client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=self.mistral_ocr_model
            )

            content_by_heading = {}
            references_content = {}
            
            for page in ocr_response.pages:
                lines = page.markdown.split('\n')
                current_heading = None
                current_content = []

                for line in lines:
                    if line.startswith("#"):
                        if current_heading is not None:
                            # Check if this is a references section
                            is_references = any(ref_kw in current_heading.lower() for ref_kw in ['references', 'bibliography', 'citations'])
                            if is_references:
                                references_content[current_heading] = '\n'.join(current_content)
                            else:
                                content_by_heading[current_heading] = '\n'.join(current_content)
                        
                        current_heading = line[1:].strip()
                        current_content = []
                    else:
                        if current_heading is not None:
                            current_content.append(line)

                # Save the last section
                if current_heading is not None:
                    is_references = any(ref_kw in current_heading.lower() for ref_kw in ['references', 'bibliography', 'citations'])
                    if is_references:
                        references_content[current_heading] = '\n'.join(current_content)
                    else:
                        content_by_heading[current_heading] = '\n'.join(current_content)

            # Store references separately
            self.references_content = references_content
            
            print(f"  ✓ Extracted {len(content_by_heading)} content sections")
            if references_content:
                print(f"  ✓ Extracted {len(references_content)} reference sections")
            
            return content_by_heading

        except Exception as e:
            import traceback
            print(f"  [OCR] Mistral OCR failed: {e}")
            traceback.print_exc()
            print(f"  [OCR] Falling back to PyMuPDF text extraction...")
            import sys
            sys.stdout.flush()
            return self._extract_with_pymupdf_fallback()
    
    def _extract_with_pymupdf_fallback(self) -> Dict:
        """Fallback text extraction using PyMuPDF when Mistral OCR fails."""
        try:
            import fitz  # PyMuPDF
            
            print(f"  [PyMuPDF] Opening PDF: {self.pdf_path}")
            doc = fitz.open(self.pdf_path)
            content_by_heading = {}
            all_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            
            # Try to parse into sections based on common headings
            full_text = "\n".join(all_text)
            
            # Common section headings in clinical papers
            section_keywords = [
                'abstract', 'introduction', 'background', 'methods', 'materials and methods',
                'results', 'discussion', 'conclusion', 'conclusions', 'acknowledgments',
                'references', 'supplementary', 'appendix', 'patients', 'study design',
                'statistical analysis', 'outcomes', 'endpoints', 'treatment', 'eligibility'
            ]
            
            # Simple section detection
            lines = full_text.split('\n')
            current_section = "Content"
            current_content = []
            
            for line in lines:
                line_lower = line.lower().strip()
                # Check if this line is a section header
                is_header = False
                for keyword in section_keywords:
                    if line_lower == keyword or line_lower.startswith(keyword + ':') or line_lower.startswith(keyword + ' '):
                        if len(line_lower) < 50:  # Headers are usually short
                            is_header = True
                            # Save previous section
                            if current_content:
                                content_by_heading[current_section] = '\n'.join(current_content)
                            current_section = line.strip().title()
                            current_content = []
                            break
                
                if not is_header:
                    current_content.append(line)
            
            # Save last section
            if current_content:
                content_by_heading[current_section] = '\n'.join(current_content)
            
            # If no sections detected, put everything under "Content"
            if not content_by_heading:
                content_by_heading["Content"] = full_text
            
            total_chars = sum(len(v) for v in content_by_heading.values())
            print(f"  [PyMuPDF] Extracted {len(content_by_heading)} sections ({total_chars} chars)")
            
            # Also store as pixtral content fallback for indexing
            if not self.pixtral_content and full_text:
                self.pixtral_content = full_text
                print(f"  [PyMuPDF] Also set pixtral_content fallback ({len(full_text)} chars)")
            
            return content_by_heading
            
        except Exception as e:
            import traceback
            print(f"  [PyMuPDF] Fallback also failed: {e}")
            traceback.print_exc()
            return {}

    def _extract_with_pixtral(self) -> str:
        """Extract detailed content using Pixtral, with PyMuPDF fallback."""
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(self.pdf_path, dpi=200)
            print(f"  [Pixtral] Converted to {len(images)} images")
            
            extracted_text = ""
            key_pages = min(len(images), 10)
            
            for i in range(key_pages):
                print(f"  [Pixtral] Processing page {i+1}/{key_pages}...")
                
                buffer = BytesIO()
                images[i].save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                response = self.mistral_client.chat.complete(
                    model=self.mistral_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract ALL text from this medical document page with attention to:
1. Drug names and dosages
2. Treatment schedules
3. Numbers and measurements
4. Tables and structured data
5. Patient characteristics
6. Survival data and outcomes

Preserve exact formatting and numbers."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{image_base64}"
                                }
                            ]
                        }
                    ]
                )
                
                page_text = response.choices[0].message.content
                if page_text:
                    extracted_text += f"\n--- Page {i+1} (Pixtral) ---\n{page_text}\n"
            
            print(f"  [Pixtral] Extracted {len(extracted_text)} characters")
            return extracted_text
            
        except Exception as e:
            import traceback
            print(f"  [Pixtral] Extraction failed: {e}")
            traceback.print_exc()
            print(f"  [Pixtral] Using PyMuPDF text as fallback...")
            return self._extract_pixtral_fallback()
    
    def _extract_pixtral_fallback(self) -> str:
        """Fallback for Pixtral using PyMuPDF text extraction."""
        try:
            import fitz  # PyMuPDF
            
            # Check if we already have content from structured extraction fallback
            if self.pixtral_content:
                print(f"  [Pixtral Fallback] Using existing content ({len(self.pixtral_content)} chars)")
                return self.pixtral_content
            
            print(f"  [Pixtral Fallback] Extracting with PyMuPDF...")
            doc = fitz.open(self.pdf_path)
            all_text = []
            
            key_pages = min(len(doc), 10)
            for page_num in range(key_pages):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    all_text.append(f"--- Page {page_num + 1} (PyMuPDF) ---\n{text}")
            
            doc.close()
            
            result = "\n".join(all_text)
            print(f"  [Pixtral Fallback] Extracted {len(result)} characters from {key_pages} pages")
            return result
            
        except Exception as e:
            import traceback
            print(f"  [Pixtral Fallback] Also failed: {e}")
            traceback.print_exc()
            return ""

    def _create_document_index(self) -> Dict:
        """Create document index using DocumentIndexer."""
        indexer = DocumentIndexer()
        index = indexer.create_index(
            structured_content=self.structured_content,
            pixtral_content=self.pixtral_content
        )
        
        return index

    def _extract_tables_and_figures(self):
        """Extract tables and figures from PDF."""
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(self.pdf_path, dpi=200)
            
            for i, image in enumerate(images):
                page_num = i + 1
                print(f"  📊 Page {page_num}/{len(images)}...")
                
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Extract tables
                tables = self._extract_tables_from_page(image_base64, page_num)
                self.extracted_tables.extend(tables)
                
                # Extract figures
                figures = self._extract_figures_from_page(image_base64, page_num)
                self.extracted_figures.extend(figures)
            
            print(f"  ✓ Extracted {len(self.extracted_tables)} tables, {len(self.extracted_figures)} figures")
            
        except Exception as e:
            print(f"  ⚠ Error: {e}")

    def _extract_tables_from_page(self, image_base64: str, page_num: int) -> List[Dict]:
        """Extract tables from a page."""
        prompt = """Extract ALL tables from this page EXACTLY as they appear.

Preserve exact structure, values, and formatting.

RESPOND WITH VALID JSON:
{
  "tables": [
    {
      "table_number": "Table 1",
      "title": "Exact title",
      "headers": ["Col1", "Col2"],
      "rows": [["val1", "val2"]],
      "footnotes": ["* footnote"]
    }
  ]
}"""

        try:
            response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                        ]
                    }
                ]
            )
            
            result_text = response.choices[0].message.content
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            tables = result.get("tables", [])
            
            for table in tables:
                table["page"] = page_num
            
            return tables
            
        except Exception as e:
            return []

    def _extract_figures_from_page(self, image_base64: str, page_num: int) -> List[Dict]:
        """Extract figures from a page."""
        prompt = """Extract ALL figures and charts from this page.

For survival curves, extract median survival, p-values, and survival rates.

RESPOND WITH VALID JSON:
{
  "figures": [
    {
      "figure_number": "Figure 1",
      "title": "Exact title",
      "type": "survival_curve",
      "p_value": "0.023",
      "groups": [{"name": "Treatment A", "median_survival_months": 24.1}]
    }
  ]
}"""

        try:
            response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                        ]
                    }
                ]
            )
            
            result_text = response.choices[0].message.content
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            if isinstance(result, dict):
                figures = result.get("figures", [])
            elif isinstance(result, list):
                figures = result
            else:
                figures = []
            
            for figure in figures:
                if isinstance(figure, dict):
                    figure["page"] = page_num
            
            return figures
            
        except Exception as e:
            return []

    def _extract_metadata_from_content(self) -> Dict:
        """Extract document metadata from structured content using AI."""
        try:
            # Combine first few sections for metadata extraction
            content_sample = ""
            section_count = 0
            for heading, content in self.structured_content.items():
                content_sample += f"{heading}\n{content}\n\n"
                section_count += 1
                if section_count >= 5:  # Use first 5 sections
                    break
            
            prompt = f"""Extract document metadata from this clinical trial document.

Document content:
{content_sample[:4000]}

Extract and return ONLY a valid JSON object with this structure:
{{
  "title": "full document title",
  "journal": "journal name",
  "volume": "volume number",
  "issue": "issue number", 
  "pages": "page range (e.g., 259-269)",
  "publication_date": "publication date (e.g., February 2022)",
  "online_date": "online publication date",
  "doi": "DOI link or identifier",
  "total_pages": number,
  "authors": ["author1", "author2"],
  "nct_number": "NCT number if mentioned",
  "trial_name": "trial name if mentioned",
  "disease_area": "disease/cancer type",
  "intervention": "main intervention/treatment"
}}

Return ONLY the JSON object, no other text."""

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            extracted = json.loads(result_text)
            
            # Create metadata extractor and populate it
            metadata_extractor = DocumentMetadataExtractor(extracted.get("title"))
            
            # Add document info
            metadata_extractor.extract_document_info(
                title=extracted.get("title", "Unknown"),
                authors=extracted.get("authors", []),
                total_pages=extracted.get("total_pages")
            )
            
            # Add publication info
            if extracted.get("journal"):
                metadata_extractor.extract_publication_info(
                    journal_name=extracted.get("journal"),
                    volume=extracted.get("volume"),
                    issue=extracted.get("issue"),
                    pages=extracted.get("pages"),
                    publication_date=extracted.get("publication_date"),
                    online_date=extracted.get("online_date"),
                    doi=extracted.get("doi")
                )
            
            # Add trial info if available
            if extracted.get("nct_number") or extracted.get("trial_name"):
                metadata_extractor.extract_trial_info(
                    nct_number=extracted.get("nct_number"),
                    trial_name=extracted.get("trial_name"),
                    disease_area=extracted.get("disease_area"),
                    intervention=extracted.get("intervention")
                )
            
            print(f"  ✓ Extracted document metadata")
            return metadata_extractor.get_metadata()
            
        except Exception as e:
            print(f"  ⚠ Metadata extraction failed: {e}")
            return {}

    def _save_structured_content(self) -> Path:
        """Save Mistral OCR structured content with metadata."""
        file_path = self.output_dir / f"{self.doc_name}_structured_content.json"
        
        # Extract metadata from content
        metadata = self._extract_metadata_from_content()
        
        data = {
            "document_metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "source": "mistral_ocr",
            "sections": self.structured_content
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path

    def _save_references(self) -> Path:
        """Save references section to separate file."""
        file_path = self.output_dir / f"{self.doc_name}_references.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "source": "mistral_ocr",
            "sections": self.references_content
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path

    def _save_pixtral_content(self) -> Path:
        """Save Pixtral detailed content."""
        file_path = self.output_dir / f"{self.doc_name}_pixtral_content.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.pixtral_content)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path

    def _save_document_index(self) -> Path:
        """Save document index."""
        file_path = self.output_dir / f"{self.doc_name}_document_index.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.document_index, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path

    def _save_tables_json(self) -> Path:
        """Save extracted tables as JSON."""
        file_path = self.output_dir / f"{self.doc_name}_tables.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_tables": len(self.extracted_tables),
            "total_figures": len(self.extracted_figures),
            "tables": self.extracted_tables,
            "figures": self.extracted_figures
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path

    def _save_tables_as_csv(self) -> List[str]:
        """Save each table as CSV."""
        csv_dir = self.output_dir / "tables_csv"
        csv_dir.mkdir(exist_ok=True)
        
        csv_files = []
        
        for i, table in enumerate(self.extracted_tables):
            table_number = table.get('table_number', f'Table_{i+1}')
            table_title = table.get('title', 'Untitled')
            page = table.get('page', 0)
            
            safe_title = "".join(c for c in table_title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            filename = f"Page{page}_{table_number.replace(' ', '_')}_{safe_title}.csv"
            filepath = csv_dir / filename
            
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    if table_title:
                        writer.writerow([f"# {table_title}"])
                    
                    headers = table.get('headers', [])
                    if headers:
                        writer.writerow(headers)
                    
                    rows = table.get('rows', [])
                    for row in rows:
                        if len(row) < len(headers):
                            row = row + [''] * (len(headers) - len(row))
                        elif len(row) > len(headers):
                            row = row[:len(headers)]
                        writer.writerow(row)
                    
                    footnotes = table.get('footnotes', [])
                    if footnotes:
                        writer.writerow([])
                        for footnote in footnotes:
                            writer.writerow([f"# {footnote}"])
                
                csv_files.append(str(filepath))
                
            except Exception as e:
                print(f"  ⚠ Error saving {filename}: {e}")
        
        print(f"  ✓ Saved {len(csv_files)} CSV files")
        return csv_files

    def _create_table_dictionaries(self) -> Path:
        """Create queryable table dictionaries."""
        file_path = self.output_dir / f"{self.doc_name}_table_dictionaries.json"
        
        # Convert tables to dictionaries
        tables_dict = {}
        for i, table in enumerate(self.extracted_tables):
            table_number = table.get('table_number', f'Table_{i+1}')
            table_key = table_number.replace(' ', '_').replace('.', '_')
            
            tables_dict[table_key] = {
                "metadata": {
                    "title": table.get('title', 'Untitled'),
                    "page": table.get('page', 0),
                    "table_number": table_number
                },
                "headers": table.get('headers', []),
                "data": {},
                "footnotes": table.get('footnotes', [])
            }
            
            # Convert rows to dict
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            
            for row in rows:
                if len(row) > 0:
                    row_label = row[0]
                    row_data = {}
                    for j, value in enumerate(row[1:], start=1):
                        if j < len(headers):
                            row_data[headers[j]] = value
                    tables_dict[table_key]["data"][row_label] = row_data
        
        # Create queryable index
        queryable_index = {
            "patient_characteristics": [],
            "outcomes": [],
            "adverse_events": [],
            "dosage": [],
            "other": []
        }
        
        for table_key, table_data in tables_dict.items():
            title_lower = table_data["metadata"]["title"].lower()
            
            if any(kw in title_lower for kw in ["baseline", "characteristic", "demographic"]):
                queryable_index["patient_characteristics"].append(table_key)
            elif any(kw in title_lower for kw in ["survival", "outcome", "response"]):
                queryable_index["outcomes"].append(table_key)
            elif any(kw in title_lower for kw in ["adverse", "toxicity", "safety"]):
                queryable_index["adverse_events"].append(table_key)
            elif any(kw in title_lower for kw in ["dose", "treatment", "regimen"]):
                queryable_index["dosage"].append(table_key)
            else:
                queryable_index["other"].append(table_key)
        
        output = {
            "tables_as_dicts": tables_dict,
            "queryable_index": queryable_index
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path

    def _save_processed_content_to_cache(self) -> Path:
        """Save complete processed content to cache directory."""
        # Save to cache/ directory instead of document folder
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        file_path = cache_dir / f"{self.doc_name}_processed_content.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "source_pdf": self.pdf_path,
            "structured_content": self.structured_content,
            "pixtral_content": self.pixtral_content,
            "document_index": self.document_index,
            "processing_metadata": {
                "mistral_sections": len(self.structured_content),
                "pixtral_content_length": len(self.pixtral_content),
                "total_paragraphs": self.document_index.get("metadata", {}).get("total_paragraphs", 0),
                "total_sentences": self.document_index.get("metadata", {}).get("total_sentences", 0),
                "total_tables": len(self.extracted_tables),
                "total_figures": len(self.extracted_figures)
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved to cache: {file_path.name}")
        return file_path

    def _generate_summary_report(self, generated_files: Dict) -> Path:
        """Generate human-readable summary report."""
        file_path = self.output_dir / f"{self.doc_name}_SUMMARY.txt"
        
        report = f"""
{'='*70}
DOCUMENT PROCESSING SUMMARY
{'='*70}

Document: {self.doc_name}
Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.output_dir}

{'='*70}
EXTRACTION STATISTICS
{'='*70}

Structured Content (Mistral OCR):
  - Sections extracted: {len(self.structured_content)}
  - Total characters: {sum(len(content) for content in self.structured_content.values())}

Detailed Content (Pixtral):
  - Pages processed: {self.pixtral_content.count('--- Page')}
  - Total characters: {len(self.pixtral_content)}

Document Index:
  - Total paragraphs: {self.document_index.get('metadata', {}).get('total_paragraphs', 0)}

Tables & Figures:
  - Tables extracted: {len(self.extracted_tables)}
  - Figures extracted: {len(self.extracted_figures)}

{'='*70}
GENERATED FILES
{'='*70}

"""
        
        for file_type, file_path in generated_files["files"].items():
            if isinstance(file_path, list):
                report += f"\n{file_type}:\n"
                for fp in file_path[:5]:  # Show first 5
                    report += f"  - {Path(fp).name}\n"
                if len(file_path) > 5:
                    report += f"  ... and {len(file_path) - 5} more\n"
            else:
                report += f"\n{file_type}:\n  - {Path(file_path).name}\n"
        
        report += f"""
{'='*70}
TABLES EXTRACTED
{'='*70}

"""
        
        for i, table in enumerate(self.extracted_tables, 1):
            title = table.get('title', 'Untitled')
            page = table.get('page', 0)
            rows = len(table.get('rows', []))
            cols = len(table.get('headers', []))
            report += f"{i}. {title}\n"
            report += f"   Page: {page}, Size: {rows} rows × {cols} columns\n\n"
        
        if self.extracted_figures:
            report += f"""
{'='*70}
FIGURES EXTRACTED
{'='*70}

"""
            for i, figure in enumerate(self.extracted_figures, 1):
                title = figure.get('title', 'Untitled')
                page = figure.get('page', 0)
                fig_type = figure.get('type', 'unknown')
                report += f"{i}. {title}\n"
                report += f"   Page: {page}, Type: {fig_type}\n\n"
        
        report += f"""
{'='*70}
NEXT STEPS
{'='*70}

1. Review CSV tables in: {self.output_dir}/tables_csv/
2. Use processed_content.json for RAG/extraction
3. Query table_dictionaries.json for structured data
4. Check SUMMARY.txt for overview

{'='*70}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  ✓ Saved: {Path(file_path).name}")
        return file_path

    def extract_study_profile(self) -> Dict:
        """
        Extract structured study profile from the processed document.
        
        This uses the StudyProfileExtractor to analyze the processed
        document files and extract structured clinical trial data.
        
        Returns:
            Dictionary with extracted study profile data
        """
        try:
            from .study_profile_extractor import StudyProfileExtractor
            
            extractor = StudyProfileExtractor()
            result = extractor.extract_from_processed_dir(self.output_dir)
            
            return result
            
        except Exception as e:
            print(f"  ⚠️  Study profile extraction error: {e}")
            return {"error": str(e)}
    
    def _save_study_profile(self, profile_result: Dict) -> Path:
        """Save extracted study profile to JSON file."""
        file_path = self.output_dir / f"{self.doc_name}_study_profile.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profile_result, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {file_path.name}")
        return file_path
