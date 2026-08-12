#!/usr/bin/env python3
"""
Figure and Table Extractor Implementation

Extracts tables, figures, and charts from clinical trial PDFs using Pixtral vision model.
"""

import json
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()


class FigureTableExtractor:
    """Extract figures and tables from PDF using Pixtral vision model."""

    def __init__(self, pdf_path: str):
        """Initialize with PDF path."""
        from mistralai import Mistral
        
        mistral_api_key = os.getenv('MISTRAL_API_KEY')
        if not mistral_api_key:
            raise ValueError("Mistral API key not found. Please set MISTRAL_API_KEY in .env file.")
        
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.mistral_model = os.getenv('MISTRAL_MODEL', 'pixtral-large-latest')
        self.pdf_path = pdf_path
        self.extracted_tables = []
        self.extracted_figures = []
        self.timestamp = datetime.now().isoformat()
        
        print(f"✓ Initialized Figure/Table Extractor for: {os.path.basename(pdf_path)}")

    def extract_all(self) -> Dict:
        """Extract all tables and figures from the PDF."""
        print("\n" + "="*70)
        print("FIGURE AND TABLE EXTRACTION")
        print("="*70)
        
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            print(f"\n📄 Converting PDF to images...")
            images = convert_from_path(self.pdf_path, dpi=200)
            print(f"✓ Converted to {len(images)} pages")
            
            # Process each page
            for i, image in enumerate(images):
                page_num = i + 1
                print(f"\n📊 Processing page {page_num}/{len(images)}...")
                
                # Convert image to base64
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Extract tables from this page
                tables = self._extract_tables_from_page(image_base64, page_num)
                self.extracted_tables.extend(tables)
                
                # Extract figures from this page
                figures = self._extract_figures_from_page(image_base64, page_num)
                self.extracted_figures.extend(figures)
            
            print(f"\n✓ Extraction complete!")
            print(f"  Tables extracted: {len(self.extracted_tables)}")
            print(f"  Figures extracted: {len(self.extracted_figures)}")
            
            return self._create_output()
            
        except Exception as e:
            print(f"✗ Error during extraction: {e}")
            return {}

    def _extract_tables_from_page(self, image_base64: str, page_num: int) -> List[Dict]:
        """Extract tables from a single page using Pixtral."""
        
        prompt = """Analyze this page from a clinical trial document and extract ALL tables.

For EACH table found, provide:

1. TABLE IDENTIFICATION
   - Table number or identifier
   - Table title/caption
   - Page location (top/middle/bottom)

2. TABLE STRUCTURE
   - Column headers (exact text)
   - All row data (preserve exact values)
   - Units for each column (mg, mg/m², %, months, etc.)

3. CRITICAL: PRESERVE EXACT VALUES
   - Do NOT round numbers (60.5 NOT ~60)
   - Do NOT summarize ranges (60-80 NOT "around 60-80")
   - Include all decimal places shown
   - Preserve statistical notations (p<0.001, CI: 24.1-28.5)

4. METADATA
   - Footnotes (*, †, ‡, etc.)
   - Table notes
   - Abbreviations explained

5. CONTEXT
   - What type of data (dosage, outcomes, patient characteristics, etc.)
   - Number of patients (n) if shown

Respond with ONLY valid JSON (no markdown, no code blocks):
{
  "tables": [
    {
      "table_number": "Table 1",
      "title": "exact title",
      "type": "dosage|outcomes|patient_characteristics|adverse_events|other",
      "headers": ["Header 1", "Header 2", "Header 3"],
      "units": {"Header 1": "mg/m²", "Header 2": "%"},
      "rows": [
        ["Row 1 Label", "Value 1", "Value 2"],
        ["Row 2 Label", "Value 1", "Value 2"]
      ],
      "footnotes": ["* footnote text"],
      "sample_size": "n=123",
      "confidence": 0.95
    }
  ]
}

If NO tables found on this page, return: {"tables": []}"""

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
            
            # Clean up markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            tables = result.get("tables", [])
            
            # Add page number to each table
            for table in tables:
                table["page"] = page_num
            
            if tables:
                print(f"  ✓ Found {len(tables)} table(s) on page {page_num}")
                for table in tables:
                    print(f"    - {table.get('title', 'Untitled table')}")
            
            return tables
            
        except Exception as e:
            print(f"  ⚠ Error extracting tables from page {page_num}: {e}")
            return []

    def _extract_figures_from_page(self, image_base64: str, page_num: int) -> List[Dict]:
        """Extract figures and charts from a single page using Pixtral."""
        
        prompt = """Analyze this page from a clinical trial document and extract ALL figures, charts, and graphs.

For EACH figure/chart found, provide:

1. FIGURE IDENTIFICATION
   - Figure number
   - Figure title/caption
   - Type: survival_curve|bar_chart|line_plot|forest_plot|scatter_plot|diagram|other

2. FOR SURVIVAL CURVES (Kaplan-Meier):
   - Treatment groups being compared
   - Median survival time for each group (exact months)
   - Survival rates at key timepoints (6mo, 12mo, 24mo)
   - P-value from log-rank test
   - Number at risk (n) at timepoints
   - Confidence intervals if shown

3. FOR BAR CHARTS:
   - Category labels
   - Exact values/percentages for each bar
   - Error bars if present
   - Statistical significance markers

4. FOR LINE PLOTS:
   - X-axis label and values
   - Y-axis label and values
   - Data points visible
   - Legend groups

5. AXIS INFORMATION:
   - X-axis: label, units, range
   - Y-axis: label, units, range
   - Scale type (linear/log)

6. EXTRACTED DATA:
   - All numeric values visible
   - Statistical annotations
   - Legend text

Respond with ONLY valid JSON (no markdown, no code blocks):
{
  "figures": [
    {
      "figure_number": "Figure 1",
      "title": "exact title",
      "type": "survival_curve|bar_chart|line_plot|other",
      "caption": "full caption text",
      "x_axis": {"label": "Time (months)", "range": "0-60"},
      "y_axis": {"label": "Survival probability", "range": "0-1.0"},
      "groups": [
        {
          "name": "Treatment A",
          "median_survival_months": 24.1,
          "survival_rates": {"6mo": 0.95, "12mo": 0.78, "24mo": 0.42}
        }
      ],
      "p_value": "0.023",
      "extracted_values": {},
      "confidence": 0.90
    }
  ]
}

If NO figures found on this page, return: {"figures": []}"""

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
            
            # Clean up markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            figures = result.get("figures", [])
            
            # Add page number to each figure
            for figure in figures:
                figure["page"] = page_num
            
            if figures:
                print(f"  ✓ Found {len(figures)} figure(s) on page {page_num}")
                for figure in figures:
                    print(f"    - {figure.get('title', 'Untitled figure')} ({figure.get('type', 'unknown')})")
            
            return figures
            
        except Exception as e:
            print(f"  ⚠ Error extracting figures from page {page_num}: {e}")
            return []

    def _create_output(self) -> Dict:
        """Create structured output with all extracted data."""
        
        return {
            "timestamp": self.timestamp,
            "source_pdf": self.pdf_path,
            "extraction_summary": {
                "total_tables": len(self.extracted_tables),
                "total_figures": len(self.extracted_figures),
                "tables_by_type": self._count_by_type(self.extracted_tables, "type"),
                "figures_by_type": self._count_by_type(self.extracted_figures, "type")
            },
            "tables": self.extracted_tables,
            "figures": self.extracted_figures
        }

    def _count_by_type(self, items: List[Dict], type_key: str) -> Dict:
        """Count items by their type."""
        counts = {}
        for item in items:
            item_type = item.get(type_key, "unknown")
            counts[item_type] = counts.get(item_type, 0) + 1
        return counts

    def save_to_json(self, output_path: str) -> bool:
        """Save extracted data to JSON file."""
        try:
            output_data = self._create_output()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"\n✗ Error saving results: {e}")
            return False


def main():
    """Main function to run figure/table extraction."""
    
    # Get PDF path from environment or use provided path
    pdf_path = os.getenv('DOCUMENT_PATH', 
        "/Users/ahalfon/Downloads/References for RAG Questions/Trastuzumab with trimodality treatment for esophageal adenocarcinoma with HER2 overexpression - Q76.pdf")
    
    if pdf_path.startswith('"') and pdf_path.endswith('"'):
        pdf_path = pdf_path[1:-1]
    
    if not os.path.exists(pdf_path):
        print(f"✗ PDF file not found: {pdf_path}")
        return
    
    # Initialize extractor
    try:
        extractor = FigureTableExtractor(pdf_path)
    except ValueError as e:
        print(f"✗ {e}")
        return
    
    # Extract all tables and figures
    results = extractor.extract_all()
    
    if results:
        # Save results
        output_dir = Path("extracted_figures_tables")
        output_dir.mkdir(exist_ok=True)
        
        doc_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{doc_name}_figures_tables_{timestamp}.json"
        
        extractor.save_to_json(str(output_file))
        
        # Print summary
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY")
        print("="*70)
        print(f"Tables extracted: {len(extractor.extracted_tables)}")
        print(f"Figures extracted: {len(extractor.extracted_figures)}")
        
        if extractor.extracted_tables:
            print("\nTables:")
            for table in extractor.extracted_tables:
                print(f"  - Page {table['page']}: {table.get('title', 'Untitled')} ({table.get('type', 'unknown')})")
        
        if extractor.extracted_figures:
            print("\nFigures:")
            for figure in extractor.extracted_figures:
                print(f"  - Page {figure['page']}: {figure.get('title', 'Untitled')} ({figure.get('type', 'unknown')})")


if __name__ == "__main__":
    main()
