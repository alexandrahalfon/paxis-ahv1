#!/usr/bin/env python3
"""
Extract Tables to CSV

Extracts tables from PDFs and saves each table as a separate CSV file,
preserving the exact structure as it appears in the document.
"""

import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class TableToCSVExtractor:
    """Extract tables from PDF and save as CSV files."""

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
        self.timestamp = datetime.now().isoformat()
        
        print(f"✓ Initialized Table to CSV Extractor for: {os.path.basename(pdf_path)}")

    def extract_tables(self) -> List[Dict]:
        """Extract all tables from the PDF."""
        print("\n" + "="*70)
        print("TABLE EXTRACTION TO CSV")
        print("="*70)
        
        try:
            from pdf2image import convert_from_path
            import base64
            from io import BytesIO
            
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
            
            print(f"\n✓ Extraction complete!")
            print(f"  Total tables extracted: {len(self.extracted_tables)}")
            
            return self.extracted_tables
            
        except Exception as e:
            print(f"✗ Error during extraction: {e}")
            return []

    def _extract_tables_from_page(self, image_base64: str, page_num: int) -> List[Dict]:
        """Extract tables from a single page using Pixtral."""
        
        prompt = """Extract ALL tables from this page EXACTLY as they appear.

CRITICAL INSTRUCTIONS:

1. PRESERVE EXACT STRUCTURE
   - Extract EVERY row and EVERY column
   - Maintain exact cell alignment
   - Do NOT skip any cells
   - Do NOT merge cells unless they are merged in the original
   - Empty cells should be represented as empty strings ""

2. PRESERVE EXACT VALUES
   - Copy numbers EXACTLY as shown (including decimals)
   - Copy text EXACTLY as written
   - Preserve special characters (±, ≥, ≤, etc.)
   - Keep units attached to values (e.g., "60 mg/m²")
   - Preserve formatting like "(95% CI: 20.5-28.7)"

3. HEADERS
   - First row should be column headers
   - If table has row headers, include them as first column
   - Multi-line headers should be combined with space

4. FOOTNOTES
   - Extract all footnotes separately
   - Include footnote markers (*, †, ‡, etc.)

5. TABLE IDENTIFICATION
   - Extract table number/title if present
   - Note page location

RESPOND WITH VALID JSON (no markdown, no code blocks):
{
  "tables": [
    {
      "table_number": "Table 1",
      "title": "Exact table title from document",
      "page": 1,
      "headers": ["Column 1", "Column 2", "Column 3"],
      "rows": [
        ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
        ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
      ],
      "footnotes": ["* Footnote text", "† Another footnote"]
    }
  ]
}

If NO tables found, return: {"tables": []}

CRITICAL: Every row must have the SAME number of columns as headers."""

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
                    title = table.get('title', 'Untitled table')
                    rows = len(table.get('rows', []))
                    cols = len(table.get('headers', []))
                    print(f"    - {title} ({rows} rows × {cols} columns)")
            
            return tables
            
        except Exception as e:
            print(f"  ⚠ Error extracting tables from page {page_num}: {e}")
            return []

    def save_tables_as_csv(self, output_dir: str = "extracted_tables_csv") -> List[str]:
        """
        Save each extracted table as a separate CSV file.
        
        Returns:
            List of created CSV file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        doc_name = Path(self.pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_files = []
        
        print("\n" + "="*70)
        print("SAVING TABLES AS CSV")
        print("="*70)
        
        for i, table in enumerate(self.extracted_tables):
            # Create filename
            table_number = table.get('table_number', f'Table_{i+1}')
            table_title = table.get('title', 'Untitled')
            page = table.get('page', 0)
            
            # Sanitize filename
            safe_title = "".join(c for c in table_title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title[:50]  # Limit length
            
            filename = f"{doc_name}_Page{page}_{table_number.replace(' ', '_')}_{safe_title}.csv"
            filepath = output_path / filename
            
            # Write CSV
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write title as comment (if CSV reader supports it)
                    if table_title:
                        writer.writerow([f"# {table_title}"])
                    
                    # Write headers
                    headers = table.get('headers', [])
                    if headers:
                        writer.writerow(headers)
                    
                    # Write data rows
                    rows = table.get('rows', [])
                    for row in rows:
                        # Ensure row has same length as headers
                        if len(row) < len(headers):
                            row = row + [''] * (len(headers) - len(row))
                        elif len(row) > len(headers):
                            row = row[:len(headers)]
                        writer.writerow(row)
                    
                    # Write footnotes as comments
                    footnotes = table.get('footnotes', [])
                    if footnotes:
                        writer.writerow([])  # Empty row
                        for footnote in footnotes:
                            writer.writerow([f"# {footnote}"])
                
                csv_files.append(str(filepath))
                print(f"✓ Saved: {filename}")
                
            except Exception as e:
                print(f"✗ Error saving {filename}: {e}")
        
        return csv_files

    def save_metadata(self, output_dir: str = "extracted_tables_csv"):
        """Save metadata about all extracted tables."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        doc_name = Path(self.pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metadata_file = output_path / f"{doc_name}_tables_metadata_{timestamp}.json"
        
        metadata = {
            "timestamp": self.timestamp,
            "source_pdf": self.pdf_path,
            "total_tables": len(self.extracted_tables),
            "tables": []
        }
        
        for i, table in enumerate(self.extracted_tables):
            table_info = {
                "index": i + 1,
                "table_number": table.get('table_number', f'Table {i+1}'),
                "title": table.get('title', 'Untitled'),
                "page": table.get('page', 0),
                "dimensions": {
                    "rows": len(table.get('rows', [])),
                    "columns": len(table.get('headers', []))
                },
                "headers": table.get('headers', []),
                "has_footnotes": len(table.get('footnotes', [])) > 0
            }
            metadata["tables"].append(table_info)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Metadata saved to: {metadata_file}")


def main():
    """Main function to extract tables and save as CSV."""
    
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
        extractor = TableToCSVExtractor(pdf_path)
    except ValueError as e:
        print(f"✗ {e}")
        return
    
    # Extract tables
    tables = extractor.extract_tables()
    
    if tables:
        # Save as CSV files
        csv_files = extractor.save_tables_as_csv()
        
        # Save metadata
        extractor.save_metadata()
        
        # Print summary
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY")
        print("="*70)
        print(f"Total tables extracted: {len(tables)}")
        print(f"CSV files created: {len(csv_files)}")
        print(f"\nOutput directory: extracted_tables_csv/")
        
        print("\nExtracted tables:")
        for i, table in enumerate(tables, 1):
            title = table.get('title', 'Untitled')
            page = table.get('page', 0)
            rows = len(table.get('rows', []))
            cols = len(table.get('headers', []))
            print(f"  {i}. {title}")
            print(f"     Page: {page}, Size: {rows} rows × {cols} columns")


if __name__ == "__main__":
    main()
