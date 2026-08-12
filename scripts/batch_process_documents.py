#!/usr/bin/env python3
"""
Batch Document Processing Script

Processes multiple PDF documents using the document processor.
Equivalent to the Colab batch processing workflow.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.document_processor import CompleteDocumentProcessor
from core.config import get_settings, ensure_directories


class BatchDocumentProcessor:
    """Batch process multiple PDF documents."""
    
    def __init__(self, input_dir: Path, output_root: Path = None):
        """
        Initialize batch processor.
        
        Args:
            input_dir: Directory containing PDF files
            output_root: Root directory for processed outputs
        """
        self.input_dir = Path(input_dir)
        self.output_root = Path(output_root) if output_root else Path("processed_documents")
        
        ensure_directories()
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Initialized BatchDocumentProcessor")
        print(f"  Input: {self.input_dir}")
        print(f"  Output: {self.output_root}")
    
    def process_all_documents(
        self, 
        skip_duplicates: bool = True,
        file_pattern: str = "*.pdf"
    ) -> Dict[str, Any]:
        """
        Process all PDF documents in the input directory.
        
        Args:
            skip_duplicates: Whether to skip already processed documents
            file_pattern: File pattern to match (default: *.pdf)
            
        Returns:
            Processing statistics
        """
        # Find all PDF files
        pdf_files = sorted(list(self.input_dir.glob(file_pattern)))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {self.input_dir}")
            return {"total_docs": 0, "processed": 0, "skipped": 0, "failed": 0}
        
        print(f"\n📄 Found {len(pdf_files)} PDF files to process")
        print("="*70)
        
        stats = {
            "total_docs": len(pdf_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "results": []
        }
        
        for idx, pdf_path in enumerate(pdf_files, start=1):
            print(f"\n📄 ({idx}/{len(pdf_files)}) Processing: {pdf_path.name}")
            print("="*70)
            
            doc_stem = pdf_path.stem
            output_dir = self.output_root / doc_stem
            
            # Check for duplicates
            if skip_duplicates and output_dir.exists():
                print(f"⚠️ Duplicate detected: {doc_stem} — Skipping.")
                stats["skipped"] += 1
                stats["results"].append({
                    "document": doc_stem,
                    "status": "skipped",
                    "reason": "duplicate"
                })
                continue
            
            # Process document
            try:
                processor = CompleteDocumentProcessor(
                    pdf_path=str(pdf_path),
                    output_dir=str(self.output_root)
                )
                
                result = processor.process_complete()
                
                stats["processed"] += 1
                stats["results"].append({
                    "document": doc_stem,
                    "status": "success",
                    "files": result.get("files", {}),
                    "output_directory": result.get("output_directory")
                })
                
                print(f"✅ Successfully processed: {doc_stem}")
                
            except Exception as e:
                print(f"❌ Failed to process {doc_stem}: {e}")
                stats["failed"] += 1
                stats["results"].append({
                    "document": doc_stem,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Print final summary
        self._print_batch_summary(stats)
        
        return stats
    
    def process_single_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing result
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 Processing single document: {pdf_path.name}")
        
        try:
            processor = CompleteDocumentProcessor(
                pdf_path=str(pdf_path),
                output_dir=str(self.output_root)
            )
            
            result = processor.process_complete()
            
            print(f"✅ Successfully processed: {pdf_path.stem}")
            return {
                "status": "success",
                "document": pdf_path.stem,
                "result": result
            }
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.stem}: {e}")
            return {
                "status": "failed",
                "document": pdf_path.stem,
                "error": str(e)
            }
    
    def _print_batch_summary(self, stats: Dict[str, Any]):
        """Print batch processing summary."""
        print("\n" + "="*70)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*70)
        
        print(f"Total documents: {stats['total_docs']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Skipped (duplicates): {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['processed'] > 0:
            print(f"\n✅ Processing complete!")
            print(f"📁 Output directory: {self.output_root}")
        
        if stats['failed'] > 0:
            print(f"\n⚠️ {stats['failed']} documents failed to process:")
            for result in stats['results']:
                if result['status'] == 'failed':
                    print(f"  - {result['document']}: {result.get('error', 'Unknown error')}")


def main():
    """Main function for batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process PDF documents")
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument("-o", "--output", help="Output directory (default: processed_documents)")
    parser.add_argument("--no-skip-duplicates", action="store_true", 
                       help="Process all documents, even if already processed")
    parser.add_argument("--pattern", default="*.pdf", 
                       help="File pattern to match (default: *.pdf)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output) if args.output else None
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    try:
        processor = BatchDocumentProcessor(
            input_dir=input_dir,
            output_root=output_dir
        )
        
        stats = processor.process_all_documents(
            skip_duplicates=not args.no_skip_duplicates,
            file_pattern=args.pattern
        )
        
        if stats['failed'] > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()