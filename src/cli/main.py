#!/usr/bin/env python3
"""
Paxis Document Processing CLI

Main entry point for document processing and ingestion.
"""

import sys
import argparse
from pathlib import Path

# Import from processing module
# (Your actual process_document_complete code should be refactored here)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process PDF documents and ingest into knowledge base"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF document to process"
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip automatic ingestion"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_documents"),
        help="Output directory for processed documents"
    )
    
    args = parser.parse_args()
    
    print(f"Processing: {args.pdf_path}")
    
    # TODO: Import and call your actual processing code
    # from src.processing.process_document_complete import process_document
    # process_document(args.pdf_path, args.output_dir, auto_ingest=not args.no_ingest)
    
    print("✓ Processing complete")

if __name__ == "__main__":
    main()
