#!/usr/bin/env python3
"""
Run the Colab-style ingestion pipeline.

This script runs the exact same pipeline as your Colab notebook,
but in the organized repository structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.colab_pipeline import ColabIngestionPipeline
from core.config import get_settings, validate_settings


def main():
    """Main function for running the Colab-style pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Colab-style ingestion pipeline")
    parser.add_argument("input_root", help="Root directory with processed documents")
    parser.add_argument("output_root", help="Output directory for pipeline results")
    parser.add_argument("-k", "--keywords", help="Path to extractor_keywords.json")
    parser.add_argument("--no-recreate", action="store_true", 
                       help="Don't recreate Qdrant collection")
    
    args = parser.parse_args()
    
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    
    if not input_root.exists():
        print(f"❌ Input directory not found: {input_root}")
        sys.exit(1)
    
    # Validate configuration
    try:
        validate_settings()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Set up keyword path
    keyword_path = None
    if args.keywords:
        keyword_path = Path(args.keywords)
        if not keyword_path.exists():
            print(f"❌ Keyword file not found: {keyword_path}")
            sys.exit(1)
    
    # Initialize and run pipeline
    try:
        pipeline = ColabIngestionPipeline(keyword_json_path=keyword_path)
        
        stats = pipeline.run_complete_pipeline(
            input_root=input_root,
            output_root=output_root,
            recreate_collection=not args.no_recreate
        )
        
        print("\n" + "="*70)
        print("✅ COLAB PIPELINE COMPLETE!")
        print("="*70)
        
        # Print summary
        stages = stats.get("stages", {})
        
        if "normalize" in stages:
            normalize_stats = stages["normalize"]
            print(f"Documents processed: {normalize_stats.get('docs_processed', 0)}")
            print(f"Initial chunks: {normalize_stats.get('chunks_written', 0)}")
        
        if "windowing" in stages:
            window_stats = stages["windowing"]
            print(f"Section windows: {window_stats.get('total_output_chunks', 0)}")
        
        if "ingestion" in stages:
            ingest_stats = stages["ingestion"]
            print(f"Points ingested: {ingest_stats.get('total_points', 0)}")
        
        settings = get_settings()
        print(f"\n🎉 Your documents are now in Qdrant!")
        print(f"Collection: {settings.qdrant_collection}")
        print(f"Endpoint: {settings.qdrant_url}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()