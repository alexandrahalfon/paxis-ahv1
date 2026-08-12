#!/usr/bin/env python3
"""
Migration script to help transition from Colab workflow to organized pipeline.

This script helps users migrate their existing Colab-based workflow to the
new organized pipeline structure.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.colab_pipeline import ColabIngestionPipeline
from core.config import get_settings


def migrate_colab_output(
    colab_output_dir: Path,
    organized_output_dir: Path = None,
    run_ingestion: bool = True
) -> Dict[str, Any]:
    """
    Migrate Colab processing output to organized structure and run ingestion.
    
    Args:
        colab_output_dir: Directory with Colab-processed documents
        organized_output_dir: Target directory for organized output
        run_ingestion: Whether to run the ingestion pipeline
        
    Returns:
        Migration and ingestion statistics
    """
    if organized_output_dir is None:
        organized_output_dir = Path("processed_documents_organized")
    
    print("🔄 Migrating Colab output to organized structure...")
    print(f"  Source: {colab_output_dir}")
    print(f"  Target: {organized_output_dir}")
    
    # Create organized structure
    organized_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a single category for migrated documents
    category_dir = organized_output_dir / "migrated_documents"
    category_dir.mkdir(exist_ok=True)
    
    migrated_count = 0
    
    # Copy document directories
    for doc_dir in colab_output_dir.iterdir():
        if doc_dir.is_dir():
            target_dir = category_dir / doc_dir.name
            
            if target_dir.exists():
                print(f"  ⚠️ Skipping existing: {doc_dir.name}")
                continue
            
            print(f"  📁 Copying: {doc_dir.name}")
            shutil.copytree(doc_dir, target_dir)
            migrated_count += 1
    
    print(f"✅ Migrated {migrated_count} documents")
    
    migration_stats = {
        "migrated_documents": migrated_count,
        "source_dir": str(colab_output_dir),
        "target_dir": str(organized_output_dir)
    }
    
    if run_ingestion:
        print("\n🚀 Running ingestion pipeline on migrated documents...")
        
        # Initialize and run pipeline
        pipeline = ColabIngestionPipeline()
        
        ingestion_output = organized_output_dir.parent / "ingestion_output"
        
        ingestion_stats = pipeline.run_complete_pipeline(
            input_root=organized_output_dir,
            output_root=ingestion_output,
            recreate_collection=True
        )
        
        migration_stats["ingestion"] = ingestion_stats
    
    return migration_stats


def setup_keyword_dictionary(source_path: Path = None):
    """
    Set up keyword dictionary from Colab workflow.
    
    Args:
        source_path: Path to existing extractor_keywords.json
    """
    target_path = Path("data/keywords/extractor_keywords.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if source_path and source_path.exists():
        print(f"📋 Copying keyword dictionary from: {source_path}")
        shutil.copy2(source_path, target_path)
        print(f"✅ Keyword dictionary installed at: {target_path}")
    else:
        print("⚠️ No keyword dictionary provided. Creating example...")
        
        example_keywords = {
            "treatments": {
                "chemotherapy": [
                    "cisplatin", "carboplatin", "paclitaxel", "docetaxel",
                    "5-fluorouracil", "capecitabine", "oxaliplatin"
                ],
                "immunotherapy": [
                    "pembrolizumab", "nivolumab", "atezolizumab", 
                    "durvalumab", "ipilimumab"
                ],
                "targeted_therapy": [
                    "trastuzumab", "bevacizumab", "cetuximab", 
                    "panitumumab", "ramucirumab"
                ]
            },
            "outcomes": {
                "survival": [
                    "overall survival", "progression-free survival",
                    "disease-free survival", "median survival"
                ],
                "response": [
                    "complete response", "partial response", 
                    "stable disease", "progressive disease"
                ]
            },
            "adverse_events": {
                "hematologic": [
                    "neutropenia", "anemia", "thrombocytopenia"
                ],
                "gastrointestinal": [
                    "nausea", "vomiting", "diarrhea", "mucositis"
                ]
            }
        }
        
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(example_keywords, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Example keyword dictionary created at: {target_path}")
        print("   Please customize it for your domain!")


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate from Colab workflow")
    parser.add_argument("colab_output", help="Directory with Colab-processed documents")
    parser.add_argument("-o", "--output", help="Organized output directory")
    parser.add_argument("-k", "--keywords", help="Path to extractor_keywords.json")
    parser.add_argument("--no-ingestion", action="store_true", 
                       help="Skip ingestion pipeline")
    
    args = parser.parse_args()
    
    colab_output = Path(args.colab_output)
    if not colab_output.exists():
        print(f"❌ Colab output directory not found: {colab_output}")
        sys.exit(1)
    
    # Set up keyword dictionary first
    keyword_path = Path(args.keywords) if args.keywords else None
    setup_keyword_dictionary(keyword_path)
    
    # Run migration
    try:
        organized_output = Path(args.output) if args.output else None
        
        stats = migrate_colab_output(
            colab_output_dir=colab_output,
            organized_output_dir=organized_output,
            run_ingestion=not args.no_ingestion
        )
        
        print("\n" + "="*70)
        print("✅ MIGRATION COMPLETE!")
        print("="*70)
        
        print(f"Migrated documents: {stats['migrated_documents']}")
        print(f"Organized structure: {stats['target_dir']}")
        
        if "ingestion" in stats:
            ingestion = stats["ingestion"]
            final_stats = ingestion.get("final_stats", {})
            collection_info = final_stats.get("collection_info", {})
            
            if "points_count" in collection_info:
                print(f"Vector database points: {collection_info['points_count']}")
        
        print("\n🎉 Your documents are now ready for retrieval and generation!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()