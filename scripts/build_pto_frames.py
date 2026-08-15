#!/usr/bin/env python3
"""
Build PTO (Patient→Treatment→Outcome) frames from ingested chunks.

This script extracts structured PTO relationships from your tagged chunks
using existing keyword metadata - no LLM calls required.

Usage:
    # Basic - build frames from tagged chunks
    python scripts/build_pto_frames.py ingestion_output/03_tagged_chunks.jsonl
    
    # With custom output path
    python scripts/build_pto_frames.py ingestion_output/03_tagged_chunks.jsonl -o pto_frames.jsonl
    
    # Only high/medium confidence frames
    python scripts/build_pto_frames.py ingestion_output/03_tagged_chunks.jsonl --min-confidence medium
    
    # Build and upsert to Qdrant
    python scripts/build_pto_frames.py ingestion_output/03_tagged_chunks.jsonl --upsert

Author: Built for Paxis RAG platform
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build PTO frames from tagged chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingestion_output/03_tagged_chunks.jsonl
  %(prog)s ingestion_output/03_tagged_chunks.jsonl --min-confidence medium
  %(prog)s ingestion_output/03_tagged_chunks.jsonl --upsert
        """
    )
    parser.add_argument(
        "input_jsonl", 
        type=Path, 
        help="Path to tagged chunks JSONL (e.g., 03_tagged_chunks.jsonl)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        help="Output JSONL path (default: input_dir/05_pto_frames.jsonl)"
    )
    parser.add_argument(
        "--min-confidence", 
        choices=["low", "medium", "high"],
        default="low", 
        help="Minimum confidence threshold (default: low)"
    )
    parser.add_argument(
        "--upsert", 
        action="store_true", 
        help="Upsert frames to Qdrant after building"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build frames but don't save or upsert"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_jsonl.exists():
        print(f"❌ Input file not found: {args.input_jsonl}")
        sys.exit(1)
    
    # Default output path
    output_path = args.output or args.input_jsonl.parent / "05_pto_frames.jsonl"
    
    # Import after arg parsing to speed up --help
    from ingestion.pto_frame_builder import PTOFrameBuilder, QdrantFrameUpserter
    
    print("\n" + "=" * 60)
    print("PTO FRAME BUILDER")
    print("=" * 60)
    print(f"Input:  {args.input_jsonl}")
    print(f"Output: {output_path}")
    print(f"Min confidence: {args.min_confidence}")
    print("=" * 60)
    
    # Build frames
    builder = PTOFrameBuilder()
    
    print(f"\n📂 Loading chunks...")
    chunks_by_doc = builder.load_chunks(args.input_jsonl)
    
    print(f"\n🔧 Building PTO frames...")
    frames = builder.build_all_frames(chunks_by_doc, min_confidence=args.min_confidence)
    
    # Print summary
    builder.print_summary(frames)
    
    if args.dry_run:
        print("\n⚠️  Dry run - not saving or upserting")
        return
    
    # Save frames
    print(f"\n💾 Saving frames to: {output_path}")
    builder.save_frames(frames, output_path)
    
    # Optionally upsert to Qdrant
    if args.upsert:
        print(f"\n☁️  Upserting to Qdrant...")
        
        try:
            from core.config import get_settings
            settings = get_settings()
            
            upserter = QdrantFrameUpserter(
                qdrant_url=settings.qdrant_url,
                qdrant_api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection,
                openai_api_key=settings.openai_api_key
            )
            upserter.ensure_payload_index()
            upserter.upsert_frames(frames)
            
        except ImportError:
            print("❌ Could not import settings. Make sure you have:")
            print("   - src/core/config.py with get_settings()")
            print("   - .env file with QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY")
            print("\nAlternatively, run with explicit credentials:")
            print("   python pto_frame_builder.py input.jsonl output.jsonl --upsert \\")
            print("       --qdrant-url ... --qdrant-api-key ... --openai-api-key ...")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            sys.exit(1)
    
    print("\n✅ Done!")
    
    # Print next steps
    print("\n" + "-" * 60)
    print("NEXT STEPS:")
    print("-" * 60)
    if not args.upsert:
        print("1. Review the generated frames:")
        print(f"   head -5 {output_path} | python -m json.tool")
        print("\n2. Upsert to Qdrant:")
        print(f"   python scripts/build_pto_frames.py {args.input_jsonl} --upsert")
    else:
        print("1. Test PTO retrieval:")
        print("   python -c \"from src.api.services.pto_retriever import PTOQueryRouter; \\")
        print("              r = PTOQueryRouter(); \\")
        print("              print(r.analyze_query('treatment for stage II breast cancer'))\"")
        print("\n2. Query via API:")
        print("   curl -X POST http://localhost:8000/api/query/query \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"question\": \"What is the treatment for T1N1 breast cancer?\"}'")
    print("-" * 60)


if __name__ == "__main__":
    main()
