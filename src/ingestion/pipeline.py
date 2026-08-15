"""
Complete document ingestion pipeline.

Orchestrates the full process from processed documents to vector storage:
1. Document chunking and normalization
2. Keyword tagging
3. Embedding generation
4. Vector database ingestion
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.config import get_settings, ensure_directories
from .chunk_processor import ChunkProcessor
from .keyword_tagger import KeywordTagger
from .embeddings import EmbeddingGenerator
from .qdrant_client import QdrantIngestionClient


class IngestionPipeline:
    """Complete document ingestion pipeline."""
    
    def __init__(self, keyword_json_path: Path = None):
        """
        Initialize ingestion pipeline.
        
        Args:
            keyword_json_path: Path to keyword dictionary JSON
        """
        self.settings = get_settings()
        ensure_directories()
        
        # Initialize components
        self.chunk_processor = ChunkProcessor()
        self.keyword_tagger = KeywordTagger(keyword_json_path)
        self.embedding_generator = EmbeddingGenerator()
        self.qdrant_client = QdrantIngestionClient()
        
        print("✓ Initialized IngestionPipeline")
    
    def run_complete_pipeline(
        self,
        input_root: Path,
        output_dir: Path = None,
        recreate_collection: bool = False,
        include_references: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.
        
        Args:
            input_root: Root directory with processed documents
            output_dir: Output directory for intermediate files
            recreate_collection: Whether to recreate Qdrant collection
            include_references: Whether to include reference sections
            
        Returns:
            Pipeline execution statistics
        """
        if output_dir is None:
            output_dir = Path("ingestion_output")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("DOCUMENT INGESTION PIPELINE")
        print("="*70)
        print(f"Input: {input_root}")
        print(f"Output: {output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        pipeline_stats = {
            "timestamp": datetime.now().isoformat(),
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "settings": {
                "embed_model": self.settings.embed_model,
                "embed_dim": self.settings.embed_dim,
                "max_tokens_per_chunk": self.settings.max_tokens_per_chunk,
                "qdrant_collection": self.settings.qdrant_collection,
            },
            "stages": {}
        }
        
        try:
            # Stage 1: Document Chunking
            print("\n📦 STAGE 1: Document Chunking and Normalization")
            print("-" * 50)
            
            chunks_jsonl = output_dir / "01_chunks.jsonl"
            chunk_stats = self.chunk_processor.process_documents_to_chunks(
                input_root=input_root,
                output_jsonl=chunks_jsonl,
                include_references=include_references
            )
            pipeline_stats["stages"]["chunking"] = chunk_stats
            
            # Stage 2: Section Windowing
            print("\n🪟 STAGE 2: Section Window Creation")
            print("-" * 50)
            
            windows_jsonl = output_dir / "02_section_windows.jsonl"
            window_stats = self.chunk_processor.create_section_windows(
                input_jsonl=chunks_jsonl,
                output_jsonl=windows_jsonl
            )
            pipeline_stats["stages"]["windowing"] = window_stats
            
            # Stage 3: Keyword Tagging
            print("\n🏷️  STAGE 3: Keyword Tagging")
            print("-" * 50)
            
            tagged_jsonl = output_dir / "03_tagged_chunks.jsonl"
            tag_stats = self.keyword_tagger.tag_chunks_in_jsonl(
                input_jsonl=windows_jsonl,
                output_jsonl=tagged_jsonl,
                add_to_text_for_embedding=False  # Keep embeddings clean
            )
            pipeline_stats["stages"]["tagging"] = tag_stats
            
            # Stage 4: Embedding Generation
            print("\n🔢 STAGE 4: Embedding Generation")
            print("-" * 50)
            
            embedded_jsonl = output_dir / "04_embedded_chunks.jsonl"
            embed_stats = self.embedding_generator.embed_and_save_jsonl(
                input_jsonl=tagged_jsonl,
                output_jsonl=embedded_jsonl
            )
            pipeline_stats["stages"]["embedding"] = embed_stats
            
            # Stage 5: Qdrant Ingestion
            print("\n🗄️  STAGE 5: Vector Database Ingestion")
            print("-" * 50)
            
            # Ensure collection exists
            self.qdrant_client.ensure_collection(recreate=recreate_collection)
            
            # Ingest chunks
            ingest_stats = self.qdrant_client.ingest_chunks_from_jsonl(embedded_jsonl)
            pipeline_stats["stages"]["ingestion"] = ingest_stats
            
            # Stage 6: Final Statistics
            print("\n📊 STAGE 6: Final Statistics")
            print("-" * 50)
            
            collection_info = self.qdrant_client.get_collection_info()
            keyword_stats = self.keyword_tagger.get_category_stats(tagged_jsonl)
            
            pipeline_stats["final_stats"] = {
                "collection_info": collection_info,
                "keyword_category_stats": keyword_stats,
            }
            
            # Save pipeline stats
            stats_file = output_dir / "pipeline_stats.json"
            with stats_file.open("w", encoding="utf-8") as f:
                json.dump(pipeline_stats, f, indent=2, ensure_ascii=False)
            
            print("\n" + "="*70)
            print("✅ INGESTION PIPELINE COMPLETE!")
            print("="*70)
            
            self._print_final_summary(pipeline_stats)
            
            return pipeline_stats
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            pipeline_stats["error"] = str(e)
            raise
    
    def run_single_document(
        self,
        doc_dir: Path,
        output_dir: Path = None,
        recreate_collection: bool = False
    ) -> Dict[str, Any]:
        """
        Run ingestion pipeline for a single document.
        
        Args:
            doc_dir: Path to processed document directory
            output_dir: Output directory for intermediate files
            recreate_collection: Whether to recreate Qdrant collection
            
        Returns:
            Processing statistics
        """
        if output_dir is None:
            output_dir = Path("single_doc_ingestion") / doc_dir.name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Processing single document: {doc_dir.name}")
        
        # Create temporary input structure
        temp_input = output_dir / "temp_input"
        temp_input.mkdir(exist_ok=True)
        
        # Create category directory and symlink document
        category_dir = temp_input / "single_doc"
        category_dir.mkdir(exist_ok=True)
        
        doc_link = category_dir / doc_dir.name
        if not doc_link.exists():
            doc_link.symlink_to(doc_dir.absolute())
        
        try:
            # Run pipeline on single document
            stats = self.run_complete_pipeline(
                input_root=temp_input,
                output_dir=output_dir,
                recreate_collection=recreate_collection
            )
            
            return stats
            
        finally:
            # Clean up temporary structure
            if doc_link.exists():
                doc_link.unlink()
            if category_dir.exists():
                category_dir.rmdir()
            if temp_input.exists():
                temp_input.rmdir()
    
    def _print_final_summary(self, stats: Dict[str, Any]):
        """Print final pipeline summary."""
        stages = stats.get("stages", {})
        
        print(f"\n📈 PIPELINE SUMMARY")
        print(f"{'='*50}")
        
        # Chunking stats
        if "chunking" in stages:
            chunk_stats = stages["chunking"]
            print(f"Documents processed: {chunk_stats.get('docs_processed', 0)}")
            print(f"Initial chunks created: {chunk_stats.get('chunks_created', 0)}")
        
        # Windowing stats
        if "windowing" in stages:
            window_stats = stages["windowing"]
            print(f"Section windows created: {window_stats.get('total_output_chunks', 0)}")
        
        # Embedding stats
        if "embedding" in stages:
            embed_stats = stages["embedding"]
            print(f"Chunks embedded: {embed_stats.get('embedded_chunks', 0)}")
            print(f"Embedding success rate: {embed_stats.get('success_rate', 0):.2%}")
        
        # Ingestion stats
        if "ingestion" in stages:
            ingest_stats = stages["ingestion"]
            print(f"Points ingested: {ingest_stats.get('successful_points', 0)}")
            print(f"Ingestion success rate: {ingest_stats.get('success_rate', 0):.2%}")
        
        # Collection info
        final_stats = stats.get("final_stats", {})
        collection_info = final_stats.get("collection_info", {})
        if collection_info and "points_count" in collection_info:
            print(f"Total points in collection: {collection_info['points_count']}")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Collection: {self.settings.qdrant_collection}")
        print(f"Endpoint: {self.settings.qdrant_url}")


def main():
    """Main function for running the ingestion pipeline."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pipeline <input_root> [output_dir]")
        print("Example: python -m src.ingestion.pipeline /path/to/processed_documents")
        return
    
    input_root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_root.exists():
        print(f"❌ Input directory not found: {input_root}")
        return
    
    # Initialize and run pipeline
    pipeline = IngestionPipeline()
    
    try:
        stats = pipeline.run_complete_pipeline(
            input_root=input_root,
            output_dir=output_dir,
            recreate_collection=True  # Recreate for fresh start
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()