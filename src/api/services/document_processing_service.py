"""
Service for processing approved documents through the full pipeline.
"""

from pathlib import Path
from typing import Dict, Any

from ...core.config import get_settings
from ...processing.document_processor import CompleteDocumentProcessor, persist_study_profile_if_present
from ...ingestion.colab_pipeline import ColabIngestionPipeline


class DocumentProcessingService:
    """Service for processing documents through the full pipeline."""
    
    def __init__(self):
        """Initialize processing service."""
        self.settings = get_settings()
    
    async def process_document(self, file_path: Path, upload_id: str) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline:
        1. Document processing (OCR, vision, tables, figures)
        2. Ingestion (chunking, embedding, Qdrant storage)
        
        Args:
            file_path: Path to the PDF file
            upload_id: Upload ID for tracking
            
        Returns:
            Processing results and statistics
        """
        try:
            # Step 1: Document Processing
            print(f"\n[Processing {upload_id}] Starting document processing...")
            processor = CompleteDocumentProcessor(str(file_path))
            processing_result = processor.process_complete()

            # Persist any study profile process_complete() extracted (Phase
            # 10, gated by settings.extract_study_profiles) -- a real
            # await now, not a second event loop nested inside this one.
            # See persist_study_profile_if_present()'s docstring.
            await persist_study_profile_if_present(processing_result, processor.doc_name)

            # Step 2: Ingestion Pipeline
            print(f"\n[Processing {upload_id}] Starting ingestion pipeline...")
            
            # Set up paths for ingestion
            processed_doc_dir = Path(processing_result['output_directory'])
            doc_name = processor.doc_name
            
            # Create temporary structure for single document ingestion
            temp_input_root = Path("temp_ingestion_input")
            temp_category_dir = temp_input_root / "single_document"
            temp_category_dir.mkdir(parents=True, exist_ok=True)
            
            # Create symlink to processed document
            doc_symlink = temp_category_dir / doc_name
            if doc_symlink.exists():
                doc_symlink.unlink()
            doc_symlink.symlink_to(processed_doc_dir.absolute())
            
            # Set up output directory
            ingestion_output = Path("ingestion_output") / doc_name
            ingestion_output.mkdir(parents=True, exist_ok=True)
            
            # Initialize and run ingestion pipeline
            pipeline = ColabIngestionPipeline()
            
            ingestion_stats = pipeline.run_complete_pipeline(
                input_root=temp_input_root,
                output_root=ingestion_output,
                recreate_collection=False  # Don't recreate for single document
            )
            
            # Clean up temporary structure
            if doc_symlink.exists():
                doc_symlink.unlink()
            if temp_category_dir.exists():
                temp_category_dir.rmdir()
            if temp_input_root.exists():
                temp_input_root.rmdir()
            
            return {
                "success": True,
                "upload_id": upload_id,
                "processing_result": processing_result,
                "ingestion_stats": ingestion_stats,
                "message": "Document processed and ingested successfully"
            }
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            print(f"\n[Processing {upload_id}] {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "upload_id": upload_id,
                "error": error_msg,
                "message": f"Failed to process document: {str(e)}"
            }
