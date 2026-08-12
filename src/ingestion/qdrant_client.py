"""
Qdrant vector database client for document ingestion.

Handles collection management, indexing, and batch ingestion of document chunks.
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Iterable
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, PointStruct

from ..core.config import get_settings


class QdrantIngestionClient:
    """Client for ingesting document chunks into Qdrant."""
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.settings = get_settings()
        
        if not self.settings.qdrant_url:
            raise ValueError("Qdrant URL not configured. Please set QDRANT_URL.")
        
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            timeout=120
        )
        
        self.collection_name = self.settings.qdrant_collection
        self.batch_size = self.settings.qdrant_batch_size
        
        print(f"✓ Initialized QdrantIngestionClient")
        print(f"  URL: {self.settings.qdrant_url}")
        print(f"  Collection: {self.collection_name}")
    
    def ensure_collection(self, recreate: bool = False) -> bool:
        """
        Ensure collection exists with correct configuration.
        
        Args:
            recreate: Whether to delete and recreate existing collection
            
        Returns:
            True if collection is ready
        """
        if recreate and self.client.collection_exists(self.collection_name):
            print(f"🗑️ Deleting existing collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
        
        if not self.client.collection_exists(self.collection_name):
            print(f"🗂️ Creating collection '{self.collection_name}' with dim={self.settings.embed_dim}...")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embed_dim,
                    distance=Distance.COSINE
                ),
            )

        # Idempotent: Qdrant silently accepts re-indexing an already-indexed
        # field, so this also backfills indexes for new canonical detection
        # fields (cancer_types_detected, stages_detected, sites_detected,
        # biomarkers_detected, drugs_detected, histologies_detected,
        # genomic_alterations) on existing collections without re-ingestion.
        self._create_payload_indexes()

        print("✅ Collection ready.")
        return True
    
    def _create_payload_indexes(self):
        """Create payload indexes for filtering and search optimization."""
        index_specs = [
            ("metadata.keywords_flat", models.PayloadSchemaType.KEYWORD),
            # Canonical detection fields written by KeywordTagger.tag_chunk;
            # query-time token resolver targets these for typed MatchAny.
            ("metadata.cancer_types_detected", models.PayloadSchemaType.KEYWORD),
            ("metadata.sites_detected", models.PayloadSchemaType.KEYWORD),
            ("metadata.histologies_detected", models.PayloadSchemaType.KEYWORD),
            ("metadata.stages_detected", models.PayloadSchemaType.KEYWORD),
            ("metadata.biomarkers_detected", models.PayloadSchemaType.KEYWORD),
            ("metadata.drugs_detected", models.PayloadSchemaType.KEYWORD),
            ("metadata.genomic_alterations", models.PayloadSchemaType.KEYWORD),
            ("chunk_id", models.PayloadSchemaType.KEYWORD),
            ("original_chunk_id", models.PayloadSchemaType.KEYWORD),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("doc_id_raw", models.PayloadSchemaType.KEYWORD),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_granularity", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("table_number", models.PayloadSchemaType.KEYWORD),
            ("doc_meta.doi", models.PayloadSchemaType.KEYWORD),
            ("doc_meta.author_et_al", models.PayloadSchemaType.KEYWORD),
            ("section_window_idx", models.PayloadSchemaType.INTEGER),
            ("row_index", models.PayloadSchemaType.INTEGER),
            ("doc_meta.year", models.PayloadSchemaType.INTEGER),
        ]
        
        for field_name, field_schema in index_specs:
            try:
                print(f"↓ Creating payload index for '{field_name}' ({field_schema})...")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                print(f"⚠️ Index create failed for {field_name}: {e}")
        
        print("✅ Payload indexing attempted for all configured fields.")
    
    def ingest_chunks_from_jsonl(
        self, 
        jsonl_path: Path,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Ingest chunks from JSONL file into Qdrant.
        
        Args:
            jsonl_path: Path to JSONL file with embedded chunks
            batch_size: Batch size for ingestion
            
        Returns:
            Ingestion statistics
        """
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        print(f"🚀 Ingesting chunks from: {jsonl_path}")
        print(f"  Batch size: {batch_size}")
        
        total_points = 0
        successful_points = 0
        failed_points = 0
        
        for batch in self._jsonl_batches(jsonl_path, batch_size):
            points = []
            
            for chunk in batch:
                # Skip chunks without embeddings
                embedding = chunk.get("embedding")
                if embedding is None:
                    failed_points += 1
                    continue
                
                # Create point
                point = self._chunk_to_point(chunk)
                if point:
                    points.append(point)
                    successful_points += 1
                else:
                    failed_points += 1
            
            # Ingest batch
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_points += len(points)
                except Exception as e:
                    print(f"❌ Batch ingestion failed: {e}")
                    failed_points += len(points)
                    successful_points -= len(points)
        
        stats = {
            "total_points": total_points,
            "successful_points": successful_points,
            "failed_points": failed_points,
            "success_rate": successful_points / (successful_points + failed_points) if (successful_points + failed_points) > 0 else 0,
            "collection_name": self.collection_name,
            "source_file": str(jsonl_path)
        }
        
        print(f"✅ Ingestion complete:")
        print(f"  Total points ingested: {total_points}")
        print(f"  Successful: {successful_points}")
        print(f"  Failed: {failed_points}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        
        return stats
    
    def _jsonl_batches(self, path: Path, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
        """Yield batches of chunks from JSONL file."""
        batch = []
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    batch.append(obj)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON: {e}")
                    continue
        
        if batch:
            yield batch
    
    def _chunk_to_point(self, chunk: Dict[str, Any]) -> PointStruct:
        """Convert chunk to Qdrant point."""
        try:
            # Get embedding
            embedding = chunk.get("embedding")
            if not embedding:
                return None
            
            # Create stable UUID from chunk_id
            original_id = chunk.get("chunk_id")
            if not original_id:
                return None
            
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(original_id)))
            
            # Prepare payload (copy chunk and add traceable ID)
            payload = chunk.copy()
            payload["original_chunk_id"] = original_id
            
            # Remove embedding from payload to save space
            if "embedding" in payload:
                del payload["embedding"]
            
            return PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
        except Exception as e:
            print(f"⚠️ Error creating point for chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            return None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance,
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                print(f"✅ Deleted collection: {self.collection_name}")
                return True
            else:
                print(f"⚠️ Collection does not exist: {self.collection_name}")
                return False
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False
    
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters for search
            
        Returns:
            List of search results
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters
            )
            
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []