"""
Embedding generation using OpenAI's text-embedding models.

Handles batch processing, token limits, and safe text preprocessing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterable
import tiktoken
from openai import OpenAI

from ..core.config import get_settings


class EmbeddingGenerator:
    """Generate embeddings for document chunks using OpenAI."""
    
    def __init__(self):
        """Initialize embedding generator."""
        self.settings = get_settings()
        
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
        
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embed_model
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.max_tokens = 7000  # Safe limit for OpenAI embeddings
        
        print(f"✓ Initialized EmbeddingGenerator")
        print(f"  Model: {self.model}")
        print(f"  Dimensions: {self.settings.embed_dim}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Preprocess texts to ensure they're within token limits
        safe_texts = []
        for text in texts:
            text = text or ""
            token_ids = self.encoding.encode(text)
            
            if len(token_ids) > self.max_tokens:
                # Truncate to max tokens
                token_ids = token_ids[:self.max_tokens]
                text = self.encoding.decode(token_ids)
            
            safe_texts.append(text)
        
        # Generate embeddings
        response = self.client.embeddings.create(
            model=self.model,
            input=safe_texts
        )
        
        return [d.embedding for d in response.data]
    
    def embed_chunks_from_jsonl(
        self, 
        input_jsonl: Path,
        batch_size: int = None
    ) -> Iterable[Dict[str, Any]]:
        """
        Generate embeddings for chunks in a JSONL file.
        
        Args:
            input_jsonl: Path to JSONL file with chunks
            batch_size: Batch size for embedding generation
            
        Yields:
            Chunks with embeddings added
        """
        if batch_size is None:
            batch_size = self.settings.embed_batch_size
        
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
        
        print(f"🔄 Generating embeddings from: {input_jsonl}")
        print(f"  Batch size: {batch_size}")
        
        batch = []
        
        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    batch.append(chunk)
                    
                    if len(batch) >= batch_size:
                        # Process batch
                        yield from self._process_batch(batch)
                        batch = []
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON: {e}")
                    continue
        
        # Process remaining batch
        if batch:
            yield from self._process_batch(batch)
    
    def _process_batch(self, chunks: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        """Process a batch of chunks and add embeddings."""
        # Extract texts for embedding (use text_for_embedding if available)
        texts = []
        for chunk in chunks:
            text = chunk.get("text_for_embedding") or chunk.get("text", "")
            texts.append(text)
        
        # Generate embeddings
        try:
            embeddings = self.embed_texts(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
                chunk["embedding_model"] = self.model
                chunk["embedding_dim"] = len(embedding)
                yield chunk
                
        except Exception as e:
            print(f"❌ Error generating embeddings for batch: {e}")
            # Yield chunks without embeddings
            for chunk in chunks:
                chunk["embedding"] = None
                chunk["embedding_error"] = str(e)
                yield chunk
    
    def embed_and_save_jsonl(
        self, 
        input_jsonl: Path, 
        output_jsonl: Path,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all chunks and save to new JSONL.
        
        Args:
            input_jsonl: Input JSONL file path
            output_jsonl: Output JSONL file path
            batch_size: Batch size for processing
            
        Returns:
            Processing statistics
        """
        if output_jsonl.exists():
            print(f"⚠️ Output file exists, overwriting: {output_jsonl}")
        
        total_chunks = 0
        embedded_chunks = 0
        failed_chunks = 0
        
        with output_jsonl.open("w", encoding="utf-8") as f_out:
            for chunk in self.embed_chunks_from_jsonl(input_jsonl, batch_size):
                total_chunks += 1
                
                if chunk.get("embedding") is not None:
                    embedded_chunks += 1
                else:
                    failed_chunks += 1
                
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        stats = {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "failed_chunks": failed_chunks,
            "success_rate": embedded_chunks / total_chunks if total_chunks > 0 else 0,
            "model": self.model,
            "embedding_dim": self.settings.embed_dim,
            "input_file": str(input_jsonl),
            "output_file": str(output_jsonl)
        }
        
        print(f"✅ Embedding generation complete:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Successfully embedded: {embedded_chunks}")
        print(f"  Failed: {failed_chunks}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Output: {output_jsonl}")
        
        return stats
    
    def get_embedding_stats(self, jsonl_path: Path) -> Dict[str, Any]:
        """
        Get statistics about embeddings in a JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file with embeddings
            
        Returns:
            Statistics dictionary
        """
        total_chunks = 0
        embedded_chunks = 0
        embedding_dims = set()
        models_used = set()
        
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    total_chunks += 1
                    
                    if chunk.get("embedding") is not None:
                        embedded_chunks += 1
                        embedding_dims.add(chunk.get("embedding_dim", 0))
                        models_used.add(chunk.get("embedding_model", "unknown"))
                        
                except json.JSONDecodeError:
                    continue
        
        return {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "embedding_coverage": embedded_chunks / total_chunks if total_chunks > 0 else 0,
            "embedding_dimensions": sorted(embedding_dims),
            "models_used": sorted(models_used),
            "file_path": str(jsonl_path)
        }