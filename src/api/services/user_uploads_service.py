"""
User Uploads Service

Handles user-uploaded documents with local embedding storage.
Documents are processed, chunked, and embedded, then stored in PostgreSQL
for logged-in users or returned for session storage.

Embeddings are stored separately from metadata for efficient loading and search.
"""

import json
import re
import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .account_db import get_account_db
from ...processing.document_processor import CompleteDocumentProcessor, persist_study_profile_if_present
from ...ingestion.embeddings import EmbeddingGenerator
from ...core.config import get_settings


def extract_doi_from_filename(filename: str) -> Optional[str]:
    """
    Extract and normalize DOI from filename.
    
    Handles various formats:
    - "DOI 10.1016j.ijrobp.2024.09.040-Bladder.pdf"
    - "doi_10.1016_j.ijrobp.2024.09.040.pdf"
    - "10.1016-j.ijrobp.2024.09.040.pdf"
    
    Returns normalized DOI like: "10.1016/j.ijrobp.2024.09.040"
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Common DOI patterns
    # DOI format: 10.XXXX/... where XXXX is registrant code
    patterns = [
        # Standard DOI with slash: 10.1016/j.ijrobp.2024.09.040
        r'(10\.\d{4,}/[^\s]+)',
        # DOI with underscore instead of slash: 10.1016_j.ijrobp.2024.09.040
        r'(10\.\d{4,}_[a-zA-Z][^\s_-]*(?:\.[^\s_-]+)*)',
        # DOI prefix in filename: DOI 10.1016j.ijrobp... or doi_10.1016_j...
        r'(?:DOI|doi)[_\s]*(10\.\d{4,}[a-zA-Z]?[^\s_-]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            doi = match.group(1)
            # Normalize: convert underscores to slashes/dots appropriately
            doi = normalize_doi(doi)
            return doi
    
    return None


def normalize_doi(doi: str) -> str:
    """
    Normalize DOI to standard format: 10.XXXX/suffix
    
    Converts:
    - "10.1016_j.ijrobp.2024.09.040" -> "10.1016/j.ijrobp.2024.09.040"
    - "10.1016j.ijrobp.2024.09.040" -> "10.1016/j.ijrobp.2024.09.040"
    """
    # Remove any leading/trailing whitespace
    doi = doi.strip()
    
    # If already has slash after registrant, it's normalized
    if re.match(r'10\.\d{4,}/', doi):
        return doi
    
    # Pattern: 10.XXXX_ or 10.XXXXletter (where letter starts suffix)
    # Convert first underscore after registrant to slash
    match = re.match(r'(10\.\d{4,})_(.+)', doi)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    
    # Pattern: 10.XXXXletter... (no separator, letter indicates start of suffix)
    # e.g., "10.1016j.ijrobp..." -> "10.1016/j.ijrobp..."
    match = re.match(r'(10\.\d{4,})([a-zA-Z].+)', doi)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    
    return doi


def doi_to_qdrant_pattern(doi: str) -> str:
    """
    Convert DOI to pattern used in Qdrant doc_id.
    
    Qdrant uses: doi_10.1016_j.ijrobp.2024.09.040_HASH
    """
    # Normalize first
    normalized = normalize_doi(doi)
    # Convert to Qdrant format: replace / with _ and add doi_ prefix
    qdrant_pattern = "doi_" + normalized.replace("/", "_")
    return qdrant_pattern


class UserUploadsService:
    """Service for managing user-uploaded documents with local embeddings."""
    
    def __init__(self):
        self.uploads_dir = Path("user_uploads")
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        
    async def _ensure_schema(self):
        """Ensure the user_uploads table exists in the account database."""
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Create comprehensive user_uploads table with embeddings and study profile
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_uploads (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    upload_id VARCHAR(64) UNIQUE NOT NULL,
                    doc_id VARCHAR(500),
                    filename VARCHAR(255) NOT NULL,
                    title TEXT,
                    status VARCHAR(50) DEFAULT 'completed',
                    
                    -- Document metadata
                    doc_meta JSONB,
                    
                    -- Embeddings stored as binary (numpy array serialized)
                    embeddings BYTEA,
                    embedding_dim INTEGER,
                    
                    -- Chunk metadata (text, section, etc. - without embeddings)
                    chunk_metadata JSONB,
                    chunk_count INTEGER DEFAULT 0,
                    
                    -- Study profile (extracted study details as JSONB)
                    study_profile JSONB,
                    
                    -- Identifiers from study profile
                    doi VARCHAR(255),
                    pmid VARCHAR(50),
                    
                    -- Processing info
                    error_message TEXT,
                    reused_existing BOOLEAN DEFAULT FALSE,
                    
                    -- Timestamps
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    processed_at TIMESTAMPTZ,
                    
                    UNIQUE(user_id, upload_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_user_uploads_user_id 
                ON user_uploads(user_id);
                
                CREATE INDEX IF NOT EXISTS idx_user_uploads_user_created 
                ON user_uploads(user_id, created_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_user_uploads_doc_id
                ON user_uploads(doc_id);
            """)
            
            # Add new columns if they don't exist (for existing tables)
            try:
                await conn.execute("""
                    ALTER TABLE user_uploads ADD COLUMN IF NOT EXISTS doc_id VARCHAR(500);
                    ALTER TABLE user_uploads ADD COLUMN IF NOT EXISTS study_profile JSONB;
                    ALTER TABLE user_uploads ADD COLUMN IF NOT EXISTS doi VARCHAR(255);
                    ALTER TABLE user_uploads ADD COLUMN IF NOT EXISTS pmid VARCHAR(50);
                    ALTER TABLE user_uploads ADD COLUMN IF NOT EXISTS reused_existing BOOLEAN DEFAULT FALSE;
                """)
            except Exception:
                pass  # Columns may already exist
    
    def _serialize_embeddings(self, embeddings: List[List[float]]) -> bytes:
        """Serialize embeddings list to bytes (numpy format)."""
        arr = np.array(embeddings, dtype=np.float32)
        return arr.tobytes()
    
    def _deserialize_embeddings(self, data: bytes, dim: int, count: int) -> np.ndarray:
        """Deserialize bytes back to numpy array."""
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.reshape(count, dim)
    
    async def _check_existing_processed_document(
        self,
        doc_name: str,
        filename: str,
        user_id: Optional[str],
        upload_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if document already exists in processed documents or other user uploads.
        
        NOTE: ALL DUPLICATE CHECKS DISABLED - was causing false matches
        Always returns None to force fresh processing.
        
        Args:
            doc_name: Document name (filename stem)
            filename: Original filename (for DOI extraction)
            user_id: User ID if logged in
            upload_id: Upload ID for this request
            
        Returns:
            None - always process fresh
        """
        # ALL DUPLICATE DETECTION DISABLED
        # Was causing false positive matches with wrong studies
        print(f"  Duplicate detection disabled - processing fresh")
        return None
    
    async def _check_qdrant_by_doi(
        self,
        doi: str,
        user_id: Optional[str],
        upload_id: str,
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if document exists in Qdrant by EXACT DOI match.
        
        Uses semantic search to find candidates, then verifies the DOI pattern
        is EXACTLY present in the doc_id (not just semantically similar).
        
        Args:
            doi: Normalized DOI (e.g., "10.1016/j.ijrobp.2024.09.040")
            user_id: User ID if logged in
            upload_id: Upload ID for this request
            filename: Original filename
            
        Returns:
            Processing result if found, None if not found
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            settings = get_settings()
            
            if not settings.qdrant_url:
                return None
            
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=30
            )
            
            # Convert DOI to Qdrant doc_id pattern (without hash suffix)
            # e.g., "10.1016/j.ijrobp.2024.09.040" -> "doi_10.1016_j.ijrobp.2024.09.040"
            qdrant_pattern = doi_to_qdrant_pattern(doi)
            print(f"  Checking Qdrant for EXACT DOI pattern: {qdrant_pattern}")
            
            # Do a semantic search to find candidates
            embedder = EmbeddingGenerator()
            query_text = f"DOI {doi}"
            query_embedding = embedder.embed_texts([query_text])[0]
            
            initial_results = client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_embedding,
                limit=10,
                with_payload=True
            )
            
            # Find a doc_id that EXACTLY contains our DOI pattern
            # The doc_id format is: doi_10.1016_j.ijrobp.2024.09.040_HASH
            # We need to match the DOI part exactly, not just similar DOIs
            matching_doc_id = None
            for point in initial_results.points:
                doc_id = point.payload.get("doc_id", "")
                # Check if the doc_id starts with our exact pattern
                # This ensures we match "doi_10.1016_j.ijrobp.2024.09.040_xxx" 
                # but NOT "doi_10.1016_j.ijrobp.2024.09.041_xxx" (different DOI)
                if doc_id.lower().startswith(qdrant_pattern.lower()):
                    matching_doc_id = doc_id
                    print(f"  Found EXACT DOI match: {doc_id}")
                    break
            
            if not matching_doc_id:
                print(f"  No EXACT DOI match found in Qdrant for: {doi}")
                return None
            
            # Now scroll through ALL chunks with this exact doc_id
            matching_chunks = []
            offset = None
            
            while True:
                # Use scroll to get all chunks for this document
                results, offset = client.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchText(text=matching_doc_id)
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in results:
                    payload = point.payload
                    matching_chunks.append({
                        "chunk_id": str(point.id),
                        "doc_id": payload.get("doc_id", ""),
                        "text": payload.get("text", ""),
                        "section": payload.get("section"),
                        "chunk_type": payload.get("chunk_type", "paragraph"),
                        "title": payload.get("title"),
                        "doi": payload.get("doi") or doi,
                    })
                
                if offset is None or len(results) == 0:
                    break
            
            if not matching_chunks:
                print(f"  No chunks found for doc_id: {matching_doc_id}")
                return None
            
            print(f"  Found {len(matching_chunks)} chunks in Qdrant for DOI: {doi}")
            
            # Get document metadata from first chunk
            first_chunk = matching_chunks[0]
            title = first_chunk.get("title") or Path(filename).stem
            
            # Generate embeddings for the chunks we found
            print(f"  Generating embeddings for {len(matching_chunks)} Qdrant chunks...")
            texts = [c["text"] for c in matching_chunks]
            embeddings = embedder.embed_texts(texts)
            
            # Build chunk metadata
            chunk_metadata = []
            for i, chunk in enumerate(matching_chunks):
                chunk_metadata.append({
                    "id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "chunk_type": chunk["chunk_type"],
                    "word_count": len(chunk["text"].split()),
                })
            
            # Store for user if logged in
            if user_id:
                await self._ensure_schema()
                
                # Build a simple doc_meta
                doc_meta = {
                    "title": title,
                    "doi": doi,
                    "source": "qdrant_existing"
                }
                
                # Create chunks list for storage
                chunks = []
                for meta in chunk_metadata:
                    chunks.append({
                        "chunk_id": meta["id"],
                        "text": meta["text"],
                        "section": meta["section"],
                        "chunk_type": meta["chunk_type"],
                        "word_count": meta["word_count"],
                    })
                
                await self._store_in_postgres(
                    user_id=user_id,
                    upload_id=upload_id,
                    filename=filename,
                    title=title,
                    doc_meta=doc_meta,
                    chunks=chunks,
                    embeddings=embeddings,
                    doc_id=matching_doc_id,
                    study_profile=None,  # Will need to fetch from studies table if available
                    reused_existing=True
                )
                
                return {
                    "success": True,
                    "upload_id": upload_id,
                    "doc_id": matching_doc_id,
                    "filename": filename,
                    "title": title,
                    "chunk_count": len(matching_chunks),
                    "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    "stored": "database",
                    "message": f"Document found in knowledge base (DOI: {doi}) - linked to your account",
                    "study_profile": None,
                    "reused_existing": True
                }
            else:
                # Return for session storage
                return {
                    "success": True,
                    "upload_id": upload_id,
                    "doc_id": matching_doc_id,
                    "filename": filename,
                    "title": title,
                    "chunk_count": len(matching_chunks),
                    "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    "embeddings": embeddings,
                    "chunk_metadata": chunk_metadata,
                    "doc_meta": {"title": title, "doi": doi},
                    "stored": "session",
                    "message": f"Document found in knowledge base (DOI: {doi}) - loaded without reprocessing",
                    "study_profile": None,
                    "reused_existing": True
                }
                
        except Exception as e:
            print(f"  Qdrant DOI check failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _check_cross_user_uploads(
        self,
        doi: Optional[str],
        doc_name: str,
        user_id: str,
        upload_id: str,
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if another user has already uploaded this document.
        
        If found, copies the embeddings and study profile to create a new
        entry for the current user without reprocessing.
        
        Args:
            doi: Normalized DOI (if extracted from filename)
            doc_name: Document name (filename stem)
            user_id: Current user's ID
            upload_id: Upload ID for this request
            filename: Original filename
            
        Returns:
            Processing result if found, None if not found
        """
        try:
            await self._ensure_schema()
            db = get_account_db()
            pool = await db.get_pool()
            
            async with pool.acquire() as conn:
                existing_upload = None
                
                # First try to find by DOI (most reliable)
                if doi:
                    existing_upload = await conn.fetchrow("""
                        SELECT upload_id, doc_id, filename, title, doc_meta,
                               embeddings, embedding_dim, chunk_metadata, chunk_count,
                               study_profile, doi, pmid
                        FROM user_uploads
                        WHERE doi = $1 
                          AND user_id != $2 
                          AND status = 'completed'
                          AND embeddings IS NOT NULL
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, doi, user_id)
                
                # If not found by DOI, try by doc_id pattern
                if not existing_upload:
                    # Normalize doc_name for comparison
                    search_pattern = f"%{doc_name}%"
                    existing_upload = await conn.fetchrow("""
                        SELECT upload_id, doc_id, filename, title, doc_meta,
                               embeddings, embedding_dim, chunk_metadata, chunk_count,
                               study_profile, doi, pmid
                        FROM user_uploads
                        WHERE doc_id ILIKE $1 
                          AND user_id != $2 
                          AND status = 'completed'
                          AND embeddings IS NOT NULL
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, search_pattern, user_id)
                
                if not existing_upload:
                    return None
                
                print(f"  Found existing upload from another user: {existing_upload['doc_id']}")
                
                # Copy the data to create a new entry for this user
                source_doc_id = existing_upload['doc_id']
                source_title = existing_upload['title']
                source_doi = existing_upload['doi']
                source_pmid = existing_upload['pmid']
                
                # Parse JSON fields
                doc_meta = existing_upload['doc_meta']
                if isinstance(doc_meta, str):
                    doc_meta = json.loads(doc_meta)
                
                chunk_metadata = existing_upload['chunk_metadata']
                if isinstance(chunk_metadata, str):
                    chunk_metadata = json.loads(chunk_metadata)
                
                study_profile = existing_upload['study_profile']
                if isinstance(study_profile, str):
                    study_profile = json.loads(study_profile)
                
                # Insert new row for this user with copied data
                await conn.execute("""
                    INSERT INTO user_uploads (
                        user_id, upload_id, doc_id, filename, title, status,
                        doc_meta, embeddings, embedding_dim, chunk_metadata, 
                        chunk_count, study_profile, doi, pmid, reused_existing, processed_at
                    ) VALUES ($1, $2, $3, $4, $5, 'completed', $6, $7, $8, $9, $10, $11, $12, $13, TRUE, NOW())
                """,
                    user_id,
                    upload_id,
                    source_doc_id,
                    filename,  # Use the new user's filename
                    source_title,
                    json.dumps(doc_meta) if doc_meta else None,
                    existing_upload['embeddings'],  # Copy binary embeddings directly
                    existing_upload['embedding_dim'],
                    json.dumps(chunk_metadata) if chunk_metadata else None,
                    existing_upload['chunk_count'],
                    json.dumps(study_profile) if study_profile else None,
                    source_doi,
                    source_pmid,
                )
                
                print(f"  Copied upload to user's account (reused from another user)")
                
                return {
                    "success": True,
                    "upload_id": upload_id,
                    "doc_id": source_doc_id,
                    "filename": filename,
                    "title": source_title,
                    "chunk_count": existing_upload['chunk_count'],
                    "embedding_dim": existing_upload['embedding_dim'],
                    "stored": "database",
                    "message": "Document already processed - linked to your account (no reprocessing needed)",
                    "study_profile": study_profile,
                    "reused_existing": True
                }
                
        except Exception as e:
            print(f"  Cross-user upload check failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _link_existing_document(
        self,
        doc_name: str,
        processed_dir: Path,
        user_id: Optional[str],
        upload_id: str,
        source: str
    ) -> Dict[str, Any]:
        """
        Link an existing processed document to a user's account.
        
        Creates chunks and embeddings from the existing processed files,
        then stores in user_uploads table.
        """
        print(f"  Linking existing document from {source}...")
        
        # Create chunks from existing processed document
        chunks = self._create_chunks_from_processed(processed_dir, upload_id)
        
        if not chunks:
            return None  # Fall back to full processing
        
        # Generate embeddings
        print(f"  Generating embeddings for {len(chunks)} chunks...")
        embedder = EmbeddingGenerator()
        texts = [c.get("text_for_embedding") or c.get("text", "") for c in chunks]
        embeddings = embedder.embed_texts(texts)
        
        # Extract metadata
        doc_meta = chunks[0].get("doc_meta", {}) if chunks else {}
        title = doc_meta.get("title") or doc_name
        
        # Get study profile from studies table if exists
        study_profile = None
        try:
            db = get_account_db()
            pool = await db.get_pool()
            
            async with pool.acquire() as conn:
                study_row = await conn.fetchrow(
                    "SELECT * FROM studies WHERE doc_id = $1 OR document_name ILIKE $2 LIMIT 1",
                    doc_name, f"%{doc_name}%"
                )
                
                if study_row:
                    # Build study profile from existing data
                    print(f"  Found study profile in database")
                    study_profile = await self._build_study_profile_from_row(study_row)
                else:
                    print(f"  No study profile in database, extracting from processed files...")
        except Exception as e:
            print(f"  Could not fetch study profile from DB: {e}")
        
        # If no study profile found in DB, extract it from the processed files
        if not study_profile:
            try:
                from ...processing.study_profile_extractor import StudyProfileExtractor
                
                print(f"  Extracting study profile from processed files...")
                extractor = StudyProfileExtractor()
                profile_result = extractor.extract_from_processed_dir(processed_dir)
                
                if profile_result and profile_result.get("extracted_data"):
                    study_profile = profile_result.get("extracted_data")
                    print(f"  Study profile extracted successfully")
            except Exception as e:
                print(f"  Study profile extraction failed: {e}")
        
        # Store for user
        if user_id:
            await self._ensure_schema()
            await self._store_in_postgres(
                user_id=user_id,
                upload_id=upload_id,
                filename=f"{doc_name}.pdf",
                title=title,
                doc_meta=doc_meta,
                chunks=chunks,
                embeddings=embeddings,
                doc_id=doc_name,
                study_profile=study_profile,
                reused_existing=True  # This is from existing document
            )
            
            return {
                "success": True,
                "upload_id": upload_id,
                "doc_id": doc_name,
                "filename": f"{doc_name}.pdf",
                "title": title,
                "chunk_count": len(chunks),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "stored": "database",
                "message": f"Document found in {source} - linked to your account (no reprocessing needed)",
                "study_profile": study_profile,
                "reused_existing": True
            }
        else:
            # Session storage
            chunk_metadata = []
            for c in chunks:
                chunk_metadata.append({
                    "id": c.get("chunk_id"),
                    "text": c.get("text"),
                    "section": c.get("section"),
                    "chunk_type": c.get("chunk_type"),
                    "word_count": len(c.get("text", "").split()),
                })
            
            return {
                "success": True,
                "upload_id": upload_id,
                "doc_id": doc_name,
                "filename": f"{doc_name}.pdf",
                "title": title,
                "chunk_count": len(chunks),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "embeddings": embeddings,
                "chunk_metadata": chunk_metadata,
                "doc_meta": doc_meta,
                "stored": "session",
                "message": f"Document found in {source} - loaded without reprocessing",
                "study_profile": study_profile,
                "reused_existing": True
            }
    
    async def _link_existing_study(
        self,
        study_row,
        user_id: Optional[str],
        upload_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Link an existing study from the studies table to a user.
        
        This is used when we find the study in PostgreSQL but don't have
        the processed files locally. We can still provide the study profile.
        """
        doc_name = study_row['doc_id'] or study_row['document_name']
        title = study_row.get('study_name') or doc_name
        
        # Build study profile from the row
        study_profile = await self._build_study_profile_from_row(study_row)
        
        # We don't have chunks/embeddings, so we can't fully link
        # Return None to trigger full processing, but the study profile
        # will be available from the studies table
        print(f"  Study found in DB but no processed files available")
        return None
    
    async def _build_study_profile_from_row(self, study_row) -> Dict[str, Any]:
        """Build study profile dict from a studies table row."""
        # This builds a simplified profile from the main studies table
        # The full profile would need to query related tables
        
        def get_field(row, field):
            value = row.get(field)
            evidence = row.get(f"{field}_evidence")
            if value is not None:
                return {"value": value, "evidence_quote": evidence}
            return None
        
        study_details = {}
        for field in ['study_name', 'protocol_name', 'trial_registration_number', 
                      'publish_date', 'study_type', 'study_phase', 'analysis_type',
                      'number_of_patients', 'study_institution', 'country', 'doi', 'pmid']:
            val = get_field(study_row, field)
            if val:
                study_details[field] = val
        
        patient_chars = {}
        for field in ['age_range', 'median_age', 'gender_distribution', 
                      'race_ethnicity', 'performance_status']:
            val = get_field(study_row, field)
            if val:
                patient_chars[field] = val
        
        diagnosis = {}
        for field in ['cancer_location', 'cancer_type', 'histopathologic_type',
                      'tumor_grade', 'molecular_subtype']:
            val = get_field(study_row, field)
            if val:
                diagnosis[field] = val
        
        staging = {}
        for field in ['staging_system_used', 'risk_stratification', 
                      'metastatic_status', 'extent_of_resection']:
            val = get_field(study_row, field)
            if val:
                staging[field] = val
        
        outcomes = {}
        for field in ['primary_endpoint', 'event_free_survival', 'overall_survival',
                      'progression_free_survival', 'disease_free_survival', 
                      'local_control', 'median_followup']:
            val = get_field(study_row, field)
            if val:
                outcomes[field] = val
        
        return {
            "study_details": study_details,
            "patient_characteristics": patient_chars,
            "diagnosis": diagnosis,
            "staging": staging,
            "treatment": {},  # Would need to query related tables
            "outcomes": outcomes,
            "biomarkers": [],
            "toxicity": [],
            "dose_constraints": []
        }
    
    async def process_and_store_document(
        self,
        file_path: Path,
        filename: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document and store with embeddings.
        
        First checks if the document already exists in admin processed documents
        (GCP bucket or local). If so, links to user account without reprocessing.
        
        Args:
            file_path: Path to the uploaded PDF
            filename: Original filename
            user_id: User ID if logged in, None for session storage
            
        Returns:
            Processing result with upload_id and chunks (for session) or success status
        """
        upload_id = str(uuid.uuid4())
        study_profile = None
        
        try:
            # Generate doc_name from filename (same logic as CompleteDocumentProcessor)
            doc_name = Path(filename).stem
            
            # Step 0: Check if document already exists in admin processed documents
            print(f"[UserUpload {upload_id}] Checking for existing processed document...")
            existing_result = await self._check_existing_processed_document(
                doc_name=doc_name,
                filename=filename,
                user_id=user_id,
                upload_id=upload_id
            )
            
            if existing_result:
                print(f"[UserUpload {upload_id}] Found existing processed document, linking to user account")
                # Cleanup uploaded file since we don't need to process it
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except:
                        pass
                return existing_result
            
            # Step 1: Document Processing (OCR, vision, etc.)
            print(f"[UserUpload {upload_id}] Processing document...")
            processor = CompleteDocumentProcessor(str(file_path))
            processing_result = processor.process_complete()

            # Persist any study profile process_complete() extracted
            # internally (Phase 10, gated by settings.extract_study_profiles)
            # -- separate from this method's own StudyProfileExtractor call
            # below, which is for immediate display, not Postgres storage.
            # See persist_study_profile_if_present()'s docstring for why
            # this can't happen inside process_complete() itself.
            await persist_study_profile_if_present(processing_result, processor.doc_name)

            # Use original filename for doc_name (not temp file name)
            doc_name = Path(filename).stem
            print(f"[UserUpload {upload_id}] Doc name: {doc_name}")
            
            # Step 1b: Extract study profile for display
            print(f"[UserUpload {upload_id}] Extracting study profile...")
            try:
                from ...processing.study_profile_extractor import StudyProfileExtractor
                
                processed_dir = Path(processing_result['output_directory'])
                extractor = StudyProfileExtractor()
                profile_result = extractor.extract_from_processed_dir(processed_dir)
                
                if profile_result and profile_result.get("extracted_data"):
                    study_profile = profile_result.get("extracted_data")
                    print(f"[UserUpload {upload_id}] Study profile extracted successfully")
                else:
                    print(f"[UserUpload {upload_id}] Study profile extraction returned no data")
            except Exception as e:
                print(f"[UserUpload {upload_id}] Study profile extraction failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 1c: Sync to GCP user_uploads bucket
            settings = get_settings()
            if settings.auto_sync_gcp:
                try:
                    from ...utils.gcp_sync import UserUploadsGCPSync
                    
                    print(f"[UserUpload {upload_id}] Syncing to GCP user_uploads bucket...")
                    gcp_sync = UserUploadsGCPSync()
                    sync_stats = gcp_sync.sync_user_document(
                        doc_name=processor.doc_name,
                        local_base=str(Path(processing_result['output_directory']).parent),
                        user_id=user_id
                    )
                    print(f"[UserUpload {upload_id}] GCP sync complete: {sync_stats['uploaded']} files")
                except Exception as e:
                    print(f"[UserUpload {upload_id}] GCP sync failed: {e}")
            
            # Step 2: Create chunks from processed document
            print(f"[UserUpload {upload_id}] Creating chunks...")
            processed_dir = Path(processing_result['output_directory'])
            chunks = self._create_chunks_from_processed(processed_dir, upload_id)
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Step 3: Generate embeddings (same as ingestion pipeline)
            print(f"[UserUpload {upload_id}] Generating embeddings for {len(chunks)} chunks...")
            embedder = EmbeddingGenerator()
            texts = [c.get("text_for_embedding") or c.get("text", "") for c in chunks]
            embeddings = embedder.embed_texts(texts)
            
            # Extract document metadata
            doc_meta = chunks[0].get("doc_meta", {}) if chunks else {}
            title = doc_meta.get("title") or filename
            # doc_name was set earlier in Step 1b
            
            # Step 4: Store based on user status
            if user_id:
                # Store in PostgreSQL with separate embeddings array
                await self._ensure_schema()
                await self._store_in_postgres(
                    user_id=user_id,
                    upload_id=upload_id,
                    filename=filename,
                    title=title,
                    doc_meta=doc_meta,
                    chunks=chunks,
                    embeddings=embeddings,
                    doc_id=doc_name,
                    study_profile=study_profile,
                    reused_existing=False  # Newly processed document
                )
                
                return {
                    "success": True,
                    "upload_id": upload_id,
                    "doc_id": doc_name,  # For study details lookup
                    "filename": filename,
                    "title": title,
                    "chunk_count": len(chunks),
                    "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    "stored": "database",
                    "message": "Document processed and saved to your account",
                    "study_profile": study_profile  # Extracted study profile for display
                }
            else:
                # Return for session storage
                # Build session data with embeddings array separate from metadata
                chunk_metadata = []
                for i, c in enumerate(chunks):
                    chunk_metadata.append({
                        "id": c.get("chunk_id"),
                        "text": c.get("text"),
                        "section": c.get("section"),
                        "chunk_type": c.get("chunk_type"),
                        "word_count": len(c.get("text", "").split()),
                    })
                
                return {
                    "success": True,
                    "upload_id": upload_id,
                    "doc_id": doc_name,  # For study details lookup
                    "filename": filename,
                    "title": title,
                    "chunk_count": len(chunks),
                    "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    "embeddings": embeddings,  # List of embedding vectors
                    "chunk_metadata": chunk_metadata,  # Metadata without embeddings
                    "doc_meta": doc_meta,
                    "stored": "session",
                    "message": "Document processed. Log in to save permanently.",
                    "study_profile": study_profile  # Extracted study profile for display
                }
                
        except Exception as e:
            print(f"[UserUpload {upload_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "upload_id": upload_id,
                "error": str(e),
                "message": f"Failed to process document: {str(e)}"
            }
        finally:
            # Cleanup temporary files
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
    
    def _create_chunks_from_processed(
        self, 
        processed_dir: Path, 
        upload_id: str
    ) -> List[Dict[str, Any]]:
        """Create chunks from a processed document directory."""
        chunks = []
        
        # Find document index file
        idx_files = list(processed_dir.glob("*_document_index.json"))
        if not idx_files:
            raise ValueError("No document index found in processed output")
        
        with idx_files[0].open("r", encoding="utf-8") as f:
            document_index = json.load(f)
        
        # Load structured content for metadata
        doc_meta = {}
        structured_files = list(processed_dir.glob("*_structured_content.json"))
        if structured_files:
            try:
                with structured_files[0].open("r", encoding="utf-8") as f:
                    structured = json.load(f)
                doc_meta = self._build_doc_meta(structured)
            except:
                pass
        
        # Create paragraph chunks
        paragraphs = document_index.get("paragraphs", [])
        for i, p in enumerate(paragraphs):
            text = p.get("text", "").strip()
            if not text or len(text) < 20:
                continue
            
            chunk = {
                "chunk_id": f"{upload_id}_p{i:04d}",
                "doc_id": upload_id,
                "chunk_type": "paragraph",
                "section": p.get("section"),
                "page": p.get("page", 1),
                "text": text,
                "text_for_embedding": text,
                "word_count": len(text.split()),
                "doc_meta": doc_meta,
            }
            chunks.append(chunk)
        
        # Create table chunks if available
        table_files = list(processed_dir.glob("*_tables.json"))
        if table_files:
            try:
                with table_files[0].open("r", encoding="utf-8") as f:
                    tables_obj = json.load(f)
                
                for t_idx, tbl in enumerate(tables_obj.get("tables", [])):
                    table_number = tbl.get("table_number", f"Table_{t_idx+1}")
                    title = tbl.get("title", "")
                    headers = tbl.get("headers") or []
                    page = tbl.get("page", 1)
                    
                    for r_idx, row in enumerate(tbl.get("rows", [])):
                        row_values = [str(v) for v in row]
                        if headers and len(headers) == len(row_values):
                            cells_text = "; ".join(f"{h}: {v}" for h, v in zip(headers, row_values))
                        else:
                            cells_text = "; ".join(row_values)
                        
                        full_text = f"{table_number} - {title}. {cells_text}".strip()
                        if len(full_text) < 20:
                            continue
                        
                        chunk = {
                            "chunk_id": f"{upload_id}_t{t_idx+1:02d}_r{r_idx+1:03d}",
                            "doc_id": upload_id,
                            "chunk_type": "table_row",
                            "section": f"Table: {table_number}",
                            "page": page,
                            "text": full_text,
                            "text_for_embedding": full_text,
                            "word_count": len(full_text.split()),
                            "doc_meta": doc_meta,
                        }
                        chunks.append(chunk)
            except:
                pass
        
        return chunks
    
    def _build_doc_meta(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Build document metadata from structured content."""
        dm = structured.get("document_metadata", {})
        info = dm.get("document_info", {})
        pub = dm.get("publication_info", {})
        
        return {
            "title": info.get("title"),
            "authors": info.get("authors", []),
            "journal": pub.get("journal"),
            "year": self._extract_year(pub),
            "doi": pub.get("doi"),
        }
    
    def _extract_year(self, pub: Dict[str, Any]) -> Optional[int]:
        """Extract year from publication info."""
        import re
        for key in ("publication_date", "online_date"):
            val = pub.get(key)
            if val:
                m = re.search(r"\b(19|20)\d{2}\b", str(val))
                if m:
                    return int(m.group(0))
        return None
    
    async def _store_in_postgres(
        self,
        user_id: str,
        upload_id: str,
        filename: str,
        title: str,
        doc_meta: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_id: Optional[str] = None,
        study_profile: Optional[Dict[str, Any]] = None,
        reused_existing: bool = False
    ):
        """
        Store processed document in PostgreSQL.
        
        Embeddings stored as binary array, metadata stored as JSON.
        Study profile stored as JSONB for display.
        """
        db = get_account_db()
        pool = await db.get_pool()
        
        # Serialize embeddings to binary
        embeddings_bytes = self._serialize_embeddings(embeddings)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        # Build chunk metadata (without embeddings)
        chunk_metadata = []
        for c in chunks:
            chunk_metadata.append({
                "id": c.get("chunk_id"),
                "text": c.get("text"),
                "section": c.get("section"),
                "chunk_type": c.get("chunk_type"),
                "page": c.get("page", 1),
                "word_count": c.get("word_count", 0),
            })
        
        # Extract DOI and PMID from study profile if available
        doi = None
        pmid = None
        if study_profile:
            study_details = study_profile.get("study_details", {})
            doi_data = study_details.get("doi", {})
            pmid_data = study_details.get("pmid", {})
            doi = doi_data.get("value") if isinstance(doi_data, dict) else doi_data
            pmid = pmid_data.get("value") if isinstance(pmid_data, dict) else pmid_data
        
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_uploads (
                    user_id, upload_id, doc_id, filename, title, status,
                    doc_meta, embeddings, embedding_dim, chunk_metadata, 
                    chunk_count, study_profile, doi, pmid, reused_existing, processed_at
                ) VALUES ($1, $2, $3, $4, $5, 'completed', $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW())
            """,
                user_id,
                upload_id,
                doc_id,
                filename,
                title,
                json.dumps(doc_meta),
                embeddings_bytes,
                embedding_dim,
                json.dumps(chunk_metadata),
                len(chunks),
                json.dumps(study_profile) if study_profile else None,
                doi,
                pmid,
                reused_existing
            )
        
        print(f"[UserUpload {upload_id}] Saved {len(embeddings)} embeddings ({embedding_dim}D) to PostgreSQL")
        if study_profile:
            print(f"[UserUpload {upload_id}] Study profile stored (DOI: {doi}, PMID: {pmid})")
        
        # Clear user's query cache so new uploads are included in future queries
        try:
            from .cache_service import get_cache_service
            cache_service = get_cache_service()
            await cache_service.clear_user_cache(user_id, "rag_query")
            print(f"[UserUpload {upload_id}] Cleared query cache for user")
        except Exception as e:
            print(f"[UserUpload {upload_id}] Warning: Failed to clear cache: {e}")
    
    async def get_user_uploads(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all uploads for a user (without embeddings)."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, upload_id, filename, title, status, 
                       doc_meta, embedding_dim, chunk_count, 
                       error_message, created_at, processed_at
                FROM user_uploads
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)
            
            return [
                {
                    "id": row["id"],
                    "upload_id": row["upload_id"],
                    "filename": row["filename"],
                    "title": row["title"],
                    "status": row["status"],
                    "doc_meta": row["doc_meta"] if isinstance(row["doc_meta"], dict) else json.loads(row["doc_meta"] or "{}"),
                    "embedding_dim": row["embedding_dim"],
                    "chunk_count": row["chunk_count"],
                    "error_message": row["error_message"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "processed_at": row["processed_at"].isoformat() if row["processed_at"] else None,
                }
                for row in rows
            ]
    
    async def get_upload_with_embeddings(
        self,
        user_id: str,
        upload_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get upload with embeddings for search."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT upload_id, filename, title, doc_meta,
                       embeddings, embedding_dim, chunk_metadata, chunk_count
                FROM user_uploads
                WHERE user_id = $1 AND upload_id = $2
            """, user_id, upload_id)
            
            if not row:
                return None
            
            # Deserialize embeddings
            embeddings_array = None
            if row["embeddings"] and row["embedding_dim"] and row["chunk_count"]:
                embeddings_array = self._deserialize_embeddings(
                    row["embeddings"], 
                    row["embedding_dim"], 
                    row["chunk_count"]
                )
            
            chunk_metadata = row["chunk_metadata"]
            if isinstance(chunk_metadata, str):
                chunk_metadata = json.loads(chunk_metadata)
            
            return {
                "upload_id": row["upload_id"],
                "filename": row["filename"],
                "title": row["title"],
                "doc_meta": row["doc_meta"] if isinstance(row["doc_meta"], dict) else json.loads(row["doc_meta"] or "{}"),
                "embeddings": embeddings_array,  # numpy array (N, dim)
                "chunk_metadata": chunk_metadata,
                "chunk_count": row["chunk_count"],
                "embedding_dim": row["embedding_dim"],
            }
    
    async def get_all_user_embeddings(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get all uploads with embeddings for a user (for search)."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT upload_id, filename, title, doc_meta,
                       embeddings, embedding_dim, chunk_metadata, chunk_count
                FROM user_uploads
                WHERE user_id = $1 AND status = 'completed'
                ORDER BY created_at DESC
            """, user_id)
            
            results = []
            for row in rows:
                embeddings_array = None
                if row["embeddings"] and row["embedding_dim"] and row["chunk_count"]:
                    embeddings_array = self._deserialize_embeddings(
                        row["embeddings"], 
                        row["embedding_dim"], 
                        row["chunk_count"]
                    )
                
                chunk_metadata = row["chunk_metadata"]
                if isinstance(chunk_metadata, str):
                    chunk_metadata = json.loads(chunk_metadata)
                
                results.append({
                    "upload_id": row["upload_id"],
                    "filename": row["filename"],
                    "title": row["title"],
                    "doc_meta": row["doc_meta"] if isinstance(row["doc_meta"], dict) else json.loads(row["doc_meta"] or "{}"),
                    "embeddings": embeddings_array,
                    "chunk_metadata": chunk_metadata,
                    "chunk_count": row["chunk_count"],
                    "embedding_dim": row["embedding_dim"],
                })
            
            return results
    
    async def delete_upload(self, user_id: str, upload_id: str) -> bool:
        """Delete a user upload."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM user_uploads
                WHERE user_id = $1 AND upload_id = $2
            """, user_id, upload_id)
            
            return "DELETE 1" in result
    
    async def migrate_session_upload(
        self,
        user_id: str,
        upload_data: Dict[str, Any]
    ) -> bool:
        """Migrate a session upload to user's account."""
        await self._ensure_schema()
        
        try:
            # Rebuild chunks from session data for storage
            embeddings = upload_data.get("embeddings", [])
            chunk_metadata = upload_data.get("chunk_metadata", [])
            
            # Create chunks list for storage
            chunks = []
            for meta in chunk_metadata:
                chunks.append({
                    "chunk_id": meta.get("id"),
                    "text": meta.get("text"),
                    "section": meta.get("section"),
                    "chunk_type": meta.get("chunk_type"),
                    "page": meta.get("page", 1),
                    "word_count": meta.get("word_count", 0),
                })
            
            await self._store_in_postgres(
                user_id=user_id,
                upload_id=upload_data.get("upload_id"),
                filename=upload_data.get("filename"),
                title=upload_data.get("title"),
                doc_meta=upload_data.get("doc_meta", {}),
                chunks=chunks,
                embeddings=embeddings
            )
            return True
        except Exception as e:
            print(f"Error migrating upload: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def get_upload_study_profile(
        self,
        user_id: str,
        upload_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the study profile for a user's uploaded document.
        
        Returns the study_profile JSONB stored in user_uploads table.
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT upload_id, doc_id, filename, title, study_profile, doi, pmid
                FROM user_uploads
                WHERE user_id = $1 AND upload_id = $2
            """, user_id, upload_id)
            
            if not row:
                return None
            
            study_profile = row["study_profile"]
            if isinstance(study_profile, str):
                study_profile = json.loads(study_profile)
            
            return {
                "upload_id": row["upload_id"],
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "title": row["title"],
                "study_profile": study_profile,
                "doi": row["doi"],
                "pmid": row["pmid"]
            }
    
    async def get_all_uploads_with_profiles(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all uploads with study profiles in a single query.
        Much faster than fetching uploads then profiles separately.
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT upload_id, doc_id, filename, title, study_profile, doi, pmid
                FROM user_uploads
                WHERE user_id = $1 AND study_profile IS NOT NULL
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)
            
            results = []
            for row in rows:
                study_profile = row["study_profile"]
                if isinstance(study_profile, str):
                    study_profile = json.loads(study_profile)
                
                if study_profile:  # Only include if profile exists
                    results.append({
                        "upload_id": row["upload_id"],
                        "doc_id": row["doc_id"],
                        "filename": row["filename"],
                        "title": row["title"],
                        "study_profile": study_profile,
                        "doi": row["doi"],
                        "pmid": row["pmid"]
                    })
            
            return results
    
    async def get_upload_chunks(
        self,
        user_id: str,
        upload_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get chunks for a specific upload."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT chunk_metadata
                FROM user_uploads
                WHERE user_id = $1 AND upload_id = $2
            """, user_id, upload_id)
            
            if not row:
                return None
            
            chunk_metadata = row["chunk_metadata"]
            if isinstance(chunk_metadata, str):
                chunk_metadata = json.loads(chunk_metadata)
            
            return chunk_metadata


# Singleton
_user_uploads_service: Optional[UserUploadsService] = None


def get_user_uploads_service() -> UserUploadsService:
    global _user_uploads_service
    if _user_uploads_service is None:
        _user_uploads_service = UserUploadsService()
    return _user_uploads_service
