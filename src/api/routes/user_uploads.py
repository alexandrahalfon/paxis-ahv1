"""
User Uploads API Routes

Provides endpoints for users to upload and manage their own documents.
Documents are processed, chunked, and embedded locally (not in Qdrant).
"""

import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.api.services.auth_dependencies import get_current_user_optional, require_physician
from src.api.services.user_uploads_service import get_user_uploads_service

router = APIRouter(prefix="/user-uploads", tags=["user-uploads"])


class UserUpload(BaseModel):
    """A user upload record"""
    id: int
    upload_id: str
    filename: str
    title: Optional[str]
    status: str
    doc_meta: Dict[str, Any]
    chunk_count: int
    error_message: Optional[str]
    created_at: Optional[str]
    processed_at: Optional[str]


class UserUploadsListResponse(BaseModel):
    """Response with list of user uploads"""
    success: bool
    total: int
    uploads: List[UserUpload]


class UploadProcessingResponse(BaseModel):
    """Response after processing an upload"""
    success: bool
    upload_id: str
    doc_id: Optional[str] = None  # For study details lookup
    filename: str
    title: Optional[str]
    chunk_count: int
    embedding_dim: int = 0
    stored: str  # "database" or "session"
    message: str
    embeddings: Optional[List[List[float]]] = None  # Embeddings array (session only)
    chunk_metadata: Optional[List[Dict[str, Any]]] = None  # Chunk metadata (session only)
    doc_meta: Optional[Dict[str, Any]] = None
    study_profile: Optional[Dict[str, Any]] = None  # Extracted study profile for display
    reused_existing: bool = False  # True if document was found in existing processed docs


class MigrateUploadRequest(BaseModel):
    """Request to migrate a session upload to account"""
    upload_id: str
    filename: str
    title: Optional[str]
    doc_meta: Dict[str, Any] = {}
    embeddings: List[List[float]]  # Embeddings array
    chunk_metadata: List[Dict[str, Any]]  # Chunk metadata


@router.post("/process", response_model=UploadProcessingResponse)
async def process_user_upload(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Upload and process a PDF document.
    
    The document will be:
    1. Processed (OCR, vision analysis)
    2. Chunked into paragraphs and table rows
    3. Embedded using OpenAI embeddings
    
    If logged in: stored in PostgreSQL with your account
    If not logged in: returned for session storage (temporary)
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = Path(tmp.name)
        
        service = get_user_uploads_service()
        
        # Get user_id if available
        user_id = current_user.get("id") if current_user else None
        
        result = await service.process_and_store_document(
            file_path=temp_path,
            filename=file.filename,
            user_id=user_id
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Processing failed")
            )
        
        return UploadProcessingResponse(
            success=True,
            upload_id=result["upload_id"],
            doc_id=result.get("doc_id"),
            filename=result["filename"],
            title=result.get("title"),
            chunk_count=result["chunk_count"],
            embedding_dim=result.get("embedding_dim", 0),
            stored=result["stored"],
            message=result["message"],
            embeddings=result.get("embeddings"),  # Embeddings array (session)
            chunk_metadata=result.get("chunk_metadata"),  # Metadata (session)
            doc_meta=result.get("doc_meta"),
            study_profile=result.get("study_profile"),  # Study profile for display
            reused_existing=result.get("reused_existing", False)  # Was existing doc reused
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=UserUploadsListResponse)
async def get_user_uploads(
    limit: int = 50,
    current_user: dict = Depends(require_physician)
):
    """
    Get all uploads for the current user.
    """
    try:
        service = get_user_uploads_service()
        uploads = await service.get_user_uploads(
            user_id=current_user["id"],
            limit=limit
        )
        
        return UserUploadsListResponse(
            success=True,
            total=len(uploads),
            uploads=[UserUpload(**u) for u in uploads]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving uploads: {str(e)}")


@router.get("/{upload_id}/chunks")
async def get_upload_chunks(
    upload_id: str,
    current_user: dict = Depends(require_physician)
):
    """
    Get chunks for a specific upload.
    """
    try:
        service = get_user_uploads_service()
        chunks = await service.get_upload_chunks(
            user_id=current_user["id"],
            upload_id=upload_id
        )
        
        if chunks is None:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {"success": True, "chunks": chunks}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")


@router.delete("/{upload_id}")
async def delete_user_upload(
    upload_id: str,
    current_user: dict = Depends(require_physician)
):
    """
    Delete a user upload.
    """
    try:
        service = get_user_uploads_service()
        deleted = await service.delete_upload(
            user_id=current_user["id"],
            upload_id=upload_id
        )
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {"success": True, "message": "Upload deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting upload: {str(e)}")


@router.post("/migrate")
async def migrate_session_upload(
    request: MigrateUploadRequest,
    current_user: dict = Depends(require_physician)
):
    """
    Migrate a session upload to the user's account.
    Called after login to persist session uploads.
    """
    try:
        service = get_user_uploads_service()
        success = await service.migrate_session_upload(
            user_id=current_user["id"],
            upload_data={
                "upload_id": request.upload_id,
                "filename": request.filename,
                "title": request.title,
                "doc_meta": request.doc_meta,
                "embeddings": request.embeddings,
                "chunk_metadata": request.chunk_metadata
            }
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Migration failed")
        
        return {"success": True, "message": "Upload migrated to your account"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


@router.get("/{upload_id}/study-profile")
async def get_user_study_profile(
    upload_id: str,
    current_user: dict = Depends(require_physician)
):
    """
    Get the study profile for a user's uploaded document.
    Returns the extracted study details for display.
    """
    try:
        service = get_user_uploads_service()
        result = await service.get_upload_study_profile(
            user_id=current_user["id"],
            upload_id=upload_id
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        if not result.get("study_profile"):
            raise HTTPException(status_code=404, detail="Study profile not available for this upload")
        
        return {"success": True, "profile": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving study profile: {str(e)}")


@router.get("/study-profiles/all")
async def get_all_user_study_profiles(
    limit: int = 50,
    current_user: dict = Depends(require_physician)
):
    """
    Get all study profiles for the current user.
    Returns uploads that have study_profile data.
    """
    try:
        service = get_user_uploads_service()
        # Use optimized single-query method instead of N+1 queries
        profiles = await service.get_all_uploads_with_profiles(
            user_id=current_user["id"],
            limit=limit
        )
        
        return {"success": True, "total": len(profiles), "profiles": profiles}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving study profiles: {str(e)}")


@router.get("/{upload_id}/similar-studies")
async def find_similar_studies(
    upload_id: str,
    limit: int = 10,
    current_user: dict = Depends(require_physician)
):
    """
    Find similar studies from the knowledge base for a user's uploaded study.
    
    First searches PostgreSQL study-profiles database for studies with matching 
    characteristics (cancer type, location, treatment), then falls back to 
    Qdrant vector search if needed.
    
    Returns similar studies with a comparison summary.
    """
    from qdrant_client import QdrantClient
    from src.core.config import settings
    from src.api.services.study_profiles_filtering import get_study_profiles_filtering_service
    from openai import OpenAI
    import json
    
    try:
        service = get_user_uploads_service()
        
        # Get the user's study profile
        profile_result = await service.get_upload_study_profile(
            user_id=current_user["id"],
            upload_id=upload_id
        )
        
        if not profile_result:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        study_profile = profile_result.get("study_profile")
        if not study_profile:
            raise HTTPException(status_code=404, detail="Study profile not available for this upload")
        
        # Extract key characteristics from study profile
        study_details = study_profile.get("study_details", {})
        diagnosis = study_profile.get("diagnosis", {})
        treatment = study_profile.get("treatment", {})
        outcomes = study_profile.get("outcomes", {})
        
        # Cancer type
        cancer_type = diagnosis.get("cancer_type", {}).get("value")
        # Cancer location
        cancer_location = diagnosis.get("cancer_location", {}).get("value")
        # Treatment modality
        treatment_modality = treatment.get("treatment_modality", {}).get("value")
        # Study type
        study_type = study_details.get("study_type", {}).get("value")
        # Histology
        histology = diagnosis.get("histopathologic_type", {}).get("value")
        
        # Exclude the user's own study
        user_doc_id = profile_result.get("doc_id")
        user_doi = profile_result.get("doi")
        
        similar_studies = []
        search_method = "none"
        
        # ============================================
        # Strategy 1: Search study-profiles PostgreSQL database (structured data)
        # ============================================
        if cancer_type or cancer_location or histology:
            try:
                print(f"[SimilarStudies] Searching study-profiles DB with: cancer_type={cancer_type}, location={cancer_location}, histology={histology}")
                
                # Use StudyProfilesFilteringService which connects to study-profiles database
                profiles_service = get_study_profiles_filtering_service()
                
                filter_result = await profiles_service.filter_studies_by_profile(
                    cancer_type=cancer_type,
                    anatomical_site=cancer_location,
                    histology=histology,
                    limit=limit * 2  # Get more to filter
                )
                
                # Filter out user's own study and format results
                for match in filter_result.matches:
                    doc_id = match.doc_id
                    
                    # Skip user's own study
                    if doc_id and user_doc_id and doc_id == user_doc_id:
                        continue
                    
                    similar_studies.append({
                        "doc_id": doc_id,
                        "title": match.study_name,
                        "doi": None,  # Not in this query
                        "pmid": None,  # Not in this query
                        "author": None,  # Not in this query
                        "year": None,  # Not in this query
                        "category": match.cancer_type,
                        "relevance_score": 0.9,  # High score for structured match
                        "cancer_type": match.cancer_type,
                        "cancer_location": match.cancer_location,
                        "study_type": None,
                        "number_of_patients": match.number_of_patients,
                        "overall_survival": None,
                        "source": "postgres"
                    })
                    
                    if len(similar_studies) >= limit:
                        break
                
                if similar_studies:
                    search_method = "postgres"
                    print(f"[SimilarStudies] Found {len(similar_studies)} matches in study-profiles DB")
                    
            except Exception as e:
                print(f"[SimilarStudies] PostgreSQL search failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ============================================
        # Strategy 2: Fall back to Qdrant vector search
        # ============================================
        if len(similar_studies) < limit:
            try:
                # Build search query from study profile
                search_terms = []
                if cancer_type:
                    search_terms.append(cancer_type)
                if cancer_location:
                    search_terms.append(cancer_location)
                if treatment_modality:
                    search_terms.append(treatment_modality)
                if study_type:
                    search_terms.append(study_type)
                
                # Fallback to title if no terms
                if not search_terms:
                    title = profile_result.get("title")
                    if title:
                        search_terms.append(title)
                
                if search_terms:
                    search_query = " ".join(search_terms)
                    print(f"[SimilarStudies] Searching Qdrant with: {search_query[:100]}")
                    
                    # Initialize OpenAI for embeddings
                    openai_client = OpenAI(api_key=settings.openai_api_key)
                    
                    # Generate embedding for search query - use same model as collection (text-embedding-3-large)
                    embed_response = openai_client.embeddings.create(
                        model=settings.embed_model,  # text-embedding-3-large (3072 dim)
                        input=search_query
                    )
                    query_embedding = embed_response.data[0].embedding
                    
                    # Initialize Qdrant client
                    qdrant_client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None
                    )
                    
                    # Search Qdrant
                    search_results = qdrant_client.query_points(
                        collection_name=settings.qdrant_collection,
                        query=query_embedding,
                        limit=(limit - len(similar_studies)) * 3,
                        with_payload=True,
                        with_vectors=False,
                    ).points
                    
                    # Track already-added doc_ids
                    existing_doc_ids = {s["doc_id"] for s in similar_studies if s.get("doc_id")}
                    
                    for result in search_results:
                        payload = dict(result.payload or {})
                        doc_meta = payload.get("doc_meta", {})
                        doc_id = payload.get("doc_id")
                        doi = payload.get("doi") or doc_meta.get("doi")
                        
                        # Skip user's own study
                        if doc_id and user_doc_id and doc_id == user_doc_id:
                            continue
                        if doi and user_doi and doi == user_doi:
                            continue
                        
                        # Skip duplicates
                        if doc_id in existing_doc_ids:
                            continue
                        existing_doc_ids.add(doc_id)
                        
                        # Extract title from doc_meta (where it's actually stored)
                        title = (
                            payload.get("title") or 
                            doc_meta.get("title") or 
                            doc_meta.get("study_name") or
                            "Unknown Study"
                        )
                        
                        similar_studies.append({
                            "doc_id": doc_id,
                            "title": title,
                            "doi": doi,
                            "pmid": payload.get("pmid") or doc_meta.get("pmid"),
                            "author": payload.get("author") or doc_meta.get("author_et_al") or doc_meta.get("authors"),
                            "year": payload.get("year") or doc_meta.get("year"),
                            "category": payload.get("category"),
                            "relevance_score": float(result.score) if result.score else 0,
                            "source": "qdrant"
                        })
                        
                        if len(similar_studies) >= limit:
                            break
                    
                    if search_method == "none" and similar_studies:
                        search_method = "qdrant"
                    elif similar_studies:
                        search_method = "postgres+qdrant"
                        
                    print(f"[SimilarStudies] Total after Qdrant: {len(similar_studies)}")
                    
            except Exception as e:
                print(f"[SimilarStudies] Qdrant search failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ============================================
        # Generate comparison summary
        # ============================================
        comparison_summary = None
        if similar_studies:
            try:
                openai_client = OpenAI(api_key=settings.openai_api_key)
                
                user_study_summary = f"""User's Study: {profile_result.get('title', 'Unknown')}
- Cancer Type: {cancer_type or 'N/A'}
- Location: {cancer_location or 'N/A'}
- Treatment: {treatment_modality or 'N/A'}
- Study Type: {study_type or 'N/A'}"""
                
                similar_summaries = []
                for i, s in enumerate(similar_studies[:5]):
                    info = f"{i+1}. {s.get('title', 'Unknown')}"
                    if s.get('year'):
                        info += f" ({s['year']})"
                    if s.get('cancer_type'):
                        info += f" - {s['cancer_type']}"
                    if s.get('number_of_patients'):
                        info += f" [n={s['number_of_patients']}]"
                    similar_summaries.append(info)
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are a clinical research expert. 
Briefly summarize how the user's study compares to similar studies found in the knowledge base.
Focus on: what makes these studies similar, and what key differences might exist.
Keep it to 2-3 sentences."""},
                        {"role": "user", "content": f"""{user_study_summary}

Similar studies found:
{chr(10).join(similar_summaries)}

Provide a brief comparison summary:"""}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                comparison_summary = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating comparison summary: {e}")
                comparison_summary = f"Found {len(similar_studies)} similar studies based on {cancer_type or cancer_location or 'study characteristics'}."
        
        return {
            "success": True,
            "user_study": {
                "upload_id": upload_id,
                "doc_id": user_doc_id,
                "title": profile_result.get("title"),
                "doi": user_doi,
                "cancer_type": cancer_type,
                "cancer_location": cancer_location,
                "treatment_modality": treatment_modality,
                "study_type": study_type
            },
            "search_method": search_method,
            "similar_studies": similar_studies,
            "total": len(similar_studies),
            "comparison_summary": comparison_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error finding similar studies: {str(e)}")
