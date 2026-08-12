"""
Query Classifier API Routes

Provides endpoints for classifying clinical queries into structured data
that can be matched against the PostgreSQL studies database.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.api.services.query_classifier_service import (
    get_query_classifier_service,
    StructuredQuery
)

router = APIRouter(prefix="/query-classifier", tags=["query-classifier"])


class ClassifyQueryRequest(BaseModel):
    """Request to classify a clinical query"""
    query: str
    include_sql_filters: bool = False


class ClassifiedQueryResponse(BaseModel):
    """Response with classified/structured query data"""
    success: bool
    query_summary: str
    
    # Extracted fields organized by category
    demographics: Dict[str, Any]
    diagnosis: Dict[str, Any]
    staging: Dict[str, Any]
    pathology: Dict[str, Any]
    treatment_history: Dict[str, Any]
    risk_factors: Dict[str, Any]
    
    # All extracted fields as flat dict
    all_fields: Dict[str, Any]
    
    # Optional SQL filter info
    sql_conditions: Optional[List[str]] = None
    sql_params: Optional[List[Any]] = None


@router.post("/classify", response_model=ClassifiedQueryResponse)
async def classify_query(request: ClassifyQueryRequest):
    """
    Classify a clinical query into structured data.
    
    Takes a free-text clinical description and extracts structured fields
    that can be used to search the studies database.
    
    Example input:
    "68 year old female, non-smoker, with SCC of R maxilla s/p maxillectomy, 
    pT4N0, DOI 15 mm, LVI-, PNI-, margins-, 0/42 LN involved"
    
    Returns structured data including:
    - Demographics (age, gender, performance status)
    - Diagnosis (cancer type, location, histology)
    - Staging (TNM, overall stage)
    - Pathology (DOI, LVI, PNI, margins, lymph nodes)
    - Treatment history (prior surgery, recurrence)
    - Risk factors (smoking, comorbidities)
    """
    if not request.query or len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 10 characters"
        )
    
    try:
        service = get_query_classifier_service()
        structured = service.classify_query(request.query)
        
        # Organize fields by category
        demographics = {}
        if structured.age:
            demographics["age"] = structured.age
        if structured.gender:
            demographics["gender"] = structured.gender
        if structured.race_ethnicity:
            demographics["race_ethnicity"] = structured.race_ethnicity
        if structured.performance_status:
            demographics["performance_status"] = f"ECOG {structured.performance_status}"
        
        diagnosis = {}
        if structured.cancer_type:
            diagnosis["cancer_type"] = structured.cancer_type
        if structured.cancer_location:
            diagnosis["cancer_location"] = structured.cancer_location
        if structured.histopathologic_type:
            diagnosis["histopathologic_type"] = structured.histopathologic_type
        if structured.tumor_grade:
            diagnosis["tumor_grade"] = structured.tumor_grade
        if structured.molecular_subtype:
            diagnosis["molecular_subtype"] = structured.molecular_subtype
        
        staging = {}
        if structured.tnm_t:
            staging["T_stage"] = structured.tnm_t
        if structured.tnm_n:
            staging["N_stage"] = structured.tnm_n
        if structured.tnm_m:
            staging["M_stage"] = structured.tnm_m
        if structured.overall_stage:
            staging["overall_stage"] = structured.overall_stage
        if structured.risk_stratification:
            staging["risk_stratification"] = structured.risk_stratification
        if structured.metastatic_status:
            staging["metastatic_status"] = structured.metastatic_status
        
        pathology = {}
        if structured.depth_of_invasion:
            pathology["depth_of_invasion"] = structured.depth_of_invasion
        if structured.lymphovascular_invasion:
            pathology["lymphovascular_invasion"] = structured.lymphovascular_invasion
        if structured.perineural_invasion:
            pathology["perineural_invasion"] = structured.perineural_invasion
        if structured.margin_status:
            pathology["margin_status"] = structured.margin_status
        if structured.lymph_nodes_examined is not None:
            pathology["lymph_nodes_examined"] = structured.lymph_nodes_examined
        if structured.lymph_nodes_positive is not None:
            pathology["lymph_nodes_positive"] = structured.lymph_nodes_positive
        
        treatment_history = {}
        if structured.prior_surgery:
            treatment_history["prior_surgery"] = structured.prior_surgery
        if structured.prior_radiation is not None:
            treatment_history["prior_radiation"] = structured.prior_radiation
        if structured.prior_chemotherapy is not None:
            treatment_history["prior_chemotherapy"] = structured.prior_chemotherapy
        if structured.recurrence_status:
            treatment_history["recurrence_status"] = structured.recurrence_status
        
        risk_factors = {}
        if structured.smoking_status:
            risk_factors["smoking_status"] = structured.smoking_status
        if structured.comorbidities:
            risk_factors["comorbidities"] = structured.comorbidities
        
        # Build response
        response = ClassifiedQueryResponse(
            success=True,
            query_summary=structured.get_search_summary(),
            demographics=demographics,
            diagnosis=diagnosis,
            staging=staging,
            pathology=pathology,
            treatment_history=treatment_history,
            risk_factors=risk_factors,
            all_fields=structured.to_dict()
        )
        
        # Add SQL filters if requested
        if request.include_sql_filters:
            filters = service.build_postgres_filters(structured)
            response.sql_conditions = filters["conditions"]
            response.sql_params = filters["params"]
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error classifying query: {str(e)}"
        )


class SearchStudiesRequest(BaseModel):
    """Request to search studies with a classified query"""
    query: str
    limit: int = 20
    include_qdrant_search: bool = True


class MatchingStudy(BaseModel):
    """A study matching the query criteria"""
    study_id: int
    doc_id: Optional[str]
    study_name: Optional[str]
    cancer_type: Optional[str]
    cancer_location: Optional[str]
    study_type: Optional[str]
    number_of_patients: Optional[int]
    match_score: float
    matched_fields: List[str]


class SearchStudiesResponse(BaseModel):
    """Response with matching studies"""
    success: bool
    query_summary: str
    extracted_fields: Dict[str, Any]
    total_matches: int
    studies: List[MatchingStudy]


@router.post("/search", response_model=SearchStudiesResponse)
async def search_studies_with_query(request: SearchStudiesRequest):
    """
    Search studies using a classified clinical query.
    
    This endpoint:
    1. Classifies the query into structured fields
    2. Uses Qdrant vector search for semantic relevance
    3. Optionally filters with PostgreSQL for structured fields
    4. Returns ranked results with match explanations
    """
    if not request.query or len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 10 characters"
        )
    
    try:
        from src.api.routes.user_preferences import get_study_pool
        from src.core.config import settings
        from qdrant_client import QdrantClient
        from openai import OpenAI
        
        service = get_query_classifier_service()
        structured = service.classify_query(request.query)
        
        print(f"[QueryClassifier] Query: {request.query}")
        print(f"[QueryClassifier] Extracted cancer_type: {structured.cancer_type}")
        
        studies = []
        doc_id_to_score = {}
        
        # Use Qdrant vector search for semantic relevance
        if request.include_qdrant_search:
            try:
                openai_client = OpenAI(api_key=settings.openai_api_key)
                qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )
                
                # Generate embedding for the query
                embed_response = openai_client.embeddings.create(
                    model=settings.embed_model,
                    input=request.query
                )
                query_vector = embed_response.data[0].embedding
                
                # Search Qdrant for relevant chunks using query_points (new API)
                search_results = qdrant_client.query_points(
                    collection_name=settings.qdrant_collection,
                    query=query_vector,
                    limit=request.limit * 3,  # Get more to dedupe by doc_id
                    with_payload=True
                ).points
                
                # Aggregate scores by doc_id (take max score per document)
                for result in search_results:
                    doc_id = result.payload.get("doc_id")
                    if doc_id:
                        current_score = doc_id_to_score.get(doc_id, 0)
                        doc_id_to_score[doc_id] = max(current_score, result.score)
                
                print(f"[QueryClassifier] Qdrant found {len(doc_id_to_score)} unique documents")
                
            except Exception as e:
                print(f"[QueryClassifier] Qdrant search failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Get study metadata from PostgreSQL
        pool = await get_study_pool()
        filters = service.build_postgres_filters(structured)
        
        async with pool.acquire() as conn:
            # If we have Qdrant results, fetch those specific studies
            if doc_id_to_score:
                doc_ids = list(doc_id_to_score.keys())
                placeholders = ", ".join(f"${i+1}" for i in range(len(doc_ids)))
                query = f"""
                    SELECT 
                        study_id,
                        doc_id,
                        study_name,
                        cancer_type,
                        cancer_location,
                        study_type,
                        number_of_patients
                    FROM studies
                    WHERE doc_id IN ({placeholders})
                """
                rows = await conn.fetch(query, *doc_ids)
                
                # Build studies with Qdrant relevance scores
                extracted = structured.to_dict()
                for row in rows:
                    doc_id = row['doc_id']
                    qdrant_score = doc_id_to_score.get(doc_id, 0)
                    
                    matched_fields = ["semantic_match"]
                    
                    # Boost score if structured fields also match
                    if structured.cancer_type and row['cancer_type']:
                        if structured.cancer_type.lower() in row['cancer_type'].lower():
                            matched_fields.append("cancer_type")
                            qdrant_score += 0.1
                    
                    if structured.cancer_location and row['cancer_location']:
                        if structured.cancer_location.lower() in row['cancer_location'].lower():
                            matched_fields.append("cancer_location")
                            qdrant_score += 0.05
                    
                    studies.append(MatchingStudy(
                        study_id=row['study_id'],
                        doc_id=doc_id,
                        study_name=row['study_name'],
                        cancer_type=row['cancer_type'],
                        cancer_location=row['cancer_location'],
                        study_type=row['study_type'],
                        number_of_patients=row['number_of_patients'],
                        match_score=min(qdrant_score, 1.0),
                        matched_fields=matched_fields
                    ))
            
            # Fallback to PostgreSQL filter search if no Qdrant results
            elif filters["conditions"]:
                where_clause = " AND ".join(filters["conditions"])
                query = f"""
                    SELECT 
                        study_id,
                        doc_id,
                        study_name,
                        cancer_type,
                        cancer_location,
                        study_type,
                        number_of_patients
                    FROM studies
                    WHERE {where_clause}
                    ORDER BY number_of_patients DESC NULLS LAST
                    LIMIT {request.limit}
                """
                rows = await conn.fetch(query, *filters["params"])
                
                extracted = structured.to_dict()
                for row in rows:
                    matched_fields = []
                    score = 0.5  # Base score for filter match
                    
                    if structured.cancer_type and row['cancer_type']:
                        if structured.cancer_type.lower() in row['cancer_type'].lower():
                            matched_fields.append("cancer_type")
                            score += 0.2
                    
                    if structured.cancer_location and row['cancer_location']:
                        if structured.cancer_location.lower() in row['cancer_location'].lower():
                            matched_fields.append("cancer_location")
                            score += 0.1
                    
                    studies.append(MatchingStudy(
                        study_id=row['study_id'],
                        doc_id=row['doc_id'],
                        study_name=row['study_name'],
                        cancer_type=row['cancer_type'],
                        cancer_location=row['cancer_location'],
                        study_type=row['study_type'],
                        number_of_patients=row['number_of_patients'],
                        match_score=min(score, 1.0),
                        matched_fields=matched_fields if matched_fields else ["filter_match"]
                    ))
            
            else:
                # No filters and no Qdrant - return empty with message
                print(f"[QueryClassifier] No filters extracted and Qdrant search disabled/failed")
        
        # Sort by match score
        studies.sort(key=lambda x: x.match_score, reverse=True)
        
        # Limit results
        studies = studies[:request.limit]
        
        return SearchStudiesResponse(
            success=True,
            query_summary=structured.get_search_summary(),
            extracted_fields=structured.to_dict(),
            total_matches=len(studies),
            studies=studies
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error searching studies: {str(e)}"
        )
