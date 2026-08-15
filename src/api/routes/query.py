"""
FastAPI Routes for Enhanced RAG Queries

Provides endpoints for:
- Simple queries
- Deep dive queries with site context
- Query modes and available sites
- Health check
"""

import re
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict, Any, Tuple

from src.api.models.query_models import (
    QueryRequest,
    QueryResponse,
    DeepDiveRequest,
    DeepDiveResponse,
    HealthCheckResponse,
    QueryModesResponse,
    QueryMode,
    SitesResponse,
    SiteInfo,
    RetrievalResult,
    QueryMetadata,
    TableInfo,
    PatientProfile,
    StudyMatch,
    PatientMatchResponse,
    TreatmentComparisonRequest,
    TreatmentEvidence,
    TreatmentComparisonResult,
    TreatmentComparisonResponse,
    EnhancedQueryResponse,
    PTOFrame,
    Artifact,
    ChartArtifact,
    ChartDataset,
    VisualComparisonRequest,
    VisualComparisonResponse,
    TreatmentArmResult,
    IntentAnalysisRequest,
    IntentAnalysisResponse,
    QueryIntentInfo,
    ExtractedPatientProfile,
    FollowUpOptionInfo,
    MatchingTrialInfo,
    ClassifyRequest,
    ClassifyResponse,
    StructuredSummary,
    GuidelineAlignment,
)
from src.api.services.auth_dependencies import get_current_user_optional
from src.api.services.cache_service import get_cache_service, make_cache_key

import math

router = APIRouter(prefix="/rag")


def _normalize_crossencoder_score(raw: float) -> int:
    """Convert a cross-encoder logit to a 0-100 integer percentage via sigmoid.

    If the raw value is strictly between 0 and 1 (exclusive), it is already a
    probability and is scaled directly.  Otherwise (logits, 0, and 1) sigmoid
    is applied so that a neutral logit of 0 correctly maps to 50%.
    """
    if raw is None:
        return None
    if 0 < raw < 1:
        return round(raw * 100)
    return round(100 / (1 + math.exp(-raw)))


_WEB_BLOCKED_TYPES = {"treatment_recommendation", "mechanism", "side_effects"}
_RECENCY_RE = re.compile(r"\b(recent|latest|new(?:est)?|updated?|202[3-9]|20[3-9]\d)\b", re.IGNORECASE)

_SUPPLEMENT_THRESHOLDS = {
    "trial_results":      70,
    "clinical_evidence":  65,
    "staging":            60,
    "workup":             60,
    "general":            55,
    "patient_specific":   55,
}
_DEFAULT_SUPPLEMENT_THRESHOLD = 45


def _compute_avg_score(kb_results: list) -> float:
    """Return sigmoid-normalized 0-100 avg score for top-5 KB results."""
    top5_scores = [
        r.get("score_crossencoder") or r.get("score", 0)
        for r in kb_results[:5]
        if r.get("score_crossencoder") is not None or r.get("score") is not None
    ]
    if not top5_scores:
        return 0.0
    raw_avg = sum(top5_scores) / len(top5_scores)
    if raw_avg != 0 and abs(raw_avg) > 1:
        return round(100 / (1 + math.exp(-raw_avg)))
    return raw_avg * 100


def _score_web_need(query: str, query_type: str, kb_results: list) -> Tuple[Optional[str], bool]:
    """
    Decide whether to supplement or fall back to web evidence.

    Returns:
      ("supplement", True)  — KB has some results but web would add value; mix web at the tail
      ("fallback",   True)  — KB has nothing useful; web is primary source
      (None,         False) — KB is sufficient; do not fetch web
    """
    # Step 1 — hard block
    if query_type in _WEB_BLOCKED_TYPES:
        return None, False

    avg_score = _compute_avg_score(kb_results)
    has_recency = bool(_RECENCY_RE.search(query))

    def _norm_single(r: dict) -> float:
        raw = r.get("score_crossencoder") or r.get("score", 0)
        if raw is None:
            return 0.0
        if abs(raw) > 1:
            return round(100 / (1 + math.exp(-raw)))
        return raw * 100

    n_above_50 = sum(1 for r in kb_results[:5] if _norm_single(r) >= 50)

    # Step 3 — immediate fallback (KB clearly failing)
    if len(kb_results) == 0 or avg_score < 30:
        return "fallback", True

    # Step 4 — recency override (always supplement for time-sensitive queries)
    if has_recency:
        return "supplement", True

    # Step 5 — coverage gap (fewer than 3 results with decent scores)
    if n_above_50 < 3:
        return "supplement", True

    # Step 6 — type-specific thresholds
    threshold = _SUPPLEMENT_THRESHOLDS.get(query_type, _DEFAULT_SUPPLEMENT_THRESHOLD)
    if avg_score < threshold:
        return "supplement", True

    # Step 7 — KB is sufficient
    return None, False


def _fetch_web_evidence(query: str, query_type: str, mode: str) -> list:
    """
    Fetch PubMed and/or ClinicalTrials.gov evidence.

    mode="supplement": 5 PubMed + 3 CT (trial_results only), scores 0.38 / 0.32
    mode="fallback":   8 PubMed + 5 CT,                      scores 0.35 / 0.30
    """
    try:
        from src.api.services.literature_search_service import LiteratureSearchService
        svc = LiteratureSearchService()
        web_evidence = []

        pubmed_count = 5 if mode == "supplement" else 8
        pubmed_score = 0.38 if mode == "supplement" else 0.35
        ct_count     = 3 if mode == "supplement" else 5
        ct_score     = 0.32 if mode == "supplement" else 0.30

        # PubMed
        articles = svc.search_pubmed(query, max_results=pubmed_count)
        for a in articles:
            # Authors come back as "LastName ForeName" from
            # literature_search_service, so the surname is the first token.
            # This previously took the last token and rendered the given
            # name instead: "Burtness Barbara" -> "Barbara et al."
            authors = [x for x in (a.get("authors") or []) if x]
            author_str = f"{authors[0].split()[0]} et al." if authors else None
            citation = (
                f"{author_str}, {a.get('journal', '')}, {a.get('year', '')}"
                if author_str else a.get("journal", "PubMed")
            )
            web_evidence.append({
                "doc_id": f"pubmed_{a.get('pmid', '')}",
                "title": a.get("title", ""),
                "text": a.get("abstract", ""),
                "citation": citation,
                "doi": a.get("doi"),
                "pmid": a.get("pmid"),
                "year": a.get("year"),
                "score": pubmed_score,
                "score_crossencoder": None,
                "source_type": "pubmed",
            })

        # ClinicalTrials.gov — always in fallback, only for trial_results in supplement
        if mode == "fallback" or query_type == "trial_results":
            trials = svc.search_clinical_trials_by_query(query, max_results=ct_count)
            for t in trials:
                web_evidence.append({
                    "doc_id": f"ct_{t.get('nct_id', '')}",
                    "title": t.get("title", ""),
                    "text": t.get("brief_summary", ""),
                    "citation": f"ClinicalTrials.gov {t.get('nct_id', '')}",
                    "doi": None,
                    "pmid": None,
                    "year": None,
                    "score": ct_score,
                    "score_crossencoder": None,
                    "source_type": "clinicaltrials",
                })

        print(f"[web_evidence] mode={mode} type={query_type}: {len(web_evidence)} results fetched")
        return web_evidence
    except Exception as e:
        print(f"[web_evidence] Fetch failed (non-fatal): {e}")
        return []


def _generate_structured_summary(
    justification: str,
    query_type: str,
    openai_client,
    top_relevance: float = 0.0,
) -> Optional[StructuredSummary]:
    """Extract a key-finding summary card from the justification using gpt-4o-mini.

    `top_relevance` is the highest cross-encoder relevance_score (0-100) across
    the cited evidence. We use it to CAP the LLM's self-reported evidence_level
    so the summary card can't claim "High Evidence" when the actual retrieved
    chunks score "Limited relevance" on the cross-encoder. Without this cap
    the UI shows two contradictory badges side-by-side (observed live on an
    H&N nodal-recurrence case: LLM said "High Evidence Strong", cross-encoder
    said "Limited relevance" — they were both displayed).
    """
    try:
        import json
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical summarizer. Extract a structured summary from the provided oncology response. "
                        "Return JSON with exactly these fields:\n"
                        '- "key_finding": 1-2 sentence clinical takeaway\n'
                        '- "evidence_level": one of "High", "Moderate", "Low", "Insufficient"\n'
                        '- "recommendation_strength": one of "Strong", "Conditional", "Expert Opinion"\n'
                        '- "caveats": brief caveat or null if none\n'
                        "Base evidence_level on: High=RCT/meta-analysis, Moderate=prospective/registry, "
                        "Low=retrospective/case series, Insufficient=case reports/expert opinion only."
                    )
                },
                {
                    "role": "user",
                    "content": f"Query type: {query_type}\n\nResponse:\n{justification[:3000]}"
                }
            ]
        )
        data = json.loads(response.choices[0].message.content)
        evidence_level = data.get("evidence_level", "Moderate")
        recommendation_strength = data.get("recommendation_strength", "Conditional")

        # Cap evidence_level + recommendation_strength by the actual
        # cross-encoder relevance of the cited chunks. The UI's
        # `relevance-badge` uses the same buckets:
        #   ≥75 → High, 40-74 → Moderate, <40 → Limited
        # so we tier the structured-summary LLM output the same way.
        # Order: High → Moderate → Low → Insufficient
        _LVL_ORDER = ["Insufficient", "Low", "Moderate", "High"]
        if top_relevance and top_relevance > 0:
            if top_relevance < 40:
                cap = "Low"
            elif top_relevance < 75:
                cap = "Moderate"
            else:
                cap = "High"
            if _LVL_ORDER.index(evidence_level) > _LVL_ORDER.index(cap):
                print(
                    f"[structured_summary] Capping evidence_level: "
                    f"LLM said {evidence_level!r}, cross-encoder relevance "
                    f"{top_relevance:.0f}% → {cap}"
                )
                evidence_level = cap

            # Recommendation strength similarly: "Strong" only when evidence
            # is Moderate or better. When relevance is Limited (<40), max is
            # "Expert Opinion" since the system has poor evidence support.
            _STRENGTH_ORDER = ["Expert Opinion", "Conditional", "Strong"]
            if top_relevance < 40:
                strength_cap = "Expert Opinion"
            elif top_relevance < 75:
                strength_cap = "Conditional"
            else:
                strength_cap = "Strong"
            if _STRENGTH_ORDER.index(recommendation_strength) > _STRENGTH_ORDER.index(strength_cap):
                print(
                    f"[structured_summary] Capping recommendation_strength: "
                    f"LLM said {recommendation_strength!r}, cross-encoder relevance "
                    f"{top_relevance:.0f}% → {strength_cap}"
                )
                recommendation_strength = strength_cap

        return StructuredSummary(
            key_finding=data.get("key_finding", ""),
            evidence_level=evidence_level,
            recommendation_strength=recommendation_strength,
            caveats=data.get("caveats"),
        )
    except Exception as e:
        print(f"[structured_summary] Failed: {e}")
        return None


def _generate_guideline_alignment(justification: str, openai_client) -> Optional[GuidelineAlignment]:
    """Extract guideline alignment for treatment/indication responses using gpt-4o-mini."""
    try:
        import json
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical guideline analyst. From the provided oncology response, determine if the "
                        "recommendation aligns with major clinical guidelines. Return JSON with:\n"
                        '- "guideline_body": primary guideline referenced, e.g. "NCCN", "ASTRO", "ESMO", "Multiple", "None"\n'
                        '- "alignment_status": one of "Consistent", "Inconsistent", "Not addressed"\n'
                        '- "guideline_note": 1 sentence explaining alignment or gap'
                    )
                },
                {
                    "role": "user",
                    "content": f"Oncology response:\n{justification[:3000]}"
                }
            ]
        )
        data = json.loads(response.choices[0].message.content)
        return GuidelineAlignment(
            guideline_body=data.get("guideline_body", "None"),
            alignment_status=data.get("alignment_status", "Not addressed"),
            guideline_note=data.get("guideline_note", ""),
        )
    except Exception as e:
        print(f"[guideline_alignment] Failed: {e}")
        return None


def _extract_author_from_citation(citation: str) -> Optional[str]:
    """
    Extract author from citation string as a fallback when author_et_al is missing.
    
    Common citation formats:
    - "Wang et al. (2011) - Title..."
    - "Smith et al., Journal Name, 2020"
    - "Author1, Author2 (Year)"
    """
    if not citation:
        return None
    
    # Pattern 1: "Author et al." at the start
    match = re.match(r'^([A-Z][a-z]+(?:\s+et\s+al\.?))', citation)
    if match:
        return match.group(1)
    
    # Pattern 2: "Author et al." anywhere before a year
    match = re.search(r'([A-Z][a-z]+(?:\s+et\s+al\.?))\s*[\(,]\s*\d{4}', citation)
    if match:
        return match.group(1)
    
    # Pattern 3: Single author name before year in parentheses
    match = re.match(r'^([A-Z][a-z]+)\s*\(\d{4}\)', citation)
    if match:
        return match.group(1)
    
    # Pattern 4: Author name at start followed by comma
    match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,', citation)
    if match:
        return match.group(1)
    
    return None


def _get_author_with_fallback(evidence: Dict[str, Any]) -> Optional[str]:
    """
    Get author from evidence, with fallback to extracting from citation string.
    """
    # First try author_et_al
    author = evidence.get("author_et_al")
    if author:
        return author
    
    # Fallback: try to extract from citation string
    citation = evidence.get("citation")
    if citation:
        extracted = _extract_author_from_citation(citation)
        if extracted:
            return extracted
    
    return None


# ============================================
# HELPER: Enrich retrieval results with study metadata
# ============================================

async def enrich_retrieval_results_with_metadata(
    retrieval_results: List[RetrievalResult]
) -> List[RetrievalResult]:
    """
    Enrich retrieval results with study metadata (patient count, citation count)
    from the PostgreSQL database.
    
    Tries to match by doc_id first, then falls back to DOI or PMID.
    Gracefully handles missing tables.
    """
    if not retrieval_results:
        return retrieval_results
    
    # Collect unique identifiers
    doc_ids = list(set(r.doc_id for r in retrieval_results if r.doc_id))
    dois = list(set(r.doi for r in retrieval_results if r.doi))
    pmids = list(set(r.pmid for r in retrieval_results if r.pmid))
    
    if not doc_ids and not dois and not pmids:
        return retrieval_results
    
    try:
        from src.api.services.account_db import get_account_db
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # First check if the studies table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'studies'
                )
            """)
            
            if not table_exists:
                # Table doesn't exist yet - skip enrichment silently
                return retrieval_results
            
            # Batch fetch study metadata - try multiple identifiers
            rows = await conn.fetch("""
                SELECT doc_id, doi, pmid, number_of_patients, citation_count
                FROM studies
                WHERE doc_id = ANY($1) OR doi = ANY($2) OR pmid = ANY($3)
            """, doc_ids, dois, pmids)
            
            if not rows:
                return retrieval_results
            
            # Create lookup dicts by different identifiers
            by_doc_id = {}
            by_doi = {}
            by_pmid = {}
            
            for row in rows:
                meta = {
                    'number_of_patients': row['number_of_patients'],
                    'citation_count': row['citation_count']
                }
                if row['doc_id']:
                    by_doc_id[row['doc_id']] = meta
                if row['doi']:
                    by_doi[row['doi']] = meta
                if row['pmid']:
                    by_pmid[str(row['pmid'])] = meta
            
            # Enrich results - try doc_id first, then DOI, then PMID
            enriched_count = 0
            for result in retrieval_results:
                meta = None
                
                # Try doc_id first
                if result.doc_id and result.doc_id in by_doc_id:
                    meta = by_doc_id[result.doc_id]
                # Try DOI
                elif result.doi and result.doi in by_doi:
                    meta = by_doi[result.doi]
                # Try PMID
                elif result.pmid and str(result.pmid) in by_pmid:
                    meta = by_pmid[str(result.pmid)]
                
                if meta:
                    result.number_of_patients = meta.get('number_of_patients')
                    result.citation_count = meta.get('citation_count')
                    enriched_count += 1
            
            if enriched_count > 0:
                print(f"[Query] Enriched {enriched_count}/{len(retrieval_results)} results with study metadata")
        
    except Exception as e:
        # Don't log expected errors like missing tables
        if "does not exist" not in str(e):
            print(f"[Query] Warning: Failed to enrich results with metadata: {e}")
    
    return retrieval_results


# ============================================
# MAIN QUERY ENDPOINT
# ============================================

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Query the medical knowledge base using enhanced RAG.
    
    Features:
    - Automatic query expansion (50+ medical abbreviations)
    - Query type classification (8 types)
    - Query-type-specific answer generation
    - Cross-encoder reranking for improved relevance
    - NCCN guideline gap detection
    - Optional site inference
    - User uploaded documents included in search (if logged in)
    
    **Example Request:**
    ```json
    {
        "question": "What is the standard RT dose for breast cancer?",
        "top_k": 10,
        "use_site_inference": false
    }
    ```
    """
    try:
        from src.api.services.enhanced_rag_service import get_enhanced_rag_service
        
        cache_service = get_cache_service()
        cache_key = None
        cached_evidence = None
        
        if current_user:
            # Convert conversation_history to serializable format for cache key
            history_for_cache = [msg.model_dump() for msg in request.conversation_history] if request.conversation_history else []
            cache_key = make_cache_key(
                "rag_retrieval",  # Changed namespace to indicate retrieval-only cache
                {
                    "question": request.question,
                    "query_mode": request.query_mode,
                    "top_k": request.top_k,
                    "category": request.category,
                    "use_site_inference": request.use_site_inference,
                    # Note: conversation_history affects retrieval context
                },
            )
            cached = await cache_service.get(current_user["id"], cache_key)
            if cached and cached.get("evidence"):
                cached_evidence = cached.get("evidence")
                print(f"[Query] Using cached retrieval ({len(cached_evidence)} chunks)")

        # Get RAG service
        rag_service = get_enhanced_rag_service()
        
        # Convert conversation_context to list of dicts if provided
        conversation_context_dicts = None
        if request.conversation_context:
            conversation_context_dicts = [entry.model_dump() for entry in request.conversation_context]
        
        # Execute query against Qdrant (use cached evidence if available)
        #
        # Auto-enable site inference when the query contains patient
        # context and no explicit category was provided. Without this,
        # the default use_site_inference=False means patient case queries
        # get NO cancer-type filter and retrieve studies from the general
        # radiotherapy&oncology collection (anal cancer, glioma, melanoma
        # etc. instead of H&N / breast / prostate-specific studies).
        effective_site_inference = request.use_site_inference
        if not request.category and not request.use_site_inference:
            # Check if query has enough patient context to warrant auto-inference
            patient_signals = sum(1 for p in [
                r'\b\d{2,3}\s*(?:y\.?o\.?|year)', r'(?:pT|cT|ypT)\d',
                r'(?:stage|Stage)\s+[IV]{1,3}', r'\b(?:s/p|status post)\b',
                r'\b(?:adenocarcinoma|carcinoma|SCC|squamous)\b',
                r'\b(?:cancer|tumor|tumour|malignancy|neoplasm)\b',
                r'\b(?:Gleason|PSA|CPS|PD-L1|HER2|EGFR|BRCA|ER\+|PR\+)\b',
            ] if re.search(p, request.question, re.IGNORECASE))
            if patient_signals >= 2:
                effective_site_inference = True
                print(f"[Query] Auto-enabled site inference (patient_signals={patient_signals})")

        result = await rag_service.query(
            question=request.question,
            query_mode=request.query_mode,
            top_k=request.top_k,
            category=request.category,
            use_site_inference=effective_site_inference,
            conversation_history=request.conversation_history,
            cached_evidence=cached_evidence,
            user_id=current_user["id"] if current_user else None,
            accumulated_context=request.accumulated_context,
            conversation_context=conversation_context_dicts,
        )
        
        # Cache the retrieved evidence for future queries (not the answer)
        if current_user and cache_key and not cached_evidence:
            evidence_to_cache = result.get("evidence", [])
            if evidence_to_cache:
                try:
                    await cache_service.set(
                        current_user["id"],
                        cache_key,
                        {"evidence": evidence_to_cache, "metadata": result.get("metadata", {})},
                    )
                    print(f"[Query] Cached {len(evidence_to_cache)} evidence chunks")
                except Exception as cache_err:
                    print(f"[Query] Cache set failed (non-fatal): {cache_err}")
        
        # If user is logged in, also search their uploaded documents
        # Check user preference for including uploads
        user_doc_results = []
        include_user_uploads = True  # Default to true
        
        print(f"[Query] current_user: {current_user}")  # DEBUG
        if current_user:
            # Check user preference for including uploads
            try:
                from src.api.routes.user_preferences import get_preferences
                user_prefs = await get_preferences(current_user)
                include_user_uploads = user_prefs.include_user_uploads
                print(f"[Query] include_user_uploads preference: {include_user_uploads}")
            except Exception as e:
                print(f"[Query] Failed to get user preferences: {e}")
                include_user_uploads = True  # Default to true on error
            
            if include_user_uploads:
                print(f"[Query] User logged in: {current_user.get('id')}, searching user uploads...")  # DEBUG
                try:
                    from src.api.services.user_document_search import get_user_document_search_service
                    
                    user_search = get_user_document_search_service()
                    
                    # Get query embedding from the retriever
                    query_embedding = rag_service.retriever.embed_query(request.question)
                    
                    # Search user documents with hybrid search
                    user_doc_results = await user_search.search_user_documents(
                        user_id=current_user["id"],
                        query_embedding=query_embedding,
                        query_text=request.question,  # For keyword/BM25 search
                        top_k=request.top_k,
                        alpha=0.7  # 70% semantic, 30% keyword
                    )
                    
                    if user_doc_results:
                        print(f"[Query] Found {len(user_doc_results)} results from user uploads")
                except Exception as e:
                    print(f"[Query] User document search failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[Query] Skipping user uploads (disabled in preferences)")
        
        # Convert evidence to retrieval results
        retrieval_results = []
        for e in result.get("evidence", []):
            table_info = None
            if e.get("table"):
                table_info = TableInfo(**e["table"])
            
            retrieval_results.append(RetrievalResult(
                doc_id=e.get("doc_id"),
                title=e.get("title"),
                author=_get_author_with_fallback(e),
                citation=e.get("citation"),
                doi=e.get("doi"),
                pmid=e.get("pmid"),
                year=e.get("year"),
                category=e.get("category"),
                section=e.get("section"),
                chunk_type=e.get("chunk_type"),
                content=e.get("text", ""),
                score=e.get("score"),
                relevance_score=_normalize_crossencoder_score(e.get("score_crossencoder")),
                # Per-source patient_match badge data. Stamped on each
                # evidence chunk by the post-retrieval scorer in
                # EnhancedRAGService.query() — see the [PatientMatch]
                # block. Stays None for non-patient queries where the
                # scorer didn't run; the UI hides the Match badge then.
                patient_match_score=e.get("patient_match_score"),
                patient_match_breakdown=e.get("patient_match_breakdown"),
                evidence_type=e.get("evidence_type"),
                table=table_info
            ))

        # Add user document results (merge by score)
        # Note: user_document_search returns 'similarity_score', not 'score'
        for u in user_doc_results:
            retrieval_results.append(RetrievalResult(
                doc_id=u.get("doc_id"),
                title=u.get("title"),
                author=None,
                citation=f"User Upload: {u.get('filename', '')}",
                doi=None,
                pmid=None,
                year=None,
                category="user_upload",
                section=u.get("section"),
                chunk_type=u.get("chunk_type"),
                content=u.get("text", ""),
                score=u.get("similarity_score")  # User doc search uses similarity_score
            ))
        
        # Re-sort by score after merging
        retrieval_results.sort(key=lambda x: x.score or 0, reverse=True)

        # Limit to top_k
        retrieval_results = retrieval_results[:request.top_k]

        # Web supplement / fallback — relevance-based scoring
        kb_evidence_raw = result.get("evidence", [])
        web_mode_q, web_triggered_q = _score_web_need(
            query=request.question,
            query_type=result.get("query_type", "general"),
            kb_results=kb_evidence_raw,
        )
        web_fallback_q = False
        web_supplement_q = False
        if web_triggered_q:
            web_items_q = _fetch_web_evidence(request.question, result.get("query_type", "general"), web_mode_q)
            web_fallback_q = web_mode_q == "fallback"
            web_supplement_q = web_mode_q == "supplement"
            for e in web_items_q:
                retrieval_results.append(RetrievalResult(
                    doc_id=e.get("doc_id"),
                    title=e.get("title"),
                    author=None,
                    citation=e.get("citation"),
                    doi=e.get("doi"),
                    pmid=e.get("pmid"),
                    year=e.get("year"),
                    category=None,
                    section=None,
                    chunk_type=None,
                    content=e.get("text", ""),
                    score=e.get("score"),
                    relevance_score=None,
                    source_type=e.get("source_type"),
                ))

        # Enrich with study metadata (patient count, citation count) for sorting
        retrieval_results = await enrich_retrieval_results_with_metadata(retrieval_results)

        # Build metadata
        metadata_dict = result.get("metadata", {})
        metadata = QueryMetadata(**metadata_dict)
        
        # Build module classification if available
        module_classification = None
        if metadata_dict.get("module_classification"):
            from src.api.models.query_models import ModuleClassification
            module_classification = ModuleClassification(**metadata_dict["module_classification"])
        
        # Build structured response if available
        structured_response = None
        if result.get("structured_response"):
            from src.api.models.query_models import StructuredResponseData
            structured_response = StructuredResponseData(**result["structured_response"])
        
        # Build updated context entry for automatic conversation mode
        updated_context_entry = None
        if result.get("updated_context_entry"):
            from src.api.models.query_models import ConversationContextEntry
            updated_context_entry = ConversationContextEntry(**result["updated_context_entry"])
        
        response = QueryResponse(
            answer=result["answer"],
            retrieval_results=retrieval_results,
            query_type=result.get("query_type"),
            module_classification=module_classification,
            metadata=metadata,
            sources=result.get("sources"),
            source_citations=result.get("source_citations"),
            accumulated_context=result.get("accumulated_context"),
            structured_response=structured_response,
            updated_context_entry=updated_context_entry,
            answer_quality=result.get("answer_quality"),
            web_fallback=web_fallback_q,
            web_supplement=web_supplement_q,
        )

        return response

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Enhanced RAG service not available: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_detail = f"Error querying knowledge base: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# ============================================
# ENHANCED QUERY WITH PTO ROUTING
# ============================================

@router.post("/query/enhanced", response_model=EnhancedQueryResponse)
async def enhanced_query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Enhanced query that generates a direct 1-sentence answer plus detailed justification.
    
    This endpoint:
    1. Retrieves evidence using the RAG pipeline
    2. Searches user uploaded documents (if logged in)
    3. Generates a concise 1-sentence direct answer using LLM
    4. Generates detailed justification using the combined evidence
    5. Returns both for display (short answer visible, justification in dropdown)
    
    **Example Request:**
    ```json
    {
        "question": "What outcome is associated with completion axillary LND for breast cancer with sentinel node micrometastasis?",
        "top_k": 10
    }
    ```
    
    **Response Structure:**
    - short_answer: Concise 1-sentence direct answer (e.g., "No significant difference in OS or DFS")
    - justification: Detailed explanation with citations
    - retrieval_results: Supporting evidence
    """
    try:
        from src.api.services.enhanced_rag_service import get_enhanced_rag_service, gpt4o_summary_enhanced
        from src.core.config import settings
        from openai import OpenAI
        
        print(f"[Enhanced Query] Starting - current_user: {current_user}")  # DEBUG
        
        cache_service = get_cache_service()
        cache_key = None
        cached_evidence = None
        
        if current_user:
            # Convert conversation_history to serializable format for cache key
            history_for_cache = [msg.model_dump() for msg in request.conversation_history] if request.conversation_history else []
            cache_key = make_cache_key(
                "rag_retrieval_enhanced",  # Changed namespace to indicate retrieval-only cache
                {
                    "question": request.question,
                    "query_mode": request.query_mode,
                    "top_k": request.top_k,
                    "category": request.category,
                    "use_site_inference": request.use_site_inference,
                    "use_study_focused": request.use_study_focused,
                    "max_studies": request.max_studies if request.use_study_focused else None,
                    "chunks_per_study": request.chunks_per_study if request.use_study_focused else None,
                    # Note: conversation_history affects retrieval context
                },
            )
            cached = await cache_service.get(current_user["id"], cache_key)
            if cached and cached.get("evidence"):
                cached_evidence = cached.get("evidence")
                print(f"[Enhanced Query] Using cached retrieval ({len(cached_evidence)} chunks)")

        # Initialize OpenAI client for short answer generation
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Get RAG service for justification
        rag_service = get_enhanced_rag_service()
        
        # Execute RAG query with more chunks for better accuracy
        # Use at least 10 chunks for short answer generation
        effective_top_k = max(request.top_k, 10)
        
        # First, search user uploads if logged in (BEFORE generating answer)
        # Check user preference for including uploads
        user_doc_results = []
        user_evidence = []
        include_user_uploads = True  # Default to true
        
        print(f"[Enhanced Query] current_user: {current_user}")  # DEBUG
        if current_user:
            # Check user preference for including uploads
            try:
                from src.api.routes.user_preferences import get_preferences
                user_prefs = await get_preferences(current_user)
                include_user_uploads = user_prefs.include_user_uploads
                print(f"[Enhanced Query] include_user_uploads preference: {include_user_uploads}")
            except Exception as e:
                print(f"[Enhanced Query] Failed to get user preferences: {e}")
                include_user_uploads = True  # Default to true on error
            
            if include_user_uploads:
                print(f"[Enhanced Query] User logged in: {current_user.get('id')}, searching user uploads...")
                try:
                    from src.api.services.user_document_search import get_user_document_search_service
                    
                    user_search = get_user_document_search_service()
                    
                    # Get query embedding from the retriever
                    query_embedding = rag_service.retriever.embed_query(request.question)
                    
                    # Search user documents with hybrid search + cross-encoder reranking
                    user_doc_results = await user_search.search_user_documents(
                        user_id=current_user["id"],
                        query_embedding=query_embedding,
                        query_text=request.question,
                        top_k=request.top_k,
                        alpha=0.7
                    )
                    
                    if user_doc_results:
                        print(f"[Enhanced Query] Found {len(user_doc_results)} results from user uploads")
                        
                        # Convert to evidence format for answer generation
                        for u in user_doc_results:
                            user_evidence.append({
                                "doc_id": u.get("doc_id"),
                                "title": u.get("title"),
                                "text": u.get("text", ""),
                                "citation": f"User Upload: {u.get('filename', '')}",
                                "score": u.get("similarity_score"),  # Now cross-encoder score
                                "section": u.get("section"),
                                "chunk_type": u.get("chunk_type"),
                            })
                except Exception as e:
                    print(f"[Enhanced Query] User document search failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[Enhanced Query] Skipping user uploads (disabled in preferences)")
        
        # Convert conversation_context to list of dicts if provided
        conversation_context_dicts = None
        if request.conversation_context:
            conversation_context_dicts = [entry.model_dump() for entry in request.conversation_context]
        
        # Auto-enable site inference for patient case queries (same logic
        # as the standard /rag/query endpoint)
        effective_site_inference = request.use_site_inference
        if not request.category and not request.use_site_inference:
            patient_signals = sum(1 for p in [
                r'\b\d{2,3}\s*(?:y\.?o\.?|year)', r'(?:pT|cT|ypT)\d',
                r'(?:stage|Stage)\s+[IV]{1,3}', r'\b(?:s/p|status post)\b',
                r'\b(?:adenocarcinoma|carcinoma|SCC|squamous)\b',
                r'\b(?:cancer|tumor|tumour|malignancy|neoplasm)\b',
                r'\b(?:Gleason|PSA|CPS|PD-L1|HER2|EGFR|BRCA|ER\+|PR\+)\b',
            ] if re.search(p, request.question, re.IGNORECASE))
            if patient_signals >= 2:
                effective_site_inference = True
                print(f"[Enhanced Query] Auto-enabled site inference (patient_signals={patient_signals})")

        # Shadow-route via pick_surface — observability-only for now.
        # Logs which surface (P1/P3/P4/P5) the router would pick for this
        # request so we can compare against the pipeline that actually ran
        # before cutting traffic over in a follow-up. Never raises: any
        # failure here is non-fatal.
        planned_surface = None
        try:
            from src.api.services.unified_router import pick_surface, Surface
            axes_hint: Dict[str, Any] = {}
            try:
                from src.api.services.query_structuring_service import (
                    structure_query_fast,
                )
                qs = structure_query_fast(request.question, "treatment_recommendation")
                axes_hint = {
                    "has_patient_context": bool(getattr(qs, "has_patient_context", False)),
                    "prior_treatments": list(
                        getattr(getattr(qs, "treatment", None), "prior_treatments", []) or []
                    ),
                    "trajectory_flags": [],  # filled in downstream by inference
                }
            except Exception:
                pass
            planned_surface = pick_surface(
                axes_hint,
                query_mode=request.query_mode,
                force_trial_match=False,
            ).value
            print(f"[Router] pick_surface → {planned_surface}")
        except Exception as e:
            print(f"[Router] pick_surface failed (non-fatal): {e}")

        # Phase 4: `use_study_focused` is retired. The flag still works (it is
        # silently re-routed through the same ComprehensiveRetriever backbone),
        # but we log one deprecation line per request so callers get a signal.
        if request.use_study_focused:
            print(
                "[Deprecation] /query/enhanced received use_study_focused=True; "
                "this flag is retired. Request is being re-routed through the "
                "shared retrieval backbone (mode='comprehensive')."
            )

        # Execute RAG query for Qdrant evidence (use cached evidence if available)
        rag_result = await rag_service.query(
            question=request.question,
            query_mode=request.query_mode,
            top_k=effective_top_k,
            category=request.category,
            use_site_inference=effective_site_inference,
            conversation_history=request.conversation_history,
            user_id=current_user["id"] if current_user else None,
            cached_evidence=cached_evidence,
            conversation_context=conversation_context_dicts,
            use_study_focused=request.use_study_focused,
            max_studies=request.max_studies,
            chunks_per_study=request.chunks_per_study,
        )
        if isinstance(rag_result, dict) and planned_surface:
            rag_result.setdefault("metadata", {})["planned_surface"] = planned_surface
        
        # Get Qdrant evidence
        qdrant_evidence = rag_result.get("evidence", [])
        
        # Cache the retrieved evidence for future queries (not the answer)
        if current_user and cache_key and not cached_evidence:
            if qdrant_evidence:
                try:
                    await cache_service.set(
                        current_user["id"],
                        cache_key,
                        {"evidence": qdrant_evidence, "metadata": rag_result.get("metadata", {})},
                    )
                    print(f"[Enhanced Query] Cached {len(qdrant_evidence)} evidence chunks")
                except Exception as cache_err:
                    print(f"[Enhanced Query] Cache set failed (non-fatal): {cache_err}")
        
        # Merge user evidence with Qdrant evidence
        # Both now use cross-encoder scores on the same scale
        combined_evidence = list(qdrant_evidence)
        combined_evidence.extend(user_evidence)
        
        # Sort by score and limit
        combined_evidence.sort(key=lambda x: x.get("score", 0), reverse=True)
        combined_evidence = combined_evidence[:effective_top_k]
        
        # Web supplement / fallback — relevance-based scoring
        web_mode, web_triggered = _score_web_need(
            query=request.question,
            query_type=rag_result.get("query_type", "general"),
            kb_results=qdrant_evidence,
        )
        web_fallback_triggered = False
        web_supplement_triggered = False
        if web_triggered:
            web_evidence = _fetch_web_evidence(request.question, rag_result.get("query_type", "general"), web_mode)
            web_fallback_triggered = web_mode == "fallback"
            web_supplement_triggered = web_mode == "supplement"
            if web_evidence:
                combined_evidence.extend(web_evidence)
                combined_evidence.sort(key=lambda x: x.get("score", 0), reverse=True)
                combined_evidence = combined_evidence[:effective_top_k]

        # If user uploads were found, regenerate the justification with combined evidence
        justification = rag_result.get("answer", "")
        if user_evidence:
            print(f"[Enhanced Query] Regenerating answer with {len(user_evidence)} user uploads included")
            try:
                justification = gpt4o_summary_enhanced(
                    openai_client=openai_client,
                    question=request.question,
                    evidence=combined_evidence,
                    query_type=rag_result.get("query_type", "general"),
                    nccn_assessment=rag_result.get("metadata", {}).get("nccn_assessment"),
                    staging_context=rag_result.get("metadata", {}).get("staging_info", {}).get("staging_context"),
                )
            except Exception as e:
                print(f"[Enhanced Query] Failed to regenerate answer: {e}")
                # Fall back to original answer
        
        # Convert combined evidence to retrieval results
        retrieval_results = []
        for e in combined_evidence:
            table_info = None
            if e.get("table"):
                table_info = TableInfo(**e["table"])
            
            # Determine if this is a user upload
            is_user_upload = "User Upload:" in (e.get("citation") or "")
            
            retrieval_results.append(RetrievalResult(
                doc_id=e.get("doc_id"),
                title=e.get("title"),
                author=_get_author_with_fallback(e) if not is_user_upload else None,
                citation=e.get("citation"),
                doi=e.get("doi") if not is_user_upload else None,
                pmid=e.get("pmid") if not is_user_upload else None,
                year=e.get("year") if not is_user_upload else None,
                category="user_upload" if is_user_upload else e.get("category"),
                section=e.get("section"),
                chunk_type=e.get("chunk_type"),
                content=e.get("text", ""),
                score=e.get("score") or e.get("similarity_score"),
                relevance_score=_normalize_crossencoder_score(e.get("score_crossencoder")),
                source_type=e.get("source_type", "user_upload" if is_user_upload else "kb"),
                # Per-source patient_match badge data, stamped on each
                # chunk by the post-retrieval scorer in
                # EnhancedRAGService.query(). Stays None for
                # non-patient queries and user-upload chunks.
                patient_match_score=e.get("patient_match_score") if not is_user_upload else None,
                patient_match_breakdown=e.get("patient_match_breakdown") if not is_user_upload else None,
                evidence_type=e.get("evidence_type") if not is_user_upload else None,
                table=table_info
            ))

        # Enrich with study metadata (patient count, citation count) for sorting
        retrieval_results = await enrich_retrieval_results_with_metadata(retrieval_results)
        
        # Generate concise 1-sentence short answer using LLM
        short_answer = _generate_short_answer_with_llm(
            question=request.question,
            evidence=combined_evidence,
            openai_client=openai_client,
            justification=justification  # Pass regenerated justification
        )
        
        # Fallback if LLM generation fails
        if not short_answer:
            first_period = justification.find('. ')
            if first_period > 20 and first_period < len(justification) - 10:
                short_answer = justification[:first_period + 1]
            else:
                short_answer = justification[:150] + "..." if len(justification) > 150 else justification
        
        # Generate follow-up suggestions if in conversation mode
        suggested_followups = []
        print(f"[Enhanced Query] query_mode={request.query_mode}, history_len={len(request.conversation_history)}")
        if request.query_mode in {"conversation", "chat"}:
            try:
                from src.api.services.enhanced_rag_service import generate_followup_suggestions, _extract_clinical_context
                
                # Extract clinical context from conversation history + current question
                user_messages = []
                for msg in request.conversation_history:
                    if msg.role == "user":
                        user_messages.append(msg.content)
                user_messages.append(request.question)  # Include current question
                
                clinical_context = _extract_clinical_context(user_messages)
                print(f"[Enhanced Query] Generating followups with context: {clinical_context}")
                
                suggested_followups = generate_followup_suggestions(
                    openai_client=openai_client,
                    question=request.question,
                    answer=short_answer,
                    clinical_context=clinical_context,
                    query_type=rag_result.get("query_type", "general")
                )
                print(f"[Enhanced Query] Generated followups: {suggested_followups}")
            except Exception as e:
                print(f"[Enhanced Query] Failed to generate followup suggestions: {e}")
                import traceback
                traceback.print_exc()
        
        # Prepend staging clarification questions if stage is ambiguous
        # DISABLED: Staging clarification temporarily disabled
        # try:
        #     rag_metadata = rag_result.get("metadata", {})
        #     if rag_metadata.get("stage_ambiguous"):
        #         from src.api.services.staging_clarification import (
        #             generate_staging_clarifications,
        #             clarifications_to_followup_strings,
        #         )
        #
        #         clarification = generate_staging_clarifications(
        #             required_factors=rag_metadata.get("stage_required_factors", []),
        #             possible_stages=rag_metadata.get("stage_possible_stages", []),
        #             inference_notes=rag_metadata.get("stage_inference_notes", []),
        #         )
        #
        #         if clarification.needs_clarification:
        #             staging_suggestions = clarifications_to_followup_strings(clarification)
        #             suggested_followups = staging_suggestions + suggested_followups
        #             print(f"[Enhanced Query] Prepended {len(staging_suggestions)} staging clarification followups")
        # except Exception as e:
        #     print(f"[Enhanced Query] Staging clarification followups failed: {e}")
        
        # Generate chart artifact if visualization is requested
        # Note: These functions are planned but not yet implemented
        artifact = None
        try:
            from src.api.services.enhanced_rag_service import detect_visualization_intent, generate_chart_artifact
            
            if detect_visualization_intent(request.question):
                print(f"[Enhanced Query] Visualization intent detected, generating chart...")
                artifact = generate_chart_artifact(
                    openai_client=openai_client,
                    question=request.question,
                    evidence=combined_evidence,
                    answer=justification
                )
                if artifact:
                    print(f"[Enhanced Query] Chart artifact generated: {artifact.get('chart', {}).get('title', 'Unknown')}")
        except ImportError:
            # Visualization functions not yet implemented - skip silently
            pass
        except Exception as e:
            print(f"[Enhanced Query] Failed to generate chart artifact: {e}")
            import traceback
            traceback.print_exc()
        
        # Build metadata
        metadata_dict = rag_result.get("metadata", {})
        # Populate query_confidence from classification result if not already present
        if "query_confidence" not in metadata_dict:
            classification = metadata_dict.get("query_classification", {})
            if isinstance(classification, dict) and "confidence" in classification:
                metadata_dict = {**metadata_dict, "query_confidence": classification["confidence"]}
        metadata = QueryMetadata(**metadata_dict)
        
        # Build module classification if available
        module_classification = None
        if metadata_dict.get("module_classification"):
            from src.api.models.query_models import ModuleClassification
            module_classification = ModuleClassification(**metadata_dict["module_classification"])
        
        # Build structured response if available
        structured_response = None
        if rag_result.get("structured_response"):
            from src.api.models.query_models import StructuredResponseData
            structured_response = StructuredResponseData(**rag_result["structured_response"])
        
        # Build updated context entry for automatic conversation mode
        updated_context_entry = None
        if rag_result.get("updated_context_entry"):
            from src.api.models.query_models import ConversationContextEntry
            updated_context_entry = ConversationContextEntry(**rag_result["updated_context_entry"])
        
        # Extract routing info from metadata
        routing_info = None
        if metadata_dict.get("routing"):
            routing_info = metadata_dict["routing"]
        
        # Generate dynamic follow-up suggestions using Follow_Up_Generator
        follow_up_suggestions = []
        try:
            from src.api.services.follow_up_generator import get_follow_up_generator
            
            # Get doc titles from retrieval results
            doc_titles_for_followups = [r.title for r in retrieval_results if r.title]
            
            generator = get_follow_up_generator()
            follow_up_suggestions = await generator.generate_follow_ups(
                query=request.question,
                response=justification,
                module=routing_info.get("module", "general_knowledge") if routing_info else "general_knowledge",
                doc_titles=doc_titles_for_followups,
                max_suggestions=4
            )
            print(f"[Enhanced Query] Generated {len(follow_up_suggestions)} dynamic follow-ups")
        except Exception as e:
            print(f"[Enhanced Query] Follow-up generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate structured summary card (Task 2). Pass the top cross-encoder
        # relevance score so evidence_level / recommendation_strength can't
        # exceed what the actual retrieved chunks support — prevents the
        # "High Evidence Strong" + "Limited relevance" contradiction.
        top_relevance = 0.0
        for r in retrieval_results or []:
            v = getattr(r, "relevance_score", None) or 0.0
            if isinstance(v, (int, float)) and v > top_relevance:
                top_relevance = float(v)
        structured_summary = None
        try:
            structured_summary = _generate_structured_summary(
                justification,
                rag_result.get("query_type", "general"),
                openai_client,
                top_relevance=top_relevance,
            )
        except Exception as e:
            print(f"[Enhanced Query] Structured summary failed (non-fatal): {e}")

        # Generate guideline alignment (Task 5) — only for treatment/indication query types
        guideline_alignment = None
        _guide_types = {"treatment_recommendation", "indication_question"}
        if rag_result.get("query_type") in _guide_types:
            try:
                guideline_alignment = _generate_guideline_alignment(justification, openai_client)
            except Exception as e:
                print(f"[Enhanced Query] Guideline alignment failed (non-fatal): {e}")

        # Extract doc_ids and doc_titles for response
        doc_ids = [r.doc_id for r in retrieval_results if r.doc_id]
        doc_titles = [r.title for r in retrieval_results if r.title]

        response = EnhancedQueryResponse(
            short_answer=short_answer,
            justification=justification,
            pto_frames=[],  # Not using PTO frames in this approach
            retrieval_results=retrieval_results,
            used_pto=False,
            query_type=rag_result.get("query_type"),
            suggested_followups=suggested_followups,
            module_classification=module_classification,
            artifact=artifact,
            metadata=metadata,
            sources=rag_result.get("sources"),
            source_citations=rag_result.get("source_citations"),
            structured_response=structured_response,
            updated_context_entry=updated_context_entry,
            routing=routing_info,
            follow_up_suggestions=follow_up_suggestions,
            doc_ids=doc_ids,
            doc_titles=doc_titles,
            structured_summary=structured_summary,
            guideline_alignment=guideline_alignment,
            web_fallback=web_fallback_triggered,
            web_supplement=web_supplement_triggered,
        )

        return response

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not available: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_detail = f"Error in enhanced query: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


def _generate_short_answer_from_pto(pto_frames: list, query: str) -> str:
    """Generate a concise short answer from PTO frames."""
    if not pto_frames:
        return ""
    
    # Get the best matching frame
    best_frame = pto_frames[0]
    
    # Build short answer from outcomes
    outcomes = best_frame.get("outcomes", {})
    treatments = best_frame.get("treatment_modalities", [])
    
    if outcomes:
        # Format outcomes into a readable string
        outcome_parts = []
        for metric, value in outcomes.items():
            # Clean up metric name
            metric_clean = metric.replace("_", " ").title()
            outcome_parts.append(f"{metric_clean}: {value}")
        
        if treatments:
            treatment_str = ", ".join(treatments)
            return f"{treatment_str} - {'; '.join(outcome_parts)}"
        else:
            return "; ".join(outcome_parts)
    
    # If no outcomes, describe the treatment
    if treatments:
        return f"Recommended: {', '.join(treatments)}"
    
    # Fallback to frame text if available
    frame_text = best_frame.get("frame_text", "")
    if frame_text:
        return frame_text[:200] + "..." if len(frame_text) > 200 else frame_text
    
    return ""


def _generate_short_answer_with_llm(question: str, evidence: list, openai_client, justification: str = "") -> str:
    """
    Generate a concise 1-sentence direct answer using LLM.
    
    Args:
        question: The user's question
        evidence: List of evidence chunks from retrieval
        openai_client: OpenAI client instance
        justification: The full RAG-generated answer (used to ensure consistency)
        
    Returns:
        A single sentence direct answer
    """
    # Build context from top evidence - include more text for better accuracy
    context_parts = []
    for i, e in enumerate(evidence[:7], 1):
        text = e.get("text", e.get("content", ""))[:800]
        citation = e.get("citation", e.get("title", ""))
        context_parts.append(f"[{i}] {text}\nSource: {citation}")
    
    context = "\n\n".join(context_parts)
    
    # If we have a justification, use it to guide the short answer
    # This ensures consistency between short answer and justification
    justification_guidance = ""
    if justification:
        justification_guidance = f"""
IMPORTANT - The detailed analysis concluded:
{justification[:1000]}

Your short answer MUST be consistent with this analysis. Extract the KEY RECOMMENDATION from above.
"""
    
    system_prompt = """You are a radiation oncology expert providing DIRECT, EVIDENCE-BASED answers.

CRITICAL RULES:
1. Answer in ONE SENTENCE ONLY (maximum 30 words)
2. Be DIRECT and SPECIFIC - state the exact answer from the evidence
3. Extract the SPECIFIC treatment, dose, outcome, or recommendation from the evidence
4. NEVER hedge with phrases like "may", "could", "might", "it depends"
5. NEVER say "based on the context", "according to", "the evidence suggests"
6. NEVER include citations or references in the answer
7. If asking about outcomes, state the SPECIFIC outcome with numbers if available (e.g., "No difference in 10-year OS (83.3% vs 84.4%)")
8. If asking about treatment, state the SPECIFIC treatment recommendation
9. If asking about dose, state the SPECIFIC dose/fractionation
10. If multiple options exist, state the PRIMARY/RECOMMENDED option
11. For recurrence score questions: midrange (11-25) = endocrine therapy alone is non-inferior to chemoendocrine therapy
12. For RT technique questions: state the specific technique AND dose/fractionation
13. For "which feature" or "risk factor" questions: identify the SPECIFIC factor mentioned in the trial/study, not general knowledge

ANSWER FORMAT EXAMPLES:
- Treatment question: "Endocrine therapy alone for patients with midrange recurrence scores (11-25)."
- Outcome question: "No significant difference in invasive disease-free survival between endocrine therapy alone and chemoendocrine therapy."
- Dose question: "40 Gy in 15 fractions over 3 weeks."
- RT technique question: "Para-aortic strip irradiation with 30 Gy in 15 fractions."
- Risk factor question: "Clinical detection (vs mammographic detection) is associated with elevated recurrence risk."
- Comparison question: "SLNB alone is non-inferior to completion ALND with no difference in OS or DFS."
"""

    user_prompt = f"""QUESTION: {question}

EVIDENCE FROM CLINICAL TRIALS:
{context}
{justification_guidance}
Extract the SPECIFIC answer from the evidence above. Provide ONE SENTENCE (max 30 words). 
State the answer directly - no hedging, no citations, no "based on" phrases.
Include the specific technique/treatment AND dose if the question asks about RT or treatment."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Use 0 for most deterministic/accurate response
            max_tokens=100
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up the answer - remove any "Answer:" prefix
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        
        # Remove any trailing citation patterns like "(Author, Year)" or "[1]"
        import re
        answer = re.sub(r'\s*\([^)]*\d{4}[^)]*\)\s*\.?$', '.', answer)
        answer = re.sub(r'\s*\[\d+\]\s*\.?$', '.', answer)
        
        # Ensure it ends with a period
        if answer and not answer.endswith('.'):
            answer += '.'
        
        return answer
        
    except Exception as e:
        print(f"Error generating short answer: {e}")
        return ""


async def _get_consensus_answer_from_pto(
    evidence: list,
    question: str,
    qdrant_client,
    collection_name: str
) -> tuple:
    """
    Get consensus answer by finding PTO frames for each doc_id in evidence
    and returning the most frequent outcome/treatment.
    
    Args:
        evidence: List of evidence chunks from RAG retrieval
        question: The user's question
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        
    Returns:
        Tuple of (consensus_answer, confidence, supporting_frames)
    """
    from collections import Counter
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    
    # Extract unique doc_ids from evidence
    doc_ids = list(set(
        e.get("doc_id") for e in evidence 
        if e.get("doc_id")
    ))
    
    if not doc_ids:
        return None, 0.0, []
    
    # Find PTO frames for these doc_ids
    pto_frames = []
    try:
        # Search for PTO frames matching these doc_ids
        results, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="pto_frame")),
                    FieldCondition(key="doc_id", match=MatchAny(any=doc_ids[:20]))
                ]
            ),
            limit=50,
            with_payload=True
        )
        
        for r in results:
            pto_frames.append(r.payload)
            
    except Exception as e:
        print(f"Error fetching PTO frames: {e}")
        # Fallback: search without filter
        try:
            results, _ = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True
            )
            for r in results:
                payload = r.payload
                if payload.get("node_type") == "pto_frame" and payload.get("doc_id") in doc_ids:
                    pto_frames.append(payload)
        except Exception as e2:
            print(f"Fallback PTO search also failed: {e2}")
            return None, 0.0, []
    
    if not pto_frames:
        return None, 0.0, []
    
    # Determine what type of answer we're looking for based on the question
    question_lower = question.lower()
    
    # Extract answers based on question type
    answers = []
    
    if any(word in question_lower for word in ["outcome", "survival", "os", "dfs", "pfs", "control", "recurrence"]):
        # Looking for outcomes
        for frame in pto_frames:
            outcomes = frame.get("outcomes", {})
            if outcomes:
                # Format outcomes as a string
                outcome_str = "; ".join(f"{k}: {v}" for k, v in outcomes.items())
                if outcome_str:
                    answers.append(outcome_str)
    
    elif any(word in question_lower for word in ["treatment", "therapy", "regimen", "recommend", "best", "should"]):
        # Looking for treatments
        for frame in pto_frames:
            treatments = frame.get("treatment_modalities", [])
            dose = frame.get("dose_fractionation", "")
            chemo = frame.get("chemo_agents", [])
            
            treatment_parts = []
            if treatments:
                treatment_parts.append(", ".join(treatments))
            if dose:
                treatment_parts.append(dose)
            if chemo:
                treatment_parts.append(", ".join(chemo))
            
            if treatment_parts:
                answers.append(" + ".join(treatment_parts))
    
    elif any(word in question_lower for word in ["dose", "gy", "fractionation", "fraction"]):
        # Looking for dose information
        for frame in pto_frames:
            dose = frame.get("dose_fractionation", "")
            if dose:
                answers.append(dose)
    
    else:
        # General - try to extract any relevant info
        for frame in pto_frames:
            frame_text = frame.get("frame_text", "")
            if frame_text:
                # Take first sentence
                first_sentence = frame_text.split('.')[0] + '.'
                if len(first_sentence) > 10:
                    answers.append(first_sentence)
    
    if not answers:
        return None, 0.0, pto_frames
    
    # Count frequency of each answer
    answer_counts = Counter(answers)
    
    # Get the most common answer
    most_common = answer_counts.most_common(1)[0]
    consensus_answer = most_common[0]
    frequency = most_common[1]
    total = len(answers)
    confidence = frequency / total if total > 0 else 0.0
    
    return consensus_answer, confidence, pto_frames


# ============================================
# STUDY-SPECIFIC Q&A ENDPOINT
# ============================================

from src.api.models.query_models import StudyQueryRequest, StudyQueryResponse

@router.post("/query/study", response_model=StudyQueryResponse)
async def query_study(
    request: StudyQueryRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Answer questions about a specific study.
    
    This endpoint:
    1. Retrieves study chunks from Qdrant filtered by doc_id, DOI, PMID, or title
    2. Uses the study content as context for answering questions
    3. Maintains conversation history for follow-up questions
    """
    try:
        from src.core.config import settings
        from openai import OpenAI
        from src.api.services.enhanced_rag_service import get_enhanced_rag_service
        from qdrant_client import models as qm
        
        print(f"[Study Query] Starting - study_id={request.study_id}, doi={request.study_doi}, pmid={request.study_pmid}, title={request.study_title[:50] if request.study_title else None}")
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Get RAG service for retrieval
        rag_service = get_enhanced_rag_service()
        retriever = rag_service.retriever
        
        # Determine filter value - include title as fallback
        filter_value = request.study_id or request.study_doi or request.study_pmid or request.study_title
        
        if not filter_value:
            raise HTTPException(status_code=400, detail="No study identifier provided")
        
        print(f"[Study Query] Filter value: '{filter_value[:50] if filter_value else None}'")
        
        # Embed the question
        qvec = retriever.embed_query(request.question)
        
        # Try multiple filter strategies to find the study
        hits = []
        
        # Strategy 1: Exact match on doc_id (includes hash suffix)
        if request.study_id:
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=request.study_id))
            ])
            hits = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            print(f"[Study Query] Strategy 1 (exact doc_id): {len(hits)} hits")
        
        # Strategy 2: Try doc_id_raw field (without hash suffix)
        if not hits and request.study_id:
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id_raw", match=qm.MatchValue(value=request.study_id))
            ])
            hits = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            print(f"[Study Query] Strategy 2 (doc_id_raw): {len(hits)} hits")
        
        # Strategy 3: Try source_doc_dir_name field
        if not hits and request.study_id:
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="source_doc_dir_name", match=qm.MatchValue(value=request.study_id))
            ])
            hits = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            print(f"[Study Query] Strategy 3 (source_doc_dir_name): {len(hits)} hits")
        
        # Strategy 4: Convert DOI to doc_id_raw format and try
        if not hits and request.study_doi:
            # Convert DOI "10.1200/jco.2014.59.5132" -> "doi_10.1200_jco.2014.59.5132"
            doi_as_doc_id = "doi_" + request.study_doi.replace("/", "_")
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id_raw", match=qm.MatchValue(value=doi_as_doc_id))
            ])
            hits = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            print(f"[Study Query] Strategy 4 (doi->doc_id_raw '{doi_as_doc_id}'): {len(hits)} hits")
        
        # Strategy 5: Try source_doc_dir_name with DOI format
        if not hits and request.study_doi:
            doi_as_dir = "doi_" + request.study_doi.replace("/", "_")
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="source_doc_dir_name", match=qm.MatchValue(value=doi_as_dir))
            ])
            hits = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            print(f"[Study Query] Strategy 5 (doi->source_doc_dir_name '{doi_as_dir}'): {len(hits)} hits")
        
        # Strategy 6: Scroll and filter manually (last resort)
        if not hits:
            print(f"[Study Query] Strategy 6: Scroll-based search")
            try:
                # Get points and filter manually
                scroll_results, _ = retriever.qdrant.scroll(
                    collection_name=retriever.collection,
                    limit=200,
                    with_payload=True,
                    with_vectors=False,
                )
                
                # Build search terms
                search_terms = []
                if request.study_id:
                    search_terms.append(request.study_id.lower())
                if request.study_doi:
                    search_terms.append(request.study_doi.lower())
                    # Also try DOI in doc_id format
                    search_terms.append(("doi_" + request.study_doi.replace("/", "_")).lower())
                if request.study_pmid:
                    search_terms.append(request.study_pmid.lower())
                
                print(f"[Study Query] Strategy 6 search terms: {search_terms}")
                
                # Filter by doc_id/doc_id_raw/source_doc_dir_name containing any search term
                matching_points = []
                for point in scroll_results:
                    doc_id = (point.payload.get("doc_id") or "").lower()
                    doc_id_raw = (point.payload.get("doc_id_raw") or "").lower()
                    source_dir = (point.payload.get("source_doc_dir_name") or "").lower()
                    
                    for term in search_terms:
                        if term and (term in doc_id or term == doc_id_raw or term == source_dir):
                            matching_points.append(point)
                            break
                
                if matching_points:
                    # Convert to same format as query_points results
                    class FakeHit:
                        def __init__(self, point):
                            self.id = point.id
                            self.payload = point.payload
                            self.score = 1.0  # No score from scroll
                    
                    hits = [FakeHit(p) for p in matching_points[:50]]
                    print(f"[Study Query] Strategy 6 found {len(hits)} matching points")
            except Exception as e:
                print(f"[Study Query] Strategy 6 failed: {e}")
        
        # Strategy 7: Search by title using text match (fallback when no ID/DOI/PMID match)
        if not hits and request.study_title:
            print(f"[Study Query] Strategy 7: Title-based search")
            try:
                # Use semantic search with the title to find relevant chunks
                title_vec = retriever.embed_query(request.study_title)
                title_hits = retriever.qdrant.query_points(
                    collection_name=retriever.collection,
                    query=title_vec,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                ).points
                
                # Filter to only include chunks where the title matches closely
                title_lower = request.study_title.lower()
                # Extract key words from title (at least 4 chars, not common words)
                stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'study', 'trial', 'phase'}
                title_words = [w for w in title_lower.split() if len(w) >= 4 and w not in stop_words][:5]
                
                matching_hits = []
                for hit in title_hits:
                    payload = hit.payload
                    chunk_title = (payload.get("title") or payload.get("doc_meta", {}).get("title") or "").lower()
                    chunk_text = (payload.get("text") or "").lower()
                    
                    # Check if enough title words appear in the chunk's title or text
                    matches = sum(1 for w in title_words if w in chunk_title or w in chunk_text[:500])
                    if matches >= min(3, len(title_words)):
                        matching_hits.append(hit)
                
                if matching_hits:
                    hits = matching_hits[:50]
                    print(f"[Study Query] Strategy 7 found {len(hits)} matching points by title")
            except Exception as e:
                print(f"[Study Query] Strategy 7 failed: {e}")
        
        print(f"[Study Query] Final result: {len(hits)} chunks for study {filter_value[:50] if filter_value else 'unknown'}")
        
        if not hits:
            raise HTTPException(status_code=404, detail="Study not found in database")
        
        # Extract study info and chunks
        study_title = "Unknown Study"
        study_chunks = []
        
        for hit in hits:
            payload = hit.payload
            # Try multiple places for title
            if not study_title or study_title == "Unknown Study":
                study_title = (
                    payload.get("title") or 
                    payload.get("doc_meta", {}).get("title") or
                    payload.get("doc_meta", {}).get("study_name") or
                    study_title
                )
            
            study_chunks.append({
                "text": payload.get("text", ""),
                "section": payload.get("section", ""),
                "chunk_type": payload.get("chunk_type", ""),
                "score": hit.score
            })
        
        # Format context from chunks (top 15 most relevant)
        context_parts = []
        for i, chunk in enumerate(study_chunks[:15], 1):
            section = chunk.get("section", "")
            text = chunk.get("text", "")
            if text:
                context_parts.append(f"[{i}] {section}:\n{text[:1000]}")
        
        study_context = "\n\n".join(context_parts)
        
        # Format conversation history
        conversation_context = ""
        if request.conversation_history:
            history_lines = []
            for msg in request.conversation_history[-6:]:
                role = "User" if msg.role == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.content}")
            conversation_context = "\n".join(history_lines)
        
        # Generate answer using LLM
        system_prompt = """You are a clinical research assistant helping answer questions about a specific clinical study.
You have access to the study content and should answer questions based ONLY on the information provided.

RULES:
1. Answer based ONLY on the study information provided
2. If the information is not available in the study data, say so clearly
3. Be specific and cite relevant details from the study
4. Keep answers concise but informative
5. If asked about something not covered in the study, acknowledge the limitation
"""

        user_prompt = f"""STUDY: {study_title}

STUDY CONTENT:
{study_context}

{f"CONVERSATION HISTORY:{chr(10)}{conversation_context}{chr(10)}{chr(10)}" if conversation_context else ""}QUESTION: {request.question}

Please answer the question based on the study information provided."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Generate follow-up suggestions
        suggested_followups = _generate_study_followups(
            openai_client=openai_client,
            question=request.question,
            answer=answer,
            study_title=study_title
        )
        
        return StudyQueryResponse(
            answer=answer,
            study_title=study_title,
            suggested_followups=suggested_followups
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error querying study: {str(e)}\n{traceback.format_exc()}"
        print(f"[Study Query] {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_study_followups(
    openai_client,
    question: str,
    answer: str,
    study_title: str
) -> List[str]:
    """Generate follow-up questions for study Q&A."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Generate 2-3 relevant follow-up questions about a clinical study.
Keep questions concise (under 10 words). Format as JSON array.
Focus on: endpoints, patient population, treatment details, outcomes, toxicities, methodology."""},
                {"role": "user", "content": f"""Study: {study_title}
Question asked: {question}
Answer given: {answer[:300]}...

Generate follow-up questions as JSON array:"""}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON
        import json
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        suggestions = json.loads(result)
        if isinstance(suggestions, list):
            return [s.strip() for s in suggestions[:3] if isinstance(s, str)]
        
        return []
    except Exception as e:
        print(f"[Study Q&A] Failed to generate followups: {e}")
        return []


# ============================================
# VISUAL TREATMENT COMPARISON ENDPOINT
# ============================================

@router.post("/compare/visual", response_model=VisualComparisonResponse)
async def visual_treatment_comparison(
    request: VisualComparisonRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Generate a visual treatment comparison with per-arm retrieval.

    Pipeline:
    1. Extract treatment arms from the comparison query (LLM)
    2. Run SEPARATE retrieval for each arm (Qdrant + PostgreSQL in parallel)
    3. Fetch structured profiles for each arm's studies
    4. Generate charts and summary from combined evidence
    5. Return per-arm grouped results
    """
    try:
        import asyncio
        import json
        from src.core.config import settings
        from openai import OpenAI
        from src.api.services.enhanced_rag_service import (
            get_enhanced_rag_service,
            gpt4o_summary_enhanced,
            infer_site_key,
            normalize_category_filter,
            SITE_LABELS,
        )

        print(f"[Visual Comparison] Starting - query: {request.query[:100]}...")

        openai_client = OpenAI(api_key=settings.openai_api_key)
        rag_service = get_enhanced_rag_service()
        user_id = current_user["id"] if current_user else None

        # ── Run LLM extraction ONCE on the full patient narrative ─────────
        # The per-arm queries are short search strings that lose the
        # clinical complexity of the original narrative (comorbidities,
        # surgical history, progression timeline, biomarkers). By running
        # LLM extraction here once, we get a rich query_structure that
        # can be passed as accumulated_context to every per-arm call,
        # ensuring ALL arms share the full patient context.
        from src.api.services.query_structuring_service import (
            structure_query_fast,
            structure_query_with_llm_if_needed,
        )
        # arm carries the full clinical context (site_detail, histology,
        # stage, TNM, biomarkers, disease_descriptor, metastatic_sites).
        # This eliminates the LLM drift we observed in the v4 live run,
        # where the Immunotherapy arm dropped "maxilla"/"oral cavity"
        # from its query and caused OCAT to be false-rejected by the
        # PatientEligibility layer.
        from src.api.services.arm_query_builder import (
            build_patient_summary_for_arm_queries,
        )

        # Use LLM extraction (not just regex) on the full patient narrative
        # so comorbidities, surgical history, progression timeline, and
        # biomarkers are all captured in the query_structure.
        query_structure, used_llm = await structure_query_with_llm_if_needed(
            request.query, query_type="treatment_recommendation"
        )
        if used_llm:
            print(f"[Visual Comparison] LLM extraction ran on full patient narrative")
        patient_summary = build_patient_summary_for_arm_queries(query_structure)
        # Build accumulated_context dict from the rich query_structure so
        # per-arm rag_service.query() calls inherit the full patient context
        # (comorbidities, prior treatments, biomarkers, staging) even though
        # their arm queries are short search strings.
        accumulated_context = query_structure.to_dict() if query_structure else None
        print(
            f"[Visual Comparison] Patient summary for arm queries: "
            f"'{patient_summary}'" if patient_summary else
            "[Visual Comparison] No patient context detected — using raw query"
        )

        # ── HARD primary-cancer filter ────────────────────────────────────
        # Infer the patient's PRIMARY cancer site from the original
        # narrative ONCE and pass it as a strict category filter to every
        # per-arm retrieval. We deliberately use `infer_site_key` on the
        # patient summary (or, when absent, the raw query) instead of on
        # the per-arm query — the per-arm queries inject "Reirradiation",
        # "Immunotherapy", etc., which can drift the inference toward
        # CNS / cutaneous. The primary cancer is fixed by the case, not
        # by the treatment being asked about.
        #
        # Secondary / metastatic sites are intentionally ignored here:
        # a head-and-neck patient with a "right ventricular metastasis"
        # must still be filtered to head_neck, not cardiac/CNS. This is
        # what `infer_site_key` already does (keyword scoring, not first-
        # match), so we just trust it on the primary-cancer narrative.
        primary_site_source = patient_summary or request.query
        try:
            primary_site_key = infer_site_key(primary_site_source)
            # If patient_summary produced the default (no site found),
            # retry with the full raw query which has more context
            if primary_site_key == "Radiotherapy&Oncology" and patient_summary:
                primary_site_key = infer_site_key(request.query)
        except Exception:
            primary_site_key = None
        # The "Radiotherapy&Oncology" default is the general bucket —
        # using it as a filter provides no benefit (matches everything).
        # Treat it as "no site found".
        if primary_site_key == "Radiotherapy&Oncology":
            primary_site_key = None
        primary_category = normalize_category_filter(primary_site_key) if primary_site_key else None
        if primary_category:
            print(
                f"[Visual Comparison] HARD primary-cancer filter: "
                f"site_key='{primary_site_key}' "
                f"({SITE_LABELS.get(primary_site_key, primary_site_key)}) "
                f"→ category='{primary_category}'"
            )
        else:
            print(
                "[Visual Comparison] No primary cancer site could be "
                "inferred from the narrative — running without category filter"
            )

        arms = _extract_treatment_arms(openai_client, request.query)
        # Re-render each arm's query from (label + patient_summary + outcomes)
        # when a patient summary is available. Fallback arms (from the
        # regex path or from LLM failure) still keep their own queries if
        # we have no structured context.
        if patient_summary:
            for arm in arms:
                label = arm.get("label", "").strip()
                if label:
                    arm["query"] = f"{label} {patient_summary} outcomes"

        print(f"[Visual Comparison] Extracted {len(arms)} treatment arms: {[a['label'] for a in arms]}")

        # Step 2: Run per-arm retrieval in parallel
        async def retrieve_for_arm(arm: Dict[str, Any]) -> Dict[str, Any]:
            """Retrieve evidence for a single treatment arm."""
            arm_query = arm["query"]
            arm_label = arm["label"]
            print(
                f"[Visual Comparison] Retrieving for arm: {arm_label} -> "
                f"'{arm_query[:80]}...' (category={primary_category}, strict=True)"
            )

            try:
                arm_result = await rag_service.query(
                    question=arm_query,
                    query_mode="hybrid",
                    top_k=request.top_k,
                    # HARD primary-cancer filter — pinned once for the
                    # whole comparison and never silently bypassed.
                    category=primary_category,
                    # When we have a confirmed primary site, do NOT let
                    # the inference layer override it with the per-arm
                    # query (e.g. "Immunotherapy ..." → cutaneous).
                    use_site_inference=(primary_category is None),
                    strict_category=(primary_category is not None),
                    user_id=user_id,
                    # Pass the rich patient context extracted once from
                    # the full narrative so every arm inherits
                    # comorbidities, prior treatments, biomarkers, etc.
                    accumulated_context=accumulated_context,
                    # Per-arm answers are intermediate evidence summaries
                    # that feed into the comparison analysis — not user-
                    # facing. Use mini to cut cost (~$0.10 → ~$0.01).
                    generation_model="gpt-4o-mini",
                )
                evidence = arm_result.get("evidence", [])
                print(f"[Visual Comparison] Arm '{arm_label}': {len(evidence)} evidence chunks")
                return {
                    "label": arm_label,
                    "query": arm_query,
                    "evidence": evidence,
                    "answer": arm_result.get("answer", ""),
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Visual Comparison] Arm '{arm_label}' retrieval failed: {e}")
                return {
                    "label": arm_label,
                    "query": arm_query,
                    "evidence": [],
                    "answer": "",
                }

        arm_results = await asyncio.gather(*[retrieve_for_arm(arm) for arm in arms])

        # Step 3: Fetch structured profiles per arm from PostgreSQL
        from src.api.services.postgres_study_details_service import PostgresStudyDetailsService
        postgres_service = PostgresStudyDetailsService()

        treatment_arms = []
        all_evidence = []
        all_profiles = []
        all_retrieval_results = []

        for arm_data in arm_results:
            evidence = arm_data["evidence"]
            all_evidence.extend(evidence)

            # Collect unique doc_ids for this arm
            seen_doc_ids = set()
            for e in evidence:
                did = e.get("doc_id")
                if did and did not in seen_doc_ids:
                    seen_doc_ids.add(did)

            # Fetch profiles for this arm's studies
            arm_profiles = []
            for did in list(seen_doc_ids)[:8]:
                try:
                    profile = await postgres_service.get_study_details(doc_id=did)
                    if profile and "error" not in profile:
                        arm_profiles.append(profile)
                except Exception as pe:
                    print(f"[Visual Comparison] Profile fetch failed for {did}: {pe}")

            all_profiles.extend(arm_profiles)
            print(f"[Visual Comparison] Arm '{arm_data['label']}': {len(arm_profiles)} structured profiles")

            # Build RetrievalResult objects for this arm with match tags
            arm_retrieval_results = []
            for e in evidence:
                table_info = None
                if e.get("table"):
                    table_info = TableInfo(**e["table"])

                # Generate match tags showing which patient criteria this
                # study matches. Uses simple keyword matching against the
                # study text + metadata to produce human-readable tags.
                tags = _generate_match_tags(e, request.query, query_structure)

                rr = RetrievalResult(
                    doc_id=e.get("doc_id"),
                    title=e.get("title"),
                    author=_get_author_with_fallback(e),
                    citation=e.get("citation"),
                    doi=e.get("doi"),
                    pmid=e.get("pmid"),
                    year=e.get("year"),
                    category=e.get("category"),
                    section=e.get("section"),
                    chunk_type=e.get("chunk_type"),
                    content=e.get("text", ""),
                    score=e.get("score"),
                    relevance_score=_normalize_crossencoder_score(e.get("score_crossencoder")),
                    table=table_info,
                    match_tags=tags,
                )
                arm_retrieval_results.append(rr)
                all_retrieval_results.append(rr)

            treatment_arms.append(TreatmentArmResult(
                arm_label=arm_data["label"],
                arm_query=arm_data["query"],
                retrieval_results=arm_retrieval_results,
                study_profiles=arm_profiles,
            ))

        # Step 4: Generate combined analysis from all evidence
        # Build a combined evidence context for the LLM
        combined_evidence_text = ""
        for arm_data in arm_results:
            combined_evidence_text += f"\n--- {arm_data['label']} ---\n"
            for e in arm_data["evidence"][:5]:
                combined_evidence_text += f"[{e.get('title', 'Unknown')}]: {e.get('text', '')[:400]}\n\n"

        # Generate detailed analysis using the combined evidence
        detailed_analysis = _generate_arm_comparison_analysis(
            openai_client=openai_client,
            query=request.query,
            arm_results=arm_results,
        )

        # Generate summary
        summary = _generate_comparison_summary(
            openai_client=openai_client,
            query=request.query,
            evidence=all_evidence,
            detailed_analysis=detailed_analysis,
        )

        # Charts disabled — the LLM call to extract numerical data and
        # build chart JSON was expensive (gpt-4o, ~4k tokens) and the
        # extracted values were often hallucinated or imprecise. Pass an
        # empty list so the response shape stays stable for the frontend.
        charts: List[Any] = []

        print(f"[Visual Comparison] Done - {len(treatment_arms)} arms, {len(all_retrieval_results)} total chunks (charts disabled)")

        return VisualComparisonResponse(
            summary=summary,
            detailed_analysis=detailed_analysis,
            charts=charts,
            retrieval_results=all_retrieval_results,
            study_profiles=all_profiles,
            treatment_arms=treatment_arms,
            query_type="treatment_comparison",
        )

    except Exception as e:
        import traceback
        error_detail = f"Error in visual comparison: {str(e)}\n{traceback.format_exc()}"
        print(f"[Visual Comparison] {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_treatment_arms(
    openai_client,
    query: str,
) -> List[Dict[str, str]]:
    """
    Extract treatment arms from a comparison or discovery query using LLM.

    Handles two types of queries:
    1. Comparison queries: "Compare X vs Y for cancer type"
    2. Discovery queries: "What are the treatment options for cancer type"

    Returns a list of dicts with 'label' and 'query' keys.
    Falls back to regex splitting on 'vs'/'versus' if LLM fails.

    Note (RF-4): when the caller passes the returned arms through
    `_build_patient_summary_for_arm_queries`, the LLM-generated `query`
    field is OVERWRITTEN with a deterministic `{label} {patient_summary}
    outcomes` string so per-arm context drift cannot drop clinical
    axes. The LLM `query` field is only used as a fallback when no
    patient summary is available.
    """
    import json

    system_prompt = """You extract treatment arms from oncology queries for evidence retrieval.

You handle TWO types of queries:

1. COMPARISON QUERIES (contain "vs", "versus", "compare"):
   - Extract the specific treatments being compared
   - Example: "Compare pembrolizumab vs chemotherapy for NSCLC"

2. DISCOVERY QUERIES (ask about treatment options, or contain a patient description):
   - Identify ALL major treatment modalities relevant to this specific patient
   - Consider the patient's cancer type, stage, biomarkers, prior treatments, age, and comorbidities
   - Include standard-of-care AND emerging/alternative options
   - Example: "What are the treatment options for a 66-year-old female with pT1 pN0 grade 2 ER+ breast cancer after breast conserving surgery"
   - For this, return treatments specific to THIS patient, not generic options

CRITICAL — TREATMENT PROGRESSION AWARENESS:
When the patient has PROGRESSED on or FAILED a treatment, do NOT include
that treatment as an arm. Instead, include POST-FAILURE salvage options.

Examples:
- "progressing on pembrolizumab" → do NOT create a "Pembrolizumab" arm.
  Instead create arms for post-ICI salvage options (e.g. "Salvage
  Chemotherapy (post-ICI)", "Reirradiation", "Cetuximab-based therapy",
  "Clinical Trials (ICI-refractory)")
- "failed cisplatin-based chemoRT" → do NOT create a "Cisplatin + RT" arm.
  Instead create arms for second-line options.
- "status post surgery, now recurrent" → do NOT create a "Surgery" arm
  if the patient is no longer a surgical candidate. Create "Reirradiation",
  "Systemic salvage", etc.

Read the ENTIRE patient narrative carefully. Look for these signals:
- "progressing on", "progression on", "failed", "refractory to"
- "no longer a surgical candidate", "unresectable", "inoperable"
- "s/p" (status post = already had this treatment)
- "declined" (patient refused this modality)

The arms you generate must represent NEXT treatment options, not
treatments that have already been tried and failed.

Return a JSON array where each element has:
- "label": Short treatment name (e.g., "Whole Breast RT", "Partial Breast RT", "Endocrine Therapy Alone")
- "query": A complete search query for that treatment INCLUDING the full patient clinical context

COMPARISON QUERY EXAMPLE:
Input: "Compare pembrolizumab vs chemotherapy for stage IV NSCLC overall survival"
Output:
[
  {"label": "Pembrolizumab", "query": "pembrolizumab immunotherapy stage IV NSCLC overall survival outcomes"},
  {"label": "Chemotherapy", "query": "chemotherapy stage IV NSCLC overall survival outcomes"}
]

DISCOVERY QUERY EXAMPLES:
Input: "What are the treatment options for a 66-year-old female with pT1 pN0 grade 2 ER+ breast cancer treated with breast conserving surgery and endocrine therapy"
Output:
[
  {"label": "Whole Breast RT", "query": "whole breast radiation therapy pT1 N0 ER+ breast cancer after breast conserving surgery outcomes"},
  {"label": "Partial Breast RT (APBI)", "query": "accelerated partial breast irradiation APBI pT1 N0 low-risk breast cancer outcomes"},
  {"label": "Endocrine Therapy Alone", "query": "endocrine therapy alone omission radiation pT1 N0 ER+ elderly breast cancer outcomes"},
  {"label": "Hypofractionated RT", "query": "hypofractionated whole breast radiation pT1 N0 breast cancer outcomes"},
  {"label": "Boost vs No Boost", "query": "tumor bed boost versus no boost breast conserving surgery pT1 N0 outcomes"}
]

Input: "What are the treatment options for a 72-year-old male with Gleason 3+4 prostate cancer, PSA 5.6"
Output:
[
  {"label": "Active Surveillance", "query": "active surveillance Gleason 3+4 intermediate risk prostate cancer outcomes"},
  {"label": "Radical Prostatectomy", "query": "radical prostatectomy Gleason 3+4 intermediate risk prostate cancer outcomes"},
  {"label": "EBRT + ADT", "query": "external beam radiation androgen deprivation Gleason 3+4 prostate cancer outcomes"},
  {"label": "Brachytherapy", "query": "brachytherapy seed implant Gleason 3+4 prostate cancer outcomes"},
  {"label": "SBRT", "query": "SBRT stereotactic body radiation Gleason 3+4 prostate cancer outcomes"}
]

Input: "80 y.o. male with recurrent SCC oral tongue, progressing on pembrolizumab, no longer a surgical candidate, radiographic concern for metastatic disease"
Output:
[
  {"label": "Reirradiation + Chemotherapy", "query": "reirradiation chemotherapy recurrent head and neck SCC salvage post-ICI ICI-refractory outcomes"},
  {"label": "Cetuximab-based Salvage", "query": "cetuximab recurrent HNSCC post-pembrolizumab ICI-refractory second-line outcomes"},
  {"label": "Palliative RT", "query": "palliative radiation therapy recurrent HNSCC symptom control unresectable outcomes"},
  {"label": "Taxane-based Chemotherapy", "query": "docetaxel paclitaxel recurrent metastatic HNSCC second-line post-ICI outcomes"},
  {"label": "Clinical Trials (ICI-refractory)", "query": "clinical trials ICI-refractory recurrent HNSCC novel agents salvage outcomes"}
]

RULES:
- For discovery queries, identify 4-6 treatment modalities specific to this patient's profile
- Do NOT include treatments the patient has already progressed on or declined
- Consider the patient's stage, biomarkers, prior treatments, and comorbidities when selecting arms
- Always include the cancer type/stage/biomarker context in each arm query
- Keep labels short and clear (specific treatment names, not just "Radiation")
- Include relevant clinical details in queries so retrieval finds patient-relevant studies
- Return ONLY the JSON array, no other text"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=500,
        )

        result = response.choices[0].message.content.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()

        arms = json.loads(result)
        if isinstance(arms, list) and len(arms) >= 2:
            # Validate structure
            valid_arms = []
            for arm in arms:
                if isinstance(arm, dict) and arm.get("label") and arm.get("query"):
                    valid_arms.append({"label": arm["label"], "query": arm["query"]})
            if len(valid_arms) >= 2:
                return valid_arms
    except Exception as e:
        print(f"[Visual Comparison] LLM arm extraction failed: {e}")

    # Fallback: regex split on vs/versus
    return _extract_treatment_arms_regex(query)


def _generate_match_tags(
    evidence_chunk: Dict[str, Any],
    patient_query: str,
    query_structure: Any = None,
) -> List[str]:
    """
    Generate human-readable match tags showing which patient criteria
    a retrieved study matches. Tags appear as pill/badge elements in
    the source list so the user can instantly see WHY a study was
    retrieved.

    Tags are generated by keyword matching against the study's text +
    metadata, compared against the patient's extracted profile.
    """
    import re as _re
    tags: List[str] = []
    text = (evidence_chunk.get("text") or "").lower()
    title = (evidence_chunk.get("title") or "").lower()
    combined = f"{title} {text}"
    category = (evidence_chunk.get("category") or "").lower()

    # Cancer site
    site_tags = {
        "h&n": "Cancer: Head & Neck",
        "breast": "Cancer: Breast",
        "lung": "Cancer: Lung",
        "prostate": "Cancer: Prostate",
        "gi": "Cancer: GI",
        "gu": "Cancer: GU",
        "gyn": "Cancer: GYN",
        "cns": "Cancer: CNS",
        "cutaneous": "Cancer: Skin/Melanoma",
        "lymphoma": "Cancer: Lymphoma",
        "sarcoma": "Cancer: Sarcoma",
        "peds": "Cancer: Pediatric",
    }
    for key, label in site_tags.items():
        if key in category:
            tags.append(label)
            break

    # Histology
    if query_structure:
        hist = getattr(getattr(query_structure, "cancer", None), "histology", None)
        if hist:
            hist_lower = hist.lower()
            if hist_lower in combined or hist_lower.replace("_", " ") in combined:
                tags.append(f"Histology: {hist.upper() if hist.lower() == 'scc' else hist}")

    # Stage
    stage_keywords = [
        ("recurrent", "Stage: Recurrent"),
        ("metastatic", "Stage: Metastatic"),
        ("locally advanced", "Stage: Locally Advanced"),
        ("stage iv", "Stage: IV"), ("stage iii", "Stage: III"),
        ("stage ii", "Stage: II"), ("stage i", "Stage: I"),
        ("unresectable", "Stage: Unresectable"),
    ]
    for kw, label in stage_keywords:
        if kw in combined:
            tags.append(label)
            break

    # Treatment modalities mentioned
    tx_keywords = [
        ("reirradiation", "Tx: Reirradiation"),
        ("sbrt", "Tx: SBRT"), ("stereotactic", "Tx: SBRT"),
        ("chemoradiation", "Tx: Chemoradiation"),
        ("immunotherapy", "Tx: Immunotherapy"), ("checkpoint", "Tx: Immunotherapy"),
        ("pembrolizumab", "Tx: Pembrolizumab"), ("nivolumab", "Tx: Nivolumab"),
        ("cetuximab", "Tx: Cetuximab"),
        ("cisplatin", "Tx: Cisplatin"), ("carboplatin", "Tx: Carboplatin"),
        ("docetaxel", "Tx: Docetaxel"), ("paclitaxel", "Tx: Paclitaxel"),
        ("surgery", "Tx: Surgery"), ("resection", "Tx: Surgery"),
        ("brachytherapy", "Tx: Brachytherapy"),
        ("palliative", "Tx: Palliative"),
    ]
    tx_seen = set()
    for kw, label in tx_keywords:
        if kw in combined and label not in tx_seen:
            tags.append(label)
            tx_seen.add(label)
            if len(tx_seen) >= 3:
                break

    # Biomarkers
    biomarker_keywords = [
        ("cps", "Biomarker: CPS/PD-L1"),
        ("pd-l1", "Biomarker: PD-L1"),
        ("hpv", "Biomarker: HPV"), ("p16", "Biomarker: p16"),
        ("her2", "Biomarker: HER2"),
        ("egfr", "Biomarker: EGFR"),
        ("brca", "Biomarker: BRCA"),
        ("msi", "Biomarker: MSI"),
        ("kras", "Biomarker: KRAS"),
        ("alk", "Biomarker: ALK"),
    ]
    bm_seen = set()
    for kw, label in biomarker_keywords:
        if kw in combined and label not in bm_seen:
            tags.append(label)
            bm_seen.add(label)
            if len(bm_seen) >= 2:
                break

    # Line of therapy
    line_keywords = [
        ("first-line", "Line: 1L"), ("first line", "Line: 1L"),
        ("1l ", "Line: 1L"), ("frontline", "Line: 1L"),
        ("second-line", "Line: 2L+"), ("second line", "Line: 2L+"),
        ("2l ", "Line: 2L+"), ("salvage", "Line: Salvage"),
        ("ici-refractory", "Line: Post-ICI"), ("post-ici", "Line: Post-ICI"),
        ("refractory", "Line: Refractory"),
    ]
    for kw, label in line_keywords:
        if kw in combined:
            tags.append(label)
            break

    return tags


def _extract_treatment_arms_regex(query: str) -> List[Dict[str, str]]:
    """Fallback regex-based treatment arm extraction for comparison and discovery queries."""
    import re
    
    # Try splitting on vs/versus/compared to (for comparison queries)
    parts = re.split(r'\s+(?:vs\.?|versus|compared\s+to|compared\s+with)\s+', query, flags=re.I)
    
    if len(parts) >= 2:
        # Extract shared context (cancer type, stage, etc.) from the full query
        context_patterns = [
            r'(?:for|in)\s+(.+?)(?:\s+(?:overall|progression|disease|local|survival|outcomes|toxicity))',
            r'(?:for|in)\s+(.+?)$',
        ]
        context = ""
        for pattern in context_patterns:
            match = re.search(pattern, query, re.I)
            if match:
                context = match.group(1).strip()
                break
        
        arms = []
        for part in parts:
            # Clean up the part - remove context that's already extracted
            label = part.strip()
            # Remove trailing context phrases
            label = re.sub(r'\s+(?:for|in)\s+.*$', '', label, flags=re.I)
            label = re.sub(r'\s+(?:overall|progression|disease).*$', '', label, flags=re.I)
            label = label.strip()
            
            arm_query = f"{label} {context}" if context else label
            arms.append({"label": label.title(), "query": arm_query})
        
        return arms
    
    # For discovery queries (no vs/versus), extract context and create generic treatment arms
    # This handles queries like "What are the treatment options for stage IV NSCLC"
    context_match = re.search(r'(?:treatment\s+options?\s+for|treatments?\s+for|options?\s+for)\s+(.+?)(?:\s+regarding|\s*$)', query, flags=re.I)
    if context_match:
        context = context_match.group(1).strip()
        print(f"[Visual Comparison] Discovery query detected, context: {context}")
        # Return generic treatment modalities with the context
        return [
            {"label": "Systemic Therapy", "query": f"systemic therapy chemotherapy immunotherapy {context} outcomes"},
            {"label": "Radiation Therapy", "query": f"radiation therapy radiotherapy {context} outcomes"},
            {"label": "Surgery", "query": f"surgical treatment resection {context} outcomes"},
        ]
    
    # Last resort: use the query itself as a single comprehensive search
    print(f"[Visual Comparison] Fallback: using query as single arm")
    return [
        {"label": "Treatment Options", "query": query},
    ]


def _generate_arm_comparison_analysis(
    openai_client,
    query: str,
    arm_results: List[Dict[str, Any]],
) -> str:
    """Generate a detailed comparative analysis across treatment arms.

    Each arm is analysed SEPARATELY with its own full evidence budget so
    no arm's content is truncated by the others. The per-arm analyses are
    then concatenated into one document. This eliminates the old failure
    mode where a 25k-char combined cap meant arms 3-5 had almost no
    evidence in the LLM context and got "no data available" filler.
    """
    try:
        MAX_CHUNKS_PER_ARM = 8
        MAX_CHUNKS_PER_DOC_PER_ARM = 2
        PER_CHUNK_TEXT_LIMIT = 2000

        arm_labels = [arm['label'] for arm in arm_results]
        arm_label_list = ", ".join(arm_labels)
        all_arm_analyses: List[str] = []

        for arm in arm_results:
            # Build per-arm evidence with per-doc diversity cap
            evidence_text = ""
            chunks_per_doc: Dict[str, int] = {}
            distinct_docs: set = set()
            kept = 0

            for e in arm["evidence"][:30]:
                if kept >= MAX_CHUNKS_PER_ARM:
                    break
                doc_id = e.get("doc_id") or ""
                if doc_id and chunks_per_doc.get(doc_id, 0) >= MAX_CHUNKS_PER_DOC_PER_ARM:
                    continue

                title = e.get("title", "Unknown")
                text = (e.get("text") or "")[:PER_CHUNK_TEXT_LIMIT]
                author = e.get("author_et_al", "")
                year = e.get("year", "")
                journal = e.get("journal", "")

                if author and year:
                    citation = f"({author}, {year}, {journal})" if journal else f"({author}, {year})"
                else:
                    citation = ""

                kept += 1
                evidence_text += f"[{kept}] {title} {citation}:\n{text}\n\n"

                if doc_id:
                    chunks_per_doc[doc_id] = chunks_per_doc.get(doc_id, 0) + 1
                    distinct_docs.add(doc_id)

            n_docs = len(distinct_docs)
            print(
                f"[Visual Comparison] Arm '{arm['label']}': "
                f"{kept} chunks from {n_docs} distinct studies"
            )

            if not evidence_text.strip():
                all_arm_analyses.append(
                    f"#### {arm['label']}\n\n"
                    f"No studies were retrieved for this treatment arm "
                    f"in the patient's cancer type."
                )
                continue

            # Generate analysis for THIS ARM ONLY — full evidence budget
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"""You are a clinical oncology expert writing a treatment analysis for ONE treatment arm: "{arm['label']}".

The patient query is: {query[:500]}

Write a focused analysis of THIS treatment arm covering:
1. **Efficacy**: OS, PFS, DFS, local control, response rates — with EXACT values and HRs from the evidence
2. **Toxicity**: Grade 3+ adverse event rates, specific toxicities
3. **Patient selection**: Which patients benefit most from this approach
4. **Applicability to THIS patient**: How the evidence applies to the specific patient described

HARD RULES:
- You are given {n_docs} studies below. READ THEM CAREFULLY and extract
  the specific outcomes they report. Do NOT claim "no evidence is
  available" or "specific data is not provided" when the studies below
  contain relevant data — that is a failure mode.
- Every clinical claim must include a specific number (%, HR, n=, dose)
  OR a named trial from the evidence below.
- If a study is about a DIFFERENT cancer site (e.g. brain metastases,
  anal cancer, lung cancer, melanoma) but the patient has head and neck
  cancer, SKIP that study entirely — do NOT cite it or extrapolate from
  it. Only use studies that match the patient's PRIMARY cancer type.
- If after filtering out wrong-cancer studies the evidence is truly
  sparse, state in ONE sentence: "Only [N] studies were retrieved for
  [arm label] in [patient's cancer type]. [Study Name] reported
  [specific number]." Then stop. Do NOT pad with filler.
- When a study's outcome data is absent because the study was conducted
  before a certain endpoint matured, or because it focused on a
  different endpoint, STATE WHY the data is missing (e.g. "OS data
  not yet mature at median follow-up of 12 months" or "This study
  reported local control but not OS").
- BANNED phrases: "may be effective", "should be considered", "improves
  outcomes", "the evidence does not provide", "specific data is not
  available", "is not directly addressed"
- Cite at the END of sentences: "...finding (Author et al., Year, Journal)."
- Cite as many different studies as possible from the evidence below."""},
                        {"role": "user", "content": f"""EVIDENCE FOR {arm['label']}:

{evidence_text}

Write the analysis for {arm['label']} now. Include inline citations for every claim. Be specific with numbers."""}
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                )
                arm_analysis = resp.choices[0].message.content.strip()
                all_arm_analyses.append(
                    f"#### {arm['label']}\n\n{arm_analysis}"
                )
            except Exception as arm_err:
                print(f"[Visual Comparison] Arm '{arm['label']}' analysis failed: {arm_err}")
                # Fallback to the per-arm answer from retrieval
                if arm.get("answer"):
                    all_arm_analyses.append(
                        f"#### {arm['label']}\n\n{arm['answer']}"
                    )

        # Concatenate all per-arm analyses
        combined_analysis = "\n\n---\n\n".join(all_arm_analyses)

        # Generate a brief comparative synthesis across all arms
        try:
            synth_resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """You are a clinical oncology expert. Given per-arm analyses, write a brief COMPARATIVE SYNTHESIS (200-300 words) that:
1. Directly compares the arms against each other on efficacy (which has better OS/PFS?)
2. Compares toxicity profiles
3. States which arm is most appropriate for THIS specific patient and why
4. Every claim must cite a specific number or trial name from the per-arm analyses.
BANNED: "may be effective", "should be considered", generic filler."""},
                    {"role": "user", "content": f"""Patient: {query[:500]}

Per-arm analyses:
{combined_analysis[:8000]}

Write the comparative synthesis now:"""}
                ],
                temperature=0.2,
                max_tokens=800,
            )
            synthesis = synth_resp.choices[0].message.content.strip()
            combined_analysis += f"\n\n---\n\n#### Comparative Synthesis\n\n{synthesis}"
        except Exception:
            pass  # Synthesis is optional — per-arm analyses are the core content

        return combined_analysis

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Visual Comparison] Arm comparison analysis failed: {e}")
        parts = []
        for arm in arm_results:
            if arm.get("answer"):
                parts.append(f"**{arm['label']}**\n{arm['answer']}")
        return "\n\n".join(parts) if parts else "Comparison analysis could not be generated."


def _generate_comparison_summary(
    openai_client,
    query: str,
    evidence: List[Dict[str, Any]],
    detailed_analysis: str
) -> str:
    """Generate a concise summary of the comparison."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a clinical oncology expert.

Distill the detailed comparative analysis into a 2-4 sentence summary
that names ONE specific treatment as the leading option for THIS
patient and quotes at least TWO specific quantitative results from
DIFFERENT trials in the analysis (e.g. "5-year LRC 72% vs 58% in
EORTC 22931" + "median OS 13 vs 9 mo, HR 0.70 in RTOG 9501").

HARD RULES:
- Name a specific regimen / dose / fractionation, not a treatment
  category. ("Concurrent cisplatin + 60 Gy / 30 fx", not "chemoradiation".)
- Cite at least 2 DIFFERENT trials from the analysis with specific
  numbers from each.
- BANNED phrases: "may be effective", "should be considered",
  "improves outcomes", "is generally considered", "warrants" — replace
  with the specific number or trial finding.
- If the analysis itself is sparse for the patient's cancer type,
  state that explicitly in one sentence; do not fill space with
  generic prose.
- Do NOT mention outcomes from a different primary cancer type as if
  they apply to the patient."""},
                {"role": "user", "content": f"""Query: {query}

Detailed analysis (use ONLY the studies and numbers below — do not
invent any):
{detailed_analysis[:4000]}

Write the 2-4 sentence summary now. Name one leading regimen with
its dose / fractionation and quote at least two specific quantitative
results from at least two DIFFERENT trials in the analysis above:"""}
            ],
            temperature=0.2,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Visual Comparison] Summary generation failed: {e}")
        return detailed_analysis[:500] if detailed_analysis else "Comparison analysis generated."


def _generate_comparison_charts(
    openai_client,
    query: str,
    evidence: List[Dict[str, Any]],
    detailed_analysis: str
) -> List[Artifact]:
    """Generate multiple chart artifacts from evidence data."""
    import json
    
    # Build evidence context
    evidence_text = "\n\n".join([
        f"[{i+1}] {e.get('title', 'Unknown')}: {e.get('text', '')[:600]}"
        for i, e in enumerate(evidence[:10])
    ])
    
    system_prompt = """You are a data extraction assistant for clinical oncology research.
Extract ALL numerical comparison data from the evidence and create multiple charts.

IMPORTANT: Generate AS MANY charts as possible from the data. Look for:
1. Survival outcomes (OS, PFS, DFS) - create bar charts comparing treatments
2. Response rates (CR, PR, ORR) - create bar charts
3. Toxicity rates by grade - create grouped bar charts
4. Hazard ratios with confidence intervals - create forest plot style data
5. Time-to-event data - create line chart data if available

OUTPUT FORMAT (JSON array of charts):
[
    {
        "chart_type": "bar",
        "title": "Overall Survival Comparison",
        "labels": ["Treatment A", "Treatment B"],
        "datasets": [
            {
                "label": "5-Year OS",
                "data": [57.6, 44.6],
                "unit": "%"
            }
        ],
        "source": "Study citation"
    },
    {
        "chart_type": "bar",
        "title": "Grade 3+ Toxicity Rates",
        "labels": ["Pneumonitis", "Esophagitis", "Fatigue"],
        "datasets": [
            {"label": "Treatment A", "data": [5, 12, 8], "unit": "%"},
            {"label": "Treatment B", "data": [3, 15, 10], "unit": "%"}
        ],
        "source": "Study citation"
    }
]

RULES:
1. Extract ONLY explicitly stated numerical data
2. Create separate charts for different outcome types
3. Always include units (%, months, patients, etc.)
4. Include source citation for each chart
5. If no numerical data found, return empty array []
6. Aim for 2-4 charts if data is available"""

    user_prompt = f"""COMPARISON QUERY: {query}

EVIDENCE FROM CLINICAL TRIALS:
{evidence_text}

ANALYSIS:
{detailed_analysis[:1500]}

Extract numerical data and create multiple comparison charts.
Return a JSON array of chart objects. If no numerical data is available, return [].

JSON array:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        charts_data = json.loads(result)
        
        if not isinstance(charts_data, list):
            print("[Visual Comparison] Response is not a list")
            return []
        
        # Convert to Artifact objects
        artifacts = []
        colors = [
            ["rgba(59, 130, 246, 0.8)", "rgba(16, 185, 129, 0.8)", "rgba(245, 158, 11, 0.8)", "rgba(239, 68, 68, 0.8)"],
            ["rgba(139, 92, 246, 0.8)", "rgba(236, 72, 153, 0.8)", "rgba(34, 197, 94, 0.8)", "rgba(251, 146, 60, 0.8)"],
        ]
        
        for chart_data in charts_data:
            if not chart_data.get("title") or not chart_data.get("labels"):
                continue
            
            datasets = []
            for i, ds in enumerate(chart_data.get("datasets", [])):
                num_values = len(ds.get("data", []))
                color_set = colors[i % len(colors)]
                bg_colors = [color_set[j % len(color_set)] for j in range(num_values)]
                border_colors = [c.replace("0.8", "1") for c in bg_colors]
                
                datasets.append(ChartDataset(
                    label=ds.get("label", f"Dataset {i+1}"),
                    data=ds.get("data", []),
                    backgroundColor=bg_colors,
                    borderColor=border_colors,
                ))
            
            if datasets:
                chart = ChartArtifact(
                    type=chart_data.get("chart_type", "bar"),
                    title=chart_data["title"],
                    labels=chart_data["labels"],
                    datasets=datasets,
                    unit=chart_data.get("datasets", [{}])[0].get("unit"),
                    source=chart_data.get("source"),
                )
                artifacts.append(Artifact(artifact_type="chart", chart=chart))
        
        return artifacts
        
    except json.JSONDecodeError as e:
        print(f"[Visual Comparison] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"[Visual Comparison] Chart generation error: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================
# DEEP DIVE ENDPOINT
# ============================================

@router.post("/deep-dive", response_model=DeepDiveResponse)
async def deep_dive_query(request: DeepDiveRequest):
    """
    Deep dive query with explicit or inferred tumor site context.
    
    This endpoint:
    - Infers tumor site from the query if not provided
    - Constructs a site-specific query for better retrieval
    - Returns more comprehensive results (up to 30 chunks)
    
    **Example Request:**
    ```json
    {
        "question": "What is the recommended chemotherapy regimen?",
        "site_key": "Breast",
        "top_k": 15
    }
    ```
    """
    try:
        from src.api.services.enhanced_rag_service import get_enhanced_rag_service
        
        # Get RAG service
        rag_service = get_enhanced_rag_service()
        
        # Execute deep dive query
        result = rag_service.deep_dive(
            question=request.question,
            site_key=request.site_key,
            top_k=request.top_k,
            category_filter=request.category_filter
        )
        
        # Convert evidence to retrieval results
        retrieval_results = []
        for e in result.get("evidence", []):
            table_info = None
            if e.get("table"):
                table_info = TableInfo(**e["table"])
            
            retrieval_results.append(RetrievalResult(
                doc_id=e.get("doc_id"),
                title=e.get("title"),
                author=_get_author_with_fallback(e),
                citation=e.get("citation"),
                doi=e.get("doi"),
                pmid=e.get("pmid"),
                year=e.get("year"),
                category=e.get("category"),
                section=e.get("section"),
                chunk_type=e.get("chunk_type"),
                content=e.get("text", ""),
                score=e.get("score"),
                relevance_score=_normalize_crossencoder_score(e.get("score_crossencoder")),
                table=table_info
            ))
        
        # Build metadata
        metadata_dict = result.get("metadata", {})
        metadata = QueryMetadata(**metadata_dict)
        
        return DeepDiveResponse(
            query=result["query"],
            site_key=result["site_key"],
            site_label=result["site_label"],
            summary=result["summary"],
            evidence=retrieval_results,
            metadata=metadata
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Enhanced RAG service not available: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_detail = f"Error in deep dive query: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# ============================================
# INTENT ANALYSIS - Smart Query Understanding
# ============================================

@router.post("/analyze-intent", response_model=IntentAnalysisResponse)
async def analyze_query_intent(
    request: IntentAnalysisRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Analyze a user's query to detect intent and extract patient information.
    
    This endpoint is useful when a user provides patient information without
    an explicit question. It will:
    1. Detect if the query contains an explicit question
    2. Extract patient profile information
    3. Generate relevant follow-up options
    
    **Use Cases:**
    - User pastes patient details without asking a question
    - User provides a clinical scenario and needs guidance
    - Conversational AI that offers intelligent suggestions
    
    **Example Input:**
    ```
    "68 year old female, non-smoker, with SCC of R maxilla s/p maxillectomy, 
    pT4N0, negative margins, no LVI/PNI"
    ```
    
    **Response includes:**
    - Detected intent (patient_description, explicit_question, etc.)
    - Extracted patient profile
    - Suggested follow-up actions with pre-filled queries
    """
    try:
        from src.api.services.query_intent_service import get_query_intent_service
        
        # Get user_id for preference filtering
        user_id = current_user["id"] if current_user else None
        
        service = get_query_intent_service()
        result = await service.analyze_query(
            request.query,
            find_matching_trials=True,
            force_trial_match=request.force_trial_match,
            user_id=user_id,
        )
        
        # Convert to response models
        intent_info = QueryIntentInfo(
            intent_type=result.intent.intent_type,
            has_explicit_question=result.intent.has_explicit_question,
            confidence=result.intent.confidence,
            detected_question_type=result.intent.detected_question_type
        )
        
        patient_profile = None
        if result.patient_profile:
            patient_profile = ExtractedPatientProfile(
                age=result.patient_profile.age,
                gender=result.patient_profile.gender,
                ethnicity=result.patient_profile.ethnicity,
                smoking_status=result.patient_profile.smoking_status,
                cancer_type=result.patient_profile.cancer_type,
                cancer_location=result.patient_profile.cancer_location,
                histology=result.patient_profile.histology,
                stage=result.patient_profile.stage,
                tnm_t=result.patient_profile.tnm_t,
                tnm_n=result.patient_profile.tnm_n,
                tnm_m=result.patient_profile.tnm_m,
                tumor_size=result.patient_profile.tumor_size,
                doi=result.patient_profile.doi,
                lvi=result.patient_profile.lvi,
                pni=result.patient_profile.pni,
                margins=result.patient_profile.margins,
                lymph_nodes=result.patient_profile.lymph_nodes,
                other_pathology=result.patient_profile.other_pathology,
                molecular_markers=result.patient_profile.molecular_markers,
                prior_treatment=result.patient_profile.prior_treatment,
                comorbidities=result.patient_profile.comorbidities,
                performance_status=result.patient_profile.performance_status,
                recurrence_status=result.patient_profile.recurrence_status,
                treatment_setting=result.patient_profile.treatment_setting
            )
        
        follow_up_options = [
            FollowUpOptionInfo(
                action_type=opt.action_type,
                label=opt.label,
                description=opt.description,
                query_template=opt.query_template
            )
            for opt in result.follow_up_options
        ]
        
        # Generate user-facing message
        if result.should_prompt_user:
            message = f"I understand you're describing: **{result.patient_summary}**\n\nHow would you like me to help? Select an option below or ask a specific question."
        elif result.intent.has_explicit_question:
            message = "Processing your question..."
        else:
            message = "I've analyzed your input. What would you like to know?"
        
        # Convert matching trials
        matching_trials = [
            MatchingTrialInfo(
                title=trial.title,
                author=trial.author,
                year=trial.year,
                match_score=trial.match_score,
                match_reasons=trial.match_reasons,
                relevant_excerpt=trial.relevant_excerpt,
                doi=trial.doi,
                treatment=trial.treatment,
                inclusion_criteria=trial.inclusion_criteria,
                exclusion_criteria=trial.exclusion_criteria,
                eligibility_notes=trial.eligibility_notes,
                population_details=getattr(trial, 'population_details', None)
            )
            for trial in result.matching_trials
        ]
        
        return IntentAnalysisResponse(
            intent=intent_info,
            patient_profile=patient_profile,
            patient_summary=result.patient_summary,
            follow_up_options=follow_up_options,
            should_prompt_user=result.should_prompt_user,
            auto_action=result.auto_action,
            message=message,
            formatted_response=result.formatted_response,
            matching_trials=matching_trials
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Intent analysis failed: {str(e)}"
        )


# ============================================
# HEALTH CHECK
# ============================================

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify RAG service availability.
    
    Tests:
    - Service initialization
    - Qdrant connection
    - Cross-encoder availability
    - Test query execution
    """
    try:
        from src.api.services.enhanced_rag_service import (
            get_enhanced_rag_service, 
            QDRANT_COLLECTION,
            CROSS_ENCODER_AVAILABLE
        )
        
        # Get service
        rag_service = get_enhanced_rag_service()
        
        # Try a simple test query
        try:
            test_result = await rag_service.query(
                question="lung cancer treatment",
                top_k=1
            )
            test_query_success = True
            results_count = len(test_result.get("evidence", []))
        except Exception as e:
            test_query_success = False
            results_count = 0
            print(f"Test query failed: {e}")
        
        return HealthCheckResponse(
            status="healthy",
            collection=QDRANT_COLLECTION,
            cross_encoder_available=CROSS_ENCODER_AVAILABLE,
            test_query_success=test_query_success,
            results_count=results_count
        )
        
    except Exception as e:
        return HealthCheckResponse(
            status=f"unhealthy: {str(e)}",
            collection=None,
            cross_encoder_available=None,
            test_query_success=False,
            results_count=0
        )


# ============================================
# STANDALONE CLASSIFICATION
# ============================================

@router.post("/query/classify")
async def classify_query(request: ClassifyRequest):
    """
    Standalone query classification endpoint.
    
    Uses the Unified_Router to classify a query and return routing information
    without executing the full RAG pipeline. Useful for:
    - Frontend pre-routing decisions
    - Debugging classification behavior
    - Testing query patterns
    
    **Example Request:**
    ```json
    {
        "query": "68 yo male with T3N1 rectal cancer, what treatment?",
        "include_format_hints": true
    }
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "routing": {
            "module": "patient_specific",
            "module_confidence": 0.88,
            "query_type": "treatment_recommendation",
            "format_hints": {...},
            "signals_matched": ["demographics:68 yo", "staging:T3N1"]
        }
    }
    ```
    """
    try:
        from src.api.services.unified_router import get_unified_router
        
        print(f"[Classify] Classifying query: {request.query[:100]}...")
        
        router = get_unified_router()
        routing_result = router.route_query(request.query)
        
        # Convert to dict
        routing_dict = routing_result.to_dict()
        
        # Optionally exclude format hints
        if not request.include_format_hints:
            routing_dict.pop("format_hints", None)
            routing_dict.pop("retrieval_strategy", None)
        
        print(f"[Classify] Result: module={routing_dict.get('module')}, confidence={routing_dict.get('module_confidence')}")
        
        return {
            "success": True,
            "routing": routing_dict
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# QUERY MODES
# ============================================

@router.get("/modes", response_model=QueryModesResponse)
async def get_query_modes():
    """
    Get available query modes.
    
    Note: The enhanced RAG system automatically selects the best retrieval
    strategy based on query type classification, but this endpoint maintains
    compatibility with the original interface.
    """
    return QueryModesResponse(
        modes=[
            QueryMode(
                id="naive",
                name="Naive",
                description="Simple retrieval (legacy mode)"
            ),
            QueryMode(
                id="local",
                name="Local",
                description="Local context search (legacy mode)"
            ),
            QueryMode(
                id="global",
                name="Global",
                description="Global knowledge search (legacy mode)"
            ),
            QueryMode(
                id="hybrid",
                name="Hybrid (Enhanced)",
                description="Auto-optimized retrieval with query classification (recommended)"
            )
        ]
    )


# ============================================
# AVAILABLE SITES
# ============================================

@router.get("/sites", response_model=SitesResponse)
async def get_available_sites():
    """
    Get available tumor sites for deep dive queries.
    
    These are the cancer categories available in the knowledge base.
    """
    from src.api.services.enhanced_rag_service import SITE_LABELS
    
    sites = []
    for key, label in SITE_LABELS.items():
        sites.append(SiteInfo(key=key, label=label))
    
    return SitesResponse(sites=sites)


# ============================================
# PATIENT MATCHING ENDPOINT
# ============================================

# @router.post("/patient/match", response_model=PatientMatchResponse)
# async def match_patient(patient_profile: PatientProfile):
#     """
#     Match patient characteristics to relevant clinical studies.
#     
#     This endpoint uses the enhanced patient matching service to find studies that match
#     the patient's demographics, cancer characteristics, and other factors.
#     
#     **Patient Characteristics:**
#     - age: Patient age
#     - gender: Gender (male, female)
#     - cancer_type: Type of cancer
#     - cancer_stage: Cancer stage (I, II, III, IV)
#     - histology: Histology type
#     - molecular_markers: List of molecular markers (e.g., EGFR+, PD-L1+)
#     - performance_status: Performance status (ECOG 0, 1, 2, etc.)
#     - comorbidities: List of comorbidities
#     - smoking_status: Smoking status
#     
#     **Returns:**
#     - List of matching studies with match scores
#     - Match reasons and relevant excerpts
#     """
#     try:
#         from qdrant_client import QdrantClient
#         from openai import OpenAI
#         from src.core.config import settings
#         from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
#         
#         # Create patient matching service
#         qdrant_client = QdrantClient(
#             url=settings.qdrant_url,
#             api_key=settings.qdrant_api_key,
#             timeout=60  # 60 second timeout for cloud connections
#         )
#         openai_client = OpenAI(api_key=settings.openai_api_key)
#         
#         service = SimplePatientMatchingService(
#             qdrant_client=qdrant_client,
#             openai_client=openai_client,
#             collection_name=settings.qdrant_collection,
#             embed_model=settings.embed_model
#         )
#         
#         # Build profile dict from Pydantic model
#         profile_dict = {
#             "cancer_type": patient_profile.cancer_type,
#             "age": patient_profile.age,
#             "gender": patient_profile.gender,
#             "cancer_stage": patient_profile.cancer_stage,
#             "histology": patient_profile.histology,
#             "molecular_markers": patient_profile.molecular_markers,
#             "performance_status": patient_profile.performance_status,
#             "comorbidities": patient_profile.comorbidities,
#             "smoking_status": patient_profile.smoking_status,
#         }
#         # Remove None values
#         profile_dict = {k: v for k, v in profile_dict.items() if v is not None}
#         
#         # Execute patient matching
#         result = service.match_patient(profile_dict, top_k=15)
#         
#         # Convert results to StudyMatch format
#         matches = []
#         for match in result.get("matches", []):
#             # Build match reasons from demographics, cancer_characteristics, key_matches
#             match_reasons = []
#             demographics = match.get("demographics", [])
#             cancer_chars = match.get("cancer_characteristics", [])
#             key_matches = match.get("key_matches", [])
#             
#             if demographics:
#                 match_reasons.extend([f"Demographics: {d}" for d in demographics])
#             if cancer_chars:
#                 match_reasons.extend([f"Cancer: {c}" for c in cancer_chars])
#             if key_matches:
#                 match_reasons.extend([f"Marker: {k}" for k in key_matches])
#             
#             if not match_reasons:
#                 match_reasons = ["Semantic similarity to patient profile"]
#             
#             matches.append(StudyMatch(
#                 title=match.get("title", "Unknown"),
#                 author=match.get("author"),
#                 citation=match.get("citation"),
#                 doi=match.get("doi"),
#                 year=match.get("year"),
#                 match_score=match.get("match_score", 0.0),
#                 confidence=match.get("match_score", 0.0),  # Use same score for confidence
#                 match_reasons=match_reasons,
#                 relevant_text=match.get("relevant_text", ""),
#                 treatment=match.get("treatment"),
#                 key_info=match.get("key_info"),
#                 demographics=demographics,
#                 cancer_characteristics=cancer_chars,
#                 key_matches=key_matches
#             ))
#         
#         return PatientMatchResponse(
#             matches=matches,
#             total_matches=result.get("total_matches", len(matches)),
#             patient_summary=result.get("patient_summary", "Patient")
#         )
#         
#     except Exception as e:
#         import traceback
#         error_detail = f"Error matching patient: {str(e)}\n{traceback.format_exc()}"
#         raise HTTPException(status_code=500, detail=error_detail)



# ============================================
# TREATMENT COMPARISON ENDPOINT
# ============================================

@router.post("/comparison/treatments", response_model=TreatmentComparisonResponse)
async def compare_treatments(request: TreatmentComparisonRequest):
    """
    Compare two treatments side by side across medical literature.
    
    This endpoint uses the enhanced RAG system to find and compare
    evidence for two different treatments.
    
    **Parameters:**
    - treatment_a: First treatment name
    - treatment_b: Second treatment name
    - cancer_type: Optional cancer type filter
    - stage: Optional cancer stage filter
    - top_k: Number of studies to retrieve per treatment
    
    **Returns:**
    - Side-by-side comparison of efficacy, safety, dosing
    - Supporting evidence from medical literature
    - Comparison summary and recommendations
    """
    try:
        from src.api.services.enhanced_rag_service import infer_site_key
        from src.api.services.multi_specialty_retrieval import (
            retrieve_evidence_multispecialty,
            MultiSpecialtyEvidence,
        )
        from src.api.services.tumor_board.retrieval import LightweightStudy

        # "Evaluate treatment options" runs through the SAME multi-specialty
        # retrieval pipeline used by the tumor board (and now Trial Match
        # and Patient Matching). Six specialty agents each build their own
        # specialty-aware sub-queries from a synthetic case bundle and fan
        # them out via the tumor board's lightweight Qdrant search. We
        # stop BEFORE the LLM expert-assessment step.

        def normalize_category_filter(cancer_type: Optional[str]) -> Optional[str]:
            if not cancer_type:
                return None
            cancer_type = cancer_type.strip()
            if not cancer_type:
                return None
            lower = cancer_type.lower()
            if lower.endswith("_processed_documents"):
                return lower
            site_key = infer_site_key(cancer_type)
            if site_key:
                return f"{site_key.lower()}_processed_documents"
            return None

        def _studies_to_evidence_dicts(
            studies: List[LightweightStudy],
        ) -> List[Dict[str, Any]]:
            """
            Flatten LightweightStudy objects into the per-chunk dict shape
            the rest of this route expects (text/section/score plus
            doi/pmid/author_et_al/year/citation flattened from doc_meta,
            and score_crossencoder mapped from rerank_score).
            """
            flattened: List[Dict[str, Any]] = []
            for s in studies:
                for chunk in (s.chunks or []):
                    doc_meta = dict(chunk.get("doc_meta") or {})
                    e: Dict[str, Any] = {
                        "doc_id": s.doc_id,
                        "title": doc_meta.get("title") or s.title,
                        "citation": doc_meta.get("citation") or s.citation,
                        "doi": doc_meta.get("doi"),
                        "pmid": doc_meta.get("pmid"),
                        "year": doc_meta.get("year") or s.year,
                        "author_et_al": doc_meta.get("author_et_al"),
                        "journal": doc_meta.get("journal"),
                        "category": doc_meta.get("category")
                                    or chunk.get("category"),
                        "section": chunk.get("section"),
                        "chunk_type": chunk.get("chunk_type"),
                        "text": chunk.get("text", ""),
                        "score": chunk.get("score", s.rerank_score or 0.0),
                        "score_crossencoder": s.rerank_score,
                        "doc_meta": doc_meta,
                        "_specialties": list(getattr(s, "specialties", []) or []),
                    }
                    if chunk.get("table"):
                        e["table"] = chunk["table"]
                    flattened.append(e)
            return flattened

        def _build_treatment_case_text(treatment_name: str) -> str:
            """
            Build a synthetic case-narrative string the multi-specialty
            bundle extractor can parse. We deliberately phrase this as a
            patient-style sentence so the regex extractor populates a
            CancerContext (which in turn flips `has_patient_context` to
            True and lets the specialty agents build their sub-queries).
            """
            cancer = (request.cancer_type or "cancer").strip()
            stage_part = f"stage {request.stage}" if request.stage else ""
            stage_clause = f" with {stage_part}" if stage_part else ""
            return (
                f"Patient with {cancer}{stage_clause} being considered for "
                f"{treatment_name}. Evaluate efficacy, safety, dosing, "
                f"survival outcomes, response rates, and adverse events for "
                f"{treatment_name} in {cancer}."
            )

        category_filter = normalize_category_filter(request.cancer_type)

        async def run_query_with_fallback(
            treatment_name: str,
        ) -> Tuple[List[Dict[str, Any]], bool, MultiSpecialtyEvidence]:
            """
            Run the multi-specialty pipeline for one treatment.

            Returns (evidence_list, used_fallback, ms_result). Falls back
            to no category filter if the filtered run returned nothing.
            `force_all_agents=True` ensures every specialty fires even
            though a treatment-name query lacks rich patient context.
            """
            case_text = _build_treatment_case_text(treatment_name)
            ms_result = await retrieve_evidence_multispecialty(
                case_text=case_text,
                query_type="treatment_recommendation",
                category=category_filter,
                max_studies=max(request.top_k * 2, request.top_k + 5),
                force_all_agents=True,
            )
            used_fallback = False
            if category_filter and not ms_result.merged_studies:
                used_fallback = True
                ms_result = await retrieve_evidence_multispecialty(
                    case_text=case_text,
                    query_type="treatment_recommendation",
                    category=None,
                    max_studies=max(request.top_k * 2, request.top_k + 5),
                    force_all_agents=True,
                )
            return (
                _studies_to_evidence_dicts(ms_result.merged_studies),
                used_fallback,
                ms_result,
            )

        # Query for treatment A
        evidence_a, fallback_a, ms_a = await run_query_with_fallback(
            request.treatment_a
        )

        # Query for treatment B
        evidence_b, fallback_b, ms_b = await run_query_with_fallback(
            request.treatment_b
        )

        # Reconstruct the legacy `query_a` / `query_b` strings for the
        # response metadata field
        context_parts = []
        if request.cancer_type:
            context_parts.append(request.cancer_type)
        if request.stage:
            context_parts.append(f"stage {request.stage}")
        context = " ".join(context_parts)
        query_a = f"{request.treatment_a} {context} efficacy safety dosing outcomes"
        query_b = f"{request.treatment_b} {context} efficacy safety dosing outcomes"

        # Extract supporting study citations for treatment A
        treatment_a_studies = [e.get("citation", e.get("title", "")) for e in evidence_a[:5]]
        
        # Build evidence summary for A
        efficacy_a_parts = []
        safety_a_parts = []
        dosing_a_parts = []
        
        for e in evidence_a[:3]:
            text = e.get("text", "").lower()
            if any(word in text for word in ["response", "survival", "efficacy", "outcome"]):
                efficacy_a_parts.append(e.get("text", "")[:200])
            if any(word in text for word in ["adverse", "toxicity", "side effect", "safety"]):
                safety_a_parts.append(e.get("text", "")[:200])
            if any(word in text for word in ["dose", "dosing", "mg", "administration"]):
                dosing_a_parts.append(e.get("text", "")[:200])
        
        treatment_a_evidence = TreatmentEvidence(
            efficacy=" | ".join(efficacy_a_parts[:2]) if efficacy_a_parts else "Limited efficacy data available",
            safety=" | ".join(safety_a_parts[:2]) if safety_a_parts else "Limited safety data available",
            dosing=" | ".join(dosing_a_parts[:2]) if dosing_a_parts else "Limited dosing data available",
            outcomes=f"Found in {len(evidence_a)} studies",
            studies=treatment_a_studies
        )
        
        # Extract evidence for treatment B
        treatment_b_studies = [e.get("citation", e.get("title", "")) for e in evidence_b[:5]]
        
        # Build evidence summary for B
        efficacy_b_parts = []
        safety_b_parts = []
        dosing_b_parts = []
        
        for e in evidence_b[:3]:
            text = e.get("text", "").lower()
            if any(word in text for word in ["response", "survival", "efficacy", "outcome"]):
                efficacy_b_parts.append(e.get("text", "")[:200])
            if any(word in text for word in ["adverse", "toxicity", "side effect", "safety"]):
                safety_b_parts.append(e.get("text", "")[:200])
            if any(word in text for word in ["dose", "dosing", "mg", "administration"]):
                dosing_b_parts.append(e.get("text", "")[:200])
        
        treatment_b_evidence = TreatmentEvidence(
            efficacy=" | ".join(efficacy_b_parts[:2]) if efficacy_b_parts else "Limited efficacy data available",
            safety=" | ".join(safety_b_parts[:2]) if safety_b_parts else "Limited safety data available",
            dosing=" | ".join(dosing_b_parts[:2]) if dosing_b_parts else "Limited dosing data available",
            outcomes=f"Found in {len(evidence_b)} studies",
            studies=treatment_b_studies
        )
        
        # Create comparison summary
        comparison_summary = f"Comparison of {request.treatment_a} vs {request.treatment_b}"
        if context:
            comparison_summary += f" for {context}"
        comparison_summary += f". Found evidence from {len(evidence_a)} studies for {request.treatment_a} and {len(evidence_b)} studies for {request.treatment_b}."
        if fallback_a or fallback_b:
            comparison_summary += " Category filtering returned no results for at least one treatment; fallback to unfiltered retrieval."
        
        # Combine sources
        all_sources = []
        for e in (evidence_a + evidence_b)[:10]:
            table_info = None
            if e.get("table"):
                table_info = TableInfo(**e["table"])
            
            all_sources.append(RetrievalResult(
                doc_id=e.get("doc_id"),
                title=e.get("title"),
                author=_get_author_with_fallback(e),
                citation=e.get("citation"),
                doi=e.get("doi"),
                pmid=e.get("pmid"),
                year=e.get("year"),
                category=e.get("category"),
                section=e.get("section"),
                chunk_type=e.get("chunk_type"),
                content=e.get("text", ""),
                score=e.get("score"),
                relevance_score=_normalize_crossencoder_score(e.get("score_crossencoder")),
                table=table_info
            ))
        
        # Build metadata
        metadata = QueryMetadata(
            query_type="COMPARISON_QUERY",
            expanded_query=f"{query_a} | {query_b}",
            num_results=len(all_sources),
            processing_time_ms=0,
            reranked=True
        )
        
        return TreatmentComparisonResponse(
            comparison=TreatmentComparisonResult(
                treatment_a_name=request.treatment_a,
                treatment_b_name=request.treatment_b,
                treatment_a_evidence=treatment_a_evidence,
                treatment_b_evidence=treatment_b_evidence,
                comparison_summary=comparison_summary,
                statistical_significance=None,
                recommendation=None
            ),
            sources=all_sources,
            metadata=metadata
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error comparing treatments: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
