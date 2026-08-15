"""
Pydantic models for Enhanced RAG API

All request/response models for the query endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message in conversation history"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    # Optional: sources used in assistant responses for follow-up context
    sources: Optional[List[str]] = Field(
        default=None,
        description="Doc IDs of sources used (for assistant messages)"
    )
    source_citations: Optional[List[str]] = Field(
        default=None,
        description="Citation strings of sources used (for assistant messages)"
    )


class ConversationContextEntry(BaseModel):
    """Single entry in conversation context for automatic conversation mode.
    
    Stores lightweight context data for follow-up queries without LLM summarization.
    Used for retrieval boosting and conversation history inclusion.
    """
    query: str = Field(..., description="Raw user query text (no summarization)")
    action_type: str = Field(
        ..., 
        description="Type of action: query, eval_treatment, patient_match, study_comparison, followup"
    )
    doc_ids: List[str] = Field(
        default_factory=list, 
        description="Doc IDs from retrieved chunks for retrieval boosting"
    )
    doc_titles: List[str] = Field(
        default_factory=list, 
        description="Document titles for display context"
    )
    timestamp: int = Field(..., description="Unix timestamp in milliseconds when entry was created")
    treatments: Optional[List[str]] = Field(
        default=None, 
        description="Identified treatment options (for eval_treatment action type)"
    )


class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str = Field(..., description="User's question", min_length=1)
    query_mode: str = Field(
        default="hybrid",
        description="Query mode (kept for compatibility)"
    )
    conversation_history: List[Message] = Field(
        default=[],
        description="Previous conversation messages"
    )
    top_k: int = Field(
        default=10,
        description="Number of evidence chunks to retrieve",
        ge=1,
        le=20
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category (e.g., 'Breast', 'Lung')"
    )
    use_site_inference: bool = Field(
        default=False,
        description="Automatically infer tumor site from query"
    )
    accumulated_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Accumulated structured context from previous queries in conversation"
    )
    conversation_context: Optional[List[ConversationContextEntry]] = Field(
        default=None,
        description="Conversation context entries from frontend sessionStorage for automatic conversation mode"
    )
    use_study_focused: bool = Field(
        default=False,
        description=(
            "DEPRECATED: this flag is retired. All comprehensive retrieval is now "
            "served by retrieve_evidence(mode='comprehensive'). When True the "
            "request is silently re-routed through the same backbone; a deprecation "
            "log line is emitted. The flag will be removed in a future release."
        ),
    )
    max_studies: int = Field(
        default=5,
        description="Maximum number of studies for study-focused retrieval",
        ge=1,
        le=10
    )
    chunks_per_study: int = Field(
        default=8,
        description="Maximum chunks per study for study-focused retrieval",
        ge=2,
        le=15
    )


class DeepDiveRequest(BaseModel):
    """Request model for deep dive queries"""
    question: str = Field(..., description="User's question", min_length=1)
    site_key: Optional[str] = Field(
        default=None,
        description="Tumor site key (will auto-infer if not provided)"
    )
    top_k: int = Field(
        default=15,
        description="Number of evidence chunks to retrieve",
        ge=1,
        le=30
    )
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional category filter"
    )


class TableInfo(BaseModel):
    """Table information in evidence"""
    number: Optional[str] = None
    title: Optional[str] = None
    row_index: Optional[int] = None
    headers: List[str] = []
    raw_row: List[Any] = []
    page: Optional[int] = None


class RetrievalResult(BaseModel):
    """Single evidence chunk from retrieval"""
    doc_id: Optional[str] = Field(None, description="Document ID for study details lookup")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author(s)")
    citation: Optional[str] = Field(None, description="Full citation")
    doi: Optional[str] = Field(None, description="DOI if available")
    pmid: Optional[str] = Field(None, description="PubMed ID if available")
    year: Optional[int] = Field(None, description="Publication year")
    category: Optional[str] = Field(None, description="Cancer category")
    section: Optional[str] = Field(None, description="Document section")
    chunk_type: Optional[str] = Field(None, description="Type of chunk")
    content: str = Field(..., description="Evidence text")
    score: Optional[float] = Field(None, description="Relevance score")
    relevance_score: Optional[float] = Field(None, description="Cross-encoder relevance score (0-100 percentage)")
    source_type: Optional[str] = Field(None, description="Source type: 'kb' (knowledge base), 'pubmed', 'clinicaltrials', 'user_upload'")
    table: Optional[TableInfo] = Field(None, description="Table information if table chunk")
    number_of_patients: Optional[int] = Field(None, description="Number of patients in the study")
    citation_count: Optional[int] = Field(None, description="Number of citations for the study")
    match_tags: List[str] = Field(default_factory=list, description="Patient criteria this study matches (e.g. 'Cancer: H&N SCC', 'Stage: recurrent', 'Biomarker: CPS ≥ 20')")
    # Patient–study match score (0–100): weighted overlap of the
    # patient's ClinicalProfile axes against the study's doc_level_*
    # metadata. Distinct from relevance_score (cross-encoder chunk
    # match to the query). Surfaced as the "Match X%" badge in the
    # source list when the scorer ran — non-patient queries leave
    # this null and the badge does not render.
    patient_match_score: Optional[int] = Field(None, description="Patient–study population match (0–100); only set for patient-specific queries")
    patient_match_breakdown: Optional[Dict[str, Any]] = Field(None, description="Per-axis breakdown of the patient–study match score")
    # Evidence class — "guideline" / "landmark_trial" / "trial".
    # Populated by evidence_classifier; frontend renders a tag for
    # guidelines and landmarks to distinguish them from
    # patient-specific trial results.
    evidence_type: Optional[str] = Field(None, description='Evidence class: "guideline", "landmark_trial", or "trial"')


# ============================================
# QUERY STRUCTURE MODELS (Fast regex-based extraction)
# ============================================

class PatientContextInfo(BaseModel):
    """Extracted patient demographics from query"""
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    performance_status: Optional[str] = Field(None, description="ECOG or KPS")
    ethnicity: Optional[str] = Field(None, description="Patient ethnicity")
    smoking_status: Optional[str] = Field(None, description="Smoking status")
    comorbidities: List[str] = Field(default_factory=list, description="Comorbidities")
    raw_text: Optional[str] = Field(None, description="Raw text span for semantic search")


class CancerContextInfo(BaseModel):
    """Extracted cancer-specific information from query"""
    site: Optional[str] = Field(None, description="Cancer site (breast, lung, etc)")
    site_detail: Optional[str] = Field(None, description="Specific location (oral cavity, etc)")
    histology: Optional[str] = Field(None, description="Histology type (SCC, adenocarcinoma)")
    stage: Optional[str] = Field(None, description="Stage (I, II, III, IV)")
    tnm_t: Optional[str] = Field(None, description="T stage")
    tnm_n: Optional[str] = Field(None, description="N stage")
    tnm_m: Optional[str] = Field(None, description="M stage")
    grade: Optional[str] = Field(None, description="Tumor grade")
    receptor_status: Optional[str] = Field(None, description="Receptor status (ER+/PR+/HER2+)")
    # biomarkers and metastatic_sites_detected are populated by the
    # regex extractor (BIOMARKER_PATTERNS at query_structuring_service.py
    # line 621) and surfaced in to_dict(). Were previously absent from
    # this model, causing Pydantic to silently drop them when
    # constructing QueryMetadata — the API response then showed
    # biomarkers=[] even when the internal pipeline correctly extracted
    # them. The scorer / matcher consume them via the dataclass, not
    # this model, so the omission only affected debugging visibility,
    # not actual scoring behaviour.
    biomarkers: List[str] = Field(default_factory=list, description="Biomarker tags (HPV+, EGFR mutant, etc.)")
    metastatic_sites_detected: List[str] = Field(default_factory=list, description="Sites explicitly described as metastatic")
    doi: Optional[str] = Field(None, description="Depth of invasion")
    lvi: Optional[str] = Field(None, description="Lymphovascular invasion")
    pni: Optional[str] = Field(None, description="Perineural invasion")
    margins: Optional[str] = Field(None, description="Margin status")
    lymph_nodes: Optional[str] = Field(None, description="Lymph node involvement")
    raw_text: Optional[str] = Field(None, description="Raw text span for semantic search")


class TreatmentContextInfo(BaseModel):
    """Extracted treatment-related information from query"""
    modality: Optional[str] = Field(None, description="Treatment modality (RT, chemo, surgery)")
    setting: Optional[str] = Field(None, description="Treatment setting (adjuvant, neoadjuvant)")
    prior_treatments: List[str] = Field(default_factory=list, description="Prior treatments mentioned")
    raw_text: Optional[str] = Field(None, description="Raw text span for semantic search")


class ClinicalHistoryInfo(BaseModel):
    """Extracted clinical history and imaging findings from query"""
    imaging_findings: Optional[str] = Field(None, description="Imaging findings")
    recurrence_info: Optional[str] = Field(None, description="Recurrence information")
    raw_text: Optional[str] = Field(None, description="Raw text span for semantic search")


class QueryStructureInfo(BaseModel):
    """
    Structured breakdown of user query - extracted via fast regex patterns.
    For complex queries, LLM extraction runs in parallel with embedding.
    """
    query_type: str = Field(..., description="Classified query type")
    has_explicit_question: bool = Field(True, description="Whether query contains explicit question")
    has_patient_context: bool = Field(False, description="Whether patient-specific info was detected")
    question_focus: Optional[str] = Field(None, description="What's being asked (dose, survival, indication)")
    patient: Optional[PatientContextInfo] = Field(None, description="Extracted patient demographics")
    cancer: Optional[CancerContextInfo] = Field(None, description="Extracted cancer context")
    treatment: Optional[TreatmentContextInfo] = Field(None, description="Extracted treatment context")
    clinical_history: Optional[ClinicalHistoryInfo] = Field(None, description="Extracted clinical history")
    boost_terms: Optional[List[str]] = Field(None, description="Terms used to boost retrieval")
    filter_category: Optional[str] = Field(None, description="Category filter applied")
    used_llm_extraction: bool = Field(False, description="Whether LLM extraction was used for complex query")


class QueryMetadata(BaseModel):
    """Metadata about the query and retrieval"""
    query_type: str = Field(..., description="Classified query type")
    query_classification: Dict[str, Any] = Field(
        default={},
        description="Full classification details"
    )
    expanded_query: Optional[str] = Field(
        None,
        description="Query after expansion"
    )
    nccn_assessment: Optional[Dict[str, Any]] = Field(
        None,
        description="NCCN guideline gap assessment"
    )
    retrieval_route: Optional[Dict[str, Any]] = Field(
        None,
        description="Retrieval strategy used"
    )
    inferred_site: Optional[str] = Field(
        None,
        description="Inferred tumor site if site inference was used"
    )
    site_label: Optional[str] = Field(
        None,
        description="Human-readable site label"
    )
    query_structure: Optional[QueryStructureInfo] = Field(
        None,
        description="Structured breakdown of the query (patient context, cancer info, etc)"
    )
    query_confidence: Optional[float] = Field(
        None,
        description="Classification confidence from classify_query_hybrid() (0-1)"
    )

    class Config:
        # Allow extra fields to be passed through (for backwards compatibility)
        extra = "ignore"
    
    def __init__(self, **data):
        # Convert query_structure dict to QueryStructureInfo if needed
        if 'query_structure' in data and data['query_structure'] is not None:
            qs = data['query_structure']
            if isinstance(qs, dict):
                # Convert nested dicts to models
                if qs.get('patient') and isinstance(qs['patient'], dict):
                    qs['patient'] = PatientContextInfo(**qs['patient'])
                if qs.get('cancer') and isinstance(qs['cancer'], dict):
                    qs['cancer'] = CancerContextInfo(**qs['cancer'])
                if qs.get('treatment') and isinstance(qs['treatment'], dict):
                    qs['treatment'] = TreatmentContextInfo(**qs['treatment'])
                if qs.get('clinical_history') and isinstance(qs['clinical_history'], dict):
                    qs['clinical_history'] = ClinicalHistoryInfo(**qs['clinical_history'])
                data['query_structure'] = QueryStructureInfo(**qs)
        super().__init__(**data)


class ModuleClassification(BaseModel):
    """Module classification result for query routing"""
    module: str = Field(..., description="Classified module: general_knowledge, patient_specific, or evidence_exploration")
    confidence: float = Field(..., description="Classification confidence (0-1)")
    signals_matched: List[str] = Field(default=[], description="Signals that triggered this classification")
    has_patient_context: bool = Field(default=False, description="Whether patient context was detected")
    has_explicit_question: bool = Field(default=True, description="Whether an explicit question was detected")
    has_superlative: bool = Field(default=False, description="Whether superlative/comparative language was detected")
    suggested_follow_ups: List[str] = Field(default=[], description="Suggested follow-up questions for module transfer")


class StructuredSummary(BaseModel):
    """Key-finding summary card shown above the main justification."""
    key_finding: str = Field(..., description="1-2 sentence key clinical finding")
    evidence_level: str = Field(..., description="'High' | 'Moderate' | 'Low' | 'Insufficient'")
    recommendation_strength: str = Field(..., description="'Strong' | 'Conditional' | 'Expert Opinion'")
    caveats: Optional[str] = Field(None, description="Important caveats or limitations")


class GuidelineAlignment(BaseModel):
    """Guideline alignment status for treatment/indication responses."""
    guideline_body: str = Field(..., description="'NCCN' | 'ASTRO' | 'ESMO' | 'Multiple' | 'None'")
    alignment_status: str = Field(..., description="'Consistent' | 'Inconsistent' | 'Not addressed'")
    guideline_note: str = Field(..., description="Brief explanation of alignment or gap")


class StructuredResponseData(BaseModel):
    """Structured response data for layered display"""
    brief_answer: str = Field(default="", description="Brief 1-2 sentence answer")
    structured_details: Dict[str, Any] = Field(default_factory=dict, description="Format-specific structured details")
    structured_format: str = Field(default="simple_prose", description="Format type for rendering")
    explanation: str = Field(default="", description="Detailed explanation")
    query_type: str = Field(default="general", description="Detected query type")
    raw_response: str = Field(default="", description="Original LLM response")


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str = Field(..., description="Generated answer")
    retrieval_results: List[RetrievalResult] = Field(
        default=[],
        description="Evidence chunks used"
    )
    query_type: Optional[str] = Field(
        None,
        description="Classified query type"
    )
    module_classification: Optional[ModuleClassification] = Field(
        None,
        description="Module classification for response routing"
    )
    metadata: Optional[QueryMetadata] = Field(
        None,
        description="Query and retrieval metadata"
    )
    # Source tracking for conversation follow-ups
    sources: Optional[List[str]] = Field(
        default=None,
        description="Doc IDs of sources used (for conversation tracking)"
    )
    source_citations: Optional[List[str]] = Field(
        default=None,
        description="Citation strings of sources used (for conversation tracking)"
    )
    # Accumulated context for conversation continuity
    accumulated_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Accumulated structured context to send back in follow-up queries"
    )
    # Structured response for layered display
    structured_response: Optional[StructuredResponseData] = Field(
        default=None,
        description="Parsed structured response for layered UI rendering"
    )
    # Updated context entry for automatic conversation mode
    updated_context_entry: Optional[ConversationContextEntry] = Field(
        default=None,
        description="Updated context entry for frontend to store in sessionStorage"
    )
    # Answer quality metrics (confidence, contradictions, citations)
    answer_quality: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Answer quality metrics including confidence score, contradictions, and consensus findings"
    )
    # Web fallback flag — True when KB had no/low-relevance results and web sources were used
    web_fallback: bool = Field(
        default=False,
        description="True when response is sourced from PubMed/ClinicalTrials.gov due to low KB relevance"
    )
    # Web supplement flag — True when KB results were augmented with online evidence
    web_supplement: bool = Field(
        default=False,
        description="True when KB results were supplemented with PubMed/ClinicalTrials.gov evidence"
    )


class DeepDiveResponse(BaseModel):
    """Response model for deep dive queries"""
    query: str = Field(..., description="Full query with site context")
    site_key: str = Field(..., description="Tumor site key used")
    site_label: str = Field(..., description="Human-readable site label")
    summary: str = Field(..., description="Generated answer")
    evidence: List[RetrievalResult] = Field(
        default=[],
        description="Evidence chunks used"
    )
    metadata: Optional[QueryMetadata] = Field(
        None,
        description="Query and retrieval metadata"
    )


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    collection: Optional[str] = Field(None, description="Qdrant collection name")
    cross_encoder_available: Optional[bool] = Field(
        None,
        description="Whether cross-encoder reranking is available"
    )
    test_query_success: Optional[bool] = Field(
        None,
        description="Whether test query succeeded"
    )
    results_count: Optional[int] = Field(
        None,
        description="Number of results from test query"
    )


class QueryMode(BaseModel):
    """Available query mode"""
    id: str = Field(..., description="Mode ID")
    name: str = Field(..., description="Mode display name")
    description: str = Field(..., description="Mode description")


class QueryModesResponse(BaseModel):
    """Response with available query modes"""
    modes: List[QueryMode] = Field(..., description="Available query modes")


class SiteInfo(BaseModel):
    """Tumor site information"""
    key: str = Field(..., description="Site key")
    label: str = Field(..., description="Human-readable label")


class SitesResponse(BaseModel):
    """Response with available tumor sites"""
    sites: List[SiteInfo] = Field(..., description="Available tumor sites")


# ============================================
# PATIENT MATCHING MODELS
# ============================================

class PatientProfile(BaseModel):
    """Patient profile for matching to clinical studies."""

    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Gender (male, female)")
    cancer_stage: Optional[str] = Field(None, description="Cancer stage (I, II, III, IV)")
    cancer_type: Optional[str] = Field(None, description="Type of cancer")
    anatomical_site: Optional[str] = Field(None, description="Anatomical site/location of cancer (e.g., maxilla, oral cavity, lung, breast)")
    histology: Optional[str] = Field(None, description="Histology type (e.g., adenocarcinoma, squamous)")
    molecular_markers: Optional[List[str]] = Field(None, description="Molecular markers (e.g., EGFR+, PD-L1+)")
    performance_status: Optional[str] = Field(None, description="Performance status (ECOG 0, 1, 2, etc.)")
    comorbidities: Optional[List[str]] = Field(None, description="Comorbidities")
    smoking_status: Optional[str] = Field(None, description="Smoking status (never, former, current)")
    prior_treatments: Optional[List[str]] = Field(None, description="Prior treatments (e.g., osimertinib, chemotherapy, maxillectomy)")
    tnm_t: Optional[str] = Field(None, description="T stage (e.g., pT4, cT2)")
    tnm_n: Optional[str] = Field(None, description="N stage (e.g., pN0, cN1)")
    tnm_m: Optional[str] = Field(None, description="M stage (e.g., M0, M1)")
    recurrence_status: Optional[str] = Field(None, description="Primary, recurrent, nodal recurrence, etc.")
    treatment_setting: Optional[str] = Field(None, description="Adjuvant, neoadjuvant, definitive, salvage, palliative, surveillance")
    lvi: Optional[str] = Field(None, description="Lymphovascular invasion (positive, negative)")
    pni: Optional[str] = Field(None, description="Perineural invasion (positive, negative)")
    margins: Optional[str] = Field(None, description="Surgical margin status (positive, negative, close)")
    disease_descriptor: Optional[str] = Field(None, description="Clinical descriptor like locally advanced, metastatic (not a stage number)")
    criteria_weights: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom weights for match criteria boosting. "
                    "Core keys (active now): cancer_site, histology, stage, treatment, biomarkers, age, performance_status. "
                    "Stub keys (future): race, sex, comorbidities, recurrence_status, grade, tumor_size, treatment_setting. "
                    "Values: float multiplier where 1.0 = default, 2.0 = double importance, 0.0 = ignore. "
                    "Special key 'biomarker_mode' can be 'strict' or 'partial'."
    )


class UnstructuredPatientMatchRequest(BaseModel):
    """Request for patient matching from free-text description."""
    unstructured_description: str = Field(..., min_length=10, description="Free-text patient description")
    top_k: int = Field(15, ge=1, le=30, description="Number of studies to return")
    criteria_weights: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom weights for match criteria boosting. "
                    "Core keys (active now): cancer_site, histology, stage, treatment, biomarkers, age, performance_status. "
                    "Stub keys (future): race, sex, comorbidities, recurrence_status, grade, tumor_size, treatment_setting. "
                    "Values: float multiplier where 1.0 = default, 2.0 = double importance, 0.0 = ignore. "
                    "Special key 'biomarker_mode' can be 'strict' or 'partial'."
    )
    conversation_context: Optional[List[ConversationContextEntry]] = Field(
        default=None,
        description="Conversation context entries from frontend sessionStorage for context reuse"
    )
    cached_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Previously extracted patient profile to skip LLM extraction"
    )


class StudyMatch(BaseModel):
    """Individual study match result."""
    
    title: str = Field(..., description="Study title")
    author: Optional[str] = Field(None, description="Author citation")
    citation: Optional[str] = Field(None, description="Full citation")
    doi: Optional[str] = Field(None, description="DOI")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    doc_id: Optional[str] = Field(None, description="Internal document ID")
    year: Optional[int] = Field(None, description="Publication year")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Match score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    match_reasons: List[str] = Field(default_factory=list, description="Why this study matches")
    match_rationale: Optional[str] = Field(None, description="Plain-language 1-2 sentence rationale for clinicians")
    match_strength: Optional[str] = Field(None, description="strong | partial | possible")
    relevant_text: str = Field("", description="Relevant excerpt from study")
    treatment: Optional[str] = Field(None, description="Treatment information")
    key_info: Optional[str] = Field(None, description="Key findings from the study")
    demographics: Optional[List[str]] = Field(default_factory=list, description="Matched demographics")
    cancer_characteristics: Optional[List[str]] = Field(default_factory=list, description="Matched cancer characteristics")
    key_matches: Optional[List[str]] = Field(default_factory=list, description="Matched molecular markers")
    source_type: Optional[str] = Field(None, description="Source: 'kb', 'pubmed', 'clinicaltrials'")
    # patient_match_score is the structured scorer's output (0–100)
    # mirroring what the chat pipeline puts on each citation. Computed
    # in match_patient_comprehensive via patient_match_scorer, which
    # uses NA-axis-aware weighted overlap of patient ClinicalProfile
    # axes against the study's doc_level_* metadata. Independent of
    # `match_score` (which is the cross-encoder normalized 0–1) — the
    # two scores answer different questions: relevance vs. population
    # fit. The frontend renders the same Strong / Moderate / Weak /
    # Limited bucketing it uses in chat.
    patient_match_score: Optional[int] = Field(None, ge=0, le=100, description="Patient–study population match (0–100); axis-overlap of patient profile against study's doc_level metadata")
    patient_match_breakdown: Optional[Dict[str, Any]] = Field(None, description="Per-axis breakdown of the patient–study match score")


class PatientMatchResponse(BaseModel):
    """Response for patient matching."""
    
    matches: List[StudyMatch] = Field(default_factory=list, description="Matching studies")
    total_matches: int = Field(0, description="Total number of matches found")
    patient_summary: str = Field("", description="Summary of patient characteristics")
    extracted_profile: Optional[Dict[str, Any]] = Field(None, description="Structured profile extracted from unstructured input")


# ============================================
# TREATMENT COMPARISON MODELS
# ============================================

class TreatmentComparisonRequest(BaseModel):
    """Request for treatment comparison."""
    
    treatment_a: str = Field(..., description="First treatment name")
    treatment_b: str = Field(..., description="Second treatment name")
    cancer_type: Optional[str] = Field(None, description="Cancer type (optional)")
    stage: Optional[str] = Field(None, description="Cancer stage (optional)")
    top_k: int = Field(10, ge=1, le=30, description="Number of studies to retrieve")


class TreatmentEvidence(BaseModel):
    """Evidence for a single treatment."""
    
    efficacy: Optional[str] = Field(None, description="Efficacy summary")
    safety: Optional[str] = Field(None, description="Safety profile")
    dosing: Optional[str] = Field(None, description="Dosing information")
    outcomes: Optional[str] = Field(None, description="Outcome metrics")
    studies: List[str] = Field(default_factory=list, description="Supporting studies")


class TreatmentComparisonResult(BaseModel):
    """Comparison result for two treatments."""
    
    treatment_a_name: str
    treatment_b_name: str
    treatment_a_evidence: TreatmentEvidence
    treatment_b_evidence: TreatmentEvidence
    comparison_summary: str = Field("", description="Summary of key differences")
    statistical_significance: Optional[bool] = Field(None, description="Is comparison statistically significant")
    recommendation: Optional[str] = Field(None, description="Treatment recommendation if any")


class TreatmentComparisonResponse(BaseModel):
    """Response for treatment comparison."""
    
    comparison: TreatmentComparisonResult
    sources: List[RetrievalResult] = Field(default_factory=list, description="Source documents")
    metadata: QueryMetadata


# ============================================
# ENHANCED QUERY WITH PTO MODELS
# ============================================

class PTOOutcome(BaseModel):
    """Outcome from a PTO frame."""
    metric: str = Field(..., description="Outcome metric name (e.g., 'OS', 'PFS', 'local_control')")
    value: str = Field(..., description="Outcome value")


class PTOFrame(BaseModel):
    """Patient-Treatment-Outcome frame."""
    pto_id: Optional[str] = Field(None, description="PTO frame ID")
    cancer_type: Optional[str] = Field(None, description="Cancer type")
    stage: Optional[str] = Field(None, description="Cancer stage")
    tnm: Optional[str] = Field(None, description="TNM staging")
    biomarkers: List[str] = Field(default_factory=list, description="Biomarkers")
    treatment_modalities: List[str] = Field(default_factory=list, description="Treatment modalities")
    dose_fractionation: Optional[str] = Field(None, description="Dose/fractionation")
    chemo_agents: List[str] = Field(default_factory=list, description="Chemotherapy agents")
    outcomes: Dict[str, Any] = Field(default_factory=dict, description="Outcomes")
    confidence: Optional[str] = Field(None, description="Confidence level")
    citation: Optional[str] = Field(None, description="Source citation")
    score: Optional[float] = Field(None, description="Relevance score")


# ============================================
# ARTIFACT MODELS FOR INLINE VISUALIZATIONS
# ============================================

class ChartDataset(BaseModel):
    """Dataset for a chart."""
    label: str = Field(..., description="Dataset label")
    data: List[float] = Field(..., description="Data values")
    backgroundColor: Optional[List[str]] = Field(None, description="Background colors")
    borderColor: Optional[List[str]] = Field(None, description="Border colors")


class ChartArtifact(BaseModel):
    """Chart artifact for inline visualization."""
    type: str = Field(..., description="Chart type: bar, line, pie, doughnut")
    title: str = Field(..., description="Chart title")
    labels: List[str] = Field(..., description="X-axis labels or pie segment labels")
    datasets: List[ChartDataset] = Field(..., description="Chart datasets")
    unit: Optional[str] = Field(None, description="Unit for values (e.g., '%', 'months')")
    source: Optional[str] = Field(None, description="Data source citation")


class Artifact(BaseModel):
    """Artifact container for visualizations."""
    artifact_type: str = Field(..., description="Type: chart, table, timeline")
    chart: Optional[ChartArtifact] = Field(None, description="Chart configuration if artifact_type is 'chart'")


class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with short answer and detailed justification."""
    
    # Short answer from PTO frames
    short_answer: str = Field(..., description="Concise answer from PTO frames")
    
    # Detailed justification from RAG
    justification: str = Field(..., description="Detailed justification from RAG pipeline")
    
    # PTO frames used for short answer
    pto_frames: List[PTOFrame] = Field(default_factory=list, description="PTO frames used")
    
    # Evidence from RAG pipeline
    retrieval_results: List[RetrievalResult] = Field(default_factory=list, description="Evidence chunks")
    
    # Routing information
    used_pto: bool = Field(..., description="Whether PTO frames were used")
    query_type: Optional[str] = Field(None, description="Classified query type")
    
    # Suggested follow-up questions for conversation mode
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    
    # Module classification info (for frontend follow-ups and display)
    module_classification: Optional[ModuleClassification] = Field(None, description="Module classification result")
    
    # Artifact for inline visualizations
    artifact: Optional[Artifact] = Field(None, description="Visualization artifact if applicable")
    
    # Metadata
    metadata: Optional[QueryMetadata] = Field(None, description="Query metadata")
    
    # Source tracking for conversation follow-ups
    sources: Optional[List[str]] = Field(
        default=None,
        description="Doc IDs of sources used (for conversation tracking)"
    )
    source_citations: Optional[List[str]] = Field(
        default=None,
        description="Citation strings of sources used (for conversation tracking)"
    )
    
    # Structured response for layered display
    structured_response: Optional[StructuredResponseData] = Field(
        default=None,
        description="Parsed structured response for layered UI rendering"
    )
    
    # Updated context entry for automatic conversation mode
    updated_context_entry: Optional[ConversationContextEntry] = Field(
        default=None,
        description="Updated context entry for frontend to store in sessionStorage"
    )
    
    # Unified routing information
    routing: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Unified routing result with module, query_type, and format hints"
    )
    
    # Dynamic follow-up suggestions from Follow_Up_Generator
    follow_up_suggestions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Dynamic follow-up suggestions with type and text"
    )
    
    # Document IDs and titles for conversation tracking
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Document IDs from retrieval results"
    )
    doc_titles: Optional[List[str]] = Field(
        default=None,
        description="Document titles from retrieval results"
    )

    # Key-finding summary card (Task 2)
    structured_summary: Optional[StructuredSummary] = Field(
        default=None,
        description="Key-finding summary card shown above the main justification"
    )

    # Guideline alignment section (Task 5)
    guideline_alignment: Optional[GuidelineAlignment] = Field(
        default=None,
        description="Guideline alignment status for treatment/indication responses"
    )

    # Web fallback flag — True when KB had no/low-relevance results and web sources were used
    web_fallback: bool = Field(
        default=False,
        description="True when response is sourced from PubMed/ClinicalTrials.gov due to low KB relevance"
    )
    # Web supplement flag — True when KB results were augmented with online evidence
    web_supplement: bool = Field(
        default=False,
        description="True when KB results were supplemented with PubMed/ClinicalTrials.gov evidence"
    )


# ============================================
# STUDY-SPECIFIC Q&A MODELS
# ============================================

class StudyQueryRequest(BaseModel):
    """Request for study-specific Q&A."""
    question: str = Field(..., description="Question about the study", min_length=1)
    study_id: Optional[str] = Field(None, description="Study doc_id")
    study_doi: Optional[str] = Field(None, description="Study DOI")
    study_pmid: Optional[str] = Field(None, description="Study PMID")
    study_title: Optional[str] = Field(None, description="Study title for fallback search")
    conversation_history: List[Message] = Field(
        default=[],
        description="Previous conversation messages about this study"
    )


class StudyQueryResponse(BaseModel):
    """Response for study-specific Q&A."""
    answer: str = Field(..., description="Answer to the question")
    study_title: Optional[str] = Field(None, description="Study title")
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")


class StudyExampleQuestionsRequest(BaseModel):
    """Request to get example questions grounded in a study's available content."""
    study_id: Optional[str] = Field(None, description="Study doc_id")
    study_doi: Optional[str] = Field(None, description="Study DOI")
    study_pmid: Optional[str] = Field(None, description="Study PMID")
    study_title: Optional[str] = Field(None, description="Study title for fallback search")


class StudyExampleQuestionsResponse(BaseModel):
    """Response with example questions that can be answered from the study content."""
    questions: List[str] = Field(default_factory=list, description="Example questions answerable from the study")


# ============================================
# VISUAL TREATMENT COMPARISON MODELS
# ============================================

class VisualComparisonRequest(BaseModel):
    """Request for visual treatment comparison."""
    query: str = Field(..., description="Natural language comparison query", min_length=10)
    top_k: int = Field(default=15, ge=5, le=30, description="Number of evidence chunks to retrieve")


class TreatmentArmResult(BaseModel):
    """Results for a single treatment arm in a comparison."""
    arm_label: str = Field(..., description="Treatment arm name (e.g., 'Pembrolizumab', 'Chemotherapy')")
    arm_query: str = Field(..., description="The sub-query used to retrieve evidence for this arm")
    retrieval_results: List[RetrievalResult] = Field(default_factory=list, description="Evidence chunks for this arm")
    study_profiles: List[Dict[str, Any]] = Field(default_factory=list, description="Structured profiles for this arm")



class VisualComparisonResponse(BaseModel):
    """Response for visual treatment comparison with charts."""
    summary: str = Field(..., description="Short summary of the comparison")
    detailed_analysis: str = Field(..., description="Detailed analysis with citations")
    charts: List[Artifact] = Field(default_factory=list, description="Chart artifacts for visualization")
    retrieval_results: List[RetrievalResult] = Field(default_factory=list, description="Supporting evidence (all arms combined)")
    study_profiles: List[Dict[str, Any]] = Field(default_factory=list, description="Structured study profiles from PostgreSQL")
    treatment_arms: List[TreatmentArmResult] = Field(default_factory=list, description="Per-arm grouped results")
    query_type: Optional[str] = Field(None, description="Classified query type")


# ============================================
# MULTI-STUDY COMPARISON MODELS
# ============================================

class StudyComparisonRequest(BaseModel):
    """Request for comparing multiple studies."""
    study_ids: List[str] = Field(..., description="List of study doc_ids to compare (2-4 studies)", min_length=2, max_length=4)


class StudyComparisonCategory(BaseModel):
    """Comparison data for a single category."""
    category: str = Field(..., description="Category name (e.g., 'outcomes', 'toxicity')")
    title: str = Field(..., description="Display title for the category")
    charts: List[Artifact] = Field(default_factory=list, description="Charts for this category")
    summary: Optional[str] = Field(None, description="Text summary for this category")
    data_available: bool = Field(default=True, description="Whether data was available for this category")


class StudySummary(BaseModel):
    """Summary of a single study for comparison."""
    doc_id: str = Field(..., description="Study document ID")
    title: Optional[str] = Field(None, description="Study title")
    doi: Optional[str] = Field(None, description="DOI")
    year: Optional[str] = Field(None, description="Publication year")
    cancer_type: Optional[str] = Field(None, description="Cancer type")
    number_of_patients: Optional[int] = Field(None, description="Number of patients")
    study_type: Optional[str] = Field(None, description="Study type (RCT, Phase III, etc.)")
    primary_endpoint: Optional[str] = Field(None, description="Primary endpoint")


class StudyComparisonResponse(BaseModel):
    """Response for multi-study comparison with visualizations."""
    studies: List[StudySummary] = Field(..., description="Summary of compared studies")
    categories: List[StudyComparisonCategory] = Field(..., description="Comparison by category")
    narrative: str = Field(..., description="AI-generated comparison narrative")
    generated_at: str = Field(..., description="Timestamp of generation")


# ============================================
# INTENT ANALYSIS MODELS
# ============================================

class QueryIntentInfo(BaseModel):
    """Detected intent from user query"""
    intent_type: str = Field(..., description="Type: explicit_question, patient_description, treatment_inquiry, comparison_request, unclear")
    has_explicit_question: bool = Field(..., description="Whether query contains an explicit question")
    confidence: float = Field(..., description="Confidence score 0-1")
    detected_question_type: Optional[str] = Field(None, description="Type of question if detected: outcome, treatment, dose, comparison, etc.")


class ExtractedPatientProfile(BaseModel):
    """Patient profile extracted from query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    smoking_status: Optional[str] = None
    cancer_type: Optional[str] = None
    cancer_location: Optional[str] = None
    histology: Optional[str] = None
    stage: Optional[str] = None
    tnm_t: Optional[str] = None
    tnm_n: Optional[str] = None
    tnm_m: Optional[str] = None
    tumor_size: Optional[str] = None
    doi: Optional[str] = None
    lvi: Optional[str] = None
    pni: Optional[str] = None
    margins: Optional[str] = None
    lymph_nodes: Optional[str] = None
    other_pathology: List[str] = Field(default_factory=list)
    molecular_markers: List[str] = Field(default_factory=list)
    prior_treatment: List[str] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    performance_status: Optional[str] = None
    recurrence_status: Optional[str] = None
    treatment_setting: Optional[str] = None


class FollowUpOptionInfo(BaseModel):
    """A suggested follow-up action"""
    action_type: str = Field(..., description="Action type: find_studies, treatment_options, outcomes, comparison, guidelines, dose, toxicity")
    label: str = Field(..., description="Display label with emoji")
    description: str = Field(..., description="Longer description of what this action does")
    query_template: str = Field(..., description="Pre-filled query to execute if selected")


class IntentAnalysisRequest(BaseModel):
    """Request for intent analysis"""
    query: str = Field(..., description="User's input text", min_length=1)
    force_trial_match: bool = Field(False, description="When True, always extract patient profile and find matching trials (Trial Match mode)")


class MatchingTrialInfo(BaseModel):
    """A matching clinical trial/study"""
    title: str = Field(..., description="Study title")
    author: Optional[str] = Field(None, description="Author citation")
    year: Optional[int] = Field(None, description="Publication year")
    match_score: float = Field(0.0, description="Match score 0-1")
    match_reasons: List[str] = Field(default_factory=list, description="Why this study matches")
    relevant_excerpt: str = Field("", description="Relevant excerpt from study")
    doi: Optional[str] = Field(None, description="DOI")
    treatment: Optional[str] = Field(None, description="Treatment information")
    inclusion_criteria: Optional[str] = Field(None, description="Key inclusion criteria")
    exclusion_criteria: Optional[str] = Field(None, description="Key exclusion criteria")
    eligibility_notes: List[str] = Field(default_factory=list, description="Notes about patient eligibility")
    population_details: Optional[str] = Field(None, description="Demographics, cancer type, stage, histology, biomarkers from trial")


class IntentAnalysisResponse(BaseModel):
    """Response from intent analysis"""
    intent: QueryIntentInfo = Field(..., description="Detected intent")
    patient_profile: Optional[ExtractedPatientProfile] = Field(None, description="Extracted patient profile if applicable")
    patient_summary: str = Field(..., description="Human-readable patient summary")
    follow_up_options: List[FollowUpOptionInfo] = Field(..., description="Suggested follow-up actions")
    should_prompt_user: bool = Field(..., description="Whether to show follow-up options to user")
    auto_action: Optional[str] = Field(None, description="Suggested automatic action if intent is clear")
    message: str = Field(..., description="Message to display to user")
    formatted_response: str = Field(default="", description="Formatted conversational response with structured patient summary and follow-up questions")
    matching_trials: List[MatchingTrialInfo] = Field(default_factory=list, description="Matching clinical trials/studies")


# ============================================
# STANDALONE CLASSIFICATION MODELS
# ============================================

class ClassifyRequest(BaseModel):
    """Request for standalone query classification."""
    query: str = Field(..., description="Query to classify", min_length=1)
    include_format_hints: bool = Field(True, description="Whether to include format hints in response")


class ClassifyResponse(BaseModel):
    """Response from standalone query classification."""
    success: bool = Field(..., description="Whether classification succeeded")
    routing: Dict[str, Any] = Field(..., description="Routing result with module, confidence, query_type, format_hints, signals_matched")
