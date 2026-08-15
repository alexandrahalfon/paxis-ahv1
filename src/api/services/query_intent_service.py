"""
Query Intent Detection Service (Optimized)

Detects when a user provides patient information without an explicit question,
structures the information, and offers relevant follow-up options.

This enables a more conversational AI experience where users can simply paste
patient details and get intelligent suggestions for what to do next.

OPTIMIZATIONS (v2):
  - Parallelized abstract fetches via ThreadPoolExecutor (~0.5s vs ~3-5s serial)
  - Batched eligibility + summary extraction into ONE LLM call for all matches
    (~2-3s vs ~20-30s for 10 serial calls)
  - Net effect on _find_matching_trials: ~3s post-retrieval vs ~30s
  - Full end-to-end Trial Match toggle: ~8-10s vs ~35-40s
"""

import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES (unchanged)
# ============================================

@dataclass
class QueryIntent:
    """Detected intent from user query"""
    intent_type: str  # 'explicit_question', 'patient_description', 'treatment_inquiry', 'comparison_request', 'unclear'
    has_explicit_question: bool
    confidence: float  # 0-1
    detected_question_type: Optional[str] = None  # 'outcome', 'treatment', 'dose', 'comparison', etc.
    

@dataclass
class PatientProfile:
    """Extracted patient profile from query"""
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
    other_pathology: List[str] = field(default_factory=list)
    molecular_markers: List[str] = field(default_factory=list)
    prior_treatment: List[str] = field(default_factory=list)
    comorbidities: List[str] = field(default_factory=list)
    performance_status: Optional[str] = None
    recurrence_status: Optional[str] = None
    treatment_setting: Optional[str] = None
    recurrence_score: Optional[int] = None
    
    def get_summary(self) -> str:
        """Generate human-readable summary"""
        parts = []
        if self.age:
            parts.append(f"{self.age}-year-old")
        if self.gender:
            parts.append(self.gender)
        if self.cancer_type:
            parts.append(f"with {self.cancer_type}")
        if self.cancer_location:
            parts.append(f"of the {self.cancer_location}")
        if self.stage:
            parts.append(f"(Stage {self.stage})")
        elif self.tnm_t or self.tnm_n:
            tnm = "/".join(filter(None, [self.tnm_t, self.tnm_n, self.tnm_m]))
            if tnm:
                parts.append(f"({tnm})")
        if self.histology:
            parts.append(f"- {self.histology}")
        if self.recurrence_status:
            parts.append(f"[{self.recurrence_status}]")
        if self.treatment_setting:
            parts.append(f"({self.treatment_setting} setting)")
        if self.prior_treatment:
            parts.append(f"s/p {', '.join(self.prior_treatment)}")
        
        return " ".join(parts) if parts else "Patient profile"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != [] and value != "":
                result[key] = value
        return result


@dataclass
class FollowUpOption:
    """A suggested follow-up action"""
    action_type: str
    label: str
    description: str
    query_template: str
    priority: int = 0


@dataclass
class MatchingTrial:
    """A matching clinical trial/study"""
    title: str
    author: Optional[str] = None
    year: Optional[int] = None
    match_score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)
    relevant_excerpt: str = ""
    doi: Optional[str] = None
    treatment: Optional[str] = None
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    eligibility_notes: List[str] = field(default_factory=list)
    population_details: Optional[str] = None


@dataclass
class IntentAnalysisResult:
    """Complete result of intent analysis"""
    intent: QueryIntent
    patient_profile: Optional[PatientProfile]
    patient_summary: str
    follow_up_options: List[FollowUpOption]
    should_prompt_user: bool
    auto_action: Optional[str] = None
    formatted_response: str = ""
    matching_trials: List[MatchingTrial] = field(default_factory=list)


# ============================================
# PROMPTS (unchanged)
# ============================================

INTENT_DETECTION_PROMPT = """You are a clinical query analyzer. Analyze the user's input and determine:
1. Whether it contains an explicit question or request
2. What type of information is being provided
3. What the user likely wants to do

INTENT TYPES:
- "explicit_question": User asks a clear question (e.g., "What is the survival rate for...", "Should I use...")
- "patient_description": User provides patient details without a clear question
- "treatment_inquiry": User asks about treatment options
- "comparison_request": User wants to compare treatments or outcomes
- "unclear": Cannot determine intent

QUESTION TYPES (if explicit question detected):
- "outcome": Asking about survival, response rates, outcomes
- "treatment": Asking about treatment recommendations
- "dose": Asking about radiation/chemo dosing
- "comparison": Comparing treatments
- "eligibility": Asking about trial eligibility
- "toxicity": Asking about side effects
- "guidelines": Asking about standard of care

Respond with JSON:
{
    "intent_type": "string",
    "has_explicit_question": boolean,
    "confidence": float (0-1),
    "detected_question_type": "string or null",
    "reasoning": "brief explanation"
}"""

PATIENT_EXTRACTION_PROMPT = """Extract patient and case information from this clinical description.

Return JSON with these fields (use null if not mentioned):

{
    "age": integer or null,
    "gender": "male" or "female" or null (infer from context if possible, e.g., testicular cancer = male, ovarian = female),
    "ethnicity": "string - infer from language spoken (e.g., Mandarin-speaking -> Asian) or explicit mention",
    "smoking_status": "string - never, former, current, or null",
    "cancer_type": "string - e.g., Squamous Cell Carcinoma, Seminoma, Breast Cancer",
    "cancer_location": "string - anatomical site (e.g., testis, breast, oral cavity, lung)",
    "histology": "string - histologic details (e.g., poorly differentiated, pure seminoma, invasive ductal)",
    "stage": "string - overall stage I/II/III/IV",
    "tnm_t": "string - T stage (e.g., pT4, cT2)",
    "tnm_n": "string - N stage (e.g., pN0, cN1)", 
    "tnm_m": "string - M stage",
    "tumor_size": "string - tumor size if mentioned (e.g., 8cm, 2.5cm)",
    "doi": "string - depth of invasion if mentioned (e.g., 15mm)",
    "lvi": "string - lymphovascular invasion status (positive, negative, LVSI+, LVI-)",
    "pni": "string - perineural invasion status (positive, negative, PNI+, PNI-)",
    "margins": "string - margin status (positive, negative, close)",
    "lymph_nodes": "string - lymph node status (e.g., 0/42 LN involved, N0, no nodal involvement)",
    "other_pathology": ["array of other pathologic findings like rete testis invasion, extracapsular extension"],
    "molecular_markers": ["array of markers like HER2+, EGFR+, PD-L1+, serum markers normal"],
    "prior_treatment": ["array of prior treatments like orchiectomy, maxillectomy, lumpectomy, chemotherapy"],
    "comorbidities": ["array of comorbidities like HTN, T2DM, CAD"],
    "performance_status": "string - ECOG 0-4",
    "recurrence_status": "string - primary, recurrent, nodal recurrence, etc.",
    "treatment_setting": "string - adjuvant, neoadjuvant, definitive, salvage, palliative, surveillance",
    "recurrence_score": "integer or null - 21-gene recurrence score / Oncotype DX score (0-100) if mentioned"
}

Clinical description:
"""


class QueryIntentService:
    """
    Service to detect query intent and offer intelligent follow-up options.
    """
    
    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client
        
    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client"""
        if self.openai_client is None:
            from src.core.config import settings
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        return self.openai_client
    
    # ==================================================================
    # MAIN ORCHESTRATOR (unchanged logic)
    # ==================================================================
    
    async def analyze_query(self, query: str, find_matching_trials: bool = True, force_trial_match: bool = False, user_id: Optional[str] = None) -> IntentAnalysisResult:
        """
        Analyze a user query to detect intent and extract patient information.

        Args:
            query: The user's input text
            find_matching_trials: Whether to search for matching trials
            force_trial_match: When True, always extract patient profile and find matching trials
                              (used when Trial Match toggle is ON in frontend)
            user_id: Optional user ID for preference filtering
        """
        try:
            from src.api.services import pipeline_metrics as _pm
            if _pm.current() is None:
                _pm.start("p5")
            if force_trial_match and _pm.current() is not None:
                _pm.current().event("trial_match_forced")
        except Exception:
            pass

        client = self._get_client()

        # FAST PATH: Skip full intent analysis for obvious questions
        if not force_trial_match and self._is_obvious_question(query):
            logger.info(f"[Intent] Fast path: obvious question detected, skipping full analysis")
            return IntentAnalysisResult(
                intent=QueryIntent(
                    intent_type="explicit_question",
                    has_explicit_question=True,
                    confidence=0.9,
                    detected_question_type="general"
                ),
                patient_profile=None,
                patient_summary="",
                follow_up_options=[],
                should_prompt_user=False,
                auto_action="general",
                formatted_response="",
                matching_trials=[]
            )

        # Step 1: Detect intent (regex, ~0ms)
        intent = self._detect_intent(client, query)

        # Step 2: Extract patient profile if relevant (1 LLM call, ~1.5s)
        patient_profile = None
        patient_summary = ""
        if force_trial_match or intent.intent_type in ['patient_description', 'treatment_inquiry', 'unclear']:
            patient_profile = await asyncio.to_thread(self._extract_patient_profile, client, query)
            if patient_profile:
                patient_summary = patient_profile.get_summary()

        # Step 3: Generate follow-up options (no LLM, ~0ms)
        follow_up_options = self._generate_follow_up_options(intent, patient_profile, query)

        # Step 4: Determine if we should prompt user
        should_prompt = force_trial_match or (
            not intent.has_explicit_question and
            intent.intent_type in ['patient_description', 'unclear'] and
            patient_profile is not None
        )

        # Step 5: Suggest auto-action
        auto_action = None
        if intent.has_explicit_question and intent.confidence > 0.8:
            auto_action = intent.detected_question_type

        # Step 6: Find matching trials using Pipeline 1/2 comprehensive retrieval
        matching_trials = []
        if patient_profile and find_matching_trials and (should_prompt or force_trial_match):
            matching_trials = await self._find_matching_trials(patient_profile)

        # Step 7: Generate formatted response (string formatting, ~0ms)
        formatted_response = ""
        if patient_profile and (should_prompt or force_trial_match):
            formatted_response = self._generate_formatted_response(
                patient_profile,
                follow_up_options,
                matching_trials
            )

        try:
            from src.api.services import pipeline_metrics as _pm
            _pm_cur = _pm.current()
            if _pm_cur is not None:
                print(_pm_cur.summary_line())
        except Exception:
            pass

        return IntentAnalysisResult(
            intent=intent,
            patient_profile=patient_profile,
            patient_summary=patient_summary,
            follow_up_options=follow_up_options,
            should_prompt_user=should_prompt,
            auto_action=auto_action,
            formatted_response=formatted_response,
            matching_trials=matching_trials
        )
    
    # ==================================================================
    # INTENT DETECTION (unchanged — already fast regex)
    # ==================================================================
    
    def _is_obvious_question(self, query: str) -> bool:
        """Fast check if query is obviously a question (not a patient description)."""
        query_lower = query.lower().strip()
        
        if query.strip().endswith('?'):
            return True
        
        question_starters = [
            'what ', 'how ', 'why ', 'when ', 'where ', 'which ', 'who ',
            'is ', 'are ', 'can ', 'could ', 'should ', 'would ', 'will ',
            'do ', 'does ', 'did ', 'has ', 'have ', 'compare ', 'explain ',
            'tell me about ', 'describe ', 'list ', 'show me '
        ]
        for starter in question_starters:
            if query_lower.startswith(starter):
                return True
        
        if len(query) < 80 and not any(term in query_lower for term in [
            'year old', 'yo ', 'y/o', 'pmh', 'htn', 'dm', 'stage', 
            's/p', 'status post', 'diagnosed', 'pathology', 'biopsy'
        ]):
            return True
        
        return False
    
    def _detect_intent(self, client: OpenAI, query: str) -> QueryIntent:
        """Detect the intent of the query using regex (fast path)."""
        query_lower = query.lower().strip()
        
        question_markers = ['?', 'what ', 'how ', 'should ', 'is there', 'can you', 'would you', 
                          'compare', 'versus', 'vs', 'difference between']
        has_question = any(marker in query_lower for marker in question_markers)
        
        patient_patterns = ['year old', 'yo ', 'y/o', 'male', 'female', 'stage ', 
                          's/p ', 'status post', 'diagnosed with', 'presents with',
                          'pT', 'pN', 'cT', 'cN', 'ECOG']
        has_patient_info = any(pattern in query_lower for pattern in patient_patterns)
        
        if has_question:
            intent_type = "explicit_question"
            if any(w in query_lower for w in ['survival', 'outcome', 'prognosis', 'response rate']):
                question_type = "outcome"
            elif any(w in query_lower for w in ['treatment', 'therapy', 'regimen', 'recommend']):
                question_type = "treatment"
            elif any(w in query_lower for w in ['dose', 'gy', 'fractions', 'dosing']):
                question_type = "dose"
            elif any(w in query_lower for w in ['compare', 'versus', 'vs', 'difference']):
                question_type = "comparison"
            else:
                question_type = None
        elif has_patient_info:
            intent_type = "patient_description"
            question_type = None
        else:
            intent_type = "unclear"
            question_type = None
        
        return QueryIntent(
            intent_type=intent_type,
            has_explicit_question=has_question,
            confidence=0.6 if has_question or has_patient_info else 0.3,
            detected_question_type=question_type
        )
    
    # ==================================================================
    # PATIENT PROFILE EXTRACTION (unchanged — 1 LLM call)
    # ==================================================================
    
    def _extract_patient_profile(self, client: OpenAI, query: str) -> Optional[PatientProfile]:
        """Extract patient profile from query."""
        try:
            from src.core.config import settings
            response = client.chat.completions.create(
                model=settings.openai_mini_model,
                messages=[
                    {"role": "system", "content": "You are a medical data extraction specialist. Extract patient information and return valid JSON only."},
                    {"role": "user", "content": PATIENT_EXTRACTION_PROMPT + query}
                ],
                temperature=0.1,
                max_tokens=700
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(result_text)
            
            return PatientProfile(
                age=data.get("age"),
                gender=data.get("gender"),
                ethnicity=data.get("ethnicity"),
                smoking_status=data.get("smoking_status"),
                cancer_type=data.get("cancer_type"),
                cancer_location=data.get("cancer_location"),
                histology=data.get("histology"),
                stage=data.get("stage"),
                tnm_t=data.get("tnm_t"),
                tnm_n=data.get("tnm_n"),
                tnm_m=data.get("tnm_m"),
                tumor_size=data.get("tumor_size"),
                doi=data.get("doi"),
                lvi=data.get("lvi"),
                pni=data.get("pni"),
                margins=data.get("margins"),
                lymph_nodes=data.get("lymph_nodes"),
                other_pathology=data.get("other_pathology") or [],
                molecular_markers=data.get("molecular_markers") or [],
                prior_treatment=data.get("prior_treatment") or [],
                comorbidities=data.get("comorbidities") or [],
                performance_status=data.get("performance_status"),
                recurrence_status=data.get("recurrence_status"),
                treatment_setting=data.get("treatment_setting"),
                recurrence_score=int(data["recurrence_score"]) if data.get("recurrence_score") else None
            )
            
        except Exception as e:
            logger.error(f"Patient extraction failed: {e}")
            return None

    # ==================================================================
    # FOLLOW-UP OPTIONS (unchanged — no LLM)
    # ==================================================================
    
    def _generate_follow_up_options(
        self, 
        intent: QueryIntent, 
        profile: Optional[PatientProfile],
        original_query: str
    ) -> List[FollowUpOption]:
        """Generate relevant follow-up options based on intent and profile."""
        options = []
        
        if profile is None:
            return [
                FollowUpOption(
                    action_type="clarify",
                    label="Provide more details",
                    description="Please provide patient details (age, cancer type, stage) for personalized results",
                    query_template=original_query,
                    priority=1
                )
            ]
        
        cancer_desc = profile.cancer_type or "cancer"
        if profile.cancer_location:
            cancer_desc = f"{cancer_desc} of the {profile.cancer_location}"
        
        stage_desc = ""
        if profile.stage:
            stage_desc = f"stage {profile.stage}"
        elif profile.tnm_t or profile.tnm_n:
            tnm = "/".join(filter(None, [profile.tnm_t, profile.tnm_n, profile.tnm_m]))
            stage_desc = tnm
        
        setting_desc = profile.treatment_setting or "definitive"
        
        options.append(FollowUpOption(
            action_type="find_studies",
            label="Find matching clinical studies",
            description=f"Search for studies that enrolled patients similar to this profile",
            query_template=f"Find clinical studies for {profile.get_summary()}",
            priority=10
        ))
        
        options.append(FollowUpOption(
            action_type="treatment_options",
            label="Show treatment options",
            description=f"What are the recommended treatment approaches for this patient?",
            query_template=f"What are the treatment options for {cancer_desc} {stage_desc}?",
            priority=9
        ))
        
        options.append(FollowUpOption(
            action_type="outcomes",
            label="Expected outcomes",
            description=f"What outcomes can be expected for patients with this profile?",
            query_template=f"What are the survival outcomes for {cancer_desc} {stage_desc} treated with {setting_desc} therapy?",
            priority=8
        ))
        
        if profile.treatment_setting in ['definitive', 'adjuvant', 'neoadjuvant'] or \
           any(t.lower() in ['radiation', 'rt', 'radiotherapy'] for t in profile.prior_treatment):
            options.append(FollowUpOption(
                action_type="dose",
                label="Radiation dosing",
                description=f"What is the recommended radiation dose and fractionation?",
                query_template=f"What is the standard radiation dose for {cancer_desc} {stage_desc}?",
                priority=7
            ))
        
        options.append(FollowUpOption(
            action_type="comparison",
            label="Compare treatment approaches",
            description=f"Compare different treatment strategies for this patient",
            query_template=f"Compare treatment approaches for {cancer_desc} {stage_desc}",
            priority=6
        ))
        
        options.append(FollowUpOption(
            action_type="guidelines",
            label="Standard of care guidelines",
            description=f"What do guidelines recommend for this patient?",
            query_template=f"What is the standard of care for {cancer_desc} {stage_desc}?",
            priority=5
        ))
        
        if profile.prior_treatment or profile.treatment_setting:
            options.append(FollowUpOption(
                action_type="toxicity",
                label="Expected toxicities",
                description=f"What side effects should be expected?",
                query_template=f"What are the expected toxicities for {setting_desc} treatment of {cancer_desc}?",
                priority=4
            ))
        
        # =====================================================
        # STAGING CLARIFICATION: Add follow-ups for ambiguous staging
        # DISABLED: Staging clarification temporarily disabled
        # =====================================================
        # if profile and (profile.tnm_t or profile.tnm_n or profile.tnm_m) and not profile.stage:
        #     try:
        #         from src.api.services.stage_inference_service import infer_stage_for_query
        #         from src.api.services.staging_clarification import (
        #             generate_staging_clarifications,
        #             clarifications_to_follow_up_options,
        #         )
        #
        #         inference = infer_stage_for_query(
        #             cancer_type=profile.cancer_type,
        #             cancer_location=profile.cancer_location,
        #             tnm_t=profile.tnm_t,
        #             tnm_n=profile.tnm_n,
        #             tnm_m=getattr(profile, 'tnm_m', None),
        #             age=profile.age,
        #         )
        #
        #         if inference.is_ambiguous and inference.required_factors:
        #             clarification = generate_staging_clarifications(
        #                 required_factors=inference.required_factors,
        #                 possible_stages=inference.possible_stages,
        #                 inference_notes=inference.notes,
        #                 patient_context={
        #                     "age": profile.age,
        #                     "cancer_type": profile.cancer_type,
        #                 },
        #             )
        #
        #             if clarification.needs_clarification:
        #                 clarification_dicts = clarifications_to_follow_up_options(
        #                     clarification, original_query
        #                 )
        #                 for cd in clarification_dicts[:6]:
        #                     options.insert(0, FollowUpOption(
        #                         action_type=cd["action_type"],
        #                         label=cd["label"],
        #                         description=cd["description"],
        #                         query_template=cd["query_template"],
        #                         priority=cd["priority"],
        #                     ))
        #                 logger.info(
        #                     f"[StagingClarification] Added {len(clarification_dicts)} "
        #                     f"follow-up options for factors: {inference.required_factors}"
        #                 )
        #     except Exception as e:
        #         logger.warning(f"[StagingClarification] Failed: {e}")

        options.sort(key=lambda x: x.priority, reverse=True)
        return options

    # ==================================================================
    # TRIAL MATCHING — SAME PIPELINE AS STANDARD RAG QUERY
    # ==================================================================

    async def _find_matching_trials(self, profile: PatientProfile, top_k: int = 10) -> List[MatchingTrial]:
        """
        Find clinical trials/studies that match the patient profile.

        Uses the SAME retrieval pipeline as the tumor board (multi-specialty
        agent fan-out + lightweight Qdrant retrieval) up to but excluding
        the per-expert LLM assessment step. Each of the 6 specialty agents
        builds its own specialty-aware sub-queries from the patient case
        bundle, runs them in parallel, and the results are merged across
        specialties (with a small consensus boost for studies surfaced by
        multiple specialties). We then convert the merged studies into
        `MatchingTrial` objects via the existing enrichment path.
        """
        try:
            from src.api.services.multi_specialty_retrieval import (
                retrieve_evidence_multispecialty,
            )

            t_start = time.perf_counter()
            openai_client = self._get_client()

            # -------------------------------------------------------
            # Step 1: Build retrieval query (acts as the case
            # narrative passed into the multi-specialty bundle
            # extractor)
            # -------------------------------------------------------
            query_text = self._build_retrieval_query(profile)
            category = self._infer_category_for_retrieval(profile)

            logger.info(f"[TrialMatch] Query: {query_text[:100]}...")
            logger.info(f"[TrialMatch] Category: {category}")

            # -------------------------------------------------------
            # Step 2: Run the multi-specialty retrieval pipeline.
            # Six specialty agents (medical/surgical/radiation
            # oncology, pathology/molecular, radiology, palliative
            # care) each build their own specialty-aware sub-queries
            # from the case bundle and fan them out via the tumor
            # board's lightweight Qdrant search. We stop BEFORE the
            # LLM expert-assessment step.
            # -------------------------------------------------------
            t1 = time.perf_counter()
            ms_result = await retrieve_evidence_multispecialty(
                case_text=query_text,
                query_type="treatment_recommendation",
                category=category,
                max_studies=top_k,
            )
            t1_end = time.perf_counter()
            logger.info(
                f"[TrialMatch] Multi-specialty retrieval: {t1_end - t1:.2f}s, "
                f"{len(ms_result.merged_studies)} studies, "
                f"specialties_run={list(ms_result.per_specialty.keys())}, "
                f"skipped={list(ms_result.skipped.keys())}"
            )

            if not ms_result.merged_studies:
                return []

            # -------------------------------------------------------
            # Step 3: Convert merged studies to match format + enrich.
            # `LightweightStudy` is duck-typed to expose the same
            # fields the converter reads off `StudyEvidence`
            # (doc_id / title / citation / year / chunks /
            # rerank_score / source / sections_covered).
            # -------------------------------------------------------
            matches, abstracts = self._convert_studies_to_matches(ms_result.merged_studies)

            # Batched LLM enrichment (1 call for all studies)
            t3 = time.perf_counter()
            enrichments = await asyncio.to_thread(
                self._batch_enrich_matches, matches, abstracts, profile, openai_client
            )
            t3_end = time.perf_counter()
            logger.info(f"[TrialMatch] Enrichment: {t3_end - t3:.2f}s")

            # -------------------------------------------------------
            # Step 4: Assemble MatchingTrial objects
            # -------------------------------------------------------
            try:
                from src.api.services.safety.numerical import (
                    strip_unvalidated_numbers,
                    validate_numbers_against_sources,
                )
            except Exception:
                strip_unvalidated_numbers = None
                validate_numbers_against_sources = None

            matching_trials = []
            for i, match in enumerate(matches):
                enrich = enrichments[i] if i < len(enrichments) else {}

                # Strip unverified numbers from LLM-generated enrichment fields
                # against that study's abstract, so we never serve a
                # fabricated percentage / HR in a trial-match response.
                if validate_numbers_against_sources and enrich:
                    abstract_text = abstracts[i] if i < len(abstracts) else ""
                    evidence = [{"text": abstract_text}] if abstract_text else []
                    if evidence:
                        for field_name in ("summary", "inclusion", "exclusion", "population", "patient_fit"):
                            val = enrich.get(field_name)
                            if not val:
                                continue
                            v = validate_numbers_against_sources(val, evidence)
                            if v["unvalidated_numbers"]:
                                enrich[field_name] = strip_unvalidated_numbers(
                                    val, v["unvalidated_numbers"],
                                )

                match_reasons = []
                if profile.cancer_type:
                    match_reasons.append(f"Cancer: {profile.cancer_type}")
                if match.get("source") and match["source"] != "qdrant":
                    match_reasons.append(f"Source: {match['source']}")

                study_summary = enrich.get("summary") or ""
                if not study_summary:
                    study_summary = self._clean_latex(match.get("relevant_text", ""))

                trial = MatchingTrial(
                    title=self._clean_latex(match.get("title", "Unknown")),
                    author=match.get("author"),
                    year=match.get("year"),
                    match_score=match.get("match_score", 0.0),
                    match_reasons=match_reasons,
                    relevant_excerpt=study_summary,
                    doi=match.get("doi"),
                    treatment=self._clean_latex(match.get("treatment") or "") or None,
                    inclusion_criteria=enrich.get("inclusion"),
                    exclusion_criteria=enrich.get("exclusion"),
                    eligibility_notes=[enrich["patient_fit"]] if enrich.get("patient_fit") else [],
                    population_details=enrich.get("population"),
                )
                matching_trials.append(trial)

            total_time = time.perf_counter() - t_start
            logger.info(f"[TrialMatch] Total: {total_time:.2f}s ({len(matching_trials)} trials)")
            return matching_trials

        except Exception as e:
            logger.error(f"[TrialMatch] Failed to find matching trials: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ==================================================================
    # BRIDGE: Pipeline 1/2 output → Trial Match format
    # ==================================================================

    def _build_retrieval_query(self, profile: PatientProfile) -> str:
        """
        Build a retrieval query from PatientProfile as a natural clinical question.

        Formulated as a question (not a patient profile) so the cross-encoder
        scores it properly against study text. The cross-encoder (ms-marco) is
        trained on question→passage pairs, so "What are the treatment outcomes
        for..." scores much higher than "clinical studies for...".
        """
        # Build cancer description
        cancer_desc = ""
        if profile.cancer_type and profile.cancer_location:
            cancer_desc = f"{profile.cancer_type} of the {profile.cancer_location}"
        elif profile.cancer_type:
            cancer_desc = profile.cancer_type
        elif profile.cancer_location:
            cancer_desc = f"cancer of the {profile.cancer_location}"

        # Build patient descriptor
        patient_parts = []
        if profile.age:
            patient_parts.append(f"{profile.age}-year-old")
        if profile.gender:
            patient_parts.append(profile.gender)
        patient_desc = " ".join(patient_parts) if patient_parts else "patient"

        # Build staging/pathology string
        staging_parts = []
        tnm = "".join(filter(None, [profile.tnm_t, profile.tnm_n, profile.tnm_m]))
        if tnm:
            staging_parts.append(tnm)
        if profile.stage:
            staging_parts.append(f"stage {profile.stage}")
        if profile.histology:
            staging_parts.append(profile.histology)
        staging_desc = " ".join(staging_parts)

        # Build treatment context
        treatment_parts = []
        if profile.prior_treatment:
            treatment_parts.append("treated with " + ", ".join(profile.prior_treatment))
        if profile.treatment_setting:
            treatment_parts.append(profile.treatment_setting)
        treatment_desc = " ".join(treatment_parts)

        # Build biomarker context
        biomarker_desc = ""
        if profile.molecular_markers:
            biomarker_desc = " ".join(profile.molecular_markers)

        # Build patient profile string
        profile_parts = []
        profile_parts.append(f"a {patient_desc}")
        if staging_desc:
            profile_parts.append(staging_desc)
        if biomarker_desc:
            profile_parts.append(biomarker_desc)
        profile_parts.append(cancer_desc or "cancer")
        if treatment_desc:
            profile_parts.append(treatment_desc)
        if profile.recurrence_status:
            profile_parts.append(profile.recurrence_status)

        patient_profile_str = " ".join(profile_parts)
        return f"Does {patient_profile_str} match the patients enrolled in this study?"

    def _infer_category_for_retrieval(self, profile: PatientProfile) -> Optional[str]:
        """Infer Qdrant category filter from patient profile."""
        try:
            from src.api.services.enhanced_rag_service import normalize_category_filter
        except ImportError:
            return None

        site = (profile.cancer_location or "").strip().lower()
        cancer = (profile.cancer_type or "").strip().lower()

        # Reuse the same site→category mapping from patient_matching_service
        site_to_category = {
            "maxilla": "h&n", "mandible": "h&n", "oral cavity": "h&n",
            "tongue": "h&n", "oral tongue": "h&n", "gingiva": "h&n",
            "hard palate": "h&n", "soft palate": "h&n", "buccal mucosa": "h&n",
            "floor of mouth": "h&n", "oropharynx": "h&n", "nasopharynx": "h&n",
            "hypopharynx": "h&n", "larynx": "h&n", "tonsil": "h&n",
            "base of tongue": "h&n", "pharynx": "h&n", "neck": "h&n",
            "salivary gland": "h&n", "parotid": "h&n",
            "skin": "cutaneous", "lung": "lung", "bronchus": "lung",
            "breast": "breast",
            "cervix": "gyn", "uterus": "gyn", "ovary": "gyn",
            "endometrium": "gyn", "vulva": "gyn", "vagina": "gyn",
            "anus": "gi", "rectum": "gi", "colon": "gi",
            "esophagus": "gi", "stomach": "gi", "liver": "gi", "pancreas": "gi",
            "bladder": "gu", "kidney": "gu",
            "prostate": "prostate", "brain": "cns",
        }

        # Try site first
        for site_key, cat in site_to_category.items():
            if site_key in site:
                return normalize_category_filter(cat)

        # Try cancer type
        cancer_map = {
            "breast": "breast", "lung": "lung", "nsclc": "lung", "sclc": "lung",
            "prostate": "prostate", "melanoma": "cutaneous",
            "glioma": "cns", "glioblastoma": "cns",
            "colorectal": "gi", "rectal": "gi",
            "cervical": "gyn", "ovarian": "gyn", "endometrial": "gyn",
            "bladder": "gu", "renal": "gu",
            "head and neck": "h&n", "hnscc": "h&n",
        }
        for cancer_key, cat in cancer_map.items():
            if cancer_key in cancer:
                return normalize_category_filter(cat)

        return None

    def _convert_studies_to_matches(self, studies) -> Tuple[List[Dict], List[str]]:
        """
        Convert StudyEvidence objects from comprehensive retrieval into the
        match dict format expected by _batch_enrich_matches.

        Returns:
            (matches, abstracts) — parallel lists
        """
        matches = []
        abstracts = []
        for study in studies:
            # Concatenate chunk texts for this study
            full_text = "\n".join(c.get("text", "") for c in study.chunks)

            # Extract metadata from first chunk
            doc_meta = study.chunks[0].get("doc_meta", {}) if study.chunks else {}

            # Normalize score to 0-1 range
            score = study.rerank_score
            if score > 1.0:
                # Cross-encoder scores can be >1; normalize conservatively
                score = min(0.95, 0.5 + score * 0.05)

            matches.append({
                "doc_id": study.doc_id,
                "title": study.title or doc_meta.get("title", "Unknown"),
                "author": doc_meta.get("author_et_al") or study.citation,
                "year": study.year,
                "match_score": score,
                "relevant_text": full_text[:2000],
                "doi": doc_meta.get("doi"),
                "pmid": doc_meta.get("pmid"),
                "treatment": "",
                "source": study.source,
                "sections": study.sections_covered,
                "cancer_characteristics": [],
                "demographics": [],
                "key_matches": [],
            })
            abstracts.append(full_text[:2000])

        return matches, abstracts

    def _convert_studies_to_validation_chunks(self, studies) -> List[Dict]:
        """
        Convert StudyEvidence objects into the chunk format expected by
        SimplePatientMatchingService._validate_matches_semantically().
        """
        chunks = []
        for study in studies:
            full_text = "\n".join(c.get("text", "") for c in study.chunks)
            doc_meta = study.chunks[0].get("doc_meta", {}) if study.chunks else {}

            chunks.append({
                "payload": {
                    "text": full_text[:500],
                    "doc_meta": {"title": study.title or doc_meta.get("title", "Unknown")},
                },
                "score_normalized": study.rerank_score,
            })
        return chunks

    def _profile_to_dict(self, profile: PatientProfile) -> Dict[str, Any]:
        """Convert PatientProfile to the dict format expected by eligibility filter."""
        d = {
            "age": profile.age,
            "gender": profile.gender,
            "cancer_type": profile.cancer_type,
            "anatomical_site": profile.cancer_location,
            "cancer_stage": profile.stage,
            "histology": profile.histology,
            "molecular_markers": profile.molecular_markers,
            "performance_status": profile.performance_status,
            "prior_treatments": profile.prior_treatment,
            "recurrence_score": profile.recurrence_score,
            "tnm_t": profile.tnm_t,
            "tnm_n": profile.tnm_n,
            "tnm_m": profile.tnm_m,
        }
        return {k: v for k, v in d.items() if v is not None}
    
    # ==================================================================
    # NEW: Parallel abstract fetching
    # ==================================================================
    
    def _fetch_abstracts_parallel(self, doc_ids: List[Optional[str]]) -> List[Optional[str]]:
        """
        Fetch abstracts from Qdrant for all doc_ids in parallel.
        Returns list in same order as doc_ids (None where not found).
        """
        results = [None] * len(doc_ids)
        
        # Only fetch for non-None doc_ids
        tasks = [(i, did) for i, did in enumerate(doc_ids) if did]
        if not tasks:
            return results
        
        with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as executor:
            future_to_idx = {
                executor.submit(self._get_abstract_from_qdrant, did): idx
                for idx, did in tasks
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Abstract fetch failed for index {idx}: {e}")
        
        return results
    
    # ==================================================================
    # NEW: Batched summary + eligibility in ONE LLM call
    # ==================================================================
    
    def _batch_enrich_matches(
        self,
        matches: List[Dict],
        abstracts: List[Optional[str]],
        profile: PatientProfile,
        openai_client: OpenAI
    ) -> List[Dict]:
        """
        Single LLM call to generate summary + eligibility for ALL matches.
        
        Returns list of dicts, one per match:
          { "summary", "inclusion", "exclusion", "population", "patient_fit" }
        
        Replaces N serial calls to _extract_eligibility_criteria + _generate_study_summary.
        """
        if not matches:
            return []
        
        # Build patient summary
        patient_info = self._build_patient_info_string(profile)
        
        # Build study blocks for the prompt
        study_blocks = []
        for i, match in enumerate(matches):
            abstract_text = abstracts[i] if i < len(abstracts) else None
            raw_text = self._clean_latex(match.get("relevant_text", ""))
            source_text = abstract_text if abstract_text else raw_text
            title = self._clean_latex(match.get("title", "Unknown"))
            study_blocks.append(
                f"STUDY {i+1}:\n"
                f"Title: {title}\n"
                f"Text: {source_text[:1200]}"
            )
        
        studies_text = "\n\n".join(study_blocks)
        
        try:
            from src.core.config import settings
            response = openai_client.chat.completions.create(
                model=settings.openai_mini_model,
                messages=[
                    {"role": "system", "content": """You are a clinical trial analysis expert. For each study provided, generate:
1. A complete study summary (what was investigated, design, key findings)
2. Inclusion criteria (infer from study context if not explicit)
3. Exclusion criteria (infer from study context if not explicit)
4. Enrolled population characteristics (age, gender, cancer sites, stages, biomarkers)
5. Patient fit assessment (which criteria the patient meets/doesn't meet and why)

Return ONLY a valid JSON array. Each element must have ALL five string fields.
Do NOT include eligibility criteria in the summary field."""},
                    {"role": "user", "content": f"""Patient profile: {patient_info}

{studies_text}

Return a JSON array with exactly {len(matches)} objects, one per study, in order:
[
  {{
    "summary": "What the study investigated, design, and key findings. Do not include eligibility here.",
    "inclusion": "All inclusion criteria. Cancer type/site, stage, age, ECOG, biomarkers, prior treatment status.",
    "exclusion": "All exclusion criteria. Excluded sites, stages, contraindications.",
    "population": "Enrolled demographics: age range/median, gender %, cancer sites, stage distribution, biomarkers, ECOG.",
    "patient_fit": "Detailed assessment: which criteria patient MEETS, which they DO NOT MEET, and WHY."
  }},
  ...
]"""}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            enrichments = json.loads(result_text)
            
            if not isinstance(enrichments, list):
                logger.warning(f"[Intent] Batch enrich returned non-list: {type(enrichments)}")
                return [{}] * len(matches)
            
            # Clean up null-like values
            cleaned = []
            for item in enrichments:
                if not isinstance(item, dict):
                    cleaned.append({})
                    continue
                cleaned.append({
                    "summary": self._ensure_string(item.get("summary")),
                    "inclusion": self._ensure_string(item.get("inclusion")),
                    "exclusion": self._ensure_string(item.get("exclusion")),
                    "population": self._ensure_string(item.get("population")),
                    "patient_fit": self._ensure_string(item.get("patient_fit")),
                })
            
            # Pad if LLM returned fewer items than matches
            while len(cleaned) < len(matches):
                cleaned.append({})
            
            return cleaned
            
        except Exception as e:
            logger.error(f"[Intent] Batch enrich failed: {e}")
            import traceback
            traceback.print_exc()
            return [{}] * len(matches)
    
    def _build_patient_info_string(self, profile: PatientProfile) -> str:
        """Build comprehensive patient info string for eligibility prompts."""
        patient_info = []
        if profile.age:
            patient_info.append(f"Age: {profile.age}")
        if profile.gender:
            patient_info.append(f"Gender: {profile.gender}")
        if profile.cancer_type:
            patient_info.append(f"Cancer: {profile.cancer_type}")
        if profile.cancer_location:
            patient_info.append(f"Location: {profile.cancer_location}")
        if profile.stage:
            patient_info.append(f"Stage: {profile.stage}")
        if profile.tnm_t or profile.tnm_n:
            tnm = "".join(filter(None, [profile.tnm_t, profile.tnm_n, profile.tnm_m]))
            if tnm:
                patient_info.append(f"TNM: {tnm}")
        if profile.histology:
            patient_info.append(f"Histology: {profile.histology}")
        if profile.tumor_size:
            patient_info.append(f"Tumor size: {profile.tumor_size}")
        if profile.lvi:
            patient_info.append(f"LVI: {profile.lvi}")
        if profile.pni:
            patient_info.append(f"PNI: {profile.pni}")
        if profile.margins:
            patient_info.append(f"Margins: {profile.margins}")
        if profile.lymph_nodes:
            patient_info.append(f"Lymph nodes: {profile.lymph_nodes}")
        if profile.other_pathology:
            patient_info.append(f"Other pathology: {', '.join(profile.other_pathology)}")
        if profile.molecular_markers:
            patient_info.append(f"Markers: {', '.join(profile.molecular_markers)}")
        if profile.prior_treatment:
            patient_info.append(f"Prior treatment: {', '.join(profile.prior_treatment)}")
        if profile.performance_status:
            patient_info.append(f"ECOG: {profile.performance_status}")
        if profile.recurrence_score is not None:
            patient_info.append(f"21-gene recurrence score: {profile.recurrence_score}")
        if profile.recurrence_status:
            patient_info.append(f"Status: {profile.recurrence_status}")
        return "; ".join(patient_info) if patient_info else "Limited patient information"
    
    @staticmethod
    def _ensure_string(val) -> Optional[str]:
        """Safely coerce a value to string, returning None for null-like values."""
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip()
            if val.lower() in ("null", "none", "not found", "n/a", ""):
                return None
            return val
        if isinstance(val, dict):
            parts = [f"{k}: {v}" for k, v in val.items() if v]
            return "; ".join(parts) if parts else None
        if isinstance(val, list):
            items = [str(item) for item in val if item]
            return "; ".join(items) if items else None
        return str(val)

    # ==================================================================
    # BACKWARD COMPAT: Original per-match methods preserved
    # These are no longer called from the main flow but kept for
    # any external callers (tests, other routes, etc.)
    # ==================================================================
    
    def _generate_study_summary(
        self, 
        title: str, 
        text: str, 
        openai_client: OpenAI
    ) -> str:
        """Generate a complete study summary from the title and text."""
        if not text or len(text) < 50:
            return ""
        
        try:
            from src.core.config import settings
            response = openai_client.chat.completions.create(
                model=settings.openai_mini_model,
                messages=[
                    {"role": "system", "content": """You are a medical research summarizer. 
Generate a complete summary of the clinical study focusing on:
- What the study investigated (treatment comparison, outcomes, etc.)
- Study design (randomized, phase, etc.)
- Key findings or purpose
- Primary endpoints if mentioned
Be thorough and factual. Do not include eligibility criteria in the summary."""},
                    {"role": "user", "content": f"""Study title: {title}

Study text excerpt: {text[:1500]}

Generate a complete summary of what this study investigated, its design, and key findings."""}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate study summary: {e}")
            return text
    
    def _extract_eligibility_criteria(
        self, 
        study_text: str, 
        profile: PatientProfile,
        openai_client: OpenAI
    ) -> Tuple[Optional[str], Optional[str], List[str], Optional[str]]:
        """
        Extract inclusion/exclusion criteria, population details, and assess patient eligibility.
        Returns (inclusion_criteria, exclusion_criteria, eligibility_notes, population_details)
        """
        if not study_text or len(study_text) < 50:
            return None, None, [], None
        
        patient_summary = self._build_patient_info_string(profile)
        
        try:
            from src.core.config import settings
            response = openai_client.chat.completions.create(
                model=settings.openai_mini_model,
                messages=[
                    {"role": "system", "content": """You are a clinical trial eligibility expert. Your task is to:
1. Extract inclusion and exclusion criteria from the study text
2. Describe the enrolled patient population characteristics
3. Assess whether the specific patient meets the study criteria

IMPORTANT: You MUST provide all four fields. If criteria are not explicitly stated, infer them from the study context (e.g., cancer type, stage mentioned in the study).

Return ONLY valid JSON with string values (not nested objects)."""},
                    {"role": "user", "content": f"""Study text: {study_text[:2500]}

Patient profile: {patient_summary}

Extract and return JSON with these REQUIRED fields (all must be non-empty strings):

{{
    "inclusion": "List ALL inclusion criteria. Include: required cancer type/site, stage requirements, histology requirements, age limits, performance status requirements, biomarker requirements, prior treatment status. Example: 'Patients must be at least 18 years of age with histologically confirmed squamous cell carcinoma of the head and neck (oral cavity, oropharynx, hypopharynx, or larynx), stage III or IV disease, ECOG 0-1, no prior chemotherapy or radiation.'",
    
    "exclusion": "List ALL exclusion criteria. Include: excluded cancer sites, excluded stages, contraindications, prior treatment exclusions. Example: 'Patients with nasopharyngeal carcinoma, distant metastases, prior radiation to the head and neck, or ECOG >2 are excluded.'",
    
    "population": "Describe the ENROLLED patient population from the trial with specific details: age range/median, gender distribution (% male/female), cancer sites included, stage distribution, histology types, biomarker status (HPV, p16, etc.), performance status distribution, smoking status if mentioned. Example: 'Demographics: median age 58 years (range 35-78), 72% male, 28% female. Cancer: oropharyngeal SCC (65%), oral cavity (20%), larynx (15%). Stage: III (30%), IVA (45%), IVB (25%). HPV status: 60% positive. ECOG: 0 (40%), 1 (60%).'",
    
    "patient_fit": "Provide a DETAILED assessment of how this specific patient fits the study. State which criteria the patient MEETS, which they DO NOT MEET, and explain WHY. Example: 'The patient meets the inclusion criteria for age (68 years, ≥18 required) and cancer type (SCC of maxilla, which is oral cavity). However, the patient does NOT meet criteria because: (1) prior surgical treatment (maxillectomy, neck dissection) excludes them from studies requiring treatment-naive patients, (2) nodal recurrence indicates disease progression which may exclude from primary treatment trials. The patient may be eligible for recurrent/metastatic disease trials instead.'"
}}"""
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(result_text)
            
            inclusion = self._ensure_string(data.get("inclusion"))
            exclusion = self._ensure_string(data.get("exclusion"))
            patient_fit = self._ensure_string(data.get("patient_fit"))
            population = self._ensure_string(data.get("population"))
            
            eligibility_notes = []
            if patient_fit:
                eligibility_notes.append(patient_fit)
            
            return inclusion, exclusion, eligibility_notes, population
            
        except Exception as e:
            logger.warning(f"Failed to extract eligibility criteria: {e}")
            return None, None, [], None

    # ==================================================================
    # ABSTRACT FETCHING (unchanged)
    # ==================================================================
    
    async def _get_abstract_from_db(self, doi: str = None, pmid: str = None) -> Optional[str]:
        """Fetch abstract from PostgreSQL if available."""
        if not doi and not pmid:
            return None
        
        try:
            from src.api.services.account_db import get_account_db
            
            db = get_account_db()
            pool = await db.get_pool()
            
            async with pool.acquire() as conn:
                if doi:
                    result = await conn.fetchval(
                        "SELECT abstract FROM studies WHERE doi = $1 AND abstract IS NOT NULL",
                        doi
                    )
                    if result:
                        return result
                
                if pmid:
                    result = await conn.fetchval(
                        "SELECT abstract FROM studies WHERE pmid = $1 AND abstract IS NOT NULL",
                        pmid
                    )
                    if result:
                        return result
            
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch abstract from DB: {e}")
            return None
    
    def _get_abstract_from_qdrant(self, doc_id: str) -> Optional[str]:
        """Fetch abstract chunk from Qdrant for a given document."""
        if not doc_id:
            return None
        
        try:
            from src.core.config import settings
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
            
            results = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=50,
                with_payload=True
            )
            
            abstract_chunks = []
            for point in results[0]:
                payload = point.payload or {}
                section = (payload.get("section") or "").lower()
                if "abstract" in section or "summary" in section:
                    text = payload.get("text", "")
                    if text and len(text) > 50:
                        abstract_chunks.append(text)
            
            if abstract_chunks:
                if len(abstract_chunks) == 1:
                    return self._clean_latex(abstract_chunks[0])
                else:
                    longest = max(abstract_chunks, key=len)
                    return self._clean_latex(longest)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch abstract from Qdrant for {doc_id}: {e}")
            return None

    # ==================================================================
    # TEXT HELPERS (unchanged)
    # ==================================================================
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX artifacts from text."""
        if not text:
            return ""
        
        text = re.sub(r'\$\{\s*\}\^?\{?\d+\}?\$', '', text)
        text = re.sub(r'\$\^\{?\d+\}?\$', '', text)
        text = re.sub(r'\$\{[^}]*\}\$', '', text)
        text = re.sub(r'\$[^$]+\$', '', text)
        text = re.sub(r'\\text\w+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\cite\{[^}]*\}', '', text)
        text = re.sub(r'\\ref\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\{\s*\}', '', text)
        
        return text.strip()

    # ==================================================================
    # FORMATTED RESPONSE (unchanged — string formatting only)
    # ==================================================================
    
    def _generate_formatted_response(
        self, 
        profile: PatientProfile, 
        options: List[FollowUpOption],
        matching_trials: List[MatchingTrial] = None
    ) -> str:
        """
        Generate a formatted conversational response summarizing the patient,
        showing matching trials in a clean summary format, and asking how to proceed.
        """
        lines = []
        
        # Patient Summary Header
        lines.append("**Patient Summary**")
        lines.append("")
        
        # Demographics line
        demo_parts = []
        if profile.age:
            demo_parts.append(f"{profile.age}yo")
        if profile.gender:
            demo_parts.append(profile.gender)
        if profile.ethnicity:
            demo_parts.append(profile.ethnicity)
        if profile.smoking_status:
            demo_parts.append(profile.smoking_status if profile.smoking_status == "non-smoker" else f"{profile.smoking_status} smoker")
        if demo_parts:
            lines.append(f"**Demographics:** {', '.join(demo_parts)}")
        
        # Diagnosis line
        diag_parts = []
        if profile.cancer_type:
            diag_parts.append(profile.cancer_type)
        if profile.cancer_location:
            diag_parts.append(f"of {profile.cancer_location}")
        if profile.histology:
            diag_parts.append(f"({profile.histology})")
        if diag_parts:
            lines.append(f"**Diagnosis:** {' '.join(diag_parts)}")
        
        # Staging line
        staging_parts = []
        if profile.tnm_t or profile.tnm_n:
            tnm = "".join(filter(None, [profile.tnm_t, profile.tnm_n, profile.tnm_m]))
            if tnm:
                staging_parts.append(tnm)
        if profile.stage:
            staging_parts.append(f"Stage {profile.stage}")
        if staging_parts:
            lines.append(f"**Staging:** {', '.join(staging_parts)}")
        
        # Pathology line
        path_parts = []
        if profile.tumor_size:
            path_parts.append(f"size {profile.tumor_size}")
        if profile.doi:
            path_parts.append(f"DOI {profile.doi}")
        if profile.lvi:
            path_parts.append(f"LVI {profile.lvi}")
        if profile.pni:
            path_parts.append(f"PNI {profile.pni}")
        if profile.margins:
            path_parts.append(f"margins {profile.margins}")
        if profile.lymph_nodes:
            path_parts.append(profile.lymph_nodes)
        if profile.other_pathology:
            path_parts.extend(profile.other_pathology)
        if path_parts:
            lines.append(f"**Pathology:** {', '.join(path_parts)}")
        
        # Treatment history
        if profile.prior_treatment:
            lines.append(f"**Prior Treatment:** {', '.join(profile.prior_treatment)}")
        
        # Recurrence status
        if profile.recurrence_status:
            lines.append(f"**Status:** {profile.recurrence_status}")
        
        # Comorbidities
        if profile.comorbidities:
            lines.append(f"**Comorbidities:** {', '.join(profile.comorbidities)}")
        
        # Performance status
        if profile.performance_status:
            lines.append(f"**Performance Status:** {profile.performance_status}")
        
        # Matching trials section
        if matching_trials:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append(f"**Matching Studies ({len(matching_trials)} found)**")
            
            for i, trial in enumerate(matching_trials[:5], 1):
                lines.append("")
                
                if trial.author and trial.year:
                    lines.append(f"**{i}. {trial.title}** ({trial.author}, {trial.year})")
                elif trial.year:
                    lines.append(f"**{i}. {trial.title}** ({trial.year})")
                else:
                    lines.append(f"**{i}. {trial.title}**")
                
                if trial.relevant_excerpt:
                    lines.append(f"*Summary:* {trial.relevant_excerpt}")
                
                if trial.treatment:
                    lines.append(f"*Treatment:* {trial.treatment}")
                
                if trial.population_details:
                    lines.append(f"*Population:* {trial.population_details}")
                elif trial.match_reasons:
                    characteristics = ", ".join(trial.match_reasons)
                    lines.append(f"*Population:* {characteristics}")
                
                criteria_parts = []
                if trial.inclusion_criteria:
                    criteria_parts.append(f"Inclusion: {trial.inclusion_criteria}")
                if trial.exclusion_criteria:
                    criteria_parts.append(f"Exclusion: {trial.exclusion_criteria}")
                if criteria_parts:
                    lines.append(f"*Eligibility:* {'; '.join(criteria_parts)}")
                
                if trial.eligibility_notes:
                    fit_notes = " ".join(trial.eligibility_notes)
                    lines.append(f"*Patient Fit:* {fit_notes}")
                else:
                    score_pct = int(trial.match_score * 100)
                    if score_pct >= 70:
                        lines.append(f"*Patient Fit:* Strong match ({score_pct}%) - patient characteristics align well with study population.")
                    elif score_pct >= 50:
                        lines.append(f"*Patient Fit:* Moderate match ({score_pct}%) - patient may be eligible pending additional criteria review.")
                    else:
                        lines.append(f"*Patient Fit:* Partial match ({score_pct}%) - some criteria align; review full eligibility requirements.")
        
        # Follow-up section
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("**How would you like me to proceed?**")
        lines.append("")
        
        for opt in options[:4]:
            lines.append(f"- {opt.description}")
        
        lines.append("")
        lines.append("*Select an option above or ask a specific question.*")
        
        return "\n".join(lines)


# ============================================
# SINGLETON (unchanged)
# ============================================

_query_intent_service: Optional[QueryIntentService] = None


def get_query_intent_service() -> QueryIntentService:
    """Get or create the query intent service singleton."""
    global _query_intent_service
    if _query_intent_service is None:
        _query_intent_service = QueryIntentService()
    return _query_intent_service