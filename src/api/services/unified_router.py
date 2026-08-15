"""
Unified Router Service

Single source of truth for query classification and routing decisions.
Uses Module_Classifier as primary classification, then determines
query-type as a secondary format hint within the module context.

This consolidates the dual classification systems into a single router
that provides consistent routing decisions across the RAG pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum


class QueryModule(Enum):
    """
    Primary classification modules for query routing.
    
    - GENERAL_KNOWLEDGE: Factual Q&A with ground truth answers (dose, guidelines, trial results)
    - PATIENT_SPECIFIC: Personalized guidance for a patient case (demographics, staging, biomarkers)
    - EVIDENCE_EXPLORATION: Comparative analysis and option exploration (best, optimal, compare)
    """
    GENERAL_KNOWLEDGE = "general_knowledge"
    PATIENT_SPECIFIC = "patient_specific"
    EVIDENCE_EXPLORATION = "evidence_exploration"


class QueryType(Enum):
    """
    Secondary format hints within each module.
    
    These determine response formatting and structure based on
    the specific type of question being asked.
    """
    DOSE_QUESTION = "dose_question"
    TRIAL_RESULTS = "trial_results"
    COMPARISON = "comparison"
    STAGING = "staging"
    WORKUP = "workup"
    MECHANISM = "mechanism"
    SIDE_EFFECTS = "side_effects"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    GENERAL = "general"


@dataclass
class RoutingResult:
    """
    Result of unified routing decision.
    
    Contains the primary module classification, secondary query type,
    format hints for response generation, and retrieval strategy.
    """
    module: QueryModule
    module_confidence: float
    query_type: QueryType
    format_hints: Dict[str, Any]
    retrieval_strategy: Dict[str, Any]
    signals_matched: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert routing result to dictionary for API responses."""
        return {
            "module": self.module.value,
            "module_confidence": self.module_confidence,
            "query_type": self.query_type.value,
            "format_hints": self.format_hints,
            "retrieval_strategy": self.retrieval_strategy,
            "signals_matched": self.signals_matched,
        }

from typing import Optional


class UnifiedRouter:
    """
    Single source of truth for query classification and routing.
    
    Uses Module_Classifier as primary classification, then determines
    query-type as a secondary format hint within the module context.
    """
    
    def __init__(self):
        self._format_templates = self._load_format_templates()
    
    def route_query(
        self, 
        query: str, 
        conversation_context: Optional[List[Dict]] = None
    ) -> RoutingResult:
        """
        Route a query to the appropriate module with format hints.
        
        Args:
            query: User's query string
            conversation_context: Previous conversation entries
            
        Returns:
            RoutingResult with module, query_type, and format hints
        """
        # Step 1: Primary classification via Module_Classifier
        from src.api.services.module_classifier import (
            classify_module as mc_classify_module,
            get_retrieval_strategy as mc_get_retrieval_strategy,
            QueryModule as MCQueryModule,
        )
        
        classification = mc_classify_module(query)
        
        # Map module_classifier's QueryModule to unified_router's QueryModule
        module = self._map_module(classification.module)
        
        # Step 2: Determine query-type as format hint
        query_type = self._determine_query_type(
            query, 
            module,
            classification.has_patient_context
        )
        
        # Step 3: Get format hints for module + query_type combination
        format_hints = self._get_format_hints(module, query_type)
        
        # Step 4: Get retrieval strategy from module_classifier
        retrieval_strategy = mc_get_retrieval_strategy(classification.module)
        
        return RoutingResult(
            module=module,
            module_confidence=classification.confidence,
            query_type=query_type,
            format_hints=format_hints,
            retrieval_strategy=retrieval_strategy,
            signals_matched=classification.signals_matched,
        )
    
    def _map_module(self, mc_module) -> QueryModule:
        """
        Map module_classifier's QueryModule to unified_router's QueryModule.
        
        Args:
            mc_module: QueryModule from module_classifier
            
        Returns:
            QueryModule from unified_router
        """
        # Map by value since both enums have the same string values
        return QueryModule(mc_module.value)
    
    def _determine_query_type(
        self, 
        query: str, 
        module: QueryModule,
        has_patient_context: bool
    ) -> QueryType:
        """
        Determine query-type as format hint within module context.
        
        Args:
            query: User's query string
            module: Primary module classification
            has_patient_context: Whether query contains patient context
            
        Returns:
            QueryType for response formatting
        """
        query_lower = query.lower()
        
        # Comparison (check first - takes priority over other types)
        if any(p in query_lower for p in ["compare", "versus", "vs ", "vs.", "difference between"]):
            return QueryType.COMPARISON
        
        # Dose questions
        if any(p in query_lower for p in ["dose", "gy", "gray", "fractionation", "fx"]):
            return QueryType.DOSE_QUESTION
        
        # Trial results
        if any(p in query_lower for p in ["trial", "study show", "results", "rtog", "nrg"]):
            return QueryType.TRIAL_RESULTS
        
        # Staging
        if any(p in query_lower for p in ["stage", "staging", "tnm"]):
            return QueryType.STAGING
        
        # Workup
        if any(p in query_lower for p in ["workup", "work-up", "evaluation", "diagnostic"]):
            return QueryType.WORKUP
        
        # Mechanism
        if any(p in query_lower for p in ["mechanism", "how does", "pathway"]):
            return QueryType.MECHANISM
        
        # Side effects
        if any(p in query_lower for p in ["side effect", "toxicity", "adverse"]):
            return QueryType.SIDE_EFFECTS
        
        # Treatment recommendation (patient-specific)
        if has_patient_context and module == QueryModule.PATIENT_SPECIFIC:
            return QueryType.TREATMENT_RECOMMENDATION
        
        return QueryType.GENERAL
    
    def _get_format_hints(
        self, 
        module: QueryModule, 
        query_type: QueryType
    ) -> Dict[str, Any]:
        """
        Get format hints for module + query_type combination.
        
        Args:
            module: Primary module classification
            query_type: Secondary query type
            
        Returns:
            Dictionary of format hints for response generation
        """
        key = f"{module.value}:{query_type.value}"
        return self._format_templates.get(key, self._format_templates["default"])
    
    def _load_format_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load centralized format templates for all module-querytype combinations.
        
        Returns:
            Dictionary mapping module:querytype keys to format hint dictionaries
        """
        return {
            # General Knowledge format templates
            "general_knowledge:dose_question": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "include_dose_table": True,
                "numerical_precision": "exact",
            },
            "general_knowledge:trial_results": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "include_outcome_data": True,
                "numerical_precision": "exact",
            },
            "general_knowledge:mechanism": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "include_pathway_details": True,
            },
            "general_knowledge:side_effects": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "include_toxicity_grades": True,
            },
            "general_knowledge:staging": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "include_staging_criteria": True,
            },
            "general_knowledge:general": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "numerical_precision": "exact",
            },
            
            # Patient-Specific format templates
            "patient_specific:treatment_recommendation": {
                "response_structure": "patient_guidance",
                "citation_style": "inline",
                "include_patient_summary": True,
                "include_rationale": True,
            },
            "patient_specific:staging": {
                "response_structure": "patient_guidance",
                "citation_style": "inline",
                "include_staging_assessment": True,
            },
            "patient_specific:workup": {
                "response_structure": "patient_guidance",
                "citation_style": "inline",
                "include_workup_steps": True,
            },
            "patient_specific:general": {
                "response_structure": "patient_guidance",
                "citation_style": "inline",
                "include_patient_summary": True,
            },
            
            # Evidence Exploration format templates
            "evidence_exploration:comparison": {
                "response_structure": "comparison",
                "citation_style": "inline",
                "include_comparison_table": True,
                "min_options": 2,
            },
            "evidence_exploration:trial_results": {
                "response_structure": "comparison",
                "citation_style": "inline",
                "include_outcome_data": True,
                "include_comparison_table": True,
            },
            "evidence_exploration:dose_question": {
                "response_structure": "comparison",
                "citation_style": "inline",
                "include_dose_table": True,
                "include_comparison_table": True,
            },
            "evidence_exploration:general": {
                "response_structure": "comparison",
                "citation_style": "inline",
                "include_comparison_table": True,
            },
            
            # Default template
            "default": {
                "response_structure": "direct_answer",
                "citation_style": "inline",
                "numerical_precision": "exact",
            },
        }


# Singleton instance
_router_instance: Optional[UnifiedRouter] = None


def get_unified_router() -> UnifiedRouter:
    """
    Get singleton UnifiedRouter instance.

    Returns:
        The shared UnifiedRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = UnifiedRouter()
    return _router_instance


# ─── Phase 3: surface picker ───────────────────────────────────────────────


class Surface(Enum):
    """Which end-user surface should render the response for a query.

    Surfaces correspond to the five legacy pipelines (P1..P5), re-named
    after what they actually render to the user rather than the order in
    which they were implemented:

        P1 : synthesized answer + evidence (EnhancedRAGService.query)
        P3 : raw EvidenceBundle (research surface — no synthesis)
        P4 : tumor-board multi-specialist report (TumorBoardOrchestrator)
        P5 : trial-match list (QueryIntentService.analyze_query)
    """

    P1_SYNTHESIZED = "p1"
    P3_RESEARCH_EVIDENCE = "p3"
    P4_TUMOR_BOARD = "p4"
    P5_TRIAL_MATCH = "p5"


def pick_surface(
    extracted_axes: Optional[Dict[str, Any]] = None,
    *,
    query_mode: Optional[str] = None,
    force_trial_match: bool = False,
) -> Surface:
    """Decide which surface should render a given query.

    Precedence (first rule to match wins):
      1. `force_trial_match` or `query_mode == "trial_match"`  → P5
      2. patient context + prior definitive treatment +
         disease_trajectory in {recurrence, progression}       → P4
      3. `query_mode == "research"`                            → P3
      4. default                                               → P1

    `extracted_axes` is the dict produced by the clinical inference layer
    (or by any retrieval mode's `EvidenceBundle.extracted_axes`). The
    keys this function reads are the same ones the multispecialty bundle
    exposes: `has_patient_context`, `trajectory_flags`,
    `prior_definitive_treatment`.
    """
    if force_trial_match or query_mode == "trial_match":
        return Surface.P5_TRIAL_MATCH

    axes = extracted_axes or {}
    has_patient = bool(axes.get("has_patient_context"))
    trajectory = set(axes.get("trajectory_flags") or [])
    has_trajectory_progression = bool(
        trajectory & {
            "recurrence", "recurrent",
            "progression", "progressing",
            "progressing_on_ici", "ici_refractory",
        }
    )
    prior_def_tx = bool(
        axes.get("prior_definitive_treatment")
        or axes.get("prior_treatments")
    )

    if has_patient and prior_def_tx and has_trajectory_progression:
        return Surface.P4_TUMOR_BOARD

    if query_mode == "research":
        return Surface.P3_RESEARCH_EVIDENCE

    return Surface.P1_SYNTHESIZED
