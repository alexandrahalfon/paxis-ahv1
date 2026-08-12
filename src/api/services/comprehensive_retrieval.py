"""
Comprehensive Retrieval Service

A hybrid retrieval approach that:
1. Runs standard retrieval (Qdrant + PostgreSQL) with query expansion
2. Collects all unique doc_ids from both sources
3. For each doc_id, runs parallel in-document searches for comprehensive coverage
4. Reranks complete study evidence by relevance to the query

This ensures we get the full picture from each relevant study, not just fragments.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import hashlib
import json
import time

from qdrant_client import QdrantClient
import qdrant_client.models as qm
from openai import OpenAI

from src.core.config import settings
from src.api.services.query_token_resolver import ResolvedQueryTokens
from src.api.services.clinical_extractor import ClinicalProfile


# ── PG result cache ───────────────────────────────────────────────────
# Simple in-memory cache for PostgreSQL matcher results, keyed by a hash
# of the query structure dict.  Gated behind settings.enable_perf_optimizations.
_PG_CACHE_MAX_SIZE: int = 64
_pg_cache: Dict[str, Any] = {}


def _pg_cache_key(query_structure_dict: Dict[str, Any]) -> str:
    """Compute a stable hash key from a query structure dict."""
    serialized = json.dumps(query_structure_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


# Tokens that make for useless `metadata.keywords_flat` filter clauses —
# either too generic to discriminate ("rt", "chemo") or better represented
# by another axis. Kept short; extend cautiously.
_AXIS_SKIP_TOKENS: Set[str] = {
    "", "none", "unknown", "n/a",
    "rt", "chemo", "surgery",
    "patient", "study",
}


# ── Category normalization ────────────────────────────────────────────
# Known alias map: maps variant forms to a single canonical name.
# The tricky pair is h&n ↔ head_neck; all others just need suffix
# stripping.
_CATEGORY_ALIASES: Dict[str, str] = {
    "h&n": "head_neck",
    "hn": "head_neck",
}

# Suffixes to strip from Qdrant-style category strings.
_CATEGORY_SUFFIXES = ("_processed_documents", "_docs", "_documents")


def normalize_category(value: Optional[str]) -> str:
    """Normalize a category string for comparison.

    1. Returns empty string for None / empty / whitespace-only input.
    2. Lowercases and strips whitespace.
    3. Strips ``_processed_documents``, ``_docs``, ``_documents`` suffixes.
    4. Resolves known aliases (``h&n`` → ``head_neck``).
    """
    if not value or not value.strip():
        return ""
    result = value.strip().lower()
    for suffix in _CATEGORY_SUFFIXES:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break
    return _CATEGORY_ALIASES.get(result, result)


def should_skip_phase3_candidate(
    query_category: Optional[str],
    doc_category: Optional[str],
) -> bool:
    """Decide whether to skip a Phase 3 dispatch for a candidate document.

    Rules:
    1. If the query has no known category → never skip (return False).
    2. If the doc has no known category → never skip (return False).
    3. If normalized categories differ → skip (return True).
    4. Otherwise → don't skip (return False).

    Uses :func:`normalize_category` so that ``prostate`` matches
    ``prostate_processed_documents`` and ``head_neck`` matches
    ``h&n_processed_documents``.
    """
    norm_q = normalize_category(query_category)
    norm_d = normalize_category(doc_category)
    if not norm_q or not norm_d:
        return False
    return norm_q != norm_d


def _materialize_hits(results) -> List[Dict[str, Any]]:
    """Convert Qdrant `results.points` into the hit-dict shape used
    throughout the retriever."""
    hits: List[Dict[str, Any]] = []
    for point in results.points:
        payload = dict(point.payload or {})
        hits.append({
            "point_id": point.id,
            "score": float(point.score),
            "doc_id": payload.get("doc_id"),
            "doc_meta": payload.get("doc_meta", {}),
            "category": payload.get("category"),
            "section": payload.get("section"),
            "text": payload.get("text", ""),
        })
    return hits


def _build_must_not_clauses(
    resolved_tokens: Optional[ResolvedQueryTokens],
) -> List[Any]:
    """Typed-slot negations only (biomarker / drug / histology).

    Prevents 'no HER2 amplification' from surfacing HER2+ studies.
    """
    clauses: List[Any] = []
    if resolved_tokens is None or not resolved_tokens.negated:
        return clauses
    typed_slot_terms = (
        resolved_tokens.biomarkers
        | resolved_tokens.drugs
        | resolved_tokens.histologies
    )
    typed_negated = sorted(
        n for n in resolved_tokens.negated if n in typed_slot_terms
    )
    if typed_negated:
        clauses.append(
            qm.FieldCondition(
                key="metadata.keywords_flat",
                match=qm.MatchAny(any=typed_negated),
            )
        )
    return clauses


def _profile_to_must_clauses(profile: "ClinicalProfile") -> List[Any]:
    """Build Qdrant `must` clauses from a fully-typed ClinicalProfile.

    Each clinical axis the profile populates becomes one hard
    `FieldCondition(MatchAny(...))` against the corresponding
    `metadata.*_detected` payload field. Axes left empty in the profile
    are not added — a sparse profile is a softer filter, which is the
    right behavior when the extractor couldn't pin down an axis.
    """
    clauses: List[Any] = []
    axis_to_field = (
        ("cancer_type_label",   "cancer_types_detected",   "scalar"),
        ("cancer_sites",        "sites_detected",          "list"),
        ("histologies",         "histologies_detected",    "list"),
        ("stages",              "stages_detected",         "list"),
        ("biomarkers",          "biomarkers_detected",     "list"),
        ("prior_treatments",    "drugs_detected",          "list"),
    )
    for attr, field_name, kind in axis_to_field:
        value = getattr(profile, attr, None)
        if kind == "scalar":
            if value:
                clauses.append(
                    qm.FieldCondition(
                        key=f"metadata.{field_name}",
                        match=qm.MatchAny(any=[value]),
                    )
                )
        else:
            values = value or []
            if values:
                clauses.append(
                    qm.FieldCondition(
                        key=f"metadata.{field_name}",
                        match=qm.MatchAny(any=list(values)),
                    )
                )
    return clauses


def _profile_axis_counts(profile: "ClinicalProfile") -> Dict[str, int]:
    """Short summary for logging the hard-filter shape."""
    return {
        "cancer_type":       1 if profile.cancer_type_label else 0,
        "sites":             len(profile.cancer_sites),
        "histologies":       len(profile.histologies),
        "stages":            len(profile.stages),
        "biomarkers":        len(profile.biomarkers),
        "prior_treatments":  len(profile.prior_treatments),
        "disease_status":    len(profile.disease_status),
        "bio_expressions":   len(profile.biomarker_expressions),
    }


def _profile_to_should_clauses(profile: "ClinicalProfile") -> List[Any]:
    """Soft-boost clauses from a ClinicalProfile.

    ``disease_status`` is boost-only (not hard-filtered) because it's
    semantically fuzzy — rejecting studies whose keywords_flat doesn't
    literally contain "recurrent" would exclude too much relevant
    evidence. The dense vector handles most of the semantic lift; this
    clause just nudges status-matching chunks up the ranking.
    """
    clauses: List[Any] = []
    status_terms = [s.lower() for s in (profile.disease_status or []) if s]
    if status_terms:
        clauses.append(
            qm.FieldCondition(
                key="metadata.keywords_flat",
                match=qm.MatchAny(any=sorted(set(status_terms))),
            )
        )
    return clauses


def _collect_structured_axes(
    query_structure,
    inferred_axes: Optional[Dict[str, Any]],
    resolved_tokens: Optional[ResolvedQueryTokens] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Project the patient's structured axes into filter phrases for Qdrant.

    Returns:
        flat_terms:     lowercased phrases suitable for a
                        `metadata.keywords_flat` `should` clause. Union of
                        `QueryStructure`-derived phrases, inferred-axis
                        flags, and `resolved_tokens.keywords_flat`.
        typed_filters:  `{field_name → [canonical values]}` for typed
                        `metadata.*_detected` `should` clauses. Built from
                        `resolved_tokens.as_typed_filters()`. Preserves
                        canonical casing emitted by the tagger (e.g.
                        "Head and Neck Cancer"), since ingest-time tags
                        carry the same casing.

    Draws `flat_terms` from:
      - `query_structure.cancer` (site, site_detail, histology, stage, TNM, biomarkers)
      - `query_structure.treatment` (modality, prior_treatments)
      - `inferred_axes["trajectory_flags"]` / `["metastatic_sites"]`
        (from clinical_inference.py)
      - `resolved_tokens.keywords_flat` (from KeywordTagger)

    Capped at 30 entries to keep filter payload bounded.
    """
    axes: List[str] = []

    if query_structure is not None:
        cancer = getattr(query_structure, "cancer", None)
        if cancer is not None:
            for attr in ("site", "site_detail", "histology", "stage"):
                val = getattr(cancer, attr, None)
                if isinstance(val, str):
                    axes.append(val)
            tnm = getattr(cancer, "get_tnm_string", lambda: None)()
            if tnm:
                axes.append(tnm)
                for part in str(tnm).split():
                    if part:
                        axes.append(part)
            for bm in getattr(cancer, "biomarkers", []) or []:
                if isinstance(bm, str):
                    axes.append(bm)
        treatment = getattr(query_structure, "treatment", None)
        if treatment is not None:
            modality = getattr(treatment, "modality", None)
            if isinstance(modality, str):
                axes.append(modality)
            for prior in getattr(treatment, "prior_treatments", []) or []:
                if isinstance(prior, str):
                    axes.append(prior)

    if inferred_axes:
        for flag in inferred_axes.get("trajectory_flags") or []:
            if isinstance(flag, str):
                axes.append(flag)
        for site in inferred_axes.get("metastatic_sites") or []:
            if isinstance(site, str):
                axes.append(site)

    if resolved_tokens is not None:
        axes.extend(resolved_tokens.keywords_flat)

    seen: Set[str] = set()
    cleaned: List[str] = []
    for a in axes:
        al = a.strip().lower()
        if not al or al in _AXIS_SKIP_TOKENS or al in seen:
            continue
        seen.add(al)
        cleaned.append(al)
        if len(cleaned) >= 30:
            break

    typed_filters: Dict[str, List[str]] = {}
    if resolved_tokens is not None:
        typed_filters = resolved_tokens.as_typed_filters()

    return cleaned, typed_filters


@dataclass
class StudyEvidence:
    """Complete evidence from a single study."""
    doc_id: str
    title: str
    citation: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    initial_score: float = 0.0  # Score from phase 1
    rerank_score: float = 0.0   # Score after cross-encoder reranking
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    sections_covered: Set[str] = field(default_factory=set)
    source: str = "qdrant"  # "qdrant", "postgres", or "both"
    match_score: Optional[float] = None                    # PG match score 0-100
    match_breakdown: Optional[Dict[str, Any]] = None       # PGMatchBreakdown.to_dict()
    axis_mismatches: List[str] = field(default_factory=list)
    soft_score_normalized: Optional[float] = None          # SoftScorer 0-100 score
    # Patient–study match (v1, weighted overlap of ClinicalProfile axes
    # against the study's doc_level_* metadata; see patient_match_scorer.py)
    patient_match_score: Optional[int] = None              # 0–100
    patient_match_breakdown: Optional[Dict[str, Any]] = None
    # Evidence class (two-track retrieval). Populated by
    # evidence_classifier.classify_study after chunks are gathered so
    # retrieve_comprehensive can budget guidelines/landmarks separately
    # from patient-specific trials.
    evidence_type: str = "trial"                           # "guideline" | "landmark_trial" | "trial"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "citation": self.citation,
            "year": self.year,
            "category": self.category,
            "initial_score": self.initial_score,
            "rerank_score": self.rerank_score,
            "chunks": self.chunks,
            "sections_covered": list(self.sections_covered),
            "chunk_count": len(self.chunks),
            "source": self.source,
        }
        if self.match_score is not None:
            d["match_score"] = self.match_score
        if self.match_breakdown is not None:
            d["match_breakdown"] = self.match_breakdown
        if self.axis_mismatches:
            d["axis_mismatches"] = list(self.axis_mismatches)
        if self.soft_score_normalized is not None:
            d["soft_score_normalized"] = self.soft_score_normalized
        if self.patient_match_score is not None:
            d["patient_match_score"] = self.patient_match_score
        if self.patient_match_breakdown is not None:
            d["patient_match_breakdown"] = self.patient_match_breakdown
        d["evidence_type"] = self.evidence_type
        return d


@dataclass 
class ComprehensiveRetrievalResult:
    """Result of comprehensive retrieval."""
    studies: List[StudyEvidence]
    total_chunks: int
    retrieval_time_ms: float
    phase1_qdrant_docs: int
    phase1_postgres_docs: int
    phase2_docs_searched: int
    query_structure: Optional[Dict[str, Any]] = None
    expanded_query: Optional[str] = None
    reconciled_structure: Optional[Any] = None  # ReconciledStructure from query_reconciliation
    
    def get_evidence_list(self) -> List[Dict[str, Any]]:
        """Get flattened evidence list for synthesis."""
        evidence = []
        for study in self.studies:
            for chunk in study.chunks:
                evidence.append({
                    "doc_id": study.doc_id,
                    "title": study.title,
                    "citation": study.citation,
                    "year": study.year,
                    "category": study.category,
                    "score": chunk.get("score", study.rerank_score),
                    "text": chunk.get("text", ""),
                    "section": chunk.get("section"),
                    "chunk_type": chunk.get("chunk_type"),
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_meta": chunk.get("doc_meta", {}),
                    # Expose the patient-match score per chunk so the
                    # citation UI can render it inline next to the
                    # source. Same score for every chunk within a
                    # study — it's study-level.
                    "patient_match_score": study.patient_match_score,
                    "patient_match_breakdown": study.patient_match_breakdown,
                    "evidence_type": study.evidence_type,
                    "_study_rerank_score": study.rerank_score,
                    "_sections_in_study": list(study.sections_covered),
                })
        return evidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "studies": [s.to_dict() for s in self.studies],
            "total_chunks": self.total_chunks,
            "retrieval_time_ms": self.retrieval_time_ms,
            "phase1_qdrant_docs": self.phase1_qdrant_docs,
            "phase1_postgres_docs": self.phase1_postgres_docs,
            "phase2_docs_searched": self.phase2_docs_searched,
            "expanded_query": self.expanded_query,
        }


class ComprehensiveRetriever:
    """
    Comprehensive retriever that gets full study coverage.
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        collection: Optional[str] = None,
    ):
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.collection = collection or settings.qdrant_collection
        self._embed_model = settings.embed_model
        self._cross_encoder = None
        # Phase 3 gate micro-batching state (see _gate_score_batched)
        self._gate_pending: List[Tuple[Tuple[str, str], asyncio.Future]] = []
        self._gate_flush_task: Optional[asyncio.Task] = None

    # Gate batching tuning: 60ms accumulation window is negligible next
    # to a multi-second Phase 3, and long enough to catch the burst of
    # gate calls the eager dispatcher produces. 64 pairs ≈ the max
    # candidate pool, flushed immediately if reached.
    _GATE_WINDOW_S = 0.06
    _GATE_MAX_BATCH = 64

    # ── Async wrappers for synchronous I/O ─────────────────────────────
    # The Qdrant and OpenAI clients are synchronous.  Without offloading
    # to a thread-pool executor, every `await asyncio.gather(...)` in
    # this class runs its branches *sequentially* on the event loop
    # thread — Postgres waits for Qdrant, PTO waits for Postgres, and
    # Phase 3 tasks can't start until gather() finishes.

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a synchronous function in the default thread-pool executor."""
        import functools
        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, call)

    async def _qdrant_query(self, **kwargs):
        """Async wrapper around self.qdrant.query_points()."""
        return await self._run_sync(
            self.qdrant.query_points,
            collection_name=kwargs.pop("collection_name", self.collection),
            **kwargs,
        )

    async def _embed_async(self, text: str) -> List[float]:
        """Async wrapper around self.embed_query()."""
        return await self._run_sync(self.embed_query, text)

    async def _gate_score_batched(self, cross_encoder, pair: Tuple[str, str]) -> float:
        """
        Micro-batching wrapper around cross_encoder.predict() for the
        Phase 3 gate.

        The gate runs inside each per-document Phase 3 task, so under the
        eager-dispatch design 20-100 gate scorings can be in flight at
        once — previously each made its own predict() call with a batch
        of one, which serialises on the CPU-bound model and pays full
        per-call overhead every time (the ~8s reranking cost seen in
        production logs). This wrapper accumulates concurrent requests
        for a short window and scores them in ONE predict() call, which
        is exactly how the Phase 4 fallback already batches
        (_phase4_rerank_studies). Scores are identical to the unbatched
        path — same model, same pairs, deterministic — only the call
        pattern changes.

        Failure mode is preserved: an exception from predict() is
        raised to every waiting caller, whose existing try/except
        continues without the gate (fail-open), matching the previous
        single-call behaviour.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._gate_pending.append((pair, fut))

        # First caller in this window schedules the flush; batch-size cap
        # triggers an immediate flush so a big burst doesn't wait.
        if len(self._gate_pending) >= self._GATE_MAX_BATCH:
            self._flush_gate_batch(cross_encoder)
        elif self._gate_flush_task is None or self._gate_flush_task.done():
            self._gate_flush_task = asyncio.create_task(
                self._gate_flush_after_window(cross_encoder)
            )
        return await fut

    async def _gate_flush_after_window(self, cross_encoder):
        await asyncio.sleep(self._GATE_WINDOW_S)
        self._flush_gate_batch(cross_encoder)

    def _flush_gate_batch(self, cross_encoder):
        batch = self._gate_pending
        self._gate_pending = []
        if not batch:
            return

        async def _run_batch():
            pairs = [p for p, _ in batch]
            try:
                scores = await self._run_sync(cross_encoder.predict, pairs)
                print(f"[Phase3Gate] Batched {len(pairs)} gate scorings into one predict() call")
                for (_, fut), s in zip(batch, scores):
                    if not fut.done():
                        fut.set_result(float(s))
            except Exception as e:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

        asyncio.ensure_future(_run_batch())

    def _get_cross_encoder(self):
        """Lazy load cross-encoder."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                print("[ComprehensiveRetrieval] Cross-encoder loaded")
            except Exception as e:
                print(f"[ComprehensiveRetrieval] Cross-encoder not available: {e}")
        return self._cross_encoder
    
    def embed_query(self, query_text: str) -> List[float]:
        """Generate embedding for query."""
        response = self.openai.embeddings.create(
            model=self._embed_model,
            input=query_text,
        )
        return response.data[0].embedding
    
    async def retrieve_comprehensive(
        self,
        query_text: str,
        max_studies: int = 12,
        chunks_per_study: int = 8,
        category: Optional[str] = None,
        accumulated_context: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        clinical_profile: Optional[ClinicalProfile] = None,
        max_guidelines: int = 5,
    ) -> ComprehensiveRetrievalResult:
        """
        Comprehensive retrieval with full study coverage.

        Args:
            query_text: The user's query
            max_studies: Maximum studies to include in final result
            chunks_per_study: Maximum chunks per study
            category: Optional category filter
            accumulated_context: Accumulated context from conversation
            conversation_context: List of conversation context entries for query expansion and boosting
            clinical_profile: Optional cascading-extractor output. When
                provided and populated, Phase 1 uses its axes as HARD
                filters against the canonical `metadata.*_detected`
                payload fields. Falls back to soft-boost filtering if
                the strict filter yields no hits.

        Returns:
            ComprehensiveRetrievalResult with complete study evidence
        """
        t_start = time.perf_counter()

        # Preserve the user's original question for cross-encoder relevance
        # scoring. ``query_text`` itself may be overwritten downstream by
        # query decomposition (Phase 0.5), but the cross-encoder should
        # always compare passages against the natural-language question
        # the user actually asked.
        original_query_text = query_text

        try:
            from src.api.services import pipeline_metrics as _pm
            if _pm.current() is None:
                _pm.start("p2")
        except Exception:
            pass

        print("\n" + "=" * 80)
        print("  PIPELINE START: ComprehensiveRetriever.retrieve_comprehensive()")
        print("=" * 80)
        print(f"  Query: {query_text[:200]}{'...' if len(query_text) > 200 else ''}")
        print(f"  Settings: max_studies={max_studies}, chunks_per_study={chunks_per_study}, category={category}")
        print("-" * 80)

        # Extract context from conversation for query expansion AND score boosting
        context_doc_ids: Set[str] = set()
        context_queries: List[str] = []
        context_titles: List[str] = []

        if conversation_context:
            for entry in conversation_context:
                # Collect doc_ids for score boosting
                if entry.get("doc_ids"):
                    context_doc_ids.update(entry["doc_ids"])
                # Collect previous queries for query expansion
                if entry.get("query"):
                    context_queries.append(entry["query"])
                # Collect doc titles for semantic expansion
                if entry.get("doc_titles"):
                    context_titles.extend(entry["doc_titles"][:3])  # Top 3 titles per entry

            if context_doc_ids:
                print(f"[ComprehensiveRetrieval] Conversation context: {len(context_doc_ids)} doc_ids, {len(context_queries)} queries")
        else:
            print(f"  [Conversation Context] No conversation context provided")

        # =====================================================
        # PHASE 0.5: Query Decomposition (before structuring)
        # =====================================================
        decomposition_result = None
        if settings.enable_query_decomposition:
            try:
                from src.api.services.query_decomposer import QueryDecomposer
                decomposer = QueryDecomposer()
                decomposition_result = await decomposer.decompose(query_text)
                if decomposition_result.is_decomposed:
                    print(f"[Decomp] Query decomposed into {len(decomposition_result.sub_queries)} sub-queries "
                          f"(reason={decomposition_result.decomposition_reason})")
                    for i, sq in enumerate(decomposition_result.sub_queries):
                        print(f"[Decomp]   sub_query[{i}]: {sq[:120]}")
                    # Use the first sub-query as the primary retrieval query
                    query_text = decomposition_result.sub_queries[0]
                    print(f"[Decomp] Using first sub-query as primary: {query_text[:120]}")
                else:
                    print(f"[Decomp] No decomposition needed")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Decomp] Decomposition failed (continuing with original query): {e}")

        # =====================================================
        # PHASE 1: Standard retrieval with query expansion
        # =====================================================
        print(f"\n{'─' * 80}")
        print(f"  STEP 1: Query Classification + Expansion")
        print(f"{'─' * 80}")
        
        # Expand query using existing expansion logic
        from src.api.services.enhanced_rag_service import expand_query, classify_query
        
        query_classification = classify_query(query_text)
        query_type = query_classification.get("primary_type", "general")
        print(f"  [Query Classification] Type: {query_type}")
        print(f"  [Query Classification] Full result: {query_classification}")

        expanded_query = expand_query(query_text)
        if expanded_query != query_text:
            # Show what terms were added
            original_terms = set(query_text.lower().split())
            expanded_terms = set(expanded_query.lower().split())
            new_terms = expanded_terms - original_terms
            print(f"  [Query Expansion] Added {len(new_terms)} terms: {list(new_terms)[:15]}{'...' if len(new_terms) > 15 else ''}")
        else:
            print(f"  [Query Expansion] No expansion terms added")
        
        # Add clinical entity expansion terms
        try:
            from src.api.services.clinical_entity_extractor import get_clinical_entity_extractor
            extractor = get_clinical_entity_extractor()
            clinical_terms = extractor.get_query_expansion_terms(query_text)
            if clinical_terms:
                expanded_query = f"{expanded_query} {' '.join(clinical_terms)}"
                print(f"[ComprehensiveRetrieval] Added {len(clinical_terms)} clinical terms")
        except Exception as e:
            print(f"[ComprehensiveRetrieval] Clinical entity extraction failed: {e}")
        
        # Add conversation context to query expansion
        # This ensures retrieval actively searches for context-related content
        if context_queries or context_titles:
            context_expansion_terms = []
            
            # Extract key terms from previous queries (last 2 queries)
            for prev_query in context_queries[-2:]:
                # Extract nouns/key terms (simple approach: words > 4 chars, not common words)
                common_words = {'what', 'which', 'when', 'where', 'does', 'have', 'with', 'from', 'that', 'this', 'there', 'about', 'would', 'could', 'should'}
                words = prev_query.lower().split()
                key_terms = [w for w in words if len(w) > 4 and w not in common_words]
                context_expansion_terms.extend(key_terms[:5])  # Top 5 terms per query
            
            # Extract key terms from doc titles (last 3 titles)
            for title in context_titles[-3:]:
                if title:
                    words = title.lower().split()
                    # Skip common title words
                    skip_words = {'study', 'trial', 'analysis', 'review', 'patients', 'treatment', 'results', 'outcomes'}
                    key_terms = [w for w in words if len(w) > 4 and w not in skip_words]
                    context_expansion_terms.extend(key_terms[:3])  # Top 3 terms per title
            
            # Deduplicate and add to expanded query
            if context_expansion_terms:
                unique_terms = list(dict.fromkeys(context_expansion_terms))[:10]  # Max 10 context terms
                expanded_query = f"{expanded_query} {' '.join(unique_terms)}"
                print(f"[ComprehensiveRetrieval] Added {len(unique_terms)} context expansion terms: {unique_terms[:5]}...")
        
        print(f"[ComprehensiveRetrieval] Query type: {query_type}")
        print(f"[ComprehensiveRetrieval] Expanded: {expanded_query[:100]}...")

        # Get query structure for PostgreSQL matching
        print(f"\n{'─' * 80}")
        print(f"  STEP 2: Query Structuring (regex-based)")
        print(f"{'─' * 80}")
        query_structure = None
        try:
            from src.api.services.query_structuring_service import structure_query_fast, merge_query_structures
            query_structure = structure_query_fast(query_text, query_type)

            if accumulated_context:
                query_structure = merge_query_structures(accumulated_context, query_structure)
                print(f"  [Query Structure] Merged with accumulated context from conversation")

            print(f"  [Query Structure] has_patient_context: {query_structure.has_patient_context}")
            if query_structure.has_patient_context:
                try:
                    from src.api.services import pipeline_metrics as _pm
                    if _pm.current() is not None:
                        _pm.current().event("has_patient_context")
                except Exception:
                    pass
                print(f"  [Query Structure] Cancer site: {query_structure.cancer.site}")
                print(f"  [Query Structure] Cancer site_detail: {query_structure.cancer.site_detail}")
                print(f"  [Query Structure] Histology: {query_structure.cancer.histology}")
                print(f"  [Query Structure] Stage: {query_structure.cancer.stage}")
                print(f"  [Query Structure] TNM: {query_structure.cancer.get_tnm_string()}")
                print(f"  [Query Structure] Biomarkers: {query_structure.cancer.biomarkers}")
                print(f"  [Query Structure] Treatment modality: {query_structure.treatment.modality}")
                print(f"  [Query Structure] Treatment setting: {query_structure.treatment.setting}")
                print(f"  [Query Structure] Patient age: {query_structure.patient.age}")
                print(f"  [Query Structure] Patient gender: {query_structure.patient.gender}")
                print(f"  [Query Structure] Comorbidities: {query_structure.patient.comorbidities}")
                print(f"  [Query Structure] Boost terms: {query_structure.boost_terms[:8] if query_structure.boost_terms else []}")
                print(f"  [Query Structure] Filter category: {query_structure.filter_category}")
            else:
                print(f"  [Query Structure] No patient context detected (generic query)")
        except Exception as e:
            print(f"  [Query Structure] FAILED: {e}")

        # Expand staging notation (TNM ↔ Stage Group) pre-retrieval
        # Must run AFTER query_structure is populated so we can use cancer site hint
        try:
            from src.api.services.staging_search_expander import expand_query_with_staging
            cancer_type_hint = None
            if query_structure and query_structure.cancer.site:
                cancer_type_hint = query_structure.cancer.site
            staging_terms = expand_query_with_staging(query_text, cancer_type=cancer_type_hint)
            if staging_terms.all_search_terms:
                existing_lower = set(expanded_query.lower().split())
                new_staging = [t for t in staging_terms.all_search_terms
                               if t.lower() not in existing_lower]
                if new_staging:
                    expanded_query = expanded_query + " " + " ".join(new_staging[:8])
                    print(f"[ComprehensiveRetrieval] Staging expansion: +{len(new_staging[:8])} variants: {new_staging[:4]}")
        except Exception as e:
            print(f"[ComprehensiveRetrieval] Staging expansion failed: {e}")

        # ── Task 1: Trigger LLM extraction for complex queries ────────────
        print(f"\n{'─' * 80}")
        print(f"  STEP 3: Complexity Gate + LLM Extraction")
        print(f"{'─' * 80}")
        inferred_axes = None
        _len_check = len(query_text) > 150
        _comma_check = query_text.count(',') > 4
        _keyword_hits = [t for t in [
            'progression', 'refractory', 'metastatic', 'recurrent',
            'pembrolizumab', 'nivolumab', 'ici', 's/p', 'status post', 'pmh',
            'ilo', 'locoregional', 'cardiac', 'ventricle',
        ] if t in query_text.lower()]
        is_complex = _len_check or _comma_check or bool(_keyword_hits)

        # ── Perf optimization: skip LLM extraction for low patient signal ──
        _signal_score = None
        _signal_skip = False
        if is_complex and settings.enable_perf_optimizations and query_structure:
            try:
                from src.api.services.query_structuring_service import _patient_signal_score
                _signal_score = _patient_signal_score(query_text, query_structure)
                if _signal_score < 2:
                    _signal_skip = True
                    is_complex = False
                    print(f"  [PerfOpt] _patient_signal_score={_signal_score} < 2 — skipping LLM extraction")
            except Exception as e:
                print(f"  [PerfOpt] Signal score check failed (continuing): {e}")

        print(f"  [Complexity Gate] Length > 150: {_len_check} (len={len(query_text)})")
        print(f"  [Complexity Gate] Commas > 4: {_comma_check} (count={query_text.count(',')})")
        print(f"  [Complexity Gate] Keyword hits: {_keyword_hits if _keyword_hits else 'none'}")
        if _signal_score is not None:
            print(f"  [Complexity Gate] Patient signal score: {_signal_score} (threshold=2, skip={_signal_skip})")
        print(f"  [Complexity Gate] RESULT: {'COMPLEX → trigger LLM' if is_complex else 'SIMPLE → skip LLM'}")

        if is_complex and query_structure:
            try:
                from src.api.services.query_structuring_service import (
                    structure_query_with_llm, merge_llm_extraction,
                )
                print(f"  [LLM Extraction] Calling structure_query_with_llm()...")
                llm_result = await structure_query_with_llm(query_text)
                if llm_result:
                    query_structure = merge_llm_extraction(query_structure, llm_result)
                    query_structure.used_llm_extraction = True
                    print(f"  [LLM Extraction] SUCCESS — extracted axes:")
                    for axis_name, axis_value in llm_result.items():
                        if axis_value:
                            print(f"    {axis_name}: {str(axis_value)[:100]}{'...' if len(str(axis_value)) > 100 else ''}")
                else:
                    print(f"  [LLM Extraction] Returned empty result")
            except Exception as e:
                print(f"  [LLM Extraction] FAILED (continuing without): {e}")
        elif not query_structure:
            print(f"  [LLM Extraction] Skipped — no query_structure available")

        # ── Reconciliation + Canonicalization ─────────────────────────────
        # Build ReconciledStructure from regex + LLM, then run
        # canonicalization (biomarker, cancer type, stage) when enabled.
        reconciled = None
        if query_structure and query_structure.has_patient_context:
            try:
                from src.api.services.query_reconciliation import reconcile_if_enabled
                llm_dict = getattr(query_structure, "_llm_axes", None) or {}
                reconciled = reconcile_if_enabled(query_structure, llm_dict)
                if reconciled is not None:
                    print(f"[Canon] ReconciledStructure built: site={reconciled.cancer_site} "
                          f"stage={reconciled.stage} biomarkers={len(reconciled.biomarkers)}")
                else:
                    print(f"[Canon] Reconciliation skipped (USE_RECONCILED_STRUCTURE=false)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Canon] Reconciliation failed (continuing without): {e}")

        # Canonicalization layer — gated behind settings.enable_canonicalization
        if reconciled is not None and settings.enable_canonicalization:
            try:
                from src.api.services.biomarker_canonicalizer import BiomarkerCanonicalizer
                from src.api.services.cancer_type_canonicalizer import CancerTypeCanonicalizer
                from src.api.services.stage_canonicalizer import StageCanonicalizer

                # 1. Biomarker canonicalization
                bio_canon = BiomarkerCanonicalizer()
                biomarker_tuples = [
                    (bm.name, bm.polarity) for bm in reconciled.biomarkers
                ]
                canonical_biomarkers = bio_canon.resolve_list(
                    biomarker_tuples, raw_text=query_text, source="reconciled"
                )
                reconciled.canonical_biomarkers = canonical_biomarkers
                print(f"[Canon] Biomarkers canonicalized: "
                      f"{[(cb.canonical_id, cb.polarity) for cb in canonical_biomarkers]}")

                # 2. Cancer type canonicalization
                ct_canon = CancerTypeCanonicalizer()
                canonical_cancer_type = ct_canon.canonicalize(reconciled)
                reconciled.canonical_cancer_type = canonical_cancer_type
                print(f"[Canon] Cancer type canonicalized: site={canonical_cancer_type.site} "
                      f"histology={canonical_cancer_type.histology} "
                      f"category={canonical_cancer_type.category}")

                # 3. Stage canonicalization
                stage_canon = StageCanonicalizer()
                canonical_stage, stage_history = stage_canon.canonicalize(
                    reconciled, query_text
                )
                reconciled.canonical_stage = canonical_stage
                reconciled.stage_history = stage_history
                print(f"[Canon] Stage canonicalized: tnm={canonical_stage.tnm_string()} "
                      f"group={canonical_stage.stage_group} "
                      f"type={canonical_stage.staging_type} "
                      f"recurrent={stage_history.is_recurrent}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Canon] Canonicalization failed (continuing without): {e}")
        elif reconciled is not None:
            print(f"[Canon] Canonicalization skipped (enable_canonicalization=false)")

        # ── Task 6: Ontology inference layer ──────────────────────────────
        print(f"\n{'─' * 80}")
        print(f"  STEP 4: Ontology Inference Layer")
        print(f"{'─' * 80}")
        if query_structure and query_structure.has_patient_context:
            try:
                from src.api.services.clinical_inference import apply_inference_to_query_structure
                inferred_axes = apply_inference_to_query_structure(query_structure, query_text)
                print(f"  [Inference] Trajectory flags: {inferred_axes.get('trajectory_flags', [])}")
                print(f"  [Inference] Metastatic sites: {inferred_axes.get('metastatic_sites', [])}")
                print(f"  [Inference] Surgical candidate: {inferred_axes.get('surgical_candidate')}")
                inferred_terms = inferred_axes.get('inferred_terms', {})
                total_inferred = sum(len(v) for v in inferred_terms.values())
                print(f"  [Inference] Total inferred terms: {total_inferred} across {sum(1 for v in inferred_terms.values() if v)} axes")
                for axis_name, terms in inferred_terms.items():
                    if terms:
                        print(f"    {axis_name}: +{len(terms)} terms → {terms[:6]}{'...' if len(terms) > 6 else ''}")
                expanded_axes = inferred_axes.get('expanded_axes', {})
                if expanded_axes:
                    print(f"  [Inference] Expanded axes (first 80 chars each):")
                    for ax_name, ax_val in expanded_axes.items():
                        if ax_val:
                            print(f"    {ax_name}: {ax_val[:80]}{'...' if len(ax_val) > 80 else ''}")
            except Exception as e:
                print(f"  [Inference] FAILED (continuing without): {e}")
        else:
            print(f"  [Inference] Skipped — no patient context in query_structure")

        # ── Fix E: Query-time keyword resolver ─────────────────────────────
        # Scan the raw query against every JSON ontology (cancer_type,
        # trial_ontology, extractor_keywords, ajcc_staging_tables) to get
        # canonical-label detections. Feeds typed `should` clauses in the
        # Qdrant filter, hint-seeds the Postgres matcher, and fills gaps in
        # the eligibility patient context.
        resolved_tokens = None
        try:
            from src.api.services.query_token_resolver import resolve_query_tokens
            resolved_tokens = resolve_query_tokens(query_text)
            n_typed = sum(
                1 for v in resolved_tokens.as_typed_filters().values() if v
            )
            print(
                f"  [TokenResolver] cancer={sorted(resolved_tokens.cancer_types)} "
                f"site={sorted(resolved_tokens.sites)} "
                f"hist={sorted(resolved_tokens.histologies)} "
                f"stage={sorted(resolved_tokens.stages)} "
                f"bm={sorted(resolved_tokens.biomarkers)} "
                f"drug={sorted(resolved_tokens.drugs)} "
                f"typed_filters={n_typed} categories"
            )
        except Exception as e:
            print(f"  [TokenResolver] FAILED (continuing without): {e}")

        # Apply comprehensive ontology / drug / staging / clinical-context
        # expansion on top of the core expand_query() output so the
        # embedding vector covers ALL known synonym / abbreviation / brand-
        # name / staging-notation variants of the query terms.
        try:
            from src.api.services.query_expansion import expand_query_comprehensive
            expanded_query = expand_query_comprehensive(expanded_query)
        except Exception as e:
            print(f"  [Expansion] Comprehensive expansion failed (continuing): {e}")

        # Generate embedding for expanded query (now includes context terms)
        print(f"\n{'─' * 80}")
        print(f"  STEP 5: Embed Expanded Query")
        print(f"{'─' * 80}")
        print(f"  [Embedding] Model: {self._embed_model}")
        print(f"  [Embedding] Input length: {len(expanded_query)} chars")
        print(f"  [Embedding] Input preview: {expanded_query[:150]}{'...' if len(expanded_query) > 150 else ''}")
        t_embed = time.perf_counter()
        # Threaded via _embed_async — a direct embed_query() call here
        # blocks the event loop (and every other in-flight request on
        # this single-worker process) for the full OpenAI round-trip.
        query_embedding = await self._embed_async(expanded_query)
        embed_ms = (time.perf_counter() - t_embed) * 1000
        print(f"  [Embedding] Done in {embed_ms:.0f}ms → vector dim={len(query_embedding)}")

        # =====================================================
        # EAGER DISPATCH: Three-source intake with Phase 3 firing
        # as each source completes (Tasks 2, 3, 4, 7)
        # =====================================================
        print(f"\n{'─' * 80}")
        print(f"  STEP 6: Three-Source Parallel Retrieval + Eager Phase 3 Dispatch")
        print(f"{'─' * 80}")
        print(f"  Sources: Qdrant (vector) | PostgreSQL (structured) | PTO (frame index)")
        print(f"  Thresholds: qdrant=0.50, postgres=0.35, pto=0.28, both=0.28")

        # Registry — keyed by doc_id to prevent duplicate Phase 3 tasks
        phase3_registry: Dict[str, asyncio.Task] = {}
        doc_info: Dict[str, Dict[str, Any]] = {}

        # Source → cross-encoder threshold for the Phase 3 gate (Task 4)
        SOURCE_THRESHOLDS = {
            "pto":      0.28,
            "postgres": 0.35,
            "both":     0.28,
            "qdrant":   0.50,
        }
        # Source → reserved confirmation slots (out of max_studies)
        SOURCE_RESERVED = {
            "pto":      2,
            "postgres": 2,
            "qdrant":   0,  # fills remaining slots
        }

        # Resolve query-level category once for the pre-filter
        _query_filter_category = (
            getattr(query_structure, "filter_category", None)
            if query_structure is not None
            else None
        )

        def dispatch_phase3(doc_id: str, source: str, score: float,
                            doc_meta: dict, doc_category=None):
            """Dispatch a Phase 3 task if not already running for this doc_id."""
            # ── Early site-category pre-filter (Fix 5 / Task 7) ──────
            # Skip off-site candidates before expensive cross-encoder +
            # LLM calls.  On-site and unknown-category candidates always
            # pass through.
            if should_skip_phase3_candidate(_query_filter_category, doc_category):
                print(
                    f"[Phase3 PreFilter] Skipping doc_id={doc_id[:40]} "
                    f"category={doc_category!r} "
                    f"(query_cat={_query_filter_category!r})"
                )
                return

            if doc_id in phase3_registry:
                # Upgrade trust level if new source is higher precision
                existing_source = doc_info[doc_id]["source"]
                precision_rank = {"qdrant": 0, "postgres": 1, "pto": 2, "both": 3}
                if precision_rank.get(source, 0) > precision_rank.get(existing_source, 0):
                    doc_info[doc_id]["source"] = source
                    doc_info[doc_id]["threshold"] = SOURCE_THRESHOLDS[source]
                    print(f"[EagerDispatch] Upgraded {doc_id[:40]} trust: "
                          f"{existing_source} -> {source}")
                return

            doc_info[doc_id] = {
                "source": source,
                "score": score,
                "threshold": SOURCE_THRESHOLDS.get(source, 0.45),
                "doc_meta": doc_meta,
                "category": doc_category,
            }
            elapsed_since_start = (time.perf_counter() - t_start) * 1000
            task = asyncio.create_task(
                self._tagged_phase3(
                    doc_id=doc_id,
                    query_embedding=query_embedding,
                    expanded_query=expanded_query,
                    max_chunks=chunks_per_study,
                    query_type=query_type,
                    doc_info_entry=doc_info[doc_id],
                    query_structure=query_structure,
                    inferred_axes=inferred_axes,
                    query_text=original_query_text,
                )
            )
            phase3_registry[doc_id] = task
            if len(phase3_registry) <= 3:  # Log first few dispatches
                print(f"[EagerDispatch] Phase3 dispatched for {doc_id[:30]}... "
                      f"(source={source}, t={elapsed_since_start:.0f}ms)")

        # Set up PostgreSQL task
        postgres_task = None
        _pg_cache_hit = False
        _pg_cached_result = None
        if query_structure:
            try:
                from src.api.services.structured_study_matcher import match_studies_by_structure
                pg_limit = 30 if settings.enable_perf_optimizations else 50
                qs_dict = query_structure.to_dict()

                # Check PG cache when perf optimizations are enabled
                if settings.enable_perf_optimizations:
                    cache_key = _pg_cache_key(qs_dict)
                    if cache_key in _pg_cache:
                        _pg_cached_result = _pg_cache[cache_key]
                        _pg_cache_hit = True
                        print(f"[PGCache] HIT — reusing cached PG result "
                              f"({len(_pg_cached_result.doc_ids)} docs)")

                if not _pg_cache_hit:
                    if settings.enable_perf_optimizations:
                        print(f"[PGCache] MISS — running PG matcher")
                    postgres_task = match_studies_by_structure(
                        qs_dict,
                        limit=pg_limit,
                        resolver_hints=resolved_tokens,
                    )
            except Exception as e:
                print(f"[ComprehensiveRetrieval] PostgreSQL setup failed: {e}")

        qdrant_doc_count = 0
        postgres_doc_ids: Set[str] = set()
        pto_doc_ids: Set[str] = set()
        pg_match_details: Dict[str, Dict[str, Any]] = {}  # doc_id -> match_details from PG matcher

        async def qdrant_and_dispatch():
            nonlocal qdrant_doc_count
            t0 = time.perf_counter()
            qdrant_limit = 50 if settings.enable_perf_optimizations else 100
            hits = await self._phase1_qdrant_search(
                query_embedding=query_embedding,
                expanded_query=expanded_query,
                category=category,
                limit=qdrant_limit,
                query_structure=query_structure,
                inferred_axes=inferred_axes,
                resolved_tokens=resolved_tokens,
                clinical_profile=clinical_profile,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            for hit in hits:
                did = hit.get("doc_id")
                if did:
                    dispatch_phase3(did, "qdrant", hit.get("score", 0),
                                    hit.get("doc_meta", {}), hit.get("category"))
            qdrant_doc_count = len({h.get("doc_id") for h in hits if h.get("doc_id")})
            print(f"[ComprehensiveRetrieval] Qdrant: {qdrant_doc_count} unique docs "
                  f"({elapsed:.0f}ms, dispatched {qdrant_doc_count} Phase 3 tasks)")
            return hits

        async def postgres_and_dispatch():
            nonlocal postgres_doc_ids, pg_match_details
            if not postgres_task and not _pg_cached_result:
                return None
            t0 = time.perf_counter()

            # Use cached result or await the live task
            if _pg_cached_result:
                result = _pg_cached_result
                elapsed = 0.0
            else:
                result = await postgres_task
                elapsed = (time.perf_counter() - t0) * 1000

                # Store in cache when perf optimizations are enabled
                if settings.enable_perf_optimizations and result and result.doc_ids:
                    try:
                        qs_dict = query_structure.to_dict()
                        cache_key = _pg_cache_key(qs_dict)
                        if len(_pg_cache) >= _PG_CACHE_MAX_SIZE:
                            _pg_cache.clear()
                            print(f"[PGCache] Evicted all entries (exceeded max size {_PG_CACHE_MAX_SIZE})")
                        _pg_cache[cache_key] = result
                        print(f"[PGCache] Stored result ({len(result.doc_ids)} docs)")
                    except Exception as e:
                        print(f"[PGCache] Failed to store result: {e}")
            if not result or not result.doc_ids:
                print(f"[ComprehensiveRetrieval] PostgreSQL: 0 docs ({elapsed:.0f}ms)")
                return result
            postgres_doc_ids = set(result.doc_ids)
            # Capture PG match details for later attachment to StudyEvidence
            if hasattr(result, 'match_details') and result.match_details:
                pg_match_details = result.match_details
            both_count = 0
            pg_only_count = 0
            for pg_doc_id in postgres_doc_ids:
                pg_score = result.match_scores.get(pg_doc_id, 0.5) if hasattr(result, 'match_scores') else 0.5
                source = "both" if pg_doc_id in doc_info else "postgres"
                if source == "both":
                    both_count += 1
                else:
                    pg_only_count += 1
                dispatch_phase3(pg_doc_id, source, pg_score, {})
            print(f"[ComprehensiveRetrieval] PostgreSQL: {len(postgres_doc_ids)} docs "
                  f"({elapsed:.0f}ms) → {both_count} upgraded to 'both', "
                  f"{pg_only_count} postgres-only")
            return result

        async def pto_and_dispatch():
            nonlocal pto_doc_ids
            t0 = time.perf_counter()
            pto_ids = await self._phase0_pto_search(
                query_text=query_text,
                query_embedding=query_embedding,
                query_structure=query_structure,
                inferred_axes=inferred_axes,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            pto_doc_ids = pto_ids
            upgraded_count = 0
            pto_only_count = 0
            for pto_doc_id in pto_ids:
                if pto_doc_id in doc_info:
                    # Doc already found by Qdrant or PG — upgrade to "both"
                    # so it gets the lower cross-encoder threshold
                    prev_source = doc_info[pto_doc_id]["source"]
                    doc_info[pto_doc_id]["source"] = "both"
                    doc_info[pto_doc_id]["threshold"] = SOURCE_THRESHOLDS["both"]
                    upgraded_count += 1
                    print(f"[PTO] Upgraded {pto_doc_id[:40]} source: "
                          f"{prev_source} -> both (threshold={SOURCE_THRESHOLDS['both']})")
                    # Still call dispatch_phase3 with "both" so precision_rank
                    # logic is consistent (it will see the doc already registered
                    # and skip creating a new task)
                    dispatch_phase3(pto_doc_id, "both", 0.65, {})
                else:
                    # New doc only found by PTO
                    pto_only_count += 1
                    dispatch_phase3(pto_doc_id, "pto", 0.65, {})
            print(f"[PTO] {len(pto_ids)} doc_ids ({elapsed:.0f}ms) → "
                  f"{upgraded_count} upgraded to 'both', "
                  f"{pto_only_count} pto-only")
            return pto_ids

        # Fire all three — Phase 3 tasks start mid-gather as sources return
        print(f"[ComprehensiveRetrieval] Phase 1: Three-source parallel retrieval...")
        await asyncio.gather(
            qdrant_and_dispatch(),
            postgres_and_dispatch(),
            pto_and_dispatch(),
        )

        # Apply conversation context boosting to doc_info
        if context_doc_ids:
            boosted_count = 0
            for did in doc_info:
                if did in context_doc_ids:
                    doc_info[did]["score"] += 0.15
                    doc_info[did]["from_context"] = True
                    boosted_count += 1
            if boosted_count > 0:
                print(f"[ComprehensiveRetrieval] Boosted {boosted_count} docs from conversation context")

        print(f"[ComprehensiveRetrieval] Total unique docs: {len(doc_info)}")
        print(f"[ComprehensiveRetrieval] Phase 3 tasks dispatched: {len(phase3_registry)}")

        # =====================================================
        # PRIORITY QUEUE: Collect Phase 3 results with reserved
        # slots per source (Task 7)
        # =====================================================
        print(f"\n{'─' * 80}")
        print(f"  STEP 7: Priority Queue — Collecting Phase 3 Results")
        print(f"{'─' * 80}")
        print(f"  Reserved slots: pto=2, postgres=2, qdrant=fills remaining (max_studies={max_studies})")
        confirmed_by_source: Dict[str, List[StudyEvidence]] = {
            "pto": [], "postgres": [], "both": [], "qdrant": [],
        }
        rejected: Set[str] = set()

        for coro in asyncio.as_completed(list(phase3_registry.values())):
            try:
                doc_id, chunks = await coro
            except asyncio.CancelledError:
                continue
            except Exception as e:
                print(f"[EagerDispatch] Phase 3 exception: {e}")
                continue

            if not chunks:
                # Empty chunks = gate rejected in _phase3_document_search (Task 4)
                rejected.add(doc_id)
                info = doc_info.get(doc_id, {})
                print(f"  [PriorityQueue] REJECTED: {doc_id[:40]}... (source={info.get('source', '?')}, gate filtered)")
                continue

            info = doc_info.get(doc_id, {})
            source = info.get("source", "qdrant")

            sections = {c.get("section") for c in chunks if c.get("section")}
            doc_meta = chunks[0].get("doc_meta", {}) if chunks else info.get("doc_meta", {})

            study = StudyEvidence(
                doc_id=doc_id,
                title=doc_meta.get("title", "Unknown"),
                citation=doc_meta.get("citation"),
                year=doc_meta.get("year"),
                category=info.get("category") or (chunks[0].get("category") if chunks else None),
                initial_score=info.get("score", 0),
                chunks=chunks,
                sections_covered=sections,
                source=source,
            )

            # Attach PG match breakdown if available for this study
            pg_detail = pg_match_details.get(doc_id)
            if pg_detail:
                study.match_score = pg_detail.get("raw_score")
                study.match_breakdown = pg_detail.get("match_breakdown")
                mismatches = pg_detail.get("match_breakdown", {}).get("mismatches", [])
                if mismatches:
                    study.axis_mismatches = mismatches

            # Patient–study match score (v1: weighted overlap of
            # ClinicalProfile axes against doc_level_* metadata).
            if clinical_profile is not None and clinical_profile.has_any_filter():
                doc_level_md = (
                    (chunks[0].get("metadata") or {}) if chunks else {}
                )
                try:
                    from src.api.services.patient_match_scorer import score_patient_match
                    pm = score_patient_match(clinical_profile, doc_level_md)
                    study.patient_match_score = pm["score"]
                    study.patient_match_breakdown = pm
                except Exception as e:
                    print(f"[PatientMatch] score failed for {doc_id[:40]}: {e}")

            # Classify evidence type (guideline / landmark_trial / trial)
            # so retrieve_comprehensive can apply separate budgets to
            # each lane instead of letting guidelines crowd out patient-
            # specific evidence.
            try:
                from src.api.services.evidence_classifier import classify_study
                study.evidence_type = classify_study(
                    title=study.title,
                    citation=study.citation,
                    chunk_texts=[c.get("text", "") for c in chunks],
                )
            except Exception as e:
                print(f"[EvidenceClassifier] failed for {doc_id[:40]}: {e}")

            lane = "both" if source == "both" else source if source in ("pto", "postgres") else "qdrant"
            confirmed_by_source[lane].append(study)
            try:
                from src.api.services import pipeline_metrics as _pm
                _pm.incr("source_counts", lane)
            except Exception:
                pass
            doc_meta_title = doc_meta.get("title", study.title or "?")
            print(f"  [PriorityQueue] CONFIRMED: {doc_id[:40]}... → lane={lane}, "
                  f"chunks={len(chunks)}, sections={len(sections)}, "
                  f"title={doc_meta_title[:50]}{'...' if len(doc_meta_title) > 50 else ''}")

            # ── Early termination: cancel Qdrant-only tasks if we have enough ──
            total_confirmed = sum(len(v) for v in confirmed_by_source.values())
            high_precision_confirmed = (
                len(confirmed_by_source["pto"])
                + len(confirmed_by_source["both"])
                + len(confirmed_by_source["postgres"])
            )

            if total_confirmed >= max_studies and high_precision_confirmed >= 2:
                for did, task in phase3_registry.items():
                    if (
                        did not in rejected
                        and not any(did == s.doc_id for lane_studies in confirmed_by_source.values() for s in lane_studies)
                        and doc_info.get(did, {}).get("source") == "qdrant"
                        and not task.done()
                    ):
                        task.cancel()
                break

        # ── Merge lanes into final ordered list ──────────────────────────────
        print(f"\n{'─' * 80}")
        print(f"  STEP 8: Lane Merge + Final Assembly")
        print(f"{'─' * 80}")
        print(f"  Lane counts: pto={len(confirmed_by_source['pto'])}, "
              f"both={len(confirmed_by_source['both'])}, "
              f"postgres={len(confirmed_by_source['postgres'])}, "
              f"qdrant={len(confirmed_by_source['qdrant'])}")
        print(f"  Rejected by cross-encoder gate: {len(rejected)}")

        # Reserved slots: pto (2) + postgres (2) + qdrant fills remaining
        # Deduplicate across lanes by doc_id AND by title+year (catches
        # the same physical study ingested under two different doc_ids).
        def _extract_nct_number(study: StudyEvidence) -> Optional[str]:
            """Extract NCT number from study chunks' doc_meta."""
            for chunk in (study.chunks or []):
                doc_meta = chunk.get("doc_meta", {})
                # Check trial_info.nct_number (ingested documents)
                nct = (doc_meta.get("trial_info") or {}).get("nct_number")
                if nct and str(nct).strip():
                    return str(nct).strip().upper()
                # Check top-level nct_id (online-sourced studies)
                nct = doc_meta.get("nct_id")
                if nct and str(nct).strip():
                    return str(nct).strip().upper()
            return None

        def _dedup_studies(study_list: List[StudyEvidence], seen: set) -> List[StudyEvidence]:
            """Return studies not already in seen, tracking by doc_id, title fingerprint, and NCT number."""
            result = []
            for s in study_list:
                # Primary key: doc_id
                if s.doc_id in seen:
                    continue
                # Secondary key: normalised title + year (catches duplicate ingestions)
                title_fp = (s.title or "").strip().lower()[:80]
                year_fp = s.year or 0
                fingerprint = f"{title_fp}|{year_fp}"
                if fingerprint in seen:
                    print(f"  [Dedup] Duplicate study removed: {s.title[:50]}... "
                          f"(doc_id={s.doc_id[:30]}, already seen under different doc_id)")
                    continue
                # Tertiary key: NCT number (catches same trial ingested with different titles/doc_ids)
                nct = _extract_nct_number(s)
                if nct:
                    nct_key = f"nct|{nct}"
                    if nct_key in seen:
                        print(f"  [Dedup] Duplicate study removed by NCT number: {nct} "
                              f"(doc_id={s.doc_id[:30]}, title={s.title[:50]})")
                        continue
                    seen.add(nct_key)
                seen.add(s.doc_id)
                seen.add(fingerprint)
                result.append(s)
            return result

        seen_studies: set = set()
        pto_studies = _dedup_studies(
            confirmed_by_source["pto"][:2] + confirmed_by_source["both"][:2],
            seen_studies,
        )
        postgres_studies = _dedup_studies(
            confirmed_by_source["postgres"][:2],
            seen_studies,
        )
        qdrant_studies = _dedup_studies(
            confirmed_by_source["qdrant"],
            seen_studies,
        )

        reserved_used = len(pto_studies) + len(postgres_studies)
        remaining_slots = max(0, max_studies - reserved_used)

        studies = pto_studies + postgres_studies + qdrant_studies[:remaining_slots]

        print(f"  [Lane Merge] PTO/both reserved: {len(pto_studies)} studies")
        print(f"  [Lane Merge] Postgres reserved: {len(postgres_studies)} studies")
        print(f"  [Lane Merge] Qdrant fills: {min(len(qdrant_studies), remaining_slots)} of {remaining_slots} remaining slots")
        print(f"  [Lane Merge] Total studies after merge: {len(studies)}")

        # =====================================================
        # PHASE 6: Free reranking (sort by score already computed)
        # =====================================================
        print(f"\n{'─' * 80}")
        print(f"  STEP 9: Final Reranking")
        print(f"{'─' * 80}")
        # Use cross-encoder gate scores from Phase 3 if available,
        # otherwise fall back to Phase 4 reranking
        has_gate_scores = any(
            any(c.get("score_crossencoder_gate") is not None for c in s.chunks)
            for s in studies if s.chunks
        )
        if has_gate_scores:
            # Free reranking — use the gate score already computed in Phase 3
            print(f"  [Reranking] Using FREE reranking (Phase 3 gate scores already computed)")
            for study in studies:
                gate_scores = [c.get("score_crossencoder_gate", 0) for c in study.chunks
                               if c.get("score_crossencoder_gate") is not None]
                study.rerank_score = max(gate_scores) if gate_scores else study.initial_score
            studies.sort(key=lambda s: s.rerank_score, reverse=True)
        else:
            # Fall back to Phase 4 cross-encoder reranking. Pull a
            # generous cap here; the two-track budgeting below applies
            # the real caps per lane.
            print(f"  [Reranking] Using Phase 4 cross-encoder reranking (no gate scores available)")
            studies = await self._phase4_rerank_studies(
                studies=studies,
                query_text=original_query_text,
                max_studies=max_studies + max_guidelines,
                query_structure=query_structure,
            )

        # ─── Two-track budgeting ─────────────────────────────────
        # Split reranked studies into guideline / landmark_trial /
        # trial buckets (populated on each study during the Phase 3
        # StudyEvidence build). Apply separate caps so guidelines
        # can't crowd out patient-specific trials.
        from src.api.services.evidence_classifier import bucket_studies
        buckets = bucket_studies(studies)
        guideline_lane = buckets.get("guideline", []) + buckets.get("landmark_trial", [])
        trial_lane = buckets.get("trial", [])

        guideline_lane = guideline_lane[:max_guidelines]
        trial_lane = trial_lane[:max_studies]

        print(
            f"  [Two-Track] trial lane: {len(trial_lane)}/{max_studies} | "
            f"guideline lane: {len(guideline_lane)}/{max_guidelines} "
            f"(guidelines={len(buckets.get('guideline', []))}, "
            f"landmarks={len(buckets.get('landmark_trial', []))})"
        )

        # Patient-specific trials first so the synthesis prompt leads
        # with them; guidelines follow for reference/standard-of-care.
        studies = trial_lane + guideline_lane

        print(f"  [Reranking] Final study ranking:")
        for i, study in enumerate(studies):
            print(f"    {i+1}. [{study.rerank_score:.3f}] {study.title[:60]}{'...' if len(study.title) > 60 else ''} "
                  f"(source={study.source}, chunks={len(study.chunks)}, sections={list(study.sections_covered)[:3]})")

        # =====================================================
        # HARD ELIGIBILITY FILTER: Remove studies that clearly
        # don't match the patient on cancer type, histology,
        # stage, prior therapies, or biomarkers.
        # =====================================================
        pre_elig = len(studies)
        try:
            from src.api.services.patient_eligibility_boost_service import run_patient_eligibility_check

            # Convert StudyEvidence → flat chunk dicts for the eligibility service
            elig_chunks = []
            for s in studies:
                best_text = ""
                for c in s.chunks[:3]:
                    best_text += " " + (c.get("text") or "")
                elig_chunks.append({
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "text": best_text.strip()[:1500],
                    "score": s.rerank_score,
                })

            # Reuse the retriever's shared OpenAI client — previously a
            # fresh client was constructed here on every query.
            filtered_elig_chunks, elig_meta = await run_patient_eligibility_check(
                query=query_text,
                chunks=elig_chunks,
                openai_client=self.openai,
                resolver_hints=resolved_tokens,
            )

            if elig_meta.get("patient_context_detected"):
                # Identify which doc_ids survived the filter
                surviving_ids = {
                    c.get("doc_id") for c in filtered_elig_chunks
                }
                studies = [s for s in studies if s.doc_id in surviving_ids]
                removed = pre_elig - len(studies)
                if removed:
                    print(
                        f"  [HardEligibility] Removed {removed} studies "
                        f"({pre_elig} → {len(studies)})"
                    )
                else:
                    print(f"  [HardEligibility] All {len(studies)} studies passed")

                # ── Soft Scoring (Phase 4) ────────────────────────────
                # After hard eligibility, apply SoftScorer to surviving
                # studies using per-axis verdicts from the eligibility
                # check.  Gated behind settings.enable_soft_scorer.
                if settings.enable_soft_scorer:
                    try:
                        from src.api.services.soft_scorer import SoftScorer

                        # Build doc_id → criteria_verdicts lookup from
                        # the filtered eligibility chunks.
                        doc_verdicts: Dict[str, Dict[str, str]] = {}
                        for ec in filtered_elig_chunks:
                            did = ec.get("doc_id")
                            if did:
                                pe = ec.get("patient_eligibility", {})
                                cv = pe.get("criteria_verdicts")
                                if cv:
                                    doc_verdicts[did] = cv

                        scorer = SoftScorer()
                        scored_count = 0
                        for study in studies:
                            verdicts = doc_verdicts.get(study.doc_id)
                            if verdicts:
                                result = scorer.score(study.doc_id, verdicts)
                                study.soft_score_normalized = result.normalized
                                scored_count += 1
                            else:
                                # No verdicts available — leave soft_score as None
                                print(f"[SoftScore] study={study.doc_id[:40]} skipped (no eligibility verdicts)")

                        # Re-rank using combined score: 70% cross-encoder + 30% soft score
                        # Studies without soft scores keep their original ranking position.
                        if scored_count > 0:
                            for study in studies:
                                if study.soft_score_normalized is not None:
                                    # Normalize soft score to 0-1 range for blending
                                    soft_norm = study.soft_score_normalized / 100.0
                                    study.rerank_score = (
                                        0.7 * study.rerank_score + 0.3 * soft_norm
                                    )
                            studies.sort(key=lambda s: s.rerank_score, reverse=True)
                            print(f"[SoftScore] Re-ranked {len(studies)} studies "
                                  f"({scored_count} scored, blended 70% cross-encoder + 30% soft)")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[SoftScore] Failed (continuing without): {e}")
                else:
                    print(f"[SoftScore] Skipped (enable_soft_scorer=false)")
            else:
                print(f"  [HardEligibility] No patient context — skipped")
        except Exception as e:
            print(f"  [HardEligibility] Failed (continuing without): {e}")

        # Calculate totals
        total_chunks = sum(len(s.chunks) for s in studies)
        retrieval_time_ms = (time.perf_counter() - t_start) * 1000

        print(f"\n{'=' * 80}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'=' * 80}")
        print(f"  Total studies: {len(studies)}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Total time: {retrieval_time_ms:.1f}ms")
        print(f"  Phase 1 Qdrant docs: {qdrant_doc_count}")
        print(f"  Phase 1 Postgres docs: {len(postgres_doc_ids)}")
        print(f"  Phase 1 PTO docs: {len(pto_doc_ids)}")
        print(f"  Phase 3 tasks dispatched: {len(phase3_registry)}")
        print(f"  Phase 3 rejected by gate: {len(rejected)}")
        print(f"{'=' * 80}\n")

        try:
            from src.api.services import pipeline_metrics as _pm
            _pm_cur = _pm.current()
            if _pm_cur is not None:
                print(_pm_cur.summary_line())
        except Exception:
            pass

        return ComprehensiveRetrievalResult(
            studies=studies,
            total_chunks=total_chunks,
            retrieval_time_ms=retrieval_time_ms,
            phase1_qdrant_docs=qdrant_doc_count,
            phase1_postgres_docs=len(postgres_doc_ids),
            phase2_docs_searched=len(phase3_registry),
            query_structure=query_structure.to_dict() if query_structure else None,
            expanded_query=expanded_query,
            reconciled_structure=reconciled,
        )
    
    async def _phase1_qdrant_search(
        self,
        query_embedding: List[float],
        expanded_query: str,
        category: Optional[str],
        limit: int,
        query_structure=None,
        inferred_axes: Optional[Dict[str, Any]] = None,
        resolved_tokens: Optional[ResolvedQueryTokens] = None,
        clinical_profile: Optional[ClinicalProfile] = None,
    ) -> List[Dict[str, Any]]:
        """Phase 1: Initial Qdrant vector search (offloaded to thread pool).

        Two modes:
          A. Hard-filter mode — triggered when ``clinical_profile`` is
             passed in with at least one populated axis. Cancer type,
             site, histology, stage, biomarker, and prior-treatment axes
             become ``must`` clauses against the canonical
             ``metadata.*_detected`` payload fields. If the strict filter
             returns 0 hits, we fall back to the soft-filter mode below
             so the pipeline degrades gracefully when extraction missed.
          B. Soft-filter mode — the original behavior: all clinical
             axes are ``should`` clauses that boost ranking without
             dropping anything.

        Common to both:
          - ``must``: hard ``category`` match when present.
          - ``must_not``: negated typed-slot tokens (biomarker / drug /
            histology) so "no HER2 amplification" doesn't surface
            HER2+ studies.
        """
        flat_terms, typed_filters = _collect_structured_axes(
            query_structure, inferred_axes, resolved_tokens,
        )
        must_not_clauses = _build_must_not_clauses(resolved_tokens)
        category_clause = (
            [qm.FieldCondition(key="category", match=qm.MatchValue(value=category))]
            if category else []
        )

        # ── Mode A: Hard filter from ClinicalProfile ─────────────────
        if clinical_profile is not None and clinical_profile.has_any_filter():
            hard_must = _profile_to_must_clauses(clinical_profile)
            soft_should = _profile_to_should_clauses(clinical_profile)
            must_clauses = category_clause + hard_must
            flt_hard = qm.Filter(
                must=must_clauses or None,
                should=soft_should or None,  # disease_status boost only
                must_not=must_not_clauses or None,
            )
            axes_summary = ", ".join(
                f"{lbl}={cnt}" for lbl, cnt in _profile_axis_counts(clinical_profile).items() if cnt
            )
            print(
                f"[Phase1] HARD filter active: must=category={category!r} "
                f"+ axes=[{axes_summary}] | must_not={len(must_not_clauses)}"
            )
            results = await self._qdrant_query(
                query=query_embedding,
                limit=limit,
                query_filter=flt_hard,
                with_payload=True,
                with_vectors=False,
            )
            if results.points:
                return _materialize_hits(results)
            print(
                "[Phase1] Hard filter returned 0 hits — falling back to "
                "soft-boost mode so the pipeline degrades gracefully."
            )
            # fall through to soft mode below

        # ── Mode B: Soft filter (original) ───────────────────────────
        should_clauses: List[Any] = []
        for field_name, values in typed_filters.items():
            if values:
                should_clauses.append(
                    qm.FieldCondition(
                        key=f"metadata.{field_name}",
                        match=qm.MatchAny(any=values),
                    )
                )
        if flat_terms:
            should_clauses.append(
                qm.FieldCondition(
                    key="metadata.keywords_flat",
                    match=qm.MatchAny(any=flat_terms),
                )
            )

        flt = None
        if category_clause or should_clauses or must_not_clauses:
            flt = qm.Filter(
                must=category_clause or None,
                should=should_clauses or None,
                must_not=must_not_clauses or None,
            )

        if should_clauses or must_not_clauses:
            typed_summary = ", ".join(
                f"{fn}={len(vs)}" for fn, vs in typed_filters.items() if vs
            )
            print(
                f"[Phase1] SOFT filter: must=category={category!r} "
                f"| typed_should=[{typed_summary}] "
                f"| flat_should={len(flat_terms)} terms "
                f"| must_not={len(must_not_clauses)} neg-clauses"
            )

        results = await self._qdrant_query(
            query=query_embedding,
            limit=limit,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )

        return _materialize_hits(results)

    async def _phase0_pto_search(
        self,
        query_text: str,
        query_embedding: List[float],
        query_structure,
        inferred_axes: dict = None,
        limit: int = 30,
    ) -> Set[str]:
        """
        Phase 0: Search PTO frame index for document-level patient profile matches.

        When ``settings.enable_pto_retrieval`` is True, delegates to
        ``PTORetriever.hybrid_search()`` which combines PTO frame search with
        chunk retrieval and query routing.  Falls back to the legacy direct-
        Qdrant approach when the flag is off.

        Returns set of doc_ids whose PTO frame matched the patient profile.
        """
        # ── Gate: skip when PTO retrieval is disabled ─────────────────────
        if not settings.enable_pto_retrieval:
            return await self._phase0_pto_search_legacy(
                query_text=query_text,
                query_embedding=query_embedding,
                query_structure=query_structure,
                inferred_axes=inferred_axes,
                limit=limit,
            )

        try:
            doc_ids: Set[str] = set()

            # Resolve cancer type filter from query_structure
            cancer_filter = None
            if query_structure and hasattr(query_structure, "filter_category"):
                cancer_filter = query_structure.filter_category

            print(f"    [PTO] Starting PTORetriever.hybrid_search() "
                  f"(cancer_filter={cancer_filter})...")

            from src.api.services.pto_retriever import PTORetriever

            retriever = PTORetriever(
                qdrant_url=settings.qdrant_url,
                qdrant_api_key=settings.qdrant_api_key,
                collection_name=self.collection,
                openai_api_key=settings.openai_api_key,
            )

            # hybrid_search is synchronous — run in thread pool
            result = await self._run_sync(
                retriever.hybrid_search,
                query=query_text,
                pto_limit=min(limit, 10),
                chunk_limit=0,          # we only need PTO frames here
                expand_evidence=False,  # evidence expansion handled in Phase 3
            )

            # Extract doc_ids from PTO frames
            pto_frames = result.get("pto_frames", [])
            for frame in pto_frames:
                did = frame.get("doc_id")
                if did:
                    doc_ids.add(did)

            routing = result.get("routing_info", {})
            print(f"    [PTO] PTORetriever returned {len(pto_frames)} frames → "
                  f"{len(doc_ids)} unique doc_ids "
                  f"(should_use_pto={routing.get('should_use_pto')}, "
                  f"confidence={routing.get('confidence')})")

            if not doc_ids:
                print(f"    [PTO] No PTO doc_ids found. PTO frames may not be "
                      f"ingested in collection '{self.collection}'.")

            return doc_ids

        except Exception as e:
            print(f"    [PTO] PTORetriever.hybrid_search() FAILED (continuing without): {e}")
            import traceback
            traceback.print_exc()
            return set()

    async def _phase0_pto_search_legacy(
        self,
        query_text: str,
        query_embedding: List[float],
        query_structure,
        inferred_axes: dict = None,
        limit: int = 30,
    ) -> Set[str]:
        """Legacy PTO search using direct Qdrant queries (enable_pto_retrieval=false)."""
        try:
            doc_ids: Set[str] = set()
            print(f"    [PTO] Starting legacy PTO frame search (limit={limit})...")

            # ── Search 1: Full profile vector (pre-computed) ──────────────
            pto_filter = qm.Filter(must=[
                qm.FieldCondition(
                    key="node_type",
                    match=qm.MatchValue(value="pto_frame"),
                )
            ])

            search1_method = "filtered"
            try:
                profile_results = await self._qdrant_query(
                    query=query_embedding,
                    limit=limit,
                    query_filter=pto_filter,
                    with_payload=True,
                    with_vectors=False,
                )
                print(f"    [PTO] Search 1 (filtered by node_type): {len(profile_results.points)} points returned")
                for pt in profile_results.points:
                    did = (pt.payload or {}).get("doc_id")
                    if did:
                        doc_ids.add(did)
            except Exception as e:
                print(f"    [PTO] Search 1 filtered failed: {e}")
                print(f"    [PTO] Falling back to unfiltered search + post-filter...")
                search1_method = "unfiltered_fallback"
                try:
                    fallback_limit = limit * 3 if settings.enable_perf_optimizations else limit * 5
                    unfiltered = await self._qdrant_query(
                        query=query_embedding,
                        limit=fallback_limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                    pto_points = [
                        pt for pt in unfiltered.points
                        if (pt.payload or {}).get("node_type") == "pto_frame"
                    ]
                    print(f"    [PTO] Unfiltered fallback: {len(unfiltered.points)} total points, "
                          f"{len(pto_points)} with node_type=pto_frame")
                    for pt in pto_points[:limit]:
                        did = (pt.payload or {}).get("doc_id")
                        if did:
                            doc_ids.add(did)
                except Exception as e2:
                    print(f"    [PTO] Unfiltered fallback also failed: {e2}")

            if not doc_ids:
                print(f"    [PTO] Search 1 returned 0 doc_ids via {search1_method}. "
                      f"PTO frames may not be ingested in this collection.")

            # ── Search 2: Per-axis embeddings ─────────────────────────────
            expanded_axes = (inferred_axes or {}).get("expanded_axes", {})
            llm_axes = getattr(query_structure, "_llm_axes", None) if query_structure else None
            axes_to_search = expanded_axes or llm_axes or {}

            axis_queries = {
                k: v for k, v in axes_to_search.items()
                if v and len(v.strip()) > 10
            }

            priority_order = [
                "primary_cancer", "disease_trajectory", "current_treatment",
                "biomarker_profile", "metastatic_concern", "patient_factors",
                "tnm_pathology", "prior_definitive_treatment",
            ]
            selected = []
            for axis in priority_order:
                if axis in axis_queries and len(selected) < 4:
                    selected.append((axis, axis_queries[axis]))

            if selected:
                print(f"    [PTO] Search 2: {len(selected)} axis queries: {[s[0] for s in selected]}")
            else:
                print(f"    [PTO] Search 2: no axes available "
                      f"(expanded_axes={len(expanded_axes)}, llm_axes={'yes' if llm_axes else 'no'})")
            pre_axis_count = len(doc_ids)

            AXIS_TO_PTO_SECTION = {
                "primary_cancer": "pto_frame_patient",
                "biomarker_profile": "pto_frame_patient",
                "patient_factors": "pto_frame_patient",
                "disease_trajectory": "pto_frame_eligibility",
                "current_treatment": "pto_frame_treatment",
                "prior_definitive_treatment": "pto_frame_treatment",
                "metastatic_concern": "pto_frame_patient",
                "tnm_pathology": "pto_frame_patient",
            }

            for axis_name, axis_text in selected:
                try:
                    try:
                        from src.api.services.query_expansion import expand_query_comprehensive
                        axis_text_expanded = expand_query_comprehensive(axis_text)
                    except Exception:
                        axis_text_expanded = axis_text
                    axis_emb = await self._embed_async(axis_text_expanded)
                    section_type = AXIS_TO_PTO_SECTION.get(axis_name, "pto_frame_patient")
                    axis_filter = qm.Filter(
                        should=[
                            qm.FieldCondition(key="node_type", match=qm.MatchValue(value="pto_frame")),
                            qm.FieldCondition(key="node_type", match=qm.MatchValue(value=section_type)),
                        ]
                    )
                    try:
                        axis_results = await self._qdrant_query(
                            query=axis_emb,
                            limit=10,
                            query_filter=axis_filter,
                            with_payload=True,
                            with_vectors=False,
                        )
                        points = axis_results.points
                    except Exception:
                        axis_fallback_limit = 30 if settings.enable_perf_optimizations else 50
                        axis_results = await self._qdrant_query(
                            query=axis_emb,
                            limit=axis_fallback_limit,
                            with_payload=True,
                            with_vectors=False,
                        )
                        valid_types = {"pto_frame", section_type}
                        points = [
                            p for p in axis_results.points
                            if (p.payload or {}).get("node_type") in valid_types
                        ][:10]

                    for pt in points:
                        did = (pt.payload or {}).get("doc_id")
                        if did:
                            doc_ids.add(did)
                except Exception:
                    continue

            axis_added = len(doc_ids) - pre_axis_count
            print(f"    [PTO] Search 2 added {axis_added} new doc_ids from axis queries")
            print(f"    [PTO] TOTAL: {len(doc_ids)} doc_ids "
                  f"(search1={len(doc_ids) - axis_added}, search2=+{axis_added})")
            if not doc_ids:
                print(f"    [PTO] PTO returned 0 results. Likely cause: PTO frames have not "
                      f"been ingested into collection '{self.collection}'. "
                      f"Run: python src/ingestion/pto_frame_builder.py --upsert")
            return doc_ids

        except Exception as e:
            print(f"    [PTO] PTO search FAILED (continuing without): {e}")
            import traceback
            traceback.print_exc()
            return set()

    async def _tagged_phase3(
        self,
        doc_id: str,
        query_embedding: List[float],
        expanded_query: str,
        max_chunks: int,
        query_type: str,
        doc_info_entry: Optional[Dict[str, Any]] = None,
        query_structure=None,
        inferred_axes: dict = None,
        query_text: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        """Wraps _phase3_document_search to return (doc_id, chunks).

        ``doc_info_entry`` is the mutable ``doc_info[doc_id]`` dict from the
        dispatch registry. The Phase 3 gate reads the current ``threshold`` off
        this dict at gate-check time so a later source-tag upgrade (e.g. Qdrant
        arrived first, PG then confirms the same doc and raises trust to
        ``both``) actually relaxes the gate for the in-flight task.

        ``query_text`` is the original user question — used for the cross-
        encoder gate so the relevance score reflects question↔passage match
        rather than the bloated expanded query.
        """
        chunks = await self._phase3_document_search(
            doc_id=doc_id,
            query_embedding=query_embedding,
            expanded_query=expanded_query,
            max_chunks=max_chunks,
            query_type=query_type,
            doc_info_entry=doc_info_entry,
            query_structure=query_structure,
            inferred_axes=inferred_axes,
            query_text=query_text,
        )
        return doc_id, chunks

    async def _phase3_document_search(
        self,
        doc_id: str,
        query_embedding: List[float],
        expanded_query: str,
        max_chunks: int,
        query_type: str = "general",
        doc_info_entry: Optional[Dict[str, Any]] = None,
        query_structure=None,
        inferred_axes: dict = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: Enhanced in-document search for comprehensive coverage.
        
        Features:
        1. Multi-query search: Original query + aspect-specific sub-queries
        2. Hybrid scoring: Dense vectors + lexical/BM25
        3. Section diversity: Ensures coverage across different sections
        """
        # Filter to only this document
        doc_filter = qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
        ])
        
        # Generate aspect-specific sub-queries (Task 5: clinical-axis sub-queries)
        sub_queries = self._generate_patient_axis_subqueries(
            main_query=expanded_query,
            query_type=query_type,
            query_structure=query_structure,
            inferred_axes=inferred_axes or {},
        )

        print(f"  [Phase3:{doc_id[:25]}] Sub-queries generated ({len(sub_queries)}):")
        for sq_name, sq_text in sub_queries.items():
            print(f"    {sq_name}: {sq_text[:80]}{'...' if len(sq_text) > 80 else ''}")

        # Collect all chunks from multiple queries
        all_chunks: Dict[str, Dict[str, Any]] = {}  # point_id -> chunk

        # Search with main query (offloaded to thread pool)
        main_results = await self._qdrant_query(
            query=query_embedding,
            limit=max_chunks * 2,
            query_filter=doc_filter,
            with_payload=True,
            with_vectors=False,
        )
        
        for point in main_results.points:
            payload = dict(point.payload or {})
            point_id = str(point.id)
            if point_id not in all_chunks:
                all_chunks[point_id] = {
                    "point_id": point.id,
                    "score_dense": float(point.score),
                    "doc_id": doc_id,
                    "text": payload.get("text", ""),
                    "section": payload.get("section"),
                    "chunk_type": payload.get("chunk_type"),
                    "chunk_id": payload.get("chunk_id"),
                    "section_window_idx": payload.get("section_window_idx"),
                    "doc_meta": payload.get("doc_meta", {}),
                    "category": payload.get("category"),
                    # Keep the full `metadata` dict so downstream code
                    # can read `doc_level_*` tags for patient-match
                    # scoring without a second Qdrant round-trip.
                    "metadata": payload.get("metadata", {}),
                    "query_matches": ["main"],
                }
                # Include table metadata if present
                if payload.get("chunk_type") == "table_row":
                    all_chunks[point_id]["table"] = {
                        "number": payload.get("table_number"),
                        "title": payload.get("table_title"),
                        "row_index": payload.get("row_index"),
                        "headers": (payload.get("metadata") or {}).get("headers", []),
                        "raw_row": (payload.get("metadata") or {}).get("raw_row", []),
                    }
        
        # Search with sub-queries for comprehensive coverage
        for sub_query_name, sub_query_text in sub_queries.items():
            try:
                sub_embedding = await self._embed_async(sub_query_text)
                sub_results = await self._qdrant_query(
                    query=sub_embedding,
                    limit=max_chunks,
                    query_filter=doc_filter,
                    with_payload=True,
                    with_vectors=False,
                )
                
                for point in sub_results.points:
                    payload = dict(point.payload or {})
                    point_id = str(point.id)
                    
                    if point_id in all_chunks:
                        # Boost score for chunks matching multiple queries
                        all_chunks[point_id]["query_matches"].append(sub_query_name)
                        # Average the scores
                        old_score = all_chunks[point_id]["score_dense"]
                        new_score = float(point.score)
                        all_chunks[point_id]["score_dense"] = (old_score + new_score) / 2
                    else:
                        all_chunks[point_id] = {
                            "point_id": point.id,
                            "score_dense": float(point.score),
                            "doc_id": doc_id,
                            "text": payload.get("text", ""),
                            "section": payload.get("section"),
                            "chunk_type": payload.get("chunk_type"),
                            "chunk_id": payload.get("chunk_id"),
                            "section_window_idx": payload.get("section_window_idx"),
                            "doc_meta": payload.get("doc_meta", {}),
                            "category": payload.get("category"),
                            "query_matches": [sub_query_name],
                        }
                        if payload.get("chunk_type") == "table_row":
                            all_chunks[point_id]["table"] = {
                                "number": payload.get("table_number"),
                                "title": payload.get("table_title"),
                                "row_index": payload.get("row_index"),
                                "headers": (payload.get("metadata") or {}).get("headers", []),
                                "raw_row": (payload.get("metadata") or {}).get("raw_row", []),
                            }
            except Exception as e:
                print(f"[ComprehensiveRetrieval] Sub-query '{sub_query_name}' failed: {e}")
        
        # Apply hybrid scoring (dense + lexical)
        chunks_list = list(all_chunks.values())
        print(f"  [Phase3:{doc_id[:25]}] Raw chunks collected: {len(chunks_list)} "
              f"(main query + {len(sub_queries)} sub-queries)")
        chunks_list = self._apply_hybrid_scoring(chunks_list, expanded_query)

        # Apply multi-query boost (chunks found by multiple queries are more relevant)
        multi_boosted = 0
        for chunk in chunks_list:
            num_matches = len(chunk.get("query_matches", []))
            if num_matches > 1:
                chunk["score"] += 0.05 * (num_matches - 1)
                chunk["multi_query_boost"] = True
                multi_boosted += 1
        if multi_boosted:
            print(f"  [Phase3:{doc_id[:25]}] Multi-query boosted: {multi_boosted} chunks")

        # Sort by final score
        chunks_list.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Deduplicate by text similarity
        final_chunks = []
        seen_texts = set()
        dupes_removed = 0

        for chunk in chunks_list:
            text = chunk.get("text", "")
            text_key = text[:200].lower().strip()
            if text_key in seen_texts:
                dupes_removed += 1
                continue
            seen_texts.add(text_key)
            final_chunks.append(chunk)

            if len(final_chunks) >= max_chunks:
                break

        print(f"  [Phase3:{doc_id[:25]}] After dedup: {len(final_chunks)} chunks kept, {dupes_removed} duplicates removed")
        if final_chunks:
            print(f"  [Phase3:{doc_id[:25]}] Top chunk scores: {[round(c.get('score', 0), 3) for c in final_chunks[:5]]}")
            print(f"  [Phase3:{doc_id[:25]}] Sections covered: {list({c.get('section') for c in final_chunks if c.get('section')})[:5]}")

        # ── Task 4: Cross-encoder gate ────────────────────────────────────
        # Read the threshold from the live ``doc_info`` entry so a later
        # source-tag upgrade propagates to the already-running Phase 3 task
        # (the upgrade path mutates this same dict at dispatch time).
        gate_threshold = 0.45
        if doc_info_entry is not None:
            gate_threshold = doc_info_entry.get("threshold", 0.45)
        print(f"  [Phase3:{doc_id[:25]}] Cross-encoder gate (threshold={gate_threshold:.2f})...")
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is not None and final_chunks:
            # Prepend the study title so the cross-encoder sees canonical
            # literature vocabulary regardless of which sections the top
            # chunks came from (a Methods chunk doesn't say "Glioblastoma"
            # verbatim, but its study's title always does). Matches the
            # passage construction used by cross_encoder_rerank in
            # enhanced_rag_service.py.
            study_title = ""
            for chunk in final_chunks[:4]:
                t = (chunk.get("doc_meta") or {}).get("title") or ""
                if t.strip():
                    study_title = t.strip()
                    break

            combined_text = ""
            for chunk in final_chunks[:4]:
                t = chunk.get("text", "")
                if len(combined_text) + len(t) < 2000:
                    combined_text += " " + t
                else:
                    break
            combined_text = combined_text.strip()

            if study_title and len(study_title) >= 15:
                title_part = study_title[:200].rstrip(".") + "."
                combined_text = f"{title_part} {combined_text}"[:2200]

            if combined_text:
                try:
                    # Build a short, structure-derived keyword query for
                    # ms-marco-MiniLM. The expanded_query is ~10K chars
                    # of appended vocabulary terms and the raw query is
                    # the full patient narrative — both blow past the
                    # cross-encoder's training distribution and saturate
                    # the logits negative (every score collapses to 0%).
                    from src.api.services.enhanced_rag_service import (
                        build_reranker_query,
                    )
                    ce_query = build_reranker_query(
                        query_text or expanded_query, query_structure
                    )
                    # Micro-batched: concurrent Phase 3 gate calls are
                    # accumulated and scored in one predict() call (see
                    # _gate_score_batched). Same scores, far less CPU.
                    ce_score = await self._gate_score_batched(
                        cross_encoder, (ce_query, combined_text)
                    )

                    # ── Site-level sanity check ─────────────────────────
                    # If the query's resolved category disagrees with the
                    # document's category, apply a penalty. This catches
                    # cases where a strong embedding match pulls a study
                    # from an unrelated cancer category (e.g. NCCN-NSCLC
                    # content embedding-matching a head-and-neck case's
                    # pathology/NGS sub-query).
                    query_category = None
                    if query_structure is not None:
                        query_category = getattr(query_structure, "filter_category", None)
                    doc_category = None
                    if doc_info_entry is not None:
                        doc_category = doc_info_entry.get("category")
                    if (
                        query_category
                        and doc_category
                        and normalize_category(query_category) != normalize_category(doc_category)
                    ):
                        penalty = 0.15
                        ce_score -= penalty
                        print(
                            f"[Phase3Gate] site-mismatch penalty applied to "
                            f"{doc_id[:40]}: "
                            f"query_cat={query_category!r} vs "
                            f"doc_cat={doc_category!r} (-{penalty:.2f})"
                        )

                    # Store (possibly penalised) score for Phase 6 free reranking
                    for chunk in final_chunks:
                        chunk["score_crossencoder_gate"] = ce_score

                    if ce_score < gate_threshold:
                        print(f"[Phase3Gate] REJECTED {doc_id[:40]} "
                              f"ce_score={ce_score:.3f} < threshold={gate_threshold:.3f}")
                        return []
                    else:
                        print(f"[Phase3Gate] PASSED   {doc_id[:40]} "
                              f"ce_score={ce_score:.3f}")
                except Exception as e:
                    print(f"[Phase3Gate] Cross-encoder failed for {doc_id[:40]}: {e}")

        return final_chunks
    
    def _generate_sub_queries(self, main_query: str, query_type: str) -> Dict[str, str]:
        """Legacy sub-query generator. Delegates to patient-axis version."""
        return self._generate_patient_axis_subqueries(main_query, query_type)

    def _generate_patient_axis_subqueries(
        self,
        main_query: str,
        query_type: str,
        query_structure=None,
        inferred_axes: dict = None,
    ) -> Dict[str, str]:
        """
        Generate sub-queries derived from the patient's specific clinical axes.

        Uses extractor_keywords.json and ajcc_staging_tables.json to enrich
        sub-query templates with ontology-sourced terms rather than relying
        solely on hardcoded strings.

        Falls back to generic sub-queries when no query_structure is available
        (preserves backward compatibility for non-patient queries).
        """
        inferred_axes = inferred_axes or {}

        # Load ontology keywords (cached after first call)
        try:
            from src.api.services.ontology_loader import (
                get_axis_keywords,
                expand_cancer_site_synonyms,
                get_cancer_type_context,
                get_biomarker_keywords,
                get_treatment_keywords,
                get_outcome_keywords,
                get_ici_resistance_terms,
                get_metastatic_pattern_terms,
            )
            _ontology_available = True
        except Exception:
            _ontology_available = False

        def _pick_keywords(axis: str, limit: int = 6) -> str:
            """Pick a few ontology keywords for an axis, deduped against existing text."""
            if not _ontology_available:
                return ""
            kws = get_axis_keywords(axis)
            return " ".join(kws[:limit])

        # ── Fallback: generic sub-queries for non-patient queries ─────────
        has_patient = query_structure is not None and getattr(query_structure, 'has_patient_context', False)
        if not has_patient:
            sub = {}
            q = main_query.lower()
            # Use ontology outcome keywords instead of hardcoded list
            outcome_terms = " ".join(get_outcome_keywords().get("survival_metrics", [])[:5]) if _ontology_available else "outcomes survival results efficacy"
            if "outcome" not in q and "survival" not in q:
                sub["outcomes"] = f"{main_query} {outcome_terms}"
            if query_type in ("dose_question", "treatment_recommendation"):
                if "dose" not in q:
                    sub["dosing"] = f"{main_query} dose fractionation regimen"
            if query_type in ("side_effects", "treatment_recommendation"):
                if "toxicity" not in q:
                    safety_terms = " ".join(get_outcome_keywords().get("safety_metrics", [])[:5]) if _ontology_available else "toxicity adverse effects"
                    sub["toxicity"] = f"{main_query} {safety_terms}"
            return dict(list(sub.items())[:3])

        # ── Patient-profile-derived sub-queries ───────────────────────────
        sub = {}
        cancer = query_structure.cancer
        expanded_axes = inferred_axes.get("expanded_axes", {})

        # 1. Primary cancer identity (axis 1: primary_cancer)
        #    Enriched with AJCC aliases + cancer_type_ontology keywords/drugs
        primary_axis = expanded_axes.get("primary_cancer", "")
        if primary_axis or cancer.site:
            site_text = primary_axis or cancer.site or ""
            extra_parts = []
            if _ontology_available:
                site_synonyms = expand_cancer_site_synonyms(site_text)
                existing_lower = site_text.lower()
                extra_parts.extend(
                    s for s in site_synonyms[:4] if s.lower() not in existing_lower
                )
                # Add cancer-specific keywords from cancer_type_ontology
                ctx = get_cancer_type_context(site_text)
                if ctx:
                    extra_parts.extend(
                        k for k in ctx.get("keywords", [])[:5]
                        if k.lower() not in existing_lower
                    )
            sub["primary"] = (
                f"{site_text} {cancer.histology or ''} "
                f"{cancer.stage or ''} {' '.join(extra_parts)} treatment outcomes"
            ).strip()

        # 2. ICI trajectory (axis 6: disease_trajectory)
        #    Enriched with ontology treatment response keywords
        trajectory_flags = inferred_axes.get("trajectory_flags", [])
        current_therapy = expanded_axes.get("current_treatment", "") or getattr(query_structure.treatment, "raw_text", None) or ""
        if any(f in trajectory_flags for f in ("ici_refractory", "progressing_on_ici")):
            # Use trial ontology ICI resistance terms + immunotherapy markers
            ici_ontology = ""
            if _ontology_available:
                resist_terms = get_ici_resistance_terms()[:6]
                ici_ontology = " ".join(resist_terms)
            sub["ici_refractory"] = (
                f"{cancer.site or ''} {cancer.histology or ''} "
                f"ICI-refractory anti-PD1 failure second-line salvage "
                f"checkpoint inhibitor resistant {ici_ontology}"
            ).strip()
        elif any(
            t in current_therapy.lower()
            for t in ("pembrolizumab", "nivolumab", "ici", "checkpoint")
        ):
            response_kws = ""
            if _ontology_available:
                response_kws = " ".join(get_treatment_keywords().get("response", [])[:5])
            sub["immunotherapy"] = (
                f"{cancer.site or ''} {current_therapy[:60]} "
                f"outcomes {response_kws}"
            ).strip()

        # 3. Biomarker-specific evidence (axis 5: biomarker_profile)
        #    Enriched with ontology biomarker categories
        biomarker_axis = expanded_axes.get("biomarker_profile", "")
        biomarkers = cancer.biomarkers or []
        cps_marker = next((b for b in biomarkers if "CPS" in b.upper()), None)
        pdl1_marker = next(
            (b for b in biomarkers if "PD-L1" in b.upper() or "PDL1" in b.upper()),
            None,
        )
        if biomarker_axis or cps_marker or pdl1_marker:
            bio_kws = _pick_keywords("biomarker_profile", 6)
            sub["biomarker"] = (
                f"{biomarker_axis} CPS PD-L1 expression "
                f"{cancer.histology or ''} {cancer.site or ''} "
                f"immunotherapy {bio_kws}"
            ).strip()

        # 4. Metastatic / staging concern (axis 7: metastatic_concern)
        #    Enriched with ontology distant_spread keywords
        met_axis = expanded_axes.get("metastatic_concern", "")
        met_sites = inferred_axes.get("metastatic_sites", [])
        if met_axis or met_sites:
            met_kws = ""
            if _ontology_available:
                met_patterns = get_metastatic_pattern_terms()
                met_existing = (met_axis + " " + " ".join(met_sites)).lower()
                met_kws = " ".join(
                    t for t in met_patterns[:8] if t.lower() not in met_existing
                )
            sub["metastatic"] = (
                f"{met_axis} {' '.join(met_sites)} "
                f"metastasis outcomes prognosis {met_kws}"
            ).strip()

        # 5. Non-surgical / eligibility (axis 8: patient_factors)
        #    Enriched with ontology eligibility keywords
        patient_axis = expanded_axes.get("patient_factors", "")
        surgical_candidate = inferred_axes.get("surgical_candidate")
        if surgical_candidate is False:
            elig_kws = _pick_keywords("patient_factors", 5)
            sub["eligibility"] = (
                f"{patient_axis} unresectable inoperable locoregional advanced "
                f"systemic therapy eligibility {elig_kws}"
            ).strip()

        # 6. Treatment response / neoadjuvant trajectory
        #    Fires when the inference layer detected response-related flags
        if any(f in trajectory_flags for f in (
            "excellent_response", "pcr", "poor_response",
            "post_neoadjuvant", "de_escalation", "residual_disease",
        )):
            response_parts = []
            if "excellent_response" in trajectory_flags or "pcr" in trajectory_flags:
                response_parts.extend([
                    "pathologic complete response", "pCR", "ypT0N0",
                    "excellent response", "de-escalation", "treatment de-intensification",
                    "response-adapted", "adjuvant omission",
                ])
            if "poor_response" in trajectory_flags or "residual_disease" in trajectory_flags:
                response_parts.extend([
                    "residual disease", "poor response", "non-pCR",
                    "adjuvant escalation", "additional adjuvant",
                    "post-neoadjuvant residual", "RCB-II", "RCB-III",
                ])
            if "post_neoadjuvant" in trajectory_flags:
                response_parts.extend([
                    "neoadjuvant chemotherapy", "post-NAC", "preoperative",
                    "NAC response", "post-neoadjuvant adjuvant",
                ])
            if "de_escalation" in trajectory_flags:
                response_parts.extend([
                    "de-escalation", "de-intensification", "omission",
                    "reduced adjuvant", "response-guided",
                ])
            sub["treatment_response"] = (
                f"{cancer.site or ''} {cancer.histology or ''} "
                f"{' '.join(response_parts[:12])}"
            ).strip()

        # 7. Comorbidity → treatment eligibility inference
        comorbidities = getattr(query_structure.patient, "comorbidities", []) or []
        # Use ontology comorbidity keywords for broader matching
        comorbidity_kws = []
        if _ontology_available:
            from src.api.services.ontology_loader import get_patient_keywords
            comorbidity_kws = get_patient_keywords().get("comorbidities", [])
        cisplatin_contra = {"ckd", "renal", "kidney", "creatinine"}
        # Also check against ontology comorbidity terms
        cisplatin_contra.update(
            t.lower() for t in comorbidity_kws
            if any(w in t.lower() for w in ("renal", "kidney", "creatinine"))
        )
        if any(
            any(c_word in c.lower() for c_word in cisplatin_contra)
            for c in comorbidities
        ):
            sub["comorbidity"] = (
                "cisplatin ineligible carboplatin renal impairment "
                "modified regimen toxicity"
            )

        # Cap at 6 sub-queries (one per active axis); always include primary
        result = {}
        if "primary" in sub:
            result["primary"] = sub.pop("primary")
        for k, v in list(sub.items())[:5]:
            result[k] = v

        return result
    
    def _apply_hybrid_scoring(
        self, 
        chunks: List[Dict[str, Any]], 
        query: str,
        dense_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring combining dense vector scores with lexical/BM25 scores.
        
        This helps with exact term matching (drug names, trial names, staging terms).
        """
        if not chunks:
            return chunks
        
        # Tokenize query
        query_terms = set(self._tokenize(query.lower()))
        
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            text_terms = set(self._tokenize(text))
            
            # Calculate lexical score (Jaccard-like overlap)
            if query_terms and text_terms:
                overlap = len(query_terms & text_terms)
                lexical_score = overlap / len(query_terms) if query_terms else 0
            else:
                lexical_score = 0
            
            # Boost for exact phrase matches
            phrase_boost = 0
            for term in query_terms:
                if len(term) > 4 and term in text:
                    phrase_boost += 0.05  # 5% boost per exact term match
            
            # Combine scores
            dense_score = chunk.get("score_dense", 0)
            chunk["score_lexical"] = lexical_score
            chunk["score"] = (dense_score * dense_weight) + (lexical_score * lexical_weight) + phrase_boost
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for lexical scoring."""
        import re
        # Split on non-alphanumeric, keep words > 2 chars
        tokens = re.findall(r'\b[a-z0-9]{3,}\b', text.lower())
        return tokens
    
    async def _phase3_document_search_legacy(
        self,
        doc_id: str,
        query_embedding: List[float],
        expanded_query: str,
        max_chunks: int,
    ) -> List[Dict[str, Any]]:
        """
        Legacy Phase 3: Simple single-query search (kept for reference).
        """
        # Filter to only this document
        doc_filter = qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
        ])
        
        # Query for chunks within this document
        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=max_chunks * 2,  # Get extra for deduplication
            query_filter=doc_filter,
            with_payload=True,
            with_vectors=False,
        )
        
        # Convert and deduplicate
        chunks = []
        seen_texts = set()
        
        for point in results.points:
            payload = dict(point.payload or {})
            text = payload.get("text", "")
            
            # Skip near-duplicate text
            text_key = text[:200].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            
            chunk = {
                "point_id": point.id,
                "score": float(point.score),
                "doc_id": doc_id,
                "text": text,
                "section": payload.get("section"),
                "chunk_type": payload.get("chunk_type"),
                "chunk_id": payload.get("chunk_id"),
                "section_window_idx": payload.get("section_window_idx"),
                "doc_meta": payload.get("doc_meta", {}),
                "category": payload.get("category"),
            }
            
            # Include table metadata if present
            if payload.get("chunk_type") == "table_row":
                chunk["table"] = {
                    "number": payload.get("table_number"),
                    "title": payload.get("table_title"),
                    "row_index": payload.get("row_index"),
                    "headers": (payload.get("metadata") or {}).get("headers", []),
                    "raw_row": (payload.get("metadata") or {}).get("raw_row", []),
                }
            
            chunks.append(chunk)
            
            if len(chunks) >= max_chunks:
                break
        
        return chunks
    
    async def _phase4_rerank_studies(
        self,
        studies: List[StudyEvidence],
        query_text: str,
        max_studies: int,
        query_structure=None,
    ) -> List[StudyEvidence]:
        """
        Phase 4: Rerank studies by overall relevance using cross-encoder.

        For each study, we create a combined text from its chunks and
        score it against the query.
        """
        if not studies:
            return studies

        cross_encoder = self._get_cross_encoder()

        if cross_encoder is None:
            # Fallback: sort by initial score
            studies.sort(key=lambda s: s.initial_score, reverse=True)
            for study in studies:
                study.rerank_score = study.initial_score
            return studies[:max_studies]

        # Distill the query down to a short keyword form so ms-marco
        # MiniLM scores stay inside its training distribution. See
        # build_reranker_query for the rationale.
        try:
            from src.api.services.enhanced_rag_service import (
                build_reranker_query,
            )
            ce_query = build_reranker_query(query_text, query_structure)
        except Exception:
            ce_query = (query_text or "")[:200]

        # Build pairs for cross-encoder
        pairs = []
        for study in studies:
            # Combine chunk texts for study-level scoring
            # Use first ~2000 chars to stay within model limits
            combined_text = ""
            for chunk in study.chunks[:4]:
                text = chunk.get("text", "")
                if len(combined_text) + len(text) < 2000:
                    combined_text += " " + text
                else:
                    break
            
            # Add title for context
            study_text = f"{study.title}. {combined_text.strip()}"
            pairs.append((ce_query, study_text))
        
        # Score all pairs
        try:
            scores = await self._run_sync(cross_encoder.predict, pairs)

            # Assign scores
            for study, score in zip(studies, scores):
                study.rerank_score = float(score)
            
            # Sort by rerank score
            studies.sort(key=lambda s: s.rerank_score, reverse=True)
            
            print(f"[ComprehensiveRetrieval] Reranked {len(studies)} studies")
            for i, study in enumerate(studies[:5]):
                print(f"  {i+1}. [{study.rerank_score:.3f}] {study.title[:50]}")
            
        except Exception as e:
            print(f"[ComprehensiveRetrieval] Cross-encoder failed: {e}")
            # Fallback to initial scores
            studies.sort(key=lambda s: s.initial_score, reverse=True)
            for study in studies:
                study.rerank_score = study.initial_score
        
        return studies[:max_studies]


# Singleton instance
_retriever_instance: Optional[ComprehensiveRetriever] = None


def get_comprehensive_retriever() -> ComprehensiveRetriever:
    """Get singleton ComprehensiveRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        qdrant_timeout = 15 if settings.enable_perf_optimizations else 60
        # prefer_grpc: gRPC is a faster binary protocol than the default
        # REST/HTTP transport, Qdrant Cloud supports it natively on 6334.
        # Transport-only change — does not alter filters, ranking, or
        # results returned by any search call.
        qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=qdrant_timeout,
            prefer_grpc=True,
            grpc_port=6334,
        )
        print(f"[ComprehensiveRetrieval] Qdrant client timeout={qdrant_timeout}s "
              f"(perf_optimizations={'on' if settings.enable_perf_optimizations else 'off'})")
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        _retriever_instance = ComprehensiveRetriever(
            qdrant_client=qdrant,
            openai_client=openai_client,
        )
    return _retriever_instance


def convert_to_rag_evidence(
    result: ComprehensiveRetrievalResult,
    max_chunks: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert ComprehensiveRetrievalResult to standard RAG evidence format.
    
    Includes cross-study consensus detection for key findings.
    
    Returns:
        Tuple of (evidence_list, metadata_dict)
    """
    evidence = result.get_evidence_list()[:max_chunks]
    
    # Detect cross-study consensus (findings that appear in multiple studies)
    consensus_findings = detect_cross_study_consensus(result.studies)
    
    # Add consensus markers to evidence chunks
    for e in evidence:
        doc_id = e.get("doc_id")
        text = e.get("text", "").lower()
        
        # Check if this chunk contains consensus findings
        e["consensus_matches"] = []
        for finding in consensus_findings:
            if finding["pattern"].lower() in text:
                e["consensus_matches"].append({
                    "finding": finding["pattern"],
                    "study_count": finding["study_count"],
                    "studies": finding["studies"][:3],  # Top 3 study titles
                })
        
        if e["consensus_matches"]:
            # Boost score for chunks with consensus findings
            e["score"] = e.get("score", 0) * (1 + 0.1 * len(e["consensus_matches"]))
            e["has_consensus"] = True
    
    metadata = {
        "comprehensive_retrieval": True,
        "studies_retrieved": len(result.studies),
        "total_chunks": result.total_chunks,
        "retrieval_time_ms": result.retrieval_time_ms,
        "phase1_qdrant_docs": result.phase1_qdrant_docs,
        "phase1_postgres_docs": result.phase1_postgres_docs,
        "phase2_docs_searched": result.phase2_docs_searched,
        "expanded_query": result.expanded_query,
        "query_structure": result.query_structure,
        "consensus_findings": consensus_findings[:5],  # Top 5 consensus findings
        "study_summaries": [
            {
                "doc_id": s.doc_id,
                "title": s.title,
                "rerank_score": s.rerank_score,
                "chunks": len(s.chunks),
                "sections": list(s.sections_covered),
                "source": s.source,
                "patient_match_score": s.patient_match_score,
                "patient_match_breakdown": s.patient_match_breakdown,
                "evidence_type": s.evidence_type,
            }
            for s in result.studies
        ],
    }
    
    return evidence, metadata


def detect_cross_study_consensus(studies: List[StudyEvidence]) -> List[Dict[str, Any]]:
    """
    Detect findings that appear across multiple studies.
    
    Looks for:
    - Dose values (e.g., "50 Gy", "2 Gy/fraction")
    - Survival rates (e.g., "5-year OS 85%")
    - Key recommendations
    
    Returns list of consensus findings with study counts.
    """
    import re
    from collections import defaultdict
    
    # Patterns to detect key findings
    patterns = {
        "dose": r'\b(\d+(?:\.\d+)?)\s*(?:Gy|cGy)\b',
        "fractionation": r'\b(\d+(?:\.\d+)?)\s*(?:Gy|cGy)\s*/\s*(?:fraction|fx)\b',
        "survival_rate": r'\b(\d+(?:\.\d+)?)\s*%\s*(?:OS|DFS|PFS|survival|control)\b',
        "hazard_ratio": r'\bHR\s*[=:]?\s*(\d+\.\d+)\b',
    }
    
    # Track findings by pattern across studies
    findings_by_pattern: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    for study in studies:
        study_text = " ".join(chunk.get("text", "") for chunk in study.chunks)
        
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, study_text, re.IGNORECASE)
            for match in matches:
                # Normalize the match
                normalized = match.strip().lower()
                findings_by_pattern[pattern_name][normalized].append(study.title)
    
    # Build consensus list (findings in 2+ studies)
    consensus = []
    for pattern_name, findings in findings_by_pattern.items():
        for value, study_titles in findings.items():
            if len(study_titles) >= 2:
                # Deduplicate study titles
                unique_studies = list(dict.fromkeys(study_titles))
                if len(unique_studies) >= 2:
                    consensus.append({
                        "type": pattern_name,
                        "pattern": value,
                        "study_count": len(unique_studies),
                        "studies": unique_studies,
                    })
    
    # Sort by study count (most consensus first)
    consensus.sort(key=lambda x: x["study_count"], reverse=True)
    
    if consensus:
        print(f"[ComprehensiveRetrieval] Found {len(consensus)} cross-study consensus findings")
        for c in consensus[:3]:
            print(f"  - {c['type']}: {c['pattern']} ({c['study_count']} studies)")
    
    return consensus
