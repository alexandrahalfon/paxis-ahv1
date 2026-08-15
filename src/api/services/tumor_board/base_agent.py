"""
SpecialtyAgent base class + ExpertAssessment / StudyCitation dataclasses.

Every specialty (Medical Oncology, Radiation Oncology, Surgical Oncology,
Pathology / Molecular, Radiology, Palliative Care) extends SpecialtyAgent
and implements:

    - build_sub_queries(bundle) -> List[str]
    - relevance_filter(bundle)  -> bool

The base class handles the common flow:

    1. relevance filter → optionally skip
    2. build specialty sub-queries
    3. dispatch them in parallel via retrieve_comprehensive()
    4. merge / dedupe retrieved studies by doc_id
    5. LLM synthesis using the specialty's system prompt → ExpertAssessment
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.core.config import settings

from .case_bundle import PatientCaseBundle
from .retrieval import LightweightStudy, lightweight_search


# Allowed recommendation verdicts — kept as a plain tuple so tests / the
# API layer can assert against it without importing an enum.
RECOMMENDATION_VERDICTS = (
    "favor",
    "against",
    "conditional",
    "insufficient_evidence",
)


@dataclass
class StudyCitation:
    """Lightweight reference to a retrieved study used by the expert agent."""

    doc_id: str
    title: str
    citation: Optional[str] = None
    year: Optional[int] = None
    relevance_score: float = 0.0
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertAssessment:
    """
    Structured output from a single specialty agent. The orchestrator
    collects one per agent and returns them to the API client side-by-side.
    """

    specialty: str                                  # e.g. "medical_oncology"
    display_name: str                               # e.g. "Medical Oncology"
    recommendation: str                             # RECOMMENDATION_VERDICTS
    recommendation_text: str                        # 1-paragraph explanation
    confidence: float                               # 0.0 - 1.0
    key_questions: List[str] = field(default_factory=list)
    supporting_studies: List[StudyCitation] = field(default_factory=list)
    conflicting_studies: List[StudyCitation] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    # ─── constructors for edge cases ──────────────────────────────────────

    @classmethod
    def skipped_assessment(
        cls,
        specialty: str,
        display_name: str,
        reason: str,
    ) -> "ExpertAssessment":
        return cls(
            specialty=specialty,
            display_name=display_name,
            recommendation="insufficient_evidence",
            recommendation_text=(
                f"{display_name} review skipped: {reason}"
            ),
            confidence=0.0,
            skipped=True,
            skip_reason=reason,
        )

    @classmethod
    def error_assessment(
        cls,
        specialty: str,
        display_name: str,
        error: str,
    ) -> "ExpertAssessment":
        return cls(
            specialty=specialty,
            display_name=display_name,
            recommendation="insufficient_evidence",
            recommendation_text=(
                f"{display_name} review failed: {error}"
            ),
            confidence=0.0,
            error=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supporting_studies"] = [s.to_dict() if isinstance(s, StudyCitation) else s
                                    for s in self.supporting_studies]
        d["conflicting_studies"] = [s.to_dict() if isinstance(s, StudyCitation) else s
                                     for s in self.conflicting_studies]
        return d


class SpecialtyAgent(ABC):
    """
    Base class for one specialty on the virtual tumor board.

    Subclasses MUST set the class attributes `specialty`, `display_name`,
    and `system_prompt`, and MUST implement `build_sub_queries` and
    `relevance_filter`.
    """

    # Subclasses override these
    specialty: str = ""
    display_name: str = ""
    system_prompt: str = ""

    # Retrieval knobs — reasonable defaults; subclasses may override
    max_sub_queries: int = 3
    points_per_query: int = 25
    max_chunks_per_study: int = 4

    # ─── abstract hooks ───────────────────────────────────────────────────

    @abstractmethod
    def build_sub_queries(self, bundle: PatientCaseBundle) -> List[str]:
        """Return up to `max_sub_queries` specialty-specific query strings."""

    @abstractmethod
    def relevance_filter(self, bundle: PatientCaseBundle) -> Optional[str]:
        """
        Return None if this agent should evaluate the case.
        Return a short skip-reason string to skip it.
        """

    # ─── retrieval hook (override for tests) ─────────────────────────────

    async def _retrieve(
        self,
        query_text: str,
        category: Optional[str] = None,
    ) -> List[LightweightStudy]:
        """Fetch studies for one specialty sub-query.

        Default: direct-Qdrant lightweight_search — avoids the heavy
        /rag pipeline entirely so the tumor board stays fast and doesn't
        get blocked by its cross-encoder gate or PTO latency.

        `category` is forwarded to `lightweight_search` so per-specialty
        sub-queries can't match literature from unrelated cancer
        categories (e.g. pathology NGS queries embedding-matching
        NCCN-NSCLC content for a head-and-neck case).

        Unit tests override this method on an instance to inject fakes.
        """
        return await lightweight_search(
            query_text,
            limit_points=self.points_per_query,
            max_chunks_per_study=self.max_chunks_per_study,
            category=category,
        )

    # ─── main entry point used by the orchestrator ───────────────────────

    async def evaluate(
        self,
        bundle: PatientCaseBundle,
        retriever: Any = None,
    ) -> ExpertAssessment:
        """Run this specialty's full evaluation flow.

        `retriever` is accepted for backwards compatibility but is ignored;
        retrieval is now handled via `self._retrieve()`.
        """
        t0 = time.perf_counter()

        skip_reason = self.relevance_filter(bundle)
        if skip_reason:
            assess = ExpertAssessment.skipped_assessment(
                specialty=self.specialty,
                display_name=self.display_name,
                reason=skip_reason,
            )
            assess.elapsed_ms = (time.perf_counter() - t0) * 1000
            return assess

        # 1. Build specialty sub-queries
        sub_queries = [q for q in self.build_sub_queries(bundle) if q and q.strip()]
        sub_queries = sub_queries[: self.max_sub_queries]
        if not sub_queries:
            assess = ExpertAssessment.skipped_assessment(
                specialty=self.specialty,
                display_name=self.display_name,
                reason="no specialty sub-queries could be built from the extracted case",
            )
            assess.elapsed_ms = (time.perf_counter() - t0) * 1000
            return assess

        # 2. Retrieve evidence for each sub-query in parallel.
        #    Each task is a direct Qdrant query — ~1–2 s each.
        #    `bundle.category` constrains every sub-query to the
        #    patient's cancer category so pathology / NGS sub-queries
        #    can't leak across categories (e.g. NCCN-NSCLC for an H&N case).
        category = bundle.category
        retrieval_tasks = [
            self._retrieve(q, category=category) for q in sub_queries
        ]
        retrieval_results = await asyncio.gather(
            *retrieval_tasks, return_exceptions=True
        )

        # 3. Merge studies across sub-queries, dedup by doc_id
        merged_studies = self._merge_studies(retrieval_results)
        print(
            f"[TumorBoard:{self.specialty}] retrieved {len(merged_studies)} "
            f"unique studies from {len(sub_queries)} sub-queries"
        )

        if not merged_studies:
            assess = ExpertAssessment(
                specialty=self.specialty,
                display_name=self.display_name,
                recommendation="insufficient_evidence",
                recommendation_text=(
                    f"No {self.display_name.lower()} evidence could be retrieved "
                    f"from the knowledge base for the sub-queries generated from "
                    f"this case."
                ),
                confidence=0.0,
                sub_queries=sub_queries,
            )
            assess.elapsed_ms = (time.perf_counter() - t0) * 1000
            return assess

        # 4. LLM synthesis
        try:
            assessment = await self._synthesize(bundle, sub_queries, merged_studies)
        except Exception as e:  # pragma: no cover — defensive
            assessment = ExpertAssessment.error_assessment(
                specialty=self.specialty,
                display_name=self.display_name,
                error=f"LLM synthesis failed: {e}",
            )

        assessment.sub_queries = sub_queries
        assessment.elapsed_ms = (time.perf_counter() - t0) * 1000
        return assessment

    # ─── helpers ─────────────────────────────────────────────────────────

    def _merge_studies(
        self,
        retrieval_results: List[Any],
    ) -> List[Any]:
        """
        Merge studies across sub-query retrieval results, dedup by doc_id,
        keep the highest score per study, and sort descending.

        Accepts both the new shape (each result is a `List[LightweightStudy]`
        from `lightweight_search`) and the legacy shape (each result is a
        `ComprehensiveRetrievalResult`-like object with a `.studies` list).
        """
        merged: Dict[str, Any] = {}
        for res in retrieval_results:
            if isinstance(res, Exception):
                print(f"[TumorBoard:{self.specialty}] sub-query retrieval failed: {res}")
                continue
            if res is None:
                continue
            # New shape: res is a list of study-like objects
            if isinstance(res, list):
                studies_iter = res
            else:
                # Legacy shape (still accepted for test shims)
                studies_iter = getattr(res, "studies", None) or []
            for study in studies_iter:
                doc_id = getattr(study, "doc_id", None)
                if not doc_id:
                    continue
                score = float(getattr(study, "rerank_score", 0) or 0)
                existing = merged.get(doc_id)
                if existing is None or score > float(
                    getattr(existing, "rerank_score", 0) or 0
                ):
                    merged[doc_id] = study
        return sorted(
            merged.values(),
            key=lambda s: (
                float(getattr(s, "rerank_score", 0) or 0),
                float(getattr(s, "initial_score", 0) or 0),
            ),
            reverse=True,
        )

    def _build_user_message(
        self,
        bundle: PatientCaseBundle,
        sub_queries: List[str],
        studies: List[Any],
    ) -> str:
        """Build the user-role message sent to the LLM for synthesis."""
        lines: List[str] = []
        lines.append("PATIENT CASE (structured summary):")
        lines.append(bundle.summary_text())
        lines.append("")
        lines.append("ORIGINAL NARRATIVE:")
        lines.append(bundle.raw_text.strip())
        lines.append("")
        lines.append(f"YOUR SPECIALTY SUB-QUERIES ({len(sub_queries)}):")
        for q in sub_queries:
            lines.append(f"  - {q}")
        lines.append("")
        lines.append("RETRIEVED EVIDENCE (one block per study):")
        for i, study in enumerate(studies[:8]):
            title = getattr(study, "title", "Unknown")
            citation = getattr(study, "citation", None) or ""
            year = getattr(study, "year", None)
            rerank = getattr(study, "rerank_score", 0) or 0
            lines.append(
                f"\n[STUDY {i + 1}] doc_id={study.doc_id} | "
                f"score={rerank:.3f} | year={year} | {citation}"
            )
            lines.append(f"  Title: {title}")
            chunks = getattr(study, "chunks", []) or []
            for j, chunk in enumerate(chunks[:3]):
                section = chunk.get("section") or "body"
                text = (chunk.get("text") or "").strip()
                if text:
                    lines.append(f"  [{section}] {text[:600]}")
        lines.append("")
        lines.append(
            "Produce a JSON object matching the schema in the system prompt. "
            "Cite studies using their doc_id only. Do not invent evidence not "
            "present in the retrieved blocks above."
        )
        return "\n".join(lines)

    async def _synthesize(
        self,
        bundle: PatientCaseBundle,
        sub_queries: List[str],
        studies: List[Any],
    ) -> ExpertAssessment:
        """Call the LLM with the specialty system prompt and parse the JSON."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        user_message = self._build_user_message(bundle, sub_queries, studies)

        print(
            f"[TumorBoard:{self.specialty}] synthesizing over "
            f"{len(studies)} studies, {len(sub_queries)} sub-queries"
        )

        response = await client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = self._parse_llm_json(content)

        assessment = self._assessment_from_parsed(parsed, studies)

        # Strip unverified statistics from the LLM narrative before it
        # leaves this agent. Without this, P4 specialists were free to
        # hallucinate percentages / HRs that weren't in retrieved evidence.
        try:
            from src.api.services.safety.numerical import (
                strip_unvalidated_numbers,
                validate_numbers_against_sources,
            )
            evidence = []
            for st in studies:
                for c in (getattr(st, "chunks", None) or []):
                    txt = c.get("text") if isinstance(c, dict) else None
                    if txt:
                        evidence.append({"text": txt})
            if evidence and assessment.recommendation_text:
                v = validate_numbers_against_sources(
                    assessment.recommendation_text, evidence,
                )
                if v["unvalidated_numbers"]:
                    assessment.recommendation_text = strip_unvalidated_numbers(
                        assessment.recommendation_text,
                        v["unvalidated_numbers"],
                    )
        except Exception as e:  # pragma: no cover — defensive
            print(f"[TumorBoard:{self.specialty}] numerical validation failed: {e}")

        return assessment

    def _parse_llm_json(self, content: str) -> Dict[str, Any]:
        """Robust JSON parse — strips markdown fences if present."""
        s = content.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            print(f"[TumorBoard:{self.specialty}] JSON parse failed: {e}")
            return {}

    def _assessment_from_parsed(
        self,
        parsed: Dict[str, Any],
        studies: List[Any],
    ) -> ExpertAssessment:
        """Convert the parsed LLM JSON into a typed ExpertAssessment."""
        # doc_id → StudyEvidence lookup for citation enrichment
        study_index = {s.doc_id: s for s in studies}

        def build_citations(ref_list: Any) -> List[StudyCitation]:
            citations: List[StudyCitation] = []
            if not isinstance(ref_list, list):
                return citations
            for ref in ref_list:
                if isinstance(ref, dict):
                    doc_id = ref.get("doc_id") or ref.get("id") or ""
                    snippet = ref.get("snippet") or ref.get("quote")
                elif isinstance(ref, str):
                    doc_id = ref
                    snippet = None
                else:
                    continue
                study = study_index.get(doc_id)
                if study is None:
                    continue
                citations.append(
                    StudyCitation(
                        doc_id=doc_id,
                        title=getattr(study, "title", "Unknown") or "Unknown",
                        citation=getattr(study, "citation", None),
                        year=getattr(study, "year", None),
                        relevance_score=float(getattr(study, "rerank_score", 0) or 0),
                        snippet=snippet,
                    )
                )
            return citations

        recommendation = str(parsed.get("recommendation", "insufficient_evidence")).lower()
        if recommendation not in RECOMMENDATION_VERDICTS:
            recommendation = "insufficient_evidence"

        try:
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return ExpertAssessment(
            specialty=self.specialty,
            display_name=self.display_name,
            recommendation=recommendation,
            recommendation_text=str(
                parsed.get("recommendation_text")
                or parsed.get("assessment")
                or "No assessment produced."
            ),
            confidence=confidence,
            key_questions=[str(q) for q in (parsed.get("key_questions") or []) if q],
            supporting_studies=build_citations(parsed.get("supporting_studies")),
            conflicting_studies=build_citations(parsed.get("conflicting_studies")),
            next_steps=[str(s) for s in (parsed.get("next_steps") or []) if s],
        )
