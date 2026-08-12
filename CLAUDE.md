# Paxis RAG Pipeline — Implementation Brief for Claude Code

> ## ⚠️ STATUS: ALL 7 TASKS IN THIS BRIEF ARE COMPLETE (verified 2026-08-08)
>
> **This brief is kept for design rationale only. Its "current state" tables
> and task list are HISTORICAL and no longer describe the code.** Do not use
> it to decide what still needs building. Verify against the code first.
>
> All seven tasks below are implemented and wired, several more thoroughly
> than the brief specifies:
>
> | Task | Status | Where |
> |---|---|---|
> | 1. LLM extraction complexity gate | Done | `enhanced_rag_service.py:4069` (runs in parallel with embedding, not serially) |
> | 2. PTO frames wired | Done | `comprehensive_retrieval.py:_phase0_pto_search` + third branch of the Phase 1 `asyncio.gather` |
> | 3. Eager Phase 3 dispatch | Done | `comprehensive_retrieval.py` `dispatch_phase3` / `*_and_dispatch` |
> | 4. Phase 3 cross-encoder gate | Done | `_phase3_document_search`, plus a site-mismatch penalty not in the brief |
> | 5. Clinical-axis sub-queries | Done | `_generate_patient_axis_subqueries` |
> | 6. Ontology inference layer | Done | `clinical_inference.py` (~970 lines) |
> | 7. Priority queue + reserved slots | Done | Lane merge with doc_id / title / NCT dedup |
>
> Section 4's "staged but not wired" and "missing, must be built" tables are
> the most misleading part of this file: everything they list now exists.
>
> **For current beta-readiness work, read `BETA_OPTIMIZATION_AUDIT.md`
> instead** — it is code-verified as of 2026-08-08 and lists what actually
> remains. Section 10 ("Do not change") below is still accurate and still
> applies.
>
> **2026-08-12 update**: a separate, later 27-item patient/physician
> convergence program (one canonical patient record + one shared
> evidence/provenance/grounding layer, with audience-specific patient and
> physician RAG policies) is also complete — see
> `PHYSICIAN_BETA_DEPLOYMENT_GATE.md` for its full item list, test
> coverage, and the deployment gate checklist for the new
> `/api/physician-beta/*` routes it added (still off by default behind
> `settings.physician_rag_beta_enabled`). Section 10's "do not change"
> list is what that program built its physician-side legacy retrieval
> adapter around, and remains accurate.

---

## 1. What this repository is

Paxis is a clinical evidence RAG system for oncology. It ingests medical
literature (PDFs → structured chunks), stores them in Qdrant (vector DB) and
PostgreSQL (structured metadata), and answers complex clinical queries by
retrieving and synthesising evidence.

The core user action is submitting a complex patient profile — e.g. an 80-year-old
male with recurrent SCC of the oral tongue, CPS 100, progressing on
pembrolizumab, no longer surgical, with radiographic concern for right
ventricular metastasis — and receiving matched clinical studies with relevant
evidence extracted.

---

## 2. Key files — read these first

```
src/api/services/comprehensive_retrieval.py   ← main retrieval pipeline (modify most)
src/api/services/enhanced_rag_service.py      ← orchestrator, query expansion, cross-encoder
src/api/services/query_structuring_service.py ← regex + LLM axis extraction
src/api/services/structured_study_matcher.py  ← PostgreSQL structured matching
src/api/services/pto_frame_builder.py         ← PTO frame builder (built, not wired)
src/api/services/pto_retriever.py             ← PTO retriever (built, not wired)
src/ingestion/colab_pipeline.py               ← ingestion pipeline
src/ingestion/chunk_processor.py              ← chunking + section windows
```

---

## 3. The problem being solved

**Context fragmentation**: chunk-level cosine similarity misses clinically
relevant studies when the decisive context is spread across multiple sections
(eligibility criteria, methods, results, subgroup analyses) and no single chunk
contains the conjunction of axes that defines relevance.

**Single-vector dilution**: embedding a 200-word patient narrative with 8+
clinical axes as one vector averages the axes together. "ICI-refractory" competes
with "oral tongue SCC" in the same embedding space.

**Vocabulary mismatch**: implicit clinical status ("no longer surgical candidate")
never appears verbatim in literature. Studies say "unresectable" or "inoperable".
Queries say "declined surgery" or "locoregional progression". Neither the query
nor the document contains the other's exact string.

---

## 4. Current state — what is already built vs. what is missing

### ✅ Built and working

| Component | File | Notes |
|---|---|---|
| Section-window chunks | `colab_pipeline.py`, `chunk_processor.py` | Paragraphs aggregated into section windows with keyword union |
| Regex query structuring | `query_structuring_service.py` | Extracts age, gender, TNM, site, histology, treatment, stage, biomarkers |
| PostgreSQL structured matcher | `structured_study_matcher.py` | 9-field dynamic scoring, 3-tier biomarker matching, runs parallel with Qdrant |
| Bidirectional query expansion | `enhanced_rag_service.py` | `ONCOLOGY_EXPANSIONS`, `REVERSE_EXPANSIONS`, `STAGING_SYNONYMS`, `CLINICAL_SYNONYMS` |
| 4-phase comprehensive retrieval | `comprehensive_retrieval.py` | Phase 1: Qdrant+PG parallel. Phase 2: doc_id merge. Phase 3: per-doc search. Phase 4: reranking |
| Cross-encoder reranking | `comprehensive_retrieval.py` | ms-marco-MiniLM-L-6-v2, runs in Phase 4 only |

### 🔶 Staged but not wired

| Component | File | What's missing |
|---|---|---|
| LLM extraction for complex queries | `query_structuring_service.py` | `structure_query_with_llm()` and `_needs_llm_extraction()` exist. Never called from `enhanced_rag_service.py`. Need complexity gate. |
| PTO frame retrieval | `pto_frame_builder.py`, `pto_retriever.py` | Builds and stores Patient→Treatment→Outcome frames in a separate Qdrant collection. `pto_retriever.py` exists but is never called from `comprehensive_retrieval.py`. |

### ❌ Missing — must be built

| Component | Where to build | Description |
|---|---|---|
| Clinical-axis sub-query generation | `comprehensive_retrieval.py` | Replace `_generate_sub_queries()` with patient-profile-derived sub-queries built from extracted axes |
| Eager Phase 3 dispatch | `comprehensive_retrieval.py` | Phase 3 tasks fire as each source completes, not after all three finish |
| Phase 3 cross-encoder gate | `comprehensive_retrieval.py` | Move cross-encoder from Phase 4 into Phase 3 step to reject before Phase 4 |
| Priority queue with reserved slots | `comprehensive_retrieval.py` | Three lanes (PTO/Postgres/Qdrant-only) with separate thresholds and slot reservation |
| Multi-axis query decomposition | `comprehensive_retrieval.py` | Decompose patient profile into 8 clinical axes, embed each separately |
| Ontology inference layer | `src/api/services/clinical_inference.py` (new file) | Implicit → explicit clinical label mapping |

---

## 5. Target pipeline — six layers

```
L0  Clinical axis extraction      LLM fires for complex queries → 8 labelled axes
L1  Ontology expansion per axis   Synonyms + abbreviations + INFERENCE per axis
L2  Three-source intake           Qdrant + PTO + Postgres all parallel
L3  Eager Phase 3 dispatch        Phase 3 fires as each source returns doc_ids
L4  Phase 3: axis sub-queries     Clinical-axis sub-queries + cross-encoder GATE
    + cross-encoder gate
L5  Priority queue                PTO (threshold 0.28, 2 slots) /
                                  Postgres (threshold 0.35, 2 slots) /
                                  Qdrant-only (threshold 0.50, remaining)
L6  Free reranking                Sort by score already computed in L4 — no second model call
```

---

## 6. Implementation tasks — in priority order

Work through these in order. Each task is scoped to specific files and functions.

---

### TASK 1 — Trigger LLM extraction for complex queries

**File**: `src/api/services/enhanced_rag_service.py`

**Where**: In the `query()` method, after `structure_query_fast()` is called
(around the `1b_query_structuring` timing block).

**What to add**:

```python
# After: query_structure = structure_query_fast(query_text, query_type)

from src.api.services.query_structuring_service import (
    _needs_llm_extraction,
    structure_query_with_llm,
    merge_llm_into_structure,  # see note below
)

is_complex = (
    len(query_text) > 150
    or query_text.count(',') > 4
    or any(t in query_text.lower() for t in [
        'progression', 'refractory', 'metastatic', 'recurrent',
        'pembrolizumab', 'nivolumab', 'ici', 's/p', 'status post', 'pmh',
        'ilo', 'locoregional', 'cardiac', 'ventricle',
    ])
)

if is_complex:
    try:
        llm_result = await structure_query_with_llm(query_text)
        if llm_result:
            query_structure = _merge_llm_into_structure(query_structure, llm_result)
            print(f"[Query Structure] LLM extraction applied for complex query")
    except Exception as e:
        print(f"[Query Structure] LLM extraction failed (continuing without): {e}")
```

**Also needed**: Add `_merge_llm_into_structure()` to
`query_structuring_service.py` if it doesn't exist. It should populate
`raw_text` fields on the `QueryStructure` dataclass from the LLM's extracted
spans (the function `populate_structure_raw_text` may already do this —
check and alias or extend).

**Also add** the same trigger to `comprehensive_retrieval.py` in the
`retrieve()` method where `structure_query_fast()` is called.

---

### TASK 2 — Wire PTO frames into `comprehensive_retrieval.py`

**File**: `src/api/services/comprehensive_retrieval.py`

**Goal**: Add PTO frame search as a third parallel branch alongside the existing
Qdrant and PostgreSQL branches.

**Step 2a** — Add PTO search method to `ComprehensiveRetriever`:

```python
async def _phase0_pto_search(
    self,
    query_text: str,
    query_embedding: List[float],
    query_structure,
    limit: int = 30,
) -> Set[str]:
    """
    Phase 0: Search PTO frame index for document-level patient profile matches.
    Returns set of doc_ids whose PTO frame matched the patient profile.
    """
    try:
        from src.api.services.pto_retriever import PTORetriever
        retriever = PTORetriever(
            qdrant_client=self.qdrant,
            openai_client=self.openai,
        )
        results = await retriever.search(
            query_embedding=query_embedding,
            limit=limit,
        )
        doc_ids = set()
        for r in results:
            doc_id = r.get("doc_id") or r.get("payload", {}).get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)
        print(f"[ComprehensiveRetrieval] PTO search: {len(doc_ids)} doc_ids")
        return doc_ids
    except Exception as e:
        print(f"[ComprehensiveRetrieval] PTO search failed (continuing without): {e}")
        return set()
```

**Step 2b** — In the `retrieve()` method, add PTO as a third task in the
`asyncio.gather` call that currently only runs `qdrant_task` and `postgres_task`:

```python
pto_task = self._phase0_pto_search(
    query_text=query_text,
    query_embedding=query_embedding,
    query_structure=query_structure,
    limit=30,
)

if postgres_task:
    qdrant_results, postgres_result, pto_doc_ids = await asyncio.gather(
        qdrant_task, postgres_task, pto_task
    )
else:
    qdrant_results, pto_doc_ids = await asyncio.gather(qdrant_task, pto_task)
    postgres_result = None
```

**Step 2c** — In the Phase 2 doc_id merge block, handle PTO doc_ids:

```python
# After processing Qdrant and Postgres results into doc_info...
for pto_doc_id in pto_doc_ids:
    if pto_doc_id in doc_info:
        doc_info[pto_doc_id]["source"] = "pto"  # or "pto+qdrant" / "pto+both"
        doc_info[pto_doc_id]["score"] = max(doc_info[pto_doc_id]["score"], 0.7)
    else:
        doc_info[pto_doc_id] = {
            "score": 0.65,
            "source": "pto",
            "doc_meta": {},
            "category": None,
        }
```

---

### TASK 3 — Eager Phase 3 dispatch

**File**: `src/api/services/comprehensive_retrieval.py`

**Goal**: Phase 3 tasks fire as each source completes its search, not after all
three sources have finished. This uses `asyncio.create_task()` inside each
source's completion handler so Phase 3 runs on Qdrant results during the
~320ms that Postgres is still computing.

**Replace** the current Phase 1+2+3 block with this pattern:

```python
# Registry — keyed by doc_id to prevent duplicate Phase 3 tasks
phase3_registry: Dict[str, asyncio.Task] = {}
doc_info: Dict[str, Dict] = {}

# Source → cross-encoder threshold for the Phase 3 gate (Task 4)
SOURCE_THRESHOLDS = {
    "pto":      0.28,
    "postgres": 0.35,
    "both":     0.28,  # matched both → lowest threshold
    "qdrant":   0.50,
}
# Source → reserved confirmation slots (out of max_studies=5)
SOURCE_RESERVED = {
    "pto":      2,
    "postgres": 2,
    "qdrant":   0,  # fills remaining slots
}

def dispatch_phase3(doc_id: str, source: str, score: float,
                    doc_meta: dict, category=None):
    """Dispatch a Phase 3 task if not already running for this doc_id."""
    if doc_id in phase3_registry:
        # Upgrade trust level if new source is higher precision
        existing_source = doc_info[doc_id]["source"]
        precision_rank = {"qdrant": 0, "postgres": 1, "pto": 2, "both": 3}
        if precision_rank.get(source, 0) > precision_rank.get(existing_source, 0):
            doc_info[doc_id]["source"] = source
            doc_info[doc_id]["threshold"] = SOURCE_THRESHOLDS[source]
            print(f"[EagerDispatch] Upgraded {doc_id[:40]} trust: "
                  f"{existing_source} → {source}")
        return

    doc_info[doc_id] = {
        "source": source,
        "score": score,
        "threshold": SOURCE_THRESHOLDS.get(source, 0.45),
        "doc_meta": doc_meta,
        "category": category,
    }
    task = asyncio.create_task(
        self._tagged_phase3(
            doc_id=doc_id,
            query_embedding=query_embedding,
            expanded_query=expanded_query,
            max_chunks=chunks_per_study,
            query_type=query_type,
        )
    )
    phase3_registry[doc_id] = task

async def qdrant_and_dispatch():
    hits = await self._phase1_qdrant_search(
        query_embedding=query_embedding,
        expanded_query=expanded_query,
        category=category,
        limit=100,  # increase from 50 for better recall
    )
    for hit in hits:
        doc_id = hit.get("doc_id")
        if doc_id:
            dispatch_phase3(doc_id, "qdrant", hit.get("score", 0),
                           hit.get("doc_meta", {}), hit.get("category"))
    return hits

async def postgres_and_dispatch():
    if not postgres_task:
        return None
    result = await postgres_task
    if not result or not result.doc_ids:
        return result
    for pg_doc_id in result.doc_ids:
        pg_score = result.match_scores.get(pg_doc_id, 0.5)
        source = "both" if pg_doc_id in doc_info else "postgres"
        dispatch_phase3(pg_doc_id, source, pg_score, {})
    return result

async def pto_and_dispatch():
    pto_ids = await self._phase0_pto_search(
        query_text=query_text,
        query_embedding=query_embedding,
        query_structure=query_structure,
    )
    for pto_doc_id in pto_ids:
        source = "both" if pto_doc_id in doc_info else "pto"
        dispatch_phase3(pto_doc_id, source, 0.65, {})
    return pto_ids

# Fire all three — Phase 3 tasks start mid-gather as sources return
await asyncio.gather(
    qdrant_and_dispatch(),
    postgres_and_dispatch(),
    pto_and_dispatch(),
)
```

**Add the tagged wrapper** (needed so `asyncio.as_completed` can match results
to doc_ids):

```python
async def _tagged_phase3(
    self,
    doc_id: str,
    query_embedding: List[float],
    expanded_query: str,
    max_chunks: int,
    query_type: str,
) -> Tuple[str, List[Dict]]:
    """Wraps _phase3_document_search to return (doc_id, chunks)."""
    chunks = await self._phase3_document_search(
        doc_id=doc_id,
        query_embedding=query_embedding,
        expanded_query=expanded_query,
        max_chunks=max_chunks,
        query_type=query_type,
    )
    return doc_id, chunks
```

---

### TASK 4 — Cross-encoder gate inside Phase 3

**File**: `src/api/services/comprehensive_retrieval.py`

**Goal**: Add cross-encoder scoring as step 3c inside `_phase3_document_search()`,
before returning chunks. Studies that fail the gate are rejected at this point
rather than at Phase 4.

**Add to `_phase3_document_search()`** after hybrid scoring and multi-query
boost, before deduplication:

```python
# Step 3c: Cross-encoder gate
# threshold is carried in via a parameter (sourced from doc_info in the
# dispatch registry) or falls back to 0.45 for safety
gate_threshold = kwargs.get("gate_threshold", 0.45)

cross_encoder = self._get_cross_encoder()
if cross_encoder is not None and chunks_list:
    # Build combined study text (first ~2000 chars of top chunks)
    combined_text = ""
    for chunk in chunks_list[:4]:
        t = chunk.get("text", "")
        if len(combined_text) + len(t) < 2000:
            combined_text += " " + t
        else:
            break
    combined_text = combined_text.strip()

    if combined_text:
        ce_score = float(cross_encoder.predict([(expanded_query, combined_text)]))
        # Store on each chunk for Phase 6 reranking (free — already computed)
        for chunk in chunks_list:
            chunk["score_crossencoder_gate"] = ce_score

        if ce_score < gate_threshold:
            print(f"[Phase3Gate] REJECTED {doc_id[:40]} "
                  f"ce_score={ce_score:.3f} < threshold={gate_threshold:.3f}")
            return []   # empty list → study will not be confirmed
        else:
            print(f"[Phase3Gate] PASSED   {doc_id[:40]} "
                  f"ce_score={ce_score:.3f}")
```

**Pass `gate_threshold` from the dispatch registry** by threading it through
`_tagged_phase3()` → `_phase3_document_search()` as a kwarg or explicit param.

**Update `_tagged_phase3()`**:
```python
async def _tagged_phase3(self, doc_id, query_embedding, expanded_query,
                          max_chunks, query_type, gate_threshold=0.45):
    chunks = await self._phase3_document_search(
        doc_id=doc_id,
        query_embedding=query_embedding,
        expanded_query=expanded_query,
        max_chunks=max_chunks,
        query_type=query_type,
        gate_threshold=gate_threshold,
    )
    return doc_id, chunks
```

And pass `doc_info[doc_id]["threshold"]` when creating the task in `dispatch_phase3()`.

---

### TASK 5 — Replace `_generate_sub_queries()` with clinical-axis sub-queries

**File**: `src/api/services/comprehensive_retrieval.py`

**Goal**: Replace the current generic sub-query generator with one that derives
sub-queries from the patient's extracted clinical axes.

**Replace** `_generate_sub_queries()` entirely:

```python
def _generate_patient_axis_subqueries(
    self,
    main_query: str,
    query_type: str,
    query_structure=None,
    inferred_axes: dict = None,
) -> Dict[str, str]:
    """
    Generate sub-queries derived from the patient's specific clinical axes.

    Falls back to generic sub-queries when no query_structure is available
    (preserves backward compatibility for non-patient queries).
    """
    inferred_axes = inferred_axes or {}

    # ── Fallback: generic sub-queries for non-patient queries ─────────────
    if query_structure is None or not query_structure.has_patient_context:
        sub = {}
        q = main_query.lower()
        if "outcome" not in q and "survival" not in q:
            sub["outcomes"] = f"{main_query} outcomes survival results efficacy"
        if query_type in ("dose_question", "treatment_recommendation"):
            if "dose" not in q:
                sub["dosing"] = f"{main_query} dose fractionation regimen"
        if query_type in ("side_effects", "treatment_recommendation"):
            if "toxicity" not in q:
                sub["toxicity"] = f"{main_query} toxicity adverse effects"
        return dict(list(sub.items())[:3])

    # ── Patient-profile-derived sub-queries ───────────────────────────────
    sub = {}

    # 1. Primary cancer identity
    cancer = query_structure.cancer
    if cancer.site:
        sub["primary"] = (
            f"{cancer.site} {cancer.histology or ''} "
            f"{cancer.stage or ''} treatment outcomes"
        ).strip()

    # 2. ICI trajectory — only if detected
    trajectory_flags = inferred_axes.get("trajectory_flags", [])
    current_therapy = getattr(query_structure.treatment, "current_therapy", None)
    if any(f in trajectory_flags for f in ("ici_refractory", "progressing_on_ici")):
        sub["ici_refractory"] = (
            f"{cancer.site or ''} {cancer.histology or ''} "
            "ICI-refractory anti-PD1 failure second-line salvage "
            "checkpoint inhibitor resistant"
        ).strip()
    elif current_therapy and any(
        t in current_therapy.lower()
        for t in ("pembrolizumab", "nivolumab", "ici", "checkpoint")
    ):
        sub["immunotherapy"] = (
            f"{cancer.site or ''} {current_therapy} "
            "outcomes response progression"
        ).strip()

    # 3. Biomarker-specific evidence
    biomarkers = cancer.biomarkers or []
    cps_marker = next((b for b in biomarkers if "CPS" in b.upper()), None)
    pdl1_marker = next(
        (b for b in biomarkers if "PD-L1" in b.upper() or "PDL1" in b.upper()),
        None,
    )
    if cps_marker or pdl1_marker:
        sub["biomarker"] = (
            f"CPS PD-L1 expression pembrolizumab "
            f"{cancer.histology or ''} {cancer.site or ''} immunotherapy"
        ).strip()

    # 4. Metastatic / staging concern
    met_sites = inferred_axes.get("metastatic_sites", [])
    if met_sites:
        sub["metastatic"] = (
            f"{' '.join(met_sites)} metastasis outcomes prognosis systemic"
        )

    # 5. Non-surgical / eligibility
    surgical_candidate = inferred_axes.get("surgical_candidate")
    if surgical_candidate is False:
        sub["eligibility"] = (
            "unresectable inoperable locoregional advanced "
            "systemic therapy eligibility"
        )

    # 6. Comorbidity → treatment eligibility inference
    comorbidities = getattr(query_structure.patient, "comorbidities", [])
    CISPLATIN_CONTRAINDICATIONS = {"CKD", "renal", "kidney", "creatinine"}
    if any(
        any(c_word in c.lower() for c_word in CISPLATIN_CONTRAINDICATIONS)
        for c in comorbidities
    ):
        sub["comorbidity"] = (
            "cisplatin ineligible carboplatin renal impairment "
            "modified regimen toxicity"
        )

    # Cap at 5 sub-queries; always include primary
    result = {}
    if "primary" in sub:
        result["primary"] = sub.pop("primary")
    for k, v in list(sub.items())[:4]:
        result[k] = v

    return result
```

**Update the call site** in `_phase3_document_search()` — replace:
```python
sub_queries = self._generate_sub_queries(expanded_query, query_type)
```
with:
```python
sub_queries = self._generate_patient_axis_subqueries(
    main_query=expanded_query,
    query_type=query_type,
    query_structure=query_structure,      # thread this through from retrieve()
    inferred_axes=inferred_axes or {},    # from Task 6
)
```

You will need to thread `query_structure` and `inferred_axes` through
`_tagged_phase3()` → `_phase3_document_search()` as kwargs.

---

### TASK 6 — Build the ontology inference layer (new file)

**File**: `src/api/services/clinical_inference.py` (create new)

**Goal**: Map implicit clinical status from a patient narrative to the explicit
labels that appear in literature. This is the most impactful change for recall
of complex patient profiles.

```python
"""
Clinical Inference Layer

Maps implicit clinical facts in patient narratives to explicit clinical labels
used in the literature. Called after LLM axis extraction and before embedding.

Example:
    "no longer a surgical candidate following locoregional progression"
    → adds: unresectable, inoperable, salvage not feasible, locoregional failure

    "progressing on pembrolizumab"
    → adds: ICI-refractory, anti-PD1 failure, checkpoint refractory, 2nd-line
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ── Inference map ──────────────────────────────────────────────────────────────
# Each entry: trigger pattern → list of terms to add to that axis
# Patterns are matched case-insensitively against the raw narrative text.

INFERENCE_MAP: Dict[str, List[str]] = {

    # ── Surgical candidacy ──────────────────────────────────────────────────
    r"no longer (?:a )?surgical candidate": [
        "unresectable", "inoperable", "salvage surgery not feasible",
        "non-surgical management",
    ],
    r"not (?:a )?surgical candidate": [
        "unresectable", "inoperable", "non-surgical",
    ],
    r"declined surgery": [
        "surgery declined", "non-surgical", "systemic therapy only",
    ],
    r"unresectable": [
        "inoperable", "non-surgical candidate", "locoregional advanced",
    ],

    # ── ICI / checkpoint status ─────────────────────────────────────────────
    r"progress(?:ing|ion) on (?:pembrolizumab|nivolumab|atezolizumab|durvalumab|ICI|immunotherapy|checkpoint)": [
        "ICI-refractory", "anti-PD1 failure", "anti-PD-L1 failure",
        "checkpoint inhibitor refractory", "post-immunotherapy",
        "second-line", "2nd-line systemic", "salvage",
    ],
    r"refractory to (?:pembrolizumab|nivolumab|ICI|immunotherapy|checkpoint)": [
        "ICI-refractory", "checkpoint inhibitor refractory",
        "anti-PD1 failure", "post-ICI progression",
    ],
    r"ICI.{0,15}(?:progress|fail|refract)": [
        "ICI-refractory", "checkpoint inhibitor failure",
        "2nd-line", "post-checkpoint progression",
    ],
    r"locoregional progression on ICI": [
        "ICI-refractory HNSCC", "anti-PD1 failure", "locoregional failure",
        "2nd-line systemic therapy",
    ],

    # ── Line of therapy ─────────────────────────────────────────────────────
    r"started on (?:pembrolizumab|nivolumab|atezolizumab)": [
        "first-line immunotherapy", "1L checkpoint inhibitor",
    ],
    r"second.line|2nd.line|salvage therapy": [
        "second-line systemic", "salvage", "post-progression treatment",
    ],

    # ── CPS / PD-L1 thresholds ──────────────────────────────────────────────
    # A patient with CPS 100 is eligible for all lower thresholds
    r"CPS\s*(?:score\s*(?:of\s*)?)?(?:=\s*)?100": [
        "CPS ≥ 1", "CPS ≥ 20", "CPS ≥ 50", "CPS ≥ 80", "CPS 100",
        "PD-L1 high expression", "immunotherapy eligible",
        "pembrolizumab indicated",
    ],
    r"CPS\s*(?:score\s*(?:of\s*)?)?(?:=\s*)?\d{2,}": [
        "CPS ≥ 1", "CPS ≥ 20", "PD-L1 positive", "immunotherapy eligible",
    ],
    r"CPS\s*(?:score\s*(?:of\s*)?)?(?:=\s*)?\d+": [
        "CPS ≥ 1", "PD-L1 expression",
    ],

    # ── Metastatic / cardiac ────────────────────────────────────────────────
    r"right ventricl": [
        "cardiac metastasis", "right ventricular metastasis",
        "intracardiac metastasis", "right heart involvement",
        "cardiac involvement", "distant metastasis",
    ],
    r"(?:concern for|suspected|radiographic)\s+metastatic disease": [
        "suspected distant metastasis", "metastatic workup",
        "systemic disease", "M1 disease",
    ],
    r"metastatic disease to the (?:right ventricle|heart|cardiac)": [
        "cardiac metastasis", "intracardiac metastasis", "M1 disease",
        "distant metastasis", "right ventricular involvement",
    ],

    # ── Comorbidity → treatment implications ────────────────────────────────
    r"\bCKD\b|chronic kidney disease|renal impairment": [
        "cisplatin ineligible", "carboplatin preferred",
        "renal insufficiency", "dose modification required",
    ],
    r"\bHep(?:atitis)?\s*C\b|HCV": [
        "hepatic comorbidity", "liver disease",
        "immunosuppression risk", "hepatic function impaired",
    ],
    r"declined (?:combination with )?chemotherapy": [
        "chemotherapy refused", "immunotherapy monotherapy",
        "chemo-free regimen", "single-agent systemic",
    ],

    # ── Recurrence patterns ─────────────────────────────────────────────────
    r"biopsy.proven recurrent": [
        "biopsy-confirmed recurrence", "pathologically confirmed recurrence",
        "recurrent/metastatic", "R/M disease",
    ],
    r"recurrent (?:SCC|squamous|lesion|disease)": [
        "locoregional recurrence", "recurrent/metastatic HNSCC",
        "R/M HNSCC", "recurrent disease", "salvage setting",
    ],
    r"multiloculated.*collection|sublingual.*collection": [
        "abscess formation", "locoregional complication",
        "post-operative collection", "neck abscess",
    ],

    # ── HPV / p16 ───────────────────────────────────────────────────────────
    r"\bp16\+|HPV.positive|HPV.associated": [
        "HPV-positive HNSCC", "p16-positive", "HPV-related oropharyngeal",
        "favorable biology",
    ],
    r"\bp16-|HPV.negative|HPV.unrelated": [
        "HPV-negative HNSCC", "p16-negative",
        "non-HPV-associated", "unfavorable biology",
    ],
}


@dataclass
class InferenceResult:
    """Result of inference layer processing."""
    original_axes: Dict[str, str]
    expanded_axes: Dict[str, str]
    inferred_terms: Dict[str, List[str]]   # axis_name → list of added terms
    trajectory_flags: List[str] = field(default_factory=list)
    metastatic_sites: List[str] = field(default_factory=list)
    surgical_candidate: Optional[bool] = None


def run_inference(
    raw_text: str,
    axes: Dict[str, str],
) -> InferenceResult:
    """
    Apply inference rules to raw patient narrative and expand axes.

    Args:
        raw_text: Full raw patient narrative (used for pattern matching)
        axes: Dict of axis_name → axis_string (from LLM extraction)

    Returns:
        InferenceResult with expanded axes and extracted flags
    """
    raw_lower = raw_text.lower()
    added: Dict[str, List[str]] = {k: [] for k in axes}
    trajectory_flags: List[str] = []
    metastatic_sites: List[str] = []
    surgical_candidate: Optional[bool] = None

    # Apply each inference rule against the full narrative
    for pattern, terms in INFERENCE_MAP.items():
        if re.search(pattern, raw_text, re.IGNORECASE):
            # Determine which axis this inference best belongs to
            axis = _assign_to_axis(pattern, terms, axes)
            added[axis] = list(set(added[axis] + terms))

    # Extract trajectory flags
    if re.search(r"progress(?:ing|ion) on .{0,40}(?:pembrolizumab|ICI|checkpoint|immunotherapy)", raw_text, re.IGNORECASE):
        trajectory_flags.append("ici_refractory")
        trajectory_flags.append("progressing_on_ici")
    if re.search(r"no longer (?:a )?surgical|not (?:a )?surgical candidate|unresectable", raw_text, re.IGNORECASE):
        surgical_candidate = False

    # Extract metastatic sites
    met_patterns = [
        (r"right ventricl", "right ventricle"),
        (r"cardiac|heart", "cardiac"),
        (r"lung metastasis|pulmonary met", "lung"),
        (r"liver metastasis|hepatic met", "liver"),
        (r"bone metastasis|osseous met", "bone"),
        (r"brain metastasis|cerebral met", "brain"),
    ]
    for pat, label in met_patterns:
        if re.search(pat, raw_text, re.IGNORECASE):
            metastatic_sites.append(label)

    # Build expanded axes
    expanded_axes = {}
    for axis_name, axis_str in axes.items():
        extra = added.get(axis_name, [])
        if extra:
            expanded_axes[axis_name] = axis_str + " " + " ".join(extra)
        else:
            expanded_axes[axis_name] = axis_str

    return InferenceResult(
        original_axes=axes,
        expanded_axes=expanded_axes,
        inferred_terms=added,
        trajectory_flags=trajectory_flags,
        metastatic_sites=metastatic_sites,
        surgical_candidate=surgical_candidate,
    )


def _assign_to_axis(pattern: str, terms: List[str], axes: Dict[str, str]) -> str:
    """Heuristically assign inferred terms to the most relevant axis."""
    # Pattern-to-axis hints
    axis_hints = {
        "ventricl": "metastatic_concern",
        "surgical": "patient_factors",
        "ICI|checkpoint|pembrolizumab|nivolumab": "disease_trajectory",
        "CPS|PD-L1": "biomarker_profile",
        "CKD|Hep|renal|hepatic": "patient_factors",
        "recurrent": "disease_trajectory",
        "line|salvage": "disease_trajectory",
        "HPV|p16": "biomarker_profile",
    }
    for hint_pat, axis_name in axis_hints.items():
        if re.search(hint_pat, pattern, re.IGNORECASE):
            if axis_name in axes:
                return axis_name
    # Fallback: first available axis
    return next(iter(axes), "primary_cancer")


def apply_inference_to_query_structure(query_structure, raw_text: str) -> dict:
    """
    Convenience wrapper: extract axes from query_structure, run inference,
    return inferred_axes dict for use in sub-query generation.
    """
    # Build basic axis dict from query_structure
    cancer = query_structure.cancer
    treatment = getattr(query_structure, "treatment", None)
    patient = getattr(query_structure, "patient", None)

    axes = {
        "primary_cancer": f"{cancer.site or ''} {cancer.histology or ''} {cancer.stage or ''}".strip(),
        "biomarker_profile": " ".join(cancer.biomarkers or []),
        "disease_trajectory": getattr(treatment, "raw_text", "") or "",
        "patient_factors": " ".join(getattr(patient, "comorbidities", []) or []),
        "metastatic_concern": "",
        "treatment_history": getattr(treatment, "raw_text", "") or "",
    }

    result = run_inference(raw_text, axes)

    return {
        "expanded_axes": result.expanded_axes,
        "trajectory_flags": result.trajectory_flags,
        "metastatic_sites": result.metastatic_sites,
        "surgical_candidate": result.surgical_candidate,
        "inferred_terms": result.inferred_terms,
    }
```

**Wire into the query flow** in `enhanced_rag_service.py` after LLM extraction
(Task 1):

```python
# After: query_structure = _merge_llm_into_structure(query_structure, llm_result)

from src.api.services.clinical_inference import apply_inference_to_query_structure
inferred_axes = apply_inference_to_query_structure(query_structure, query_text)
print(f"[Inference] Flags: {inferred_axes['trajectory_flags']}")
print(f"[Inference] Met sites: {inferred_axes['metastatic_sites']}")
```

Pass `inferred_axes` through to `comprehensive_retrieval.py` `retrieve()` and
then into `_phase3_document_search()` so it reaches sub-query generation.

---

### TASK 7 — Priority queue with reserved slots and `as_completed` gate

**File**: `src/api/services/comprehensive_retrieval.py`

**Goal**: Replace the current "sort → take top N → run all Phase 3" pattern
with an `as_completed` loop that confirms studies into source-specific lanes
with reserved slots.

**Replace** the current task collection block after `asyncio.gather` (the loop
that builds `StudyEvidence` objects) with:

```python
# ── Priority queue ────────────────────────────────────────────────────────
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
        category=info.get("category"),
        initial_score=info.get("score", 0),
        chunks=chunks,
        sections_covered=sections,
        source=source,
    )

    lane = "both" if source == "both" else source if source in ("pto", "postgres") else "qdrant"
    confirmed_by_source[lane].append(study)

    # ── Early termination: cancel Qdrant-only tasks if we have enough ──
    total_confirmed = sum(len(v) for v in confirmed_by_source.values())
    qdrant_confirmed = len(confirmed_by_source["qdrant"])
    high_precision_confirmed = (
        len(confirmed_by_source["pto"])
        + len(confirmed_by_source["both"])
        + len(confirmed_by_source["postgres"])
    )

    # Only cancel Qdrant-only tasks — never cancel PTO or Postgres tasks
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

# ── Merge lanes into final ordered list ──────────────────────────────────
# Reserved slots: pto (2) + postgres (2) + qdrant fills remaining
pto_studies    = confirmed_by_source["pto"][:2] + confirmed_by_source["both"][:2]
postgres_studies = [s for s in confirmed_by_source["postgres"]
                    if s not in pto_studies][:2]
qdrant_studies = confirmed_by_source["qdrant"]

reserved_used = len(pto_studies) + len(postgres_studies)
remaining_slots = max(0, max_studies - reserved_used)

studies = pto_studies + postgres_studies + qdrant_studies[:remaining_slots]
```

---

## 7. Ontology synonym extensions needed in `enhanced_rag_service.py`

Add these groups to `CLINICAL_SYNONYMS` and the bidirectional expansion tables:

```python
# ICI/checkpoint synonyms
"pembrolizumab": "nivolumab atezolizumab durvalumab anti-PD1 anti-PD-L1 checkpoint inhibitor CPI immunotherapy",
"ICI": "immune checkpoint inhibitor anti-PD1 anti-PD-L1 checkpoint blockade immunotherapy",

# Disease status synonyms
"recurrent/metastatic": "R/M locoregional recurrence distant metastasis unresectable recurrent metastatic",

# Line of therapy
"second-line": "2nd-line salvage subsequent therapy post-progression post-ICI",
"first-line": "1st-line frontline initial treatment naive",

# HNSCC sub-site
"oral tongue": "mobile tongue tongue body anterior tongue",
"oropharynx": "tonsil base of tongue BOT soft palate posterior pharyngeal wall",

# CPS numeric ranges (add to a new CPS_SYNONYMS dict, not string-based)
# See: clinical_inference.py CPS threshold expansion
```

---

## 8. Files to modify — summary

| File | Change type |
|---|---|
| `src/api/services/enhanced_rag_service.py` | Add complexity gate → trigger LLM extraction; wire inference layer |
| `src/api/services/comprehensive_retrieval.py` | Eager dispatch; PTO branch; Phase 3 gate; axis sub-queries; priority queue |
| `src/api/services/query_structuring_service.py` | Add `_merge_llm_into_structure()` if missing |
| `src/api/services/clinical_inference.py` | **New file** — create from scratch per Task 6 |
| `src/api/services/enhanced_rag_service.py` | Extend `CLINICAL_SYNONYMS` with ICI/checkpoint/disease-status groups |

---

## 9. Testing after each task

After each task, test with this patient profile string before moving to the next:

```
80 y.o. male non-smoker with a PMH HTN, Hep C, BPH, CKD, latent syphilis,
transverse colon adenocarcinoma complicated by LBO s/p (6/16/21) diagnostic lap,
ex lap with extended right hemicolectomy 6/2021 and ileostomy reversal 10/6/2021,
and initial Stage II (pT2pN0M0R0, DOI 5.1 mm, PNI-, LVSI-) squamous cell carcinoma
of the left oral tongue, status post left partial glossectomy, left neck dissection
levels I-III, and radial forearm free flap reconstruction, and left STSG performed
at Bellevue Hospital on 12/2/2024 with Dr. Moses. In August 2025, he developed a
recurrent lesion in the left level I neck associated with a multiloculated left
sub-lingual collection, which was biopsy-proven recurrent SCC with a CPS score of
100, started on pembrolizumab (declined combination with chemotherapy) and is no
longer a surgical candidate following significant locoregional progression on ICI
with radiographic concern for metastatic disease to the right ventricle and
progressing on systemic therapy.
```

**Expected after Task 1**: LLM extraction fires, 7 clinical fields populated.

**Expected after Task 2**: PTO search returns doc_ids, prints count.

**Expected after Task 3**: Logs show Phase 3 tasks starting before Postgres
completes (watch timestamps).

**Expected after Task 4**: Logs show "REJECTED" and "PASSED" for each doc_id
with cross-encoder scores.

**Expected after Task 5**: Sub-query dict includes `ici_refractory`, `biomarker`,
`metastatic`, `eligibility` keys — not just `outcomes`/`dosing`/`toxicity`.

**Expected after Task 6**: Inferred terms include "ICI-refractory",
"unresectable", "CPS ≥ 1", "cardiac metastasis", "cisplatin ineligible".

**Expected after Task 7**: Final confirmed studies include at least 2 from PTO
or Postgres lanes for this complex profile.

---

## 10. Do not change

- The `StudyEvidence` and `ComprehensiveRetrievalResult` dataclasses (only add fields)
- The `_phase4_rerank_studies()` method signature (it now uses the score from
  Task 4 — just ensure `score_crossencoder_gate` is read when available)
- The ingestion pipeline (`colab_pipeline.py`, `chunk_processor.py`)
- The PostgreSQL structured matcher (`structured_study_matcher.py`) — it is
  already well-built and needs no changes for this sprint
- Authentication, upload, or frontend code

---

*End of implementation brief. Start with Task 1.*
