# Beta Optimization Audit

Full audit of the site and repo ahead of beta, focused on two questions: why would a user wait too long for a response, and where does the logic have holes in how queries are processed, patient profiles are parsed, and evidence is retrieved. Every claim below was verified against the current code in this session, with file and line references. This complements BETA_SCALING_PROPOSAL.md (which covers infrastructure and the patient DB); where a finding from that document was re-checked here, its current status is stated.

A note up front: the core pipeline is in much better shape than the CLAUDE.md brief implies. All seven of the brief's tasks (LLM extraction gate, PTO wiring, eager dispatch, cross-encoder gate, axis sub-queries, inference layer, priority queue) are implemented, and several are more sophisticated than what the brief asked for. The findings below are the gaps that remain.

---

## Part 1: Response speed

### 1.1 No streaming anywhere (biggest perceived-wait lever)

There is no `StreamingResponse` or server-sent-events usage in `query.py` or anywhere in the API. The frontend shows a static "Thinking..." message (`index.html:1853`) and then waits for the entire pipeline to finish: classification, expansion, embedding, three-source retrieval, per-document cross-encoder gates, an LLM eligibility check, and a full GPT-4o answer, before showing a single word. For a complex patient query that whole chain is plausibly 20 to 40 seconds of blank waiting.

Streaming the final answer token by token would put first words on screen seconds after generation starts instead of after it ends. Even without full streaming, staged progress messages ("Matching studies... 4 found... Verifying eligibility... Writing answer") would transform how the wait feels. The backend already prints exactly these stage transitions to logs; surfacing them to the client is the cheap version of this fix.

### 1.2 Cross-encoder gate scores one document at a time (biggest actual-latency lever)

Confirmed still current. The Phase 3 gate calls `cross_encoder.predict()` once per candidate document (`comprehensive_retrieval.py:2172`), and those calls run inside each document's Phase 3 task. A correctly batched version of the same operation already exists in the same file (`:2622`, the Phase 4 fallback). Batching all pending gate scorings into one predict call is the single biggest retrieval latency win available and the pattern to copy is 450 lines away.

### 1.3 Blocking calls still on the event loop

Two spots freeze the whole server process for every other concurrent user while they run:

- `comprehensive_retrieval.py:900`: `self.embed_query(expanded_query)` is called directly in the async `retrieve()`, even though the async wrapper `_embed_async()` exists at line 462 of the same file and is used correctly elsewhere (lines 1861, 2035).
- `enhanced_rag_service.py:4303`: the else-branch embedding call (taken by every query that does not trigger LLM extraction, meaning most simple queries) is a raw blocking call. The parallel branch 80 lines up correctly uses `asyncio.to_thread`.

Also: the eligibility service builds a fresh `QdrantClient` and runs up to 10 sequential blocking `.scroll()` calls per query (`patient_eligibility_boost_service.py:1367-1381`). These should go through the thread-pool wrapper and reuse a shared client.

For contrast, answer generation is already handled correctly (`enhanced_rag_service.py:5952`, wrapped in `to_thread` with a comment explaining exactly why). The same treatment just needs to reach the remaining spots.

### 1.4 One worker, no concurrency cap (unchanged)

`Dockerfile` still runs `gunicorn --workers 1`, and the deploy command in DEPLOY.md still sets no `--concurrency`, so Cloud Run's default of 80 concurrent requests can pile onto a single Python process. This remains the highest-priority infrastructure fix from BETA_SCALING_PROPOSAL.md. With one worker, every blocking call in 1.3 stalls every concurrent user.

### 1.5 Feature flags: production is running with all optimizations off

All the Phase 10 flags in `config.py` default to False, and the local `.env` sets none of them. Unless they are set in Cloud Run's environment variables (not confirmable from the repo, worth checking in the Console), production is running with:

- `enable_perf_optimizations=False`: the Postgres result cache is off, and the Qdrant client timeout is 60s instead of 15s.
- `enable_pto_retrieval=False`: PTO search uses the legacy direct-Qdrant path rather than the newer PTORetriever path. PTO still runs either way, but which implementation runs in prod should be a decision, not an accident.
- `enable_soft_scorer=False`: the post-eligibility soft scoring layer is dead code in prod.

Recommendation: decide explicitly which flags beta runs with, set them in Cloud Run, and record the choice in DEPLOY.md.

### 1.6 Structured matcher still opens a raw connection per query

`structured_study_matcher.py:1515` still calls `asyncpg.connect()` per invocation. It now has retry with backoff (an improvement since the code comment at :1504 describes the observed 5-way concurrency failure), but retry is a bandage: under concurrent beta load, every query still opens and closes its own Postgres connection on the busiest path in the product. Convert to a shared pool, same pattern as `account_db.py` / `patient_db.py`.

### 1.7 Cost and latency waste: collection seeding generates a discarded answer

`patient_collection_seeder.py:98` calls the full `rag_service.query()`, which runs GPT-4o answer generation, then uses only the evidence list and throws the answer away. Every patient intake with auto-seed pays several seconds and a GPT-4o bill for text nobody sees. It should call the retrieval path directly (`retriever.retrieve()` or `retrieve_evidence`), skipping generation entirely.

### 1.8 Per-query client construction

The comprehensive path constructs a fresh OpenAI client for the eligibility check on every query (`comprehensive_retrieval.py:1436`), and the eligibility service constructs a fresh QdrantClient (1.3 above). The retriever singleton itself is done right (gRPC transport, shared client, `:2649-2668`); the stragglers should reuse those.

### 1.9 No warmup

Nothing preloads the cross-encoder model or opens DB pools at startup, so the first query after any new instance boots (including the min-instance after a deploy) eats a multi-second cold penalty. A small startup hook that loads the model and touches each pool fixes this.

### 1.10 Frontend never times out

No `AbortController` or fetch timeout in the frontend API layer, and gunicorn allows 300s per request. If the backend hangs, the user watches "Thinking..." indefinitely. Add a client-side timeout (60-90s) with a friendly retry message.

---

## Part 2: Logic holes

### 2.1 Patient age never reaches the seeded search

`patient_collection_seeder.py:28` reads `date_of_birth` into a variable and then never uses it. The seed query gets sex, diagnosis, biomarkers, and treatment history, but not age, despite age being an axis the query pipeline extracts and matches on. An 80-year-old and a 40-year-old with identical diagnoses seed identical collections. Small fix: compute age from DOB and prepend it (the narrative format the pipeline expects starts with exactly this, "80 y.o. male...").

### 2.2 Comorbidities are a dead end for structured patients

The free-text path handles comorbidities well: `clinical_inference.py` maps CKD to cisplatin-ineligible, hepatitis to immunosuppression risk, and so on. But for structured patients this axis is lost twice over. The patient DB schema has no comorbidities field (patients, diagnosis, biomarkers, treatment_history, timeline only), so intake cannot store them, and the seeder cannot include them. And in the matching preferences panel, `_USER_KEY_MAP` (`structured_study_matcher.py:898`) maps `comorbidities`, `race`, `recurrence_status`, `grade`, and `tumor_size` to empty lists, so any user weight applied to them silently does nothing. The UI-side inputs were hidden in a previous session; the deeper fix (store comorbidities on the patient record and route them into matching) is a schema-plus-seeder change worth scheduling.

The practical consequence today: a physician who enters a patient through intake gets weaker matching than one who pastes the same patient as a free-text narrative into chat. That is backwards from what users will expect.

### 2.3 The two retrieval paths give different quality answers

The comprehensive path (`retrieve_comprehensive`) has the priority queue, PTO lane, per-study cross-encoder gate, hard eligibility filter, and patient match scoring. The standard path (`retriever.retrieve()`, used by `/query/enhanced` when not routed to comprehensive) has a good but different stack (RRF fusion, structure rerank, preference filters) and only gained match scoring via a post-hoc patch (`enhanced_rag_service.py:5593-5640`). Which path a user's query takes determines which quality bar it gets. Worth confirming the router's surface-picking rules send every patient-context query to the comprehensive path, and adding a log-based check during beta that no patient query slips down the standard path.

### 2.4 Early termination can drop in-flight high-precision results

The priority queue breaks out of collection as soon as `total_confirmed >= max_studies` and two high-precision confirmations exist (`comprehensive_retrieval.py:1261-1270`). It explicitly only cancels Qdrant-lane tasks, which is right, but the `break` also abandons still-running Postgres and PTO tasks whose results are then never collected. A Postgres-matched study that needed 300ms more simply vanishes from consideration. Deliberate latency tradeoff, but worth knowing it exists when debugging "why didn't study X show up." An alternative: on break, await only the already-dispatched postgres/pto tasks with a short timeout (500ms) before final merge.

### 2.5 Match Criteria stubs still silently absorb user weights

Restating 2.2's second half as its own item because it is user-facing: five criteria in the Match Criteria panel map to nothing. If any of these inputs are still reachable anywhere in the UI (worth a fresh check beyond the panel that was hidden), a user's adjustment silently no-ops.

### 2.6 "Continuous monitoring" is per-case manual subscribe

The standing alerts feature is real, but it is a per-case opt-in toggle. The automatic version described in the product vision (timeline change triggers a re-match diff) is Phase 4, still unbuilt (`pattern_diff_service.py` does not exist). Fine for beta; keep outward-facing wording aligned with what ships.

### 2.7 Patient list pagination

`GET /patients` still has `limit` (default 100) and no `offset`. A physician's 101st patient is unreachable through the API. Not a beta-scale problem, but it is a two-line fix now versus a confusing support ticket later.

### 2.8 CLAUDE.md is stale and actively misleading

The implementation brief still describes PTO as "staged but not wired" and lists build tasks that were all completed. It cost this session a wrong answer to you about what is built, and it would mislead any collaborator or future coding session that trusts it. Worth updating the "current state" table or adding a completion note at the top.

---

## Part 3: What is working well (verified, leave alone)

- The eager-dispatch three-source retrieval with per-source thresholds, trust upgrading, dedup by doc_id, title fingerprint, and NCT number: well designed and correctly implemented.
- Free reranking from gate scores (no second model call) with the Phase 4 fallback when gate scores are absent.
- The hard eligibility filter's batched LLM verdict call (single call for all 10 studies, JSON response format, truncation-aware token budget).
- Answer generation properly threaded off the event loop, with numerical validation of quoted values against sources afterward.
- `patient_match_scorer` v2: the not-assessed handling, sparse-metadata fallback denominator, and wrong-cancer cap are all sound choices honestly implemented.
- The category filter's variant-spelling `should` matching (fixed the old silent fall-through that leaked wrong-cancer studies).
- The strict-category no-fallback rule (returning zero results rather than wrong-cancer results when a category is pinned).
- patientSignal.js now wired into all tool pages; footer localhost links fixed; GCS credentials now point at the real paxis-prod service account.

---

## Suggested order of work

Before beta, in this order:

1. Cloud Run: set `--concurrency` (start 4-8), confirm CPU/memory, and set the feature flags deliberately (1.4, 1.5). Config only, no code.
2. Batch the Phase 3 cross-encoder gate (1.2). Biggest per-query latency win.
3. Fix the three blocking-call sites (1.3). Small diffs, existing patterns.
4. Streaming or staged progress in the chat UI (1.1). Biggest perceived-speed win; staged progress is the low-risk version.
5. Pool the structured matcher's connections (1.6).
6. Seeder: skip answer generation, add age to the seed query (1.7, 2.1).
7. Frontend fetch timeout (1.10) and warmup hook (1.9).

Soon after beta opens: comorbidities on the patient record and into matching (2.2), pagination offset (2.7), the router check on which path patient queries take (2.3), and the CLAUDE.md refresh (2.8).

Then load-test with 5 to 10 simulated concurrent complex-patient queries before inviting real physicians, so the first stress test is not the beta itself.
