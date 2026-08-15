# Beta Scaling & Patient Data Proposal

Prepared ahead of opening Paxis to real concurrent beta users. Covers two things: (1) why queries are slow today and what breaks under concurrent load, and (2) the current state of the patient-tracking database and what "cleanly accessible" requires. Based on a full read of the retrieval pipeline, connection/pooling code, deploy config, patient DB schema, service layer, and API surface — not guesses. Every claim below is either confirmed by reading the code directly or flagged as uncertain (needs a Console check, not visible from the repo).

---

## Part 1: Why queries are slow, and what breaks under concurrency

### The one fact that explains almost everything else

The production container runs **one worker process** (`Dockerfile`, `gunicorn --workers 1`). One Python process, one event loop, per Cloud Run instance. That single fact is why every blocking call below doesn't just slow down the one request that made it — it stalls **every other concurrent user hitting that same instance** for the duration.

On top of that, the actual deploy command (`DEPLOY.md`) never sets `--concurrency`. Cloud Run's default is 80 requests routed to one instance at a time. Combined with one worker and CPU-bound reranking, that default is a bad match — it lets far more concurrent load pile onto a single instance than that instance can actually handle, instead of Cloud Run spinning up a second instance sooner.

**This is the highest-priority fix.** Everything else compounds it, but this alone determines whether 5 simultaneous beta users degrade gracefully (new instances spin up) or all pile onto one already-struggling instance.

### The specific things making individual queries slow

1. **The reranking step calls the model once per candidate study, not in a batch.** The cross-encoder gate that runs during retrieval (`comprehensive_retrieval.py`, Phase 3) scores each candidate document with its own separate call to the model, for as many as 50-100 candidates per query. Cross-encoder models are built to score many pairs at once in a single batched call — that's dramatically faster than the same work done one call at a time. This is almost certainly the direct cause of the ~8.4s reranking time you saw in the logs, and it's a CPU-bound cost that gets worse, not just linearly slower, when multiple users' requests are competing for the same CPU at once. There's already a correctly-batched version of this exact reranking step elsewhere in the same file (the old Phase 4 fallback) — the fix is applying that same batching to the Phase 3 path.

2. **Two spots make blocking network calls without yielding the event loop.** The query embedding call (one per query, every query) and a small loop that checks patient eligibility against Qdrant both make direct, synchronous calls rather than routing through the async-safe wrapper the codebase already has for exactly this purpose elsewhere. In a single-worker process, each of these calls freezes the whole instance for every other in-flight user until it returns.

3. **The busiest database path opens a brand-new connection every single query, with no pool.** The structured Postgres matcher — which runs on nearly every query — opens a raw connection per call instead of reusing a pool. This one already has a code comment describing exactly this problem happening in production: *parallel calls racing against Postgres, one flapping with a timeout, under just 5-way concurrency.* That's not a hypothetical risk, it's already been observed once.

4. **Several services build a fresh Qdrant/OpenAI client on every call instead of reusing one.** Small overhead each time, but it adds up, and it's the same pattern the codebase already recognized and fixed once for a different service (there's a comment explicitly warning against this exact mistake in the study-details service).

5. **There's effectively no caching.** Redis is listed as a dependency but is never actually used anywhere in the code — it's dead weight in the requirements file. The one cache that does exist is a small in-memory dictionary, capped at 64 entries, that only caches Postgres results, is gated behind a flag whose production value can't be confirmed from the repo, and — since it's in-memory — won't be shared once the service scales to more than one instance. Every query, even a near-identical repeat, re-embeds and re-searches from scratch.

6. **The model reloads cold on every new instance's first request.** Nothing warms the cross-encoder or the DB pools when a fresh Cloud Run instance boots. That means the exact moment concurrent load causes Cloud Run to spin up a new instance to help, that new instance's first user pays a cold-load penalty on top of everything else.

### What this means concretely for beta

Today, this architecture is tuned for "one query at a time, patiently." A handful of simultaneous beta users each submitting a complex patient profile is a realistic scenario that this setup has not been tested against, and the codebase's own comments show it's already seen cracks (the 5-way Postgres timeout) under far lower concurrency than a real beta launch would produce.

### Recommended fixes, in priority order

**Before opening beta — cheap, high-impact, low-risk:**

- Set explicit `--concurrency` on the Cloud Run deploy (start conservative — something like 4-8 — and raise it based on real load testing, not guesswork), along with explicit `--cpu`/`--memory`/`--max-instances` rather than leaving them at whatever's currently configured (which can't be confirmed from the repo — worth checking the Console directly before beta regardless).
- Batch the cross-encoder reranking calls in the Phase 3 path the same way the existing Phase 4 fallback already does it. This is very likely the single biggest latency win available, and it's a contained, well-precedented change (the correct pattern already exists in the same file).
- Fix the two blocking-call sites (query embedding, the eligibility Qdrant loop) to route through the async-safe wrapper pattern already used elsewhere in the same file.
- Convert the structured Postgres matcher to use a real connection pool instead of a raw per-call connection — same pattern already used successfully for the accounts and patients databases.
- Add a startup hook that warms the cross-encoder model and DB pools when an instance boots, so autoscaled instances don't eat a cold-load penalty on their first real request.

**Soon after, once the above is stable:**

- Reuse singleton Qdrant/OpenAI clients everywhere instead of constructing fresh ones per call.
- Add explicit timeouts and retry tuning on the OpenAI client calls, and consider a simple concurrency cap (a semaphore) around outbound OpenAI calls per instance, so a burst of concurrent users doesn't compound into OpenAI-side rate-limit errors.
- Increase the Postgres connection pool sizes (currently capped at 5 for both the accounts and patients databases) — but only in coordination with the Cloud Run concurrency setting and the actual max-connections limit on the Cloud SQL instance, so the numbers stay consistent with each other rather than just raised in isolation.

**Worth doing, not urgent for the first beta wave:**

- Actually wire up Redis (or drop it from the dependency list if it's staying unused) for a real shared query/embedding cache — this matters more once the service is running multiple instances at once, since an in-memory cache doesn't help across instances.
- Clean up the leftover, currently-unused `gunicorn_config.py` file, which computes a worker count that the actual Dockerfile command ignores — confusing for anyone reading it later, worth consolidating into one source of truth.

---

## Part 2: Patient tracking database — current state and what "cleanly accessible" needs

### What's already solid

The core schema (patients, diagnosis, biomarkers, treatment history, and an append-only timeline of every change) is a reasonable, well-normalized design. Every write to a patient record correctly generates a timeline event in the same transaction — that part of the audit trail works as intended. Every API endpoint that touches a specific patient correctly scopes the query to the requesting physician; there's no route today where one physician could pull up another's patient by guessing an ID. That's the right foundation to build on.

### What's missing or risky before real patient data goes in

**There is currently no way to delete a patient's data.** No delete endpoint exists anywhere in the product. If a physician (or, down the line, a patient) asks for their data to be removed, there's no supported path to do that today beyond going into the database directly. This is worth closing before any real health data accumulates, both because it's the right thing to do and because it's the kind of gap that's much easier to fix now than after there's a backlog of records to retroactively handle.

**Patient data is stored in plaintext, and the database connection doesn't request an encrypted connection.** Name, date of birth, and similar fields are stored as plain text with no encryption at rest, and the connection pool doesn't set any SSL/TLS parameters — so whether the connection to Postgres is even encrypted in transit depends entirely on Cloud SQL's own default settings, not on anything the app explicitly requires. Requesting an encrypted connection is a small, low-risk code change. Encrypting the most sensitive fields at rest (date of birth, MRN) is a larger piece of work worth doing but not necessarily a blocker for the first beta wave, depending on your risk tolerance.

**There's no way to see or query across patients as a business owner.** Right now the only way to look at patient data is one record at a time, through a physician's own scoped view. There's no "how many patients are in the system," no "how many were added this week," nothing. The analytics that already exist in the product are entirely about study/literature metadata — they don't touch the patient database at all. This is the core of what "cleanly accessible" was asking for, and it doesn't exist yet. Building even a simple, protected internal view (counts, growth over time, basic activity) would close this gap without much engineering effort.

**There's no export.** A physician (or you, for a support or data request) can't currently pull a clean full dump of a patient's record — the only way to see it is through the app's own UI, one screen at a time.

**Creating a patient has no duplicate check.** Two "Jane Doe" entries for the same physician are currently allowed with no warning — there's a name-matching check, but it only runs in one specific automated-capture flow, not on the direct "add patient" action most physicians will actually use.

**Listing patients silently caps at 100 with no way to see more.** Not a problem yet at beta scale, but worth fixing before it becomes a confusing "why can't I find this patient" support issue.

**No documentation on database backups exists** for either the patient database or the accounts database — only Qdrant's backup process is documented anywhere in the repo. Worth a direct check in the Cloud SQL console to confirm automated backups and point-in-time recovery are actually turned on, since that can't be confirmed from the code.

**One internal inconsistency worth knowing about:** the schema-creation code has a comment claiming it's "not wired up yet, nothing runs against the live database until this is reviewed" — but that's no longer true. The patient routes are live and registered, and the first real request to any patient endpoint will create the schema and start writing real rows. This isn't dangerous by itself, but it means the safety net the comment implies doesn't actually exist anymore, and the schema-creation step currently runs on every single request instead of once at startup (harmless, just wasteful).

### Recommended build-out, in priority order

**Before real patient data accumulates:**

- Add a delete endpoint (decide between soft-delete, keeping the timeline intact but marking a record inactive, versus a real hard-delete path for genuine removal requests — probably want both, for different use cases).
- Request an encrypted connection on the database pool (small change, meaningfully reduces risk).
- Move schema creation to happen once at app startup instead of on every request, and update the stale comment.
- Add a basic duplicate check on patient creation (at minimum, warn on a same-name match the way the automated-capture flow already does).
- Fix pagination on the patient list so it doesn't silently truncate.

**Soon after — this is the "cleanly accessible" ask directly:**

- Build a simple internal/admin view over the patient database: counts, growth, basic activity — protected, not exposed to physician accounts.
- Add a clean export (JSON to start, PDF later using the existing report-generation pattern already in the product) for a full patient record.

**Worth doing, not urgent for the first beta wave:**

- Confirm Cloud SQL backups and point-in-time recovery are actually enabled, and document it.
- Column-level encryption for the most sensitive fields (DOB, MRN).
- Reconcile the patient database with the older case-based alerts system — right now they're two parallel systems that aren't fully connected to each other, which matters less today but will matter more as the "Paxis tracks your patient continuously" story becomes more central to the product.

---

## Suggested sequencing overall

If tackling both pieces together, the highest-leverage order is roughly:

1. Cloud Run concurrency/resource settings (quick config change, immediately reduces risk of a bad multi-user experience)
2. Batch the reranking calls (biggest single latency win, contained change)
3. Fix the two blocking-call sites + pool the structured matcher's connections (closes the concurrency failure mode already observed once)
4. Patient data: delete capability + encrypted connection + fix the stale schema-creation pattern (closes the biggest before-real-data risks)
5. Everything else in the "soon after" tiers, once the above is deployed and holding up under real usage

None of this needs to happen all at once, and nothing above touches the frontend or the demo work from today — it's entirely backend/infra. Happy to start on any piece whenever you want to move on it; the reranking batching and the Cloud Run concurrency setting are both good candidates for a first pass since they're contained, well-understood changes with an outsized effect on what beta users will actually experience.
