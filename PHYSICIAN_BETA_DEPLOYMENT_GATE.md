# Physician RAG Beta — Deployment Gate Checklist

> Sprint D item 27 of the 2026-08-12 patient/physician convergence
> program. This is the final item of that 27-item program — see
> `CLAUDE.md` for the original architecture brief and
> `BETA_OPTIMIZATION_AUDIT.md` for the pre-existing (patient-side, older)
> beta-readiness audit this document does not duplicate.
>
> **Purpose**: the single place a reviewer checks before flipping
> `settings.physician_rag_beta_enabled` to `True` in a real deployment.
> Every item below is either a verified, automated gate (with the exact
> test that proves it) or an honestly-flagged open item that needs a
> human/operational action this document cannot itself perform from a
> sandboxed coding session — the two are kept visually distinct rather
> than blended into one undifferentiated checklist.

---

## 1. What shipped — all 27 items

Every item below is implemented, tested, and pushed to `main`. Sprint
letters/numbers match the convergence plan's own numbering.

| # | Item | Where |
|---|---|---|
| A1 | Shared `EvidenceCandidate` dataclass | `src/api/services/evidence/evidence_candidate.py` |
| A2 | `EvidencePacket` expanded to the full shared contract | `src/api/services/evidence/evidence_packet_builder.py` |
| A3 | Interpretation policy for patient labs/state | `src/api/services/patient/lab_interpretation.py` |
| A4 | Shared `claim_grounding_validator.py` | `src/api/services/evidence/claim_grounding_validator.py` |
| A5 | `evidence_hierarchy.py` ranking policies | `src/api/services/evidence/evidence_hierarchy.py` |
| A6 | Unified trace schema (patient + physician) | `src/api/services/evidence/retrieval_debug_trace.py` |
| B7 | Deterministic state freshness via revision counters | `src/api/services/patient/patient_state_service.py`, `patient_context_service.py` |
| B8 | Source-governance enforcement at retrieval time | `src/api/services/evidence/source_governance.py` |
| B9 | Restrict patient live-PubMed fallback | `patient_pubmed_fallback_enabled` flag, `patient_chat_service.py` |
| B10 | Enforce production GCS requirement for patient documents | `require_gcs_for_patient_documents` flag, `patient_document_storage.py` |
| B11 | Claim validation on high-risk patient claims | `patient_chat_service.answer()` step 6c |
| C12 | `QueryAnalysis` adapter (audience + intent) | `src/api/services/evidence/query_analysis.py` |
| C13 | Physician context selector | `src/api/services/physician/physician_context_service.py` |
| C14 | Verified physician↔patient authorization | `authorize_physician_patient_access()`, `patient_care_team_service.py` |
| C15 | Legacy clinical retrieval adapter | `src/api/services/physician/clinical_retrieval_adapter.py` |
| C16/C17 | Physician applicability scorer + typed incompatibility taxonomy | `src/api/services/physician/physician_applicability_scorer.py` |
| C18 | Physician answer generator + care-team precedence | `src/api/services/physician/physician_answer_generator.py` |
| C19 | Physician grounding gate (mechanical + claim-level) | `src/api/services/physician/physician_grounding_gate.py` |
| C20 | Physician RAG orchestrator | `src/api/services/physician/physician_rag_orchestrator.py` |
| C21 | Protected `/api/physician-beta/*` routes | `src/api/routes/physician_beta.py`, `physician_rag_beta_enabled` flag |
| D22 | Patient golden eval scenarios | `tests/api/services/test_patient_golden_scenarios.py` (9 scenarios) |
| D23/D24 | Physician golden eval + state-sensitivity scenarios | `tests/api/services/test_physician_golden_scenarios.py` (10 scenarios) |
| D25 | Staging integration / app-wiring checks | `tests/api/test_app_wiring.py` |
| D26 | Old-vs-new pipeline comparison tests | `tests/api/services/test_physician_old_vs_new_pipeline.py` |
| D27 | This checklist | — |

Full `tests/api` sweep as of this item: **792 passed**, 0 regressions
(excluding pre-existing `hypothesis`-based property tests and
`patient_eligibility_boost_service` tests unrelated to this program —
both fail identically with or without this program's changes).

---

## 2. Automated gates — verified, with proof

Every row here is checked by a real, currently-passing test. Re-running
the referenced test is sufficient to re-verify the gate.

| Gate | Verified by |
|---|---|
| The new route is off by default; enabling it is an explicit, per-deployment choice | `tests/api/test_app_wiring.py::TestNewFeatureFlagsDefaultToOff` |
| No two routers claim the same (method, path) — a silent-shadowing risk in an app with ~20 routers | `tests/api/test_app_wiring.py::TestNoRouteCollisions` |
| `POST /api/physician-beta/query` is actually mounted | `tests/api/test_app_wiring.py::TestPhysicianBetaRouteIsMounted` |
| The whole app (all ~20 routers, including every module this program added) imports without error | `tests/api/test_app_wiring.py::TestAppAssemblesCleanly` |
| `StudyEvidence` / `ComprehensiveRetrievalResult` / `retrieve_comprehensive()`'s signature — CLAUDE.md's "do not change" list — are unmodified | `tests/api/test_app_wiring.py::TestDoNotChangeDataclassShapesAreIntact` |
| The legacy retriever is called with unmodified kwargs, unaffected by the new orchestrator | `tests/api/services/test_physician_old_vs_new_pipeline.py::TestOrchestratorOnlyPassesDocumentedKwargsToTheLegacyRetriever` |
| Toggling the new flag has zero effect on the legacy `/rag/patient-query` route or the orchestrator function itself | `tests/api/services/test_physician_old_vs_new_pipeline.py::TestFeatureFlagIsolation` |
| A denied authorization returns a generic response and never reaches the retriever | `tests/api/services/test_physician_rag_orchestrator.py::TestAuthorizationGate`, `test_physician_golden_scenarios.py::TestAuthorizationDeniedIsAGoldenSafetyScenario` |
| An answer that never cites its evidence is retried once, then falls back to a safe, non-fabricated response — never silently shipped | `tests/api/services/test_physician_grounding_gate.py`, `test_physician_golden_scenarios.py::TestToxicityManagementUnsupportedClaimIsRepaired` / `TestTrialEligibilityUnrepairableClaimFallsBack` |
| Patient state actually changes retrieval ranking (biomarker match/mismatch, care-team instruction precedence) — not just that it's accepted without crashing | `tests/api/services/test_physician_golden_scenarios.py` (D24 state-sensitivity classes) |
| Both the patient and physician chat paths behave correctly across their representative safety/evidence/escalation scenarios | `tests/api/services/test_patient_golden_scenarios.py`, `test_physician_golden_scenarios.py` |

---

## 3. Open items — need a human or operational action, not more code

These are honestly out of reach from this sandboxed session (no real
staging DB, no real OpenAI credentials for live output, no Cloud Run
console access) and should not be treated as done just because every
automated gate above is green.

- [ ] **Clinician review of real (non-scripted) model output.** Every
  test in this program uses a fake/scripted OpenAI client by design (for
  determinism — see every golden-scenario file's own docstring). Nobody
  has reviewed what the physician generator actually says against real
  evidence for real questions. Before wide rollout, run a batch of the
  golden-scenario questions (and a handful of real de-identified cases)
  through the real pipeline with a live model and have an oncologist
  review the output for clinical accuracy and tone — this is a judgment
  call automated tests cannot make.
- [ ] **Alembic migrations applied to the real staging `exueed-patients`
  DB and verified.** Migrations `0001`–`0005` are frozen and tested
  against the migration-freeze regression suite
  (`tests/.../test_patient_schema_migrations.py`), but that only proves
  the *statements* are correct and stable — nobody has run `alembic
  upgrade head` against a real staging database from this session.
- [ ] **`physician_rag_beta_enabled` must be set explicitly in the
  deployment environment (e.g. Cloud Run env vars), not left at its
  code default.** `BETA_OPTIMIZATION_AUDIT.md` §1.5 already found that
  several existing flags are set in code but their actual production
  value is unconfirmed from the repo alone — the same risk applies
  here. Record the chosen value in `DEPLOY.md` when this ships.
- [ ] **No rate limiting or per-physician cost cap on the new route.**
  Each request can make 1–3 OpenAI calls (draft generation, an optional
  mechanical retry, an optional claim-validation pass for strict
  intents) on top of whatever the legacy retriever itself already
  costs (embeddings, its own eligibility LLM check). The route inherits
  whatever app-wide protection exists today, which is none found in
  this codebase — worth a decision before opening this to real
  physician traffic at volume.
- [ ] **No monitoring/alerting wired for this route's health signals.**
  `sources_valid=False` rate (grounding fell back to the safe response),
  `authorized=False` rate (denied access attempts — could indicate a
  broken care-team link flow, or someone probing patient IDs they
  shouldn't have), and claim-repair rate are all meaningful operational
  signals `PhysicianAnswer`/`GroundedAnswer` already carry, but nothing
  currently logs or graphs them. Recommend wiring these into whatever
  logging/metrics stack the rest of the app uses before wide rollout.
- [ ] **No streaming.** `BETA_OPTIMIZATION_AUDIT.md` §1.1 found this gap
  on the patient side; it is equally true here — the physician route
  waits for the entire pipeline (retrieval, scoring, generation,
  grounding, possibly a retry and a claim-validation pass) before
  returning anything. Not a correctness gap, but a real latency-feel
  gap for a clinician-facing tool.
- [ ] **Item #48 from the pre-convergence review (security sweep: legacy
  route auth + stop returning tracebacks) is still open.** It predates
  and is independent of this 27-item program, but is called out here so
  it isn't lost — it does not block flipping `physician_rag_beta_enabled`
  specifically, since it concerns different, older routes.

---

## 4. Known, documented limitations carried into beta

Not blockers — each was a deliberate choice, documented in its own
module at the time, honestly reflecting what the current corpus/data
model can support today rather than faking a capability that doesn't
exist yet. Collected here so a reviewer sees the full list in one place
instead of finding them one docstring at a time.

- **`version_id` / `rrf_score` are always `None`** for candidates from
  the legacy clinical corpus — it predates version tracking and uses
  weighted-sum fusion, not literal RRF. (`clinical_retrieval_adapter.py`)
- **Hard incompatibility detection (biomarker/cancer-type/organ-function
  mismatch) only fires for candidates carrying structured
  `applicability_meta` tags** — the legacy corpus doesn't have this
  metadata, so today it's reachable only via the extra-corpora search
  path (`include_extra_corpora=True`), which is **off by default** (see
  next point). Legacy-only candidates still get real, if coarser, text-
  containment-based sensitivity (1.0 vs. 0.5, not 1.0 vs. 0.0).
  (`physician_applicability_scorer.py`, proven end-to-end in
  `test_physician_golden_scenarios.py::TestBiomarkerStateChangesTheApplicabilityScore`)
- **Extra-corpora search (`include_extra_corpora`) defaults to `False`.**
  `retrieval_planner.build_plan()`'s collection table is keyed by patient
  intent names; none of the four physician intents match it, so every
  physician call falls through to the same generic collection set
  regardless of intent — a real but undifferentiated result. It's also
  a live network call with no dependency-injection seam yet. Both are
  solvable, just not required for this beta. (`physician_rag_orchestrator.py`)
- **`performance_status`, `treatment_cycles`, `study_populations`,
  `outcomes`, and `ecog` have no dedicated `PatientState` field yet** —
  they're listed in `PHYSICIAN_CONTEXT_POLICY` and the scorer's
  `patient_values` shape (ready to activate the moment real fields
  exist) but currently select/score as neutral, not fabricated.
  (`physician_context_service.py`, `physician_rag_orchestrator.py`)
- **`prior_treatments` reuses `active_treatment`** — `PatientState` only
  tracks active episodes today, not a distinct completed/prior list.
- **`detect_policy_conflicts()` is an honest stub returning `[]`.**
  Automatic detection that a care-team instruction and a guideline
  actually contradict needs real semantic judgment (an LLM pass or
  clinician-authored rules); the `PRECEDENCE_ORDER` ranking is the real
  mechanism shipped now — it tells the model how to resolve a conflict
  it encounters, rather than pretending to pre-detect one.
  (`physician_answer_generator.py`)
- **No physician-side `safety_policy` content exists yet** — the
  precedence directive's "DETERMINISTIC SAFETY POLICY" prompt section
  is wired and tested, but nothing in the pipeline populates
  `packet["safety_policy"]` today (same as the patient path, which also
  never sets it), so that section never actually renders in production
  yet.
- **Claim-level validation only runs for `STRICT_VALIDATION_INTENTS`**
  (`therapy_selection`, `treatment_sequencing`, `dose_modification`,
  `drug_interaction`, `toxicity_management`, `trial_eligibility`,
  `prognosis`, `lab_interpretation`) — a general physician question
  gets the cheaper mechanical citation check only, by design (an LLM
  pass on every turn is not free).

---

## 5. Rollback plan

Because every new capability in this program is gated behind
`physician_rag_beta_enabled` (route-level) and additive-only schema
changes (never a dropped column/table, per this program's own standing
constraint), rollback is a single environment-variable flip:

1. Set `physician_rag_beta_enabled=False` (or unset it — the code
   default is already `False`).
2. `/api/physician-beta/query` immediately starts returning `404` again
   — verified by `tests/api/routes/test_physician_beta_route.py::TestDisabledByDefault`.
3. No other route, no existing schema column, and no existing behavior
   is affected — verified by `test_physician_old_vs_new_pipeline.py::TestFeatureFlagIsolation`.

No data migration or code rollback is required for this specific flag;
the new Postgres columns/tables added across this program
(`state_revision`, `link_status`/`verified_physician_user_id`/etc. on
`patient_care_team_links`, `patient_state_snapshots.source_revision`)
are all additive and harmless to leave in place regardless of the
flag's state.
