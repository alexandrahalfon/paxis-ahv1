# Migrations — `exueed-patients`

Real, reviewable migrations for the `exueed-patients` database, replacing
runtime `ensure_schema()` (`src/api/services/patient_db.py`) as the
long-term schema-change strategy. Scoped to this one database only — the
other two Postgres databases this app uses (`display-study-details`,
`exueed_cache`) are unaffected and out of scope here.

## Why this exists

`ensure_schema()` (additive `CREATE ... IF NOT EXISTS` run at request
time) was fine for solo, additive-only development but doesn't give
production deployments the two things a schema change needs: a reviewable
diff of exactly what will run against the live database, and the ability
to write a real down-migration for anything that isn't purely additive
(renaming a column, tightening a constraint, backfilling data).

## How the two systems relate

`src/api/services/patient_schema.py` holds `SCHEMA_STATEMENTS` — the
single list of every `CREATE TABLE` / `CREATE INDEX` / `ALTER TABLE`
statement this app has ever issued against this database, all idempotent.
`ensure_schema()` executes the live list directly (it always wants
"everything up to today"). Migration revisions do **not** — each one
embeds a `FROZEN_STATEMENTS` list that is a literal, historical **copy**
of the slice of `SCHEMA_STATEMENTS` current when that revision was
written, not a live import. See "Why frozen, not imported" below for why
this changed from the original design.

`0001_baseline.py` is a **baseline, not a from-scratch build**: since
`ensure_schema()` already runs in every existing deployment, most
environments already have every one of these objects. Because every
statement is `IF NOT EXISTS`, running the baseline against such a
database is a safe no-op that only adds the `alembic_version` bookkeeping
row — that's what "stamping a baseline" means in practice here.

## Why frozen, not imported (2026-08-12)

The first version of this setup had `0001_baseline.py` do
`for statement in SCHEMA_STATEMENTS: op.execute(statement)` — importing
the live, still-growing list directly, on the theory that this kept
`ensure_schema()` and the migration from ever drifting apart. In practice
it meant `0001`'s behavior silently changed every time someone added a
statement to `patient_schema.py`, regardless of whether they meant to
change what the *baseline* covers: on the day this was caught, `0001`
would have executed all 117 current statements (not the 92 that define
"the schema as it stood before real migrations existed"), and `0002`
— which computed the open-ended `SCHEMA_STATEMENTS[92:]` and asserted the
result was exactly 18 — would have raised an `AssertionError` on a real
`alembic upgrade head` run, because 25 statements had accumulated past
the baseline by then.

Migrations are supposed to be frozen snapshots of what already shipped;
importing a mutable source defeats that. Every revision below now embeds
its own `FROZEN_STATEMENTS` — a literal copy taken at the time it was
written — and `tests/api/services/test_patient_schema_migrations.py`
enforces the other half: it fails the moment `SCHEMA_STATEMENTS` grows
past what the newest frozen revision covers, so drift is caught as a
failing test instead of a runtime assertion inside `alembic upgrade`.

## Going forward

**New schema changes should be a new Alembic revision, not a new
`await conn.execute(...)` appended to `ensure_schema()`.** Concretely:

1. Add the new statement(s) to `SCHEMA_STATEMENTS` in
   `src/api/services/patient_schema.py` (keeps `ensure_schema()` — the
   request-time self-heal safety net — in sync for dev/test environments
   that don't run migrations).
2. Write a new Alembic revision under `versions/` with its **own**
   `FROZEN_STATEMENTS` — a literal copy of just the new statements, not a
   slice of the live list — and a real `downgrade()` for anything that
   isn't purely additive.
3. Add the new file to `REVISION_FILES` in
   `tests/api/services/test_patient_schema_migrations.py` with its
   expected statement count, and run that test — it will tell you
   immediately if the new revision's `FROZEN_STATEMENTS` doesn't line up
   with the rest of `SCHEMA_STATEMENTS`.
4. Run the migration against staging/production instead of relying on
   `ensure_schema()` to apply it — `ensure_schema()` stays as the local/CI
   safety net, not the production deployment mechanism.
5. **Never edit an already-shipped revision's `FROZEN_STATEMENTS` or
   `down_revision`** — that revision is history; fix drift with a new
   revision, the same way you'd never rewrite a committed migration in
   any other Alembic project.

## Usage

From the repo root, with `PATIENTS_POSTGRES_*` (or the shared `POSTGRES_*`
fallback — see `src/core/config.py`) set in the environment:

```bash
# Stamp an existing (ensure_schema()-created) database as up to date,
# without re-running the baseline's statements:
alembic -c migrations/patients_db/alembic.ini stamp head

# Or, on a genuinely empty database, actually run the baseline:
alembic -c migrations/patients_db/alembic.ini upgrade head

# After adding a new revision file:
alembic -c migrations/patients_db/alembic.ini upgrade head

# Check current revision:
alembic -c migrations/patients_db/alembic.ini current
```

`env.py` builds its own synchronous connection URL from the same
`patients_postgres_*` / `postgres_*` fallback settings the app's asyncpg
pool uses (`psycopg2`, sync-only, used for migrations exclusively — the
running application never uses this driver).

## Revisions

| Revision | Adds | downgrade() |
|---|---|---|
| `0001_baseline` | Every table that predated real migrations (frozen copy of `SCHEMA_STATEMENTS[0:92]`) | Refuses — see its docstring |
| `0002_phase1_finalization` | Multi-primary/recurrence diagnoses, `tumor_profiles`, `symptom_observations`, `nutrition_assessments`, `care_team_instructions`, normalization columns (frozen copy of `SCHEMA_STATEMENTS[92:110]`) | Real — drops exactly what it added |
| `0003_evidence_versioning_and_debug_trace` | `is_current`/`superseded_by` on `evidence_document_versions`, `current_version_id`/`latest_content_hash` on `evidence_documents`, `evidence_document_version_id`/`section_title` on `evidence_chunk_registry`, `query_debug_traces` table (frozen copy of `SCHEMA_STATEMENTS[110:117]`) | Real — drops exactly what it added |

## Verified

Offline (`alembic ... upgrade head --sql` / `downgrade head:0001_baseline
--sql`, no live Postgres needed): the revision chain resolves
`<base> → 0001_baseline → 0002_phase1_finalization →
0003_evidence_versioning_and_debug_trace (head)` cleanly, `upgrade head`
emits valid SQL for all three revisions (47 `CREATE TABLE` statements),
and `downgrade head:0001_baseline` emits valid SQL exercising both
`0002` and `0003`'s real `downgrade()` bodies. `tests/api/services/
test_patient_schema_migrations.py` additionally verifies the three
frozen revisions concatenate to exactly `patient_schema.py`'s current
`SCHEMA_STATEMENTS`, that every frozen statement is idempotent, and that
the revision chain is contiguous.

## Not done here

This has **not been run against a live Postgres database** — this
sandbox has no network access to one. The offline verification above
confirms the SQL is well-formed and the revision chain resolves, but not
that it applies cleanly against real data. Run `alembic ... upgrade
head` against a real staging database (both from empty, and from an
existing `ensure_schema()`-created database via `stamp` first) and
confirm `\dt`/`\d+` output matches expectations before trusting this in
production.
