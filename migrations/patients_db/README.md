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
**Both** `ensure_schema()` and `versions/0001_baseline.py` (below) execute
this exact same list, so there is nowhere for the two to drift apart.

`0001_baseline.py` is a **baseline, not a from-scratch build**: since
`ensure_schema()` already runs in every existing deployment, most
environments already have every one of these objects. Because every
statement is `IF NOT EXISTS`, running the baseline against such a
database is a safe no-op that only adds the `alembic_version` bookkeeping
row — that's what "stamping a baseline" means in practice here.

## Going forward

**New schema changes should be a new Alembic revision, not a new
`await conn.execute(...)` appended to `ensure_schema()`.** Concretely:

1. Add the new statement(s) to `SCHEMA_STATEMENTS` in
   `src/api/services/patient_schema.py` (keeps `ensure_schema()` — the
   request-time self-heal safety net — in sync for dev/test environments
   that don't run migrations).
2. Write a new Alembic revision under `versions/` that performs the same
   change, **with a real `downgrade()`** for anything that isn't purely
   additive.
3. Run the migration against staging/production instead of relying on
   `ensure_schema()` to apply it — `ensure_schema()` stays as the local/CI
   safety net, not the production deployment mechanism.

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
| `0001_baseline` | Every table that predated real migrations (`SCHEMA_STATEMENTS[0:92]`) | Refuses — see its docstring |
| `0002_phase1_finalization` | Multi-primary/recurrence diagnoses, `tumor_profiles`, `symptom_observations`, `nutrition_assessments`, `care_team_instructions`, normalization columns (`SCHEMA_STATEMENTS[92:110]`) | Real — drops exactly what it added |

## Not done here

This baseline was written and reviewed against `patient_schema.py`
line-for-line (see the commit that introduced it for the equivalence
check) but has **not been run against a live database** — this sandbox
has no network access to a real Postgres instance. Run
`alembic ... upgrade head` against a real staging database and confirm
`\dt`/`\d+` output matches expectations before trusting this in
production.
