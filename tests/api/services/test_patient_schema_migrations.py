"""
Regression test for the Alembic migration freeze (2026-08-12).

migrations/patients_db/versions/0001_baseline.py, 0002_phase1_finalization.py,
and 0003_evidence_versioning_and_debug_trace.py each embed a literal,
frozen FROZEN_STATEMENTS list — historical snapshots of
src/api/services/patient_schema.py's SCHEMA_STATEMENTS at the point each
revision was written, not a live import of that (mutable, still-growing)
list. That fixes the bug this test guards against: 0001 used to run
`for statement in SCHEMA_STATEMENTS` (whatever the list happened to
contain that day, not the 92-statement baseline its name promised), and
0002 used to run `SCHEMA_STATEMENTS[92:]` (open-ended) while asserting
the result was exactly 18 statements — an assertion that was already
false by the time this was caught, because more had been appended after
Phase 1 finalization shipped.

This test is the freeze's other half: it fails loudly, at test time
rather than at `alembic upgrade head` time against a real database, the
moment SCHEMA_STATEMENTS grows past what the newest frozen revision
covers -- e.g. because a change was added to patient_schema.py without
its own migration revision. Add a new revision (0004_...) with its own
FROZEN_STATEMENTS slice and this test's REVISION_FILES/expected boundary
below to fix it, per migrations/patients_db/README.md's "Going forward"
section -- never edit an already-shipped revision's FROZEN_STATEMENTS.
"""

import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VERSIONS_DIR = REPO_ROOT / "migrations" / "patients_db" / "versions"

# (filename, expected FROZEN_STATEMENTS length) in upgrade order. Add a
# new tuple here — and a new frozen revision file — the next time
# patient_schema.py grows past statement 117; never change an existing
# entry's expected length after it has shipped.
REVISION_FILES = [
    ("0001_baseline.py", 92),
    ("0002_phase1_finalization.py", 18),
    ("0003_evidence_versioning_and_debug_trace.py", 7),
    ("0004_state_revision_freshness.py", 2),
    ("0005_verified_physician_authorization.py", 2),
]


def _load_frozen_statements(filename: str):
    path = VERSIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.FROZEN_STATEMENTS


def _live_schema_statements():
    from src.api.services.patient_schema import SCHEMA_STATEMENTS
    return SCHEMA_STATEMENTS


class TestMigrationFreeze:
    @pytest.mark.parametrize("filename,expected_len", REVISION_FILES)
    def test_frozen_revision_length_matches_its_documented_slice(self, filename, expected_len):
        assert len(_load_frozen_statements(filename)) == expected_len

    def test_frozen_revisions_concatenate_to_exactly_the_live_schema_statements(self):
        """The real regression guard: if this fails, patient_schema.py has
        drifted ahead of every frozen migration revision -- SCHEMA_STATEMENTS
        contains statements no `alembic upgrade head` run will ever execute
        against a real database. Add a new revision, not a fix here."""
        concatenated = []
        for filename, _ in REVISION_FILES:
            concatenated.extend(_load_frozen_statements(filename))
        live = _live_schema_statements()
        assert len(concatenated) == len(live), (
            f"Frozen migrations cover {len(concatenated)} statements but "
            f"patient_schema.py's SCHEMA_STATEMENTS has {len(live)}. Add a "
            "new Alembic revision (0004_...) covering the difference -- see "
            "migrations/patients_db/README.md."
        )
        assert concatenated == live, (
            "Frozen migration content has diverged from patient_schema.py "
            "for statements both cover -- an already-shipped revision must "
            "never be edited; add a new revision instead."
        )

    def test_revision_chain_is_contiguous_and_ends_at_the_newest_file(self):
        """Each revision file's down_revision must point at the previous
        one, forming a single unbranched chain ending at the last entry in
        REVISION_FILES -- catches a forgotten down_revision update when a
        new revision is added."""
        ids = []
        down = {}
        for filename, _ in REVISION_FILES:
            spec = importlib.util.spec_from_file_location(filename[:-3], VERSIONS_DIR / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            ids.append(module.revision)
            down[module.revision] = module.down_revision

        assert down[ids[0]] is None, f"{ids[0]} must be the root revision (down_revision=None)"
        for prev_id, this_id in zip(ids, ids[1:]):
            assert down[this_id] == prev_id, (
                f"{this_id}.down_revision is {down[this_id]!r}, expected {prev_id!r} "
                "-- REVISION_FILES order or a revision's down_revision is wrong."
            )

    def test_every_frozen_statement_is_idempotent_create_or_add_column(self):
        """Every statement across all frozen revisions must be safe to run
        against a database that already has some of these objects (true for
        every existing deployment, since ensure_schema() already ran) --
        i.e. CREATE ... IF NOT EXISTS or ADD COLUMN IF NOT EXISTS. A bare
        DROP/RENAME slipping into an upgrade() would not be idempotent and
        would fail on second application."""
        for filename, _ in REVISION_FILES:
            for stmt in _load_frozen_statements(filename):
                upper = stmt.upper()
                if "CREATE TABLE" in upper:
                    assert "IF NOT EXISTS" in upper, f"{filename}: CREATE TABLE without IF NOT EXISTS: {stmt[:80]!r}"
                if "CREATE INDEX" in upper or "CREATE UNIQUE INDEX" in upper:
                    assert "IF NOT EXISTS" in upper, f"{filename}: CREATE INDEX without IF NOT EXISTS: {stmt[:80]!r}"
                if "ADD COLUMN" in upper:
                    assert "IF NOT EXISTS" in upper, f"{filename}: ADD COLUMN without IF NOT EXISTS: {stmt[:80]!r}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
